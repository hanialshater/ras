use hnsw_rs::prelude::{DistCosine, Hnsw};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const D: usize = 384;

#[derive(Debug)]
struct Args {
    assets: PathBuf,
    programs: PathBuf,
    positive: Vec<String>,
    negative: Vec<String>,
    queries: usize,
    k: usize,
    ef: usize,
    m: usize,
    ef_construction: usize,
    gate_logprob: f32,
    postfilter_oversample: usize,
    out: PathBuf,
}

fn parse_list(s: &str) -> Vec<String> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|x| x.trim().to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

fn args() -> Args {
    let xs: Vec<String> = env::args().collect();
    let mut a = Args {
        assets: PathBuf::from("results/native_finalists_first_seed"),
        programs: PathBuf::from("results/native_finalists_first_seed/sidecar_programs"),
        positive: vec!["minimalist".into(), "office_appropriate".into()],
        negative: vec!["technical_sporty".into()],
        queries: 100,
        k: 50,
        ef: 128,
        m: 24,
        ef_construction: 200,
        gate_logprob: -1.0,
        postfilter_oversample: 8,
        out: PathBuf::from("semantic_hnsw_live_results.csv"),
    };

    let mut i = 1;
    while i < xs.len() {
        match xs[i].as_str() {
            "--assets" => {
                i += 1;
                a.assets = PathBuf::from(&xs[i]);
            }
            "--programs" => {
                i += 1;
                a.programs = PathBuf::from(&xs[i]);
            }
            "--positive" => {
                i += 1;
                a.positive = parse_list(&xs[i]);
            }
            "--negative" => {
                i += 1;
                a.negative = parse_list(&xs[i]);
            }
            "--queries" => {
                i += 1;
                a.queries = xs[i].parse().unwrap();
            }
            "--k" => {
                i += 1;
                a.k = xs[i].parse().unwrap();
            }
            "--ef" => {
                i += 1;
                a.ef = xs[i].parse().unwrap();
            }
            "--m" => {
                i += 1;
                a.m = xs[i].parse().unwrap();
            }
            "--ef-construction" => {
                i += 1;
                a.ef_construction = xs[i].parse().unwrap();
            }
            "--gate-logprob" => {
                i += 1;
                a.gate_logprob = xs[i].parse().unwrap();
            }
            "--postfilter-oversample" => {
                i += 1;
                a.postfilter_oversample = xs[i].parse().unwrap();
            }
            "--out" => {
                i += 1;
                a.out = PathBuf::from(&xs[i]);
            }
            "--help" | "-h" => {
                println!(
                    "semantic_hnsw_live --assets DIR --programs DIR --positive a,b --negative c \
                     --queries 100 --k 50 --ef 128 --m 24 --ef-construction 200 \
                     --gate-logprob -1.0 --postfilter-oversample 8 --out FILE"
                );
                std::process::exit(0);
            }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }

    assert!(
        !a.positive.is_empty() || !a.negative.is_empty(),
        "at least one predicate required"
    );
    assert!(a.ef >= a.k, "--ef must be at least --k");
    a
}

fn read_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

#[derive(Debug, Clone)]
struct Program {
    packed_bytes: usize,
    planes: Vec<u8>,
    weight_lo: f32,
    weight_scale: f32,
    base: f32,
    sum_w: f32,
    cal_a: f32,
    cal_b: f32,
}

fn load_program(root: &Path, name: &str) -> Program {
    let p = root.join(name);
    let planes = fs::read(p.join("bitplanes.u8")).unwrap();
    assert_eq!(planes.len() % 4, 0);
    let packed_bytes = planes.len() / 4;
    let s = read_f32(p.join("scalars.f32"));
    assert!(s.len() >= 6);
    Program {
        packed_bytes,
        planes,
        weight_lo: s[0],
        weight_scale: s[1],
        base: s[2],
        sum_w: s[3],
        cal_a: s[4],
        cal_b: s[5],
    }
}

#[derive(Clone)]
struct PredRef {
    p: Program,
    positive: bool,
}

struct Sidecar {
    bits: Vec<u8>,
    corr: Vec<f32>,
    packed_bytes: usize,
    refs: Vec<PredRef>,
}

impl Sidecar {
    fn load(index_root: &Path, program_root: &Path, positive: &[String], negative: &[String]) -> Self {
        let mut refs = Vec::new();
        for n in positive {
            refs.push(PredRef {
                p: load_program(program_root, n),
                positive: true,
            });
        }
        for n in negative {
            refs.push(PredRef {
                p: load_program(program_root, n),
                positive: false,
            });
        }

        let packed_bytes = refs[0].p.packed_bytes;
        for r in &refs {
            assert_eq!(r.p.packed_bytes, packed_bytes);
        }
        let bits = fs::read(index_root.join("bits.u8")).unwrap();
        assert_eq!(bits.len() % packed_bytes, 0);
        let n = bits.len() / packed_bytes;
        let corr = read_f32(index_root.join("corrections.f32"));
        assert_eq!(corr.len(), n * 2);

        Self {
            bits,
            corr,
            packed_bytes,
            refs,
        }
    }

    #[inline(always)]
    fn n_predicates(&self) -> usize {
        self.refs.len()
    }

    #[inline(always)]
    fn semantic_mean_logprob(&self, item: usize) -> f32 {
        let off = item * self.packed_bytes;
        let doc = &self.bits[off..off + self.packed_bytes];
        let lo_corr = self.corr[item * 2];
        let hi_corr = self.corr[item * 2 + 1];
        let pos_count = doc.iter().map(|b| b.count_ones()).sum::<u32>();
        let mut total = 0.0f32;

        for r in &self.refs {
            let raw = raw_program_score(doc, pos_count, lo_corr, hi_corr, &r.p);
            let logit = r.p.cal_a * raw + r.p.cal_b;
            total += if r.positive {
                log_sigmoid(logit)
            } else {
                log_sigmoid(-logit)
            };
        }
        total / self.refs.len() as f32
    }
}

#[inline(always)]
fn load_u64_le(v: &[u8], off: usize) -> u64 {
    let p = unsafe { v.as_ptr().add(off) as *const u64 };
    u64::from_le(unsafe { std::ptr::read_unaligned(p) })
}

#[inline(always)]
fn raw_program_score(
    doc: &[u8],
    pos_count: u32,
    lo_corr: f32,
    hi_corr: f32,
    p: &Program,
) -> f32 {
    let words = p.packed_bytes / 8;
    let mut weighted_q_pos = 0u32;

    for bit in 0..4 {
        let poff = bit * p.packed_bytes;
        let mut c = 0u32;
        for q in 0..words {
            c += (load_u64_le(doc, q * 8) & load_u64_le(&p.planes, poff + q * 8)).count_ones();
        }
        for j in words * 8..p.packed_bytes {
            c += (doc[j] & p.planes[poff + j]).count_ones();
        }
        weighted_q_pos += c << bit;
    }

    let pos_w = p.weight_lo * pos_count as f32 + p.weight_scale * weighted_q_pos as f32;
    p.base + lo_corr * (p.sum_w - pos_w) + hi_corr * pos_w
}

#[inline(always)]
fn log_sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        -(-z).exp().ln_1p()
    } else {
        z - z.exp().ln_1p()
    }
}

#[inline(always)]
fn dot(items: &[f32], item: usize, query: &[f32]) -> f32 {
    let base = item * D;
    let mut s = 0.0f32;
    for j in 0..D {
        s += unsafe { *items.get_unchecked(base + j) } * unsafe { *query.get_unchecked(j) };
    }
    s
}

#[derive(Debug)]
struct Graph {
    neigh: Vec<Vec<Vec<usize>>>,
    level: Vec<usize>,
    max_level: usize,
    top_nodes: Vec<usize>,
}

fn extract_graph(hnsw: &Hnsw<'_, f32, DistCosine>, n: usize) -> Graph {
    let max_level = hnsw.get_max_level_observed() as usize;
    let mut neigh = (0..n)
        .map(|_| {
            (0..=max_level)
                .map(|_| Vec::<usize>::new())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut level = vec![0usize; n];

    for p in hnsw.get_point_indexation().into_iter() {
        let id = p.get_origin_id();
        let pid = p.get_point_id();
        level[id] = pid.0 as usize;
        let h = p.get_neighborhood_id();
        for l in 0..=max_level.min(h.len().saturating_sub(1)) {
            neigh[id][l] = h[l].iter().map(|x| x.d_id).collect();
        }
    }

    let mut top_nodes = Vec::new();
    for (id, &l) in level.iter().enumerate() {
        if l == max_level {
            top_nodes.push(id);
        }
    }
    assert!(!top_nodes.is_empty(), "HNSW top layer unexpectedly empty");

    Graph {
        neigh,
        level,
        max_level,
        top_nodes,
    }
}

fn dense_entry_descent(graph: &Graph, items: &[f32], query: &[f32]) -> usize {
    let mut cur = graph.top_nodes[0];
    let mut cur_score = dot(items, cur, query);
    for &id in &graph.top_nodes[1..] {
        let s = dot(items, id, query);
        if s > cur_score {
            cur = id;
            cur_score = s;
        }
    }

    for l in (1..=graph.max_level).rev() {
        if graph.level[cur] < l {
            continue;
        }
        loop {
            let mut best = cur;
            let mut best_s = dot(items, cur, query);
            for &nb in &graph.neigh[cur][l] {
                let s = dot(items, nb, query);
                if s > best_s {
                    best_s = s;
                    best = nb;
                }
            }
            if best == cur {
                break;
            }
            cur = best;
        }
    }
    cur
}

#[derive(Copy, Clone, Debug)]
struct ScoreNode {
    score: f32,
    id: usize,
}

impl PartialEq for ScoreNode {
    fn eq(&self, o: &Self) -> bool {
        self.score.to_bits() == o.score.to_bits() && self.id == o.id
    }
}
impl Eq for ScoreNode {}
impl PartialOrd for ScoreNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for ScoreNode {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| self.id.cmp(&o.id))
    }
}

fn push_topk(heap: &mut BinaryHeap<Reverse<ScoreNode>>, k: usize, id: usize, score: f32) {
    if k == 0 {
        return;
    }
    let e = ScoreNode { score, id };
    if heap.len() < k {
        heap.push(Reverse(e));
    } else if score > heap.peek().unwrap().0.score {
        heap.pop();
        heap.push(Reverse(e));
    }
}

fn heap_ids(heap: BinaryHeap<Reverse<ScoreNode>>) -> Vec<usize> {
    heap.into_iter().map(|x| x.0.id).collect()
}

#[derive(Debug)]
struct SearchResult {
    ids: Vec<usize>,
    visited: usize,
    semantic_evals: usize,
    predicate_evals: usize,
    dense_pruned_before_semantic: usize,
}

/// Dense-navigation, filter-aware custom HNSW using materialized semantic scores.
fn search_materialized(
    graph: &Graph,
    items: &[f32],
    sem: &[f32],
    query: &[f32],
    skip: usize,
    k: usize,
    ef: usize,
    gate: f32,
) -> SearchResult {
    let start = dense_entry_descent(graph, items, query);
    let n = graph.neigh.len();
    let mut seen = vec![false; n];
    let mut candidates = BinaryHeap::<ScoreNode>::new();
    let mut valid_beam = BinaryHeap::<Reverse<ScoreNode>>::new();
    let mut expanded = 0usize;
    let mut semantic_evals = 0usize;
    let mut dense_pruned = 0usize;

    let start_dense = dot(items, start, query);
    let start_node = ScoreNode {
        score: start_dense,
        id: start,
    };
    candidates.push(start_node);
    seen[start] = true;
    semantic_evals += 1;
    if start != skip && sem[start] >= gate {
        valid_beam.push(Reverse(start_node));
    }

    while let Some(c) = candidates.pop() {
        if valid_beam.len() >= ef {
            let worst_valid = valid_beam.peek().unwrap().0.score;
            if c.score < worst_valid {
                break;
            }
        }

        expanded += 1;
        let id = c.id;
        for &nb in &graph.neigh[id][0] {
            if nb >= n || seen[nb] {
                continue;
            }
            seen[nb] = true;
            let dense_s = dot(items, nb, query);
            let can_improve = valid_beam.len() < ef
                || dense_s > valid_beam.peek().unwrap().0.score;
            if !can_improve {
                dense_pruned += 1;
                continue;
            }

            semantic_evals += 1;
            let node = ScoreNode {
                score: dense_s,
                id: nb,
            };
            candidates.push(node);
            if nb != skip && sem[nb] >= gate {
                valid_beam.push(Reverse(node));
                if valid_beam.len() > ef {
                    valid_beam.pop();
                }
            }
        }
    }

    let mut results = BinaryHeap::<Reverse<ScoreNode>>::new();
    for x in valid_beam.into_iter() {
        let id = x.0.id;
        if id != skip && sem[id] >= gate {
            push_topk(&mut results, k, id, dot(items, id, query));
        }
    }

    SearchResult {
        ids: heap_ids(results),
        visited: expanded,
        semantic_evals,
        predicate_evals: 0,
        dense_pruned_before_semantic: dense_pruned,
    }
}

struct LiveSemanticCache {
    values: Vec<f32>,
    stamps: Vec<u32>,
    epoch: u32,
    semantic_evals: usize,
    predicate_evals: usize,
}

impl LiveSemanticCache {
    fn new(n: usize) -> Self {
        Self {
            values: vec![0.0; n],
            stamps: vec![0; n],
            epoch: 0,
            semantic_evals: 0,
            predicate_evals: 0,
        }
    }

    fn begin_query(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
        self.semantic_evals = 0;
        self.predicate_evals = 0;
    }

    #[inline(always)]
    fn get(&mut self, sidecar: &Sidecar, item: usize) -> f32 {
        if self.stamps[item] != self.epoch {
            self.values[item] = sidecar.semantic_mean_logprob(item);
            self.stamps[item] = self.epoch;
            self.semantic_evals += 1;
            self.predicate_evals += sidecar.n_predicates();
        }
        self.values[item]
    }
}

/// The production candidate: dense HNSW navigation with the real compiled
/// Binary1-LS2-int4 predicates executed inside the timed traversal.
///
/// Semantic evaluation is exact but lazy: a neighbour that cannot improve the
/// current dense valid beam is discarded before touching the semantic program.
/// This does not change traversal relative to the materialized lambda=0 method,
/// because semantic score never affects navigation priority.
fn search_live(
    graph: &Graph,
    items: &[f32],
    sidecar: &Sidecar,
    cache: &mut LiveSemanticCache,
    query: &[f32],
    skip: usize,
    k: usize,
    ef: usize,
    gate: f32,
) -> SearchResult {
    cache.begin_query();
    let start = dense_entry_descent(graph, items, query);
    let n = graph.neigh.len();
    let mut seen = vec![false; n];
    let mut candidates = BinaryHeap::<ScoreNode>::new();
    let mut valid_beam = BinaryHeap::<Reverse<ScoreNode>>::new();
    let mut expanded = 0usize;
    let mut dense_pruned = 0usize;

    let start_dense = dot(items, start, query);
    let start_node = ScoreNode {
        score: start_dense,
        id: start,
    };
    candidates.push(start_node);
    seen[start] = true;
    let start_sem = cache.get(sidecar, start);
    if start != skip && start_sem >= gate {
        valid_beam.push(Reverse(start_node));
    }

    while let Some(c) = candidates.pop() {
        if valid_beam.len() >= ef {
            let worst_valid = valid_beam.peek().unwrap().0.score;
            if c.score < worst_valid {
                break;
            }
        }

        expanded += 1;
        let id = c.id;
        for &nb in &graph.neigh[id][0] {
            if nb >= n || seen[nb] {
                continue;
            }
            seen[nb] = true;
            let dense_s = dot(items, nb, query);
            let can_improve = valid_beam.len() < ef
                || dense_s > valid_beam.peek().unwrap().0.score;
            if !can_improve {
                dense_pruned += 1;
                continue;
            }

            let sem_s = cache.get(sidecar, nb);
            let node = ScoreNode {
                score: dense_s,
                id: nb,
            };
            // Invalid nodes remain traversable; validity only controls the
            // result beam, not the dense graph frontier.
            candidates.push(node);
            if nb != skip && sem_s >= gate {
                valid_beam.push(Reverse(node));
                if valid_beam.len() > ef {
                    valid_beam.pop();
                }
            }
        }
    }

    let mut results = BinaryHeap::<Reverse<ScoreNode>>::new();
    for x in valid_beam.into_iter() {
        let id = x.0.id;
        push_topk(&mut results, k, id, dot(items, id, query));
    }

    SearchResult {
        ids: heap_ids(results),
        visited: expanded,
        semantic_evals: cache.semantic_evals,
        predicate_evals: cache.predicate_evals,
        dense_pruned_before_semantic: dense_pruned,
    }
}

fn brute_truth(
    items: &[f32],
    query: &[f32],
    sem: &[f32],
    gate: f32,
    k: usize,
    skip: usize,
) -> Vec<usize> {
    let mut heap = BinaryHeap::<Reverse<ScoreNode>>::new();
    for id in 0..sem.len() {
        if id == skip || sem[id] < gate {
            continue;
        }
        push_topk(&mut heap, k, id, dot(items, id, query));
    }
    heap_ids(heap)
}

fn recall_at_k(found: &[usize], truth: &[usize]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut hit = 0usize;
    for x in found {
        if truth.contains(x) {
            hit += 1;
        }
    }
    hit as f64 / truth.len() as f64
}

fn same_ids(a: &[usize], b: &[usize]) -> bool {
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

fn main() {
    let a = args();
    let items = read_f32(a.assets.join("fp32_items.f32"));
    assert_eq!(items.len() % D, 0);
    let n = items.len() / D;
    let index_root = a.assets.join("sidecar_index");
    let sidecar = Sidecar::load(&index_root, &a.programs, &a.positive, &a.negative);
    assert_eq!(sidecar.bits.len() / sidecar.packed_bytes, n);

    println!(
        "items={} dim={} predicates={} m={} ef_construction={}",
        n,
        D,
        sidecar.n_predicates(),
        a.m,
        a.ef_construction
    );
    println!("gate_logprob={}", a.gate_logprob);

    let t0 = Instant::now();
    let max_layer = 16usize.min(((n.max(2) as f64).ln().ceil() as usize).max(2));
    let mut hnsw = Hnsw::<f32, DistCosine>::new(
        a.m,
        n,
        max_layer,
        a.ef_construction,
        DistCosine {},
    );
    for id in 0..n {
        hnsw.insert_slice((&items[id * D..(id + 1) * D], id));
    }
    hnsw.set_searching_mode(true);
    println!("hnsw_build_ms={:.3}", t0.elapsed().as_secs_f64() * 1000.0);
    let graph = extract_graph(&hnsw, n);

    // Materialized semantic scores are used ONLY for ground truth and idealized
    // materialized-filter baselines. The live method below never reads this vec.
    let sem_t0 = Instant::now();
    let mut sem = Vec::with_capacity(n);
    for i in 0..n {
        sem.push(sidecar.semantic_mean_logprob(i));
    }
    println!(
        "truth_semantic_precompute_ms={:.3}",
        sem_t0.elapsed().as_secs_f64() * 1000.0
    );
    let qualified = sem.iter().filter(|&&s| s >= a.gate_logprob).count();
    let qualified_fraction = qualified as f64 / n as f64;
    println!(
        "qualified={} qualified_fraction={:.6}",
        qualified, qualified_fraction
    );
    assert!(qualified >= a.k, "semantic gate leaves fewer than k eligible items");

    let mut out = BufWriter::new(File::create(&a.out).unwrap());
    writeln!(
        out,
        "query_id,method,latency_ms,recall_at_k,returned,visited,semantic_evals,predicate_evals,dense_pruned_before_semantic,qualified_fraction,live_matches_materialized"
    )
    .unwrap();

    let qn = a.queries.min(n);
    let mut live_cache = LiveSemanticCache::new(n);
    let mut sums = [0.0f64; 4];
    let mut recalls = [0.0f64; 4];
    let mut parity_matches = 0usize;

    for qi in 0..qn {
        let qid = (qi * 9973) % n;
        let query = &items[qid * D..(qid + 1) * D];
        let truth = brute_truth(&items, query, &sem, a.gate_logprob, a.k, qid);

        // Idealized materialized post-filter baseline.
        let ask = (a.k * a.postfilter_oversample).max(a.k).min(n);
        let t = Instant::now();
        let raw = hnsw.search(query, ask, a.ef.max(ask));
        let mut post = Vec::with_capacity(a.k);
        for x in raw {
            if x.d_id != qid && sem[x.d_id] >= a.gate_logprob {
                post.push(x.d_id);
                if post.len() == a.k {
                    break;
                }
            }
        }
        let post_ms = t.elapsed().as_secs_f64() * 1000.0;
        let post_r = recall_at_k(&post, &truth);
        writeln!(
            out,
            "{qi},hnsw_postfilter_materialized,{post_ms:.6},{post_r:.6},{},0,0,0,0,{qualified_fraction:.6},true",
            post.len()
        )
        .unwrap();

        // Idealized materialized filter-aware HNSW baseline.
        let predicate = |id: &usize| *id != qid && sem[*id] >= a.gate_logprob;
        let t = Instant::now();
        let filtered = hnsw.search_filter(query, a.k, a.ef.max(a.k), Some(&predicate));
        let filtered_ms = t.elapsed().as_secs_f64() * 1000.0;
        let filtered_ids: Vec<usize> = filtered.iter().map(|x| x.d_id).collect();
        let filtered_r = recall_at_k(&filtered_ids, &truth);
        writeln!(
            out,
            "{qi},hnsw_filtered_materialized,{filtered_ms:.6},{filtered_r:.6},{},0,0,0,0,{qualified_fraction:.6},true",
            filtered_ids.len()
        )
        .unwrap();

        // Same custom traversal with free materialized semantic lookup.
        let t = Instant::now();
        let materialized = search_materialized(
            &graph,
            &items,
            &sem,
            query,
            qid,
            a.k,
            a.ef,
            a.gate_logprob,
        );
        let materialized_ms = t.elapsed().as_secs_f64() * 1000.0;
        let materialized_r = recall_at_k(&materialized.ids, &truth);
        writeln!(
            out,
            "{qi},custom_hnsw_materialized,{materialized_ms:.6},{materialized_r:.6},{},{},{},{},{},{qualified_fraction:.6},true",
            materialized.ids.len(),
            materialized.visited,
            materialized.semantic_evals,
            materialized.predicate_evals,
            materialized.dense_pruned_before_semantic,
        )
        .unwrap();

        // Real Binary1-LS2-int4 predicate execution inside the timed traversal.
        let t = Instant::now();
        let live = search_live(
            &graph,
            &items,
            &sidecar,
            &mut live_cache,
            query,
            qid,
            a.k,
            a.ef,
            a.gate_logprob,
        );
        let live_ms = t.elapsed().as_secs_f64() * 1000.0;
        let live_r = recall_at_k(&live.ids, &truth);
        let matches = same_ids(&live.ids, &materialized.ids);
        if matches {
            parity_matches += 1;
        }
        writeln!(
            out,
            "{qi},semantic_hnsw_live,{live_ms:.6},{live_r:.6},{},{},{},{},{},{qualified_fraction:.6},{}",
            live.ids.len(),
            live.visited,
            live.semantic_evals,
            live.predicate_evals,
            live.dense_pruned_before_semantic,
            matches,
        )
        .unwrap();

        sums[0] += post_ms;
        sums[1] += filtered_ms;
        sums[2] += materialized_ms;
        sums[3] += live_ms;
        recalls[0] += post_r;
        recalls[1] += filtered_r;
        recalls[2] += materialized_r;
        recalls[3] += live_r;
    }

    let q = qn as f64;
    let names = [
        "hnsw_postfilter_materialized",
        "hnsw_filtered_materialized",
        "custom_hnsw_materialized",
        "semantic_hnsw_live",
    ];
    for i in 0..names.len() {
        println!(
            "mean {:28} latency_ms={:.4} recall@{}={:.4}",
            names[i],
            sums[i] / q,
            a.k,
            recalls[i] / q
        );
    }
    println!(
        "live_matches_materialized={}/{} ({:.2}%)",
        parity_matches,
        qn,
        100.0 * parity_matches as f64 / q
    );
    println!("results={}", a.out.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_probability_is_non_positive() {
        for z in [-20.0f32, -1.0, 0.0, 1.0, 20.0] {
            assert!(log_sigmoid(z) <= 0.0);
        }
    }

    #[test]
    fn topk_heap_keeps_largest_scores() {
        let mut h = BinaryHeap::<Reverse<ScoreNode>>::new();
        push_topk(&mut h, 2, 1, 0.1);
        push_topk(&mut h, 2, 2, 0.8);
        push_topk(&mut h, 2, 3, 0.5);
        let ids = heap_ids(h);
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn live_cache_reuses_value_within_query() {
        let mut c = LiveSemanticCache::new(4);
        c.begin_query();
        assert_eq!(c.semantic_evals, 0);
        assert_eq!(c.stamps, vec![0; 4]);
        // Full scoring needs a Sidecar, so this test verifies epoch semantics.
        c.stamps[2] = c.epoch;
        c.values[2] = -0.25;
        assert_eq!(c.stamps[2], c.epoch);
        assert_eq!(c.values[2], -0.25);
        c.begin_query();
        assert_ne!(c.stamps[2], c.epoch);
    }
}
