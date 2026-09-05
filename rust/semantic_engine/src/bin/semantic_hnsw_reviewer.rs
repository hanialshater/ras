use hnsw_rs::prelude::{Distance, Hnsw};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const D: usize = 384;

#[derive(Clone, Copy, Default)]
struct DistNormalizedDot;

impl Distance<f32> for DistNormalizedDot {
    #[inline(always)]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        debug_assert_eq!(va.len(), vb.len());
        let mut s = 0.0f32;
        for i in 0..va.len() {
            s += unsafe { *va.get_unchecked(i) } * unsafe { *vb.get_unchecked(i) };
        }
        1.0 - s.clamp(-1.0, 1.0)
    }
}

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
    gates: Vec<f32>,
    overfetch_multipliers: Vec<f64>,
    progress_every: usize,
    out: PathBuf,
}

fn parse_strings(s: &str) -> Vec<String> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_f32s(s: &str) -> Vec<f32> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<f32>().unwrap())
        .collect()
}

fn parse_f64s(s: &str) -> Vec<f64> {
    s.split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<f64>().unwrap())
        .collect()
}

fn args() -> Args {
    let xs: Vec<String> = env::args().collect();
    let mut a = Args {
        assets: PathBuf::from("results/hnsw_assets"),
        programs: PathBuf::from("results/hnsw_assets/sidecar_programs"),
        positive: vec!["minimalist".into(), "office_appropriate".into()],
        negative: vec!["technical_sporty".into()],
        queries: 1000,
        k: 50,
        ef: 128,
        m: 24,
        ef_construction: 200,
        gates: vec![-1.0],
        overfetch_multipliers: vec![0.75, 1.0, 1.5, 2.0],
        progress_every: 100,
        out: PathBuf::from("semantic_hnsw_reviewer.csv"),
    };
    let mut i = 1;
    while i < xs.len() {
        match xs[i].as_str() {
            "--assets" => { i += 1; a.assets = PathBuf::from(&xs[i]); }
            "--programs" => { i += 1; a.programs = PathBuf::from(&xs[i]); }
            "--positive" => { i += 1; a.positive = parse_strings(&xs[i]); }
            "--negative" => { i += 1; a.negative = parse_strings(&xs[i]); }
            "--queries" => { i += 1; a.queries = xs[i].parse().unwrap(); }
            "--k" => { i += 1; a.k = xs[i].parse().unwrap(); }
            "--ef" => { i += 1; a.ef = xs[i].parse().unwrap(); }
            "--m" => { i += 1; a.m = xs[i].parse().unwrap(); }
            "--ef-construction" => { i += 1; a.ef_construction = xs[i].parse().unwrap(); }
            "--gates" => { i += 1; a.gates = parse_f32s(&xs[i]); }
            "--overfetch-multipliers" => { i += 1; a.overfetch_multipliers = parse_f64s(&xs[i]); }
            "--progress-every" => { i += 1; a.progress_every = xs[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = PathBuf::from(&xs[i]); }
            "--help" | "-h" => {
                println!("semantic_hnsw_reviewer --assets DIR --programs DIR --positive a,b --negative c --queries 1000 --k 50 --ef 128 --gates g1,g2 --overfetch-multipliers .75,1,1.5,2 --out FILE");
                std::process::exit(0);
            }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }
    assert!(!a.positive.is_empty() || !a.negative.is_empty());
    assert!(!a.gates.is_empty());
    assert!(!a.overfetch_multipliers.is_empty());
    assert!(a.ef >= a.k);
    a
}

fn read_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn verify_unit_norm(items: &[f32]) -> f32 {
    assert_eq!(items.len() % D, 0);
    let mut max_err = 0.0f32;
    for row in items.chunks_exact(D) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        max_err = max_err.max((norm - 1.0).abs());
    }
    assert!(max_err < 2e-3, "unit-normalized vectors required; max error={max_err}");
    max_err
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
struct PredRef { p: Program, positive: bool }

struct Sidecar {
    bits: Vec<u8>,
    corr: Vec<f32>,
    packed_bytes: usize,
    refs: Vec<PredRef>,
}

impl Sidecar {
    fn load(index_root: &Path, program_root: &Path, positive: &[String], negative: &[String]) -> Self {
        let mut refs = Vec::new();
        for n in positive { refs.push(PredRef { p: load_program(program_root, n), positive: true }); }
        for n in negative { refs.push(PredRef { p: load_program(program_root, n), positive: false }); }
        let packed_bytes = refs[0].p.packed_bytes;
        for r in &refs { assert_eq!(r.p.packed_bytes, packed_bytes); }
        let bits = fs::read(index_root.join("bits.u8")).unwrap();
        assert_eq!(bits.len() % packed_bytes, 0);
        let n = bits.len() / packed_bytes;
        let corr = read_f32(index_root.join("corrections.f32"));
        assert_eq!(corr.len(), n * 2);
        Self { bits, corr, packed_bytes, refs }
    }

    #[inline(always)]
    fn n_predicates(&self) -> usize { self.refs.len() }

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
            total += if r.positive { log_sigmoid(logit) } else { log_sigmoid(-logit) };
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
fn raw_program_score(doc: &[u8], pos_count: u32, lo_corr: f32, hi_corr: f32, p: &Program) -> f32 {
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
    if z >= 0.0 { -(-z).exp().ln_1p() } else { z - z.exp().ln_1p() }
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

fn extract_graph(hnsw: &Hnsw<'_, f32, DistNormalizedDot>, n: usize) -> Graph {
    let max_level = hnsw.get_max_level_observed() as usize;
    let mut neigh = (0..n)
        .map(|_| (0..=max_level).map(|_| Vec::<usize>::new()).collect())
        .collect::<Vec<Vec<Vec<usize>>>>();
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
    let top_nodes = level.iter().enumerate().filter_map(|(id, &l)| (l == max_level).then_some(id)).collect::<Vec<_>>();
    assert!(!top_nodes.is_empty());
    Graph { neigh, level, max_level, top_nodes }
}

fn dense_entry_descent(graph: &Graph, items: &[f32], query: &[f32]) -> usize {
    let mut cur = graph.top_nodes[0];
    let mut cur_score = dot(items, cur, query);
    for &id in &graph.top_nodes[1..] {
        let s = dot(items, id, query);
        if s > cur_score { cur = id; cur_score = s; }
    }
    for l in (1..=graph.max_level).rev() {
        if graph.level[cur] < l { continue; }
        loop {
            let mut best = cur;
            let mut best_s = dot(items, cur, query);
            for &nb in &graph.neigh[cur][l] {
                let s = dot(items, nb, query);
                if s > best_s { best_s = s; best = nb; }
            }
            if best == cur { break; }
            cur = best;
        }
    }
    cur
}

#[derive(Copy, Clone, Debug)]
struct ScoreNode { score: f32, id: usize }
impl PartialEq for ScoreNode {
    fn eq(&self, o: &Self) -> bool { self.score.to_bits() == o.score.to_bits() && self.id == o.id }
}
impl Eq for ScoreNode {}
impl PartialOrd for ScoreNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}
impl Ord for ScoreNode {
    fn cmp(&self, o: &Self) -> Ordering { self.score.total_cmp(&o.score).then_with(|| self.id.cmp(&o.id)) }
}

fn push_topk(heap: &mut BinaryHeap<Reverse<ScoreNode>>, k: usize, id: usize, score: f32) {
    if k == 0 { return; }
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

fn search_materialized(graph: &Graph, items: &[f32], sem: &[f32], query: &[f32], skip: usize, k: usize, ef: usize, gate: f32) -> SearchResult {
    let start = dense_entry_descent(graph, items, query);
    let n = graph.neigh.len();
    let mut seen = vec![false; n];
    let mut candidates = BinaryHeap::<ScoreNode>::new();
    let mut valid_beam = BinaryHeap::<Reverse<ScoreNode>>::new();
    let mut expanded = 0usize;
    let mut semantic_evals = 0usize;
    let mut dense_pruned = 0usize;
    let start_node = ScoreNode { score: dot(items, start, query), id: start };
    candidates.push(start_node);
    seen[start] = true;
    semantic_evals += 1;
    if start != skip && sem[start] >= gate { valid_beam.push(Reverse(start_node)); }
    while let Some(c) = candidates.pop() {
        if valid_beam.len() >= ef && c.score < valid_beam.peek().unwrap().0.score { break; }
        expanded += 1;
        for &nb in &graph.neigh[c.id][0] {
            if nb >= n || seen[nb] { continue; }
            seen[nb] = true;
            let dense_s = dot(items, nb, query);
            if valid_beam.len() >= ef && dense_s <= valid_beam.peek().unwrap().0.score {
                dense_pruned += 1;
                continue;
            }
            semantic_evals += 1;
            let node = ScoreNode { score: dense_s, id: nb };
            candidates.push(node);
            if nb != skip && sem[nb] >= gate {
                valid_beam.push(Reverse(node));
                if valid_beam.len() > ef { valid_beam.pop(); }
            }
        }
    }
    let mut results = BinaryHeap::<Reverse<ScoreNode>>::new();
    for x in valid_beam {
        let id = x.0.id;
        if id != skip && sem[id] >= gate { push_topk(&mut results, k, id, dot(items, id, query)); }
    }
    SearchResult { ids: heap_ids(results), visited: expanded, semantic_evals, predicate_evals: 0, dense_pruned_before_semantic: dense_pruned }
}

struct LiveSemanticCache {
    values: Vec<f32>,
    stamps: Vec<u32>,
    epoch: u32,
    semantic_evals: usize,
    predicate_evals: usize,
}

impl LiveSemanticCache {
    fn new(n: usize) -> Self { Self { values: vec![0.0; n], stamps: vec![0; n], epoch: 0, semantic_evals: 0, predicate_evals: 0 } }
    fn begin_query(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 { self.stamps.fill(0); self.epoch = 1; }
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

fn search_live(graph: &Graph, items: &[f32], sidecar: &Sidecar, cache: &mut LiveSemanticCache, query: &[f32], skip: usize, k: usize, ef: usize, gate: f32) -> SearchResult {
    cache.begin_query();
    let start = dense_entry_descent(graph, items, query);
    let n = graph.neigh.len();
    let mut seen = vec![false; n];
    let mut candidates = BinaryHeap::<ScoreNode>::new();
    let mut valid_beam = BinaryHeap::<Reverse<ScoreNode>>::new();
    let mut expanded = 0usize;
    let mut dense_pruned = 0usize;
    let start_node = ScoreNode { score: dot(items, start, query), id: start };
    candidates.push(start_node);
    seen[start] = true;
    if start != skip && cache.get(sidecar, start) >= gate { valid_beam.push(Reverse(start_node)); }
    while let Some(c) = candidates.pop() {
        if valid_beam.len() >= ef && c.score < valid_beam.peek().unwrap().0.score { break; }
        expanded += 1;
        for &nb in &graph.neigh[c.id][0] {
            if nb >= n || seen[nb] { continue; }
            seen[nb] = true;
            let dense_s = dot(items, nb, query);
            if valid_beam.len() >= ef && dense_s <= valid_beam.peek().unwrap().0.score {
                dense_pruned += 1;
                continue;
            }
            let node = ScoreNode { score: dense_s, id: nb };
            candidates.push(node);
            if nb != skip && cache.get(sidecar, nb) >= gate {
                valid_beam.push(Reverse(node));
                if valid_beam.len() > ef { valid_beam.pop(); }
            }
        }
    }
    let mut results = BinaryHeap::<Reverse<ScoreNode>>::new();
    for x in valid_beam { push_topk(&mut results, k, x.0.id, dot(items, x.0.id, query)); }
    SearchResult { ids: heap_ids(results), visited: expanded, semantic_evals: cache.semantic_evals, predicate_evals: cache.predicate_evals, dense_pruned_before_semantic: dense_pruned }
}

fn truth_from_dense(dense: &[f32], sem: &[f32], gate: f32, k: usize, skip: usize) -> Vec<usize> {
    let mut heap = BinaryHeap::<Reverse<ScoreNode>>::new();
    for id in 0..sem.len() {
        if id == skip || sem[id] < gate { continue; }
        push_topk(&mut heap, k, id, dense[id]);
    }
    heap_ids(heap)
}

fn recall_at_k(found: &[usize], truth: &[usize]) -> f64 {
    if truth.is_empty() { return 1.0; }
    found.iter().filter(|x| truth.contains(x)).count() as f64 / truth.len() as f64
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
    let norm_err = verify_unit_norm(&items);
    let sidecar = Sidecar::load(&a.assets.join("sidecar_index"), &a.programs, &a.positive, &a.negative);
    assert_eq!(sidecar.bits.len() / sidecar.packed_bytes, n);

    println!("[reviewer] items={} dim={} predicates={} gates={} queries={} max_norm_error={:.6}", n, D, sidecar.n_predicates(), a.gates.len(), a.queries.min(n), norm_err);
    println!("[reviewer] building HNSW once ...");
    let t0 = Instant::now();
    let max_layer = 16usize.min(((n.max(2) as f64).ln().ceil() as usize).max(2));
    let mut hnsw = Hnsw::<f32, DistNormalizedDot>::new(a.m, n, max_layer, a.ef_construction, DistNormalizedDot {});
    for id in 0..n { hnsw.insert_slice((&items[id * D..(id + 1) * D], id)); }
    hnsw.set_searching_mode(true);
    let graph = extract_graph(&hnsw, n);
    println!("[reviewer] HNSW built in {:.2}s", t0.elapsed().as_secs_f64());

    println!("[reviewer] materializing semantic truth scores once ...");
    let sem_t0 = Instant::now();
    let sem = (0..n).map(|i| sidecar.semantic_mean_logprob(i)).collect::<Vec<_>>();
    let qualified = a.gates.iter().map(|&g| sem.iter().filter(|&&s| s >= g).count()).collect::<Vec<_>>();
    for (gi, &g) in a.gates.iter().enumerate() {
        println!("[reviewer] gate {} value={:.6} eligible={} fraction={:.6}", gi, g, qualified[gi], qualified[gi] as f64 / n as f64);
        assert!(qualified[gi] >= a.k);
    }
    println!("[reviewer] semantic scores ready in {:.2}s", sem_t0.elapsed().as_secs_f64());

    let mut out = BufWriter::new(File::create(&a.out).unwrap());
    writeln!(out, "query_id,gate_index,gate_logprob,method,overfetch_multiplier,ask,latency_ms,recall_at_k,returned,visited,semantic_evals,predicate_evals,dense_pruned_before_semantic,qualified_fraction,live_matches_materialized").unwrap();

    let qn = a.queries.min(n);
    let mut live_cache = LiveSemanticCache::new(n);
    let bench_t0 = Instant::now();
    for qi in 0..qn {
        if a.progress_every > 0 && qi % a.progress_every == 0 {
            println!("[reviewer] query {}/{} elapsed={:.1}s", qi, qn, bench_t0.elapsed().as_secs_f64());
        }
        let qid = (qi * 9973) % n;
        let query = &items[qid * D..(qid + 1) * D];
        let mut dense = vec![0.0f32; n];
        for id in 0..n { dense[id] = dot(&items, id, query); }

        for (gi, &gate) in a.gates.iter().enumerate() {
            let qfrac = qualified[gi] as f64 / n as f64;
            let truth = truth_from_dense(&dense, &sem, gate, a.k, qid);

            let t = Instant::now();
            let materialized = search_materialized(&graph, &items, &sem, query, qid, a.k, a.ef, gate);
            let materialized_ms = t.elapsed().as_secs_f64() * 1000.0;
            let materialized_r = recall_at_k(&materialized.ids, &truth);

            let t = Instant::now();
            let live = search_live(&graph, &items, &sidecar, &mut live_cache, query, qid, a.k, a.ef, gate);
            let live_ms = t.elapsed().as_secs_f64() * 1000.0;
            let live_r = recall_at_k(&live.ids, &truth);
            let parity = same_ids(&live.ids, &materialized.ids);

            writeln!(out, "{qi},{gi},{gate:.8},custom_hnsw_materialized,0,0,{materialized_ms:.6},{materialized_r:.6},{},{},{},{},{},{qfrac:.6},true",
                materialized.ids.len(), materialized.visited, materialized.semantic_evals, materialized.predicate_evals, materialized.dense_pruned_before_semantic).unwrap();
            writeln!(out, "{qi},{gi},{gate:.8},semantic_hnsw_live,0,0,{live_ms:.6},{live_r:.6},{},{},{},{},{},{qfrac:.6},{}",
                live.ids.len(), live.visited, live.semantic_evals, live.predicate_evals, live.dense_pruned_before_semantic, parity).unwrap();

            for &mult in &a.overfetch_multipliers {
                let ask = (((mult * a.k as f64) / qfrac).ceil() as usize).max(a.k).min(n.saturating_sub(1).max(a.k));
                let t = Instant::now();
                let raw = hnsw.search(query, ask, a.ef.max(ask));
                let mut post = Vec::with_capacity(a.k);
                for x in raw {
                    if x.d_id != qid && sem[x.d_id] >= gate {
                        post.push(x.d_id);
                        if post.len() == a.k { break; }
                    }
                }
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                let r = recall_at_k(&post, &truth);
                writeln!(out, "{qi},{gi},{gate:.8},hnsw_overfetch_materialized,{mult:.4},{ask},{ms:.6},{r:.6},{},0,0,0,0,{qfrac:.6},true", post.len()).unwrap();
            }
        }
    }
    out.flush().unwrap();
    println!("[reviewer] done in {:.2}s results={}", bench_t0.elapsed().as_secs_f64(), a.out.display());
}
