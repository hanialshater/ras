use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Args {
    index: PathBuf,
    programs: PathBuf,
    positive: Vec<String>,
    negative: Vec<String>,
    candidates_file: Option<PathBuf>,
    candidate_count: usize,
    resident_items: Option<usize>,
    topk: usize,
    repeats: usize,
    early_exit: bool,
}

fn parse_list(s: &str) -> Vec<String> {
    if s.trim().is_empty() { return Vec::new(); }
    s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
}

fn args() -> Args {
    let xs: Vec<String> = env::args().collect();
    let mut a = Args {
        index: PathBuf::from("semantic_index"),
        programs: PathBuf::from("semantic_programs"),
        positive: Vec::new(),
        negative: Vec::new(),
        candidates_file: None,
        candidate_count: 5_000,
        resident_items: None,
        topk: 500,
        repeats: 7,
        early_exit: true,
    };
    let mut i = 1;
    while i < xs.len() {
        match xs[i].as_str() {
            "--index" => { i += 1; a.index = PathBuf::from(&xs[i]); }
            "--programs" => { i += 1; a.programs = PathBuf::from(&xs[i]); }
            "--positive" => { i += 1; a.positive = parse_list(&xs[i]); }
            "--negative" => { i += 1; a.negative = parse_list(&xs[i]); }
            "--candidates" => { i += 1; a.candidates_file = Some(PathBuf::from(&xs[i])); }
            "--candidate-count" => { i += 1; a.candidate_count = xs[i].parse().unwrap(); }
            "--resident-items" => { i += 1; a.resident_items = Some(xs[i].parse().unwrap()); }
            "--topk" => { i += 1; a.topk = xs[i].parse().unwrap(); }
            "--repeats" => { i += 1; a.repeats = xs[i].parse().unwrap(); }
            "--no-early-exit" => { a.early_exit = false; }
            "--help" | "-h" => {
                println!("sidecar --index DIR --programs DIR --positive a,b --negative c [--candidates ids.u32 | --candidate-count N] [--resident-items N] --topk K --repeats N [--no-early-exit]");
                std::process::exit(0);
            }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }
    assert!(!a.positive.is_empty() || !a.negative.is_empty(), "at least one semantic predicate is required");
    a
}

fn read_f32(path: impl AsRef<Path>) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect()
}

fn read_u32(path: impl AsRef<Path>) -> Vec<u32> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect()
}

#[derive(Debug)]
struct Program {
    name: String,
    packed_bytes: usize,
    planes: Vec<u8>,
    weight_lo: f32,
    weight_scale: f32,
    base: f32,
    sum_w: f32,
    cal_a: f32,
    cal_b: f32,
    positive_rate: f32,
}

fn load_program(root: &Path, name: &str) -> Program {
    let p = root.join(name);
    let planes = fs::read(p.join("bitplanes.u8")).unwrap();
    assert_eq!(planes.len() % 4, 0);
    let packed_bytes = planes.len() / 4;
    let s = read_f32(p.join("scalars.f32"));
    assert!(s.len() >= 7, "program scalars.f32 must contain seven f32 values");
    Program {
        name: name.to_string(), packed_bytes, planes,
        weight_lo: s[0], weight_scale: s[1], base: s[2], sum_w: s[3],
        cal_a: s[4], cal_b: s[5], positive_rate: s[6],
    }
}

#[derive(Debug)]
struct PredRef {
    program: Program,
    positive: bool,
}

impl PredRef {
    fn expected_acceptance(&self) -> f32 {
        if self.positive { self.program.positive_rate } else { 1.0 - self.program.positive_rate }
    }
}

struct Index {
    bits: Vec<u8>,
    corrections: Vec<f32>,
    packed_bytes: usize,
    source_items: usize,
    n_items: usize,
}

fn tile_rows<T: Copy>(src: &[T], row: usize, n: usize) -> Vec<T> {
    assert_eq!(src.len() % row, 0);
    let source_n = src.len() / row;
    assert!(source_n > 0);
    if n == source_n { return src.to_vec(); }
    let mut out = Vec::with_capacity(n * row);
    for i in 0..n {
        let s = (i % source_n) * row;
        out.extend_from_slice(&src[s..s + row]);
    }
    out
}

fn load_index(root: &Path, packed_bytes: usize, resident_items: Option<usize>) -> Index {
    let source_bits = fs::read(root.join("bits.u8")).unwrap();
    assert_eq!(source_bits.len() % packed_bytes, 0);
    let source_items = source_bits.len() / packed_bytes;
    let source_corrections = read_f32(root.join("corrections.f32"));
    assert_eq!(source_corrections.len(), source_items * 2);
    let n_items = resident_items.unwrap_or(source_items).max(1);
    let bits = tile_rows(&source_bits, packed_bytes, n_items);
    let corrections = tile_rows(&source_corrections, 2, n_items);
    Index { bits, corrections, packed_bytes, source_items, n_items }
}

#[inline(always)]
fn load_u64_le(v: &[u8], off: usize) -> u64 {
    let p = unsafe { v.as_ptr().add(off) as *const u64 };
    u64::from_le(unsafe { std::ptr::read_unaligned(p) })
}

#[inline(always)]
fn doc_pos_count(doc: &[u8]) -> u32 {
    let words = doc.len() / 8;
    let mut count = 0u32;
    for q in 0..words { count += load_u64_le(doc, q * 8).count_ones(); }
    for &b in &doc[words * 8..] { count += b.count_ones(); }
    count
}

#[inline(always)]
fn raw_program_score(doc: &[u8], pos_count: u32, lo_corr: f32, hi_corr: f32, p: &Program) -> f32 {
    let mut weighted_q_pos = 0u32;
    let words = p.packed_bytes / 8;
    for bit in 0..4 {
        let plane_off = bit * p.packed_bytes;
        let mut c = 0u32;
        for q in 0..words {
            let d = load_u64_le(doc, q * 8);
            let m = load_u64_le(&p.planes, plane_off + q * 8);
            c += (d & m).count_ones();
        }
        for j in words * 8..p.packed_bytes {
            c += (doc[j] & p.planes[plane_off + j]).count_ones();
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

#[derive(Copy, Clone, Debug)]
struct HeapEntry { score: f32, row: u32 }
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool { self.score.to_bits() == other.score.to_bits() && self.row == other.row }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then_with(|| self.row.cmp(&other.row))
    }
}

struct RunResult {
    elapsed: Duration,
    evaluated_terms: usize,
    early_rejects: usize,
    top: Vec<HeapEntry>,
}

fn execute(index: &Index, refs: &[PredRef], candidates: &[u32], topk: usize, early_exit: bool) -> RunResult {
    let k = topk.min(candidates.len());
    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::with_capacity(k + 1);
    let mut evaluated_terms = 0usize;
    let mut early_rejects = 0usize;
    let t0 = Instant::now();

    for &row in candidates {
        let i = row as usize;
        assert!(i < index.n_items);
        let off = i * index.packed_bytes;
        let doc = &index.bits[off..off + index.packed_bytes];
        let pos_count = doc_pos_count(doc);
        let lo_corr = index.corrections[i * 2];
        let hi_corr = index.corrections[i * 2 + 1];
        let mut partial = 0.0f32;
        let mut rejected = false;

        for r in refs {
            let raw = raw_program_score(doc, pos_count, lo_corr, hi_corr, &r.program);
            let logit = r.program.cal_a * raw + r.program.cal_b;
            partial += if r.positive { log_sigmoid(logit) } else { log_sigmoid(-logit) };
            evaluated_terms += 1;

            if early_exit && heap.len() == k && k > 0 {
                let threshold = heap.peek().unwrap().0.score;
                // Every remaining log-probability term is <= 0, so partial is
                // an upper bound on the final score. Once it falls below the
                // top-k threshold the item cannot recover.
                if partial <= threshold {
                    rejected = true;
                    early_rejects += 1;
                    break;
                }
            }
        }

        if rejected || k == 0 { continue; }
        let e = HeapEntry { score: partial, row };
        if heap.len() < k {
            heap.push(Reverse(e));
        } else if e.score > heap.peek().unwrap().0.score {
            heap.pop();
            heap.push(Reverse(e));
        }
    }

    let elapsed = t0.elapsed();
    let mut top: Vec<HeapEntry> = heap.into_iter().map(|x| x.0).collect();
    top.sort_by(|a, b| b.score.total_cmp(&a.score));
    RunResult { elapsed, evaluated_terms, early_rejects, top }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
}

fn random_candidates(n_items: usize, n: usize) -> Vec<u32> {
    let mut r = Rng::new(1234567);
    (0..n).map(|_| (r.next() as usize % n_items) as u32).collect()
}

fn main() {
    let a = args();
    let mut refs: Vec<PredRef> = Vec::new();
    for name in &a.positive { refs.push(PredRef { program: load_program(&a.programs, name), positive: true }); }
    for name in &a.negative { refs.push(PredRef { program: load_program(&a.programs, name), positive: false }); }
    let packed = refs[0].program.packed_bytes;
    for r in &refs { assert_eq!(r.program.packed_bytes, packed, "program dimensions differ"); }
    refs.sort_by(|a, b| a.expected_acceptance().total_cmp(&b.expected_acceptance()));
    let index = load_index(&a.index, packed, a.resident_items);

    let candidates: Vec<u32> = match &a.candidates_file {
        Some(p) => read_u32(p),
        None => random_candidates(index.n_items, a.candidate_count.min(index.n_items)),
    };
    for &id in &candidates { assert!((id as usize) < index.n_items, "candidate ID out of range"); }

    println!("source_items={}", index.source_items);
    println!("index_items={}", index.n_items);
    println!("item_bytes={}", index.packed_bytes + 8);
    println!("resident_mb={:.3}", index.n_items as f64 * (index.packed_bytes + 8) as f64 / 1e6);
    println!("candidates={}", candidates.len());
    println!("predicates={}", refs.len());
    println!("topk={}", a.topk.min(candidates.len()));
    println!("early_exit={}", a.early_exit);
    println!("plan={}", refs.iter().map(|r| format!("{}{}", if r.positive { "+" } else { "-" }, r.program.name)).collect::<Vec<_>>().join(","));

    let _ = execute(&index, &refs, &candidates, a.topk, a.early_exit);
    let mut runs = Vec::new();
    let mut last: Option<RunResult> = None;
    for _ in 0..a.repeats {
        let r = execute(&index, &refs, &candidates, a.topk, a.early_exit);
        runs.push(r.elapsed.as_secs_f64() * 1000.0);
        last = Some(r);
    }
    runs.sort_by(|a, b| a.total_cmp(b));
    let median = runs[runs.len() / 2];
    let p95 = runs[((runs.len() - 1) * 95) / 100];
    let r = last.unwrap();
    let full_terms = candidates.len() * refs.len();
    let eval_fraction = if full_terms == 0 { 0.0 } else { r.evaluated_terms as f64 / full_terms as f64 };
    let mps = candidates.len() as f64 / (median / 1000.0) / 1e6;

    println!("median_ms={:.6}", median);
    println!("p95_ms={:.6}", p95);
    println!("million_candidates_per_s={:.6}", mps);
    println!("evaluated_predicate_fraction={:.6}", eval_fraction);
    println!("early_reject_fraction={:.6}", r.early_rejects as f64 / candidates.len().max(1) as f64);
    println!("top_results:");
    for e in r.top.iter().take(20) { println!("{}\t{:.7}", e.row, e.score); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_prob_terms_are_non_positive() {
        for &z in &[-10.0f32, -1.0, 0.0, 1.0, 10.0] {
            assert!(log_sigmoid(z) <= 0.0);
            assert!(log_sigmoid(-z) <= 0.0);
        }
    }

    #[test]
    fn heap_ordering_uses_score() {
        let mut h: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
        h.push(Reverse(HeapEntry { score: -1.0, row: 1 }));
        h.push(Reverse(HeapEntry { score: -3.0, row: 2 }));
        assert_eq!(h.peek().unwrap().0.row, 2);
    }

    #[test]
    fn tiling_repeats_real_rows() {
        let x = vec![1u8, 2, 3, 4];
        assert_eq!(tile_rows(&x, 2, 3), vec![1, 2, 3, 4, 1, 2]);
    }
}
