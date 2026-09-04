use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::Instant;

const D: usize = 384;
const RSA_BYTES: usize = D / 2;
const BINS: usize = 16;
const UNARY: usize = 24;
const PAIRS: usize = 2;
const PQ_M: usize = 64;
const MAX_PRED: usize = 8;

#[derive(Clone)]
struct Rng64(u64);
impl Rng64 {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline(always)]
    fn next_f32(&mut self) -> f32 {
        let x = (self.next_u64() >> 40) as u32;
        (x as f32) / ((1u32 << 24) as f32) - 0.5
    }
}

#[derive(Clone)]
struct ProgramF32 {
    coord: [u16; UNARY],
    unary: [[f32; BINS]; UNARY],
    pair: [[f32; BINS * BINS]; PAIRS],
    intercept: f32,
}

#[derive(Clone)]
struct ProgramI16 {
    coord: [u16; UNARY],
    unary: [[i16; BINS]; UNARY],
    pair: [[i16; BINS * BINS]; PAIRS],
    intercept: i32,
    scale_inv: f32,
}

struct FusedPlan {
    coords: Vec<u16>,
    pos: Vec<[u8; UNARY]>,
}

struct Args {
    throughput_items: usize,
    capacity_items: usize,
    repeats: usize,
    seed: u64,
    out: String,
}

fn parse_usize(s: &str, name: &str) -> usize {
    s.parse::<usize>().unwrap_or_else(|_| panic!("invalid {}: {}", name, s))
}

fn parse_args() -> Args {
    let mut a = Args {
        throughput_items: 500_000,
        capacity_items: 5_000_000,
        repeats: 5,
        seed: 7,
        out: "systems_results.csv".to_string(),
    };
    let xs: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < xs.len() {
        match xs[i].as_str() {
            "--throughput-items" => { i += 1; a.throughput_items = parse_usize(&xs[i], "throughput-items"); }
            "--capacity-items" => { i += 1; a.capacity_items = parse_usize(&xs[i], "capacity-items"); }
            "--repeats" => { i += 1; a.repeats = parse_usize(&xs[i], "repeats"); }
            "--seed" => { i += 1; a.seed = xs[i].parse().expect("invalid seed"); }
            "--out" => { i += 1; a.out = xs[i].clone(); }
            "--help" | "-h" => {
                println!("rsa-semantic-engine [--throughput-items N] [--capacity-items N] [--repeats N] [--seed N] [--out PATH]");
                std::process::exit(0);
            }
            other => panic!("unknown argument: {}", other),
        }
        i += 1;
    }
    a
}

fn fill_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut out = vec![0u8; len];
    let mut rng = Rng64::new(seed);
    let mut chunks = out.chunks_exact_mut(8);
    for c in &mut chunks {
        c.copy_from_slice(&rng.next_u64().to_le_bytes());
    }
    for b in chunks.into_remainder() { *b = rng.next_u64() as u8; }
    out
}

fn make_candidates(items: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng64::new(seed);
    (0..n).map(|_| (rng.next_u64() as usize) % items).collect()
}

fn make_programs(n: usize, overlap: f32, seed: u64) -> Vec<ProgramF32> {
    assert!(n <= MAX_PRED);
    let common = ((UNARY as f32) * overlap).round() as usize;
    let mut rng = Rng64::new(seed);
    let mut out = Vec::with_capacity(n);
    for p in 0..n {
        let mut coord = [0u16; UNARY];
        for t in 0..UNARY {
            let c = if t < common {
                t
            } else {
                common + p * (UNARY - common) + (t - common)
            };
            coord[t] = (c % D) as u16;
        }
        let mut unary = [[0.0f32; BINS]; UNARY];
        for t in 0..UNARY {
            for b in 0..BINS { unary[t][b] = rng.next_f32(); }
        }
        let mut pair = [[0.0f32; BINS * BINS]; PAIRS];
        for m in 0..PAIRS {
            for b in 0..(BINS * BINS) { pair[m][b] = rng.next_f32(); }
        }
        out.push(ProgramF32 { coord, unary, pair, intercept: rng.next_f32() });
    }
    out
}

fn quantize_program(p: &ProgramF32, scale: f32) -> ProgramI16 {
    let mut unary = [[0i16; BINS]; UNARY];
    let mut pair = [[0i16; BINS * BINS]; PAIRS];
    for t in 0..UNARY {
        for b in 0..BINS {
            unary[t][b] = (p.unary[t][b] * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
    }
    for m in 0..PAIRS {
        for b in 0..(BINS * BINS) {
            pair[m][b] = (p.pair[m][b] * scale).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        }
    }
    ProgramI16 {
        coord: p.coord,
        unary,
        pair,
        intercept: (p.intercept * scale).round() as i32,
        scale_inv: 1.0 / scale,
    }
}

fn build_fused_plan(programs: &[ProgramF32]) -> FusedPlan {
    let mut coords: Vec<u16> = Vec::new();
    let mut pos: Vec<[u8; UNARY]> = Vec::with_capacity(programs.len());
    for p in programs {
        let mut pp = [0u8; UNARY];
        for t in 0..UNARY {
            let c = p.coord[t];
            let k = match coords.iter().position(|&x| x == c) {
                Some(k) => k,
                None => { coords.push(c); coords.len() - 1 }
            };
            assert!(k < 256);
            pp[t] = k as u8;
        }
        pos.push(pp);
    }
    FusedPlan { coords, pos }
}

#[inline(always)]
fn nibble_item(code: &[u8], item: usize, coord: usize) -> u8 {
    let b = unsafe { *code.get_unchecked(item * RSA_BYTES + (coord >> 1)) };
    if (coord & 1) == 0 { b & 0x0f } else { b >> 4 }
}

#[inline(always)]
fn nibble_col(code: &[u8], items: usize, item: usize, coord: usize) -> u8 {
    let stride = (items + 1) >> 1;
    let b = unsafe { *code.get_unchecked(coord * stride + (item >> 1)) };
    if (item & 1) == 0 { b & 0x0f } else { b >> 4 }
}

fn item_to_columnar(item_code: &[u8], items: usize) -> Vec<u8> {
    let stride = (items + 1) >> 1;
    let mut out = vec![0u8; D * stride];
    for item in 0..items {
        for coord in 0..D {
            let q = nibble_item(item_code, item, coord);
            let idx = coord * stride + (item >> 1);
            if (item & 1) == 0 { out[idx] = (out[idx] & 0xf0) | q; }
            else { out[idx] = (out[idx] & 0x0f) | (q << 4); }
        }
    }
    out
}

#[inline(always)]
fn score_rsa_item(code: &[u8], item: usize, p: &ProgramF32) -> f32 {
    let mut s = p.intercept;
    let mut q4 = [0u8; 4];
    for t in 0..UNARY {
        let q = nibble_item(code, item, p.coord[t] as usize);
        if t < 4 { q4[t] = q; }
        s += unsafe { *p.unary[t].get_unchecked(q as usize) };
    }
    s += unsafe { *p.pair[0].get_unchecked((q4[0] as usize) * BINS + q4[1] as usize) };
    s += unsafe { *p.pair[1].get_unchecked((q4[2] as usize) * BINS + q4[3] as usize) };
    s
}

#[inline(always)]
fn score_rsa_item_i16(code: &[u8], item: usize, p: &ProgramI16) -> f32 {
    let mut s = p.intercept;
    let mut q4 = [0u8; 4];
    for t in 0..UNARY {
        let q = nibble_item(code, item, p.coord[t] as usize);
        if t < 4 { q4[t] = q; }
        s += unsafe { *p.unary[t].get_unchecked(q as usize) as i32 };
    }
    s += unsafe { *p.pair[0].get_unchecked((q4[0] as usize) * BINS + q4[1] as usize) as i32 };
    s += unsafe { *p.pair[1].get_unchecked((q4[2] as usize) * BINS + q4[3] as usize) as i32 };
    (s as f32) * p.scale_inv
}

#[inline(always)]
fn score_rsa_col(code: &[u8], items: usize, item: usize, p: &ProgramF32) -> f32 {
    let mut s = p.intercept;
    let mut q4 = [0u8; 4];
    for t in 0..UNARY {
        let q = nibble_col(code, items, item, p.coord[t] as usize);
        if t < 4 { q4[t] = q; }
        s += unsafe { *p.unary[t].get_unchecked(q as usize) };
    }
    s += unsafe { *p.pair[0].get_unchecked((q4[0] as usize) * BINS + q4[1] as usize) };
    s += unsafe { *p.pair[1].get_unchecked((q4[2] as usize) * BINS + q4[3] as usize) };
    s
}

fn score_rsa_item_multi(code: &[u8], idx: &[usize], programs: &[ProgramF32]) -> f64 {
    let mut total = 0.0f64;
    for &item in idx {
        for p in programs { total += score_rsa_item(code, item, p) as f64; }
    }
    total
}

fn score_rsa_item_i16_multi(code: &[u8], idx: &[usize], programs: &[ProgramI16]) -> f64 {
    let mut total = 0.0f64;
    for &item in idx {
        for p in programs { total += score_rsa_item_i16(code, item, p) as f64; }
    }
    total
}

fn score_rsa_col_multi(code: &[u8], items: usize, idx: &[usize], programs: &[ProgramF32]) -> f64 {
    let mut total = 0.0f64;
    for &item in idx {
        for p in programs { total += score_rsa_col(code, items, item, p) as f64; }
    }
    total
}

fn score_rsa_fused_item(code: &[u8], idx: &[usize], programs: &[ProgramF32], plan: &FusedPlan) -> f64 {
    let mut total = 0.0f64;
    let mut vals = vec![0u8; plan.coords.len()];
    for &item in idx {
        for (k, &coord) in plan.coords.iter().enumerate() { vals[k] = nibble_item(code, item, coord as usize); }
        for (pi, p) in programs.iter().enumerate() {
            let mut s = p.intercept;
            let pp = &plan.pos[pi];
            for t in 0..UNARY {
                let q = vals[pp[t] as usize];
                s += unsafe { *p.unary[t].get_unchecked(q as usize) };
            }
            let q0 = vals[pp[0] as usize] as usize;
            let q1 = vals[pp[1] as usize] as usize;
            let q2 = vals[pp[2] as usize] as usize;
            let q3 = vals[pp[3] as usize] as usize;
            s += unsafe { *p.pair[0].get_unchecked(q0 * BINS + q1) };
            s += unsafe { *p.pair[1].get_unchecked(q2 * BINS + q3) };
            total += s as f64;
        }
    }
    total
}

fn score_rsa_fused_col(code: &[u8], items: usize, idx: &[usize], programs: &[ProgramF32], plan: &FusedPlan) -> f64 {
    let mut total = 0.0f64;
    let mut vals = vec![0u8; plan.coords.len()];
    for &item in idx {
        for (k, &coord) in plan.coords.iter().enumerate() { vals[k] = nibble_col(code, items, item, coord as usize); }
        for (pi, p) in programs.iter().enumerate() {
            let mut s = p.intercept;
            let pp = &plan.pos[pi];
            for t in 0..UNARY {
                let q = vals[pp[t] as usize];
                s += unsafe { *p.unary[t].get_unchecked(q as usize) };
            }
            let q0 = vals[pp[0] as usize] as usize;
            let q1 = vals[pp[1] as usize] as usize;
            let q2 = vals[pp[2] as usize] as usize;
            let q3 = vals[pp[3] as usize] as usize;
            s += unsafe { *p.pair[0].get_unchecked(q0 * BINS + q1) };
            s += unsafe { *p.pair[1].get_unchecked(q2 * BINS + q3) };
            total += s as f64;
        }
    }
    total
}

fn score_fp32_multi(x: &[f32], idx: &[usize], weights: &[f32], predicates: usize) -> f64 {
    let mut total = 0.0f64;
    for &item in idx {
        let base = item * D;
        let mut acc = [0.0f32; MAX_PRED];
        for j in 0..D {
            let v = unsafe { *x.get_unchecked(base + j) };
            for p in 0..predicates {
                acc[p] += v * unsafe { *weights.get_unchecked(p * D + j) };
            }
        }
        for p in 0..predicates { total += acc[p] as f64; }
    }
    total
}

fn score_pq64_multi(code: &[u8], idx: &[usize], luts: &[f32], predicates: usize) -> f64 {
    let mut total = 0.0f64;
    for &item in idx {
        let base = item * PQ_M;
        let mut acc = [0.0f32; MAX_PRED];
        for m in 0..PQ_M {
            let q = unsafe { *code.get_unchecked(base + m) } as usize;
            for p in 0..predicates {
                let off = (p * PQ_M + m) * 256 + q;
                acc[p] += unsafe { *luts.get_unchecked(off) };
            }
        }
        for p in 0..predicates { total += acc[p] as f64; }
    }
    total
}

fn bench<F: FnMut() -> f64>(mut f: F, repeats: usize) -> (f64, f64) {
    black_box(f());
    let mut times = Vec::with_capacity(repeats);
    let mut checksum = 0.0f64;
    for _ in 0..repeats {
        let t0 = Instant::now();
        checksum = black_box(f());
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|a,b| a.partial_cmp(b).unwrap());
    (times[times.len()/2], checksum)
}

fn write_row(w: &mut BufWriter<File>, repr: &str, layout: &str, items: usize, candidates: usize, predicates: usize, bytes_per_item: usize, seconds: f64, checksum: f64) {
    let ns_item = seconds * 1e9 / (candidates as f64);
    let items_s = (candidates as f64) / seconds;
    let logical_gbs = items_s * (bytes_per_item as f64) / 1e9;
    writeln!(w, "{},{},{},{},{},{},{:.6},{:.3},{:.3},{:.6},{:.6}", repr, layout, items, candidates, predicates, bytes_per_item, seconds, ns_item, items_s/1e6, logical_gbs, checksum).unwrap();
}

fn main() {
    let args = parse_args();
    assert!(args.throughput_items > 1000);
    println!("RSA systems benchmark: throughput_items={}, capacity_items={}, repeats={}", args.throughput_items, args.capacity_items, args.repeats);
    println!("capacity: FP32={:.3} GB, RSA4={:.3} GB, PQ64={:.3} GB", args.capacity_items as f64 * 1536.0 / 1e9, args.capacity_items as f64 * 192.0 / 1e9, args.capacity_items as f64 * 64.0 / 1e9);

    let file = File::create(&args.out).expect("create output");
    let mut w = BufWriter::new(file);
    writeln!(w, "representation,layout,catalog_items,candidates,predicates,bytes_per_item,seconds,ns_per_candidate,million_candidates_per_s,logical_gb_per_s,checksum").unwrap();

    let counts: Vec<usize> = [5_000usize, 20_000, 100_000].iter().map(|&n| n.min(args.throughput_items)).collect();
    let pred_counts = [1usize, 2, 4, 8];

    {
        println!("allocating FP32 catalog ({:.3} GB)", args.throughput_items as f64 * 1536.0 / 1e9);
        let x = vec![0.125f32; args.throughput_items * D];
        let mut rng = Rng64::new(args.seed + 10);
        let mut weights = vec![0.0f32; MAX_PRED * D];
        for v in &mut weights { *v = rng.next_f32(); }
        for &p in &pred_counts {
            for &n in &counts {
                let idx = make_candidates(args.throughput_items, n, args.seed + n as u64 + p as u64);
                let (sec, sum) = bench(|| score_fp32_multi(&x, &idx, &weights, p), args.repeats);
                write_row(&mut w, "fp32_linear", "item_major_fused", args.throughput_items, n, p, 1536, sec, sum);
            }
        }
        black_box(x);
    }

    {
        println!("allocating PQ64 catalog ({:.3} GB)", args.throughput_items as f64 * 64.0 / 1e9);
        let code = fill_bytes(args.throughput_items * PQ_M, args.seed + 20);
        let mut rng = Rng64::new(args.seed + 21);
        let mut luts = vec![0.0f32; MAX_PRED * PQ_M * 256];
        for v in &mut luts { *v = rng.next_f32(); }
        for &p in &pred_counts {
            for &n in &counts {
                let idx = make_candidates(args.throughput_items, n, args.seed + 1000 + n as u64 + p as u64);
                let (sec, sum) = bench(|| score_pq64_multi(&code, &idx, &luts, p), args.repeats);
                write_row(&mut w, "pq64_lut_head", "item_major_fused", args.throughput_items, n, p, 64, sec, sum);
            }
        }
        black_box(code);
    }

    {
        println!("allocating RSA4 item-major catalog ({:.3} GB)", args.throughput_items as f64 * 192.0 / 1e9);
        let code = fill_bytes(args.throughput_items * RSA_BYTES, args.seed + 30);
        for &p in &pred_counts {
            let programs = make_programs(p, 0.5, args.seed + 300 + p as u64);
            let programs_i16: Vec<ProgramI16> = programs.iter().map(|x| quantize_program(x, 1024.0)).collect();
            let plan = build_fused_plan(&programs);
            for &n in &counts {
                let idx = make_candidates(args.throughput_items, n, args.seed + 2000 + n as u64 + p as u64);
                let (sec, sum) = bench(|| score_rsa_item_multi(&code, &idx, &programs), args.repeats);
                write_row(&mut w, "rsa4_f32", "item_major_fixed24x2", args.throughput_items, n, p, 192, sec, sum);
                let (sec, sum) = bench(|| score_rsa_item_i16_multi(&code, &idx, &programs_i16), args.repeats);
                write_row(&mut w, "rsa4_i16", "item_major_fixed24x2", args.throughput_items, n, p, 192, sec, sum);
                let (sec, sum) = bench(|| score_rsa_fused_item(&code, &idx, &programs, &plan), args.repeats);
                write_row(&mut w, "rsa4_f32", "item_major_fused_union50", args.throughput_items, n, p, 192, sec, sum);
            }
        }

        println!("transposing RSA4 to coordinate-major layout");
        let col = item_to_columnar(&code, args.throughput_items);
        for &p in &pred_counts {
            let programs = make_programs(p, 0.5, args.seed + 300 + p as u64);
            let plan = build_fused_plan(&programs);
            for &n in &counts {
                let idx = make_candidates(args.throughput_items, n, args.seed + 3000 + n as u64 + p as u64);
                let (sec, sum) = bench(|| score_rsa_col_multi(&col, args.throughput_items, &idx, &programs), args.repeats);
                write_row(&mut w, "rsa4_f32", "coordinate_major_fixed24x2", args.throughput_items, n, p, 192, sec, sum);
                let (sec, sum) = bench(|| score_rsa_fused_col(&col, args.throughput_items, &idx, &programs, &plan), args.repeats);
                write_row(&mut w, "rsa4_f32", "coordinate_major_fused_union50", args.throughput_items, n, p, 192, sec, sum);
            }
        }
        black_box(col);
        black_box(code);
    }

    w.flush().unwrap();
    println!("wrote {}", args.out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nibble_round_trip_layouts() {
        let items = 17;
        let code = fill_bytes(items * RSA_BYTES, 9);
        let col = item_to_columnar(&code, items);
        for i in 0..items {
            for c in 0..D {
                assert_eq!(nibble_item(&code, i, c), nibble_col(&col, items, i, c));
            }
        }
    }

    #[test]
    fn fused_matches_independent() {
        let items = 100;
        let code = fill_bytes(items * RSA_BYTES, 11);
        let programs = make_programs(4, 0.5, 12);
        let plan = build_fused_plan(&programs);
        let idx: Vec<usize> = (0..items).collect();
        let a = score_rsa_item_multi(&code, &idx, &programs);
        let b = score_rsa_fused_item(&code, &idx, &programs, &plan);
        assert!((a - b).abs() < 1e-2);
    }

    #[test]
    fn i16_tracks_f32() {
        let items = 100;
        let code = fill_bytes(items * RSA_BYTES, 21);
        let p = make_programs(1, 0.0, 22).remove(0);
        let q = quantize_program(&p, 1024.0);
        for i in 0..items {
            assert!((score_rsa_item(&code, i, &p) - score_rsa_item_i16(&code, i, &q)).abs() < 0.03);
        }
    }
}
