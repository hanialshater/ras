use std::env;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const D: usize = 384;
const CONCEPTS: usize = 8;
const MAX_P: usize = 8;
const PQ_M: usize = 64;
const PQ_K: usize = 256;
const BBQ_BYTES: usize = 48;
const BBQ_WORDS: usize = 6;
const RSA2_BYTES: usize = 96;
const UNARY: usize = 24;
const PAIRS: usize = 2;
const RSA2_BINS: usize = 4;

struct Args {
    assets: PathBuf,
    resident_items: usize,
    repeats: usize,
    out: String,
}

fn args() -> Args {
    let xs: Vec<String> = env::args().collect();
    let mut a = Args {
        assets: PathBuf::from("results/native_finalists_first_seed"),
        resident_items: 500_000,
        repeats: 5,
        out: "actual_finalists_results.csv".into(),
    };
    let mut i = 1;
    while i < xs.len() {
        match xs[i].as_str() {
            "--assets" => { i += 1; a.assets = PathBuf::from(&xs[i]); }
            "--resident-items" => { i += 1; a.resident_items = xs[i].parse().unwrap(); }
            "--repeats" => { i += 1; a.repeats = xs[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = xs[i].clone(); }
            "--help" | "-h" => {
                println!("actual --assets DIR --resident-items N --repeats N --out FILE");
                std::process::exit(0);
            }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }
    a
}

fn file(dir: &Path, name: &str) -> PathBuf { dir.join(name) }
fn read_u8(path: PathBuf) -> Vec<u8> { fs::read(path).unwrap() }
fn read_f32(path: PathBuf) -> Vec<f32> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect()
}
fn read_u16(path: PathBuf) -> Vec<u16> {
    let b = fs::read(path).unwrap();
    assert_eq!(b.len() % 2, 0);
    b.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect()
}

fn tile_rows<T: Copy>(src: &[T], row: usize, n: usize) -> Vec<T> {
    assert_eq!(src.len() % row, 0);
    let source_n = src.len() / row;
    assert!(source_n > 0);
    let mut out = Vec::with_capacity(n * row);
    for i in 0..n {
        let s = (i % source_n) * row;
        out.extend_from_slice(&src[s..s + row]);
    }
    out
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    #[inline(always)] fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
}
fn candidates(items: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| (r.next() as usize) % items).collect()
}

fn bench<F: FnMut() -> f64>(mut f: F, repeats: usize) -> (f64, f64) {
    black_box(f());
    let mut ts = Vec::with_capacity(repeats);
    let mut z = 0.0;
    for _ in 0..repeats {
        let t = Instant::now();
        z = black_box(f());
        ts.push(t.elapsed().as_secs_f64());
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (ts[ts.len() / 2], z)
}

fn row(
    w: &mut BufWriter<File>, repr: &str, layout: &str, resident: usize, source: usize,
    n: usize, p: usize, bpi: usize, prog: usize, sec: f64, z: f64,
) {
    let ips = n as f64 / sec;
    writeln!(w, "{},{},{},{},{},{},{},{},{:.6},{:.3},{:.3},{:.6}",
        repr, layout, resident, source, n, p, bpi, prog, sec,
        sec * 1e9 / n as f64, ips / 1e6, z).unwrap();
}

fn linear_multi(x: &[f32], idx: &[usize], w: &[f32], intercepts: &[f32], np: usize) -> f64 {
    let mut z = 0.0f64;
    for &i in idx {
        let base = i * D;
        let mut a = [0.0f32; MAX_P];
        for p in 0..np { a[p] = intercepts[p]; }
        for j in 0..D {
            let q = unsafe { *x.get_unchecked(base + j) };
            for p in 0..np {
                a[p] += q * unsafe { *w.get_unchecked(p * D + j) };
            }
        }
        for p in 0..np { z += a[p] as f64; }
    }
    z
}

fn pq_multi(code: &[u8], idx: &[usize], lut: &[f32], intercepts: &[f32], np: usize) -> f64 {
    let mut z = 0.0f64;
    for &i in idx {
        let base = i * PQ_M;
        let mut a = [0.0f32; MAX_P];
        for p in 0..np { a[p] = intercepts[p]; }
        for m in 0..PQ_M {
            let q = unsafe { *code.get_unchecked(base + m) } as usize;
            for p in 0..np {
                a[p] += unsafe { *lut.get_unchecked((p * PQ_M + m) * PQ_K + q) };
            }
        }
        for p in 0..np { z += a[p] as f64; }
    }
    z
}

#[inline(always)]
fn load_u64_le(v: &[u8], off: usize) -> u64 {
    let p = unsafe { v.as_ptr().add(off) as *const u64 };
    u64::from_le(unsafe { std::ptr::read_unaligned(p) })
}

fn bbq_multi(
    bits: &[u8], corr: &[f32], idx: &[usize], planes: &[u8],
    weight_lo: &[f32], weight_scale: &[f32], base: &[f32], sum_w: &[f32], np: usize,
) -> f64 {
    let mut z = 0.0f64;
    for &i in idx {
        let off = i * BBQ_BYTES;
        let mut doc = [0u64; BBQ_WORDS];
        let mut pos_count = 0u32;
        for q in 0..BBQ_WORDS {
            doc[q] = load_u64_le(bits, off + q * 8);
            pos_count += doc[q].count_ones();
        }
        let lo_corr = unsafe { *corr.get_unchecked(i * 2) };
        let hi_corr = unsafe { *corr.get_unchecked(i * 2 + 1) };
        for p in 0..np {
            let mut sum_q_pos = 0u32;
            for b in 0..4 {
                let mut c = 0u32;
                let poff = (p * 4 + b) * BBQ_BYTES;
                for q in 0..BBQ_WORDS {
                    let mask = load_u64_le(planes, poff + q * 8);
                    c += (doc[q] & mask).count_ones();
                }
                sum_q_pos += c << b;
            }
            let pos_w = weight_lo[p] * pos_count as f32 + weight_scale[p] * sum_q_pos as f32;
            let s = base[p] + lo_corr * (sum_w[p] - pos_w) + hi_corr * pos_w;
            z += s as f64;
        }
    }
    z
}

#[inline(always)]
fn get2(v: &[u8], item: usize, coord: usize) -> usize {
    let b = unsafe { *v.get_unchecked(item * RSA2_BYTES + (coord >> 2)) };
    ((b >> ((coord & 3) * 2)) & 3) as usize
}

fn rsa2_multi(
    code: &[u8], idx: &[usize], unary_idx: &[u16], unary_tables: &[f32],
    pair_idx: &[u16], pair_tables: &[f32], intercepts: &[f32], np: usize,
) -> f64 {
    let mut z = 0.0f64;
    for &i in idx {
        for p in 0..np {
            let mut s = intercepts[p];
            for t in 0..UNARY {
                let c = unsafe { *unary_idx.get_unchecked(p * UNARY + t) } as usize;
                let q = get2(code, i, c);
                s += unsafe { *unary_tables.get_unchecked((p * UNARY + t) * RSA2_BINS + q) };
            }
            for m in 0..PAIRS {
                let j = unsafe { *pair_idx.get_unchecked((p * PAIRS + m) * 2) } as usize;
                let k = unsafe { *pair_idx.get_unchecked((p * PAIRS + m) * 2 + 1) } as usize;
                let qj = get2(code, i, j);
                let qk = get2(code, i, k);
                s += unsafe { *pair_tables.get_unchecked((p * PAIRS + m) * 16 + qj * 4 + qk) };
            }
            z += s as f64;
        }
    }
    z
}

fn main() {
    let a = args();
    assert!(a.resident_items >= 5_000);
    let mut out = BufWriter::new(File::create(&a.out).unwrap());
    writeln!(out, "representation,layout,resident_items,source_items,candidates,predicates,bytes_per_item,program_bytes_per_concept,seconds,ns_per_candidate,million_candidates_per_s,checksum").unwrap();
    let counts: Vec<usize> = [5_000usize, 20_000, 100_000]
        .into_iter().map(|n| n.min(a.resident_items)).collect();
    let pcs = [1usize, 2, 4, 8];

    // Read tiny programs once.
    let fp32_w = read_f32(file(&a.assets, "fp32_weights.f32"));
    let fp32_b = read_f32(file(&a.assets, "fp32_intercepts.f32"));
    let pq_lut = read_f32(file(&a.assets, "pq64_luts.f32"));
    let pq_b = read_f32(file(&a.assets, "pq64_intercepts.f32"));
    let bbq_planes = read_u8(file(&a.assets, "bbq_weight_bitplanes.u8"));
    let bbq_wlo = read_f32(file(&a.assets, "bbq_weight_lo.f32"));
    let bbq_scale = read_f32(file(&a.assets, "bbq_weight_scale.f32"));
    let bbq_base = read_f32(file(&a.assets, "bbq_base.f32"));
    let bbq_sum_w = read_f32(file(&a.assets, "bbq_sum_w.f32"));
    let rsa2_ui = read_u16(file(&a.assets, "rsa2_unary_idx.u16"));
    let rsa2_ut = read_f32(file(&a.assets, "rsa2_unary_tables.f32"));
    let rsa2_pi = read_u16(file(&a.assets, "rsa2_pair_idx.u16"));
    let rsa2_pt = read_f32(file(&a.assets, "rsa2_pair_tables.f32"));
    let rsa2_b = read_f32(file(&a.assets, "rsa2_intercepts.f32"));
    assert_eq!(fp32_w.len(), CONCEPTS * D);
    assert_eq!(pq_lut.len(), CONCEPTS * PQ_M * PQ_K);
    assert_eq!(bbq_planes.len(), CONCEPTS * 4 * BBQ_BYTES);
    assert_eq!(rsa2_ui.len(), CONCEPTS * UNARY);

    // FP32: physically tile real held-out vectors to the resident working set.
    let src_fp32 = read_f32(file(&a.assets, "fp32_items.f32"));
    let source_n = src_fp32.len() / D;
    println!("source test items: {} | resident target: {}", source_n, a.resident_items);
    println!("FP32 resident {:.3} GB", a.resident_items as f64 * 1536.0 / 1e9);
    let x = tile_rows(&src_fp32, D, a.resident_items);
    drop(src_fp32);
    for &p in &pcs { for &n in &counts {
        let ids = candidates(a.resident_items, n, 7000 + n as u64 + p as u64);
        let (s, z) = bench(|| linear_multi(&x, &ids, &fp32_w, &fp32_b, p), a.repeats);
        row(&mut out, "fp32_linear", "real_item_major_fused", a.resident_items, source_n, n, p, 1536, 1548, s, z);
    }}
    drop(x);

    // PQ64: real PQ codes and real compiled linear LUTs.
    let src_pq = read_u8(file(&a.assets, "pq64_codes.u8"));
    assert_eq!(src_pq.len() / PQ_M, source_n);
    println!("PQ64 resident {:.3} GB", a.resident_items as f64 * 64.0 / 1e9);
    let pq = tile_rows(&src_pq, PQ_M, a.resident_items);
    drop(src_pq);
    for &p in &pcs { for &n in &counts {
        let ids = candidates(a.resident_items, n, 7000 + n as u64 + p as u64);
        let (s, z) = bench(|| pq_multi(&pq, &ids, &pq_lut, &pq_b, p), a.repeats);
        row(&mut out, "pq64_linear_lut", "real_item_major_fused", a.resident_items, source_n, n, p, 64, 65548, s, z);
    }}
    drop(pq);

    // BBQ-inspired 1-bit documents + real LS2 corrections + int4 predicate bitplanes.
    let src_bits = read_u8(file(&a.assets, "bbq_bits.u8"));
    let src_corr = read_f32(file(&a.assets, "bbq_corrections.f32"));
    assert_eq!(src_bits.len() / BBQ_BYTES, source_n);
    assert_eq!(src_corr.len() / 2, source_n);
    println!("BBQ1-int4 resident {:.3} GB", a.resident_items as f64 * 56.0 / 1e9);
    let bits = tile_rows(&src_bits, BBQ_BYTES, a.resident_items);
    let corr = tile_rows(&src_corr, 2, a.resident_items);
    drop(src_bits); drop(src_corr);
    for &p in &pcs { for &n in &counts {
        let ids = candidates(a.resident_items, n, 7000 + n as u64 + p as u64);
        let (s, z) = bench(|| bbq_multi(&bits, &corr, &ids, &bbq_planes, &bbq_wlo, &bbq_scale, &bbq_base, &bbq_sum_w, p), a.repeats);
        row(&mut out, "bbq1_ls2_int4q", "real_bitpacked_popcnt", a.resident_items, source_n, n, p, 56, 216, s, z);
    }}
    drop(bits); drop(corr);

    // RSA2: real 2-bit codes and learned 24-unary + 2-pair programs.
    let src_rsa2 = read_u8(file(&a.assets, "rsa2_codes.u8"));
    assert_eq!(src_rsa2.len() / RSA2_BYTES, source_n);
    println!("RSA2 resident {:.3} GB", a.resident_items as f64 * 96.0 / 1e9);
    let rsa2 = tile_rows(&src_rsa2, RSA2_BYTES, a.resident_items);
    drop(src_rsa2);
    for &p in &pcs { for &n in &counts {
        let ids = candidates(a.resident_items, n, 7000 + n as u64 + p as u64);
        let (s, z) = bench(|| rsa2_multi(&rsa2, &ids, &rsa2_ui, &rsa2_ut, &rsa2_pi, &rsa2_pt, &rsa2_b, p), a.repeats);
        row(&mut out, "rsa2_random", "real_item_major_sparse_lut", a.resident_items, source_n, n, p, 96, 580, s, z);
    }}
    out.flush().unwrap();
    println!("wrote {}", a.out);
}
