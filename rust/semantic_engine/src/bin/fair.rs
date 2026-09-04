use std::env;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::Instant;

const D: usize = 384;
const PACKED_BYTES: usize = D / 2;
const UNARY: usize = 24;
const BINS: usize = 16;
const PAIRS: usize = 2;
const PQ_M: usize = 64;
const MAX_P: usize = 8;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed.max(1)) }
    #[inline(always)] fn u64(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x
    }
    #[inline(always)] fn f32(&mut self) -> f32 {
        ((self.u64() >> 40) as f32) / ((1u32 << 24) as f32) - 0.5
    }
}

#[derive(Clone)]
struct Program {
    coord: [u16; UNARY],
    unary: [[f32; BINS]; UNARY],
    pair: [[f32; BINS * BINS]; PAIRS],
    intercept: f32,
}

struct Args { items: usize, capacity: usize, repeats: usize, out: String }
fn args() -> Args {
    let xs: Vec<String> = env::args().collect();
    let mut a = Args { items: 500_000, capacity: 5_000_000, repeats: 5, out: "systems_results_fair.csv".into() };
    let mut i = 1;
    while i < xs.len() {
        match xs[i].as_str() {
            "--throughput-items" => { i += 1; a.items = xs[i].parse().unwrap(); }
            "--capacity-items" => { i += 1; a.capacity = xs[i].parse().unwrap(); }
            "--repeats" => { i += 1; a.repeats = xs[i].parse().unwrap(); }
            "--out" => { i += 1; a.out = xs[i].clone(); }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }
    a
}

fn bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut v = vec![0u8; len]; let mut r = Rng::new(seed);
    let mut cs = v.chunks_exact_mut(8);
    for c in &mut cs { c.copy_from_slice(&r.u64().to_le_bytes()); }
    for b in cs.into_remainder() { *b = r.u64() as u8; }
    v
}
fn floats(len: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0f32; len]; let mut r = Rng::new(seed);
    for x in &mut v { *x = r.f32(); } v
}
fn candidates(items: usize, n: usize, seed: u64) -> Vec<usize> {
    let mut r = Rng::new(seed); (0..n).map(|_| (r.u64() as usize) % items).collect()
}
fn programs(n: usize, seed: u64) -> Vec<Program> {
    let mut r = Rng::new(seed); let mut out = Vec::new();
    for p in 0..n {
        let mut coord = [0u16; UNARY];
        for t in 0..UNARY { coord[t] = ((t * 73 + p * 37 + 19) % D) as u16; }
        let mut unary = [[0f32; BINS]; UNARY];
        for t in 0..UNARY { for b in 0..BINS { unary[t][b] = r.f32(); } }
        let mut pair = [[0f32; BINS * BINS]; PAIRS];
        for m in 0..PAIRS { for b in 0..BINS*BINS { pair[m][b] = r.f32(); } }
        out.push(Program { coord, unary, pair, intercept: r.f32() });
    }
    out
}

#[inline(always)] fn get_item(v: &[u8], item: usize, c: usize) -> u8 {
    let b = unsafe { *v.get_unchecked(item * PACKED_BYTES + (c >> 1)) };
    if c & 1 == 0 { b & 15 } else { b >> 4 }
}
#[inline(always)] fn get_col(v: &[u8], items: usize, item: usize, c: usize) -> u8 {
    let stride = (items + 1) >> 1;
    let b = unsafe { *v.get_unchecked(c * stride + (item >> 1)) };
    if item & 1 == 0 { b & 15 } else { b >> 4 }
}
fn transpose(item_major: &[u8], items: usize) -> Vec<u8> {
    let stride = (items + 1) >> 1; let mut out = vec![0u8; D * stride];
    for item in 0..items {
        for c in 0..D {
            let q = get_item(item_major, item, c); let k = c * stride + (item >> 1);
            if item & 1 == 0 { out[k] = (out[k] & 0xf0) | q; } else { out[k] = (out[k] & 0x0f) | (q << 4); }
        }
    }
    out
}

#[inline(always)] fn rsa_item(v: &[u8], item: usize, p: &Program) -> f32 {
    let mut s = p.intercept; let mut q4 = [0u8; 4];
    for t in 0..UNARY {
        let q = get_item(v, item, p.coord[t] as usize); if t < 4 { q4[t] = q; }
        s += unsafe { *p.unary[t].get_unchecked(q as usize) };
    }
    s += unsafe { *p.pair[0].get_unchecked(q4[0] as usize * BINS + q4[1] as usize) };
    s += unsafe { *p.pair[1].get_unchecked(q4[2] as usize * BINS + q4[3] as usize) };
    s
}
#[inline(always)] fn rsa_col(v: &[u8], items: usize, item: usize, p: &Program) -> f32 {
    let mut s = p.intercept; let mut q4 = [0u8; 4];
    for t in 0..UNARY {
        let q = get_col(v, items, item, p.coord[t] as usize); if t < 4 { q4[t] = q; }
        s += unsafe { *p.unary[t].get_unchecked(q as usize) };
    }
    s += unsafe { *p.pair[0].get_unchecked(q4[0] as usize * BINS + q4[1] as usize) };
    s += unsafe { *p.pair[1].get_unchecked(q4[2] as usize * BINS + q4[3] as usize) };
    s
}
fn rsa_item_multi(v: &[u8], idx: &[usize], ps: &[Program]) -> f64 {
    let mut z = 0f64; for &i in idx { for p in ps { z += rsa_item(v, i, p) as f64; } } z
}
fn rsa_col_multi(v: &[u8], items: usize, idx: &[usize], ps: &[Program]) -> f64 {
    let mut z = 0f64; for &i in idx { for p in ps { z += rsa_col(v, items, i, p) as f64; } } z
}
fn linear_multi(x: &[f32], idx: &[usize], w: &[f32], np: usize) -> f64 {
    let mut z = 0f64;
    for &i in idx {
        let base = i * D; let mut a = [0f32; MAX_P];
        for j in 0..D {
            let q = unsafe { *x.get_unchecked(base + j) };
            for p in 0..np { a[p] += q * unsafe { *w.get_unchecked(p * D + j) }; }
        }
        for p in 0..np { z += a[p] as f64; }
    }
    z
}
fn pq_multi(code: &[u8], idx: &[usize], lut: &[f32], np: usize) -> f64 {
    let mut z = 0f64;
    for &i in idx {
        let base = i * PQ_M; let mut a = [0f32; MAX_P];
        for m in 0..PQ_M {
            let q = unsafe { *code.get_unchecked(base + m) } as usize;
            for p in 0..np { a[p] += unsafe { *lut.get_unchecked((p * PQ_M + m) * 256 + q) }; }
        }
        for p in 0..np { z += a[p] as f64; }
    }
    z
}
fn bench<F: FnMut() -> f64>(mut f: F, repeats: usize) -> (f64, f64) {
    black_box(f()); let mut ts = Vec::new(); let mut z = 0f64;
    for _ in 0..repeats { let t = Instant::now(); z = black_box(f()); ts.push(t.elapsed().as_secs_f64()); }
    ts.sort_by(|a,b| a.partial_cmp(b).unwrap()); (ts[ts.len()/2], z)
}
fn row(w: &mut BufWriter<File>, repr: &str, layout: &str, items: usize, n: usize, p: usize, bpi: usize, sec: f64, z: f64) {
    let ips = n as f64 / sec;
    writeln!(w, "{},{},{},{},{},{},{:.6},{:.3},{:.3},{:.6},{:.6}", repr,layout,items,n,p,bpi,sec,sec*1e9/n as f64,ips/1e6,ips*bpi as f64/1e9,z).unwrap();
}

fn main() {
    let a = args(); assert!(a.items >= 5_000);
    println!("capacity @ {} items: FP32 {:.3} GB | RSA4 {:.3} GB | PQ64 {:.3} GB", a.capacity, a.capacity as f64*1536.0/1e9, a.capacity as f64*192.0/1e9, a.capacity as f64*64.0/1e9);
    let mut out = BufWriter::new(File::create(&a.out).unwrap());
    writeln!(out,"representation,layout,catalog_items,candidates,predicates,bytes_per_item,seconds,ns_per_candidate,million_candidates_per_s,logical_gb_per_s,checksum").unwrap();
    let counts: Vec<usize> = [5000usize,20000,100000].into_iter().map(|n| n.min(a.items)).collect();
    let pcs = [1usize,2,4,8];

    println!("FP32 random resident catalog {:.3} GB", a.items as f64*1536.0/1e9);
    let x = floats(a.items*D, 101); let w = floats(MAX_P*D, 102);
    for &p in &pcs { for &n in &counts { let idx=candidates(a.items,n,1000+n as u64+p as u64); let (s,z)=bench(||linear_multi(&x,&idx,&w,p),a.repeats); row(&mut out,"fp32_linear","item_major_fused",a.items,n,p,1536,s,z); }}
    drop(x);

    println!("PQ64 random resident catalog {:.3} GB", a.items as f64*64.0/1e9);
    let pq=bytes(a.items*PQ_M,201); let lut=floats(MAX_P*PQ_M*256,202);
    for &p in &pcs { for &n in &counts { let idx=candidates(a.items,n,2000+n as u64+p as u64); let (s,z)=bench(||pq_multi(&pq,&idx,&lut,p),a.repeats); row(&mut out,"pq64_lut_head","item_major_fused",a.items,n,p,64,s,z); }}
    drop(pq);

    println!("RSA4 random packed catalog {:.3} GB", a.items as f64*192.0/1e9);
    let packed=bytes(a.items*PACKED_BYTES,301); let col=transpose(&packed,a.items);
    for &p in &pcs {
        let ps=programs(p,302+p as u64);
        for &n in &counts {
            let idx=candidates(a.items,n,3000+n as u64+p as u64);
            let (s,z)=bench(||rsa_item_multi(&packed,&idx,&ps),a.repeats); row(&mut out,"rsa4_f32","item_major_random_coords",a.items,n,p,192,s,z);
            let (s,z)=bench(||rsa_col_multi(&col,a.items,&idx,&ps),a.repeats); row(&mut out,"rsa4_f32","coordinate_major_random_coords",a.items,n,p,192,s,z);
        }
    }
    out.flush().unwrap(); println!("wrote {}",a.out);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn layouts_and_scores_match() {
        let n=31; let v=bytes(n*PACKED_BYTES,9); let c=transpose(&v,n); let p=programs(1,10).remove(0);
        for i in 0..n { for j in 0..D { assert_eq!(get_item(&v,i,j),get_col(&c,n,i,j)); } assert!((rsa_item(&v,i,&p)-rsa_col(&c,n,i,&p)).abs()<1e-5); }
    }
}
