use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

const D: usize = 384;

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let b = fs::read(path).expect("read items");
    assert_eq!(b.len() % 4, 0);
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

#[inline(always)]
fn dot(items: &[f32], a: usize, b: usize) -> f32 {
    let aa = a * D;
    let bb = b * D;
    let mut s = 0.0f32;
    for j in 0..D {
        s += unsafe { *items.get_unchecked(aa + j) } * unsafe { *items.get_unchecked(bb + j) };
    }
    s
}

fn main() {
    let xs: Vec<String> = env::args().collect();
    let mut items_path = PathBuf::from("results/native_finalists_first_seed/fp32_items.f32");
    let mut evals = 2_000_000usize;
    let mut warmup = 100_000usize;
    let mut i = 1usize;
    while i < xs.len() {
        match xs[i].as_str() {
            "--items" => { i += 1; items_path = PathBuf::from(&xs[i]); }
            "--evals" => { i += 1; evals = xs[i].parse().unwrap(); }
            "--warmup" => { i += 1; warmup = xs[i].parse().unwrap(); }
            "--help" | "-h" => {
                println!("dot_cost --items fp32_items.f32 --evals 2000000 --warmup 100000");
                return;
            }
            _ => panic!("unknown argument {}", xs[i]),
        }
        i += 1;
    }

    let items = read_f32(&items_path);
    assert_eq!(items.len() % D, 0);
    let n = items.len() / D;
    assert!(n >= 2);

    let mut acc = 0.0f32;
    for t in 0..warmup {
        let a = (t.wrapping_mul(9973)) % n;
        let b = (t.wrapping_mul(7919).wrapping_add(17)) % n;
        acc += dot(&items, a, b);
    }
    black_box(acc);

    let t0 = Instant::now();
    let mut acc2 = 0.0f32;
    for t in 0..evals {
        let a = (t.wrapping_mul(9973)) % n;
        let b = (t.wrapping_mul(7919).wrapping_add(17)) % n;
        acc2 += dot(&items, a, b);
    }
    black_box(acc2);
    let elapsed = t0.elapsed().as_secs_f64();
    let ns = elapsed * 1e9 / evals as f64;
    println!("items={} dim={} evals={} elapsed_s={:.6} ns_per_dot={:.3}", n, D, evals, elapsed, ns);
}
