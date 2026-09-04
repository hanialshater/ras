# RSA Rust semantic execution benchmark

CPU microbenchmark for the systems claim in the Random Semantic Algebra paper.
It compares a fused 384-D FP32 linear semantic head, a PQ64-style LUT head,
and several packed 4-bit RSA executors (item-major, coordinate-major, fused
multi-predicate plan, and i16 LUTs).

The default benchmark reports memory capacity for a 5M-item catalog but uses a
500k-item resident catalog for throughput so it is safe on standard Colab RAM.
Increase `--throughput-items` on a high-memory host.

```bash
cargo test --release
cargo run --release -- \
  --throughput-items 500000 \
  --capacity-items 5000000 \
  --repeats 5 \
  --out systems_results.csv
```

The benchmark evaluates random post-ANN candidate pools of 5k, 20k and 100k
items and compound queries with 1, 2, 4 and 8 semantic predicates. FP32 and
PQ baselines fuse heads so each item representation is loaded once per compound
query; RSA includes a union-of-coordinates fused plan with 50% coordinate
overlap to quantify the benefit of query compilation.

This is a throughput/memory benchmark, not a quality benchmark. Quality is
measured separately in the Python reviewer-baseline experiment.
