# RSA Rust semantic execution benchmark

CPU microbenchmarks for the systems claim in the Random Semantic Algebra paper.

The headline benchmark is `src/bin/fair.rs`. It compares:

- a fused 384-D FP32 linear semantic head on randomized FP32 catalogs;
- a PQ64-style compiled LUT head (64 B/item);
- packed 4-bit RSA with 24 unary + 2 pair LUTs (192 B/item), using randomized selected coordinates;
- item-major versus coordinate-major packed layouts.

The default run reports memory capacity for a 5M-item catalog but uses a 500k-item resident catalog for throughput so it is safe on standard Colab RAM. Increase `--throughput-items` on a high-memory host.

```bash
cargo test --release --manifest-path rust/semantic_engine/Cargo.toml
cargo run --release --manifest-path rust/semantic_engine/Cargo.toml --bin fair -- \
  --throughput-items 500000 \
  --capacity-items 5000000 \
  --repeats 5 \
  --out systems_results_fair.csv
```

The benchmark evaluates random post-ANN candidate pools of 5k, 20k and 100k items and compound queries with 1, 2, 4 and 8 semantic predicates. FP32 and PQ baselines fuse heads so each item representation is loaded once per compound query.

`src/main.rs` is the earlier exploratory executor benchmark containing additional experiments with fused coordinate unions and i16 LUTs. It is retained because those negative/neutral results are useful: on the first CI run, explicit union fusion and i16 tables did not outperform the simpler fixed F32 executor.

This is a throughput/memory benchmark, not a quality benchmark. Quality is measured separately in `experiments/reviewer_baselines.py`.
