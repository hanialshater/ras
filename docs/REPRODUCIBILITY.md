# Reproducing the paper

This repository separates three kinds of evidence so that latency and quality claims are not mixed accidentally:

1. **Semantic quality**: how well compiled predicates allocate a candidate budget.
2. **Scoring microkernels**: how quickly learned item codes and programs can be evaluated in isolation.
3. **Live semantic HNSW**: graph traversal with real Binary1-LS2-int4 predicate execution inside the timed loop.

The paper reports each scope explicitly.

## Environment

Recommended Python setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,benchmark,demo]"
```

Rust stable is required for native experiments:

```bash
rustc --version
cargo --version
```

Run unit tests before experiments:

```bash
pytest -q
cargo test --release --manifest-path rust/semantic_engine/Cargo.toml
```

## 1. Main semantic-quality benchmark

```bash
python -m experiments.binary_bbq_predicates \
  --config configs/binary_bbq.yaml
```

This uses the 44,072-product fashion source dataset, MiniLM title embeddings as the search-side representation, CLIP image semantics as an independent teacher, strict fit/calibration/test splits, and the compound-query protocol described in the paper.

Important scope: recall and purity are measured **within the exact-filtered top-5,000 ANN candidate pool**. They are not global catalog recall.

## 2. Export native assets

```bash
python -m experiments.export_native_finalists \
  --config configs/binary_bbq.yaml \
  --out-dir results/native_finalists_first_seed
```

The export contains real first-split held-out representations and learned programs, including:

```text
sidecar_index/
  bits.u8
  corrections.f32
  centroid.f32
  projection.f32
  manifest.json

sidecar_programs/<concept>/
  bitplanes.u8
  scalars.f32
  manifest.json
```

For D=384, Binary1-LS2-int4 uses a 56-byte item payload and about a 216-byte predicate scoring payload.

## 3. Fair live semantic-HNSW benchmark

The preferred systems entry point is:

```bash
python -m experiments.semantic_hnsw_live_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_live_fair
```

Defaults:

```text
positive predicates: minimalist, office_appropriate
negative predicates: technical_sporty
queries:             100
K:                   50
ef:                  128
M:                   24
ef construction:     200
eligible fractions:  50%, 20%, 10%, 5%, 2%
```

The harness:

1. exports the strict held-out assets if they are absent;
2. verifies the MiniLM retrieval vectors are unit-normalized;
3. derives semantic thresholds that produce the requested eligibility fractions;
4. builds `semantic_hnsw_live`;
5. runs four methods on the same graph and normalized-dot geometry;
6. records raw query rows, aggregate summaries, fairness comparisons, and environment metadata.

The four methods are:

```text
hnsw_postfilter_materialized
hnsw_filtered_materialized
custom_hnsw_materialized
semantic_hnsw_live
```

`semantic_hnsw_live` executes the actual 56-byte item code × 216-byte predicate program inside the timed traversal. The materialized custom method uses the same traversal but free precomputed semantic scores, so their latency difference isolates the program execution overhead.

Expected output files:

```text
results/semantic_hnsw_live_fair/
  assets/
  gates.csv
  run_0.500.csv
  run_0.200.csv
  run_0.100.csv
  run_0.050.csv
  run_0.020.csv
  raw.csv
  summary.csv
  fairness.csv
  environment.json
```

`environment.json` records the repository commit, Python version, Rust version, CPU model, query parameters, and normalization statistics.

The checked-in paper summary is:

```text
paper/icml/data/semantic_hnsw_live_fair_full.csv
```

It should be treated as a recorded result, not silently regenerated during paper compilation.

## 4. Memory accounting

Memory numbers used in the paper and demo come from one source of truth:

```python
from ras.accounting import METHOD_FOOTPRINTS, memory_rows
```

For N items and C concepts:

```text
payload = N * item_bytes + C * program_bytes
```

The illustrative 5M-item / 100k-concept table excludes HNSW graph edges, allocator/container overhead, filesystem metadata, and offline teacher/model weights.

## 5. Interactive demo

```bash
pip install -e ".[demo,benchmark]"
```

Then use the Colab notebook:

```text
notebooks/rsa_fashion_search_demo_colab.ipynb
```

or call `demos.fashion_app.prepare_demo(...)` and `build_app(...)` directly.

The demo intentionally separates **Python UI latency** from the **Rust HNSW systems latency** displayed in its systems panel.

## 6. Build the paper

```bash
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The systems tables distinguish:

- candidate-side microkernel latency;
- live HNSW search-stage latency;
- theoretical item/program memory payloads.

Do not reinterpret microkernel latency as end-to-end search latency.

## Current limits

The current evidence is single-domain and research-prototype scale. The live-HNSW benchmark is single-threaded and resident-memory. It does not establish production p99, distributed RPC cost, update/tombstone behavior, or large-vocabulary cache behavior. The 100k-concept memory table is representation arithmetic; the quality benchmark currently learns eight latent concepts.
