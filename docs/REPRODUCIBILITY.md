# Reproducing the paper

This repository separates three kinds of evidence so that latency and quality claims are not mixed accidentally:

1. **Semantic quality**: how well compiled predicates allocate a candidate budget.
2. **Scoring microkernels**: how quickly learned item codes and programs can be evaluated in isolation.
3. **Reviewer-controlled live semantic HNSW**: graph traversal with real Binary1-LS2-int4 predicate execution inside the timed loop, compared with selectivity-aware over-fetch on the same graph.

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

## 2. Native finalist assets and microkernels

```bash
python -m experiments.export_native_finalists \
  --config configs/binary_bbq.yaml \
  --out-dir results/native_finalists_first_seed
```

The export contains real first-split held-out representations and learned programs. For D=384, Binary1-LS2-int4 uses a 56-byte item payload and about a 216-byte predicate scoring payload.

PQ64 storage has two distinct scopes. The native executor consumes a materialized 64×256 FP32 LUT, about 65.5 KB per active predicate. Persistent concept storage can instead retain the 1,548-byte FP32 linear head plus a single shared ~0.393 MB PQ codebook and materialize the LUT on activation. The paper does not treat 65.5 KB as mandatory persistent state for every concept.

## 3. Reviewer-controlled semantic-HNSW benchmark

The preferred systems entry point is:

```bash
python -m experiments.semantic_hnsw_reviewer_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_reviewer \
  --queries 1000
```

Public Colab:

```text
notebooks/rsa_semantic_hnsw_reviewer_colab.ipynb
```

Default paper protocol:

```text
predicate sets:       3 predetermined sets, 3 active predicates each
queries:              1000 per predicate set
K:                    50
ef:                   128
M:                    24
ef construction:      200
eligible fractions:   50%, 20%, 10%, 5%, 2%
over-fetch budgets:   .75, 1, 1.5, 2, 3, 4, 6, 8 × K/selectivity
recall tolerance:     .005
```

The focused exporter `experiments/export_hnsw_assets.py` writes only the assets needed by this benchmark and prints explicit progress stages.

The optimized harness:

1. exports/reuses the strict held-out HNSW assets;
2. verifies MiniLM retrieval vectors are unit-normalized;
3. derives all semantic gates for one predicate set;
4. builds `semantic_hnsw_reviewer` once for that set;
5. builds HNSW once, not once per over-fetch point;
6. materializes semantic scores once;
7. computes all 384-D dense query-to-item scores once per query;
8. sweeps all gates and all over-fetch budgets in the same process;
9. records raw rows, aggregate summaries, matched-recall comparisons, predicate-cost estimates, and environment metadata.

Expected outputs:

```text
results/semantic_hnsw_reviewer/
  assets/
  run_office_minimal_not_sporty.csv
  run_elegant_quiet_not_chunky.csv
  run_retro_relaxed_not_office.csv
  gates.csv
  raw.csv
  summary.csv
  same_run_pairs.csv
  matched_recall.csv
  environment.json
```

### Traversal recall definition

`Traversal Recall@50` is recall against **brute-force dense top-K among items passing the same compiled semantic predicate**. It measures ANN execution fidelity. It is not end-to-end semantic relevance.

`live_matches_materialized` compares live predicate execution with the same custom traversal using precomputed semantic scores. Exact ID parity is a correctness check, not a relevance result.

### Completed 2× result

The completed reviewer run swept `.75, 1, 1.5, 2 × K/selectivity`. No over-fetch point reached live traversal recall within `.005` in any of the 15 predicate-set × selectivity conditions. The aggregate at the largest tested budget is checked in at:

```text
paper/icml/data/semantic_hnsw_reviewer_2x_summary.csv
```

The current harness extends the sweep through `3, 4, 6, 8×` to find a genuine matched-recall point. Do not infer or report those results before running the extended benchmark.

### Predicate-cost measurement

`approx_ns_per_predicate_eval` is computed from the latency difference between:

```text
semantic_hnsw_live - custom_hnsw_materialized
```

divided by the number of live predicate invocations. It is an incremental end-to-end traversal estimate, not a standalone instruction microbenchmark. In the completed run it is approximately 96–134 ns/predicate. `dot_cost.rs` measures the same scalar 384-D dot kernel used by the custom traversal; on the recorded Intel Xeon 2.20 GHz run it was 617.155 ns/evaluation.

## 4. Persistent memory accounting

Memory numbers used by current code come from:

```python
from ras.accounting import METHOD_FOOTPRINTS, memory_rows
```

For N items and C concepts:

```text
persistent payload = N * item_bytes + C * stored_program_bytes + shared_bytes
```

`MethodFootprint.active_program_bytes` records activation-time state separately when it differs from stored state. For PQ64 this is 65,548 active bytes versus 1,548 persistent bytes per concept, plus one shared 393,216-byte codebook.

The illustrative 5M-item / 100k-concept table excludes HNSW graph edges, allocator/container overhead, filesystem metadata, and offline teacher/model weights.

## 5. Interactive demo

```bash
pip install -e ".[demo,benchmark]"
```

Then use:

```text
notebooks/rsa_fashion_search_demo_colab.ipynb
```

or call `demos.fashion_app.prepare_demo(...)` and `build_app(...)` directly. The demo intentionally separates Python UI latency from Rust systems measurements.

## 6. Build the paper

```bash
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The manuscript distinguishes candidate-side quality, microkernel latency, traversal latency, persistent program state, and activation-time program state. Do not reinterpret one scope as another.

## Current limits

The current evidence is single-domain and research-prototype scale. The reviewer HNSW graph is only the strict held-out split (~15.4k items), single-threaded and resident-memory. It does not establish million-scale QPS/core, production p99, distributed RPC cost, update/tombstone behavior, or large-vocabulary cache behavior. The quality benchmark currently learns eight latent concepts.
