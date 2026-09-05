# Review follow-up experiments

Open [the Colab notebook](https://colab.research.google.com/github/hanialshater/ras/blob/codex/ras-review-experiments/notebooks/ras_review_experiments_colab.ipynb), select a GPU runtime, and run all cells. Drive checkpoints are enabled by default. The final cell downloads `ras_review_results.zip`.

## What this run resolves

- **Predicate-head precision:** PQ64-FP32 and PQ64-int4 share the exact same fitted codebook and item codes. Each gets its own scalar calibration on calibration rows. The int4 head is physically packed. Its 212-byte persistent payload includes weight offset/scale, intercept and calibration. The 65,548-byte active LUT remains separate.
- **Materialized scores:** save eight calibrated logits per item as real FP32 files (32 B/item). Both FP32-head and compiled-head tables are evaluated for quality. The native table path composes compiled logits inside the same HNSW traversal, rather than receiving a precomposed score for free. Native logits are regenerated with the identical Rust arithmetic, checked against the Python export, and saved; exact table/live eligibility and result parity are asserted. The precomposed-score control is still reported separately.
- **Controlled traversal:** one `search_custom` function implements live filtering, materialized filtering and dense over-fetch. Over-fetch sweeps returned candidate count and search effort separately. Methods are measured in randomized order for each query/gate/repetition. An optional library baseline is contextual only.

The run also exports corrected all-query purity, explicit recall denominators, deployed memory with existing FP32 vectors, and a ZIP of raw/replay evidence. It does not automatically replace any reported paper result.

## Run locally

```bash
pip install -e ".[dev,benchmark]"
python -u -m experiments.review_followup --output-dir results/review_followup
```

This needs model/data downloads for the real fashion run. For execution validation without model downloads:

```bash
python -u -m experiments.review_followup --synthetic \
  --output-dir results/review_smoke --hnsw-queries 4 --timing-repeats 1 \
  --overfetch-multipliers 1,2 --ef-multipliers 1,2
```

Synthetic output is not research evidence about fashion relevance.

## Artifacts and resumption

Each completed quality split gets a `complete.json` checkpoint. Restarting an identical run reuses completed splits and completed HNSW predicate sets. An output directory is bound to its full configuration and Git commit; use another directory after code/config changes. Interrupted current splits or predicate sets are restarted.

Key files:

- `per_query.csv`, `summary.csv`, `paired_deltas.csv`
- `seed_*/queries.json`, `predicate_metrics.csv`, `calibrators.csv`
- `seed_*/replay.npz`: exact fit/calibration/test row identities, teacher thresholds, test labels, query vectors, learned weights, PQ codebook/codes and calibrated logits
- `hnsw/raw.csv`, `summary.csv`, `same_run_pairs.csv`, `matched_recall.csv`
- `deployment_memory.csv`, `compiler_payloads.json`, `scope.json`
- `retrieval_embeddings.f32`, `teacher_scores.f32`, `source_arrays.json`: exact inputs for replay without model downloads
- `hnsw/run_*.graph.json`, `hnsw/run_*.logits.f32`, `hnsw/Cargo.lock`: exact native graph, native-rounded tables and dependencies
- `request.json`, `environment.json`, `checksums.json`
- `ras_review_results.zip` (excludes model/dataset caches)

The larger vocabulary sweep is representation arithmetic, not measured behavior of newly learned concepts. There are still eight learned semantic concepts. Timings are single-thread, resident-memory, warm-cache prototype measurements, not production p99.

## Serving compatibility

Programs now carry an encoder fingerprint and immutable versioned payload files. The manifest is published atomically. Python executors check the manifest revision and refresh changed programs; a scoring call snapshots its requested programs. Old program format v1 cannot establish encoder identity and must be recompiled. Item bits need not change when the encoder itself is unchanged.

The portable Rust sidecar and controlled Rust reviewer validate the program/index fingerprint and reads versioned payloads. Canonical `bitplanes.u8` and `scalars.f32` exports remain solely for historical benchmark compatibility; those legacy binaries assume immutable exported assets.

## Result interpretation

First assess PQ-int4's loss relative to PQ-FP32, then compare Binary1's quality, deployed memory and timing. Check whether a score table is preferable at small vocabulary sizes. For traversal, inspect the custom over-fetch frontier and require matched recall within the declared tolerance before interpreting latency ratios. Negative live-minus-materialized timing estimates remain visible rather than being clipped.

Zero-truth candidate pools count toward purity with zero hits; recall excludes undefined denominators. Empty exact-filtered candidate pools are recorded in query metadata and reported separately. Query-level bootstrap intervals describe this reused catalog/vocabulary; they do not create independent domains or users.
