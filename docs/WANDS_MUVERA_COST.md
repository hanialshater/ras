# WANDS MUVERA fidelity and serving costs

[Open the Colab notebook](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_wands_muvera_cost_colab.ipynb).

This follows the completed WANDS DenseOn/LateOn quality comparison. It preserves
the saved corpus, query order, qrels, input limits and model revisions. It does
not retrain the models or regenerate judgments. The quality notebook saved
scores, not document embeddings: this benchmark re-encodes documents once and
checks numerical parity on five seeded queries against every corpus document.
Near-tie top-k changes are reported separately from the score tolerance.

## Comparisons

| Pipeline | Candidate search | Final scoring |
|---|---|---|
| DenseOn | Exact normalized dot over corpus | Same dot product |
| LateOn | Exhaustive token search | FP32 MaxSim |
| BM25 | Full-corpus BM25 | BM25 |
| DenseOn + BM25 + CE | Live union of both top-N lists | GTE ModernBERT cross-encoder |
| MUVERA flat | Exact FDE inner product | FP32 LateOn MaxSim |
| MUVERA HNSW | Approximate FDE inner product | FP32 LateOn MaxSim |

The CE candidate pool must match the original quality run. Its latency includes
both candidate generators and joint pair scoring. It is a shared-pool pipeline;
the other cost arms search the full corpus. No filters are applied to MUVERA.

Default sweep: 4096/8192 FDE dimensions, 8 repetitions, 4 partition bits,
seed 7; candidate budgets 100, 500, 1000, 2000, 5000, 10000; k=10. HNSW uses
M=32, efConstruction=200, efSearch=max(200, candidate budget). The explicit
efSearch rule is part of the measured configuration, not an independently tuned
HNSW optimum. One seed is exploratory; repeat with other seeds/new directories
before making robust configuration claims.

## What MUVERA implementation is measured?

`ras.late_interaction.MuveraFDE` is the repository's transparent NumPy reference:
SimHash buckets, query sums, document means, nearest-Hamming empty-bucket filling,
and optional final CountSketch. Actual random maps are saved. FDE vectors are
**not cosine-normalized**. This is not a reproduction of Google's optimized
C++/DiskANN/PQ system or its paper speedups. See the
[Google research description](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
and [construction source](https://github.com/google/graph-mining/tree/main/sketching/point_cloud).

## Relevance versus approximation

`fidelity.csv` measures overlap with the completed exact LateOn top-k. Flat FDE
search exposes representation loss; HNSW-versus-flat FDE candidate overlap exposes
additional ANN effects. Per-query candidate recall and fill rates are retained.
`muvera_quality.csv` separately reports the same human-label WANDS metrics, with
paired nDCG differences against exact LateOn and full-corpus denominators.
Offline fidelity reranking uses saved exact scores. This avoids recomputing the
same quality calculation repeatedly; **timed search never reads those scores**.

Targets 0.90/0.95/0.99 are selected from achieved mean fidelity in this run, choosing
the measured lowest-p50 budget among those passing. They are exploratory,
evaluation-set selections, not held-out guarantees. A target can be unreached.
Do not interpret a high ColBERT overlap as semantic relevance.

## Latency and memory scope

Encoders use the requested GPU (or explicit CPU); dense search, FDE construction,
FAISS flat/HNSW and token MaxSim run on CPU with two threads by default. MaxSim is
the chunked NumPy scorer, parity checked against the native PyLate quality scores.
This is an implementation comparison, not the fastest possible ColBERT serving
engine. GPU token scoring, IVF, binary quantization and concurrent throughput
are future experiments.

Each serving pipeline starts a separate process. Model/index loading is recorded
as startup time. Two warmups per configuration precede 30 seeded queries with
three repeats, randomized across queries and budgets. CUDA is synchronized.
Stage p50 values, total p50/p95/p99 and per-request samples are exported. Component
percentiles do not necessarily sum to total percentiles. `excluding_encoding_ms`
is computed per request before aggregation. Sequential QPS is the reciprocal of
mean request time, not saturated concurrent throughput. P99 from 90 requests is
exploratory.

Token and dense arrays use read-only memory mapping. Warmups do not guarantee
every token page remains resident; page reads/evictions affect latency and are
part of this setup. Run on local disk, not Drive, and compare identical hardware.
RSS before load, after requests, sampled peak and process lifetime peak are
reported per pipeline. RSS includes resident mapped pages, text, models,
Python/native allocators and metadata; it excludes CUDA. CUDA allocated/reserved
peaks come from PyTorch's allocator, not total physical GPU utilization. CE also
holds candidate IDs for validation. Model tensor bytes are a payload estimate,
not a substitute for measured memory.

## Build and storage

Document encoding, shard serialization/packing, FDE transformation and FAISS
construction/serialization are measured separately. Build totals exclude model
downloads/loading, query encoding, source preparation, validation and copies to
Drive. Resumed encoding totals sum completed shard measurements, so hardware
must remain the same. BM25 construction/serialization is measured in the report
phase; worker reconstruction is included in startup, not request latency.

Serving disk bytes include metadata/text plus exactly the necessary corpus
representations. A FAISS flat/HNSW index already holds its FDE vectors; the
separate transformation matrix is not counted again. Both MUVERA pipelines also
need full-precision tokens for exact reranking. Model tensor payloads are listed
separately; checkpoint download directories, reference scores, cached queries,
offline rankings, logs, temporary shards and Drive backups are excluded. Metadata
currently includes all query texts for the harness, so counts are conservative
benchmark footprints rather than minimal deployment packages.

## Running and resuming

```bash
pip install -e '.[wands,wands-systems,dev]'
python -m experiments.wands_systems \
  --reference-dir /local/completed-wands-quality \
  --output-dir /local/wands-muvera-cost \
  --dimensions 4096 8192 --candidates 100 500 1000 2000 5000 10000 \
  --device cuda --threads 2
```

The Colab notebook stages the reference on local disk and backs up between
phases. On ordinary exceptions, its finally block also copies completed shards.
A runtime reset can lose progress since the previous backup; only completed
files are restored. No Drive synchronization runs during measured requests.
Document encoding, FDE transformation and index construction skip completed
artifacts. Fidelity and latency phases rerun intentionally; reruns overwrite
their reports. Use a new directory if the locked code, settings, dependencies,
hardware or reference changes. The original WANDS quality run is never modified.

Outputs: `fidelity.csv`, `muvera_quality.csv`, `latency_at_fidelity.csv`,
`latency.csv`, `memory.csv`, `storage_build.csv`, per-query fidelity/quality,
individual request samples, model settings, build components, `parity.json`,
`manifest.json`, and `scope.json`. The notebook exports a small report ZIP,
excluding large token arrays and indexes.

Tests include negative-token MaxSim parity, memory-mapped packing, full-budget
flat/HNSW recovery, inner-product index scoring, resume markers, reference guards,
live timing without score-cache access, cost reports and an opt-in real-checkpoint
test that runs each serving pipeline in a separate process on a tiny artificial
corpus. The latter checks compatibility, not relevance or performance quality.
