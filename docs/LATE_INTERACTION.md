# ColBERT and MUVERA: first controlled baselines

This addition starts from `codex/ras-review-experiments` (PR #1, `e795f3a`).
It leaves historical paper results unchanged. No real ColBERT/fashion result is
claimed until the model run completes.

[Run in Colab](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_late_interaction_colab.ipynb).
Choose a GPU runtime and run all cells. The output ZIP contains raw rankings,
metrics, representations and replay information.

## Questions the experiment separates

1. **Matching quality within a shared pool.** MiniLM selects up to 5,000 held-out
   products, then the existing exact filters apply. Compare dense ordering,
   supervised Binary1-LS2-int4 scores, supervised FP32 scores, and pretrained
   ColBERT MaxSim on those same candidates. Recall is conditional on this pool.
2. **Retrieval quality over the held-out catalogue.** Compare full dense scan,
   exact ColBERT scan, and MUVERA candidate generation followed by exact MaxSim.
   Recall is relative to all teacher-relevant products satisfying exact filters.
3. **Approximation fidelity.** Measure MUVERA candidate coverage and final top-K
   overlap against exact ColBERT, independently of the teacher relevance metric.

The full corpus is the **held-out 35% split**, not the whole 44k source dataset.
The split, fit-derived teacher thresholds, vocabulary, title input, query
construction and binary/linear calibration follow the existing strict protocol.
The ColBERT checkpoint uses upstream query/document tokenizers, token markers,
query augmentation, and document punctuation masking.

## Baselines and interpretation

| Method | Retrieval/scoring | Supervision |
|---|---|---|
| `dense` | Original MiniLM dot product | Pretrained MiniLM |
| `binary_predicates` | Existing compiled scores, composed over shared pool | Fashion CLIP-derived fit labels |
| `linear_predicates` | FP32 head scores, composed over shared pool | Same fashion labels |
| `colbert_exact` | Exhaustive token MaxSim in the declared scope | Pretrained ColBERTv2 |
| `muvera_flat_maxsim` | Exhaustive FDE inner product, global top-B, exact filtering, MaxSim | Same ColBERT vectors |
| `muvera_hnsw_maxsim` | Faiss HNSW FDE inner product, global top-B, exact filtering, MaxSim | Same ColBERT vectors |

The flat FDE arm isolates representation approximation from ANN approximation.
It scans the entire FDE matrix and **is not an ANN speed claim**. HNSW is an
optional additional arm; flat remains present as its control. HNSW efSearch is
at least the returned candidate budget and its actual setting is recorded.
The same FDE seed/maps are used for queries and documents. FDEs are never
cosine-normalized. Original ColBERT vectors remain available for rescoring.

This is an independent NumPy reference construction of MUVERA's SimHash
partitions, query sums, document means, nearest-Hamming-token filling of empty
document partitions, and optional final CountSketch. Ordinary binary bucket IDs
replace Gray-code bucket IDs (a relabeling). NumPy random draws are not bitwise
compatible with Google's code. Repetition scores are averaged. No PQ compression,
DiskANN integration, PLAID implementation, or reproduction of the paper's
optimized system is claimed. The original C++ reference is
[Google graph-mining](https://github.com/google/graph-mining/tree/main/sketching/point_cloud);
see also the [MUVERA paper](https://arxiv.org/abs/2405.19504) and
[upstream ColBERT](https://github.com/stanford-futuredata/ColBERT).

## Run

From the repository root, use Python 3.10+ and install:

```bash
pip install -e '.[dev,benchmark,late-interaction]'
```

A small execution-only test needs neither model weights nor torch:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.late_interaction_baselines \
  --synthetic --output-dir results/late_smoke \
  --queries 4 --k 10 --candidates 20 1000 \
  --repetitions 2 --partition-bits 2 --fde-dim 128 --backend hnsw
```

For the initial fashion run:

```bash
OPENBLAS_NUM_THREADS=1 python -m experiments.late_interaction_baselines \
  --output-dir results/late_fashion_seed7 \
  --queries 30 --k 50 --candidates 100 500 1000 \
  --repetitions 8 --partition-bits 4 --fde-dim 4096 --backend hnsw
```

This encodes the existing public fashion dataset with MiniLM and an image CLIP
teacher, fits binary and FP32 semantic heads on fit rows only, and encodes
held-out titles and queries with ColBERT. The GPU accelerates encoding. Search
and reference MaxSim are CPU computations. The resolved checkpoint snapshot and
library versions are recorded. `--revision` can pin a Hugging Face model commit.
CPU-only upstream ColBERT may compile its own C++ scoring extension.

For larger runs, use `--queries 200`, repeat `--seed 7/17/27` and several
`--fde-seed` values in separate directories. Tune dimensions, repetitions and
candidate budgets on a development protocol before treating held-out results as
confirmatory. The initial 30-query run is exploratory.

Repeated identical runs verify input checksums and reuse prepared embeddings;
evaluation restarts. A changed configuration or source implementation requires a
new output directory. Interrupted query evaluation leaves a partial
`per_query.csv`; only a finished run writes a complete summary and ZIP.

## Metrics and artifacts

- `per_query.csv`: explicit scope, candidate/scored/returned counts, binary
  teacher recall, precision@K, binary nDCG@K, fill rate, negation/positive-count
  slices, approximation recall, and timing scope. Missing result slots count as
  nonrelevant for precision@K. Empty-truth queries stay in precision/fill metrics;
  recall and nDCG are undefined and excluded only from those averages.
- `summary.csv`: query-level bootstrap intervals (repetitions are averaged first)
  and observed timing percentiles. These intervals condition on the reused
  catalogue/vocabulary; they are not independent user or domain samples.
- `rankings.json`: actual result row IDs/product IDs and MUVERA candidate IDs.
- `documents.npz`, `queries.npz`: shared ragged ColBERT token matrices, without
  document padding; retained unchanged for all exact rescoring.
- `document_fde.npy`, `fde_parameters.npz`, optional `fde_hnsw.faiss`: actual
  encodings, random maps, and serialized ANN index. `MuveraFDE.load` restores maps.
- `prepared.npz`, `metadata.json`, `queries.json`, `fit_replay.npz`,
  `calibrations.json`, binary index/programs: row identities, splits, thresholds,
  labels, dense vectors, learned heads and composed-score inputs.
- `memory.json`: representation array bytes and serialized index size. HNSW
  serialization already contains its own FDE copy; avoid double-counting it in a
  hypothetical deployment. The prototype intentionally retains the flat matrix
  for the control. This is not measured process RSS.
- `encoding.json`, `environment.json`, `request.json`, `scope.json`,
  `checksums.json`, `late_interaction_results.zip`: provenance and declared limits.

## Limits to keep visible

- RSA is supervised for eight named image-teacher concepts. Pretrained ColBERT is
  a title-only relevance model. A win for either does not isolate architecture.
  Same-supervision fine-tuning and human judgments are subsequent experiments.
- Negation remains in natural-language queries; all methods receive the same
  exact metadata filters. MUVERA applies filters after its global candidate
  budget, so selective filters can reduce fill/recall. This harness does not
  implement native filtered HNSW or RSA inside MUVERA traversal.
- Full-corpus relevance here means satisfying a conjunction of teacher-defined
  concepts plus exact metadata. It is not an independent human judgment of the
  complete query/product relationship.
- Timings exclude token-model encoding, reported separately as batched time.
  Shared-pool timings also exclude initial pool generation. Binary/FP32 scores
  are materialized offline: their measured times cover composition, not native
  predicate execution. Do not compare these milliseconds with paper Rust times.
- Synthetic mode validates execution and invariants only. Its output is marked
  synthetic in every metric row and is not fashion or ColBERT evidence.
