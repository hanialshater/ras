# Retrieval comparison v2

[Open the updated Colab notebook](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_late_interaction_colab.ipynb).

The runner is `python -m experiments.retrieval_comparison`. The v1 runner remains
available for historical reproduction. The new notebook uses a separate v2 output
directory and can import v1 prepared embeddings without replacing the old results.

## 1. Scoring quality: dot product, ColBERT and cross-encoder

Freeze one MiniLM top-5,000 pool per query, apply exact metadata filters, then
score the **same rows** with:

- `dense_dot_product`: original pretrained MiniLM query/title vectors, normalized
  before dot product. This is equivalent to cosine similarity for these vectors;
  it is not the supervised linear predicate model.
- `colbert_exact`: upstream ColBERT query/title tokens, exact sum-of-max scores.
- `cross_encoder`: joint query/title transformer scoring, default
  `cross-encoder/ms-marco-MiniLM-L6-v2`, raw logits. It does not create reusable
  document embeddings. This is a specific pretrained model comparison, not a
  controlled architecture study or an assumption that a cross-encoder is an oracle.
- `binary_predicates` and `linear_predicates`: supervised task-specific controls,
  with scores materialized during preparation.

All title/query content and exact filters are shared; models retain their own
native tokenizers and truncation settings. The cross-encoder defaults to 256 total
pair tokens, ColBERT to 32 query / 180 document tokens. Model snapshots, devices,
tensor bytes and downloaded snapshot file bytes are recorded in `models.json`.
Snapshot disk bytes count the files actually present, including alternate formats
if already cached; they are separate from corpus storage and are not a minimal
model deployment footprint.

`ranking_quality.csv` reports precision@K, binary nDCG@K, conditional pool recall
and fill rate. `pool_coverage.csv` reports pool truth / full eligible corpus truth.
`paired_quality_deltas.csv` bootstraps **per-query differences** (A minus B), after
averaging timing repetitions. Repeats never become independent quality samples.

The current labels are the existing CLIP image teacher, not human search relevance.
The learned predicates directly target these labels; pretrained models do not.
`judgment_pool.csv` exports candidate query/title pairs with an empty relevance
column for independent annotation. The runner does not yet ingest that annotation
file: do not interpret CE predictions as ground truth or claim customer relevance
from these teacher labels.

## 2. MUVERA approximation

Use the same ColBERT document/query embeddings and the full held-out corpus:

1. Exact ColBERT establishes a reference top-K after exact metadata filtering.
2. MUVERA Flat scans the FDE matrix by inner product, selects global top-B,
   applies metadata filters and rescoring with exact ColBERT MaxSim.
3. MUVERA HNSW replaces the FDE scan with Faiss HNSW; the remaining stages match.
4. Exact full-corpus MiniLM dot product provides the single-vector baseline.

`approximation.csv` reports ColBERT top-K overlap, candidate coverage of the
eligible ColBERT top-K, candidate survival after filtering, and fill rate.
`retrieval_quality.csv` separately reports teacher relevance against the full
eligible catalogue. It has a different recall denominator from the shared pool.

A budget of 100 can underfill K=50 when filtering removes candidates. There is
no hidden refill. `candidate_survival_rate` exposes this loss. Full candidate
budgets must recover exact ColBERT's ordering; tests cover Flat and HNSW.

`matched_fidelity.csv` chooses the fastest measured setting reaching each requested
**mean** top-K recall target (default .90/.95/.99). Unreached targets are labeled
`not_reached`, not extrapolated. These operating points are selected on the same
queries: validate chosen parameters on separate queries before confirmatory claims.
Do not equate equal ColBERT fidelity with equal human relevance.

## 3. Latency, storage and build time

The loop warms each query/method once, then randomizes method order during three
measured rounds. Every neural pipeline starts from query text: it includes query
encoding, exact filtering, pool/candidate search and final scoring/top-K.

`latency.csv` reports observed p50/p95/p99 for query encoding, pool generation,
FDE construction, FDE retrieval, scoring and total request time. Cross-encoder
pair tokenization and neural inference belong to `scoring_ms`. Scoring-only
numbers isolate the ranker; total time includes the dense candidate retriever for
shared-pool pipelines. CUDA operations are synchronized at stage boundaries.

Prepared query vectors and the shared quality pool are immutable for scoring, so
batched/single-query floating-point drift cannot change the ColBERT oracle. Live
query encoding is still measured per request. Online dense pool generation is timed again,
and its overlap with the frozen pool appears in `per_query.csv`. Small numeric
encoding drift cannot silently give different candidate sets to different rankers.
The total therefore measures a reference request with a frozen scoring pool; it is
not a live production service test. Check `runtime_pool_overlap` before interpreting.

Concurrency is **1**, model load is excluded, and no network request or queueing is
included. ColBERT MaxSim/dot product/HNSW run on CPU; encoders and cross-encoder
use their actual available device. This is not PLAID or an optimized GPU MaxSim
comparison. Few queries cannot establish production p99 or target-QPS behavior.
Predicate timing excludes actual predicate execution and is explicitly labeled.

- `storage_components.csv`: actual serialized corpus artifact sizes.
- `storage.csv`: per-pipeline corpus disk footprints, including required shared
  titles/metadata. The cross-encoder pipeline includes the dense candidate index.
  HNSW serialization **already includes FDE vectors**; its plan does not add the
  standalone Flat matrix. Models, queries and research replay files are excluded.
- `models.json`: model tensor payload and cached snapshot disk bytes, separately.
- `memory.json`: document array payloads and whole-harness peak RSS. This includes
  all experimental arms and preparation, **not per-method serving RAM**. GPU memory
  and allocator overhead cannot be inferred from tensor payloads.
- `build_time.csv`: model load, dense document encoding, ColBERT document encoding,
  FDE map construction, FDE document transformation, HNSW build and serialization.
  Exact scans have no graph construction. CE has no standalone document encoding.
- `build_totals.csv`: required corpus encoding + transformation + index construction
  per pipeline; excludes downloads, model load, serialization, teacher generation
  and predicate training. A dense-pool CE pipeline still pays dense index build.

Imported ColBERT encoding times come from the source run; the provenance column
marks this. Dense encoding is measured with a fresh held-out-title pass because
upstream vectors may have come from cache. Use fresh preparation on one machine
for a same-runtime build comparison. Fresh dense outputs and ColBERT queries are
checked against prepared vectors to catch accidental checkpoint changes.

## Commands

```bash
pip install -e '.[dev,benchmark,late-interaction]'
python -m experiments.retrieval_comparison \
  --output-dir results/retrieval_v2 --queries 30 --seed 7 \
  --k 50 --pool-size 5000 --candidates 100 500 1000 2000 5000 \
  --backend hnsw --warmup 1 --timing-repeats 3
```

To reuse a completed v1 preparation add `--prepared-from /path/to/v1/run`.
Preparation configuration, input checksums and model/row compatibility are checked;
new search settings are allowed. Do not use the source directory as output.

Use `--cross-encoder-checkpoint`, `--cross-encoder-revision`,
`--cross-encoder-maxlen` and `--cross-encoder-batch-size` to change the pair model;
use a new output directory. `--skip-cross-encoder` allows a retrieval-only run.

An execution-only smoke test requires no model downloads or torch:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m experiments.retrieval_comparison \
  --synthetic --output-dir /tmp/ras-comparison-smoke \
  --queries 4 --k 5 --pool-size 50 --candidates 10 1000 \
  --repetitions 2 --partition-bits 2 --fde-dim 32 --backend hnsw
```

Every synthetic row is marked. The cross-encoder is a deterministic score stub in
synthetic mode, not an executed neural model. `rankings.json` saves candidate,
scored and returned identities; the ZIP includes checksums and all reports.

## Matched modern backbone profile

[Open the ModernBERT Colab notebook](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_modern_backbones_colab.ipynb).
Use a fresh GPU runtime. The historical notebook and `--model-family legacy`
remain available.

| Role | Checkpoint | Backbone / output |
|---|---|---|
| Dot product | [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | ModernBERT-base, ~149M; one 768-dimensional vector |
| ColBERT | [lightonai/GTE-ModernColBERT-v1](https://huggingface.co/lightonai/GTE-ModernColBERT-v1) | Initialized from that GTE dense model; adds a trained 768→128 token projection |
| Cross-encoder | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | ModernBERT-base, ~149M; one joint query/title score |

This approximately matches backbone family and model capacity. It does **not**
match fine-tuning supervision: ModernColBERT uses MS MARCO/BGE-Gemma distillation;
the GTE dense and reranker models follow their respective GTE training recipes.
All three are English checkpoints. An architecture-only study still requires
training the three heads with controlled data/teacher/budget.

The profile loads all neural models in FP32 with SDPA and disables ModernBERT's
internal reference compilation consistently. Actual model type, hidden width,
layer/head counts, parameter counts and weight dtype are recorded in
`backbones.csv` and `models.json`. The run refuses mismatched architecture fields,
precision or parameter counts differing by more than 5%. Model revisions resolve
to recorded snapshots; dense/ColBERT snapshots are reused on prepared-input imports.

PyLate 1.6.0 performs ModernColBERT encoding, preserving the checkpoint's trained
projection, token markers, punctuation masking, normalization and query expansion.
The adapter rejects checkpoints without a saved Dense projection. It does not
pass ModernBERT through the legacy `colbert-ai` BERT wrapper or initialize a new
random token projection. Reference MaxSim and MUVERA still share identical saved
token matrices and CPU scoring.

```bash
pip install -e '.[dev,benchmark,modern-retrieval]'
python -m experiments.retrieval_comparison \
  --model-family modernbert-base \
  --output-dir results/modernbert_base_seed7 \
  --queries 30 --seed 7 --k 50 --pool-size 5000 \
  --candidates 100 500 1000 2000 5000 --backend hnsw
```

This selects `configs/modern_backbones.yaml`, the matched checkpoints, 48/300
ColBERT query/document limits and a 512-token CE pair limit. The saved ModernColBERT
configuration specifies query length 48 and disables query expansion; it takes
precedence over the model card's older 32-token description. Model tokenization and
truncation formats remain native, not identical token budgets. The model IDs are
fixed by this profile; incompatible overrides are rejected.

Dense item/query embeddings and the supervised predicate controls are rebuilt.
The source dataset, split seed, query-generation protocol and CLIP label definition
remain the same. **The pool is now selected by GTE-ModernBERT**, and results use
`shared_modernbert_pool` with a `model_family` column. All five rankers receive
identical candidates within this run, but the new pool is not identical to the
old MiniLM pool. Do not attribute historical-to-modern shared-pool changes solely
to the scorer; use pool coverage and full-corpus controls when comparing runs.

Legacy prepared inputs are explicitly rejected by the modern profile. Only a
matching completed modern run can be supplied through `--prepared-from`. The
notebook defaults to fresh encoding and a separate `ras_modernbert_base_seed7_v1`
Drive directory. Library pins are compatible with the existing Transformers 4.49
teacher path; downloaded wheel metadata was checked. Real neural execution still
requires the GPU notebook; synthetic validation does not load the checkpoints.
