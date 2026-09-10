# WANDS predicate representation study

[Open the Colab](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_wands_predicates_colab.ipynb).

This returns to RAS after the retrieval and official-FDE diagnostics. It asks whether better representations and local predicate heads reduce the **FP32 modelling gap**, then measures how much Binary1 and int4 compilation lose. It does not assume LateOn will win or that MUVERA is necessary.

## Comparisons

| Representation | Predicate head | Precision controls |
|---|---|---|
| all-MiniLM-L6-v2 | Global / local logistic regression | FP32 / Binary1 items / Binary1 items + int4 weights |
| DenseOn | Global / local logistic regression | Same three controls |
| Mean-pooled, normalized LateOn tokens | Global / local logistic regression | Same three controls |
| LateOn tokens | Learned `max_token(w · token) + bias` | Same three controls, with Binary1 per token |

Backbone checkpoints are pinned. All encoders remain frozen. This is a predicate experiment: the learned token-max head is **not** the pretrained LateOn query/document MaxSim retriever. The pooled LateOn control helps isolate retaining token structure, although head architecture and optimization also differ. MiniLM is the historical small baseline, not a size-matched backbone.

For local heads, KMeans fits only training products; K=8 and K=16 are defaults. Each item has a fixed nearest-center route shared by its FP32 and compiled controls. A concept/cluster with fewer than ten positives or ten negatives falls back to its global head. The fallback count is reported. Token local heads initialize from the global token head and update only supported entries. Global/local heads have different parameter counts; those counts are included.

## Labels and input text

Twelve fixed concepts come from explicit WANDS product metadata: modern, traditional, industrial and glam primary style; upholstered; outdoor use; water/stain resistance; storage; assembly; drawers; shelves. Missing, conflicting and invalid values stay **unknown**, never negative. Another primary style is a negative for the specified style; these are literal catalog classifications, not independent human assessments of appearance.

The default **redacted** input uses titles and descriptions only, omitting structured features and product class. Existing full-text WANDS embeddings included the label fields, so this control requires a new encoding pass. It is cached thereafter. Descriptions may naturally state the property; success demonstrates property recognition from text, not visual reasoning or subjective aesthetic understanding. Optional `--view full` is explicitly a metadata-decoding control and gets a separate embedding cache.

This supplies better-controlled predicate supervision than a CLIP teacher, but does not replace the separate human WANDS search relevance evaluation. Later soft predicates need independent reviewed annotations; ESCI can provide a separate product-search validation task, not these predicate labels.

## Splits and shared candidates

The pilot selects 12,000 products by a fixed hash of product ID, independent of labels. Exact normalized-title groups are split 60% fitting / 20% calibration / 20% test. Product-family IDs are unavailable; this does not guarantee family-disjointness. Embedding centroids, local clusters and predicate weights fit on the fitting partition only. Scalar calibrators and F1 thresholds fit on calibration products, separately for each precision arm. Test labels never select concepts or compositions.

Concepts need at least 30 positives and negatives in fitting and 10 each in calibration. Up to twelve two-term conjunction/negation plans are selected using fitting support and a fixed seed, in addition to single concepts. Unsupported test plans remain in the report. Recall for a plan with no test positives is undefined and its count is reported; it is not silently converted to zero.

For each plan, its evaluation universe consists of held-out products with known labels for **all participating concepts**. The same DenseOn top-500 candidates within that universe feed every head and precision arm; return budget K=50. Catalog products with unknown labels are outside this measured universe. Consequently this is **conditional recall on the annotated test universe**, not unrestricted catalog recall. Unknown-label eligibility is an evaluation restriction, not a deployable filter. `candidate_ids.json` records every shared pool. The `dense_query` baseline uses the same pool and the pretrained query score.

## The gap accounting

For a plan with R relevant products in its annotated test universe and C relevant products in the shared pool:

- Candidate coverage = C/R.
- Oracle recall@K = min(K,C)/R.
- Coverage loss = 1 − C/R.
- Return-budget loss = C/R − oracle recall@K.
- FP32 modelling loss = oracle recall@K − FP32 recall@K.
- Item compilation loss = FP32 recall − Binary1 recall.
- Weight compilation loss = Binary1 recall − Binary1+int4 recall.

These five signed gaps sum to **1 − Binary1+int4 recall**. Negative compilation losses are retained: quantization can occasionally improve a ranking. Compilation gaps include each arm's recalibration. Oracle is a label-informed upper bound within the fixed pool and return budget, not a deployable model. It is separate from agreement with exact LateOn, which the earlier MUVERA diagnostic measured.

`summary.csv` reports macro AP/F1 and composition precision/recall. `gap_summary.csv` gives the decomposition. Per-concept, per-plan and per-seed files retain the denominators. A one-seed pilot with correlated synthetic plans is diagnostic, not a statistical equivalence test or a representation ceiling. Compare local-vs-global and token-vs-pooled results on the same plans; confirm any selected recipe on additional seeds before advancing.

## Compilation and storage scope

Binary1 stores packed residual signs plus two FP32 correction scalars per vector, using a fitting-only centroid. Int4 predicate weights are stored as four packed bitplanes with per-weight-vector offset/scale. Local item routes are exported as uint8. Token representations retain a code per token and document offsets; they do **not** become one small code per product.

The quality evaluator reconstructs packed values and int4 weights into FP32 to score them. This verifies numerical quality, not optimized bitwise serving speed. A regression test checks the reconstruction against the existing RAS Binary1/int4 scorer. Payload accounting includes representation bytes, heads, calibration parameters, local routing IDs and centers; it excludes file headers, model checkpoints, ANN navigation and allocator overhead. F1 decision thresholds are evaluation artifacts; the composition ranker uses calibrated scores. Training checkpoints and cached scores are not serving payloads. No new serving latency or peak-memory claim is made by this study; isolated serving measurement follows only after choosing a useful head.

## Short running cycle

1. Audit the labels and examples without loading models.
2. Encode the fixed product texts once with the three pinned encoders. Interrupted encoding resumes in 512-product shards.
3. Fit and score the small heads. Completed heads and raw scores are reused on an identical rerun.
4. Inspect oracle gaps and payloads before adding models, seeds or larger corpora.

The Colab keeps its embedding cache separate from study outputs. Changing epochs/clusters/seeds or code creates a fresh study directory while reusing identical embeddings. Changing text, model revision, sequence cap or package versions creates a new embedding cache. The first encoding pass is longer; subsequent head experiments do not need a full retrieval/MUVERA sweep. No fixed runtime estimate is promised across Colab devices.

Local invocation after installing `pip install -e '.[wands,wands-systems,dev]'`:

```bash
python -m experiments.wands_predicates --phase all \
  --data-dir runs/predicates/redacted-12000 \
  --embedding-dir runs/predicates/embeddings \
  --output-dir runs/predicates/study-v1 \
  --device cuda --max-products 12000 --clusters 8 16 --seeds 7
```

Use `--phase audit`, `--phase encode`, or `--phase study` independently. `--models dense colbert` omits MiniLM. `--clusters` with no numbers runs only global heads. Use a fresh data directory for a different sample/text view and a fresh study directory when its locked configuration changes. `--max-products 0` selects all products; first establish a useful result on the pilot.
