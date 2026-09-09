# WANDS product-search comparison

[Open the Colab notebook](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_wands_colab.ipynb).

Use a fresh GPU runtime and run all cells. The notebook installs the pinned
`wands` extra, runs native integration tests, audits the real data, and exercises
all three real neural models on two queries/four products before the full run.

```bash
pip install -e '.[wands,dev]'
python -m experiments.wands_comparison --phase data --output-dir results/wands
python -m experiments.wands_comparison --phase smoke --output-dir results/wands
python -m experiments.wands_comparison --phase compare --output-dir results/wands
```

Default: 480 Wayfair shopping queries against all 42,994 products. The source is
the original [WANDS repository](https://github.com/wayfair/WANDS), pinned to
`3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`. Input files are TSV despite their
`.csv` extension. Text includes product name, class, features and description,
in that order. It excludes query classes, ratings and label-derived features.
`--text-mode title` provides a title-only ablation in a new output directory.

## Models and scope

* DenseOn and LateOn: published paired ModernBERT-base models, about 149M parameters.
* Cross-encoder: `Alibaba-NLP/gte-reranker-modernbert-base`, similar size but
  separately trained; it is not an architecture-only controlled comparison.
* BM25: rank-bm25 with lowercase Unicode word tokenization, a lexical control,
  not a reproduction of a Lucene baseline.

All model revisions are immutable and actual settings are exported. FP32, SDPA,
512-token document caps; CE's cap applies to the entire pair. Source text is
identical, but query prefixes, query expansion, and pair truncation follow each
model's interface. Dense uses native ST encode_query/encode_document and normalized
dot product. LateOn uses PyLate encoding and its native MaxSim implementation.
Documents are grouped by encoded length before scoring so zero padding cannot
outscore a negative real token match. No custom MaxSim formula or ANN is used.

Full-corpus scoring produces dense, ColBERT and BM25 results. The shared pool is
the union of dense top-100 and BM25 top-100 (at most 200 candidates). All four
models rank exactly these same IDs. Relevant products are never injected. Pool
coverage is reported separately. Corpus and pool results always use the same
full-corpus judgment denominator.

## Labels and metrics

Original source: 233,448 label rows, 231,873 unique query-product pairs,
1,575 duplicate rows, and 14 conflicting pairs. Identical duplicates collapse;
conflicts conservatively take the minimum grade, with every conflict saved in
`data_audit.json`. This explicit policy is our protocol, not a claimed official
WANDS evaluation rule. Query 366 has no positive judgments and remains with zero
quality scores rather than disappearing from the average.

`ir_measures` computes nDCG@10 with linear gains Exact=2, Partial=1, Irrelevant=0.
MRR@10, precision@10, and recall@100 count only Exact as relevant. Unjudged items
score zero and are retained in rankings; `judged_at10` reports judged items / 10.
Recall covers known judged Exact products, not all truly relevant products.
Paired bootstrap intervals resample queries (5,000 draws), independently per
scope. No training on WANDS is performed; pretrained exposure is not established.

## Resumption and artifacts

Score shards save atomically after every 256 products, and CE outputs after each
query. Resumption reuses completed shards; configuration, data, library versions,
and source hashes must match. Model loading and query encoding repeat on resume.
Use a new output directory after code/configuration changes. Optional
`--queries 50` draws a seeded subset while retaining the entire product corpus.

Outputs: quality/per-query/paired-interval CSVs, pool coverage, candidate IDs,
top-1,000 full rankings, complete shared rankings, all dense/ColBERT score shards,
source audit, model settings and dataset snapshot. Colab exports a ZIP.

This is a relevance-quality run. Printed shard times are offline progress only;
they do not establish serving latency, memory, storage or index build-time gains.
WANDS is English home furnishings. Fashion and multilingual applicability still
need separate evaluation. MUVERA/IVF benchmarking follows a validated relevance
baseline and is not part of this notebook.
