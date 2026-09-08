# Upstream NanoBEIR baseline

This experiment establishes an external correctness reference before revisiting the fashion
benchmark, MUVERA, filtering or serving performance. It imports no RAS scoring code.

[Open Colab](https://colab.research.google.com/github/hanialshater/ras/blob/codex/colbert-muvera-baselines/notebooks/ras_upstream_nanobeir_colab.ipynb)

## Run

Use a fresh GPU runtime. Install `pip install -e '.[upstream-retrieval,dev]'`.

```bash
python -m experiments.upstream_nanobeir --phase reference --output-dir runs/upstream_nanobeir
python -m experiments.upstream_nanobeir --phase compare --output-dir runs/upstream_nanobeir
```

All 13 NanoBEIR datasets are the default. For a smoke run append
`--datasets scifact nfcorpus` to both commands. A subset is not the full NanoBEIR mean.
The runner resumes completed datasets with the same environment, source and settings.
Use a new output directory when changing batch/chunk sizes or packages. A source-only
fix can resume data preparation if no reference/comparison artifacts exist; the prior
manifest is archived. After scoring starts, source changes require a new directory.
`--pool-size 100` is the comparison default; 500 or 1000 creates another comparison directory.

## Reference reproduction

Use the checkpoint's native PyLate encoding settings and PyLate's
`PyLateInformationRetrievalEvaluator`, the same per-dataset evaluator used by
`NanoBEIREvaluator`. Load the original zeta-alpha-ai datasets exactly as PyLate does:
nonempty text fields, binary relevant-document sets, train splits, no self-ID exclusion.
Resolve each dataset to an immutable Hub revision and save the actual corpus, queries,
qrels and a content checksum. Both phases consume that saved snapshot.
Use `--phase data --device cpu` to audit all 13 real datasets without loading models.

Model revisions are pinned in the script. PyLate 1.6.0, Sentence Transformers 5.3.0,
Transformers 4.49.0 and rank-bm25 0.2.2 are pinned; all actual key package versions are
recorded, along with model parameter counts, dtype and token limits. FP32/SDPA and
PyLate's torch scoring backend avoid optional kernel changes. The default corpus chunk
is 128 to bound the query-document-token similarity tensor on a Colab GPU.

The evaluator retains full-corpus predictions by adding corpus size as a recall cutoff;
this leaves the target nDCG@10, MRR@10 and MAP@100 definitions unchanged. Tie ordering
and padded-document behavior remain upstream semantics. We do not substitute the old
ragged NumPy MaxSim implementation. This is a reference reproduction attempt, not a
claim that upstream zero-padding has the same behavior for negative matches as ragged scoring.

`reference_check.csv` compares every dataset with the published model-card nDCG@10.
`reference_summary.json` distinguishes the subset mean from the full 13-dataset
published mean of 0.6758. A difference above 0.01 on any dataset blocks the comparison
command by default. That tolerance is an investigation trigger, not an equivalence test.
The model card does not provide a complete historical environment lock, and its prose
query length differs from its saved checkpoint configuration. Actual settings are recorded;
we do not silently change them to chase the reported score. A diagnostic comparison can
explicitly use `--allow-reference-mismatch`; the failed reference datasets stay in metadata.

## Comparison

- Dense: upstream Sentence Transformers cosine scoring, mathematically a dot product
  of normalized vectors. This is labeled `dense_normalized_dot`.
- ColBERT: the saved upstream full-corpus scores from the reference phase.
- CE: Sentence Transformers CrossEncoder, one raw logit per identical query-document pair.
- BM25: rank-bm25 Okapi, lowercased Unicode word tokens, library defaults. A lightweight
  lexical control, not a reproduction of Lucene/BEIR BM25 scores.

First report dense, ColBERT and BM25 full-NanoBEIR-corpus retrieval. Then freeze the
same dense top-100 candidate IDs for all four methods. Preserve scores for every method,
check ID-set equality and evaluate all using Sentence Transformers' IR metric implementation.
This pool favors the dense retriever's candidates; it measures reranking conditional on
that retriever. We do not inject positives. Candidate relevance coverage is reported.

All relevance denominators, including shared-pool nDCG's ideal ranking, use all
upstream qrels, including relevant IDs absent from the loaded nonempty-text corpus.
This matches the upstream evaluator; missing IDs are not silently removed.
`data_audit.json` lists affected queries, counts and missing IDs for each dataset. Therefore missing relevant candidates remain a penalty, consistently.
This differs from the old experiment's conditional shared-pool recall denominator.
The input strings are identical across models, but native tokenization/truncation differs:
ColBERT checkpoint defaults, dense checkpoint defaults and CE pair max length 512.
Matching parameter scale does not match training supervision or compute.

`paired_ndcg.csv` contains paired query bootstrap differences within each dataset
(5,000 resamples). Repeated queries across methods are paired, not independent samples.
The notebook reports dataset-macro means; do not pool all query rows into a claimed
BEIR mean or infer equivalence from a confidence interval crossing zero.

## Outputs and limits

Root: manifest, scope, frozen dataset snapshots, ColBERT configuration, reference checks,
reference summary, native metric JSON and complete native rankings.

`comparison_pool100/`: quality, per-query nDCG, paired differences, pool coverage,
model metadata, shared candidate IDs and per-method scores. Per-dataset completion markers
are written only after all artifacts for that dataset are saved. Notebook ZIP includes these
artifacts and logs, excluding model downloads.

Offline phase duration is diagnostic only. It includes batching, file I/O and preprocessing;
it is **not request latency**. No serving-speed, storage or MUVERA claim is made here.
The next experiments are official-FDE parity, filtering and an isolated optimized systems
comparison, after the quality reference has been assessed.

Sources:
- https://huggingface.co/lightonai/GTE-ModernColBERT-v1
- https://github.com/lightonai/pylate
- https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
