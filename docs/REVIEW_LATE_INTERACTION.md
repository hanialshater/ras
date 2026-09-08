# Review before adding ColBERT and MUVERA

Reviewed 2026-09-08: main `c74bbc1` and draft PR #1 at `e795f3a`.
This review focuses on representation, retrieval, evaluation and baseline design.
It does not revalidate historical measurements or certify the native engine.

The strongest idea remains **compiling learned semantic predicates into a small,
reusable sidecar that can run during retrieval**. ColBERT and MUVERA address query
matching and efficient multi-vector retrieval. They are useful comparators and
possible complements to the sidecar.

## Findings

| Priority | Finding | Consequence / action |
|---|---|---|
| High | Existing semantic recall is conditional on a MiniLM top-5,000 pool, then exact filtering (`experiments/large_scale_search.py::_evaluate_queries`, `review_followup.py::evaluate`). | It cannot establish full-catalogue recall improvements. Keep that experiment and add a separate full-held-out-catalogue evaluation. Implemented in the new runner. |
| High | Binary/FP32 predicates fit CLIP-derived fashion labels; the dense baseline is pretrained and receives no equivalent predicate training. | ColBERT zero-shot comparisons also confound supervision and representation. Record the distinction explicitly; future work needs comparable fine-tuning and human relevance labels. |
| High | The published 56 B/item is a semantic sidecar; dense navigation still needs its own representation. | Do not treat the roughly 302 MB illustrative sidecar/program payload as total search memory. With the current 384-D FP32 navigation vectors, that example is roughly 7.98 GB before graph/runtime overhead. PR #1 adds deployed accounting. Multi-vector storage must likewise include retained token matrices and ANN/FDE state. |
| High | Main still lacks PR #1's all-query purity and serving identity/cache corrections. PR #1 remains open. | Baseline work is stacked on that branch so results do not inherit the corrected defects. Merging this work should preserve the parent fixes. |
| Medium | HNSW traversal truth is dense top-K among items passing the same compiled gate; that measures traversal fidelity rather than semantic truth. | ColBERT-top-K overlap must likewise be reported separately from teacher relevance. The new runner separates approximation fidelity and relevance metrics. |
| Medium | The benchmark uses eight image-teacher concepts, generated compound queries, a single fashion dataset and approximately 15.4k held-out indexed items. | This is useful controlled prototype evidence, not a demonstrated Zalando-scale retrieval replacement. Add independent relevance judgments, broader queries/domains, and scale/load experiments after the baseline quality run. |
| Medium | Existing embedding cache metadata includes dataset name, length and model name, but not the full ordered content or immutable model/dataset revisions (`large_scale_search.py::_retrieval_embeddings`). | Upstream changes can make cache reuse stale. The new runner persists its exact prepared inputs/checksums and resolved ColBERT snapshot; immutable upstream dense/teacher cache identity remains follow-up work. |

## What the first addition establishes

- A real upstream ColBERTv2 checkpoint/tokenizer adapter, with exact MaxSim.
- A transparent MUVERA reference FDE and optional Faiss HNSW retrieval.
- Shared token representations for exact ColBERT and every MUVERA arm.
- Separate shared-pool semantic-quality, full-catalogue retrieval-quality and
  approximation-fidelity results, with raw rankings and explicit denominators.
- Reusable prepared representations, checksums, memory components and a Colab run.

It does not claim to reproduce Google's optimized implementation, add learned
ColBERT fine-tuning, or integrate RSA predicates into MUVERA traversal. Those are
separate follow-ups. The immediate decision is whether multi-vector matching
improves the actual query cases and how much of that gain FDE/ANN retrieval retains.

## Validation status

The inherited 22 focused tests pass. New tests check negative-token MaxSim,
variable lengths, query-sum/document-mean asymmetry, empty-partition filling,
projection determinism, stable result IDs and zero-truth/underfilled metrics.
Synthetic flat and HNSW runs validate execution and the full-candidate parity
invariant. Real checkpoint encoding and fashion relevance measurements require
the supplied Colab/model run and are not reported as completed evidence.
