# Short official MUVERA diagnostic

Open `notebooks/ras_wands_official_fde_colab.ipynb`. CPU runtime is sufficient.
It reuses the completed WANDS comparison and systems-run embeddings. It does
not download models, re-encode product text, or rerun the serving benchmark.

The default is **one configuration, 10 fixed queries, the entire 42,994-product
corpus**, and candidate budgets 100/1,000/5,000. Query IDs are the first 10 of
the earlier seed-7 60-query diagnostic sample. Selecting 60 preserves that
sample. These are exploratory development queries, not a held-out test set.

## Actual upstream code

We compile the unchanged `.cc`, `.h`, and `.proto` files from
[google/graph-mining at dca52ed1](https://github.com/google/graph-mining/tree/dca52ed1c35ca530d11ea7f76b0bf74634cdeb5c/sketching/point_cloud).
`third_party/google_fde/upstream.json` records their SHA-256 hashes. The adapter
only converts ragged arrays, sets the config, calls the official functions,
and parallelizes independent documents. CMake dependencies are pinned and
archive hashes are checked. This is the public FDE generator, **not** the
paper's complete DiskANN/PQ search stack.

The official implementation uses sparse inner projections and optional final
CountSketch. Its inner projection differs from our dense random-sign
`paper_*` controls. Same numeric seeds also do not imply identical random maps
between NumPy and C++. The comparison does not isolate a single code change.

## Short cycle

1. First run: stage saved tokens, compile native code, hash inputs, generate
   document FDEs for one configuration, and score the selected queries.
2. Same runtime: document FDEs are reused when query count changes; query
   scores are reused when only candidate budgets change. Configuration or
   token changes get a separate cache. Interrupted document construction
   resumes after its last flushed block.
3. Only small CSV/JSON reports are copied back to Drive. There are no hundreds
   of score-shard copies. Native build and document FDE cache stay local to
   the Colab runtime; a runtime reset requires rebuilding these.

Fewer queries reduce scoring and evaluation, but **do not reduce the first
full-corpus FDE generation**. Progress prints elapsed time and an observed
remaining-time estimate. No hardware-independent runtime promise is made.

Outputs include exact LateOn and dense quality **on the same selected queries**,
dense top-100 → LateOn, the existing dense/BM25 union → LateOn, and official
FDE candidates → LateOn. Full-corpus judgments are used throughout. The
`dense_full` fidelity row is top-k ranking overlap, while candidate controls
measure candidate recall. Top-1 recall is reported separately from top-k.
No serving latency or memory claims can be made from this offline run.

## Local check

```bash
pip install 'cmake>=3.24,<4' ninja numpy pandas threadpoolctl ir-measures pytest
python -m experiments.build_google_fde --build-dir /tmp/google-fde-build
GOOGLE_FDE_LIBRARY=/tmp/google-fde-build/libgoogle_fde.so PYTHONPATH=src:. \
  python -m pytest -q tests/test_google_fde.py
```

The native tests cover query/document sum/mean behavior, identity scoring,
batch and thread parity, error handling, full-budget exact-quality recovery,
query-score reuse, document-FDE reuse, and rejection of mismatched references.
