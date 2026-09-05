# Compiled semantic search systems design

This document defines the deployment boundary for the current strongest design:
**dense ANN geometry + tiny learned soft predicates over one shared binary item substrate**.

The semantic layer supports two serving modes:

1. **Candidate sidecar** — ANN/exact filtering runs first, then semantic programs prune a candidate array.
2. **Integrated HNSW** — dense similarity navigates the graph; the compiled semantic program decides whether a visited node is eligible for the result beam. Invalid nodes remain traversable.

The second mode is the stronger systems result.

## 1. Data model

Bring any precomputed embedding matrix:

```python
import numpy as np
from ras import BinarySemanticIndex

X = np.load("item_embeddings.npy")
index = BinarySemanticIndex.build(
    "semantic_index",
    X,
    projection_kind="identity",
)
```

For D=384, Binary1-LS2 stores:

```text
48 B  centered sign bits
 8 B  two f32 reconstruction scalars
----
56 B / item
```

Portable layout:

```text
semantic_index/
  manifest.json
  bits.u8
  corrections.f32
  centroid.f32
  projection.f32
```

The Python loader can mmap the item payload. The current Rust research binaries load resident files into memory.

## 2. Compile soft concepts independently

Labels may come from humans, a VLM/LLM teacher, a specialist model, or another supervision source.

```python
from ras import ProgramStore, fit_binary_predicate

y = np.load("office_appropriate.npy")
program = fit_binary_predicate(index, X, y, name="office_appropriate")
ProgramStore("semantic_programs").save(program)
```

An existing linear head can also be compiled directly with `compile_linear_program(...)`.

For D=384, Binary1-LS2-int4 uses about **216 B/predicate**: four packed int4 weight bit planes plus scalar scoring/calibration metadata.

Adding a new predicate does **not** rewrite the item index.

## 3. Candidate-side API

```python
from ras import SemanticExecutor

executor = SemanticExecutor.open("semantic_index", "semantic_programs")
result = executor.topk(
    candidate_ids,
    positive=["minimalist", "office_appropriate"],
    negative=["technical_sporty"],
    k=1000,
)
```

This mode is useful when ANN must remain completely external.

## 4. Integrated semantic HNSW

The final traversal design is intentionally simple:

```text
query embedding
      ↓
dense HNSW navigation
      ↓
discovered node
      ↓
dense score competitive?
   no ───────→ discard before semantic work
   yes
      ↓
live compiled soft predicate
   ┌──────────┴──────────┐
 eligible             ineligible
   ↓                      ↓
valid result beam      still traversable
```

Important design choice: **semantic score does not steer graph priority**.

Experiments showed that direct `dense + λ * semantic` navigation damaged traversal recall, and bounded two-hop bridge expansion added substantial work without a meaningful quality gain. Dense geometry owns navigation; semantics own eligibility.

The reviewer executable is:

```text
rust/semantic_engine/src/bin/semantic_hnsw_reviewer.rs
```

It uses normalized-dot geometry, builds the graph once per predicate set, materializes semantic truth once, computes dense query-to-item scores once per query, and then evaluates all semantic gates and over-fetch budgets inside that same process. `semantic_hnsw_live.rs` remains as the earlier single-condition controlled traversal harness.

## 5. Reviewer-controlled benchmark

Run:

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

Protocol:

```text
strict held-out graph: ~15,426 items
1000 queries / predicate set
3 predetermined predicate sets
3 active predicates / set
K=50
EF=128
M=24
efConstruction=200
semantic eligibility: 50%, 20%, 10%, 5%, 2%
completed over-fetch sweep: .75, 1, 1.5, 2 × K/selectivity
extended harness: .75, 1, 1.5, 2, 3, 4, 6, 8 × K/selectivity
matched-recall tolerance: .005
```

Predicate sets:

```text
+minimalist +office_appropriate -technical_sporty
+elegant +quiet_luxury -chunky
+retro +relaxed -office_appropriate
```

### Recall definition

**Traversal Recall@50** compares an ANN result with brute-force dense top-K among items passing the **same compiled semantic predicate**. It measures search execution fidelity, not semantic relevance.

Exact live/materialized result-ID parity compares live program execution with the same custom traversal using precomputed semantic scores. It is a correctness check, not a relevance result.

### Completed result through 2× over-fetch

The largest completed selectivity-aware over-fetch point, averaged across the three predicate sets, is:

| Eligible | Live Traversal Recall@50 | 2× over-fetch recall | Live ms | 2× over-fetch ms |
|---:|---:|---:|---:|---:|
| 50% | .9833 | .7923 | 2.23 | 1.61 |
| 20% | .9826 | .7098 | 4.58 | 3.46 |
| 10% | .9794 | .7180 | 6.79 | 5.95 |
| 5% | .9752 | .7482 | 9.82 | 10.27 |
| 2% | .9778 | .8420 | 14.57 | 20.14 |

**No tested over-fetch point through 2× reaches live traversal recall within .005 in any of the 15 predicate-set × selectivity conditions.** Therefore this completed run does not support a matched-recall latency ratio.

At 50–10% eligibility, the largest tested over-fetch budget is faster but materially lower recall. At 5% and 2%, live traversal is both faster and substantially higher recall than that 2× point.

The public harness extends the sweep to 3×, 4×, 6× and 8× specifically to locate a genuine matched-recall point. Those results must be measured before making a universal comparison with over-fetch.

Checked-in aggregate:

```text
paper/icml/data/semantic_hnsw_reviewer_2x_summary.csv
```

## 6. What does the semantic program itself cost?

The controlled comparison is:

```text
semantic_hnsw_live
minus
custom_hnsw_materialized
```

Both use the same custom traversal and graph. The difference isolates the incremental cost of executing the real Binary1-LS2-int4 programs online.

Across the completed reviewer conditions, the estimate is approximately **96–134 ns per predicate invocation**. The same scalar 384-D normalized-dot kernel costs **617.155 ns** on the recorded Intel Xeon 2.20 GHz CPU, so one predicate invocation is roughly **0.16–0.22 dense-distance equivalents** in this implementation.

This is an incremental traversal estimate, not a standalone instruction benchmark.

## 7. Persistent item + program memory

For N items and C compiled concepts:

```text
persistent payload = N * item_bytes + C * stored_program_bytes + shared_bytes
```

Illustrative 5M-item / 100k-concept payload:

| Method | Items | Persistent programs/store | Total |
|---|---:|---:|---:|
| Binary1-LS2-int4 | 280.0 MB | 21.6 MB | **301.6 MB** |
| PQ64 + compact linear heads | 320.0 MB | ~155.2 MB | ~475.2 MB |
| RSA2 sparse LUT | 480.0 MB | 58.0 MB | 538.0 MB |
| FP32 linear | 7,680.0 MB | 154.8 MB | 7,834.8 MB |

PQ64 requires 64 B/item. Persistently, a semantic concept can remain a 1,548-byte FP32 linear head plus one ~0.393 MB codebook shared by all concepts. The native scoring kernel materializes a **65,548-byte LUT per active concept**, but that is activation-time state rather than mandatory persistent state.

These figures are representation payload only. They exclude graph edges, allocators, containers, and offline model weights. The source of truth for current persistent accounting is `ras.accounting`.

## 8. Composition and early exit

For calibrated logits `L_i`:

```text
S(x) = sum positive log sigmoid(L_i)
     + sum negative log sigmoid(-L_i)
```

Every term is <= 0, so a partial sum upper-bounds the final score. Candidate-side execution can therefore stop evaluating remaining predicates once the partial score is below the current top-k threshold.

The current HNSW eligibility benchmark evaluates the active predicate set exactly, but the same property remains available for future multi-predicate short-circuiting.

## 9. Updates and versioning

- **New predicate:** publish one new program; no item re-encoding.
- **New item:** encode with the frozen binary transform.
- **Delete:** host index stops emitting/traversing the row ID; production integration may add tombstones.
- **Embedding-model or centroid change:** create a new semantic-index generation and recompile predicates.

Programs should be tied to an index generation in production.

## 10. Remaining systems work

The current result establishes that learned soft predicates are cheap enough to participate directly in ANN traversal. It does not yet establish:

- the matched-recall frontier beyond 2× over-fetch;
- QPS/core on a million-scale graph;
- production mmap/segment behavior;
- p50/p95/p99 under concurrency;
- distributed RPC overhead;
- production ANN integration and update/tombstone handling;
- realistic 100k–1M concept-store cache behavior;
- natural search/recommendation relevance from LLM-generated plans;
- exact parity with Lucene/Elasticsearch BBQ or another production filtered-vector engine.

See `docs/REPRODUCIBILITY.md` for the exact artifact protocol.
