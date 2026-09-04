# Compiled semantic search systems design

This document defines the deployment boundary for the current strongest design:
**dense ANN geometry + tiny learned soft predicates over one shared binary item substrate**.

The semantic layer supports two serving modes:

1. **Candidate sidecar** — ANN/exact filtering runs first, then semantic programs prune a candidate array.
2. **Integrated HNSW** — dense similarity navigates the graph; the compiled semantic program decides whether a visited node is eligible for the result beam. Invalid nodes remain traversable.

The second mode is now the stronger systems result.

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

Important design choice: **semantic score does not steer the graph priority**.

Experiments showed that direct `dense + λ * semantic` navigation damaged recall, and bounded two-hop bridge expansion added substantial work without a meaningful quality gain. Dense geometry owns navigation; semantics own eligibility.

The live Rust executable is:

```text
rust/semantic_engine/src/bin/semantic_hnsw_live.rs
```

It uses the same normalized-dot distance for both the `hnsw_rs` baseline and the extracted custom traversal, verifies unit-normalized input vectors, caches each semantic result once per query, and evaluates semantics only after a dense admissibility check.

## 5. Fair live benchmark

Run:

```bash
python -m experiments.semantic_hnsw_live_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_live_fair
```

Defaults:

```text
100 queries
K=50
EF=128
M=24
three active predicates:
  +minimalist
  +office_appropriate
  -technical_sporty
semantic eligibility: 50%, 20%, 10%, 5%, 2%
```

The harness writes:

```text
gates.csv
raw.csv
summary.csv
fairness.csv
environment.json
```

and records repository commit, CPU model, Python/Rust versions, normalization statistics, and benchmark parameters.

Checked-in paper summary:

```text
paper/icml/data/semantic_hnsw_live_fair_full.csv
```

### Recorded fair result

| Eligible | Live semantic HNSW | Materialized filtered HNSW | Live Recall@50 | Filtered Recall@50 |
|---:|---:|---:|---:|---:|
| 50% | 2.13 ms | 5.00 ms | .9816 | .9808 |
| 20% | 4.90 ms | 10.44 ms | .9774 | .9738 |
| 10% | 7.09 ms | 14.37 ms | .9730 | .9718 |
| 5% | 10.18 ms | 20.19 ms | .9786 | .9758 |
| 2% | 13.71 ms | 26.39 ms | .9828 | .9798 |

The actual compiled program adds roughly **108–114 ns per predicate invocation** in this run and exactly matches the result IDs of the free-materialized custom traversal.

This is a controlled, single-thread research result, not a production p99 claim.

## 6. Why post-filtering is insufficient

At low selectivity, retrieving a fixed dense pool and filtering afterward loses most valid nearest neighbors. In the same full-data experiment, post-filter Recall@50 falls from .927 at 50% eligibility to .135 at 2% eligibility.

Filter-aware traversal keeps invalid points available for navigation while preventing them from consuming the valid result beam.

## 7. Joint item + program memory

For N items and C compiled concepts:

```text
M = N * item_bytes + C * program_bytes
```

Illustrative 5M-item / 100k-concept payload:

| Method | Items | Programs | Total |
|---|---:|---:|---:|
| Binary1-LS2-int4 | 280 MB | 21.6 MB | **301.6 MB** |
| RSA2 sparse LUT | 480 MB | 58 MB | 538 MB |
| PQ64 compiled linear | 320 MB | 6,554.8 MB | 6,874.8 MB |
| FP32 linear | 7,680 MB | 154.8 MB | 7,834.8 MB |

These figures are representation payload only. They exclude graph edges, allocators, containers, and offline model weights.

The source of truth is `ras.accounting`.

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

- production mmap/segment behavior;
- p50/p95/p99 under concurrency;
- distributed RPC overhead;
- production ANN integration and update/tombstone handling;
- realistic 100k–1M concept-store cache behavior;
- natural search/recommendation relevance from LLM-generated plans;
- exact parity with Lucene/Elasticsearch BBQ or another production filtered-vector engine.

See `docs/REPRODUCIBILITY.md` for the exact artifact protocol.
