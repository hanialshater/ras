# Semantic sidecar systems design

This document defines the deployment boundary for compiled semantic predicates.
The low-bit layer is **not** an ANN replacement.  It is a sidecar execution stage
that receives candidate row IDs from an existing search stack.

## 1. System boundary

```text
query
  |
  +--> host ANN / lexical retrieval --------------------+
  |                                                     |
  +--> host exact filters (brand, color, policy, ...) --+--> candidate row IDs
                                                               |
                                                               v
                                                binary semantic sidecar
                                                + compiled predicates
                                                               |
                                                        semantic top-k
                                                               |
                                                               v
                                                     expensive ranker
```

The host remains responsible for document lifecycle, retrieval, exact filters,
query parsing, and final ranking.  The sidecar owns only:

1. a concept-independent binary item index;
2. a store of independently deployable semantic predicates;
3. candidate scoring, calibrated composition, and semantic pruning.

This boundary lets an existing Elasticsearch/Lucene, Vespa, FAISS, vector DB,
or custom retrieval system keep its first-stage index unchanged.

## 2. Bring your own data

The Python API accepts arbitrary precomputed embeddings.  No encoder is required
inside RSA.

```python
import numpy as np
from ras import BinarySemanticIndex

X = np.load("item_embeddings.npy")       # [N, D], any embedding model
ids = np.load("item_ids.npy")            # optional external IDs

index = BinarySemanticIndex.build(
    "semantic_index",
    X,
    item_ids=ids,
    projection_kind="identity",          # strongest current compact result
)
```

The serving representation contains packed centered sign bits plus two per-item
least-squares reconstruction values.  At D=384 the payload is 56 bytes/item.
The current portable format is:

```text
semantic_index/
  manifest.json
  bits.u8
  corrections.f32
  centroid.f32
  projection.f32       # only for non-identity experimental variants
  item_ids.npy         # optional
```

`BinarySemanticIndex.load(..., mmap=True)` memory-maps the item payload in the
Python reference executor.

## 3. Bring your own semantic supervision

A predicate can be learned from human labels, a VLM/LLM teacher, a specialist
classifier, or any other binary semantic target.  Full-precision embeddings are
needed only while fitting the semantic head.

```python
from ras import ProgramStore, fit_binary_predicate

labels = np.load("office_appropriate.npy")  # one 0/1 label per item
program = fit_binary_predicate(
    index,
    X,
    labels,
    name="office_appropriate",
)
ProgramStore("semantic_programs").save(program)
```

Alternatively, an existing linear head can be compiled directly:

```python
from ras import compile_linear_program
program = compile_linear_program(
    index,
    name="office_appropriate",
    weight=w,
    intercept=b,
)
```

For D=384 the Binary1-LS2-int4 scoring payload is approximately 216 bytes per
predicate: four packed int4 weight bit planes plus six f32 scoring/calibration
scalars.  Planner metadata is stored separately.

**Adding or updating a predicate does not rewrite `bits.u8` or
`corrections.f32`.**  This is the key operational distinction from adding a new
item representation.

## 4. Inference API

The host search engine passes candidate **row IDs**, not full embeddings, to the
sidecar.

```python
from ras import SemanticExecutor

executor = SemanticExecutor.open("semantic_index", "semantic_programs")
result = executor.topk(
    candidate_ids,                        # output of ANN + exact filters
    positive=["minimalist", "office_appropriate"],
    negative=["technical_sporty"],
    k=1000,
)
```

`result.row_ids` are returned to the host ranker.  Optional external item IDs are
metadata; production integrations should normally maintain the row-ID mapping in
the host index so the candidate handoff is an integer array.

## 5. Native Rust executor

The Rust binary consumes exactly the portable item/program files:

```bash
cd rust/semantic_engine
cargo build --release --bin sidecar

./target/release/sidecar \
  --index ../../semantic_index \
  --programs ../../semantic_programs \
  --positive minimalist,office_appropriate \
  --negative technical_sporty \
  --candidate-count 5000 \
  --topk 1000
```

A real integration can instead provide a little-endian `u32` candidate file with
`--candidates candidates.u32`.  The current Rust prototype loads the resident
item files into process memory; the Python reference loader already supports
memory mapping.  OS-backed mmap/direct integration is a serving-engine follow-up.

## 6. Safe early exit from semantic composition

For calibrated logits `L_i`, the query score is

```text
S(x) = sum positive log sigmoid(L_i)
     + sum negative log sigmoid(-L_i).
```

Every term is <= 0.  Therefore after evaluating any prefix of predicates,

```text
S_partial(x) >= S_final(x).
```

If the current top-k heap threshold is `tau` and `S_partial(x) <= tau`, the item
cannot recover after evaluating the remaining predicates.  The native sidecar
therefore orders likely-selective predicates first and can reject an item before
executing the rest of the program list.  This is analogous in spirit to
WAND/MaxScore pruning, but the upper bound follows directly from the calibrated
log-probability composition rule.

The benchmark harness reports `evaluated_predicate_fraction`, making the benefit
measurable rather than assumed.

## 7. What latency means

End-to-end search latency should be decomposed as

```text
T_search = T_retrieval
         + T_exact_filter
         + T_semantic
         + T_topk
         + T_downstream_rank
         + T_rpc/serialization   (if the sidecar is remote)
```

The existing `actual.rs` benchmark measures a semantic **microkernel** only.  The
new `sidecar` benchmark measures semantic scoring + calibrated composition +
top-k maintenance + optional early exit.  Neither benchmark includes ANN
traversal, exact-filter evaluation, network RPC, or the downstream ranker.

Run the stage-level benchmark with CPU metadata recorded automatically:

```bash
python -m experiments.sidecar_systems \
  --index results/native_finalists_first_seed/sidecar_index \
  --programs results/native_finalists_first_seed/sidecar_programs \
  --resident-items 500000
```

The output contains median and p95 latency for 5k/20k/100k candidate pools,
1/2/4/8 active predicates, early exit on/off, and the fraction of predicate
invocations actually executed.  `environment.json` records the CPU model,
platform, Rust version, resident-set size, and benchmark scope.

## 8. Item updates, deletes, and versioning

The current research implementation intentionally keeps mutation policy outside
the serving kernel.

- **New predicate:** compile and atomically publish one new program directory;
  no item re-indexing.
- **New item:** encode it with the frozen `CenteredBinaryEncoder`.  The Python API
  exposes `index.encode_batch(new_embeddings)`.  Append/segment management is an
  integration concern not yet implemented in the raw-file writer.
- **Delete:** the host retrieval layer should stop returning the row ID.  A
  production sidecar may additionally maintain tombstones if direct scans are
  added later.
- **Embedding-model change:** build a new semantic-index generation in the
  background and switch generations atomically.  Programs must be recompiled
  because they are tied to the embedding/index generation.
- **Centroid drift:** small catalog changes use the frozen centroid.  Material
  distribution drift should trigger a new generation rather than mutating the
  centroid in place, because changing it changes every sign bit.

A production manifest should therefore carry an index-generation ID and each
program should declare the generation it was compiled against.  The current
prototype checks dimensional compatibility; generation IDs are a planned format
extension.

## 9. Remaining systems work

Before making production-latency claims, the research needs:

- mmap/zero-copy native loading and direct candidate-array integration;
- SIMD/popcount variants and thread scaling;
- random vs sorted document-ID locality tests;
- p50/p95/p99 under concurrent query load;
- large semantic-vocabulary cache experiments (100 to 100k programs);
- update/segment compaction and generation versioning;
- a real ANN/filter integration reporting end-to-end latency and global recall or
  NDCG, not only conditional recall within an ANN candidate pool;
- an exact Lucene/Elasticsearch BBQ comparison if the paper wants to claim
  parity with that production implementation.

The goal is a narrow, testable systems claim: **given a candidate set from an
existing search engine, how cheaply can many reusable semantic constraints be
executed before the expensive ranker?**
