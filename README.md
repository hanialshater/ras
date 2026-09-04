# Random Semantic Algebra (RSA)

Random Semantic Algebra studies whether expensive semantic supervision can be
**compiled offline into tiny reusable search-time programs over a shared low-bit
item representation**.

The current strongest compact design is **Binary1-LS2-int4**: centered 1-bit
items, two per-item reconstruction scalars, int4 semantic predicate weights,
and calibrated composition.

## Current paper result

Main benchmark: 44,072 fashion products, MiniLM title embeddings, independent
CLIP image semantics, three strict fit/calibration/test splits, and 200 compound
queries per split.

At 20% candidate retention:

| Method | Item bytes | Program bytes / concept | Recall | Purity |
|---|---:|---:|---:|---:|
| Dense MiniLM | 1536 | 0 | .207 | .425 |
| RSA2 sparse LUT | 96 | 580 | .321 | .561 |
| RSA4 sparse LUT | 192 | 3652 | .330 | .571 |
| **Binary1-LS2-int4** | **56** | **216** | **.337** | **.578** |
| PQ64 compiled linear | 64 | 65,548 | .352 | .594 |
| FP32 linear | 1536 | 1548 | .363 | .606 |

Binary1-LS2-int4 recovers about 83% of the incremental FP32 semantic gain over
dense retrieval while using a ~303x smaller predicate scoring payload than the
PQ64 compiled-linear baseline.

The Rust microkernel benchmark on real learned rows/programs shows, for a 5k
candidate pool, about **0.48 ms for one Binary1 predicate and 1.99 ms for eight**
in the recorded run.  These are semantic-kernel measurements, not end-to-end
search latency; the CPU model was unfortunately not recorded for that historical
artifact.

## Bring your own data

RSA is designed as a semantic **sidecar**, not a replacement ANN engine.
Use any embedding model and keep your existing retrieval/filter stack.

```python
import numpy as np
from ras import BinarySemanticIndex

X = np.load("item_embeddings.npy")   # [N, D], your encoder
index = BinarySemanticIndex.build(
    "semantic_index",
    X,
    projection_kind="identity",
)
```

For D=384 the sidecar writes a 56-byte/item serving representation: 48 packed
sign-bit bytes + two f32 correction values.

## Compile your own semantic predicates

Labels may come from humans, a VLM/LLM teacher, a specialist model, or another
supervision source.

```python
from ras import ProgramStore, fit_binary_predicate

y = np.load("office_appropriate.npy")
program = fit_binary_predicate(index, X, y, name="office_appropriate")
ProgramStore("semantic_programs").save(program)
```

Adding a new predicate writes only a new tiny program; it does **not** re-encode
the catalog.

An existing linear head can also be compiled directly with
`compile_linear_program(...)`.

## Inference: candidate IDs in, semantic top-k out

Your host search engine performs ANN/lexical retrieval and exact filters, then
passes integer row IDs to the semantic sidecar:

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

The intended production path is:

```text
ANN / lexical retrieval
        -> exact filters
        -> binary semantic sidecar
        -> semantic top-k
        -> expensive neural ranker
```

See [`docs/SYSTEMS.md`](docs/SYSTEMS.md) for the index format, lifecycle,
update/versioning model, latency decomposition, early-exit rule, and remaining
systems work.

## Native Rust sidecar

The portable Python index/program files are directly consumable by the Rust
executor:

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

The native executor fuses calibrated composition with top-k maintenance and uses
a safe early-termination rule: every additional log-probability term is <= 0, so
once a partial score falls below the current top-k threshold the item cannot
recover.

## Systems benchmark

Export the first real held-out split into the portable sidecar format:

```bash
python -m experiments.export_native_finalists \
  --config configs/binary_bbq.yaml \
  --out-dir results/native_finalists_first_seed
```

Then measure the semantic stage, including composition + top-k + early exit:

```bash
python -m experiments.sidecar_systems \
  --index results/native_finalists_first_seed/sidecar_index \
  --programs results/native_finalists_first_seed/sidecar_programs \
  --resident-items 500000
```

The harness records CPU model, Rust version, median/p95 latency, resident-set
size, candidate count, predicate count, and the fraction of predicate
invocations actually executed.  It intentionally excludes ANN traversal, exact
filters, RPC, and downstream ranking.

## Research code

```text
src/ras/
  binary.py             # centered binary encoder + int4 primitives
  semantic_index.py     # portable BYO item index
  semantic_program.py   # predicate compiler + program store
  serving.py            # Python candidate-side executor
  substrate.py          # experimental random low-bit substrates
  predicates.py         # sparse boosted LUT predicates
  calibration.py        # scalar calibration
  composition.py        # calibrated AND / NOT algebra
  retrieval.py          # MiniLM helpers used by experiments
  teachers.py           # independent CLIP teacher benchmark
  queries.py            # fit-only compound-query generation

rust/semantic_engine/src/bin/
  actual.rs              # real-code finalist microkernel benchmark
  fair.rs                # synthetic fairness benchmark
  sidecar.rs             # portable composition + top-k executor
```

Useful entry points:

- `examples/byo_semantic_sidecar.py` — minimal bring-your-own-data integration.
- `experiments/binary_bbq_predicates.py` — quality benchmark.
- `experiments/export_native_finalists.py` — real native/portable export.
- `experiments/sidecar_systems.py` — stage-level latency benchmark.
- `notebooks/rsa_fashion_search_demo_colab.ipynb` — interactive fashion image demo.

## Reproduction

```bash
pip install -e .
pytest -q

# Paper-scale quality benchmark
python -m experiments.binary_bbq_predicates --config configs/binary_bbq.yaml

# Export real native assets and portable sidecar
python -m experiments.export_native_finalists --config configs/binary_bbq.yaml

# Rust tests
cargo test --release --manifest-path rust/semantic_engine/Cargo.toml

# ICML-style paper
cd paper/icml
pdflatex main.tex
pdflatex main.tex
```

## Paper

The conference-style manuscript is under [`paper/icml`](paper/icml).  The latest
version includes the Binary1/PQ/RSA2/RSA4 comparison, real native throughput,
explicit microkernel latency, a bring-your-own systems interface, joint item +
predicate memory accounting, and a clear separation between kernel latency and
end-to-end search latency.

## Research status

The strongest supported thesis is now **compiled semantic predicate execution**,
not randomness or four-bit LUTs by themselves.  The next evidence step is a
fully instrumented search-stage benchmark with p50/p95/p99 latency and CPU
metadata, followed by a real ANN/filter integration and a large semantic
vocabulary cache experiment.
