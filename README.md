# Random Semantic Algebra (RSA)

Random Semantic Algebra studies whether expensive semantic supervision can be
**compiled offline into tiny reusable search-time programs over a shared low-bit
item representation**.

The strongest compact design in the current experiments is
**Binary1-LS2-int4**:

- 56 B/item: 384 centered sign bits + two per-item reconstruction scalars.
- ~216 B/semantic predicate: int4 weights packed into four bit planes plus scalar metadata.
- Learned predicates can be added independently without re-encoding the catalog.
- The same program can run after retrieval or directly inside ANN traversal.

## Headline results

### Semantic quality

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

Binary1-LS2-int4 recovers about **83% of the incremental FP32 semantic gain**
over dense retrieval while using a ~303x smaller predicate payload than the
compiled-PQ64 baseline.

### Live soft predicates inside HNSW

The latest systems benchmark executes the real Binary1-LS2-int4 predicates
**inside timed HNSW traversal**. Dense similarity still navigates the graph;
semantic programs only decide whether a visited node may enter the valid result
beam. Invalid nodes remain traversable.

The fair benchmark uses the same normalized-dot geometry and same graph for our
custom traversal and the `hnsw_rs` filtered baseline, 100 queries, K=50, EF=128,
and three active predicates:

| Eligible catalog fraction | Live semantic HNSW | Filtered HNSW | Live Recall@50 | Filtered Recall@50 |
|---:|---:|---:|---:|---:|
| 50% | 2.13 ms | 5.00 ms | .9816 | .9808 |
| 20% | 4.90 ms | 10.44 ms | .9774 | .9738 |
| 10% | 7.09 ms | 14.37 ms | .9730 | .9718 |
| 5% | 10.18 ms | 20.19 ms | .9786 | .9758 |
| 2% | 13.71 ms | 26.39 ms | .9828 | .9798 |

Across the sweep, live execution exactly matches the corresponding materialized
custom traversal. The incremental compiled-program cost is roughly
**108–114 ns per predicate invocation** in this run.

These are single-thread prototype measurements, not production p99 claims. The
important result is that the soft predicate is cheap enough to participate in
ANN traversal and avoids the severe recall loss of post-filtering at low
selectivity.

### Joint memory

For a catalog with `N` items and `C` learned concepts, the relevant payload is:

```text
N * item_bytes + C * program_bytes
```

Illustrative 5M-item / 100k-concept payload:

| Method | Item payload | Program payload | Total |
|---|---:|---:|---:|
| **Binary1-LS2-int4** | 280 MB | 21.6 MB | **301.6 MB** |
| RSA2 sparse LUT | 480 MB | 58 MB | 538 MB |
| PQ64 compiled linear | 320 MB | 6,554.8 MB | 6,874.8 MB |
| FP32 linear | 7,680 MB | 154.8 MB | 7,834.8 MB |

This is representation payload only; it excludes HNSW graph edges, containers,
allocator overhead, and offline model weights.

## Architecture

A natural-language or LLM planner can emit a structured search plan:

```text
Dense query:  "flowy midi skirt"
Exact:        category=skirts, gender=women
Soft:         +fluid +refined -sporty
```

Then:

```text
query embedding
      ↓
dense HNSW navigation
      ↓
visited node
      ↓
compiled soft predicate
   ┌───────┴────────┐
 valid            invalid
   ↓                 ↓
result beam      still traversable
      ↓
downstream personalized / neural ranker
```

The final HNSW design deliberately does **not** steer graph geometry with the
semantic score. Dense similarity owns navigation; semantics own eligibility.

## Bring your own embeddings

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

For D=384, this writes a 56-byte/item serving representation.

## Compile semantic predicates

Labels may come from humans, a VLM/LLM teacher, a specialist model, or another
supervision source.

```python
from ras import ProgramStore, fit_binary_predicate

y = np.load("office_appropriate.npy")
program = fit_binary_predicate(index, X, y, name="office_appropriate")
ProgramStore("semantic_programs").save(program)
```

Adding a predicate writes only the tiny program; the item index is unchanged.
Existing linear heads can also be compiled with `compile_linear_program(...)`.

## Candidate-side API

If you want to keep ANN completely external:

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

## Live HNSW experiment

The native executable is:

```text
rust/semantic_engine/src/bin/semantic_hnsw_live.rs
```

The reproducible non-notebook harness is:

```bash
python -m experiments.semantic_hnsw_live_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_live_fair
```

It:

1. exports the strict held-out real item codes and programs;
2. verifies unit-normalized retrieval embeddings;
3. derives semantic thresholds for 50%, 20%, 10%, 5%, and 2% selectivity;
4. builds the Rust live-HNSW executable;
5. runs post-filter, filtered-HNSW, free-materialized custom traversal, and live compiled predicates;
6. writes `raw.csv`, `summary.csv`, `fairness.csv`, `gates.csv`, and `environment.json`.

The paper's checked-in result table is
[`paper/icml/data/semantic_hnsw_live_fair_full.csv`](paper/icml/data/semantic_hnsw_live_fair_full.csv).

## Interactive fashion demo

The demo shows held-out fashion images, parses exact vs soft intent, compares
Binary1-LS2-int4 with PQ64 and other baselines, and displays the paper's memory
and live-HNSW systems results separately from Python UI latency.

```bash
pip install -e ".[demo,benchmark]"
```

Colab:

[`notebooks/rsa_fashion_search_demo_colab.ipynb`](notebooks/rsa_fashion_search_demo_colab.ipynb)

Core demo code:

[`demos/fashion_app.py`](demos/fashion_app.py)

## Research code

```text
src/ras/
  accounting.py         # centralized item/program memory accounting
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
  sidecar.rs             # portable candidate-side executor
  semantic_hnsw.rs       # traversal prototypes / ablations
  semantic_hnsw_live.rs  # live compiled predicates inside HNSW
```

## Reproduction

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,benchmark,demo]"
```

Then:

```bash
# Unit tests
pytest -q

# Paper-scale semantic quality benchmark
python -m experiments.binary_bbq_predicates --config configs/binary_bbq.yaml

# Fair live semantic-HNSW systems benchmark
python -m experiments.semantic_hnsw_live_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_live_fair

# Rust tests
cargo test --release --manifest-path rust/semantic_engine/Cargo.toml

# ICML-style manuscript
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact artifact
contract and what each benchmark does and does not measure.

## Paper

The manuscript is under [`paper/icml`](paper/icml). The current story is no
longer "random 4-bit codes are best." The stronger supported thesis is:

> **Soft semantic predicates can be compiled into tiny reusable programs over a
> shared low-bit item substrate and executed cheaply enough to participate
> directly in ANN traversal.**

PQ64 remains the stronger compressed-quality baseline. Binary1-LS2-int4 is the
stronger tiny-program / large-vocabulary point.

## Research status

Current evidence is controlled and single-domain. The next work is external
validity rather than more graph variants: natural LLM-generated plans, a second
dataset/domain, larger semantic vocabularies, and production-grade ANN/service
integration with concurrency and p99 measurements.
