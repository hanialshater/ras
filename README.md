# Compiled Semantic Predicates for ANN Search

This repository studies whether expensive semantic supervision can be **compiled
offline into tiny reusable search-time programs over a shared low-bit item
representation**, and whether those programs are cheap enough to execute directly
inside approximate-nearest-neighbor traversal.

The project began as *Random Semantic Algebra (RSA)*. The experiments no longer
support randomness or 4-bit sparse LUTs as the central result. The strongest
compact design is now **Binary1-LS2-int4**:

- **56 B/item**: 384 centered sign bits + two per-item reconstruction scalars.
- **~216 B/semantic predicate**: int4 weights packed into four bit planes plus scalar metadata.
- New learned predicates can be added without re-encoding the catalog.
- The same program can run after retrieval or directly inside ANN traversal.

The manuscript is titled **Compiled Semantic Predicates for Approximate Nearest
Neighbor Search**.

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
| PQ64 compiled linear | 64 | 1,548 stored / 65,548 active LUT | .352 | .594 |
| FP32 linear | 1536 | 1548 | .363 | .606 |
| Oracle | — | — | .609 | .888 |

Binary1-LS2-int4 recovers about **83% of the incremental FP32 semantic gain**
over dense retrieval.

For PQ64, the 65.5 KB number is **not mandatory persistent state per concept**.
A concept can persist as the 1,548-byte FP32 linear head plus one shared PQ
codebook; the 64×256 FP32 lookup table can be materialized when that concept is
activated. The native PQ microbenchmark uses the pre-materialized LUT.

### Reviewer-controlled semantic HNSW benchmark

The systems benchmark executes the real Binary1-LS2-int4 predicates **inside
timed HNSW traversal**. Dense similarity navigates the graph; semantic programs
only decide whether a visited node may enter the valid result beam. Invalid nodes
remain traversable.

Protocol:

- strict first held-out split of the 44,072-product source catalog (~15.4k indexed items),
- **1,000 queries per predicate set**,
- **3 predetermined predicate sets**, each with 3 active predicates,
- K=50, EF=128, M=24, efConstruction=200,
- eligibility fractions 50%, 20%, 10%, 5%, 2%,
- traversal truth = brute-force dense top-K among items passing the **same compiled predicate**.

The completed selectivity-aware over-fetch run sweeps 0.75×, 1×, 1.5× and 2×
`K/selectivity`. The table below shows the largest tested budget, averaged over
the three predicate sets:

| Eligible fraction | Live Traversal Recall@50 | 2× over-fetch recall | Live ms | 2× over-fetch ms |
|---:|---:|---:|---:|---:|
| 50% | .9833 | .7923 | 2.23 | 1.61 |
| 20% | .9826 | .7098 | 4.58 | 3.46 |
| 10% | .9794 | .7180 | 6.79 | 5.95 |
| 5% | .9752 | .7482 | 9.82 | 10.27 |
| 2% | .9778 | .8420 | 14.57 | 20.14 |

**None of the 15 predicate-set × selectivity conditions reached live traversal
recall within 0.005 at any completed over-fetch point.** Therefore the repository
does **not** claim a matched-recall speedup from this completed sweep. At 50–10%
eligibility, over-fetch is faster but materially lower recall; at 5% and 2%, live
traversal is both faster and substantially higher recall than the largest tested
2× budget.

Across individual conditions, live Traversal Recall@50 is **0.955–0.992**. Exact
live/materialized result-ID parity is used only as a correctness check.

The incremental cost of executing the compiled predicates, measured as live
minus the same custom traversal with materialized semantics, is approximately
**96–134 ns per predicate invocation**. The scalar 384-D normalized-dot kernel on
the same Intel Xeon 2.20 GHz CPU measures **617 ns**, so one predicate invocation
adds roughly **0.16–0.22 dense-distance equivalents** in this prototype.

The public reviewer harness now extends the over-fetch sweep to
`0.75,1,1.5,2,3,4,6,8 × K/selectivity` to locate a genuine matched-recall point.
Those >2× results have **not yet been reported** in the paper.

### Persistent semantic payload

For a catalog with `N` items and `C` learned concepts, persistent payload is:

```text
N * item_bytes + C * stored_program_bytes
```

Illustrative 5M-item / 100k-concept payload:

| Method | Item payload | Persistent programs/store | Total |
|---|---:|---:|---:|
| **Binary1-LS2-int4** | 280.0 MB | 21.6 MB | **301.6 MB** |
| PQ64 + compact linear heads | 320.0 MB | ~155.2 MB | ~475.2 MB |
| RSA2 sparse LUT | 480.0 MB | 58.0 MB | 538.0 MB |
| FP32 linear | 7,680.0 MB | 154.8 MB | 7,834.8 MB |

PQ64 includes one shared ~0.4 MB float codebook in the ~155.2 MB store figure.
This accounting deliberately does not charge a permanent 65.5 KB LUT for every
PQ concept. Representation payload excludes HNSW graph edges, containers,
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

If ANN remains external:

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

## Reviewer HNSW reproduction

Public Colab:

[`notebooks/rsa_semantic_hnsw_reviewer_colab.ipynb`](notebooks/rsa_semantic_hnsw_reviewer_colab.ipynb)

Command-line run:

```bash
python -m experiments.semantic_hnsw_reviewer_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_reviewer \
  --queries 1000
```

The optimized harness:

1. exports only the HNSW-required held-out assets;
2. verifies normalized retrieval vectors;
3. derives all five semantic selectivity gates;
4. builds HNSW once per predicate set instead of once per over-fetch point;
5. computes brute-force dense scores once per query;
6. sweeps live traversal, the materialized custom control, and all over-fetch budgets;
7. writes `raw.csv`, `summary.csv`, `same_run_pairs.csv`, `matched_recall.csv`, `gates.csv`, and `environment.json`.

The completed 2× aggregate used by the manuscript is
[`paper/icml/data/semantic_hnsw_reviewer_2x_summary.csv`](paper/icml/data/semantic_hnsw_reviewer_2x_summary.csv).

Key native files:

```text
rust/semantic_engine/src/bin/
  actual.rs                  # real-code finalist microkernel benchmark
  sidecar.rs                 # portable candidate-side executor
  semantic_hnsw.rs           # traversal prototypes / ablations
  semantic_hnsw_live.rs      # earlier controlled live traversal harness
  semantic_hnsw_reviewer.rs  # single-build reviewer benchmark
  dot_cost.rs                # same-kernel 384-D dot benchmark
```

## Interactive fashion demo

The demo shows held-out fashion images, parses exact vs soft intent, compares
Binary1-LS2-int4 with PQ64 and other baselines, and keeps systems benchmark
latency separate from Python UI latency.

```bash
pip install -e ".[demo,benchmark]"
```

Colab:

[`notebooks/rsa_fashion_search_demo_colab.ipynb`](notebooks/rsa_fashion_search_demo_colab.ipynb)

Core demo code:

[`demos/fashion_app.py`](demos/fashion_app.py)

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
pytest -q
python -m experiments.binary_bbq_predicates --config configs/binary_bbq.yaml
python -m experiments.semantic_hnsw_reviewer_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_reviewer \
  --queries 1000
cargo test --release --manifest-path rust/semantic_engine/Cargo.toml
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for artifact details.

## Research status

The current evidence is controlled and single-domain. The strongest supported
thesis is:

> **Soft semantic predicates can be compiled into tiny reusable programs over a
> shared low-bit item substrate and executed cheaply enough to participate
> directly in ANN traversal.**

The most important next experiments are the extended matched-recall over-fetch
frontier, million-scale graph/QPS measurements, external datasets and stronger or
human semantic judgments, and representation learning that makes semantic
predicates more locally linear before compilation.
