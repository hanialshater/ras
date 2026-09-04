# Random Semantic Algebra (RSA)

Random Semantic Algebra explores whether expensive semantic supervision can be **compiled into tiny reusable programs over a fixed low-bit item representation** for retrieval and search.

## Current result

The prototype uses a 384-dimensional semantic representation, one fixed orthogonal random rotation, and 4-bit quantization (**192 bytes/item**). Sparse residual-boosted LUT programs recover much of a full-precision semantic classifier and support calibrated `AND` / `NOT` composition.

Mechanism benchmark:

- FP32 linear classifier: **0.9331 mean F1**
- all-coordinate 4-bit LUT compilation: **0.9334**
- sparse 28-unary + 4-pair RSA program: **~0.9015**
- calibrated 2-way composition: **0.814 F1 / 0.852 AP**
- calibrated 3-way composition: **~0.747 F1 / 0.780 AP**

Independent-teacher search pilot:

- retrieval / item substrate: MiniLM title embeddings
- latent semantic teacher: CLIP image semantics
- executor: RSA 4-bit codes + sparse LUT programs
- query: `minimalist black office shoes not sporty`

At a 40% retained candidate budget, RSA increased teacher-relevant recall from **0.420 -> 0.652** while purity rose from **0.492 -> 0.763**. At a 20% budget, recall improved **0.246 -> 0.333** and purity **0.586 -> 0.793**.

These remain pilot results; the new large-scale harness below is intended to replace the single-query evidence with a multi-query, multi-seed benchmark.

## Research code structure

The exact historical implementation remains in `src/rsa_v2.py` so existing numbers are reproducible. New work should use the modular API under `src/ras/`:

```text
src/ras/
  substrate.py      # random rotation, quantization, geometry
  predicates.py     # boosted LUTs, LLR factors, pair interactions
  calibration.py    # scalar calibration
  composition.py    # calibrated AND / NOT algebra
  teachers.py       # independent CLIP semantic supervision
  retrieval.py      # MiniLM retrieval embeddings
  queries.py        # fit-only compound-query benchmark generation
  metrics.py        # ranking metrics + bootstrap confidence intervals
  cache.py          # reusable embedding cache
  repro.py          # environment + git manifests
  config.py         # YAML experiment configuration
```

Paper experiments live separately from the library:

- `experiments/independent_teacher_search.py` - original readable search pilot.
- `experiments/large_scale_search.py` - paper-grade multi-query / multi-seed benchmark.
- `configs/smoke.yaml` - quick 8k-product / 20-query validation run.
- `configs/large_scale.yaml` - full ~44k-product / 600 query-seed benchmark.
- `tests/test_smoke.py` - synthetic regression/smoke tests.
- `scripts/reproduce_paper.sh` - mechanism + large-scale reproduction entry point.

## Large-scale paper benchmark

The default large-scale config uses:

- **all ~44k products** in Fashion Product Images Small;
- **3 strict seeds** (`7, 17, 27`);
- **200 compound queries per seed** generated using fit data only;
- a **5,000-candidate ANN pool** before exact filtering and semantic pruning;
- MiniLM title embeddings for retrieval;
- CLIP image semantics as an independent latent teacher;
- RSA vs dense-only vs a full-precision linear semantic proxy vs an oracle upper bound;
- retention budgets from 100% down to 5%;
- bootstrap confidence intervals and paired deltas versus dense retrieval.

Expensive MiniLM/CLIP embeddings are cached and reused across seeds.

```bash
pip install -e .

# Validate the full pipeline first
python -m experiments.large_scale_search --config configs/smoke.yaml

# Paper-scale run
python -m experiments.large_scale_search --config configs/large_scale.yaml
```

Each run writes an immutable result directory containing:

```text
results/<run_id>/
  config.yaml
  environment.json
  headline.json
  predicate_metrics.csv
  queries.csv
  per_query.csv
  summary.csv
  paired_deltas.csv
  figures/
```

## Core idea

```text
expensive semantic supervision
          |
          v  offline
compile reusable predicates
          |
          v
fixed 192-byte/item RSA code
          |
          v  online
ANN candidates -> semantic program -> cheap pruning -> expensive ranker
```

## Quick start

```bash
pip install -e .
pytest -q

# Original mechanism experiment
python src/random_semantic_algebra_full.py

# Original independent-teacher pilot
python -m experiments.independent_teacher_search

# New paper-scale benchmark
python -m experiments.large_scale_search --config configs/large_scale.yaml

# Full experiment suite
bash scripts/reproduce_paper.sh

# ICML-style paper
cd paper/icml
pdflatex main.tex
pdflatex main.tex
```

A GPU is recommended for CLIP image embedding. RSA predicate compilation and online LUT scoring are lightweight compared with the teacher stage.

## Paper

- `paper/Random_Semantic_Algebra.md` - long-form source.
- `paper/icml/main.tex` - professional two-column ICML-style manuscript with equations, pseudocode algorithms, TikZ/PGFPlots figures, tables, impact statement, references, and appendix.
- `paper/icml/README.md` - build instructions.

## Research status

The codebase now supports the next evidence step: a **large multi-query, multi-seed independent-teacher benchmark** with strong proxy/oracle baselines and confidence intervals. Still missing for a mature submission are a second dataset/domain, direct composition ceilings at larger scale, and measured packed CPU/SIMD latency.
