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

These are pilot results on one dataset and one detailed compound-query sweep, not yet production or multi-query evidence.

## Repository layout

- `src/random_semantic_algebra_full.py` - exact runnable snapshot of the original reproducible experiment.
- `src/rsa_v2.py` - exact runnable snapshot of the extended RSA v2 implementation: boosted LUTs, pair interactions, calibration, composition, whitening/random-dictionary ablations, teacher compilation and distillation.
- `experiments/independent_teacher_search.py` - readable MiniLM-retrieval / CLIP-image-teacher search experiment with retention sweeps.
- `paper/Random_Semantic_Algebra.md` - current paper source, including algorithms, pedagogical Mermaid figures, results, systems interpretation, and limitations.
- `requirements.txt` - Python dependencies.

The two exact implementation snapshots are stored as self-extracting gzip payloads so the exact research-session code is preserved byte-for-byte while remaining executable through the repository connector.

## Core idea

Offline, an expensive teacher (VLM, human labels, or another semantic model) defines reusable concepts such as `minimalist`, `office appropriate`, or `technical/sporty`. Each concept is compiled into a small sparse program over the universal 4-bit item code. At query time, exact catalog filters remain ordinary filters, while latent predicates are combined using calibrated log-probability composition and evaluated with a small number of LUT reads and additions.

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
pip install -r requirements.txt

# Original mechanism experiment
python src/random_semantic_algebra_full.py

# Independent-teacher search pilot
python -m experiments.independent_teacher_search
```

For the search pilot, a GPU is recommended because CLIP image embeddings are generated once offline. The RSA predicate compiler and online scoring are lightweight CPU operations in this prototype.

## Run the experiment series in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hanialshater/ras/blob/claude/repo-check-rrudyh/experiments/rsa_lab.ipynb)

`experiments/rsa_lab.ipynb` runs the seven-stage plan in `experiments/COLAB_PLAN.md` on a T4: kill tests, probe ceilings, a 200-query composition benchmark, a LUT budget sweep, an optional VLM teacher check, a throughput comparison, and four improvement experiments. It caches embeddings to Drive and writes `results/summary.md`. The notebook is generated from `experiments/rsa_lab.py`; set `RSA_SYNTHETIC=1 RSA_FAST=1` to smoke-test it without downloads.

## Research status

Exploratory research prototype. The next decisive experiment is a **50-200 compound-query benchmark** with independent semantic supervision, repeated seeds, aggregate quality-vs-candidate-budget curves, direct-conjunction ceilings, and measured packed CPU/SIMD latency.
