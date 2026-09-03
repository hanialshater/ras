# Random Semantic Algebra (RSA)

Random Semantic Algebra explores whether expensive semantic supervision can be compiled into tiny reusable programs over a fixed low-bit item representation for retrieval and search.

## Current result

The current prototype uses a 384-dimensional semantic representation, a fixed orthogonal random rotation, and 4-bit quantization (192 bytes/item). Sparse residual-boosted LUT programs recover much of a full-precision semantic classifier and support calibrated AND/NOT composition.

In the independent-teacher search pilot:

- retrieval: MiniLM title embeddings
- latent semantic teacher: CLIP image semantics
- executor: RSA 4-bit codes + sparse LUT programs
- query example: `minimalist black office shoes not sporty`

At a 40% retained candidate budget, RSA increased recall of teacher-defined relevant items from 0.420 to 0.652 while improving purity from 0.492 to 0.763. At a 20% budget, recall improved from 0.246 to 0.333 and purity from 0.586 to 0.793.

These are pilot results on one query / one dataset, not yet a production or multi-query benchmark.

## Repository layout

- `src/random_semantic_algebra_full.py` — original reproducible experiment
- `src/rsa_v2_lib.py` — extended RSA implementation (boosted LUTs, interactions, calibration, composition, ablations)
- `notebooks/RSA_Independent_Teacher_Search_Experiment.ipynb` — independent MiniLM retrieval vs CLIP image-teacher search experiment
- `notebooks/random_semantic_algebra_v2_colab.ipynb` — full ablation / F1 experiment suite
- `paper/Random_Semantic_Algebra_Paper.pdf` — current paper draft with algorithms and pedagogical figures
- `paper/Random_Semantic_Algebra_Paper.docx` — editable paper draft

## Core idea

Offline, an expensive teacher (VLM, human labels, or another semantic model) defines reusable concepts such as `minimalist`, `office appropriate`, or `technical/sporty`. Each concept is compiled into a small sparse program over the universal 4-bit item code. At query time, exact catalog filters remain ordinary filters, while latent predicates are combined using calibrated log-probability composition and evaluated with a small number of LUT reads and additions.

## Status

Exploratory research prototype. The next decisive experiment is a 50–200 query benchmark with independent semantic supervision, repeated seeds, quality-vs-candidate-budget curves, and measured CPU/SIMD latency.
