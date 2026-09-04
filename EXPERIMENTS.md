# RSA paper experiments

This file tracks the experiments added in response to critical review of the ICML-style draft.

## One-click reviewer suite

Open:

https://colab.research.google.com/github/hanialshater/ras/blob/main/notebooks/rsa_reviewer_experiments_colab.ipynb

Start with `FULL_QUALITY = False`; then rerun with `FULL_QUALITY = True`.

The quality suite compares:

- dense compound-query retrieval;
- zero-shot MiniLM concept-name vectors;
- zero-shot MiniLM positive-minus-negative prompt vectors;
- 4-bit sparse LUTs with no rotation;
- 4-bit sparse LUTs with PCA rotation;
- 4-bit sparse LUTs with random orthogonal rotation (RSA);
- PQ64 (64 B/item) with a compiled linear LUT head;
- the FP32 linear semantic proxy;
- a 384 -> 64 -> 8 MLP ceiling;
- an oracle ordering.

All supervised methods use strict fit/calibration/test splits. Query templates are generated from fit labels only. The script also repeats the composition-depth analysis after stratifying by conjunction prevalence.

## Rust systems benchmark

The fairness-focused benchmark is:

`rust/semantic_engine/src/bin/fair.rs`

It uses randomized FP32 catalog values and randomized selected RSA coordinates, and compares a fused FP32 linear head, a fused PQ64 LUT head, and packed 4-bit RSA in item-major and coordinate-major layouts. Candidate indices are random samples from a resident catalog to approximate post-ANN semantic filtering.

### 5M-item capacity

| Representation | Bytes / item | 5M items |
|---|---:|---:|
| FP32 MiniLM 384D | 1,536 B | 7.68 GB |
| RSA 384 x 4-bit | 192 B | 0.96 GB |
| PQ64 | 64 B | 0.32 GB |

### Preliminary CI throughput

GitHub Actions, x86_64, one process / one scoring thread, 500k resident items, median of 3 runs, 100k random candidates. These numbers are a reproducible microbenchmark, not yet the paper's final hardware result.

| Predicates in compound query | FP32 linear (M cand/s) | PQ64 LUT (M cand/s) | Best RSA (M cand/s) | RSA / FP32 | RSA / PQ64 | Best RSA layout |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.232 | 20.218 | 30.535 | 13.68x | 1.51x | coordinate-major |
| 2 | 1.990 | 11.464 | 16.122 | 8.10x | 1.41x | coordinate-major |
| 4 | 1.725 | 4.309 | 8.302 | 4.81x | 1.93x | coordinate-major |
| 8 | 1.077 | 2.164 | 3.317 | 3.08x | 1.53x | item-major |

The layout crossover is useful: sparse coordinate-major access is strongest for 1--4 predicates, whereas item-major locality becomes better for 8 predicates. A blocked/hybrid layout is therefore a natural next systems optimization.

The earlier exploratory Rust benchmark also tested explicit union-of-coordinates fusion and i16 LUTs. Neither improved over the simpler fixed F32 executor on the first CI run, so they should not be assumed to help without further work.

## Composition-depth confound

Reanalysis of the existing 600 query-split cases confirms the reviewer's concern. At 20% retention, the raw RSA-vs-dense recall gain rises with the number of latent predicates, but conjunction prevalence falls at the same time:

| Latent predicates | Mean pool prevalence | Raw RSA recall delta |
|---:|---:|---:|
| 1 | 0.577 | +0.076 |
| 2 | 0.392 | +0.129 |
| 3 | 0.270 | +0.156 |
| 4 | 0.186 | +0.216 |

Within prevalence quartiles the monotonic depth trend disappears. An exploratory OLS with heteroscedasticity-robust errors, using logit prevalence as a covariate, gives a depth coefficient of approximately -0.0015 (p = 0.816). The paper should therefore remove the claim that RSA's advantage intrinsically increases with composition depth.

## What remains before changing the paper's central claim

1. Run the full reviewer quality suite and establish where random RSA sits relative to zero-shot, no-rotation, PCA, PQ64, linear, and MLP baselines.
2. Repeat the Rust benchmark on the Colab/target CPU with 5 repeats and record CPU model, compiler version, and thread count.
3. Combine quality, storage, and measured throughput into a Pareto table/figure.
4. If possible, run an actual 5M-item RSA-resident benchmark and a second dataset/domain.

Until those are complete, the Rust numbers should be described as preliminary systems evidence rather than a final production claim.
