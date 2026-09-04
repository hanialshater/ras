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

## Binary / BBQ-inspired kill test

Open:

https://colab.research.google.com/github/hanialshater/ras/blob/main/notebooks/rsa_binary_bbq_experiment_colab.ipynb

Start with `FULL_RUN = False`; if the smoke run is sensible, rerun with `FULL_RUN = True`.

This experiment asks whether the semantic-substrate idea survives at roughly BBQ/PQ memory scale. It compares:

- FP32 supervised linear predicates;
- PQ64 + compiled linear LUTs;
- a **BBQ-inspired** globally centered 1-bit document code with two per-item least-squares reconstruction values, evaluated with either FP32 or int4 predicate weights;
- RSA1 sparse learned predicates on centered identity bits (48 B/item);
- RSA1 sparse learned predicates on centered random-orthogonal bits (48 B/item);
- RSA2 random quantile predicates (96 B/item);
- RSA4 random quantile predicates (192 B/item);
- zero-shot MiniLM, dense, and oracle baselines.

The BBQ-inspired method is intentionally not called an exact Lucene BBQ implementation. Lucene BBQ uses additional correction terms and a specialized asymmetric bitwise estimator. Our baseline isolates the relevant design principles for semantic predicate execution: a global centroid, one stored bit per document dimension, tiny per-item correction state, and optional int4 predicate weights.

The run writes `pareto_at_20pct.csv` with recall, purity, item bytes, and approximate predicate-program bytes. It also writes `native_export_first_seed.npz`, containing real packed test codes and int4 predicate bitplanes, so a successful binary quality result can immediately be moved into the Rust popcount executor.

The decision rule is simple:

1. If RSA1 retains most of RSA4 quality, implement/benchmark the native popcount mask executor.
2. If the BBQ-inspired int4 compiled linear head dominates RSA1, treat it as the stronger binary baseline and test whether RSA earns its complexity through nonlinear predicates or sparse compositional programs.
3. If 1-bit quality collapses but RSA2 is strong, optimize the 2-bit representation instead.

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

1. Run the binary / BBQ-inspired smoke experiment and determine whether 1-bit or 2-bit is a viable semantic substrate.
2. If binary quality is viable, execute the exported real codes and real predicate bitplanes in Rust rather than synthetic codes.
3. Repeat the Rust benchmark on the target CPU with 5 repeats and record CPU model, compiler version, and thread count.
4. Combine quality, storage, predicate-program storage, and measured throughput into a Pareto table/figure.
5. If possible, run an actual 5M-item resident benchmark and a second dataset/domain.

Until those are complete, the Rust numbers should be described as preliminary systems evidence rather than a final production claim.
