# RSA Colab experiment series

One notebook, run top to bottom on a T4. Every stage reads from the cache built in Stage 0 and appends rows to `results/*.csv` on Drive, so stages can be re-run independently after a runtime reset. Each stage ends with a decision rule so the loop knows what to do next without a human.

Prerequisite: `src/rsa_v2.py` is corrupt, so Stage 0 must include a fresh implementation of the boosted-LUT compiler (`build_substrate`, `fit_boosted_lut`, `add_pair_interactions`, `score_boosted`, `fit_scalar_calibrator`). About 150 lines of NumPy following Algorithm 1 in the paper.

---

## Stage 0 - Cache everything once

Data: `ashraq/fashion-product-images-small`, all ~44k rows, drop rows with missing colour/category.

Embeddings to cache (`.npy`, float16, on Drive):

| name | model | dim | purpose |
|---|---|---:|---|
| `minilm` | all-MiniLM-L6-v2 on title | 384 | current substrate |
| `bge` | bge-small-en-v1.5 on title | 384 | second text substrate, same dim |
| `clip_txt` | CLIP ViT-B/32 text on title | 512 | text side of the teacher space |
| `clip_img` | CLIP ViT-B/32 image | 512 | teacher space |

Teacher labels: extend the 8 concepts to 16 (add `streetwear`, `bohemian`, `preppy`, `edgy`, `festive`, `outdoor`, `vintage`, `luxury`). Score = mean positive prompt minus mean negative prompt in CLIP image space. Cache raw scores, not binary labels, so prevalence can be varied later.

Splits: 5 seeds x (fit 52% / cal 13% / test 35%). Cache index arrays.

Also cache: 5 random orthogonal 384x384 rotations, one PCA rotation per substrate.

Budget: CLIP images ~10 min on T4, everything else under a minute.

---

## Stage 1 - Kill tests on the existing claims

Cheap, and each one can delete a sentence from the paper.

**1a. Is the rotation doing anything?**
Substrate = {identity, random orthogonal, PCA} x {4-bit quantile}. Fit sparse boosted LUT (K=28, 4 pairs) on the 14 structured-facet concepts and the 16 latent ones. 5 seeds.
Decision: if identity is within 1 sd of random, drop "random" from the method name and the substrate story. If PCA wins, the story becomes "quantized principal coordinates".

**1b. Is the pair gain real?**
Same runs, pairs in {0, 2, 4, 8}. Report mean and sd across seeds.
Decision: if +pairs is inside seed noise at 30k items, remove pairs from the default method.

**1c. Fair sparse baselines at equal K.**
For K in {16, 28, 64}: (i) L1-logistic on FP32 substrate with exactly K nonzeros, (ii) L1-logistic on the 16-bin one-hot expansion restricted to K coordinates, (iii) boosted LUT. Same seeds.
Decision: if (ii) matches (iii), the Newton boosting contributes nothing over a sparse GAM fit by any solver, and the paper should say so.

---

## Stage 2 - Probe ceilings: which substrate can even see the concept?

For each of the 16 latent concepts, fit a plain logistic probe on FP32 `minilm`, `bge`, `clip_txt`, `clip_img`. Report test F1 and AP with 5-seed CIs. `clip_img` is the teacher's own space and is the ceiling.

Decision: the gap between `clip_img` and the best text substrate is the cross-modal bound. Everything RSA does on text is capped by it. If `bge` beats `minilm` by more than 0.05 F1, switch the substrate for all later stages. If `clip_txt` on titles beats both, note that the "independent teacher" is less independent than claimed.

---

## Stage 3 - The decisive one: composition in query space vs predicate space

Auto-generate 200 compound queries from templates:
`[latent+] [colour] [category] [latent+]? [not latent-]?`
sampled so that each query has at least 30 teacher-relevant items in its filtered ANN pool of 500. Truth = conjunction of teacher labels inside the exact-filtered pool.

Arms, all scored on the same pool:

| arm | what it is | needs labels? |
|---|---|---|
| A | one dense vector for the full query string | no |
| B | decomposed: per-concept prompt similarity in the substrate space, z-scored on fit set, fused by log-sigmoid sum, negation by sign flip | no |
| C | FP32 logistic probe per concept + same fusion | yes |
| D | RSA sparse LUT program per concept + same fusion (the paper) | yes |
| E | direct conjunction probe trained per query on FP32 (upper bound for composition) | yes, per query |

Metrics per query: recall and purity at retention {5, 10, 20, 40}%, nDCG@20 against teacher score. Aggregate across 200 queries with bootstrap CIs. Plot the retention curves with bands.

Decision tree:
- B is within CI of C and D: supervision adds nothing on this data. RSA's value, if any, is compute only. Go to Stage 6 before anything else.
- C clearly beats D (more than 0.05 recall at 20%): the 4-bit sparse program is losing too much. Go to Stage 4 and find the K where D catches C.
- D is within CI of C and both beat B: RSA is a valid cheap executor of supervision. Go to Stage 4 for the Pareto curve, then Stage 7.
- E is far above C: composition itself is the bottleneck, not the executor. Prioritise the correlation-aware fusion in Stage 7.

---

## Stage 4 - Budget sweep (only if D survived Stage 3)

Grid: K in {8, 16, 24, 32, 48, 64, 96, 192, 384}, bits in {2, 4, 8}, pairs in {0, 2}. Report F1, AP, and the Stage 3 recall@20% for the 200 queries, against (a) LUT operations and (b) bytes per item. Also refit the K=24 program on fit sets of 2k, 4k, 8k, 16k items to test the sample-size explanation for the weak pair gain.

Output: two Pareto plots. Decision: pick the smallest configuration within 0.01 of the FP32 probe as the default.

---

## Stage 5 - Is the teacher the story?

Take a 2,000-item subset and 4 concepts (`minimalist`, `office_appropriate`, `technical_sporty`, `quiet_luxury`). Label them with a VLM instead of CLIP prompt differences. Options in Colab: an open VLM such as Qwen2-VL-2B or SmolVLM on T4, or an API call on the subset. Measure agreement (Cohen's kappa) between CLIP labels and VLM labels, then re-run arms A to D on VLM truth for the queries that use those concepts.

Decision: if kappa is below 0.4, the CLIP teacher is noise on those concepts and the paper's pilot numbers should be re-labelled "CLIP-prompt agreement", not "semantic relevance". If arm rankings change under VLM truth, report both.

---

## Stage 6 - Systems reality check

The hidden competitor is not dense retrieval. It is the FP32 or int8 probe from arm C, which costs one 384-dim dot product per concept, the same as a similarity score. RSA has to beat that.

Implement: (i) nibble-packed uint8 codes, (ii) LUT scoring in NumPy gather, (iii) LUT scoring in Numba or a torch gather kernel, (iv) int8 dot product via NumPy/torch. Measure items per second on CPU for 1 concept and for 3 composed concepts, on 100k items.

Decision: if int8 dot products are within 2x of LUT scoring in the best implementation, drop the "hardware-cheap" claim and reframe RSA as a memory story (192 bytes for any number of concepts vs one byte per concept per item for stored scores). If LUT scoring is 5x faster or more, keep the systems section and put the number in it.

---

## Stage 7 - Improving the idea (pick by Stage 3 outcome)

**7a. Shared coordinate dictionary.** Select coordinates jointly across all 16 concepts (multi-task greedy) so programs read from a shared subset of 48 or 64 coordinates. Measures: accuracy loss vs per-concept selection, and distinct bytes read per composed query. This is the version that could actually be fast.

**7b. Correlation-aware fusion.** If Stage 3 shows E far above C: fit one pairwise term per concept pair on the calibration split (a 16x16 matrix of corrections) and add it to the log-prob sum. Closes the independence-assumption gap without training per-query models.

**7c. Zero-shot compile.** Compile a program with no labels: pseudo-label the fit set with the concept's prompt similarity in `clip_txt` or `bge` space, then run Algorithm 1 on the pseudo-labels. Compare to the supervised program. If within 0.05 F1, new concepts become instant, which is the strongest product argument for the method.

**7d. Few-shot personal predicates.** For 200 synthetic "users" defined by 30 liked items, compile a program from those positives against random negatives. Measure held-out precision@20 vs K. This is the case where per-item precomputation is impossible and RSA has a structural reason to exist.

---

## Reporting

One `results/summary.md` regenerated at the end of every run: seed-averaged tables with CIs, the Stage 3 retention plot, the Stage 4 Pareto plot, the Stage 6 throughput table, and the decision taken at each branch. The paper gets rewritten from that file, not from memory.

## Rough compute

| stage | T4 time |
|---|---|
| 0 | 15 min |
| 1 | 10 min |
| 2 | 2 min |
| 3 | 20 min (arm E is the slow one) |
| 4 | 30 min |
| 5 | 30 to 60 min depending on the VLM |
| 6 | 5 min |
| 7 | 20 min |

About three hours end to end, which fits in one Colab session with the cache on Drive.
