# ICML-style LaTeX manuscript

This directory contains the conference-style LaTeX version of **Random Semantic Algebra: Compiling Latent Search Predicates over Low-Bit Substrates**.

The manuscript now covers both the semantic-quality benchmark and the integrated ANN systems result:

- random 1/2/4-bit sparse boosted-LUT compilers;
- the centered **Binary1-LS2-int4** compiler (56 B/item, ~216 B/predicate);
- the strong **PQ64 compiled-linear** baseline (64 B/item, ~65.5 KB/predicate);
- FP32 and zero-shot baselines, calibrated AND/NOT composition, and strict fit/calibration/test evaluation;
- quality--item-memory--program-memory Pareto results;
- joint memory accounting, including an illustrative 5M-item / 100k-concept comparison;
- native single-threaded candidate-scoring microkernels;
- **live Binary1-LS2-int4 predicates executed inside HNSW traversal** with the same normalized-dot geometry as the filtered-HNSW baseline;
- selectivity-dependent recall/latency from 50% to 2% semantic eligibility;
- the prevalence-confound correction for the earlier composition-depth observation;
- related work, limitations, impact statement, and reproducibility details.

The framing is intentionally conservative. Random rotation and 4-bit quantization are not presented as uniquely optimal; the calibrated probability product is not claimed as a new mathematical operator; and the BBQ-inspired binary compiler is explicitly distinguished from exact Lucene/Elasticsearch BBQ. The HNSW latency comparison is also described as a controlled single-thread prototype result rather than a universal production-speed claim.

The checked-in HNSW summary used by the paper is:

```text
paper/icml/data/semantic_hnsw_live_fair_full.csv
```

The non-notebook reproduction entry point is:

```bash
python -m experiments.semantic_hnsw_live_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_live_fair
```

## Build

```bash
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The local `rsa_icml_like.sty` is self-contained and follows an ICML-like two-column geometry for a reproducible preprint. For an actual ICML submission, use the official conference author kit and anonymize the author block as required.
