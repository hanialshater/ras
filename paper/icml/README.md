# ICML-style LaTeX manuscript

This directory contains the conference-style LaTeX version of **Random Semantic Algebra: Compiling Latent Search Predicates over Low-Bit Substrates**.

The current manuscript reflects the full 44,072-product / 600-query independent-teacher benchmark and the real native Rust execution benchmark. It covers:

- the original random 1/2/4-bit sparse boosted-LUT compiler;
- the centered **Binary1-LS2-int4** compiler (56 B/item, ~216 B/predicate);
- the strong **PQ64 compiled-linear** baseline (64 B/item, ~65.5 KB/predicate);
- FP32 and zero-shot baselines, calibrated AND/NOT composition, and strict fit/calibration/test evaluation;
- quality--item-memory--program-memory Pareto results;
- native single-threaded execution on the actual held-out codes and learned programs;
- the prevalence-confound correction for the earlier composition-depth observation;
- related work, limitations, impact statement, and reproducibility details.

The framing is intentionally conservative: random rotation and 4-bit quantization are no longer presented as uniquely optimal, the calibrated probability product is not claimed as a new mathematical operator, and the BBQ-inspired binary compiler is explicitly distinguished from exact Lucene/Elasticsearch BBQ.

## Build

```bash
cd paper/icml
pdflatex main.tex
pdflatex main.tex
```

The local `rsa_icml_like.sty` is self-contained and follows an ICML-like two-column geometry for a reproducible preprint. For an actual ICML submission, use the official conference author kit and anonymize the author block as required.
