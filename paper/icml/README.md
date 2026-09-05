# ICML-style LaTeX manuscript

This directory contains the conference-style LaTeX manuscript **Compiled Semantic Predicates for Approximate Nearest Neighbor Search**.

The manuscript covers both the semantic-quality benchmark and the integrated ANN systems result:

- random 1/2/4-bit sparse boosted-LUT compilers as historical/mechanism baselines;
- the centered **Binary1-LS2-int4** compiler (56 B/item, ~216 B/predicate);
- the strong **PQ64 compiled-linear** baseline (64 B/item, 1,548 B persistent head, ~65.5 KB active materialized LUT);
- FP32 and zero-shot baselines, calibrated AND/NOT composition, and strict fit/calibration/test evaluation;
- quality--item-memory--program-memory Pareto results with persistent versus active state separated;
- native single-threaded candidate-scoring microkernels;
- **live Binary1-LS2-int4 predicates executed inside HNSW traversal**;
- a reviewer-controlled selectivity-aware over-fetch comparison with 1,000 queries for each of three predetermined predicate sets;
- Traversal Recall@50 from 50% to 2% semantic eligibility;
- the prevalence-confound correction for the earlier composition-depth observation;
- limitations, related work, impact statement, and reproducibility details.

The framing is intentionally conservative. Random rotation and 4-bit quantization are not presented as uniquely optimal; the calibrated probability product is not claimed as a new mathematical operator; the BBQ-inspired binary compiler is explicitly distinguished from exact Lucene/Elasticsearch BBQ; PQ's 65.5 KB active LUT is not treated as mandatory persistent storage; and exact live/materialized ID parity is used only as a correctness check.

The completed reviewer sweep through `2 × K/selectivity` found **no matched-recall over-fetch point** within the predeclared 0.005 tolerance in any of the 15 predicate-set × selectivity conditions. The manuscript therefore does not claim a universal matched-recall speedup from that run. The checked-in aggregate is:

```text
paper/icml/data/semantic_hnsw_reviewer_2x_summary.csv
```

The current public harness extends the over-fetch range through 3×, 4×, 6× and 8× so that a genuine matched-recall frontier can be measured:

```bash
python -m experiments.semantic_hnsw_reviewer_sweep \
  --config configs/binary_bbq.yaml \
  --output-dir results/semantic_hnsw_reviewer \
  --queries 1000
```

Public Colab:

```text
notebooks/rsa_semantic_hnsw_reviewer_colab.ipynb
```

## Build

```bash
cd paper/icml
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The local `rsa_icml_like.sty` is self-contained and follows an ICML-like two-column geometry for a reproducible preprint. For an actual ICML submission, use the official conference author kit and anonymize the author block as required.
