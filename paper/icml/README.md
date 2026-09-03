# ICML-style LaTeX manuscript

This directory contains the conference-style LaTeX version of **Random Semantic Algebra: Compiling Latent Search Predicates into Low-Bit Programs**.

It includes:

- two-column ICML-like US Letter layout;
- full mathematical formulation;
- Algorithm 1: residual Newton-boosted predicate compilation;
- Algorithm 2: query-time exact-filter + semantic-program execution;
- TikZ architecture and predicate diagrams;
- PGFPlots result figures generated directly from the reported experiment values;
- publication tables, related work, limitations, impact statement, references, and appendix.

## Build

```bash
cd paper/icml
pdflatex main.tex
pdflatex main.tex
```

The local `rsa_icml_like.sty` is self-contained and closely follows ICML's two-column geometry for a reproducible preprint. For an actual ICML submission, use the official conference `icml2026.sty`/author kit and anonymize the author block as required by the conference.

The current manuscript is intentionally conservative about the evidence: the mechanism benchmark is exploratory, and the independent-teacher search result is reported as a single-query pilot rather than a production-scale claim.
