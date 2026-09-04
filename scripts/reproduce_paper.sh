#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e .
pytest -q

# Mechanism benchmark used for core representation / ablation evidence.
python src/random_semantic_algebra_full.py

# Larger independent-teacher search benchmark.
python -m experiments.large_scale_search --config configs/large_scale.yaml
