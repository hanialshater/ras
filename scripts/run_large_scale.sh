#!/usr/bin/env bash
set -euo pipefail
python -m experiments.large_scale_search --config configs/large_scale.yaml "$@"
