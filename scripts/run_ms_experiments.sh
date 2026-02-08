#!/bin/bash
# Run Multi-Scale Temporal experiments (18 experiments)
#
# Priorities:
#   MS-P1: Hierarchical Pool (6 experiments) - Pool patches at multiple scales
#   MS-P2: Multi-Patch (6 experiments) - Parallel patch sizes
#   MS-P3: Dilated Conv (4 experiments) - Dilated temporal convolutions
#   MS-P4: Cross-Scale Attention (2 experiments) - Attention between scales

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Multi-Scale Temporal Experiments (18 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4; do
    echo "--- Running MS-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "MS-P${p}" $DRY_RUN
    echo ""
done

echo "=== Multi-Scale Temporal experiments complete at $(date) ==="
