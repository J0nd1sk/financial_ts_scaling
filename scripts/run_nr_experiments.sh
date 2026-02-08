#!/bin/bash
# Run Noise-Robust Training experiments (18 experiments)
#
# Priorities:
#   NR-P1: Bootstrap Loss (6 experiments) - Blend predicted labels
#   NR-P2: Co-teaching (6 experiments) - Dual networks
#   NR-P3: Forward Correction (4 experiments) - Transition matrix
#   NR-P4: Confidence Learning (2 experiments) - Identify mislabeled samples

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Noise-Robust Training Experiments (18 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4; do
    echo "--- Running NR-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "NR-P${p}" $DRY_RUN
    echo ""
done

echo "=== Noise-Robust Training experiments complete at $(date) ==="
