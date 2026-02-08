#!/bin/bash
# Run Data Augmentation experiments (24 experiments)
#
# Priorities:
#   DA-P1: Jitter (6 experiments) - Gaussian noise
#   DA-P2: Scale (6 experiments) - Feature scaling
#   DA-P3: Mixup (6 experiments) - Sample interpolation
#   DA-P4: Time Warp (4 experiments) - DTW-based warping
#   DA-P5: Combined (2 experiments) - Best combinations

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Data Augmentation Experiments (24 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4 5; do
    echo "--- Running DA-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "DA-P${p}" $DRY_RUN
    echo ""
done

echo "=== Data Augmentation experiments complete at $(date) ==="
