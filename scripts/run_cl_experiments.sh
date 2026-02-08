#!/bin/bash
# Run Curriculum Learning experiments (18 experiments)
#
# Priorities:
#   CL-P1: Loss-based (6 experiments) - Rank by per-sample loss
#   CL-P2: Confidence-based (6 experiments) - Start with high-confidence predictions
#   CL-P3: Volatility-based (4 experiments) - Low volatility periods first
#   CL-P4: Anti-curriculum (2 experiments) - Hard samples first (baseline)

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Curriculum Learning Experiments (18 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4; do
    echo "--- Running CL-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "CL-P${p}" $DRY_RUN
    echo ""
done

echo "=== Curriculum Learning experiments complete at $(date) ==="
