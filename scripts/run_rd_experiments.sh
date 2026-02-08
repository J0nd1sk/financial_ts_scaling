#!/bin/bash
# Run Regime Detection experiments (18 experiments)
#
# Priorities:
#   RD-P1: Volatility (6 experiments) - High/med/low vol regimes
#   RD-P2: Trend (6 experiments) - Bull/bear/sideways
#   RD-P3: Learned (4 experiments) - Cluster-based regime discovery
#   RD-P4: Regime-Gated (2 experiments) - Different model heads per regime

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Regime Detection Experiments (18 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4; do
    echo "--- Running RD-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "RD-P${p}" $DRY_RUN
    echo ""
done

echo "=== Regime Detection experiments complete at $(date) ==="
