#!/bin/bash
# Run Contrastive Pre-training experiments (18 experiments)
#
# Priorities:
#   CP-P1: SimCLR (6 experiments) - Contrastive loss with temperature
#   CP-P2: TS2Vec (6 experiments) - Temporal contrastive (hierarchical)
#   CP-P3: BYOL (4 experiments) - Bootstrap Your Own Latent (no negatives)
#   CP-P4: Fine-tune (2 experiments) - Fine-tune from best pretrained

set -e
cd "$(dirname "$0")/.."

PYTHON="./venv/bin/python"
RUNNER="experiments/feature_embedding/run_experiments.py"
DRY_RUN=""

[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=== Contrastive Pre-training Experiments (18 total) ==="
echo "Starting at $(date)"
echo ""

for p in 1 2 3 4; do
    echo "--- Running CP-P${p} ---"
    caffeinate $PYTHON $RUNNER --priority "CP-P${p}" $DRY_RUN
    echo ""
done

echo "=== Contrastive Pre-training experiments complete at $(date) ==="
