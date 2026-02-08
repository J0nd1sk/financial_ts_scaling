#!/bin/bash
# Run all Advanced Embedding experiments (AE-P1 through AE-P5)
# Total: 21 experiments

set -e  # Exit on error

cd "$(dirname "$0")/.."

echo "=== Advanced Embedding Experiments ==="
echo "Starting at $(date)"
echo ""

for priority in AE-P1 AE-P2 AE-P3 AE-P4 AE-P5; do
    echo ">>> Running $priority at $(date)"
    caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority "$priority"
    echo "<<< Completed $priority at $(date)"
    echo ""
done

echo "=== All AE experiments complete at $(date) ==="
