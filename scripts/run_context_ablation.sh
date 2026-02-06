#!/bin/bash
# Context ablation experiments for iTransformer and Informer
# Run with: caffeinate -dims ./scripts/run_context_ablation.sh

set -e

PYTHON="./venv/bin/python"
SCRIPT="experiments/architectures/context_ablation_nf.py"

CONTEXT_LENGTHS=(60 80 120 180 220)
MODELS=("itransformer" "informer")

echo "=========================================="
echo "Context Ablation Experiment Runner"
echo "Models: ${MODELS[*]}"
echo "Context lengths: ${CONTEXT_LENGTHS[*]}"
echo "Total runs: $((${#CONTEXT_LENGTHS[@]} * ${#MODELS[@]}))"
echo "Estimated time: ~5 hours (30 min each)"
echo "=========================================="
echo ""

for model in "${MODELS[@]}"; do
    for ctx in "${CONTEXT_LENGTHS[@]}"; do
        echo ""
        echo "[$(date '+%H:%M:%S')] Starting: ${model} @ ${ctx}d"
        echo "-------------------------------------------"
        $PYTHON $SCRIPT --model "$model" --context-length "$ctx"
        echo ""
        echo "[$(date '+%H:%M:%S')] Completed: ${model} @ ${ctx}d"
        echo ""
    done
done

echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Results in: outputs/architectures/context_ablation/"
echo "=========================================="
