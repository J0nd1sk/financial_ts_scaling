#!/bin/bash
# HPO Batch Runner - Runs all budgets for a given tier using improved methodology
#
# Usage:
#   ./scripts/run_hpo_batch.sh a100       # Run 2M, 20M, 200M for tier a100
#   ./scripts/run_hpo_batch.sh a50        # Run 2M, 20M, 200M for tier a50
#   ./scripts/run_hpo_batch.sh a100 20M   # Start from 20M budget (skip 2M)
#
# Run in tmux with:
#   caffeinate -i ./scripts/run_hpo_batch.sh a100 2>&1 | tee outputs/hpo_a100_batch.log

set -e  # Exit on error

TIER=${1:-a100}
START_BUDGET=${2:-2M}

# Validate tier
if [[ ! "$TIER" =~ ^(a20|a50|a100|a200)$ ]]; then
    echo "Error: Invalid tier '$TIER'. Use a20, a50, a100, or a200."
    exit 1
fi

# Check data exists
DATA_FILE="data/processed/v1/SPY_dataset_${TIER}_combined.parquet"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: Data file not found: $DATA_FILE"
    exit 1
fi

echo "============================================================"
echo "HPO Batch Runner"
echo "============================================================"
echo "Tier: $TIER"
echo "Data: $DATA_FILE"
echo "Start budget: $START_BUDGET"
echo "Started: $(date)"
echo "============================================================"
echo ""

# Determine which budgets to run
BUDGETS=()
case $START_BUDGET in
    2M)  BUDGETS=(2M 20M 200M) ;;
    20M) BUDGETS=(20M 200M) ;;
    200M) BUDGETS=(200M) ;;
    *)
        echo "Error: Invalid start budget '$START_BUDGET'. Use 2M, 20M, or 200M."
        exit 1
        ;;
esac

# Create output directory
OUTPUT_DIR="outputs/phase6c_${TIER}"
mkdir -p "$OUTPUT_DIR"

# Run HPO for each budget
for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Starting HPO: $BUDGET budget, tier $TIER"
    echo "Time: $(date)"
    echo "============================================================"
    echo ""

    # Run HPO with improved template
    ./venv/bin/python experiments/templates/hpo_template.py \
        --budget "$BUDGET" \
        --tier "$TIER" \
        --horizon 1 \
        --trials 50

    echo ""
    echo "Completed: $BUDGET budget"
    echo "Time: $(date)"
    echo ""
done

echo ""
echo "============================================================"
echo "HPO Batch Complete"
echo "============================================================"
echo "Tier: $TIER"
echo "Budgets completed: ${BUDGETS[*]}"
echo "Finished: $(date)"
echo "Results in: $OUTPUT_DIR"
echo "============================================================"

# Run cross-budget validation if all budgets completed
if [[ "${#BUDGETS[@]}" -eq 3 ]]; then
    echo ""
    echo "Running cross-budget validation..."
    ./venv/bin/python scripts/validate_cross_budget.py --tier "$TIER" --horizon 1
fi
