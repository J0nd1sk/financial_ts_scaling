#!/bin/bash
# Run Phase 6C A100 Stage 1 Baseline Experiments
# 12 models: 3 budgets × 4 horizons

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=================================================="
echo "Phase 6C A100: Stage 1 Baseline Experiments"
echo "12 models: 2M/20M/200M × H1/H2/H3/H5"
echo "=================================================="

# Array of experiment scripts
EXPERIMENTS=(
    "s1_01_2m_h1"
    "s1_02_20m_h1"
    "s1_03_200m_h1"
    "s1_04_2m_h2"
    "s1_05_20m_h2"
    "s1_06_200m_h2"
    "s1_07_2m_h3"
    "s1_08_20m_h3"
    "s1_09_200m_h3"
    "s1_10_2m_h5"
    "s1_11_20m_h5"
    "s1_12_200m_h5"
)

# Track results
RESULTS_FILE="outputs/phase6c_a100/s1_summary.json"

echo "[" > "$RESULTS_FILE"
FIRST=true

for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running: $exp"
    echo "=================================================="

    python experiments/phase6c_a100/${exp}.py 2>&1 | tee outputs/phase6c_a100/${exp}/log.txt

    # Append to summary (skip comma for first entry)
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$RESULTS_FILE"
    fi

    # Extract key metrics
    cat outputs/phase6c_a100/${exp}/results.json >> "$RESULTS_FILE"

    # Thermal pause between experiments
    echo ""
    echo "Cooling pause (30s)..."
    sleep 30
done

echo "]" >> "$RESULTS_FILE"

echo ""
echo "=================================================="
echo "All experiments complete!"
echo "Summary saved to: $RESULTS_FILE"
echo "=================================================="
