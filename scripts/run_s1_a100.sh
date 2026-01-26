#!/bin/bash
#
# Phase 6C A100: Stage 1 Baseline Experiments
# 12 models: 3 budgets (2M/20M/200M) × 4 horizons (H1/H2/H3/H5)
#
# Usage: caffeinate ./scripts/run_s1_a100.sh
#
# Each experiment runs ~20-40 epochs with early stopping
# Total estimated time: 2-4 hours depending on thermal conditions
#

set -e  # Exit on error
set -o pipefail  # Catch errors in pipelines
export PYTHONUNBUFFERED=1  # Ensure real-time output

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment (disable set -e temporarily due to unalias in activate)
set +e
source venv/bin/activate
set -e

# Ensure base output directory exists
mkdir -p outputs/phase6c_a100

echo "=================================================="
echo "Phase 6C A100: Stage 1 Baseline Experiments"
echo "12 models: 2M/20M/200M × H1/H2/H3/H5"
echo "=================================================="
echo ""
date
echo ""

# Track counters
TOTAL=0
PASSED=0
FAILED=0

run_experiment() {
    local exp=$1
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "[$TOTAL/12] Running: $exp"
    echo "----------------------------------------"

    if python "experiments/phase6c_a100/${exp}.py"; then
        PASSED=$((PASSED + 1))
        echo "[$TOTAL/12] PASSED: $exp"
    else
        FAILED=$((FAILED + 1))
        echo "[$TOTAL/12] FAILED: $exp"
    fi

    # Thermal pause between experiments
    echo ""
    echo "Cooling pause (30s)..."
    sleep 30
}

# Run all 12 Stage 1 experiments
echo ""
echo "========================================"
echo "HORIZON 1 (H1) EXPERIMENTS"
echo "========================================"
run_experiment "s1_01_2m_h1"
run_experiment "s1_02_20m_h1"
run_experiment "s1_03_200m_h1"

echo ""
echo "========================================"
echo "HORIZON 2 (H2) EXPERIMENTS"
echo "========================================"
run_experiment "s1_04_2m_h2"
run_experiment "s1_05_20m_h2"
run_experiment "s1_06_200m_h2"

echo ""
echo "========================================"
echo "HORIZON 3 (H3) EXPERIMENTS"
echo "========================================"
run_experiment "s1_07_2m_h3"
run_experiment "s1_08_20m_h3"
run_experiment "s1_09_200m_h3"

echo ""
echo "========================================"
echo "HORIZON 5 (H5) EXPERIMENTS"
echo "========================================"
run_experiment "s1_10_2m_h5"
run_experiment "s1_11_20m_h5"
run_experiment "s1_12_200m_h5"

echo ""
echo "=================================================="
echo "SUMMARY"
echo "=================================================="
echo ""
echo "Total: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
date
echo ""
echo "Results saved to: outputs/phase6c_a100/s1_*/results.json"
echo ""
