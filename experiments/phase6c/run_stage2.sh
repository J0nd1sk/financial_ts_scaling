#!/bin/bash
#
# Phase 6C Stage 2: Run all 23 exploration experiments
#
# Usage: caffeinate ./experiments/phase6c/run_stage2.sh
#
# Estimated time: ~15-20 minutes
#

set -e  # Exit on error
export PYTHONUNBUFFERED=1  # Ensure real-time output

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment (disable set -e temporarily due to unalias in activate)
set +e
source venv/bin/activate
set -e

echo "========================================"
echo "PHASE 6C STAGE 2: EXPLORATION EXPERIMENTS"
echo "========================================"
echo ""
echo "Total experiments: 23"
echo "  - Track 1 (Horizons): 9"
echo "  - Track 2 (Architecture): 6"
echo "  - Track 3 (Training): 8"
echo ""
date
echo ""

# Track counters
TOTAL=0
PASSED=0
FAILED=0

run_experiment() {
    local script=$1
    local name=$(basename "$script" .py)
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "[$TOTAL/23] Running: $name"
    echo "----------------------------------------"

    if python "$script"; then
        PASSED=$((PASSED + 1))
        echo "[$TOTAL/23] PASSED: $name"
    else
        FAILED=$((FAILED + 1))
        echo "[$TOTAL/23] FAILED: $name"
    fi
}

echo ""
echo "========================================"
echo "TRACK 1: HORIZON EXPERIMENTS (9)"
echo "========================================"

run_experiment "$SCRIPT_DIR/s2_horizon_2m_h2_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_2m_h3_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_2m_h5_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_20m_h2_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_20m_h3_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_20m_h5_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_200m_h2_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_200m_h3_a50.py"
run_experiment "$SCRIPT_DIR/s2_horizon_200m_h5_a50.py"

echo ""
echo "========================================"
echo "TRACK 2: ARCHITECTURE EXPERIMENTS (6)"
echo "========================================"

run_experiment "$SCRIPT_DIR/s2_arch_2m_h1_heads8.py"
run_experiment "$SCRIPT_DIR/s2_arch_20m_h1_heads16.py"
run_experiment "$SCRIPT_DIR/s2_arch_200m_h1_heads16.py"
run_experiment "$SCRIPT_DIR/s2_arch_200m_h1_shallow.py"
run_experiment "$SCRIPT_DIR/s2_arch_200m_h1_wide.py"
run_experiment "$SCRIPT_DIR/s2_arch_200m_h1_balanced.py"

echo ""
echo "========================================"
echo "TRACK 3: TRAINING PARAMETER EXPERIMENTS (8)"
echo "========================================"

run_experiment "$SCRIPT_DIR/s2_train_20m_h1_drop03.py"
run_experiment "$SCRIPT_DIR/s2_train_20m_h1_drop07.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_drop03.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_drop07.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_lr5e5.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_lr2e4.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_wd1e4.py"
run_experiment "$SCRIPT_DIR/s2_train_200m_h1_wd1e3.py"

echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo ""
echo "Total: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
date
echo ""
echo "Results saved to: outputs/phase6c/s2_*/results.json"
echo ""
echo "Run analysis with:"
echo "  python experiments/phase6c/analyze_stage2.py"
