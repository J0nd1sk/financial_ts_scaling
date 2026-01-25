#!/bin/bash
# Phase 6C Stage 1: a50 Baseline Experiments
# Run all three budget configurations (2M, 20M, 200M) at H1 with 55 features

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source venv/bin/activate

echo "=============================================================================="
echo "PHASE 6C STAGE 1: a50 Baseline Experiments"
echo "=============================================================================="
echo "Features: 55 (5 OHLCV + 50 indicators)"
echo "Horizon: H1 (1-day)"
echo "Budgets: 2M, 20M, 200M"
echo ""

# S1-01: 2M budget
echo "[1/3] Running S1-01: 2M budget..."
./venv/bin/python experiments/phase6c/s1_01_2m_h1_a50.py
echo ""

# S1-02: 20M budget
echo "[2/3] Running S1-02: 20M budget..."
./venv/bin/python experiments/phase6c/s1_02_20m_h1_a50.py
echo ""

# S1-03: 200M budget
echo "[3/3] Running S1-03: 200M budget..."
./venv/bin/python experiments/phase6c/s1_03_200m_h1_a50.py
echo ""

echo "=============================================================================="
echo "STAGE 1 COMPLETE"
echo "=============================================================================="
echo ""
echo "Results saved to:"
echo "  - outputs/phase6c/s1_01_2m_h1_a50/results.json"
echo "  - outputs/phase6c/s1_02_20m_h1_a50/results.json"
echo "  - outputs/phase6c/s1_03_200m_h1_a50/results.json"
echo ""
echo "Compare with Phase 6A baselines:"
echo "  2M:   a20=0.706  vs  a50=?"
echo "  20M:  a20=0.715  vs  a50=?"
echo "  200M: a20=0.718  vs  a50=?"
