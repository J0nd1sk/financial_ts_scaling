#!/bin/bash
#
# Phase 6C A100: HPO Experiments
# 6 HPO runs: 3 budgets (2M/20M/200M) × 2 horizons (H1/H5)
# 50 trials each = 300 total trials
#
# Usage: caffeinate ./scripts/run_hpo_a100.sh
#
# Each HPO run: 50 trials × ~1-2 min/trial = ~1-2 hours
# Total estimated time: 6-12 hours depending on thermal conditions
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
echo "Phase 6C A100: HPO Experiments"
echo "6 runs: 2M/20M/200M × H1/H5"
echo "50 trials each = 300 total trials"
echo "=================================================="
echo ""
date
echo ""

# Track counters
TOTAL=0
PASSED=0
FAILED=0

run_hpo() {
    local exp=$1
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "[$TOTAL/6] Running HPO: $exp"
    echo "----------------------------------------"

    if python "experiments/phase6c_a100/${exp}.py"; then
        PASSED=$((PASSED + 1))
        echo "[$TOTAL/6] PASSED: $exp"
    else
        FAILED=$((FAILED + 1))
        echo "[$TOTAL/6] FAILED: $exp"
    fi

    # Longer thermal pause for HPO (intensive computation)
    echo ""
    echo "Cooling pause (2 min)..."
    sleep 120
}

# Run all 6 HPO experiments
echo ""
echo "========================================"
echo "HORIZON 1 (H1) HPO"
echo "========================================"
run_hpo "hpo_2m_h1"
run_hpo "hpo_20m_h1"
run_hpo "hpo_200m_h1"

echo ""
echo "========================================"
echo "HORIZON 5 (H5) HPO"
echo "========================================"
run_hpo "hpo_2m_h5"
run_hpo "hpo_20m_h5"
run_hpo "hpo_200m_h5"

# Aggregate HPO results
echo ""
echo "========================================"
echo "Aggregating HPO results..."
echo "========================================"

python -c "
import json
from pathlib import Path

hpo_dirs = [
    'outputs/phase6c_a100/hpo_2m_h1',
    'outputs/phase6c_a100/hpo_20m_h1',
    'outputs/phase6c_a100/hpo_200m_h1',
    'outputs/phase6c_a100/hpo_2m_h5',
    'outputs/phase6c_a100/hpo_20m_h5',
    'outputs/phase6c_a100/hpo_200m_h5',
]

results = []
for exp_dir in hpo_dirs:
    results_path = Path(exp_dir) / 'best_params.json'
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            data['source'] = exp_dir
            results.append(data)
        print(f'  Found: {results_path}')
    else:
        print(f'  Missing: {results_path}')

summary_path = Path('outputs/phase6c_a100/hpo_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'HPO summary saved to {summary_path} ({len(results)} results)')
"

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
echo "Results saved to: outputs/phase6c_a100/hpo_*/best_params.json"
echo "Summary saved to: outputs/phase6c_a100/hpo_summary.json"
echo ""
