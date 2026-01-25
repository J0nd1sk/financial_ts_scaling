#!/bin/bash
# Run Phase 6C A100 HPO Experiments
# 6 HPO runs: 3 budgets × 2 horizons (H1, H5)

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=================================================="
echo "Phase 6C A100: HPO Experiments"
echo "6 runs: 2M/20M/200M × H1/H5"
echo "50 trials each = 300 total trials"
echo "=================================================="

# HPO experiments
HPO_EXPERIMENTS=(
    "hpo_2m_h1"
    "hpo_20m_h1"
    "hpo_200m_h1"
    "hpo_2m_h5"
    "hpo_20m_h5"
    "hpo_200m_h5"
)

for exp in "${HPO_EXPERIMENTS[@]}"; do
    echo ""
    echo "=================================================="
    echo "Running: $exp"
    echo "=================================================="

    python experiments/phase6c_a100/${exp}.py 2>&1 | tee outputs/phase6c_a100/${exp}/log.txt

    # Thermal pause between HPO runs (longer due to intensive computation)
    echo ""
    echo "Cooling pause (2 min)..."
    sleep 120
done

# Aggregate HPO results
echo ""
echo "=================================================="
echo "Aggregating HPO results..."
echo "=================================================="

python -c "
import json
from pathlib import Path

hpo_experiments = $( echo "[$(printf '\"outputs/phase6c_a100/%s\",' "${HPO_EXPERIMENTS[@]}" | sed 's/,$//')"]")

results = []
for exp_dir in hpo_experiments:
    results_path = Path(exp_dir) / 'best_params.json'
    if results_path.exists():
        with open(results_path) as f:
            results.append(json.load(f))

summary_path = Path('outputs/phase6c_a100/hpo_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'HPO summary saved to {summary_path}')
"

echo ""
echo "=================================================="
echo "All HPO experiments complete!"
echo "=================================================="
