# Phase 6A HPO Runbook

## Overview

12 HPO experiments testing whether optimal hyperparameters vary by:
- **Parameter scale**: 2M, 20M, 200M, 2B
- **Prediction horizon**: 1-day, 3-day, 5-day

**Research question**: Do optimal hyperparameters vary significantly by horizon at each scale?

---

## Scripts

| Script | Budget | Horizon | Est. Time |
|--------|--------|---------|-----------|
| `hpo_2M_h1_threshold_1pct.py` | 2M | 1-day | ~45 min |
| `hpo_2M_h3_threshold_1pct.py` | 2M | 3-day | ~45 min |
| `hpo_2M_h5_threshold_1pct.py` | 2M | 5-day | ~45 min |
| `hpo_20M_h1_threshold_1pct.py` | 20M | 1-day | ~2-3 hrs |
| `hpo_20M_h3_threshold_1pct.py` | 20M | 3-day | ~2-3 hrs |
| `hpo_20M_h5_threshold_1pct.py` | 20M | 5-day | ~2-3 hrs |
| `hpo_200M_h1_threshold_1pct.py` | 200M | 1-day | ~8-12 hrs |
| `hpo_200M_h3_threshold_1pct.py` | 200M | 3-day | ~8-12 hrs |
| `hpo_200M_h5_threshold_1pct.py` | 200M | 5-day | ~8-12 hrs |
| `hpo_2B_h1_threshold_1pct.py` | 2B | 1-day | ~24-48 hrs |
| `hpo_2B_h3_threshold_1pct.py` | 2B | 3-day | ~24-48 hrs |
| `hpo_2B_h5_threshold_1pct.py` | 2B | 5-day | ~24-48 hrs |

**Total estimated time**: ~150-250 hours (run sequentially)

---

## Running Experiments

### Basic Execution

```bash
cd /Users/alexanderthomson/Documents/financial_ts_scaling
./venv/bin/python experiments/phase6a/hpo_2M_h1_threshold_1pct.py
```

### Recommended: Use tmux/screen

For long-running experiments, use tmux so you can detach and reattach:

```bash
# Start new tmux session
tmux new -s hpo

# Run experiment
./venv/bin/python experiments/phase6a/hpo_2M_h1_threshold_1pct.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t hpo
```

### Run Order Recommendation

Start with smallest models to validate pipeline, then scale up:

```bash
# Day 1: 2M models (~2.5 hrs total)
./venv/bin/python experiments/phase6a/hpo_2M_h1_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_2M_h3_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_2M_h5_threshold_1pct.py

# Day 2: 20M models (~8 hrs total)
./venv/bin/python experiments/phase6a/hpo_20M_h1_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_20M_h3_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_20M_h5_threshold_1pct.py

# Days 3-5: 200M models (~30 hrs total)
./venv/bin/python experiments/phase6a/hpo_200M_h1_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_200M_h3_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_200M_h5_threshold_1pct.py

# Days 6-12: 2B models (~100+ hrs total)
./venv/bin/python experiments/phase6a/hpo_2B_h1_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_2B_h3_threshold_1pct.py
./venv/bin/python experiments/phase6a/hpo_2B_h5_threshold_1pct.py
```

---

## Monitoring

### CLI Output (Live)

Each script logs to stdout with timestamps:

```
2025-12-11 16:17:13 [INFO] === phase6a_2M_h1_threshold_1pct ===
2025-12-11 16:17:13 [INFO] Device: mps, Budget: 2M, Horizon: 1d, N_TRIALS: 50
2025-12-11 16:17:13 [INFO] Thermal: unknown (-1.0°C)
2025-12-11 16:17:13 [INFO] ✓ Data loaded: 8073 rows
2025-12-11 16:17:13 [INFO] ✓ Splits: train=4075, val=19, test=19
2025-12-11 16:17:13 [INFO] Starting HPO (50 trials)...
[I 2025-12-11 16:17:13,404] Trial 0 starting: device=mps, lr=0.003583, epochs=110
[I 2025-12-11 16:17:50,816] Trial 0 finished with value: 0.8312 ...
[I 2025-12-11 16:17:50,817] Trial 1 starting: ...
```

**Key things to watch:**
- Trial completion messages (shows progress)
- `Best is trial X with value: Y` (current best)
- Any ERROR or WARNING messages
- Thermal status (should stay below 85°C)

### Thermal Monitoring (Separate Terminal)

```bash
# macOS thermal monitoring
sudo powermetrics --samplers smc -i 5000 | grep -i temp

# Or simpler:
while true; do
  echo "$(date): $(sudo powermetrics --samplers smc -n 1 2>/dev/null | grep -i 'cpu die' | head -1)"
  sleep 30
done
```

**Thermal thresholds:**
- < 70°C: Normal
- 70-85°C: Acceptable
- 85-95°C: Warning - consider pausing
- > 95°C: **STOP** - let cool down

### Check Results Log

```bash
# View latest results
tail -20 docs/experiment_results.csv

# Count completed experiments
grep -c "success" docs/experiment_results.csv

# View specific experiment
grep "phase6a_2M_h1" docs/experiment_results.csv
```

### Check HPO Output Files

```bash
# List all HPO outputs
ls -la outputs/hpo/

# View best params for specific experiment
cat outputs/hpo/phase6a_2M_h1_threshold_1pct/best_params.json
```

---

## Analyzing Results

### Quick Summary

After all experiments complete:

```bash
# View all results
cat docs/experiment_results.csv | column -t -s,

# Extract best val_loss by experiment
grep "success" docs/experiment_results.csv | cut -d, -f2,4,5,11 | column -t -s,
```

### Compare Hyperparameters Across Horizons

```python
import json
from pathlib import Path

# Load all best params
results = {}
for budget in ['2M', '20M', '200M', '2B']:
    for horizon in [1, 3, 5]:
        exp = f"phase6a_{budget}_h{horizon}_threshold_1pct"
        path = Path(f"outputs/hpo/{exp}/best_params.json")
        if path.exists():
            with open(path) as f:
                results[exp] = json.load(f)

# Compare learning rates across horizons for each budget
for budget in ['2M', '20M', '200M', '2B']:
    print(f"\n{budget}:")
    for horizon in [1, 3, 5]:
        exp = f"phase6a_{budget}_h{horizon}_threshold_1pct"
        if exp in results:
            lr = results[exp]['best_params']['learning_rate']
            val = results[exp]['best_value']
            print(f"  h{horizon}: lr={lr:.2e}, val_loss={val:.4f}")
```

### Key Analysis Questions

1. **Do params vary by horizon?**
   - Compare lr, epochs, dropout across h1/h3/h5 for each budget
   - If variance > 20%, need separate HPO per horizon
   - If similar, can borrow params (saves significant time)

2. **Do params vary by scale?**
   - Compare across 2M/20M/200M/2B for same horizon
   - Larger models may need different lr, dropout

3. **Does val_loss improve with scale?**
   - Plot val_loss vs parameter count
   - Look for power law: error ∝ N^(-α)

### Generate Analysis Report

```python
# Use scaling curves module
from src.analysis.scaling_curves import fit_power_law, plot_scaling_curve

# After collecting results, fit power law
# See src/analysis/scaling_curves.py for full API
```

---

## Troubleshooting

### Script Crashes

1. Check error message in terminal
2. Check `docs/experiment_results.csv` for logged error
3. Common issues:
   - OOM: Reduce batch size in config
   - Thermal: Let machine cool, restart
   - Data: Verify `data/processed/v1/SPY_dataset_a25.parquet` exists

### Thermal Throttling

If temperature exceeds 85°C:

1. Pause experiment (Ctrl+C is safe - Optuna saves progress)
2. Wait 10-15 minutes for cooldown
3. Consider reducing batch size or epochs in search space
4. Restart script (Optuna won't re-run completed trials if using same study)

### Resume After Interruption

Scripts use in-memory Optuna storage, so interrupted runs lose progress. For long runs (200M, 2B), consider:

1. Reducing N_TRIALS to run in shorter batches
2. Using Optuna's SQLite storage for persistence (requires script modification)

---

## Outputs

### Per-Experiment

```
outputs/hpo/{experiment}/
└── best_params.json    # Best hyperparameters found
```

### Aggregated

```
docs/experiment_results.csv   # All runs with metadata
```

### best_params.json Format

```json
{
  "experiment": "phase6a_2M_h1_threshold_1pct",
  "budget": "2M",
  "horizon": 1,
  "best_params": {
    "learning_rate": 3.78e-05,
    "epochs": 140,
    "weight_decay": 2.35e-06,
    "warmup_steps": 600,
    "dropout": 0.0217
  },
  "best_value": 0.3914,
  "n_trials": 50,
  "timestamp": "2025-12-12T00:19:44.799132+00:00"
}
```

---

## Next Steps After HPO

Once all 12 HPO runs complete:

1. **Analyze horizon variance** - Do params differ significantly?
2. **Generate training scripts** - Use best params for final evaluation
3. **Run final training** - Full training with optimal hyperparameters
4. **Collect test metrics** - Evaluate on held-out test set
5. **Fit scaling curves** - Plot error vs parameters, check power law

---

*Generated: 2025-12-11*
