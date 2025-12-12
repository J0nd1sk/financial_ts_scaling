# Phase 6A HPO Runbook

## Overview

12 HPO experiments with **architectural search** - finding the best model architecture AND training parameters for each parameter budget.

**What we search:**
- **Architecture**: d_model, n_layers, n_heads, d_ff (model structure)
- **Training**: learning_rate, epochs, batch_size, weight_decay, warmup_steps

**Experiment matrix:**
- **Parameter scale**: 2M, 20M, 200M, 2B
- **Prediction horizon**: 1-day, 3-day, 5-day

**Research questions:**
1. What is the optimal architecture at each parameter budget?
2. Do optimal architectures differ by prediction horizon?
3. Is depth (more layers) or width (larger d_model) better at each scale?

### How Architectural Search Works

Each budget has a **pre-computed architecture grid** - all valid combinations of (d_model, n_layers, n_heads, d_ff) that fit within ±25% of the target parameter count. During HPO, Optuna samples from this grid along with training parameters.

For details on the architecture grid and parameter estimation, see [`docs/architectural_hpo_design.md`](./architectural_hpo_design.md).

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

### Recommended: Automated Runner Script

Run all 12 experiments sequentially with logging:

```bash
# Start tmux session
tmux new -s hpo

# Run all experiments (smallest to largest)
./scripts/run_phase6a_hpo.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t hpo
```

**Features:**
- Runs all 12 experiments in order (2M→20M→200M→2B)
- Logs all output to `outputs/logs/phase6a_hpo_YYYYMMDD_HHMMSS.log`
- Continues even if one experiment fails
- Prints summary with pass/fail status and duration for each experiment

**Monitoring while running:**
```bash
# Watch the log file
tail -f outputs/logs/phase6a_hpo_*.log

# Check current experiment progress
tmux attach -t hpo
```

### Manual Execution (Alternative)

For running individual experiments or debugging:

```bash
tmux new -s hpo
./venv/bin/python experiments/phase6a/hpo_2M_h1_threshold_1pct.py
# Detach: Ctrl+B, then D
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

Each script logs to stdout with timestamps. Note the **architecture grid** printed at startup:

```
✓ Architecture grid: 28 valid configs for 2M
✓ Data validated: 8073 rows, 20 features
✓ Splits: train=7569, val=252, test=252
✓ Training params: ['learning_rate', 'epochs', 'batch_size', 'weight_decay', 'warmup_steps']

Starting HPO: 50 trials, 28 architectures...
[I 2025-12-12 10:15:00,123] Trial 0 starting: arch_idx=5, d_model=128, n_layers=8, lr=0.000342
[I 2025-12-12 10:15:45,456] Trial 0 finished with value: 0.4521 (params: 1,847,000)
[I 2025-12-12 10:15:45,789] Trial 1 starting: arch_idx=12, d_model=192, n_layers=4, lr=0.000567
[I 2025-12-12 10:16:28,012] Trial 1 finished with value: 0.4389 (params: 2,103,000)
[I 2025-12-12 10:16:28,345] Best is trial 1 with value: 0.4389
...

✓ HPO complete in 42.3 min
  Best val_loss: 0.4102
  Best arch: d_model=128, n_layers=8, params=1,847,000
  Results saved to: outputs/hpo/phase6a_2M_h1_threshold_1pct/best_params.json
```

**Key things to watch:**
- Architecture grid size at startup (should be 25-35 configs per budget)
- Trial messages showing architecture variation (`arch_idx`, `d_model`, `n_layers`)
- `Best is trial X with value: Y` (current best)
- Final summary shows best architecture details
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

### Compare Architectures Across Budgets

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

# Compare architectures across budgets for each horizon
for horizon in [1, 3, 5]:
    print(f"\n=== Horizon {horizon}d ===")
    for budget in ['2M', '20M', '200M', '2B']:
        exp = f"phase6a_{budget}_h{horizon}_threshold_1pct"
        if exp in results:
            r = results[exp]
            arch = r['architecture']
            print(f"  {budget}: d={arch['d_model']}, L={arch['n_layers']}, "
                  f"params={r['param_count']:,}, val_loss={r['best_val_loss']:.4f}")
```

### Compare Training Params Across Horizons

```python
# Compare learning rates and epochs across horizons for each budget
for budget in ['2M', '20M', '200M', '2B']:
    print(f"\n{budget}:")
    for horizon in [1, 3, 5]:
        exp = f"phase6a_{budget}_h{horizon}_threshold_1pct"
        if exp in results:
            t = results[exp]['training']
            val = results[exp]['best_val_loss']
            print(f"  h{horizon}: lr={t['learning_rate']:.2e}, epochs={t['epochs']}, val_loss={val:.4f}")
```

### Key Analysis Questions

1. **What architecture wins at each budget?**
   - Compare d_model vs n_layers across budgets
   - Does "deep & narrow" or "shallow & wide" perform better?
   - Is there a consistent pattern or does it vary by scale?

2. **Do optimal architectures vary by horizon?**
   - Compare architectures across h1/h3/h5 for same budget
   - Longer horizons might benefit from different depth/width ratios

3. **Do training params vary by horizon?**
   - Compare lr, epochs, batch_size across h1/h3/h5 for each budget
   - If variance > 20%, horizon-specific HPO was valuable
   - If similar, could have borrowed params

4. **Does val_loss improve with scale?**
   - Plot val_loss vs parameter count
   - Look for power law: error ∝ N^(-α)
   - Use `src/analysis/scaling_curves.py` for fitting

### Generate Analysis Report

```python
# Use scaling curves module
from src.analysis.scaling_curves import fit_power_law, plot_scaling_curve

# After collecting results, fit power law
# See src/analysis/scaling_curves.py for full API
```

---

## Interpreting Architectural Results

### Depth vs Width Tradeoffs

Transformer architectures can be configured along two main dimensions:

| Configuration | Characteristics | When it might win |
|--------------|-----------------|-------------------|
| **Deep & Narrow** (many layers, small d_model) | More sequential processing, better at capturing long-range dependencies | Complex temporal patterns, longer horizons |
| **Shallow & Wide** (few layers, large d_model) | More parallel processing, larger per-layer capacity | Simple patterns, shorter horizons |

**Example at 2M budget:**
- Deep: d_model=64, n_layers=24, d_ff=256 → ~2.1M params
- Wide: d_model=256, n_layers=4, d_ff=1024 → ~1.9M params

### Budget Utilization

The ±25% tolerance means actual param counts vary. Check budget utilization:

```python
# Check how well each experiment utilized its budget
targets = {'2M': 2e6, '20M': 20e6, '200M': 200e6, '2B': 2e9}
for exp, r in results.items():
    budget = exp.split('_')[1]
    target = targets[budget]
    actual = r['param_count']
    util = (actual / target) * 100
    print(f"{exp}: {actual:,} params ({util:.0f}% of {budget})")
```

Ideal utilization is 90-110% of target. Lower means the architecture grid might need adjustment.

### Comparing Across Horizons

If optimal architectures differ significantly by horizon:

| Finding | Interpretation |
|---------|----------------|
| Same architecture wins for all horizons | Architecture choice is horizon-independent at this budget |
| Longer horizons prefer deeper models | Sequential patterns become more important for longer predictions |
| Longer horizons prefer wider models | Each layer needs more capacity for complex multi-step reasoning |
| No clear pattern | Horizon effect is weaker than random variation in HPO |

### Signs of Good HPO Coverage

After running experiments, check that:

1. **Architecture diversity**: Different architectures were tried (check trial logs)
2. **Budget coverage**: Best models are in 75-125% of target budget
3. **Training param stability**: Similar training params across architectures suggest robust search
4. **Val loss variation**: Meaningful difference between best and worst architectures (~10%+ difference)

If best trials cluster around similar architectures, the search found a clear optimum. If results are scattered, the landscape may be flat or noisy.

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
└── best_params.json    # Best architecture + hyperparameters found
```

### Aggregated

```
docs/experiment_results.csv   # All runs with metadata (includes architecture columns)
```

### best_params.json Format

The output now includes separate `architecture` and `training` sections:

```json
{
  "experiment": "phase6a_2M_h1_threshold_1pct",
  "architecture": {
    "d_model": 128,
    "n_layers": 8,
    "n_heads": 4,
    "d_ff": 512,
    "patch_len": 10,
    "stride": 5
  },
  "training": {
    "learning_rate": 0.0003,
    "epochs": 75,
    "batch_size": 128,
    "weight_decay": 0.0001,
    "warmup_steps": 200
  },
  "param_count": 1847000,
  "best_val_loss": 0.4102,
  "n_trials": 50,
  "timestamp": "2025-12-12T10:57:00.000000+00:00"
}
```

### experiment_results.csv Columns

The CSV now includes architecture columns:

```
experiment, phase, budget, task, horizon, timescale, script_path, run_type, status,
d_model, n_layers, n_heads, d_ff, param_count,
val_loss, hyperparameters, duration_seconds, timestamp
```

**New columns** (added for architectural HPO):
- `d_model`: Embedding dimension (64-2048)
- `n_layers`: Number of transformer layers (2-48)
- `n_heads`: Number of attention heads (2-32)
- `d_ff`: Feedforward hidden dimension
- `param_count`: Actual parameter count of best model

---

## Next Steps After HPO

Once all 12 HPO runs complete:

1. **Analyze architectural results** - Which architectures won at each budget? (see "Interpreting Architectural Results")
2. **Analyze horizon variance** - Do architectures/params differ by horizon?
3. **Generate training scripts** - Use best architecture + params for final evaluation
4. **Run final training** - Full training with optimal architecture and hyperparameters
5. **Collect test metrics** - Evaluate on held-out test set
6. **Fit scaling curves** - Plot error vs parameters, check power law (error ∝ N^(-α))

---

*Updated: 2025-12-12 (architectural HPO)*
