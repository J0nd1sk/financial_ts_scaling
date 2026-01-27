# HPO Methodology

This document describes the hyperparameter optimization methodology for Phase 6C experiments. Use this as the authoritative reference for all HPO runs.

## Overview

HPO finds optimal model configurations for each parameter budget (2M, 20M, 200M). The methodology addresses two critical challenges:

1. **TPE premature convergence** - Bayesian optimization can focus too narrowly before exploring architecture extremes
2. **Probability collapse** - Models may output constant predictions, achieving 50% AUC with 0% recall

Our solution: Two-phase search with comprehensive metrics capture.

## Two-Phase Strategy

### Phase A: Forced Extreme Trials (6 trials)

The first 6 trials test architecture extremes before TPE sampling begins. This ensures we understand the impact of each architecture dimension.

| Trial | Tests | Configuration |
|-------|-------|---------------|
| 0 | Min d_model | Smallest embedding, middle n_layers/n_heads |
| 1 | Max d_model | Largest embedding, middle n_layers/n_heads |
| 2 | Min n_layers | Shallowest, middle d_model/n_heads |
| 3 | Max n_layers | Deepest, middle d_model/n_heads |
| 4 | Min n_heads | Fewest heads, middle d_model |
| 5 | Max n_heads | Most heads, middle d_model |

**Why this matters:** TPE learns from successful trials and may never explore architecture extremes if early random trials happen to succeed with mid-range values. Forcing extremes first provides signal about the entire search space.

**Training parameters for forced trials:** Use middle values (lr=5e-5, dropout=0.3, weight_decay=1e-4) to isolate architecture effects.

### Phase B: TPE Exploration (remaining trials)

After forced extremes, Optuna's TPE sampler explores the search space:

1. **n_startup_trials=20**: First 20 TPE trials are random (ensuring exploration)
2. **Bayesian optimization**: Subsequent trials use tree-structured Parzen estimators
3. **Coverage-aware sampling**: Duplicate architecture combinations are redirected to untested regions

**Coverage tracking logic** (`src/training/hpo_coverage.py`):
- Tracks tested (d_model, n_layers, n_heads) combinations
- When TPE suggests a duplicate, redirects to an untested combination
- Preserves training parameters (learning_rate, dropout, weight_decay) from original suggestion
- Ensures broader architecture exploration despite TPE's tendency to exploit

## Configuration by Budget

### Search Spaces

```python
SEARCH_SPACES = {
    "2M": {
        "d_model": [32, 48, 64, 80, 96],
        "n_layers": [2, 3, 4, 5, 6],
        "n_heads": [2, 4, 8],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
    "20M": {
        "d_model": [64, 96, 128, 160, 192],
        "n_layers": [4, 5, 6, 7, 8],
        "n_heads": [4, 8],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
    "200M": {
        "d_model": [128, 192, 256, 320, 384],
        "n_layers": [6, 8, 10, 12],
        "n_heads": [8, 16],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
}
```

### Fixed Hyperparameters (Ablation-Validated)

These are fixed across all HPO runs based on ablation study results:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| context_length | 80 | Optimal from context ablation |
| epochs | 50 | Sufficient for convergence with early stopping |
| patch_length | 16 | Standard for financial time series |
| stride | 8 | 50% overlap |
| early_stopping_patience | 10 | Prevents overfitting |
| early_stopping_min_delta | 0.001 | Meaningful improvement threshold |
| use_revin | True | RevIN normalization for non-stationary data |

### Batch Size Selection

Batch size is determined by model size to prevent OOM:

| d_model | Batch Size |
|---------|------------|
| >= 256 | 32 |
| >= 128 | 64 |
| < 128 | 128 |

## Metrics Capture

**All metrics are stored for every trial**, not just the optimization target. This enables post-hoc analysis.

### Stored Metrics

| Metric | Purpose |
|--------|---------|
| `val_auc` | **Optimization target** - discrimination ability |
| `val_precision` | Positive prediction quality |
| `val_recall` | **Critical** - 0% = useless model |
| `val_pred_min` | Lower bound of predictions |
| `val_pred_max` | Upper bound of predictions |
| `epochs_run` | Actual epochs (may stop early) |
| `duration_sec` | Trial wall time |

### Probability Collapse Detection

**Probability collapse** occurs when a model outputs near-constant predictions (e.g., all ~0.5). These models achieve ~50% AUC but have 0% recall and are useless for trading.

**Detection via `pred_range`:**
- `pred_range = [pred_min, pred_max]`
- **Healthy range:** `[0.1, 0.9]` or wider
- **Collapsed:** `[0.48, 0.52]` or similar narrow band

**Always check pred_range in HPO results.** A trial with 0.55 AUC but collapsed predictions is worse than 0.52 AUC with healthy predictions.

## Output Files

Each HPO run produces:

| File | Content |
|------|---------|
| `all_trials.json` | Full trial data with all metrics and user_attrs |
| `best_params.json` | Best configuration with search space metadata |
| `trial_metrics.csv` | Tabular format for analysis |
| `study.db` | SQLite database for resumability |
| `trial_NNN/` | Individual trial checkpoints (if needed) |

### Output Directory Structure

```
outputs/phase6c_{tier}/
├── hpo_{budget}_h{horizon}/
│   ├── all_trials.json
│   ├── best_params.json
│   ├── trial_metrics.csv
│   ├── study.db
│   └── trial_000/
│       └── ...
```

## Cross-Budget Validation

After HPO completes for all budgets, cross-budget validation tests whether optimal configurations transfer.

### Purpose

Test if the 2M-optimal config works on 20M/200M budgets (and vice versa). This reveals:
- **Budget-specific optima**: Different budgets need different architectures
- **Transferable hyperparameters**: Training params (lr, dropout) may transfer better than architecture

### Methodology

Train a 3x3 matrix:
- Rows: Configuration source (2M, 20M, 200M best params)
- Columns: Budget used for training

### Running Cross-Budget Validation

```bash
python scripts/validate_cross_budget.py --tier a100 --horizon 1
```

### Interpreting Results

| Diagonal vs Off-Diagonal | Interpretation |
|--------------------------|----------------|
| Diagonal > Off-diagonal | Configs are budget-specific (expected) |
| Similar performance | Configs may transfer (surprising) |
| High transfer gap (>0.02 AUC) | Strong budget specialization |

Output files:
- `outputs/phase6c_{tier}/cross_budget_validation.json`
- `outputs/phase6c_{tier}/cross_budget_validation.md`

## Running HPO

### Basic Run

```bash
# Configure experiments/templates/hpo_template.py, then:
python experiments/templates/hpo_template.py --budget 2M --tier a100 --horizon 1 --trials 50
```

### Resume Interrupted Run

```bash
python experiments/templates/hpo_template.py --budget 2M --tier a100 --resume
```

The SQLite study database preserves all trial history. Resuming continues from where it left off.

### Dry Run (Test Configuration)

```bash
python experiments/templates/hpo_template.py --dry-run
```

Returns dummy results to validate configuration without training.

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--budget` | 2M | Parameter budget (2M, 20M, 200M) |
| `--tier` | a100 | Feature tier |
| `--horizon` | 1 | Prediction horizon |
| `--trials` | 50 | Total trials including forced extremes |
| `--resume` | False | Resume existing study |
| `--dry-run` | False | Test without training |

## Thermal Management

### Temperature Thresholds

| Temperature | Action |
|-------------|--------|
| < 70C | Normal operation |
| 70-85C | Acceptable, monitor |
| 85-95C | Warning, automatic 60s pause |
| > 95C | **CRITICAL** - abort HPO |

### Pause Behavior

When temperature exceeds warning threshold (85C), HPO pauses for `THERMAL_PAUSE_SECONDS=60` before the next trial. This allows cooling without losing progress.

### Monitoring

```bash
# In a separate terminal:
sudo powermetrics --samplers smc -i 1000 | grep -i temp
```

## Post-HPO Analysis

### Analyze HPO Results

```bash
# View best parameters and trial summary
cat outputs/phase6c_{tier}/hpo_{budget}_h{horizon}/best_params.json

# Analyze trial distribution
python scripts/analyze_hpo_results.py --tier a100 --budget 2M
```

### Coverage Analysis

```bash
# Check architecture coverage
python scripts/analyze_hpo_coverage.py --tier a100 --budget 2M
```

Coverage analysis shows:
- Total valid architecture combinations
- Number tested
- Untested combinations (if any)
- Coverage percentage

## Checklist for HPO Runs

### Pre-Run

- [ ] Data exists: `data/processed/v1/SPY_dataset_{tier}_combined.parquet`
- [ ] Feature count matches tier (a20=25, a50=55, a100=105, a200=211)
- [ ] Thermal monitoring active in separate terminal
- [ ] Previous study backed up (if overwriting)
- [ ] Git status clean

### During Run

- [ ] Monitor for probability collapse (check pred_range in output)
- [ ] Monitor thermal state
- [ ] Verify trials are completing (not all pruned)

### Post-Run

- [ ] Run coverage analysis
- [ ] Check best_params.json for sensible values
- [ ] Verify best trial has healthy pred_range
- [ ] Check for recall > 0 in best trial
- [ ] When all budgets done: run cross-budget validation

## Common Issues & Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| OOM errors | Trial pruned with "OOM" in error | Reduce batch size, check d_model |
| Probability collapse | pred_range ~[0.48, 0.52], recall=0 | Check class balance, increase dropout |
| All trials pruned | n_complete=0 | Check data path, feature count |
| Thermal throttling | Slow trials, pauses | Improve cooling, reduce concurrency |
| NaN loss | Trial fails with NaN | Check learning rate (too high?) |
| Resume fails | KeyError on load | Verify study.db path, study name |

### Interpreting Poor Results

| Pattern | Likely Cause | Investigation |
|---------|--------------|---------------|
| Best AUC ~0.50 | No signal OR collapsed | Check pred_range, try more epochs |
| High AUC, 0% recall | Collapsed, threshold issue | Model predicts constant negative |
| Extreme LR wins | Underfitting | Try larger d_model, more layers |
| High dropout wins | Overfitting | Feature redundancy, need regularization |

## Reproducibility

All HPO runs should be reproducible:

1. **Fixed seed**: TPESampler uses `seed=42`
2. **SQLite storage**: Full trial history preserved
3. **Config logging**: Search space saved in `best_params.json`
4. **Deterministic operations**: Where possible on MPS

To reproduce a specific trial:
1. Load `all_trials.json`
2. Extract trial parameters
3. Train with those exact parameters using Trainer directly

## Version History

| Date | Change |
|------|--------|
| 2026-01 | Initial methodology with two-phase strategy |
| 2026-01 | Added coverage tracking for architecture exploration |
| 2026-01 | Added cross-budget validation |
