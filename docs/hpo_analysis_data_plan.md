# HPO Analysis Data Infrastructure Plan

**Status:** Ready for implementation
**Created:** 2026-01-10
**Purpose:** Transform 600 HPO trials into analysis-ready formats

## Context

All 12 HPO studies complete (4 budgets × 3 horizons × 50 trials = 600 total).
14 diverged trials identified (all 2B scale, val_loss=100.0).

## Analysis Priorities

1. **Architecture optimization and patterns** - Which d_model/n_layers/n_heads work best
2. **Training dynamics** - Learning curves, early stopping, divergence causes
3. **Horizon effects** - How optimal architectures differ across h1/h3/h5

Note: Scaling laws deprioritized - HPO used limited data (~53% for training). Scaling analysis deferred to final experiments with full data.

## Output Structure

```
outputs/analysis/
├── hpo_summary.csv          # 600 rows, ~40 columns, spreadsheet-ready
├── hpo_full.json            # 600 trials, complete data + curves
├── diverged_analysis.csv    # 14 rows, extra diagnostic columns
├── diverged_analysis.json   # 14 trials, full curves for diagnosis
└── README.md                # Column definitions, query examples
```

## File Specifications

### hpo_summary.csv (Tier 1: Spreadsheet)

One row per trial, flattened for easy filtering/pivoting.

**Identifier columns:**
- `study` - Full study name (e.g., "phase6a_2M_h1_threshold_1pct")
- `budget` - Parameter budget: 2M, 20M, 200M, 2B
- `horizon` - Prediction horizon: h1, h3, h5
- `trial_num` - Trial number within study (0-49)

**Architecture columns:**
- `d_model` - Model dimension
- `n_layers` - Number of transformer layers
- `n_heads` - Number of attention heads
- `d_ff` - Feed-forward dimension
- `param_count` - Total parameter count
- `arch_idx` - Architecture index in grid

**Training parameter columns:**
- `lr` - Learning rate
- `epochs_config` - Configured epochs (before early stopping)
- `weight_decay` - Weight decay
- `warmup_steps` - Warmup steps
- `dropout` - Dropout rate

**Result columns:**
- `val_loss` - Final validation loss (or 100.0 if diverged)
- `train_loss` - Final training loss
- `val_accuracy` - Validation accuracy (if available)

**Training dynamics columns:**
- `best_epoch` - Epoch with best val_loss
- `best_val_loss` - Best val_loss achieved (may differ from final)
- `epochs_trained` - Actual epochs trained (after early stopping)
- `early_stopped` - Boolean: did early stopping trigger?
- `duration_sec` - Total training duration

**Flag columns:**
- `diverged` - Boolean: val_loss >= 10.0
- `rank_in_study` - Rank by val_loss within study (1=best)
- `is_best_in_study` - Boolean: best trial in study

### hpo_full.json (Tier 2: Deep Dive)

Complete data for programmatic access:

```json
{
  "metadata": {
    "generated": "2026-01-10T...",
    "n_studies": 12,
    "n_trials": 600,
    "n_diverged": 14
  },
  "trials": [
    {
      "study": "phase6a_2M_h1_threshold_1pct",
      "budget": "2M",
      "horizon": "h1",
      "trial_num": 0,
      "architecture": { "d_model": 64, "n_layers": 48, ... },
      "training_params": { "lr": 0.001, ... },
      "results": { "val_loss": 0.3634, "train_loss": 0.35, ... },
      "dynamics": {
        "best_epoch": 29,
        "epochs_trained": 33,
        "early_stopped": true,
        "duration_sec": 33.08
      },
      "learning_curve": [
        { "epoch": 1, "train_loss": 0.45, "val_loss": 0.42 },
        ...
      ],
      "flags": { "diverged": false, "rank_in_study": 5, "is_best": false }
    },
    ...
  ]
}
```

### diverged_analysis.csv

Same columns as hpo_summary.csv, plus diagnostic columns:

- `diverge_epoch` - First epoch where loss > 10 (if detectable)
- `last_good_loss` - Loss value before divergence
- `loss_at_diverge` - First abnormal loss value
- `arch_category` - "narrow-deep", "wide-shallow", "balanced"
- `similar_successful_trial` - Study/trial of nearest successful architecture

### diverged_analysis.json

Full trial data for the 14 diverged trials, including complete learning curves for diagnosing failure patterns.

### README.md

- Column definitions
- Example queries (jq, pandas)
- Data lineage (source files)
- Known patterns/findings

## Implementation Tasks

1. **Create extraction script** - `scripts/extract_hpo_analysis.py`
   - Read all trial JSON files from `outputs/hpo/phase6a_*/trials/`
   - Compute derived fields (rank, best_epoch, early_stopped, etc.)
   - Output all 4 files + README

2. **Validate outputs**
   - Row counts match expectations
   - No missing required fields
   - Diverged trials correctly flagged

3. **Generate README** - Column definitions, example queries

## Usage Patterns

**Conversational exploration (primary):**
- Load hpo_full.json into memory
- Query interactively during chat sessions
- Drill down from summary stats to specific trials

**Spreadsheet analysis (secondary):**
- Import hpo_summary.csv into Excel/Sheets
- Pivot tables by budget/horizon
- Filter and sort to find patterns
- Manual annotation of interesting trials

**Failure analysis:**
- Load diverged_analysis.json
- Compare learning curves with similar successful trials
- Identify common failure patterns

## Questions This Data Enables

**Architecture patterns:**
- What d_model/n_layers combinations perform best per budget?
- Is there a depth limit for reliable training at each scale?
- Does optimal n_heads vary with d_model?

**Training dynamics:**
- How often does early stopping trigger?
- What's the typical gap between best_epoch and epochs_trained?
- Do higher dropout rates correlate with better generalization?

**Horizon effects:**
- Do optimal architectures transfer across horizons?
- Which horizon is easiest to predict (lowest val_loss)?
- Do deeper models help more for longer horizons?

**Divergence patterns:**
- What's the max reliable depth at 2B scale?
- Do diverged trials share common hyperparameters?
- At what epoch do they typically diverge?
