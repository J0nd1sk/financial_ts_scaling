# Experiment Skills Design

**Status:** Approved
**Date:** 2025-12-11
**Phase:** 6A Preparation

---

## Overview

Two Claude Code skills to systematize experiment generation and execution for scaling law research.

### Skills

1. **experiment-generation**: Generate HPO and training scripts from templates
2. **experiment-execution**: Execute experiments with thermal monitoring and logging

### Key Design Decisions

- **Thin wrapper scripts** (~50-80 lines): All parameters visible inline for reproducibility
- **Dynamic data assembly**: Load parquet, select features at runtime (no pre-built datasets)
- **Per-budget HPO**: 12 HPO runs (4 budgets × 3 tasks: 1%, 3%, 5%), borrow params for 2%
- **Dual logging**: Append-only CSV (raw history) + regenerated markdown (summary)

---

## Skill 1: experiment-generation

### Purpose

Generate experiment scripts (HPO + training) from templates with all parameters visible inline for reproducibility and publication.

### When to Use

- "Generate experiment for 2M threshold_1pct"
- "Create Phase 6A experiments"
- "Set up HPO and training scripts for 20M budget"
- Starting a new experiment phase

### Execution Steps

1. **Gather Parameters**
   - Budget: 2M | 20M | 200M | 2B
   - Task: threshold_1pct | threshold_2pct | threshold_3pct | threshold_5pct
   - Horizon: 1 | 2 | 3 | 5 | 7 (days)
   - Timescale: daily | 2d | 3d | 5d | weekly
   - Data path: path to processed parquet
   - Phase: phase6a | phase6b | phase6c

2. **Validate Prerequisites**
   - Data file exists
   - Manifest entry verified
   - Budget config exists (`configs/model/patchtst_{budget}.yaml`)

3. **Generate HPO Script** (if HPO needed for this experiment)
   - Output: `experiments/{phase}/hpo_{budget}_{task}.py`

4. **Generate Training Script**
   - Output: `experiments/{phase}/train_{budget}_{task}.py`

5. **Validate Generated Scripts**
   - Run `python -m py_compile` on each
   - Verify imports resolve

6. **Update Experiment Manifest**
   - Append to `experiments/{phase}/README.md`

### Script Template Structure

```python
#!/usr/bin/env python3
"""
Phase 6A Experiment: {budget} parameters, {task} task
Generated: {timestamp}
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import run_hpo, run_training

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "{phase}_{budget}_{task}"
BUDGET = "{budget}"
TASK = "{task}"
HORIZON = {horizon}
TIMESCALE = "{timescale}"
DATA_PATH = "{data_path}"
FEATURE_COLUMNS = {feature_list}

# HPO settings (hpo_*.py only)
N_TRIALS = 50
TIMEOUT_HOURS = 4.0

# Training settings (train_*.py only)
HPO_PARAMS_PATH = "outputs/hpo/{experiment}/{budget}_best.json"
BORROWED_FROM = {borrowed_from}  # For 2% task

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data():
    import pandas as pd
    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
    assert all(col in df.columns for col in FEATURE_COLUMNS)
    assert not df[FEATURE_COLUMNS].isna().any().any()
    print(f"✓ Data validated: {len(df)} rows, {len(FEATURE_COLUMNS)} features")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    validate_data()
    run_hpo(...)  # or run_training(...)
```

---

## Skill 2: experiment-execution

### Purpose

Execute experiment scripts with thermal monitoring, logging, and automatic report updates.

### When to Use

- "Run HPO for 2M threshold_1pct"
- "Execute training for phase6a experiments"
- "Run the 20M experiments"
- After experiment-generation has created scripts

### Execution Steps

1. **Pre-Flight Checks**
   - Script exists and is valid Python
   - Thermal status is NORMAL or ACCEPTABLE
   - Data manifest verified
   - Sufficient disk space

2. **Execute Experiment**
   - Run script with thermal monitoring
   - Capture stdout/stderr
   - Pause if temp >85°C
   - Abort gracefully if >95°C

3. **On Completion**
   - Append row to `outputs/results/experiment_log.csv`

4. **On Success**
   - Verify outputs exist
   - Regenerate `docs/experiment_results.md`

5. **On Failure**
   - Log error details
   - Do NOT regenerate report

---

## Output Specifications

### CSV Log Schema (`outputs/results/experiment_log.csv`)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | When experiment started |
| experiment | string | Full experiment name |
| phase | string | phase6a, phase6b, etc. |
| budget | string | 2M, 20M, 200M, 2B |
| task | string | threshold_1pct, etc. |
| horizon | int | Prediction horizon in days |
| timescale | string | daily, weekly, etc. |
| script_path | string | Path to executed script |
| run_type | string | hpo or training |
| status | string | success, failed, thermal_abort |
| duration_seconds | float | Total runtime |
| val_loss | float | Best validation loss |
| test_accuracy | float | Test accuracy |
| hyperparameters | json | JSON string of params |
| error_message | string | Error details if failed |
| thermal_max_temp | float | Peak temperature |
| data_md5 | string | MD5 of data file |

### Markdown Report (`docs/experiment_results.md`)

```markdown
# Experiment Results

Generated: {timestamp}
Total Experiments: {count}
Success Rate: {rate}%

## Phase 6A: Parameter Scaling

### Summary Table
| Budget | 1% | 2% | 3% | 5% | Avg Loss |
|--------|----|----|----|----|----------|

### Scaling Analysis
- Alpha (α): {value}
- R²: {value}
- Interpretation: {text}

### Individual Results
[Detailed tables per budget]
```

---

## Implementation

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `.claude/skills/experiment_generation/SKILL.md` | ~200 | Generation skill |
| `.claude/skills/experiment_execution/SKILL.md` | ~200 | Execution skill |
| `src/experiments/__init__.py` | ~10 | Module init |
| `src/experiments/runner.py` | ~150 | Core execution logic |
| `src/experiments/templates.py` | ~100 | Script templates |
| `tests/test_experiment_runner.py` | ~120 | Unit tests |

### Integration Points

- `src/training/hpo.py` → `run_hpo()` function
- `src/training/trainer.py` → `Trainer` class
- `src/training/thermal.py` → `ThermalCallback`
- `src/analysis/aggregate_results.py` → result collection
- `src/analysis/scaling_curves.py` → power law fitting

### Test Strategy

1. **Template validation**: `python -m py_compile` on generated scripts
2. **Runner tests**: Mocked training, verify CSV/report updates
3. **Manual verification**: Generate one experiment, inspect output

---

## Execution Order

1. Create `src/experiments/` module with runner and templates
2. Write tests for runner module
3. Implement runner functions (TDD)
4. Create experiment-generation skill
5. Create experiment-execution skill
6. Manual test: generate and run one experiment

---

## HPO Strategy (from brainstorm)

**Phase 6A Hybrid Approach:**

```
Budget × Task HPO (12 runs):
         1%    2%    3%    5%
  2M     HPO   skip  HPO   HPO
  20M    HPO   skip  HPO   HPO
  200M   HPO   skip  HPO   HPO
  2B     HPO   skip  HPO   HPO

2% task uses interpolated params from 1% and 3%

Feature scaling HPO (3-4 runs):
  Fixed: 2M budget, threshold_1pct
  Vary: 20 → 100 → 250 → 500 features
```

Total: ~15-16 HPO runs instead of 320.

---

*Document Version: 1.0*
*Memory Entity: ExperimentSkills_Plan*
