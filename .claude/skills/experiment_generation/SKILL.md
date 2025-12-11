---
name: experiment_generation
description: Generate HPO and training experiment scripts from templates. Use when starting experiments, creating Phase 6A/6B/6C scripts, or setting up HPO and training runs. Produces self-contained Python scripts with all parameters visible inline.
---

# Experiment Generation Skill

Generate self-contained experiment scripts for HPO and training runs using `src/experiments/templates.py`.

## When to Use

- "Generate experiment for 2M threshold_1pct"
- "Create Phase 6A experiments"
- "Set up HPO and training scripts for 20M budget"
- Starting a new experiment phase
- Need to create scripts for a specific budget/task combination

## 🔴 CRITICAL: Data Split Requirements

**ALL experiments MUST use ChunkSplitter for proper train/val/test splits.**

Previous experiments trained on ALL data without validation splits - this is a critical methodological error. All new experiments MUST:

1. Use `ChunkSplitter` from `src/data/dataset.py`
2. Pass `split_indices` to `create_objective()` for HPO
3. Pass `split_indices` to `Trainer` for training
4. HPO optimizes `val_loss`, NOT `train_loss`

### Split Protocol

| Split | Purpose | Method |
|-------|---------|--------|
| Train | Model training | Sliding window (maximizes samples) |
| Val | HPO optimization, early stopping | Non-overlapping chunks (isolation) |
| Test | Final evaluation | Non-overlapping chunks (strict isolation) |

### Split Ratios (Fixed)

- **Train**: ~70% (sliding window on non-val/test data)
- **Val**: 15% (non-overlapping chunks)
- **Test**: 15% (non-overlapping chunks)

## Parameter Reference

| Parameter | Valid Values | Description |
|-----------|--------------|-------------|
| budget | 2M, 20M, 200M, 2B | Parameter budget |
| task | threshold_1pct, threshold_2pct, threshold_3pct, threshold_5pct | Prediction task |
| horizon | 1, 2, 3, 5, 7 | Prediction horizon in days |
| timescale | daily, 2d, 3d, 5d, weekly | Data timescale |
| phase | phase6a, phase6b, phase6c | Experiment phase |
| data_path | e.g., data/processed/SPY_dataset_c.parquet | Path to data file |
| context_length | 60 (default) | Days in input sequence |

## Execution Steps

### Step 1: Gather Parameters

Collect required parameters from user. If not specified, ask:

```python
# Required parameters
experiment = "{phase}_{budget}_{task}"  # e.g., "phase6a_2M_threshold_1pct"
phase = "phase6a"        # phase6a | phase6b | phase6c
budget = "2M"            # 2M | 20M | 200M | 2B
task = "threshold_1pct"  # threshold_1pct | threshold_2pct | threshold_3pct | threshold_5pct
horizon = 1              # 1 | 2 | 3 | 5 | 7 days
timescale = "daily"      # daily | 2d | 3d | 5d | weekly
data_path = "data/processed/SPY_dataset_c.parquet"
```

### Step 2: Validate Prerequisites

Before generating, verify:

```bash
# 1. Data file exists
ls -la data/processed/SPY_dataset_c.parquet

# 2. Manifest entry exists
python scripts/manage_data_versions.py verify

# 3. Budget config exists
ls configs/model/patchtst_{budget}.yaml
```

### Step 3: Determine Script Types Needed

**HPO Strategy (Per design doc):**

| Task | HPO Needed? | Notes |
|------|-------------|-------|
| threshold_1pct | YES | Full HPO |
| threshold_2pct | NO | Borrow params from 1% and 3% |
| threshold_3pct | YES | Full HPO |
| threshold_5pct | YES | Full HPO |

For **threshold_2pct**: Skip HPO script, generate training script with `borrowed_from` set.

### Step 4: Create Data Splits (MANDATORY)

```python
import pandas as pd
from src.data.dataset import ChunkSplitter

# Load data to get total_days
df = pd.read_parquet("data/processed/SPY_dataset_c.parquet")
total_days = len(df)

# Create ChunkSplitter with standard settings
splitter = ChunkSplitter(
    total_days=total_days,
    context_length=60,   # Standard for all experiments
    horizon=1,           # Adjust based on task
    val_ratio=0.15,      # 15% validation
    test_ratio=0.15,     # 15% test
    seed=42,             # Reproducible splits
)

# Get split indices
splits = splitter.split()
print(f"Train samples: {len(splits.train_indices)}")
print(f"Val chunks: {len(splits.val_indices)}")
print(f"Test chunks: {len(splits.test_indices)}")

# For HPO: use 30% subset of train for faster iteration
hpo_train_indices = splitter.get_hpo_subset(splits, fraction=0.3)
print(f"HPO train samples: {len(hpo_train_indices)}")
```

### Step 5: Generate Scripts with Split Support

```python
from src.experiments.templates import generate_hpo_script, generate_training_script
import pandas as pd

# Load feature columns from data
df = pd.read_parquet("data/processed/SPY_dataset_c.parquet")
feature_columns = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Generate HPO script (if needed - NOT for threshold_2pct)
if task != "threshold_2pct":
    hpo_script = generate_hpo_script(
        experiment=f"{phase}_{budget}_{task}",
        phase=phase,
        budget=budget,
        task=task,
        horizon=horizon,
        timescale=timescale,
        data_path=data_path,
        feature_columns=feature_columns,
        n_trials=50,
        timeout_hours=4.0,
        # Split parameters (MANDATORY)
        context_length=60,
        val_ratio=0.15,
        test_ratio=0.15,
        hpo_train_fraction=0.3,
    )

# Generate training script
training_script = generate_training_script(
    experiment=f"{phase}_{budget}_{task}",
    phase=phase,
    budget=budget,
    task=task,
    horizon=horizon,
    timescale=timescale,
    data_path=data_path,
    feature_columns=feature_columns,
    hyperparameters={},  # Will be filled from HPO results
    borrowed_from="phase6a_2M_threshold_1pct" if task == "threshold_2pct" else None,
    # Split parameters (MANDATORY)
    context_length=60,
    val_ratio=0.15,
    test_ratio=0.15,
)
```

### Step 6: Write Scripts to Disk

```python
from pathlib import Path

# Create output directory
output_dir = Path(f"experiments/{phase}")
output_dir.mkdir(parents=True, exist_ok=True)

# Write HPO script (if generated)
if task != "threshold_2pct":
    hpo_path = output_dir / f"hpo_{budget}_{task}.py"
    hpo_path.write_text(hpo_script)
    print(f"Written: {hpo_path}")

# Write training script
train_path = output_dir / f"train_{budget}_{task}.py"
train_path.write_text(training_script)
print(f"Written: {train_path}")
```

### Step 7: Validate Generated Scripts

```bash
# Verify scripts compile without syntax errors
python -m py_compile experiments/{phase}/hpo_{budget}_{task}.py
python -m py_compile experiments/{phase}/train_{budget}_{task}.py

# Check imports resolve (dry run)
python -c "import experiments.{phase}.hpo_{budget}_{task}"
```

### Step 8: Update Experiment Manifest

Append to `experiments/{phase}/README.md`:

```markdown
## {budget} - {task}

- **Generated:** {timestamp}
- **HPO Script:** `hpo_{budget}_{task}.py`
- **Training Script:** `train_{budget}_{task}.py`
- **Data:** {data_path}
- **Horizon:** {horizon} days
- **Timescale:** {timescale}
```

## Output Paths

| Script Type | Path Pattern |
|-------------|--------------|
| HPO | `experiments/{phase}/hpo_{budget}_{task}.py` |
| Training | `experiments/{phase}/train_{budget}_{task}.py` |
| Manifest | `experiments/{phase}/README.md` |

## Complete Example

Generate Phase 6A experiments for 2M budget, threshold_1pct task with proper data splits:

```python
from pathlib import Path
import pandas as pd
from src.data.dataset import ChunkSplitter
from src.experiments.templates import generate_hpo_script, generate_training_script

# Parameters
phase = "phase6a"
budget = "2M"
task = "threshold_1pct"
horizon = 1
timescale = "daily"
data_path = "data/processed/SPY_dataset_c.parquet"
context_length = 60  # Standard for all experiments

# Load data and features
df = pd.read_parquet(data_path)
feature_columns = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
total_days = len(df)

# MANDATORY: Create data splits
splitter = ChunkSplitter(
    total_days=total_days,
    context_length=context_length,
    horizon=horizon,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)
splits = splitter.split()
print(f"Train: {len(splits.train_indices)}, Val: {len(splits.val_indices)}, Test: {len(splits.test_indices)}")

# Generate HPO script (with split parameters)
hpo_script = generate_hpo_script(
    experiment=f"{phase}_{budget}_{task}",
    phase=phase,
    budget=budget,
    task=task,
    horizon=horizon,
    timescale=timescale,
    data_path=data_path,
    feature_columns=feature_columns,
    context_length=context_length,
    val_ratio=0.15,
    test_ratio=0.15,
    hpo_train_fraction=0.3,  # Use 30% subset for faster HPO
)

# Write to disk
output_dir = Path(f"experiments/{phase}")
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / f"hpo_{budget}_{task}.py").write_text(hpo_script)

# Validate
import py_compile
py_compile.compile(str(output_dir / f"hpo_{budget}_{task}.py"), doraise=True)
print("Script validated successfully")
```

## Phase 6A HPO Matrix

For reference, Phase 6A requires 12 HPO runs:

```
Budget × Task HPO Matrix:
         1%    2%    3%    5%
  2M     HPO   skip  HPO   HPO
  20M    HPO   skip  HPO   HPO
  200M   HPO   skip  HPO   HPO
  2B     HPO   skip  HPO   HPO

2% task borrows interpolated params from 1% and 3%
```

Total: 12 HPO runs (4 budgets × 3 tasks excluding 2%)
