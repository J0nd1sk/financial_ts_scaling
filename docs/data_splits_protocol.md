# Data Splits Protocol

## Overview

This document describes the mandatory train/val/test data splitting protocol for all experiments in the financial time-series scaling project. **All experiments MUST use proper data splits** - training on ALL data without validation is a critical methodological error that invalidates results.

## Problem Statement

### Why Splits Matter

Without proper data splits:
- HPO optimizes for training loss, leading to overfitting
- Model performance is evaluated on data seen during training
- Results are not reproducible or generalizable
- Scaling law conclusions are invalid

### Previous Issue

Early experiments trained on ALL data without held-out validation/test sets. This is now fixed with the ChunkSplitter class.

## Split Architecture

### Hybrid Chunk-Based Approach

We use a **hybrid approach** that maximizes training samples while ensuring strict isolation of validation and test sets:

| Split | Method | Purpose |
|-------|--------|---------|
| **Val** | Non-overlapping chunks | HPO optimization, early stopping |
| **Test** | Non-overlapping chunks | Final evaluation (never touched during training) |
| **Train** | Sliding window | Maximizes training samples from remaining data |

### Why Hybrid?

- **Pure non-overlapping chunks**: Only ~132 training samples (too few)
- **Pure sliding window for all**: Data leakage between train/val/test
- **Hybrid**: ~5,000+ training samples with strict val/test isolation

## Implementation

### ChunkSplitter Class

Location: `src/data/dataset.py`

```python
from src.data.dataset import ChunkSplitter, SplitIndices

# Standard configuration
splitter = ChunkSplitter(
    total_days=8073,      # Total rows in dataset
    context_length=60,    # Input sequence length
    horizon=1,            # Prediction horizon
    val_ratio=0.15,       # 15% validation
    test_ratio=0.15,      # 15% test
    seed=42,              # Reproducibility
)

# Get split indices
splits = splitter.split()

# splits contains:
#   train_indices: Array of valid training start positions (sliding window)
#   val_indices: Array of validation chunk start positions (non-overlapping)
#   test_indices: Array of test chunk start positions (non-overlapping)
#   chunk_size: 61 (context_length + horizon)
```

### SplitIndices Dataclass

```python
@dataclass
class SplitIndices:
    train_indices: np.ndarray  # Sliding window start positions
    val_indices: np.ndarray    # Chunk start positions
    test_indices: np.ndarray   # Chunk start positions
    chunk_size: int            # 61 (context_length + horizon)
```

## Parameters (Fixed)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| context_length | 60 days | Standard input window |
| horizon | 1 day | Single-step prediction |
| chunk_size | 61 days | context_length + horizon |
| val_ratio | 0.15 | 15% of chunks for validation |
| test_ratio | 0.15 | 15% of chunks for test |
| seed | 42 | Reproducibility |

## Usage in HPO

### Creating Objective with Splits

```python
from src.data.dataset import ChunkSplitter
from src.training.hpo import create_objective

# Create splits
splitter = ChunkSplitter(
    total_days=len(df),
    context_length=60,
    horizon=1,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
)
splits = splitter.split()

# HPO subset: 30% of train for faster iteration
hpo_train_subset = splitter.get_hpo_subset(splits, fraction=0.3)

# Create HPO split indices
from src.data.dataset import SplitIndices
hpo_splits = SplitIndices(
    train_indices=hpo_train_subset,
    val_indices=splits.val_indices,
    test_indices=splits.test_indices,
    chunk_size=splits.chunk_size,
)

# Create objective that optimizes val_loss
objective = create_objective(
    config_path="configs/experiments/threshold_1pct.yaml",
    budget="2M",
    search_space=search_space,
    split_indices=hpo_splits,  # MANDATORY
)
```

### HPO Optimizes val_loss

When `split_indices` is provided:
- Trainer creates separate train and val dataloaders
- Each epoch computes both train_loss and val_loss
- HPO objective returns val_loss (NOT train_loss)

```python
# In create_objective():
result = trainer.train()

if split_indices is not None:
    return result["val_loss"]  # Optimize validation loss
return result["train_loss"]    # Only if no splits (backward compat)
```

## Usage in Training

### Creating Trainer with Splits

```python
from src.training.trainer import Trainer

trainer = Trainer(
    experiment_config=config,
    model_config=model_config,
    batch_size=32,
    learning_rate=0.001,
    epochs=100,
    device="mps",
    checkpoint_dir=Path("outputs/training/exp1"),
    split_indices=splits,  # MANDATORY
)

result = trainer.train()
# Returns: {"train_loss": ..., "val_loss": ..., "stopped_early": ..., "stop_reason": ...}
```

## Validation Protocol

### Before Running ANY Experiment

```bash
# Verify script uses ChunkSplitter
grep -q "ChunkSplitter" experiments/phase6a/hpo_2M_threshold_1pct.py
grep -q "split_indices" experiments/phase6a/hpo_2M_threshold_1pct.py

# Both must succeed, otherwise script is INVALID
```

### Verify Split Isolation

```python
# Ensure no overlap between train and val/test
train_set = set(range(idx, idx + chunk_size) for idx in splits.train_indices)
val_set = set(range(idx, idx + chunk_size) for idx in splits.val_indices)
test_set = set(range(idx, idx + chunk_size) for idx in splits.test_indices)

# These must all be empty sets
assert not (train_set & val_set)  # No train/val overlap
assert not (train_set & test_set) # No train/test overlap
assert not (val_set & test_set)   # No val/test overlap
```

## Expected Sample Counts

For SPY dataset (8073 days):

| Split | Chunks/Samples | Calculation |
|-------|----------------|-------------|
| Total chunks | ~132 | 8073 // 61 |
| Val chunks | ~20 | 132 * 0.15 |
| Test chunks | ~20 | 132 * 0.15 |
| Train (sliding) | ~5000+ | Sliding window on remaining |
| HPO subset | ~1500 | 5000 * 0.30 |

## Troubleshooting

### "Insufficient data" Error

```
ValueError: Insufficient data: X days is too small.
```

**Solution**: Ensure data file has enough rows. Minimum: 5 * chunk_size = 305 days.

### "val_loss is None"

**Cause**: split_indices not passed to Trainer or create_objective.

**Solution**: Ensure split_indices parameter is provided.

### "No overlap but few train samples"

**Cause**: High val/test ratios leaving little train data.

**Solution**: Use standard ratios (15%/15%) or reduce if necessary.

## References

- Implementation: `src/data/dataset.py` (ChunkSplitter, SplitIndices)
- Trainer: `src/training/trainer.py` (split support)
- HPO: `src/training/hpo.py` (val_loss objective)
- Tests: `tests/data/test_dataset.py`, `tests/test_training.py`, `tests/test_hpo.py`
