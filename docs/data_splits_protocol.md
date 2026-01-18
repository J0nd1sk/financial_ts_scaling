# Data Splits Protocol

## Overview

This document describes the mandatory train/val/test data splitting protocol for all experiments. **All experiments MUST use proper data splits** - training without held-out validation is a critical methodological error that invalidates results.

## Two Split Modes

ChunkSplitter supports two modes:

| Mode | Chunk Distribution | Use Case |
|------|-------------------|----------|
| **scattered** (default) | Val/test chunks randomly distributed | HPO - val covers all market regimes |
| **contiguous** | Test at end, val before test | Final training - production-realistic evaluation |

## Scattered Mode (HPO)

Val and test chunks are randomly distributed throughout the dataset. This ensures validation covers diverse market conditions.

```python
splitter = ChunkSplitter(
    total_days=8073,
    context_length=60,
    horizon=1,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    mode="scattered",  # Default
)
```

**When to use**: HPO experiments where you want validation to cover all market regimes.

See `docs/archive/data_splits_protocol_hpo.md` for detailed HPO usage.

## Contiguous Mode (Final Training)

Test chunks are at the END of the dataset (most recent data), val chunks immediately BEFORE test. This is production-realistic: train on history, validate on recent, test on most recent.

```python
splitter = ChunkSplitter(
    total_days=8100,
    context_length=60,
    horizon=1,
    val_ratio=0.008,   # ~60 days for early stopping
    test_ratio=0.031,  # ~250 days for backtest (2025)
    mode="contiguous",
)
```

**When to use**: Final training where you want production-realistic evaluation.

### Final Training Split Strategy

For Phase 6A final training with SPY data (8100 rows through Jan 2026):

| Split | Date Range | Days | Purpose |
|-------|-----------|------|---------|
| **Train** | 1993 — Sept 2024 | ~7,700 | Gradient updates |
| **Val** | Oct — Dec 2024 | ~60 | Early stopping only |
| **Test** | 2025 | ~250 | Backtest (never touched until final eval) |

**Rationale**:
- HPO already determined hyperparameters, so val only needed for early stopping
- Maximize training data (2x more than HPO's scattered approach)
- Test on most recent data for realistic backtest

## Implementation

### ChunkSplitter Class

Location: `src/data/dataset.py`

```python
from src.data.dataset import ChunkSplitter, SplitIndices

splitter = ChunkSplitter(
    total_days=8100,
    context_length=60,
    horizon=1,
    val_ratio=0.008,
    test_ratio=0.031,
    mode="contiguous",
)

splits = splitter.split()
# splits.train_indices: Sliding window positions for training
# splits.val_indices: Contiguous val chunk positions (before test)
# splits.test_indices: Contiguous test chunk positions (at end)
```

### SplitIndices Dataclass

```python
@dataclass
class SplitIndices:
    train_indices: np.ndarray  # Sliding window start positions
    val_indices: np.ndarray    # Chunk start positions
    test_indices: np.ndarray   # Chunk start positions
    chunk_size: int            # context_length + horizon
```

## Usage in Final Training

```python
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer

# Create contiguous splits
splitter = ChunkSplitter(
    total_days=len(df),
    context_length=60,
    horizon=horizon,
    val_ratio=0.008,
    test_ratio=0.031,
    mode="contiguous",
)
splits = splitter.split()

# Create trainer with splits
trainer = Trainer(
    experiment_config=config,
    model_config=model_config,
    batch_size=batch_size,
    learning_rate=lr,
    epochs=50,
    device="mps",
    checkpoint_dir=checkpoint_dir,
    split_indices=splits,
    early_stopping_patience=5,
)

result = trainer.train()
# Best checkpoint saved to checkpoint_dir/best_checkpoint.pt
```

## Best Checkpoint Saving

With validation set provided:
- Checkpoint saved only when val_loss improves
- Saved to `best_checkpoint.pt` (overwrites previous best)
- Contains: `model_state_dict`, `optimizer_state_dict`, `val_loss`, `epoch`

Without validation set:
- Legacy behavior: final checkpoint saved at end
- Not recommended for production use

## Validation

### Verify Split Isolation

```python
# Ensure no overlap between train and val/test
train_days = set()
for idx in splits.train_indices:
    train_days.update(range(idx, idx + splits.chunk_size))

val_days = set()
for idx in splits.val_indices:
    val_days.update(range(idx, idx + splits.chunk_size))

test_days = set()
for idx in splits.test_indices:
    test_days.update(range(idx, idx + splits.chunk_size))

assert not (train_days & val_days), "Train/val overlap!"
assert not (train_days & test_days), "Train/test overlap!"
assert not (val_days & test_days), "Val/test overlap!"
```

### Verify Mode

```python
# For contiguous mode, test should be at the end
if splitter.mode == "contiguous":
    assert splits.test_indices[-1] + splits.chunk_size >= splitter.total_days - 10
    assert splits.val_indices[-1] < splits.test_indices[0]
```

## References

- Implementation: `src/data/dataset.py` (ChunkSplitter, SplitIndices)
- Trainer: `src/training/trainer.py` (split support, best checkpoint)
- HPO details: `docs/archive/data_splits_protocol_hpo.md`
- Tests: `tests/data/test_dataset.py`, `tests/test_training.py`
