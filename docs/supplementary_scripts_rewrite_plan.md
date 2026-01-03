# Supplementary Training Scripts Rewrite Plan

**Created:** 2025-12-31
**Status:** Task A complete, Tasks B-C pending
**Purpose:** Fix broken supplementary scripts that use fabricated API

---

## Background

### What We're Trying to Do

Run 10 supplementary training experiments to properly compare h3-optimal config across horizons:

| Set | Purpose | Scripts |
|-----|---------|---------|
| 1 | Cross-horizon validation | h3-optimal (d=64, L=32, h=2, dropout=0.10) on h1 and h5 |
| 2 | n_heads sensitivity | Vary h={2, 8, 16} with dropout=0.10 fixed |
| 3 | Dropout sensitivity | Vary dropout={0.10, 0.20, 0.30} with h=2 fixed |

### Why This Matters

HPO explored different architecture regions for each horizon. We cannot claim optimal params differ between horizons without testing the same configs on all horizons. These experiments provide apples-to-apples comparison.

### What Went Wrong

Scripts generated on 2025-12-30 used a fabricated API:
- `TimeSeriesDataset` - doesn't exist (actual: `FinancialDataset`, but not used directly)
- `train_model()` - doesn't exist (actual: `Trainer` class)
- Wrong parameter names throughout

All 10 scripts fail with `ImportError` on line 25.

---

## Correct API Pattern

Based on codebase analysis, scripts must use:

```python
#!/usr/bin/env python3
"""
Supplementary Training: [name]
Purpose: [description]
Architecture: d_model=64, n_layers=32, n_heads=[N], d_ff=256
Horizon: [H]-day prediction
Dropout: [D]
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer
from src.training.thermal import ThermalMonitor

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "[name]"
HORIZON = [1 or 5]

# Architecture (fixed for all supplementary: d=64, L=32)
D_MODEL = 64
N_LAYERS = 32
N_HEADS = [2, 8, or 16]
D_FF = 256  # 4 * d_model

# Training (h3-optimal baseline)
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
WEIGHT_DECAY = 0.0007
DROPOUT = [0.10, 0.20, or 0.30]

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a25.parquet"
CONTEXT_LENGTH = 60
NUM_FEATURES = 25

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/supplementary" / EXPERIMENT_NAME

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"SUPPLEMENTARY: {EXPERIMENT_NAME}")
    print("=" * 70)

    # Check MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data to get row count for splitter
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,  # Disable tracking for supplementary
        mlflow_experiment=None,
    )

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=10,
        stride=5,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    print(f"\nArchitecture: d={D_MODEL}, L={N_LAYERS}, h={N_HEADS}, d_ff={D_FF}")
    print(f"Horizon: {HORIZON}-day, Dropout: {DROPOUT}")

    # Create splits
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    splits = splitter.split()
    print(f"Splits: train={len(splits.train_indices)}, val={len(splits.val_indices)}, test={len(splits.test_indices)}")

    # Create trainer
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        split_indices=splits,
    )

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    # Report results
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {EXPERIMENT_NAME}")
    print(f"{'=' * 70}")
    print(f"Train loss: {result['train_loss']:.6f}")
    print(f"Val loss:   {result.get('val_loss', 'N/A')}")
    print(f"Time:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if result.get('stopped_early'):
        print(f"Early stop: {result.get('stop_reason')}")

    # Save results
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "experiment": EXPERIMENT_NAME,
            "horizon": HORIZON,
            "architecture": {
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
            },
            "training": {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "weight_decay": WEIGHT_DECAY,
            },
            "results": {
                "train_loss": result["train_loss"],
                "val_loss": result.get("val_loss"),
                "stopped_early": result.get("stopped_early", False),
                "stop_reason": result.get("stop_reason"),
            },
            "runtime_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return result


if __name__ == "__main__":
    main()
```

---

## 10 Scripts to Generate

All share: d=64, L=32, epochs=50, lr=0.001, batch=256

| Script | Horizon | n_heads | Dropout | Purpose |
|--------|---------|---------|---------|---------|
| `train_h1_d64_L32_h2_drop010.py` | 1 | 2 | 0.10 | Cross-horizon: h3-optimal on h1 |
| `train_h5_d64_L32_h2_drop010.py` | 5 | 2 | 0.10 | Cross-horizon: h3-optimal on h5 |
| `train_h1_d64_L32_h8_drop010.py` | 1 | 8 | 0.10 | n_heads sensitivity |
| `train_h1_d64_L32_h16_drop010.py` | 1 | 16 | 0.10 | n_heads sensitivity |
| `train_h5_d64_L32_h8_drop010.py` | 5 | 8 | 0.10 | n_heads sensitivity |
| `train_h5_d64_L32_h16_drop010.py` | 5 | 16 | 0.10 | n_heads sensitivity |
| `train_h1_d64_L32_h2_drop020.py` | 1 | 2 | 0.20 | Dropout sensitivity |
| `train_h1_d64_L32_h2_drop030.py` | 1 | 2 | 0.30 | Dropout sensitivity |
| `train_h5_d64_L32_h2_drop020.py` | 5 | 2 | 0.20 | Dropout sensitivity |
| `train_h5_d64_L32_h2_drop030.py` | 5 | 2 | 0.30 | Dropout sensitivity |

---

## Task Breakdown

### Task A: Fix runner script pipefail ✅ COMPLETE
- Added `set -o pipefail` to `scripts/run_supplementary_2M.sh`
- Now correctly detects Python failures through pipe to tee

### Task B: Write ONE template script, validate it runs (PENDING)
1. Write `train_h1_d64_L32_h2_drop010.py` using pattern above
2. Syntax check: `python -m py_compile script.py`
3. Run it manually and verify:
   - No import errors
   - Training starts and completes
   - Results saved to `outputs/supplementary/h1_d64_L32_h2_drop010/results.json`

### Task C: Generate remaining 9 scripts from template (PENDING)
1. Copy template and modify HORIZON, N_HEADS, DROPOUT, EXPERIMENT_NAME
2. Syntax check all 9 scripts
3. (Optional) Run one more to double-check

---

## Validation Checklist

Before marking complete:
- [ ] All 10 scripts pass `python -m py_compile`
- [ ] At least 1 script runs end-to-end successfully
- [ ] Runner script correctly reports FAILED on import error
- [ ] Results JSON produced in correct location

---

## Key Imports Reference

```python
# CORRECT imports for supplementary scripts:
from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer
from src.training.thermal import ThermalMonitor  # optional

# WRONG imports (do not use):
from src.data.dataset import TimeSeriesDataset  # DOES NOT EXIST
from src.training.trainer import train_model     # DOES NOT EXIST
```

---

## Expected Runtime

- Per script: 2-5 minutes (based on 2M HPO timing)
- All 10: 20-50 minutes
- Runner script already exists and is fixed

---

*Document created for session continuity. Delete after Task C is complete.*
