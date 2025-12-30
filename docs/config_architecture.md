# Configuration Architecture

**Document Version:** 2.0
**Date:** 2025-12-29
**Status:** Approved
**Supersedes:** v1.0 (2025-12-08)

---

## Overview

This document defines the configuration architecture for the financial time-series scaling experiments. The design separates **experiment definition** (what we're testing) from **execution parameters** (how to run it optimally).

---

## Design Principles

1. **Separation of Concerns**: Experiment configs define WHAT we're testing. Execution params (batch size, learning rate) are discovered/tuned separately.

2. **Per-Budget Optimization**: Optimal batch size, learning rate, and epochs vary by parameter budget. Each is discovered/tuned independently.

3. **Reproducibility**: All parameters needed to reproduce a run are tracked. Experiment config + discovered params + HPO results = full specification.

4. **Future-Proof**: Same experiment config runs across different hardware (different batch sizes) and can be HPO-tuned without modification.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT CONFIG                                        │
│  configs/experiments/spy_daily_threshold_1pct.yaml                               │
│  Defines: seed, data_path, task, timescale, context_length, horizon             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                ┌───────────┬───────────┼───────────┬───────────┐
                ▼           ▼           ▼           ▼           │
          ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
          │   2M    │ │   20M   │ │  200M   │ │   2B    │  ← --budget CLI arg
          └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
               │           │           │           │            │
┌──────────────┼───────────┼───────────┼───────────┼────────────┼─────────────────┐
│              ▼           ▼           ▼           ▼            │                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │              ARCHITECTURE GRID (src/models/arch_grid.py)                  │  │
│  │  Pre-computed valid (d_model, n_layers, n_heads, d_ff) per budget         │  │
│  │  Filter: param_count within ±25% of target budget                         │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│              │           │           │           │                              │
│              ▼           ▼           ▼           ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │              DYNAMIC BATCH SIZING (arch_grid.get_memory_safe_batch_config)│  │
│  │  Per-architecture batch size based on memory heuristic                    │  │
│  │  Gradient accumulation to achieve effective_batch_size=128                │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│              │           │           │           │                              │
│              ▼           ▼           ▼           ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │              ARCHITECTURAL HPO (Phase 6A)                                 │  │
│  │  Optuna samples: architecture from grid + training params from config     │  │
│  │  Search space: configs/hpo/architectural_search.yaml                      │  │
│  │  Results: outputs/hpo/{budget}_h{horizon}/best_params.json                │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│              │           │           │           │                              │
│              ▼           ▼           ▼           ▼                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │              TRAINING                                                     │  │
│  │  Features: gradient accumulation, early stopping, dropout tuning         │  │
│  │  Logs: architecture + training params to CSV + W&B + MLflow               │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
configs/
├── experiments/                    # Experiment definitions
│   ├── spy_daily_direction.yaml
│   ├── spy_daily_threshold_1pct.yaml
│   ├── threshold_2pct.yaml
│   ├── threshold_3pct.yaml
│   └── threshold_5pct.yaml
├── model/                          # Static model configs (legacy, superseded by arch_grid.py)
│   ├── patchtst_2m.yaml           # ~2M parameters
│   ├── patchtst_20m.yaml          # ~20M parameters
│   ├── patchtst_200m.yaml         # ~200M parameters
│   └── patchtst_2b.yaml           # ~2B parameters
└── hpo/                            # HPO search spaces
    └── architectural_search.yaml   # Training params only; arch from arch_grid.py

src/models/
└── arch_grid.py                    # Architecture grid generator
    # - estimate_param_count(): Calculate params for any config
    # - get_architectures_for_budget(): Valid architectures per budget
    # - get_memory_safe_batch_config(): Dynamic batch sizing

outputs/
├── hpo/                            # Optuna HPO results
│   ├── 2M_h1/                     # Per (budget, horizon)
│   │   ├── study.db               # Optuna SQLite database
│   │   └── best_params.json       # Best hyperparameters + architecture
│   ├── 2M_h3/
│   ├── 2M_h5/
│   ├── 20M_h1/ ... 20M_h5/
│   ├── 200M_h1/ ... 200M_h5/
│   └── 2B_h1/ ... 2B_h5/
├── logs/                           # Training logs
│   └── experiment_results.csv      # Append-only experiment log
├── checkpoints/                    # Model weights
└── results/                        # Experiment results
```

---

## Config Schemas

### ExperimentConfig (Task 1)

Defines WHAT experiment we're running. Does NOT include execution parameters.

```yaml
# configs/experiments/spy_daily_threshold_1pct.yaml
seed: 42
data_path: data/processed/v1/SPY_features_a20.parquet
task: threshold_1pct          # direction | threshold_1pct | threshold_2pct | threshold_3pct | threshold_5pct | regression
timescale: daily              # daily | 2d | 3d | 5d | weekly | 2wk | monthly
context_length: 60            # Input sequence length (days of history)
horizon: 5                    # Prediction horizon (days ahead)
wandb_project: financial-scaling
mlflow_experiment: spy-scaling
```

**Python Dataclass:**
```python
@dataclass
class ExperimentConfig:
    seed: int = 42
    data_path: str
    task: str
    timescale: str
    context_length: int = 60
    horizon: int = 5
    wandb_project: str | None = None
    mlflow_experiment: str | None = None
```

**Validation Rules:**
- `task` must be one of: `direction`, `threshold_1pct`, `threshold_2pct`, `threshold_3pct`, `threshold_5pct`, `regression`
- `timescale` must be one of: `daily`, `2d`, `3d`, `5d`, `weekly`, `2wk`, `monthly`
- `data_path` must exist (validated at load time)
- `context_length` and `horizon` must be positive integers

### ModelConfig (Discovered by Architectural HPO)

Model architecture is now **discovered by HPO**, not statically defined.

The `configs/model/patchtst_{budget}.yaml` files are retained for reference but are
**superseded** by `src/models/arch_grid.py` which generates valid architectures per budget.

**Architecture Search Space** (from `arch_grid.py`):
```python
ARCH_SEARCH_SPACE = {
    "d_model": [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
    "n_layers": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180, 192, 256],
    "n_heads": [2, 4, 8, 16, 32],
    "d_ff_ratio": [2, 4],  # d_ff = d_model * ratio
}

BUDGET_TARGETS = {
    "2m": 2_000_000,
    "20m": 20_000_000,
    "200m": 200_000_000,
    "2b": 2_000_000_000,
}
```

Architectures are filtered to within ±25% of target budget. Optuna samples from this
pre-computed list during HPO, ensuring all tested architectures are valid.

### HPO Search Space (Phase 6A — Architectural HPO)

Defines Optuna search space for **training parameters only**. Architecture parameters
are sampled from `arch_grid.py` (see above).

```yaml
# configs/hpo/architectural_search.yaml
n_trials: 50
timeout_hours: null  # No timeout - rely on n_trials
direction: minimize  # Minimize validation loss

# Training parameter search space (narrower ranges for architectural HPO)
# Note: batch_size REMOVED - now determined dynamically by get_memory_safe_batch_config()
training_search_space:
  learning_rate:
    type: log_uniform
    low: 1.0e-4
    high: 1.0e-3

  epochs:
    type: categorical
    choices: [50, 75, 100]

  weight_decay:
    type: log_uniform
    low: 1.0e-4
    high: 5.0e-3

  warmup_steps:
    type: categorical
    choices: [100, 200, 300, 500]

  dropout:              # NEW: PatchTST dropout (was hardcoded at 0.1)
    type: uniform
    low: 0.1
    high: 0.3

# Early stopping configuration
early_stopping:
  patience: 10
  min_delta: 0.001
```

**Key differences from legacy `default_search.yaml`:**

| Parameter | Legacy | Architectural HPO |
|-----------|--------|-------------------|
| batch_size | Searched (32-256) | **REMOVED** — dynamic via `get_memory_safe_batch_config()` |
| dropout | Not searched | **ADDED** (0.1-0.3 uniform) |
| learning_rate | 1e-5 to 1e-2 | Narrower: 1e-4 to 1e-3 |
| epochs | 10-200 range | Categorical: [50, 75, 100] |
| weight_decay | 1e-6 to 1e-2 | Narrower: 1e-4 to 5e-3 |
| warmup_steps | 0-1000 range | Categorical: [100, 200, 300, 500] |
| early_stopping | Not present | **ADDED** (patience=10, min_delta=0.001) |
| key name | `search_space` | `training_search_space` |

### Dynamic Batch Sizing (Supersedes Static batch_sizes.json)

Batch size is now determined **per-architecture** at runtime using `get_memory_safe_batch_config()`.

**Why dynamic?** The 2B budget exposed that static per-budget batch sizes don't work —
a shallow-wide 2B model (d=2048, L=32) needs much smaller batches than a narrow-deep one
(d=768, L=256). Memory requirements vary by architecture, not just budget.

```python
from src.models.arch_grid import get_memory_safe_batch_config

# Memory heuristic: memory_score = (d_model² × n_layers) / 1e9
config = get_memory_safe_batch_config(d_model=1024, n_layers=24)
# Returns: {"batch_size": 64, "accumulation_steps": 2, "effective_batch_size": 128}
```

**Batch size tiers based on memory score:**

| Memory Score | Batch Size | Accumulation Steps | Effective Batch |
|--------------|------------|-------------------|-----------------|
| < 0.05       | 128        | 1                 | 128             |
| 0.05 - 0.2   | 64         | 2                 | 128             |
| 0.2 - 0.5    | 32         | 4                 | 128             |
| 0.5 - 1.0    | 16         | 8                 | 128             |
| > 1.0        | 8          | 16                | 128             |

All configurations target `effective_batch_size=128` for training consistency.

### HPO Results (Phase 6A — Architectural HPO)

Best hyperparameters **and architecture** per (budget, horizon).

```json
// outputs/hpo/20M_h3/best_params.json
{
  "architecture": {
    "d_model": 768,
    "n_layers": 24,
    "n_heads": 16,
    "d_ff": 3072,
    "param_count": 170000000
  },
  "training": {
    "learning_rate": 0.00045,
    "epochs": 75,
    "weight_decay": 0.0012,
    "warmup_steps": 200,
    "dropout": 0.15
  },
  "batch_config": {
    "batch_size": 32,
    "accumulation_steps": 4,
    "effective_batch_size": 128
  },
  "best_val_loss": 0.3081,
  "n_trials": 50,
  "timestamp": "2025-12-21T14:30:00Z"
}
```

**Schema changes from v1.0:**
- Added `architecture` object with d_model, n_layers, n_heads, d_ff, param_count
- Added `batch_config` object with batch_size, accumulation_steps, effective_batch_size
- Added `dropout` to training parameters
- Path changed from `{experiment}/{budget}_best.json` to `{budget}_h{horizon}/best_params.json`

---

## Training Pipeline

### Complete Workflow (Phase 6A — Architectural HPO)

```bash
# Step 0: Build processed features (prerequisite)
python scripts/build_dataset_combined.py --include-vix
# → data/processed/SPY_dataset_a25.parquet

# Step 1: Run Architectural HPO (finds best architecture + training params)
# Uses generated scripts in experiments/phase6a/
python experiments/phase6a/hpo_20M_h3_threshold_1pct.py
# → outputs/hpo/20M_h3/study.db (Optuna database)
# → outputs/hpo/20M_h3/best_params.json (best architecture + training params)

# Step 2: (Alternative) Run HPO via runner script with thermal monitoring
./scripts/run_phase6a_hpo.sh
# Runs all 12 HPO experiments sequentially with:
#   - Pre-flight checks (MPS, memory, data file)
#   - Background hardware monitoring (5-min CSV logging)
#   - Graceful stop: touch outputs/logs/STOP_HPO to stop between experiments

# Step 3: Train with discovered params (using HPO results)
python scripts/train.py \
  --config configs/experiments/spy_daily_threshold_1pct.yaml \
  --budget 20M \
  --horizon 3
# Automatically loads:
#   - architecture from outputs/hpo/20M_h3/best_params.json
#   - batch_config from get_memory_safe_batch_config(d_model, n_layers)
#   - training params from best_params.json
```

### Training Script Parameter Resolution (v2.0)

The training script assembles the full configuration from HPO results:

```python
def resolve_training_params(config_path: str, budget: str, horizon: int) -> FullTrainingParams:
    # 1. Load experiment config
    experiment = load_experiment_config(config_path)

    # 2. Load HPO results (architecture + training params)
    hpo_path = f"outputs/hpo/{budget}_h{horizon}/best_params.json"
    if Path(hpo_path).exists():
        hpo_results = json.load(open(hpo_path))
        arch = hpo_results["architecture"]
        training = hpo_results["training"]
    else:
        raise ValueError(f"No HPO results found at {hpo_path}. Run HPO first.")

    # 3. Compute dynamic batch config based on architecture
    batch_config = get_memory_safe_batch_config(arch["d_model"], arch["n_layers"])

    return FullTrainingParams(
        experiment=experiment,
        architecture=arch,
        batch_size=batch_config["batch_size"],
        accumulation_steps=batch_config["accumulation_steps"],
        learning_rate=training["learning_rate"],
        epochs=training["epochs"],
        dropout=training["dropout"],
        early_stopping={"patience": 10, "min_delta": 0.001},
        ...
    )
```

---

## Default Hyperparameters

> **Note (v2.0):** HPO is now **required** before final training. Default hyperparameters
> are only used for smoke tests or debugging.

When HPO results are not available, these fallback defaults are used:

```python
DEFAULT_HYPERPARAMS = {
    "learning_rate": 5e-4,
    "epochs": 75,
    "weight_decay": 1e-3,
    "warmup_steps": 200,
    "dropout": 0.15,
}

DEFAULT_ARCHITECTURE = {  # Only for smoke tests
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "d_ff": 1024,
}
```

For valid scaling law experiments, always run HPO first.

---

## Scaling Law Experiment Workflow (v2.0)

For valid scaling law analysis, run architectural HPO across all budgets, then compare:

```bash
# Phase 6A: Architectural HPO for each (budget, horizon)
# 12 experiments: 4 budgets × 3 horizons
./scripts/run_phase6a_hpo.sh

# Results are in outputs/hpo/{budget}_h{horizon}/best_params.json
# Analysis extracts: best_val_loss, architecture, param_count

# Same horizon, four budgets → scaling curve
# Compare best_val_loss vs param_count for h=3:
#   2M_h3:   ~2M params,   val_loss = X
#   20M_h3:  ~20M params,  val_loss = Y
#   200M_h3: ~200M params, val_loss = Z
#   2B_h3:   ~2B params,   val_loss = W
```

Results are logged to:
- `outputs/logs/experiment_results.csv` — Append-only CSV with all trials
- `outputs/hpo/{budget}_h{horizon}/study.db` — Optuna SQLite database
- W&B/MLflow — For visualization and comparison

Logged fields include:
- Architecture (d_model, n_layers, n_heads, d_ff, param_count)
- Training params (learning_rate, epochs, dropout, etc.)
- Batch config (batch_size, accumulation_steps, effective_batch_size)
- Results (val_loss, training_time, max_temperature)

---

## Rationale

### Why separate experiment config from execution params?

1. **Same experiment, different hardware**: Batch size varies by GPU/MPS memory. Config shouldn't change.

2. **HPO independence**: Can tune hyperparams without modifying experiment definition.

3. **Reproducibility**: Clear separation of "what" vs "how" makes reproducing results easier.

4. **Scaling law validity**: Comparing 2M vs 200M models requires optimal execution for each. Hardcoded params would unfairly handicap some models.

### Why YAML for all configs?

Single format, single loader, less complexity. JSON considered but YAML is more readable for nested configs.

### Why dynamic batch sizing? (v2.0 update)

> **v1.0 approach (superseded):** Static batch sizes per budget in `batch_sizes.json`.

**v2.0 approach:** Dynamic per-architecture batch sizing via `get_memory_safe_batch_config()`.

The 2B budget revealed that static per-budget batch sizes don't scale:
- A shallow-wide 2B model (d=2048, L=32) uses ~10GB memory per batch
- A narrow-deep 2B model (d=768, L=256) uses ~50GB memory per batch
- Same budget, vastly different memory requirements

Dynamic sizing solves this by computing batch size from architecture parameters,
using gradient accumulation to maintain consistent effective batch size.

---

## Architectural HPO (v2.0)

Phase 6A introduced **architectural HPO** — searching both model architecture and
training parameters simultaneously.

**Key insight:** Optimal architecture varies by parameter budget. A 2M model might
prefer d=256, L=8 (wide-shallow), while a 200M model might prefer d=768, L=24
(narrow-deep). HPO discovers these optimal configurations.

**Design decisions documented in:** `docs/architectural_hpo_design.md`

**Implementation:**
- `src/models/arch_grid.py` — Pre-computes valid architectures per budget
- `src/training/hpo.py::create_architectural_objective()` — Optuna objective function
- `configs/hpo/architectural_search.yaml` — Training parameter search space
- `experiments/phase6a/hpo_{budget}_h{horizon}_*.py` — Generated HPO scripts

---

## Training Optimizations (v2.0)

The Trainer (`src/training/trainer.py`) now includes three key optimizations:

### Gradient Accumulation
Enables effective batch sizes larger than physical memory allows:
```python
trainer = Trainer(..., accumulation_steps=4)
# physical_batch=32, accumulation_steps=4 → effective_batch=128
```

### Early Stopping
Prevents overfitting and reduces training time:
```python
trainer = Trainer(..., early_stopping_patience=10, early_stopping_min_delta=0.001)
# Stops if val_loss doesn't improve by min_delta for patience epochs
```

### Dropout Tuning
PatchTST dropout is now tunable (previously hardcoded at 0.1):
```python
# HPO samples dropout in [0.1, 0.3] range
model = PatchTST(..., dropout=0.15)
```

---

## Production Readiness (v2.0)

| Budget | HPO Status | Notes |
|--------|------------|-------|
| 2M | ✅ Complete | 50 trials × 3 horizons (h1, h3, h5) |
| 20M | ✅ Complete | 50 trials × 3 horizons |
| 200M | ✅ Complete | 50 trials × 3 horizons |
| 2B | 🔄 In Progress | Smoke test passed (3 trials), full HPO pending |

**Key findings from 200M HPO:**
- h1/h3 prefer wide-medium architectures (d=768-1024, L=12-24)
- h5 prefers narrow-deep architectures (d=256, L=256)
- n_heads=16 optimal across all horizons; n_heads=8 underperforms

---

## Phase Implementation

*This table tracks config **infrastructure**, not experiment execution.
For experiment progress, see `.claude/context/phase_tracker.md`.*

| Phase | Config Component | Status |
|-------|------------------|--------|
| Phase 4 | ExperimentConfig loader | ✅ Complete |
| Phase 4 | ModelConfig loader | ✅ Superseded by `arch_grid.py` |
| Phase 4 | Batch size discovery | ✅ Superseded by `get_memory_safe_batch_config()` |
| Phase 6A | Architecture grid generator | ✅ Complete (`arch_grid.py`) |
| Phase 6A | Architectural HPO objective | ✅ Complete (`hpo.py`) |
| Phase 6A | HPO search config | ✅ Complete (`architectural_search.yaml`) |
| Phase 6A | Dynamic batch sizing | ✅ Complete |
| Phase 6A | Gradient accumulation | ✅ Complete |
| Phase 6A | Early stopping | ✅ Complete |
| Phase 6A | 12 HPO scripts | ✅ Generated |

**See Also:**
- `.claude/context/phase_tracker.md` — Experiment progress
- `docs/phase6a_execution_plan.md` — Phase 6A execution details
- `docs/architectural_hpo_design.md` — HPO design decisions
- `docs/phase6a_implementation_history.md` — Implementation history

---

*Document Version: 2.0*
*Author: Claude + Alex*
*Approved: 2025-12-29*
*Supersedes: v1.0 (2025-12-08)*
