# Configuration Architecture

**Document Version:** 1.0
**Date:** 2025-12-08
**Status:** Approved

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
┌─────────────────────────────────────────────────────────────────────────┐
│                     EXPERIMENT CONFIG (Task 1)                          │
│  configs/experiments/spy_daily_threshold_1pct.yaml                      │
│  Defines: seed, data_path, task, timescale, context_length, horizon     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │   2M    │     │   20M   │     │  200M   │  ← --budget CLI arg
              └────┬────┘     └────┬────┘     └────┬────┘
                   │               │               │
┌──────────────────┼───────────────┼───────────────┼──────────────────────┐
│                  ▼               ▼               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              MODEL CONFIG (Task 3)                              │    │
│  │  configs/model/patchtst_{budget}.yaml                           │    │
│  │  Defines: d_model, n_heads, n_layers, patch_len, etc.           │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                  │               │               │                      │
│                  ▼               ▼               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              BATCH SIZE DISCOVERY (Task 7)                      │    │
│  │  outputs/batch_sizes.json: {"2M": 64, "20M": 32, "200M": 8}     │    │
│  │  Hardware-dependent, discovered once per machine                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                  │               │               │                      │
│                  ▼               ▼               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              OPTUNA HPO (Phase 6)                               │    │
│  │  Per (experiment, budget): tune learning_rate, epochs, etc.     │    │
│  │  Search space: configs/hpo/default_search.yaml                  │    │
│  │  Results: outputs/hpo/{experiment}/{budget}_best.json           │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                  │               │               │                      │
│                  ▼               ▼               ▼                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              TRAINING (Task 6)                                  │    │
│  │  Merges: experiment config + model config + batch_size + hpo    │    │
│  │  Logs: all params to W&B + MLflow for reproducibility           │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
configs/
├── experiments/                    # Experiment definitions (Task 1)
│   ├── spy_daily_direction.yaml
│   ├── spy_daily_threshold_1pct.yaml
│   ├── spy_daily_threshold_2pct.yaml
│   └── ...
├── model/                          # Model architecture configs (Task 3)
│   ├── patchtst_2m.yaml           # ~2M parameters
│   ├── patchtst_20m.yaml          # ~20M parameters
│   └── patchtst_200m.yaml         # ~200M parameters
└── hpo/                            # HPO search spaces (Phase 6)
    └── default_search.yaml

outputs/
├── batch_sizes.json                # Discovered batch sizes per budget
├── hpo/                            # Optuna HPO results
│   └── {experiment_name}/
│       ├── 2M_best.json
│       ├── 20M_best.json
│       └── 200M_best.json
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

### ModelConfig (Task 3)

Defines model architecture for each parameter budget.

```yaml
# configs/model/patchtst_20m.yaml
d_model: 512
n_heads: 8
n_layers: 6
d_ff: 2048
patch_len: 16
stride: 8
dropout: 0.1
```

### HPO Search Space (Phase 6)

Defines Optuna search space for hyperparameter tuning.

```yaml
# configs/hpo/default_search.yaml
n_trials: 50
timeout_hours: 4
search_space:
  learning_rate:
    type: log_uniform
    low: 1.0e-5
    high: 1.0e-2
  epochs:
    type: int
    low: 10
    high: 200
  weight_decay:
    type: log_uniform
    low: 1.0e-6
    high: 1.0e-2
  warmup_steps:
    type: int
    low: 0
    high: 1000
```

### Batch Sizes (Task 7)

Discovered per budget, stored as JSON.

```json
// outputs/batch_sizes.json
{
  "2M": 64,
  "20M": 32,
  "200M": 8,
  "discovered_on": "2025-12-08",
  "hardware": "M4 MacBook Pro 128GB"
}
```

### HPO Results (Phase 6)

Best hyperparameters per (experiment, budget).

```json
// outputs/hpo/spy_daily_threshold_1pct/20M_best.json
{
  "learning_rate": 0.00032,
  "epochs": 87,
  "weight_decay": 0.00001,
  "warmup_steps": 500,
  "best_val_loss": 0.4523,
  "n_trials": 50,
  "timestamp": "2025-12-15T10:30:00Z"
}
```

---

## Training Pipeline

### Complete Workflow

```bash
# Step 0: Build processed features (prerequisite)
python scripts/build_features_a20.py
# → data/processed/v1/SPY_features_a20.parquet

# Step 1: Discover batch sizes per budget (once per hardware)
python scripts/find_batch_size.py --budget 2M
python scripts/find_batch_size.py --budget 20M
python scripts/find_batch_size.py --budget 200M
# → outputs/batch_sizes.json

# Step 2: (Optional) HPO to find optimal hyperparams per (experiment, budget)
python scripts/hpo.py \
  --config configs/experiments/spy_daily_threshold_1pct.yaml \
  --budget 20M \
  --hpo-config configs/hpo/default_search.yaml
# → outputs/hpo/spy_daily_threshold_1pct/20M_best.json

# Step 3: Train with discovered/tuned params
python scripts/train.py \
  --config configs/experiments/spy_daily_threshold_1pct.yaml \
  --budget 20M
# Automatically loads:
#   - batch_size from outputs/batch_sizes.json["20M"]
#   - model config from configs/model/patchtst_20m.yaml
#   - hyperparams from outputs/hpo/.../20M_best.json (if exists, else defaults)
```

### Training Script Parameter Resolution

The training script (`scripts/train.py`) assembles the full configuration:

```python
def resolve_training_params(config_path: str, budget: str) -> FullTrainingParams:
    # 1. Load experiment config
    experiment = load_experiment_config(config_path)

    # 2. Load model config (derived from budget)
    model = load_model_config(f"configs/model/patchtst_{budget.lower()}.yaml")

    # 3. Load discovered batch size
    batch_sizes = json.load(open("outputs/batch_sizes.json"))
    batch_size = batch_sizes[budget]

    # 4. Load HPO results if available, else use defaults
    hpo_path = f"outputs/hpo/{experiment.name}/{budget}_best.json"
    if Path(hpo_path).exists():
        hpo_params = json.load(open(hpo_path))
    else:
        hpo_params = DEFAULT_HYPERPARAMS

    return FullTrainingParams(
        experiment=experiment,
        model=model,
        batch_size=batch_size,
        learning_rate=hpo_params["learning_rate"],
        epochs=hpo_params["epochs"],
        ...
    )
```

---

## Default Hyperparameters

When HPO results are not available, these defaults are used:

```python
DEFAULT_HYPERPARAMS = {
    "learning_rate": 1e-4,
    "epochs": 100,
    "weight_decay": 1e-5,
    "warmup_steps": 100,
}
```

These defaults are intentionally conservative. HPO is recommended for final experiments.

---

## Scaling Law Experiment Workflow

For valid scaling law analysis, run the same experiment across all budgets:

```bash
# Same experiment, three budgets → scaling curve
python scripts/train.py --config configs/experiments/spy_daily_threshold_1pct.yaml --budget 2M
python scripts/train.py --config configs/experiments/spy_daily_threshold_1pct.yaml --budget 20M
python scripts/train.py --config configs/experiments/spy_daily_threshold_1pct.yaml --budget 200M
```

Results are logged to W&B/MLflow with:
- Experiment config (seed, task, timescale, etc.)
- Model config (architecture params)
- Execution params (batch_size, learning_rate, epochs)
- Data version (MD5 hash from manifest)
- Results (loss, accuracy, training time)

---

## Rationale

### Why separate experiment config from execution params?

1. **Same experiment, different hardware**: Batch size varies by GPU/MPS memory. Config shouldn't change.

2. **HPO independence**: Can tune hyperparams without modifying experiment definition.

3. **Reproducibility**: Clear separation of "what" vs "how" makes reproducing results easier.

4. **Scaling law validity**: Comparing 2M vs 200M models requires optimal execution for each. Hardcoded params would unfairly handicap some models.

### Why YAML for all configs?

Single format, single loader, less complexity. JSON considered but YAML is more readable for nested configs.

### Why store batch sizes separately?

Batch sizes are hardware-dependent and need to be rediscovered when:
- Moving to new hardware
- Changing model architecture significantly
- Changing input dimensions

Storing them separately makes this explicit and avoids config file proliferation.

---

## Phase Implementation

| Phase | Config Component | Status |
|-------|------------------|--------|
| Phase 4, Task 1 | ExperimentConfig loader | Next |
| Phase 4, Task 3 | ModelConfig loader | Planned |
| Phase 4, Task 7 | Batch size discovery | Planned |
| Phase 6 | HPO search space + results | Future |

---

*Document Version: 1.0*
*Author: Claude + Alex*
*Approved: 2025-12-08*
