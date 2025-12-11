---
name: experiment_execution
description: Execute HPO and training experiment scripts with thermal monitoring and logging. Use after experiment-generation has created scripts, when running experiments, or when user wants to execute HPO/training runs. Enforces pre-flight checks and logs all results.
---

# Experiment Execution Skill

Execute experiment scripts with thermal monitoring, automatic logging, and report generation.

## When to Use

- "Run HPO for 2M threshold_1pct"
- "Execute training for phase6a experiments"
- "Run the 20M experiments"
- "Start the HPO sweep"
- After experiment-generation has created scripts
- When ready to run actual experiments

## Prerequisites

Before using this skill:
1. Experiment scripts must exist (use `experiment_generation` skill first)
2. Data files must be registered in manifest
3. System should be in suitable thermal state

## Pre-Flight Checklist

Before executing ANY experiment, verify ALL of the following:

### 1. Script Validation

```bash
# Verify script exists
ls -la experiments/{phase}/hpo_{budget}_{task}.py

# Verify script compiles
python -m py_compile experiments/{phase}/hpo_{budget}_{task}.py
```

### 2. Thermal Status

Use the `thermal_management` skill or check directly:

```bash
sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i temp
```

| Status | Temperature | Action |
|--------|-------------|--------|
| NORMAL | <70°C | Proceed |
| ACCEPTABLE | 70-85°C | Proceed with monitoring |
| WARNING | 85-95°C | Wait for cooldown |
| CRITICAL | >95°C | DO NOT START |

### 3. Data Manifest

```bash
# Verify data manifest
python scripts/manage_data_versions.py verify

# Verify specific data file exists
ls -la data/processed/v1/SPY_dataset_c.parquet
```

### 4. Disk Space

```bash
# Check available space (need ~10GB for outputs)
df -h outputs/
```

### 5. Output Directory

```bash
# Create output directories if needed
mkdir -p outputs/hpo/{experiment}
mkdir -p outputs/training/{experiment}
mkdir -p outputs/results
```

## Execution Workflow

### For HPO Experiments

```python
from pathlib import Path
from src.experiments.runner import run_hpo_experiment, update_experiment_log, regenerate_results_report

# Define paths
PROJECT_ROOT = Path("/Users/alexanderthomson/Documents/financial_ts_scaling")
LOG_PATH = PROJECT_ROOT / "outputs/results/experiment_log.csv"
REPORT_PATH = PROJECT_ROOT / "docs/experiment_results.md"

# Run HPO experiment
result = run_hpo_experiment(
    experiment="phase6a_2M_threshold_1pct",
    budget="2M",
    task="threshold_1pct",
    data_path=PROJECT_ROOT / "data/processed/v1/SPY_dataset_c.parquet",
    output_dir=PROJECT_ROOT / "outputs/hpo/phase6a_2M_threshold_1pct",
    n_trials=50,
    timeout_hours=4.0,
)

# Add metadata for logging
result.update({
    "experiment": "phase6a_2M_threshold_1pct",
    "phase": "phase6a",
    "budget": "2M",
    "task": "threshold_1pct",
    "horizon": 1,
    "timescale": "daily",
    "script_path": "experiments/phase6a/hpo_2M_threshold_1pct.py",
    "run_type": "hpo",
})

# Log result (always, success or failure)
update_experiment_log(result, LOG_PATH)

# Regenerate report only on success
if result["status"] == "success":
    regenerate_results_report(LOG_PATH, REPORT_PATH)
```

### For Training Experiments

```python
from pathlib import Path
from src.experiments.runner import run_training_experiment, update_experiment_log, regenerate_results_report

# Define paths
PROJECT_ROOT = Path("/Users/alexanderthomson/Documents/financial_ts_scaling")
LOG_PATH = PROJECT_ROOT / "outputs/results/experiment_log.csv"
REPORT_PATH = PROJECT_ROOT / "docs/experiment_results.md"

# Load hyperparameters from HPO results
import json
hpo_results_path = PROJECT_ROOT / "outputs/hpo/phase6a_2M_threshold_1pct/best_params.json"
with open(hpo_results_path) as f:
    hyperparameters = json.load(f)["params"]

# Run training experiment
result = run_training_experiment(
    experiment="phase6a_2M_threshold_1pct",
    budget="2M",
    task="threshold_1pct",
    data_path=PROJECT_ROOT / "data/processed/v1/SPY_dataset_c.parquet",
    hyperparameters=hyperparameters,
    output_dir=PROJECT_ROOT / "outputs/training/phase6a_2M_threshold_1pct",
)

# Add metadata for logging
result.update({
    "experiment": "phase6a_2M_threshold_1pct",
    "phase": "phase6a",
    "budget": "2M",
    "task": "threshold_1pct",
    "horizon": 1,
    "timescale": "daily",
    "script_path": "experiments/phase6a/train_2M_threshold_1pct.py",
    "run_type": "training",
})

# Log result
update_experiment_log(result, LOG_PATH)

# Regenerate report only on success
if result["status"] == "success":
    regenerate_results_report(LOG_PATH, REPORT_PATH)
```

## During Execution

### Thermal Monitoring

The `run_hpo_experiment()` and `run_training_experiment()` functions include built-in thermal checks:

- **Pre-flight check**: Aborts immediately if temperature is CRITICAL (>95°C)
- **Records max temperature**: Logged in results for analysis

For long runs, periodically check thermal status:

```bash
# Manual check during execution
sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i temp
```

### If Temperature Rises

| Condition | Action |
|-----------|--------|
| Reaches WARNING (85-95°C) | Consider pausing, reduce batch size |
| Reaches CRITICAL (>95°C) | Stop immediately, wait for cooldown |

See `thermal_management` skill for detailed protocols.

## Result Handling

### On Success

1. Result logged to CSV with `status: success`
2. Markdown report regenerated
3. Outputs saved to `outputs/{hpo|training}/{experiment}/`

```
outputs/
├── hpo/
│   └── phase6a_2M_threshold_1pct/
│       ├── study.db          # Optuna study
│       └── best_params.json  # Best hyperparameters
├── training/
│   └── phase6a_2M_threshold_1pct/
│       ├── checkpoints/      # Model checkpoints
│       └── metrics.json      # Training metrics
└── results/
    └── experiment_log.csv    # All experiment results
```

### On Failure

1. Result logged to CSV with `status: failed` and `error_message`
2. Markdown report NOT regenerated (preserves last good state)
3. Investigate error before retrying

### On Thermal Abort

1. Result logged to CSV with `status: thermal_abort`
2. Markdown report NOT regenerated
3. Wait for cooldown before retrying

## Output Paths

| Output | Path |
|--------|------|
| Experiment Log (CSV) | `outputs/results/experiment_log.csv` |
| Results Report (MD) | `docs/experiment_results.md` |
| HPO Outputs | `outputs/hpo/{experiment}/` |
| Training Outputs | `outputs/training/{experiment}/` |

## CSV Log Schema

The experiment log contains these columns:

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
| test_accuracy | float | Test accuracy (training only) |
| hyperparameters | json | JSON string of params |
| error_message | string | Error details if failed |
| thermal_max_temp | float | Peak temperature |
| data_md5 | string | MD5 of data file |

## Complete Example

Execute HPO for Phase 6A, 2M budget, threshold_1pct task:

```python
from pathlib import Path
from src.experiments.runner import (
    run_hpo_experiment,
    update_experiment_log,
    regenerate_results_report,
)
from src.training.thermal import ThermalCallback

PROJECT_ROOT = Path("/Users/alexanderthomson/Documents/financial_ts_scaling")

# Pre-flight thermal check
thermal = ThermalCallback()
status = thermal.check()
print(f"Thermal status: {status.status} ({status.temperature}°C)")

if status.status == "critical":
    print("ABORT: Temperature too high")
else:
    # Run experiment
    result = run_hpo_experiment(
        experiment="phase6a_2M_threshold_1pct",
        budget="2M",
        task="threshold_1pct",
        data_path=PROJECT_ROOT / "data/processed/v1/SPY_dataset_c.parquet",
        output_dir=PROJECT_ROOT / "outputs/hpo/phase6a_2M_threshold_1pct",
        n_trials=50,
        timeout_hours=4.0,
    )

    # Add metadata
    result.update({
        "experiment": "phase6a_2M_threshold_1pct",
        "phase": "phase6a",
        "budget": "2M",
        "task": "threshold_1pct",
        "horizon": 1,
        "timescale": "daily",
        "script_path": "experiments/phase6a/hpo_2M_threshold_1pct.py",
        "run_type": "hpo",
    })

    # Log and report
    log_path = PROJECT_ROOT / "outputs/results/experiment_log.csv"
    update_experiment_log(result, log_path)

    if result["status"] == "success":
        regenerate_results_report(log_path, PROJECT_ROOT / "docs/experiment_results.md")
        print(f"SUCCESS: val_loss={result['val_loss']:.4f}")
    else:
        print(f"FAILED: {result['error_message']}")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Script not found | Run experiment_generation skill first |
| Thermal abort | Wait 15-20 min, check cooling setup |
| Data not found | Verify manifest with `make verify` |
| Import errors | Activate venv: `source venv/bin/activate` |
| Disk full | Clear old outputs, check `df -h` |
