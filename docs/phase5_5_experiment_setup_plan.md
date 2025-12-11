# Phase 5.5: Experiment Setup Plan

**Status:** Approved, Ready for Implementation
**Date:** 2025-12-10
**Version:** 1.0
**Prerequisite:** Phase 5 Complete (SPY_dataset_c.parquet ready)

---

## Executive Summary

Phase 5.5 builds the infrastructure required to run systematic scaling law experiments. Upon completion, we can execute Phase 6A (Parameter Scaling Baseline) with proper HPO, tracking, and analysis tools.

**Total Effort:** 10-14 hours across 6 tasks
**Execution Strategy:** One task per session, sequential with approval gates

---

## Objective

Build experiment infrastructure for:
- **4 tasks:** threshold_1pct, threshold_2pct, threshold_3pct, threshold_5pct
- **4 parameter budgets:** 2M, 20M, 200M, 2B
- **5 horizons:** 1-day, 2-day, 3-day, 5-day, weekly

**Outcome:** Can run `scripts/run_hpo.py` and `scripts/train.py` systematically, document data comprehensively, then analyze results with scaling curve tools.

---

## Scope

### In Scope

1. Config templates for 4 threshold tasks
2. Timescale resampling utilities (daily → 2d/3d/5d/weekly)
3. Comprehensive data dictionary documentation
4. Optuna HPO integration with W&B/MLflow tracking
5. Scaling curve analysis (plot error vs parameters, fit power law)
6. Result aggregation utilities

### Out of Scope

- Direction prediction task (only threshold tasks)
- Regression task (only classification)
- Feature tier expansion (a100, a250, a500) - Phase 6C
- Multi-asset training - Phase 6D
- New model architectures - use existing PatchTST

---

## Task Summary

| Task | Name | Est. Hours | Dependencies | Deliverables |
|------|------|------------|--------------|--------------|
| 5.5.1 | Config Templates | 0.5 | None | 3 YAML configs |
| 5.5.2 | Timescale Resampling | 2-3 | 5.5.1 | resample.py, CLI |
| 5.5.3 | Data Dictionary | 1-2 | 5.5.2 | docs/data_dictionary.md |
| 5.5.4 | Optuna HPO Integration | 3-4 | 5.5.3 | hpo.py, run_hpo.py |
| 5.5.5 | Scaling Curve Analysis | 2 | 5.5.4 | scaling_curves.py |
| 5.5.6 | Result Aggregation | 1-2 | 5.5.5 | aggregate_results.py |

---

## Detailed Task Specifications

### Task 5.5.1: Config Templates for 4 Threshold Tasks

**Purpose:** Create experiment configs for threshold_2pct, threshold_3pct, threshold_5pct. (threshold_1pct already exists)

**Estimated Time:** 30 minutes

**Prerequisites:**
- Phase 5 complete
- `configs/daily/threshold_1pct.yaml` exists as template
- `data/processed/v1/SPY_dataset_c.parquet` exists

**Files to Create:**

| File Path | Purpose |
|-----------|---------|
| `configs/experiments/spy_daily_threshold_2pct.yaml` | 2% threshold config |
| `configs/experiments/spy_daily_threshold_3pct.yaml` | 3% threshold config |
| `configs/experiments/spy_daily_threshold_5pct.yaml` | 5% threshold config |

**Config Template:**
```yaml
# configs/experiments/spy_daily_threshold_Xpct.yaml
seed: 42
data_path: data/processed/v1/SPY_dataset_c.parquet
task: threshold_Xpct  # threshold_2pct | threshold_3pct | threshold_5pct
timescale: daily
context_length: 60
horizon: 5
wandb_project: financial-ts-scaling
mlflow_experiment: phase6a-parameter-scaling
```

**Tests to Add:**
- `tests/test_config.py::test_load_threshold_2pct_config`
- `tests/test_config.py::test_load_threshold_3pct_config`
- `tests/test_config.py::test_load_threshold_5pct_config`

**Success Criteria:**
- [ ] All 3 new config files exist
- [ ] `load_experiment_config()` loads each without error
- [ ] `make test` passes

**Verification Command:**
```bash
python -c "from src.config.experiment import load_experiment_config; load_experiment_config('configs/experiments/spy_daily_threshold_2pct.yaml')"
```

---

### Task 5.5.2: Timescale Resampling

**Purpose:** Build OHLCV resampling to 2d/3d/5d/weekly frequencies, then regenerate features for multi-timescale experiments.

**Estimated Time:** 2-3 hours

**Prerequisites:**
- Task 5.5.1 complete
- Raw OHLCV data in `data/raw/`
- Feature engineering pipeline functional

**Files to Create:**

| File Path | Purpose | ~Lines |
|-----------|---------|--------|
| `src/features/resample.py` | Core resampling functions | 80 |
| `scripts/resample_timescales.py` | CLI entry point | 60 |
| `tests/features/test_resample.py` | Unit tests | 60 |

**Core Functions in `src/features/resample.py`:**

```python
def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV data to lower frequency.

    Args:
        df: DataFrame with Date index and OHLCV columns
        freq: Target frequency ('2D', '3D', '5D', 'W-FRI')

    Returns:
        Resampled DataFrame with aggregated OHLCV:
        - Open: first
        - High: max
        - Low: min
        - Close: last
        - Volume: sum

    Note: Uses end-of-period values to avoid look-ahead bias.
    """

def get_freq_string(timescale: str) -> str:
    """Convert timescale name to pandas frequency string.

    Args:
        timescale: '2d', '3d', '5d', 'weekly'

    Returns:
        Pandas frequency: '2D', '3D', '5D', 'W-FRI'
    """
```

**CLI Usage:**
```bash
# Resample SPY to weekly
python scripts/resample_timescales.py --ticker SPY --timescale weekly

# Output: data/processed/v1/SPY_OHLCV_weekly.parquet
# Then rebuild features on resampled data
```

**Tests:**
- `test_resample_ohlcv_to_2d`: 2-day aggregation correct (Open=first, High=max, etc.)
- `test_resample_ohlcv_to_weekly`: Weekly aggregation uses Friday close
- `test_resample_preserves_date_index`: Date column remains datetime
- `test_resample_no_lookahead`: Resampled Close matches actual period-end price

**Success Criteria:**
- [ ] `resample_ohlcv()` produces correct aggregations
- [ ] CLI generates resampled parquet files
- [ ] `make test` passes
- [ ] Manual verification: weekly row count ≈ daily / 5

**Technical Notes:**
- Weekly resampling already partially exists in `tier_a20.py` for weekly RSI
- Use `W-FRI` for weekly to align with trading week end
- Forward-fill any gaps from market holidays

---

### Task 5.5.3: Data Dictionary

**Purpose:** Create comprehensive documentation of all raw and processed data files with schema and statistics.

**Estimated Time:** 1-2 hours

**Prerequisites:**
- Task 5.5.2 complete (so resampled data is included)
- All data files from Phase 5 exist

**Files to Create:**

| File Path | Purpose | ~Lines |
|-----------|---------|--------|
| `docs/data_dictionary.md` | Human-readable data documentation | 500+ |
| `scripts/generate_data_dictionary.py` | Auto-generation script | 150 |
| `tests/test_data_dictionary.py` | Verification tests | 50 |

**Script Functionality (`scripts/generate_data_dictionary.py`):**

```python
def generate_data_dictionary(output_path: str = "docs/data_dictionary.md"):
    """Generate comprehensive data dictionary for all parquet files.

    Scans:
    - data/raw/*.parquet
    - data/processed/v1/*.parquet

    For each file, documents:
    - File metadata: path, shape, date range, MD5, source
    - Column schema: name, dtype, category, description
    - Statistics: count, mean, std, min, 25%, 50%, 75%, max

    Outputs markdown to docs/data_dictionary.md
    """

def get_file_stats(filepath: Path) -> dict:
    """Extract statistics from a parquet file."""

def format_statistics_table(df: pd.DataFrame) -> str:
    """Format df.describe() as markdown table."""

def get_column_descriptions(columns: list, file_type: str) -> dict:
    """Return known descriptions for standard columns."""
```

**Output Document Structure:**

```markdown
# Data Dictionary

Generated: YYYY-MM-DD HH:MM:SS
Generator: scripts/generate_data_dictionary.py

## Summary

| Location | Files | Total Rows | Total Columns |
|----------|-------|------------|---------------|
| data/raw/ | N | X | Y |
| data/processed/v1/ | N | X | Y |

## Data Lineage

```
Raw OHLCV (yfinance)
    │
    ├── SPY.parquet ──► SPY_features_a20.parquet ──► SPY_dataset_c.parquet
    ├── DIA.parquet ──► DIA_features_a20.parquet
    ├── QQQ.parquet ──► QQQ_features_a20.parquet
    └── VIX.parquet ──► VIX_features_c.parquet ──┘
```

---

## Raw Data Files

### SPY.OHLCV.daily

| Property | Value |
|----------|-------|
| Path | data/raw/SPY.parquet |
| Shape | 8,272 rows × 6 columns |
| Date Range | 1993-01-29 to 2025-12-09 |
| Source | yfinance (Yahoo Finance) |
| Manifest ID | SPY.OHLCV.daily |
| MD5 | 805e73ad... |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high (USD) |
| Low | float64 | Intraday low (USD) |
| Close | float64 | Closing price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|-------|------|-----|-----|-----|-----|-----|-----|
| Open | 8272 | 187.34 | 127.45 | 25.12 | 89.23 | 134.56 | 287.12 | 604.21 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

[Repeat for each file...]

## Processed Data Files

### SPY.features.a20

[Same structure with additional indicator descriptions...]

### SPY.dataset.c

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_dataset_c.parquet |
| Shape | 8,073 rows × 34 columns |
| Composition | 1 Date + 5 OHLCV + 20 indicators + 8 VIX features |
| Source Files | SPY_features_a20.parquet + VIX_features_c.parquet |

#### Feature Categories

| Category | Count | Features |
|----------|-------|----------|
| OHLCV | 5 | Open, High, Low, Close, Volume |
| Moving Averages | 7 | DEMA_9, DEMA_10, SMA_12, DEMA_20, DEMA_25, SMA_50, ... |
| Momentum | 4 | RSI_14, RSI_5d, StochRSI, StochRSI_5d |
| Volume | 2 | OBV, ADOSC |
| Volatility | 3 | ATR_14, ADX_14, BB_pctB |
| VIX | 8 | vix_close, vix_sma_10, vix_sma_20, vix_percentile_60d, ... |

[Full column schema and statistics...]

---

## Appendix: Column Descriptions

### OHLCV Columns
- **Date**: Trading date, datetime64, used as index
- **Open**: First trade price of the day
- **High**: Highest trade price of the day
- **Low**: Lowest trade price of the day
- **Close**: Last trade price of the day (adjusted)
- **Volume**: Total shares traded

### Indicator Columns (Tier a20)
- **DEMA_N**: Double Exponential Moving Average, N-period
- **SMA_N**: Simple Moving Average, N-period
- **RSI_14**: Relative Strength Index, 14-period (0-100)
- **RSI_5d**: RSI computed on 5-day resampled data
- **StochRSI**: Stochastic RSI (0-1)
- **MACD**: Moving Average Convergence Divergence line
- **OBV**: On-Balance Volume (cumulative)
- **ADOSC**: Accumulation/Distribution Oscillator
- **ATR_14**: Average True Range, 14-period (volatility)
- **ADX_14**: Average Directional Index, 14-period (trend strength)
- **BB_pctB**: Bollinger Band %B (position within bands, 0-1)
- **VWAP_20**: Volume-Weighted Average Price, 20-day rolling

### VIX Columns (Tier c)
- **vix_close**: Raw VIX closing value
- **vix_sma_10**: 10-day SMA of VIX
- **vix_sma_20**: 20-day SMA of VIX
- **vix_percentile_60d**: VIX percentile rank over 60 days (0-100)
- **vix_zscore_20d**: VIX z-score over 20 days
- **vix_regime**: Categorical: 'low' (<15), 'normal' (15-25), 'high' (>25)
- **vix_change_1d**: 1-day VIX percentage change
- **vix_change_5d**: 5-day VIX percentage change
```

**CLI Usage:**
```bash
# Generate the data dictionary
python scripts/generate_data_dictionary.py

# Regenerate after data changes
python scripts/generate_data_dictionary.py --force
```

**Tests:**
- `test_generate_data_dictionary_creates_file`: Output file created
- `test_data_dictionary_covers_all_parquets`: All files documented
- `test_data_dictionary_has_statistics`: Numeric stats present

**Success Criteria:**
- [ ] `docs/data_dictionary.md` exists with complete documentation
- [ ] All raw and processed files documented
- [ ] Statistics accurate (spot-check against `df.describe()`)
- [ ] `make test` passes

---

### Task 5.5.4: Optuna HPO Integration

**Purpose:** Build hyperparameter optimization infrastructure with Optuna, integrated with W&B and MLflow for tracking.

**Estimated Time:** 3-4 hours

**Prerequisites:**
- Task 5.5.3 complete
- Training infrastructure functional (`scripts/train.py`, `src/training/trainer.py`)
- W&B and MLflow configured

**Files to Create:**

| File Path | Purpose | ~Lines |
|-----------|---------|--------|
| `src/training/hpo.py` | Core HPO functions | 200 |
| `scripts/run_hpo.py` | CLI entry point | 80 |
| `configs/hpo/default_search.yaml` | Default search space | 30 |
| `tests/test_hpo.py` | Unit tests (mocked) | 100 |

**Core Functions in `src/training/hpo.py`:**

```python
import optuna
from optuna.integration import WeightsAndBiasesCallback
from pathlib import Path

def create_study(
    experiment_name: str,
    budget: str,
    storage: str | None = None,
    direction: str = "minimize"
) -> optuna.Study:
    """Create Optuna study with optional persistent storage.

    Args:
        experiment_name: Name for the study (e.g., 'spy_daily_threshold_1pct')
        budget: Parameter budget ('2M', '20M', '200M', '2B')
        storage: SQLite path for persistence, or None for in-memory
        direction: 'minimize' for loss, 'maximize' for accuracy

    Returns:
        Optuna Study object
    """

def create_objective(
    config_path: str,
    budget: str,
    search_space: dict
) -> callable:
    """Create Optuna objective function for HPO.

    The objective function:
    1. Samples hyperparameters from search space
    2. Trains model with sampled params
    3. Returns validation loss

    Args:
        config_path: Path to experiment config YAML
        budget: Parameter budget
        search_space: Dict defining parameter ranges

    Returns:
        Objective function for study.optimize()
    """

def run_hpo(
    config_path: str,
    budget: str,
    n_trials: int = 50,
    timeout_hours: float = 4.0,
    search_space_path: str = "configs/hpo/default_search.yaml"
) -> dict:
    """Run complete HPO workflow.

    1. Creates study
    2. Sets up W&B and MLflow callbacks
    3. Runs optimization
    4. Saves best params to JSON
    5. Returns best params dict

    Args:
        config_path: Experiment config path
        budget: Parameter budget
        n_trials: Max trials to run
        timeout_hours: Max time in hours
        search_space_path: Path to search space YAML

    Returns:
        Dict with best hyperparameters and metrics

    Output:
        Saves to: outputs/hpo/{experiment_name}/{budget}_best.json
    """

def load_search_space(path: str) -> dict:
    """Load search space definition from YAML."""

def save_best_params(
    study: optuna.Study,
    experiment_name: str,
    budget: str
) -> Path:
    """Save best params to JSON file."""
```

**Search Space Config (`configs/hpo/default_search.yaml`):**

```yaml
# Default HPO search space for scaling experiments
n_trials: 50
timeout_hours: 4.0
direction: minimize  # minimize validation loss

search_space:
  learning_rate:
    type: log_uniform
    low: 1.0e-5
    high: 1.0e-2

  epochs:
    type: int
    low: 20
    high: 200
    step: 10

  weight_decay:
    type: log_uniform
    low: 1.0e-6
    high: 1.0e-2

  warmup_steps:
    type: int
    low: 0
    high: 1000
    step: 100

  dropout:
    type: uniform
    low: 0.0
    high: 0.3

# Pruning configuration
pruning:
  enabled: true
  patience: 5  # epochs without improvement before pruning
  min_trials: 10  # minimum trials before pruning kicks in
```

**HPO Result Format (`outputs/hpo/{experiment}/{budget}_best.json`):**

```json
{
  "experiment": "spy_daily_threshold_1pct",
  "budget": "20M",
  "best_params": {
    "learning_rate": 0.00032,
    "epochs": 87,
    "weight_decay": 0.00001,
    "warmup_steps": 500,
    "dropout": 0.1
  },
  "best_value": 0.4523,
  "n_trials_completed": 50,
  "n_trials_pruned": 12,
  "total_time_seconds": 14400,
  "timestamp": "2025-12-15T10:30:00Z",
  "study_name": "spy_daily_threshold_1pct_20M",
  "optuna_version": "4.6.0"
}
```

**CLI Usage:**
```bash
# Run HPO for 2M budget on threshold_1pct
python scripts/run_hpo.py \
  --config configs/experiments/spy_daily_threshold_1pct.yaml \
  --budget 2M \
  --n-trials 50 \
  --timeout 4

# Use custom search space
python scripts/run_hpo.py \
  --config configs/experiments/spy_daily_threshold_1pct.yaml \
  --budget 20M \
  --search-space configs/hpo/aggressive_search.yaml
```

**Tests (all mocked, no actual training):**
- `test_create_optuna_study`: Study creates with correct name
- `test_load_search_space`: YAML loads correctly
- `test_hpo_objective_samples_params`: Objective samples from search space
- `test_hpo_saves_best_json`: Best params saved to correct path
- `test_hpo_integrates_wandb`: W&B callback triggered (mocked)
- `test_hpo_respects_timeout`: Stops after timeout

**Success Criteria:**
- [ ] `run_hpo()` executes without error
- [ ] Best params saved to `outputs/hpo/` as JSON
- [ ] W&B logs HPO trials
- [ ] MLflow logs HPO results
- [ ] `make test` passes

**Thermal Considerations:**
- Add thermal check between trials
- If temp > 85°C, pause for 60 seconds
- If temp > 95°C, save study state and abort

---

### Task 5.5.5: Scaling Curve Analysis

**Purpose:** Build analysis tools to visualize scaling laws and fit power law relationships.

**Estimated Time:** 2 hours

**Prerequisites:**
- Task 5.5.4 complete
- HPO results available (or mock data for testing)

**Files to Create:**

| File Path | Purpose | ~Lines |
|-----------|---------|--------|
| `src/analysis/scaling_curves.py` | Analysis functions | 120 |
| `tests/analysis/test_scaling_curves.py` | Unit tests | 60 |

**Core Functions in `src/analysis/scaling_curves.py`:**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def load_experiment_results(
    experiment_name: str,
    budgets: list[str] = ["2M", "20M", "200M", "2B"]
) -> pd.DataFrame:
    """Load training results for an experiment across budgets.

    Args:
        experiment_name: e.g., 'spy_daily_threshold_1pct'
        budgets: List of parameter budgets

    Returns:
        DataFrame with columns: budget, params, val_loss, test_loss, accuracy
    """

def plot_scaling_curve(
    results: pd.DataFrame,
    metric: str = "val_loss",
    output_path: str | None = None,
    title: str | None = None
) -> plt.Figure:
    """Plot log-log scaling curve.

    Args:
        results: DataFrame with 'params' and metric columns
        metric: Column name for y-axis
        output_path: If provided, save figure here
        title: Plot title

    Returns:
        Matplotlib figure with:
        - Log-log scatter of params vs metric
        - Power law fit line
        - Annotation with alpha and R^2
    """

def fit_power_law(
    params: np.ndarray,
    errors: np.ndarray
) -> tuple[float, float, float]:
    """Fit power law: error = a * params^(-alpha)

    Args:
        params: Array of parameter counts
        errors: Array of error values (loss or 1-accuracy)

    Returns:
        Tuple of (alpha, a, r_squared):
        - alpha: Scaling exponent (higher = better scaling)
        - a: Scaling coefficient
        - r_squared: Goodness of fit

    Notes:
        - Fit performed in log-log space for numerical stability
        - error ∝ N^(-α) implies log(error) = log(a) - α*log(N)
    """

def generate_scaling_report(
    experiment_name: str,
    output_dir: str = "outputs/figures"
) -> dict:
    """Generate complete scaling analysis report.

    Creates:
    - Scaling curve plot (PNG)
    - Summary statistics (JSON)

    Returns:
        Dict with alpha, r_squared, and interpretation
    """

def compare_scaling_across_tasks(
    tasks: list[str],
    output_path: str
) -> plt.Figure:
    """Compare scaling curves across different tasks.

    Creates multi-panel figure showing scaling for each task.
    """
```

**Output Example:**

```
outputs/figures/
├── spy_daily_threshold_1pct_scaling.png
├── spy_daily_threshold_1pct_scaling.json
└── scaling_comparison_all_tasks.png
```

**Scaling Report JSON:**
```json
{
  "experiment": "spy_daily_threshold_1pct",
  "metric": "val_loss",
  "fit_results": {
    "alpha": 0.123,
    "coefficient": 1.45,
    "r_squared": 0.94
  },
  "data_points": [
    {"budget": "2M", "params": 1820000, "val_loss": 0.52},
    {"budget": "20M", "params": 19000000, "val_loss": 0.48},
    {"budget": "200M", "params": 202000000, "val_loss": 0.44},
    {"budget": "2B", "params": 2040000000, "val_loss": 0.41}
  ],
  "interpretation": "Moderate scaling: 10x params yields ~8% error reduction"
}
```

**Tests:**
- `test_fit_power_law_perfect_data`: Known power law recovers alpha
- `test_fit_power_law_noisy_data`: Returns reasonable R^2
- `test_plot_scaling_curve_creates_figure`: Figure object returned
- `test_plot_scaling_curve_saves_file`: PNG saved when path provided
- `test_load_experiment_results`: Correctly aggregates results

**Success Criteria:**
- [ ] `fit_power_law()` returns alpha and R^2
- [ ] `plot_scaling_curve()` generates valid figure
- [ ] Works with synthetic data in tests
- [ ] `make test` passes

---

### Task 5.5.6: Result Aggregation

**Purpose:** Build utilities to aggregate and summarize experiment results for analysis and reporting.

**Estimated Time:** 1-2 hours

**Prerequisites:**
- Task 5.5.5 complete

**Files to Create:**

| File Path | Purpose | ~Lines |
|-----------|---------|--------|
| `src/analysis/aggregate_results.py` | Aggregation functions | 80 |
| `tests/analysis/test_aggregate_results.py` | Unit tests | 40 |

**Core Functions in `src/analysis/aggregate_results.py`:**

```python
from pathlib import Path
import json
import pandas as pd

def aggregate_hpo_results(
    experiment_name: str | None = None,
    hpo_dir: str = "outputs/hpo"
) -> pd.DataFrame:
    """Aggregate all HPO best.json files into DataFrame.

    Args:
        experiment_name: Filter to specific experiment, or None for all
        hpo_dir: Directory containing HPO results

    Returns:
        DataFrame with columns:
        - experiment, budget, learning_rate, epochs, weight_decay,
        - warmup_steps, dropout, best_value, n_trials, timestamp
    """

def aggregate_training_results(
    results_dir: str = "outputs/results"
) -> pd.DataFrame:
    """Aggregate all training result JSONs into DataFrame.

    Returns:
        DataFrame with columns:
        - experiment, budget, train_loss, val_loss, test_loss,
        - accuracy, precision, recall, f1, training_time, timestamp
    """

def summarize_experiment(
    experiment_name: str
) -> dict:
    """Generate summary statistics for an experiment.

    Returns dict with:
    - best_budget: Budget with lowest val_loss
    - scaling_factor: Improvement from 2M to 200M
    - hpo_summary: Best params per budget
    - training_summary: Final metrics per budget
    """

def export_results_csv(
    output_path: str = "outputs/results/all_results.csv"
) -> Path:
    """Export all results to CSV for external analysis."""

def generate_experiment_summary_report(
    output_path: str = "outputs/results/summary_report.md"
) -> Path:
    """Generate markdown summary report of all experiments."""
```

**Output Format (`outputs/results/all_results.csv`):**

```csv
experiment,budget,params,val_loss,test_loss,accuracy,alpha_fit,r_squared,timestamp
spy_daily_threshold_1pct,2M,1820000,0.52,0.53,0.68,,,2025-12-15
spy_daily_threshold_1pct,20M,19000000,0.48,0.49,0.71,,,2025-12-16
spy_daily_threshold_1pct,200M,202000000,0.44,0.45,0.74,0.123,0.94,2025-12-17
...
```

**Tests:**
- `test_aggregate_hpo_results_empty_dir`: Returns empty DataFrame
- `test_aggregate_hpo_results_multiple_files`: Correctly combines
- `test_summarize_experiment`: Returns expected structure
- `test_export_results_csv`: CSV written correctly

**Success Criteria:**
- [ ] `aggregate_hpo_results()` collects all HPO results
- [ ] `aggregate_training_results()` collects all training results
- [ ] CSV export works
- [ ] `make test` passes

---

## Execution Order

```
Task 5.5.1: Config templates
    │ Create threshold_2pct, threshold_3pct, threshold_5pct configs
    │
    └──► Task 5.5.2: Timescale resampling
            │ Build OHLCV resampling for 2d/3d/5d/weekly
            │
            └──► Task 5.5.3: Data dictionary
                    │ Document all data files comprehensively
                    │
                    └──► Task 5.5.4: Optuna HPO integration
                            │ Build HPO with W&B/MLflow tracking
                            │
                            └──► Task 5.5.5: Scaling curve analysis
                                    │ Build visualization and power law fitting
                                    │
                                    └──► Task 5.5.6: Result aggregation
                                            │ Build result collection utilities
                                            │
                                            └──► PHASE 6A READY
```

---

## Assumptions

1. **Existing infrastructure works:** train.py, trainer.py, thermal callbacks functional
2. **Data ready:** SPY_dataset_c.parquet (8,073 rows, 34 cols) available
3. **Optuna compatible:** Optuna 4.6+ integrates with W&B/MLflow
4. **Config schema sufficient:** ExperimentConfig works without changes
5. **MPS memory:** Batch sizes from find_batch_size.py accurate
6. **Horizon via dataset:** Horizon variation via config `horizon` param, not resampling

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Optuna + W&B conflict | Medium | Test 2-trial pilot early |
| HPO thermal throttle | High | Thermal check between trials |
| Config proliferation | Medium | Template-based generation |
| Resampling look-ahead bias | Low | Use end-of-period values |
| Power law doesn't fit | Medium | Document as negative result |

---

## Success Criteria Summary

Phase 5.5 is complete when ALL of the following are true:

- [ ] All 4 threshold task configs exist and load
- [ ] Timescale resampling produces valid datasets
- [ ] Data dictionary documents all files with stats
- [ ] HPO runs with W&B/MLflow tracking
- [ ] Scaling curve analysis generates plots and fits
- [ ] Result aggregation collects all outputs
- [ ] `make test` passes (all new tests)
- [ ] Can execute: `python scripts/run_hpo.py --config configs/experiments/spy_daily_threshold_1pct.yaml --budget 2M --n-trials 5`

---

## Branching Strategy

**Branch name:** `feature/phase-5.5-experiment-setup`
**Base:** `main`
**Merge target:** `main`

Each task committed separately:
- `feat: add threshold_2pct/3pct/5pct config templates (5.5.1)`
- `feat: add timescale resampling utilities (5.5.2)`
- `docs: add comprehensive data dictionary (5.5.3)`
- `feat: add Optuna HPO integration (5.5.4)`
- `feat: add scaling curve analysis tools (5.5.5)`
- `feat: add result aggregation utilities (5.5.6)`

---

## Agent Handoff Instructions

**For any coding agent picking up a task:**

1. **Session Start:**
   - Run `session restore` or read `.claude/context/session_context.md`
   - Run `make test` to verify environment
   - Read this plan document thoroughly

2. **Before Starting a Task:**
   - Verify prerequisite tasks are complete
   - Read the detailed task specification above
   - Follow TDD: write tests first, then implementation

3. **During Task:**
   - Use `TodoWrite` to track progress
   - Follow approval gates for any changes
   - Run `make test` frequently

4. **After Task:**
   - All tests must pass: `make test`
   - Update `phase_tracker.md` with task status
   - Commit with descriptive message
   - Run `session handoff` skill

5. **Key Files to Read:**
   - `CLAUDE.md` - Project rules
   - `docs/config_architecture.md` - Config system design
   - `src/config/experiment.py` - Existing config loader
   - `src/training/trainer.py` - Training infrastructure

---

*Document Version: 1.0*
*Author: Claude (Planning Session)*
*Date: 2025-12-10*
*Status: Approved*
