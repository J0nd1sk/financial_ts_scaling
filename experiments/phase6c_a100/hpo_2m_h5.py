#!/usr/bin/env python3
"""
Phase 6C A100 HPO: 2M parameters, horizon=5

Hyperparameter optimization for architecture within 2M budget.
Search space focuses on architecture (training params already validated).

Trials: 50
Metric: val_auc (maximize)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "hpo_2m_h5"
BUDGET = "2M"
HORIZON = 5
N_TRIALS = 50

# Architecture search space (within 2M budget)
SEARCH_SPACE = {
    "d_model": [32, 48, 64, 80, 96],
    "n_layers": [2, 3, 4, 5, 6],
    "n_heads": [2, 4, 8],
    "d_ff_ratio": [2, 4],
}

# Fixed hyperparameters (ablation-validated)
LEARNING_RATE = 1e-4
DROPOUT = 0.5
CONTEXT_LENGTH = 80
EPOCHS = 50

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet"
NUM_FEATURES = 100

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a100" / EXPERIMENT_NAME


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def objective(trial):
    """Optuna objective function."""
    # Sample architecture
    d_model = trial.suggest_categorical("d_model", SEARCH_SPACE["d_model"])
    n_layers = trial.suggest_categorical("n_layers", SEARCH_SPACE["n_layers"])
    n_heads = trial.suggest_categorical("n_heads", SEARCH_SPACE["n_heads"])
    d_ff_ratio = trial.suggest_categorical("d_ff_ratio", SEARCH_SPACE["d_ff_ratio"])
    d_ff = d_model * d_ff_ratio

    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        return float("-inf")

    # Estimate parameter count (rough)
    param_estimate = d_model * n_layers * (4 * d_model + d_ff)
    budget_target = {"2M": 2e6, "20M": 20e6, "200M": 200e6}[BUDGET]

    # Skip if too far from target
    if param_estimate > budget_target * 1.5 or param_estimate < budget_target * 0.3:
        return float("-inf")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load data
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values

    # Batch size based on model size
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    trial_dir = OUTPUT_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=trial_dir,
        split_indices=split_indices,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric="val_auc",
        use_revin=True,
        high_prices=high_prices,
    )

    try:
        result = trainer.train(verbose=False)
        val_auc = result.get("val_auc", 0.5)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        val_auc = 0.5

    return val_auc if val_auc else 0.5


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 6C A100 HPO: " + BUDGET + " / horizon=" + str(HORIZON))
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print("Starting HPO with", N_TRIALS, "trials...")
    start_time = time.time()

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("HPO RESULTS")
    print("=" * 70)
    print("Best trial:", study.best_trial.number)
    print("Best AUC:", round(study.best_value, 4))
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("Total time:", round(elapsed/60, 1), "min")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "budget": BUDGET,
        "horizon": HORIZON,
        "n_trials": N_TRIALS,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "search_space": SEARCH_SPACE,
        "total_time_min": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "best_params.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to", results_path)

    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        })

    trials_path = OUTPUT_DIR / "all_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)

    return study.best_value


if __name__ == "__main__":
    main()
