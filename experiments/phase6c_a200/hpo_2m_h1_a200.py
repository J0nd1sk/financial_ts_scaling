#!/usr/bin/env python3
"""
Phase 6C A200 HPO: 2M parameters, horizon=1

Hyperparameter optimization for architecture and training within 2M budget.
Search space includes architecture and training hyperparameters.

This is for the a200 feature tier (206 features) - separate from a100's HPO.

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

EXPERIMENT_NAME = "hpo_2m_h1_a200"
BUDGET = "2M"
HORIZON = 1
N_TRIALS = 50

# Search space (within 2M budget)
SEARCH_SPACE = {
    # Architecture
    "d_model": [32, 48, 64, 80, 96],
    "n_layers": [2, 3, 4, 5, 6],
    "n_heads": [2, 4, 8],
    "d_ff_ratio": [2, 4],
    # Training hyperparameters
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
    "dropout": [0.1, 0.3, 0.5, 0.7],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
}

# Fixed hyperparameters (ablation-validated)
CONTEXT_LENGTH = 80
EPOCHS = 50

# Data - A200 tier (206 features)
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet"
NUM_FEATURES = 211  # 206 indicators + 5 OHLCV (auto-adjusted by Trainer)

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a200" / EXPERIMENT_NAME


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

    # Sample training hyperparameters
    learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"])
    dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
    weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"])

    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        raise optuna.TrialPruned("d_model not divisible by n_heads")

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
        dropout=dropout,
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
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # OOM or MPS memory errors - prune trial, don't poison study with 0.5
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
            trial.set_user_attr("error", f"OOM: {error_msg[:200]}")
            raise optuna.TrialPruned(f"OOM error: {error_msg[:100]}")
        # Other RuntimeErrors - log and return 0.5 (but log it)
        trial.set_user_attr("error", f"RuntimeError: {error_msg[:200]}")
        print(f"Trial {trial.number} failed (RuntimeError): {e}")
        val_auc = 0.5
    except ValueError as e:
        # NaN/Inf or data issues - prune trial
        error_msg = str(e)
        trial.set_user_attr("error", f"ValueError: {error_msg[:200]}")
        if "nan" in error_msg.lower() or "inf" in error_msg.lower():
            raise optuna.TrialPruned(f"NaN/Inf error: {error_msg[:100]}")
        print(f"Trial {trial.number} failed (ValueError): {e}")
        val_auc = 0.5
    except KeyboardInterrupt:
        # Don't catch keyboard interrupt - let user stop
        raise
    except Exception as e:
        # Catch-all for unexpected errors - log but don't silently ignore
        error_msg = str(e)
        trial.set_user_attr("error", f"Unexpected: {type(e).__name__}: {error_msg[:200]}")
        print(f"Trial {trial.number} failed ({type(e).__name__}): {e}")
        val_auc = 0.5

    return val_auc if val_auc else 0.5


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("PHASE 6C A200 HPO: " + BUDGET + " / horizon=" + str(HORIZON))
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
