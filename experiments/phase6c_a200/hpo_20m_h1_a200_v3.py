#!/usr/bin/env python3
"""
Phase 6C A200 HPO v3: 20M parameters, horizon=1

Precision-first optimization with:
1. Composite objective: precision*2 + recall*1 + auc*0.1
2. Loss type as hyperparameter: focal vs weighted_bce
3. Conditional loss parameters (alpha/gamma for focal, pos_weight for BCE)
4. Multi-threshold metrics logging (t30, t40, t50, t60, t70)
5. 80d context length (standard per CLAUDE.md)

Key insight: "Precision increases with more strict probability score thresholds.
We need to test ALL of it in HPO in a consistent way and not be so focused
exclusively on AUC."

Trials: 50
Metric: composite_score (maximize) = precision*2 + recall*1 + auc*0.1
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
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer
from src.training.losses import FocalLoss, WeightedBCELoss

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "hpo_20m_h1_a200_v3"
BUDGET = "20M"
HORIZON = 1
N_TRIALS = 50
DIRECTION = "maximize"

# v3 search space with loss parameters
SEARCH_SPACE_V3 = {
    # Architecture (unchanged from v2)
    "d_model": [64, 96, 128, 160, 192],
    "n_layers": [4, 5, 6, 7, 8],
    "n_heads": [4, 8],
    "d_ff_ratio": [2, 4],
    # Training (slightly refined from v2)
    "learning_rate": [5e-5, 7e-5, 1e-4, 1.5e-4],
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
    # Loss function (NEW in v3)
    "loss_type": ["focal", "weighted_bce"],
    "focal_alpha": [0.3, 0.5, 0.7, 0.9],
    "focal_gamma": [0.0, 0.5, 1.0, 2.0],
    "bce_pos_weight": [1.0, 2.0, 3.0, 5.0],
}

# Fixed hyperparameters (ablation-validated)
# 80d from CLAUDE.md standard
CONTEXT_LENGTH = 80
EPOCHS = 50

# Data - A200 tier (206 features)
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet"
NUM_FEATURES = 211  # 206 indicators + 5 OHLCV (auto-adjusted by Trainer)

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a200" / EXPERIMENT_NAME

# Thresholds for multi-threshold evaluation
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# ============================================================================
# COMPOSITE OBJECTIVE FUNCTION
# ============================================================================


def compute_composite_score(precision: float, recall: float, auc: float) -> float:
    """Compute precision-first composite score.

    Formula: precision * 2.0 + recall * 1.0 + auc * 0.1

    Weights:
    - Precision: 2.0 (primary - when we say buy, be right)
    - Recall: 1.0 (secondary - catch opportunities)
    - AUC: 0.1 (tertiary - tie-breaking only)

    Args:
        precision: Precision at threshold 0.5
        recall: Recall at threshold 0.5
        auc: AUC-ROC score

    Returns:
        Composite score (float)
    """
    return (precision * 2.0) + (recall * 1.0) + (auc * 0.1)


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================


def objective(trial):
    """Optuna objective function with precision-first composite scoring."""
    # Sample architecture params
    d_model = trial.suggest_categorical("d_model", SEARCH_SPACE_V3["d_model"])
    n_layers = trial.suggest_categorical("n_layers", SEARCH_SPACE_V3["n_layers"])
    n_heads = trial.suggest_categorical("n_heads", SEARCH_SPACE_V3["n_heads"])
    d_ff_ratio = trial.suggest_categorical("d_ff_ratio", SEARCH_SPACE_V3["d_ff_ratio"])

    # Sample training params
    learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE_V3["learning_rate"])
    dropout = trial.suggest_categorical("dropout", SEARCH_SPACE_V3["dropout"])
    weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE_V3["weight_decay"])

    # Sample loss type and conditional params
    loss_type = trial.suggest_categorical("loss_type", SEARCH_SPACE_V3["loss_type"])

    if loss_type == "focal":
        focal_alpha = trial.suggest_categorical("focal_alpha", SEARCH_SPACE_V3["focal_alpha"])
        focal_gamma = trial.suggest_categorical("focal_gamma", SEARCH_SPACE_V3["focal_gamma"])
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        trial.set_user_attr("loss_alpha", focal_alpha)
        trial.set_user_attr("loss_gamma", focal_gamma)
    else:  # weighted_bce
        bce_pos_weight = trial.suggest_categorical("bce_pos_weight", SEARCH_SPACE_V3["bce_pos_weight"])
        loss_fn = WeightedBCELoss(pos_weight=bce_pos_weight)
        trial.set_user_attr("loss_pos_weight", bce_pos_weight)

    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        raise optuna.TrialPruned("d_model not divisible by n_heads")

    d_ff = d_model * d_ff_ratio
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load data
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values

    # Batch size based on model size
    if d_model >= 192:
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
        loss_fn=loss_fn,
    )

    try:
        result = trainer.train(verbose=False)

        # Get core metrics
        val_auc = result.get("val_auc", 0.5)
        val_precision = result.get("val_precision", 0.0)
        val_recall = result.get("val_recall", 0.0)

        # Compute composite score
        composite = compute_composite_score(val_precision, val_recall, val_auc)

        # Log core metrics
        trial.set_user_attr("val_auc", val_auc)
        trial.set_user_attr("val_precision", val_precision)
        trial.set_user_attr("val_recall", val_recall)
        trial.set_user_attr("composite_score", composite)

        # Log pred_range for probability collapse detection
        pred_range = result.get("val_pred_range")
        if pred_range:
            trial.set_user_attr("pred_min", pred_range[0])
            trial.set_user_attr("pred_max", pred_range[1])

        # Log multi-threshold metrics
        for t in THRESHOLDS:
            t_key = f"t{int(t * 100)}"
            prec_key = f"val_precision_{t_key}"
            rec_key = f"val_recall_{t_key}"

            trial.set_user_attr(prec_key, result.get(prec_key, 0.0))
            trial.set_user_attr(rec_key, result.get(rec_key, 0.0))

        return composite

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
            trial.set_user_attr("error", f"OOM: {error_msg[:200]}")
            raise optuna.TrialPruned(f"OOM error: {error_msg[:100]}")
        trial.set_user_attr("error", f"RuntimeError: {error_msg[:200]}")
        print(f"Trial {trial.number} failed (RuntimeError): {e}")
        return 0.0
    except ValueError as e:
        error_msg = str(e)
        trial.set_user_attr("error", f"ValueError: {error_msg[:200]}")
        if "nan" in error_msg.lower() or "inf" in error_msg.lower():
            raise optuna.TrialPruned(f"NaN/Inf error: {error_msg[:100]}")
        print(f"Trial {trial.number} failed (ValueError): {e}")
        return 0.0
    except KeyboardInterrupt:
        raise
    except Exception as e:
        error_msg = str(e)
        trial.set_user_attr("error", f"Unexpected: {type(e).__name__}: {error_msg[:200]}")
        print(f"Trial {trial.number} failed ({type(e).__name__}): {e}")
        return 0.0


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print(f"PHASE 6C A200 HPO v3: {BUDGET} / horizon={HORIZON}")
    print("=" * 70)
    print(f"Optimization: PRECISION-FIRST (composite = prec*2 + rec*1 + auc*0.1)")
    print(f"Context length: {CONTEXT_LENGTH}d")
    print(f"Features: {NUM_FEATURES} (a200 tier)")
    print(f"Loss types: {SEARCH_SPACE_V3['loss_type']}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Total trials: {N_TRIALS}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        direction=DIRECTION,
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print("Starting HPO with", N_TRIALS, "trials...")
    print("  - Optimizing: composite_score (precision-first)")
    print("  - Loss options: focal (alpha/gamma), weighted_bce (pos_weight)")
    print()
    start_time = time.time()

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("HPO RESULTS (Precision-First)")
    print("=" * 70)
    print("Best trial:", study.best_trial.number)
    print("Best composite score:", round(study.best_value, 4))
    print()
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Show best trial user_attrs
    best_attrs = study.best_trial.user_attrs
    print()
    print("Best trial metrics:")
    print(f"  precision (t50): {best_attrs.get('val_precision', 'N/A')}")
    print(f"  recall (t50): {best_attrs.get('val_recall', 'N/A')}")
    print(f"  AUC: {best_attrs.get('val_auc', 'N/A')}")
    print(f"  composite: {best_attrs.get('composite_score', 'N/A')}")

    if "pred_min" in best_attrs:
        print(f"  pred_range: [{best_attrs.get('pred_min'):.3f}, {best_attrs.get('pred_max'):.3f}]")

    # Show multi-threshold metrics
    print()
    print("Best trial precision/recall at thresholds:")
    for t in THRESHOLDS:
        t_key = f"t{int(t * 100)}"
        prec = best_attrs.get(f"val_precision_{t_key}", "N/A")
        rec = best_attrs.get(f"val_recall_{t_key}", "N/A")
        prec_str = f"{prec:.3f}" if isinstance(prec, float) else prec
        rec_str = f"{rec:.3f}" if isinstance(rec, float) else rec
        print(f"  t={t}: precision={prec_str}, recall={rec_str}")

    print()
    print("Total time:", round(elapsed / 60, 1), "min")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "budget": BUDGET,
        "horizon": HORIZON,
        "n_trials": N_TRIALS,
        "context_length": CONTEXT_LENGTH,
        "optimization_target": "composite_score",
        "composite_formula": "precision*2 + recall*1 + auc*0.1",
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": dict(best_attrs),
        "search_space": SEARCH_SPACE_V3,
        "thresholds": THRESHOLDS,
        "total_time_min": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "best_params.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to", results_path)

    # Save all trials with full details
    trials_data = []
    for trial in study.trials:
        trial_info = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": dict(trial.user_attrs),
            "state": str(trial.state),
        }
        trials_data.append(trial_info)

    trials_path = OUTPUT_DIR / "all_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)

    # Sort trials by composite score and show top 5
    print()
    print("Top 5 trials by composite score:")
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:5]
    for t in sorted_trials:
        attrs = t.user_attrs
        print(
            f"  Trial {t.number}: composite={t.value:.3f}, "
            f"prec={attrs.get('val_precision', 0):.3f}, "
            f"rec={attrs.get('val_recall', 0):.3f}, "
            f"auc={attrs.get('val_auc', 0):.3f}"
        )

    return study.best_value


if __name__ == "__main__":
    main()
