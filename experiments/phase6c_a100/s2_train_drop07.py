#!/usr/bin/env python3
"""
Phase 6C A100 Architecture Variation: drop07

More Regularization

Base: Best performing budget from S1 baseline
Modification: dropout=0.7 (more regularization)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "s2_train_drop07"
HORIZON = 1  # Test on H1

# Architecture variation
D_MODEL = 128
N_LAYERS = 6
N_HEADS = 8
D_FF = 512

# Hyperparameters
LEARNING_RATE = 0.0001
DROPOUT = 0.7
CONTEXT_LENGTH = 80

# Training
BATCH_SIZE = 64
EPOCHS = 50

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet"
NUM_FEATURES = 100

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a100" / EXPERIMENT_NAME


def evaluate_model(model, dataloader, device):
    """Evaluate model with full metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None

    return dict(
        auc=auc,
        accuracy=accuracy_score(labels, binary_preds),
        precision=precision_score(labels, binary_preds, zero_division=0),
        recall=recall_score(labels, binary_preds, zero_division=0),
        f1=f1_score(labels, binary_preds, zero_division=0),
        pred_min=float(preds.min()),
        pred_max=float(preds.max()),
        pred_mean=float(preds.mean()),
        pred_std=float(preds.std()),
        n_positive_preds=int((preds >= 0.5).sum()),
        n_samples=len(labels),
        class_balance=float(labels.mean()),
    )


def main():
    print("=" * 70)
    print("PHASE 6C A100 Architecture Variation: " + EXPERIMENT_NAME)
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values
    print("Data:", len(df), "rows")

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
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    print("Architecture: d=" + str(D_MODEL) + ", L=" + str(N_LAYERS) + ", h=" + str(N_HEADS) + ", d_ff=" + str(D_FF))
    print("Training: lr=" + str(LEARNING_RATE) + ", dropout=" + str(DROPOUT))

    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        split_indices=split_indices,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric="val_auc",
        use_revin=True,
        high_prices=high_prices,
    )

    print("Training for", EPOCHS, "epochs...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    val_metrics = evaluate_model(trainer.model, trainer.val_dataloader, trainer.device)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print("Training time:", round(elapsed/60, 1), "min")
    if val_metrics["auc"]:
        print("AUC:", round(val_metrics["auc"], 4))
    print("Accuracy:", round(val_metrics["accuracy"], 4))
    print("Precision:", round(val_metrics["precision"], 4))
    print("Recall:", round(val_metrics["recall"], 4))
    print("F1:", round(val_metrics["f1"], 4))

    results = dict(
        experiment=EXPERIMENT_NAME,
        phase="6C",
        stage="S2_arch_variation",
        variation="drop07",
        horizon=HORIZON,
        feature_tier="a100",
        num_features=NUM_FEATURES,
        architecture=dict(
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            d_ff=D_FF,
        ),
        hyperparameters=dict(
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
            context_length=CONTEXT_LENGTH,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            use_revin=True,
        ),
        training=dict(
            train_loss=result.get("train_loss"),
            val_loss=result.get("val_loss"),
            val_auc=result.get("val_auc"),
            stopped_early=result.get("stopped_early", False),
            training_time_min=elapsed / 60,
        ),
        val_metrics=val_metrics,
        timestamp=datetime.now().isoformat(),
    )

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to", results_path)

    return val_metrics["auc"]


if __name__ == "__main__":
    main()
