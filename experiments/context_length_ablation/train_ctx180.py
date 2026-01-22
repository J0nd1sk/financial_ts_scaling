#!/usr/bin/env python3
"""
Context Length Ablation: 180 days (3x baseline)
Purpose: Test impact of context window size on PatchTST performance
Architecture: d_model=64, n_layers=4, n_heads=4 (PatchTST-recommended)
Context: 60 days (current default)
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "ctx_ablation_180"
HORIZON = 1

# Architecture (PatchTST-recommended: 2-4 layers, not 48!)
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 256  # 4 * d_model

# Training
LEARNING_RATE = 1e-4  # Slower for stability
BATCH_SIZE = 128
EPOCHS = 50
DROPOUT = 0.50  # Higher for regularization

# Context length (ABLATION VARIABLE)
CONTEXT_LENGTH = 180

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
NUM_FEATURES = 20

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/context_length_ablation" / EXPERIMENT_NAME


# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions + labels."""
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

    # Binary predictions at 0.5 threshold
    binary_preds = (preds >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"CONTEXT LENGTH ABLATION: {EXPERIMENT_NAME}")
    print(f"Context Length: {CONTEXT_LENGTH} days")
    print("=" * 70)

    # Check MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values
    print(f"Data: {len(df)} rows")

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,  # Standard PatchTST
        stride=8,  # Standard PatchTST
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    # Print config
    print(f"\nModel Config:")
    print(f"  d_model: {D_MODEL}, n_layers: {N_LAYERS}, n_heads: {N_HEADS}")
    print(f"  context_length: {CONTEXT_LENGTH}, patch_length: 16, stride: 8")
    print(f"  dropout: {DROPOUT}, RevIN: enabled")
    print(f"  num_patches: {model_config.num_patches}")

    # Create date-based splits using SimpleSplitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"\nSplit sizes:")
    print(f"  Train: {len(split_indices.train_indices)} samples")
    print(f"  Val: {len(split_indices.val_indices)} samples")
    print(f"  Test: {len(split_indices.test_indices)} samples")

    # Create trainer with RevIN enabled
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,  # RevIN for non-stationary financial data
            high_prices=high_prices,
        )

        # Train
        print(f"\nTraining for {EPOCHS} epochs...")
        start_time = time.time()

        result = trainer.train(verbose=True)

        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")

        # Get metrics from training
        val_auc = result.get("val_auc")
        val_loss = result.get("val_loss")
        stopped_early = result.get("stopped_early", False)

        # Evaluate on validation set for accuracy/recall
        eval_metrics = evaluate_model(trainer.model, trainer.val_dataloader, trainer.device)

    print(f"\nResults:")
    print(f"  Val AUC: {val_auc:.4f}" if val_auc else "  Val AUC: N/A")
    print(f"  Val Loss: {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"  F1: {eval_metrics['f1']:.4f}")
    print(f"  Pred Range: [{eval_metrics['pred_min']:.4f}, {eval_metrics['pred_max']:.4f}]")
    print(f"  Stopped Early: {stopped_early}")

    # Save results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "context_length": CONTEXT_LENGTH,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "use_revin": True,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
        "results": {
            "val_auc": val_auc,
            "val_loss": val_loss,
            "train_loss": result.get("train_loss"),
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "pred_min": eval_metrics["pred_min"],
            "pred_max": eval_metrics["pred_max"],
            "pred_std": eval_metrics["pred_std"],
            "stopped_early": stopped_early,
            "training_time_minutes": elapsed / 60,
        },
        "splits": {
            "train_samples": len(split_indices.train_indices),
            "val_samples": len(split_indices.val_indices),
            "test_samples": len(split_indices.test_indices),
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return val_auc


if __name__ == "__main__":
    main()
