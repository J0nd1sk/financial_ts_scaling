#!/usr/bin/env python3
"""
Sweep classification thresholds on trained models to find optimal accuracy/precision tradeoff.

This loads the saved predictions and sweeps different thresholds to find
configurations that maximize accuracy or precision at the cost of recall.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.data.dataset import SimpleSplitter, FinancialDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
CONTEXT_LENGTH = 80
HORIZON = 1
THRESHOLD = 0.005

# Models to evaluate
MODELS = {
    "2M_h2": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/2M_narrow_deep_threshold_05pct_HIGH/checkpoints",
        "d_model": 64, "n_layers": 32, "n_heads": 2, "d_ff": 256, "dropout": 0.5,
    },
    "2M_h4": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/2M_narrow_h4_threshold_05pct_HIGH/checkpoints",
        "d_model": 64, "n_layers": 32, "n_heads": 4, "d_ff": 256, "dropout": 0.5,
    },
    "2M_h8": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/2M_narrow_h8_threshold_05pct_HIGH/checkpoints",
        "d_model": 64, "n_layers": 32, "n_heads": 8, "d_ff": 256, "dropout": 0.5,
    },
    "20M_h2": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/20M_wide_h2_threshold_05pct_HIGH/checkpoints",
        "d_model": 512, "n_layers": 6, "n_heads": 2, "d_ff": 2048, "dropout": 0.5,
    },
    "20M_h4": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/20M_wide_h4_threshold_05pct_HIGH/checkpoints",
        "d_model": 512, "n_layers": 6, "n_heads": 4, "d_ff": 2048, "dropout": 0.5,
    },
    "20M_h8": {
        "checkpoint": PROJECT_ROOT / "outputs/threshold_05pct_high/20M_wide_h8_threshold_05pct_HIGH/checkpoints",
        "d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048, "dropout": 0.5,
    },
}

# Thresholds to sweep
THRESHOLDS = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# ============================================================================
# MAIN
# ============================================================================

def get_predictions(model_config, checkpoint_dir, test_loader, device):
    """Load model and get predictions on test set."""
    # Find checkpoint file
    ckpt_files = list(checkpoint_dir.glob("*.pt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    ckpt_path = ckpt_files[0]

    # Create model
    config = PatchTSTConfig(
        num_features=25,  # OHLCV (5) + indicators (20)
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        d_ff=model_config["d_ff"],
        dropout=model_config["dropout"],
        head_dropout=0.0,  # Match training config
    )
    model = PatchTST(config, use_revin=True).to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def evaluate_threshold(preds, labels, threshold):
    """Evaluate metrics at a given classification threshold."""
    binary_preds = (preds >= threshold).astype(int)

    # Handle edge case where all predictions are same class
    if len(np.unique(binary_preds)) == 1:
        return {
            "threshold": threshold,
            "accuracy": accuracy_score(labels, binary_preds),
            "precision": 0.0 if binary_preds[0] == 0 else precision_score(labels, binary_preds, zero_division=0),
            "recall": 0.0 if binary_preds[0] == 0 else recall_score(labels, binary_preds, zero_division=0),
            "f1": 0.0,
            "n_positive_preds": int(binary_preds.sum()),
            "n_total": len(binary_preds),
        }

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "n_positive_preds": int(binary_preds.sum()),
        "n_total": len(binary_preds),
    }


def main():
    print("=" * 70)
    print("THRESHOLD SWEEP: Finding optimal classification threshold")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    close_prices = df["Close"].values
    high_prices = df["High"].values

    # Create splits
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create dataset
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=THRESHOLD,
        high_prices=high_prices,
    )

    test_subset = torch.utils.data.Subset(full_dataset, split_indices.test_indices.tolist())
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    print(f"Test samples: {len(split_indices.test_indices)}")

    # Evaluate each model
    all_results = []

    for model_name, model_config in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        try:
            preds, labels = get_predictions(model_config, model_config["checkpoint"], test_loader, device)
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")
            continue

        auc = roc_auc_score(labels, preds)
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}]")

        print(f"\n{'Thresh':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'#Pos':<6}")
        print("-" * 50)

        for thresh in THRESHOLDS:
            result = evaluate_threshold(preds, labels, thresh)
            result["model"] = model_name
            result["auc_roc"] = auc
            all_results.append(result)

            print(f"{thresh:<8.2f} {result['accuracy']:<8.4f} {result['precision']:<8.4f} "
                  f"{result['recall']:<8.4f} {result['f1']:<8.4f} {result['n_positive_preds']:<6}")

    # Find best configurations
    print(f"\n{'=' * 70}")
    print("BEST CONFIGURATIONS")
    print(f"{'=' * 70}")

    results_df = pd.DataFrame(all_results)

    # Best accuracy
    best_acc = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\nBest Accuracy: {best_acc['accuracy']:.4f}")
    print(f"  Model: {best_acc['model']}, Threshold: {best_acc['threshold']:.2f}")
    print(f"  Precision: {best_acc['precision']:.4f}, Recall: {best_acc['recall']:.4f}")

    # Best precision (with at least 5 predictions)
    valid_prec = results_df[results_df['n_positive_preds'] >= 5]
    if len(valid_prec) > 0:
        best_prec = valid_prec.loc[valid_prec['precision'].idxmax()]
        print(f"\nBest Precision (≥5 trades): {best_prec['precision']:.4f}")
        print(f"  Model: {best_prec['model']}, Threshold: {best_prec['threshold']:.2f}")
        print(f"  Accuracy: {best_prec['accuracy']:.4f}, Recall: {best_prec['recall']:.4f}")
        print(f"  Trades: {best_prec['n_positive_preds']}")

    # Best F1
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    print(f"\nBest F1: {best_f1['f1']:.4f}")
    print(f"  Model: {best_acc['model']}, Threshold: {best_f1['threshold']:.2f}")
    print(f"  Precision: {best_f1['precision']:.4f}, Recall: {best_f1['recall']:.4f}")

    # Save results
    output_path = PROJECT_ROOT / "outputs/threshold_05pct_high/threshold_sweep_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
