#!/usr/bin/env python3
"""
Phase 6C Stage 1: Threshold Sweep Analysis

Evaluates Stage 1 models at different decision thresholds to find optimal
operating points for different objectives (precision, recall, F1, etc.)
"""
import sys
import json
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig, PatchTST
from src.data.dataset import SimpleSplitter, FinancialDataset
from torch.utils.data import DataLoader, Subset

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet"
NUM_FEATURES = 55  # 5 OHLCV + 50 indicators (auto-discovered by Trainer)
CONTEXT_LENGTH = 80
HORIZON = 1

EXPERIMENTS = [
    {
        "name": "s1_01_2m_h1_a50",
        "budget": "2M",
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 256,
    },
    {
        "name": "s1_02_20m_h1_a50",
        "budget": "20M",
        "d_model": 128,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 512,
    },
    {
        "name": "s1_03_200m_h1_a50",
        "budget": "200M",
        "d_model": 256,
        "n_layers": 8,
        "n_heads": 8,
        "d_ff": 1024,
    },
]

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_model_and_get_predictions(exp_config, checkpoint_path, df, split_indices, device):
    """Load model from checkpoint and get validation predictions."""

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=exp_config["d_model"],
        n_heads=exp_config["n_heads"],
        n_layers=exp_config["n_layers"],
        d_ff=exp_config["d_ff"],
        dropout=0.5,
        head_dropout=0.0,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model and load weights (use_revin=True as training used RevIN)
    model = PatchTST(model_config, use_revin=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Prepare validation data
    close_prices = df["Close"].values
    high_prices = df["High"].values

    # Create full dataset (1% threshold for threshold_1pct task)
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=0.01,  # 1% threshold
        high_prices=high_prices,
    )

    # Create validation subset using indices
    val_subset = Subset(full_dataset, split_indices.val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def compute_metrics_at_threshold(preds, labels, threshold):
    """Compute metrics at a specific decision threshold."""
    binary_preds = (preds >= threshold).astype(int)

    # Handle edge cases
    n_positive_preds = binary_preds.sum()
    n_negative_preds = len(binary_preds) - n_positive_preds

    if n_positive_preds == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif n_negative_preds == 0:
        precision = labels.mean()  # All positive predictions
        recall = 1.0
        f1 = 2 * precision / (precision + 1) if precision > 0 else 0
    else:
        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_positive_preds": int(n_positive_preds),
        "n_negative_preds": int(n_negative_preds),
        "pred_positive_rate": n_positive_preds / len(binary_preds),
    }


def find_optimal_thresholds(sweep_results):
    """Find optimal thresholds for different objectives."""
    df = pd.DataFrame(sweep_results)

    # Filter to reasonable thresholds (at least some predictions in each class)
    valid = df[(df["n_positive_preds"] > 0) & (df["n_negative_preds"] > 0)]

    if len(valid) == 0:
        return {}

    optimal = {}

    # Best F1
    best_f1_idx = valid["f1"].idxmax()
    optimal["best_f1"] = valid.loc[best_f1_idx].to_dict()

    # Best precision (with at least 5% recall)
    high_precision = valid[valid["recall"] >= 0.05]
    if len(high_precision) > 0:
        best_prec_idx = high_precision["precision"].idxmax()
        optimal["best_precision_5pct_recall"] = high_precision.loc[best_prec_idx].to_dict()

    # Best recall (with at least 30% precision)
    high_recall = valid[valid["precision"] >= 0.30]
    if len(high_recall) > 0:
        best_recall_idx = high_recall["recall"].idxmax()
        optimal["best_recall_30pct_precision"] = high_recall.loc[best_recall_idx].to_dict()

    # Best accuracy
    best_acc_idx = valid["accuracy"].idxmax()
    optimal["best_accuracy"] = valid.loc[best_acc_idx].to_dict()

    # Balanced (precision >= 40%, recall >= 10%)
    balanced = valid[(valid["precision"] >= 0.40) & (valid["recall"] >= 0.10)]
    if len(balanced) > 0:
        best_balanced_idx = balanced["f1"].idxmax()
        optimal["balanced"] = balanced.loc[best_balanced_idx].to_dict()

    return optimal


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6C STAGE 1: THRESHOLD SWEEP ANALYSIS")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()
    print(f"Validation samples: {len(split_indices.val_indices)}")

    all_results = {}

    for exp in EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {exp['name']} ({exp['budget']})")
        print("=" * 80)

        checkpoint_path = PROJECT_ROOT / f"outputs/phase6c/{exp['name']}/best_checkpoint.pt"

        if not checkpoint_path.exists():
            print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
            continue

        # Get predictions
        print("  Loading model and getting predictions...")
        preds, labels = load_model_and_get_predictions(exp, checkpoint_path, df, split_indices, device)

        # Compute AUC (threshold-independent)
        auc = roc_auc_score(labels, preds)
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Class balance: {labels.mean():.3f} ({int(labels.sum())}/{len(labels)} positives)")

        # Sweep thresholds
        print(f"\n  Threshold Sweep:")
        print(f"  {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Pos Preds':>10}")
        print("  " + "-" * 60)

        sweep_results = []
        for thresh in THRESHOLDS:
            metrics = compute_metrics_at_threshold(preds, labels, thresh)
            sweep_results.append(metrics)
            print(f"  {thresh:>6.2f} | {metrics['accuracy']:>6.3f} | {metrics['precision']:>6.3f} | "
                  f"{metrics['recall']:>6.3f} | {metrics['f1']:>6.3f} | {metrics['n_positive_preds']:>10}")

        # Find optimal thresholds
        optimal = find_optimal_thresholds(sweep_results)

        print(f"\n  OPTIMAL THRESHOLDS:")
        print("  " + "-" * 60)
        for objective, result in optimal.items():
            print(f"  {objective}:")
            print(f"    Threshold: {result['threshold']:.2f}")
            print(f"    Accuracy: {result['accuracy']:.3f}, Precision: {result['precision']:.3f}, "
                  f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}")

        all_results[exp['name']] = {
            "budget": exp['budget'],
            "auc": auc,
            "pred_range": [float(preds.min()), float(preds.max())],
            "sweep": sweep_results,
            "optimal": optimal,
        }

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: BEST OPERATING POINTS BY BUDGET")
    print("=" * 80)

    print(f"\n{'Budget':<8} | {'Objective':<25} | {'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}")
    print("-" * 85)

    for exp_name, results in all_results.items():
        budget = results['budget']
        for obj_name, obj_result in results['optimal'].items():
            print(f"{budget:<8} | {obj_name:<25} | {obj_result['threshold']:>6.2f} | "
                  f"{obj_result['accuracy']:>6.3f} | {obj_result['precision']:>6.3f} | "
                  f"{obj_result['recall']:>6.3f} | {obj_result['f1']:>6.3f}")
        print("-" * 85)

    # Save results
    output_path = PROJECT_ROOT / "outputs/phase6c/stage1_threshold_sweep.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
