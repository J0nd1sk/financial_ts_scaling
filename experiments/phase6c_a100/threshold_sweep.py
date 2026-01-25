#!/usr/bin/env python3
"""
Phase 6C A100: Threshold Sweeping

Sweep classification thresholds to find optimal operating points.
Runs on all S1 baseline models.

Thresholds tested: [0.50, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]

Operating points identified:
- Max F1 threshold
- 60% precision threshold
- 65% precision threshold
- 70% precision threshold
- 5% recall minimum threshold
"""
import sys
import json
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
# CONFIGURATION
# ============================================================================

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet"
NUM_FEATURES = 100

THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]

# All S1 baseline experiments
S1_EXPERIMENTS = [
    ("s1_01_2m_h1", "2M", 1, 64, 4, 4, 256, 128),
    ("s1_02_20m_h1", "20M", 1, 128, 6, 8, 512, 64),
    ("s1_03_200m_h1", "200M", 1, 256, 8, 8, 1024, 32),
    ("s1_04_2m_h2", "2M", 2, 64, 4, 4, 256, 128),
    ("s1_05_20m_h2", "20M", 2, 128, 6, 8, 512, 64),
    ("s1_06_200m_h2", "200M", 2, 256, 8, 8, 1024, 32),
    ("s1_07_2m_h3", "2M", 3, 64, 4, 4, 256, 128),
    ("s1_08_20m_h3", "20M", 3, 128, 6, 8, 512, 64),
    ("s1_09_200m_h3", "200M", 3, 256, 8, 8, 1024, 32),
    ("s1_10_2m_h5", "2M", 5, 64, 4, 4, 256, 128),
    ("s1_11_20m_h5", "20M", 5, 128, 6, 8, 512, 64),
    ("s1_12_200m_h5", "200M", 5, 256, 8, 8, 1024, 32),
]

OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a100"


# ============================================================================
# THRESHOLD SWEEP
# ============================================================================

def sweep_thresholds(model, dataloader, device, thresholds):
    """Sweep classification thresholds and compute metrics at each."""
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

    results = []
    for threshold in thresholds:
        binary_preds = (preds >= threshold).astype(int)

        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)
        accuracy = accuracy_score(labels, binary_preds)
        positive_rate = binary_preds.mean()

        results.append(dict(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            positive_prediction_rate=positive_rate,
            n_positive_preds=int(binary_preds.sum()),
        ))

    return results


def find_operating_points(sweep_results):
    """Find key operating points from threshold sweep."""
    points = {}

    # Max F1
    max_f1_idx = np.argmax([r["f1"] for r in sweep_results])
    points["max_f1"] = sweep_results[max_f1_idx]

    # Precision targets
    for target in [0.60, 0.65, 0.70]:
        key = f"precision_{int(target*100)}pct"
        valid = [r for r in sweep_results if r["precision"] >= target]
        if valid:
            # Pick the one with highest F1 among valid
            best = max(valid, key=lambda x: x["f1"])
            points[key] = best
        else:
            points[key] = None

    # 5% recall minimum
    valid = [r for r in sweep_results if r["recall"] >= 0.05]
    if valid:
        # Pick highest precision with at least 5% recall
        best = max(valid, key=lambda x: x["precision"])
        points["min_recall_5pct"] = best
    else:
        points["min_recall_5pct"] = None

    return points


def load_model(exp_name, budget, horizon, d_model, n_layers, n_heads, d_ff, batch_size, df, device):
    """Load a trained model from checkpoint."""
    checkpoint_path = OUTPUT_DIR / exp_name / "best_checkpoint.pt"

    if not checkpoint_path.exists():
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
        return None, None

    # Create config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=80,
        horizon=horizon,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=80,
        patch_length=16,
        stride=8,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.5,
        head_dropout=0.0,
    )

    # Create splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=80,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create trainer (to get dataloader)
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=1e-4,
        epochs=1,  # Not training
        device=device,
        checkpoint_dir=OUTPUT_DIR / exp_name,
        split_indices=split_indices,
        use_revin=True,
        high_prices=df["High"].values,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])

    return trainer.model, trainer.val_dataloader


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 6C A100: Threshold Sweeping")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    # Load data once
    print("\nLoading data...")
    df = pd.read_parquet(DATA_PATH)
    print("Data:", len(df), "rows")

    all_results = []

    for exp_name, budget, horizon, d_model, n_layers, n_heads, d_ff, batch_size in S1_EXPERIMENTS:
        print()
        print("-" * 50)
        print(f"Sweeping: {exp_name}")
        print("-" * 50)

        model, val_dataloader = load_model(
            exp_name, budget, horizon, d_model, n_layers, n_heads, d_ff, batch_size, df, device
        )

        if model is None:
            print("  Skipping (no checkpoint)")
            continue

        # Run sweep
        sweep_results = sweep_thresholds(model, val_dataloader, device, THRESHOLDS)

        # Find operating points
        operating_points = find_operating_points(sweep_results)

        # Print summary
        print("  Threshold sweep:")
        for r in sweep_results[::2]:  # Print every other for brevity
            print(f"    τ={r['threshold']:.2f}: P={r['precision']:.3f}, R={r['recall']:.3f}, F1={r['f1']:.3f}")

        print("  Operating points:")
        for name, point in operating_points.items():
            if point:
                print(f"    {name}: τ={point['threshold']:.2f}, P={point['precision']:.3f}, R={point['recall']:.3f}")
            else:
                print(f"    {name}: Not achievable")

        # Store results
        exp_results = dict(
            experiment=exp_name,
            budget=budget,
            horizon=horizon,
            sweep=sweep_results,
            operating_points={k: v for k, v in operating_points.items() if v is not None},
        )
        all_results.append(exp_results)

    # Save results
    output_path = OUTPUT_DIR / "threshold_sweep_results.json"
    with open(output_path, "w") as f:
        json.dump(dict(
            experiments=all_results,
            thresholds=THRESHOLDS,
            timestamp=datetime.now().isoformat(),
        ), f, indent=2)

    print()
    print("=" * 70)
    print("Threshold sweep complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
