#!/usr/bin/env python3
"""Threshold sweep analysis for top supplementary HPO models.

Analyzes precision/recall trade-off at different probability thresholds.

Usage:
    ./venv/bin/python scripts/threshold_sweep.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import SimpleSplitter
from src.models.patchtst import PatchTSTConfig
from src.config.experiment import ExperimentConfig
from src.training.trainer import Trainer


# Top models to analyze (from Round 1 and Round 2)
TOP_MODELS = {
    # Round 1 champions
    "D2_recall": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                  "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "F1_precision": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                     "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 3e-4},
    "E4_balance": {"d_model": 160, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                   "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
    # Round 2 champions
    "G4_recall": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                  "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 3e-4},
    "G2_balance": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                   "dropout": 0.45, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "H1_precision": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                     "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 5e-4},
}

# Thresholds to sweep
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def train_and_get_predictions(config: dict, tier: str = "a50") -> tuple[np.ndarray, np.ndarray]:
    """Train a model and return validation probabilities and labels.

    Args:
        config: Hyperparameter dictionary
        tier: Feature tier

    Returns:
        Tuple of (probabilities, labels) arrays
    """
    NUM_FEATURES = {"a50": 55, "a100": 105}
    CONTEXT_LENGTH = 80
    EPOCHS = 50

    # Load data
    data_path = PROJECT_ROOT / f"data/processed/v1/SPY_dataset_{tier}_combined.parquet"
    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    num_features = NUM_FEATURES[tier]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Batch size based on d_model
    d_model = config["d_model"]
    batch_size = 32 if d_model >= 256 else (64 if d_model >= 128 else 128)

    # Experiment config
    exp_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=1,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Model config
    model_config = PatchTSTConfig(
        num_features=num_features,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_model"] * config["d_ff_ratio"],
        dropout=config["dropout"],
        head_dropout=0.0,
    )

    # Splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=1,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Train
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=exp_config,
            model_config=model_config,
            batch_size=batch_size,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
            high_prices=high_prices,
        )

        # Train and get raw predictions
        trainer.train(verbose=False)

        # Get validation predictions
        trainer.model.eval()
        val_probs = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in trainer.val_dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = trainer.model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = batch_y.cpu().numpy().flatten()

                val_probs.extend(probs)
                val_labels.extend(labels)

        return np.array(val_probs), np.array(val_labels)


def compute_metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Compute precision, recall, F1 at a given threshold.

    Args:
        probs: Predicted probabilities
        labels: True binary labels
        threshold: Classification threshold

    Returns:
        Dictionary with metrics
    """
    preds = (probs >= threshold).astype(int)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n_predictions": int(tp + fp),
    }


def sweep_thresholds(probs: np.ndarray, labels: np.ndarray, thresholds: list[float]) -> list[dict]:
    """Sweep all thresholds and compute metrics.

    Args:
        probs: Predicted probabilities
        labels: True binary labels
        thresholds: List of thresholds to try

    Returns:
        List of metric dictionaries
    """
    return [compute_metrics_at_threshold(probs, labels, t) for t in thresholds]


def format_sweep_table(results: list[dict]) -> str:
    """Format sweep results as markdown table."""
    lines = [
        "| Threshold | Precision | Recall | F1 | Accuracy | TPs | Predictions |",
        "|-----------|-----------|--------|-----|----------|-----|-------------|",
    ]

    for r in results:
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']*100:5.1f}% | {r['recall']*100:5.1f}% | "
            f"{r['f1']:.3f} | {r['accuracy']*100:5.1f}% | {r['tp']:3d} | {r['n_predictions']:3d} |"
        )

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Threshold Sweep Analysis - Top Supplementary HPO Models")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "outputs/phase6c_a50/threshold_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, config in TOP_MODELS.items():
        print(f"\n{'='*70}")
        print(f"Model: {name}")
        print(f"Config: {config}")
        print("=" * 70)

        start = time.time()
        print("Training model...")
        probs, labels = train_and_get_predictions(config)
        duration = time.time() - start
        print(f"Training complete ({duration:.1f}s)")

        print(f"Validation samples: {len(labels)}")
        print(f"Positive rate: {labels.mean()*100:.1f}%")
        print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")

        # Sweep thresholds
        sweep_results = sweep_thresholds(probs, labels, THRESHOLDS)
        all_results[name] = {
            "config": config,
            "sweep": sweep_results,
            "prob_stats": {
                "min": float(probs.min()),
                "max": float(probs.max()),
                "mean": float(probs.mean()),
                "std": float(probs.std()),
            }
        }

        print("\n" + format_sweep_table(sweep_results))

        # Find best F1
        best_f1 = max(sweep_results, key=lambda x: x["f1"])
        print(f"\nBest F1: {best_f1['f1']:.3f} at threshold {best_f1['threshold']:.2f}")
        print(f"  -> Precision: {best_f1['precision']*100:.1f}%, Recall: {best_f1['recall']*100:.1f}%")

        # Find operating points
        for target_prec in [0.6, 0.7, 0.8]:
            for r in sweep_results:
                if r["precision"] >= target_prec and r["recall"] > 0:
                    print(f"  {int(target_prec*100)}%+ precision at t={r['threshold']:.2f}: "
                          f"P={r['precision']*100:.1f}%, R={r['recall']*100:.1f}%, {r['tp']} TPs")
                    break

    # Save results
    with open(output_dir / "threshold_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary report
    report_lines = [
        "# Threshold Sweep Analysis Results",
        f"\nGenerated: {pd.Timestamp.now().isoformat()}",
        "\n## Summary",
        f"\nModels analyzed: {len(TOP_MODELS)}",
        f"Thresholds tested: {len(THRESHOLDS)} ({min(THRESHOLDS):.2f} - {max(THRESHOLDS):.2f})",
        "\n## Best Operating Points by Model",
        "\n| Model | Best F1 | Threshold | Precision | Recall | TPs |",
        "|-------|---------|-----------|-----------|--------|-----|",
    ]

    for name, data in all_results.items():
        best = max(data["sweep"], key=lambda x: x["f1"])
        report_lines.append(
            f"| {name} | {best['f1']:.3f} | {best['threshold']:.2f} | "
            f"{best['precision']*100:.1f}% | {best['recall']*100:.1f}% | {best['tp']} |"
        )

    report_lines.extend([
        "\n## High Precision Operating Points (>=60%)",
        "\n| Model | Threshold | Precision | Recall | TPs |",
        "|-------|-----------|-----------|--------|-----|",
    ])

    for name, data in all_results.items():
        for r in data["sweep"]:
            if r["precision"] >= 0.6 and r["recall"] > 0:
                report_lines.append(
                    f"| {name} | {r['threshold']:.2f} | {r['precision']*100:.1f}% | "
                    f"{r['recall']*100:.1f}% | {r['tp']} |"
                )
                break

    report_lines.extend([
        "\n## Detailed Results by Model",
    ])

    for name, data in all_results.items():
        report_lines.extend([
            f"\n### {name}",
            f"\nConfig: `{data['config']}`",
            f"\nProbability range: [{data['prob_stats']['min']:.3f}, {data['prob_stats']['max']:.3f}]",
            f"\n{format_sweep_table(data['sweep'])}",
        ])

    with open(output_dir / "threshold_sweep_report.md", "w") as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print(f"  - threshold_sweep_results.json")
    print(f"  - threshold_sweep_report.md")


if __name__ == "__main__":
    main()
