#!/usr/bin/env python3
"""Loss function hyperparameter sweep.

Fixes architecture at best BCE baseline and sweeps over loss function
parameters (gamma, alpha) and dropout to find optimal ranges before full HPO.

Sweep space:
- gamma: [0, 0.5, 1.0, 2.0] - 0 is weighted BCE, higher adds focal effect
- alpha: [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9] - higher weights positives more
- dropout: [0.3, 0.4, 0.5] - regularization strength

Total: 4 × 10 × 3 = 120 combinations per tier

Usage:
    ./venv/bin/python scripts/run_loss_sweep.py --tier a50
    ./venv/bin/python scripts/run_loss_sweep.py --tier a100
    ./venv/bin/python scripts/run_loss_sweep.py --tier a200
    ./venv/bin/python scripts/run_loss_sweep.py --tier a50 --gamma 0 --alpha 0.8 --dropout 0.5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Sweep ranges
GAMMA_VALUES = [0.0, 0.5, 1.0, 2.0]
ALPHA_VALUES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
DROPOUT_VALUES = [0.3, 0.4, 0.5]

# Best BCE baseline architectures (fixed during sweep)
BEST_ARCHITECTURES = {
    "a50": {
        "d_model": 128,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff_ratio": 4,
        "dropout": 0.3,
        "learning_rate": 5e-5,
        "weight_decay": 1e-4,
    },
    "a100": {
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff_ratio": 2,
        "dropout": 0.7,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
    },
    "a200": {  # Same as a100 - higher regularization for more features
        "d_model": 64,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff_ratio": 2,
        "dropout": 0.7,
        "learning_rate": 5e-4,
        "weight_decay": 1e-3,
    },
}

# Feature counts by tier
NUM_FEATURES_BY_TIER = {
    "a20": 25,
    "a50": 55,
    "a100": 105,
    "a200": 211,
}

# Fixed training parameters
CONTEXT_LENGTH = 80
EPOCHS = 50
HORIZON = 1


def get_thermal_reading() -> float | None:
    """Get current CPU temperature using powermetrics."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.split("\n"):
            if "CPU die temperature" in line:
                temp_str = line.split(":")[1].strip().replace(" C", "")
                return float(temp_str)
    except Exception:
        pass
    return None


def train_single_config(
    tier: str,
    gamma: float,
    alpha: float,
    dropout: float,
    arch_config: dict,
    output_dir: Path,
) -> dict:
    """Train a single gamma/alpha/dropout combination and return metrics."""
    import pandas as pd
    import torch

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.config.experiment import ExperimentConfig
    from src.data.dataset import SimpleSplitter
    from src.models.patchtst import PatchTSTConfig
    from src.training.losses import FocalLoss
    from src.training.trainer import Trainer

    start_time = time.time()

    # Load dataset
    data_path = PROJECT_ROOT / f"data/processed/v1/SPY_dataset_{tier}_combined.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    num_features = NUM_FEATURES_BY_TIER.get(tier, 105)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Batch size based on model size
    d_model = arch_config["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    # Experiment config
    exp_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Model config - use passed dropout, not arch_config's default
    model_config = PatchTSTConfig(
        num_features=num_features,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=arch_config["d_model"],
        n_heads=arch_config["n_heads"],
        n_layers=arch_config["n_layers"],
        d_ff=arch_config["d_model"] * arch_config["d_ff_ratio"],
        dropout=dropout,
        head_dropout=0.0,
    )

    # Splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Loss function - FocalLoss with specified gamma and alpha
    criterion = FocalLoss(gamma=gamma, alpha=alpha)

    # Train
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=exp_config,
            model_config=model_config,
            batch_size=batch_size,
            learning_rate=arch_config["learning_rate"],
            weight_decay=arch_config["weight_decay"],
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
            high_prices=high_prices,
            criterion=criterion,
        )

        result = trainer.train(verbose=True)

    duration = time.time() - start_time

    return {
        "gamma": gamma,
        "alpha": alpha,
        "dropout": dropout,
        "tier": tier,
        "architecture": arch_config,
        "duration_sec": duration,
        "epochs_run": result.get("epochs_run", EPOCHS),
        "stopped_early": result.get("stopped_early", False),
        "val_auc": result.get("val_auc"),
        "val_precision": result.get("val_precision"),
        "val_recall": result.get("val_recall"),
        "val_pred_range": result.get("val_pred_range"),
        "train_auc": result.get("train_auc"),
        "train_pred_range": result.get("train_pred_range"),
        "timestamp": datetime.now().isoformat(),
    }


def run_loss_sweep(
    tier: str,
    gamma_values: list[float] | None = None,
    alpha_values: list[float] | None = None,
    dropout_values: list[float] | None = None,
    output_base: Path | None = None,
    thermal_pause_threshold: float = 85.0,
) -> dict:
    """Run loss function parameter sweep.

    Args:
        tier: Feature tier ("a50" or "a100")
        gamma_values: List of gamma values to sweep (default: GAMMA_VALUES)
        alpha_values: List of alpha values to sweep (default: ALPHA_VALUES)
        dropout_values: List of dropout values to sweep (default: DROPOUT_VALUES)
        output_base: Output directory
        thermal_pause_threshold: Pause if temperature exceeds this

    Returns:
        Dictionary with results and summary
    """
    if gamma_values is None:
        gamma_values = GAMMA_VALUES
    if alpha_values is None:
        alpha_values = ALPHA_VALUES
    if dropout_values is None:
        dropout_values = DROPOUT_VALUES

    if tier not in BEST_ARCHITECTURES:
        raise ValueError(f"Unknown tier: {tier}. Must be one of {list(BEST_ARCHITECTURES.keys())}")

    arch_config = BEST_ARCHITECTURES[tier]

    # Generate all combinations (gamma × alpha × dropout)
    combinations = list(product(gamma_values, alpha_values, dropout_values))
    total_trials = len(combinations)

    # Setup output directory
    if output_base is None:
        output_base = PROJECT_ROOT / f"outputs/phase6c_{tier}/loss_sweep"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Loss Function Sweep - Tier {tier}")
    print(f"{'='*60}")
    print(f"Architecture (fixed): {arch_config}")
    print(f"Gamma values: {gamma_values}")
    print(f"Alpha values: {alpha_values}")
    print(f"Dropout values: {dropout_values}")
    print(f"Total trials: {total_trials}")
    print(f"Output: {output_base}")
    print(f"{'='*60}\n")

    all_results = []
    results_file = output_base / "all_results.json"

    for i, (gamma, alpha, dropout) in enumerate(combinations, 1):
        print(f"\n[{i}/{total_trials}] gamma={gamma}, alpha={alpha}, dropout={dropout}")

        # Thermal check
        temp = get_thermal_reading()
        if temp is not None:
            print(f"  Temperature: {temp:.1f}C")
            if temp > thermal_pause_threshold:
                print(f"  Warning: Above {thermal_pause_threshold}C, pausing 60s...")
                time.sleep(60)

        try:
            result = train_single_config(
                tier=tier,
                gamma=gamma,
                alpha=alpha,
                dropout=dropout,
                arch_config=arch_config,
                output_dir=output_base,
            )
            all_results.append(result)

            # Report
            print(f"  AUC: {result['val_auc']:.4f}")
            prec = result.get('val_precision')
            recall = result.get('val_recall')
            pred_range = result.get('val_pred_range')
            print(f"  Precision: {prec:.4f}" if prec else "  Precision: N/A")
            print(f"  Recall: {recall:.4f}" if recall else "  Recall: N/A")
            print(f"  Pred range: {pred_range}")
            print(f"  Duration: {result['duration_sec']:.1f}s")

        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({
                "gamma": gamma,
                "alpha": alpha,
                "dropout": dropout,
                "tier": tier,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        # Incremental save
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(all_results, tier, output_base)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Completed: {summary['completed']}/{summary['total']}")
    if summary['completed'] > 0:
        print(f"Best AUC: {summary['best_auc']:.4f} (gamma={summary['best_gamma']}, alpha={summary['best_alpha']}, dropout={summary['best_dropout']})")
        print(f"Best Precision: {summary.get('best_precision_value', 'N/A')}")
        print(f"Best Recall: {summary.get('best_recall_value', 'N/A')}")
    print(f"Results: {output_base}")
    print(f"{'='*60}\n")

    return {"results": all_results, "summary": summary}


def generate_summary(results: list[dict], tier: str, output_dir: Path) -> dict:
    """Generate summary files from results."""
    import pandas as pd

    successful = [r for r in results if "val_auc" in r and r["val_auc"] is not None]

    if not successful:
        return {"completed": 0, "total": len(results), "best_auc": 0}

    # Find best by different criteria
    best_auc = max(successful, key=lambda x: x["val_auc"])
    best_precision = max(successful, key=lambda x: x.get("val_precision") or 0)
    best_recall = max(successful, key=lambda x: x.get("val_recall") or 0)

    summary = {
        "tier": tier,
        "total": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        # Best by AUC
        "best_gamma": best_auc["gamma"],
        "best_alpha": best_auc["alpha"],
        "best_dropout": best_auc["dropout"],
        "best_auc": best_auc["val_auc"],
        # Best by precision
        "best_precision_gamma": best_precision["gamma"],
        "best_precision_alpha": best_precision["alpha"],
        "best_precision_dropout": best_precision["dropout"],
        "best_precision_value": best_precision.get("val_precision"),
        # Best by recall
        "best_recall_gamma": best_recall["gamma"],
        "best_recall_alpha": best_recall["alpha"],
        "best_recall_dropout": best_recall["dropout"],
        "best_recall_value": best_recall.get("val_recall"),
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV with all results
    df_data = []
    for r in successful:
        pred_range = r.get("val_pred_range")
        row = {
            "gamma": r["gamma"],
            "alpha": r["alpha"],
            "dropout": r["dropout"],
            "val_auc": r["val_auc"],
            "val_precision": r.get("val_precision"),
            "val_recall": r.get("val_recall"),
            "pred_min": pred_range[0] if pred_range else None,
            "pred_max": pred_range[1] if pred_range else None,
            "epochs_run": r.get("epochs_run"),
            "duration_sec": r.get("duration_sec"),
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df = df.sort_values(["dropout", "gamma", "alpha"])
    df.to_csv(output_dir / "results_sweep.csv", index=False)

    # Save pivot tables for analysis - one per dropout level
    if len(df) > 0:
        dropout_values = df["dropout"].unique()
        for dropout in dropout_values:
            df_dropout = df[df["dropout"] == dropout]

            # AUC pivot
            auc_pivot = df_dropout.pivot(index="gamma", columns="alpha", values="val_auc")
            auc_pivot.to_csv(output_dir / f"pivot_auc_dropout_{dropout}.csv")

            # Precision pivot
            prec_pivot = df_dropout.pivot(index="gamma", columns="alpha", values="val_precision")
            prec_pivot.to_csv(output_dir / f"pivot_precision_dropout_{dropout}.csv")

            # Recall pivot
            recall_pivot = df_dropout.pivot(index="gamma", columns="alpha", values="val_recall")
            recall_pivot.to_csv(output_dir / f"pivot_recall_dropout_{dropout}.csv")

    # Markdown report
    md_lines = [
        f"# Loss Function Sweep - Tier {tier}",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Configuration",
        f"- Tier: {tier}",
        f"- Architecture: {BEST_ARCHITECTURES[tier]}",
        f"- Gamma values: {GAMMA_VALUES}",
        f"- Alpha values: {ALPHA_VALUES}",
        f"- Dropout values: {DROPOUT_VALUES}",
        f"\n## Best Results",
        f"\n### By AUC",
        f"- gamma={summary['best_gamma']}, alpha={summary['best_alpha']}, dropout={summary['best_dropout']}",
        f"- AUC: {summary['best_auc']:.4f}",
        f"\n### By Precision",
        f"- gamma={summary['best_precision_gamma']}, alpha={summary['best_precision_alpha']}, dropout={summary['best_precision_dropout']}",
        f"- Precision: {summary.get('best_precision_value', 'N/A')}",
        f"\n### By Recall",
        f"- gamma={summary['best_recall_gamma']}, alpha={summary['best_recall_alpha']}, dropout={summary['best_recall_dropout']}",
        f"- Recall: {summary.get('best_recall_value', 'N/A')}",
        f"\n## All Results",
        f"\n| Dropout | Gamma | Alpha | AUC | Precision | Recall | Pred Range |",
        f"|---------|-------|-------|-----|-----------|--------|------------|",
    ]

    for _, row in df.iterrows():
        pred_range = f"[{row['pred_min']:.3f}, {row['pred_max']:.3f}]" if row['pred_min'] is not None else "N/A"
        prec = f"{row['val_precision']:.3f}" if pd.notna(row['val_precision']) else "N/A"
        recall = f"{row['val_recall']:.3f}" if pd.notna(row['val_recall']) else "N/A"
        md_lines.append(
            f"| {row['dropout']} | {row['gamma']} | {row['alpha']} | {row['val_auc']:.4f} | {prec} | {recall} | {pred_range} |"
        )

    with open(output_dir / "results_sweep.md", "w") as f:
        f.write("\n".join(md_lines))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Loss function parameter sweep")
    parser.add_argument("--tier", required=True, choices=["a50", "a100", "a200"],
                        help="Feature tier")
    parser.add_argument("--gamma", type=float, nargs="+",
                        help="Specific gamma values (default: 0, 0.5, 1, 2)")
    parser.add_argument("--alpha", type=float, nargs="+",
                        help="Specific alpha values (default: 0.3-0.9)")
    parser.add_argument("--dropout", type=float, nargs="+",
                        help="Specific dropout values (default: 0.3, 0.4, 0.5)")
    parser.add_argument("--thermal-threshold", type=float, default=85.0,
                        help="Thermal pause threshold (C)")
    args = parser.parse_args()

    gamma_values = args.gamma if args.gamma else None
    alpha_values = args.alpha if args.alpha else None
    dropout_values = args.dropout if args.dropout else None

    run_loss_sweep(
        tier=args.tier,
        gamma_values=gamma_values,
        alpha_values=alpha_values,
        dropout_values=dropout_values,
        thermal_pause_threshold=args.thermal_threshold,
    )


if __name__ == "__main__":
    main()
