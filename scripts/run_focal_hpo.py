#!/usr/bin/env python3
"""Focal Loss HPO experiments for precision optimization.

Runs HPO with Focal Loss instead of BCE to test whether it improves
precision/recall tradeoffs by better handling class imbalance.

Focal Loss down-weights easy examples and focuses learning on hard examples,
which should produce wider probability distributions (less clustered around 0.5)
and better discrimination between positive and negative cases.

Usage:
    ./venv/bin/python scripts/run_focal_hpo.py --tier a50
    ./venv/bin/python scripts/run_focal_hpo.py --tier a50 --trials 25
    ./venv/bin/python scripts/run_focal_hpo.py --tier a50 --gamma 3.0 --alpha 0.3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Project root for imports and data paths
PROJECT_ROOT = Path(__file__).parent.parent

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    import pandas as pd
    from src.data.dataset import SimpleSplitter
    from src.models.patchtst import PatchTSTConfig
    from src.config.experiment import ExperimentConfig
    from src.training.trainer import Trainer
    from src.training.losses import FocalLoss


# Default Focal Loss parameters (from original paper)
DEFAULT_GAMMA = 2.0  # Focusing parameter - higher = more focus on hard examples
DEFAULT_ALPHA = 0.25  # Positive class weight (our base rate is ~18%, so slightly upweight)


def get_thermal_reading() -> float | None:
    """Get current CPU temperature using powermetrics (requires prior sudo auth).

    Returns:
        Temperature in Celsius, or None if unable to read.
    """
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


def train_single_trial(
    trial_num: int,
    config: dict,
    tier: str,
    output_dir: Path,
    gamma: float,
    alpha: float,
    budget: str = "20M",
    horizon: int = 1,
) -> dict:
    """Train a single trial with Focal Loss and return full metrics.

    Args:
        trial_num: Trial number for tracking
        config: Hyperparameter dictionary
        tier: Feature tier (e.g., "a50")
        output_dir: Directory to save results
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
        budget: Parameter budget (default "20M")
        horizon: Prediction horizon in days (default 1)

    Returns:
        Dictionary with all metrics and metadata
    """
    import tempfile

    import pandas as pd
    import torch

    # Lazy imports to avoid import errors in tests
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.dataset import SimpleSplitter
    from src.models.patchtst import PatchTSTConfig
    from src.config.experiment import ExperimentConfig
    from src.training.trainer import Trainer
    from src.training.losses import FocalLoss

    # Feature counts by tier
    NUM_FEATURES_BY_TIER = {
        "a20": 25,
        "a50": 55,
        "a100": 105,
        "a200": 211,
    }

    # Fixed parameters (ablation-validated)
    CONTEXT_LENGTH = 80
    EPOCHS = 50

    start_time = time.time()

    # Load dataset
    data_path = PROJECT_ROOT / f"data/processed/v1/SPY_dataset_{tier}_combined.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    num_features = NUM_FEATURES_BY_TIER.get(tier, 105)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Determine batch size based on model size
    d_model = config["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    # Build experiment config
    exp_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Build model config
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

    # Create splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create Focal Loss criterion
    criterion = FocalLoss(gamma=gamma, alpha=alpha)

    # Create temp directory for checkpoints
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
            criterion=criterion,
        )

        # Train with verbose=True to get all metrics
        result = trainer.train(verbose=True)

    duration = time.time() - start_time

    # Build result dict
    metrics = {
        "trial": trial_num,
        "tier": tier,
        "budget": budget,
        "horizon": horizon,
        "loss_type": "focal",
        "focal_gamma": gamma,
        "focal_alpha": alpha,
        "params": config,
        "duration_sec": duration,
        "epochs_run": result.get("epochs_run", EPOCHS),
        "stopped_early": result.get("stopped_early", False),
        # Validation metrics
        "val_auc": result.get("val_auc"),
        "val_accuracy": result.get("val_accuracy"),
        "val_precision": result.get("val_precision"),
        "val_recall": result.get("val_recall"),
        "val_pred_range": result.get("val_pred_range"),
        "val_confusion": result.get("val_confusion"),
        # Training metrics
        "train_auc": result.get("train_auc"),
        "train_precision": result.get("train_precision"),
        "train_recall": result.get("train_recall"),
        "train_pred_range": result.get("train_pred_range"),
        "timestamp": datetime.now().isoformat(),
    }

    return metrics


def get_focal_hpo_configs(tier: str) -> dict:
    """Get hyperparameter configs for Focal Loss HPO.

    Uses best-performing configs from BCE HPO baseline as starting points.

    Args:
        tier: Feature tier ("a50" or "a100")

    Returns:
        Dictionary of config_name -> config dict
    """
    # a50 configs based on best BCE results
    # Best BCE: d_model=128, n_layers=6, dropout=0.3, lr=5e-5, wd=1e-4
    A50_CONFIGS = {
        # Baseline architecture with different training HPs
        "FL01": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL02": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL03": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL04": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 7e-5, "weight_decay": 1e-4},
        "FL05": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 1e-4, "weight_decay": 1e-4},

        # Architecture variants (proven good with BCE)
        "FL06": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},  # D2 style
        "FL07": {"d_model": 160, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},  # E4 style
        "FL08": {"d_model": 128, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 1e-4},

        # Higher regularization (Focal may benefit from less)
        "FL09": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.5, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL10": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 3e-4},

        # Lower regularization (Focal may benefit from more)
        "FL11": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.2, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL12": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-5},

        # Combined variations
        "FL13": {"d_model": 160, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL14": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 1e-4},
        "FL15": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 3e-4},

        # Additional exploration
        "FL16": {"d_model": 96, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL17": {"d_model": 160, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL18": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.25, "learning_rate": 7e-5, "weight_decay": 1e-4},
        "FL19": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.45, "learning_rate": 5e-5, "weight_decay": 5e-4},
        "FL20": {"d_model": 192, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},

        # Deeper models
        "FL21": {"d_model": 128, "n_layers": 8, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL22": {"d_model": 128, "n_layers": 8, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 3e-4},

        # Shallower with higher capacity
        "FL23": {"d_model": 192, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
        "FL24": {"d_model": 160, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 7e-5, "weight_decay": 1e-4},
        "FL25": {"d_model": 128, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 4,
                 "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
    }

    # a100 configs based on best BCE results
    # Best BCE: d_model=64, n_layers=4, dropout=0.7, lr=5e-4, wd=1e-3
    A100_CONFIGS = {
        "FL01": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2,
                 "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 1e-3},
        "FL02": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2,
                 "dropout": 0.6, "learning_rate": 5e-4, "weight_decay": 1e-3},
        "FL03": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2,
                 "dropout": 0.5, "learning_rate": 5e-4, "weight_decay": 1e-3},
        "FL04": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2,
                 "dropout": 0.7, "learning_rate": 3e-4, "weight_decay": 1e-3},
        "FL05": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2,
                 "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 2e-3},
        # ... additional configs as needed
    }

    if tier == "a50":
        return A50_CONFIGS
    elif tier == "a100":
        return A100_CONFIGS
    else:
        raise ValueError(f"Unknown tier: {tier}. Must be 'a50' or 'a100'")


def run_focal_hpo(
    tier: str,
    trial_names: list[str] | None = None,
    output_base: Path | None = None,
    thermal_pause_threshold: float = 85.0,
    gamma: float = DEFAULT_GAMMA,
    alpha: float = DEFAULT_ALPHA,
    n_trials: int | None = None,
) -> dict:
    """Run Focal Loss HPO trials for a tier.

    Args:
        tier: Feature tier ("a50" or "a100")
        trial_names: Optional list of specific trials to run (e.g., ["FL01", "FL02"])
        output_base: Base output directory
        thermal_pause_threshold: Pause if temperature exceeds this (Celsius)
        gamma: Focal loss gamma parameter (default 2.0)
        alpha: Focal loss alpha parameter (default 0.25)
        n_trials: Limit number of trials (runs first N if specified)

    Returns:
        Dictionary with all results and summary
    """
    configs = get_focal_hpo_configs(tier)

    # Filter to specific trials if requested
    if trial_names:
        configs = {k: v for k, v in configs.items() if k in trial_names}
        if not configs:
            available = list(get_focal_hpo_configs(tier).keys())
            raise ValueError(f"No matching trials found. Available: {available}")

    # Limit number of trials if specified
    if n_trials is not None and n_trials < len(configs):
        config_items = list(configs.items())[:n_trials]
        configs = dict(config_items)

    # Setup output directory
    if output_base is None:
        output_base = PROJECT_ROOT / f"outputs/phase6c_{tier}/focal_loss_hpo"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Focal Loss HPO - Tier {tier}")
    print(f"{'='*60}")
    print(f"Loss: Focal (gamma={gamma}, alpha={alpha})")
    print(f"Total trials: {len(configs)}")
    print(f"Output: {output_base}")
    print(f"Thermal threshold: {thermal_pause_threshold}C")
    print(f"{'='*60}\n")

    all_results = []
    results_file = output_base / "all_results.json"

    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n[{i}/{len(configs)}] Running trial {name}...")
        print(f"  Config: {config}")

        # Thermal check between trials
        temp = get_thermal_reading()
        if temp is not None:
            print(f"  Temperature: {temp:.1f}C")
            if temp > thermal_pause_threshold:
                print(f"  Warning: Temperature above {thermal_pause_threshold}C, pausing 60s...")
                time.sleep(60)

        try:
            result = train_single_trial(
                trial_num=i,
                config=config,
                tier=tier,
                output_dir=output_base,
                gamma=gamma,
                alpha=alpha,
            )
            result["config_name"] = name
            all_results.append(result)

            # Report key metrics
            print(f"  Done: AUC: {result['val_auc']:.4f}")
            prec_str = f"{result['val_precision']:.4f}" if result.get('val_precision') else 'N/A'
            recall_str = f"{result['val_recall']:.4f}" if result.get('val_recall') else 'N/A'
            print(f"    Precision: {prec_str}")
            print(f"    Recall: {recall_str}")
            print(f"    Pred range: {result['val_pred_range']}")
            print(f"    Duration: {result['duration_sec']:.1f}s")

        except Exception as e:
            print(f"  Failed: {e}")
            all_results.append({
                "config_name": name,
                "tier": tier,
                "params": config,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        # Incremental save after each trial
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(all_results, tier, output_base, gamma, alpha)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Completed: {summary['completed']}/{summary['total']}")
    if summary['completed'] > 0:
        print(f"Best AUC: {summary['best_auc']:.4f} ({summary['best_trial']})")
    print(f"Results saved to: {output_base}")
    print(f"{'='*60}\n")

    return {"results": all_results, "summary": summary}


def generate_summary(
    results: list[dict],
    tier: str,
    output_dir: Path,
    gamma: float,
    alpha: float,
) -> dict:
    """Generate summary files from results.

    Args:
        results: List of result dictionaries
        tier: Feature tier
        output_dir: Output directory
        gamma: Focal loss gamma used
        alpha: Focal loss alpha used

    Returns:
        Summary dictionary
    """
    import pandas as pd

    # Filter successful results
    successful = [r for r in results if "val_auc" in r and r["val_auc"] is not None]

    if not successful:
        return {"completed": 0, "total": len(results), "best_auc": 0, "best_trial": "N/A"}

    # Find best
    best = max(successful, key=lambda x: x["val_auc"])

    summary = {
        "tier": tier,
        "loss_type": "focal",
        "focal_gamma": gamma,
        "focal_alpha": alpha,
        "total": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "best_trial": best["config_name"],
        "best_auc": best["val_auc"],
        "best_precision": best.get("val_precision"),
        "best_recall": best.get("val_recall"),
        "best_params": best["params"],
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    df_data = []
    for r in successful:
        row = {
            "config_name": r["config_name"],
            "val_auc": r["val_auc"],
            "val_precision": r.get("val_precision"),
            "val_recall": r.get("val_recall"),
            "pred_min": r["val_pred_range"][0] if r.get("val_pred_range") else None,
            "pred_max": r["val_pred_range"][1] if r.get("val_pred_range") else None,
            "epochs_run": r.get("epochs_run"),
            "duration_sec": r.get("duration_sec"),
            **r["params"],
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df = df.sort_values("val_auc", ascending=False)
    df.to_csv(output_dir / "results_summary.csv", index=False)

    # Save markdown report
    md_lines = [
        f"# Focal Loss HPO Results - Tier {tier}",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Configuration",
        f"- Loss: Focal (gamma={gamma}, alpha={alpha})",
        f"- Tier: {tier}",
        f"\n## Summary",
        f"- Total trials: {summary['total']}",
        f"- Completed: {summary['completed']}",
        f"- Failed: {summary['failed']}",
        f"- Best AUC: {summary['best_auc']:.4f} ({summary['best_trial']})",
        f"- Best Precision: {summary.get('best_precision', 'N/A')}",
        f"- Best Recall: {summary.get('best_recall', 'N/A')}",
        f"\n## Best Config",
        f"```json",
        json.dumps(summary['best_params'], indent=2),
        f"```",
        f"\n## All Results (sorted by AUC)",
        f"\n| Config | AUC | Precision | Recall | Pred Range |",
        f"|--------|-----|-----------|--------|------------|",
    ]

    for _, row in df.iterrows():
        pred_range = f"[{row['pred_min']:.3f}, {row['pred_max']:.3f}]" if row['pred_min'] is not None else "N/A"
        precision_str = f"{row['val_precision']:.3f}" if row['val_precision'] is not None else "N/A"
        recall_str = f"{row['val_recall']:.3f}" if row['val_recall'] is not None else "N/A"
        md_lines.append(
            f"| {row['config_name']} | {row['val_auc']:.4f} | "
            f"{precision_str} | {recall_str} | {pred_range} |"
        )

    with open(output_dir / "results_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Focal Loss HPO trials")
    parser.add_argument("--tier", required=True, choices=["a50", "a100"],
                        help="Feature tier")
    parser.add_argument("--trials", type=str,
                        help="Comma-separated list of specific trials (e.g., FL01,FL02)")
    parser.add_argument("--n-trials", type=int,
                        help="Limit to first N trials")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help=f"Focal loss gamma parameter (default {DEFAULT_GAMMA})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help=f"Focal loss alpha parameter (default {DEFAULT_ALPHA})")
    parser.add_argument("--thermal-threshold", type=float, default=85.0,
                        help="Thermal pause threshold (C)")
    args = parser.parse_args()

    trial_names = None
    if args.trials:
        trial_names = [t.strip() for t in args.trials.split(",")]

    run_focal_hpo(
        tier=args.tier,
        trial_names=trial_names,
        thermal_pause_threshold=args.thermal_threshold,
        gamma=args.gamma,
        alpha=args.alpha,
        n_trials=args.n_trials,
    )


if __name__ == "__main__":
    main()
