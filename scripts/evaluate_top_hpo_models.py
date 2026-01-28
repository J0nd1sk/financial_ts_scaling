#!/usr/bin/env python3
"""Evaluate top HPO models to capture precision/recall/pred_range.

HPO runs used verbose=False for speed, which skips detailed metrics.
This script re-trains top N models from each HPO run with verbose=True
to capture precision, recall, pred_range, and confusion matrix.

Usage:
    python scripts/evaluate_top_hpo_models.py --tier a50 --top-n 3
    python scripts/evaluate_top_hpo_models.py --tier a100 --top-n 3 --dry-run

Output:
    outputs/phase6c_{tier}/top_models_detailed_metrics.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
import numpy as np

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# Fixed hyperparameters
CONTEXT_LENGTH = 80
EPOCHS = 50  # Same as HPO

# Feature counts by tier
NUM_FEATURES_BY_TIER = {
    "a20": 25,
    "a50": 55,
    "a100": 105,
    "a200": 211,
}


def load_all_trials(tier: str, budget: str, horizon: int) -> list[dict]:
    """Load all trials from HPO run, merging data from all_trials.json and trial_metrics.csv.

    Args:
        tier: Feature tier (a50, a100)
        budget: Budget level (2M, 20M, 200M)
        horizon: Prediction horizon

    Returns:
        List of trial dicts with 'number', 'value', 'params', 'state'
    """
    budget_lower = budget.lower()

    # Try multiple directory patterns
    patterns = [
        f"hpo_{budget_lower}_{horizon}_{tier}",
        f"hpo_{budget_lower}_h{horizon}",
    ]

    trials = []
    for pattern in patterns:
        trials_file = PROJECT_ROOT / "outputs" / f"phase6c_{tier}" / pattern / "all_trials.json"
        if trials_file.exists():
            with open(trials_file) as f:
                trials = json.load(f)

            # Also load trial_metrics.csv to get params for forced trials
            metrics_file = PROJECT_ROOT / "outputs" / f"phase6c_{tier}" / pattern / "trial_metrics.csv"
            if metrics_file.exists():
                metrics_df = pd.read_csv(metrics_file)
                # Build a dict of trial_number -> params from CSV
                csv_params = {}
                for _, row in metrics_df.iterrows():
                    trial_num = int(row["trial"])
                    csv_params[trial_num] = {
                        "d_model": int(row["d_model"]),
                        "n_layers": int(row["n_layers"]),
                        "n_heads": int(row["n_heads"]),
                        "d_ff_ratio": int(row["d_ff_ratio"]),
                        "learning_rate": float(row["learning_rate"]),
                        "dropout": float(row["dropout"]),
                        "weight_decay": float(row["weight_decay"]),
                    }

                # Merge params into trials
                for trial in trials:
                    trial_num = trial["number"]
                    if not trial.get("params") and trial_num in csv_params:
                        trial["params"] = csv_params[trial_num]

            break

    return trials


def train_and_evaluate(
    params: dict,
    tier: str,
    horizon: int,
) -> dict:
    """Train model with given params and return detailed metrics.

    Args:
        params: Hyperparameter dict from HPO trial
        tier: Feature tier
        horizon: Prediction horizon

    Returns:
        Dict with all metrics including precision, recall, pred_range
    """
    # Compute d_ff
    d_ff = params["d_model"] * params.get("d_ff_ratio", 4)

    # Determine batch size based on model size
    d_model = params["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    # Load data
    data_path = PROJECT_ROOT / f"data/processed/v1/SPY_dataset_{tier}_combined.parquet"
    if not data_path.exists():
        return {"error": f"Data file not found: {data_path}"}

    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    num_features = NUM_FEATURES_BY_TIER.get(tier, 105)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    experiment_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=num_features,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        n_layers=params["n_layers"],
        d_ff=d_ff,
        dropout=params.get("dropout", 0.3),
        head_dropout=0.0,
    )

    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create temp directory for checkpoints
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=batch_size,
            learning_rate=params.get("learning_rate", 1e-4),
            weight_decay=params.get("weight_decay", 0.0),
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

        # Train with verbose=True to get all metrics
        result = trainer.train(verbose=True)

        return {
            "val_auc": result.get("val_auc"),
            "val_precision": result.get("val_precision"),
            "val_recall": result.get("val_recall"),
            "val_pred_range": result.get("val_pred_range"),
            "val_accuracy": result.get("val_accuracy"),
            "val_confusion": result.get("val_confusion"),
            "train_auc": result.get("train_auc"),
            "train_precision": result.get("train_precision"),
            "train_recall": result.get("train_recall"),
            "train_pred_range": result.get("train_pred_range"),
            "epochs_run": result.get("epochs_run", EPOCHS),
            "stopped_early": result.get("stopped_early", False),
            "params": params,
        }


def evaluate_top_models(
    tier: str,
    horizon: int = 1,
    top_n: int = 3,
    dry_run: bool = False,
) -> dict:
    """Evaluate top N models from each budget's HPO results.

    Args:
        tier: Feature tier (a50, a100)
        horizon: Prediction horizon
        top_n: Number of top models to evaluate per budget
        dry_run: If True, return dummy results without training

    Returns:
        Dict with evaluation results for all top models
    """
    budgets = ["2M", "20M", "200M"]
    all_results = {
        "tier": tier,
        "horizon": horizon,
        "top_n": top_n,
        "timestamp": datetime.now().isoformat(),
        "results_by_budget": {},
    }

    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Processing {tier} {budget}")
        print(f"{'='*60}")

        trials = load_all_trials(tier, budget, horizon)
        if not trials:
            print(f"  No trials found for {tier} {budget}")
            all_results["results_by_budget"][budget] = {"error": "No trials found"}
            continue

        # Filter to complete trials and sort by AUC
        complete_trials = [t for t in trials if "COMPLETE" in t.get("state", "")]
        sorted_trials = sorted(complete_trials, key=lambda t: t.get("value", 0), reverse=True)
        top_trials = sorted_trials[:top_n]

        print(f"  Found {len(complete_trials)} complete trials, evaluating top {len(top_trials)}")

        budget_results = []
        for i, trial in enumerate(top_trials):
            trial_num = trial["number"]
            original_auc = trial["value"]
            params = trial["params"]

            print(f"\n  Trial {trial_num} (rank {i+1}): original AUC = {original_auc:.4f}")
            print(f"    d_model={params.get('d_model')}, n_layers={params.get('n_layers')}, "
                  f"n_heads={params.get('n_heads')}, dropout={params.get('dropout', 0.3):.2f}")

            if dry_run:
                result = {
                    "trial_number": trial_num,
                    "original_auc": original_auc,
                    "val_auc": original_auc,
                    "val_precision": 0.65,
                    "val_recall": 0.10,
                    "val_pred_range": (0.4, 0.6),
                    "dry_run": True,
                    "params": params,
                }
            else:
                result = train_and_evaluate(params, tier, horizon)
                result["trial_number"] = trial_num
                result["original_auc"] = original_auc

            # Print key metrics
            auc = result.get('val_auc')
            precision = result.get('val_precision')
            recall = result.get('val_recall')
            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A"
            prec_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else "N/A"
            rec_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else "N/A"
            print(f"    Evaluated: AUC={auc_str}, Precision={prec_str}, Recall={rec_str}")

            pred_range = result.get("val_pred_range")
            if pred_range:
                print(f"    Pred range: [{pred_range[0]:.4f}, {pred_range[1]:.4f}]")

            budget_results.append(result)

        all_results["results_by_budget"][budget] = budget_results

    return all_results


def generate_summary_report(results: dict) -> str:
    """Generate markdown summary report.

    Args:
        results: Dict from evaluate_top_models

    Returns:
        Markdown formatted report string
    """
    lines = [
        "# Top HPO Models Detailed Evaluation",
        "",
        f"**Tier**: {results['tier']}",
        f"**Horizon**: {results['horizon']}",
        f"**Top N**: {results['top_n']}",
        f"**Timestamp**: {results.get('timestamp', 'N/A')}",
        "",
        "## Summary by Budget",
        "",
    ]

    for budget in ["2M", "20M", "200M"]:
        budget_data = results["results_by_budget"].get(budget, [])
        if isinstance(budget_data, dict) and "error" in budget_data:
            lines.append(f"### {budget}: {budget_data['error']}")
            lines.append("")
            continue

        lines.append(f"### {budget}")
        lines.append("")
        lines.append("| Rank | Trial | AUC | Precision | Recall | Pred Range |")
        lines.append("|------|-------|-----|-----------|--------|------------|")

        for i, model in enumerate(budget_data):
            auc = model.get("val_auc")
            precision = model.get("val_precision")
            recall = model.get("val_recall")
            pred_range = model.get("val_pred_range")

            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A"
            prec_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else "N/A"
            rec_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else "N/A"

            if pred_range and len(pred_range) == 2:
                range_str = f"[{pred_range[0]:.3f}, {pred_range[1]:.3f}]"
            else:
                range_str = "N/A"

            lines.append(f"| {i+1} | {model.get('trial_number', 'N/A')} | {auc_str} | {prec_str} | {rec_str} | {range_str} |")

        lines.append("")

    # Key findings
    lines.extend([
        "## Key Findings",
        "",
    ])

    # Compute averages
    all_recalls = []
    all_precisions = []
    zero_recall_count = 0
    total_models = 0

    for budget, budget_data in results["results_by_budget"].items():
        if isinstance(budget_data, list):
            for model in budget_data:
                total_models += 1
                recall = model.get("val_recall")
                precision = model.get("val_precision")
                if isinstance(recall, (int, float)):
                    all_recalls.append(recall)
                    if recall == 0:
                        zero_recall_count += 1
                if isinstance(precision, (int, float)):
                    all_precisions.append(precision)

    if all_recalls:
        avg_recall = sum(all_recalls) / len(all_recalls)
        lines.append(f"- **Average Recall**: {avg_recall:.4f}")
        lines.append(f"- **Zero Recall Models**: {zero_recall_count}/{total_models}")

    if all_precisions:
        avg_precision = sum(all_precisions) / len(all_precisions)
        lines.append(f"- **Average Precision**: {avg_precision:.4f}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate top HPO models")
    parser.add_argument("--tier", type=str, required=True, help="Feature tier (a50, a100)")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top models per budget")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual training")
    args = parser.parse_args()

    print(f"Evaluating top {args.top_n} models for tier={args.tier}, horizon={args.horizon}")

    results = evaluate_top_models(
        tier=args.tier,
        horizon=args.horizon,
        top_n=args.top_n,
        dry_run=args.dry_run,
    )

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / f"phase6c_{args.tier}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "top_models_detailed_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save markdown report
    report = generate_summary_report(results)
    md_path = output_dir / "top_models_detailed_metrics.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {md_path}")

    # Print report
    print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
