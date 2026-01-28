#!/usr/bin/env python3
"""Cross-Budget Validation Script.

Loads best hyperparameters from HPO runs and tests them across all budget levels
to validate whether optimal configs transfer across parameter budgets.

Usage:
    python scripts/validate_cross_budget.py --tier a100 --horizon 1 [--dry-run]

Output:
    outputs/phase6c_{tier}/cross_budget_validation.json
    outputs/phase6c_{tier}/cross_budget_validation.md
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

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# Fixed hyperparameters
CONTEXT_LENGTH = 80
EPOCHS = 50

# Feature counts by tier
NUM_FEATURES_BY_TIER = {
    "a20": 25,
    "a50": 55,
    "a100": 105,
    "a200": 211,
}

# Budget configurations (simplified for validation - not full search space)
BUDGET_CONFIGS = {
    "2M": {"max_d_model": 96, "typical_batch": 128},
    "20M": {"max_d_model": 192, "typical_batch": 64},
    "200M": {"max_d_model": 384, "typical_batch": 32},
}


def load_best_params(budget: str, output_dir: str, horizon: int, tier: str) -> dict | None:
    """Load best parameters from HPO results.

    Args:
        budget: Budget level (2M, 20M, 200M)
        output_dir: Base output directory path
        horizon: Prediction horizon
        tier: Feature tier (a20, a50, a100, a200)

    Returns:
        Dict with best_params and metadata, or None if not found
    """
    budget_lower = budget.lower()

    # Try multiple directory patterns (different naming conventions used)
    patterns = [
        f"hpo_{budget_lower}_{horizon}_{tier}",    # Current format: hpo_2m_1_a50
        f"hpo_{budget_lower}_h{horizon}",          # Legacy format: hpo_2m_h1
        f"hpo_{budget_lower}_h{horizon}_{tier}",   # Hybrid format
    ]

    for pattern in patterns:
        hpo_dir = Path(output_dir) / pattern
        params_file = hpo_dir / "best_params.json"
        if params_file.exists():
            with open(params_file) as f:
                return json.load(f)

    return None


def train_with_config(
    config: dict,
    budget: str,
    tier: str,
    horizon: int,
    dry_run: bool = False,
) -> dict:
    """Train a model with given config and return metrics.

    Args:
        config: Hyperparameter configuration dict
        budget: Budget level for batch size selection
        tier: Feature tier (a20, a50, a100, a200)
        horizon: Prediction horizon
        dry_run: If True, return dummy metrics without training

    Returns:
        Dict with val_auc, precision, recall, pred_range
    """
    if dry_run:
        return {
            "dry_run": True,
            "auc": 0.5,
            "precision": None,
            "recall": None,
            "pred_range": None,
            "config": config,
        }

    # Determine batch size based on model size
    d_model = config["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    # Compute d_ff
    d_ff = config["d_model"] * config.get("d_ff_ratio", 4)

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
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=d_ff,
        dropout=config.get("dropout", 0.3),
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
            learning_rate=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.0),
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

        result = trainer.train(verbose=False)

        return {
            "auc": result.get("val_auc", 0.5),
            "precision": result.get("val_precision"),
            "recall": result.get("val_recall"),
            "pred_range": result.get("val_pred_range"),
            "epochs_run": result.get("epochs_run", EPOCHS),
            "config": config,
        }


def run_cross_budget_validation(
    tier: str,
    horizon: int,
    output_dir: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Run cross-budget validation for all budget combinations.

    For each budget's best config, train models at all budget levels
    and collect metrics to create a comparison matrix.

    Args:
        tier: Feature tier
        horizon: Prediction horizon
        output_dir: Override output directory (for testing)
        dry_run: If True, skip actual training

    Returns:
        Dict with configs_found, matrix of results, and metadata
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / f"phase6c_{tier}")

    budgets = ["2M", "20M", "200M"]
    configs_found = []
    best_configs = {}

    # Load best params for each budget
    for budget in budgets:
        params = load_best_params(budget, output_dir, horizon, tier)
        if params is not None:
            configs_found.append(budget)
            best_configs[budget] = params["best_params"]

    if not configs_found:
        return {
            "tier": tier,
            "horizon": horizon,
            "configs_found": [],
            "matrix": {},
            "error": "No HPO results found",
            "timestamp": datetime.now().isoformat(),
        }

    # Build validation matrix
    matrix = {}

    for config_from in configs_found:
        config = best_configs[config_from]

        for train_on in budgets:
            key = f"{config_from}_config_on_{train_on}_budget"

            if dry_run:
                matrix[key] = {
                    "auc": 0.50 + 0.02 * (budgets.index(train_on)),
                    "precision": 0.6,
                    "recall": 0.5,
                    "config_source": config_from,
                    "budget_used": train_on,
                    "dry_run": True,
                }
            else:
                print(f"Training {config_from} config on {train_on} budget...")
                result = train_with_config(config, train_on, tier, horizon, dry_run=False)
                result["config_source"] = config_from
                result["budget_used"] = train_on
                matrix[key] = result

    return {
        "tier": tier,
        "horizon": horizon,
        "configs_found": configs_found,
        "matrix": matrix,
        "timestamp": datetime.now().isoformat(),
    }


def generate_markdown_report(results: dict) -> str:
    """Generate markdown report from validation results.

    Args:
        results: Dict from run_cross_budget_validation

    Returns:
        Markdown formatted report string
    """
    lines = [
        f"# Cross-Budget Validation Report",
        "",
        f"**Tier**: {results['tier']}",
        f"**Horizon**: {results['horizon']}",
        f"**Timestamp**: {results.get('timestamp', 'N/A')}",
        "",
    ]

    configs_found = results.get("configs_found", [])
    if not configs_found:
        lines.append("**Error**: No HPO results found")
        return "\n".join(lines)

    lines.append(f"**Configs Found**: {', '.join(configs_found)}")
    lines.append("")
    lines.append("## Validation Matrix")
    lines.append("")
    lines.append("Each cell shows AUC when using row's config on column's budget.")
    lines.append("")

    # Build table header
    budgets = ["2M", "20M", "200M"]
    lines.append("| Config \\ Budget | " + " | ".join(budgets) + " |")
    lines.append("|" + "---|" * (len(budgets) + 1))

    matrix = results.get("matrix", {})

    # Build table rows
    for config_from in configs_found:
        row = [f"**{config_from}**"]
        for budget in budgets:
            key = f"{config_from}_config_on_{budget}_budget"
            if key in matrix:
                auc = matrix[key].get("auc", "N/A")
                if isinstance(auc, float):
                    row.append(f"{auc:.4f}")
                else:
                    row.append(str(auc))
            else:
                row.append("--")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Analysis")
    lines.append("")

    # Find diagonal vs off-diagonal performance
    diag_values = []
    off_diag_values = []
    for config_from in configs_found:
        for budget in budgets:
            key = f"{config_from}_config_on_{budget}_budget"
            if key in matrix:
                auc = matrix[key].get("auc")
                if isinstance(auc, (int, float)):
                    if config_from == budget:
                        diag_values.append(auc)
                    else:
                        off_diag_values.append(auc)

    if diag_values and off_diag_values:
        diag_avg = sum(diag_values) / len(diag_values)
        off_diag_avg = sum(off_diag_values) / len(off_diag_values)
        lines.append(f"- **Diagonal average** (matched config/budget): {diag_avg:.4f}")
        lines.append(f"- **Off-diagonal average** (cross-budget): {off_diag_avg:.4f}")
        lines.append(f"- **Transfer gap**: {diag_avg - off_diag_avg:.4f}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Cross-budget HPO validation")
    parser.add_argument("--tier", type=str, default="a100", help="Feature tier")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual training")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Cross-Budget Validation: tier={args.tier}, horizon={args.horizon}")
    print("=" * 70)

    results = run_cross_budget_validation(
        tier=args.tier,
        horizon=args.horizon,
        dry_run=args.dry_run,
    )

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / f"phase6c_{args.tier}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "cross_budget_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    report = generate_markdown_report(results)
    with open(output_dir / "cross_budget_validation.md", "w") as f:
        f.write(report)

    print()
    print(report)
    print()
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
