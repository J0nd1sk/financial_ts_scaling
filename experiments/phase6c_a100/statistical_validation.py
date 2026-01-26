#!/usr/bin/env python3
"""
Phase 6C A100: Statistical Validation

Bootstrap CI analysis comparing a100 vs a50 and a100 vs a20.
10,000 bootstrap iterations for 95% confidence intervals.

Key metrics:
- AUC difference with 95% CI
- Statistical significance (CI excludes 0)
- Effect size estimation
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
from sklearn.metrics import roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# CONFIGURATION
# ============================================================================

N_BOOTSTRAP = 10000
ALPHA = 0.05  # 95% CI
RANDOM_SEED = 42

NUM_FEATURES = {"a20": 20, "a50": 50, "a100": 100}
DATA_PATHS = {
    "a20": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet",
    "a50": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet",
    "a100": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet",
}

RESULT_DIRS = {
    "a20": PROJECT_ROOT / "outputs/phase6a_final",  # Phase 6A experiments (different naming)
    "a50": PROJECT_ROOT / "outputs/phase6c",
    "a100": PROJECT_ROOT / "outputs/phase6c_a100",
}

# Naming conventions differ by tier:
# - a20 (Phase 6A): phase6a_2m_h1, phase6a_20m_h1, etc.
# - a50/a100 (Phase 6C): s1_01_2m_h1, s1_02_20m_h1, etc. (sequential numbering)

# Experiments to compare
BUDGETS = ["2M", "20M", "200M"]
HORIZONS = [1, 2, 3, 5]
ARCHITECTURES = {
    "2M": (64, 4, 4, 256, 128),    # d_model, n_layers, n_heads, d_ff, batch_size
    "20M": (128, 6, 8, 512, 64),
    "200M": (256, 8, 8, 1024, 32),
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a100"


# ============================================================================
# BOOTSTRAP FUNCTIONS
# ============================================================================

def bootstrap_auc_ci(labels, preds, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=RANDOM_SEED):
    """Compute bootstrap confidence interval for AUC."""
    np.random.seed(seed)
    n = len(labels)

    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            auc = roc_auc_score(labels[idx], preds[idx])
            aucs.append(auc)
        except ValueError:
            continue

    aucs = np.array(aucs)
    lower = np.percentile(aucs, 100 * alpha / 2)
    upper = np.percentile(aucs, 100 * (1 - alpha / 2))
    mean = np.mean(aucs)

    return {"mean": mean, "lower": lower, "upper": upper, "std": np.std(aucs)}


def bootstrap_auc_difference_ci(labels, preds1, preds2, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=RANDOM_SEED):
    """Compute bootstrap CI for AUC difference (preds1 - preds2)."""
    np.random.seed(seed)
    n = len(labels)

    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        try:
            auc1 = roc_auc_score(labels[idx], preds1[idx])
            auc2 = roc_auc_score(labels[idx], preds2[idx])
            diffs.append(auc1 - auc2)
        except ValueError:
            continue

    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    mean = np.mean(diffs)

    # Significant if CI excludes 0
    significant = (lower > 0) or (upper < 0)

    return {
        "mean_diff": mean,
        "lower": lower,
        "upper": upper,
        "std": np.std(diffs),
        "significant": significant,
    }


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_model_predictions(tier, budget, horizon, device):
    """Load model and get validation predictions."""
    # Construct experiment name (naming convention differs by tier)
    if tier == "a20":
        # Phase 6A naming: phase6a_2m_h1, phase6a_20m_h1, etc.
        exp_name = f"phase6a_{budget.lower()}_h{horizon}"
    else:
        # Phase 6C naming: s1_01_2m_h1, s1_02_20m_h1, etc. (sequential numbering)
        budget_idx = BUDGETS.index(budget)
        horizon_idx = HORIZONS.index(horizon)
        exp_num = budget_idx + horizon_idx * 3 + 1
        exp_name = f"s1_{exp_num:02d}_{budget.lower()}_h{horizon}"

    checkpoint_path = RESULT_DIRS[tier] / exp_name / "best_checkpoint.pt"
    data_path = DATA_PATHS[tier]

    if not checkpoint_path.exists():
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None, None

    if not data_path.exists():
        print(f"  Data not found: {data_path}")
        return None, None

    # Load data
    df = pd.read_parquet(data_path)

    # Get architecture
    d_model, n_layers, n_heads, d_ff, batch_size = ARCHITECTURES[budget]

    # Create config
    experiment_config = ExperimentConfig(
        data_path=str(data_path),
        task="threshold_1pct",
        timescale="daily",
        context_length=80,
        horizon=horizon,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES[tier],
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

    # Determine high_prices column
    if "High" in df.columns:
        high_prices = df["High"].values
    else:
        # For a20 which might not have OHLCV
        high_prices = None

    # Create trainer
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=1e-4,
        epochs=1,
        device=device,
        checkpoint_dir=RESULT_DIRS[tier] / exp_name,
        split_indices=split_indices,
        use_revin=True,
        high_prices=high_prices,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in trainer.val_dataloader:
            batch_x = batch_x.to(device)
            preds = trainer.model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_labels), np.array(all_preds)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 6C A100: Statistical Validation")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Confidence level: {100*(1-ALPHA)}%")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    results = {
        "comparisons": [],
        "config": {
            "n_bootstrap": N_BOOTSTRAP,
            "alpha": ALPHA,
            "seed": RANDOM_SEED,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Compare a100 vs a50
    print("\n" + "=" * 70)
    print("COMPARISON: a100 vs a50")
    print("=" * 70)

    for budget in BUDGETS:
        for horizon in HORIZONS:
            print(f"\n{budget} H{horizon}:")

            labels_a100, preds_a100 = get_model_predictions("a100", budget, horizon, device)
            labels_a50, preds_a50 = get_model_predictions("a50", budget, horizon, device)

            if labels_a100 is None or labels_a50 is None:
                print("  Skipping (missing data)")
                continue

            # Ensure same validation set
            if len(labels_a100) != len(labels_a50):
                print(f"  Warning: Different val sizes ({len(labels_a100)} vs {len(labels_a50)})")
                continue

            # Bootstrap CI for difference
            diff_ci = bootstrap_auc_difference_ci(labels_a100, preds_a100, preds_a50)

            # Individual CIs
            ci_a100 = bootstrap_auc_ci(labels_a100, preds_a100)
            ci_a50 = bootstrap_auc_ci(labels_a50, preds_a50)

            print(f"  a100 AUC: {ci_a100['mean']:.4f} [{ci_a100['lower']:.4f}, {ci_a100['upper']:.4f}]")
            print(f"  a50  AUC: {ci_a50['mean']:.4f} [{ci_a50['lower']:.4f}, {ci_a50['upper']:.4f}]")
            print(f"  Diff: {diff_ci['mean_diff']:+.4f} [{diff_ci['lower']:+.4f}, {diff_ci['upper']:+.4f}]")
            print(f"  Significant: {'YES' if diff_ci['significant'] else 'NO'}")

            results["comparisons"].append({
                "comparison": "a100_vs_a50",
                "budget": budget,
                "horizon": horizon,
                "a100_auc": ci_a100,
                "a50_auc": ci_a50,
                "difference": diff_ci,
            })

    # Compare a100 vs a20
    print("\n" + "=" * 70)
    print("COMPARISON: a100 vs a20")
    print("=" * 70)

    for budget in BUDGETS:
        for horizon in HORIZONS:
            print(f"\n{budget} H{horizon}:")

            labels_a100, preds_a100 = get_model_predictions("a100", budget, horizon, device)
            labels_a20, preds_a20 = get_model_predictions("a20", budget, horizon, device)

            if labels_a100 is None or labels_a20 is None:
                print("  Skipping (missing data)")
                continue

            if len(labels_a100) != len(labels_a20):
                print(f"  Warning: Different val sizes ({len(labels_a100)} vs {len(labels_a20)})")
                continue

            diff_ci = bootstrap_auc_difference_ci(labels_a100, preds_a100, preds_a20)
            ci_a100 = bootstrap_auc_ci(labels_a100, preds_a100)
            ci_a20 = bootstrap_auc_ci(labels_a20, preds_a20)

            print(f"  a100 AUC: {ci_a100['mean']:.4f} [{ci_a100['lower']:.4f}, {ci_a100['upper']:.4f}]")
            print(f"  a20  AUC: {ci_a20['mean']:.4f} [{ci_a20['lower']:.4f}, {ci_a20['upper']:.4f}]")
            print(f"  Diff: {diff_ci['mean_diff']:+.4f} [{diff_ci['lower']:+.4f}, {diff_ci['upper']:+.4f}]")
            print(f"  Significant: {'YES' if diff_ci['significant'] else 'NO'}")

            results["comparisons"].append({
                "comparison": "a100_vs_a20",
                "budget": budget,
                "horizon": horizon,
                "a100_auc": ci_a100,
                "a20_auc": ci_a20,
                "difference": diff_ci,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    a100_vs_a50_sig = sum(1 for r in results["comparisons"]
                         if r["comparison"] == "a100_vs_a50" and r["difference"]["significant"])
    a100_vs_a50_total = sum(1 for r in results["comparisons"]
                           if r["comparison"] == "a100_vs_a50")

    a100_vs_a20_sig = sum(1 for r in results["comparisons"]
                         if r["comparison"] == "a100_vs_a20" and r["difference"]["significant"])
    a100_vs_a20_total = sum(1 for r in results["comparisons"]
                           if r["comparison"] == "a100_vs_a20")

    print(f"\na100 vs a50: {a100_vs_a50_sig}/{a100_vs_a50_total} significant differences")
    print(f"a100 vs a20: {a100_vs_a20_sig}/{a100_vs_a20_total} significant differences")

    # Save results
    output_path = OUTPUT_DIR / "statistical_validation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
