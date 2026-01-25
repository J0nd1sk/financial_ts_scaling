#!/usr/bin/env python3
"""
Statistical Validation of a20 vs a50 AUC Differences.

This script performs rigorous statistical testing to determine if observed
AUC differences between a20 and a50 feature tiers are statistically significant.

Tests performed:
1. Bootstrap confidence intervals for AUC differences (10,000 iterations)
2. 2-way ANOVA for tier x budget interaction effect

Usage:
    python experiments/phase6c/statistical_validation.py

Output:
    outputs/phase6c/statistical_validation_results.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchtst import PatchTST, PatchTSTConfig
from src.data.dataset import FinancialDataset, SimpleSplitter

# ============================================================================
# CONFIGURATION
# ============================================================================

BUDGETS = ["2M", "20M", "200M"]
HORIZONS = [1, 2, 3, 5]
BOOTSTRAP_ITERATIONS = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# Architecture configs for each budget
ARCHITECTURES = {
    "2M": {"d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 256},
    "20M": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 512},
    "200M": {"d_model": 256, "n_layers": 8, "n_heads": 16, "d_ff": 1024},
}

# Paths
A20_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "phase6a_final"
A50_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "phase6c"

A20_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a20.parquet"
A50_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a50_combined.parquet"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "phase6c" / "statistical_validation_results.json"


# ============================================================================
# MODEL LOADING
# ============================================================================


def get_a20_checkpoint_path(budget: str, horizon: int) -> Path:
    """Get path to a20 (Phase 6A) checkpoint."""
    return A20_CHECKPOINT_DIR / f"phase6a_{budget.lower()}_h{horizon}" / "best_checkpoint.pt"


def get_a50_checkpoint_path(budget: str, horizon: int) -> Path:
    """Get path to a50 (Phase 6C) checkpoint."""
    if horizon == 1:
        # Stage 1 naming: s1_01_2m_h1_a50, s1_02_20m_h1_a50, s1_03_200m_h1_a50
        idx = {"2M": "01", "20M": "02", "200M": "03"}[budget]
        return A50_CHECKPOINT_DIR / f"s1_{idx}_{budget.lower()}_h1_a50" / "best_checkpoint.pt"
    else:
        # Stage 2 naming: s2_horizon_{budget}_h{horizon}_a50
        return A50_CHECKPOINT_DIR / f"s2_horizon_{budget.lower()}_h{horizon}_a50" / "best_checkpoint.pt"


def get_num_features_from_checkpoint(checkpoint_path: Path) -> int:
    """Infer num_features from checkpoint revin.gamma shape."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint and "revin.gamma" in checkpoint["model_state_dict"]:
        return checkpoint["model_state_dict"]["revin.gamma"].shape[0]
    raise ValueError(f"Cannot infer num_features from checkpoint: {checkpoint_path}")


def load_model(checkpoint_path: Path, budget: str, num_features: int = None, use_revin: bool = True) -> PatchTST:
    """Load a trained PatchTST model from checkpoint.

    If num_features is None, infers from checkpoint.
    """
    # Infer num_features if not provided
    if num_features is None:
        num_features = get_num_features_from_checkpoint(checkpoint_path)

    arch = ARCHITECTURES[budget]
    config = PatchTSTConfig(
        num_features=num_features,
        context_length=80,
        patch_length=16,
        stride=8,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=0.5,
        head_dropout=0.0,
    )

    model = PatchTST(config, use_revin=use_revin)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def get_predictions(model: PatchTST, dataloader: torch.utils.data.DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions and labels for a dataset."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


# ============================================================================
# BOOTSTRAP CI
# ============================================================================


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_pred_a20: np.ndarray,
    y_pred_a50: np.ndarray,
    n_iterations: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap CI for AUC difference (a50 - a20).

    Returns:
        dict with point_estimate, ci_lower, ci_upper, p_value, significant
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    # Compute point estimate
    try:
        auc_a20 = roc_auc_score(y_true, y_pred_a20)
        auc_a50 = roc_auc_score(y_true, y_pred_a50)
        point_estimate = auc_a50 - auc_a20
    except ValueError:
        return {
            "point_estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "significant": None,
            "auc_a20": None,
            "auc_a50": None,
        }

    # Bootstrap
    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        try:
            boot_auc_a20 = roc_auc_score(y_true[idx], y_pred_a20[idx])
            boot_auc_a50 = roc_auc_score(y_true[idx], y_pred_a50[idx])
            diffs.append(boot_auc_a50 - boot_auc_a20)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    diffs = np.array(diffs)

    if len(diffs) < 100:
        return {
            "point_estimate": point_estimate,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
            "significant": None,
            "auc_a20": auc_a20,
            "auc_a50": auc_a50,
            "note": "Insufficient valid bootstrap samples",
        }

    alpha = 1 - ci
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Approximate two-tailed p-value: proportion of samples with opposite sign
    if point_estimate >= 0:
        p_value = float(np.mean(diffs < 0) * 2)
    else:
        p_value = float(np.mean(diffs > 0) * 2)
    p_value = min(p_value, 1.0)

    # CI includes 0 -> not significant
    significant = not (ci_lower <= 0 <= ci_upper)

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": significant,
        "auc_a20": float(auc_a20),
        "auc_a50": float(auc_a50),
        "n_bootstrap_samples": len(diffs),
    }


# ============================================================================
# ANOVA TEST
# ============================================================================


def perform_anova(auc_data: list[dict]) -> dict:
    """
    Perform 2-way ANOVA to test for tier x budget interaction effect.

    Args:
        auc_data: List of dicts with keys: tier, budget, horizon, auc

    Returns:
        dict with F-statistics and p-values for main effects and interaction
    """
    # Convert to DataFrame
    df = pd.DataFrame(auc_data)

    # Focus on H1 (where the "interaction" claim was made)
    df_h1 = df[df["horizon"] == 1].copy()

    if len(df_h1) < 6:  # Need at least 2 tiers x 3 budgets
        return {"error": "Insufficient data for ANOVA"}

    # Simple 2-way ANOVA using statsmodels
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        # Fit model
        model = ols("auc ~ C(tier) * C(budget)", data=df_h1).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Extract results
        results = {
            "tier_F": float(anova_table.loc["C(tier)", "F"]),
            "tier_p": float(anova_table.loc["C(tier)", "PR(>F)"]),
            "budget_F": float(anova_table.loc["C(budget)", "F"]),
            "budget_p": float(anova_table.loc["C(budget)", "PR(>F)"]),
            "interaction_F": float(anova_table.loc["C(tier):C(budget)", "F"]),
            "interaction_p": float(anova_table.loc["C(tier):C(budget)", "PR(>F)"]),
            "interaction_significant": bool(anova_table.loc["C(tier):C(budget)", "PR(>F)"] < 0.05),
        }

        return results

    except ImportError:
        # Fallback: simple F-test without statsmodels
        # Group by tier x budget and compute means
        grouped = df_h1.groupby(["tier", "budget"])["auc"].mean().reset_index()

        # Get variance between groups vs within groups
        overall_mean = df_h1["auc"].mean()
        ss_total = ((df_h1["auc"] - overall_mean) ** 2).sum()

        return {
            "error": "statsmodels not available - manual ANOVA not implemented",
            "overall_mean_auc": float(overall_mean),
            "ss_total": float(ss_total),
        }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def run_bootstrap_analysis(device: str = "mps") -> tuple[list[dict], list[dict]]:
    """
    Run bootstrap CI analysis for all budget/horizon combinations.

    Returns:
        bootstrap_results: List of bootstrap CI results
        auc_data: List of AUC values for ANOVA
    """
    print("\n" + "=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print(f"Iterations: {BOOTSTRAP_ITERATIONS}, CI: {CONFIDENCE_LEVEL * 100}%")
    print("=" * 70)

    bootstrap_results = []
    auc_data = []

    # Load datasets
    print("\nLoading datasets...")
    df_a20 = pd.read_parquet(A20_DATA_PATH)
    df_a50 = pd.read_parquet(A50_DATA_PATH)

    # Feature columns = all numeric columns except Date
    # Note: OHLCV are INCLUDED as features (this matches how models were trained)
    a20_feature_cols = [c for c in df_a20.columns if c != "Date"]
    a50_feature_cols = [c for c in df_a50.columns if c != "Date"]

    # Filter to numeric only
    a20_feature_cols = [c for c in a20_feature_cols if df_a20[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    a50_feature_cols = [c for c in a50_feature_cols if df_a50[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    num_features_a20 = len(a20_feature_cols)
    num_features_a50 = len(a50_feature_cols)

    print(f"  a20: {len(df_a20)} rows, {num_features_a20} features (OHLCV + indicators)")
    print(f"  a50: {len(df_a50)} rows, {num_features_a50} features (OHLCV + indicators)")

    for horizon in HORIZONS:
        for budget in BUDGETS:
            print(f"\n--- {budget} / H{horizon} ---")

            # Get checkpoint paths
            a20_checkpoint = get_a20_checkpoint_path(budget, horizon)
            a50_checkpoint = get_a50_checkpoint_path(budget, horizon)

            if not a20_checkpoint.exists():
                print(f"  SKIP: a20 checkpoint not found: {a20_checkpoint}")
                continue
            if not a50_checkpoint.exists():
                print(f"  SKIP: a50 checkpoint not found: {a50_checkpoint}")
                continue

            # Load models - infer num_features from checkpoint
            print(f"  Loading models...")
            try:
                model_a20 = load_model(a20_checkpoint, budget)  # Infers features from checkpoint
                model_a50 = load_model(a50_checkpoint, budget)  # Infers features from checkpoint
            except Exception as e:
                print(f"  ERROR loading model: {e}")
                continue

            # Create datasets and dataloaders for validation set
            # a20 dataset
            splitter_a20 = SimpleSplitter(
                dates=df_a20["Date"],
                context_length=80,
                horizon=horizon,
                val_start="2023-01-01",
                test_start="2025-01-01",
            )
            split_a20 = splitter_a20.split()

            dataset_a20 = FinancialDataset(
                features_df=df_a20,
                close_prices=df_a20["Close"].values,
                context_length=80,
                horizon=horizon,
                threshold=0.01,
                feature_columns=a20_feature_cols,
                high_prices=df_a20["High"].values,
            )

            val_indices_a20 = split_a20.val_indices
            val_dataset_a20 = torch.utils.data.Subset(dataset_a20, val_indices_a20)
            val_loader_a20 = torch.utils.data.DataLoader(val_dataset_a20, batch_size=64, shuffle=False)

            # a50 dataset
            splitter_a50 = SimpleSplitter(
                dates=df_a50["Date"],
                context_length=80,
                horizon=horizon,
                val_start="2023-01-01",
                test_start="2025-01-01",
            )
            split_a50 = splitter_a50.split()

            dataset_a50 = FinancialDataset(
                features_df=df_a50,
                close_prices=df_a50["Close"].values,
                context_length=80,
                horizon=horizon,
                threshold=0.01,
                feature_columns=a50_feature_cols,
                high_prices=df_a50["High"].values,
            )

            val_indices_a50 = split_a50.val_indices
            val_dataset_a50 = torch.utils.data.Subset(dataset_a50, val_indices_a50)
            val_loader_a50 = torch.utils.data.DataLoader(val_dataset_a50, batch_size=64, shuffle=False)

            # Get predictions
            print(f"  Getting predictions (a20: {len(val_indices_a20)}, a50: {len(val_indices_a50)} samples)...")
            preds_a20, labels_a20 = get_predictions(model_a20, val_loader_a20, device)
            preds_a50, labels_a50 = get_predictions(model_a50, val_loader_a50, device)

            # Verify labels match (same underlying data, same splits)
            # Note: Due to different feature counts, dataset sizes may differ slightly
            # We need to align on the same samples
            min_len = min(len(labels_a20), len(labels_a50))

            # For fair comparison, we'll use the same labels
            # Since both datasets are from SPY with same dates, labels should match
            if not np.array_equal(labels_a20[:min_len], labels_a50[:min_len]):
                print(f"  WARNING: Label mismatch! Using a20 labels as reference.")
                # Use a20 as reference (more conservative)

            # Bootstrap CI
            print(f"  Running bootstrap ({BOOTSTRAP_ITERATIONS} iterations)...")
            # Use the aligned portion
            y_true = labels_a20[:min_len]
            y_a20 = preds_a20[:min_len]
            y_a50 = preds_a50[:min_len]

            ci_result = bootstrap_auc_ci(
                y_true, y_a20, y_a50,
                n_iterations=BOOTSTRAP_ITERATIONS,
                ci=CONFIDENCE_LEVEL,
                seed=RANDOM_SEED + horizon * 10 + BUDGETS.index(budget),
            )

            ci_result["budget"] = budget
            ci_result["horizon"] = horizon
            ci_result["n_samples"] = min_len

            bootstrap_results.append(ci_result)

            # Collect AUC data for ANOVA
            if ci_result["auc_a20"] is not None:
                auc_data.append({"tier": "a20", "budget": budget, "horizon": horizon, "auc": ci_result["auc_a20"]})
            if ci_result["auc_a50"] is not None:
                auc_data.append({"tier": "a50", "budget": budget, "horizon": horizon, "auc": ci_result["auc_a50"]})

            # Print result
            if ci_result["significant"] is not None:
                sig_marker = "**" if ci_result["significant"] else ""
                print(f"  AUC a20={ci_result['auc_a20']:.4f}, a50={ci_result['auc_a50']:.4f}")
                print(f"  Delta={ci_result['point_estimate']:+.4f}, 95% CI=[{ci_result['ci_lower']:+.4f}, {ci_result['ci_upper']:+.4f}] {sig_marker}")

    return bootstrap_results, auc_data


def print_summary(bootstrap_results: list[dict], anova_results: dict) -> None:
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("SUMMARY: BOOTSTRAP CI FOR AUC DIFFERENCE (a50 - a20)")
    print("=" * 70)
    print("\n| Budget | Horizon | AUC(a20) | AUC(a50) | Delta | 95% CI | Sig? |")
    print("|--------|---------|----------|----------|-------|--------|------|")

    for r in bootstrap_results:
        if r.get("point_estimate") is None:
            print(f"| {r['budget']:6} | H{r['horizon']} | N/A | N/A | N/A | N/A | N/A |")
        else:
            sig = "YES" if r["significant"] else "no"
            ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
            print(f"| {r['budget']:6} | H{r['horizon']} | {r['auc_a20']:.4f} | {r['auc_a50']:.4f} | {r['point_estimate']:+.4f} | {ci_str} | {sig} |")

    # Count significant results
    significant_count = sum(1 for r in bootstrap_results if r.get("significant") is True)
    total_count = sum(1 for r in bootstrap_results if r.get("significant") is not None)

    print(f"\nSignificant differences: {significant_count}/{total_count}")

    # ANOVA results
    print("\n" + "=" * 70)
    print("2-WAY ANOVA: TIER x BUDGET INTERACTION (H1 only)")
    print("=" * 70)

    if "error" in anova_results:
        print(f"\nError: {anova_results['error']}")
    else:
        print(f"\nMain Effects:")
        print(f"  Tier effect:   F = {anova_results['tier_F']:.3f}, p = {anova_results['tier_p']:.4f}")
        print(f"  Budget effect: F = {anova_results['budget_F']:.3f}, p = {anova_results['budget_p']:.4f}")
        print(f"\nInteraction:")
        print(f"  Tier x Budget: F = {anova_results['interaction_F']:.3f}, p = {anova_results['interaction_p']:.4f}")

        if anova_results["interaction_significant"]:
            print("\n  ** SIGNIFICANT INTERACTION (p < 0.05) **")
        else:
            print("\n  No significant interaction (p >= 0.05)")

    # Decision
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    any_significant = significant_count > 0
    interaction_significant = anova_results.get("interaction_significant", False)

    if not any_significant and not interaction_significant:
        print("\nRESULT: NULL FINDING")
        print("  - No AUC differences are statistically significant (CI includes 0)")
        print("  - No tier x budget interaction effect")
        print("  - Observed differences are likely due to sampling noise")
    elif any_significant and interaction_significant:
        print("\nRESULT: SIGNIFICANT EFFECT")
        print("  - Some AUC differences are statistically significant")
        print("  - Tier x budget interaction is significant")
        print("  - BUT: Requires test set replication to confirm")
    else:
        print("\nRESULT: INCONCLUSIVE")
        print(f"  - Significant AUC differences: {significant_count}/{total_count}")
        print(f"  - Interaction significant: {interaction_significant}")
        print("  - Mixed evidence - requires further investigation")


def main():
    """Main entry point."""
    print("=" * 70)
    print("STATISTICAL VALIDATION: a20 vs a50 AUC COMPARISON")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Run bootstrap analysis
    bootstrap_results, auc_data = run_bootstrap_analysis(device)

    # Run ANOVA
    print("\n" + "=" * 70)
    print("RUNNING 2-WAY ANOVA")
    print("=" * 70)
    anova_results = perform_anova(auc_data)

    # Print summary
    print_summary(bootstrap_results, anova_results)

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "confidence_level": CONFIDENCE_LEVEL,
            "random_seed": RANDOM_SEED,
        },
        "bootstrap_results": bootstrap_results,
        "auc_data": auc_data,
        "anova_results": anova_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
