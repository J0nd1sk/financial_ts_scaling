#!/usr/bin/env python3
"""
Compare a20 vs a50 threshold sweeps.

Loads threshold sweep data from Phase 6A (a20) and Phase 6C (a50) and produces:
1. Unified comparison table at common thresholds (0.5, 0.6, 0.7, 0.8)
2. Operating point comparisons (at target precision levels)
3. AUC comparison across tiers

Key question: Does a50 achieve same precision with better recall?

Usage:
    python experiments/phase6c/compare_a20_a50_thresholds.py
"""

import json
from pathlib import Path
import pandas as pd

# Constants
COMMON_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
PRECISION_TARGETS = [0.50, 0.55, 0.60, 0.65, 0.70]
BUDGETS = ["2M", "20M", "200M"]
HORIZONS = [1, 2, 3, 5]


def load_a20_data(csv_path: Path) -> pd.DataFrame:
    """Load Phase 6A (a20) threshold sweep data from CSV."""
    df = pd.read_csv(csv_path)
    df["tier"] = "a20"
    return df


def load_a50_data(stage1_path: Path, comprehensive_path: Path) -> pd.DataFrame:
    """Load Phase 6C (a50) threshold sweep data from JSON files."""
    rows = []

    # Load stage1 (H1 models)
    with open(stage1_path) as f:
        stage1_data = json.load(f)

    for model_key, model_data in stage1_data.items():
        if not model_key.startswith("s1_"):
            continue
        budget = model_data["budget"]
        auc = model_data["auc"]
        pred_range = model_data["pred_range"]

        for sweep_point in model_data["sweep"]:
            rows.append({
                "tier": "a50",
                "budget": budget,
                "horizon": 1,
                "threshold": sweep_point["threshold"],
                "precision": sweep_point["precision"],
                "recall": sweep_point["recall"],
                "f1": sweep_point["f1"],
                "n_positive_preds": sweep_point["n_positive_preds"],
                "n_samples": sweep_point.get("n_negative_preds", 0) + sweep_point["n_positive_preds"],
                "auc": auc,
                "pred_min": pred_range[0],
                "pred_max": pred_range[1],
            })

    # Load comprehensive (H2, H3, H5 models)
    with open(comprehensive_path) as f:
        comprehensive_data = json.load(f)

    for model_key, model_data in comprehensive_data.items():
        if not model_key.startswith("s2_horizon_"):
            continue

        # Parse budget and horizon from key like "s2_horizon_2m_h2_a50"
        parts = model_key.split("_")
        budget = parts[2].upper()  # "2m" -> "2M"
        horizon = int(parts[3][1:])  # "h2" -> 2

        auc = model_data["auc"]
        pred_range = model_data["pred_range"]
        n_samples = model_data["n_samples"]

        for sweep_point in model_data["sweep"]:
            rows.append({
                "tier": "a50",
                "budget": budget,
                "horizon": horizon,
                "threshold": sweep_point["threshold"],
                "precision": sweep_point["precision"],
                "recall": sweep_point["recall"],
                "f1": sweep_point["f1"],
                "n_positive_preds": sweep_point["n_positive_preds"],
                "n_samples": n_samples,
                "auc": auc,
                "pred_min": pred_range[0],
                "pred_max": pred_range[1],
            })

    return pd.DataFrame(rows)


def filter_to_common_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include common thresholds (with tolerance)."""
    tolerance = 0.01
    mask = df["threshold"].apply(
        lambda t: any(abs(t - ct) < tolerance for ct in COMMON_THRESHOLDS)
    )
    return df[mask].copy()


def normalize_threshold(t: float) -> float:
    """Normalize threshold to nearest common threshold."""
    for ct in COMMON_THRESHOLDS:
        if abs(t - ct) < 0.01:
            return ct
    return t


def create_comparison_table(a20_df: pd.DataFrame, a50_df: pd.DataFrame) -> pd.DataFrame:
    """Create side-by-side comparison at common thresholds."""
    # Filter to common thresholds
    a20_common = filter_to_common_thresholds(a20_df)
    a50_common = filter_to_common_thresholds(a50_df)

    # Normalize thresholds
    a20_common["threshold"] = a20_common["threshold"].apply(normalize_threshold)
    a50_common["threshold"] = a50_common["threshold"].apply(normalize_threshold)

    # Merge on budget, horizon, threshold
    merged = pd.merge(
        a20_common[["budget", "horizon", "threshold", "precision", "recall", "f1", "auc"]],
        a50_common[["budget", "horizon", "threshold", "precision", "recall", "f1", "auc"]],
        on=["budget", "horizon", "threshold"],
        suffixes=("_a20", "_a50"),
        how="outer",
    )

    # Calculate deltas
    merged["precision_delta"] = merged["precision_a50"] - merged["precision_a20"]
    merged["recall_delta"] = merged["recall_a50"] - merged["recall_a20"]
    merged["auc_delta"] = merged["auc_a50"] - merged["auc_a20"]

    return merged.sort_values(["horizon", "budget", "threshold"])


def compute_auc_comparison(a20_df: pd.DataFrame, a50_df: pd.DataFrame) -> pd.DataFrame:
    """Compare AUC across tiers by budget and horizon."""
    # Get unique AUC per budget/horizon
    a20_auc = a20_df.groupby(["budget", "horizon"])["auc"].first().reset_index()
    a20_auc["tier"] = "a20"

    a50_auc = a50_df.groupby(["budget", "horizon"])["auc"].first().reset_index()
    a50_auc["tier"] = "a50"

    # Pivot for comparison
    merged = pd.merge(
        a20_auc[["budget", "horizon", "auc"]].rename(columns={"auc": "auc_a20"}),
        a50_auc[["budget", "horizon", "auc"]].rename(columns={"auc": "auc_a50"}),
        on=["budget", "horizon"],
        how="outer",
    )
    merged["auc_delta"] = merged["auc_a50"] - merged["auc_a20"]

    return merged.sort_values(["horizon", "budget"])


def find_operating_point(df: pd.DataFrame, target_precision: float) -> dict:
    """Find the threshold/recall at a target precision level using interpolation."""
    # Sort by threshold ascending
    df_sorted = df.sort_values("threshold")

    # Find thresholds that bracket the target precision
    for i in range(len(df_sorted) - 1):
        p1 = df_sorted.iloc[i]["precision"]
        p2 = df_sorted.iloc[i + 1]["precision"]

        if p1 <= target_precision <= p2 or p2 <= target_precision <= p1:
            t1 = df_sorted.iloc[i]["threshold"]
            t2 = df_sorted.iloc[i + 1]["threshold"]
            r1 = df_sorted.iloc[i]["recall"]
            r2 = df_sorted.iloc[i + 1]["recall"]

            # Linear interpolation
            if abs(p2 - p1) < 1e-6:
                frac = 0.5
            else:
                frac = (target_precision - p1) / (p2 - p1)

            return {
                "threshold": t1 + frac * (t2 - t1),
                "precision": target_precision,
                "recall": r1 + frac * (r2 - r1),
            }

    # If target not achievable, find closest
    closest_idx = (df_sorted["precision"] - target_precision).abs().idxmin()
    row = df_sorted.loc[closest_idx]
    return {
        "threshold": row["threshold"],
        "precision": row["precision"],
        "recall": row["recall"],
    }


def operating_point_comparison(
    a20_df: pd.DataFrame, a50_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare recall at various precision targets."""
    rows = []

    for horizon in HORIZONS:
        for budget in BUDGETS:
            a20_subset = a20_df[
                (a20_df["horizon"] == horizon) & (a20_df["budget"] == budget)
            ]
            a50_subset = a50_df[
                (a50_df["horizon"] == horizon) & (a50_df["budget"] == budget)
            ]

            if a20_subset.empty or a50_subset.empty:
                continue

            for target in PRECISION_TARGETS:
                a20_op = find_operating_point(a20_subset, target)
                a50_op = find_operating_point(a50_subset, target)

                rows.append({
                    "budget": budget,
                    "horizon": horizon,
                    "target_precision": target,
                    "a20_threshold": a20_op["threshold"],
                    "a20_precision": a20_op["precision"],
                    "a20_recall": a20_op["recall"],
                    "a50_threshold": a50_op["threshold"],
                    "a50_precision": a50_op["precision"],
                    "a50_recall": a50_op["recall"],
                    "recall_improvement": a50_op["recall"] - a20_op["recall"],
                })

    return pd.DataFrame(rows)


def print_summary_tables(
    comparison_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    op_df: pd.DataFrame,
) -> None:
    """Print markdown summary tables to console."""
    print("\n" + "=" * 80)
    print("PHASE 6C: a20 vs a50 THRESHOLD COMPARISON")
    print("=" * 80)

    # AUC Summary
    print("\n## AUC Comparison by Budget and Horizon\n")
    print("| Horizon | Budget | AUC (a20) | AUC (a50) | Delta |")
    print("|---------|--------|-----------|-----------|-------|")
    for _, row in auc_df.iterrows():
        print(f"| H{row['horizon']} | {row['budget']} | {row['auc_a20']:.4f} | {row['auc_a50']:.4f} | {row['auc_delta']:+.4f} |")

    # Key finding: scaling x feature interaction
    print("\n## Key Finding: Scaling × Feature Interaction")
    print("\nOptimal budget by tier:")
    h1_auc = auc_df[auc_df["horizon"] == 1].copy()
    for tier in ["auc_a20", "auc_a50"]:
        tier_label = "a20" if "a20" in tier else "a50"
        best_idx = h1_auc[tier].idxmax()
        best_budget = h1_auc.loc[best_idx, "budget"]
        best_auc = h1_auc.loc[best_idx, tier]
        print(f"  - {tier_label}: {best_budget} (AUC={best_auc:.4f})")

    # Threshold comparison (H1 only for brevity)
    print("\n## Threshold Comparison at H1\n")
    h1_comp = comparison_df[comparison_df["horizon"] == 1]
    print("| Budget | Thresh | Prec(a20) | Prec(a50) | Δ Prec | Rec(a20) | Rec(a50) | Δ Rec |")
    print("|--------|--------|-----------|-----------|--------|----------|----------|-------|")
    for _, row in h1_comp.iterrows():
        prec_a20 = row["precision_a20"] if pd.notna(row["precision_a20"]) else 0
        prec_a50 = row["precision_a50"] if pd.notna(row["precision_a50"]) else 0
        rec_a20 = row["recall_a20"] if pd.notna(row["recall_a20"]) else 0
        rec_a50 = row["recall_a50"] if pd.notna(row["recall_a50"]) else 0
        print(f"| {row['budget']} | {row['threshold']:.1f} | {prec_a20:.3f} | {prec_a50:.3f} | {row['precision_delta']:+.3f} | {rec_a20:.3f} | {rec_a50:.3f} | {row['recall_delta']:+.3f} |")

    # Operating point comparison (65% precision)
    print("\n## Operating Point: 65% Precision Target\n")
    op_65 = op_df[op_df["target_precision"] == 0.65]
    print("| Horizon | Budget | Recall(a20) | Recall(a50) | Improvement |")
    print("|---------|--------|-------------|-------------|-------------|")
    for _, row in op_65.iterrows():
        print(f"| H{row['horizon']} | {row['budget']} | {row['a20_recall']:.3f} | {row['a50_recall']:.3f} | {row['recall_improvement']:+.3f} |")

    # Summary statistics
    print("\n## Summary Statistics")
    avg_auc_improvement = auc_df["auc_delta"].mean()
    print(f"\nAverage AUC improvement (a50 over a20): {avg_auc_improvement:+.4f}")

    h1_auc_delta = auc_df[auc_df["horizon"] == 1]["auc_delta"].mean()
    print(f"H1 average AUC improvement: {h1_auc_delta:+.4f}")

    avg_recall_at_65 = op_65["recall_improvement"].mean()
    print(f"Average recall improvement at 65% precision: {avg_recall_at_65:+.3f}")


def save_results(
    comparison_df: pd.DataFrame,
    auc_df: pd.DataFrame,
    op_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save results to JSON file."""
    results = {
        "comparison_table": comparison_df.to_dict(orient="records"),
        "auc_comparison": auc_df.to_dict(orient="records"),
        "operating_points": op_df.to_dict(orient="records"),
        "summary": {
            "avg_auc_improvement": auc_df["auc_delta"].mean(),
            "h1_avg_auc_improvement": auc_df[auc_df["horizon"] == 1]["auc_delta"].mean(),
            "avg_recall_improvement_at_65pct": op_df[op_df["target_precision"] == 0.65]["recall_improvement"].mean(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    a20_csv = base_dir / "outputs" / "phase6a_final" / "threshold_sweep.csv"
    a50_stage1 = base_dir / "outputs" / "phase6c" / "stage1_threshold_sweep.json"
    a50_comprehensive = base_dir / "outputs" / "phase6c" / "comprehensive_threshold_sweep.json"
    output_path = base_dir / "outputs" / "phase6c" / "a20_vs_a50_comparison.json"

    # Load data
    print("Loading data...")
    a20_df = load_a20_data(a20_csv)
    a50_df = load_a50_data(a50_stage1, a50_comprehensive)

    print(f"  a20: {len(a20_df)} rows, horizons: {sorted(a20_df['horizon'].unique())}")
    print(f"  a50: {len(a50_df)} rows, horizons: {sorted(a50_df['horizon'].unique())}")

    # Create comparison tables
    print("\nCreating comparisons...")
    comparison_df = create_comparison_table(a20_df, a50_df)
    auc_df = compute_auc_comparison(a20_df, a50_df)
    op_df = operating_point_comparison(a20_df, a50_df)

    # Print summary
    print_summary_tables(comparison_df, auc_df, op_df)

    # Save results
    save_results(comparison_df, auc_df, op_df, output_path)


if __name__ == "__main__":
    main()
