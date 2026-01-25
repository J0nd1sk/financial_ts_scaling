#!/usr/bin/env python3
"""
Generate figures for Phase 6C scaling analysis.

Creates:
1. Precision-Recall curves by horizon (a20 vs a50)
2. Scaling × Feature interaction heatmap (AUC improvement)

Usage:
    python experiments/phase6c/generate_scaling_figures.py
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "a20-2M": "#1f77b4",
    "a20-20M": "#2ca02c",
    "a20-200M": "#d62728",
    "a50-2M": "#17becf",
    "a50-20M": "#bcbd22",
    "a50-200M": "#e377c2",
}
LINESTYLES = {
    "a20": "-",
    "a50": "--",
}
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

        for sweep_point in model_data["sweep"]:
            rows.append({
                "tier": "a50",
                "budget": budget,
                "horizon": 1,
                "threshold": sweep_point["threshold"],
                "precision": sweep_point["precision"],
                "recall": sweep_point["recall"],
                "f1": sweep_point["f1"],
                "auc": auc,
            })

    # Load comprehensive (H2, H3, H5 models)
    with open(comprehensive_path) as f:
        comprehensive_data = json.load(f)

    for model_key, model_data in comprehensive_data.items():
        if not model_key.startswith("s2_horizon_"):
            continue

        parts = model_key.split("_")
        budget = parts[2].upper()
        horizon = int(parts[3][1:])
        auc = model_data["auc"]

        for sweep_point in model_data["sweep"]:
            rows.append({
                "tier": "a50",
                "budget": budget,
                "horizon": horizon,
                "threshold": sweep_point["threshold"],
                "precision": sweep_point["precision"],
                "recall": sweep_point["recall"],
                "f1": sweep_point["f1"],
                "auc": auc,
            })

    return pd.DataFrame(rows)


def plot_precision_recall_curve(
    a20_df: pd.DataFrame,
    a50_df: pd.DataFrame,
    horizon: int,
    output_path: Path,
) -> None:
    """Create precision-recall curve for a specific horizon."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter by horizon
    a20_h = a20_df[a20_df["horizon"] == horizon]
    a50_h = a50_df[a50_df["horizon"] == horizon]

    for budget in BUDGETS:
        # a20 line
        a20_budget = a20_h[a20_h["budget"] == budget].sort_values("recall")
        if not a20_budget.empty:
            label = f"a20-{budget}"
            ax.plot(
                a20_budget["recall"],
                a20_budget["precision"],
                color=COLORS[label],
                linestyle=LINESTYLES["a20"],
                linewidth=2,
                marker="o",
                markersize=6,
                label=label,
            )

        # a50 line
        a50_budget = a50_h[a50_h["budget"] == budget].sort_values("recall")
        if not a50_budget.empty:
            label = f"a50-{budget}"
            ax.plot(
                a50_budget["recall"],
                a50_budget["precision"],
                color=COLORS[label],
                linestyle=LINESTYLES["a50"],
                linewidth=2,
                marker="s",
                markersize=5,
                label=label,
            )

    # Reference lines for precision targets
    for target, alpha in [(0.65, 0.4), (0.70, 0.3), (0.75, 0.2)]:
        ax.axhline(
            y=target,
            color="gray",
            linestyle=":",
            alpha=alpha,
            label=f"{int(target*100)}% precision" if target == 0.65 else None,
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve: H{horizon} Direction Prediction", fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation about tier difference
    a20_auc = a20_h.groupby("budget")["auc"].first().mean() if not a20_h.empty else 0
    a50_auc = a50_h.groupby("budget")["auc"].first().mean() if not a50_h.empty else 0
    auc_delta = a50_auc - a20_auc
    ax.text(
        0.02, 0.02,
        f"Mean AUC: a20={a20_auc:.3f}, a50={a50_auc:.3f} (Δ={auc_delta:+.3f})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_scaling_feature_heatmap(
    a20_df: pd.DataFrame,
    a50_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create heatmap showing AUC improvement from a20 to a50."""
    # Get unique AUC per budget/horizon
    a20_auc = a20_df.groupby(["horizon", "budget"])["auc"].first().unstack()
    a50_auc = a50_df.groupby(["horizon", "budget"])["auc"].first().unstack()

    # Reorder columns
    a20_auc = a20_auc[BUDGETS]
    a50_auc = a50_auc[BUDGETS]

    # Calculate delta
    delta = a50_auc - a20_auc

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # AUC heatmaps
    data_sets = [
        (a20_auc, "a20 AUC", "Blues", 0.5, 0.8),
        (a50_auc, "a50 AUC", "Greens", 0.5, 0.8),
        (delta, "AUC Improvement (a50 - a20)", "RdYlGn", -0.1, 0.1),
    ]

    for ax, (data, title, cmap, vmin, vmax) in zip(axes, data_sets):
        im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # Add text annotations
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if pd.notna(val):
                    text_color = "white" if cmap in ["Blues", "Greens"] and val > 0.65 else "black"
                    if cmap == "RdYlGn":
                        text_color = "white" if abs(val) > 0.05 else "black"
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=text_color,
                    )

        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, fontsize=11)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([f"H{h}" for h in data.index], fontsize=11)
        ax.set_xlabel("Parameter Budget", fontsize=12)
        ax.set_ylabel("Horizon", fontsize=12)
        ax.set_title(title, fontsize=13)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(
        "Scaling × Feature Tier Interaction: AUC Comparison",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_optimal_budget_by_tier(
    a20_df: pd.DataFrame,
    a50_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create bar chart showing optimal budget shifts between tiers."""
    # Get AUC by budget and horizon
    a20_auc = a20_df.groupby(["horizon", "budget"])["auc"].first().unstack()
    a50_auc = a50_df.groupby(["horizon", "budget"])["auc"].first().unstack()

    # Reorder columns
    a20_auc = a20_auc[BUDGETS]
    a50_auc = a50_auc[BUDGETS]

    # Find optimal budget for each horizon
    results = []
    for horizon in a20_auc.index:
        if horizon in a50_auc.index:
            a20_best = a20_auc.loc[horizon].idxmax()
            a50_best = a50_auc.loc[horizon].idxmax()
            results.append({
                "horizon": horizon,
                "a20_best": a20_best,
                "a20_auc": a20_auc.loc[horizon, a20_best],
                "a50_best": a50_best,
                "a50_auc": a50_auc.loc[horizon, a50_best],
            })

    results_df = pd.DataFrame(results)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax.bar(
        x - width/2,
        results_df["a20_auc"],
        width,
        label="a20 (25 features)",
        color="#1f77b4",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width/2,
        results_df["a50_auc"],
        width,
        label="a50 (55 features)",
        color="#2ca02c",
        alpha=0.8,
    )

    # Add budget labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(
            bar1.get_x() + bar1.get_width()/2,
            bar1.get_height() + 0.005,
            results_df.iloc[i]["a20_best"],
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
        ax.text(
            bar2.get_x() + bar2.get_width()/2,
            bar2.get_height() + 0.005,
            results_df.iloc[i]["a50_best"],
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Prediction Horizon", fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Optimal Parameter Budget by Feature Tier", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in results_df["horizon"]], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0.5, 0.8)
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation
    ax.text(
        0.02, 0.98,
        "Labels show optimal budget for each tier",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontstyle="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main entry point."""
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    a20_csv = base_dir / "outputs" / "phase6a_final" / "threshold_sweep.csv"
    a50_stage1 = base_dir / "outputs" / "phase6c" / "stage1_threshold_sweep.json"
    a50_comprehensive = base_dir / "outputs" / "phase6c" / "comprehensive_threshold_sweep.json"
    figures_dir = base_dir / "outputs" / "phase6c" / "figures"

    # Load data
    print("Loading data...")
    a20_df = load_a20_data(a20_csv)
    a50_df = load_a50_data(a50_stage1, a50_comprehensive)

    print(f"  a20: {len(a20_df)} rows")
    print(f"  a50: {len(a50_df)} rows")

    # Generate figures
    print("\nGenerating figures...")

    # PR curves by horizon
    for horizon in HORIZONS:
        output_path = figures_dir / f"pr_curve_h{horizon}.png"
        plot_precision_recall_curve(a20_df, a50_df, horizon, output_path)

    # Heatmap
    plot_scaling_feature_heatmap(
        a20_df, a50_df,
        figures_dir / "scaling_feature_heatmap.png",
    )

    # Optimal budget chart
    plot_optimal_budget_by_tier(
        a20_df, a50_df,
        figures_dir / "optimal_budget_by_tier.png",
    )

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
