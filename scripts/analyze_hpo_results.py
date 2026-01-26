#!/usr/bin/env python3
"""
Standardized HPO Results Analysis

Comprehensive analysis pipeline for any HPO run:
1. Coverage heatmaps (as tables)
2. Parameter importance analysis (fANOVA-style)
3. Trend plots per hyperparameter
4. Cross-budget comparison tables
5. Probability collapse detection
6. Full metrics analysis (precision, recall, pred_range)

Supports both old format (all_trials.json) and new format (trial_metrics.csv).

Usage:
    python scripts/analyze_hpo_results.py [--budget 2M] [--tier a100] [--all]
    python scripts/analyze_hpo_results.py --all  # Analyze all available budgets
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_hpo_data(budget: str, tier: str = "a100") -> tuple[pd.DataFrame, dict]:
    """Load HPO data in either new or old format.

    Returns:
        (df: DataFrame with trial data, metadata: dict with search space etc.)
    """
    base_dir = PROJECT_ROOT / f"outputs/phase6c_{tier}"

    # Try new format first (trial_metrics.csv from template)
    new_format_dirs = list(base_dir.glob(f"hpo_{budget.lower()}_*_{tier}"))
    if new_format_dirs:
        exp_dir = new_format_dirs[0]
        if (exp_dir / "trial_metrics.csv").exists():
            df = pd.read_csv(exp_dir / "trial_metrics.csv")
            metadata = {}
            if (exp_dir / "best_params.json").exists():
                with open(exp_dir / "best_params.json") as f:
                    metadata = json.load(f)
            return df, metadata

    # Fall back to old format (all_trials.json)
    old_dir = base_dir / f"hpo_{budget.lower()}_h1"
    if old_dir.exists() and (old_dir / "all_trials.json").exists():
        with open(old_dir / "all_trials.json") as f:
            trials = json.load(f)

        # Convert to DataFrame
        rows = []
        for trial in trials:
            if trial["state"] != "TrialState.COMPLETE":
                continue
            row = {
                "trial": trial["number"],
                "auc": trial["value"],
                **trial["params"],
            }
            # Check for user_attrs in old format (some may have them)
            if "user_attrs" in trial:
                for k, v in trial["user_attrs"].items():
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows)
        metadata = {}
        if (old_dir / "best_params.json").exists():
            with open(old_dir / "best_params.json") as f:
                metadata = json.load(f)
        return df, metadata

    return pd.DataFrame(), {}


def compute_coverage_matrix(df: pd.DataFrame, param1: str, param2: str) -> pd.DataFrame:
    """Compute coverage matrix for two parameters."""
    if param1 not in df.columns or param2 not in df.columns:
        return pd.DataFrame()

    # Count combinations
    counts = df.groupby([param1, param2]).size().unstack(fill_value=0)
    return counts


def df_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without tabulate dependency."""
    if df.empty:
        return ""

    lines = []
    # Header row with index name
    index_name = df.index.name or ""
    headers = [str(index_name)] + [str(c) for c in df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Data rows
    for idx, row in df.iterrows():
        values = [str(idx)] + [str(v) for v in row.values]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def compute_parameter_importance(df: pd.DataFrame, target: str = "auc") -> dict[str, float]:
    """Estimate parameter importance using variance-based analysis.

    This is a simplified version of fANOVA - computing variance in target
    explained by each parameter.
    """
    if target not in df.columns:
        return {}

    importance = {}
    total_var = df[target].var()

    if total_var == 0:
        return {}

    for param in ["d_model", "n_layers", "n_heads", "d_ff_ratio",
                  "learning_rate", "dropout", "weight_decay"]:
        if param not in df.columns:
            continue

        # Compute mean target by parameter value
        group_means = df.groupby(param)[target].mean()

        # Compute between-group variance
        overall_mean = df[target].mean()
        group_sizes = df.groupby(param).size()

        between_var = sum(
            group_sizes[g] * (group_means[g] - overall_mean) ** 2
            for g in group_means.index
        ) / len(df)

        # Importance as fraction of variance explained
        importance[param] = round(between_var / total_var, 4)

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def detect_probability_collapse(df: pd.DataFrame) -> dict:
    """Detect probability collapse patterns."""
    collapse_info = {
        "detected": False,
        "details": []
    }

    # Check if pred_range columns exist
    if "pred_min" in df.columns and "pred_max" in df.columns:
        valid_preds = df.dropna(subset=["pred_min", "pred_max"])
        if not valid_preds.empty:
            avg_min = valid_preds["pred_min"].mean()
            avg_max = valid_preds["pred_max"].mean()
            avg_range = avg_max - avg_min

            collapse_info["avg_pred_min"] = round(avg_min, 4)
            collapse_info["avg_pred_max"] = round(avg_max, 4)
            collapse_info["avg_pred_range"] = round(avg_range, 4)

            # Collapse if range is very narrow (< 0.2)
            if avg_range < 0.2:
                collapse_info["detected"] = True
                collapse_info["details"].append(
                    f"Narrow prediction range: [{avg_min:.3f}, {avg_max:.3f}] (range={avg_range:.3f})"
                )

            # Also check for extreme values
            if avg_max < 0.7:
                collapse_info["details"].append(
                    f"Low max probability: {avg_max:.3f} (should be closer to 1.0)"
                )
            if avg_min > 0.4:
                collapse_info["details"].append(
                    f"High min probability: {avg_min:.3f} (should be closer to 0.0)"
                )

    # Check recall patterns
    if "recall" in df.columns:
        valid_recall = df["recall"].dropna()
        if not valid_recall.empty:
            avg_recall = valid_recall.mean()
            zero_recall_count = (valid_recall == 0).sum()
            collapse_info["avg_recall"] = round(avg_recall, 4)
            collapse_info["zero_recall_trials"] = int(zero_recall_count)

            if zero_recall_count > len(valid_recall) * 0.3:
                collapse_info["detected"] = True
                collapse_info["details"].append(
                    f"High zero-recall trials: {zero_recall_count}/{len(valid_recall)} ({100*zero_recall_count/len(valid_recall):.0f}%)"
                )

    return collapse_info


def analyze_trends_by_parameter(df: pd.DataFrame, param: str, target: str = "auc") -> dict:
    """Analyze trend of target vs parameter values."""
    if param not in df.columns or target not in df.columns:
        return {}

    grouped = df.groupby(param)[target].agg(["mean", "std", "count", "min", "max"])
    grouped = grouped.sort_index()

    # Detect trend direction
    means = list(grouped["mean"])
    if len(means) >= 2:
        trend = "increasing" if means[-1] > means[0] else "decreasing"
        if abs(means[-1] - means[0]) < 0.01:
            trend = "flat"
    else:
        trend = "insufficient_data"

    return {
        "param": param,
        "trend": trend,
        "best_value": grouped["mean"].idxmax(),
        "best_mean": round(grouped["mean"].max(), 4),
        "worst_value": grouped["mean"].idxmin(),
        "worst_mean": round(grouped["mean"].min(), 4),
        "by_value": {
            str(idx): {
                "mean": round(row["mean"], 4),
                "std": round(row["std"], 4) if not pd.isna(row["std"]) else 0,
                "count": int(row["count"]),
            }
            for idx, row in grouped.iterrows()
        }
    }


def generate_report(
    budget: str,
    df: pd.DataFrame,
    metadata: dict,
    tier: str = "a100"
) -> str:
    """Generate comprehensive analysis report."""
    lines = []
    lines.append(f"# HPO Analysis: {budget} ({tier})\n")

    if df.empty:
        lines.append("No data available for this budget.\n")
        return "\n".join(lines)

    # Summary statistics
    lines.append("## Summary Statistics\n")
    lines.append(f"- Total trials: {len(df)}")
    lines.append(f"- Best AUC: {df['auc'].max():.4f}")
    lines.append(f"- Mean AUC: {df['auc'].mean():.4f}")
    lines.append(f"- Std AUC: {df['auc'].std():.4f}")

    if metadata.get("best_params"):
        lines.append("\n**Best Configuration:**")
        for k, v in metadata["best_params"].items():
            lines.append(f"  - {k}: {v}")
    lines.append("")

    # Probability collapse detection
    collapse = detect_probability_collapse(df)
    lines.append("## Probability Collapse Analysis\n")
    if collapse["detected"]:
        lines.append("**WARNING: Probability collapse detected!**")
        for detail in collapse["details"]:
            lines.append(f"  - {detail}")
    else:
        lines.append("No probability collapse detected.")

    if "avg_pred_range" in collapse:
        lines.append(f"\n- Avg pred range: [{collapse.get('avg_pred_min', 'N/A')}, {collapse.get('avg_pred_max', 'N/A')}]")
    if "avg_recall" in collapse:
        lines.append(f"- Avg recall: {collapse['avg_recall']:.4f}")
    if "zero_recall_trials" in collapse:
        lines.append(f"- Zero-recall trials: {collapse['zero_recall_trials']}")
    lines.append("")

    # Parameter importance
    importance = compute_parameter_importance(df)
    lines.append("## Parameter Importance\n")
    lines.append("(Fraction of AUC variance explained by each parameter)")
    lines.append("")
    for param, imp in importance.items():
        bar_len = int(imp * 50)
        bar = "#" * bar_len
        lines.append(f"  {param:15} | {bar:50} | {imp:.4f}")
    lines.append("")

    # Trends per parameter
    lines.append("## Parameter Trends\n")
    for param in ["d_model", "n_layers", "n_heads", "learning_rate", "dropout", "weight_decay"]:
        trends = analyze_trends_by_parameter(df, param)
        if not trends:
            continue

        lines.append(f"### {param}\n")
        lines.append(f"- Trend: **{trends['trend']}**")
        lines.append(f"- Best value: {trends['best_value']} (mean AUC: {trends['best_mean']})")
        lines.append(f"- Worst value: {trends['worst_value']} (mean AUC: {trends['worst_mean']})")
        lines.append("")

        # Value breakdown table
        lines.append("| Value | Mean AUC | Std | Count |")
        lines.append("|-------|----------|-----|-------|")
        for val, stats in trends["by_value"].items():
            lines.append(f"| {val} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['count']} |")
        lines.append("")

    # Coverage matrices
    lines.append("## Coverage Matrices\n")
    lines.append("(Count of trials testing each combination)")
    lines.append("")

    # d_model x n_layers
    coverage = compute_coverage_matrix(df, "d_model", "n_layers")
    if not coverage.empty:
        lines.append("### d_model × n_layers\n")
        lines.append(df_to_markdown(coverage))
        lines.append("")

    # d_model x n_heads
    coverage = compute_coverage_matrix(df, "d_model", "n_heads")
    if not coverage.empty:
        lines.append("### d_model × n_heads\n")
        lines.append(df_to_markdown(coverage))
        lines.append("")

    # learning_rate x dropout
    coverage = compute_coverage_matrix(df, "learning_rate", "dropout")
    if not coverage.empty:
        lines.append("### learning_rate × dropout\n")
        lines.append(df_to_markdown(coverage))
        lines.append("")

    # Forced extreme analysis (if available)
    if "forced_extreme" in df.columns:
        forced = df[df["forced_extreme"] == True]
        if len(forced) > 0:
            lines.append("## Forced Extreme Trials\n")
            lines.append("| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |")
            lines.append("|-------|--------------|-----|---------|----------|---------|")
            for _, row in forced.iterrows():
                lines.append(
                    f"| {int(row['trial'])} | {row.get('extreme_type', 'N/A')} | "
                    f"{row['auc']:.4f} | {int(row['d_model'])} | {int(row['n_layers'])} | "
                    f"{int(row['n_heads'])} |"
                )
            lines.append("")

    return "\n".join(lines)


def generate_cross_budget_report(all_data: dict[str, tuple[pd.DataFrame, dict]]) -> str:
    """Generate cross-budget comparison report."""
    lines = []
    lines.append("# Cross-Budget HPO Comparison\n")

    # Summary table
    lines.append("## Best Results by Budget\n")
    lines.append("| Budget | Best AUC | Best d_model | Best n_layers | Best LR | Best dropout | Best WD |")
    lines.append("|--------|----------|--------------|---------------|---------|--------------|---------|")

    for budget in ["2M", "20M", "200M"]:
        if budget not in all_data:
            continue
        df, metadata = all_data[budget]
        if df.empty:
            continue

        bp = metadata.get("best_params", {})
        best_auc = df["auc"].max()
        lines.append(
            f"| {budget} | {best_auc:.4f} | {bp.get('d_model', 'N/A')} | "
            f"{bp.get('n_layers', 'N/A')} | {bp.get('learning_rate', 'N/A')} | "
            f"{bp.get('dropout', 'N/A')} | {bp.get('weight_decay', 'N/A')} |"
        )
    lines.append("")

    # Scaling law check
    aucs = {}
    for budget in ["2M", "20M", "200M"]:
        if budget in all_data and not all_data[budget][0].empty:
            aucs[budget] = all_data[budget][0]["auc"].max()

    if len(aucs) >= 2:
        lines.append("## Scaling Law Analysis\n")
        for budget, auc in sorted(aucs.items(), key=lambda x: {"2M": 1, "20M": 2, "200M": 3}[x[0]]):
            lines.append(f"- {budget}: {auc:.4f}")

        # Check pattern
        if all(k in aucs for k in ["2M", "20M", "200M"]):
            if aucs["200M"] > aucs["20M"] > aucs["2M"]:
                lines.append("\n**Scaling law HOLDS**: 200M > 20M > 2M")
            elif aucs["20M"] > aucs["2M"] > aucs["200M"]:
                lines.append("\n**Scaling law VIOLATED**: 20M > 2M > 200M")
                lines.append("This suggests larger models may be overfitting or need different regularization.")
            else:
                lines.append("\n**Scaling law PARTIAL**: Non-monotonic relationship")
        lines.append("")

    # Cross-budget parameter consistency
    lines.append("## Parameter Consistency Across Budgets\n")

    params_to_check = ["dropout", "learning_rate", "weight_decay"]
    for param in params_to_check:
        values = {}
        for budget in ["2M", "20M", "200M"]:
            if budget in all_data:
                bp = all_data[budget][1].get("best_params", {})
                if param in bp:
                    values[budget] = bp[param]

        if len(values) > 1:
            unique_vals = set(values.values())
            consistency = "Consistent" if len(unique_vals) == 1 else "Inconsistent"
            lines.append(f"- **{param}**: {consistency} - {values}")
    lines.append("")

    # Probability collapse across budgets
    lines.append("## Probability Collapse by Budget\n")
    lines.append("| Budget | Collapse? | Pred Range | Avg Recall |")
    lines.append("|--------|-----------|------------|------------|")

    for budget in ["2M", "20M", "200M"]:
        if budget not in all_data:
            continue
        df, _ = all_data[budget]
        if df.empty:
            continue

        collapse = detect_probability_collapse(df)
        pred_range = f"[{collapse.get('avg_pred_min', 'N/A'):.3f}, {collapse.get('avg_pred_max', 'N/A'):.3f}]" if "avg_pred_min" in collapse else "N/A"
        recall = f"{collapse.get('avg_recall', 'N/A'):.4f}" if "avg_recall" in collapse else "N/A"
        detected = "YES" if collapse["detected"] else "No"
        lines.append(f"| {budget} | {detected} | {pred_range} | {recall} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze HPO results")
    parser.add_argument("--budget", type=str, default=None, help="Budget to analyze (2M, 20M, 200M)")
    parser.add_argument("--tier", type=str, default="a100", help="Feature tier")
    parser.add_argument("--all", action="store_true", help="Analyze all budgets")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    print("=" * 70)
    print("HPO RESULTS ANALYSIS")
    print("=" * 70)

    reports = []
    all_data = {}

    budgets = ["2M", "20M", "200M"] if args.all or args.budget is None else [args.budget.upper()]

    for budget in budgets:
        print(f"\nLoading {budget} data...")
        df, metadata = load_hpo_data(budget, args.tier)

        if df.empty:
            print(f"  No data found for {budget}")
            continue

        print(f"  Loaded {len(df)} trials")
        all_data[budget] = (df, metadata)

        report = generate_report(budget, df, metadata, args.tier)
        reports.append(report)

    # Generate cross-budget comparison if multiple budgets
    if len(all_data) > 1:
        cross_report = generate_cross_budget_report(all_data)
        reports.insert(0, cross_report)

    full_report = "\n---\n\n".join(reports)

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / f"outputs/phase6c_{args.tier}/hpo_analysis_report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_report)
    print(f"\nReport saved to: {output_path}")

    # Also print summary to console
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for budget, (df, metadata) in all_data.items():
        if not df.empty:
            print(f"\n{budget}:")
            print(f"  Best AUC: {df['auc'].max():.4f}")
            if metadata.get("best_params"):
                bp = metadata["best_params"]
                print(f"  Best config: d={bp.get('d_model')}, L={bp.get('n_layers')}, "
                      f"h={bp.get('n_heads')}, drop={bp.get('dropout')}")


if __name__ == "__main__":
    main()
