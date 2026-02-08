#!/usr/bin/env python3
"""
Feature Embedding Results Analyzer

Aggregates results from feature embedding experiments and produces comparison tables.

Usage:
    ./venv/bin/python experiments/feature_embedding/analyze_results.py

Design doc: docs/feature_embedding_experiments.md
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

OUTPUT_DIR = PROJECT_ROOT / "outputs/feature_embedding"
SUMMARY_DIR = OUTPUT_DIR / "summary"


def load_all_results() -> list[dict]:
    """Load all results.json files from experiment directories."""
    results = []

    if not OUTPUT_DIR.exists():
        print(f"ERROR: Output directory not found: {OUTPUT_DIR}")
        return results

    for exp_dir in OUTPUT_DIR.iterdir():
        if exp_dir.is_dir() and exp_dir.name != "summary":
            results_path = exp_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                    data["exp_dir"] = exp_dir.name
                    results.append(data)

    return results


def create_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Create a comparison table from experiment results."""
    rows = []

    for r in results:
        if "error" in r:
            rows.append({
                "Exp ID": r.get("exp_id", "?"),
                "Priority": r.get("priority", "?"),
                "Tier": r.get("tier", "?"),
                "d_embed": r.get("d_embed"),
                "Params": None,
                "Precision": None,
                "Recall": None,
                "AUC": None,
                "Pred Range": None,
                "Error": r["error"],
            })
            continue

        val = r.get("val_metrics", {})
        params = r.get("parameters", {}).get("actual", 0)

        rows.append({
            "Exp ID": r.get("exp_id", "?"),
            "Priority": r.get("priority", "?"),
            "Tier": r.get("tier", "?"),
            "d_embed": r.get("d_embed"),
            "Params": params,
            "Precision": val.get("precision"),
            "Recall": val.get("recall"),
            "AUC": val.get("auc"),
            "Pred Range": val.get("pred_range"),
            "Pred Mean": val.get("pred_mean"),
            "N Positive": val.get("n_positive_preds"),
            "Train Time (min)": r.get("training", {}).get("training_time_min"),
            "Error": None,
        })

    df = pd.DataFrame(rows)

    # Sort by tier, then d_embed
    tier_order = {"a100": 0, "a200": 1, "a500": 2}
    df["tier_order"] = df["Tier"].map(tier_order)
    df["dembed_order"] = df["d_embed"].fillna(-1).astype(int)
    df = df.sort_values(["tier_order", "dembed_order"])
    df = df.drop(columns=["tier_order", "dembed_order"])

    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta from baseline (d_embed=None) for each tier."""
    delta_rows = []

    for tier in df["Tier"].unique():
        tier_df = df[df["Tier"] == tier]
        baseline = tier_df[tier_df["d_embed"].isna()]

        if len(baseline) == 0:
            continue

        baseline = baseline.iloc[0]
        base_precision = baseline["Precision"]
        base_recall = baseline["Recall"]
        base_auc = baseline["AUC"]
        base_params = baseline["Params"]

        for _, row in tier_df.iterrows():
            if pd.isna(row["d_embed"]):
                # This is the baseline
                delta_rows.append({
                    "Exp ID": row["Exp ID"],
                    "Tier": tier,
                    "d_embed": row["d_embed"],
                    "Params": row["Params"],
                    "Precision": row["Precision"],
                    "Delta Precision": 0.0,
                    "Delta Precision %": 0.0,
                    "Recall": row["Recall"],
                    "Delta Recall": 0.0,
                    "AUC": row["AUC"],
                    "Delta AUC": 0.0,
                    "Params Reduction %": 0.0,
                })
            else:
                prec_delta = (row["Precision"] - base_precision) if base_precision else None
                prec_delta_pct = (prec_delta / base_precision * 100) if base_precision and prec_delta else None
                recall_delta = (row["Recall"] - base_recall) if base_recall else None
                auc_delta = (row["AUC"] - base_auc) if (base_auc and row["AUC"]) else None
                params_reduction = ((base_params - row["Params"]) / base_params * 100) if base_params else None

                delta_rows.append({
                    "Exp ID": row["Exp ID"],
                    "Tier": tier,
                    "d_embed": row["d_embed"],
                    "Params": row["Params"],
                    "Precision": row["Precision"],
                    "Delta Precision": prec_delta,
                    "Delta Precision %": prec_delta_pct,
                    "Recall": row["Recall"],
                    "Delta Recall": recall_delta,
                    "AUC": row["AUC"],
                    "Delta AUC": auc_delta,
                    "Params Reduction %": params_reduction,
                })

    return pd.DataFrame(delta_rows)


def print_summary(df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    """Print human-readable summary."""
    print("=" * 80)
    print("FEATURE EMBEDDING EXPERIMENT RESULTS")
    print("=" * 80)

    print("\n## COMPARISON TABLE (all experiments)\n")

    # Format for display
    display_df = df.copy()
    if "Params" in display_df.columns:
        display_df["Params"] = display_df["Params"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
        )
    for col in ["Precision", "Recall", "AUC", "Pred Range", "Pred Mean"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "-"
            )
    if "Train Time (min)" in display_df.columns:
        display_df["Train Time (min)"] = display_df["Train Time (min)"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "-"
        )

    # Select columns for display
    display_cols = ["Exp ID", "Tier", "d_embed", "Params", "Precision", "Recall", "AUC", "Pred Range"]
    display_df = display_df[[c for c in display_cols if c in display_df.columns]]

    print(display_df.to_string(index=False))

    if len(delta_df) > 0:
        print("\n\n## DELTA FROM BASELINE (d_embed=None)\n")

        delta_display = delta_df.copy()
        delta_display["Params"] = delta_display["Params"].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
        )
        for col in ["Precision", "Recall", "AUC"]:
            if col in delta_display.columns:
                delta_display[col] = delta_display[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "-"
                )
        for col in ["Delta Precision", "Delta Recall", "Delta AUC"]:
            if col in delta_display.columns:
                delta_display[col] = delta_display[col].apply(
                    lambda x: f"{x:+.4f}" if pd.notna(x) else "-"
                )
        for col in ["Delta Precision %", "Params Reduction %"]:
            if col in delta_display.columns:
                delta_display[col] = delta_display[col].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
                )

        delta_cols = ["Exp ID", "Tier", "d_embed", "Params", "Precision", "Delta Precision %",
                      "Recall", "Delta Recall", "AUC", "Params Reduction %"]
        delta_display = delta_display[[c for c in delta_cols if c in delta_display.columns]]

        print(delta_display.to_string(index=False))

    # Key insights
    print("\n\n## KEY INSIGHTS\n")

    if len(delta_df) > 0:
        # Best precision improvement
        best_prec = delta_df[delta_df["Delta Precision %"] > 0].nlargest(1, "Delta Precision %")
        if len(best_prec) > 0:
            row = best_prec.iloc[0]
            print(f"Best Precision Improvement: {row['Exp ID']} ({row['Tier']}, d_embed={row['d_embed']:.0f})")
            print(f"  -> +{row['Delta Precision %']:.1f}% precision, {row['Params Reduction %']:.1f}% fewer params")
        else:
            print("No experiments showed precision improvement over baseline.")

        # Worst precision degradation
        worst_prec = delta_df[delta_df["Delta Precision %"] < 0].nsmallest(1, "Delta Precision %")
        if len(worst_prec) > 0:
            row = worst_prec.iloc[0]
            print(f"\nWorst Precision Degradation: {row['Exp ID']} ({row['Tier']}, d_embed={row['d_embed']:.0f})")
            print(f"  -> {row['Delta Precision %']:.1f}% precision, {row['Params Reduction %']:.1f}% fewer params")

        # Best efficiency (highest param reduction with acceptable precision)
        efficient = delta_df[
            (delta_df["Delta Precision %"] > -5) &  # Within 5% of baseline
            (delta_df["Params Reduction %"] > 0)
        ].nlargest(1, "Params Reduction %")
        if len(efficient) > 0:
            row = efficient.iloc[0]
            print(f"\nMost Efficient: {row['Exp ID']} ({row['Tier']}, d_embed={row['d_embed']:.0f})")
            print(f"  -> {row['Params Reduction %']:.1f}% fewer params, only {row['Delta Precision %']:+.1f}% precision change")

    # Check for probability collapse
    collapsed = df[df["Pred Range"] < 0.1]
    if len(collapsed) > 0:
        print("\nWARNING: Probability collapse detected in:")
        for _, row in collapsed.iterrows():
            print(f"  - {row['Exp ID']} ({row['Tier']}, d_embed={row['d_embed']}): Pred Range = {row['Pred Range']:.4f}")


def generate_markdown_report(df: pd.DataFrame, delta_df: pd.DataFrame) -> str:
    """Generate markdown report for the design doc."""
    lines = [
        "## Phase 1 Results",
        "",
        "### Comparison Table",
        "",
    ]

    # Markdown table
    lines.append("| Exp ID | Tier | d_embed | Params | Precision | Recall | AUC | Pred Range |")
    lines.append("|--------|------|---------|--------|-----------|--------|-----|------------|")

    for _, row in df.iterrows():
        params = f"{row['Params']:,.0f}" if pd.notna(row['Params']) else "-"
        prec = f"{row['Precision']:.4f}" if pd.notna(row['Precision']) else "-"
        recall = f"{row['Recall']:.4f}" if pd.notna(row['Recall']) else "-"
        auc = f"{row['AUC']:.4f}" if pd.notna(row['AUC']) else "-"
        pred_range = f"{row['Pred Range']:.4f}" if pd.notna(row['Pred Range']) else "-"
        d_embed = str(int(row['d_embed'])) if pd.notna(row['d_embed']) else "None"

        lines.append(f"| {row['Exp ID']} | {row['Tier']} | {d_embed} | {params} | {prec} | {recall} | {auc} | {pred_range} |")

    lines.extend([
        "",
        "### Delta from Baseline",
        "",
    ])

    if len(delta_df) > 0:
        lines.append("| Exp ID | Tier | d_embed | Delta Precision % | Params Reduction % |")
        lines.append("|--------|------|---------|-------------------|-------------------|")

        for _, row in delta_df.iterrows():
            d_embed = str(int(row['d_embed'])) if pd.notna(row['d_embed']) else "None"
            delta_prec = f"{row['Delta Precision %']:+.1f}%" if pd.notna(row['Delta Precision %']) else "-"
            params_red = f"{row['Params Reduction %']:.1f}%" if pd.notna(row['Params Reduction %']) else "-"
            lines.append(f"| {row['Exp ID']} | {row['Tier']} | {d_embed} | {delta_prec} | {params_red} |")

    return "\n".join(lines)


def main():
    print("Loading results...")
    results = load_all_results()

    if len(results) == 0:
        print("\nNo experiment results found.")
        print(f"Run experiments first with: ./venv/bin/python experiments/feature_embedding/run_experiments.py")
        return

    print(f"Found {len(results)} experiment results\n")

    # Create tables
    df = create_comparison_table(results)
    delta_df = compute_deltas(df)

    # Print summary
    print_summary(df, delta_df)

    # Save to files
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = SUMMARY_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n\nSaved comparison table to {csv_path}")

    if len(delta_df) > 0:
        delta_path = SUMMARY_DIR / "delta_table.csv"
        delta_df.to_csv(delta_path, index=False)
        print(f"Saved delta table to {delta_path}")

    # Generate markdown
    md_report = generate_markdown_report(df, delta_df)
    md_path = SUMMARY_DIR / "analysis.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Saved markdown report to {md_path}")

    # Copy key results to design doc
    print("\n" + "=" * 80)
    print("MARKDOWN FOR DESIGN DOC (copy to docs/feature_embedding_experiments.md)")
    print("=" * 80)
    print(md_report)


if __name__ == "__main__":
    main()
