"""Result aggregation utilities for scaling experiments.

This module provides functions to:
- Aggregate HPO results across experiments
- Aggregate training results across experiments
- Generate summary statistics and reports
- Export results to CSV for external analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def aggregate_hpo_results(
    experiment_name: str | None = None,
    hpo_dir: str = "outputs/hpo",
) -> pd.DataFrame:
    """Aggregate all HPO best.json files into DataFrame.

    Args:
        experiment_name: Filter to specific experiment, or None for all.
        hpo_dir: Directory containing HPO results.

    Returns:
        DataFrame with columns:
        - experiment, budget, best_value, n_trials_completed, timestamp
    """
    hpo_path = Path(hpo_dir)
    results = []

    # Handle non-existent or empty directory
    if not hpo_path.exists():
        return pd.DataFrame()

    # Find all *_best.json files in subdirectories
    for json_file in hpo_path.glob("*/*_best.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Extract fields with defaults
            exp_name = data.get("experiment", "")

            # Apply experiment filter if specified
            if experiment_name is not None and exp_name != experiment_name:
                continue

            results.append({
                "experiment": exp_name,
                "budget": data.get("budget", ""),
                "best_value": data.get("best_value", 0.0),
                "n_trials_completed": data.get("n_trials_completed", 0),
                "timestamp": data.get("timestamp", ""),
                "best_params": data.get("best_params", {}),
            })
        except (json.JSONDecodeError, OSError):
            # Skip malformed or unreadable files
            continue

    return pd.DataFrame(results)


def aggregate_training_results(
    results_dir: str = "outputs/results",
) -> pd.DataFrame:
    """Aggregate all training result JSONs into DataFrame.

    Args:
        results_dir: Directory containing training results.

    Returns:
        DataFrame with training metrics per experiment/budget.

    Note:
        Training result format not yet finalized. This is a placeholder
        that will be updated when Trainer saves result JSONs.
    """
    results_path = Path(results_dir)
    results = []

    if not results_path.exists():
        return pd.DataFrame()

    # Placeholder: scan for training result JSONs when format is defined
    for json_file in results_path.glob("**/training_result.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    return pd.DataFrame(results)


def summarize_experiment(
    experiment_name: str,
    hpo_dir: str = "outputs/hpo",
) -> dict:
    """Generate summary statistics for an experiment.

    Args:
        experiment_name: Experiment identifier.
        hpo_dir: Directory containing HPO results.

    Returns:
        Dict with:
        - best_budget: Budget with lowest best_value
        - scaling_factor: Ratio of worst to best value (improvement factor)
        - hpo_summary: Dict mapping budget to best_value
    """
    df = aggregate_hpo_results(experiment_name=experiment_name, hpo_dir=hpo_dir)

    if df.empty:
        return {
            "best_budget": None,
            "scaling_factor": None,
            "hpo_summary": {},
        }

    # Find best budget (lowest best_value)
    best_idx = df["best_value"].idxmin()
    best_budget = df.loc[best_idx, "budget"]

    # Calculate scaling factor (improvement from worst to best)
    max_val = df["best_value"].max()
    min_val = df["best_value"].min()
    scaling_factor = max_val / min_val if min_val > 0 else None

    # Build HPO summary
    hpo_summary = dict(zip(df["budget"], df["best_value"]))

    return {
        "best_budget": best_budget,
        "scaling_factor": scaling_factor,
        "hpo_summary": hpo_summary,
    }


def export_results_csv(
    hpo_dir: str = "outputs/hpo",
    output_path: str = "outputs/results/all_results.csv",
) -> Path:
    """Export all results to CSV for external analysis.

    Args:
        hpo_dir: Directory containing HPO results.
        output_path: Path for output CSV file.

    Returns:
        Path to created CSV file.
    """
    df = aggregate_hpo_results(hpo_dir=hpo_dir)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Drop best_params column for CSV (dict doesn't serialize well)
    if "best_params" in df.columns:
        df = df.drop(columns=["best_params"])

    df.to_csv(output, index=False)

    return output


def generate_experiment_summary_report(
    hpo_dir: str = "outputs/hpo",
    output_path: str = "outputs/results/summary_report.md",
) -> Path:
    """Generate markdown summary report of all experiments.

    Args:
        hpo_dir: Directory containing HPO results.
        output_path: Path for output markdown file.

    Returns:
        Path to created markdown file.
    """
    df = aggregate_hpo_results(hpo_dir=hpo_dir)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build markdown content
    lines = [
        "# Experiment Summary Report",
        "",
        f"Generated from: `{hpo_dir}`",
        "",
    ]

    if df.empty:
        lines.append("No experiment results found.")
    else:
        # Overall statistics
        experiments = df["experiment"].unique()
        lines.extend([
            f"## Overview",
            "",
            f"- **Total experiments:** {len(experiments)}",
            f"- **Total HPO runs:** {len(df)}",
            "",
        ])

        # Results table
        lines.extend([
            "## All Results",
            "",
            "| Experiment | Budget | Best Value | Trials |",
            "|------------|--------|------------|--------|",
        ])

        for _, row in df.iterrows():
            lines.append(
                f"| {row['experiment']} | {row['budget']} | "
                f"{row['best_value']:.4f} | {row['n_trials_completed']} |"
            )

        lines.append("")

        # Per-experiment summaries
        lines.extend([
            "## Experiment Summaries",
            "",
        ])

        for exp in sorted(experiments):
            summary = summarize_experiment(exp, hpo_dir=hpo_dir)
            lines.extend([
                f"### {exp}",
                "",
                f"- **Best budget:** {summary['best_budget']}",
                f"- **Scaling factor:** {summary['scaling_factor']:.2f}"
                if summary['scaling_factor'] else "- **Scaling factor:** N/A",
                "",
            ])

    # Write file
    output.write_text("\n".join(lines))

    return output
