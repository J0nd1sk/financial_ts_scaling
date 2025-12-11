"""Scaling curve analysis for neural scaling law experiments.

This module provides functions to:
- Fit power law relationships (error ∝ N^(-α))
- Visualize scaling curves on log-log plots
- Load and aggregate experiment results
- Generate scaling analysis reports
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_power_law(
    params: np.ndarray,
    errors: np.ndarray,
) -> tuple[float, float, float]:
    """Fit power law: error = a * params^(-alpha).

    Args:
        params: Array of parameter counts (must be positive).
        errors: Array of error values (must be positive).

    Returns:
        Tuple of (alpha, a, r_squared):
        - alpha: Scaling exponent (higher = better scaling)
        - a: Scaling coefficient
        - r_squared: Goodness of fit (0 to 1)

    Raises:
        ValueError: If inputs are invalid (wrong length, negative values, etc.)

    Notes:
        Fit performed in log-log space: log(error) = log(a) - α*log(N)
    """
    params = np.asarray(params, dtype=float)
    errors = np.asarray(errors, dtype=float)

    # Input validation
    if len(params) != len(errors):
        raise ValueError(
            f"params and errors must have same length, "
            f"got {len(params)} and {len(errors)}"
        )

    if len(params) < 2:
        raise ValueError(
            f"Need at least 2 data points for fitting, got {len(params)}"
        )

    if np.any(params <= 0):
        raise ValueError("All params must be positive for log transform")

    if np.any(errors <= 0):
        raise ValueError("All errors must be positive for log transform")

    # Transform to log-log space
    log_params = np.log(params)
    log_errors = np.log(errors)

    # Linear regression in log space: log(error) = log(a) - alpha * log(params)
    # This is y = intercept + slope * x, where slope = -alpha
    coeffs = np.polyfit(log_params, log_errors, 1)
    slope, intercept = coeffs

    # Extract parameters
    alpha = -slope  # Negate because error decreases with params
    a = np.exp(intercept)

    # Calculate R-squared
    log_errors_pred = np.polyval(coeffs, log_params)
    ss_res = np.sum((log_errors - log_errors_pred) ** 2)
    ss_tot = np.sum((log_errors - np.mean(log_errors)) ** 2)

    # Handle edge case where all errors are identical
    if ss_tot == 0:
        r_squared = 1.0 if ss_res == 0 else 0.0
    else:
        r_squared = 1.0 - (ss_res / ss_tot)

    return float(alpha), float(a), float(r_squared)


def plot_scaling_curve(
    results: pd.DataFrame,
    metric: str = "val_loss",
    output_path: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot log-log scaling curve with power law fit.

    Args:
        results: DataFrame with 'params' and metric columns.
        metric: Column name for y-axis values.
        output_path: If provided, save figure to this path.
        title: Plot title (auto-generated if None).

    Returns:
        Matplotlib Figure with:
        - Log-log scatter of params vs metric
        - Power law fit line
        - Annotation with alpha and R²
    """
    params = results["params"].values
    errors = results[metric].values

    # Fit power law
    alpha, a, r_squared = fit_power_law(params, errors)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot data points
    ax.scatter(params, errors, s=100, c="blue", edgecolors="black", zorder=3)

    # Plot fit line
    params_sorted = np.sort(params)
    fit_line = a * params_sorted ** (-alpha)
    ax.plot(params_sorted, fit_line, "r--", linewidth=2, label=f"Fit: α={alpha:.3f}")

    # Set log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)

    if title is None:
        title = f"Scaling Curve (α={alpha:.3f}, R²={r_squared:.3f})"
    ax.set_title(title, fontsize=14)

    # Add annotation
    ax.annotate(
        f"α = {alpha:.4f}\nR² = {r_squared:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def load_experiment_results(
    experiment_name: str,
    budgets: list[str] | None = None,
    hpo_dir: str = "outputs/hpo",
) -> pd.DataFrame:
    """Load training results for an experiment across parameter budgets.

    Args:
        experiment_name: Experiment identifier (e.g., 'spy_daily_threshold_1pct').
        budgets: List of budgets to load (default: ["2M", "20M", "200M", "2B"]).
        hpo_dir: Directory containing HPO results.

    Returns:
        DataFrame with columns: budget, params, val_loss, best_value

    Raises:
        FileNotFoundError: If experiment directory doesn't exist.
    """
    if budgets is None:
        budgets = ["2M", "20M", "200M", "2B"]

    hpo_path = Path(hpo_dir)
    experiment_path = hpo_path / experiment_name

    if not experiment_path.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_path}"
        )

    results = []
    for budget in budgets:
        result_file = experiment_path / f"{budget}_best.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                results.append({
                    "budget": data.get("budget", budget),
                    "params": data.get("params", 0),
                    "best_value": data.get("best_value", 0),
                })

    return pd.DataFrame(results)


def generate_scaling_report(
    experiment_name: str,
    output_dir: str = "outputs/figures",
    hpo_dir: str = "outputs/hpo",
) -> dict:
    """Generate complete scaling analysis report.

    Creates:
    - Scaling curve plot (PNG)
    - Summary statistics (JSON)

    Args:
        experiment_name: Experiment identifier.
        output_dir: Directory for output files.
        hpo_dir: Directory containing HPO results.

    Returns:
        Dict with alpha, r_squared, and interpretation.
    """
    # Load results
    results = load_experiment_results(experiment_name, hpo_dir=hpo_dir)

    # Fit power law
    params = results["params"].values
    errors = results["best_value"].values
    alpha, a, r_squared = fit_power_law(params, errors)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate plot
    plot_path = output_path / f"{experiment_name}_scaling.png"
    fig = plot_scaling_curve(
        results,
        metric="best_value",
        output_path=str(plot_path),
        title=f"Scaling Curve: {experiment_name}",
    )
    plt.close(fig)

    # Generate interpretation
    if alpha > 0.15:
        interpretation = f"Strong scaling: 10x params yields ~{(1 - 10**(-alpha))*100:.0f}% error reduction"
    elif alpha > 0.08:
        interpretation = f"Moderate scaling: 10x params yields ~{(1 - 10**(-alpha))*100:.0f}% error reduction"
    else:
        interpretation = f"Weak scaling: 10x params yields ~{(1 - 10**(-alpha))*100:.0f}% error reduction"

    # Build report
    report = {
        "experiment": experiment_name,
        "metric": "best_value",
        "fit_results": {
            "alpha": alpha,
            "coefficient": a,
            "r_squared": r_squared,
        },
        "data_points": [
            {
                "budget": row["budget"],
                "params": int(row["params"]),
                "best_value": row["best_value"],
            }
            for _, row in results.iterrows()
        ],
        "interpretation": interpretation,
    }

    # Save JSON report
    json_path = output_path / f"{experiment_name}_scaling.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
