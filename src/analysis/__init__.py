"""Analysis module for scaling law experiments."""

from .scaling_curves import (
    fit_power_law,
    plot_scaling_curve,
    load_experiment_results,
    generate_scaling_report,
)
from .aggregate_results import (
    aggregate_hpo_results,
    aggregate_training_results,
    summarize_experiment,
    export_results_csv,
    generate_experiment_summary_report,
)

__all__ = [
    # scaling_curves
    "fit_power_law",
    "plot_scaling_curve",
    "load_experiment_results",
    "generate_scaling_report",
    # aggregate_results
    "aggregate_hpo_results",
    "aggregate_training_results",
    "summarize_experiment",
    "export_results_csv",
    "generate_experiment_summary_report",
]
