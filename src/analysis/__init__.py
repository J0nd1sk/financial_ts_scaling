"""Analysis module for scaling law experiments."""

from .scaling_curves import (
    fit_power_law,
    plot_scaling_curve,
    load_experiment_results,
    generate_scaling_report,
)

__all__ = [
    "fit_power_law",
    "plot_scaling_curve",
    "load_experiment_results",
    "generate_scaling_report",
]
