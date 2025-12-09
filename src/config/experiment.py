"""Experiment configuration dataclass and loader.

This module defines the ExperimentConfig dataclass which specifies WHAT experiment
we're running (task, data, timescale) but NOT how to run it optimally (batch size,
learning rate, etc.). Those execution parameters are discovered/tuned separately.

See docs/config_architecture.md for full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Valid values for categorical fields
VALID_TASKS = frozenset({
    "direction",
    "threshold_1pct",
    "threshold_2pct",
    "threshold_3pct",
    "threshold_5pct",
    "regression",
})

VALID_TIMESCALES = frozenset({
    "daily",
    "2d",
    "3d",
    "5d",
    "weekly",
    "2wk",
    "monthly",
})


@dataclass
class ExperimentConfig:
    """Configuration for a scaling law experiment.

    Defines WHAT experiment we're running:
    - Which task (direction prediction, threshold classification, regression)
    - Which timescale (daily, weekly, etc.)
    - Which data source (path to processed features)
    - Sequence parameters (context_length, horizon)

    Does NOT include execution parameters like batch_size, learning_rate, epochs.
    Those are discovered via batch size discovery and HPO, stored separately.

    Attributes:
        seed: Random seed for reproducibility.
        data_path: Path to processed features parquet file.
        task: Prediction task type.
        timescale: Time resolution of predictions.
        context_length: Number of time steps in input sequence.
        horizon: Number of time steps ahead to predict.
        wandb_project: W&B project name (None to disable).
        mlflow_experiment: MLflow experiment name (None to disable).
    """

    data_path: str
    task: str
    timescale: str
    seed: int = 42
    context_length: int = 60
    horizon: int = 5
    wandb_project: str | None = None
    mlflow_experiment: str | None = None


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If required fields are missing, values are invalid,
            or data_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Validate required fields
    _validate_required_fields(data)

    # Validate categorical fields
    _validate_task(data.get("task"))
    _validate_timescale(data.get("timescale"))

    # Validate data path exists
    _validate_data_path(data.get("data_path"))

    # Build config with defaults
    return ExperimentConfig(
        seed=data.get("seed", 42),
        data_path=data["data_path"],
        task=data["task"],
        timescale=data["timescale"],
        context_length=data.get("context_length", 60),
        horizon=data.get("horizon", 5),
        wandb_project=data.get("wandb_project"),
        mlflow_experiment=data.get("mlflow_experiment"),
    )


def _validate_required_fields(data: dict[str, Any]) -> None:
    """Check that all required fields are present."""
    required = ["data_path", "task", "timescale"]
    missing = [f for f in required if f not in data or data[f] is None]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")


def _validate_task(task: str | None) -> None:
    """Validate task is one of the allowed values."""
    if task not in VALID_TASKS:
        raise ValueError(
            f"Invalid task '{task}'. Must be one of: {sorted(VALID_TASKS)}"
        )


def _validate_timescale(timescale: str | None) -> None:
    """Validate timescale is one of the allowed values."""
    if timescale not in VALID_TIMESCALES:
        raise ValueError(
            f"Invalid timescale '{timescale}'. Must be one of: {sorted(VALID_TIMESCALES)}"
        )


def _validate_data_path(data_path: str | None) -> None:
    """Validate data_path exists on filesystem."""
    if data_path is None:
        raise ValueError("data_path is required")
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"data_path does not exist: {data_path}")
