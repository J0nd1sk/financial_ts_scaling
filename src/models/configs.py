"""Configuration loading for PatchTST models."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.models.patchtst import PatchTSTConfig


def load_patchtst_config(budget: str) -> PatchTSTConfig:
    """Load a PatchTST configuration for a given parameter budget.

    Args:
        budget: Parameter budget identifier ("2m", "20m", or "200m").

    Returns:
        PatchTSTConfig dataclass with the configuration.

    Raises:
        ValueError: If budget is not recognized.
        FileNotFoundError: If config file doesn't exist.
    """
    valid_budgets = ("2m", "20m", "200m")
    if budget.lower() not in valid_budgets:
        raise ValueError(f"Budget must be one of {valid_budgets}, got '{budget}'")

    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "model" / f"patchtst_{budget.lower()}.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return PatchTSTConfig(**config_dict)
