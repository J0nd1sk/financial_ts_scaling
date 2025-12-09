"""Utility functions for model operations."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count total parameters in a PyTorch model.

    Args:
        model: PyTorch model to count parameters for.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total number of parameters as an integer.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
