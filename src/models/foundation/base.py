"""Abstract base class for foundation model wrappers."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class FoundationModel(ABC):
    """Abstract base class for foundation model wrappers.

    All foundation model implementations should inherit from this class
    to ensure consistent interface for training and inference.
    """

    @abstractmethod
    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pre-trained weights from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint file.
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Output tensor with predictions.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return model configuration dictionary."""
        pass
