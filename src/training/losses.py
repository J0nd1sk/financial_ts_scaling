"""Custom loss functions for financial time-series models.

Provides alternative loss functions to address issues like prior collapse
with standard BCE loss on imbalanced classification tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SoftAUCLoss(nn.Module):
    """Differentiable approximation of AUC loss.

    Computes average sigmoid of (neg_score - pos_score) over all pairs.
    When model correctly ranks positives above negatives, loss approaches 0.
    When model incorrectly ranks negatives above positives, loss approaches 1.

    This loss directly optimizes ranking, avoiding the prior collapse problem
    where BCE loss causes models to predict the class prior for all samples.

    Args:
        gamma: Steepness of sigmoid. Higher values create sharper ranking
            boundaries. Default 2.0 provides good gradient flow.

    Example:
        >>> loss_fn = SoftAUCLoss(gamma=2.0)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)

    Note:
        Complexity is O(n_pos × n_neg). For very large batches with many
        samples of each class, consider subsampling.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute soft AUC loss.

        Args:
            predictions: Model predictions, shape (N,) or (N, 1).
            targets: Binary targets (0 or 1), shape (N,) or (N, 1).

        Returns:
            Scalar loss tensor.
        """
        # Flatten to 1D
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Separate positive and negative predictions
        pos_mask = targets == 1
        neg_mask = targets == 0

        pos_preds = predictions[pos_mask]
        neg_preds = predictions[neg_mask]

        # Handle degenerate cases
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.5, device=predictions.device, dtype=predictions.dtype)

        # Compute all pairwise differences: diff[i,j] = neg[j] - pos[i]
        # Shape: (n_pos, n_neg)
        diff = neg_preds.unsqueeze(0) - pos_preds.unsqueeze(1)

        # Sigmoid of differences:
        # - When pos > neg (correct): diff < 0, sigmoid < 0.5
        # - When neg > pos (wrong): diff > 0, sigmoid > 0.5
        # Loss is mean of all sigmoids, so lower is better
        loss = torch.sigmoid(self.gamma * diff).mean()

        return loss
