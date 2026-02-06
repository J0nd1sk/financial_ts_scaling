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

        # Handle degenerate cases (maintain computation graph for gradient flow)
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return predictions.mean() * 0 + 0.5

        # Compute all pairwise differences: diff[i,j] = neg[j] - pos[i]
        # Shape: (n_pos, n_neg)
        diff = neg_preds.unsqueeze(0) - pos_preds.unsqueeze(1)

        # Sigmoid of differences:
        # - When pos > neg (correct): diff < 0, sigmoid < 0.5
        # - When neg > pos (wrong): diff > 0, sigmoid > 0.5
        # Loss is mean of all sigmoids, so lower is better
        loss = torch.sigmoid(self.gamma * diff).mean()

        return loss


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    Focal Loss down-weights easy examples and focuses learning on hard examples.
    This addresses class imbalance by reducing the contribution of well-classified
    samples to the total loss.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        p_t = p if y=1, else 1-p (probability of correct class)
        α_t = α if y=1, else 1-α (class weight)
        γ = focusing parameter (higher = more focus on hard examples)

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
            γ=0 is equivalent to weighted BCE. Default 2.0 is standard.
        alpha: Weight for positive class. α=0.25 is common for imbalanced data.
            α=0.5 gives equal weight to both classes. Default 0.25.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            predictions: Model predictions (probabilities), shape (N,) or (N, 1).
            targets: Binary targets (0 or 1), shape (N,) or (N, 1).

        Returns:
            Scalar loss tensor.
        """
        # Flatten to 1D
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Clamp predictions for numerical stability
        predictions = predictions.clamp(self.eps, 1 - self.eps)

        # Compute p_t (probability of correct class)
        # p_t = p if y=1, else 1-p
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # Compute α_t (class weight)
        # α_t = α if y=1, else 1-α
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal term: (1 - p_t)^γ
        # Down-weights easy examples (high p_t) and up-weights hard examples (low p_t)
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy term: -log(p_t)
        ce = -torch.log(p_t)

        # Combine: α_t * (1 - p_t)^γ * (-log(p_t))
        loss = alpha_t * focal_weight * ce

        return loss.mean()


class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss for class imbalance.

    Simpler alternative to FocalLoss - just weights positive class more heavily.
    This encourages the model to focus on correctly classifying the minority
    positive class by penalizing false negatives more than false positives.

    Formula:
        L = -[pos_weight * y * log(p) + (1-y) * log(1-p)]

    Args:
        pos_weight: Weight for positive class. Default 1.0 (no weighting).
            Typical: n_negative / n_positive (e.g., 4.0 for 20% positive rate)
        eps: Numerical stability constant. Default 1e-7.

    Example:
        >>> loss_fn = WeightedBCELoss(pos_weight=4.0)  # For ~20% positive rate
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, pos_weight: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted BCE loss.

        Args:
            predictions: Model predictions (probabilities), shape (N,) or (N, 1).
            targets: Binary targets (0 or 1), shape (N,) or (N, 1).

        Returns:
            Scalar loss tensor.
        """
        # Flatten to 1D
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Clamp predictions for numerical stability
        predictions = predictions.clamp(self.eps, 1 - self.eps)

        # Weighted BCE: weight positive class by pos_weight
        pos_loss = -targets * torch.log(predictions) * self.pos_weight
        neg_loss = -(1 - targets) * torch.log(1 - predictions)

        return (pos_loss + neg_loss).mean()


class WeightedSumLoss(nn.Module):
    """Weighted sum of BCE and SoftAUC losses for multi-objective optimization.

    Combines binary cross-entropy (for probability calibration) with SoftAUC
    (for ranking optimization) using a tunable weight parameter.

    Formula: L = α * BCE + (1 - α) * SoftAUC

    Where:
        α = 1.0: Pure BCE (probability calibration)
        α = 0.5: Balanced (default)
        α = 0.0: Pure SoftAUC (ranking optimization)

    This allows trading off between:
    - BCE: Well-calibrated probabilities, good for threshold-based decisions
    - SoftAUC: Good ranking/separation, directly optimizes AUC-ROC

    Args:
        alpha: Weight for BCE loss. Must be in [0, 1]. Default 0.5.
        gamma: Steepness parameter for SoftAUC component. Default 2.0.

    Example:
        >>> loss_fn = WeightedSumLoss(alpha=0.7)  # 70% BCE, 30% SoftAUC
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.softauc = SoftAUCLoss(gamma=gamma)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute weighted sum of BCE and SoftAUC losses.

        Args:
            predictions: Model predictions (probabilities), shape (N,) or (N, 1).
            targets: Binary targets (0 or 1), shape (N,) or (N, 1).

        Returns:
            Scalar loss tensor.
        """
        # Flatten to 1D for consistent handling
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        bce_loss = self.bce(predictions, targets)
        softauc_loss = self.softauc(predictions, targets)

        return self.alpha * bce_loss + (1 - self.alpha) * softauc_loss


class LabelSmoothingBCELoss(nn.Module):
    """Binary Cross-Entropy with Label Smoothing.

    Label smoothing prevents overconfident predictions by softening the target
    labels. Instead of hard 0/1 targets, uses soft targets that are slightly
    pulled toward 0.5.

    For target y and smoothing parameter ε:
        - y=1 becomes 1-ε (e.g., 0.9 for ε=0.1)
        - y=0 becomes ε (e.g., 0.1 for ε=0.1)

    This encourages the model to be less certain, which can improve calibration
    and reduce overfitting.

    Args:
        epsilon: Smoothing parameter in [0, 1]. ε=0 is standard BCE.
            Typical values are 0.1 or 0.2. Default 0.1.

    Example:
        >>> loss_fn = LabelSmoothingBCELoss(epsilon=0.1)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute label-smoothed BCE loss.

        Args:
            predictions: Model predictions (probabilities), shape (N,) or (N, 1).
            targets: Binary targets (0 or 1), shape (N,) or (N, 1).

        Returns:
            Scalar loss tensor.
        """
        # Flatten to 1D
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Apply label smoothing
        # y=1 -> 1-ε, y=0 -> ε
        smoothed_targets = targets * (1 - self.epsilon) + (1 - targets) * self.epsilon

        # Standard BCE with smoothed targets
        loss = nn.functional.binary_cross_entropy(predictions, smoothed_targets)

        return loss
