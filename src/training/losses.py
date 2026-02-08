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


class MildFocalLoss(nn.Module):
    """Mild Focal Loss with lower gamma for gentler focusing.

    Standard Focal Loss with gamma=2.0 can be too aggressive, completely
    ignoring easy examples. MildFocalLoss uses gamma in [0.5, 1.0] range
    for a gentler balance between BCE and strong focal focusing.

    With lower gamma:
        - Easy examples still contribute to learning
        - Hard examples get moderate emphasis
        - Better gradient flow for all samples

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        gamma: Focusing parameter in [0.5, 1.0] range. Default 0.75.
            Lower values are closer to weighted BCE.
        alpha: Weight for positive class. Default 0.5 (balanced).
            Use values < 0.5 for more negative weight, > 0.5 for more positive.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> loss_fn = MildFocalLoss(gamma=0.75, alpha=0.5)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self, gamma: float = 0.75, alpha: float = 0.5, eps: float = 1e-7
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute mild focal loss.

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
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # Compute α_t (class weight)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Mild focal term with lower gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Cross-entropy term
        ce = -torch.log(p_t)

        # Combine
        loss = alpha_t * focal_weight * ce

        return loss.mean()


class AsymmetricFocalLoss(nn.Module):
    """Asymmetric Focal Loss with separate gamma for positive and negative samples.

    Standard Focal Loss applies the same focusing parameter to both classes.
    AsymmetricFocalLoss allows independent control:
        - gamma_pos: How much to focus on hard positive examples
        - gamma_neg: How much to focus on hard negative examples

    This is useful when the importance of hard examples differs by class:
        - High gamma_pos: Focus on hard positives (recall-oriented)
        - High gamma_neg: Focus on hard negatives (precision-oriented)

    Formula:
        For y=1: FL = -α * (1-p)^γ_pos * log(p)
        For y=0: FL = -(1-α) * p^γ_neg * log(1-p)

    Args:
        gamma_pos: Focusing parameter for positive samples. Default 1.0.
        gamma_neg: Focusing parameter for negative samples. Default 2.0.
        alpha: Weight for positive class. Default 0.5.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> # Precision-focused: harder on false positives
        >>> loss_fn = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=2.0)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.alpha = alpha
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute asymmetric focal loss.

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

        # Positive samples: -α * (1-p)^γ_pos * log(p)
        pos_loss = (
            -self.alpha
            * ((1 - predictions) ** self.gamma_pos)
            * torch.log(predictions)
            * targets
        )

        # Negative samples: -(1-α) * p^γ_neg * log(1-p)
        neg_loss = (
            -(1 - self.alpha)
            * (predictions ** self.gamma_neg)
            * torch.log(1 - predictions)
            * (1 - targets)
        )

        return (pos_loss + neg_loss).mean()


class EntropyRegularizedBCE(nn.Module):
    """Binary Cross-Entropy with Entropy Regularization.

    Adds an entropy penalty to BCE to prevent probability collapse, where
    the model predicts the same probability for all samples (e.g., all ~0.5).

    The entropy term encourages confident, differentiated predictions:
        - High entropy = uncertain predictions (diverse probabilities)
        - Low entropy = confident predictions (probabilities near 0 or 1)

    Penalty is applied when entropy is BELOW the target, encouraging
    more diverse/confident predictions.

    Formula: L = BCE - λ * mean_entropy

    The negative sign means maximizing entropy (up to natural BCE minimization).

    Args:
        lambda_entropy: Weight for entropy penalty. Default 0.1.
            Higher values push predictions away from 0.5.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> loss_fn = EntropyRegularizedBCE(lambda_entropy=0.1)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, lambda_entropy: float = 0.1, eps: float = 1e-7) -> None:
        super().__init__()
        self.lambda_entropy = lambda_entropy
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute entropy-regularized BCE loss.

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
        p = predictions.clamp(self.eps, 1 - self.eps)

        # Standard BCE
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))

        # Entropy of predictions: H(p) = -p*log(p) - (1-p)*log(1-p)
        # High entropy = uncertain (p near 0.5), low entropy = confident (p near 0 or 1)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

        # We want to encourage confident predictions (low entropy)
        # But not at the expense of accuracy
        # Subtracting entropy means minimizing loss increases entropy pressure
        # But BCE term still dominates for accuracy
        loss = bce - self.lambda_entropy * entropy

        return loss.mean()


class VarianceRegularizedBCE(nn.Module):
    """Binary Cross-Entropy with Variance Regularization.

    Adds a variance penalty to BCE to directly penalize narrow prediction ranges.
    This addresses probability collapse where all predictions cluster around
    a single value (e.g., all ~0.45).

    The variance term encourages diverse predictions across the batch:
        - Low variance = all predictions similar (collapse)
        - High variance = differentiated predictions (desired)

    Formula: L = BCE - λ * var(predictions)

    The negative sign rewards higher variance in predictions.

    Args:
        lambda_var: Weight for variance penalty. Default 0.5.
            Higher values push for more diverse predictions.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> loss_fn = VarianceRegularizedBCE(lambda_var=0.5)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(self, lambda_var: float = 0.5, eps: float = 1e-7) -> None:
        super().__init__()
        self.lambda_var = lambda_var
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute variance-regularized BCE loss.

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
        p = predictions.clamp(self.eps, 1 - self.eps)

        # Standard BCE
        bce = nn.functional.binary_cross_entropy(p, targets)

        # Variance of predictions across batch
        # Subtracting variance means lower loss for higher variance (more diverse predictions)
        # Handle single-sample case: variance is undefined, so use 0
        if predictions.numel() < 2:
            variance = torch.tensor(0.0, device=predictions.device, dtype=predictions.dtype)
        else:
            variance = torch.var(predictions)

        # Combined loss: minimize BCE, maximize variance
        loss = bce - self.lambda_var * variance

        return loss


class CalibratedFocalLoss(nn.Module):
    """Focal Loss with Temperature-Scaled Calibration.

    Combines Focal Loss (for ranking) with a calibration term that penalizes
    the difference between predicted probabilities and actual class frequencies.

    The calibration term uses temperature scaling to control the sharpness
    of probability adjustments:
        - High temperature: Smoother, less aggressive calibration
        - Low temperature: Sharper, more aggressive calibration

    Formula: L = FocalLoss + λ * CalibrationLoss

    Where CalibrationLoss = |mean(p) - mean(y)|^2 (mean prediction vs actual rate)

    Args:
        gamma: Focusing parameter for focal loss. Default 2.0.
        alpha: Weight for positive class. Default 0.25.
        lambda_cal: Weight for calibration term. Default 0.1.
        temperature: Temperature for calibration scaling. Default 1.0.
        eps: Small constant for numerical stability. Default 1e-7.

    Example:
        >>> loss_fn = CalibratedFocalLoss(gamma=2.0, lambda_cal=0.1)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        lambda_cal: float = 0.1,
        temperature: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_cal = lambda_cal
        self.temperature = temperature
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute calibrated focal loss.

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

        # === Focal Loss Component ===
        # p_t = p if y=1, else 1-p
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # α_t = α if y=1, else 1-α
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal weight and cross-entropy
        focal_weight = (1 - p_t) ** self.gamma
        ce = -torch.log(p_t)
        focal_loss = (alpha_t * focal_weight * ce).mean()

        # === Calibration Component ===
        # Mean prediction should match mean target (actual positive rate)
        # Temperature controls the sharpness of this penalty
        mean_pred = predictions.mean()
        mean_target = targets.mean()

        # Squared difference scaled by temperature
        # Higher temp = smaller gradients (gentler calibration)
        calibration_loss = ((mean_pred - mean_target) / self.temperature) ** 2

        # Combined loss
        loss = focal_loss + self.lambda_cal * calibration_loss

        return loss


# ============================================================================
# NOISE-ROBUST LOSS FUNCTIONS (NR Category)
# For handling label noise in financial data where targets may be unreliable.
# ============================================================================


class BootstrapLoss(nn.Module):
    """Bootstrap Loss for training with noisy labels.

    Blends the target label with the model's own prediction, effectively
    allowing the model to "correct" potentially noisy labels over time.

    Formula: L = -[(β * y + (1-β) * p) * log(p) + (1 - β * y - (1-β) * p) * log(1-p)]

    Where:
        y = target label
        p = model prediction
        β = blend coefficient (1.0 = trust labels, 0.0 = trust model)

    As training progresses, the model becomes more confident and the bootstrap
    term helps filter out noisy labels that contradict confident predictions.

    Reference: Reed et al. "Training Deep Neural Networks on Noisy Labels
               with Bootstrapping" (2014)

    Args:
        beta: Blend coefficient in [0, 1]. Default 0.8.
            Higher values trust the original labels more.
            Typical: 0.6-0.9 depending on expected noise level.
        mode: "soft" or "hard". Default "soft".
            - soft: Uses model's probability p directly
            - hard: Uses thresholded prediction (p >= 0.5 -> 1, else 0)
        eps: Numerical stability constant. Default 1e-7.

    Example:
        >>> loss_fn = BootstrapLoss(beta=0.8, mode="soft")
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Some noisy labels
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        beta: float = 0.8,
        mode: str = "soft",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if mode not in ("soft", "hard"):
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode}")
        self.beta = beta
        self.mode = mode
        self.eps = eps

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute bootstrap loss.

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
        p = predictions.clamp(self.eps, 1 - self.eps)

        # Bootstrap target
        if self.mode == "hard":
            # Hard bootstrap: use thresholded predictions
            p_bootstrap = (predictions >= 0.5).float()
        else:
            # Soft bootstrap: use probabilities directly
            p_bootstrap = predictions

        # Blended target: β * original_label + (1-β) * model_prediction
        blended_target = self.beta * targets + (1 - self.beta) * p_bootstrap

        # Binary cross-entropy with blended target
        loss = -(
            blended_target * torch.log(p)
            + (1 - blended_target) * torch.log(1 - p)
        )

        return loss.mean()


class ForwardCorrectionLoss(nn.Module):
    """Forward Correction Loss using estimated noise transition matrix.

    Corrects for label noise by estimating how clean labels are flipped
    to noisy labels. Uses a transition matrix T where T[i,j] = P(noisy=j|clean=i).

    For binary classification:
        T = [[1-e0, e0],
             [e1, 1-e1]]
    Where e0 = P(noisy=1|clean=0) and e1 = P(noisy=0|clean=1).

    The loss is computed using the forward-corrected probabilities:
        p_noisy = T^T @ p_clean

    Reference: Patrini et al. "Making Deep Neural Networks Robust to Label
               Noise: a Loss Correction Approach" (2017)

    Args:
        noise_rate_0: Probability of clean=0 being labeled as 1. Default 0.1.
        noise_rate_1: Probability of clean=1 being labeled as 0. Default 0.1.
        eps: Numerical stability constant. Default 1e-7.

    Example:
        >>> loss_fn = ForwardCorrectionLoss(noise_rate_0=0.1, noise_rate_1=0.15)
        >>> predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        >>> targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        noise_rate_0: float = 0.1,
        noise_rate_1: float = 0.1,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not 0.0 <= noise_rate_0 < 0.5:
            raise ValueError(f"noise_rate_0 must be in [0, 0.5), got {noise_rate_0}")
        if not 0.0 <= noise_rate_1 < 0.5:
            raise ValueError(f"noise_rate_1 must be in [0, 0.5), got {noise_rate_1}")
        self.noise_rate_0 = noise_rate_0
        self.noise_rate_1 = noise_rate_1
        self.eps = eps

        # Build transition matrix
        # T[i,j] = P(noisy=j | clean=i)
        # T = [[1-e0, e0],
        #      [e1, 1-e1]]
        self.register_buffer(
            "transition_matrix",
            torch.tensor([
                [1 - noise_rate_0, noise_rate_0],
                [noise_rate_1, 1 - noise_rate_1],
            ]),
        )

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute forward-corrected loss.

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
        p = predictions.clamp(self.eps, 1 - self.eps)

        # Stack predictions as [P(y=0), P(y=1)]
        p_clean = torch.stack([1 - p, p], dim=1)  # (N, 2)

        # Forward correction: p_noisy = T^T @ p_clean
        # This gives P(noisy_label | features)
        T = self.transition_matrix.to(p.device)
        p_noisy = torch.matmul(p_clean, T)  # (N, 2)

        # Clamp for stability
        p_noisy = p_noisy.clamp(self.eps, 1 - self.eps)

        # Cross-entropy with noisy label probabilities
        # For target y, use p_noisy[:, y]
        targets_long = targets.long()
        p_target = p_noisy[torch.arange(len(targets)), targets_long]

        loss = -torch.log(p_target)

        return loss.mean()


class ConfidenceLearningLoss(nn.Module):
    """Confidence Learning Loss for identifying and handling mislabeled samples.

    Uses confident predictions to identify potentially mislabeled samples
    and either down-weights them or prunes them from the batch.

    The key insight is that a well-trained model will disagree with mislabeled
    samples more strongly than correctly labeled ones.

    Args:
        threshold: Confidence threshold for identifying clean samples. Default 0.7.
            Samples where |prediction - label| > threshold are considered potentially
            mislabeled.
        mode: "weight" or "prune". Default "weight".
            - weight: Down-weight suspected mislabeled samples
            - prune: Exclude suspected mislabeled samples from loss
        weight_factor: Weight for suspected mislabeled samples (mode="weight").
            Default 0.1 (90% down-weighting).
        warmup_epochs: Number of epochs before applying the correction.
            Default 5 (allow model to learn before trusting its confidence).
        eps: Numerical stability constant. Default 1e-7.

    Example:
        >>> loss_fn = ConfidenceLearningLoss(threshold=0.7, mode="weight")
        >>> predictions = torch.tensor([0.95, 0.85, 0.15, 0.05])
        >>> targets = torch.tensor([1.0, 0.0, 0.0, 1.0])  # Second and fourth may be wrong
        >>> loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        mode: str = "weight",
        weight_factor: float = 0.1,
        warmup_epochs: int = 5,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.weight_factor = weight_factor
        self.warmup_epochs = warmup_epochs
        self.eps = eps
        self._current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for warmup tracking.

        Args:
            epoch: Current training epoch.
        """
        self._current_epoch = epoch

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute confidence learning loss.

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
        p = predictions.clamp(self.eps, 1 - self.eps)

        # Standard BCE
        bce = -(targets * torch.log(p) + (1 - targets) * torch.log(1 - p))

        # During warmup, just return standard BCE
        if self._current_epoch < self.warmup_epochs:
            return bce.mean()

        # Confidence: how sure is the model about the label?
        # For correct labels: high prediction for positive, low for negative
        # Agreement = p if target=1, (1-p) if target=0
        agreement = predictions * targets + (1 - predictions) * (1 - targets)

        # Low agreement = potential mislabel
        is_mislabeled = agreement < (1 - self.threshold)

        if self.mode == "prune":
            # Exclude suspected mislabeled samples
            is_clean = ~is_mislabeled
            if is_clean.sum() == 0:
                # Fallback: keep all if nothing would remain
                return bce.mean()
            return bce[is_clean].mean()
        else:
            # Weight mode: down-weight suspected mislabeled samples
            weights = torch.where(
                is_mislabeled,
                torch.tensor(self.weight_factor, device=bce.device),
                torch.tensor(1.0, device=bce.device),
            )
            return (bce * weights).sum() / weights.sum()
