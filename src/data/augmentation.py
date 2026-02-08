"""Data augmentation transforms for financial time-series.

Provides transform classes for augmenting time-series data during training
to improve model robustness and generalization.

Available transforms:
- JitterTransform: Add Gaussian noise to features
- ScaleTransform: Random scaling of features
- MixupTransform: Interpolate between samples (applied at batch level)
- TimeWarpTransform: Time-domain warping using DTW principles

Usage:
    transform = get_augmentation_transform(spec)
    if transform is not None:
        augmented_x = transform(x)  # x: (seq_len, n_features)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from typing import Any


class BaseTransform(ABC):
    """Base class for time-series augmentation transforms."""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Apply transform to input tensor.

        Args:
            x: Input tensor of shape (seq_len, n_features).

        Returns:
            Transformed tensor of same shape.
        """
        pass


class JitterTransform(BaseTransform):
    """Add Gaussian noise to features.

    This simple augmentation adds random noise to help the model become
    robust to small variations in feature values.

    Args:
        std: Standard deviation of Gaussian noise. Default 0.01.
            Typical values: 0.005 (subtle), 0.01 (moderate), 0.02 (aggressive)
        prob: Probability of applying the transform. Default 0.5.

    Example:
        >>> transform = JitterTransform(std=0.01, prob=0.5)
        >>> x = torch.randn(60, 100)  # 60 timesteps, 100 features
        >>> x_aug = transform(x)
    """

    def __init__(self, std: float = 0.01, prob: float = 0.5) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, x: Tensor) -> Tensor:
        """Apply jitter transform.

        Args:
            x: Input tensor of shape (seq_len, n_features).

        Returns:
            Jittered tensor of same shape.
        """
        if torch.rand(1).item() > self.prob:
            return x

        noise = torch.randn_like(x) * self.std
        return x + noise


class ScaleTransform(BaseTransform):
    """Random scaling of features.

    Multiplies features by a random scale factor drawn from a uniform
    distribution [1-scale_range, 1+scale_range].

    Args:
        scale_range: Range for scaling factor. Default 0.1.
            Scale factor is drawn from [1-range, 1+range].
            Examples: 0.1 gives [0.9, 1.1], 0.2 gives [0.8, 1.2]
        prob: Probability of applying the transform. Default 0.5.
        per_feature: If True, apply different scale to each feature.
            If False, apply same scale to all features. Default False.

    Example:
        >>> transform = ScaleTransform(scale_range=0.1, prob=0.5)
        >>> x = torch.randn(60, 100)
        >>> x_aug = transform(x)
    """

    def __init__(
        self,
        scale_range: float = 0.1,
        prob: float = 0.5,
        per_feature: bool = False,
    ) -> None:
        self.scale_range = scale_range
        self.prob = prob
        self.per_feature = per_feature

    def __call__(self, x: Tensor) -> Tensor:
        """Apply scale transform.

        Args:
            x: Input tensor of shape (seq_len, n_features).

        Returns:
            Scaled tensor of same shape.
        """
        if torch.rand(1).item() > self.prob:
            return x

        if self.per_feature:
            # Different scale for each feature
            n_features = x.shape[-1]
            scale = 1 + (torch.rand(n_features) * 2 - 1) * self.scale_range
            scale = scale.to(x.device)
        else:
            # Same scale for all features
            scale = 1 + (torch.rand(1).item() * 2 - 1) * self.scale_range

        return x * scale


class MixupTransform(BaseTransform):
    """Mixup augmentation for time-series.

    Mixup creates virtual training examples by interpolating between
    pairs of samples. This is applied at the batch level in the training
    loop, not per-sample.

    Note: This transform stores parameters but actual mixup is applied
    during training by mixing features and labels together.

    Args:
        alpha: Mixup interpolation coefficient. Default 0.2.
            Higher values create more interpolated samples.
            Typical: 0.1 (subtle), 0.2 (moderate), 0.4 (aggressive)
        prob: Probability of applying mixup to a batch. Default 0.5.

    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)

    Example:
        >>> # In training loop:
        >>> transform = MixupTransform(alpha=0.2)
        >>> x_mixed, y_mixed, lam = transform.apply_batch(x_batch, y_batch)
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5) -> None:
        self.alpha = alpha
        self.prob = prob

    def __call__(self, x: Tensor) -> Tensor:
        """Mixup is a batch-level operation; single sample pass-through.

        For proper mixup, use apply_batch() in the training loop.

        Args:
            x: Input tensor.

        Returns:
            Unchanged tensor.
        """
        # Single sample - no mixup possible
        return x

    def apply_batch(
        self, x: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor, float]:
        """Apply mixup to a batch of samples.

        Args:
            x: Batch of inputs, shape (batch, seq_len, n_features).
            y: Batch of targets, shape (batch, 1) or (batch,).

        Returns:
            Tuple of (mixed_x, mixed_y, lambda) where lambda is the
            interpolation coefficient used.
        """
        if torch.rand(1).item() > self.prob:
            return x, y, 1.0

        batch_size = x.size(0)

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation of batch indices
        index = torch.randperm(batch_size, device=x.device)

        # Mix samples
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]

        return mixed_x, mixed_y, lam


class TimeWarpTransform(BaseTransform):
    """Time-domain warping for time-series augmentation.

    Warps the temporal dimension of the input by stretching or
    compressing different segments of the sequence.

    Args:
        warp_factor: Maximum warping factor. Default 0.1.
            Values >0 allow up to (1+factor) stretch or (1-factor) compression.
        n_knots: Number of knot points for piecewise warping. Default 4.
        prob: Probability of applying the transform. Default 0.5.

    Example:
        >>> transform = TimeWarpTransform(warp_factor=0.1, prob=0.5)
        >>> x = torch.randn(60, 100)
        >>> x_warped = transform(x)
    """

    def __init__(
        self,
        warp_factor: float = 0.1,
        n_knots: int = 4,
        prob: float = 0.5,
    ) -> None:
        self.warp_factor = warp_factor
        self.n_knots = n_knots
        self.prob = prob

    def __call__(self, x: Tensor) -> Tensor:
        """Apply time warp transform.

        Args:
            x: Input tensor of shape (seq_len, n_features).

        Returns:
            Time-warped tensor of same shape.
        """
        if torch.rand(1).item() > self.prob:
            return x

        seq_len, n_features = x.shape
        device = x.device

        # Generate random warp path
        # Original time indices
        orig_time = torch.linspace(0, 1, seq_len, device=device)

        # Generate random knot displacements
        # Knots are evenly spaced with random vertical shifts
        knot_times = torch.linspace(0, 1, self.n_knots + 2, device=device)
        # First and last knot stay fixed at 0 and 1
        displacements = torch.zeros(self.n_knots + 2, device=device)
        displacements[1:-1] = (
            torch.rand(self.n_knots, device=device) * 2 - 1
        ) * self.warp_factor

        # Warped knot values (cumulative to maintain monotonicity)
        warped_knots = knot_times + displacements
        # Ensure monotonicity and bounds
        warped_knots = torch.clamp(warped_knots, 0, 1)
        warped_knots, _ = torch.sort(warped_knots)
        warped_knots[0] = 0
        warped_knots[-1] = 1

        # Interpolate to get warped time for each original position
        # Use simple linear interpolation between knots
        warped_time = torch.zeros(seq_len, device=device)
        for i in range(len(knot_times) - 1):
            mask = (orig_time >= knot_times[i]) & (orig_time <= knot_times[i + 1])
            if mask.any():
                # Linear interpolation within segment
                t = (orig_time[mask] - knot_times[i]) / (
                    knot_times[i + 1] - knot_times[i] + 1e-8
                )
                warped_time[mask] = (
                    warped_knots[i] + t * (warped_knots[i + 1] - warped_knots[i])
                )

        # Convert warped time back to indices
        warped_indices = warped_time * (seq_len - 1)

        # Interpolate features at warped positions
        x_warped = self._interpolate_features(x, warped_indices)

        return x_warped

    def _interpolate_features(self, x: Tensor, indices: Tensor) -> Tensor:
        """Interpolate features at fractional indices.

        Uses linear interpolation between neighboring samples.

        Args:
            x: Input tensor of shape (seq_len, n_features).
            indices: Fractional indices of shape (seq_len,).

        Returns:
            Interpolated tensor of same shape as x.
        """
        seq_len = x.shape[0]

        # Get floor and ceil indices
        idx_floor = torch.floor(indices).long()
        idx_ceil = torch.ceil(indices).long()

        # Clamp to valid range
        idx_floor = torch.clamp(idx_floor, 0, seq_len - 1)
        idx_ceil = torch.clamp(idx_ceil, 0, seq_len - 1)

        # Interpolation weights
        weights = indices - idx_floor.float()
        weights = weights.unsqueeze(-1)  # (seq_len, 1)

        # Linear interpolation
        x_warped = x[idx_floor] * (1 - weights) + x[idx_ceil] * weights

        return x_warped


class ComposedTransform(BaseTransform):
    """Compose multiple transforms together.

    Applies transforms in sequence. Each transform may or may not be
    applied based on its own probability.

    Args:
        transforms: List of transform instances to compose.

    Example:
        >>> jitter = JitterTransform(std=0.01)
        >>> scale = ScaleTransform(scale_range=0.1)
        >>> composed = ComposedTransform([jitter, scale])
        >>> x_aug = composed(x)
    """

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, x: Tensor) -> Tensor:
        """Apply all transforms in sequence.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        for transform in self.transforms:
            x = transform(x)
        return x


def get_augmentation_transform(
    augmentation_type: str | None,
    augmentation_params: dict[str, Any] | None = None,
) -> BaseTransform | None:
    """Factory function to create augmentation transform from experiment spec.

    Args:
        augmentation_type: Type of augmentation ("jitter", "scale", "mixup",
            "timewarp", "combined"). None returns None.
        augmentation_params: Parameters for the transform. Available params
            depend on augmentation_type:
            - jitter: {"std": 0.01, "prob": 0.5}
            - scale: {"scale_range": 0.1, "prob": 0.5, "per_feature": False}
            - mixup: {"alpha": 0.2, "prob": 0.5}
            - timewarp: {"warp_factor": 0.1, "n_knots": 4, "prob": 0.5}
            - combined: {"jitter_std": 0.01, "mixup_alpha": 0.2, ...}

    Returns:
        Transform instance, or None if augmentation_type is None.

    Raises:
        ValueError: If augmentation_type is unknown.

    Example:
        >>> transform = get_augmentation_transform(
        ...     augmentation_type="jitter",
        ...     augmentation_params={"std": 0.01, "prob": 0.5}
        ... )
    """
    if augmentation_type is None:
        return None

    params = augmentation_params or {}

    if augmentation_type == "jitter":
        return JitterTransform(
            std=params.get("std", 0.01),
            prob=params.get("prob", 0.5),
        )

    elif augmentation_type == "scale":
        return ScaleTransform(
            scale_range=params.get("scale_range", 0.1),
            prob=params.get("prob", 0.5),
            per_feature=params.get("per_feature", False),
        )

    elif augmentation_type == "mixup":
        return MixupTransform(
            alpha=params.get("alpha", 0.2),
            prob=params.get("prob", 0.5),
        )

    elif augmentation_type == "timewarp":
        return TimeWarpTransform(
            warp_factor=params.get("warp_factor", 0.1),
            n_knots=params.get("n_knots", 4),
            prob=params.get("prob", 0.5),
        )

    elif augmentation_type == "combined":
        # Combine jitter and mixup (the most effective combination)
        transforms = []
        if params.get("jitter_std", 0.01) > 0:
            transforms.append(
                JitterTransform(
                    std=params.get("jitter_std", 0.01),
                    prob=params.get("jitter_prob", 0.5),
                )
            )
        # Note: Mixup is a batch-level operation, handled separately
        # Include scale if specified
        if params.get("scale_range", 0) > 0:
            transforms.append(
                ScaleTransform(
                    scale_range=params.get("scale_range", 0.1),
                    prob=params.get("scale_prob", 0.5),
                )
            )
        return ComposedTransform(transforms) if transforms else None

    else:
        raise ValueError(f"Unknown augmentation_type: {augmentation_type}")
