"""Lag-Llama wrapper for binary classification.

Adapts the pre-trained Lag-Llama probabilistic forecaster for threshold-based
binary classification tasks in financial time series.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood

from src.models.foundation.base import FoundationModel


def _differentiable_studentt_cdf(
    z: torch.Tensor,
    df: torch.Tensor,
) -> torch.Tensor:
    """Differentiable StudentT CDF using normal approximation.

    Uses the approximation: StudentT_df(z) ≈ Normal(z * sqrt(df / (df + 2)))

    This approximation is exact as df → ∞ and accurate for df > 5.
    For financial applications (typical df ~5-10), this provides sufficient
    accuracy while maintaining full gradient flow.

    Args:
        z: Standardized values (already scaled by loc and scale).
        df: Degrees of freedom tensor.

    Returns:
        CDF values as tensor.
    """
    # Ensure numerical stability
    eps = 1e-8
    df_safe = torch.clamp(df, min=1.0)

    # Scale factor for normal approximation
    # As df → ∞, scale → 1 (StudentT → Normal)
    # For smaller df, we scale down z to account for heavier tails
    scale = torch.sqrt(df_safe / (df_safe + 2 + eps))

    # Apply scaled normal CDF (ndtr = standard normal CDF)
    scaled_z = z * scale
    cdf = torch.special.ndtr(scaled_z)

    return cdf


def distribution_to_threshold_prob(
    df: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Convert StudentT distribution parameters to P(X > threshold).

    Uses a differentiable implementation based on the incomplete beta function.

    Args:
        df: Degrees of freedom tensor.
        loc: Location (mean) tensor.
        scale: Scale tensor.
        threshold: Threshold value to compute probability above.

    Returns:
        Probability tensor of shape matching inputs.
    """
    # Standardize threshold: z = (threshold - loc) / scale
    eps = 1e-8
    z = (threshold - loc) / (scale + eps)

    # Compute CDF using differentiable implementation
    prob_below = _differentiable_studentt_cdf(z, df)

    # P(X > threshold) = 1 - CDF(threshold)
    prob_above = 1 - prob_below
    return prob_above


class LagLlamaWrapper(FoundationModel, nn.Module):
    """Wrapper for Lag-Llama foundation model adapted for binary classification.

    Lag-Llama is a decoder-only transformer pre-trained on diverse time series
    data. It outputs a StudentT distribution for next-step forecasting. This
    wrapper adapts it for threshold-based binary classification by:

    1. Loading pre-trained weights from checkpoint
    2. Optionally projecting multivariate input to univariate (Close prices)
    3. Converting StudentT distribution to P(return > threshold)
    4. Supporting fine-tuning modes (head-only or full)

    Attributes:
        context_length: Number of historical time steps to use.
        prediction_length: Forecast horizon (always 1 for classification).
        threshold: Return threshold for binary classification.
        num_features: Number of input features (1=univariate, >1=multivariate).
        fine_tune_mode: Whether to freeze backbone ("head_only") or not ("full").

    Example:
        >>> wrapper = LagLlamaWrapper(
        ...     context_length=80,
        ...     prediction_length=1,
        ...     threshold=0.01,
        ...     num_features=25,
        ... )
        >>> wrapper.load_pretrained("models/pretrained/lag-llama.ckpt")
        >>> x = torch.randn(8, 80, 25)  # batch, seq, features
        >>> prob = wrapper(x)  # (8, 1) - P(return > 1%)
    """

    # Constants from pre-trained model
    N_TIME_FEATURES = 6  # Required time features for Lag-Llama
    MODEL_CONTEXT_LENGTH = 32  # Built-in context length of the model

    def __init__(
        self,
        context_length: int,
        prediction_length: int = 1,
        threshold: float = 0.01,
        num_features: int = 1,
        fine_tune_mode: Literal["head_only", "full"] = "full",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Lag-Llama wrapper.

        Args:
            context_length: Number of historical time steps. Must be at least
                max_lag + 32 after loading pretrained weights.
            prediction_length: Forecast horizon. Currently only 1 is supported.
            threshold: Return threshold for P(X > threshold) calculation.
            num_features: Number of input features. If > 1, a projection layer
                is added to map to univariate.
            fine_tune_mode: "head_only" freezes backbone, "full" trains all.
            dropout: Dropout rate for classification head.
        """
        nn.Module.__init__(self)

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.threshold = threshold
        self.num_features = num_features
        self.fine_tune_mode = fine_tune_mode
        self.dropout_rate = dropout

        # Backbone will be loaded from checkpoint
        self.backbone: nn.Module | None = None
        self.max_lag: int | None = None

        # Feature projection for multivariate input
        if num_features > 1:
            self.feature_projection = nn.Sequential(
                nn.Linear(num_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.feature_projection = None

        # Classification head (maps distribution to probability)
        # This is optional - we can use distribution CDF directly
        self.classification_head: nn.Module | None = None

    def load_pretrained(self, checkpoint_path: str) -> None:
        """Load pre-trained Lag-Llama weights from checkpoint.

        Args:
            checkpoint_path: Path to the .ckpt file.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint is incompatible.
        """
        # Register safe globals for unpickling
        torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

        # Import here to avoid circular dependencies
        from lag_llama.gluon.lightning_module import LagLlamaLightningModule

        # Load the Lightning module
        lightning_module = LagLlamaLightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )

        # Extract the core model
        self.backbone = lightning_module.model
        self.max_lag = max(self.backbone.lags_seq)

        # Verify context length is sufficient
        min_context = self.max_lag + self.MODEL_CONTEXT_LENGTH
        if self.context_length < min_context:
            raise ValueError(
                f"context_length ({self.context_length}) must be >= "
                f"{min_context} (max_lag={self.max_lag} + model_context={self.MODEL_CONTEXT_LENGTH})"
            )

        # Apply fine-tuning mode
        if self.fine_tune_mode == "head_only":
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True

        # Ensure projection and head layers are trainable
        if self.feature_projection is not None:
            for param in self.feature_projection.parameters():
                param.requires_grad = True

    def _prepare_time_features(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create dummy time features.

        Lag-Llama requires 6 time features. For simplicity, we use zeros.
        In practice, these could be day-of-week, month-of-year, etc.

        Args:
            batch_size: Batch dimension.
            seq_len: Sequence length (past).
            device: Target device.

        Returns:
            Tuple of (past_time_feat, future_time_feat).
        """
        past_time_feat = torch.zeros(
            batch_size, seq_len, self.N_TIME_FEATURES, device=device
        )
        future_time_feat = torch.zeros(
            batch_size, self.prediction_length, self.N_TIME_FEATURES, device=device
        )
        return past_time_feat, future_time_feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Lag-Llama for binary classification.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).

        Returns:
            Probability tensor of shape (batch, 1) representing P(return > threshold).

        Raises:
            RuntimeError: If backbone is not loaded.
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not loaded. Call load_pretrained() first.")

        batch_size, seq_len, num_feat = x.shape

        # Project multivariate to univariate if needed
        if self.feature_projection is not None:
            # x: (batch, seq_len, num_features) -> (batch, seq_len, 1)
            x = self.feature_projection(x)
        x = x.squeeze(-1)  # (batch, seq_len)

        # Prepare inputs for backbone
        device = x.device
        past_target = x  # (batch, seq_len)
        past_observed = torch.ones_like(past_target)  # All observed
        past_time_feat, future_time_feat = self._prepare_time_features(
            batch_size, seq_len, device
        )

        # Forward through backbone
        distr_args, loc, scale = self.backbone(
            past_target=past_target,
            past_observed_values=past_observed,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )

        # Extract StudentT distribution parameters
        # distr_args is a tuple of (df,) for StudentT
        # loc and scale are already returned by the backbone
        df = distr_args[0]  # Degrees of freedom

        # Compute P(X > threshold) using differentiable CDF
        # Standardize: z = (threshold - loc) / scale
        eps = 1e-8
        z = (self.threshold - loc) / (scale + eps)

        # Get CDF at threshold
        prob_below = _differentiable_studentt_cdf(z, df)
        prob_above = 1 - prob_below

        # Take only the last position (forecast) from each batch
        # Shape: (batch, context_length) -> (batch, 1)
        prob = prob_above[:, -1:]

        # Clamp to valid probability range
        prob = torch.clamp(prob, min=0.0, max=1.0)

        return prob

    def get_config(self) -> dict[str, Any]:
        """Return model configuration dictionary.

        Returns:
            Dictionary containing model configuration.
        """
        return {
            "context_length": self.context_length,
            "prediction_length": self.prediction_length,
            "threshold": self.threshold,
            "num_features": self.num_features,
            "fine_tune_mode": self.fine_tune_mode,
            "dropout": self.dropout_rate,
            "max_lag": self.max_lag,
            "backbone_loaded": self.backbone is not None,
        }
