"""Tests for Lag-Llama wrapper implementation.

TDD tests for LagLlamaWrapper class that adapts the pre-trained
probabilistic forecaster for binary classification.
"""

from pathlib import Path

import pytest
import torch
import numpy as np

# Constants
CHECKPOINT_PATH = Path("models/pretrained/lag-llama.ckpt")
BATCH_SIZE = 8
# Lag-Llama requires max_lag + 32 context, where max_lag=1092
# So minimum context is 1124 (we use 1150 for padding)
SEQ_LEN = 1150  # Must be >= 1124 for Lag-Llama
NUM_FEATURES = 25  # SPY OHLCV + 20 indicators


class TestLagLlamaWrapperInstantiation:
    """Tests for LagLlamaWrapper instantiation."""

    def test_lag_llama_wrapper_instantiation(self):
        """Test that LagLlamaWrapper can be instantiated."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,  # 1% threshold
        )
        assert wrapper is not None
        assert isinstance(wrapper, torch.nn.Module)

    def test_lag_llama_wrapper_with_feature_projection(self):
        """Test LagLlamaWrapper with feature projection layer."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,  # Enable projection layer
        )
        assert wrapper.feature_projection is not None

    def test_lag_llama_wrapper_univariate_mode(self):
        """Test LagLlamaWrapper in univariate (Close-only) mode."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=1,  # Close only
        )
        assert wrapper.feature_projection is None  # No projection needed


class TestLagLlamaLoadPretrained:
    """Tests for loading pre-trained weights."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_load_pretrained(self):
        """Test that pre-trained weights can be loaded."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))

        # Backbone should be loaded
        assert wrapper.backbone is not None

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_load_sets_correct_device(self):
        """Test that loaded model is on correct device."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))

        # Check backbone parameters are on a device
        for param in wrapper.parameters():
            assert param.device is not None


class TestLagLlamaForward:
    """Tests for forward pass."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_forward_shape_univariate(self):
        """Test forward pass output shape for univariate input."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=1,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))
        wrapper.eval()

        # Input: (batch, seq_len, 1) - Close prices only
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 1)
        with torch.no_grad():
            output = wrapper(x)

        # Output should be (batch, 1) - probability of exceeding threshold
        assert output.shape == (BATCH_SIZE, 1)

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_forward_shape_multivariate(self):
        """Test forward pass output shape for multivariate input with projection."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))
        wrapper.eval()

        # Input: (batch, seq_len, num_features)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
        with torch.no_grad():
            output = wrapper(x)

        # Output should be (batch, 1)
        assert output.shape == (BATCH_SIZE, 1)


class TestDistributionToThresholdProbability:
    """Tests for converting distribution to threshold probability."""

    def test_distribution_to_threshold_probability(self):
        """Test conversion from StudentT distribution to P(X > threshold)."""
        from src.models.foundation.lag_llama import distribution_to_threshold_prob

        # StudentT params: df, loc, scale
        # P(X > threshold) should be computed from CDF
        df = torch.tensor([5.0, 5.0])  # Degrees of freedom
        loc = torch.tensor([0.0, 0.02])  # Location (mean)
        scale = torch.tensor([0.01, 0.01])  # Scale

        threshold = 0.01  # 1% threshold

        probs = distribution_to_threshold_prob(df, loc, scale, threshold)

        # For loc=0.0, threshold=0.01: P(X > 0.01) should be < 0.5
        # For loc=0.02, threshold=0.01: P(X > 0.01) should be > 0.5
        assert probs[0] < 0.5, f"Expected < 0.5, got {probs[0]}"
        assert probs[1] > 0.5, f"Expected > 0.5, got {probs[1]}"

    def test_distribution_to_threshold_probability_extreme(self):
        """Test extreme cases for distribution conversion."""
        from src.models.foundation.lag_llama import distribution_to_threshold_prob

        # Very high loc should give prob ~1.0
        df = torch.tensor([5.0])
        loc = torch.tensor([0.5])  # 50% return - way above threshold
        scale = torch.tensor([0.01])
        threshold = 0.01

        probs = distribution_to_threshold_prob(df, loc, scale, threshold)
        assert probs[0] > 0.99, f"Expected near 1.0, got {probs[0]}"


class TestLagLlamaGetConfig:
    """Tests for get_config method."""

    def test_lag_llama_get_config(self):
        """Test that get_config returns expected keys."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,
        )

        config = wrapper.get_config()

        # Should contain key configuration
        assert "context_length" in config
        assert "prediction_length" in config
        assert "threshold" in config
        assert "num_features" in config
        assert config["context_length"] == SEQ_LEN
        assert config["threshold"] == 0.01


class TestLagLlamaOutputRange:
    """Tests for output value ranges."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_output_range(self):
        """Test that output probabilities are in [0, 1] range."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=1,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))
        wrapper.eval()

        # Various input magnitudes
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(BATCH_SIZE, SEQ_LEN, 1) * scale
            with torch.no_grad():
                output = wrapper(x)

            assert torch.all(output >= 0), f"Output below 0: {output.min()}"
            assert torch.all(output <= 1), f"Output above 1: {output.max()}"


class TestLagLlamaFineTuning:
    """Tests for fine-tuning modes."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_fine_tuning_mode(self):
        """Test that fine-tuning mode freezes backbone."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,
            fine_tune_mode="head_only",  # Freeze backbone
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))

        # Check backbone params are frozen
        for name, param in wrapper.named_parameters():
            if "backbone" in name:
                assert not param.requires_grad, f"Backbone param {name} should be frozen"
            elif "classification_head" in name or "feature_projection" in name:
                assert param.requires_grad, f"Head param {name} should be trainable"

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_full_fine_tuning(self):
        """Test that full fine-tuning enables all gradients."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,
            fine_tune_mode="full",  # All parameters trainable
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))

        # All params should have gradients enabled
        for name, param in wrapper.named_parameters():
            assert param.requires_grad, f"Param {name} should be trainable in full mode"


class TestLagLlamaPredictionMode:
    """Tests for prediction/inference mode."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_prediction_mode(self):
        """Test that model works in eval mode for inference."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=1,
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))
        wrapper.eval()

        x = torch.randn(1, SEQ_LEN, 1)  # Single sample

        with torch.no_grad():
            output1 = wrapper(x)
            output2 = wrapper(x)

        # Deterministic in eval mode
        assert torch.allclose(output1, output2), "Outputs should be deterministic in eval mode"


class TestLagLlamaTrainerIntegration:
    """Tests for integration with our Trainer."""

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_backward_pass(self):
        """Test that gradients can flow through the model."""
        from src.models.foundation.lag_llama import LagLlamaWrapper

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
            num_features=NUM_FEATURES,
            fine_tune_mode="full",
        )
        wrapper.load_pretrained(str(CHECKPOINT_PATH))
        wrapper.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES, requires_grad=True)
        target = torch.randint(0, 2, (BATCH_SIZE, 1)).float()

        output = wrapper(x)
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None, "Input should have gradients"

    @pytest.mark.skipif(
        not CHECKPOINT_PATH.exists(),
        reason="Lag-Llama checkpoint not found"
    )
    def test_lag_llama_model_interface_compatible(self):
        """Test that LagLlamaWrapper is compatible with FoundationModel interface."""
        from src.models.foundation.lag_llama import LagLlamaWrapper
        from src.models.foundation.base import FoundationModel

        wrapper = LagLlamaWrapper(
            context_length=SEQ_LEN,
            prediction_length=1,
            threshold=0.01,
        )

        # Should be a FoundationModel subclass
        assert isinstance(wrapper, FoundationModel)

        # Should implement all abstract methods
        assert hasattr(wrapper, "load_pretrained")
        assert hasattr(wrapper, "forward")
        assert hasattr(wrapper, "get_config")
