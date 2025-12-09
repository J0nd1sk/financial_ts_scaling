"""Tests for PatchTST model implementation (Task 3a).

Test cases:
1. test_patch_embedding_output_shape - Input (B, seq_len, features) -> (B, n_patches, d_model)
2. test_transformer_encoder_output_shape - Maintains (B, n_patches, d_model)
3. test_patchtst_forward_pass_output_shape - Full model -> (B, 1) output
4. test_patchtst_output_range_sigmoid - Output in [0, 1] range
5. test_patchtst_gradient_flow - Gradients flow through all parameters
6. test_patchtst_deterministic_with_seed - Same seed -> same output
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchtst import PatchTST, PatchTSTConfig


@pytest.fixture
def default_config() -> PatchTSTConfig:
    """Default PatchTST configuration for testing."""
    return PatchTSTConfig(
        num_features=20,
        context_length=60,
        patch_length=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        head_dropout=0.0,
        num_classes=1,
    )


@pytest.fixture
def sample_input(default_config: PatchTSTConfig) -> torch.Tensor:
    """Sample input tensor for testing."""
    batch_size = 4
    return torch.randn(
        batch_size, default_config.context_length, default_config.num_features
    )


class TestPatchEmbedding:
    """Tests for PatchEmbedding component."""

    def test_patch_embedding_output_shape(self, default_config: PatchTSTConfig) -> None:
        """PatchEmbedding should transform (B, seq_len, features) to (B, n_patches, d_model)."""
        model = PatchTST(default_config)
        batch_size = 4
        x = torch.randn(
            batch_size, default_config.context_length, default_config.num_features
        )

        # Calculate expected number of patches
        # n_patches = (context_length - patch_length) // stride + 1
        expected_n_patches = (
            default_config.context_length - default_config.patch_length
        ) // default_config.stride + 1

        # Access patch embedding output (model should expose this or we test full forward)
        # For now, we verify the model can be instantiated and has correct patch calculation
        assert hasattr(model, "patch_embed"), "Model should have patch_embed component"

        # Test patch embedding directly
        patches = model.patch_embed(x)
        assert patches.shape == (
            batch_size,
            expected_n_patches,
            default_config.d_model,
        ), f"Expected shape ({batch_size}, {expected_n_patches}, {default_config.d_model}), got {patches.shape}"


class TestTransformerEncoder:
    """Tests for TransformerEncoder component."""

    def test_transformer_encoder_output_shape(
        self, default_config: PatchTSTConfig
    ) -> None:
        """TransformerEncoder should maintain (B, n_patches, d_model) shape."""
        model = PatchTST(default_config)
        batch_size = 4

        # Calculate expected number of patches
        expected_n_patches = (
            default_config.context_length - default_config.patch_length
        ) // default_config.stride + 1

        # Create input in the shape transformer expects
        encoder_input = torch.randn(batch_size, expected_n_patches, default_config.d_model)

        assert hasattr(model, "encoder"), "Model should have encoder component"

        # Test encoder directly
        encoded = model.encoder(encoder_input)
        assert encoded.shape == encoder_input.shape, (
            f"Encoder should maintain shape. Expected {encoder_input.shape}, got {encoded.shape}"
        )


class TestPatchTSTForward:
    """Tests for full PatchTST forward pass."""

    def test_patchtst_forward_pass_output_shape(
        self, default_config: PatchTSTConfig, sample_input: torch.Tensor
    ) -> None:
        """Full PatchTST forward should produce (B, 1) output for binary classification."""
        model = PatchTST(default_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        batch_size = sample_input.shape[0]
        expected_shape = (batch_size, default_config.num_classes)
        assert output.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {output.shape}"
        )

    def test_patchtst_output_range_sigmoid(
        self, default_config: PatchTSTConfig, sample_input: torch.Tensor
    ) -> None:
        """PatchTST output should be in [0, 1] range (sigmoid activation)."""
        model = PatchTST(default_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_input)

        assert torch.all(output >= 0.0), f"Output has values < 0: min={output.min()}"
        assert torch.all(output <= 1.0), f"Output has values > 1: max={output.max()}"


class TestPatchTSTGradients:
    """Tests for gradient flow through PatchTST."""

    def test_patchtst_gradient_flow(
        self, default_config: PatchTSTConfig, sample_input: torch.Tensor
    ) -> None:
        """Gradients should flow through all model parameters."""
        model = PatchTST(default_config)
        model.train()

        # Forward pass
        output = model(sample_input)

        # Create dummy target and loss
        target = torch.rand(sample_input.shape[0], default_config.num_classes)
        loss = torch.nn.functional.binary_cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Check that all parameters have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_without_grad.append(name)
                elif torch.all(param.grad == 0):
                    params_without_grad.append(f"{name} (all zeros)")

        assert len(params_without_grad) == 0, (
            f"Parameters without proper gradients: {params_without_grad}"
        )


class TestPatchTSTDeterminism:
    """Tests for deterministic behavior with fixed seeds."""

    def test_patchtst_deterministic_with_seed(
        self, default_config: PatchTSTConfig
    ) -> None:
        """Same seed should produce same output."""
        seed = 42

        # First run
        torch.manual_seed(seed)
        model1 = PatchTST(default_config)
        model1.eval()
        torch.manual_seed(seed + 1000)  # Different seed for input
        x1 = torch.randn(2, default_config.context_length, default_config.num_features)

        with torch.no_grad():
            output1 = model1(x1)

        # Second run with same seeds
        torch.manual_seed(seed)
        model2 = PatchTST(default_config)
        model2.eval()
        torch.manual_seed(seed + 1000)
        x2 = torch.randn(2, default_config.context_length, default_config.num_features)

        with torch.no_grad():
            output2 = model2(x2)

        # Verify inputs are identical
        assert torch.allclose(x1, x2), "Inputs should be identical with same seed"

        # Verify outputs are identical
        assert torch.allclose(output1, output2), (
            f"Outputs should be identical with same seed. "
            f"Max diff: {(output1 - output2).abs().max()}"
        )
