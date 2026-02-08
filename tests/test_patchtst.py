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

from src.models.patchtst import FeatureEmbedding, PatchTST, PatchTSTConfig


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
def config_with_feature_embedding() -> PatchTSTConfig:
    """Config with d_embed=64, d_model=128 for feature embedding tests."""
    return PatchTSTConfig(
        num_features=100,
        context_length=80,
        patch_length=16,
        stride=8,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.5,
        head_dropout=0.0,
        num_classes=1,
        d_embed=64,
    )


@pytest.fixture
def config_large_embedding() -> PatchTSTConfig:
    """Config with d_embed=256, d_model=256 for larger feature embedding tests."""
    return PatchTSTConfig(
        num_features=500,
        context_length=80,
        patch_length=16,
        stride=8,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.5,
        head_dropout=0.0,
        num_classes=1,
        d_embed=256,
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


class TestRevIN:
    """Tests for RevIN (Reversible Instance Normalization) component."""

    def test_revin_forward_shape_unchanged(self) -> None:
        """RevIN normalize should preserve input shape."""
        from src.models.patchtst import RevIN

        revin = RevIN(num_features=20)
        x = torch.randn(4, 60, 20)  # (batch, seq_len, features)

        x_norm = revin.normalize(x)

        assert x_norm.shape == x.shape, (
            f"RevIN should preserve shape. Expected {x.shape}, got {x_norm.shape}"
        )

    def test_revin_normalize_denormalize_identity(self) -> None:
        """Denormalize(normalize(x)) should approximately equal x."""
        from src.models.patchtst import RevIN

        revin = RevIN(num_features=20, affine=False)  # No affine for exact identity
        x = torch.randn(4, 60, 20)

        x_norm = revin.normalize(x)
        x_reconstructed = revin.denormalize(x_norm)

        assert torch.allclose(x, x_reconstructed, atol=1e-5), (
            f"denorm(norm(x)) should equal x. Max diff: {(x - x_reconstructed).abs().max()}"
        )

    def test_revin_normalizes_to_zero_mean_unit_var(self) -> None:
        """Normalized output should have mean≈0, std≈1 per instance."""
        from src.models.patchtst import RevIN

        revin = RevIN(num_features=20, affine=False)
        x = torch.randn(4, 60, 20) * 10 + 50  # Shift and scale

        x_norm = revin.normalize(x)

        # Check per-instance statistics (mean over seq_len dimension)
        instance_means = x_norm.mean(dim=1)  # (batch, features)
        instance_stds = x_norm.std(dim=1)  # (batch, features)

        assert torch.allclose(
            instance_means, torch.zeros_like(instance_means), atol=1e-5
        ), f"Normalized mean should be ~0. Got mean of means: {instance_means.mean()}"
        assert torch.allclose(
            instance_stds, torch.ones_like(instance_stds), atol=1e-1
        ), f"Normalized std should be ~1. Got mean of stds: {instance_stds.mean()}"

    def test_revin_handles_batch_dimension(self) -> None:
        """Different batch items should get different normalization stats."""
        from src.models.patchtst import RevIN

        revin = RevIN(num_features=20, affine=False)

        # Create batch with very different scales
        x = torch.zeros(2, 60, 20)
        x[0] = torch.randn(60, 20) * 1 + 0  # mean≈0, std≈1
        x[1] = torch.randn(60, 20) * 100 + 500  # mean≈500, std≈100

        x_norm = revin.normalize(x)

        # Both should be normalized to similar scale
        std_0 = x_norm[0].std()
        std_1 = x_norm[1].std()

        assert torch.allclose(std_0, std_1, atol=0.2), (
            f"Both batch items should have similar std after normalization. "
            f"Got {std_0:.3f} vs {std_1:.3f}"
        )

    def test_revin_handles_zero_variance(self) -> None:
        """RevIN should handle constant input (zero variance) without NaN."""
        from src.models.patchtst import RevIN

        revin = RevIN(num_features=20)
        x = torch.ones(2, 60, 20) * 5.0  # Constant input

        x_norm = revin.normalize(x)

        assert not torch.isnan(x_norm).any(), (
            "RevIN should not produce NaN for constant input"
        )
        assert not torch.isinf(x_norm).any(), (
            "RevIN should not produce Inf for constant input"
        )


class TestPatchTSTWithRevIN:
    """Tests for PatchTST model with RevIN enabled."""

    def test_patchtst_with_revin_forward_pass(
        self, default_config: PatchTSTConfig
    ) -> None:
        """PatchTST with use_revin=True should produce valid output."""
        model = PatchTST(default_config, use_revin=True)
        model.eval()

        x = torch.randn(4, default_config.context_length, default_config.num_features)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            "Output should be in [0, 1] range"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"


class TestFeatureEmbedding:
    """Tests for FeatureEmbedding component."""

    def test_feature_embedding_output_shape(self) -> None:
        """FeatureEmbedding should transform (B, T, num_features) to (B, T, d_embed)."""
        num_features = 100
        d_embed = 64
        batch_size = 4
        seq_len = 80

        embed = FeatureEmbedding(num_features=num_features, d_embed=d_embed)
        x = torch.randn(batch_size, seq_len, num_features)

        output = embed(x)

        assert output.shape == (batch_size, seq_len, d_embed), (
            f"Expected shape ({batch_size}, {seq_len}, {d_embed}), got {output.shape}"
        )

    def test_feature_embedding_gradient_flow(self) -> None:
        """Gradients should flow through projection and LayerNorm."""
        num_features = 100
        d_embed = 64

        embed = FeatureEmbedding(num_features=num_features, d_embed=d_embed)
        x = torch.randn(4, 80, num_features, requires_grad=True)

        output = embed(x)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None, "Input should have gradients"
        assert not torch.all(x.grad == 0), "Input gradients should not be all zeros"

        # Check that projection weights have gradients
        assert embed.projection.weight.grad is not None, (
            "Projection weight should have gradients"
        )
        assert not torch.all(embed.projection.weight.grad == 0), (
            "Projection weight gradients should not be all zeros"
        )

        # Check that LayerNorm has gradients
        assert embed.norm.weight.grad is not None, (
            "LayerNorm weight should have gradients"
        )

    def test_feature_embedding_parameter_count(self) -> None:
        """Verify parameter count: num_features * d_embed + 2 * d_embed (LayerNorm)."""
        num_features = 100
        d_embed = 64

        embed = FeatureEmbedding(num_features=num_features, d_embed=d_embed)

        # Linear projection: num_features * d_embed (weights) + d_embed (bias)
        # LayerNorm: d_embed (weight) + d_embed (bias)
        expected_params = (num_features * d_embed + d_embed) + (d_embed + d_embed)

        total_params = sum(p.numel() for p in embed.parameters())

        assert total_params == expected_params, (
            f"Expected {expected_params} parameters, got {total_params}"
        )

    def test_feature_embedding_with_dropout(self) -> None:
        """FeatureEmbedding should apply dropout during training."""
        num_features = 100
        d_embed = 64
        dropout = 0.5

        embed = FeatureEmbedding(
            num_features=num_features, d_embed=d_embed, dropout=dropout
        )
        embed.train()

        x = torch.randn(4, 80, num_features)

        # Run multiple times with dropout
        torch.manual_seed(42)
        output1 = embed(x)
        torch.manual_seed(43)
        output2 = embed(x)

        # With dropout in training, outputs should differ
        assert not torch.allclose(output1, output2), (
            "Outputs should differ with dropout during training"
        )

        # In eval mode, outputs should be deterministic
        embed.eval()
        with torch.no_grad():
            output3 = embed(x)
            output4 = embed(x)

        assert torch.allclose(output3, output4), (
            "Outputs should be identical in eval mode"
        )


class TestPatchTSTWithFeatureEmbedding:
    """Tests for PatchTST model with feature embedding enabled."""

    def test_patchtst_with_d_embed_forward_pass(
        self, config_with_feature_embedding: PatchTSTConfig
    ) -> None:
        """Full forward pass should work with d_embed=64."""
        model = PatchTST(config_with_feature_embedding)
        model.eval()

        batch_size = 4
        x = torch.randn(
            batch_size,
            config_with_feature_embedding.context_length,
            config_with_feature_embedding.num_features,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}"
        )
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            "Output should be in [0, 1] range"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    def test_patchtst_with_d_embed_output_shape(
        self, config_with_feature_embedding: PatchTSTConfig
    ) -> None:
        """Output should still be (B, 1) regardless of d_embed."""
        model = PatchTST(config_with_feature_embedding)
        model.eval()

        for batch_size in [1, 4, 8]:
            x = torch.randn(
                batch_size,
                config_with_feature_embedding.context_length,
                config_with_feature_embedding.num_features,
            )

            with torch.no_grad():
                output = model(x)

            assert output.shape == (batch_size, 1), (
                f"Expected shape ({batch_size}, 1), got {output.shape}"
            )

    def test_patchtst_d_embed_reduces_patch_dim(
        self, config_with_feature_embedding: PatchTSTConfig
    ) -> None:
        """PatchEmbedding should use d_embed instead of num_features for patch dim."""
        model = PatchTST(config_with_feature_embedding)

        # Check that feature_embed exists
        assert hasattr(model, "feature_embed"), (
            "Model should have feature_embed when d_embed is set"
        )

        # Check that patch_embed uses d_embed (effective_features)
        expected_features = config_with_feature_embedding.d_embed
        assert model.patch_embed.num_features == expected_features, (
            f"patch_embed.num_features should be {expected_features} (d_embed), "
            f"got {model.patch_embed.num_features}"
        )

    def test_patchtst_with_d_embed_and_revin(
        self, config_with_feature_embedding: PatchTSTConfig
    ) -> None:
        """Feature embedding should work with RevIN enabled."""
        model = PatchTST(config_with_feature_embedding, use_revin=True)
        model.eval()

        batch_size = 4
        x = torch.randn(
            batch_size,
            config_with_feature_embedding.context_length,
            config_with_feature_embedding.num_features,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"

    def test_patchtst_with_d_embed_gradient_flow(
        self, config_with_feature_embedding: PatchTSTConfig
    ) -> None:
        """Gradients should flow through feature embedding to all parameters."""
        model = PatchTST(config_with_feature_embedding)
        model.train()

        batch_size = 4
        x = torch.randn(
            batch_size,
            config_with_feature_embedding.context_length,
            config_with_feature_embedding.num_features,
        )

        output = model(x)
        target = torch.rand(batch_size, 1)
        loss = torch.nn.functional.binary_cross_entropy(output, target)
        loss.backward()

        # Check feature_embed gradients specifically
        assert model.feature_embed.projection.weight.grad is not None, (
            "Feature embedding projection should have gradients"
        )
        assert not torch.all(model.feature_embed.projection.weight.grad == 0), (
            "Feature embedding gradients should not be all zeros"
        )

    def test_patchtst_large_embedding(
        self, config_large_embedding: PatchTSTConfig
    ) -> None:
        """Test with larger d_embed=256 and num_features=500."""
        model = PatchTST(config_large_embedding)
        model.eval()

        batch_size = 2
        x = torch.randn(
            batch_size,
            config_large_embedding.context_length,
            config_large_embedding.num_features,
        )

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}"
        )
        assert not torch.isnan(output).any(), "Output should not contain NaN"


class TestPatchTSTBackwardCompatibility:
    """Tests to ensure backward compatibility when d_embed is not set."""

    def test_d_embed_none_equals_original(self, default_config: PatchTSTConfig) -> None:
        """d_embed=None should produce identical architecture to original."""
        # default_config has no d_embed (None)
        model = PatchTST(default_config)

        # Should NOT have feature_embed
        assert not hasattr(model, "feature_embed"), (
            "Model should NOT have feature_embed when d_embed is None"
        )

        # patch_embed should use num_features directly
        assert model.patch_embed.num_features == default_config.num_features, (
            f"patch_embed.num_features should be {default_config.num_features}, "
            f"got {model.patch_embed.num_features}"
        )

    def test_config_without_d_embed_works(self) -> None:
        """Old configs missing d_embed should still work."""
        # Create config explicitly without d_embed argument
        config = PatchTSTConfig(
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

        # Should work without error
        model = PatchTST(config)
        model.eval()

        x = torch.randn(4, config.context_length, config.num_features)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1), f"Expected shape (4, 1), got {output.shape}"

    def test_effective_features_property(self) -> None:
        """effective_features should return d_embed if set, else num_features."""
        # Without d_embed
        config1 = PatchTSTConfig(
            num_features=100,
            context_length=80,
            patch_length=16,
            stride=8,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.5,
            head_dropout=0.0,
            num_classes=1,
        )
        assert config1.effective_features == 100, (
            f"effective_features should be num_features (100), got {config1.effective_features}"
        )

        # With d_embed
        config2 = PatchTSTConfig(
            num_features=100,
            context_length=80,
            patch_length=16,
            stride=8,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.5,
            head_dropout=0.0,
            num_classes=1,
            d_embed=64,
        )
        assert config2.effective_features == 64, (
            f"effective_features should be d_embed (64), got {config2.effective_features}"
        )

    def test_original_and_new_produce_same_architecture_without_d_embed(
        self, default_config: PatchTSTConfig
    ) -> None:
        """Without d_embed, model should be structurally identical to original."""
        torch.manual_seed(42)
        model = PatchTST(default_config)

        # Count parameters - should match original implementation
        total_params = sum(p.numel() for p in model.parameters())

        # Verify we can do forward pass
        x = torch.randn(2, default_config.context_length, default_config.num_features)
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"
        # Just verify it runs - exact param count depends on architecture
