"""Integration tests for PatchTST model (Task 3c).

Test cases:
1. test_patchtst_with_real_feature_dimensions - Model works with actual 20-feature tier a20 data
2. test_patchtst_backward_pass_on_mps - Gradients flow correctly on MPS device
3. test_patchtst_batch_inference - DataLoader batching works end-to-end
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FinancialDataset
from src.models.configs import load_patchtst_config
from src.models.patchtst import PatchTST


# Paths to real data
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_features_a20.parquet"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "SPY.parquet"


@pytest.fixture
def real_features_df() -> pd.DataFrame:
    """Load real SPY features from processed data."""
    if not FEATURES_PATH.exists():
        pytest.skip(f"Features file not found: {FEATURES_PATH}")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture
def real_close_prices() -> pd.Series:
    """Load real SPY close prices from raw data."""
    if not RAW_PATH.exists():
        pytest.skip(f"Raw file not found: {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    return df["Close"].values


@pytest.fixture
def real_dataset(real_features_df: pd.DataFrame, real_close_prices: pd.Series) -> FinancialDataset:
    """Create FinancialDataset from real SPY data."""
    # Use same context_length as 2M config
    context_length = 60
    horizon = 5
    threshold = 0.01  # 1%

    return FinancialDataset(
        features_df=real_features_df,
        close_prices=real_close_prices,
        context_length=context_length,
        horizon=horizon,
        threshold=threshold,
    )


class TestPatchTSTRealFeatures:
    """Tests for PatchTST with real feature dimensions."""

    def test_patchtst_with_real_feature_dimensions(
        self, real_dataset: FinancialDataset
    ) -> None:
        """PatchTST should work with actual 20-feature tier a20 data."""
        # Load 2M config (configured for 20 features)
        config = load_patchtst_config("2m")

        # Verify config matches real data
        sample_x, sample_y = real_dataset[0]
        assert sample_x.shape[1] == config.num_features, (
            f"Feature mismatch: data has {sample_x.shape[1]}, "
            f"config expects {config.num_features}"
        )

        # Create model
        model = PatchTST(config)
        model.eval()

        # Run forward pass with real data sample
        # Add batch dimension: (context_length, features) -> (1, context_length, features)
        x = sample_x.unsqueeze(0)

        with torch.no_grad():
            output = model(x)

        # Verify output shape
        assert output.shape == (1, config.num_classes), (
            f"Expected output shape (1, {config.num_classes}), got {output.shape}"
        )

        # Verify output is valid probability
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            f"Output should be in [0, 1], got {output.item()}"
        )


class TestPatchTSTMPS:
    """Tests for PatchTST on MPS device."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available on this system"
    )
    def test_patchtst_backward_pass_on_mps(self) -> None:
        """Gradients should flow correctly on MPS device."""
        device = torch.device("mps")

        # Load 2M config
        config = load_patchtst_config("2m")

        # Create model on MPS
        model = PatchTST(config).to(device)
        model.train()

        # Create input on MPS
        batch_size = 4
        x = torch.randn(
            batch_size, config.context_length, config.num_features, device=device
        )
        target = torch.rand(batch_size, config.num_classes, device=device)

        # Forward pass
        output = model(x)

        # Compute loss
        loss = torch.nn.functional.binary_cross_entropy(output, target)

        # Backward pass
        loss.backward()

        # Verify all parameters have gradients
        params_without_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_without_grad.append(name)
                elif torch.all(param.grad == 0):
                    params_without_grad.append(f"{name} (all zeros)")

        assert len(params_without_grad) == 0, (
            f"Parameters without proper gradients on MPS: {params_without_grad}"
        )

        # Verify loss is finite
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


class TestPatchTSTBatchInference:
    """Tests for PatchTST batch inference with DataLoader."""

    def test_patchtst_batch_inference(self, real_dataset: FinancialDataset) -> None:
        """DataLoader batching should work end-to-end with PatchTST."""
        # Load 2M config
        config = load_patchtst_config("2m")

        # Create model
        model = PatchTST(config)
        model.eval()

        # Create DataLoader with batch_size=8
        batch_size = 8
        dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)

        # Get first batch
        batch_x, batch_y = next(iter(dataloader))

        # Verify batch shapes
        assert batch_x.shape == (batch_size, config.context_length, config.num_features), (
            f"Expected batch_x shape ({batch_size}, {config.context_length}, {config.num_features}), "
            f"got {batch_x.shape}"
        )
        assert batch_y.shape == (batch_size, 1), (
            f"Expected batch_y shape ({batch_size}, 1), got {batch_y.shape}"
        )

        # Run batch inference
        with torch.no_grad():
            output = model(batch_x)

        # Verify output shape
        assert output.shape == (batch_size, config.num_classes), (
            f"Expected output shape ({batch_size}, {config.num_classes}), got {output.shape}"
        )

        # Verify all outputs are valid probabilities
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
            f"All outputs should be in [0, 1], got min={output.min()}, max={output.max()}"
        )
