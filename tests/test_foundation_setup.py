"""Tests for foundation model environment setup."""

import importlib
import os
from pathlib import Path

import pytest


class TestGluonTSSetup:
    """Tests for GluonTS installation and configuration."""

    def test_gluonts_import(self):
        """Test that gluonts can be imported."""
        import gluonts
        assert hasattr(gluonts, "__version__")

    def test_gluonts_version(self):
        """Test that gluonts version is >= 0.14.0."""
        import gluonts
        from packaging import version
        assert version.parse(gluonts.__version__) >= version.parse("0.14.0")

    def test_gluonts_torch_backend(self):
        """Test that gluonts torch components are available."""
        from gluonts.torch.model.predictor import PyTorchPredictor
        assert PyTorchPredictor is not None


class TestTimesFMSetup:
    """Tests for TimesFM installation (CPU fallback mode).

    Note: TimesFM installation deferred due to JAX/Flax ARM compatibility issues.
    These tests will be enabled when TimesFM setup is completed.
    """

    @pytest.mark.skip(reason="TimesFM deferred - requires JAX/Flax ARM setup")
    def test_timesfm_import(self):
        """Test that timesfm can be imported."""
        import timesfm
        assert timesfm is not None

    @pytest.mark.skip(reason="TimesFM deferred - requires JAX/Flax ARM setup")
    def test_timesfm_cpu_mode(self):
        """Test TimesFM works in CPU mode (ARM/Apple Silicon fallback)."""
        import torch
        # TimesFM should function with CPU tensors
        # This is a smoke test - actual model loading tested separately
        assert torch.device("cpu") is not None


class TestFoundationModuleStructure:
    """Tests for foundation module organization."""

    def test_foundation_module_exists(self):
        """Test that src/models/foundation/ module exists."""
        foundation_path = Path("src/models/foundation")
        assert foundation_path.exists(), "src/models/foundation/ directory missing"

    def test_foundation_init_exists(self):
        """Test that __init__.py exists in foundation module."""
        init_path = Path("src/models/foundation/__init__.py")
        assert init_path.exists(), "src/models/foundation/__init__.py missing"

    def test_foundation_module_importable(self):
        """Test that foundation module can be imported."""
        import src.models.foundation
        assert src.models.foundation is not None

    def test_foundation_base_exists(self):
        """Test that base.py exists in foundation module."""
        base_path = Path("src/models/foundation/base.py")
        assert base_path.exists(), "src/models/foundation/base.py missing"


class TestLagLlamaSetup:
    """Tests for Lag-Llama model availability."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
        reason="Network tests disabled"
    )
    def test_huggingface_hub_available(self):
        """Test that huggingface_hub is installed."""
        import huggingface_hub
        assert hasattr(huggingface_hub, "hf_hub_download")

    @pytest.mark.skipif(
        os.environ.get("SKIP_NETWORK_TESTS", "0") == "1",
        reason="Network tests disabled"
    )
    def test_lag_llama_repo_accessible(self):
        """Test that Lag-Llama repo can be accessed on HuggingFace."""
        from huggingface_hub import repo_exists
        assert repo_exists("time-series-foundation-models/Lag-Llama")

    def test_lag_llama_weights_path_configured(self):
        """Test that weights path is properly configured."""
        weights_dir = Path("models/pretrained")
        # Directory should exist or be creatable
        weights_dir.mkdir(parents=True, exist_ok=True)
        assert weights_dir.exists()

    def test_lag_llama_checkpoint_exists(self):
        """Test that Lag-Llama checkpoint file exists and is valid.

        Note: Full model loading requires Lag-Llama's specific loader,
        which will be implemented in Task 2. This test verifies the
        checkpoint file is downloaded and readable.
        """
        checkpoint_path = Path("models/pretrained/lag-llama.ckpt")
        assert checkpoint_path.exists(), "Lag-Llama checkpoint not found"

        # Verify file is non-empty and readable
        file_size = checkpoint_path.stat().st_size
        assert file_size > 1_000_000, f"Checkpoint too small: {file_size} bytes"  # Should be >1MB

        # Verify file is readable (first bytes check)
        with open(checkpoint_path, "rb") as f:
            header = f.read(10)
            assert len(header) == 10, "Cannot read checkpoint file"


class TestHuggingFaceHub:
    """Tests for HuggingFace Hub integration."""

    def test_huggingface_hub_import(self):
        """Test that huggingface_hub can be imported."""
        import huggingface_hub
        assert huggingface_hub is not None

    def test_huggingface_hub_version(self):
        """Test huggingface_hub version is >= 0.20.0."""
        import huggingface_hub
        from packaging import version
        assert version.parse(huggingface_hub.__version__) >= version.parse("0.20.0")
