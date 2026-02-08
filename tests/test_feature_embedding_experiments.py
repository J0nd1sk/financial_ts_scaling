"""Tests for Feature Embedding experiment infrastructure.

Validates the experiment runner configuration and parameter estimation.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class TestExperimentConfiguration:
    """Test experiment matrix configuration."""

    def test_all_experiments_have_required_fields(self):
        """All experiments should have exp_id, priority, tier, num_features, d_embed."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        for exp in EXPERIMENTS:
            assert hasattr(exp, "exp_id")
            assert hasattr(exp, "priority")
            assert hasattr(exp, "tier")
            assert hasattr(exp, "num_features")
            assert hasattr(exp, "d_embed")  # Can be None

    def test_p1_experiments_include_all_baselines(self):
        """P1 should include baseline (d_embed=None) for each tier."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        p1_baselines = [
            exp for exp in EXPERIMENTS
            if exp.priority == "P1" and exp.d_embed is None
        ]

        tiers = {exp.tier for exp in p1_baselines}
        assert "a100" in tiers, "P1 should have a100 baseline"
        assert "a200" in tiers, "P1 should have a200 baseline"
        assert "a500" in tiers, "P1 should have a500 baseline"

    def test_experiment_ids_are_unique(self):
        """All experiment IDs should be unique."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        exp_ids = [exp.exp_id for exp in EXPERIMENTS]
        assert len(exp_ids) == len(set(exp_ids)), "Duplicate experiment IDs found"

    def test_data_paths_exist(self):
        """Data paths for all tiers should exist."""
        from experiments.feature_embedding.run_experiments import DATA_PATHS

        for tier, path in DATA_PATHS.items():
            assert path.exists(), f"Data file missing for {tier}: {path}"


class TestParameterEstimation:
    """Test parameter count estimation."""

    def test_estimate_matches_actual_for_baseline(self):
        """Parameter estimation should match actual model for d_embed=None."""
        from src.models.arch_grid import estimate_param_count_with_embedding
        from src.models.patchtst import PatchTST, PatchTSTConfig

        config = PatchTSTConfig(
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
            d_embed=None,
        )
        model = PatchTST(config, use_revin=False)
        actual = sum(p.numel() for p in model.parameters())

        estimated = estimate_param_count_with_embedding(
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            num_features=100,
            d_embed=None,
            context_length=80,
            patch_len=16,
            stride=8,
        )

        assert abs(estimated - actual) / actual < 0.001, (
            f"Estimated {estimated:,} but actual is {actual:,}"
        )

    def test_estimate_matches_actual_with_d_embed(self):
        """Parameter estimation should match actual model for d_embed=64."""
        from src.models.arch_grid import estimate_param_count_with_embedding
        from src.models.patchtst import PatchTST, PatchTSTConfig

        config = PatchTSTConfig(
            num_features=500,
            context_length=80,
            patch_length=16,
            stride=8,
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.5,
            head_dropout=0.0,
            d_embed=64,
        )
        model = PatchTST(config, use_revin=False)
        actual = sum(p.numel() for p in model.parameters())

        estimated = estimate_param_count_with_embedding(
            d_model=128,
            n_layers=4,
            n_heads=8,
            d_ff=512,
            num_features=500,
            d_embed=64,
            context_length=80,
            patch_len=16,
            stride=8,
        )

        assert abs(estimated - actual) / actual < 0.001, (
            f"Estimated {estimated:,} but actual is {actual:,}"
        )

    def test_d_embed_reduces_params_for_a500(self):
        """d_embed=64 should reduce params for a500 tier."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        a500_baseline = next(
            exp for exp in EXPERIMENTS if exp.exp_id == "FE-01"
        )
        a500_embed = next(
            exp for exp in EXPERIMENTS if exp.exp_id == "FE-02"
        )

        from src.models.arch_grid import estimate_param_count_with_embedding

        baseline_params = estimate_param_count_with_embedding(
            d_model=128, n_layers=4, n_heads=8, d_ff=512,
            num_features=a500_baseline.num_features,
            d_embed=a500_baseline.d_embed,
            context_length=80, patch_len=16, stride=8,
        )

        embed_params = estimate_param_count_with_embedding(
            d_model=128, n_layers=4, n_heads=8, d_ff=512,
            num_features=a500_embed.num_features,
            d_embed=a500_embed.d_embed,
            context_length=80, patch_len=16, stride=8,
        )

        reduction = (baseline_params - embed_params) / baseline_params
        assert reduction > 0.4, f"Expected >40% reduction, got {reduction*100:.1f}%"


class TestLossFunctionExperiments:
    """Test loss function experiment configuration (Phase 2)."""

    def test_loss_function_experiments_defined(self):
        """All 70 LF experiments (LF-01 to LF-70) should be defined."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        lf_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("LF-")]
        assert len(lf_experiments) == 70, f"Expected 70 LF experiments, got {len(lf_experiments)}"

        # Verify all IDs exist
        lf_ids = {exp.exp_id for exp in lf_experiments}
        for i in range(1, 71):
            exp_id = f"LF-{i:02d}"
            assert exp_id in lf_ids, f"Missing experiment {exp_id}"

    def test_loss_function_priorities(self):
        """LF experiments should have LF-P1 through LF-P11 priorities."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        lf_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("LF-")]
        priorities = {exp.priority for exp in lf_experiments}

        expected = {"LF-P1", "LF-P2", "LF-P3", "LF-P4", "LF-P5", "LF-P6",
                    "LF-P7", "LF-P8", "LF-P9", "LF-P10", "LF-P11"}
        assert priorities == expected, f"Got priorities {priorities}, expected {expected}"

    def test_experiment_spec_has_loss_fields(self):
        """ExperimentSpec should have loss_fn and loss_params fields."""
        from experiments.feature_embedding.run_experiments import ExperimentSpec

        # Check dataclass has the fields
        spec = ExperimentSpec(
            exp_id="test",
            priority="P1",
            tier="a100",
            num_features=100,
            d_embed=32,
            description="Test experiment",
            loss_fn="softauc",
            loss_params={"gamma": 2.0},
        )
        assert spec.loss_fn == "softauc"
        assert spec.loss_params == {"gamma": 2.0}

    def test_get_loss_function_returns_correct_types(self):
        """get_loss_function should return correct loss class instances."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_loss_function,
        )
        from src.training.losses import (
            FocalLoss,
            LabelSmoothingBCELoss,
            SoftAUCLoss,
            WeightedBCELoss,
            WeightedSumLoss,
        )

        # Test None returns None
        spec_none = ExperimentSpec("t", "P1", "a100", 100, 32, "test")
        assert get_loss_function(spec_none) is None

        # Test each loss function type
        test_cases = [
            ("softauc", {"gamma": 2.0}, SoftAUCLoss),
            ("focal", {"gamma": 2.0, "alpha": 0.25}, FocalLoss),
            ("weightedsum", {"alpha": 0.5, "gamma": 2.0}, WeightedSumLoss),
            ("weightedbce", {"pos_weight": 4.0}, WeightedBCELoss),
            ("labelsmoothing", {"epsilon": 0.1}, LabelSmoothingBCELoss),
        ]

        for loss_fn, params, expected_type in test_cases:
            spec = ExperimentSpec(
                exp_id="test",
                priority="P1",
                tier="a100",
                num_features=100,
                d_embed=32,
                description="Test",
                loss_fn=loss_fn,
                loss_params=params,
            )
            result = get_loss_function(spec)
            assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    def test_get_early_stop_metric(self):
        """get_early_stop_metric should return val_loss for labelsmoothing, val_auc otherwise."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_early_stop_metric,
        )

        # Default (no loss_fn)
        spec_none = ExperimentSpec("t", "P1", "a100", 100, 32, "test")
        assert get_early_stop_metric(spec_none) == "val_auc"

        # Label smoothing uses val_loss
        spec_ls = ExperimentSpec("t", "P1", "a100", 100, 32, "test",
                                  loss_fn="labelsmoothing", loss_params={"epsilon": 0.1})
        assert get_early_stop_metric(spec_ls) == "val_loss"

        # All others use val_auc
        for loss_fn in ["softauc", "focal", "weightedsum", "weightedbce"]:
            spec = ExperimentSpec("t", "P1", "a100", 100, 32, "test",
                                   loss_fn=loss_fn, loss_params={})
            assert get_early_stop_metric(spec) == "val_auc", f"Expected val_auc for {loss_fn}"

    def test_lf_experiments_have_valid_loss_configs(self):
        """All LF experiments should have valid loss_fn and loss_params."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        valid_loss_fns = {
            "softauc", "focal", "weightedsum", "weightedbce", "labelsmoothing",
            "mildfocal", "asymmetricfocal", "entropyreg", "variancereg", "calibratedfocal"
        }

        lf_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("LF-")]
        for exp in lf_experiments:
            assert exp.loss_fn is not None, f"{exp.exp_id} missing loss_fn"
            assert exp.loss_fn in valid_loss_fns, f"{exp.exp_id} has invalid loss_fn: {exp.loss_fn}"
            assert exp.loss_params is not None, f"{exp.exp_id} missing loss_params"
            assert isinstance(exp.loss_params, dict), f"{exp.exp_id} loss_params should be dict"


class TestModelForwardPass:
    """Test model forward pass with d_embed configurations."""

    def test_model_with_d_embed_produces_valid_output(self):
        """Model with d_embed should produce valid [0,1] output."""
        import torch
        from src.models.patchtst import PatchTST, PatchTSTConfig

        config = PatchTSTConfig(
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
            d_embed=32,
        )
        model = PatchTST(config, use_revin=True)
        model.eval()

        x = torch.randn(4, 80, 100)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"
        assert torch.all(out >= 0), "Output should be >= 0"
        assert torch.all(out <= 1), "Output should be <= 1"
        assert not torch.isnan(out).any(), "Output should not contain NaN"

    def test_gradient_flow_with_d_embed(self):
        """Gradients should flow through feature embedding."""
        import torch
        from src.models.patchtst import PatchTST, PatchTSTConfig

        config = PatchTSTConfig(
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
            d_embed=64,
        )
        model = PatchTST(config)
        model.train()

        x = torch.randn(4, 80, 100)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check feature_embed has gradients
        assert model.feature_embed.projection.weight.grad is not None
        assert not torch.all(model.feature_embed.projection.weight.grad == 0)


class TestAdvancedEmbeddingExperiments:
    """Test advanced embedding experiment configuration (Phase 3)."""

    def test_experiment_spec_has_embedding_fields(self):
        """ExperimentSpec should have embedding_type and embedding_params fields."""
        from experiments.feature_embedding.run_experiments import ExperimentSpec

        # Check dataclass has the fields
        spec = ExperimentSpec(
            exp_id="test",
            priority="AE-P1",
            tier="a100",
            num_features=100,
            d_embed=32,
            description="Test experiment",
            embedding_type="progressive",
            embedding_params={"num_layers": 2},
        )
        assert spec.embedding_type == "progressive"
        assert spec.embedding_params == {"num_layers": 2}

    def test_get_embedding_layer_returns_none_for_default(self):
        """get_embedding_layer should return None when embedding_type is None."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_embedding_layer,
        )

        spec = ExperimentSpec("t", "P1", "a100", 100, 32, "test")
        result = get_embedding_layer(spec, dropout=0.5)
        assert result is None

    def test_get_embedding_layer_returns_correct_types(self):
        """get_embedding_layer should return correct embedding class instances."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_embedding_layer,
        )
        from src.models.feature_embeddings import (
            AttentionFeatureEmbedding,
            BottleneckEmbedding,
            GatedResidualEmbedding,
            MultiHeadFeatureEmbedding,
            ProgressiveEmbedding,
        )

        test_cases = [
            ("progressive", {"num_layers": 2}, ProgressiveEmbedding),
            ("bottleneck", {"compression_ratio": 0.25}, BottleneckEmbedding),
            ("multihead", {"num_heads": 4}, MultiHeadFeatureEmbedding),
            ("gated_residual", {}, GatedResidualEmbedding),
            ("attention", {"num_heads": 4}, AttentionFeatureEmbedding),
        ]

        for emb_type, params, expected_type in test_cases:
            spec = ExperimentSpec(
                exp_id="test",
                priority="AE-P1",
                tier="a100",
                num_features=100,
                d_embed=32,
                description="Test",
                embedding_type=emb_type,
                embedding_params=params,
            )
            result = get_embedding_layer(spec, dropout=0.5)
            assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    def test_get_embedding_layer_raises_without_d_embed(self):
        """get_embedding_layer should raise ValueError if d_embed is None."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_embedding_layer,
        )

        spec = ExperimentSpec(
            exp_id="test",
            priority="AE-P1",
            tier="a100",
            num_features=100,
            d_embed=None,  # No d_embed
            description="Test",
            embedding_type="progressive",
            embedding_params={"num_layers": 2},
        )
        with pytest.raises(ValueError, match="requires d_embed"):
            get_embedding_layer(spec, dropout=0.5)

    def test_ae_experiments_defined(self):
        """All 21 AE experiments (AE-01 to AE-21) should be defined."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        ae_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("AE-")]
        assert len(ae_experiments) == 21, f"Expected 21 AE experiments, got {len(ae_experiments)}"

        # Verify all IDs exist
        ae_ids = {exp.exp_id for exp in ae_experiments}
        for i in range(1, 22):
            exp_id = f"AE-{i:02d}"
            assert exp_id in ae_ids, f"Missing experiment {exp_id}"

    def test_ae_experiment_priorities(self):
        """AE experiments should have AE-P1 through AE-P5 priorities."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        ae_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("AE-")]
        priorities = {exp.priority for exp in ae_experiments}

        expected = {"AE-P1", "AE-P2", "AE-P3", "AE-P4", "AE-P5"}
        assert priorities == expected, f"Got priorities {priorities}, expected {expected}"


class TestSubtleLossFunctionExperiments:
    """Test subtle loss function experiment configuration (LF-49 to LF-70)."""

    def test_lf_49_to_70_defined(self):
        """All 22 new LF experiments (LF-49 to LF-70) should be defined."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        lf_experiments = [exp for exp in EXPERIMENTS if exp.exp_id.startswith("LF-")]

        # Should have 48 original + 22 new = 70 total
        assert len(lf_experiments) == 70, f"Expected 70 LF experiments, got {len(lf_experiments)}"

        # Verify LF-49 to LF-70 exist
        lf_ids = {exp.exp_id for exp in lf_experiments}
        for i in range(49, 71):
            exp_id = f"LF-{i:02d}"
            assert exp_id in lf_ids, f"Missing experiment {exp_id}"

    def test_subtle_loss_priorities(self):
        """LF-49 to LF-70 should have LF-P7 through LF-P11 priorities."""
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        subtle_experiments = [
            exp for exp in EXPERIMENTS
            if exp.exp_id.startswith("LF-") and int(exp.exp_id[3:]) >= 49
        ]
        priorities = {exp.priority for exp in subtle_experiments}

        expected = {"LF-P7", "LF-P8", "LF-P9", "LF-P10", "LF-P11"}
        assert priorities == expected, f"Got priorities {priorities}, expected {expected}"

    def test_get_loss_function_new_types(self):
        """get_loss_function should handle 5 new loss types."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_loss_function,
        )
        from src.training.losses import (
            AsymmetricFocalLoss,
            CalibratedFocalLoss,
            EntropyRegularizedBCE,
            MildFocalLoss,
            VarianceRegularizedBCE,
        )

        test_cases = [
            ("mildfocal", {"gamma": 0.75, "alpha": 0.5}, MildFocalLoss),
            ("asymmetricfocal", {"gamma_pos": 1.0, "gamma_neg": 1.5}, AsymmetricFocalLoss),
            ("entropyreg", {"lambda_entropy": 0.1}, EntropyRegularizedBCE),
            ("variancereg", {"lambda_var": 0.5}, VarianceRegularizedBCE),
            ("calibratedfocal", {"gamma": 1.0, "lambda_cal": 0.1}, CalibratedFocalLoss),
        ]

        for loss_fn, params, expected_type in test_cases:
            spec = ExperimentSpec(
                exp_id="test",
                priority="LF-P7",
                tier="a100",
                num_features=100,
                d_embed=32,
                description="Test",
                loss_fn=loss_fn,
                loss_params=params,
            )
            result = get_loss_function(spec)
            assert isinstance(result, expected_type), f"Expected {expected_type}, got {type(result)}"

    def test_get_early_stop_metric_new_losses(self):
        """get_early_stop_metric should return val_loss for calibration-focused losses."""
        from experiments.feature_embedding.run_experiments import (
            ExperimentSpec,
            get_early_stop_metric,
        )

        # These should use val_loss (calibration-focused)
        for loss_fn in ["entropyreg", "variancereg", "calibratedfocal"]:
            spec = ExperimentSpec("t", "LF-P9", "a100", 100, 32, "test",
                                   loss_fn=loss_fn, loss_params={})
            assert get_early_stop_metric(spec) == "val_loss", f"Expected val_loss for {loss_fn}"

        # These should use val_auc (ranking-focused)
        for loss_fn in ["mildfocal", "asymmetricfocal"]:
            spec = ExperimentSpec("t", "LF-P7", "a100", 100, 32, "test",
                                   loss_fn=loss_fn, loss_params={})
            assert get_early_stop_metric(spec) == "val_auc", f"Expected val_auc for {loss_fn}"
