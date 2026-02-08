"""Tests for extended experiment categories.

Tests the 6 new experiment categories:
- Data Augmentation (DA)
- Noise-Robust Training (NR)
- Curriculum Learning (CL)
- Regime Detection (RD)
- Multi-Scale Temporal (MS)
- Contrastive Pre-training (CP)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# Data Augmentation Tests
# ============================================================================


class TestDataAugmentation:
    """Tests for src/data/augmentation.py."""

    def test_jitter_transform_shape_preserved(self):
        """Test JitterTransform preserves input shape."""
        from src.data.augmentation import JitterTransform

        transform = JitterTransform(std=0.01)
        x = torch.randn(80, 100)  # (seq_len, features)
        y = transform(x)
        assert y.shape == x.shape

    def test_jitter_transform_adds_noise(self):
        """Test JitterTransform actually modifies the input."""
        from src.data.augmentation import JitterTransform

        # Force prob=1.0 to ensure transform always applies
        transform = JitterTransform(std=0.1, prob=1.0)
        x = torch.randn(80, 100)
        y = transform(x)
        # Should not be identical (with very high probability)
        assert not torch.allclose(x, y)

    def test_scale_transform_shape_preserved(self):
        """Test ScaleTransform preserves input shape."""
        from src.data.augmentation import ScaleTransform

        # scale_range is a scalar that defines the max deviation from 1.0
        transform = ScaleTransform(scale_range=0.1)
        x = torch.randn(80, 100)
        y = transform(x)
        assert y.shape == x.shape

    def test_scale_transform_scales_data(self):
        """Test ScaleTransform actually scales the input."""
        from src.data.augmentation import ScaleTransform

        # With prob=1.0, transform always applies
        transform = ScaleTransform(scale_range=0.1, prob=1.0)
        x = torch.ones(80, 100)
        y = transform(x)
        # Just check that values are scaled (between 0.9 and 1.1 of original)
        assert y.min() >= 0.8 and y.max() <= 1.2

    def test_mixup_transform_shape_preserved(self):
        """Test MixupTransform preserves input shape."""
        from src.data.augmentation import MixupTransform

        transform = MixupTransform(alpha=0.2)
        x = torch.randn(80, 100)
        y = transform(x)
        assert y.shape == x.shape

    def test_timewarp_transform_shape_preserved(self):
        """Test TimeWarpTransform preserves input shape."""
        from src.data.augmentation import TimeWarpTransform

        transform = TimeWarpTransform(warp_factor=0.1)
        x = torch.randn(80, 100)
        y = transform(x)
        assert y.shape == x.shape

    def test_composed_transform(self):
        """Test ComposedTransform chains multiple transforms."""
        from src.data.augmentation import (
            ComposedTransform,
            JitterTransform,
            ScaleTransform,
        )

        transform = ComposedTransform([
            JitterTransform(std=0.01),
            ScaleTransform(scale_range=0.1),  # scalar, not tuple
        ])
        x = torch.randn(80, 100)
        y = transform(x)
        assert y.shape == x.shape

    def test_get_augmentation_transform_factory(self):
        """Test get_augmentation_transform factory function."""
        from src.data.augmentation import get_augmentation_transform

        # Test jitter
        transform = get_augmentation_transform("jitter", {"std": 0.01})
        assert transform is not None

        # Test scale
        transform = get_augmentation_transform("scale", {"scale_range": (0.9, 1.1)})
        assert transform is not None

        # Test mixup
        transform = get_augmentation_transform("mixup", {"alpha": 0.2})
        assert transform is not None

        # Test timewarp
        transform = get_augmentation_transform("timewarp", {"warp_factor": 0.1})
        assert transform is not None

        # Test None
        transform = get_augmentation_transform(None, None)
        assert transform is None


# ============================================================================
# Noise-Robust Training Tests
# ============================================================================


class TestNoiseRobustLosses:
    """Tests for noise-robust loss functions in src/training/losses.py."""

    def test_bootstrap_loss_forward(self):
        """Test BootstrapLoss forward pass."""
        from src.training.losses import BootstrapLoss

        loss_fn = BootstrapLoss(beta=0.8)
        pred = torch.sigmoid(torch.randn(32))
        target = torch.randint(0, 2, (32,)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_bootstrap_loss_beta_range(self):
        """Test BootstrapLoss with different beta values."""
        from src.training.losses import BootstrapLoss

        pred = torch.sigmoid(torch.randn(32))
        target = torch.randint(0, 2, (32,)).float()

        for beta in [0.5, 0.7, 0.9, 1.0]:
            loss_fn = BootstrapLoss(beta=beta)
            loss = loss_fn(pred, target)
            assert loss.item() >= 0

    def test_forward_correction_loss_forward(self):
        """Test ForwardCorrectionLoss forward pass."""
        from src.training.losses import ForwardCorrectionLoss

        loss_fn = ForwardCorrectionLoss(noise_rate_0=0.1, noise_rate_1=0.1)
        pred = torch.sigmoid(torch.randn(32))
        target = torch.randint(0, 2, (32,)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_confidence_learning_loss_forward(self):
        """Test ConfidenceLearningLoss forward pass."""
        from src.training.losses import ConfidenceLearningLoss

        loss_fn = ConfidenceLearningLoss(threshold=0.7)
        pred = torch.sigmoid(torch.randn(32))
        target = torch.randint(0, 2, (32,)).float()
        loss = loss_fn(pred, target)
        assert loss.shape == ()
        assert loss.item() >= 0


class TestCoTeaching:
    """Tests for src/training/coteaching.py."""

    def test_coteaching_trainer_init(self):
        """Test CoTeachingTrainer initialization."""
        from src.training.coteaching import CoTeachingTrainer
        from src.models.patchtst import PatchTSTConfig

        # CoTeachingTrainer takes a model_config, not individual models
        config = PatchTSTConfig(
            num_features=100,
            context_length=80,
            patch_length=16,
            stride=8,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            dropout=0.3,
            head_dropout=0.0,
        )
        trainer = CoTeachingTrainer(
            model_config=config,
            forget_rate=0.2,
            device="cpu",
        )
        assert trainer is not None

    def test_ensemble_model_forward(self):
        """Test EnsembleModel forward pass."""
        from src.training.coteaching import EnsembleModel

        model1 = nn.Linear(10, 1)
        model2 = nn.Linear(10, 1)
        ensemble = EnsembleModel(model1, model2)

        x = torch.randn(32, 10)
        y = ensemble(x)
        assert y.shape == (32, 1)


# ============================================================================
# Curriculum Learning Tests
# ============================================================================


class TestCurriculumLearning:
    """Tests for src/training/curriculum.py."""

    def test_curriculum_sampler_init(self):
        """Test CurriculumSampler initialization."""
        from src.training.curriculum import CurriculumSampler

        dataset = TensorDataset(torch.randn(100, 80, 10), torch.randint(0, 2, (100,)))
        scores = np.random.rand(100)

        sampler = CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=0.3,
            growth_rate=0.1,
        )
        assert sampler is not None

    def test_curriculum_sampler_len(self):
        """Test CurriculumSampler length at different epochs."""
        from src.training.curriculum import CurriculumSampler

        dataset = TensorDataset(torch.randn(100, 80, 10), torch.randint(0, 2, (100,)))
        scores = np.random.rand(100)

        sampler = CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=0.3,
            growth_rate=0.1,
        )

        # Epoch 0: 30% of samples
        sampler.set_epoch(0)
        assert len(sampler) == 30

        # Epoch 1: 40% of samples
        sampler.set_epoch(1)
        assert len(sampler) == 40

        # Epoch 7: 100% of samples (capped)
        sampler.set_epoch(7)
        assert len(sampler) == 100

    def test_curriculum_sampler_iter(self):
        """Test CurriculumSampler iteration."""
        from src.training.curriculum import CurriculumSampler

        dataset = TensorDataset(torch.randn(100, 80, 10), torch.randint(0, 2, (100,)))
        scores = np.random.rand(100)

        sampler = CurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=0.3,
            growth_rate=0.1,
        )
        sampler.set_epoch(0)

        indices = list(sampler)
        assert len(indices) == 30
        assert all(0 <= i < 100 for i in indices)

    def test_loss_difficulty_scorer(self):
        """Test LossDifficultyScorer computes scores."""
        from src.training.curriculum import LossDifficultyScorer

        scorer = LossDifficultyScorer()
        # Model must output shape matching targets (view(-1) applied, so need (batch, 1))
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        # Targets need to be (N, 1) to match pred after view(-1)
        dataset = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20, 1)).float())
        dataloader = DataLoader(dataset, batch_size=4)

        scores = scorer.compute_scores(model, dataloader, "cpu")
        assert scores.shape == (20,)
        assert all(s >= 0 for s in scores)

    def test_volatility_difficulty_scorer(self):
        """Test VolatilityDifficultyScorer computes scores."""
        from src.training.curriculum import VolatilityDifficultyScorer

        scorer = VolatilityDifficultyScorer(close_col_idx=0)
        dataset = TensorDataset(torch.randn(20, 80, 10), torch.randint(0, 2, (20,)))
        dataloader = DataLoader(dataset, batch_size=4)

        scores = scorer.compute_scores(None, dataloader, "cpu")
        assert scores.shape == (20,)
        assert all(s >= 0 for s in scores)

    def test_anti_curriculum_sampler(self):
        """Test AntiCurriculumSampler starts with hardest samples."""
        from src.training.curriculum import AntiCurriculumSampler

        dataset = TensorDataset(torch.randn(100, 80, 10), torch.randint(0, 2, (100,)))
        # Scores: 0, 1, 2, ..., 99 (hardest = 99)
        scores = np.arange(100).astype(float)

        sampler = AntiCurriculumSampler(
            dataset=dataset,
            difficulty_scores=scores,
            initial_pct=0.1,
            growth_rate=0.1,
        )
        sampler.set_epoch(0)

        indices = list(sampler)
        # Should contain hardest 10 samples (90-99)
        assert all(i >= 90 for i in indices)


# ============================================================================
# Regime Detection Tests
# ============================================================================


class TestRegimeDetection:
    """Tests for src/training/regime.py."""

    def test_volatility_regime_detector(self):
        """Test VolatilityRegimeDetector detects regimes."""
        from src.training.regime import VolatilityRegimeDetector

        detector = VolatilityRegimeDetector(
            volatility_thresholds=(0.01, 0.02),
            close_col_idx=0,
        )
        assert detector.n_regimes == 3

        # Create synthetic data with different volatility levels
        low_vol = np.ones((80, 10)) * 100  # Constant = low volatility
        regime = detector.detect(low_vol, 0)
        assert regime == 0  # Low volatility

    def test_trend_regime_detector_sma(self):
        """Test TrendRegimeDetector with SMA method."""
        from src.training.regime import TrendRegimeDetector

        detector = TrendRegimeDetector(
            method="sma",
            short_window=10,
            long_window=30,
        )
        assert detector.n_regimes == 3

        # Create uptrend data
        uptrend = np.linspace(100, 150, 80).reshape(-1, 1)
        uptrend = np.tile(uptrend, (1, 10))
        regime = detector.detect(uptrend, 0)
        assert regime == 2  # Bull

    def test_cluster_regime_detector(self):
        """Test ClusterRegimeDetector with clustering."""
        from src.training.regime import ClusterRegimeDetector

        detector = ClusterRegimeDetector(n_clusters=3)
        assert detector.n_regimes == 3

        # Fit on synthetic data
        data = np.random.randn(100, 80, 10)
        detector.fit(data)

        # Detect regime
        regime = detector.detect(data[0], 0)
        assert 0 <= regime < 3

    def test_regime_loss_weighter(self):
        """Test RegimeLossWeighter computes weights."""
        from src.training.regime import RegimeLossWeighter, VolatilityRegimeDetector

        detector = VolatilityRegimeDetector()
        weighter = RegimeLossWeighter(
            detector,
            regime_weights={0: 1.0, 1: 1.5, 2: 2.0},
        )

        features = np.random.randn(10, 80, 10)
        indices = np.arange(10)
        weights = weighter.get_weights(features, indices)
        assert weights.shape == (10,)

    def test_regime_embedding(self):
        """Test RegimeEmbedding forward pass."""
        from src.training.regime import RegimeEmbedding

        embed = RegimeEmbedding(n_regimes=3, embedding_dim=128)
        regime_ids = torch.randint(0, 3, (32,))
        embeddings = embed(regime_ids)
        assert embeddings.shape == (32, 128)

    def test_get_regime_detector_factory(self):
        """Test get_regime_detector factory function."""
        from src.training.regime import get_regime_detector

        # Volatility
        detector = get_regime_detector("volatility", {"thresholds": (0.01, 0.02)})
        assert detector is not None
        assert detector.n_regimes == 3

        # Trend
        detector = get_regime_detector("trend", {"method": "sma"})
        assert detector is not None

        # Cluster
        detector = get_regime_detector("cluster", {"n_clusters": 4})
        assert detector is not None
        assert detector.n_regimes == 4

        # None
        detector = get_regime_detector(None, None)
        assert detector is None


# ============================================================================
# Multi-Scale Temporal Tests
# ============================================================================


class TestMultiScaleTemporal:
    """Tests for src/models/multiscale.py."""

    def test_hierarchical_temporal_pool(self):
        """Test HierarchicalTemporalPool forward pass."""
        from src.models.multiscale import HierarchicalTemporalPool

        pool = HierarchicalTemporalPool(
            d_model=128,
            scales=[1, 2, 4],
            fusion="concat",
        )

        x = torch.randn(32, 16, 128)  # (batch, seq, d_model)
        y = pool(x)
        # concat fusion projects back to d_model after concatenating
        assert y.shape == (32, 16, 128)

    def test_hierarchical_temporal_pool_sum_fusion(self):
        """Test HierarchicalTemporalPool with sum fusion."""
        from src.models.multiscale import HierarchicalTemporalPool

        pool = HierarchicalTemporalPool(
            d_model=128,
            scales=[1, 2, 4],
            fusion="sum",
        )

        x = torch.randn(32, 16, 128)
        y = pool(x)
        assert y.shape == (32, 16, 128)  # Same as input d_model

    def test_multi_scale_patch_embedding(self):
        """Test MultiScalePatchEmbedding forward pass."""
        from src.models.multiscale import MultiScalePatchEmbedding

        embed = MultiScalePatchEmbedding(
            num_features=100,
            d_model=128,
            patch_sizes=[8, 16],
            context_length=80,
            fusion="concat",
        )

        x = torch.randn(32, 80, 100)  # (batch, context_len, features)
        y = embed(x)
        # Should produce patch embeddings - d_model stays the same
        assert y.dim() == 3
        assert y.shape[0] == 32
        assert y.shape[2] == 128  # d_model

    def test_dilated_temporal_conv(self):
        """Test DilatedTemporalConv forward pass."""
        from src.models.multiscale import DilatedTemporalConv

        conv = DilatedTemporalConv(
            d_model=128,
            dilation_rates=[1, 2, 4],
        )

        x = torch.randn(32, 16, 128)  # (batch, seq, d_model)
        y = conv(x)
        assert y.shape == x.shape

    def test_cross_scale_attention(self):
        """Test CrossScaleAttention forward pass."""
        from src.models.multiscale import CrossScaleAttention

        attn = CrossScaleAttention(
            d_model=128,
            n_heads=4,
        )

        fine = torch.randn(32, 16, 128)
        coarse = torch.randn(32, 8, 128)
        y = attn(fine, coarse)
        assert y.shape == fine.shape

    def test_get_multiscale_module_factory(self):
        """Test get_multiscale_module factory function."""
        from src.models.multiscale import get_multiscale_module

        # Hierarchical pool
        module = get_multiscale_module(
            "hierarchical_pool",
            {"scales": [1, 2, 4], "fusion": "concat"},
            d_model=128,
        )
        assert module is not None

        # Multi-patch
        module = get_multiscale_module(
            "multi_patch",
            {"patch_sizes": [8, 16], "fusion": "concat"},
            d_model=128,
            num_features=100,
        )
        assert module is not None

        # Dilated conv
        module = get_multiscale_module(
            "dilated_conv",
            {"dilation_rates": [1, 2, 4]},
            d_model=128,
        )
        assert module is not None

        # None
        module = get_multiscale_module(None, None, d_model=128)
        assert module is None


# ============================================================================
# Contrastive Pre-training Tests
# ============================================================================


class TestContrastivePretraining:
    """Tests for src/training/contrastive.py."""

    def test_contrastive_loss(self):
        """Test ContrastiveLoss (NT-Xent) forward pass."""
        from src.training.contrastive import ContrastiveLoss

        loss_fn = ContrastiveLoss(temperature=0.1)
        z1 = torch.randn(32, 128)
        z2 = torch.randn(32, 128)
        loss = loss_fn(z1, z2)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_hierarchical_contrastive_loss(self):
        """Test HierarchicalContrastiveLoss forward pass."""
        from src.training.contrastive import HierarchicalContrastiveLoss

        loss_fn = HierarchicalContrastiveLoss(temperature=0.1, lambda_temporal=0.5)
        z1 = torch.randn(32, 16, 128)  # (batch, seq, d)
        z2 = torch.randn(32, 16, 128)
        loss = loss_fn(z1, z2)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_projection_head(self):
        """Test ProjectionHead forward pass."""
        from src.training.contrastive import ProjectionHead

        head = ProjectionHead(input_dim=256, hidden_dim=512, output_dim=128)
        x = torch.randn(32, 256)
        y = head(x)
        assert y.shape == (32, 128)

    def test_contrastive_encoder(self):
        """Test ContrastiveEncoder forward pass."""
        from src.training.contrastive import ContrastiveEncoder

        base_encoder = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        encoder = ContrastiveEncoder(
            base_encoder=base_encoder,
            d_model=128,
            projection_dim=64,
        )

        x = torch.randn(32, 100)
        z = encoder(x)
        assert z.shape == (32, 64)

    def test_contrastive_encoder_get_base(self):
        """Test ContrastiveEncoder.get_base_encoder returns base encoder."""
        from src.training.contrastive import ContrastiveEncoder

        base_encoder = nn.Linear(100, 128)
        encoder = ContrastiveEncoder(
            base_encoder=base_encoder,
            d_model=128,
            projection_dim=64,
        )

        recovered = encoder.get_base_encoder()
        assert recovered is base_encoder


# ============================================================================
# Experiment Configuration Tests
# ============================================================================


class TestExperimentConfiguration:
    """Tests for extended experiment configurations."""

    def test_extended_experiment_spec_fields(self):
        """Test ExperimentSpec has all new fields."""
        import sys
        sys.path.insert(0, str(__file__).split("tests")[0])
        from experiments.feature_embedding.run_experiments import ExperimentSpec

        spec = ExperimentSpec(
            exp_id="TEST-01",
            priority="DA-P1",
            tier="a100",
            num_features=100,
            d_embed=32,
            description="Test experiment",
            # New fields
            curriculum_strategy="loss",
            curriculum_params={"initial_pct": 0.3},
            multiscale_type="hierarchical_pool",
            multiscale_params={"scales": [1, 2, 4]},
            regime_strategy="volatility",
            regime_params={"thresholds": (0.01, 0.02)},
            augmentation_type="jitter",
            augmentation_params={"std": 0.01},
            contrastive_type="simclr",
            contrastive_params={"temperature": 0.1},
            pretrain_checkpoint=None,
            noise_robust_type="bootstrap",
            noise_robust_params={"beta": 0.8},
        )
        assert spec.curriculum_strategy == "loss"
        assert spec.multiscale_type == "hierarchical_pool"
        assert spec.regime_strategy == "volatility"
        assert spec.augmentation_type == "jitter"
        assert spec.contrastive_type == "simclr"
        assert spec.noise_robust_type == "bootstrap"

    def test_experiment_count(self):
        """Test that all 114 extended experiments are defined."""
        import sys
        sys.path.insert(0, str(__file__).split("tests")[0])
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        # Count extended experiments (DA, NR, CL, RD, MS, CP)
        extended_prefixes = ["DA-", "NR-", "CL-", "RD-", "MS-", "CP-"]
        extended_count = sum(
            1 for exp in EXPERIMENTS
            if any(exp.exp_id.startswith(prefix) for prefix in extended_prefixes)
        )
        assert extended_count == 114, f"Expected 114 extended experiments, got {extended_count}"

    def test_experiment_priorities(self):
        """Test that all extended priorities are covered."""
        import sys
        sys.path.insert(0, str(__file__).split("tests")[0])
        from experiments.feature_embedding.run_experiments import EXPERIMENTS

        # Expected priorities
        expected_priorities = {
            "DA-P1", "DA-P2", "DA-P3", "DA-P4", "DA-P5",
            "NR-P1", "NR-P2", "NR-P3", "NR-P4",
            "CL-P1", "CL-P2", "CL-P3", "CL-P4",
            "RD-P1", "RD-P2", "RD-P3", "RD-P4",
            "MS-P1", "MS-P2", "MS-P3", "MS-P4",
            "CP-P1", "CP-P2", "CP-P3", "CP-P4",
        }

        actual_priorities = {
            exp.priority for exp in EXPERIMENTS
            if exp.priority.startswith(("DA-", "NR-", "CL-", "RD-", "MS-", "CP-"))
        }

        assert actual_priorities == expected_priorities, f"Missing priorities: {expected_priorities - actual_priorities}"
