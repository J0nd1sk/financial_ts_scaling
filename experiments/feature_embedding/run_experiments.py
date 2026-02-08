#!/usr/bin/env python3
"""
Feature Embedding Experiment Runner

Runs experiments from the feature embedding experiment matrix to test whether
projecting features into a learned embedding space improves PatchTST performance.

Usage:
    # Run single experiment
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id FE-01

    # Run all P1 (baseline) experiments
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority P1

    # Run all experiments
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --all

Design doc: docs/feature_embedding_experiments.md
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.experiment import ExperimentConfig
from src.data.dataset import SimpleSplitter
from src.models.arch_grid import estimate_param_count_with_embedding
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.trainer import Trainer


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentSpec:
    """Specification for a single experiment."""
    exp_id: str
    priority: str  # P1-P12 for architecture, LF-P1 to LF-P11 for loss, AE-P1 to AE-P5 for embedding
    tier: str  # a100, a200, a500
    num_features: int
    d_embed: int | None
    description: str
    # Optional overrides (None = use FIXED_CONFIG)
    d_model: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    d_ff: int | None = None
    dropout: float | None = None
    weight_decay: float | None = None
    # Loss function configuration (Phase 2)
    loss_fn: str | None = None  # "softauc", "focal", "weightedsum", "weightedbce", "labelsmoothing", etc.
    loss_params: dict | None = None  # {"gamma": 2.0, "alpha": 0.25} etc.
    # Advanced embedding configuration (Phase 3)
    embedding_type: str | None = None  # "progressive", "bottleneck", "multihead", "gated_residual", "attention"
    embedding_params: dict | None = None  # Type-specific params: {"num_layers": 2}, {"compression_ratio": 0.25}, etc.

    def get_config(self, key: str, fixed_config: dict) -> any:
        """Get config value, using override if set, otherwise fixed_config."""
        override = getattr(self, key, None)
        return override if override is not None else fixed_config[key]


# Experiment matrix from design doc
EXPERIMENTS = [
    # P1: Baseline comparisons
    ExperimentSpec("FE-01", "P1", "a500", 500, None, "a500 baseline (no embedding)"),
    ExperimentSpec("FE-02", "P1", "a500", 500, 64, "a500 with d_embed=64"),
    ExperimentSpec("FE-03", "P1", "a200", 206, None, "a200 baseline (no embedding)"),
    ExperimentSpec("FE-04", "P1", "a200", 206, 64, "a200 with d_embed=64"),
    ExperimentSpec("FE-05", "P1", "a100", 100, None, "a100 baseline (no embedding)"),
    ExperimentSpec("FE-06", "P1", "a100", 100, 32, "a100 with d_embed=32"),
    # P2: d_embed sensitivity
    ExperimentSpec("FE-07", "P2", "a500", 500, 128, "a500 with d_embed=128"),
    ExperimentSpec("FE-08", "P2", "a200", 206, 128, "a200 with d_embed=128"),
    ExperimentSpec("FE-09", "P2", "a100", 100, 64, "a100 with d_embed=64"),
    # P3: Additional exploration
    ExperimentSpec("FE-10", "P3", "a500", 500, 256, "a500 with d_embed=256 (light compression)"),
    ExperimentSpec("FE-11", "P3", "a100", 100, 128, "a100 with d_embed=128 (expansion)"),
    ExperimentSpec("FE-12", "P3", "a500", 500, 32, "a500 with d_embed=32 (aggressive compression)"),
    # P4: Architecture Scaling (a100 focus)
    ExperimentSpec("FE-13", "P4", "a100", 100, 32, "a100 d_model=256 (2x wider)",
                   d_model=256, d_ff=1024),
    ExperimentSpec("FE-14", "P4", "a100", 100, 32, "a100 d_model=512 (4x wider)",
                   d_model=512, d_ff=2048),
    ExperimentSpec("FE-15", "P4", "a100", 100, 64, "a100 d_embed=64, d_model=256",
                   d_model=256, d_ff=1024),
    ExperimentSpec("FE-16", "P4", "a100", 100, 64, "a100 d_embed=64, d_model=512",
                   d_model=512, d_ff=2048),
    ExperimentSpec("FE-17", "P4", "a100", 100, 32, "a100 d_model=1024 (8x wider, 51M params)",
                   d_model=1024, d_ff=4096),
    # P5a: Depth vs Width Trade-offs (a100)
    ExperimentSpec("FE-18", "P5", "a100", 100, 32, "Wide+Shallow: d=256, L=2",
                   d_model=256, n_layers=2, d_ff=1024),
    ExperimentSpec("FE-19", "P5", "a100", 100, 32, "Narrow+Deep: d=128, L=8",
                   d_model=128, n_layers=8, d_ff=512),
    ExperimentSpec("FE-20", "P5", "a100", 100, 32, "Very Wide+Shallow: d=512, L=2",
                   d_model=512, n_layers=2, d_ff=2048),
    ExperimentSpec("FE-21", "P5", "a100", 100, 32, "Wide+Deep: d=256, L=8",
                   d_model=256, n_layers=8, d_ff=1024),
    # P5b: Embedding Extremes (all tiers)
    ExperimentSpec("FE-22", "P5", "a500", 500, 16, "Aggressive compress: 500->16",
                   d_model=128),
    ExperimentSpec("FE-23", "P5", "a500", 500, 32, "a500 compress + wider: d=256",
                   d_model=256, d_ff=1024),
    ExperimentSpec("FE-24", "P5", "a100", 100, 256, "Expansion: 100->256",
                   d_model=128),
    ExperimentSpec("FE-25", "P5", "a200", 206, 32, "a200 compress + wider: d=256",
                   d_model=256, d_ff=1024),
    ExperimentSpec("FE-26", "P5", "a200", 206, 128, "a200 match embed to model: d=512",
                   d_model=512, d_ff=2048),
    # P5c: Regularization Extremes (a100 with best d_embed=32)
    ExperimentSpec("FE-27", "P5", "a100", 100, 32, "Low regularization: drop=0.3, wd=0",
                   dropout=0.3, weight_decay=0.0),
    ExperimentSpec("FE-28", "P5", "a100", 100, 32, "High regularization: drop=0.7, wd=1e-2",
                   dropout=0.7, weight_decay=1e-2),
    ExperimentSpec("FE-29", "P5", "a100", 100, 32, "Low dropout, moderate wd: drop=0.3, wd=1e-3",
                   dropout=0.3, weight_decay=1e-3),
    ExperimentSpec("FE-30", "P5", "a100", 100, 32, "Balanced: drop=0.6, wd=1e-4",
                   dropout=0.6, weight_decay=1e-4),
    # P5d: Attention Variations (a100, d=256)
    ExperimentSpec("FE-31", "P5", "a100", 100, 32, "More heads: h=16, d=256",
                   d_model=256, n_heads=16, d_ff=1024),
    ExperimentSpec("FE-32", "P5", "a100", 100, 32, "Fewer heads: h=4, d=256",
                   d_model=256, n_heads=4, d_ff=1024),
    ExperimentSpec("FE-33", "P5", "a100", 100, 32, "Larger FFN: d=256, d_ff=8x",
                   d_model=256, d_ff=2048),
    ExperimentSpec("FE-34", "P5", "a100", 100, 32, "Smaller FFN: d=256, d_ff=2x",
                   d_model=256, d_ff=512),
    # P5e: Cross-tier with promising configs
    ExperimentSpec("FE-35", "P5", "a200", 206, 32, "a200 with a100-best config: d=256, L=4",
                   d_model=256, d_ff=1024),
    ExperimentSpec("FE-36", "P5", "a500", 500, 32, "a500 with a100-best config: d=256, L=4",
                   d_model=256, d_ff=1024),
    # =========================================================================
    # P6: d_embed Extremes - Testing aggressive compression and expansion
    # =========================================================================
    ExperimentSpec("FE-37", "P6", "a100", 100, 16, "a100 d_embed=16 (aggressive compress)"),
    ExperimentSpec("FE-38", "P6", "a100", 100, 8, "a100 d_embed=8 (extreme compress)"),
    ExperimentSpec("FE-39", "P6", "a200", 206, 16, "a200 d_embed=16 (aggressive compress)"),
    ExperimentSpec("FE-40", "P6", "a200", 206, 8, "a200 d_embed=8 (extreme compress)"),
    ExperimentSpec("FE-41", "P6", "a500", 500, 8, "a500 d_embed=8 (maximum compress)"),
    ExperimentSpec("FE-42", "P6", "a100", 100, 512, "a100 d_embed=512 expand + huge model",
                   d_model=1024, d_ff=4096),
    ExperimentSpec("FE-43", "P6", "a500", 500, 512, "a500 d_embed=512, d=1024, L=2, h=2",
                   d_model=1024, n_layers=2, n_heads=2, d_ff=4096),
    ExperimentSpec("FE-44", "P6", "a100", 100, 256, "a100 expand 100->256 then compress to d=128"),
    # =========================================================================
    # P7: Tiny d_model - Testing if smaller models work better
    # =========================================================================
    ExperimentSpec("FE-45", "P7", "a100", 100, 32, "a100 d_model=64 (half winner)",
                   d_model=64, d_ff=256),
    ExperimentSpec("FE-46", "P7", "a100", 100, 32, "a100 d=64, L=8, h=4 (tiny+deep+fewer heads)",
                   d_model=64, n_layers=8, n_heads=4, d_ff=256),
    ExperimentSpec("FE-47", "P7", "a100", 100, 16, "a100 d_embed=16, d=64, h=4 (tiny everything)",
                   d_model=64, n_heads=4, d_ff=256),
    ExperimentSpec("FE-48", "P7", "a100", 100, 32, "a100 d_model=32, L=8, h=4 (extremely tiny)",
                   d_model=32, n_layers=8, n_heads=4, d_ff=128),
    ExperimentSpec("FE-49", "P7", "a100", 100, 16, "a100 d_embed=16, d=64, L=8, h=2",
                   d_model=64, n_layers=8, n_heads=2, d_ff=256),
    ExperimentSpec("FE-50", "P7", "a500", 500, 16, "a500 d_embed=16, d=64, h=4 (tiny on noisy)",
                   d_model=64, n_heads=4, d_ff=256),
    # =========================================================================
    # P8: Deep Networks - Testing depth with narrow models
    # =========================================================================
    ExperimentSpec("FE-51", "P8", "a100", 100, 32, "a100 L=12 (3x depth of winner)",
                   n_layers=12),
    ExperimentSpec("FE-52", "P8", "a100", 100, 32, "a100 L=16 (4x depth)",
                   n_layers=16),
    ExperimentSpec("FE-53", "P8", "a100", 100, 32, "a100 L=12, h=4 (deep + fewer heads)",
                   n_layers=12, n_heads=4),
    ExperimentSpec("FE-54", "P8", "a100", 100, 32, "a100 L=16, h=4 (deeper + fewer heads)",
                   n_layers=16, n_heads=4),
    ExperimentSpec("FE-55", "P8", "a100", 100, 32, "a100 d=64, L=16, h=4 (tiny + very deep)",
                   d_model=64, n_layers=16, n_heads=4, d_ff=256),
    ExperimentSpec("FE-56", "P8", "a100", 100, 32, "a100 d=64, L=20, h=4 (tiny + extremely deep)",
                   d_model=64, n_layers=20, n_heads=4, d_ff=256),
    ExperimentSpec("FE-57", "P8", "a100", 100, 16, "a100 d_embed=16, L=12, h=4 (compress + deep)",
                   n_layers=12, n_heads=4),
    ExperimentSpec("FE-58", "P8", "a100", 100, 16, "a100 d_embed=16, d=64, L=16, h=2 (triple extreme)",
                   d_model=64, n_layers=16, n_heads=2, d_ff=256),
    # =========================================================================
    # P9: Minimal Heads - Testing h=1 and h=2
    # =========================================================================
    ExperimentSpec("FE-59", "P9", "a100", 100, 32, "a100 h=2 (winner config but h=2)",
                   n_heads=2),
    ExperimentSpec("FE-60", "P9", "a100", 100, 32, "a100 h=1 (winner config but h=1)",
                   n_heads=1),
    ExperimentSpec("FE-61", "P9", "a100", 100, 32, "a100 L=8, h=2 (deep + h=2)",
                   n_layers=8, n_heads=2),
    ExperimentSpec("FE-62", "P9", "a100", 100, 32, "a100 L=8, h=1 (deep + h=1)",
                   n_layers=8, n_heads=1),
    ExperimentSpec("FE-63", "P9", "a100", 100, 32, "a100 d=256, h=1 (wider but single head)",
                   d_model=256, n_heads=1, d_ff=1024),
    ExperimentSpec("FE-64", "P9", "a100", 100, 32, "a100 d=64, L=8, h=1 (tiny+deep+single head)",
                   d_model=64, n_layers=8, n_heads=1, d_ff=256),
    # =========================================================================
    # P10: Fewer Features - Testing if a50/a20 outperform a100
    # =========================================================================
    ExperimentSpec("FE-65", "P10", "a50", 50, None, "a50 baseline (no embedding)"),
    ExperimentSpec("FE-66", "P10", "a50", 50, 16, "a50 d_embed=16"),
    ExperimentSpec("FE-67", "P10", "a50", 50, 32, "a50 d_embed=32"),
    ExperimentSpec("FE-68", "P10", "a20", 20, None, "a20 baseline (no embedding)"),
    ExperimentSpec("FE-69", "P10", "a20", 20, 16, "a20 d_embed=16 (expansion)"),
    ExperimentSpec("FE-70", "P10", "a20", 20, 32, "a20 d_embed=32 (more expansion)"),
    # =========================================================================
    # P11: Combined Extremes - Multi-factor combinations
    # =========================================================================
    ExperimentSpec("FE-71", "P11", "a100", 100, 16, "a100 d_embed=16, d=64, L=12, h=2 (all good dirs)",
                   d_model=64, n_layers=12, n_heads=2, d_ff=256),
    ExperimentSpec("FE-72", "P11", "a100", 100, 8, "a100 d_embed=8, d=64, L=16, h=1 (maximum extreme)",
                   d_model=64, n_layers=16, n_heads=1, d_ff=256),
    ExperimentSpec("FE-73", "P11", "a100", 100, 16, "a100 d_embed=16, L=8, h=2 (balanced extreme)",
                   n_layers=8, n_heads=2),
    ExperimentSpec("FE-74", "P11", "a200", 206, 16, "a200 d_embed=16, d=64, L=12, h=2 (a100 extremes)",
                   d_model=64, n_layers=12, n_heads=2, d_ff=256),
    ExperimentSpec("FE-75", "P11", "a500", 500, 8, "a500 d_embed=8, d=64, L=12, h=2 (extremes on 500)",
                   d_model=64, n_layers=12, n_heads=2, d_ff=256),
    ExperimentSpec("FE-76", "P11", "a100", 100, 32, "a100 L=8, h=2 (FE-19 but h=2)",
                   n_layers=8, n_heads=2),
    ExperimentSpec("FE-77", "P11", "a100", 100, 16, "a100 d_embed=16, L=8, h=2 (FE-19 smaller embed)",
                   n_layers=8, n_heads=2),
    ExperimentSpec("FE-78", "P11", "a100", 100, 32, "a100 d=96, L=8, h=4 (odd d_model)",
                   d_model=96, n_layers=8, n_heads=4, d_ff=384),
    ExperimentSpec("FE-79", "P11", "a100", 100, 32, "a100 L=6, h=4 (between L=4 and L=8)",
                   n_layers=6, n_heads=4),
    ExperimentSpec("FE-80", "P11", "a100", 100, 24, "a100 d_embed=24 (between 16 and 32)"),
    # P11 continued: Regularization with best configs
    ExperimentSpec("FE-81", "P11", "a100", 100, 32, "a100 winner + dropout=0.6",
                   dropout=0.6),
    ExperimentSpec("FE-82", "P11", "a100", 100, 32, "a100 winner + dropout=0.4",
                   dropout=0.4),
    ExperimentSpec("FE-83", "P11", "a100", 100, 32, "a100 L=8, h=4, dropout=0.6 (deep+reg)",
                   n_layers=8, n_heads=4, dropout=0.6),
    ExperimentSpec("FE-84", "P11", "a100", 100, 16, "a100 d_embed=16, d=64, L=12, h=2, drop=0.6",
                   d_model=64, n_layers=12, n_heads=2, d_ff=256, dropout=0.6),
    # =========================================================================
    # P12: FE-50 Variants - Exploring the a500+tiny breakthrough
    # FE-50 (a500, d_embed=16, d=64, h=4) achieved 60% precision - best a500 ever
    # =========================================================================
    ExperimentSpec("FE-85", "P12", "a500", 500, 16, "FE-50 but deeper: L=8",
                   d_model=64, n_layers=8, n_heads=4, d_ff=256),
    ExperimentSpec("FE-86", "P12", "a500", 500, 16, "FE-50 but fewer heads: h=2",
                   d_model=64, n_layers=4, n_heads=2, d_ff=256),
    ExperimentSpec("FE-87", "P12", "a500", 500, 8, "FE-50 but more compression: d_embed=8",
                   d_model=64, n_layers=4, n_heads=4, d_ff=256),
    ExperimentSpec("FE-88", "P12", "a500", 500, 16, "FE-50 but tinier: d=32",
                   d_model=32, n_layers=8, n_heads=4, d_ff=128),
    ExperimentSpec("FE-89", "P12", "a200", 206, 16, "Apply FE-50 config to a200",
                   d_model=64, n_layers=4, n_heads=4, d_ff=256),
    ExperimentSpec("FE-90", "P12", "a100", 100, 16, "Apply FE-50 config to a100",
                   d_model=64, n_layers=4, n_heads=4, d_ff=256),
    ExperimentSpec("FE-91", "P12", "a500", 500, 16, "FE-50 + deep + fewer heads: L=8, h=2",
                   d_model=64, n_layers=8, n_heads=2, d_ff=256),
    ExperimentSpec("FE-92", "P12", "a500", 500, 16, "FE-50 but L=12",
                   d_model=64, n_layers=12, n_heads=4, d_ff=256),
    # =========================================================================
    # PHASE 2: LOSS FUNCTION EXPERIMENTS (LF-01 to LF-48)
    # Test whether alternative loss functions improve precision on best
    # feature embedding architectures from Phase 1.
    # =========================================================================
    # =========================================================================
    # LF-P1: SoftAUC Loss (8 experiments)
    # Directly optimizes ranking, avoiding prior collapse problem.
    # =========================================================================
    # FE-06 architecture: a100, d_embed=32, d_model=128, n_layers=4, n_heads=8
    ExperimentSpec("LF-01", "LF-P1", "a100", 100, 32, "a100 FE-06 + SoftAUC gamma=1.0",
                   loss_fn="softauc", loss_params={"gamma": 1.0}),
    ExperimentSpec("LF-02", "LF-P1", "a100", 100, 32, "a100 FE-06 + SoftAUC gamma=2.0",
                   loss_fn="softauc", loss_params={"gamma": 2.0}),
    ExperimentSpec("LF-03", "LF-P1", "a100", 100, 32, "a100 FE-06 + SoftAUC gamma=3.0",
                   loss_fn="softauc", loss_params={"gamma": 3.0}),
    ExperimentSpec("LF-04", "LF-P1", "a100", 100, 32, "a100 FE-06 + SoftAUC gamma=5.0",
                   loss_fn="softauc", loss_params={"gamma": 5.0}),
    # FE-50 architecture: a500, d_embed=16, d_model=64, n_layers=4, n_heads=4, d_ff=256
    ExperimentSpec("LF-05", "LF-P1", "a500", 500, 16, "a500 FE-50 + SoftAUC gamma=1.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="softauc", loss_params={"gamma": 1.0}),
    ExperimentSpec("LF-06", "LF-P1", "a500", 500, 16, "a500 FE-50 + SoftAUC gamma=2.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="softauc", loss_params={"gamma": 2.0}),
    ExperimentSpec("LF-07", "LF-P1", "a500", 500, 16, "a500 FE-50 + SoftAUC gamma=3.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="softauc", loss_params={"gamma": 3.0}),
    ExperimentSpec("LF-08", "LF-P1", "a500", 500, 16, "a500 FE-50 + SoftAUC gamma=5.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="softauc", loss_params={"gamma": 5.0}),
    # =========================================================================
    # LF-P2: Focal Loss (12 experiments)
    # Down-weights easy examples, focuses on hard examples.
    # =========================================================================
    # FE-06 architecture
    ExperimentSpec("LF-09", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=2.0, a=0.25",
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.25}),
    ExperimentSpec("LF-10", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=2.0, a=0.50",
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.50}),
    ExperimentSpec("LF-11", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=2.0, a=0.75",
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.75}),
    ExperimentSpec("LF-12", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=3.0, a=0.25",
                   loss_fn="focal", loss_params={"gamma": 3.0, "alpha": 0.25}),
    ExperimentSpec("LF-13", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=1.0, a=0.50",
                   loss_fn="focal", loss_params={"gamma": 1.0, "alpha": 0.50}),
    ExperimentSpec("LF-14", "LF-P2", "a100", 100, 32, "a100 FE-06 + Focal g=3.0, a=0.75",
                   loss_fn="focal", loss_params={"gamma": 3.0, "alpha": 0.75}),
    # FE-50 architecture
    ExperimentSpec("LF-15", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=2.0, a=0.25",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.25}),
    ExperimentSpec("LF-16", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=2.0, a=0.50",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.50}),
    ExperimentSpec("LF-17", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=2.0, a=0.75",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.75}),
    ExperimentSpec("LF-18", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=3.0, a=0.25",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 3.0, "alpha": 0.25}),
    ExperimentSpec("LF-19", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=1.0, a=0.50",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 1.0, "alpha": 0.50}),
    ExperimentSpec("LF-20", "LF-P2", "a500", 500, 16, "a500 FE-50 + Focal g=3.0, a=0.75",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="focal", loss_params={"gamma": 3.0, "alpha": 0.75}),
    # =========================================================================
    # LF-P3: WeightedSum Loss (8 experiments)
    # Blend of BCE (calibration) + SoftAUC (ranking).
    # =========================================================================
    # FE-06 architecture
    ExperimentSpec("LF-21", "LF-P3", "a100", 100, 32, "a100 FE-06 + WeightedSum a=0.3, g=2.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.3, "gamma": 2.0}),
    ExperimentSpec("LF-22", "LF-P3", "a100", 100, 32, "a100 FE-06 + WeightedSum a=0.5, g=2.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.5, "gamma": 2.0}),
    ExperimentSpec("LF-23", "LF-P3", "a100", 100, 32, "a100 FE-06 + WeightedSum a=0.7, g=2.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.7, "gamma": 2.0}),
    ExperimentSpec("LF-24", "LF-P3", "a100", 100, 32, "a100 FE-06 + WeightedSum a=0.5, g=3.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.5, "gamma": 3.0}),
    # FE-50 architecture
    ExperimentSpec("LF-25", "LF-P3", "a500", 500, 16, "a500 FE-50 + WeightedSum a=0.3, g=2.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedsum", loss_params={"alpha": 0.3, "gamma": 2.0}),
    ExperimentSpec("LF-26", "LF-P3", "a500", 500, 16, "a500 FE-50 + WeightedSum a=0.5, g=2.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedsum", loss_params={"alpha": 0.5, "gamma": 2.0}),
    ExperimentSpec("LF-27", "LF-P3", "a500", 500, 16, "a500 FE-50 + WeightedSum a=0.7, g=2.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedsum", loss_params={"alpha": 0.7, "gamma": 2.0}),
    ExperimentSpec("LF-28", "LF-P3", "a500", 500, 16, "a500 FE-50 + WeightedSum a=0.5, g=3.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedsum", loss_params={"alpha": 0.5, "gamma": 3.0}),
    # =========================================================================
    # LF-P4: WeightedBCE Loss (6 experiments)
    # Simple positive class weighting for class imbalance.
    # =========================================================================
    # FE-06 architecture
    ExperimentSpec("LF-29", "LF-P4", "a100", 100, 32, "a100 FE-06 + WeightedBCE pw=2.0",
                   loss_fn="weightedbce", loss_params={"pos_weight": 2.0}),
    ExperimentSpec("LF-30", "LF-P4", "a100", 100, 32, "a100 FE-06 + WeightedBCE pw=4.0",
                   loss_fn="weightedbce", loss_params={"pos_weight": 4.0}),
    ExperimentSpec("LF-31", "LF-P4", "a100", 100, 32, "a100 FE-06 + WeightedBCE pw=8.0",
                   loss_fn="weightedbce", loss_params={"pos_weight": 8.0}),
    # FE-50 architecture
    ExperimentSpec("LF-32", "LF-P4", "a500", 500, 16, "a500 FE-50 + WeightedBCE pw=2.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedbce", loss_params={"pos_weight": 2.0}),
    ExperimentSpec("LF-33", "LF-P4", "a500", 500, 16, "a500 FE-50 + WeightedBCE pw=4.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedbce", loss_params={"pos_weight": 4.0}),
    ExperimentSpec("LF-34", "LF-P4", "a500", 500, 16, "a500 FE-50 + WeightedBCE pw=8.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="weightedbce", loss_params={"pos_weight": 8.0}),
    # =========================================================================
    # LF-P5: LabelSmoothing Loss (6 experiments)
    # Prevents overconfident predictions, improves calibration.
    # =========================================================================
    # FE-06 architecture
    ExperimentSpec("LF-35", "LF-P5", "a100", 100, 32, "a100 FE-06 + LabelSmoothing e=0.05",
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.05}),
    ExperimentSpec("LF-36", "LF-P5", "a100", 100, 32, "a100 FE-06 + LabelSmoothing e=0.10",
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.10}),
    ExperimentSpec("LF-37", "LF-P5", "a100", 100, 32, "a100 FE-06 + LabelSmoothing e=0.20",
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.20}),
    # FE-50 architecture
    ExperimentSpec("LF-38", "LF-P5", "a500", 500, 16, "a500 FE-50 + LabelSmoothing e=0.05",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.05}),
    ExperimentSpec("LF-39", "LF-P5", "a500", 500, 16, "a500 FE-50 + LabelSmoothing e=0.10",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.10}),
    ExperimentSpec("LF-40", "LF-P5", "a500", 500, 16, "a500 FE-50 + LabelSmoothing e=0.20",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.20}),
    # =========================================================================
    # LF-P6: a200 Cross-Validation (8 experiments)
    # Apply best loss functions to FE-04 (a200 baseline) for cross-validation.
    # FE-04 architecture: a200, d_embed=64, d_model=128, n_layers=4, n_heads=8
    # =========================================================================
    ExperimentSpec("LF-41", "LF-P6", "a200", 206, 64, "a200 FE-04 + SoftAUC g=2.0",
                   loss_fn="softauc", loss_params={"gamma": 2.0}),
    ExperimentSpec("LF-42", "LF-P6", "a200", 206, 64, "a200 FE-04 + SoftAUC g=3.0",
                   loss_fn="softauc", loss_params={"gamma": 3.0}),
    ExperimentSpec("LF-43", "LF-P6", "a200", 206, 64, "a200 FE-04 + Focal g=2.0, a=0.50",
                   loss_fn="focal", loss_params={"gamma": 2.0, "alpha": 0.50}),
    ExperimentSpec("LF-44", "LF-P6", "a200", 206, 64, "a200 FE-04 + Focal g=3.0, a=0.75",
                   loss_fn="focal", loss_params={"gamma": 3.0, "alpha": 0.75}),
    ExperimentSpec("LF-45", "LF-P6", "a200", 206, 64, "a200 FE-04 + WeightedSum a=0.5, g=2.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.5, "gamma": 2.0}),
    ExperimentSpec("LF-46", "LF-P6", "a200", 206, 64, "a200 FE-04 + WeightedSum a=0.3, g=2.0",
                   loss_fn="weightedsum", loss_params={"alpha": 0.3, "gamma": 2.0}),
    ExperimentSpec("LF-47", "LF-P6", "a200", 206, 64, "a200 FE-04 + WeightedBCE pw=4.0",
                   loss_fn="weightedbce", loss_params={"pos_weight": 4.0}),
    ExperimentSpec("LF-48", "LF-P6", "a200", 206, 64, "a200 FE-04 + LabelSmoothing e=0.10",
                   loss_fn="labelsmoothing", loss_params={"epsilon": 0.10}),
    # =========================================================================
    # LF-P7: Mild Focal Loss (6 experiments)
    # Subtle focus: gamma 0.5-1.0, alpha=0.5 (balanced)
    # User requirement: FP penalized at most 50% more than FN (ratio ≤ 1.5)
    # =========================================================================
    ExperimentSpec("LF-49", "LF-P7", "a100", 100, 32, "a100 FE-06 + MildFocal g=0.5, a=0.5",
                   loss_fn="mildfocal", loss_params={"gamma": 0.5, "alpha": 0.5}),
    ExperimentSpec("LF-50", "LF-P7", "a100", 100, 32, "a100 FE-06 + MildFocal g=0.75, a=0.5",
                   loss_fn="mildfocal", loss_params={"gamma": 0.75, "alpha": 0.5}),
    ExperimentSpec("LF-51", "LF-P7", "a100", 100, 32, "a100 FE-06 + MildFocal g=1.0, a=0.5",
                   loss_fn="mildfocal", loss_params={"gamma": 1.0, "alpha": 0.5}),
    ExperimentSpec("LF-52", "LF-P7", "a500", 500, 16, "a500 FE-50 + MildFocal g=0.5, a=0.5",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="mildfocal", loss_params={"gamma": 0.5, "alpha": 0.5}),
    ExperimentSpec("LF-53", "LF-P7", "a500", 500, 16, "a500 FE-50 + MildFocal g=0.75, a=0.5",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="mildfocal", loss_params={"gamma": 0.75, "alpha": 0.5}),
    ExperimentSpec("LF-54", "LF-P7", "a500", 500, 16, "a500 FE-50 + MildFocal g=1.0, a=0.5",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="mildfocal", loss_params={"gamma": 1.0, "alpha": 0.5}),
    # =========================================================================
    # LF-P8: Asymmetric Focal Loss (6 experiments)
    # Subtle asymmetry: gamma_neg/gamma_pos ≤ 1.5
    # =========================================================================
    ExperimentSpec("LF-55", "LF-P8", "a100", 100, 32, "a100 FE-06 + AsymFocal gp=1.0, gn=1.5",
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 1.0, "gamma_neg": 1.5, "alpha": 0.5}),
    ExperimentSpec("LF-56", "LF-P8", "a100", 100, 32, "a100 FE-06 + AsymFocal gp=0.75, gn=1.0",
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 0.75, "gamma_neg": 1.0, "alpha": 0.5}),
    ExperimentSpec("LF-57", "LF-P8", "a100", 100, 32, "a100 FE-06 + AsymFocal gp=1.0, gn=1.25",
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 1.0, "gamma_neg": 1.25, "alpha": 0.5}),
    ExperimentSpec("LF-58", "LF-P8", "a500", 500, 16, "a500 FE-50 + AsymFocal gp=1.0, gn=1.5",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 1.0, "gamma_neg": 1.5, "alpha": 0.5}),
    ExperimentSpec("LF-59", "LF-P8", "a500", 500, 16, "a500 FE-50 + AsymFocal gp=0.75, gn=1.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 0.75, "gamma_neg": 1.0, "alpha": 0.5}),
    ExperimentSpec("LF-60", "LF-P8", "a500", 500, 16, "a500 FE-50 + AsymFocal gp=1.0, gn=1.25",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="asymmetricfocal", loss_params={"gamma_pos": 1.0, "gamma_neg": 1.25, "alpha": 0.5}),
    # =========================================================================
    # LF-P9: Entropy Regularized BCE (4 experiments)
    # Encourages confident, differentiated predictions
    # =========================================================================
    ExperimentSpec("LF-61", "LF-P9", "a100", 100, 32, "a100 FE-06 + EntropyReg lambda=0.05",
                   loss_fn="entropyreg", loss_params={"lambda_entropy": 0.05}),
    ExperimentSpec("LF-62", "LF-P9", "a100", 100, 32, "a100 FE-06 + EntropyReg lambda=0.1",
                   loss_fn="entropyreg", loss_params={"lambda_entropy": 0.1}),
    ExperimentSpec("LF-63", "LF-P9", "a500", 500, 16, "a500 FE-50 + EntropyReg lambda=0.05",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="entropyreg", loss_params={"lambda_entropy": 0.05}),
    ExperimentSpec("LF-64", "LF-P9", "a500", 500, 16, "a500 FE-50 + EntropyReg lambda=0.1",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="entropyreg", loss_params={"lambda_entropy": 0.1}),
    # =========================================================================
    # LF-P10: Variance Regularized BCE (4 experiments)
    # Penalizes narrow prediction ranges (probability collapse)
    # =========================================================================
    ExperimentSpec("LF-65", "LF-P10", "a100", 100, 32, "a100 FE-06 + VarianceReg lambda=0.25",
                   loss_fn="variancereg", loss_params={"lambda_var": 0.25}),
    ExperimentSpec("LF-66", "LF-P10", "a100", 100, 32, "a100 FE-06 + VarianceReg lambda=0.5",
                   loss_fn="variancereg", loss_params={"lambda_var": 0.5}),
    ExperimentSpec("LF-67", "LF-P10", "a500", 500, 16, "a500 FE-50 + VarianceReg lambda=0.25",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="variancereg", loss_params={"lambda_var": 0.25}),
    ExperimentSpec("LF-68", "LF-P10", "a500", 500, 16, "a500 FE-50 + VarianceReg lambda=0.5",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="variancereg", loss_params={"lambda_var": 0.5}),
    # =========================================================================
    # LF-P11: Calibrated Focal Loss (2 experiments)
    # Focal loss with calibration term
    # =========================================================================
    ExperimentSpec("LF-69", "LF-P11", "a100", 100, 32, "a100 FE-06 + CalibratedFocal g=1.0",
                   loss_fn="calibratedfocal", loss_params={"gamma": 1.0, "alpha": 0.25, "lambda_cal": 0.1}),
    ExperimentSpec("LF-70", "LF-P11", "a500", 500, 16, "a500 FE-50 + CalibratedFocal g=1.0",
                   d_model=64, n_heads=4, d_ff=256,
                   loss_fn="calibratedfocal", loss_params={"gamma": 1.0, "alpha": 0.25, "lambda_cal": 0.1}),
    # =========================================================================
    # PHASE 3: ALTERNATIVE EMBEDDING EXPERIMENTS (AE-01 to AE-21)
    # Test advanced embedding architectures from feature_embeddings.py
    # =========================================================================
    # =========================================================================
    # AE-P1: Progressive Embedding (4 experiments)
    # Multi-layer compression with nonlinearity
    # =========================================================================
    ExperimentSpec("AE-01", "AE-P1", "a100", 100, 32, "a100 FE-06 + Progressive 2-layer",
                   embedding_type="progressive", embedding_params={"num_layers": 2}),
    ExperimentSpec("AE-02", "AE-P1", "a100", 100, 32, "a100 FE-06 + Progressive 3-layer",
                   embedding_type="progressive", embedding_params={"num_layers": 3}),
    ExperimentSpec("AE-03", "AE-P1", "a500", 500, 16, "a500 FE-50 + Progressive 2-layer",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="progressive", embedding_params={"num_layers": 2}),
    ExperimentSpec("AE-04", "AE-P1", "a500", 500, 16, "a500 FE-50 + Progressive 3-layer",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="progressive", embedding_params={"num_layers": 3}),
    # =========================================================================
    # AE-P2: Bottleneck Embedding (4 experiments)
    # Compress then expand (information bottleneck)
    # =========================================================================
    ExperimentSpec("AE-05", "AE-P2", "a100", 100, 32, "a100 FE-06 + Bottleneck cr=0.25",
                   embedding_type="bottleneck", embedding_params={"compression_ratio": 0.25}),
    ExperimentSpec("AE-06", "AE-P2", "a100", 100, 32, "a100 FE-06 + Bottleneck cr=0.5",
                   embedding_type="bottleneck", embedding_params={"compression_ratio": 0.5}),
    ExperimentSpec("AE-07", "AE-P2", "a500", 500, 16, "a500 FE-50 + Bottleneck cr=0.25",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="bottleneck", embedding_params={"compression_ratio": 0.25}),
    ExperimentSpec("AE-08", "AE-P2", "a500", 500, 16, "a500 FE-50 + Bottleneck cr=0.5",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="bottleneck", embedding_params={"compression_ratio": 0.5}),
    # =========================================================================
    # AE-P3: Multi-Head Embedding (4 experiments)
    # Parallel projections with combination
    # =========================================================================
    ExperimentSpec("AE-09", "AE-P3", "a100", 100, 32, "a100 FE-06 + MultiHead h=4 concat",
                   embedding_type="multihead", embedding_params={"num_heads": 4, "combine_method": "concat"}),
    ExperimentSpec("AE-10", "AE-P3", "a100", 100, 32, "a100 FE-06 + MultiHead h=8 sum",
                   embedding_type="multihead", embedding_params={"num_heads": 8, "combine_method": "sum"}),
    ExperimentSpec("AE-11", "AE-P3", "a500", 500, 16, "a500 FE-50 + MultiHead h=4 concat",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="multihead", embedding_params={"num_heads": 4, "combine_method": "concat"}),
    ExperimentSpec("AE-12", "AE-P3", "a500", 500, 16, "a500 FE-50 + MultiHead h=4 sum",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="multihead", embedding_params={"num_heads": 4, "combine_method": "sum"}),
    # =========================================================================
    # AE-P4: Gated Residual Embedding (4 experiments)
    # GRN style with GLU gating
    # =========================================================================
    ExperimentSpec("AE-13", "AE-P4", "a100", 100, 32, "a100 FE-06 + GatedResidual default",
                   embedding_type="gated_residual", embedding_params={}),
    ExperimentSpec("AE-14", "AE-P4", "a100", 100, 32, "a100 FE-06 + GatedResidual h=128",
                   embedding_type="gated_residual", embedding_params={"hidden_dim": 128}),
    ExperimentSpec("AE-15", "AE-P4", "a500", 500, 16, "a500 FE-50 + GatedResidual default",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="gated_residual", embedding_params={}),
    ExperimentSpec("AE-16", "AE-P4", "a500", 500, 16, "a500 FE-50 + GatedResidual h=64",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="gated_residual", embedding_params={"hidden_dim": 64}),
    # =========================================================================
    # AE-P5: Attention Embedding (5 experiments)
    # Self-attention across features
    # =========================================================================
    ExperimentSpec("AE-17", "AE-P5", "a100", 100, 32, "a100 FE-06 + Attention h=4 pos",
                   embedding_type="attention", embedding_params={"num_heads": 4, "use_position": True}),
    ExperimentSpec("AE-18", "AE-P5", "a100", 100, 32, "a100 FE-06 + Attention h=2 pos",
                   embedding_type="attention", embedding_params={"num_heads": 2, "use_position": True}),
    ExperimentSpec("AE-19", "AE-P5", "a500", 500, 16, "a500 FE-50 + Attention h=4 pos",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="attention", embedding_params={"num_heads": 4, "use_position": True}),
    ExperimentSpec("AE-20", "AE-P5", "a500", 500, 16, "a500 FE-50 + Attention h=2 nopos",
                   d_model=64, n_heads=4, d_ff=256,
                   embedding_type="attention", embedding_params={"num_heads": 2, "use_position": False}),
    ExperimentSpec("AE-21", "AE-P5", "a100", 100, 32, "a100 FE-83 (deep) + Attention h=4",
                   n_layers=8, n_heads=4, dropout=0.6,
                   embedding_type="attention", embedding_params={"num_heads": 4, "use_position": True}),
]

EXPERIMENT_BY_ID = {exp.exp_id: exp for exp in EXPERIMENTS}

# Fixed architecture configuration (ablation-validated)
FIXED_CONFIG = {
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 8,
    "d_ff": 512,
    "context_length": 80,
    "patch_length": 16,
    "stride": 8,
    "dropout": 0.5,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "epochs": 50,
    "use_revin": True,
    "horizon": 1,
}

# Data paths
DATA_PATHS = {
    "a20": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet",
    "a50": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet",
    "a100": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet",
    "a200": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet",
    "a500": PROJECT_ROOT / "data/processed/v1/SPY_dataset_a500_combined.parquet",
}

OUTPUT_DIR = PROJECT_ROOT / "outputs/feature_embedding"


# ============================================================================
# LOSS FUNCTION HELPERS
# ============================================================================

def get_loss_function(spec: ExperimentSpec) -> torch.nn.Module | None:
    """Create loss function from experiment spec.

    Args:
        spec: ExperimentSpec with optional loss_fn and loss_params.

    Returns:
        Loss function instance, or None to use default BCE.
    """
    if spec.loss_fn is None:
        return None  # Use default BCE

    params = spec.loss_params or {}

    if spec.loss_fn == "softauc":
        from src.training.losses import SoftAUCLoss
        return SoftAUCLoss(**params)
    elif spec.loss_fn == "focal":
        from src.training.losses import FocalLoss
        return FocalLoss(**params)
    elif spec.loss_fn == "weightedsum":
        from src.training.losses import WeightedSumLoss
        return WeightedSumLoss(**params)
    elif spec.loss_fn == "weightedbce":
        from src.training.losses import WeightedBCELoss
        return WeightedBCELoss(**params)
    elif spec.loss_fn == "labelsmoothing":
        from src.training.losses import LabelSmoothingBCELoss
        return LabelSmoothingBCELoss(**params)
    elif spec.loss_fn == "mildfocal":
        from src.training.losses import MildFocalLoss
        return MildFocalLoss(**params)
    elif spec.loss_fn == "asymmetricfocal":
        from src.training.losses import AsymmetricFocalLoss
        return AsymmetricFocalLoss(**params)
    elif spec.loss_fn == "entropyreg":
        from src.training.losses import EntropyRegularizedBCE
        return EntropyRegularizedBCE(**params)
    elif spec.loss_fn == "variancereg":
        from src.training.losses import VarianceRegularizedBCE
        return VarianceRegularizedBCE(**params)
    elif spec.loss_fn == "calibratedfocal":
        from src.training.losses import CalibratedFocalLoss
        return CalibratedFocalLoss(**params)
    else:
        raise ValueError(f"Unknown loss function: {spec.loss_fn}")


def get_early_stop_metric(spec: ExperimentSpec) -> str:
    """Determine early stopping metric based on loss function.

    Calibration-focused losses use val_loss.
    All others optimize ranking, so use val_auc.

    Args:
        spec: ExperimentSpec with optional loss_fn.

    Returns:
        Early stopping metric name.
    """
    if spec.loss_fn in ("labelsmoothing", "entropyreg", "variancereg", "calibratedfocal"):
        return "val_loss"  # Calibration-focused
    return "val_auc"  # Ranking-focused (default)


def get_embedding_layer(spec: ExperimentSpec, dropout: float):
    """Create embedding layer from experiment spec.

    Args:
        spec: ExperimentSpec with optional embedding_type and embedding_params.
        dropout: Dropout rate to use in the embedding layer.

    Returns:
        Configured embedding module, or None if no custom embedding specified.

    Raises:
        ValueError: If embedding_type requires d_embed but it's not set.
    """
    if spec.embedding_type is None:
        return None

    if spec.d_embed is None:
        raise ValueError(f"embedding_type={spec.embedding_type} requires d_embed to be set")

    from src.models.feature_embeddings import create_feature_embedding

    return create_feature_embedding(
        embedding_type=spec.embedding_type,
        num_features=spec.num_features,
        d_embed=spec.d_embed,
        dropout=dropout,
        **(spec.embedding_params or {}),
    )


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> dict[str, Any]:
    """Evaluate model with comprehensive metrics.

    Returns:
        Dict with precision, recall, auc, prediction statistics, etc.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(int)

    # Handle edge case where only one class in labels
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None

    return {
        # Primary metrics (decision-making)
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        # Secondary metrics
        "auc": auc,
        "accuracy": accuracy_score(labels, binary_preds),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        # Prediction diagnostics
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "pred_range": float(preds.max() - preds.min()),
        # Sample statistics
        "n_positive_preds": int((preds >= 0.5).sum()),
        "n_samples": len(labels),
        "class_balance": float(labels.mean()),
        "positive_rate": float(labels.sum() / len(labels)),
    }


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(spec: ExperimentSpec, dry_run: bool = False) -> dict[str, Any]:
    """Run a single experiment from the matrix.

    Args:
        spec: ExperimentSpec defining the experiment
        dry_run: If True, only estimate params without training

    Returns:
        Dict with all experiment results
    """
    print("=" * 70)
    print(f"EXPERIMENT: {spec.exp_id} - {spec.description}")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    d_embed_str = f"dembed_{spec.d_embed}" if spec.d_embed else "dembed_none"
    exp_output_dir = OUTPUT_DIR / f"{spec.tier}_{d_embed_str}"
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Get effective config values (spec overrides take precedence)
    eff_d_model = spec.get_config("d_model", FIXED_CONFIG)
    eff_n_layers = spec.get_config("n_layers", FIXED_CONFIG)
    eff_n_heads = spec.get_config("n_heads", FIXED_CONFIG)
    eff_d_ff = spec.get_config("d_ff", FIXED_CONFIG)
    eff_dropout = spec.get_config("dropout", FIXED_CONFIG)
    eff_weight_decay = spec.weight_decay if spec.weight_decay is not None else 0.0

    # Estimate parameters
    est_params = estimate_param_count_with_embedding(
        d_model=eff_d_model,
        n_layers=eff_n_layers,
        n_heads=eff_n_heads,
        d_ff=eff_d_ff,
        num_features=spec.num_features,
        d_embed=spec.d_embed,
        context_length=FIXED_CONFIG["context_length"],
        patch_len=FIXED_CONFIG["patch_length"],
        stride=FIXED_CONFIG["stride"],
    )
    print(f"\nEstimated parameters: {est_params:,}")

    if dry_run:
        return {
            "exp_id": spec.exp_id,
            "tier": spec.tier,
            "d_embed": spec.d_embed,
            "estimated_params": est_params,
            "dry_run": True,
        }

    # Load data
    data_path = DATA_PATHS[spec.tier]
    print(f"\nLoading {data_path}...")
    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    print(f"Data: {len(df)} rows, {len(df.columns)} columns")

    # Experiment config
    experiment_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=FIXED_CONFIG["context_length"],
        horizon=FIXED_CONFIG["horizon"],
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Model config with d_embed and overrides
    model_config = PatchTSTConfig(
        num_features=spec.num_features,
        context_length=FIXED_CONFIG["context_length"],
        patch_length=FIXED_CONFIG["patch_length"],
        stride=FIXED_CONFIG["stride"],
        d_model=eff_d_model,
        n_heads=eff_n_heads,
        n_layers=eff_n_layers,
        d_ff=eff_d_ff,
        dropout=eff_dropout,
        head_dropout=0.0,
        d_embed=spec.d_embed,
    )

    print(f"\nArchitecture: d={eff_d_model}, L={eff_n_layers}, "
          f"h={eff_n_heads}, d_ff={eff_d_ff}")
    print(f"Feature Embedding: d_embed={spec.d_embed}")
    print(f"Training: lr={FIXED_CONFIG['learning_rate']}, dropout={eff_dropout}, "
          f"wd={eff_weight_decay}, ctx={FIXED_CONFIG['context_length']}")
    print(f"Features: {spec.num_features} ({spec.tier} tier)")

    # SimpleSplitter for proper validation
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=FIXED_CONFIG["context_length"],
        horizon=FIXED_CONFIG["horizon"],
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"\nSplits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Get loss function and early stopping metric
    criterion = get_loss_function(spec)
    early_stop_metric = get_early_stop_metric(spec)

    if spec.loss_fn:
        print(f"Loss: {spec.loss_fn} {spec.loss_params or {}}")
        print(f"Early stopping: {early_stop_metric}")

    # Trainer with RevIN
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=FIXED_CONFIG["batch_size"],
        learning_rate=FIXED_CONFIG["learning_rate"],
        epochs=FIXED_CONFIG["epochs"],
        device=device,
        checkpoint_dir=exp_output_dir,
        split_indices=split_indices,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
        early_stopping_metric=early_stop_metric,
        criterion=criterion,
        use_revin=FIXED_CONFIG["use_revin"],
        high_prices=high_prices,
        weight_decay=eff_weight_decay,
    )

    # Replace feature embedding if custom embedding type specified
    if spec.embedding_type is not None:
        custom_embed = get_embedding_layer(spec, eff_dropout)
        if custom_embed is not None:
            trainer.model.feature_embed = custom_embed
            trainer.model.feature_embed.to(trainer.device)
            print(f"Custom Embedding: {spec.embedding_type} {spec.embedding_params or {}}")

    # Verify actual parameter count
    actual_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Actual parameters: {actual_params:,}")

    # Train
    print(f"\nTraining for {FIXED_CONFIG['epochs']} epochs...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    # Evaluate on validation
    val_metrics = evaluate_model(trainer.model, trainer.val_dataloader, trainer.device)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"Stopped early: {result.get('stopped_early', False)}")
    print(f"\nValidation (2023-2024, {val_metrics['n_samples']} samples):")
    print(f"  PRECISION: {val_metrics['precision']:.4f}")
    print(f"  RECALL: {val_metrics['recall']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Pred Range: [{val_metrics['pred_min']:.4f}, {val_metrics['pred_max']:.4f}]")
    print(f"  Class Balance: {val_metrics['class_balance']:.3f} ({int(val_metrics['positive_rate']*val_metrics['n_samples'])} positives)")

    # Compile results
    results = {
        "exp_id": spec.exp_id,
        "priority": spec.priority,
        "description": spec.description,
        "tier": spec.tier,
        "num_features": spec.num_features,
        "d_embed": spec.d_embed,
        "architecture": {
            "d_model": eff_d_model,
            "n_layers": eff_n_layers,
            "n_heads": eff_n_heads,
            "d_ff": eff_d_ff,
            "context_length": FIXED_CONFIG["context_length"],
            "patch_length": FIXED_CONFIG["patch_length"],
            "stride": FIXED_CONFIG["stride"],
        },
        "hyperparameters": {
            "dropout": eff_dropout,
            "learning_rate": FIXED_CONFIG["learning_rate"],
            "batch_size": FIXED_CONFIG["batch_size"],
            "epochs": FIXED_CONFIG["epochs"],
            "use_revin": FIXED_CONFIG["use_revin"],
        },
        "loss_function": {
            "loss_fn": spec.loss_fn,
            "loss_params": spec.loss_params,
            "early_stopping_metric": early_stop_metric,
        },
        "parameters": {
            "estimated": est_params,
            "actual": actual_params,
        },
        "splits": {
            "train": len(split_indices.train_indices),
            "val": len(split_indices.val_indices),
            "test": len(split_indices.test_indices),
        },
        "training": {
            "train_loss": result.get("train_loss"),
            "val_loss": result.get("val_loss"),
            "val_auc": result.get("val_auc"),
            "stopped_early": result.get("stopped_early", False),
            "training_time_min": elapsed / 60,
        },
        "val_metrics": val_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_path = exp_output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def run_experiments_by_priority(priority: str, dry_run: bool = False) -> list[dict]:
    """Run all experiments with given priority."""
    specs = [exp for exp in EXPERIMENTS if exp.priority == priority]
    print(f"\nRunning {len(specs)} experiments with priority {priority}")

    all_results = []
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] Starting {spec.exp_id}...")
        try:
            results = run_experiment(spec, dry_run=dry_run)
            all_results.append(results)

        except Exception as e:
            print(f"ERROR in {spec.exp_id}: {e}")
            all_results.append({
                "exp_id": spec.exp_id,
                "error": str(e),
            })

    return all_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Feature Embedding experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id FE-01
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority P1
    ./venv/bin/python experiments/feature_embedding/run_experiments.py --all --dry-run
        """,
    )
    parser.add_argument("--exp-id", type=str, help="Single experiment ID (e.g., FE-01, LF-01, AE-01)")
    parser.add_argument("--priority", type=str,
                        choices=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12",
                                 "LF-P1", "LF-P2", "LF-P3", "LF-P4", "LF-P5", "LF-P6",
                                 "LF-P7", "LF-P8", "LF-P9", "LF-P10", "LF-P11",
                                 "AE-P1", "AE-P2", "AE-P3", "AE-P4", "AE-P5"],
                        help="Run all experiments with given priority (P1-P12 architecture, LF-P1-P11 loss, AE-P1-P5 embedding)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only estimate params, don't train")
    parser.add_argument("--list", action="store_true", help="List all experiments")

    args = parser.parse_args()

    if args.list:
        print("\nFeature Embedding Experiments:")
        print("-" * 70)
        for exp in EXPERIMENTS:
            est = estimate_param_count_with_embedding(
                d_model=exp.get_config("d_model", FIXED_CONFIG),
                n_layers=exp.get_config("n_layers", FIXED_CONFIG),
                n_heads=exp.get_config("n_heads", FIXED_CONFIG),
                d_ff=exp.get_config("d_ff", FIXED_CONFIG),
                num_features=exp.num_features,
                d_embed=exp.d_embed,
                context_length=FIXED_CONFIG["context_length"],
                patch_len=FIXED_CONFIG["patch_length"],
                stride=FIXED_CONFIG["stride"],
            )
            print(f"  {exp.exp_id} [{exp.priority}] {exp.tier:5} d_embed={str(exp.d_embed):4} -> {est:>10,} params")
            print(f"      {exp.description}")
        return

    if args.exp_id:
        if args.exp_id not in EXPERIMENT_BY_ID:
            print(f"ERROR: Unknown experiment ID '{args.exp_id}'")
            print(f"Valid IDs: {list(EXPERIMENT_BY_ID.keys())}")
            sys.exit(1)
        spec = EXPERIMENT_BY_ID[args.exp_id]
        run_experiment(spec, dry_run=args.dry_run)

    elif args.priority:
        run_experiments_by_priority(args.priority, dry_run=args.dry_run)

    elif args.all:
        all_priorities = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12",
                          "LF-P1", "LF-P2", "LF-P3", "LF-P4", "LF-P5", "LF-P6",
                          "LF-P7", "LF-P8", "LF-P9", "LF-P10", "LF-P11",
                          "AE-P1", "AE-P2", "AE-P3", "AE-P4", "AE-P5"]
        for priority in all_priorities:
            run_experiments_by_priority(priority, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nTip: Use --list to see all experiments")


if __name__ == "__main__":
    main()
