# Extended Experiments: Advanced Training Techniques

**Phase**: 6C Extension (Advanced Training Methods)
**Status**: Implementation Complete, Ready for Execution
**Created**: 2026-02-08
**Total Experiments**: 114 across 6 categories

---

## Executive Summary

This document describes 6 new experiment categories testing innovative approaches for training neural networks on chaotic financial market data. These extend the feature embedding experiments (FE-01 to FE-56) with advanced training techniques.

### Categories Overview

| Category | Prefix | Experiments | Goal |
|----------|--------|-------------|------|
| Data Augmentation | DA | 24 | Improve generalization through synthetic data |
| Noise-Robust Training | NR | 18 | Handle label noise inherent in financial data |
| Curriculum Learning | CL | 18 | Train on easier samples first |
| Regime Detection | RD | 18 | Condition training on market regimes |
| Multi-Scale Temporal | MS | 18 | Capture patterns at multiple time horizons |
| Contrastive Pre-training | CP | 18 | Self-supervised learning before fine-tuning |

---

## Research Questions

### Core Hypothesis

Financial time-series prediction is fundamentally different from typical ML tasks because:
1. **Labels are noisy**: Price movements are inherently unpredictable; "correct" labels may still be wrong
2. **Regime shifts exist**: Bull/bear/sideways markets have different dynamics
3. **Multiple timescales matter**: Short-term noise vs long-term trends
4. **Data is limited**: Can't simply collect more labeled data

### Specific Questions

1. **Data Augmentation (DA)**: Can we improve generalization by synthetically expanding training data while preserving financial dynamics?

2. **Noise-Robust Training (NR)**: Do noise-robust loss functions help when labels themselves are uncertain?

3. **Curriculum Learning (CL)**: Does training on "easier" samples first (low volatility, high confidence) improve final performance?

4. **Regime Detection (RD)**: Can we improve predictions by conditioning on detected market regimes (volatility, trend)?

5. **Multi-Scale Temporal (MS)**: Do multi-scale architectures capture both short-term and long-term patterns better?

6. **Contrastive Pre-training (CP)**: Can self-supervised pre-training learn useful representations before fine-tuning?

---

## Category Details

### 1. Data Augmentation (DA) - 24 Experiments

**Goal**: Improve generalization through synthetic training data augmentation.

**Methods**:
- **Jitter**: Add Gaussian noise to features (std: 0.005, 0.01, 0.02)
- **Scale**: Random scaling of feature values (range: 0.1, 0.2)
- **Mixup**: Interpolate between training samples (alpha: 0.1, 0.2, 0.4)
- **Time Warp**: DTW-based temporal warping (factor: 0.1, 0.2)
- **Combined**: Best jitter + mixup combinations

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| DA-P1 | Jitter | DA-01 to DA-06 | 6 |
| DA-P2 | Scale | DA-07 to DA-12 | 6 |
| DA-P3 | Mixup | DA-13 to DA-18 | 6 |
| DA-P4 | Time Warp | DA-19 to DA-22 | 4 |
| DA-P5 | Combined | DA-23 to DA-24 | 2 |

**Implementation**: `src/data/augmentation.py`
- `JitterTransform`: Gaussian noise with configurable std
- `ScaleTransform`: Random feature scaling
- `MixupTransform`: Sample interpolation
- `TimeWarpTransform`: Temporal warping via interpolation
- `get_augmentation_transform()`: Factory function

---

### 2. Noise-Robust Training (NR) - 18 Experiments

**Goal**: Handle inherent label noise in financial data.

**Hypothesis**: Financial labels are inherently noisy because:
- Price movements are partially random
- Threshold-based labels create boundary uncertainty
- Future information is unknown at prediction time

**Methods**:
- **Bootstrap Loss**: Blend model predictions with targets (beta: 0.6-0.9)
- **Co-teaching**: Train two networks that teach each other (forget_rate: 0.1-0.3)
- **Forward Correction**: Use noise transition matrix estimation
- **Confidence Learning**: Down-weight suspected mislabeled samples

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| NR-P1 | Bootstrap | NR-01 to NR-06 | 6 |
| NR-P2 | Co-teaching | NR-07 to NR-12 | 6 |
| NR-P3 | Forward Correction | NR-13 to NR-16 | 4 |
| NR-P4 | Confidence Learning | NR-17 to NR-18 | 2 |

**Implementation**:
- `src/training/losses.py`: BootstrapLoss, ForwardCorrectionLoss, ConfidenceLearningLoss
- `src/training/coteaching.py`: CoTeachingTrainer, EnsembleModel

---

### 3. Curriculum Learning (CL) - 18 Experiments

**Goal**: Train on easier samples first, gradually introducing harder ones.

**Hypothesis**: Models may learn better by:
1. First learning clear patterns (low volatility, high confidence)
2. Then generalizing to harder cases (high volatility, low confidence)

**Methods**:
- **Loss-based**: Rank samples by training loss (easy = low loss)
- **Confidence-based**: Start with high-confidence predictions
- **Volatility-based**: Start with low-volatility periods (financial-specific)
- **Anti-curriculum**: Start with hard samples (baseline comparison)

**Parameters**:
- `initial_pct`: Start with X% of easiest samples (20%, 30%, 50%)
- `growth_rate`: Add X% more samples per epoch (5%, 10%)

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| CL-P1 | Loss-based | CL-01 to CL-06 | 6 |
| CL-P2 | Confidence-based | CL-07 to CL-12 | 6 |
| CL-P3 | Volatility-based | CL-13 to CL-16 | 4 |
| CL-P4 | Anti-curriculum | CL-17 to CL-18 | 2 |

**Implementation**: `src/training/curriculum.py`
- `CurriculumSampler`: Progressive sample inclusion
- `LossDifficultyScorer`, `ConfidenceDifficultyScorer`, `VolatilityDifficultyScorer`
- `AntiCurriculumSampler`: Hardest samples first
- `get_curriculum_sampler()`: Factory function

---

### 4. Regime Detection (RD) - 18 Experiments

**Goal**: Detect market regimes and condition training accordingly.

**Hypothesis**: Different market regimes have different dynamics:
- High volatility: More noise, larger price swings
- Bull/Bear/Sideways: Different trend patterns
- Regime conditioning may improve predictions within each regime

**Methods**:
- **Volatility**: Classify as low/medium/high volatility (thresholds: 1%, 2%)
- **Trend**: SMA or ADX-based bull/bear/sideways detection
- **Cluster**: Learn regimes via K-means clustering
- **Regime-Gated**: Different model heads per regime

**Conditioning Options**:
- Loss weighting: Weight loss higher for certain regimes
- Embedding: Add regime embedding to model input
- Multi-head: Separate prediction heads per regime

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| RD-P1 | Volatility | RD-01 to RD-06 | 6 |
| RD-P2 | Trend | RD-07 to RD-12 | 6 |
| RD-P3 | Learned (Cluster) | RD-13 to RD-16 | 4 |
| RD-P4 | Regime-Gated | RD-17 to RD-18 | 2 |

**Implementation**: `src/training/regime.py`
- `VolatilityRegimeDetector`, `TrendRegimeDetector`, `ClusterRegimeDetector`
- `RegimeLossWeighter`: Apply regime-dependent loss weighting
- `RegimeEmbedding`: Learnable regime embedding
- `get_regime_detector()`: Factory function

---

### 5. Multi-Scale Temporal (MS) - 18 Experiments

**Goal**: Capture patterns at different temporal scales.

**Hypothesis**: Financial data has patterns at multiple timescales:
- Short-term: Intraday momentum, mean reversion
- Medium-term: Multi-day trends, volatility clusters
- Long-term: Macro trends, seasonal patterns

**Methods**:
- **Hierarchical Pool**: Pool patches at scales [1,2,4] or [1,2,4,8]
- **Multi-Patch**: Parallel patch sizes (8,16), (8,16,32), (5,10,20)
- **Dilated Conv**: Dilated temporal convolutions (rates 1,2,4,8)
- **Cross-Scale Attention**: Attention between different temporal scales

**Fusion Options**: concat, sum, attention-weighted

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| MS-P1 | Hierarchical Pool | MS-01 to MS-06 | 6 |
| MS-P2 | Multi-Patch | MS-07 to MS-12 | 6 |
| MS-P3 | Dilated Conv | MS-13 to MS-16 | 4 |
| MS-P4 | Cross-Scale Attention | MS-17 to MS-18 | 2 |

**Implementation**: `src/models/multiscale.py`
- `HierarchicalTemporalPool`: Multi-scale pooling with fusion
- `MultiScalePatchEmbedding`: Parallel patch sizes
- `DilatedTemporalConv`: Dilated convolutions
- `CrossScaleAttention`: Cross-scale attention mechanism
- `get_multiscale_module()`: Factory function

---

### 6. Contrastive Pre-training (CP) - 18 Experiments

**Goal**: Learn useful representations through self-supervised pre-training.

**Hypothesis**: Self-supervised contrastive learning can:
1. Learn useful features without labels
2. Capture temporal structure in time-series
3. Provide better initialization for fine-tuning

**Methods**:
- **SimCLR**: Contrastive learning with augmented views (temperature: 0.05, 0.1, 0.2)
- **TS2Vec**: Hierarchical temporal contrastive learning
- **BYOL**: Bootstrap Your Own Latent (no negative samples)
- **Fine-tune**: Load best pretrained encoder, fine-tune downstream

**Workflow**:
1. Pre-train encoder with contrastive loss (unsupervised)
2. Save encoder checkpoint
3. Load encoder and fine-tune with classification head

**Priorities**:
| Priority | Focus | Experiments | Count |
|----------|-------|-------------|-------|
| CP-P1 | SimCLR | CP-01 to CP-06 | 6 |
| CP-P2 | TS2Vec | CP-07 to CP-12 | 6 |
| CP-P3 | BYOL | CP-13 to CP-16 | 4 |
| CP-P4 | Fine-tune | CP-17 to CP-18 | 2 |

**Implementation**: `src/training/contrastive.py`
- `ContrastiveLoss`: NT-Xent loss for SimCLR
- `HierarchicalContrastiveLoss`: TS2Vec-style loss
- `ProjectionHead`: MLP projection for contrastive space
- `ContrastiveEncoder`: Wrapper with projection head
- `ContrastiveTrainer`: Pre-training workflow
- `get_contrastive_trainer()`: Factory function

---

## Experiment Execution

### Implementation Phases

```
Phase 1: Foundation (no dependencies)
├── DA: Data Augmentation (24 experiments)
└── NR: Noise-Robust Training (18 experiments)

Phase 2: Training Modifications
├── CL: Curriculum Learning (18 experiments)
└── RD: Regime Detection (18 experiments)

Phase 3: Architecture Extensions
└── MS: Multi-Scale Temporal (18 experiments)

Phase 4: Pre-training (depends on DA)
└── CP: Contrastive Pre-training (18 experiments)
```

### Running Experiments

**Single experiment**:
```bash
./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id DA-01
```

**By priority**:
```bash
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority DA-P1
```

**All extended experiments**:
```bash
./scripts/run_extended_experiments.sh
```

**By category**:
```bash
./scripts/run_da_experiments.sh  # Data Augmentation
./scripts/run_nr_experiments.sh  # Noise-Robust
./scripts/run_cl_experiments.sh  # Curriculum Learning
./scripts/run_rd_experiments.sh  # Regime Detection
./scripts/run_ms_experiments.sh  # Multi-Scale Temporal
./scripts/run_cp_experiments.sh  # Contrastive Pre-training
```

**Dry run (estimate params only)**:
```bash
./scripts/run_extended_experiments.sh --dry-run
```

### Runner Script Options

```bash
# Run specific phase
./scripts/run_extended_experiments.sh --phase 1  # DA + NR
./scripts/run_extended_experiments.sh --phase 2  # CL + RD
./scripts/run_extended_experiments.sh --phase 3  # MS
./scripts/run_extended_experiments.sh --phase 4  # CP
```

---

## Complete Experiment Matrix

### Data Augmentation (DA-01 to DA-24)

| ID | Priority | Tier | d_embed | Augmentation | Parameters |
|----|----------|------|---------|--------------|------------|
| DA-01 | DA-P1 | a100 | 32 | jitter | std=0.005 |
| DA-02 | DA-P1 | a100 | 32 | jitter | std=0.01 |
| DA-03 | DA-P1 | a100 | 32 | jitter | std=0.02 |
| DA-04 | DA-P1 | a500 | 16 | jitter | std=0.005 |
| DA-05 | DA-P1 | a500 | 16 | jitter | std=0.01 |
| DA-06 | DA-P1 | a500 | 16 | jitter | std=0.02 |
| DA-07 | DA-P2 | a100 | 32 | scale | range=0.1 |
| DA-08 | DA-P2 | a100 | 32 | scale | range=0.2 |
| DA-09 | DA-P2 | a500 | 16 | scale | range=0.1 |
| DA-10 | DA-P2 | a500 | 16 | scale | range=0.2 |
| DA-11 | DA-P2 | a200 | 64 | scale | range=0.1 |
| DA-12 | DA-P2 | a200 | 64 | scale | range=0.2 |
| DA-13 | DA-P3 | a100 | 32 | mixup | alpha=0.1 |
| DA-14 | DA-P3 | a100 | 32 | mixup | alpha=0.2 |
| DA-15 | DA-P3 | a100 | 32 | mixup | alpha=0.4 |
| DA-16 | DA-P3 | a500 | 16 | mixup | alpha=0.1 |
| DA-17 | DA-P3 | a500 | 16 | mixup | alpha=0.2 |
| DA-18 | DA-P3 | a500 | 16 | mixup | alpha=0.4 |
| DA-19 | DA-P4 | a100 | 32 | timewarp | factor=0.1 |
| DA-20 | DA-P4 | a100 | 32 | timewarp | factor=0.2 |
| DA-21 | DA-P4 | a500 | 16 | timewarp | factor=0.1 |
| DA-22 | DA-P4 | a500 | 16 | timewarp | factor=0.2 |
| DA-23 | DA-P5 | a100 | 32 | combined | jitter+mixup |
| DA-24 | DA-P5 | a500 | 16 | combined | jitter+mixup |

### Noise-Robust Training (NR-01 to NR-18)

| ID | Priority | Tier | d_embed | Method | Parameters |
|----|----------|------|---------|--------|------------|
| NR-01 | NR-P1 | a100 | 32 | bootstrap | beta=0.6 |
| NR-02 | NR-P1 | a100 | 32 | bootstrap | beta=0.7 |
| NR-03 | NR-P1 | a100 | 32 | bootstrap | beta=0.8 |
| NR-04 | NR-P1 | a100 | 32 | bootstrap | beta=0.9 |
| NR-05 | NR-P1 | a500 | 16 | bootstrap | beta=0.8 |
| NR-06 | NR-P1 | a500 | 16 | bootstrap | beta=0.9 |
| NR-07 | NR-P2 | a100 | 32 | coteaching | forget=0.1 |
| NR-08 | NR-P2 | a100 | 32 | coteaching | forget=0.2 |
| NR-09 | NR-P2 | a100 | 32 | coteaching | forget=0.3 |
| NR-10 | NR-P2 | a500 | 16 | coteaching | forget=0.1 |
| NR-11 | NR-P2 | a500 | 16 | coteaching | forget=0.2 |
| NR-12 | NR-P2 | a500 | 16 | coteaching | forget=0.3 |
| NR-13 | NR-P3 | a100 | 32 | forward | noise_rate=0.1 |
| NR-14 | NR-P3 | a100 | 32 | forward | noise_rate=0.2 |
| NR-15 | NR-P3 | a500 | 16 | forward | noise_rate=0.1 |
| NR-16 | NR-P3 | a500 | 16 | forward | noise_rate=0.2 |
| NR-17 | NR-P4 | a100 | 32 | confidence | threshold=0.7 |
| NR-18 | NR-P4 | a500 | 16 | confidence | threshold=0.7 |

### Curriculum Learning (CL-01 to CL-18)

| ID | Priority | Tier | d_embed | Strategy | Parameters |
|----|----------|------|---------|----------|------------|
| CL-01 | CL-P1 | a100 | 32 | loss | init=0.2, grow=0.05 |
| CL-02 | CL-P1 | a100 | 32 | loss | init=0.3, grow=0.1 |
| CL-03 | CL-P1 | a100 | 32 | loss | init=0.5, grow=0.1 |
| CL-04 | CL-P1 | a500 | 16 | loss | init=0.2, grow=0.05 |
| CL-05 | CL-P1 | a500 | 16 | loss | init=0.3, grow=0.1 |
| CL-06 | CL-P1 | a500 | 16 | loss | init=0.5, grow=0.1 |
| CL-07 | CL-P2 | a100 | 32 | confidence | init=0.2, grow=0.05 |
| CL-08 | CL-P2 | a100 | 32 | confidence | init=0.3, grow=0.1 |
| CL-09 | CL-P2 | a100 | 32 | confidence | init=0.5, grow=0.1 |
| CL-10 | CL-P2 | a500 | 16 | confidence | init=0.2, grow=0.05 |
| CL-11 | CL-P2 | a500 | 16 | confidence | init=0.3, grow=0.1 |
| CL-12 | CL-P2 | a500 | 16 | confidence | init=0.5, grow=0.1 |
| CL-13 | CL-P3 | a100 | 32 | volatility | init=0.3, grow=0.1 |
| CL-14 | CL-P3 | a100 | 32 | volatility | init=0.5, grow=0.1 |
| CL-15 | CL-P3 | a500 | 16 | volatility | init=0.3, grow=0.1 |
| CL-16 | CL-P3 | a500 | 16 | volatility | init=0.5, grow=0.1 |
| CL-17 | CL-P4 | a100 | 32 | anti | init=0.3, grow=0.1 |
| CL-18 | CL-P4 | a500 | 16 | anti | init=0.3, grow=0.1 |

### Regime Detection (RD-01 to RD-18)

| ID | Priority | Tier | d_embed | Strategy | Parameters |
|----|----------|------|---------|----------|------------|
| RD-01 | RD-P1 | a100 | 32 | volatility | thresholds=(0.01,0.02), loss_weight |
| RD-02 | RD-P1 | a100 | 32 | volatility | thresholds=(0.01,0.02), embedding |
| RD-03 | RD-P1 | a100 | 32 | volatility | thresholds=(0.005,0.015), loss_weight |
| RD-04 | RD-P1 | a500 | 16 | volatility | thresholds=(0.01,0.02), loss_weight |
| RD-05 | RD-P1 | a500 | 16 | volatility | thresholds=(0.01,0.02), embedding |
| RD-06 | RD-P1 | a500 | 16 | volatility | thresholds=(0.005,0.015), loss_weight |
| RD-07 | RD-P2 | a100 | 32 | trend | method=sma, short=10, long=30 |
| RD-08 | RD-P2 | a100 | 32 | trend | method=sma, short=5, long=20 |
| RD-09 | RD-P2 | a100 | 32 | trend | method=adx, threshold=25 |
| RD-10 | RD-P2 | a500 | 16 | trend | method=sma, short=10, long=30 |
| RD-11 | RD-P2 | a500 | 16 | trend | method=sma, short=5, long=20 |
| RD-12 | RD-P2 | a500 | 16 | trend | method=adx, threshold=25 |
| RD-13 | RD-P3 | a100 | 32 | cluster | n_clusters=2 |
| RD-14 | RD-P3 | a100 | 32 | cluster | n_clusters=3 |
| RD-15 | RD-P3 | a100 | 32 | cluster | n_clusters=4 |
| RD-16 | RD-P3 | a500 | 16 | cluster | n_clusters=3 |
| RD-17 | RD-P4 | a100 | 32 | regime_gated | n_heads=3, volatility |
| RD-18 | RD-P4 | a500 | 16 | regime_gated | n_heads=3, volatility |

### Multi-Scale Temporal (MS-01 to MS-18)

| ID | Priority | Tier | d_embed | Type | Parameters |
|----|----------|------|---------|------|------------|
| MS-01 | MS-P1 | a100 | 32 | hierarchical_pool | scales=[1,2,4], fusion=concat |
| MS-02 | MS-P1 | a100 | 32 | hierarchical_pool | scales=[1,2,4], fusion=sum |
| MS-03 | MS-P1 | a100 | 32 | hierarchical_pool | scales=[1,2,4,8], fusion=concat |
| MS-04 | MS-P1 | a500 | 16 | hierarchical_pool | scales=[1,2,4], fusion=concat |
| MS-05 | MS-P1 | a500 | 16 | hierarchical_pool | scales=[1,2,4], fusion=attention |
| MS-06 | MS-P1 | a500 | 16 | hierarchical_pool | scales=[1,2,4,8], fusion=concat |
| MS-07 | MS-P2 | a100 | 32 | multi_patch | sizes=[8,16], fusion=concat |
| MS-08 | MS-P2 | a100 | 32 | multi_patch | sizes=[8,16,32], fusion=concat |
| MS-09 | MS-P2 | a100 | 32 | multi_patch | sizes=[5,10,20], fusion=concat |
| MS-10 | MS-P2 | a500 | 16 | multi_patch | sizes=[8,16], fusion=concat |
| MS-11 | MS-P2 | a500 | 16 | multi_patch | sizes=[8,16,32], fusion=concat |
| MS-12 | MS-P2 | a500 | 16 | multi_patch | sizes=[5,10,20], fusion=concat |
| MS-13 | MS-P3 | a100 | 32 | dilated_conv | rates=[1,2,4] |
| MS-14 | MS-P3 | a100 | 32 | dilated_conv | rates=[1,2,4,8] |
| MS-15 | MS-P3 | a500 | 16 | dilated_conv | rates=[1,2,4] |
| MS-16 | MS-P3 | a500 | 16 | dilated_conv | rates=[1,2,4,8] |
| MS-17 | MS-P4 | a100 | 32 | cross_attention | n_heads=4 |
| MS-18 | MS-P4 | a500 | 16 | cross_attention | n_heads=4 |

### Contrastive Pre-training (CP-01 to CP-18)

| ID | Priority | Tier | d_embed | Type | Parameters |
|----|----------|------|---------|------|------------|
| CP-01 | CP-P1 | a100 | 32 | simclr | temp=0.05, epochs=20 |
| CP-02 | CP-P1 | a100 | 32 | simclr | temp=0.1, epochs=20 |
| CP-03 | CP-P1 | a100 | 32 | simclr | temp=0.2, epochs=20 |
| CP-04 | CP-P1 | a500 | 16 | simclr | temp=0.05, epochs=20 |
| CP-05 | CP-P1 | a500 | 16 | simclr | temp=0.1, epochs=20 |
| CP-06 | CP-P1 | a500 | 16 | simclr | temp=0.2, epochs=20 |
| CP-07 | CP-P2 | a100 | 32 | ts2vec | temp=0.1, lambda=0.5 |
| CP-08 | CP-P2 | a100 | 32 | ts2vec | temp=0.1, lambda=1.0 |
| CP-09 | CP-P2 | a100 | 32 | ts2vec | temp=0.05, lambda=0.5 |
| CP-10 | CP-P2 | a500 | 16 | ts2vec | temp=0.1, lambda=0.5 |
| CP-11 | CP-P2 | a500 | 16 | ts2vec | temp=0.1, lambda=1.0 |
| CP-12 | CP-P2 | a500 | 16 | ts2vec | temp=0.05, lambda=0.5 |
| CP-13 | CP-P3 | a100 | 32 | byol | epochs=30 |
| CP-14 | CP-P3 | a100 | 32 | byol | epochs=50 |
| CP-15 | CP-P3 | a500 | 16 | byol | epochs=30 |
| CP-16 | CP-P3 | a500 | 16 | byol | epochs=50 |
| CP-17 | CP-P4 | a100 | 32 | finetune | from simclr |
| CP-18 | CP-P4 | a500 | 16 | finetune | from simclr |

---

## Implementation Files

### New Source Files

| File | Purpose |
|------|---------|
| `src/data/augmentation.py` | Data augmentation transforms |
| `src/training/coteaching.py` | Co-teaching trainer |
| `src/training/curriculum.py` | Curriculum learning sampler |
| `src/training/regime.py` | Regime detection |
| `src/models/multiscale.py` | Multi-scale modules |
| `src/training/contrastive.py` | Contrastive pre-training |

### Modified Files

| File | Changes |
|------|---------|
| `src/training/losses.py` | Added BootstrapLoss, ForwardCorrectionLoss, ConfidenceLearningLoss |
| `experiments/feature_embedding/run_experiments.py` | Added 12 ExperimentSpec fields, 114 experiments |

### Runner Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_extended_experiments.sh` | All 114 experiments |
| `scripts/run_da_experiments.sh` | Data Augmentation |
| `scripts/run_nr_experiments.sh` | Noise-Robust |
| `scripts/run_cl_experiments.sh` | Curriculum Learning |
| `scripts/run_rd_experiments.sh` | Regime Detection |
| `scripts/run_ms_experiments.sh` | Multi-Scale Temporal |
| `scripts/run_cp_experiments.sh` | Contrastive Pre-training |

### Tests

| File | Coverage |
|------|----------|
| `tests/test_extended_experiments.py` | 40 tests for all new modules |

---

## Expected Outcomes

### Success Criteria

1. **Data Augmentation**: >5% improvement in precision with augmentation vs baseline
2. **Noise-Robust**: Reduced variance across runs, more stable training
3. **Curriculum Learning**: Faster convergence, better final performance
4. **Regime Detection**: Better performance in specific regimes
5. **Multi-Scale**: Improved recall by capturing multi-timescale patterns
6. **Contrastive**: Better feature representations, improved fine-tuning

### Metrics to Track

- **Primary**: Precision (when model says "buy", how often correct?)
- **Secondary**: Recall (of all opportunities, how many caught?)
- **Tertiary**: AUC-ROC (ranking quality)
- **Diagnostic**: Prediction range (detect probability collapse)

---

## References

- **SimCLR**: Chen et al. "A Simple Framework for Contrastive Learning" (2020)
- **TS2Vec**: Yue et al. "TS2Vec: Universal Representation of Time Series" (2022)
- **BYOL**: Grill et al. "Bootstrap Your Own Latent" (2020)
- **Co-teaching**: Han et al. "Co-teaching: Robust Training with Noisy Labels" (2018)
- **Forward Correction**: Patrini et al. "Making DNNs Robust to Label Noise" (2017)
- **Curriculum Learning**: Bengio et al. "Curriculum Learning" (2009)
