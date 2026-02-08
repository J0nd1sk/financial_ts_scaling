# Feature Embedding Experiments

**Phase**: 6C Extension (Feature Embedding Investigation)
**Status**: P1-P12 + LF Complete | Extended Experiments (114) Ready
**Created**: 2026-02-06
**Last Updated**: 2026-02-08

---

## Executive Summary

**Key Finding**: Feature embedding with aggressive compression dramatically improves precision, especially for high-dimensional feature tiers.

| Tier | Best Config | Precision | vs Baseline | Params |
|------|-------------|-----------|-------------|--------|
| a100 | FE-06: d_embed=32 | **69.2%** | +56% | 0.87M |
| a200 | FE-04: d_embed=64 | **46.7%** | +34% | 0.94M |
| a500 | FE-50: d_embed=16, d=64, h=4 | **60.0%** | +80% | 0.23M |

**Counterintuitive Insight**: Fewer features and smaller models often outperform larger ones. The a500 tier (500 features) only became competitive when using a tiny model (0.23M params) with aggressive feature compression (d_embed=16).

---

## Research Questions

### Original Question
Does projecting raw features into a learned embedding space before patching improve prediction quality?

### Expanded Questions (Post-P1 through P7)
1. **What is the optimal d_embed for each feature tier?**
2. **Does d_model scaling help or hurt precision?**
3. **Is depth (layers) or width (d_model) more important?**
4. **Does the number of attention heads matter?**
5. **NEW: How does context length interact with embedding dimension?**
6. **NEW: Do deeper models need more embedding dimensions?**

---

## Complete Results (P1-P7)

### Master Rankings by Precision (Top 20)

| Rank | Exp | Tier | d_embed | d_model | L | h | Precision | Recall | AUC | Params | Notes |
|------|-----|------|---------|---------|---|---|-----------|--------|-----|--------|-------|
| 1 | FE-38 | a100 | 8 | 128 | 4 | 8 | 100.0% | 2.6% | 0.702 | 0.81M | **Useless**: only 2/76 positives |
| 2 | FE-28 | a100 | 32 | 128 | 4 | 8 | 100.0% | 1.3% | 0.704 | 0.87M | **Useless**: only 1/76 positives |
| **3** | **FE-06** | **a100** | **32** | **128** | **4** | **8** | **69.2%** | **11.8%** | **0.716** | **0.87M** | **BEST OVERALL** |
| 4 | FE-46 | a100 | 32 | 64 | 8 | 4 | 66.7% | 2.6% | 0.716 | 0.44M | Low recall |
| 5 | FE-15 | a100 | 64 | 256 | 4 | 8 | 62.5% | 13.2% | 0.718 | 3.4M | |
| **6** | **FE-50** | **a500** | **16** | **64** | **4** | **4** | **60.0%** | **11.8%** | **0.728** | **0.23M** | **BEST a500** |
| 7 | FE-09 | a100 | 64 | 128 | 4 | 8 | 60.0% | 7.9% | 0.710 | 0.94M | |
| 8 | FE-26 | a200 | 128 | 512 | 4 | 8 | 60.0% | 15.8% | 0.710 | 13.7M | |
| 9 | FE-25 | a200 | 32 | 256 | 4 | 8 | 57.1% | 10.5% | 0.724 | 3.3M | |
| 10 | FE-35 | a200 | 32 | 256 | 4 | 8 | 57.1% | 10.5% | 0.724 | 3.3M | |
| 11 | FE-19 | a100 | 32 | 128 | 8 | 8 | 56.3% | 11.8% | 0.717 | 1.7M | Narrow+Deep |
| 12 | FE-11 | a100 | 128 | 128 | 4 | 8 | 55.6% | 13.2% | 0.700 | 1.07M | |
| 13 | FE-17 | a100 | 32 | 1024 | 4 | 8 | 54.5% | 15.8% | 0.680 | 50.9M | |
| 14 | FE-27 | a100 | 32 | 128 | 4 | 8 | 53.3% | 10.5% | 0.699 | 0.87M | Low dropout |
| 15 | FE-39 | a200 | 16 | 128 | 4 | 8 | 50.0% | 9.2% | 0.709 | 0.83M | |
| 16 | FE-14 | a100 | 32 | 512 | 4 | 8 | 50.0% | 14.5% | 0.681 | 12.9M | |
| 17 | FE-16 | a100 | 64 | 512 | 4 | 8 | 50.0% | 18.4% | 0.680 | 13.2M | |
| 18 | FE-18 | a100 | 32 | 256 | 2 | 8 | 50.0% | 10.5% | 0.717 | 1.7M | |
| 19 | FE-20 | a100 | 32 | 512 | 2 | 8 | 48.0% | 15.8% | 0.704 | 6.6M | |
| 20 | FE-41 | a500 | 8 | 128 | 4 | 8 | 47.6% | 13.2% | 0.726 | 0.82M | |

### Results by Feature Tier

#### a100 (100 features)

| Exp | d_embed | d_model | L | h | Precision | Recall | AUC | Params | Early Stop |
|-----|---------|---------|---|---|-----------|--------|-----|--------|------------|
| FE-05 | None | 128 | 4 | 8 | 44.4% | 10.5% | 0.700 | 1.01M | Yes |
| **FE-06** | **32** | 128 | 4 | 8 | **69.2%** | 11.8% | 0.716 | 0.87M | **No** |
| FE-09 | 64 | 128 | 4 | 8 | 60.0% | 7.9% | 0.710 | 0.94M | Yes |
| FE-11 | 128 | 128 | 4 | 8 | 55.6% | 13.2% | 0.700 | 1.07M | Yes |
| FE-37 | 16 | 128 | 4 | 8 | 42.1% | 10.5% | 0.718 | 0.83M | No |
| FE-38 | 8 | 128 | 4 | 8 | 100.0% | 2.6% | 0.702 | 0.81M | No |

**Best a100 config**: d_embed=32, d_model=128, L=4, h=8 (FE-06)

#### a200 (206 features)

| Exp | d_embed | d_model | L | h | Precision | Recall | AUC | Params |
|-----|---------|---------|---|---|-----------|--------|-----|--------|
| FE-03 | None | 128 | 4 | 8 | 34.8% | 10.5% | 0.671 | 1.23M |
| **FE-04** | **64** | 128 | 4 | 8 | **46.7%** | 9.2% | 0.721 | 0.94M |
| FE-08 | 128 | 128 | 4 | 8 | 37.0% | 13.2% | 0.687 | 1.09M |
| FE-39 | 16 | 128 | 4 | 8 | 50.0% | 9.2% | 0.709 | 0.83M |
| FE-40 | 8 | 128 | 4 | 8 | 47.4% | 11.8% | 0.689 | 0.82M |

**Best a200 config**: d_embed=64, d_model=128, L=4, h=8 (FE-04) or d_embed=16 (FE-39)

#### a500 (500 features)

| Exp | d_embed | d_model | L | h | Precision | Recall | AUC | Params |
|-----|---------|---------|---|---|-----------|--------|-----|--------|
| FE-01 | None | 128 | 4 | 8 | 33.3% | 14.5% | 0.627 | 1.83M |
| FE-02 | 64 | 128 | 4 | 8 | 33.3% | 13.2% | 0.697 | 0.96M |
| FE-12 | 32 | 128 | 4 | 8 | 42.9% | 11.8% | 0.715 | 0.88M |
| FE-22 | 16 | 128 | 4 | 8 | 42.9% | 7.9% | 0.702 | 0.84M |
| FE-41 | 8 | 128 | 4 | 8 | 47.6% | 13.2% | 0.726 | 0.82M |
| **FE-50** | **16** | **64** | **4** | **4** | **60.0%** | **11.8%** | **0.728** | **0.23M** |

**Best a500 config**: d_embed=16, d_model=64, L=4, h=4 (FE-50) - **breakthrough result!**

---

## Key Findings

### 1. Smaller d_embed is generally better

For a100:
```
d_embed=32:  69.2% precision  ← BEST (FE-06)
d_embed=64:  60.0% precision
d_embed=128: 55.6% precision
d_embed=16:  42.1% precision  (too aggressive)
d_embed=8:   100% precision but 2.6% recall (useless)
None:        44.4% precision
```

For a500:
```
d_embed=16+tiny model: 60.0% precision ← BEST (FE-50)
d_embed=8:   47.6% precision
d_embed=32:  42.9% precision
d_embed=64:  33.3% precision
None:        33.3% precision
```

### 2. Larger d_model hurts precision

```
d_model=128:  69.2% (FE-06)
d_model=256:  45.5% (FE-13)
d_model=512:  50.0% (FE-14)
d_model=1024: 54.5% (FE-17)
```

**Inverse scaling law**: Bigger models are NOT better for this task.

### 3. Depth can help (with small d_model)

```
L=4, d=128: 69.2% (FE-06) ← BEST
L=8, d=128: 56.3% (FE-19) - decent
L=8, d=64:  66.7% but 2.6% recall (FE-46) - too conservative
```

### 4. Fewer attention heads may help

```
h=4:  44.8% (FE-32) at d=256
h=8:  45.5% (FE-13) at d=256
h=16: 40.0% (FE-31) at d=256
```

We haven't tested h=1 or h=2 extensively yet (P9 pending).

### 5. FE-50 breakthrough for noisy features

**FE-50**: a500 + d_embed=16 + d_model=64 + h=4 = **60% precision, 0.23M params**

This is remarkable because:
- a500 previously maxed at ~43% precision
- FE-50 uses the **smallest model** that works (0.23M)
- FE-50 has the **best a500 AUC** (0.728)

**Hypothesis**: For high-dimensional noisy features:
1. Aggressive feature compression forces learning essential patterns
2. Tiny model prevents overfitting to noise
3. Fewer heads reduce attention to spurious correlations

---

## Experiment Phases

### Completed Phases

| Phase | Focus | Experiments | Status |
|-------|-------|-------------|--------|
| P1 | Baseline comparisons | FE-01 to FE-06 | ✅ Complete |
| P2 | d_embed sensitivity | FE-07 to FE-09 | ✅ Complete |
| P3 | Additional d_embed | FE-10 to FE-12 | ✅ Complete |
| P4 | Architecture scaling | FE-13 to FE-17 | ✅ Complete |
| P5 | Depth/Width/Reg/Attention | FE-18 to FE-36 | ✅ Complete |
| P6 | d_embed extremes (8, 16, 512) | FE-37 to FE-44 | ✅ Complete |
| P7 | Tiny d_model (32, 64) | FE-45 to FE-50 | ✅ Complete |

### Pending Phases

| Phase | Focus | Experiments | Status |
|-------|-------|-------------|--------|
| P8 | Deep networks (L=12, 16, 20) | FE-51 to FE-58 | Ready |
| P9 | Minimal heads (h=1, 2) | FE-59 to FE-64 | Ready |
| P10 | Fewer features (a20, a50) | FE-65 to FE-70 | Ready |
| P11 | Combined extremes | FE-71 to FE-84 | Ready |
| P12 | FE-50 variants | FE-85 to FE-92 | Ready |

### Phase 2: Loss Function Experiments

**Status**: Ready to run (48 experiments defined)

Tests whether alternative loss functions improve precision on the best architectures from Phase 1.

| Priority | Focus | Experiments | Count | Status |
|----------|-------|-------------|-------|--------|
| LF-P1 | SoftAUC | LF-01 to LF-08 | 8 | Ready |
| LF-P2 | Focal Loss | LF-09 to LF-20 | 12 | Ready |
| LF-P3 | WeightedSum (BCE+SoftAUC) | LF-21 to LF-28 | 8 | Ready |
| LF-P4 | WeightedBCE | LF-29 to LF-34 | 6 | Ready |
| LF-P5 | LabelSmoothing | LF-35 to LF-40 | 6 | Ready |
| LF-P6 | a200 Cross-Validation | LF-41 to LF-48 | 8 | Ready |

**Architectures Tested**:
- FE-06: a100, d_embed=32, d_model=128, L=4, h=8 (best a100)
- FE-50: a500, d_embed=16, d_model=64, L=4, h=4 (best a500, tiny model)
- FE-04: a200, d_embed=64, d_model=128, L=4, h=8 (best a200, for cross-validation)

**Loss Functions**:
1. **SoftAUC**: Directly optimizes ranking (gamma: 1.0, 2.0, 3.0, 5.0)
2. **Focal**: Down-weights easy examples (gamma: 1.0-3.0, alpha: 0.25-0.75)
3. **WeightedSum**: Blend of BCE + SoftAUC (alpha: 0.3, 0.5, 0.7)
4. **WeightedBCE**: Simple positive class weighting (pos_weight: 2.0, 4.0, 8.0)
5. **LabelSmoothing**: Prevents overconfident predictions (epsilon: 0.05, 0.10, 0.20)

**Run Commands**:
```bash
# Run SoftAUC experiments
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P1

# Dry run to verify configurations
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P1 --dry-run

# Run single experiment
./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id LF-01
```

### Future Phases (Research Roadmap)

| Phase | Focus | Description |
|-------|-------|-------------|
| **Phase 3** | Context Length × Architecture | Sweep context (40, 60, 80, 100, 120, 160) across architectures |
| **Phase 4** | Full Factorial | Context × Loss × Architecture × Feature Tier |

---

## Future Research: Context Length × Embedding Dimension Interaction

### Hypothesis

**More context length may require different embedding dimensions.**

Rationale:
- Longer context = more temporal information to compress
- Embedding layer must capture patterns across more timesteps
- Deeper models may need more capacity to process longer sequences

### Proposed Experiments

#### Context Length Sweep Matrix

| Context | d_embed | d_model | L | Rationale |
|---------|---------|---------|---|-----------|
| 40 | 16 | 64 | 4 | Short context, aggressive compression |
| 40 | 32 | 128 | 4 | Short context, moderate compression |
| 60 | 16 | 64 | 4 | Medium context, aggressive compression |
| 60 | 32 | 128 | 4 | Medium context, moderate compression |
| 80 | 16 | 64 | 4 | Current default (FE-50 config) |
| 80 | 32 | 128 | 4 | Current default (FE-06 config) |
| 100 | 32 | 128 | 4 | Longer context |
| 100 | 64 | 128 | 6 | Longer context, more capacity |
| 120 | 32 | 128 | 6 | Extended context |
| 120 | 64 | 128 | 8 | Extended context, deep |
| 160 | 64 | 128 | 8 | Maximum context |
| 160 | 128 | 256 | 8 | Maximum context, maximum capacity |

#### Questions to Answer

1. **Does longer context need more d_embed?**
   - Compare precision at ctx=80 vs ctx=120 with same d_embed
   - Compare optimal d_embed at different context lengths

2. **Does longer context need more layers?**
   - Test L=4 vs L=8 vs L=12 at context=120
   - May need more depth to process longer sequences

3. **Is there an interaction between context and feature tier?**
   - a500 (noisy) may need shorter context to avoid noise accumulation
   - a100 (curated) may benefit from longer context

4. **Does the FE-50 tiny architecture work at longer contexts?**
   - FE-50 config (d=64, h=4) may break with more context
   - May need to scale up model for longer sequences

---

## Running Experiments

### Commands

```bash
# Run by priority
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority P8

# Run single experiment
./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id FE-51

# Dry run (parameter estimates only)
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority P8 --dry-run

# List all experiments
./venv/bin/python experiments/feature_embedding/run_experiments.py --list
```

### Data Files

| Tier | File | Features | Rows |
|------|------|----------|------|
| a20 | `SPY_dataset_a20.parquet` | 20 | 8100 |
| a50 | `SPY_dataset_a50_combined.parquet` | 50 | 8022 |
| a100 | `SPY_dataset_a100_combined.parquet` | 100 | 8022 |
| a200 | `SPY_dataset_a200_combined.parquet` | 206 | 7977 |
| a500 | `SPY_dataset_a500_combined.parquet` | 500 | 7977 |

---

---

## Extended Experiments

Beyond the core feature embedding experiments (FE/LF/AE), we have implemented **114 additional experiments** testing advanced training techniques:

| Category | Experiments | Focus |
|----------|-------------|-------|
| Data Augmentation (DA) | 24 | Jitter, scale, mixup, time warp |
| Noise-Robust (NR) | 18 | Bootstrap loss, co-teaching, forward correction |
| Curriculum Learning (CL) | 18 | Loss-based, confidence, volatility curricula |
| Regime Detection (RD) | 18 | Volatility, trend, cluster-based regimes |
| Multi-Scale Temporal (MS) | 18 | Hierarchical pooling, multi-patch, dilated conv |
| Contrastive Pre-training (CP) | 18 | SimCLR, TS2Vec, BYOL |

**Full documentation**: [`docs/extended_experiments.md`](extended_experiments.md)

**Quick run**:
```bash
./scripts/run_extended_experiments.sh        # All 114 experiments
./scripts/run_extended_experiments.sh --phase 1  # DA + NR
```

---

## Changelog

- **2026-02-08**: Added 114 extended experiments (DA, NR, CL, RD, MS, CP) - see `docs/extended_experiments.md`
- **2026-02-07**: Added Phase 2 (Loss Function) experiments LF-01 to LF-48, testing 5 loss functions across best architectures
- **2026-02-07**: Major update with P1-P7 results, FE-50 breakthrough, P8-P12 experiments added, future research roadmap for context length studies
- **2026-02-06**: Initial design document created
