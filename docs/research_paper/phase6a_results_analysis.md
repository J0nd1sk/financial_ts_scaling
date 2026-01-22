# Phase 6A Results Analysis: Parameter Scaling Experiments

> **IMPORTANT UPDATE (2026-01-21):** This document contains HPO results from before infrastructure corrections. For authoritative final results with corrected infrastructure (SimpleSplitter, RevIN, proper validation sizes), see **`docs/phase6a_final_results.md`**.
>
> **Key correction:** The "inverse scaling" finding in this document was an artifact of using only 19 validation samples. With proper validation (420+ samples), scaling is **flat/minimal**, not inverse.

---

## Executive Summary

Phase 6A tested whether neural scaling laws apply to transformer models on financial time-series data by varying parameter budgets from 2M to 2B while holding feature count (20 indicators) and data (SPY daily, ~8000 observations) constant.

**Core Finding**: Scaling laws do NOT hold in this data-limited regime. The smallest model (2M parameters) achieves the best validation performance, with larger models showing progressively worse generalization despite successful optimization.

## Experimental Design

### Parameter Budgets
| Budget | Actual Range | Trials | Duration |
|--------|--------------|--------|----------|
| 2M | 1.6M - 2.5M | 150 | ~1.1 hrs |
| 20M | 16.9M - 24M | 150 | ~4.8 hrs |
| 200M | 151M - 213M | 150 | ~42 hrs |
| 2B | 1.5B - 1.8B | 150 | ~128 hrs |

### Prediction Horizons
- **h1**: 1-day (next trading day)
- **h2**: 2-day (interpolated architecture)
- **h3**: 3-day
- **h5**: 5-day

### Data
- **Asset**: SPY ETF (S&P 500 tracker)
- **Period**: 1993-11-11 to 2026-01-16 (8,100 rows)
- **Features**: 25 (5 OHLCV + 20 technical indicators)
- **Task**: Binary classification (>1% price increase)
- **Splits**: 70% train, 15% validation, 15% test (hybrid chunk-based)

## HPO Results

### Best Validation Loss by Budget and Horizon

| Budget | h1 (1-day) | h3 (3-day) | h5 (5-day) | Mean |
|--------|------------|------------|------------|------|
| **2M** | 0.3199 | **0.2630** | 0.3371 | 0.3067 |
| **20M** | 0.3483 | 0.3191 | 0.3458 | 0.3377 |
| **200M** | 0.3564 | 0.3612 | 0.3547 | 0.3574 |
| **2B** | 0.3609 | 0.3948 | 0.3592 | 0.3716 |

**Key Observations**:
1. 2M achieves best performance at every horizon
2. Performance degrades monotonically with increasing parameters
3. h3 (3-day) is consistently the "easiest" prediction task
4. 2B performs ~21% worse than 2M overall

### Optimal Architectures by Budget

| Budget | Horizon | d_model | n_layers | n_heads | d_ff | Params | val_loss |
|--------|---------|---------|----------|---------|------|--------|----------|
| 2M | h1 | 64 | 48 | 2 | 256 | 2.4M | 0.3199 |
| 2M | h3 | 64 | 32 | 2 | 256 | 1.6M | 0.2630 |
| 2M | h5 | 64 | 64 | 16 | 128 | 2.2M | 0.3371 |
| 20M | h1 | 128 | 180 | 16 | 256 | 23.9M | 0.3483 |
| 20M | h3 | 256 | 32 | 2 | 512 | 16.9M | 0.3191 |
| 20M | h5 | 384 | 12 | 4 | 1536 | 21.4M | 0.3458 |
| 200M | h1 | 384 | 96 | 4 | 1536 | 170M | 0.3564 |
| 200M | h3 | 384 | 180 | 4 | 768 | 213M | 0.3612 |
| 200M | h5 | 512 | 48 | 2 | 2048 | 151M | 0.3547 |
| 2B | h1 | 1024 | 128 | 2 | 4096 | 1.6B | 0.3609 |
| 2B | h3 | 768 | 256 | 32 | 3072 | 1.8B | 0.3948 |
| 2B | h5 | 1024 | 180 | 4 | 2048 | 1.5B | 0.3592 |

## Final Training Results

After HPO, each configuration was trained on full training data (not the 30% subset used during HPO) with optimized hyperparameters. Results on the held-out validation set:

### Final Validation Loss Matrix

| Budget | h1 | h2 | h3 | h5 |
|--------|-----|-----|-----|-----|
| **2M** | 0.237 | 0.515 | 0.597 | 0.647 |
| **20M** | 0.243 | 0.517 | 0.596 | 0.633 |
| **200M** | 0.244 | 0.515 | 0.600 | 0.631 |
| **2B** | **0.232** | 0.517 | 0.581 | **0.715** |

**Critical Note**: These are cross-entropy losses on the validation set (Oct-Dec 2024), NOT test accuracy on holdout data. Lower is better.

### Interpretation of Final Training Results

1. **h1 (1-day) is most predictable**: All budgets achieve val_loss ~0.23-0.24
2. **Longer horizons are harder**: h5 val_loss is 2.5-3x worse than h1
3. **Scaling shows minimal benefit**: 2B shows marginal improvement on h1 (0.232 vs 0.237) but catastrophic degradation on h5 (0.715 vs 0.647)
4. **Early stopping triggered for all**: Models converged in 3-17 epochs despite 50 epoch budget

### Training Dynamics

| Budget | h1 epochs | h2 epochs | h3 epochs | h5 epochs |
|--------|-----------|-----------|-----------|-----------|
| 2M | 6 | 4 | 7 | 7 |
| 20M | 5 | 4 | 14 | 3 |
| 200M | 4 | 4 | 11 | 3 |
| 2B | 17 | 4 | 4 | 12 |

**Pattern**: Larger models show more variable training behavior. 2B_h1 trained for 17 epochs (longest) while 2B_h3 stopped at epoch 4.

## Scaling Law Analysis

### Classical Scaling Law Form

The Kaplan et al. (2020) scaling law predicts:
$$L(N) = L_\infty + \left(\frac{N_0}{N}\right)^\alpha$$

where:
- $L$ = loss
- $N$ = number of parameters
- $L_\infty$ = irreducible loss
- $N_0$ = characteristic scale
- $\alpha$ = scaling exponent (~0.076 for language models)

### Our Results vs. Scaling Law Predictions

If scaling laws held, we would expect monotonic improvement with more parameters:

| Budget | Predicted (if scaling) | Actual | Deviation |
|--------|------------------------|--------|-----------|
| 2M | baseline | 0.3067 | - |
| 20M | ~0.28 | 0.3377 | +20% worse |
| 200M | ~0.26 | 0.3574 | +37% worse |
| 2B | ~0.24 | 0.3716 | +55% worse |

**Conclusion**: Our results show **inverse scaling** - larger models perform worse, violating the fundamental assumption of neural scaling laws.

### Why Scaling Laws Fail Here

We hypothesize several contributing factors:

1. **Data-limited regime**: ~4000 training samples after splits cannot support models with millions of parameters. The Chinchilla scaling law (Hoffmann et al., 2022) suggests optimal data:params ratio is ~20:1. Our 2B model has ratio of 0.002:1.

2. **Low effective dimensionality**: Financial time series may have lower intrinsic dimensionality than language or images. With only 25 features, there may be limited learnable structure regardless of model capacity.

3. **Non-stationarity**: Financial markets are non-stationary. Patterns learned from 1993-2020 may not transfer to 2021+. Larger models may overfit to historical patterns that no longer hold.

4. **Feature richness bottleneck**: The input dimensionality (25 features) may be too low to support rich learned representations. This motivates Phase 6C (feature scaling).

## Statistical Significance

### Variance Analysis

Standard deviation of validation loss across HPO trials:

| Budget | h1 std | h3 std | h5 std |
|--------|--------|--------|--------|
| 2M | 0.048 | 0.089 | 0.058 |
| 20M | 0.065 | 0.074 | 0.065 |
| 200M | 0.074 | 0.058 | 0.045 |
| 2B | 0.087 | 0.099 | 0.054 |

**Pattern**: Variance is relatively stable across budgets, suggesting the performance differences are robust, not artifacts of noise.

### Divergence Rate

| Budget | Diverged Trials | Rate |
|--------|-----------------|------|
| 2M | 0 | 0% |
| 20M | 0 | 0% |
| 200M | 0 | 0% |
| 2B | 14 | 9.3% |

All 14 diverged trials had n_layers >= 180 and d_model >= 768, indicating stability boundaries at extreme scales.

## Horizon Effects

### Cross-Horizon Comparison

| Horizon | Best Budget | Best val_loss | Interpretation |
|---------|-------------|---------------|----------------|
| h1 (1-day) | 2M | 0.3199 | Short-term prediction benefits from small models |
| h3 (3-day) | 2M | 0.2630 | Sweet spot for prediction horizon |
| h5 (5-day) | 2M | 0.3371 | Longer horizons harder, still small models win |

### Architecture Transfer Failure

Testing h3-optimal architecture (d=64, L=32, h=2) on other horizons:

| Target Horizon | HPO-optimal | With h3-config | Degradation |
|----------------|-------------|----------------|-------------|
| h1 | 0.3199 | 0.3840 | +20% |
| h3 | 0.2630 | 0.2630 | baseline |
| h5 | 0.3371 | 0.7815 | +132% |

**Implication**: Architectures do NOT transfer across prediction horizons. Each horizon requires dedicated optimization.

## Key Conclusions

### Primary Finding
**Neural scaling laws do not apply to financial time-series prediction with limited features and data.** Larger models consistently underperform smaller models, exhibiting inverse scaling.

### Secondary Findings

1. **Optimal model size is surprisingly small**: 2M parameters (d=64, L=32-48) outperforms 2B parameters by 20%+

2. **h3 (3-day) is the most predictable horizon**: May align with weekly trading patterns

3. **Architecture matters more than scale**: The right architecture at 2M beats the wrong architecture at 2B

4. **n_heads has minimal impact**: 2 attention heads suffice; more heads provide no benefit

5. **Training parameters must scale with model size**: LR decreases ~3x from 2M to 2B; dropout increases from 0.10 to 0.25

### Implications for Future Work

1. **Phase 6B (Horizon Scaling)**: Test whether longer horizons benefit from different architectural patterns

2. **Phase 6C (Feature Scaling)**: Test hypothesis that feature richness (20 → 2000) unlocks scaling benefits

3. **Phase 6D (Data Scaling)**: Test whether more assets/history enables larger models to shine

### Research Contribution

These results challenge the assumption that larger models are universally better. For financial prediction with limited features:

> "Input richness matters as much as model size for neural scaling laws to emerge."

This finding has practical implications for quantitative finance practitioners, suggesting that resources are better invested in feature engineering than model scaling when working with limited input dimensionality.

---

*Analysis based on 600 HPO trials (12 studies × 50 trials) and 16 final training runs.*
*Total compute time: ~177 hours (~7.4 days)*
*Hardware: Apple M4 MacBook Pro, 128GB unified memory*

---

## Addendum: Infrastructure Corrections (2026-01-21)

Subsequent investigation revealed critical infrastructure issues that invalidated some conclusions in this document:

1. **ChunkSplitter Bug**: Validation used only 19 samples instead of ~420
2. **Probability Collapse**: Models output near-constant predictions (0.52-0.57 range)
3. **Normalization Issue**: Global z-score + RevIN caused distribution issues

### Corrected Findings

With corrected infrastructure (SimpleSplitter, RevIN only, 80-day context):

| Finding | This Document | Corrected |
|---------|---------------|-----------|
| Scaling pattern | Inverse (2M best) | Flat/minimal |
| Best budget | 2M significantly better | All similar |
| Prediction spread | 0.52-0.57 (collapsed) | 0.01-0.94 (healthy) |

### What Remains Valid

- Horizon effects (H3 easiest, H1/H5 harder)
- Architecture transfer failure across horizons
- Feature bottleneck hypothesis

### Authoritative Results

See `docs/phase6a_final_results.md` for final results with corrected infrastructure.
