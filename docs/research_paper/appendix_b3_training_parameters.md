# Appendix B.3: Training Parameter Analysis

## Overview

This appendix analyzes training hyperparameters (learning rate, dropout, weight decay, warmup) across 586 non-diverged trials from 12 HPO studies. While Appendix B.2 examined architecture (d_model, n_layers, n_heads), this analysis focuses on optimization parameters.

## Key Findings Summary

| Parameter | Correlation with val_loss | Importance | Pattern |
|-----------|---------------------------|------------|---------|
| Learning Rate | r = -0.233 | **High** | Inversely scales with model size |
| Dropout | r = +0.073 | Moderate | Scales with model size |
| Weight Decay | r = -0.023 | Low | No clear pattern |
| Warmup Steps | r = +0.036 | Low | Minor effect |

**Critical finding**: Learning rate is the most actionable training parameter, and it must be scaled with model size. Larger models require smaller learning rates.

## Learning Rate Analysis

### Overall Pattern

Higher learning rates correlate with better performance (r = -0.233), but this relationship is mediated by model size:

| LR Range | Mean val_loss | n trials |
|----------|---------------|----------|
| Low (<3e-4) | 0.3587 | 175 |
| Mid (3-6e-4) | 0.3569 | 184 |
| High (6e-4-1e-3) | 0.3461 | 227 |

### Learning Rate by Model Size

The optimal LR clearly scales inversely with parameter count:

| Param Quartile | Optimal LR | Model Size Range |
|----------------|------------|------------------|
| Q1 (smallest) | ~0.9e-3 | 2M-scale |
| Q2 | ~0.8e-3 | 20M-scale |
| Q3 | ~0.6e-3 | 200M-scale |
| Q4 (largest) | ~0.3e-3 | 2B-scale |

### Learning Rate by Budget

| Budget | Optimal LR Range | Best Trial LR |
|--------|------------------|---------------|
| 2M | 0.7-1.0e-3 | 0.000683 (mean) |
| 20M | 0.5-0.6e-3 | 0.000526 (mean) |
| 200M | 0.5-0.8e-3 | 0.000580 (mean) |
| 2B | 0.1-0.4e-3 | 0.000404 (mean) |

**Interpretation**: 2B models initially struggled because they used learning rates appropriate for smaller models. The ~3x reduction in LR from 2M to 2B is necessary for stable training.

## Dropout Analysis

### Overall Pattern

Dropout shows weak correlation with performance (r = +0.073), but the effect is budget-dependent:

| Dropout Range | Mean val_loss | n trials |
|---------------|---------------|----------|
| Low (<0.15) | 0.3463 | 122 |
| Mid (0.15-0.25) | 0.3552 | 325 |
| High (>0.25) | 0.3547 | 139 |

### Dropout by Budget

| Budget | Low Dropout | Mid Dropout | High Dropout | Optimal |
|--------|-------------|-------------|--------------|---------|
| 2M | **0.3166** | 0.3378 | 0.3428 | Low (0.10-0.15) |
| 20M | 0.3567 | 0.3492 | **0.3455** | Mid-High |
| 200M | 0.3588 | 0.3610 | 0.3628 | Low-Mid |
| 2B | 0.3752 | 0.3710 | **0.3615** | High (>0.25) |

**Pattern**:
- **2M models**: Low dropout (0.10-0.15) — don't over-regularize small models
- **20M-200M models**: Mid dropout (0.15-0.27) — moderate regularization
- **2B models**: High dropout (0.20-0.30) — larger models benefit from stronger regularization

### Recommended Dropout by Budget

| Budget | Recommended Dropout |
|--------|---------------------|
| 2M | 0.10-0.15 |
| 20M | 0.13-0.27 |
| 200M | 0.21-0.30 |
| 2B | 0.20-0.25 |

## Weight Decay Analysis

Weight decay shows no clear pattern (r = -0.023):

- Range tested: 0.0001 - 0.005
- No significant difference across values
- Best trials mostly used 0.0003 - 0.0015

**Recommendation**: Use 0.5-1.0e-3, not sensitive to tuning.

## Warmup Steps Analysis

Warmup shows minor effect (r = +0.036):

| Warmup Range | Mean val_loss |
|--------------|---------------|
| 100 | 0.3512 |
| 200 | 0.3538 |
| 300 | 0.3551 |
| 500 | 0.3572 |

### Warmup by Horizon

| Horizon | Optimal Warmup | Interpretation |
|---------|----------------|----------------|
| h1 (1-day) | 275 (higher) | Shorter horizons need gradual warmup |
| h3 (3-day) | 125 (lower) | "Easiest" horizon, trains quickly |
| h5 (5-day) | 200 (moderate) | Intermediate |

**Recommendation**: 100-200 warmup steps; slightly higher for short horizons.

## Early Stopping Analysis

Early stopping with patience=10 proved highly effective:

| Metric | Value |
|--------|-------|
| Trials that early-stopped | 570/586 (97.3%) |
| Mean epochs trained | 28 |
| Median epochs trained | 29 |
| Best trials mean epochs | 40 |
| Config epochs | 50, 75, or 100 |

**Key insight**: Despite configuring 50-100 epochs, most trials converged in ~28 epochs. Best trials trained longer (~40 epochs), suggesting they found better optima worth pursuing.

**Recommendation**: Configure 50 epochs with patience=10. This provides sufficient runway for good trials while allowing early termination for poor configurations.

## Recommended Training Parameters

### By Budget

| Budget | LR | Dropout | Weight Decay | Warmup | Epochs |
|--------|-----|---------|--------------|--------|--------|
| 2M | 0.7-1.0e-3 | 0.10-0.15 | 0.5-2.0e-3 | 100 | 50 |
| 20M | 0.5-0.6e-3 | 0.13-0.27 | 0.5-1.2e-3 | 100 | 50 |
| 200M | 0.5-0.8e-3 | 0.21-0.30 | 0.2-0.4e-3 | 200 | 50 |
| 2B | 0.1-0.4e-3 | 0.20-0.25 | 0.2-1.7e-3 | 200 | 50 |

### By Horizon (2M Budget)

| Horizon | LR | Dropout | Warmup |
|---------|-----|---------|--------|
| h1 (1-day) | 0.23e-3 | 0.30 | 500 |
| h3 (3-day) | 1.0e-3 | 0.10 | 100 |
| h5 (5-day) | 0.82e-3 | 0.24 | 200 |

## Conclusions

1. **Learning rate scales inversely with model size**: This is the most important training parameter finding. Use ~1e-3 for 2M models, ~3e-4 for 2B models.

2. **Dropout scales with model size**: Small models need minimal regularization (0.10), large models need more (0.25).

3. **Weight decay and warmup are not sensitive**: Default values (0.5-1.0e-3 weight decay, 100-200 warmup) work across configurations.

4. **Early stopping is essential**: 97% of trials benefited from early stopping. Configure 50 epochs with patience=10.

5. **Best trials train longer**: Top performers averaged 40 epochs vs 28 for all trials, suggesting good configurations warrant more training time.

---

*Analysis based on 586 non-diverged trials from 12 HPO studies (4 budgets × 3 horizons, 50 trials each).*
