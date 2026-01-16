# Appendix B.2: Architecture Analysis Results

## Overview

This appendix presents the architecture analysis from 600 HPO trials across 12 studies (4 parameter budgets × 3 prediction horizons). The analysis addresses the question: what architectural patterns optimize performance at each scale?

## Key Findings Summary

| Budget | Optimal d_model | Optimal n_layers | Style | Best val_loss |
|--------|-----------------|------------------|-------|---------------|
| 2M | 64 | 32-48 | Balanced | 0.2630 |
| 20M | 256 | 32 | Wide-shallow | 0.3191 |
| 200M | 384-512 | 48-96 | Wide-shallow | 0.3547 |
| 2B | 1024 | 180 | Balanced | 0.3592 |

**Critical finding**: Smaller models (2M) achieve the best absolute performance. Larger models show diminishing returns and eventual degradation, indicating a data-limited regime.

## Architecture Style Analysis

We categorize architectures by depth-to-width ratio:
- **Narrow-deep**: n_layers / d_model > 0.5
- **Balanced**: 0.15 ≤ n_layers / d_model ≤ 0.5
- **Wide-shallow**: n_layers / d_model < 0.15

### Performance by Architecture Style

**2M Budget:**
| Style | Best | Top-10 Mean ± Std | n |
|-------|------|-------------------|---|
| Balanced | 0.2630 | 0.2669 ± 0.003 | 24 |
| Narrow-deep | 0.2694 | 0.3117 ± 0.021 | 32 |
| Wide-shallow | 0.3396 | 0.3496 ± 0.004 | 94 |

At 2M parameters, **balanced architectures win decisively**. Wide-shallow architectures underperform by ~30%.

**20M Budget:**
| Style | Best | Top-10 Mean ± Std | n |
|-------|------|-------------------|---|
| Wide-shallow | 0.3191 | 0.3400 ± 0.008 | 98 |
| Narrow-deep | 0.3245 | 0.3477 ± 0.008 | 39 |
| Balanced | 0.3497 | 0.3587 ± 0.010 | 13 |

At 20M parameters, **wide-shallow becomes optimal**. The transition from balanced to wide-shallow suggests that at higher parameter counts, width matters more than depth.

**200M Budget:**
| Style | Best | Top-10 Mean ± Std | n |
|-------|------|-------------------|---|
| Wide-shallow | 0.3547 | 0.3603 ± 0.003 | 90 |
| Balanced | 0.3564 | 0.3575 ± 0.001 | 49 |
| Narrow-deep | 0.3603 | 0.3821 ± 0.018 | 11 |

At 200M, **wide-shallow and balanced perform similarly**. Narrow-deep architectures show high variance and worse mean performance.

**2B Budget:**
| Style | Best | Top-10 Mean ± Std | n |
|-------|------|-------------------|---|
| Balanced | 0.3592 | 0.3612 ± 0.001 | 62 |
| Wide-shallow | 0.3608 | 0.3628 ± 0.002 | 88 |

At 2B, **balanced returns as optimal** (no narrow-deep trials converged without diverging). The return to balanced suggests an upper limit on effective width.

## Model Dimension (d_model) Analysis

Strong consensus emerges for optimal d_model at each budget:

| Budget | Top-20 d_model Distribution |
|--------|----------------------------|
| 2M | 64 (100% - unanimous) |
| 20M | 256 (60%), 128 (25%) |
| 200M | 384 (65%), 512 (30%) |
| 2B | 1024 (80%), 1536 (15%) |

**Finding**: d_model scales approximately with the square root of parameter budget. This suggests a consistent relationship between model capacity and optimal embedding dimension.

## Layer Count (n_layers) Analysis

Less consensus than d_model, but clear patterns emerge:

| Budget | Optimal n_layers (Top-20) |
|--------|---------------------------|
| 2M | 32 (60%), 48 (40%) |
| 20M | 32 (55%), 128 (15%), varied |
| 200M | 96 (40%), 48 (25%), 160 (15%) |
| 2B | 180 (25%), 128 (25%), 256 (20%) |

**Finding**: Optimal depth increases with budget but plateaus. Beyond ~180 layers, training becomes unstable (14 diverged trials, all at L≥180 with 2B budget).

## Attention Heads (n_heads) Analysis

**Surprising finding**: n_heads has minimal impact on performance.

### Correlation with validation loss:
| Budget | Correlation (r) |
|--------|-----------------|
| 2M | 0.047 |
| 20M | 0.124 |
| 200M | 0.176 |
| 2B | 0.186 |

All correlations are weak positive (more heads slightly worse), but the effect is small.

### Controlled comparison (same d_model and n_layers):

**2M, d=64, L=32:**
- n_heads=2: val_loss=0.2630
- n_heads=32: val_loss=0.2631
- Difference: 0.0001 (negligible)

**200M, d=384, L=96:**
- n_heads=4: val_loss=0.3564
- n_heads=32: val_loss=0.3614
- Difference: 0.0050 (1.4%)

**Conclusion**: For this task, n_heads=2 is sufficient. Additional heads provide no benefit and may slightly harm performance. This suggests the attention mechanism's expressivity is not the bottleneck for financial time series prediction.

## Feed-Forward Dimension (d_ff) Analysis

The d_ff/d_model ratio in top-performing architectures:

| Budget | Mean Ratio | Range |
|--------|------------|-------|
| 2M | 3.7x | 2-4x |
| 20M | 2.5x | 2-4x |
| 200M | 3.3x | 2-4x |
| 2B | 2.7x | 2-4x |

**Finding**: The standard transformer ratio of 4x is near-optimal. No strong evidence for deviating from this convention.

## Horizon Effects

Optimal architectures vary modestly across prediction horizons:

| Horizon | Best Budget | Best Architecture | val_loss |
|---------|-------------|-------------------|----------|
| h1 (1-day) | 2M | d=64, L=48, h=2 | 0.3199 |
| h3 (3-day) | 2M | d=64, L=32, h=2 | 0.2630 |
| h5 (5-day) | 2M | d=64, L=64, h=16 | 0.3371 |

**Finding**: h3 (3-day prediction) is consistently easiest across all budgets. The 3-day horizon may capture weekly patterns more effectively than 1-day (too noisy) or 5-day (too much uncertainty).

## Divergence Analysis

14 trials diverged (val_loss=100.0), all at 2B scale:
- All had n_layers ≥ 180
- All had d_model ≥ 768
- Common pattern: very deep AND very wide simultaneously

This suggests a stability boundary where large architectures become untrainable with standard hyperparameters. Future work could investigate gradient clipping, learning rate warmup, or architectural modifications to stabilize training at extreme scales.

## Recommendations

Based on this analysis:

1. **For limited data**: Use 2M parameters with d=64, L=32-48, h=2
2. **For scaling experiments**: Expect diminishing returns beyond 20M parameters
3. **Ignore n_heads**: Default to 2 or 4; additional heads provide no benefit
4. **d_ff ratio**: Use standard 4x ratio
5. **Avoid extreme depth**: Beyond L=180, training stability degrades

## Implications for Scaling Laws

The finding that 2M outperforms larger models indicates we are firmly in a **data-limited regime**. Classical neural scaling laws predict monotonic improvement with parameters when data is sufficient. Our results suggest:

1. Financial time series data may have lower effective dimensionality than natural language or images
2. The information content in ~8000 daily price observations is insufficient to train models beyond ~2M parameters effectively
3. Scaling laws research in this domain requires either more data (longer history, more assets) or different experimental designs

This motivates Phase 6B (horizon scaling) and Phase 6C (feature scaling) to increase effective data complexity before revisiting parameter scaling.
