# Scaling Observations

## Phase 6A: Parameter Scaling (Limited Features)

### Setup
- Features: 20 technical indicators
- Samples: ~8,000 (SPY daily, through 2020 train)
- Parameter budgets: 2M, 20M, 200M, 2B
- Task: threshold_1pct (predict >1% returns)

### Results (200M Budget - Complete)

| Horizon | Best val_loss | Best Architecture | Params |
|---------|---------------|-------------------|--------|
| h1 (1-day) | 0.3633 | d=1024, L=12, h=16 | 151M |
| h3 (3-day) | **0.3081** | d=768, L=24, h=16 | 170M |
| h5 (5-day) | 0.3507 | d=256, L=256, h=16 | 202M |

### Results (2B Budget - In Progress)

| Trial | val_loss | Architecture | Observation |
|-------|----------|--------------|-------------|
| 0 | 0.3886 | d=768, L=256, 1.8B | Worse than 200M! |
| 1 | TBD | d=2048, L=32, 1.6B | In progress |

### Key Observation: No Scaling Benefit

**200M (170M actual) achieved val_loss=0.3081**
**2B (1.8B actual) achieved val_loss=0.3886**

Larger model performed **worse**, not better.

### Interpretation

With only 20 features and 8K samples:
- Parameter/sample ratio at 200M: ~25,000:1
- Parameter/sample ratio at 2B: ~250,000:1

The model has far more capacity than learnable signal. Additional parameters just mean:
1. Undertrained weights (insufficient gradient signal)
2. Overfitting (memorizing noise)
3. Harder optimization landscape

### Architecture Patterns

- **h=16 consistently optimal** (mean val_loss 0.3626)
- **h=8 underperforms** (mean val_loss 0.4157)
- **Wide-medium (d=768-1024, L=12-24)** best for h1/h3
- **Narrow-deep (d=256, L=256)** preferred by h5 (longer horizon)

## Hypothesis: Feature Richness Required

Scaling laws assume sufficient input complexity. With 20 features:
- Only ~190 pairwise relationships to learn
- Transformer attention has limited "vocabulary"
- Larger models can't extract more signal

With 2000 features:
- ~2,000,000 pairwise relationships
- Rich patterns across indicator families
- Larger models should find more complex interactions

**Prediction**: Phase 6C (feature scaling) will show clearer power-law behavior.

## Date

2025-12-22
