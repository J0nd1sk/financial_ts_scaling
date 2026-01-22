# Head Dropout Ablation Results

**Date:** 2026-01-21
**Phase:** 6A (Parameter Scaling)
**Purpose:** Test whether adding dropout to the prediction head improves generalization

## Experiment Design

### Hypothesis
The prediction head (linear layer after transformer encoder) might benefit from its own dropout, separate from the encoder dropout (0.5).

### Variables
- **Independent:** head_dropout ∈ {0.05, 0.15, 0.30}
- **Control:** head_dropout = 0.0 (baseline)
- **Fixed:** encoder dropout = 0.5, lr = 1e-4, context = 80, RevIN, SimpleSplitter

### Models Tested
| Scale | Architecture | Baseline AUC |
|-------|--------------|--------------|
| 2M | d=64, L=32, h=8 | 0.713 |
| 20M | d=512, L=6, h=4 | 0.712 |

## Results

### 2M Scale (d=64, L=32, h=8)

| head_dropout | Test AUC | Δ vs Baseline | Accuracy | Precision | Recall | Pred Spread |
|--------------|----------|---------------|----------|-----------|--------|-------------|
| **0.00** (baseline) | **0.713** | — | 67.96% | 67.65% | 56.10% | — |
| 0.05 | 0.711 | -0.3% | 65.75% | 76.32% | 35.37% | 0.514 |
| 0.15 | 0.712 | -0.1% | 66.85% | 78.95% | 36.59% | 0.505 |
| 0.30 | 0.713 | 0.0% | 65.19% | 73.17% | 36.59% | 0.488 |

**Observation:** Head dropout has negligible effect on AUC at 2M scale. All values produce essentially identical discrimination ability.

### 20M Scale (d=512, L=6, h=4)

| head_dropout | Test AUC | Δ vs Baseline | Accuracy | Precision | Recall | Pred Spread |
|--------------|----------|---------------|----------|-----------|--------|-------------|
| **0.00** (baseline) | **0.712** | — | — | — | — | — |
| 0.05 | 0.614 | **-13.8%** | 56.91% | 55.00% | 26.83% | 0.763 |
| 0.15 | 0.612 | **-14.0%** | 58.01% | 54.29% | 46.34% | 0.791 |
| 0.30 | 0.708 | -0.6% | 62.98% | 70.27% | 31.71% | 0.607 |

**Observation:** Head dropout HURTS 20M performance at low-medium values. Only high dropout (0.30) recovers to near-baseline.

## Key Findings

### 1. Head Dropout Provides No Benefit at 2M Scale
- All three values (0.05, 0.15, 0.30) produce identical AUC to baseline (~0.71)
- Encoder dropout (0.5) already provides sufficient regularization
- The prediction head is too simple (single linear layer) to benefit from additional dropout

### 2. Head Dropout Hurts 20M Scale at Low-Medium Values
- 0.05 and 0.15 cause **14% AUC degradation** — catastrophic
- 0.30 recovers to near-baseline but still slightly worse
- The 20M model is more sensitive to head regularization, possibly due to:
  - Larger head (512 → 1 vs 64 → 1)
  - Different optimization dynamics with wider model

### 3. Recall vs Precision Trade-off
- Adding head dropout **increases precision** but **decreases recall**
- 2M baseline recall: 56% → with dropout: 35-37%
- This means more missed trading opportunities (false negatives)
- For trading applications, this is generally undesirable

### 4. Prediction Spread
- 2M: Spread decreases slightly with more dropout (0.51 → 0.49)
- 20M: Spread is much wider (0.6-0.8), indicating more variance
- Neither shows probability collapse (good)

## Conclusion

**Recommendation: Keep head_dropout = 0.0**

The prediction head is a simple linear layer that doesn't need its own dropout when:
1. The encoder is already heavily regularized (dropout = 0.5)
2. The head has minimal capacity to overfit independently
3. Adding head dropout only trades recall for precision without improving discrimination

This parameter should remain at 0.0 for all future experiments.

## Files

- Scripts: `experiments/head_dropout_ablation/train_*.py`
- Results: `outputs/head_dropout_ablation/*/results.json`
- Runner: `experiments/head_dropout_ablation/run_all.py`

## Related

- Context length ablation: `docs/context_length_ablation_results.md`
- Threshold experiments: `docs/threshold_05pct_high_experiments.md`
