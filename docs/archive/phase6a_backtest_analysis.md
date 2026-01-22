# Phase 6A Backtest Analysis: 2025 Holdout Evaluation

> **SUPERSEDED (2026-01-21):** The issues documented here have been **RESOLVED**. SimpleSplitter + RevIN normalization eliminated probability collapse. For authoritative final results, see **`docs/phase6a_final_results.md`**.
>
> This document is preserved as a record of the debugging journey.

**Date**: 2026-01-19
**Status**: ~~ROOT CAUSE IDENTIFIED - Prior collapse due to BCE loss~~ **RESOLVED** - Infrastructure fixes applied

## Executive Summary

Evaluation of all 16 Phase 6A models on 2025 holdout data revealed:

1. **Double-sigmoid bug (FIXED)**: Evaluation script applied sigmoid twice, compressing outputs to 52-57%
2. **Prior collapse (ROOT CAUSE)**: After fixing the bug, models output 7-42% - but this is just the class prior for all samples

The models learn to predict the class prior (e.g., ~10% for h1) for ALL inputs, achieving low BCE loss without developing discriminative features. AUC-ROC (0.53-0.65) proves some ranking signal exists, but predictions have <3% spread.

**Key insight**: Random Forest achieves AUC 0.68-0.82 with 54-59% spread on the same data. The DATA has signal; the transformer training objective (BCE) allows degenerate solutions.

---

## Evaluation Methodology

### Data
- **Dataset**: SPY_dataset_a20.parquet (8,100 rows, 25 features)
- **Test Period**: 2025-01-01 to 2026-01-16
- **Test Samples**: 256-260 per model (depending on horizon)

### Models Evaluated
- **16 models**: 4 budgets (2M, 20M, 200M, 2B) × 4 horizons (h1, h2, h3, h5)
- **Task**: Binary classification (1% threshold exceeded within horizon)
- **Checkpoints**: `outputs/final_training/train_{budget}_h{horizon}/best_checkpoint.pt`

### Metrics Computed
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confidence analysis at various thresholds

---

## Results

### Accuracy by Budget × Horizon

| Budget | h1 | h2 | h3 | h5 |
|--------|-----|-----|-----|-----|
| 2M | 10.4% | 21.6% | 33.7% | 48.8% |
| 20M | 10.4% | 21.6% | 33.7% | 48.8% |
| 200M | 10.4% | 21.6% | 33.7% | 48.8% |
| 2B | 10.4% | 21.6% | 33.7% | 48.8% |

**Observation**: All budgets achieve identical accuracy, which equals the positive class base rate. This means models predict "positive" for ALL samples.

### AUC-ROC by Budget × Horizon (Better Metric)

| Budget | h1 | h2 | h3 | h5 |
|--------|-----|-----|-----|-----|
| 2M | 0.571 | 0.565 | 0.603 | 0.533 |
| 20M | 0.604 | 0.583 | 0.621 | 0.536 |
| 200M | 0.624 | 0.587 | 0.610 | 0.535 |
| **2B** | **0.645** | 0.583 | 0.591 | 0.583 |

**Observation**: AUC-ROC > 0.5 indicates models learned some ranking signal. Larger models show modest improvement at h1 (2B: 0.645 vs 2M: 0.571).

---

## Critical Finding: Probability Collapse

### Prediction Probability Ranges

| Model | Min Prob | Max Prob | Spread |
|-------|----------|----------|--------|
| 2M/h1 | 0.518 | 0.524 | **0.6%** |
| 2B/h1 | 0.519 | 0.520 | **0.1%** |
| 2M/h3 | 0.564 | 0.567 | **0.3%** |
| 2B/h3 | 0.572 | 0.575 | **0.3%** |

### Implications

1. **No confidence differentiation**: All predictions are essentially the same confidence level
2. **Cannot filter by threshold**: Zero predictions exceed 0.55 for h1 models
3. **Models collapsed to constant output**: Predicting ~52-57% regardless of input

### Confidence-Correctness Correlation

| Budget | h1 | h3 | h5 |
|--------|-----|-----|-----|
| 2M | +0.065 | +0.174 | +0.092 |
| 20M | +0.107 | +0.214 | +0.127 |
| 200M | +0.115 | +0.192 | +0.122 |
| 2B | +0.177 | +0.170 | +0.180 |

Small positive correlations exist but are unexploitable due to the narrow prediction range.

---

## Root Cause Hypotheses

### 1. Class Imbalance
- h1 positive class rate: ~10%
- h5 positive class rate: ~49%
- BCE loss may converge to predicting the prior probability

### 2. BCE Loss Convergence
- Binary cross-entropy can settle at outputting the class prior
- With 10% positive class, optimal constant prediction is ~0.1, but sigmoid(small_positive) ≈ 0.52

### 3. Sigmoid Saturation / Logit Compression
- Final layer may output near-zero logits
- Need to inspect raw logits (pre-sigmoid) to verify

### 4. Training Dynamics
- Early stopping may have stopped before proper calibration
- Validation loss (cross-entropy) doesn't penalize poor calibration

---

## Next Steps (Pending Investigation)

### Immediate: Diagnose
1. **Inspect raw logits** (pre-sigmoid) - is there more variation in logit space?
2. **Check training curves** - did loss converge too early?
3. **Analyze training set predictions** - same collapse pattern?

### Potential Fixes
1. **Temperature scaling** - post-hoc calibration method
2. **Focal loss** - better handles class imbalance
3. **Label smoothing** - prevents overconfident predictions
4. **Class weighting** - explicit handling of imbalance
5. **Platt scaling** - learns calibration mapping

---

## Files Created

| File | Description |
|------|-------------|
| `scripts/evaluate_final_models.py` | Evaluation script (315 lines) |
| `tests/test_evaluate_final_models.py` | 6 unit tests |
| `outputs/results/phase6a_backtest_2025.csv` | Full results (16 rows) |
| `docs/phase6a_backtest_analysis.md` | This analysis document |

---

## Conclusion

~~The Phase 6A models learned meaningful predictive signal (AUC-ROC > 0.5), but suffer from severe probability collapse that makes them practically unusable for confidence-filtered trading strategies.~~

**UPDATED (2026-01-19)**: Root cause identified as **prior collapse** - models learn to predict the class prior for all samples. This is a mathematically correct but practically useless solution that BCE loss allows.

Investigation ongoing into alternative loss functions:
- Margin-based loss (forces separation)
- AUC loss (directly optimizes ranking)
- Post-hoc calibration (temperature/Platt scaling)

See: `docs/research_paper/notes/prior_collapse_investigation.md`

## Memory Entities

- `Phase6A_DoubleSigmoidBug`
- `Phase6A_PriorCollapse_RootCause`
- `Phase6A_ArchitecturalIssues`
- `Phase6A_FixesAttempted`
- `Phase6A_ProposedFixes`
