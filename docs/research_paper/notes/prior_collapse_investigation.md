# Prior Collapse Investigation Notes

**Date**: 2026-01-19
**Status**: Investigation ongoing, loss function experiments running

---

## Executive Summary

Investigation into poor backtest performance revealed two issues:
1. **Double-sigmoid bug** (FIXED): Evaluation script applied sigmoid twice
2. **Prior collapse** (ROOT CAUSE): Models learn to predict class prior for all samples

The data DOES contain predictive signal (Random Forest AUC 0.68-0.82), but transformers are not learning to use it effectively due to BCE loss allowing degenerate solutions.

---

## Bug #1: Double Sigmoid (FIXED)

**Location**: `scripts/evaluate_final_models.py` lines 256-258

**Symptom**: All predictions compressed to 52-57% range

**Cause**:
- Model's `PredictionHead` applies sigmoid in `forward()`
- Evaluation script applied `torch.sigmoid()` again
- `sigmoid(sigmoid(x))` compresses to ~0.5-0.73 range

**Fix**: Remove redundant sigmoid call in evaluation script

**After fix**: Real model outputs are 7-42% depending on horizon

---

## Root Cause: Prior Collapse

### What Happens

Models learn to predict the class prior for ALL samples:

| Horizon | Predicted | True Rate | Gap |
|---------|-----------|-----------|-----|
| h1 | ~8% | 10.4% | -2.4% |
| h3 | ~27% | 33.7% | -6.7% |
| h5 | ~37% | 48.8% | -11.8% |

### When It Happens

**From Epoch 0**. The model immediately learns to predict the prior and never develops discriminative features.

### Why It Happens

1. **BCE measures calibration, not discrimination**: A model that outputs 0.15 for ALL samples achieves good BCE loss when 15% of samples are positive

2. **No penalty for uniformity**: BCE doesn't require the model to output higher probabilities for positive samples

3. **Class imbalance exacerbates**: h1 has 10% positive rate, so model sees 9x more negative examples

### Evidence of Signal

| Method | AUC-ROC | Prediction Spread |
|--------|---------|-------------------|
| Random Forest | 0.68-0.82 | 54-59% |
| Transformer (current) | 0.53-0.66 | <3% |

The data HAS predictive signal. The transformer is not learning to use it.

---

## Architectural Issues Identified

### Issue 1: Sigmoid in Model Forward Pass

**Location**: `src/models/patchtst.py:218` (PredictionHead)

```python
x = torch.sigmoid(x)  # Applied in forward()
```

**Problem**: Should output raw logits for numerical stability with BCEWithLogitsLoss

### Issue 2: No Class Weighting

**Location**: `src/training/trainer.py:146`

```python
self.criterion = nn.BCELoss()  # No pos_weight
```

**Problem**: With 10% positive rate, model is incentivized to predict low values

---

## Fixes Attempted

| Fix | Result |
|-----|--------|
| pos_weight only | Shifted predictions to ~50%, still collapsed |
| Focal loss only | Helped slightly, still collapsed |
| Combined (pos_weight + focal + logits) | AUC improved to 0.62-0.66, still collapsed |

**Key insight**: BCE-family losses (even with modifications) don't REQUIRE discrimination. They allow the "easy path" of predicting uniform values.

---

## Proposed Solutions

### User Preferences (from discussion)

1. **Temperature scaling / Platt scaling** (post-hoc calibration) - preferred
2. **Margin-based loss** to force separation between classes - preferred
3. Open to outputting logits (but may not be necessary with correct loss)
4. Willing to retrain with new loss AND re-run HPO if needed

### Recommended Comprehensive Fix

1. **Model change**: Output logits, not probabilities
2. **Loss change**: BCEWithLogitsLoss with pos_weight
3. **Add discrimination objective**: Margin-based, contrastive, or AUC loss
4. **Post-hoc calibration**: Temperature scaling or Platt scaling

### Loss Functions Tested

**RESULT (2026-01-20): Soft AUC loss WINS**

| Loss Function | Result |
|---------------|--------|
| **Soft AUC** | **WINNER** - directly optimizes ranking |
| Margin-based | Tested |
| Contrastive | Tested |

Soft AUC directly optimizes what we care about (ranking/discrimination) rather than calibration. This addresses the core problem: BCE-family losses don't require discrimination.

---

## Implications for Paper Narrative

### Original Finding (now questionable)
> "Inverse scaling - smaller models achieve better validation loss"

### Revised Understanding
The "inverse scaling" measured speed of convergence to a degenerate solution (prior prediction), not predictive ability. All models collapsed to predicting the prior.

### What We Can Claim

1. **Val loss is misleading**: Lower BCE loss can be achieved by predicting uniformly
2. **AUC-ROC tells the real story**: Models DO learn ranking signal (0.53-0.66)
3. **Data has signal**: Random Forest proves this (AUC 0.68-0.82)
4. **Transformers need correct objectives**: BCE allows degenerate solutions

### What We Cannot Claim (yet)

- Whether scaling laws hold with a proper loss function
- Whether larger models would outperform with discrimination-focused training

---

## Next Steps

1. ~~Complete loss function experiments~~ **DONE - Soft AUC wins**
2. ~~Select best loss function~~ **DONE - Soft AUC**
3. Decide: Post-hoc calibration vs. full retraining with Soft AUC
4. If retraining: Implement Soft AUC loss in trainer, re-run HPO
5. Consider: Should paper include the BCE failure + Soft AUC fix narrative?

---

## Memory Entities

- `Phase6A_DoubleSigmoidBug`
- `Phase6A_PriorCollapse_RootCause`
- `Phase6A_ArchitecturalIssues`
- `Phase6A_FixesAttempted`
- `Phase6A_ProposedFixes`
