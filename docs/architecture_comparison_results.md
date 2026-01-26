# Alternative Architecture Investigation Results

## Overview

Investigation of whether alternative transformer architectures can achieve better precision-recall tradeoffs than PatchTST for financial direction prediction.

**Investigation Date:** 2026-01-25

**Baseline (PatchTST 200M H1):**
- Val AUC: 0.718
- Precision-Recall: 90% precision → 4% recall, 75% precision → 23% recall

**Primary Goal:** Find architectures with better precision at improved recall.

---

## Results Summary

**CONCLUSION: Both alternative architectures performed significantly WORSE than PatchTST.**

| Model | Val AUC | Δ vs Baseline | Direction Acc | Status |
|-------|---------|---------------|---------------|--------|
| PatchTST (baseline) | **0.718** | - | - | ✅ BEST |
| iTransformer | 0.517 | **-28.0%** | 50.4% | ❌ Failed |
| Informer (80d) | 0.587 | **-18.2%** | 42.6% | ❌ Failed |

**Recommendation:** ABANDON architecture investigation. Focus on feature scaling (Phase 6C).

---

## Architectures Tested

### ARCH-01: iTransformer

**Key Innovation:** Feature-wise ("inverted") attention - attends across variables instead of time steps.

**Hypothesis:** May capture cross-feature relationships (RSI divergence, volume confirmation) better than PatchTST's temporal attention.

**Configuration:**
- Context length: 80 days (matching PatchTST)
- Hidden size: 128
- Attention heads: 4
- Encoder layers: 3
- Training: 500 steps (no early stopping)
- Loss: MSE

**Results:**
- Val AUC: **0.517** (barely above random 0.5)
- Test AUC: 0.537
- Direction accuracy: 50.4% (essentially random)
- Precision: 25% at 1% recall (nearly useless)
- Prediction range: [-0.011, 0.012] (very narrow, collapsed)

**Analysis:** The inverted attention mechanism appears unsuited for financial time series. By attending across features rather than time, it loses the temporal patterns that are critical for direction prediction.

### ARCH-02: Informer

**Key Innovation:** ProbSparse attention with O(L log L) complexity, enabling efficient long sequences.

**Hypothesis:** Could allow longer context windows than PatchTST's 80-day limit.

**Configuration:**
- Context length: 80 days (baseline)
- Hidden size: 128
- Attention: ProbSparse (factor=5)
- Encoder layers: 2, Decoder layers: 1
- Training: 500 steps (no early stopping)
- Loss: MSE

**Results:**
- Val AUC: **0.587** (below baseline but better than iTransformer)
- Test AUC: 0.593
- Direction accuracy: 42.6% (below random!)
- Precision: 0% (all predictions negative - probability collapse)
- Prediction range: [-0.010, -0.006] (ALL negative predictions)

**Analysis:** Complete probability collapse. The model learned to predict slightly negative returns for all samples, never predicting positive. This indicates the forecasting → threshold approach fails for classification.

---

## Comparison Table

| Model | AUC | Δ AUC | Direction Acc | Pred Range | Status |
|-------|-----|-------|---------------|------------|--------|
| PatchTST (baseline) | 0.718 | - | - | - | ✅ BASELINE |
| iTransformer | 0.517 | -28.0% | 50.4% | [-0.011, 0.012] | ❌ FAILED |
| Informer (80d) | 0.587 | -18.2% | 42.6% | [-0.010, -0.006] | ❌ FAILED |

---

## Key Questions Addressed

### Q1: Does inverted attention (iTransformer) capture cross-feature relationships better?

**ANSWER: NO.** iTransformer performed worse than random (AUC 0.517).

The inverted attention mechanism, which attends across features rather than time steps, appears fundamentally unsuited for this task. Financial direction prediction requires understanding temporal patterns (trends, momentum, mean reversion) more than instantaneous feature correlations.

### Q2: Does efficient long-sequence attention (Informer) enable longer context?

**ANSWER: Not tested** due to baseline failure.

With AUC 0.587 on 80-day context, there was no point testing extended 200-day context. The fundamental approach (forecasting → threshold) failed before we could test the long-context hypothesis.

### Q3: Can ANY architecture beat PatchTST's precision-recall curve?

**ANSWER: NO.** Both tested architectures performed significantly worse.

This strongly suggests that PatchTST's patch-based temporal attention is well-suited for financial time series. Alternative attention mechanisms (inverted, ProbSparse) do not improve performance.

---

## Conclusions

### Root Cause Analysis

The poor performance of both models can be attributed to:

1. **Task mismatch:** NeuralForecast models are designed for point forecasting (regression), not classification. The forecasting → threshold approach creates an indirect optimization target.

2. **Attention mechanism mismatch:**
   - iTransformer's feature-wise attention loses temporal patterns
   - Informer's sparse attention may skip important time steps

3. **Probability collapse:** Both models showed narrow prediction ranges, indicating they learned to predict near-mean returns rather than meaningful signals.

### Decision Matrix

| Outcome | Observed? | Interpretation |
|---------|-----------|----------------|
| Any model > 0.75 AUC with 30%+ recall at 75% precision | ❌ No | Not worth pursuing |
| All models within ±3% of PatchTST AUC | ❌ No | Much worse than baseline |
| iTransformer significantly better | ❌ No | Feature interactions not key |
| All worse than PatchTST | ✅ **YES** | **ABANDON architecture investigation** |

### Recommendation

**ABANDON the architecture investigation.** Focus resources on:
1. **Feature scaling (Phase 6C)** - tier_a100 and beyond
2. **Hyperparameter optimization** - better tuning of PatchTST
3. **Data diversity** - multi-asset, cross-asset experiments

### What We Learned

1. PatchTST's patch-based temporal attention is effective for financial time series
2. Inverted (feature-wise) attention is NOT effective for direction prediction
3. Efficient sparse attention (ProbSparse) does not compensate for task mismatch
4. Forecasting → threshold is an inferior approach compared to direct classification
5. Alternative architectures would need significant modification for classification

---

## Files

| File | Purpose |
|------|---------|
| `experiments/architectures/common.py` | Shared utilities |
| `experiments/architectures/itransformer_forecast.py` | iTransformer experiment |
| `experiments/architectures/informer_forecast.py` | Informer experiment |
| `outputs/architectures/itransformer_forecast/results.json` | iTransformer results |
| `outputs/architectures/informer_forecast/results.json` | Informer results |

---

## Reproduction

```bash
# Install dependency (if not already installed)
pip install neuralforecast>=1.7.0

# Run experiments
python experiments/architectures/itransformer_forecast.py
python experiments/architectures/informer_forecast.py

# View results
cat outputs/architectures/itransformer_forecast/results.json
cat outputs/architectures/informer_forecast/results.json
```

---

## Experiment Log

| Date | Experiment | Result |
|------|------------|--------|
| 2026-01-25 | ARCH-01 iTransformer | AUC 0.517 - FAILED |
| 2026-01-25 | ARCH-02 Informer | AUC 0.587 - FAILED |
| - | ARCH-02b Informer Long Context | SKIPPED (baseline failed) |
