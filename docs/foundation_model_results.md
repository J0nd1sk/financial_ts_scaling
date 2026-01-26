# Foundation Model Investigation Results

**Status:** Complete (Phase 1: Foundation Models)
**Branch:** `experiment/foundation-decoder-investigation`
**Last Updated:** 2026-01-26

---

## Executive Summary

Foundation models **significantly underperform** task-specific PatchTST on SPY direction prediction:

| Model | Best Val AUC | vs PatchTST 200M |
|-------|--------------|------------------|
| **PatchTST 200M** | **0.718** | Baseline |
| TimesFM (inverted) | 0.636 | -11.4% |
| Lag-Llama (best) | 0.576 | -19.8% |
| TimesFM (raw) | 0.364 | -49.3% |

**Key Findings:**
1. **Lag-Llama FAILED** - All modes (fine-tune, projection, head-only) ≤ 0.576 AUC
2. **TimesFM is ANTI-CORRELATED** - Predictions inversely relate to target (AUC 0.364)
3. **Covariates are IGNORED** - Adding 50 features had ZERO effect on TimesFM predictions
4. **Feature engineering is essential** - Foundation models cannot substitute for good features

**Recommendation:** Abandon foundation model path; proceed with feature scaling (Phase 6C).

---

## Lag-Llama Experiments

### Results Summary

| Experiment | Mode | Features | Val AUC | Test AUC | Status |
|------------|------|----------|---------|----------|--------|
| H1 Close | Fine-tune | 1 (close) | 0.576 | - | Below PatchTST |
| H1 Proj | Projection | 1 | ~0.50 | - | Random |
| H1 Headonly | Head-only | 1 | ~0.50 | - | Random |
| H3 variants | Various | 1 | 0.499-0.536 | - | Failed |

### Analysis

- Fine-tuning mode achieved best results but still 19.8% below PatchTST
- Projection and head-only modes produced near-random predictions
- H3 (3-day horizon) variants performed worse than H1
- No mode approached PatchTST baseline

### Conclusion

Lag-Llama's probabilistic forecasting approach does not transfer well to financial direction prediction. The model's pre-training on general time series data provides no benefit over task-specific training.

---

## TimesFM Experiments

### Experiment Matrix

| ID | Features | Mode | Val AUC | Test AUC | Inverted AUC | Status |
|----|----------|------|---------|----------|--------------|--------|
| TFM-01 | 1 (close) | Zero-shot | 0.3640 | 0.276 | **0.636** | Anti-correlated |
| TFM-07 | 50 (a50) | Zero-shot | 0.3640 | 0.275 | **0.636** | **IDENTICAL to TFM-01** |

### Detailed Metrics Comparison

| Metric | TFM-01 (1 feature) | TFM-07 (50 features) | Difference |
|--------|-------------------|---------------------|------------|
| Val AUC | 0.3639663218 | 0.3639663218 | **0.0000** |
| Test AUC | 0.2759 | 0.2746 | -0.0013 |
| Inverted AUC | 0.6360 | 0.6360 | **0.0000** |
| Best F1 | 0.391 | 0.391 | **0.0000** |
| Pred Min | -0.01096426 | -0.01096426 | 0.0000 |
| Pred Max | 0.00604074 | 0.00604074 | 0.0000 |
| Pred Mean | 0.00096157 | 0.00096157 | 0.0000 |
| Pred Std | 0.00139825 | 0.00139825 | 0.0000 |

---

## Critical Finding: Covariates Completely Ignored

### Raw Prediction Correlation Analysis

```
=== Validation Predictions (502 samples) ===
Correlation:        1.0000000000
Max difference:     8.15e-09
Mean difference:    1.35e-09
Identical (<1e-6):  502/502 (100.0%)

=== Difference Distribution ===
Differ by > 1e-09:  246/502 (49.0%)  # Floating point noise
Differ by > 1e-08:  0/502 (0.0%)
Differ by > 1e-07:  0/502 (0.0%)
Differ by > 1e-06:  0/502 (0.0%)
Differ by > 0.001:  0/502 (0.0%)
```

### Interpretation

The predictions are **mathematically identical**. The correlation of exactly 1.0000000000 and maximum difference of 8.15e-09 (floating point rounding error) proves that:

1. **TimesFM completely ignores covariate inputs**
2. Predictions are computed from the target series alone
3. All 50 engineered features (RSI, MACD, ATR, returns, etc.) had **ZERO effect**
4. Feature engineering effort for TimesFM is wasted

This is not "similar performance" - it is **numerically identical output**. TimesFM's covariate pathway appears to be non-functional or the implementation does not integrate covariates into the forecasting pipeline.

### Features That Were Ignored

TFM-07 included 50 carefully engineered features:
- Trend: DEMA (9,10,20,25,90), SMA (12,50,100,200)
- Momentum: RSI (daily/weekly), StochRSI, MACD line/histogram, QQE (fast/slow)
- Volatility: ATR, BB %, BB width, VRP (10d/21d)
- Volume: OBV, ADOSC, Volume ratio, KVO signal
- Returns: 1d, 5d, 21d, 63d, 252d
- Derived: RSI slope, ATR %, SMA proximity, overnight gap

**None of these affected the model's predictions in any measurable way.**

---

## Anti-Correlation Analysis

### What Anti-Correlation Means

TimesFM's raw predictions are **inversely correlated** with actual outcomes:
- When model predicts UP, market more likely goes DOWN
- When model predicts DOWN, market more likely goes UP
- This produces AUC < 0.5 (worse than random)

### Inverted Predictions

By inverting predictions (multiply by -1), AUC improves from 0.364 to 0.636. However:
- Still 11.4% below PatchTST (0.718)
- This is a symptom, not a solution
- Indicates fundamental mismatch between TimesFM's learned patterns and financial data

### Threshold Sweep (TFM-07, Inverted)

| Threshold | Precision | Recall | F1 | Accuracy |
|-----------|-----------|--------|-------|----------|
| -0.0060 (all positive) | 0.201 | 1.000 | 0.335 | 0.201 |
| -0.0009 (optimal F1) | 0.368 | 0.416 | **0.391** | 0.739 |
| +0.0017 | 0.444 | 0.079 | 0.134 | 0.795 |

Best F1 of 0.391 achieved with inverted predictions at threshold -0.00009.

---

## Comparison to PatchTST Baseline

### Performance Gap

| Model | Val AUC | Gap vs PatchTST | Notes |
|-------|---------|-----------------|-------|
| PatchTST 200M | **0.718** | - | Task-specific, 20 features |
| PatchTST 2M | 0.706 | -1.7% | Minimal parameter scaling benefit |
| TimesFM (inv) | 0.636 | -11.4% | 200M params, anti-correlated |
| Lag-Llama | 0.576 | -19.8% | Fine-tuned on our data |

### Why Foundation Models Underperform

1. **Domain Mismatch**: Pre-trained on diverse time series (energy, retail, weather), not financial data
2. **Task Mismatch**: Designed for forecasting, not binary classification
3. **Anti-Correlation**: TimesFM learned patterns that are inverted for financial data
4. **Covariate Blindness**: TimesFM ignores additional features entirely
5. **No Fine-Tuning Benefit**: Even fine-tuned Lag-Llama underperforms

---

## Hypothesis Outcomes

From `foundation_decoder_investigation_plan.md`:

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Foundation models outperform from-scratch | **REJECTED** | 11-20% below PatchTST |
| H2: Decoder ≈ Encoder for fixed-horizon classification | Unclear | Would need decoder trained from scratch |
| H3: Probabilistic outputs improve threshold classification | **REJECTED** | Lag-Llama probabilistic ≤ 0.576 AUC |

---

## Conclusions

### Foundation Model Path: NOT VIABLE

1. **Transfer learning does not help** - Pre-trained models underperform task-specific training
2. **Covariates are wasted** - TimesFM ignores feature engineering entirely
3. **Anti-correlation persists** - 50 features did not fix TimesFM's inversion problem
4. **Fine-tuning is insufficient** - Even fine-tuned Lag-Llama is 19.8% below baseline

### Feature Scaling Path: RECOMMENDED

The evidence strongly supports focusing on **Phase 6C (Feature Scaling)** instead:
- PatchTST can actually use engineered features
- 20 → 100 → 200 features may provide the signal needed
- Task-specific architecture with proper feature integration

---

## Files and Artifacts

### Results Files
- `outputs/foundation/timesfm_tfm-01_results.json` - TFM-01 metrics
- `outputs/foundation/timesfm_tfm-01_predictions.npz` - TFM-01 raw predictions
- `outputs/foundation/timesfm_tfm-07_a50_results.json` - TFM-07 metrics
- `outputs/foundation/timesfm_tfm-07_a50_predictions.npz` - TFM-07 raw predictions

### Notebooks
- `experiments/foundation/TimesFM_SPY_Experiments.ipynb` - TFM-01 notebook
- `experiments/foundation/TimesFM_a50_Experiments.ipynb` - TFM-07/08 notebook
- `experiments/foundation/TimesFM_a100_Experiments.ipynb` - TFM-09/10 notebook

---

## Next Steps

### Immediate
- [x] Document TFM-07 covariate findings
- [ ] Decision: Run TFM-08 (fine-tuned) or abandon TimesFM entirely

### If Continuing TimesFM Investigation
- [ ] TFM-08: Fine-tune with 50 features (may enable covariate usage)
- [ ] TFM-09/10: Test with 100 features

### Recommended Path
- [ ] Return focus to Phase 6C (feature scaling with PatchTST)
- [ ] Use a50 and a100 feature tiers with PatchTST
- [ ] Feature scaling more likely to improve AUC than foundation models

---

## References

- [TimesFM Paper](https://arxiv.org/abs/2402.03885)
- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [PatchTST Paper](https://arxiv.org/abs/2211.14730)

---

*Document Version: 1.0*
*Created: 2026-01-26*
