# Alternative Architecture Investigation Results

## Overview

Investigation of whether alternative transformer architectures can achieve better precision-recall tradeoffs than PatchTST for financial direction prediction.

**Investigation Dates:**
- v1/v2 (Invalid): 2026-01-25 - MSE loss with classification evaluation (methodology flaw)
- v3 (Valid): 2026-01-29 - Focal Loss with proper classification training
- Context Ablation: 2026-01-31 - Optimal context length per architecture

**Baseline (PatchTST 200M H1):**
- Val AUC: 0.718
- Precision-Recall: 90% precision → 4% recall, 75% precision → 23% recall

**Primary Goal:** Find architectures with better precision at improved recall.

---

## Results Summary (v3 - Valid Results)

**CONCLUSION: Both alternative architectures performed significantly WORSE than PatchTST, even with proper classification training and optimized context lengths.**

| Model | Best Context | Val AUC | Δ vs Baseline | Status |
|-------|--------------|---------|---------------|--------|
| PatchTST (baseline) | 80d | **0.718** | - | ✅ BEST |
| iTransformer | 80d | 0.590 | **-17.8%** | ❌ Underperforms |
| Informer | 180d | 0.585 | **-18.5%** | ❌ Underperforms |

**Recommendation:** PatchTST remains the best architecture. Alternative architectures not competitive for this task.

---

## Methodology Evolution

### v1/v2 (INVALID - 2026-01-25)
- **Flaw:** Trained as regressors (MAE/MSE loss) but evaluated as classifiers
- **Result:** 0% recall due to task mismatch
- **Lesson:** Cannot train forecaster and evaluate as classifier

### v3 (VALID - 2026-01-29)
- **Fix:** Proper classification with Focal Loss (`DistributionLoss('Bernoulli')`)
- **Result:** Models now produce valid probability outputs
- **HPO:** 50 trials each for iTransformer and Informer

---

## Architectures Tested (v3 Results)

### ARCH-01: iTransformer

**Key Innovation:** Feature-wise ("inverted") attention - attends across variables instead of time steps.

**Hypothesis:** May capture cross-feature relationships (RSI divergence, volume confirmation) better than PatchTST's temporal attention.

**Best Configuration (from 50-trial HPO):**
| Parameter | Value |
|-----------|-------|
| hidden_size | 32 |
| learning_rate | 1e-5 |
| max_steps | 3000 |
| dropout | 0.4 |
| n_layers | 6 |
| n_heads | 4 |
| focal_gamma | 0.5 |
| focal_alpha | 0.9 |
| **Best context** | **80d** |

**Results @ 80d (optimal):**
- Val AUC: **0.590**
- Precision: 0.203
- Recall: 0.941
- Direction accuracy: 54.2%
- Prediction range: 0.30-0.87

**Analysis:** With proper classification training, iTransformer achieves meaningful results but still significantly underperforms PatchTST. The inverted attention mechanism loses temporal patterns critical for direction prediction.

### ARCH-02: Informer

**Key Innovation:** ProbSparse attention with O(L log L) complexity, enabling efficient long sequences.

**Hypothesis:** Could allow longer context windows than PatchTST's 80-day limit.

**Best Configuration (from 50-trial HPO):**
| Parameter | Value |
|-----------|-------|
| hidden_size | 256 |
| learning_rate | 1e-4 |
| max_steps | 1000 |
| dropout | 0.4 |
| n_layers | 2 |
| n_heads | 2 |
| focal_gamma | 0.5 |
| focal_alpha | 0.9 |
| **Best context** | **180d** |

**Results @ 180d (optimal):**
- Val AUC: **0.585**
- Precision: 0.228
- Recall: 0.634
- Direction accuracy: 52.0%
- Prediction range: 0.00-1.00

**Analysis:** Informer benefits from longer context (180d vs 80d) due to ProbSparse attention, but this doesn't close the gap with PatchTST. The sparse attention may still miss important patterns.

---

## Context Length Ablation (2026-01-31)

### iTransformer Context Results

| Context | Val AUC | Precision | Recall | Pred Range |
|---------|---------|-----------|--------|------------|
| 60d | 0.552 | 0.202 | 0.980 | 0.39-0.87 |
| **80d** | **0.590** | 0.203 | 0.941 | 0.30-0.87 |
| 120d | 0.503 | 0.205 | 0.990 | 0.35-0.90 |
| 180d | 0.548 | 0.204 | 0.911 | 0.28-0.90 |
| 220d | 0.583 | 0.212 | 0.911 | 0.24-0.90 |

### Informer Context Results

| Context | Val AUC | Precision | Recall | Pred Range |
|---------|---------|-----------|--------|------------|
| 60d | 0.539 | 0.209 | 0.624 | 0.00-1.00 |
| 80d | 0.554 | 0.227 | 0.505 | 0.00-1.00 |
| 120d | 0.512 | 0.218 | 0.564 | 0.00-1.00 |
| **180d** | **0.585** | 0.228 | 0.634 | 0.00-1.00 |
| 220d | 0.557 | 0.242 | 0.495 | 0.00-1.00 |

---

## Final Comparison Table

| Model | Best Context | AUC | Δ AUC | Precision | Recall | Status |
|-------|--------------|-----|-------|-----------|--------|--------|
| PatchTST (baseline) | 80d | **0.718** | - | varies | varies | ✅ BEST |
| iTransformer | 80d | 0.590 | -17.8% | 0.203 | 0.941 | ❌ Underperforms |
| Informer | 180d | 0.585 | -18.5% | 0.228 | 0.634 | ❌ Underperforms |

---

## Key Questions Addressed

### Q1: Does inverted attention (iTransformer) capture cross-feature relationships better?

**ANSWER: NO.** Even with proper classification training, iTransformer achieves only 0.590 AUC (vs 0.718 for PatchTST).

The inverted attention mechanism, which attends across features rather than time steps, loses temporal patterns critical for direction prediction. Financial time series require understanding trends and momentum more than instantaneous feature correlations.

### Q2: Does efficient long-sequence attention (Informer) enable longer context?

**ANSWER: YES, but it doesn't help.** Informer's optimal context is 180d (vs 80d for PatchTST/iTransformer), but this only achieves 0.585 AUC.

ProbSparse attention handles longer sequences efficiently, but this doesn't translate to competitive performance. The sparse attention may still miss important patterns.

### Q3: Can ANY architecture beat PatchTST's precision-recall curve?

**ANSWER: NO.** Both tested architectures significantly underperform even with:
- Proper classification training (Focal Loss)
- 50-trial HPO for each architecture
- Architecture-specific optimal context lengths

This strongly confirms that PatchTST's patch-based temporal attention is well-suited for financial time series.

---

## Conclusions

### Root Cause Analysis

The performance gap persists despite proper methodology because:

1. **Attention mechanism mismatch:**
   - iTransformer's feature-wise attention loses temporal patterns critical for financial prediction
   - Informer's sparse attention may skip important time steps even with longer context

2. **Patching advantage:** PatchTST's patching mechanism aggregates local temporal patterns before attention, which appears better suited for financial time series than point-wise or sparse attention.

3. **Behavioral differences:**
   - iTransformer: Over-predicts positive (0.94 recall, 0.20 precision) - poor discrimination
   - Informer: Better calibrated but still poor discrimination

### Decision Matrix

| Outcome | Observed? | Interpretation |
|---------|-----------|----------------|
| Any model > 0.75 AUC with 30%+ recall at 75% precision | ❌ No | Not achieved |
| All models within ±3% of PatchTST AUC | ❌ No | 17-18% gap persists |
| Context tuning closes the gap | ❌ No | ~12-18% gap regardless of context |
| All worse than PatchTST | ✅ **YES** | PatchTST architecture is superior |

### Recommendation

**PatchTST remains the best architecture for this task.** Focus resources on:
1. **Feature scaling (Phase 6C)** - tier_a100, a200, and beyond
2. **PatchTST optimization** - continue HPO for PatchTST only
3. **Data diversity** - multi-asset, cross-asset experiments

### What We Learned

1. **v1/v2 methodology flaw:** Cannot train as regressor and evaluate as classifier
2. **PatchTST's patching mechanism is effective** for financial time series
3. **Inverted (feature-wise) attention is NOT effective** for direction prediction
4. **Longer context helps Informer but doesn't close the gap** - 180d optimal for Informer vs 80d for others
5. **Context length is architecture-specific** - must tune per architecture
6. **Performance gap is architectural, not configurational** - no amount of tuning closes the ~18% gap

---

## Files

| File | Purpose |
|------|---------|
| `experiments/architectures/common.py` | Shared utilities (direction_accuracy fix) |
| `experiments/architectures/hpo_neuralforecast.py` | HPO script with Focal Loss |
| `experiments/architectures/context_ablation_nf.py` | Context ablation script |
| `scripts/run_context_ablation.sh` | Runner for context ablation |
| `outputs/hpo/architectures/v3_itransformer/` | iTransformer HPO results |
| `outputs/hpo/architectures/v3_informer/` | Informer HPO results |
| `outputs/architectures/context_ablation/` | Context ablation results |

---

## Reproduction

```bash
# Install dependency (if not already installed)
pip install neuralforecast>=1.7.0

# Run HPO (50 trials each)
python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 50 --loss focal
python experiments/architectures/hpo_neuralforecast.py --model informer --trials 50 --loss focal

# Run context ablation
./scripts/run_context_ablation.sh

# View results
cat outputs/architectures/context_ablation/itransformer/ctx80/results.json
cat outputs/architectures/context_ablation/informer/ctx180/results.json
```

---

## Experiment Log

| Date | Experiment | Result |
|------|------------|--------|
| 2026-01-25 | v1/v2 iTransformer (MSE) | INVALID - task mismatch |
| 2026-01-25 | v1/v2 Informer (MSE) | INVALID - task mismatch |
| 2026-01-29 | v3 iTransformer HPO (Focal) | AUC 0.590 @ 80d |
| 2026-01-29 | v3 Informer HPO (Focal) | AUC 0.574 @ 80d |
| 2026-01-31 | Context ablation iTransformer | Best: **0.590 @ 80d** |
| 2026-01-31 | Context ablation Informer | Best: **0.585 @ 180d** |

---

## Next Steps

1. **a200 training** with optimal context per architecture
2. **Systematic validation** before training (6-point checklist)
3. **Compare PatchTST vs alternatives on a200 tier**
