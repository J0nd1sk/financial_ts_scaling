# Methodology Lessons: Alternative Architecture HPO v1 & v2

**Created:** 2026-01-28
**Purpose:** Document the fundamental methodology flaw in alternative architecture experiments and lessons learned.

---

## Executive Summary

The iTransformer and Informer experiments (v1 and v2) were trained as **regressors** (MSE loss predicting returns) but evaluated as **classifiers** (binary AUC/precision/recall). This task mismatch caused 0% recall and scientifically invalid results.

---

## What We Did (v1 & v2)

### Training Configuration

| Aspect | What We Did | What We Should Have Done |
|--------|-------------|--------------------------|
| **Loss Function** | `MAE()` / MSE on returns | `DistributionLoss('Bernoulli')` / BCE |
| **Target Variable** | Float returns (`y` column) | Binary threshold target (0/1) |
| **Output Type** | Continuous predictions | Probabilities in [0, 1] |
| **Task Type** | Regression | Classification |

### Evaluation Configuration

| Metric | How We Evaluated |
|--------|------------------|
| AUC-ROC | Treating predictions as probabilities |
| Precision | Thresholding at 0.5 |
| Recall | Counting positive predictions |
| Accuracy | Binary classification metrics |

---

## Why This Failed

### The Core Problem

When you train a model to minimize MSE on return values, the model learns to predict **expected returns** (typically small values near 0.005 or 0.5%). When you then evaluate by thresholding at 0.5 to create binary predictions, almost NONE of the predictions exceed 0.5.

```
Training: minimize |predicted_return - actual_return|^2
          → Model outputs: 0.004, 0.005, 0.003, ...

Evaluation: if prediction > 0.5: predict positive
           → Almost no predictions > 0.5
           → 0% recall
```

### Evidence of Failure

**iTransformer v2 Results:**
- AUC: 0.621 (appears reasonable, but misleading)
- Precision: 35.3%
- Recall: **0%** (17 positive predictions out of 502 samples)
- Prediction range: [0.004, 0.004] - all below 0.5 threshold

**Informer v2 Results:**
- AUC: 0.669 (appears reasonable, but misleading)
- Precision: undefined (0/0)
- Recall: **0%** (0 positive predictions out of 502 samples)
- Prediction range: essentially constant

### Why AUC Was Misleading

AUC measures ranking ability, not calibration. A model can achieve decent AUC if its small return predictions correlate with actual outcomes, even if the predictions are all in the range [0.003, 0.006]. But this doesn't mean the model is useful for classification.

---

## The Correct Approach

### For Classification Tasks

If you want to predict "will price exceed threshold?", you must:

1. **Create binary targets**: `target = 1 if max_high[t+1:t+horizon] >= close[t] * (1 + threshold) else 0`
2. **Use classification loss**: BCE, Focal Loss, or `DistributionLoss('Bernoulli')`
3. **Output probabilities**: Model output should be P(positive) in [0, 1]
4. **Evaluate consistently**: Threshold at 0.5 (or tune threshold) for binary predictions

### NeuralForecast Specific

For NeuralForecast models (iTransformer, Informer), use:

```python
from neuralforecast.losses.pytorch import DistributionLoss

model = iTransformer(
    h=1,
    input_size=context_length,
    loss=DistributionLoss(distribution='Bernoulli'),  # <-- Classification!
    ...
)
```

This makes the model output the parameter of a Bernoulli distribution (i.e., a probability).

---

## Why PatchTST Worked

PatchTST was trained correctly from the start:

| Aspect | PatchTST Implementation |
|--------|------------------------|
| Loss | `BCEWithLogitsLoss` |
| Target | Binary (0/1) |
| Output | Sigmoid probability |
| Task | Classification |

The training and evaluation objectives were aligned.

---

## What About Foundation Models?

**Lag-Llama, TimesFM, Chronos**: These are pre-trained forecasting models designed for predicting continuous values. They cannot easily be retrained with BCE loss.

**Finding stands**: For foundation models, the domain mismatch (pre-trained on non-financial data) is the primary limitation. The regression approach was reasonable for zero-shot evaluation.

**No rerun needed** for foundation models. Focus v3 efforts on iTransformer/Informer only.

---

## Lessons for Future Experiments

### Rule 1: Match Training to Evaluation

**Always verify that the loss function matches the evaluation metrics.**

| If Evaluating With | Train With |
|-------------------|-----------|
| AUC, Precision, Recall, F1 | BCE, Focal Loss, Bernoulli |
| MSE, MAE, RMSE | MSE, MAE, Huber |
| MAPE, SMAPE | MAPE loss variants |

### Rule 2: Check Output Ranges

Before running full experiments, verify that model outputs are in the expected range:

- **Classification**: Outputs should span [0, 1] with meaningful variance
- **Regression**: Outputs should have similar scale to targets

### Rule 3: Recall = 0% is a Red Flag

If a model achieves 0% recall, investigate immediately:
- Are predictions all below threshold?
- Is the model outputting constants?
- Is there a task mismatch?

### Rule 4: Smoke Test the Full Pipeline

Run 2-3 trials and inspect:
- Raw prediction values
- Prediction range and distribution
- Sanity check: do predictions make sense for the task?

---

## Summary

| Experiment Set | Training | Evaluation | Result | Fix |
|----------------|----------|------------|--------|-----|
| PatchTST | BCE (classification) | Classification | 0.718 AUC | None needed |
| iTransformer v1/v2 | MAE (regression) | Classification | 0% recall | Rerun with Bernoulli |
| Informer v1/v2 | MAE (regression) | Classification | 0% recall | Rerun with Bernoulli |
| Lag-Llama | NLL (regression) | Classification | Domain mismatch | Not applicable |
| TimesFM | MSE (regression) | Classification | Domain mismatch | Not applicable |

---

## Related Documents

- `docs/architecture_hpo_v3_design.md` - Design for corrected experiments
- `.claude/context/decision_log.md` - Entry 2026-01-28 (methodology correction)
- `docs/foundation_model_results.md` - Foundation model investigation results

---

*Document Version: 1.0*
*This document captures a critical methodology lesson that applies to all future classification experiments.*
