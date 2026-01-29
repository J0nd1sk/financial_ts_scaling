# Alternative Architecture HPO v3 Design

**Created:** 2026-01-28
**Status:** Ready for implementation
**Purpose:** Correct the methodology flaw in v1/v2 experiments

---

## Problem Analysis

### What Went Wrong in v1/v2

The iTransformer and Informer experiments were trained as **regressors** but evaluated as **classifiers**.

| Version | Training Loss | Training Target | Evaluation | Result |
|---------|--------------|-----------------|------------|--------|
| v1 | MAE | Returns | Binary AUC | Invalid |
| v2 | MAE | Returns | Binary AUC | Invalid |
| **v3 (this)** | **Bernoulli** | **Binary** | **Binary AUC** | **Valid** |

### Evidence of Failure

**iTransformer v2:**
- AUC: 0.621
- Recall: **0%** (17/502 positive predictions)
- Prediction range: [0.004, 0.004]

**Informer v2:**
- AUC: 0.669
- Recall: **0%** (0/502 positive predictions)
- Prediction range: constant

### Root Cause

MSE/MAE loss trains the model to predict expected returns (small values ~0.005). When evaluated with threshold=0.5, no predictions are classified as positive.

---

## Correct Approach for v3

### Loss Function

```python
from neuralforecast.losses.pytorch import DistributionLoss

loss = DistributionLoss(distribution='Bernoulli')
```

This makes the model output the parameter of a Bernoulli distribution, i.e., P(positive) in [0, 1].

### Target Variable

Binary threshold target (0 or 1):

```python
# Already implemented in our data preparation
target = 1 if max(high[t+1:t+horizon]) >= close[t] * (1 + threshold) else 0
```

### Evaluation

Standard classification metrics with probability outputs:
- Threshold at 0.5 for binary predictions
- AUC-ROC from raw probabilities

---

## Technical Design

### Model Configuration

```python
from neuralforecast.models import iTransformer, Informer
from neuralforecast.losses.pytorch import DistributionLoss

# iTransformer with Bernoulli classification
model = iTransformer(
    h=1,
    input_size=context_length,
    hidden_size=hidden_size,
    n_heads=n_heads,
    e_layers=e_layers,
    dropout=dropout,
    loss=DistributionLoss(distribution='Bernoulli'),  # KEY CHANGE
    max_steps=max_steps,
    early_stop_patience_steps=patience,
    val_check_steps=val_check_steps,
    batch_size=batch_size,
    learning_rate=learning_rate,
)

# Informer with Bernoulli classification
model = Informer(
    h=1,
    input_size=context_length,
    hidden_size=hidden_size,
    n_heads=n_heads,
    e_layers=e_layers,
    d_layers=d_layers,
    dropout=dropout,
    loss=DistributionLoss(distribution='Bernoulli'),  # KEY CHANGE
    max_steps=max_steps,
    early_stop_patience_steps=patience,
    val_check_steps=val_check_steps,
    batch_size=batch_size,
    learning_rate=learning_rate,
)
```

### Data Preparation

Binary targets (already implemented in `common.py`):

```python
def prepare_hpo_data(horizon: int = 1, threshold: float = 0.01):
    """Prepare data with binary threshold target."""
    # ... existing code ...
    # target is already binary: 0 or 1
    return train_df, val_df
```

### Evaluation

```python
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Get probabilities from model
probs = model.predict(val_df)['iTransformer']  # or 'Informer'

# Evaluate
auc = roc_auc_score(y_true, probs)
preds = (probs > 0.5).astype(int)
precision = precision_score(y_true, preds, zero_division=0)
recall = recall_score(y_true, preds, zero_division=0)
```

---

## Search Space (Unchanged from v2)

The hyperparameter search space remains the same:

| Parameter | Values | Notes |
|-----------|--------|-------|
| dropout | [0.3, 0.4, 0.5] | High dropout critical for regularization |
| learning_rate | [5e-5, 1e-4, 2e-4] | Standard range |
| hidden_size | [64, 128, 256] | Model capacity |
| n_layers | [2, 3, 4] | Depth |
| n_heads | [2, 4, 8] | Attention heads |
| max_steps | [1000, 2000] | Training duration |
| batch_size | [16, 32, 64] | Batch size |

---

## Success Criteria

### Primary Criterion

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| AUC | >= 0.65 | Meaningfully above random (0.5) |
| Recall | > 0% | Must produce positive predictions |

### Comparison Baseline

| Model | H1 AUC | Target (5% improvement) |
|-------|--------|------------------------|
| PatchTST 200M | 0.718 | >= 0.754 |

### Minimum Viability

If best AUC < 0.65 or recall = 0%, the architecture is not viable for this task.

---

## Thermal Considerations

### Execution Order

1. **iTransformer first** - More efficient, runs cooler
2. **Informer second** - Has decoder, more compute-intensive

### Monitoring

- Watch for temperature spikes
- Use `caffeinate -i` to prevent sleep
- Log temperatures if available

### Estimated Runtime

- 50 trials at ~3-5 min/trial = 2.5-4 hours per model
- Total: ~5-8 hours for both models

---

## Comparison Protocol

### Fair Comparison with PatchTST

| Factor | PatchTST | iTransformer/Informer |
|--------|----------|----------------------|
| Training samples | ~7,255 | ~7,255 |
| Validation samples | ~420 | ~420 |
| Feature count | 25 | 25 |
| Context length | 80 | 80 |
| Horizon | H1 | H1 |
| Loss | BCE | Bernoulli (equivalent) |
| HPO trials | 50 | 50 |

### What We're Testing

1. Does the attention architecture matter? (iTransformer inverts axes)
2. Does the encoder-decoder design help? (Informer)
3. Can these architectures match PatchTST with fair HPO?

---

## Implementation Checklist

- [ ] Modify `hpo_neuralforecast.py` to use `DistributionLoss('Bernoulli')`
- [ ] Verify binary targets are passed correctly
- [ ] Run smoke test (3 trials) to verify:
  - [ ] Predictions in [0, 1] range
  - [ ] Prediction spread > 0.1
  - [ ] Recall > 0%
- [ ] Run full iTransformer HPO (50 trials)
- [ ] Run full Informer HPO (50 trials)
- [ ] Analyze and document results

---

## Expected Outcomes

### If v3 Succeeds (AUC >= 0.65, recall > 0%)

- Confirms methodology was the issue, not the architectures
- May reveal competitive performance with PatchTST
- Warrants further investigation (H3, H5 horizons)

### If v3 Fails (AUC < 0.65 or recall = 0%)

- Confirms these architectures are not suitable for this task
- PatchTST remains the recommended architecture
- Close the alternative architecture investigation

---

## Related Documents

- `docs/methodology_lessons_v1_v2.md` - What went wrong in v1/v2
- `docs/architectural_hpo_design.md` - PatchTST HPO design (correct approach)
- `.claude/context/workstreams/ws2_context.md` - Workstream context

---

*Document Version: 1.0*
