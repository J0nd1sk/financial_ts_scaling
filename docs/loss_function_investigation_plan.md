# Loss Function Investigation Plan

**Created:** 2026-01-20
**Phase:** 6A (Foundation Fixes Stage)
**Status:** In Progress

## Overview

This document captures research and plans for investigating loss functions to optimize both AUC and accuracy for financial time-series binary classification.

## Background

### The Problem

Our PatchTST model for financial prediction needs to optimize for:
1. **AUC-ROC** (≥0.7 target): Ranking ability - can model separate up/down days?
2. **Accuracy**: Threshold-based correctness at p=0.5

### Do AUC and Accuracy Conflict?

**Research Finding (Cortes & Mohri, NYU):**
> "On average, a classification algorithm minimizing error rate also optimizes AUC. However, different classifiers may have the same error rate but different AUC values."

| Data Balance | Conflict Likelihood |
|--------------|---------------------|
| Balanced (~50/50) | Usually aligned |
| Imbalanced (<30/70) | Often conflict |
| Our data (~55/45) | **Mostly aligned** |

**Conclusion:** For our ~55% positive class (threshold_1pct task), AUC and accuracy objectives should be mostly aligned, but we should still investigate.

---

## Implemented Loss Functions

### Single-Objective Losses

| Loss | File | Purpose | Key Parameters |
|------|------|---------|----------------|
| **BCE** | PyTorch built-in | Baseline, probability calibration | None |
| **SoftAUCLoss** | `src/training/losses.py` | Direct AUC optimization | `gamma=2.0` |
| **FocalLoss** | `src/training/losses.py` | Focus on hard examples | `gamma=2.0, alpha=0.25` |
| **LabelSmoothingBCELoss** | `src/training/losses.py` | Reduce overconfidence | `epsilon=0.1` |

### Loss Function Formulas

**BCE (Binary Cross-Entropy):**
```
L = -[y * log(p) + (1-y) * log(1-p)]
```

**SoftAUCLoss:**
```
L = mean(sigmoid(gamma * (neg_preds - pos_preds)))
```
- Pairwise loss over all (positive, negative) sample pairs
- Lower when positives ranked above negatives

**FocalLoss:**
```
L = -α_t * (1 - p_t)^γ * log(p_t)
```
- `p_t = p if y=1 else 1-p`
- `α_t = α if y=1 else 1-α`
- Down-weights easy examples via `(1-p_t)^γ`

**LabelSmoothingBCELoss:**
```
y_smooth = y * (1-ε) + (1-y) * ε
L = BCE(p, y_smooth)
```
- Targets become: 1→0.9, 0→0.1 (for ε=0.1)
- Reduces overconfidence penalty

---

## Multi-Objective Approaches (Research)

### 1. Weighted Sum (Simple)

```python
loss = α * BCE(pred, target) + (1 - α) * SoftAUC(pred, target)
```

| Aspect | Detail |
|--------|--------|
| Complexity | Low |
| Pros | Simple, intuitive |
| Cons | Requires manual α tuning, favors extreme solutions |
| Best Practice | Start with α=0.5, tune on validation |

### 2. Two-Phase Training (Recommended)

```python
# Phase 1: Pretrain with BCE (5 epochs, lr=1e-4)
criterion = nn.BCELoss()
# ... train ...

# Phase 2: Fine-tune with SoftAUC (5 epochs, lr=1e-5)
criterion = SoftAUCLoss()
# ... continue training with lower LR ...
```

| Aspect | Detail |
|--------|--------|
| Complexity | Low-Medium |
| Pros | Benefits of both, no weight tuning |
| Cons | Need to tune phase lengths and LRs |
| Source | LibAUC recommendation |

### 3. Homoscedastic Uncertainty Weighting (Advanced)

```python
log_var_bce = nn.Parameter(torch.zeros(1))
log_var_auc = nn.Parameter(torch.zeros(1))

loss = (1/torch.exp(log_var_bce)) * bce_loss + log_var_bce + \
       (1/torch.exp(log_var_auc)) * auc_loss + log_var_auc
```

| Aspect | Detail |
|--------|--------|
| Complexity | High |
| Pros | Automatic weight balancing |
| Cons | Adds learnable params, may be unstable |
| Source | Multi-task learning literature |

### 4. LibAUC Library (Specialized)

```python
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG

criterion = AUCMLoss()
optimizer = PESG(model.parameters(), loss_fn=criterion, lr=0.1)
```

| Aspect | Detail |
|--------|--------|
| Complexity | Medium |
| Pros | SOTA AUC optimization, proven results |
| Cons | External dependency, specialized optimizer |
| Source | https://libauc.org/ |

---

## Investigation Plan

### Phase 1: Single Loss Comparison (Current)

**Script:** `scripts/test_loss_comparison.py` (to be created)

**Configurations:**
1. BCE (baseline)
2. SoftAUCLoss (γ=2.0)
3. FocalLoss (γ=2.0, α=0.25)
4. LabelSmoothingBCELoss (ε=0.1)

**Settings:**
- SimpleSplitter (442 val samples)
- RevIN only (no Z-score)
- 2M model, 10 epochs
- Fixed seed for reproducibility

**Metrics to Collect:**
- val_loss
- AUC-ROC
- Accuracy (at p=0.5)
- Prediction spread (max - min)
- Training time

**Success Criteria:**
- Clear winner with AUC ≥0.65 and accuracy ≥0.55
- If no clear winner, proceed to Phase 2

### Phase 2: Multi-Objective (If Needed)

**Order of Investigation:**
1. Two-phase training (BCE → SoftAUC)
2. Weighted sum (α ∈ {0.3, 0.5, 0.7})
3. LibAUC integration (if above fail)

**Implementation Notes:**
- Two-phase requires Trainer modification for mid-training criterion swap
- Weighted sum requires new `CompositeLoss` class
- LibAUC requires new dependency and optimizer integration

### Phase 3: Integration

Once best approach identified:
1. Update HPO scripts/templates
2. Document decision in decision_log.md
3. Re-run HPO with optimal loss configuration

---

## References

### Academic
- [Cortes & Mohri - AUC Optimization vs Error Rate](https://cs.nyu.edu/~mohri/pub/auc.pdf)
- [Multi-Loss Weighting with CoV (WACV 2021)](https://openaccess.thecvf.com/content/WACV2021/papers/Groenendijk_Multi-Loss_Weighting_With_Coefficient_of_Variations_WACV_2021_paper.pdf)

### Industry
- [Google Research - Loss-Conditional Training](https://research.google/blog/optimizing-multiple-loss-functions-with-loss-conditional-training/)
- [Strategies for Balancing Multiple Losses (Medium)](https://medium.com/@baicenxiao/strategies-for-balancing-multiple-loss-functions-in-deep-learning-e1a641e0bcc0)
- [Neptune.ai - F1 vs AUC vs Accuracy](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)

### Tools
- [LibAUC Documentation](https://docs.libauc.org/)
- [LibAUC GitHub](https://github.com/Optimization-AI/LibAUC)

---

## Memory Entities

Related entities in Memory MCP:
- `Research_MultiObjectiveLoss_AUC_Accuracy` - Main research finding
- `Method_WeightedSumLoss` - Weighted sum pattern
- `Method_TwoPhaseTraining` - Two-phase training pattern
- `Method_UncertaintyWeighting` - Uncertainty weighting pattern
- `Tool_LibAUC` - LibAUC library reference
- `Plan_LossFunctions_Implementation` - Implementation plan for FocalLoss/LabelSmoothing

---

## Changelog

| Date | Change |
|------|--------|
| 2026-01-20 | Created document with research findings and plan |
| 2026-01-20 | Implemented FocalLoss and LabelSmoothingBCELoss |
