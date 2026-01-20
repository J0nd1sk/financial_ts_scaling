# Phase 6A Implementation Audit

**Created:** 2026-01-20
**Status:** IN PROGRESS
**Goal:** Find the bug causing prior collapse in our transformer implementation

---

## Problem Statement

Our PatchTST models collapse to predicting the class prior (~0.52 for h1 where true rate is ~10%) from Epoch 0. This is NOT a transformer problem - it's OUR implementation problem.

**Evidence:**
- Random Forest achieves AUC 0.68-0.82 on same data → signal exists
- Transformers work for time series (proven in literature)
- Therefore: something in our code is broken

---

## Audit Scope

### 1. PatchTST Implementation (`src/models/patchtst.py`)
- [ ] Forward pass correctness
- [ ] Prediction head appropriate for classification?
- [ ] Attention mechanism working?
- [ ] Positional encoding correct?
- [ ] Output shape and activation

### 2. Training Loop (`src/training/trainer.py`)
- [ ] Loss computation correct?
- [ ] Gradient updates happening?
- [ ] Optimizer configured correctly?
- [ ] Learning rate schedule?

### 3. Data Pipeline
- [ ] Feature normalization
- [ ] Target construction (threshold logic)
- [ ] Batching preserving temporal structure?
- [ ] Data loader shuffling appropriate?

### 4. Config/Hyperparameters
- [ ] Learning rate appropriate for model size?
- [ ] Initialization reasonable?
- [ ] Dropout values?

### 5. Loss Function Setup
- [ ] BCE applied correctly (logits vs probabilities)?
- [ ] Class imbalance handling?
- [ ] Numerical stability?

---

## Audit Log

### Audit 1: PatchTST Implementation
**Date:** 2026-01-20
**Files:** `src/models/patchtst.py`, `src/training/trainer.py`

#### Code Review Findings

**1. Sigmoid in forward pass (patchtst.py:218)**
```python
x = torch.sigmoid(x)  # Binary classification
```
- Model outputs probabilities (0-1), not logits
- Compatible with BCELoss (trainer.py:161)
- ✅ Not a bug, but limits loss function choices

**2. Positional encoding initialization (patchtst.py:98)**
```python
self.position_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))
```
- Uses randn with std=1.0
- ⚠️ SUSPICIOUS: Most implementations use smaller scale (e.g., std=0.02)
- Could add significant noise to patch embeddings

**3. Prediction head is very simple (patchtst.py:202-204)**
```python
self.flatten = nn.Flatten(start_dim=1)
self.dropout = nn.Dropout(dropout)
self.linear = nn.Linear(d_model * num_patches, num_classes)
```
- Just flatten + linear, no nonlinearity
- For 2M config (d_model=64, num_patches=6): 384 → 1
- ⚠️ May not have enough capacity for complex patterns

**4. No class imbalance handling (trainer.py:161)**
```python
self.criterion = criterion if criterion is not None else nn.BCELoss()
```
- BCELoss with no pos_weight
- h1 has 10% positive rate → 9x class imbalance
- BCE allows degenerate solution: predict class prior for all samples

**5. Default weight initialization**
- Linear layers use Kaiming uniform (PyTorch default)
- Prediction head linear layer starts with small random weights
- Initial output before sigmoid ≈ 0 → sigmoid(0) = 0.5
- ✅ Expected, should learn away from this

#### Key Insight: The Model DOES Learn

From Memory entity `Phase6A_PriorCollapse_RootCause`:
- After double-sigmoid fix, outputs are 7-42% (not 52%)
- h1 predicts ~8% (true rate 10.4%)
- h3 predicts ~27% (true rate 33.7%)
- h5 predicts ~37% (true rate 48.8%)

**The model IS learning - it learns to predict the class prior!**

This is a degenerate solution where:
1. Model outputs (nearly) same prediction for ALL samples
2. That prediction = class prior (optimal for BCE when uncertain)
3. No discrimination between samples

#### Why Does This Happen?

BCE loss is minimized when predictions equal true probabilities. If the model cannot distinguish samples, predicting the prior minimizes expected loss.

RF achieves AUC 0.68-0.82 on same features → features ARE informative.

**Conclusion: The model architecture or training isn't extracting/using features properly.**

#### Suspicious Areas Requiring Diagnostics

| Area | Suspicion Level | Why |
|------|----------------|-----|
| Positional encoding init | ⚠️ HIGH | randn(std=1) is unusually large |
| Prediction head simplicity | ⚠️ MEDIUM | May lack capacity |
| No pos_weight in BCE | ⚠️ MEDIUM | Class imbalance |
| Patch embedding | ❓ UNKNOWN | Need to check output scale |
| Attention weights | ❓ UNKNOWN | May be uniform/meaningless |

---

## Key Questions to Answer

1. What does an untrained model predict? (initialization bias?)
2. Do weights actually change during training?
3. Are gradients flowing through all layers?
4. Is the prediction head appropriate for binary classification?
5. Is there a shape mismatch or broadcasting issue?

---

## Resolution

Once bug is found, document:
- Root cause
- Fix applied
- Verification method
