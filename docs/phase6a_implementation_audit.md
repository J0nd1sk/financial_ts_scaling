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

### Audit 2: Diagnostics Run
**Date:** 2026-01-20

#### Diagnostic Results (Before Root Cause Found)

**1. Signal-to-Noise Ratio (Untrained Model)**
| Component | Std | L2 Norm |
|-----------|-----|---------|
| Patch embeddings | 0.58 | 4.63 |
| Positional encoding | 1.04 | 8.26 |
| **Ratio** | - | **0.56** |

Initial concern: Positional encoding ~2x larger than content signal.

**2. Trained Model Attention (Test Data)**
| Layer | Entropy Ratio | Interpretation |
|-------|---------------|----------------|
| 0 | 0.9994 | ~Uniform |
| 24 | 1.0000 | Exactly uniform |
| 47 | 0.9999 | ~Uniform |

All patches produce identical representations (variance = 0.000000).

**3. Output Distribution**
- Val data (2016-2021): mean=0.09, spread reasonable
- Test data (2024-2026): mean=0.52, spread=0.03

#### CRITICAL: Root Cause Found by Parallel Investigation

**See: `docs/phase6a_feature_normalization_bug.md`**

The uniform attention and collapsed outputs I observed on TEST data are **symptoms, not causes**. The actual root cause is:

**FEATURES ARE NOT NORMALIZED**

| Feature | Train (1994-2016) | Test (2024-2026) | Shift |
|---------|-------------------|-------------------|-------|
| Close | 88.57 | 575.89 | **6.5x** |
| MACD | 0.20 | 3.23 | **16x** |
| ATR | 1.23 | 6.64 | **5x** |
| RSI | 54.41 | 58.32 | ~1x (bounded) |

**What Actually Happened:**
1. Model trained on Close ≈ 88, learned valid patterns
2. Model sees Close ≈ 576 at test time - completely OOD
3. OOD inputs cause model to default to "safe" uniform attention
4. Model outputs sigmoid midpoint (~0.5) = "I don't know"

**Evidence Model DID Learn:**
- Val predictions: ~0.09 (correct for class prior ~0.10)
- Val loss: 0.203 (much better than random 0.409)
- Model works on training distribution!

#### Audit Conclusions

| Component | Status | Finding |
|-----------|--------|---------|
| PatchTST architecture | ✅ OK | Works correctly on in-distribution data |
| Training loop | ✅ OK | Model learns, val_loss improves |
| Loss function | ✅ OK | BCE works for this task |
| **Data pipeline** | ❌ **BUG** | Features not normalized |
| Positional encoding | ⚠️ Minor | Large but not root cause |

---

## Key Questions to Answer

1. What does an untrained model predict? (initialization bias?)
2. Do weights actually change during training?
3. Are gradients flowing through all layers?
4. Is the prediction head appropriate for binary classification?
5. Is there a shape mismatch or broadcasting issue?

---

## Resolution

### Root Cause
**Feature normalization missing** - raw features (Close, MACD, ATR, etc.) have massive distribution shift between training (1994-2016) and test (2024-2026) periods. Model trained on Close≈88 cannot generalize to Close≈576.

### Fix Required
See `docs/phase6a_feature_normalization_bug.md` for detailed options:
- **Option A (Recommended)**: Z-score normalization using training set statistics
- **Option B**: Percent change features
- **Option C**: Rolling window normalization
- **Option D**: Use only bounded features (RSI, percentiles, etc.)
- **Option E**: Hybrid approach

### Verification Method
1. Implement normalization (Option A first)
2. Regenerate SPY_dataset_a20.parquet with normalized features
3. Retrain one model (2M_h1)
4. Check predictions on 2024-2026 data - should NOT be ~0.52
5. If successful, re-run all Phase 6A experiments

### Status
**AUDIT COMPLETE** - Root cause identified. Implementation of fix pending.
