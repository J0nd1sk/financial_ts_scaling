# SoftAUC Validation Plan

**Created:** 2026-01-19
**Updated:** 2026-01-20
**Status:** Test 1 COMPLETE - Negative result
**Purpose:** Ensure SoftAUCLoss improves actual model performance, not just prediction spread

---

## Background

SoftAUCLoss was implemented to address prior collapse where BCE causes models to predict class prior (~15% for h1) for ALL samples with <1% spread.

**Initial validation results:**
- Spread improved 7.8x (0.078 vs <0.01)
- BUT: Need to verify this translates to better predictions on held-out data

---

## Gap Analysis (from code review)

### Critical (Pre-HPO)

| Gap | Status | Notes |
|-----|--------|-------|
| Look-ahead bias audit | ⏳ TODO | Verify no feature uses future information |
| Early stopping metric | ⏳ TODO | Must change to AUC if training on SoftAUC |
| Feature normalization | ✅ OK | Features are raw indicator values, not normalized with global stats |

### Important

| Gap | Status | Notes |
|-----|--------|-------|
| Context length rationale | ✅ OK | 60 days = ~3 months, standard for daily data |
| Patch size rationale | ✅ OK | 16 days with stride 8, from PatchTST paper |
| Target threshold | ⏳ TODO | Only tested 1%, could test 0.5%, 2% |
| Distribution shift | ✅ Known | Using contiguous splits for realistic eval |

### Worth Considering (Later)

- Regression alternative (train on returns, threshold at inference)
- Feature importance analysis
- Attention visualization

---

## Validation Tests

### Test 1: AUC Comparison on 2025 Test Data (PRIORITY)

**Goal:** Does SoftAUCLoss produce better AUC-ROC on held-out test data?

**Method:**
1. Train 2M_h1 model with BCE loss (or use existing checkpoint)
2. Train 2M_h1 model with SoftAUCLoss (same architecture, same data)
3. Evaluate both on 2025 test set (NOT validation set)
4. Compare AUC-ROC scores

**Success criteria:** SoftAUC AUC > BCE AUC on test set

**Script needed:** `experiments/compare_bce_vs_soft_auc.py`

### Test 2: AUC-Based Early Stopping

**Goal:** Align early stopping metric with training objective

**Method:**
1. Add `early_stopping_metric` parameter to Trainer ("val_loss" or "val_auc")
2. Implement AUC computation in `_check_early_stopping()`
3. Re-run validation with AUC-based stopping

**Files to modify:**
- `src/training/trainer.py` — add metric option
- `tests/test_training.py` — add tests for AUC stopping

### Test 3: Look-Ahead Bias Audit

**Goal:** Verify feature pipeline has no future information leakage

**Checklist:**
- [ ] Review `src/features/tier_a20.py` — all indicators use only past data?
- [ ] Review target construction in `src/data/dataset.py` — labels computed correctly?
- [ ] Check any normalization/scaling steps

**Key indicators to audit:**
- DEMA, SMA — should use lookback only
- RSI, StochRSI — standard TA-Lib implementation
- MACD — uses historical EMA
- ATR, ADX — rolling calculations
- Bollinger %B — rolling mean/std
- VWAP — rolling calculation

---

## Current State

### Commits This Session

| Commit | Description |
|--------|-------------|
| `f2c0e62` | Backtest evaluation + prior collapse investigation |
| `7f65bba` | SoftAUCLoss implementation (TDD, 11 tests) |
| `3c50b0a` | Bug fix (gradient flow) + validation script |

### Files Created

- `src/training/losses.py` — SoftAUCLoss class
- `tests/test_losses.py` — 11 tests
- `experiments/validate_soft_auc.py` — Quick validation script

### Test Status

- 412 tests passing
- All existing functionality preserved

---

## Test 1 Results (2026-01-20)

### Experiment
- Script: `experiments/compare_bce_vs_soft_auc.py`
- Architecture: 2M_h1 (d_model=64, n_layers=48, n_heads=2)
- Training: lr=0.0008, dropout=0.12, epochs=50, early_stopping_patience=10
- Data: SPY_dataset_a20.parquet, contiguous splits

### Results

| Model | Test AUC-ROC | Spread | Prediction Range | Val Loss | Epochs |
|-------|--------------|--------|------------------|----------|--------|
| BCE | **0.5707** | 2.59% | [0.072, 0.098] | 0.203 | 6 |
| SoftAUC | 0.5374 | 2.26% | [0.045, 0.068] | 0.459 | 5 |

**Key Finding: SoftAUC HURT performance**
- AUC-ROC decreased by 5.8% (0.5707 → 0.5374)
- Spread did NOT improve (both ~2.5%)
- Both models exhibit prior collapse (predictions near class prior ~10%)

### Possible Explanations

1. **BCE hyperparams not suited for SoftAUC**: lr, dropout, weight_decay were tuned for BCE
2. **Small validation set (19 samples)**: ChunkSplitter in contiguous mode gives sparse val set
3. **Early stopping on val_loss suboptimal**: SoftAUC loss doesn't correlate well with AUC-ROC
4. **SoftAUC overfits ranking on train but doesn't generalize**
5. **Problem isn't prior collapse but weak signal**: AUC 0.57 suggests limited predictive power

### Conclusion

**SoftAUCLoss alone does NOT solve the problem.** The spread improvement seen in validation (~7.8x) did not translate to better ranking on test data.

---

## Test 2 Results (2026-01-20)

### Experiment
- Added `early_stopping_metric="val_auc"` parameter to Trainer
- Re-ran comparison with AUC-based early stopping for SoftAUC

### Results

| Model | Test AUC-ROC | Spread | Val AUC | Epochs |
|-------|--------------|--------|---------|--------|
| BCE (val_loss stopping) | **0.5707** | 2.59% | N/A | 6 |
| SoftAUC (val_auc stopping) | 0.4867 | 0.06% | 1.0 | 1 |

**Key Finding: AUC-based stopping made things WORSE**
- Val AUC hit 1.0 after just 1 epoch (on 19 samples!)
- Model stopped training immediately
- Test AUC dropped to 0.4867 (worse than random)
- Prediction spread collapsed to 0.06%

### Root Cause
**Validation set too small (19 samples)** - ChunkSplitter in contiguous mode uses non-overlapping windows, resulting in only 19 validation samples. This is:
- Insufficient for reliable val_loss early stopping
- Catastrophic for AUC-based early stopping (instantly achieves perfect score)

### Conclusion
The small validation set is the fundamental blocker. AUC-based early stopping is a valid feature but cannot help until the validation set issue is addressed.

---

## Next Steps (Revised Again)

~~1. **Test 1:** Run AUC comparison~~ ✅ COMPLETE - Negative result
~~2. **Test 2:** AUC-based early stopping~~ ✅ COMPLETE - Made things worse due to small val set

**CRITICAL BLOCKER: Small validation set (19 samples)**

Must address before any further loss function experiments:
- Option A: Increase `val_ratio` in ChunkSplitter (e.g., 0.30 instead of 0.15)
- Option B: Change ChunkSplitter mode from "contiguous" to allow overlapping val samples
- Option C: Use time-based splits instead of ratio-based

3. **Test 3:** Look-ahead bias audit (still valid regardless of loss function)
4. **After fixing val set size:**
   - Re-test BCE vs SoftAUC with proper validation
   - Consider class weighting in BCE (pos_weight parameter)
5. **Decision:** The underlying signal may be weak (AUC ~0.57), but we can't draw conclusions until validation is fixed

---

## Memory Entities

- `SoftAUCLoss_Implementation_Plan` — Contains implementation details and validation results
- `Phase6A_Backtest_CriticalFinding` — Original prior collapse discovery
