# SoftAUC Validation Plan

**Created:** 2026-01-19
**Status:** Pending validation
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

## Next Steps (Priority Order)

1. **Test 1:** Run AUC comparison (BCE vs SoftAUC on 2025 test data)
2. **Test 2:** Implement AUC-based early stopping
3. **Test 3:** Quick look-ahead bias audit
4. **Decision:** Based on results, decide whether to proceed with HPO re-run

---

## Memory Entities

- `SoftAUCLoss_Implementation_Plan` — Contains implementation details and validation results
- `Phase6A_Backtest_CriticalFinding` — Original prior collapse discovery
