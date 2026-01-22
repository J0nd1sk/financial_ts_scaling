# Phase 6A Gap Analysis: Pre-HPO Rerun Checklist

> **TEMPORARY DOCUMENT** - This document tracks active investigation gaps.
> Will be archived to `docs/archive/` or consolidated into final analysis once HPO re-run is complete.

**Date**: 2026-01-20
**Status**: Analysis complete, action items pending
**Lifecycle**: Active investigation (delete/archive after Phase 6A HPO re-run)
**Context**: Before re-running HPO with Soft AUC loss, systematic review of potential gaps

---

## Background

After identifying prior collapse (models predict class prior for all samples) and determining Soft AUC as the winning loss function, we conducted a systematic gap analysis to ensure no other issues would invalidate a costly HPO re-run (~200 hours compute).

---

## Gap Categories

### 1. Loss Function ✅ ADDRESSED

| Issue | Status | Resolution |
|-------|--------|------------|
| Prior collapse | IDENTIFIED | BCE allows degenerate solutions |
| Loss function selection | RESOLVED | Soft AUC wins validation testing |

---

### 2. Architecture Gaps

| Question | Current State | Risk Level | Action Needed |
|----------|---------------|------------|---------------|
| Sigmoid in forward pass | Model outputs probs via `PredictionHead` | Medium | Consider outputting logits for numerical stability |
| Is PatchTST right for finance? | Designed for long-horizon forecasting | Unknown | Literature review - patch-based vs other architectures |
| Patch size | Need to verify current value | Unknown | Document rationale, test alternatives |
| Context length | Need to verify current value | Unknown | Document rationale, test alternatives |
| Positional encoding | Standard sinusoidal? | Low | May need learnable for financial patterns |

**Files to check**: `src/models/patchtst.py`, config files

---

### 3. Data Pipeline Gaps

| Question | Current State | Risk Level | Action Needed |
|----------|---------------|------------|---------------|
| **Look-ahead bias** | UNVERIFIED | 🔴 CRITICAL | Audit ALL features for future information |
| **Feature normalization** | UNVERIFIED | 🔴 HIGH | Must be rolling window, not global |
| Non-stationarity | Train 1993-2020, test 2023+ | 🟡 HIGH | 30-year gap, market regimes changed |
| Feature selection | 20 indicators fixed | 🟡 MEDIUM | Are these the right indicators? |
| Target definition | >1% binary | 🟡 MEDIUM | Why 1%? Test other thresholds |
| Data leakage in splits | Chunk-based splits | 🟢 LOW | Verify no overlap |

**Files to check**: `src/features/tier_a20.py`, `src/data/splitter.py`, feature build scripts

---

### 4. Training Pipeline Gaps

| Question | Current State | Risk Level | Action Needed |
|----------|---------------|------------|---------------|
| **Early stopping metric** | Was val_loss (BCE) | 🔴 CRITICAL | MUST change to AUC for Soft AUC training |
| Learning rate schedule | In HPO search space | 🟡 MEDIUM | Verify appropriate for transformers |
| Warmup steps | In HPO search space | 🟡 MEDIUM | Verify not too short |
| Weight decay | In HPO search space | 🟢 LOW | Already tuned |
| Dropout | Added recently | 🟢 LOW | Already in search space |
| Gradient accumulation | Implemented | 🟢 LOW | Working correctly |

**Files to check**: `src/training/trainer.py`, `src/training/hpo.py`, config files

---

### 5. Evaluation Gaps

| Question | Current State | Risk Level | Action Needed |
|----------|---------------|------------|---------------|
| Double-sigmoid bug | FIXED | ✅ RESOLVED | Evaluation script corrected |
| Metrics alignment | Train on AUC, eval on AUC | 🟢 LOW | Should be aligned |
| Confidence calibration | Unknown with Soft AUC | 🟡 MEDIUM | May need post-hoc calibration |
| Val/Test distribution shift | Same source, different years | 🟡 MEDIUM | Document regime differences |

---

### 6. Fundamental Questions

| Question | Implication | Priority |
|----------|-------------|----------|
| Is 1% threshold optimal? | Maybe 0.5% or 2% more learnable | Test before HPO |
| Classification vs regression? | Regression→threshold might work better | Consider for Phase 6B |
| Is daily frequency optimal? | Hourly more signal? Weekly less noise? | Consider for Phase 6B |
| EMH ceiling | Best achievable AUC might be ~0.55 | Sets expectations |
| Random Forest baseline | AUC 0.68-0.82 - can transformers match? | Benchmark target |

---

## Pre-HPO Checklist

### 🔴 Must Complete Before HPO

- [ ] **Look-ahead bias audit**
  - Review `src/features/tier_a20.py` for any features using future data
  - Check indicator calculations (RSI, MACD, etc.) use only past data
  - Verify target labels don't leak future information

- [ ] **Early stopping metric change**
  - Modify `src/training/trainer.py` to support AUC-based early stopping
  - Or: early stop on Soft AUC loss value

- [ ] **Feature normalization audit**
  - Verify normalization uses rolling window (e.g., past 252 days)
  - NOT global normalization (would leak future statistics)

### 🟡 Should Complete Before HPO

- [ ] **Document context length and patch size**
  - What values are currently used?
  - What's the rationale?

- [ ] **Test threshold sensitivity**
  - Quick experiment: 0.5%, 1%, 2% thresholds
  - Does changing threshold affect AUC significantly?

- [ ] **Verify Soft AUC implementation**
  - Correct gradient flow?
  - Numerical stability?

### 🟢 Nice to Have

- [ ] **Feature importance analysis**
  - Which of 20 indicators contribute most?
  - Any redundant features?

- [ ] **Attention visualization**
  - What temporal patterns is model learning?

- [ ] **Literature review**
  - Other transformer architectures for financial data?
  - State-of-the-art approaches?

---

## Risk Assessment

| If we skip... | Consequence |
|---------------|-------------|
| Look-ahead bias audit | Could invalidate ALL results - unrealistic performance |
| Early stopping change | Models may stop too early/late with wrong metric |
| Normalization audit | Subtle leakage, unreproducible results |
| Threshold testing | Might be optimizing wrong task |

---

## Recommended Order of Operations

1. **Audit** (before ANY retraining)
   - Look-ahead bias check
   - Normalization check
   - Document current architecture params

2. **Fix** (if issues found)
   - Correct any leakage
   - Implement AUC-based early stopping

3. **Quick experiments** (before full HPO)
   - Threshold sensitivity (0.5%, 1%, 2%)
   - Verify Soft AUC training works end-to-end

4. **HPO re-run** (only after above complete)
   - With Soft AUC loss
   - AUC-based early stopping
   - Clean data pipeline

---

## Memory Entities

- `Phase6A_GapAnalysis` (to be created)

---

## Related Documents

- `docs/phase6a_backtest_analysis.md` - Backtest findings
- `docs/research_paper/notes/prior_collapse_investigation.md` - Root cause analysis
- `.claude/context/phase6a_gap_checklist.md` - Actionable checklist
