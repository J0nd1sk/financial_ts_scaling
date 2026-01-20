# Phase 6A Gap Checklist

**Purpose**: Actionable checklist before HPO re-run with Soft AUC
**Created**: 2026-01-20
**Status**: ACTIVE - check off items as completed

---

## 🔴 CRITICAL - Must Complete Before HPO

### 1. Look-ahead Bias Audit
- [ ] Review `src/features/tier_a20.py` - all indicators use only past data?
- [ ] Check RSI, MACD, Bollinger calculations - no future values?
- [ ] Verify target labels constructed correctly (future returns, not including current day?)
- [ ] Check any rolling calculations for off-by-one errors

**Risk if skipped**: Invalidates ALL results - unrealistic performance

### 2. Early Stopping Metric
- [ ] Current: stops on val_loss (BCE)
- [ ] Need: stop on AUC (or Soft AUC loss)
- [ ] File to modify: `src/training/trainer.py`
- [ ] Also update: `src/training/hpo.py` if metric tracked there

**Risk if skipped**: Models stop at wrong point

### 3. Feature Normalization Audit
- [ ] Check how features are normalized in pipeline
- [ ] Must be ROLLING window (e.g., past 252 days)
- [ ] NOT global normalization (leaks future statistics)
- [ ] Files to check: `src/features/tier_a20.py`, data loading code

**Risk if skipped**: Subtle leakage, unreproducible results

---

## 🟡 SHOULD Complete Before HPO

### 4. Document Architecture Parameters
- [ ] Current context length: ___
- [ ] Current patch size: ___
- [ ] Rationale documented?
- [ ] File: `src/models/patchtst.py`, config files

### 5. Threshold Sensitivity Test
- [ ] Quick test with 0.5% threshold
- [ ] Quick test with 2% threshold
- [ ] Compare AUC to 1% threshold
- [ ] Decision: stick with 1% or change?

### 6. Verify Soft AUC Implementation
- [ ] Correct gradient flow?
- [ ] Numerically stable?
- [ ] Validation test passing?

---

## 🟢 Nice to Have

### 7. Feature Importance
- [ ] Which of 20 indicators matter most?
- [ ] Any redundant?

### 8. Attention Visualization
- [ ] What patterns is model learning?

---

## Findings Log

| Date | Item | Finding | Action Taken |
|------|------|---------|--------------|
| | | | |

---

## Memory Entities

Query these for context:
- `Phase6A_DoubleSigmoidBug`
- `Phase6A_PriorCollapse_RootCause`
- `Phase6A_ArchitecturalIssues`
- `Phase6A_FixesAttempted`
- `Phase6A_ProposedFixes`

---

## Related Docs

- `docs/phase6a_gap_analysis.md` - Full analysis (TEMPORARY)
- `docs/phase6a_backtest_analysis.md` - Backtest findings
- `docs/research_paper/notes/prior_collapse_investigation.md` - Root cause
