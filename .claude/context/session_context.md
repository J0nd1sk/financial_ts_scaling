# Session Handoff - 2026-01-20 ~03:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `3c50b0a` - fix: SoftAUCLoss gradient flow in degenerate batches
- **Uncommitted changes**: None (clean working tree)
- **Ahead of origin**: 4 commits (not pushed)

### Task Status
**SoftAUCLoss Implementation** - COMPLETE, validation in progress

## Test Status
- Last `make test`: 2026-01-20
- Result: **412 passed**
- Failing: none

---

## WHAT WE DID THIS SESSION

### 1. Committed Prior Session Work
- `f2c0e62` — Backtest evaluation script + prior collapse investigation
- Double sigmoid bug fix in evaluate_final_models.py
- Research paper artifacts (figures, tables, analysis docs)

### 2. Implemented SoftAUCLoss (TDD)
- **Planning session** completed with full test plan
- Created `src/training/losses.py` with SoftAUCLoss class
- Added `criterion` parameter to Trainer (optional, defaults to BCELoss)
- 11 new tests in `tests/test_losses.py`
- Commits: `7f65bba`, `3c50b0a` (bug fix)

### 3. Initial Validation
- Ran `experiments/validate_soft_auc.py` on 2M_h1 architecture
- **Result:** Spread improved 7.8x (0.078 vs BCE's <0.01)
- **Concern:** Need to verify this improves ACTUAL performance (AUC on test data)

---

## PENDING WORK

### Immediate: SoftAUC Validation (see `.claude/context/soft_auc_validation_plan.md`)

**Test 1 (Priority):** AUC comparison on 2025 test data
- Train BCE vs SoftAUC models
- Compare AUC-ROC on held-out 2025 test set
- Answer: Does better spread → better predictions?

**Test 2:** Implement AUC-based early stopping
- Currently stops on val_loss (BCE)
- Should stop on val_AUC when training with SoftAUC

**Test 3:** Quick look-ahead bias audit
- Verify feature pipeline has no future leakage

### Gap Analysis Highlights

| Gap | Priority | Status |
|-----|----------|--------|
| Early stopping metric | Critical | TODO — must align with loss function |
| Look-ahead bias audit | Critical | TODO — verify no future leakage |
| Feature normalization | OK | Raw indicators, no global stats |
| Context/patch size | OK | 60 days, patch=16, stride=8 |

---

## KEY DECISIONS MADE

1. **SoftAUCLoss formula:** `sigmoid(gamma * (neg - pos)).mean()` over all pairs
2. **Gamma = 2.0:** Provides good gradient flow without being too sharp
3. **Degenerate case handling:** Use `predictions.mean() * 0 + 0.5` to maintain requires_grad
4. **Validation approach:** Must test on 2025 TEST data, not just validation spread

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Insists on durability for pending actions
- Document in multiple places: Memory MCP + context files + docs/
- Code comments are secondary, not primary durability

### Documentation Philosophy
- Prefers consolidation of docs/ files over deletion
- Preserve historical context - "what we did and why"
- Flat docs/ structure - no subdirectories except research_paper/ and archive/
- Precision in language - never reduce fidelity of descriptions

### Communication Standards
- Never summarize away important details
- Maintain coherent, PRECISE history
- Evidence > assumptions
- Full validation: User expects complete verification, not spot-checks

### This Session Preferences
- **Goal:** Models that actually perform well, THEN measure scaling effects
- Wants thorough validation of SoftAUCLoss before proceeding
- Concerned about gaps in methodology (early stopping, bias audit)

---

## Memory Entities Updated

- `SoftAUCLoss_Implementation_Plan` — Implementation details + validation results

---

## Files Modified This Session

| File | Change |
|------|--------|
| `src/training/losses.py` | NEW — SoftAUCLoss class |
| `src/training/__init__.py` | Export SoftAUCLoss |
| `src/training/trainer.py` | Add criterion parameter |
| `tests/test_losses.py` | NEW — 11 tests |
| `experiments/validate_soft_auc.py` | NEW — Validation script |
| `.claude/context/soft_auc_validation_plan.md` | NEW — Validation plan |

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
cat .claude/context/soft_auc_validation_plan.md
```

---

## Next Session Should

1. Read `.claude/context/soft_auc_validation_plan.md` for detailed plan
2. Execute **Test 1**: AUC comparison (BCE vs SoftAUC on 2025 test data)
3. Based on results, proceed with Test 2 (AUC early stopping) or investigate further
