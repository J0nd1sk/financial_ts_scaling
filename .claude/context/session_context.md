# Session Handoff - 2026-01-21 ~09:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `932783e` docs: session handoff - 20M_wide breakthrough, next L=2-4 trials
- **Uncommitted changes**: Multiple new experiment scripts and outputs
- **Ahead of origin**: 10 commits (not pushed)

### Task Status
- **Working on**: Target calculation fix (upside threshold using HIGH prices)
- **Status**: APPROVED, not yet implemented
- **Next**: TDD implementation of HIGH-based target in FinancialDataset

---

## CRITICAL: Target Calculation Bug - DEFINITIVE RULE

**See Memory entity: `Target_Calculation_Definitive_Rule`**

### What We Predict (Correct)

| Target Type | Question | Formula |
|-------------|----------|---------|
| **UPSIDE threshold** ✅ | "Will HIGH in next N days be ≥X% above today's CLOSE?" | `max(high[t+1:t+1+horizon]) >= close[t] * (1+X%)` |
| **DOWNSIDE threshold** (future) ✅ | "Will LOW in next N days be ≥X% below today's CLOSE?" | `min(low[t+1:t+1+horizon]) <= close[t] * (1-X%)` |

### What We NEVER Predict

❌ **NEVER** predict future CLOSE relative to current CLOSE. This is not how trading works.
A trade entered at today's close achieves profit when the HIGH reaches the target, not when the CLOSE does.

### Current Bug (to be fixed)

```python
# WRONG - uses CLOSE prices for future window
future_max = max(close[t+1:t+1+horizon])

# CORRECT - uses HIGH prices for upside threshold
future_max = max(high[t+1:t+1+horizon])
```

### Impact on Class Balance

| Threshold | Using CLOSE (wrong) | Using HIGH (correct) |
|-----------|---------------------|----------------------|
| 0.5% | 29.1% | **50.0%** (balanced!) |
| 1.0% | 14.1% | 23.9% |

**User preference**: 0.5% threshold with HIGH-based target (50/50 balance)

### Deprecated Terminology

The terms "Close-to-Close" and "Close-to-High" are **DEPRECATED** - they caused confusion.
Use "upside threshold (HIGH-based)" and "downside threshold (LOW-based)" instead.

---

## Experiments Completed This Session

### 1. Shallow Depth Sweep (L=2-5 at 20M)
**Finding**: L=6 remains optimal. Shallower underfits.

| Config | L | d | AUC | vs L=6 (0.7342) |
|--------|---|---|-----|-----------------|
| 20M_L2 | 2 | 896 | 0.7163 | -1.8% |
| 20M_L3 | 3 | 720 | 0.7139 | -2.0% |
| 20M_L4 | 4 | 640 | 0.7177 | -1.7% |
| 20M_L5 | 5 | 560 | 0.7222 | -1.2% |

### 2. LR/Dropout Tuning
**Finding**: Current settings (LR=1e-4, dropout=0.5) are optimal. More regularization hurts.

| Config | LR | Dropout | AUC | vs ref |
|--------|-----|---------|-----|--------|
| lr8e5_d50 | 8e-5 | 0.50 | 0.7202 | -1.4% |
| lr5e5_d50 | 5e-5 | 0.50 | 0.7160 | -1.8% |
| lr1e4_d55 | 1e-4 | 0.55 | 0.7068 | -2.7% |
| lr8e5_d55 | 8e-5 | 0.55 | 0.7091 | -2.5% |
| lr5e5_d55 | 5e-5 | 0.55 | 0.7078 | -2.6% |

---

## Best Model (Current - trained with CLOSE-based target, needs retraining)

**20M_wide**: d=512, L=6, h=8, LR=1e-4, dropout=0.5 → **AUC 0.7342** (+1.8% over RF)

⚠️ This model was trained with the incorrect CLOSE-based target. Results will need to be re-validated after fixing to HIGH-based target.

---

## Next Session: Implementation Plan

### Task: Add HIGH-based Upside Threshold Target to FinancialDataset

**Approved change:**
1. Add optional `high_prices` parameter to FinancialDataset
2. When provided, use `max(high[t+1:t+1+horizon])` for upside threshold target
3. Backward compatible - default behavior unchanged (for reproducibility of old experiments)

**TDD Steps:**
1. Write failing tests for HIGH-based target calculation
2. Implement the change
3. Run `make test` to verify all pass
4. Run experiments with 0.5% threshold, HIGH-based target

**Also requested:**
- Report comprehensive metrics: AUC, accuracy, precision, recall, F1
- Test h=4 (fewer heads) variant

---

## Test Status
- Last `make test`: 2026-01-21 ~08:50 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Files Created This Session (Not Committed)

| File | Description |
|------|-------------|
| `scripts/test_shallow_depth.py` | L=2-5 depth experiment |
| `scripts/test_lr_dropout_tuning.py` | LR/dropout tuning experiment |
| `scripts/backtest_optimal_models.py` | Created in prior sub-session |
| `outputs/shallow_depth/` | Results (CSV, JSON) |
| `outputs/lr_dropout_tuning/` | Results (CSV, JSON) |

---

## Memory Entities Updated This Session

**Created:**
- `Target_Calculation_Definitive_Rule` - **CANONICAL** definition of upside/downside threshold targets (HIGH/LOW based, NEVER CLOSE)
- `Finding_ShallowDepthExperiment_20260121` - L=6 optimal, shallower underfits
- `Finding_LRDropoutTuning_20260121` - LR=1e-4, dropout=0.5 optimal

**From previous sessions (relevant):**
- `Finding_DropoutScalingExperiment_20260121` - 20M_wide 0.7342 beats RF (with CLOSE-based target, needs revalidation)
- `Finding_TrainingDynamicsExperiment_20260121` - Dropout=0.5 works at 2M

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
make verify
```

---

## Next Session Should

1. **Implement HIGH-based upside threshold target** in FinancialDataset (TDD)
2. **Run 0.5% threshold experiment** with balanced data (~50/50 class distribution)
3. **Report full metrics**: AUC, accuracy, precision, recall, F1
4. **Test h=4 variant** (fewer heads)

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Document in multiple places: Memory MCP + context files + docs/
- Code comments are secondary, not primary durability

### Documentation Philosophy
- Flat docs/ structure (no subdirs except research_paper/, archive/)
- Precision in language - never reduce fidelity
- Consolidate rather than delete - preserve historical context

### Communication Standards
- Precision over brevity
- Never summarize away important details
- Evidence-based claims

### Current Focus
- **CRITICAL**: Fix target calculation (use HIGH prices for upside threshold - see `Target_Calculation_Definitive_Rule` in Memory)
- Test 0.5% threshold with balanced data (~50/50 with HIGH-based target)
- Full metrics reporting (not just AUC)
- Test h=4 (fewer heads) variant
