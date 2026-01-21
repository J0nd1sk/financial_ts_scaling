# Session Handoff - 2026-01-21 ~11:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `078198c` feat: HIGH-based upside threshold targets + documentation audit
- **Uncommitted changes**: None (clean working tree)
- **Ahead of origin**: 11 commits (not pushed)

### Task Status
- **Completed**: HIGH-based target implementation + documentation audit
- **Status**: Ready for 0.5% threshold experiments
- **Next**: Run experiments with HIGH-based targets at 0.5% threshold

---

## CRITICAL: Target Calculation - DEFINITIVE RULE

**See Memory entity: `Target_Calculation_Definitive_Rule`**

### What We Predict (Correct)

| Target Type | Question | Formula |
|-------------|----------|---------|
| **UPSIDE threshold** ✅ | "Will HIGH in next N days be ≥X% above today's CLOSE?" | `max(high[t+1:t+1+horizon]) >= close[t] * (1+X%)` |
| **DOWNSIDE threshold** (future) ✅ | "Will LOW in next N days be ≥X% below today's CLOSE?" | `min(low[t+1:t+1+horizon]) <= close[t] * (1-X%)` |

### What We NEVER Predict

❌ **NEVER** predict future CLOSE relative to current CLOSE. This is not how trading works.
A trade entered at today's close achieves profit when the HIGH reaches the target, not when the CLOSE does.

### Implementation Complete

```python
# FinancialDataset now accepts high_prices parameter
dataset = FinancialDataset(
    features_df=df,
    close_prices=close,
    high_prices=high,  # NEW - uses HIGH for target calculation
    context_length=60,
    horizon=1,
    threshold=0.005,  # 0.5%
)
```

### Impact on Class Balance

| Threshold | Using CLOSE (wrong) | Using HIGH (correct) |
|-----------|---------------------|----------------------|
| 0.5% | 29.1% | **~50%** (balanced!) |
| 1.0% | 14.1% | ~24% |

---

## Experiments Completed (Prior Sessions)

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

---

## Best Model (trained with CLOSE-based target, needs retraining with HIGH)

**20M_wide**: d=512, L=6, h=8, LR=1e-4, dropout=0.5 → **AUC 0.7342** (+1.8% over RF)

⚠️ This model was trained with the old CLOSE-based target. Results will change with HIGH-based target.

---

## Next Session: 0.5% Threshold Experiments

### Task: Run experiments with HIGH-based targets at 0.5% threshold

**Experiment Setup:**
1. Load SPY data with both Close and High prices
2. Create FinancialDataset with `high_prices=high`, `threshold=0.005`
3. Train 20M_wide model (d=512, L=6, h=8)
4. Report full metrics: AUC, accuracy, precision, recall, F1

**Also test:**
- h=4 variant (fewer heads) - compare to h=8

**Expected Outcomes:**
- ~50/50 class balance with 0.5% threshold + HIGH-based targets
- Model should learn meaningful patterns with balanced data
- Full metrics will show if model is just predicting majority class

---

## Test Status
- Last `make test`: 2026-01-21 ~10:45 UTC
- Result: **469 passed**, 2 warnings
- Failing: none

---

## Memory Entities Updated This Session

**Created:**
- `Target_Calculation_Definitive_Rule` - **CANONICAL** definition of upside/downside threshold targets (HIGH/LOW based, NEVER CLOSE)

**From previous sessions (relevant):**
- `Finding_ShallowDepthExperiment_20260121` - L=6 optimal, shallower underfits
- `Finding_LRDropoutTuning_20260121` - LR=1e-4, dropout=0.5 optimal
- `Finding_DropoutScalingExperiment_20260121` - 20M_wide 0.7342 beats RF (with CLOSE-based target)

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

1. **Create experiment script** for 0.5% threshold with HIGH-based targets
2. **Train 20M_wide model** with new target calculation
3. **Report full metrics**: AUC, accuracy, precision, recall, F1
4. **Test h=4 variant** and compare to h=8

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
- Run 0.5% threshold experiments with HIGH-based targets
- Full metrics reporting (not just AUC)
- Test h=4 (fewer heads) variant
