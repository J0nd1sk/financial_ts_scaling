# Session Handoff - 2026-01-20 ~06:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `620eb59` feat: add AUC-based early stopping metric option
- **Uncommitted changes**: YES - new doc file (see below)
- **Ahead of origin**: 6 commits (not pushed)

### Task Status
- **Phase 6A Investigation**: **TRUE ROOT CAUSE FOUND** - Feature normalization bug
- **Previous hypotheses**: Prior collapse, small val set, loss function - all secondary issues

---

## CRITICAL DISCOVERY (This Session)

### THE ACTUAL BUG: Features Not Normalized

**Discovery path**:
1. Investigated why models output ~0.52 on 2025 data
2. Found models output ~0.09 on 2016-2021 val data (correct for 14% positive rate!)
3. Checked feature distributions across time periods
4. **FOUND**: Massive distribution shift in unnormalized features

### Feature Distribution Shift

| Feature | Train (1994-2016) | Recent (2024-2026) | Shift |
|---------|-------------------|---------------------|-------|
| Close | 88.57 | 575.89 | **6.5x** |
| OBV | 4.0B | 16.6B | **4x** |
| ATR | 1.23 | 6.64 | **5x** |
| MACD | 0.20 | 3.23 | **16x** |
| RSI | 54.41 | 58.32 | ~1x (bounded) |

### What This Means
- Model DID learn on training distribution (val_loss=0.203 is valid)
- Model outputs "I don't know" (0.52) for out-of-distribution inputs
- All Phase 6A "failures" were measuring preprocessing failure, not model failure
- The 19-sample val set was a distraction from the real bug
- RSI stability (naturally 0-100 bounded) confirms normalization is the issue

### Documented In
- `docs/phase6a_feature_normalization_bug.md` (NEW - comprehensive analysis)
- Memory: `Bug_FeatureNormalization_Phase6A`, `Solution_FeatureNormalization_Options`

---

## Previous Findings (Now Secondary)

### 1. Double-Sigmoid Bug (FIXED earlier)
- Location: `scripts/evaluate_final_models.py:256-258`
- Model outputs probabilities, script applied sigmoid again

### 2. Validation Set Too Small (19 samples)
- ChunkSplitter contiguous mode issue
- Now deprioritized - fix normalization first

### 3. Prior Collapse Hypothesis
- Was wrong interpretation - model learned fine on training distribution
- "Collapse" is actually out-of-distribution behavior

---

## Test Status
- Last `make test`: 2026-01-20
- Result: **417 passed**

---

## Files Modified/Created This Session

### New Files (Untracked)
```
docs/phase6a_feature_normalization_bug.md      # ROOT CAUSE DOCUMENTATION (NEW)
docs/phase6a_gap_analysis.md
docs/phase6a_validation_exploration_plan.md
.claude/context/phase6a_gap_checklist.md
.claude/context/phase6a_exploration_tracker.md
experiments/compare_bce_vs_soft_auc.py
```

### Modified Files (Committed)
```
src/training/trainer.py                         # AUC early stopping
tests/test_training.py                          # 5 new tests
.claude/context/soft_auc_validation_plan.md     # Test results
```

---

## Memory Entities Updated

### Created This Session
- `Bug_FeatureNormalization_Phase6A` - TRUE root cause
- `Solution_FeatureNormalization_Options` - Proposed fixes

### From Earlier This Session
- `Test1_BCE_vs_SoftAUC_Plan` - Results (now known to be invalid)
- `Test2_AUC_Early_Stopping_Plan` - Results (now known to be invalid)

---

## Next Session Should

### Immediate Priority
1. **Commit doc changes** (normalization bug documentation)
2. **Implement Option A: Z-score normalization** (~50 lines)
3. **Regenerate dataset** with normalized features
4. **Validate fix** - re-run one model, check predictions on 2025 data

### Then
5. **If successful** - plan re-run of Phase 6A experiments
6. **Consider Option E (hybrid)** for production

### Deprioritized (Until Normalization Fixed)
- Validation set size improvements
- Loss function experiments (SoftAUC vs BCE)
- Look-ahead bias audit

---

## Proposed Solutions (Priority Order)

### Option A: Z-Score Normalization (RECOMMENDED FIRST)
```python
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_normalized = (X - train_mean) / (train_std + epsilon)
```
Simple, standard practice, ~50 lines of code

### Option D: Bounded Features Only
Replace raw prices with RSI, %B, percentiles, z-scores
No normalization params needed, requires feature re-engineering

### Option E: Hybrid (PRODUCTION)
Percent changes for prices, z-scores for indicators, keep oscillators as-is

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
```

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
- Mark temporary docs clearly (will archive after use)
