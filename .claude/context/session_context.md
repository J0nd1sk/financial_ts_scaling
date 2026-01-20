# Session Handoff - 2026-01-20 ~06:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `c179b96` docs: document feature normalization bug as root cause
- **Uncommitted changes**: None (clean)
- **Ahead of origin**: 7 commits (not pushed)

### Task Status
- **Phase 6A Investigation**: ROOT CAUSE FOUND AND DOCUMENTED
- **Next action**: Implement feature normalization fix

---

## CRITICAL FINDING

### THE BUG: Features Not Normalized

Models appeared to show "prior collapse" (~0.52 predictions) on 2024-2025 data. Investigation revealed:

1. Models output ~0.09 on val data (2016-2021) - **correct for 14% positive rate**
2. Models output ~0.52 on recent data (2024-2026) - **out-of-distribution confusion**
3. **Root cause**: Massive feature distribution shift due to no normalization

### Feature Distribution Shift

| Feature | Train (1994-2016) | Recent (2024-2026) | Shift |
|---------|-------------------|---------------------|-------|
| Close | 88.57 | 575.89 | **6.5x** |
| OBV | 4.0B | 16.6B | **4x** |
| ATR | 1.23 | 6.64 | **5x** |
| MACD | 0.20 | 3.23 | **16x** |
| RSI | 54.41 | 58.32 | ~1x (bounded) |

### Implications
- Model DID learn on training distribution (val_loss=0.203 is valid)
- All Phase 6A "failures" were measuring preprocessing failure
- The 19-sample val set issue was a distraction
- RSI stability confirms normalization is the root cause

---

## Documentation Created

- `docs/phase6a_feature_normalization_bug.md` - Comprehensive root cause analysis
- Memory: `Bug_FeatureNormalization_Phase6A`, `Solution_FeatureNormalization_Options`

---

## Test Status
- Last `make test`: 2026-01-20
- Result: **417 passed**

---

## Next Session Should

### Immediate Priority
1. **Implement Option A: Z-score normalization**
   ```python
   train_mean = X_train.mean(axis=0)
   train_std = X_train.std(axis=0)
   X_normalized = (X - train_mean) / (train_std + epsilon)
   ```
2. **Regenerate dataset** with normalized features
3. **Validate fix** - re-run 2M_h1, check predictions on 2025 data

### If Successful
4. Plan re-run of Phase 6A experiments with normalized features
5. Consider Option E (hybrid) for production

### Deprioritized (Until Normalization Fixed)
- Validation set size improvements (ChunkSplitter changes)
- Loss function experiments (SoftAUC vs BCE)
- Look-ahead bias audit

---

## Solution Options (Documented in docs/phase6a_feature_normalization_bug.md)

| Option | Approach | Effort | Notes |
|--------|----------|--------|-------|
| A | Z-score normalization | ~50 lines | **Recommended first** |
| D | Bounded features only | Medium | Most robust, requires re-engineering |
| E | Hybrid | High | Best for production |

---

## Memory Entities

### This Session
- `Bug_FeatureNormalization_Phase6A` - Root cause details
- `Solution_FeatureNormalization_Options` - Fix options

### Previous
- `Test1_BCE_vs_SoftAUC_Plan` - Results (invalid due to bug)
- `Test2_AUC_Early_Stopping_Plan` - Results (invalid due to bug)

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
