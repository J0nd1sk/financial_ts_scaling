# Workstream 3 Context: Phase 6C Experiments
# Last Updated: 2026-01-26 00:30

## Identity
- **ID**: ws3
- **Name**: phase6c
- **Focus**: Phase 6C feature scaling experiments (a50 → a100 tier)
- **Status**: **Phase 1 Complete** - Ready for Phase 2 (HPO search space expansion)

## Current Task
- **Working on**: Deep Code Audit - Phase 6C HPO & Training Infrastructure
- **Status**: Phase 1 COMPLETE, Phases 2-4 pending

---

## Session 2026-01-26: Deep Code Audit (Phase 1 Complete)

### Completed This Session

#### 1. Added Recall/Precision Metrics to Trainer ✅
**File**: `src/training/trainer.py`
- Added `recall` and `precision` calculation in `_evaluate_detailed()` method
- Added `pred_range` (min, max) for detecting probability collapse
- Updated `train()` method to return these metrics when `verbose=True`
- Now tracks: recall, precision, pred_range in addition to accuracy/AUC

#### 2. Fixed HPO Exception Handling ✅ (All 6 scripts)
**Files**: All `experiments/phase6c_a100/hpo_*.py`
- Replaced bare `except Exception` with specific exception handling
- OOM/MPS errors → `raise optuna.TrialPruned()` (don't poison study with 0.5)
- NaN/Inf errors → `raise optuna.TrialPruned()`
- KeyboardInterrupt → re-raise (let user stop)
- All errors logged to trial metadata via `trial.set_user_attr("error", ...)`

#### 3. Fixed Experiment Numbering / Path Issues ✅
**Files**: `statistical_validation.py`, `compare_all_tiers.py`
- Fixed `RESULT_DIRS["a20"]` path: `phase6a` → `phase6a_final`
- Added tier-aware experiment naming:
  - a20 (Phase 6A): `phase6a_2m_h1` naming convention
  - a50/a100 (Phase 6C): `s1_01_2m_h1` sequential numbering convention
- Updated `load_tier_results()` in compare_all_tiers.py similarly

### Tests
- **786 passed**, 2 skipped - All tests passing

### Not Done Yet (Phases 2-4)

#### Phase 2: Expand HPO Search Space (User's main intent)
Add to all 6 HPO scripts:
```python
SEARCH_SPACE = {
    # Architecture (existing)
    "d_model": [...],
    "n_layers": [...],
    "n_heads": [...],
    "d_ff_ratio": [2, 4],

    # Training hyperparameters (NEW)
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
    "dropout": [0.1, 0.3, 0.5, 0.7],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [32, 64, 128],  # Independent, not d_model-derived
}
```
- Remove d_model-based batch_size derivation
- Pass `weight_decay` to Trainer (already supported but not passed!)
- Use sampled values instead of fixed LEARNING_RATE/DROPOUT constants

#### Phase 3: High Severity Fixes
- Log AUC=None with warnings
- Fix gradient accumulation edge case
- Validate high_prices column presence
- Update default context_length to 80 in experiment.py
- Add device selection logging

#### Phase 4: Medium Severity Fixes
- Division by zero risk in compare_all_tiers.py
- Float formatting crash for None values
- Bootstrap CI iteration counting
- Remove dead code (threshold_0.5pct)
- File I/O error handling

---

## Files Modified This Session
- `src/training/trainer.py` - Added recall/precision/pred_range metrics
- `experiments/phase6c_a100/hpo_2m_h1.py` - Fixed exception handling
- `experiments/phase6c_a100/hpo_2m_h5.py` - Fixed exception handling
- `experiments/phase6c_a100/hpo_20m_h1.py` - Fixed exception handling
- `experiments/phase6c_a100/hpo_20m_h5.py` - Fixed exception handling
- `experiments/phase6c_a100/hpo_200m_h1.py` - Fixed exception handling
- `experiments/phase6c_a100/hpo_200m_h5.py` - Fixed exception handling
- `experiments/phase6c_a100/statistical_validation.py` - Fixed paths and naming
- `experiments/phase6c_a100/compare_all_tiers.py` - Fixed paths and naming

---

## Outputs Status
```
outputs/phase6c_a100/
├── s1_01_2m_h1/ through s1_12_200m_h5/  (all 12 complete)
├── hpo_2m_h1/  (exists but untested with fixes)
├── hpo_20m_h1/ (exists but untested)
├── hpo_200m_h1/ (partial)
```

---

## Next Session Should

1. **Phase 2: Expand HPO search space** (user's main intent)
   - Add learning_rate, dropout, weight_decay, batch_size to search
   - Update all 6 HPO scripts
   - Remove d_model-based batch derivation
   - Pass weight_decay to Trainer

2. **Test one HPO script manually** after Phase 2:
   ```bash
   ./venv/bin/python experiments/phase6c_a100/hpo_2m_h1.py
   ```

3. **Phase 3-4**: High/Medium severity fixes (optional, lower priority)

4. **Run full HPO overnight** after fixes verified

---

## Lessons Learned (Audit Findings)

1. **HPO only searched architecture, not training params** - Fixed constants for LR, dropout, weight_decay meant we never explored their interaction with architecture

2. **Exception masking poisons Optuna studies** - Returning 0.5 for failed trials makes them indistinguishable from actual 0.5 AUC results. Use `optuna.TrialPruned()` instead.

3. **Experiment naming conventions differ between phases** - Phase 6A uses `phase6a_2m_h1`, Phase 6C uses `s1_01_2m_h1`. Analysis scripts must handle both.

4. **Missing metrics (recall/precision) hide model failures** - A model predicting all negatives passes silently without recall tracking. CLAUDE.md explicitly requires these metrics.

---

## Memory Entities (This Session)
- `Phase6C_DeepCodeAudit_Phase1_20260126` - (to be created)

---

## Session History

### 2026-01-26 00:30 (Deep Code Audit - Phase 1)
- Completed Phase 1 of deep code audit (4 critical issues)
- Added recall/precision/pred_range metrics to Trainer
- Fixed exception handling in all 6 HPO scripts
- Fixed experiment paths and naming conventions
- Ready for Phase 2 (HPO search space expansion)

### 2026-01-25 17:15 (S1 Complete, HPO Blocked)
- Ran all 12 S1 baselines successfully
- Performed threshold sweep analysis
- Started HPO but stopped due to "-inf AUC" bug
- Identified 3 bugs in HPO scripts

### 2026-01-25 16:00 (Runner Script Fixes)
- Fixed broken runner scripts
