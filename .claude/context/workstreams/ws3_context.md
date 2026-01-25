# Workstream 3 Context: Phase 6C Experiments
# Last Updated: 2026-01-25 23:50

## Identity
- **ID**: ws3
- **Name**: phase6c
- **Focus**: Phase 6C feature scaling experiments (a50 → a100 tier)
- **Status**: **NEEDS TROUBLESHOOTING** - Scripts created but not working

## Current Task
- **Working on**: Phase 6C a100 experimentation pipeline
- **Status**: ❌ BLOCKED - Scripts created but `run_s1_a100.sh` not executing

---

## CRITICAL: Issues to Troubleshoot Next Session

### Problem Summary
Created full experimentation pipeline for a100 tier, but **runner script `run_s1_a100.sh` does nothing when executed**.

### Files Created This Session (Need Verification)
All in `experiments/phase6c_a100/`:
- 12 baseline scripts: `s1_01_2m_h1.py` through `s1_12_200m_h5.py`
- 6 HPO scripts: `hpo_2m_h1.py`, `hpo_20m_h1.py`, etc.
- 6 architecture variation scripts: `s2_arch_*.py`, `s2_train_*.py`
- `threshold_sweep.py`
- `compare_all_tiers.py`
- `statistical_validation.py`

Runners:
- `scripts/run_s1_a100.sh` - **NOT WORKING** (does nothing)
- `scripts/run_hpo_a100.sh` - Untested

Data:
- `data/processed/v1/SPY_dataset_a100.parquet` - Created, 8022 rows
- `data/processed/v1/SPY_dataset_a100_combined.parquet` - Created, 106 columns

### Troubleshooting Steps for Next Session
1. **Test individual experiment script directly**:
   ```bash
   ./venv/bin/python experiments/phase6c_a100/s1_01_2m_h1.py
   ```
   Check if it produces output or errors.

2. **Check runner script syntax**:
   ```bash
   bash -x scripts/run_s1_a100.sh
   ```
   Trace execution to see where it fails.

3. **Verify data paths in scripts**:
   - Scripts reference `SPY_dataset_a100_combined.parquet`
   - Confirm this file exists and has correct structure

4. **Check experiment script imports**:
   - May have import errors not caught by syntax check

### Sanity Test Results (Passed)
- `make test` passes (692 passed)
- `py_compile` on all scripts passed
- Quick setup test loaded data and created trainer successfully
- Model parameters: 309,651 (0.31M) for 2M config

---

## Progress Summary

### Completed This Session (2026-01-25)
- ✅ Created `scripts/build_features_a100.py` - Feature builder
- ✅ Built `SPY_dataset_a100.parquet` (8,022 rows × 100 features)
- ✅ Built `SPY_dataset_a100_combined.parquet` (with OHLCV for targets)
- ✅ Registered a100 in processed manifest
- ✅ Created 12 baseline experiment scripts (S1)
- ✅ Created 6 HPO scripts
- ✅ Created 6 architecture variation scripts (S2)
- ✅ Created threshold_sweep.py
- ✅ Created compare_all_tiers.py
- ✅ Created statistical_validation.py
- ✅ Created runner scripts (run_s1_a100.sh, run_hpo_a100.sh)
- ✅ All scripts pass syntax check
- ✅ Setup sanity test passed

### NOT Working
- ❌ `run_s1_a100.sh` - Does nothing when executed
- ⚠️ Individual experiment scripts - Untested at runtime

### Previous Session Completions
- Statistical validation of a20 vs a50 comparison
- Bootstrap CI: 0/12 AUC differences statistically significant
- Test set: 17% consistency (pattern NOT replicated)
- Conclusion: NULL finding - proceed to a100 tier

---

## Files Created/Modified This Session

### New Files (experiments/phase6c_a100/)
```
s1_01_2m_h1.py    s1_07_2m_h3.py    s2_arch_wide.py
s1_02_20m_h1.py   s1_08_20m_h3.py   s2_arch_deep.py
s1_03_200m_h1.py  s1_09_200m_h3.py  s2_arch_heads16.py
s1_04_2m_h2.py    s1_10_2m_h5.py    s2_train_drop03.py
s1_05_20m_h2.py   s1_11_20m_h5.py   s2_train_drop07.py
s1_06_200m_h2.py  s1_12_200m_h5.py  s2_train_lr5e5.py
hpo_2m_h1.py      hpo_2m_h5.py      threshold_sweep.py
hpo_20m_h1.py     hpo_20m_h5.py     compare_all_tiers.py
hpo_200m_h1.py    hpo_200m_h5.py    statistical_validation.py
```

### New Files (scripts/)
- `build_features_a100.py`
- `run_s1_a100.sh` (NOT WORKING)
- `run_hpo_a100.sh` (untested)

### New Files (data/)
- `data/processed/v1/SPY_dataset_a100.parquet` (6.9 MB)
- `data/processed/v1/SPY_dataset_a100_combined.parquet` (7.2 MB)

---

## Key Decisions
- Scripts use string concatenation for print statements (not f-strings) to avoid escaping issues
- Data paths are absolute using PROJECT_ROOT pattern
- Batch sizes: 128 (2M), 64 (20M), 32 (200M)

---

## Next Session Should

1. **FIRST: Debug why `run_s1_a100.sh` does nothing**
   - Run with `bash -x` to trace
   - Test individual Python scripts directly

2. **Verify one experiment runs end-to-end**:
   ```bash
   ./venv/bin/python experiments/phase6c_a100/s1_01_2m_h1.py
   ```

3. **If scripts work individually, fix runner script**

4. **Then run baseline experiments** (12 models)

5. **Continue with plan phases 3-8**

---

## Memory Entities
- None created (context in this file)

---

## Session History

### 2026-01-25 23:50 (a100 Pipeline Creation - INCOMPLETE)
- Created full a100 experimentation pipeline
- All scripts created and pass syntax check
- Data files created and validated
- **BUT runner script does nothing when executed**
- Session ended with troubleshooting needed
- `make test` passes (692 passed)
