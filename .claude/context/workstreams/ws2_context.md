# Workstream 2 Context: foundation
# Last Updated: 2026-01-25 19:45

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model investigation - testing pre-trained time series models
- **Status**: active

---

## Current Task
- **Working on**: TimesFM notebook - comprehensive export cell (TFM-01)
- **Status**: READY TO RE-RUN IN COLAB

---

## Investigation Summary

### Research Question
Can pre-trained foundation models (Lag-Llama, TimesFM) beat task-specific PatchTST on SPY direction prediction?

### Results So Far

| Experiment | AUC | Precision | Recall | Status |
|------------|-----|-----------|--------|--------|
| **PatchTST H1** | **0.718** | 0.58 | 0.45 | BASELINE |
| Lag-Llama (all modes) | 0.499-0.576 | ~0.50 | ~0.50 | FAILED |
| TimesFM TFM-01 | **0.364** | 0.0 | 0.0 | **ANTI-CORRELATED** |

### TimesFM TFM-01 Results (2026-01-25)
**Critical Finding**: Model is ANTI-CORRELATED
- Val AUC: 0.364 (< 0.5 = worse than random)
- Inverted AUC: 0.636 (still below PatchTST 0.718)
- 0 positive predictions on val (threshold 1% too high for predictions)
- Prediction range: [-0.011, +0.006] (never reaches 1%)

---

## Last Session Work (2026-01-25 19:45)

### Fixed JSON Serialization Bug in cell-27
**Problem**: Comprehensive export cell failed with:
```
TypeError: Object of type bool is not JSON serializable
```

**Root Cause**: `is_anti_correlated = auc_normal < 0.5` returns `np.bool_`, not Python `bool`

**Fix Applied**: Changed line in cell-27:
```python
# Before:
"is_anti_correlated": is_anti_correlated

# After:
"is_anti_correlated": bool(is_anti_correlated)  # Cast numpy bool to Python bool
```

### Files Modified This Session
- `experiments/foundation/TimesFM_SPY_Experiments.ipynb` - cell-27 fixed

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
  - `TimesFM_SPY_Experiments.ipynb` - **READY** (JSON bug fixed)
  - `analyze_timesfm_thresholds.py` - Local threshold analysis script
  - `train_lagllama_h1_forecast.py` - Lag-Llama experiment script
- `outputs/foundation/` - Results storage
  - `timesfm_tfm-01_results.json` - TFM-01 results from Colab

---

## Key Decisions (Workstream-Specific)

### TimesFM Anti-Correlation Discovery (2026-01-25)
- **Finding**: TimesFM predicts OPPOSITE of correct direction
- **Evidence**: AUC 0.364 < 0.5
- **Implication**: Even inverted (AUC 0.636), still below PatchTST (0.718)
- **Next step**: Complete threshold sweep to fully characterize behavior

### JSON Serialization Fix (2026-01-25 19:45)
- **Problem**: numpy bool not JSON serializable
- **Solution**: Cast with `bool()` in comprehensive export dict

---

## Session History

### 2026-01-25 19:45
- Fixed cell-27 JSON serialization bug (numpy bool → Python bool)
- Notebook now ready to re-run in Colab

### 2026-01-25 09:30
- Analyzed TFM-01 results - discovered anti-correlation
- Restructured notebook with threshold sweep and anti-correlation cells

### 2026-01-24 17:30
- Fixed TimesFM Colab notebook for API v2.5
- Model: TimesFM-2.0-500M → TimesFM-2.5-200M

### 2026-01-24 09:00
- Completed Lag-Llama investigation (FAILED)
- Created TimesFM Colab notebook

---

## Next Session Should

### Priority 1: Re-run Notebook in Colab
Now that JSON bug is fixed:
1. Run Cells 1-6 (setup, inference)
2. Run Cell 25 (threshold sweep)
3. Run Cell 26 (anti-correlation analysis)
4. Run Cell 27 (comprehensive export) - **NOW FIXED**
5. Download `timesfm_tfm-01_comprehensive.json`

### Priority 2: Analyze Comprehensive Results
- Review threshold sweep results
- Confirm anti-correlation behavior
- Determine if any threshold gives useful precision/recall

### Priority 3: Decision on TFM-02/03/etc.
Based on TFM-01 comprehensive results:
- If anti-correlated behavior persists → may skip remaining TimesFM experiments
- If inverted predictions useful → document as finding
- Move to next foundation model or conclude investigation

### Expected Output Files
- `timesfm_tfm-01_comprehensive.json` - All data in one file
- `timesfm_tfm-01_predictions.npz` - Raw predictions for local analysis

---

## Memory Entities (Workstream-Specific)
- No Memory entities created specifically for this workstream yet
