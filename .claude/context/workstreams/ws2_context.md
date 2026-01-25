# Workstream 2 Context: foundation
# Last Updated: 2026-01-25 09:30

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model investigation - testing pre-trained time series models
- **Status**: active

---

## Current Task
- **Working on**: TimesFM notebook restructuring for proper threshold analysis
- **Status**: IN PROGRESS - notebook cells need reordering

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

## Last Session Work (2026-01-25)

### TFM-01 Ran in Colab - Results Analyzed
1. User ran notebook in Colab, downloaded `timesfm_tfm-01_results.json`
2. Discovered issues:
   - Classification threshold (1%) was higher than any prediction
   - Model produced 0 positive predictions
   - AUC 0.364 indicates anti-correlation

### Notebook Restructuring (IN PROGRESS)
User requested proper threshold sweeping and anti-correlation analysis. Changes made:

**Added new cells:**
- Cell 13: Threshold sweep on predicted returns (sweep prediction confidence levels)
- Cell 14: Anti-correlation analysis (test inverted predictions)
- Cell 16: Comprehensive export (gather ALL data at end)

**Issues to fix next session:**
1. **Threshold sweep code is MISSING** - Cell 24 has header but Cell 25 jumped to anti-correlation
2. **Cell order is wrong** - Comprehensive export was in middle, needs to be LAST
3. **Stubs to clean up** - Some placeholder cells remain

### Files Modified This Session
- `experiments/foundation/TimesFM_SPY_Experiments.ipynb` - PARTIAL (needs fixing)
- `experiments/foundation/analyze_timesfm_thresholds.py` - NEW (local analysis script)
- `outputs/foundation/timesfm_tfm-01_results.json` - Downloaded from Colab

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
  - `TimesFM_SPY_Experiments.ipynb` - **NEEDS FIXING** (cell order wrong)
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

### Notebook Restructure Requirements (2026-01-25)
User specified:
1. Threshold sweep on PREDICTED returns (not class threshold)
2. Anti-correlation analysis with inverted predictions
3. Comprehensive export at VERY END gathering ALL data
4. Remove unnecessary appendix (no data output)

---

## Session History

### 2026-01-25 (Current - Interrupted)
- Analyzed TFM-01 results - discovered anti-correlation
- Started notebook restructuring
- **INTERRUPTED** - cells out of order, needs completion

### 2026-01-24 17:30
- Fixed TimesFM Colab notebook for API v2.5
- Model: TimesFM-2.0-500M → TimesFM-2.5-200M

### 2026-01-24 09:00
- Completed Lag-Llama investigation (FAILED)
- Created TimesFM Colab notebook

---

## Next Session Should

### Priority 1: Fix Notebook Cell Order
Current state (broken):
```
Cell 24: Threshold sweep header (Cell 13)
Cell 25: Anti-correlation code (WRONG - should be threshold sweep code!)
Cell 26: Export code (old export, should be comprehensive at end)
Cell 27: Anti-correlation header (Cell 14)
```

Correct order needed:
1. Cell 13 header → Threshold sweep CODE
2. Cell 14 header → Anti-correlation CODE
3. Cell 15/16 header → Comprehensive export CODE (LAST)

### Priority 2: Add Missing Threshold Sweep Code
The sweep function exists but execution code is missing. Need:
```python
# Sweep thresholds spanning prediction range
thresholds = np.linspace(pred_min, pred_max, 15)
results = sweep_prediction_thresholds(val_preds, val_labels, thresholds)
# Print table, find best F1/precision thresholds
```

### Priority 3: Re-run in Colab
After fixing notebook:
1. Run Cells 1-6 (setup, inference)
2. Run Cell 13 (threshold sweep)
3. Run Cell 14 (anti-correlation)
4. Run Cell 16 (comprehensive export)
5. Download results

### Expected Output Files
- `timesfm_tfm-01_comprehensive.json` - All data in one file
- `timesfm_tfm-01_predictions.npz` - Raw predictions for local analysis

---

## Memory Entities (Workstream-Specific)
- No Memory entities created specifically for this workstream yet
