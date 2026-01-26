# Workstream 2 Context: foundation
# Last Updated: 2026-01-25 23:55

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model investigation - testing pre-trained time series models
- **Status**: active

---

## Current Task
- **Working on**: TimesFM a50/a100 covariate experiments (TFM-07 through TFM-10)
- **Status**: NOTEBOOKS CREATED - Ready for Colab execution

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
| TFM-01 Inverted | 0.636 | 0.33 | 0.49 | Below PatchTST |

### Key Finding: TFM-01 NOT Fair Comparison
- TFM-01 was zero-shot with ONLY 1 feature (close price)
- PatchTST was trained on 20 features
- Need to test TimesFM with covariates for fair comparison

---

## Last Session Work (2026-01-25 23:55)

### Created a50/a100 Covariate Experiment Notebooks
Created two Colab notebooks for testing TimesFM with engineered features:

| Notebook | Experiments | Features | Data File |
|----------|-------------|----------|-----------|
| `TimesFM_a50_Experiments.ipynb` | TFM-07, TFM-08 | 50 | `SPY_dataset_a50_combined.parquet` |
| `TimesFM_a100_Experiments.ipynb` | TFM-09, TFM-10 | 100 | `SPY_dataset_a100_combined.parquet` |

### Experiment Matrix
| ID | Features | Mode | Description |
|----|----------|------|-------------|
| TFM-07 | 50 (a50) | Zero-shot | Covariates with 50 features |
| TFM-08 | 50 (a50) | Fine-tuned | Train on 1% threshold with 50 features |
| TFM-09 | 100 (a100) | Zero-shot | Covariates with 100 features |
| TFM-10 | 100 (a100) | Fine-tuned | Train on 1% threshold with 100 features |

### Bug Fix: Wrong File Uploaded
User initially uploaded `SPY_dataset_a50.parquet` (features only, no OHLCV) instead of `SPY_dataset_a50_combined.parquet` (OHLCV + features).

**Fixed by**: Added clear validation in Cell 2:
- Shows warning if filename doesn't contain "_combined"
- Lists all columns for diagnostics
- Clear error message explaining which file to use

### Files Created This Session
- `experiments/foundation/TimesFM_a50_Experiments.ipynb` - NEW
- `experiments/foundation/TimesFM_a100_Experiments.ipynb` - NEW

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
  - `TimesFM_SPY_Experiments.ipynb` - Original TFM-01 notebook
  - `TimesFM_a50_Experiments.ipynb` - **NEW** (TFM-07, TFM-08)
  - `TimesFM_a100_Experiments.ipynb` - **NEW** (TFM-09, TFM-10)
  - `analyze_timesfm_thresholds.py` - Local threshold analysis script
- `outputs/foundation/` - Results storage
  - `timesfm_tfm-01_results.json` - TFM-01 results
  - `timesfm_tfm-01_comprehensive.json` - TFM-01 full analysis
  - `timesfm_tfm-01_predictions.npz` - Raw predictions

---

## Key Decisions (Workstream-Specific)

### Data File Naming Clarification (2026-01-25)
- **Two versions exist**: `SPY_dataset_a50.parquet` vs `SPY_dataset_a50_combined.parquet`
- **Without "_combined"**: Features ONLY (no OHLCV columns)
- **With "_combined"**: OHLCV + Features (needed for experiments)
- **Lesson**: Always validate column presence before proceeding

### TFM-01 Not Fair Comparison (2026-01-25)
- TFM-01 used 1 feature (close), PatchTST used 20
- Created TFM-07 through TFM-10 for fair comparison with covariates

---

## Session History

### 2026-01-25 23:55
- Created TimesFM_a50_Experiments.ipynb and TimesFM_a100_Experiments.ipynb
- Fixed Cell 2 data upload with better validation and error messages
- Diagnosed wrong file upload issue (non-combined vs combined parquet)

### 2026-01-25 19:45
- Fixed cell-27 JSON serialization bug (numpy bool → Python bool)

### 2026-01-25 09:30
- Analyzed TFM-01 results - discovered anti-correlation

### 2026-01-24
- Completed Lag-Llama investigation, created TimesFM notebook

---

## Next Session Should

### Priority 1: Run TFM-07 in Colab
1. Upload `SPY_dataset_a50_combined.parquet` (note: _combined!)
2. Run all cells
3. Check if anti-correlation persists with 50 features
4. Download results

### Priority 2: Run Remaining Experiments
Order: TFM-07 → TFM-08 → TFM-09 → TFM-10

### Priority 3: Analyze Results
- Does adding features help?
- Does fine-tuning fix anti-correlation?
- Can TimesFM beat PatchTST with proper features?

### Key Question
Does adding 50-100 engineered features help TimesFM overcome the anti-correlation issue seen in TFM-01 (1 feature)?

---

## Data Files for Upload

| File | Size | Location |
|------|------|----------|
| `SPY_dataset_a50_combined.parquet` | 4.0 MB | `data/processed/v1/` |
| `SPY_dataset_a100_combined.parquet` | 7.2 MB | `data/processed/v1/` |

**IMPORTANT**: Upload the `_combined` versions (have OHLCV + features)

---

## Memory Entities (Workstream-Specific)
- None created this session
