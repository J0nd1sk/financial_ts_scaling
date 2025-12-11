# Feature Pipeline Integration Issues and Design Fixes

**Date:** 2025-12-11
**Phase:** Phase 6A (Parameter Scaling)

## Issues Discovered

### Issue 1: Non-Numeric Feature Column (`vix_regime`)

**Problem:** The `vix_regime` column was stored as strings ("low", "normal", "high") instead of numeric values. This caused runtime failures when `FinancialDataset` tried to convert features to float32.

**Error:**
```
ValueError: could not convert string to float: 'low'
```

**Root Cause:** The VIX feature engineering (`src/features/tier_c_vix.py`) created categorical string values without encoding them numerically.

**Fix Applied:**
- Modified `_classify_regime()` to return integers: 0=low, 1=normal, 2=high
- Re-processed VIX features and SPY_dataset_c

### Issue 2: Dataset Auto-Discovery Fragility

**Problem:** `FinancialDataset` used a fragile column discovery approach:
```python
feature_cols = [c for c in features_df.columns if c != "Date"]
```

This included:
- Any non-numeric columns - caused crashes (e.g., string-based vix_regime)

**Note:** OHLCV columns (Open, High, Low, Close, Volume) are CORE data and MUST be included in training. Indicators/features are additional and will expand during feature scaling tests.

**Fix Applied:** Added defensive feature discovery with explicit filtering:
1. Only exclude Date column (OHLCV is core training data)
2. Filter to only numeric dtypes
3. Warn when non-numeric columns are skipped
4. Support optional explicit `feature_columns` parameter

### Issue 3: Disconnected Feature Specification

**Problem:** Generated experiment scripts defined `FEATURE_COLUMNS` but this list was:
- Only used for validation in the script
- Not passed through to the training pipeline
- The Trainer/Dataset ignored it and auto-discovered columns

**Impact:** The script's feature list and actual training features were not connected.

## Design Recommendations

### Current Architecture (Fragile)

```
Script defines FEATURE_COLUMNS → used only for validation
Config file specifies data_path → but not which columns to use
Trainer loads ALL columns → passes whole DataFrame to Dataset
Dataset auto-discovers features → excludes only 'Date', fails on non-numeric
```

### Recommended Architecture: Explicit Config-Driven

```
Config file specifies:
  - data_path
  - feature_columns (explicit list)

Trainer:
  - Reads feature_columns from config
  - Validates columns exist and are numeric
  - Passes explicit list to Dataset

Dataset:
  - Accepts explicit feature_columns parameter
  - Validates requested columns
  - Uses only specified columns
```

**Why Explicit is Better:**
1. **Reproducibility:** Config captures exact experiment setup
2. **Documentation:** Config serves as self-documenting record
3. **Flexibility:** Easy to experiment with feature subsets
4. **Validation:** Mismatches caught early
5. **No Surprises:** What you specify is what you get

### Data Pipeline Principles

1. **Data should be "model-ready" after processing**
   - All feature columns must be numeric
   - Categorical features encoded during processing (not at training time)
   - Boolean features stored as 0/1 integers

2. **Feature encoding conventions:**
   - Ordinal categories: Integer encoding (0, 1, 2, ...)
   - Binary flags: Integer (0 or 1)
   - Multi-class categories: One-hot or integer encoding
   - Never store as strings or Python bool

3. **Defense in depth:**
   - Data pipeline ensures numeric output
   - Config explicitly lists features
   - Dataset validates columns

## Files Modified

| File | Change |
|------|--------|
| `src/data/dataset.py` | Added defensive feature discovery, explicit feature_columns param |
| `src/features/tier_c_vix.py` | Changed vix_regime from strings to integers (0/1/2) |
| `tests/test_training.py` | Updated fixture feature count (Trainer auto-adjusts to actual data) |
| `tests/features/test_vix_features.py` | Updated regime tests to expect integers |
| `data/processed/v1/VIX_features_c.parquet` | Re-processed with numeric vix_regime |
| `data/processed/v1/SPY_dataset_c.parquet` | Re-processed with numeric vix_regime |

## Future Work

1. **Update experiment config schema** to include `feature_columns` list
2. **Update Trainer** to read feature_columns from config
3. **Update experiment_generation skill** to:
   - Verify config file exists (create if missing)
   - Filter non-numeric columns during generation
   - Pass feature list through to training pipeline
4. **Add config validation** to catch issues before runtime

## Verification

After fixes, all 239 tests pass and HPO can run without feature conversion errors.

```bash
make test  # 239 passed
./venv/bin/python3 -c "import pandas as pd; df = pd.read_parquet('data/processed/v1/SPY_dataset_c.parquet'); print(df.dtypes)"
# All columns now numeric
```
