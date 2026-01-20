# Phase 6A Critical Finding: Feature Normalization Bug

**Date Discovered**: 2026-01-20
**Severity**: Critical - invalidates all Phase 6A model evaluations on recent data
**Status**: Root cause identified, solutions proposed

## Executive Summary

All Phase 6A models appeared to show "prior collapse" (outputting near-constant ~0.52 predictions on 2024-2025 test data). Investigation revealed this is NOT a model architecture or loss function problem - it's a **data preprocessing bug**. Features are not normalized, causing massive distribution shift between training (1994-2016) and test (2024-2026) periods.

## Discovery Process

### Initial Symptoms
- Models output predictions in range [0.518, 0.524] on 2025 test data
- All model sizes (2M to 2B) showed identical behavior
- SoftAUCLoss didn't fix the problem
- AUC-ROC was 0.53-0.65 (barely better than random)

### Key Investigation Steps

1. **Checked class prior**: 14.2% positive rate, not 52% - so this isn't "prior collapse"
2. **Checked val set predictions**: Model predicts ~0.09 on val data (2016-2021), not 0.52
3. **Checked date ranges**: Train=1994-2016, Val=2016-2021, Test=2021-2026
4. **Checked feature distributions**: MASSIVE shift discovered

## Root Cause: Unnormalized Features

### Feature Distribution Shift

| Feature | Train (1994-2016) | Val (2017-2021) | Recent (2024-2026) | Shift |
|---------|-------------------|-----------------|---------------------|-------|
| Close | 88.57 | 283.75 | 575.89 | **6.5x** |
| Volume | 87.5M | 82.4M | 65.1M | 0.7x |
| OBV | 4.0B | 15.7B | 16.6B | **4x** |
| ATR | 1.23 | 3.41 | 6.64 | **5x** |
| MACD | 0.20 | 1.37 | 3.23 | **16x** |
| RSI | 54.41 | 58.32 | 58.32 | ~1x |

### Why RSI is Stable
RSI is naturally bounded (0-100) by construction. It's the only feature that doesn't show distribution shift - confirming normalization is the issue.

### What Actually Happened
1. Model trained on data where Close ≈ 88, learned valid patterns
2. Model sees Close ≈ 576 at inference - completely out of distribution
3. Transformer outputs default/confused predictions around sigmoid midpoint (0.5)
4. This happens regardless of model size because the INPUT is broken

## Evidence

### Val Set Predictions (2016-2021 data)
```
Sample predictions: 0.077 - 0.111 (mean 0.0895)
Val loss: 0.203 (better than random 0.409)
Model correctly predicts LOW for mostly-negative samples
```

### Test Set Predictions (2024-2026 data)
```
Sample predictions: 0.518 - 0.524 (mean 0.52)
Near-constant output around sigmoid midpoint
Model outputs "I don't know" for out-of-distribution inputs
```

## Implications

### What This Means
1. **Models DID learn** - val_loss=0.203 proves learning occurred on training distribution
2. **Models DON'T generalize** - due to preprocessing bug, not architecture
3. **All Phase 6A scaling conclusions are suspect** - we were measuring preprocessing failure
4. **The 19-sample val set was a distraction** - real bug was in feature engineering

### What This Doesn't Change
- The experimental infrastructure is sound
- The training pipeline works correctly
- The model architecture is likely fine
- The scaling law research question remains valid

## Proposed Solutions

### Option A: Z-Score Normalization (Recommended First)
```python
# Compute statistics on training data only
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Apply to all data
X_normalized = (X - train_mean) / (train_std + epsilon)
```
**Pros**: Simple, standard practice, preserves relative relationships
**Cons**: Requires storing/loading normalization parameters

### Option B: Percent Change Features
```python
# Use returns instead of prices
returns = close.pct_change()
```
**Pros**: Naturally stationary, no parameters to store
**Cons**: Loses absolute level information, different for each feature type

### Option C: Rolling Window Normalization
```python
# Normalize within each context window
X_window_normalized = (X_window - X_window.mean()) / X_window.std()
```
**Pros**: Adapts to local distribution, no train/test split leakage
**Cons**: Different normalization for each sample, harder to interpret

### Option D: Use Only Bounded Features
Replace raw prices/volumes with:
- RSI, Stochastic RSI (0-100)
- Bollinger Band %B (0-1)
- Percentile ranks (0-100)
- Z-scores relative to rolling window

**Pros**: Naturally bounded, no normalization needed
**Cons**: Loses some information, requires feature re-engineering

### Option E: Hybrid Approach
- **Prices**: Use percent changes or log returns
- **Volumes**: Use percent change or log
- **Indicators**: Use z-scores relative to rolling window
- **Oscillators**: Keep as-is (already bounded)

**Pros**: Best of all worlds
**Cons**: More complex implementation

## Recommended Path Forward

1. **Immediate**: Implement Option A (Z-score) as quickest fix
2. **Validate**: Re-run one model (2M_h1) with normalized features
3. **Compare**: Check if predictions are now reasonable on 2025 data
4. **If successful**: Re-run all Phase 6A experiments with normalization
5. **Long-term**: Consider Option E (hybrid) for production

## Implementation Notes

### Where to Add Normalization
- `src/features/tier_a20.py` - compute and save stats
- `src/data/dataset.py` - apply normalization in FinancialDataset
- Or create new `src/features/normalize.py` module

### Backward Compatibility
- Existing datasets are invalid - need to regenerate
- Old checkpoints won't work with new normalized data
- Document version bump required

## Files to Update
- [ ] `src/features/tier_a20.py` - add normalization
- [ ] `src/data/dataset.py` - apply normalization
- [ ] `scripts/build_features_a20.py` - save norm params
- [ ] Feature pipeline tests
- [ ] Regenerate SPY_dataset_a20.parquet
- [ ] Re-run Phase 6A experiments

---

**Lesson Learned**: Always check feature distributions across time periods before training time-series models. This is ML 101 but easy to overlook when focused on architecture and loss functions.
