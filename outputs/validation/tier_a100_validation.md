# Tier A100 Validation Report

**Generated**: 2026-01-25T14:50:12.294532

## Summary

- **Total Checks**: 69
- **Passed**: 69
- **Failed**: 0
- **Pass Rate**: 100.0%

## Results by Chunk

| Chunk | Passed | Failed | Total |
|-------|--------|--------|-------|
| ✅ Chunk 1: Momentum derivatives | 10 | 0 | 10 |
| ✅ Chunk 2: QQE/STC derivatives | 4 | 0 | 4 |
| ✅ Chunk 3: Standard oscillators | 18 | 0 | 18 |
| ✅ Chunk 4: VRP + Risk metrics | 6 | 0 | 6 |
| ✅ Chunk 5: MA extensions | 9 | 0 | 9 |
| ✅ Chunk 6: Advanced volatility | 6 | 0 | 6 |
| ✅ Chunk 7: Trend indicators | 6 | 0 | 6 |
| ✅ Chunk 8: Volume + Momentum + S/R | 7 | 0 | 7 |

## Detailed Results

### ✅ `data_quality`

- ✅ **columns_present**
  - Expected: `All 50 columns`
  - Actual: `All 50 columns present`
  - Evidence: Column check passed

- ✅ **no_nan_values**
  - Expected: `No NaN in any column`
  - Actual: `No NaN values`
  - Evidence: NaN check passed

- ✅ **no_inf_values**
  - Expected: `No Inf in any column`
  - Actual: `No Inf values`
  - Evidence: Infinity check passed

### ✅ `return_1d_acceleration`

- ✅ **formula_idx_8002**
  - Expected: `-0.173280`
  - Actual: `-0.173280`
  - Evidence: Hand-calc at idx 8002: diff=0.00e+00

- ✅ **formula_idx_4131**
  - Expected: `-1.346791`
  - Actual: `-1.346791`
  - Evidence: Hand-calc at idx 4131: diff=0.00e+00

- ✅ **formula_idx_3241**
  - Expected: `0.621480`
  - Actual: `0.621480`
  - Evidence: Hand-calc at idx 3241: diff=0.00e+00

- ✅ **formula_idx_2117**
  - Expected: `-2.768230`
  - Actual: `-2.768230`
  - Evidence: Hand-calc at idx 2117: diff=0.00e+00

- ✅ **formula_idx_3005**
  - Expected: `2.302092`
  - Actual: `2.302092`
  - Evidence: Hand-calc at idx 3005: diff=0.00e+00

### ✅ `return_5d_acceleration`

- ✅ **formula_idx_8002**
  - Expected: `-0.293233`
  - Actual: `-0.293233`
  - Evidence: Hand-calc at idx 8002: diff=0.00e+00

- ✅ **formula_idx_4131**
  - Expected: `-1.779129`
  - Actual: `-1.779129`
  - Evidence: Hand-calc at idx 4131: diff=0.00e+00

- ✅ **formula_idx_3241**
  - Expected: `0.407495`
  - Actual: `0.407495`
  - Evidence: Hand-calc at idx 3241: diff=0.00e+00

- ✅ **formula_idx_2117**
  - Expected: `-2.920294`
  - Actual: `-2.920294`
  - Evidence: Hand-calc at idx 2117: diff=0.00e+00

- ✅ **formula_idx_3005**
  - Expected: `1.673372`
  - Actual: `1.673372`
  - Evidence: Hand-calc at idx 3005: diff=0.00e+00

### ✅ `qqe_extreme_dist`

- ✅ **formula_verification**
  - Expected: `min(|qqe-20|, |qqe-80|)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Formula verified against qqe_fast

### ✅ `stc_extreme_dist`

- ✅ **formula_verification**
  - Expected: `min(|stc-25|, |stc-75|)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Formula verified against stc_value

### ✅ `qqe_slope`

- ✅ **formula_verification**
  - Expected: `qqe_fast - qqe_fast.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: 5-day change formula verified

### ✅ `stc_slope`

- ✅ **formula_verification**
  - Expected: `stc_value - stc_value.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: 5-day change formula verified

### ✅ `demarker_value`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-05`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

- ✅ **hand_calc_idx_5508**
  - Expected: `0.160606`
  - Actual: `0.160606`
  - Evidence: DeMarker hand-calc at idx 5508

- ✅ **hand_calc_idx_2597**
  - Expected: `0.360544`
  - Actual: `0.360544`
  - Evidence: DeMarker hand-calc at idx 2597

- ✅ **hand_calc_idx_3712**
  - Expected: `0.256774`
  - Actual: `0.256774`
  - Evidence: DeMarker hand-calc at idx 3712

- ✅ **hand_calc_idx_7486**
  - Expected: `0.421332`
  - Actual: `0.421332`
  - Evidence: DeMarker hand-calc at idx 7486

- ✅ **hand_calc_idx_7584**
  - Expected: `0.580055`
  - Actual: `0.580055`
  - Evidence: DeMarker hand-calc at idx 7584

- ✅ **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.02, 1.00]`
  - Evidence: Range within expected bounds

### ✅ `demarker_from_half`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-05`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

### ✅ `stoch_k_14`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

- ✅ **range_check**
  - Expected: `[0, 100]`
  - Actual: `[0.85, 100.00]`
  - Evidence: Range within expected bounds

### ✅ `stoch_d_14`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

- ✅ **range_check**
  - Expected: `[0, 100]`
  - Actual: `[1.26, 98.94]`
  - Evidence: Range within expected bounds

### ✅ `stoch_extreme_dist`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

### ✅ `cci_14`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

### ✅ `mfi_14`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

- ✅ **range_check**
  - Expected: `[0, 100]`
  - Actual: `[9.85, 97.82]`
  - Evidence: Range within expected bounds

### ✅ `williams_r_14`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-06`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib/formula reference match

- ✅ **range_check**
  - Expected: `[-100, 0]`
  - Actual: `[-100.00, 0.00]`
  - Evidence: Range within expected bounds

### ✅ `vrp_5d`

- ✅ **formula_verification**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: VRP formula verified

### ✅ `vrp_slope`

- ✅ **formula_verification**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: VRP formula verified

### ✅ `var_99`

- ✅ **ordering_constraint**
  - Expected: `var_99 <= var_95 (more extreme)`
  - Actual: `0 violations`
  - Evidence: VaR ordering verified: var_99 always <= var_95

### ✅ `cvar_95`

- ✅ **ordering_constraint**
  - Expected: `cvar_95 <= var_95`
  - Actual: `0 violations`
  - Evidence: CVaR ordering verified

### ✅ `sharpe_252d`

- ✅ **range_check**
  - Expected: `[-20, 20]`
  - Actual: `[-1.97, 4.22]`
  - Evidence: Range within expected bounds

### ✅ `sortino_252d`

- ✅ **range_check**
  - Expected: `[-20, 20]`
  - Actual: `[-2.58, 7.20]`
  - Evidence: Range within expected bounds

### ✅ `sma_9_50_proximity`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `sma_50_slope`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `sma_200_slope`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `ema_12`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `ema_26`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `days_since_sma_50_cross`

- ✅ **counter_reset**
  - Expected: `days=0 at all crosses`
  - Actual: `552/552 zeros`
  - Evidence: Counter resets correctly at crosses

- ✅ **counter_increment**
  - Expected: `monotonically increasing`
  - Actual: `[30, 31, 32, 33, 34, 35]`
  - Evidence: Counter generally increases after cross

### ✅ `days_since_sma_200_cross`

- ✅ **counter_reset**
  - Expected: `days=0 at all crosses`
  - Actual: `212/212 zeros`
  - Evidence: Counter resets correctly at crosses

- ✅ **counter_increment**
  - Expected: `monotonically increasing`
  - Actual: `[54, 55, 56, 57, 58, 59]`
  - Evidence: Counter generally increases after cross

### ✅ `parkinson_volatility`

- ✅ **formula_verification**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: parkinson_volatility formula verified (textbook formula)

- ✅ **range_check**
  - Expected: `[0, 150]`
  - Actual: `[3.63, 75.76]`
  - Evidence: Volatility range reasonable

### ✅ `garman_klass_volatility`

- ✅ **formula_verification**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: garman_klass_volatility formula verified (textbook formula)

- ✅ **range_check**
  - Expected: `[0, 150]`
  - Actual: `[3.79, 77.75]`
  - Evidence: Volatility range reasonable

### ✅ `atr_pct_percentile_60d`

- ✅ **range_check**
  - Expected: `[0, 100]`
  - Actual: `[1.67, 100.00]`
  - Evidence: Percentile range valid

### ✅ `bb_width_percentile_60d`

- ✅ **range_check**
  - Expected: `[0, 100]`
  - Actual: `[1.67, 100.00]`
  - Evidence: Percentile range valid

### ✅ `adx_slope`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `di_spread`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `aroon_oscillator`

- ✅ **reference_comparison**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Talib reference match

### ✅ `supertrend_direction`

- ✅ **bullish_consistency**
  - Expected: `price above ST when bullish (>=95%)`
  - Actual: `100.0%`
  - Evidence: 4975/4975 consistent

- ✅ **bearish_consistency**
  - Expected: `price below ST when bearish (>=95%)`
  - Actual: `100.0%`
  - Evidence: 3047/3047 consistent

- ✅ **value_range**
  - Expected: `{+1, -1}`
  - Actual: `{1.0, -1.0}`
  - Evidence: Direction values are valid

### ✅ `buying_pressure_ratio`

- ✅ **boundary_close_eq_high**
  - Expected: `bp=1 when Close=High`
  - Actual: `214/214`
  - Evidence: Boundary condition verified

- ✅ **boundary_close_eq_low**
  - Expected: `bp=0 when Close=Low`
  - Actual: `124/124`
  - Evidence: Boundary condition verified

- ✅ **range_check**
  - Expected: `[0, 1]`
  - Actual: `[-0.0000, 1.0000]`
  - Evidence: Range valid

### ✅ `fib_range_position`

- ✅ **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Range valid

### ✅ `win_rate_20d`

- ✅ **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.2000, 0.9000]`
  - Evidence: Range valid

### ✅ `prior_high_20d_dist`

- ✅ **sign_constraint**
  - Expected: `always <= 0`
  - Actual: `max=0.000000`
  - Evidence: Sign constraint verified

### ✅ `prior_low_20d_dist`

- ✅ **sign_constraint**
  - Expected: `always >= 0`
  - Actual: `min=0.000000`
  - Evidence: Sign constraint verified
