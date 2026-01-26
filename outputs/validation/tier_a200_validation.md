# Tier A200 Validation Report

**Generated**: 2026-01-26T13:10:46.396177

## Summary

- **Total Checks**: 147
- **Passed**: 147
- **Failed**: 0
- **Pass Rate**: 100.0%

## Results by Chunk

| Chunk | Passed | Failed | Total |
|-------|--------|--------|-------|
| + Chunk 1: Extended MAs (101-120) | 37 | 0 | 37 |
| + Chunk 2: Duration & Proximity (121-140) | 15 | 0 | 15 |
| + Chunk 3: BB/RSI/Mean Reversion (141-160) | 14 | 0 | 14 |
| + Chunk 4: MACD/Volume/Calendar/Candle (161-180) | 20 | 0 | 20 |
| + Chunk 5: Ichimoku/Donchian/Divergence/Entropy (181-206) | 34 | 0 | 34 |
| + Lookahead Detection | 120 | 0 | 120 |
| + Cross-Indicator Consistency | 120 | 0 | 120 |
| + Known Events | 120 | 0 | 120 |

## Detailed Results

### [PASS] `data_quality`

- [PASS] **columns_present**
  - Expected: `All 106 columns`
  - Actual: `All 106 columns present`
  - Evidence: Column check passed

- [PASS] **no_nan_values**
  - Expected: `No NaN in any column`
  - Actual: `No NaN values`
  - Evidence: NaN check passed

- [PASS] **no_inf_values**
  - Expected: `No Inf in any column`
  - Actual: `No Inf values`
  - Evidence: Infinity check passed

### [PASS] `tema_9`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.012%, max_diff=2.12e-02`
  - Evidence: TA-Lib TEMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.06, 685.90]`
  - Evidence: MA range valid

### [PASS] `tema_20`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.020%, max_diff=3.47e-02`
  - Evidence: TA-Lib TEMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.10, 683.04]`
  - Evidence: MA range valid

### [PASS] `tema_50`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.017%, max_diff=2.93e-02`
  - Evidence: TA-Lib TEMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.21, 681.09]`
  - Evidence: MA range valid

### [PASS] `tema_100`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.072%, max_diff=1.26e-01`
  - Evidence: TA-Lib TEMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.42, 687.33]`
  - Evidence: MA range valid

### [PASS] `wma_10`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib WMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.31, 682.19]`
  - Evidence: MA range valid

### [PASS] `wma_20`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib WMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.40, 676.82]`
  - Evidence: MA range valid

### [PASS] `wma_50`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib WMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.59, 673.92]`
  - Evidence: MA range valid

### [PASS] `wma_200`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib WMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[26.04, 641.72]`
  - Evidence: MA range valid

### [PASS] `kama_10`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.096%, max_diff=1.62e-01`
  - Evidence: TA-Lib KAMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.48, 679.45]`
  - Evidence: MA range valid

### [PASS] `kama_20`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.088%, max_diff=1.48e-01`
  - Evidence: TA-Lib KAMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.64, 669.01]`
  - Evidence: MA range valid

### [PASS] `kama_50`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.332%, max_diff=5.60e-01`
  - Evidence: TA-Lib KAMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.80, 669.11]`
  - Evidence: MA range valid

### [PASS] `hma_9`

- [PASS] **formula_verification**
  - Expected: `HMA formula: WMA(2*WMA(n/2)-WMA(n), sqrt(n))`
  - Actual: `max_diff=1.36e-12`
  - Evidence: HMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[24.98, 687.42]`
  - Evidence: MA range valid

### [PASS] `hma_21`

- [PASS] **formula_verification**
  - Expected: `HMA formula: WMA(2*WMA(n/2)-WMA(n), sqrt(n))`
  - Actual: `max_diff=9.09e-13`
  - Evidence: HMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.10, 687.16]`
  - Evidence: MA range valid

### [PASS] `hma_50`

- [PASS] **formula_verification**
  - Expected: `HMA formula: WMA(2*WMA(n/2)-WMA(n), sqrt(n))`
  - Actual: `max_diff=3.90e-11`
  - Evidence: HMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.13, 679.45]`
  - Evidence: MA range valid

### [PASS] `vwma_10`

- [PASS] **formula_verification**
  - Expected: `VWMA = sum(close*vol)/sum(vol)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: VWMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.31, 681.45]`
  - Evidence: MA range valid

### [PASS] `vwma_20`

- [PASS] **formula_verification**
  - Expected: `VWMA = sum(close*vol)/sum(vol)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: VWMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.35, 675.09]`
  - Evidence: MA range valid

### [PASS] `vwma_50`

- [PASS] **formula_verification**
  - Expected: `VWMA = sum(close*vol)/sum(vol)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: VWMA hand-calculation verified

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.54, 669.86]`
  - Evidence: MA range valid

### [PASS] `tema_20_slope`

- [PASS] **formula_verification**
  - Expected: `tema_20 - tema_20.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Slope formula verified

### [PASS] `price_pct_from_tema_50`

- [PASS] **formula_verification**
  - Expected: `(close - tema_50) / tema_50 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Pct formula verified

### [PASS] `price_pct_from_kama_20`

- [PASS] **formula_verification**
  - Expected: `(close - kama_20) / kama_20 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Pct formula verified

### [PASS] `days_above_sma_9`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_sma_9 verified

- [PASS] **increment_logic**
  - Expected: `counter increments when condition persists`
  - Actual: `0 issues (< 1%)`
  - Evidence: Counter increment logic verified

### [PASS] `days_above_sma_50`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_sma_50 verified

- [PASS] **increment_logic**
  - Expected: `counter increments when condition persists`
  - Actual: `0 issues (< 1%)`
  - Evidence: Counter increment logic verified

### [PASS] `days_above_sma_200`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_sma_200 verified

### [PASS] `days_above_tema_20`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_tema_20 verified

### [PASS] `days_above_kama_20`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_kama_20 verified

### [PASS] `days_above_vwma_20`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_vwma_20 verified

### [PASS] `days_since_sma_9_50_cross`

- [PASS] **sign_convention**
  - Expected: `both positive and negative values exist`
  - Actual: `pos: 5445, neg: 2532`
  - Evidence: Sign convention verified (has both bullish/bearish)

### [PASS] `days_since_sma_50_200_cross`

- [PASS] **sign_convention**
  - Expected: `both positive and negative values exist`
  - Actual: `pos: 6086, neg: 1891`
  - Evidence: Sign convention verified (has both bullish/bearish)

### [PASS] `days_since_tema_sma_50_cross`

- [PASS] **sign_convention**
  - Expected: `both positive and negative values exist`
  - Actual: `pos: 5434, neg: 2543`
  - Evidence: Sign convention verified (has both bullish/bearish)

### [PASS] `days_since_kama_sma_50_cross`

- [PASS] **sign_convention**
  - Expected: `both positive and negative values exist`
  - Actual: `pos: 5265, neg: 2712`
  - Evidence: Sign convention verified (has both bullish/bearish)

### [PASS] `days_since_sma_9_200_cross`

- [PASS] **sign_convention**
  - Expected: `both positive and negative values exist`
  - Actual: `pos: 6112, neg: 1865`
  - Evidence: Sign convention verified (has both bullish/bearish)

### [PASS] `tema_20_sma_50_proximity`

- [PASS] **formula_verification**
  - Expected: `(tema_20 - sma_50) / sma_50 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Proximity formula verified

### [PASS] `sma_9_200_proximity`

- [PASS] **formula_verification**
  - Expected: `(sma_9 - sma_200) / sma_200 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Proximity formula verified

### [PASS] `pct_from_upper_band`

- [PASS] **formula_verification**
  - Expected: `(close - upper_bb) / upper_bb * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: BB pct formula verified

### [PASS] `pct_from_lower_band`

- [PASS] **formula_verification**
  - Expected: `(close - lower_bb) / lower_bb * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: BB pct formula verified

### [PASS] `bb_squeeze_indicator`

- [PASS] **squeeze_duration_consistency**
  - Expected: `squeeze=1 implies duration>=1`
  - Actual: `0 violations`
  - Evidence: Squeeze-duration consistency verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `rsi_distance_from_50`

- [PASS] **formula_verification**
  - Expected: `RSI - 50`
  - Actual: `max_diff=0.00e+00`
  - Evidence: RSI distance formula verified

### [PASS] `rsi_percentile_60d`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0167, 1.0000]`
  - Evidence: Percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.016667, 1.000000]`
  - Evidence: Range valid

### [PASS] `zscore_from_20d_mean`

- [PASS] **formula_verification**
  - Expected: `(close - SMA20) / std20`
  - Actual: `max_diff=2.47e-12`
  - Evidence: Z-score formula verified

### [PASS] `distance_from_52wk_high_pct`

- [PASS] **sign_constraint**
  - Expected: `always <= 0`
  - Actual: `max=0.000000`
  - Evidence: Sign constraint verified (cannot exceed 52wk high)

### [PASS] `percentile_in_52wk_range`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `consecutive_up_days`

- [PASS] **streak_exclusivity**
  - Expected: `up_days > 0 implies down_days == 0`
  - Actual: `0 violations`
  - Evidence: Streak exclusivity verified

### [PASS] `up_days_ratio_20d`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.2000, 0.9000]`
  - Evidence: Ratio range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.200000, 0.900000]`
  - Evidence: Range valid

### [PASS] `macd_signal`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib MACD signal match

### [PASS] `trading_day_of_week`

- [PASS] **range_check**
  - Expected: `[0, 4]`
  - Actual: `[0, 4]`
  - Evidence: Day of week range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 4]`
  - Actual: `[0.000000, 4.000000]`
  - Evidence: Range valid

### [PASS] `is_monday`

- [PASS] **calendar_logic**
  - Expected: `is_monday == (dayofweek == 0)`
  - Actual: `0 mismatches`
  - Evidence: Monday logic verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `is_friday`

- [PASS] **calendar_logic**
  - Expected: `is_friday == (dayofweek == 4)`
  - Actual: `0 mismatches`
  - Evidence: Friday logic verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `month_of_year`

- [PASS] **calendar_logic**
  - Expected: `month_of_year == dt.month, [1, 12]`
  - Actual: `0 mismatches`
  - Evidence: Month logic verified

- [PASS] **bounded_range_check**
  - Expected: `[1, 12]`
  - Actual: `[1.000000, 12.000000]`
  - Evidence: Range valid

### [PASS] `is_quarter_end_month`

- [PASS] **calendar_logic**
  - Expected: `is_quarter_end_month == month in {3,6,9,12}`
  - Actual: `0 mismatches`
  - Evidence: Quarter end logic verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `body_to_range_ratio`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Candle ratio range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `upper_wick_pct`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 0.9820]`
  - Evidence: Candle ratio range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 0.982013]`
  - Evidence: Range valid

### [PASS] `lower_wick_pct`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Candle ratio range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `doji_indicator`

- [PASS] **doji_logic**
  - Expected: `doji=1 implies body_ratio < 0.1`
  - Actual: `0 violations`
  - Evidence: Doji logic verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `candle_body_pct`

- [PASS] **formula_verification**
  - Expected: `abs(C-O)/O * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Candle body formula verified

### [PASS] `tenkan_sen`

- [PASS] **formula_verification**
  - Expected: `(9d_high + 9d_low) / 2`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Tenkan-sen formula verified

### [PASS] `kijun_sen`

- [PASS] **formula_verification**
  - Expected: `(26d_high + 26d_low) / 2`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Kijun-sen formula verified

### [PASS] `senkou_span_b`

- [PASS] **formula_verification**
  - Expected: `(52d_high + 52d_low) / 2`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Senkou Span B formula verified

### [PASS] `price_vs_cloud`

- [PASS] **value_range**
  - Expected: `{-1, 0, 1}`
  - Actual: `{0, 1, -1}`
  - Evidence: Cloud position values valid

- [PASS] **bounded_range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-1.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `donchian_upper_20`

- [PASS] **formula_verification**
  - Expected: `rolling(20).max(high)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Donchian upper formula verified

### [PASS] `donchian_lower_20`

- [PASS] **formula_verification**
  - Expected: `rolling(20).min(low)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Donchian lower formula verified

### [PASS] `donchian_position`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Donchian position range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `price_rsi_divergence`

- [PASS] **range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-0.7000, 0.8500]`
  - Evidence: Divergence range valid

- [PASS] **bounded_range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-0.700000, 0.850000]`
  - Evidence: Range valid

### [PASS] `price_obv_divergence`

- [PASS] **range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-0.9000, 0.9000]`
  - Evidence: Divergence range valid

- [PASS] **bounded_range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-0.900000, 0.900000]`
  - Evidence: Range valid

### [PASS] `divergence_magnitude`

- [PASS] **formula_verification**
  - Expected: `max(|rsi_div|, |obv_div|)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Divergence magnitude formula verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 0.900000]`
  - Evidence: Range valid

### [PASS] `permutation_entropy_order3`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.4246, 1.0000]`
  - Evidence: Entropy range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.424586, 1.000000]`
  - Evidence: Range valid

### [PASS] `permutation_entropy_order4`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.3139, 0.8915]`
  - Evidence: Entropy range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.313907, 0.891493]`
  - Evidence: Range valid

### [PASS] `permutation_entropy_order5`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.2348, 0.5791]`
  - Evidence: Entropy range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.234786, 0.579132]`
  - Evidence: Range valid

### [PASS] `atr_regime_pct_60d`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0167, 1.0000]`
  - Evidence: Regime percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.016667, 1.000000]`
  - Evidence: Range valid

### [PASS] `atr_regime_rolling_q`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0167, 1.0000]`
  - Evidence: Regime percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.016667, 1.000000]`
  - Evidence: Range valid

### [PASS] `trend_strength_pct_60d`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0167, 1.0000]`
  - Evidence: Regime percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.016667, 1.000000]`
  - Evidence: Range valid

### [PASS] `trend_strength_rolling_q`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0167, 1.0000]`
  - Evidence: Regime percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.016667, 1.000000]`
  - Evidence: Range valid

### [PASS] `regime_transition_prob`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 0.5500]`
  - Evidence: Regime percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 0.550000]`
  - Evidence: Range valid

### [PASS] `vol_regime_state`

- [PASS] **value_range**
  - Expected: `{-1, 0, 1}`
  - Actual: `{0, 1, -1}`
  - Evidence: Regime state values valid

- [PASS] **bounded_range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-1.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `regime_consistency`

- [PASS] **constraint_check**
  - Expected: `>= 1`
  - Actual: `min=1`
  - Evidence: Consistency constraint valid

### [PASS] `sample_audit`

- [PASS] **date_2020-03-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: COVID crash - worst single day: vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED), zscore=-2.14 (negative, EXPECTED)

- [PASS] **date_2020-03-23**
  - Expected: `indicators present`
  - Actual: `REVIEW`
  - Evidence: COVID bottom - reversal: data available, manual review recommended

- [PASS] **date_2021-11-19**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: All-time high before 2022 bear: above_cloud (EXPECTED), donchian_pos=0.889 (high, EXPECTED), near_52wk_high (EXPECTED)

- [PASS] **date_2022-06-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: 2022 bear market low: vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED), zscore=-2.02 (negative, EXPECTED)

- [PASS] **date_2023-10-27**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: October 2023 correction low: vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED), zscore=-2.08 (negative, EXPECTED)

- [PASS] **date_2024-07-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: Mid-2024 high: above_cloud (EXPECTED), donchian_pos=0.988 (high, EXPECTED), near_52wk_high (EXPECTED)

### [PASS] `cross_indicator`

- [PASS] **rsi_overbought_sync**
  - Expected: `RSI>70 implies days_overbought>0 (99%)`
  - Actual: `100.0% consistent`
  - Evidence: RSI-duration sync verified

- [PASS] **rsi_oversold_sync**
  - Expected: `RSI<30 implies days_oversold>0 (99%)`
  - Actual: `100.0% consistent`
  - Evidence: RSI-duration sync verified

- [PASS] **bb_squeeze_duration_sync**
  - Expected: `squeeze=1 implies duration>=1`
  - Actual: `0 violations`
  - Evidence: BB squeeze-duration sync verified

- [PASS] **monday_dow_sync**
  - Expected: `is_monday=1 implies dow=0`
  - Actual: `0 violations`
  - Evidence: Monday-DOW sync verified

- [PASS] **friday_dow_sync**
  - Expected: `is_friday=1 implies dow=4`
  - Actual: `0 violations`
  - Evidence: Friday-DOW sync verified

- [PASS] **quarter_end_month_sync**
  - Expected: `is_quarter_end=1 implies month in {3,6,9,12}`
  - Actual: `0 violations`
  - Evidence: Quarter end-month sync verified

- [PASS] **vol_regime_atr_sync**
  - Expected: `vol_state=1 implies atr_pct>0.7 (99%)`
  - Actual: `100.0% consistent`
  - Evidence: Volatility regime sync verified

- [PASS] **vol_regime_atr_low_sync**
  - Expected: `vol_state=-1 implies atr_pct<0.3 (99%)`
  - Actual: `100.0% consistent`
  - Evidence: Low volatility regime sync verified

- [PASS] **streak_exclusivity**
  - Expected: `up>0 and down>0 cannot both be true`
  - Actual: `0 violations`
  - Evidence: Streak exclusivity verified

- [PASS] **doji_body_ratio_sync**
  - Expected: `doji=1 implies body_ratio<0.1`
  - Actual: `0 violations`
  - Evidence: Doji-body ratio sync verified

### [PASS] `known_events`

- [PASS] **covid_vol_regime**
  - Expected: `>70% high vol days`
  - Actual: `100.0%`
  - Evidence: COVID crash showed sustained high volatility

- [PASS] **covid_atr_pct**
  - Expected: `mean atr_pct > 0.8`
  - Actual: `0.994`
  - Evidence: COVID ATR percentile was high

- [PASS] **covid_zscore**
  - Expected: `mean zscore < -1`
  - Actual: `-1.78`
  - Evidence: COVID showed negative z-scores (expected)

- [PASS] **covid_52wk_dist**
  - Expected: `min distance < -20%`
  - Actual: `-33.7%`
  - Evidence: COVID showed significant drawdown from 52wk high

- [PASS] **bear2022_cloud**
  - Expected: `>30% below cloud`
  - Actual: `49.0%`
  - Evidence: 2022 bear market showed time below cloud

- [PASS] **bear2022_streak**
  - Expected: `max down streak >= 3`
  - Actual: `6 days`
  - Evidence: 2022 bear market had down streaks

- [PASS] **bull2024_cloud**
  - Expected: `>50% above cloud`
  - Actual: `86.1%`
  - Evidence: 2023-24 bull run showed time above cloud

- [PASS] **bull2024_donchian**
  - Expected: `mean donchian > 0.6`
  - Actual: `0.790`
  - Evidence: 2023-24 bull run showed high Donchian position
