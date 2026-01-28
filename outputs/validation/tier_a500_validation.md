# Tier A500 Validation Report

**Generated**: 2026-01-28T00:23:56.963041

## Summary

- **Total Checks**: 96
- **Passed**: 96
- **Failed**: 0
- **Pass Rate**: 100.0%

## Results by Chunk

| Chunk | Passed | Failed | Total |
|-------|--------|--------|-------|
| + Sub-Chunk 6a: MA Extended Part 1 (207-230) | 25 | 0 | 25 |
| + Sub-Chunk 6b: MA Durations/Crosses + OSC (231-255) | 23 | 0 | 23 |
| + Sub-Chunk 7a: VOL Complete (256-278) | 18 | 0 | 18 |
| + Sub-Chunk 7b: VLM Complete (279-300) | 13 | 0 | 13 |
| + Lookahead Detection | 79 | 0 | 79 |
| + Cross-Indicator Consistency | 79 | 0 | 79 |
| + Known Events | 79 | 0 | 79 |

## Detailed Results

### [PASS] `data_quality`

- [PASS] **columns_present**
  - Expected: `All 94 columns`
  - Actual: `All 94 columns present`
  - Evidence: Column check passed

- [PASS] **no_nan_values**
  - Expected: `No NaN in any column`
  - Actual: `No NaN values`
  - Evidence: NaN check passed

- [PASS] **no_inf_values**
  - Expected: `No Inf in any column`
  - Actual: `No Inf values`
  - Evidence: Infinity check passed

### [PASS] `sma_5`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib SMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.27, 682.85]`
  - Evidence: MA range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `sma_14`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib SMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.37, 678.29]`
  - Evidence: MA range valid

### [PASS] `sma_21`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib SMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.44, 674.95]`
  - Evidence: MA range valid

### [PASS] `sma_63`

- [PASS] **talib_reference**
  - Expected: `max_diff < 1e-6`
  - Actual: `max_diff=0.00e+00`
  - Evidence: TA-Lib SMA reference match

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.73, 669.17]`
  - Evidence: MA range valid

### [PASS] `ema_5`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.052%, max_diff=8.76e-02`
  - Evidence: TA-Lib EMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.27, 682.61]`
  - Evidence: MA range valid

### [PASS] `ema_9`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.046%, max_diff=7.85e-02`
  - Evidence: TA-Lib EMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.34, 680.51]`
  - Evidence: MA range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `ema_50`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.146%, max_diff=2.46e-01`
  - Evidence: TA-Lib EMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.75, 669.63]`
  - Evidence: MA range valid

### [PASS] `ema_100`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.127%, max_diff=2.13e-01`
  - Evidence: TA-Lib EMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.92, 654.91]`
  - Evidence: MA range valid

### [PASS] `ema_200`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 0.5%`
  - Actual: `rel_diff=0.074%, max_diff=1.23e-01`
  - Evidence: TA-Lib EMA reference match (within relative tolerance)

- [PASS] **range_check**
  - Expected: `positive, reasonable range`
  - Actual: `[25.85, 628.96]`
  - Evidence: MA range valid

### [PASS] `sma_5_slope`

- [PASS] **formula_verification**
  - Expected: `sma_5 - sma_5.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Slope formula verified

### [PASS] `sma_21_slope`

- [PASS] **formula_verification**
  - Expected: `sma_21 - sma_21.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Slope formula verified

### [PASS] `sma_63_slope`

- [PASS] **formula_verification**
  - Expected: `sma_63 - sma_63.shift(5)`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Slope formula verified

### [PASS] `price_pct_from_sma_5`

- [PASS] **formula_verification**
  - Expected: `(close - sma_5) / sma_5 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Price-to-SMA distance formula verified

### [PASS] `price_pct_from_sma_21`

- [PASS] **formula_verification**
  - Expected: `(close - sma_21) / sma_21 * 100`
  - Actual: `max_diff=0.00e+00`
  - Evidence: Price-to-SMA distance formula verified

### [PASS] `days_above_ema_9`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_ema_9 verified

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 50.000000]`
  - Evidence: Range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `days_above_ema_50`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_ema_50 verified

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 258.000000]`
  - Evidence: Range valid

### [PASS] `days_above_sma_21`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_sma_21 verified

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 76.000000]`
  - Evidence: Range valid

### [PASS] `days_above_sma_63`

- [PASS] **mutual_exclusivity**
  - Expected: `days_above > 0 implies days_below == 0`
  - Actual: `0 violations`
  - Evidence: Mutual exclusivity with days_below_sma_63 verified

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 257.000000]`
  - Evidence: Range valid

### [PASS] `rsi_5`

- [PASS] **range_check**
  - Expected: `[0, 100]`
  - Actual: `[2.95, 97.79]`
  - Evidence: RSI range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 100]`
  - Actual: `[2.951795, 97.787365]`
  - Evidence: Range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `rsi_21`

- [PASS] **range_check**
  - Expected: `[0, 100]`
  - Actual: `[22.22, 84.29]`
  - Evidence: RSI range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 100]`
  - Actual: `[22.219626, 84.290350]`
  - Evidence: Range valid

### [PASS] `stoch_k_5`

- [PASS] **range_check**
  - Expected: `[0, 100]`
  - Actual: `[-0.00, 100.00]`
  - Evidence: Stochastic range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 100]`
  - Actual: `[-0.000000, 100.000000]`
  - Evidence: Range valid

### [PASS] `stoch_d_5`

- [PASS] **range_check**
  - Expected: `[0, 100]`
  - Actual: `[1.38, 100.00]`
  - Evidence: Stochastic range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 100]`
  - Actual: `[1.382329, 100.000000]`
  - Evidence: Range valid

### [PASS] `atr_5`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 2%`
  - Actual: `rel_diff=1.349%, max_diff=3.03e-02`
  - Evidence: TA-Lib ATR reference match (within tolerance)

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `atr_21`

- [PASS] **talib_reference**
  - Expected: `rel_diff < 2%`
  - Actual: `rel_diff=0.008%, max_diff=1.90e-04`
  - Evidence: TA-Lib ATR reference match (within tolerance)

### [PASS] `atr_5_21_ratio`

- [PASS] **formula_verification**
  - Expected: `atr_5 / atr_21`
  - Actual: `max_diff=0.00e+00`
  - Evidence: ATR ratio formula verified

### [PASS] `rogers_satchell_volatility`

- [PASS] **range_check**
  - Expected: `non-negative`
  - Actual: `[0.037734, 0.809779]`
  - Evidence: Volatility range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.037734, 0.809779]`
  - Evidence: Range valid

### [PASS] `yang_zhang_volatility`

- [PASS] **range_check**
  - Expected: `non-negative`
  - Actual: `[0.042568, 0.946707]`
  - Evidence: Volatility range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.042568, 0.946707]`
  - Evidence: Range valid

### [PASS] `historical_volatility_10d`

- [PASS] **range_check**
  - Expected: `non-negative`
  - Actual: `[0.020060, 1.130634]`
  - Evidence: Volatility range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.020060, 1.130634]`
  - Evidence: Range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `nvi_signal`

- [PASS] **binary_check**
  - Expected: `values in {0, 1}`
  - Actual: `unique values: [0, 1]`
  - Evidence: NVI signal is binary

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `pvi_signal`

- [PASS] **binary_check**
  - Expected: `values in {0, 1}`
  - Actual: `unique values: [0, 1]`
  - Evidence: PVI signal is binary

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `volume_spike_price_flat`

- [PASS] **exclusivity_check**
  - Expected: `spike_both=1 implies spike_flat=0`
  - Actual: `0 violations`
  - Evidence: Volume spike exclusivity verified

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `volume_percentile_20d`

- [PASS] **range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.0000, 1.0000]`
  - Evidence: Volume percentile range valid

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `days_below_ema_9`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 28.000000]`
  - Evidence: Range valid

### [PASS] `days_below_ema_50`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 86.000000]`
  - Evidence: Range valid

### [PASS] `days_below_sma_21`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 44.000000]`
  - Evidence: Range valid

### [PASS] `days_below_sma_63`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 88.000000]`
  - Evidence: Range valid

### [PASS] `consecutive_decreasing_vol`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 7.000000]`
  - Evidence: Range valid

### [PASS] `consecutive_high_vol_days`

- [PASS] **bounded_range_check**
  - Expected: `[0, inf]`
  - Actual: `[0.000000, 16.000000]`
  - Evidence: Range valid

### [PASS] `atr_percentile_20d`

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `bb_width_percentile_20d`

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `price_bb_band_position`

- [PASS] **bounded_range_check**
  - Expected: `[-1, 2]`
  - Actual: `[-0.473077, 1.322844]`
  - Evidence: Range valid

### [PASS] `kc_position`

- [PASS] **bounded_range_check**
  - Expected: `[-2, 3]`
  - Actual: `[-1.015531, 2.043940]`
  - Evidence: Range valid

### [PASS] `volume_price_spike_both`

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `vol_breakout_confirmation`

- [PASS] **bounded_range_check**
  - Expected: `[0, 1]`
  - Actual: `[0.000000, 1.000000]`
  - Evidence: Range valid

### [PASS] `bb_kc_ratio`

- [PASS] **bounded_range_check**
  - Expected: `[0, 5]`
  - Actual: `[0.416462, 4.593990]`
  - Evidence: Range valid

### [PASS] `vol_clustering_score`

- [PASS] **bounded_range_check**
  - Expected: `[-1, 1]`
  - Actual: `[-0.790341, 0.830103]`
  - Evidence: Range valid

### [PASS] `cmf_20`

- [PASS] **lookahead_truncation_test**
  - Expected: `full == truncated at all test points`
  - Actual: `all match`
  - Evidence: No lookahead detected

### [PASS] `sample_audit`

- [PASS] **date_2020-03-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: COVID crash - worst single day: atr_ratio>1.5 (vol expansion, EXPECTED), rsi_5=28 (oversold, EXPECTED)

- [PASS] **date_2020-03-23**
  - Expected: `indicators present`
  - Actual: `REVIEW`
  - Evidence: COVID bottom - reversal: data available, manual review recommended

- [PASS] **date_2021-11-19**
  - Expected: `indicators present`
  - Actual: `REVIEW`
  - Evidence: All-time high before 2022 bear: data available, manual review recommended

- [PASS] **date_2022-06-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: 2022 bear market low: rsi_5=19 (oversold, EXPECTED)

- [PASS] **date_2023-10-27**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: October 2023 correction low: rsi_5=16 (oversold, EXPECTED)

- [PASS] **date_2024-07-16**
  - Expected: `indicators sensible for context`
  - Actual: `SENSIBLE`
  - Evidence: Mid-2024 high: rsi_5=80 (overbought, EXPECTED)

### [PASS] `rsi_5_21_spread`

- [PASS] **cross_indicator_consistency**
  - Expected: `rsi_5 - rsi_21`
  - Actual: `max_diff=0.0000`
  - Evidence: RSI spread formula verified

### [PASS] `cross_indicator`

- [PASS] **ema_9_duration_exclusivity**
  - Expected: `above>0 XOR below>0`
  - Actual: `0 violations`
  - Evidence: ema_9 duration exclusivity verified

- [PASS] **ema_50_duration_exclusivity**
  - Expected: `above>0 XOR below>0`
  - Actual: `0 violations`
  - Evidence: ema_50 duration exclusivity verified

- [PASS] **sma_21_duration_exclusivity**
  - Expected: `above>0 XOR below>0`
  - Actual: `0 violations`
  - Evidence: sma_21 duration exclusivity verified

- [PASS] **sma_63_duration_exclusivity**
  - Expected: `above>0 XOR below>0`
  - Actual: `0 violations`
  - Evidence: sma_63 duration exclusivity verified

### [PASS] `known_events`

- [PASS] **covid_atr_ratio**
  - Expected: `mean atr_5_21_ratio > 1.5`
  - Actual: `1.772`
  - Evidence: COVID showed ATR expansion (short > long)

- [PASS] **covid_rsi**
  - Expected: `min rsi_5 < 30`
  - Actual: `19.7`
  - Evidence: COVID showed oversold RSI

- [PASS] **bear2022_vol**
  - Expected: `elevated volatility periods`
  - Actual: `max=0.4007`
  - Evidence: 2022 bear market volatility data

- [PASS] **bear2022_cmf**
  - Expected: `cmf level`
  - Actual: `0.0548`
  - Evidence: 2022 bear market CMF data
