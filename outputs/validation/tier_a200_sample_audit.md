# Tier A200 Sample Date Audit Report

**Generated**: 2026-01-26T13:10:46.396177

## Audit Summary

| Date | Context | Status | Key Observations |
|------|---------|--------|------------------|
| 2020-03-16 | COVID crash - worst single day | SENSIBLE | vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED) |
| 2020-03-23 | COVID bottom - reversal | REVIEW | See details |
| 2021-11-19 | All-time high before 2022 bear | SENSIBLE | above_cloud (EXPECTED), donchian_pos=0.889 (high, EXPECTED) |
| 2022-06-16 | 2022 bear market low | SENSIBLE | vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED) |
| 2023-10-27 | October 2023 correction low | SENSIBLE | vol_regime=high (EXPECTED), atr_pct>0.7 (EXPECTED) |
| 2024-07-16 | Mid-2024 high | SENSIBLE | above_cloud (EXPECTED), donchian_pos=0.988 (high, EXPECTED) |

## Detailed Audits

### 2020-03-16 - COVID crash - worst single day

**Status**: SENSIBLE

**Indicators**:
- `vol_regime_state`: 1
- `atr_regime_pct_60d`: 1.0
- `zscore_from_20d_mean`: -2.14
- `distance_from_52wk_high_pct`: -29.11
- `days_rsi_oversold`: 0
- `days_rsi_overbought`: 0
- `price_vs_cloud`: -1
- `donchian_position`: 0.024
- `permutation_entropy_order4`: 0.606

**Sensibility Checks**:
- vol_regime=high (EXPECTED)
- atr_pct>0.7 (EXPECTED)
- zscore=-2.14 (negative, EXPECTED)

### 2020-03-23 - COVID bottom - reversal

**Status**: REVIEW

**Indicators**:
- `vol_regime_state`: 1
- `atr_regime_pct_60d`: 1.0
- `zscore_from_20d_mean`: -1.69
- `distance_from_52wk_high_pct`: -33.72
- `days_rsi_oversold`: 1
- `days_rsi_overbought`: 0
- `price_vs_cloud`: -1
- `donchian_position`: 0.045
- `permutation_entropy_order4`: 0.763

**Sensibility Checks**:

### 2021-11-19 - All-time high before 2022 bear

**Status**: SENSIBLE

**Indicators**:
- `vol_regime_state`: -1
- `atr_regime_pct_60d`: 0.167
- `zscore_from_20d_mean`: 0.96
- `distance_from_52wk_high_pct`: -0.18
- `days_rsi_oversold`: 0
- `days_rsi_overbought`: 0
- `price_vs_cloud`: 1
- `donchian_position`: 0.889
- `permutation_entropy_order4`: 0.584

**Sensibility Checks**:
- above_cloud (EXPECTED)
- donchian_pos=0.889 (high, EXPECTED)
- near_52wk_high (EXPECTED)

### 2022-06-16 - 2022 bear market low

**Status**: SENSIBLE

**Indicators**:
- `vol_regime_state`: 1
- `atr_regime_pct_60d`: 1.0
- `zscore_from_20d_mean`: -2.02
- `distance_from_52wk_high_pct`: -23.01
- `days_rsi_oversold`: 0
- `days_rsi_overbought`: 0
- `price_vs_cloud`: -1
- `donchian_position`: 0.048
- `permutation_entropy_order4`: 0.805

**Sensibility Checks**:
- vol_regime=high (EXPECTED)
- atr_pct>0.7 (EXPECTED)
- zscore=-2.02 (negative, EXPECTED)

### 2023-10-27 - October 2023 correction low

**Status**: SENSIBLE

**Indicators**:
- `vol_regime_state`: 1
- `atr_regime_pct_60d`: 1.0
- `zscore_from_20d_mean`: -2.08
- `distance_from_52wk_high_pct`: -9.97
- `days_rsi_oversold`: 1
- `days_rsi_overbought`: 0
- `price_vs_cloud`: -1
- `donchian_position`: 0.051
- `permutation_entropy_order4`: 0.712

**Sensibility Checks**:
- vol_regime=high (EXPECTED)
- atr_pct>0.7 (EXPECTED)
- zscore=-2.08 (negative, EXPECTED)

### 2024-07-16 - Mid-2024 high

**Status**: SENSIBLE

**Indicators**:
- `vol_regime_state`: -1
- `atr_regime_pct_60d`: 0.15
- `zscore_from_20d_mean`: 1.96
- `distance_from_52wk_high_pct`: 0.0
- `days_rsi_oversold`: 0
- `days_rsi_overbought`: 3
- `price_vs_cloud`: 1
- `donchian_position`: 0.988
- `permutation_entropy_order4`: 0.614

**Sensibility Checks**:
- above_cloud (EXPECTED)
- donchian_pos=0.988 (high, EXPECTED)
- near_52wk_high (EXPECTED)

