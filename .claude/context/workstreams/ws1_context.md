# Workstream 1 Context: Feature Generation (tier_a200)
# Last Updated: 2026-01-26 13:30

## Identity
- **ID**: ws1
- **Name**: feature_generation
- **Focus**: tier_a200 implementation (Chunks 1-5 COMPLETE)
- **Status**: active

---

## Current Task
- **Working on**: Tier a200 Chunk 5 (Ichimoku, Donchian, Divergence, Entropy/Regime)
- **Status**: COMPLETE - ready to commit

---

## Progress Summary

### Completed
- **tier_a100 implementation** (all 8 chunks, 100 indicators)
- **tier_a100 deep validation script** (2026-01-25):
  - `scripts/validate_tier_a100.py` - 69 checks, 100% pass
- **tier_a200 Chunk 1** (2026-01-25 21:30):
  - 20 new MA indicators (ranks 101-120)
  - Categories: TEMA, WMA, KAMA, HMA, VWMA, derived
- **tier_a200 Chunk 2** (2026-01-25 22:45):
  - 20 new indicators (ranks 121-140)
  - Categories: Duration Counters (12), MA Cross Recency (5), New Proximity (3)
- **tier_a200 Chunk 3** (2026-01-25 23:30):
  - 20 new indicators (ranks 141-160)
  - Categories: BB Extension (6), RSI Duration (4), Mean Reversion (6), Consecutive Patterns (4)
- **tier_a200 Chunk 4** (2026-01-26 01:45):
  - 20 new indicators (ranks 161-180)
  - Categories: MACD Extensions (4), Volume Dynamics (4), Calendar (6), Candle Analysis (6)
- **tier_a200 Chunk 5** (2026-01-26 13:30):
  - 26 new indicators (ranks 181-206)
  - Categories: Ichimoku Cloud (6), Donchian Channel (5), Divergence (4), Entropy/Regime (11)

### Pending
- Commit tier_a200 Chunks 1-5 files
- Generate processed tier_a200 parquet files (separate task)

---

## Last Session Work (2026-01-26 13:30)

### Implemented tier_a200 Chunk 5 (TDD approach)

1. **Added feature list** - 26 new features to A200_ADDITION_LIST (total: 106)
   - Total FEATURE_LIST now 206 features

2. **Wrote failing tests** (test_tier_a200.py):
   - TestChunk5FeatureListStructure - list composition (106 additions, 206 total)
   - TestChunk5IchimokuIndicators - Tenkan/Kijun/Senkou lines, price_vs_cloud, thickness
   - TestChunk5DonchianIndicators - Upper/lower bounds, position, width, breakout proximity
   - TestChunk5DivergenceIndicators - Price/RSI/OBV divergence, streak, magnitude
   - TestChunk5EntropyRegimeIndicators - Permutation entropy orders 3/4/5, regime percentiles, state
   - TestChunk5OutputShape - Output structure (207 columns)

3. **Implemented compute functions** (tier_a200.py):
   - `_compute_ichimoku()` - Tenkan, Kijun, Senkou spans, price vs cloud, thickness (181-186)
   - `_compute_donchian()` - Upper/lower channels, position, width, breakout % (187-191)
   - `_compute_divergence()` - Price/RSI/OBV divergence, streak, magnitude (192-195)
   - `_permutation_entropy()` - Helper for ordinal pattern entropy calculation
   - `_compute_entropy_regime()` - Entropy orders, regime percentiles, state, consistency (196-206)
   - Updated `build_feature_dataframe()` to call Chunk 5 functions

4. **Edge case handling**:
   - Flat Donchian channel: position = 0.5, width = 0
   - Entropy on constant series: returns 0
   - Regime state: -1 (low), 0 (medium), 1 (high) based on 30%/70% thresholds

5. **Verification**:
   - `make test`: 944 passed, 2 skipped
   - Feature counts: A200_ADDITION_LIST=106, FEATURE_LIST=206
   - Output columns: 207 (Date + 206 features)
   - No NaN values in output after warmup

### Chunk 5 Features (ranks 181-206)

**Ichimoku Cloud (6 features, ranks 181-186):**
| Rank | Feature | Signal |
|------|---------|--------|
| 181 | tenkan_sen | (9-day H+L)/2 - Fast line |
| 182 | kijun_sen | (26-day H+L)/2 - Baseline |
| 183 | senkou_span_a | (tenkan+kijun)/2 - Cloud edge 1 |
| 184 | senkou_span_b | (52-day H+L)/2 - Cloud edge 2 |
| 185 | price_vs_cloud | -1/0/+1 (below/inside/above cloud) |
| 186 | cloud_thickness_pct | (span_a - span_b) / close * 100 |

**Donchian Channel (5 features, ranks 187-191):**
| Rank | Feature | Signal |
|------|---------|--------|
| 187 | donchian_upper_20 | 20-day rolling max(High) |
| 188 | donchian_lower_20 | 20-day rolling min(Low) |
| 189 | donchian_position | (Close - lower) / (upper - lower) [0,1] |
| 190 | donchian_width_pct | (upper - lower) / close * 100 |
| 191 | pct_to_donchian_breakout | % to nearest boundary, signed |

**Momentum Divergence (4 features, ranks 192-195):**
| Rank | Feature | Signal |
|------|---------|--------|
| 192 | price_rsi_divergence | 20d percentile rank difference |
| 193 | price_obv_divergence | 20d percentile rank difference |
| 194 | divergence_streak | Consecutive days with |div| > 0.2 |
| 195 | divergence_magnitude | max(|price_rsi_div|, |price_obv_div|) |

**Entropy & Regime (11 features, ranks 196-206):**
| Rank | Feature | Signal |
|------|---------|--------|
| 196 | permutation_entropy_order3 | 20d rolling entropy (6 patterns) [0,1] |
| 197 | permutation_entropy_order4 | 20d rolling entropy (24 patterns) [0,1] |
| 198 | permutation_entropy_order5 | 20d rolling entropy (120 patterns) [0,1] |
| 199 | entropy_trend_5d | Change in order4 entropy over 5 days |
| 200 | atr_regime_pct_60d | 60-day percentile of ATR% [0,1] |
| 201 | atr_regime_rolling_q | Rolling 60d quantile position [0,1] |
| 202 | trend_strength_pct_60d | 60-day percentile of ADX [0,1] |
| 203 | trend_strength_rolling_q | Rolling 60d quantile position [0,1] |
| 204 | vol_regime_state | -1 (low), 0 (medium), 1 (high) |
| 205 | regime_consistency | Days in current vol_regime_state |
| 206 | regime_transition_prob | % of last 20 days with regime changes [0,1] |

---

## Files Modified
- `src/features/tier_a200.py` - Modified (uncommitted)
  - Added docstring for Chunk 5
  - Added 26 feature names to A200_ADDITION_LIST (total: 106)
  - Added 4 compute functions + 1 helper for Chunk 5
  - Updated build_feature_dataframe()
- `tests/features/test_tier_a200.py` - Modified (uncommitted)
  - Updated test counts (106 additions, 206 total)
  - Added 6 new test classes for Chunk 5 (~70 new tests)
  - Updated existing count tests to new totals

---

## Test Status
- `make test`: 944 passed, 2 skipped (2026-01-26 13:30)
- All tier_a200 tests pass (Chunks 1-5)
- ~70 new tests for Chunk 5
- 252 total tier_a200 tests

---

## Key Decisions (Workstream-Specific)

### Ichimoku Cloud Design (Chunk 5)
- **Cloud definition**: Cloud boundaries are span_a and span_b
- **price_vs_cloud**: Categorical {-1, 0, 1} - below/inside/above
- **cloud_thickness_pct**: Signed - positive = span_a > span_b (bullish)

### Donchian Channel Design (Chunk 5)
- **Position clamp**: Clamped to [0, 1] even if price outside channel
- **Flat channel edge case**: position = 0.5, width = 0
- **Breakout proximity**: Positive = closer to upper, negative = closer to lower

### Divergence Design (Chunk 5)
- **Percentile-based**: Use 20-day rolling percentile ranks for comparability
- **Range**: [-1, 1] since both series are in [0, 1]
- **Streak threshold**: |divergence| > 0.2 considered significant

### Permutation Entropy (Chunk 5)
- **Pure numpy implementation**: No scipy dependency
- **Normalized**: Divided by log(n!) for [0, 1] range
- **Multiple orders**: 3, 4, 5 to let model find optimal complexity measure

### Regime State (Chunk 5)
- **Thresholds**: High > 0.7, Low < 0.3, Middle otherwise
- **Consistency**: Counts consecutive days in same state
- **Transition prob**: Rolling 20-day mean of state changes

---

## Cumulative tier_a200 Status

| Chunk | Ranks | Features | Status |
|-------|-------|----------|--------|
| 1 | 101-120 | 20 (MA types) | Complete |
| 2 | 121-140 | 20 (Duration, Cross, Proximity) | Complete |
| 3 | 141-160 | 20 (BB, RSI, MeanReversion, Patterns) | Complete |
| 4 | 161-180 | 20 (MACD, Volume, Calendar, Candle) | Complete |
| 5 | 181-206 | 26 (Ichimoku, Donchian, Divergence, Entropy/Regime) | Complete |

**Totals**: 106/106 tier_a200 additions complete, 206 total features

---

## Next Session Should
1. `git add -A && git commit` tier_a200 Chunks 1-5 files
2. Generate processed tier_a200 parquet files
3. Or pivot to ws3 Phase 6C HPO runs with larger feature tier

---

## Memory Entities (Workstream-Specific)
- None created this session
