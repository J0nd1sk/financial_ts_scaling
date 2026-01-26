# Workstream 1 Context: Feature Generation (tier_a200)
# Last Updated: 2026-01-25 23:30

## Identity
- **ID**: ws1
- **Name**: feature_generation
- **Focus**: tier_a200 implementation (Chunks 1-3 complete)
- **Status**: active

---

## Current Task
- **Working on**: Tier a200 Chunk 3 (BB Extension, RSI Duration, Mean Reversion, Consecutive Patterns)
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

### Pending
- Commit tier_a200 Chunks 1-3 files
- tier_a200 Chunks 4-10 (ranks 161-200) - future work

---

## Last Session Work (2026-01-25 23:30)

### Implemented tier_a200 Chunk 3 (TDD approach)

1. **Added feature list** - 20 new features to A200_ADDITION_LIST (total: 60)

2. **Wrote failing tests** (test_tier_a200.py):
   - TestChunk3FeatureListStructure - list composition (60 additions, 160 total)
   - TestChunk3BollingerBandExtension - BB distance, duration features
   - TestChunk3BBSqueeze - Squeeze indicator and duration
   - TestChunk3RSIDuration - RSI duration/percentile features
   - TestChunk3MeanReversion - Z-scores, 52-week features
   - TestChunk3ConsecutivePatterns - Up/down days, range compression
   - TestChunk3OutputShape - Output structure (161 columns)

3. **Implemented compute functions** (tier_a200.py):
   - `_compute_bb_extension()` - BB distance and duration features (141-146)
   - `_compute_rsi_duration()` - RSI duration/percentile features (147-150)
   - `_compute_mean_reversion()` - Z-scores, 52-week features (151-156)
   - `_compute_consecutive_patterns()` - Consecutive up/down, range compression (157-160)
   - Updated `build_feature_dataframe()` to call Chunk 3 functions

4. **Verification**:
   - `make test`: 840 passed, 2 skipped
   - Feature counts: A200_ADDITION_LIST=60, FEATURE_LIST=160
   - No duplicate features with tier_a100
   - All value range constraints satisfied

### Chunk 3 Features (ranks 141-160)

**BB Extension (6 features, ranks 141-146):**
| Rank | Feature | Signal |
|------|---------|--------|
| 141 | pct_from_upper_band | % distance to upper BB (neg=inside) |
| 142 | pct_from_lower_band | % distance to lower BB (pos=inside) |
| 143 | days_above_upper_band | Consecutive days outside upper BB |
| 144 | days_below_lower_band | Consecutive days outside lower BB |
| 145 | bb_squeeze_indicator | BB inside Keltner Channel (0/1) |
| 146 | bb_squeeze_duration | Days in current squeeze state |

**RSI Duration (4 features, ranks 147-150):**
| Rank | Feature | Signal |
|------|---------|--------|
| 147 | rsi_distance_from_50 | RSI - 50 (-50 to +50) |
| 148 | days_rsi_overbought | Consecutive days RSI > 70 |
| 149 | days_rsi_oversold | Consecutive days RSI < 30 |
| 150 | rsi_percentile_60d | RSI rank over 60 days (0-1) |

**Mean Reversion (6 features, ranks 151-156):**
| Rank | Feature | Signal |
|------|---------|--------|
| 151 | zscore_from_20d_mean | (Price - SMA20) / StdDev20 |
| 152 | zscore_from_50d_mean | (Price - SMA50) / StdDev50 |
| 153 | percentile_in_52wk_range | Position in 252d range (0-1) |
| 154 | distance_from_52wk_high_pct | % below 252-day high |
| 155 | days_since_52wk_high | Days since 252-day high |
| 156 | days_since_52wk_low | Days since 252-day low |

**Consecutive Patterns (4 features, ranks 157-160):**
| Rank | Feature | Signal |
|------|---------|--------|
| 157 | consecutive_up_days | Count of consecutive higher closes |
| 158 | consecutive_down_days | Count of consecutive lower closes |
| 159 | up_days_ratio_20d | Fraction of up days in last 20 |
| 160 | range_compression_5d | Avg range / Avg range 20d ago |

---

## Files Modified
- `src/features/tier_a200.py` - Modified (uncommitted)
  - Added docstring for Chunk 3
  - Added 20 feature names to A200_ADDITION_LIST (total: 60)
  - Added 4 compute functions for Chunk 3
  - Updated build_feature_dataframe()
- `tests/features/test_tier_a200.py` - Modified (uncommitted)
  - Updated test counts (60 additions, 160 total)
  - Added 7 new test classes for Chunk 3
  - Updated existing count tests to new totals

---

## Test Status
- `make test`: 840 passed, 2 skipped (2026-01-25 23:25)
- 842 total tests collected
- All tier_a200 tests pass (Chunks 1-3)
- 54 new tests for Chunk 3

---

## Key Decisions (Workstream-Specific)

### Duration Counter Design (Chunks 2-3)
- **Convention**: price >= MA means "above" (inclusive)
- **Counter resets**: On cross event, counter resets to 0 for opposite state
- **Mutual exclusivity**: At any point, days_above=0 OR days_below=0
- **Maximum value**: Can exceed output length (counting starts at MA warmup)

### Cross Recency Design (Chunk 2)
- **Signed values**: Positive = bullish (short > long), Negative = bearish
- **Magnitude**: Days since last cross event

### BB Squeeze Design (Chunk 3)
- **Squeeze condition**: BB inside Keltner Channel (upper BB < upper KC AND lower BB > lower KC)
- **Keltner params**: 20-period EMA, 1.5 ATR multiplier
- **BB params**: 20-period, 2 std

### 52-week Features (Chunk 3)
- **Window**: 252 trading days (standard 1 year)
- **days_since_high/low**: Uses idxmax/idxmin over lookback window

---

## Next Session Should
1. `git add -A && git commit` tier_a200 Chunks 1-3 files
2. Plan tier_a200 Chunk 4 (ranks 161-180) if continuing
3. Or pivot to other workstream priorities (ws3 Phase 6C HPO runs)
