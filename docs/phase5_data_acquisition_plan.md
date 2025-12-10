# Phase 5: Data Acquisition - Dataset Matrix Expansion Plan

**Status:** In Progress
**Date:** 2025-12-09 (Updated 2025-12-10)
**Execution Strategy:** Sequential TDD with sub-tasks and approval gates

---

## Objective

Expand the dataset matrix to enable scaling law experiments across multiple assets and feature tiers, following the experimental design in CLAUDE.md.

**Outcome:** Can train models on datasets ranging from single-asset (SPY) to multi-asset (SPY+DIA+QQQ) with multiple feature tiers (a20 → a20+VIX → full indicator suite).

---

## Dataset Matrix Reference

From the experimental design:

### Rows (Asset Scaling)
| Row | Assets | Description |
|-----|--------|-------------|
| **A** | SPY only | ✅ Complete (Phase 2) |
| **B** | +DIA, QQQ (ETFs) | Major ETFs - ✅ Complete |
| **B'** | +^DJI, ^IXIC (Indices) | Index data for extended history - ✅ Complete |
| **C** | +stocks (AAPL, MSFT, GOOGL, AMZN, TSLA) | Individual stocks |
| **D** | +sentiment (SF Fed News Sentiment Index) | Sentiment data |
| **E** | +economic indicators (FRED) | Macro indicators |

### Index vs ETF Data (Decision 2025-12-10)

To maximize training history, we downloaded both ETF and index data:

| Asset | ETF Ticker | Index Ticker | ETF Start | Index Start |
|-------|------------|--------------|-----------|-------------|
| S&P 500 | SPY | (^GSPC avail) | 1993 | 1927 |
| Dow Jones | DIA | **^DJI** | 1998 | **1992** |
| NASDAQ | QQQ | **^IXIC** | 1999 | **1971** |

**Training window with indices: 1992+ (vs 1999+ with ETFs only)**

See `decision_log.md` entry "2025-12-10 Index Tickers for Extended Training History" for full rationale.

### Columns (Feature Tiers)
| Col | Features | Description |
|-----|----------|-------------|
| **a** | OHLCV + 20 indicators | ✅ Complete (tier a20) |
| **b** | +sentiment indicators | News/social sentiment |
| **c** | +VIX correlation features | **Phase 5 Priority** |
| **d** | +trend analysis features | Extended indicators |

---

## Scope

### In Scope

1. **Asset Row B**: Download DIA and QQQ OHLCV data
2. **Feature Tier c**: Add VIX data and VIX-correlation features
3. **Multi-asset feature pipeline**: Generalize tier_a20.py for any ticker
4. **Combined dataset builder**: Create merged multi-asset datasets
5. **Manifest registration**: Track all new data artifacts
6. **Tests**: TDD for all new functionality

### Out of Scope

- Individual stocks (AAPL, MSFT, etc.) - defer to later phase
- Sentiment data (SF Fed News Sentiment) - requires separate API research
- FRED economic indicators - requires FRED API integration
- New model architectures - use existing PatchTST
- HPO/experiments - just data infrastructure

---

## Testing Rules

All tests in this phase MUST follow these rules:

### 1. Offline/Deterministic Tests
- **All yfinance calls must be mocked** - no live API calls in tests
- Tests must be deterministic and reproducible
- CI pipeline runs without network access to external APIs

### 2. Mock Strategy
- Use `unittest.mock.patch` to mock `yfinance.download()`
- Return synthetic DataFrames with known values for assertions
- Follow existing pattern in `tests/test_data_download.py`

### 3. Live Downloads
- Live downloads happen ONLY via CLI scripts (`scripts/download_*.py`)
- Scripts are run manually, not by CI
- Downloaded data is registered in manifests and committed

### 4. Test Fixtures
- Use small synthetic DataFrames (10-100 rows) for unit tests
- For integration tests requiring realistic data, use cached snapshots
- Never depend on `data/raw/*.parquet` existing in tests

---

## Sub-Task Breakdown

### Task 1: Generalize OHLCV Download Script

**Purpose:** Extend download_ohlcv.py to support any ticker, not just SPY

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `scripts/download_ohlcv.py` | Add `download_ticker()` function | +40 |
| `tests/test_data_download.py` | Add multi-ticker tests | +60 |

**Changes:**
- Add `download_ticker(ticker: str, output_dir: str)` function
- Keep existing `download_spy()` as convenience wrapper
- Parameterize manifest dataset naming: `{TICKER}.OHLCV.daily`
- Add retry logic with exponential backoff and jitter (3 retries, 1s/2s/4s base delays)
- Catch `requests.exceptions.RequestException` and yfinance-specific errors

**Tests (all mocked, no live API calls):**
- `test_download_ticker_dia_basic`: Mock yfinance, verify DIA parquet written
- `test_download_ticker_qqq_basic`: Mock yfinance, verify QQQ parquet written
- `test_download_ticker_registers_manifest`: Mock download, verify manifest entry
- `test_download_ticker_invalid_raises`: Invalid ticker raises ValueError
- `test_download_ticker_retries_on_failure`: Verify retry logic triggers on error

**Dependencies:** None

---

### Task 2: Download DIA, QQQ, ^DJI, ^IXIC Data ✅ COMPLETE

**Purpose:** Acquire OHLCV data for ETFs and indices (indices provide extended history)

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `data/raw/DIA.parquet` | DIA ETF OHLCV (1998+) | ✅ 7,018 rows |
| `data/raw/QQQ.parquet` | QQQ ETF OHLCV (1999+) | ✅ 6,731 rows |
| `data/raw/DJI.parquet` | Dow Jones Index OHLCV (1992+) | ✅ 8,546 rows |
| `data/raw/IXIC.parquet` | NASDAQ Composite OHLCV (1971+) | ✅ 13,829 rows |
| `data/raw/manifest.json` | Updated with all entries | ✅ |

**Execution:**
```bash
python scripts/download_ohlcv.py --ticker DIA
python scripts/download_ohlcv.py --ticker QQQ
python scripts/download_ohlcv.py --ticker "^DJI"
python scripts/download_ohlcv.py --ticker "^IXIC"
```

**Validation:** ✅ All Complete
- DIA: 7,018 rows, 1998-01-20 to 2025-12-10, 0 nulls
- QQQ: 6,731 rows, 1999-03-10 to 2025-12-09, 0 nulls
- DJI: 8,546 rows, 1992-01-02 to 2025-12-09, 0 nulls
- IXIC: 13,829 rows, 1971-02-05 to 2025-12-10, 0 nulls
- All manifest entries registered with MD5 checksums

**Implementation Notes:**
- Added `_sanitize_ticker_for_filename()` to handle ^ in index tickers
- Index tickers (^DJI) saved as DJI.parquet (^ removed from filename)
- 101 tests passing after changes

**Dependencies:** Task 1

---

### Task 3: Download VIX Data

**Purpose:** Acquire CBOE Volatility Index data for feature tier c

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `scripts/download_ohlcv.py` | Add VIX download support | +20 |
| `data/raw/VIX.parquet` | VIX OHLCV data | - |
| `tests/test_data_download.py` | VIX-specific tests | +30 |

**Notes:**
- VIX ticker in yfinance: `^VIX`
- VIX is an index, not tradeable, but has OHLC data

**VIX Volume Handling:**
- yfinance returns Volume column for VIX, but values are 0 or NaN
- **Decision:** Keep Volume column as-is (don't drop, don't fill)
- Document in code that VIX volume is not meaningful
- VIX feature engineering (Task 6) does NOT use volume - all 8 features use only OHLC
- Tests should verify volume is NOT required to be valid

**Tests (all mocked, no live API calls):**
- `test_download_vix_basic`: Mock yfinance, verify VIX parquet written
- `test_vix_data_columns`: Has Date, Open, High, Low, Close (Volume present but not validated)
- `test_vix_data_range`: Synthetic data covers expected date range
- `test_vix_volume_not_required`: Verify download succeeds even with NaN/0 volume

**Dependencies:** Task 1

---

### Task 4: Generalize Feature Engineering Pipeline

**Purpose:** Extend tier_a20.py to process any ticker's OHLCV data

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/features/tier_a20.py` | Parameterize for any ticker | +30 |
| `scripts/build_features_a20.py` | Add `--ticker` argument | +20 |
| `tests/features/test_indicators.py` | Multi-ticker tests | +40 |

**Changes:**
- `build_features(ticker: str, input_path: Path, output_dir: Path)`
- Manifest naming: `{TICKER}.features.a20`
- Keep SPY as default for backwards compatibility

**Tests:**
- `test_build_features_dia`: DIA features computed correctly
- `test_build_features_qqq`: QQQ features computed correctly
- `test_features_same_columns_across_tickers`: All tickers have same 20 features

**Dependencies:** Task 2

---

### Task 5: Build DIA and QQQ Features

**Purpose:** Generate tier a20 features for new assets

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `data/processed/v1/DIA_features_a20.parquet` | DIA features | - |
| `data/processed/v1/QQQ_features_a20.parquet` | QQQ features | - |
| `data/processed/manifest.json` | Updated entries | - |

**Execution:**
```bash
python scripts/build_features_a20.py --ticker DIA
python scripts/build_features_a20.py --ticker QQQ
```

**Dependencies:** Tasks 2, 4

---

### Task 6: VIX Feature Engineering (Tier c)

**Purpose:** Create VIX-based features for feature tier c

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/features/tier_c_vix.py` | VIX feature calculations | 120 |
| `scripts/build_features_vix.py` | CLI for VIX features | 60 |
| `tests/features/test_vix_features.py` | VIX feature tests | 100 |

**VIX Features to Compute:**
1. `vix_close`: Raw VIX close value
2. `vix_sma_10`: 10-day SMA of VIX
3. `vix_sma_20`: 20-day SMA of VIX
4. `vix_percentile_60d`: VIX percentile rank over 60 days
5. `vix_zscore_20d`: Z-score of VIX over 20 days
6. `vix_regime`: High (>25) / Normal (15-25) / Low (<15) categorical
7. `vix_change_1d`: 1-day VIX change
8. `vix_change_5d`: 5-day VIX change

**Output:** Separate VIX features file that can be joined with asset features

**Tests:**
- `test_vix_features_shape`: Output has expected columns
- `test_vix_percentile_range`: Percentile in [0, 100]
- `test_vix_zscore_reasonable`: Z-score in [-4, 4] for most values
- `test_vix_regime_categories`: Only 'high', 'normal', 'low' values

**Dependencies:** Task 3

---

### Task 7: Combined Dataset Builder

**Purpose:** Create utility to merge asset features with VIX features

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/data/combine.py` | Dataset combination utilities | 100 |
| `scripts/build_combined_dataset.py` | CLI for combined datasets | 80 |
| `tests/test_dataset_combined.py` | Combined dataset tests | 80 |

**Functionality:**
- Join asset features (SPY/DIA/QQQ) with VIX features on Date
- Handle date alignment (inner join on trading days)
- Support single-asset or multi-asset combinations
- Output naming: `{TICKER}_dataset_{tier}.parquet`

**Example:**
```bash
# Single asset + VIX (tier c)
python scripts/build_combined_dataset.py --ticker SPY --include-vix --tier c

# Output: data/processed/v1/SPY_dataset_c.parquet
# Contains: OHLCV + 20 indicators + 8 VIX features = 33 features
```

**Tests (all use synthetic DataFrames, no disk dependencies):**
- `test_combine_spy_with_vix`: SPY + VIX features merged correctly
- `test_combine_registers_manifest`: Manifest entry created
- `test_combine_feature_count`: Expected column count (OHLCV + 20 indicators + 8 VIX = 33)

**Date Alignment / Join Validation Tests:**
- `test_combine_inner_join_dates`: Only dates present in BOTH inputs appear in output
- `test_combine_no_nan_after_join`: No NaN values in any feature column after join
- `test_combine_date_range_is_intersection`: Output date range equals intersection of input ranges
- `test_combine_row_count_matches_overlap`: Row count equals number of common trading days
- `test_combine_rejects_no_overlap`: Raises error if inputs have zero date overlap

**Dependencies:** Tasks 5, 6

---

### Task 8: Multi-Asset Dataset Builder (Optional Stretch)

> **⚠️ APPROVAL GATE:** Do not start Task 8 until Tasks 1-7 are complete and passing. Requires explicit user approval before beginning.

**Purpose:** Create datasets that combine multiple assets for cross-asset experiments

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `scripts/build_multi_asset_dataset.py` | Multi-asset CLI | 100 |
| `tests/test_multi_asset.py` | Multi-asset tests | 60 |

**Functionality:**
- Concatenate features from multiple tickers
- Add ticker identifier column
- Support for training on pooled data

**Example:**
```bash
python scripts/build_multi_asset_dataset.py --tickers SPY,DIA,QQQ --tier a20

# Output: data/processed/v1/MULTI_SPY_DIA_QQQ_a20.parquet
```

**Note:** This is a stretch goal. Priority is single-asset datasets with VIX features.

**Dependencies:** Tasks 5, 7

---

## Execution Order

```
Phase 5 Execution Flow
======================

Task 1: Generalize download script
    │
    ├──► Task 2: Download DIA + QQQ
    │       │
    │       └──► Task 4: Generalize feature pipeline
    │               │
    │               └──► Task 5: Build DIA/QQQ features
    │
    └──► Task 3: Download VIX
            │
            └──► Task 6: VIX feature engineering
                    │
                    └──► Task 7: Combined dataset builder
                            │
                            └──► Task 8: Multi-asset builder (optional)
```

---

## Assumptions

1. yfinance provides reliable data for DIA, QQQ, and ^VIX
2. VIX data available from ~1990 (CBOE inception)
3. DIA available from ~1998, QQQ from ~1999
4. Date alignment across assets is straightforward (common trading days)
5. Existing tier_a20.py indicators work for any equity OHLCV data
6. VIX features provide meaningful signal for equity prediction
7. Current manifest system handles multi-asset tracking

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| yfinance rate limiting | Medium | Add retry logic, cache downloads |
| VIX data gaps or anomalies | Low | Validate data quality, document gaps |
| Date misalignment between assets | Medium | Use inner join, validate overlap |
| Feature engineering differs by asset | Low | Test feature distributions per ticker |
| Manifest complexity with many files | Medium | Clear naming convention, automated verification |
| Storage growth with multi-asset data | Low | Parquet compression, monitor disk usage |

---

## Success Criteria

- [ ] `make test` passes with all new tests
- [ ] All expected artifacts exist (see below)
- [ ] All manifest entries registered and verified via `make verify`
- [ ] Can run training on new datasets:
  ```bash
  python scripts/train.py --config configs/daily/threshold_1pct_spy_c.yaml --param-budget 2m
  ```

---

## Expected Artifacts

Phase 5 is complete when ALL of the following files exist and are registered in manifests.

### Raw Data Files

| File Path | Manifest ID | Source | Status |
|-----------|-------------|--------|--------|
| `data/raw/DIA.parquet` | `DIA.OHLCV.daily` | Task 2 | ✅ Complete |
| `data/raw/QQQ.parquet` | `QQQ.OHLCV.daily` | Task 2 | ✅ Complete |
| `data/raw/DJI.parquet` | `DJI.OHLCV.daily` | Task 2 | ✅ Complete |
| `data/raw/IXIC.parquet` | `IXIC.OHLCV.daily` | Task 2 | ✅ Complete |
| `data/raw/VIX.parquet` | `VIX.OHLCV.daily` | Task 3 | ⏸️ Pending |

### Processed Data Files

| File Path | Manifest ID | Source |
|-----------|-------------|--------|
| `data/processed/v1/DIA_features_a20.parquet` | `DIA.features.a20` | Task 5 |
| `data/processed/v1/QQQ_features_a20.parquet` | `QQQ.features.a20` | Task 5 |
| `data/processed/v1/VIX_features_c.parquet` | `VIX.features.c` | Task 6 |
| `data/processed/v1/SPY_dataset_c.parquet` | `SPY.dataset.c` | Task 7 |

### Optional (Task 8 - Stretch Goal)

| File Path | Manifest ID | Source |
|-----------|-------------|--------|
| `data/processed/v1/MULTI_SPY_DIA_QQQ_a20.parquet` | `MULTI.SPY_DIA_QQQ.dataset.a20` | Task 8 |

### Verification Command

```bash
# After Phase 5 completion, this must pass:
make verify

# And these files must exist:
ls data/raw/{DIA,QQQ,VIX}.parquet
ls data/processed/v1/{DIA,QQQ}_features_a20.parquet
ls data/processed/v1/VIX_features_c.parquet
ls data/processed/v1/SPY_dataset_c.parquet
```

---

## New Files Summary

| Task | New/Modified Files | ~Lines |
|------|-------------------|--------|
| 1 | download_ohlcv.py (mod), tests (mod) | +100 |
| 2 | DIA.parquet, QQQ.parquet | data |
| 3 | download_ohlcv.py (mod), VIX.parquet | +50 |
| 4 | tier_a20.py (mod), build_features_a20.py (mod) | +50 |
| 5 | DIA_features_a20.parquet, QQQ_features_a20.parquet | data |
| 6 | tier_c_vix.py, build_features_vix.py, tests | +280 |
| 7 | combine.py, build_combined_dataset.py, tests | +260 |
| 8 | build_multi_asset_dataset.py, tests (optional) | +160 |
| **Total** | ~10-12 files | ~900 lines |

---

## Estimated Effort

| Task | Estimate |
|------|----------|
| Task 1: Generalize download | 1-2 hours |
| Task 2: Download DIA/QQQ | 30 min |
| Task 3: Download VIX | 30 min |
| Task 4: Generalize features | 1-2 hours |
| Task 5: Build DIA/QQQ features | 30 min |
| Task 6: VIX features | 2-3 hours |
| Task 7: Combined builder | 2-3 hours |
| Task 8: Multi-asset (optional) | 1-2 hours |
| **Total** | **8-14 hours** |

---

## Config Examples for New Datasets

### SPY with VIX features (tier c)
```yaml
# configs/daily/threshold_1pct_spy_c.yaml
seed: 42
data_path: data/processed/v1/SPY_dataset_c.parquet
task: threshold_1pct
timescale: daily
context_length: 60
horizon: 5
wandb_project: financial-ts-scaling
mlflow_experiment: financial-ts-scaling
```

### DIA baseline (tier a20)
```yaml
# configs/daily/threshold_1pct_dia_a20.yaml
seed: 42
data_path: data/processed/v1/DIA_features_a20.parquet
task: threshold_1pct
timescale: daily
context_length: 60
horizon: 5
wandb_project: financial-ts-scaling
mlflow_experiment: financial-ts-scaling
```

---

## Approval Gates

Each sub-task requires:
1. Planning confirmation before starting
2. Test plan review
3. RED phase verification (tests fail)
4. GREEN phase verification (tests pass)
5. Commit approval

---

*Document Version: 1.2*
*Author: Claude (Planning Session)*
*Revised: 2025-12-09 - Addressed GPT5-Codex review feedback*
*Pending: User Approval*

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-09 | Initial draft |
| 1.1 | 2025-12-09 | Added Testing Rules section (mock strategy, offline tests); Added retry logic to Task 1; Clarified VIX volume handling in Task 3; Added 5 join validation tests to Task 7; Added Expected Artifacts section with explicit file paths and manifest IDs |
| 1.2 | 2025-12-09 | Added jitter to retry specification; Added explicit approval gate to Task 8 |
| 1.3 | 2025-12-10 | Task 2 expanded to include index tickers (^DJI, ^IXIC) for extended training history; Added Index vs ETF Data section; Updated Expected Artifacts with index files; Task 2 marked complete |
