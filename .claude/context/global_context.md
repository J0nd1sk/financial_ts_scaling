# Global Project Context - 2026-01-25

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-25 23:30 | tier_a200 Chunks 1-3 COMPLETE (uncommitted) - 60 new indicators, 160 total |
| ws2 | foundation | **COMPLETE** | 2026-01-26 09:00 | **FINDING**: TimesFM ignores covariates - predictions identical with 1 vs 50 features |
| ws3 | phase6c | **Audit Phase 1 Done** | 2026-01-26 00:30 | Deep audit Phase 1 COMPLETE - recall/precision added, exception handling fixed, paths fixed |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `43da65b` feat: Add tier_a200 Chunk 2 with 20 duration/cross/proximity indicators
- **Uncommitted**:
  - `src/features/tier_a200.py` - Chunk 3 implementation (60 total features)
  - `tests/features/test_tier_a200.py` - Chunk 3 tests (840 total tests)
  - `outputs/phase6c_a100/hpo_200m_h1/` (partial HPO run)

### Test Status
- Last `make test`: 2026-01-25 23:25 - **840 passed**, 2 skipped
- 842 total tests collected
- All tier_a200 tests pass (Chunks 1-3)

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20, a50, a100 (v1) - both features-only and _combined versions
- tier_a200: Module complete (Chunks 1-3), no processed data yet

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 Phase 1 DONE]: Critical bugs fixed (recall/precision, exception handling, paths)
- [ws3 Phase 2 PENDING]: HPO search space expansion still needed before overnight runs

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a200.py` | ws1 (MODIFIED - Chunk 3) |
| `tests/features/test_tier_a200.py` | ws1 (MODIFIED - Chunk 3) |
| `experiments/foundation/TimesFM_*.ipynb` | ws2 |
| `outputs/foundation/*.json` | ws2 (RESULTS) |
| `docs/foundation_model_results.md` | ws2 (NEW) |
| `experiments/phase6c_a100/*` | ws3 (NEEDS FIXES) |
| `scripts/run_s1_a100.sh` | ws3 (WORKING) |
| `scripts/run_hpo_a100.sh` | ws3 (NEEDS FIX) |

---

## Data File Naming Convention (Important!)

Two versions of processed datasets exist:
- `SPY_dataset_a50.parquet` - Features ONLY (no OHLCV)
- `SPY_dataset_a50_combined.parquet` - OHLCV + Features

**For experiments**: Always use `_combined` versions!

**Feature count**: a100_combined has **105 features** (100 indicators + 5 OHLCV), NOT 100!

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Multiple places: Memory MCP + context files + docs/
- Code comments are secondary

### Documentation Philosophy
- Flat docs/ (no subdirs except research_paper/, archive/)
- Precision - never reduce fidelity
- Consolidate rather than delete

### Hyperparameters (Fixed - Ablation-Validated)
- Dropout: 0.5, LR: 1e-4, Context: 80d, RevIN only, SimpleSplitter

---

## Recent Findings (ws2 Foundation Models) - CRITICAL

### Foundation Model Investigation Complete
See `docs/foundation_model_results.md` for full analysis.

**Key Result**: Foundation models significantly underperform PatchTST:
- PatchTST 200M: **0.718 AUC** (baseline)
- TimesFM (inverted): 0.636 AUC (-11%)
- Lag-Llama (best): 0.576 AUC (-20%)

**Critical Discovery - Covariates Ignored**:
- TFM-07 (50 features) predictions **identical** to TFM-01 (1 feature)
- Correlation: 1.0000000000
- Max difference: 8.15e-09 (floating point noise only)
- TimesFM completely ignores feature engineering

**Recommendation**: Abandon foundation model path, focus on Phase 6C with PatchTST.

---

## Recent Findings (ws3 Phase 6C)

### S1 Baseline Results
- No clear scaling benefit observed
- H1: Slight inverse scaling (2M ≈ 20M > 200M)
- H3/H5: Marginal 200M advantage

### Precision-Recall Tradeoff
- 90% precision → only 4% recall
- 75% precision → 23% recall
- High precision trading = catching very few opportunities

---

## Recent Work Summary (ws1)

### tier_a200 Chunk 3 (2026-01-25 23:30)
Implemented 20 new features (ranks 141-160):

**BB Extension (6)**: Distance and duration outside Bollinger Bands
- pct_from_upper/lower_band, days_above_upper/below_lower_band
- bb_squeeze_indicator (BB inside Keltner), bb_squeeze_duration

**RSI Duration (4)**: RSI-based momentum duration
- rsi_distance_from_50, days_rsi_overbought/oversold, rsi_percentile_60d

**Mean Reversion (6)**: Statistical extension and 52-week features
- zscore_from_20d/50d_mean, percentile_in_52wk_range
- distance_from_52wk_high_pct, days_since_52wk_high/low

**Consecutive Patterns (4)**: Price movement patterns
- consecutive_up/down_days, up_days_ratio_20d, range_compression_5d

### tier_a200 Cumulative Status
- **Total additions**: 60 features (Chunks 1-3)
- **Total features**: 160 (100 a100 + 60 new)
- **Tests**: 840 passed, all tier_a200 tests pass
- **Status**: Ready to commit Chunks 1-3
