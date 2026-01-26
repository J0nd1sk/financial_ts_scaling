# Global Project Context - 2026-01-26

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-26 01:45 | tier_a200 Chunks 1-4 COMPLETE (uncommitted) - 80 new indicators, 180 total |
| ws2 | foundation | **COMPLETE** | 2026-01-25 15:00 | Foundation models AND alt architectures FAILED vs PatchTST; decoder experiments PENDING |
| ws3 | phase6c | **Audit Phase 1 Done** | 2026-01-26 00:30 | Deep audit Phase 1 COMPLETE - recall/precision added, exception handling fixed |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `268e328` feat: Expand HPO search space to include training hyperparameters
- **Uncommitted**:
  - `src/features/tier_a200.py` - Chunks 1-4 implementation (80 new features)
  - `tests/features/test_tier_a200.py` - Chunks 1-4 tests (~80 new tests for Chunk 4)
  - Various outputs JSON files

### Test Status
- Last `make test`: 2026-01-26 01:40 - **891 passed**, 2 skipped
- All tests pass including tier_a200 (Chunks 1-4)

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20, a50, a100 (v1) - both features-only and _combined versions
- tier_a200: Module complete (Chunks 1-4, 80 additions), no processed data yet

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 Phase 1 DONE]: Critical bugs fixed (recall/precision, exception handling, paths)
- [ws3 Phase 2 PENDING]: HPO search space expansion still needed before overnight runs

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a200.py` | ws1 (MODIFIED - Chunks 1-4) |
| `tests/features/test_tier_a200.py` | ws1 (MODIFIED - Chunks 1-4) |
| `experiments/foundation/` | ws2 |
| `experiments/architectures/` | ws2 |
| `outputs/foundation/*.json` | ws2 |
| `outputs/architectures/` | ws2 |
| `docs/foundation_model_results.md` | ws2 |
| `docs/architecture_comparison_results.md` | ws2 |
| `experiments/phase6c_a100/*` | ws3 |

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

## Recent Findings (ws2) - CRITICAL

### Foundation Model Investigation Complete
See `docs/foundation_model_results.md` for full analysis.

**Key Result**: Foundation models significantly underperform PatchTST:
- PatchTST 200M: **0.718 AUC** (baseline)
- TimesFM (inverted): 0.636 AUC (-11%)
- Lag-Llama (best): 0.576 AUC (-20%)

**Critical Discovery - Covariates Ignored**:
- TFM-07 (50 features) predictions **identical** to TFM-01 (1 feature)
- TimesFM completely ignores feature engineering

### Alternative Architecture Investigation Complete
See `docs/architecture_comparison_results.md` for full analysis.

**Key Result**: Alternative architectures significantly underperform PatchTST:
- PatchTST 200M: **0.718 AUC** (baseline)
- iTransformer: 0.517 AUC (-28%) - barely above random
- Informer: 0.587 AUC (-18%) - probability collapse

**Root Causes**:
- iTransformer's inverted (feature-wise) attention loses temporal patterns
- Informer's forecasting→threshold approach fails for classification
- Both show narrow prediction ranges (collapsed to mean)

**Recommendation**: Abandon foundation/architecture investigation, focus on Phase 6C with PatchTST.

**Note (ws2)**: Foundation model investigation COMPLETE, but decoder architecture experiments still PENDING if user wants to explore.

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

### tier_a200 Chunk 4 (2026-01-26 01:45)
Implemented 20 new features (ranks 161-180):

**MACD Extensions (4)**: Signal line, histogram slope, cross recency, proximity
**Volume Dynamics (4)**: Trend, consecutive increase, confluence, direction bias
**Calendar/Temporal (6)**: Day of week, Monday/Friday flags, month end, quarter end
**Candle Analysis (6)**: Body %, wick %, doji indicator, range vs average

### tier_a200 Cumulative Status
- **Total additions**: 80 features (Chunks 1-4)
- **Total features**: 180 (100 a100 + 80 new)
- **Tests**: 891 passed, all tier_a200 tests pass
- **Status**: Ready to commit Chunks 1-4

---

## Next Session Notes

### ws1 Ready to Commit
tier_a200 Chunks 1-4 complete with 80 new features. Ready to:
1. Commit the implementation
2. Plan Chunk 5 (ranks 181-200) or pivot to other priorities

### ws2 Note
Foundation models COMPLETE but decoder experiments still available if user wants to explore alternative approaches.

### Decision Points
- Close ws2 investigation entirely?
- Focus exclusively on PatchTST + feature scaling?
- Proceed with tier_a200 Chunk 5 or use 180-feature tier for Phase 6C experiments?
