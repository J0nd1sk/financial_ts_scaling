# Global Project Context - 2026-01-28

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-28 10:15 | tier_a500 Sub-Chunk 8a COMPLETE (323 features, 1306 tests pass) |
| ws2 | foundation | paused | 2026-01-26 12:00 | HPO script created for iTransformer/Informer, awaiting runs |
| ws3 | phase6c_hpo_analysis | paused | 2026-01-27 19:15 | HPO analysis COMPLETE - a50/a100 metrics captured, trends identified |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `6bfd785` feat: Complete HPO analysis + tier_a500 Sub-Chunks 6a/6b/7a
- **Uncommitted**: 10+ files (build scripts, validation scripts, data files, tests, 8a implementation)

### Test Status
- Last `make test`: 2026-01-28 10:10 - **1306 passed**, 2 skipped
- All tests pass (69 new tests for Sub-Chunk 8a)

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined
  - a50: features + combined (55 features)
  - a100: features + combined (105 features)
  - a200: features + combined (206 features)
  - a500: 300 features (generated, validated, registered) - **NEEDS REGENERATION** for 323 features

---

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 tier_a500]: 5/12 sub-chunks complete (323/500 features)
- [ws3 HPO Analysis]: COMPLETE - Comprehensive analysis done, supplementary trials proposed

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | MODIFIED (8a added - 323 features) |
| `tests/features/test_tier_a500.py` | ws1 | MODIFIED (291 tests) |
| `scripts/build_features_a500.py` | ws1 | NEW (uncommitted) |
| `scripts/validate_tier_a500.py` | ws1 | NEW (uncommitted) |
| `tests/script_tests/` | ws1 | NEW (uncommitted) |
| `data/processed/v1/SPY_features_a500.parquet` | ws1 | NEEDS REGENERATION |
| `data/processed/v1/SPY_dataset_a500_combined.parquet` | ws1 | NEEDS REGENERATION |
| `outputs/validation/tier_a500_*.md` | ws1 | NEW (uncommitted) |
| `experiments/templates/hpo_template.py` | ws3 | MODIFIED (verbose=True fix) |
| `docs/hpo_comprehensive_report.md` | ws3 | NEW |
| `docs/supplementary_hpo_proposal.md` | ws3 | NEW |
| `scripts/evaluate_top_hpo_models.py` | ws3 | NEW |

---

## Session Summary (2026-01-28 Morning - ws1)

### Work Completed

1. **tier_a500 Sub-Chunk 8a (TRD Complete)**
   - Wrote 69 failing tests (TDD red phase)
   - Implemented 23 TRD features (TDD green phase):
     - **ADX Extended (5)**: plus_di_14, minus_di_14, adx_14_slope, adx_acceleration, di_cross_recency
     - **Trend Exhaustion (6)**: avg_up/down_day_magnitude, up_down_magnitude_ratio, trend_persistence_20d, up_vs_down_momentum, directional_bias_strength
     - **Trend Regime (5)**: adx_regime, price_trend_direction, trend_alignment_score, trend_regime_duration, trend_strength_vs_vol
     - **Trend Channel (4)**: linreg_slope_20d, linreg_r_squared_20d, price_linreg_deviation, channel_width_linreg_20d
     - **Aroon Extended (3)**: aroon_up_25, aroon_down_25, aroon_trend_strength
   - All 1306 tests pass
   - Feature count: 323 (206 + 117 new)

### tier_a500 Progress
| Sub-Chunk | Features | Status |
|-----------|----------|--------|
| 6a | 24 | COMPLETE (COMMITTED) |
| 6b | 25 | COMPLETE (COMMITTED) |
| 7a | 23 | COMPLETE (COMMITTED) |
| 7b | 22 | COMPLETE (needs commit) |
| 8a | 23 | COMPLETE (needs commit) |
| 8b-11b | ~177 | PENDING |

---

## User Priorities

### ws1 (feature_generation) - Current Focus
1. **Commit current changes** (8a + 7b + data pipeline)
2. **Regenerate data** - Re-run build script for 323 features
3. **Continue with Sub-Chunk 8b** - SR Complete (~22 features)
4. Remaining: 8b, 9a, 9b, 10a, 10b, 11a, 11b

### ws3 (phase6c) - Queued
1. **Decide**: Run 27 supplementary trials or accept current results?
2. **If running**: ~15-30 min to implement and run
3. **If not**: Proceed to Phase 6C conclusions

### ws2 (foundation) - Queued
1. Run iTransformer HPO (50 trials)
2. Run Informer HPO (50 trials)

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

### Hyperparameters (HPO-Validated)
Based on analysis of 250 trials:
- **Dropout**: 0.5 (new finding from comprehensive analysis)
- **Learning Rate**: 1e-4 (confirmed)
- **Weight Decay**: 1e-4 to 1e-3 (higher than expected)
- **d_model**: 128 for 20M budget
- **n_layers**: 2 (shallow) or 6-7 (mid-deep) - bimodal
- **Context**: 80d (unchanged)
- **Normalization**: RevIN only (unchanged)
- **Splitter**: SimpleSplitter (unchanged)

---

## Key Insight

**Scaling laws are VIOLATED for this task:**
- Parameter scaling: 20M beats both 2M and 200M
- Feature scaling: 55 features beats 105 features consistently
- Regularization is critical: High dropout (0.5) dominates top performers
- Recall is the bottleneck: ~10% means missing 90% of opportunities
