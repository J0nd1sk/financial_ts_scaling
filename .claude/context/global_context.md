# Global Project Context - 2026-01-28

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-28 21:00 | tier_a500 Sub-Chunk 10a COMPLETE (420 features, 1608 tests pass) |
| ws2 | foundation | active | 2026-01-28 15:00 | METHODOLOGY CORRECTION - v1/v2 invalid, v3 design ready |
| ws3 | phase6c_hpo_analysis | paused | 2026-01-27 19:15 | HPO analysis COMPLETE - a50/a100 metrics captured, trends identified |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `208a992` feat: Add tier_a500 Sub-Chunks 8b+9a+9b (72 SR+CDL features)
- **Uncommitted**: 2 files (ws1: 10a implementation)

### Test Status
- Last `make test`: 2026-01-28 21:00 - **1608 passed**, 5 failed (threshold_sweep - unrelated), 2 skipped
- All tier_a500 tests pass (62 new for chunk 10a)

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined
  - a50: features + combined (55 features)
  - a100: features + combined (105 features)
  - a200: features + combined (206 features)
  - a500: 300 features (generated, validated, registered) - **NEEDS REGENERATION** for 420 features

---

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 tier_a500]: 9/12 sub-chunks complete (420/500 features)
- [ws2 foundation]: v2 results INVALID - methodology flaw discovered, v3 ready
- [ws3 HPO Analysis]: COMPLETE - Comprehensive analysis done, supplementary trials proposed

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | MODIFIED (10a added - 420 features) |
| `tests/features/test_tier_a500.py` | ws1 | MODIFIED (559 tests) |
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
| `docs/methodology_lessons_v1_v2.md` | ws2 | NEW (uncommitted) |
| `docs/architecture_hpo_v3_design.md` | ws2 | NEW (uncommitted) |

---

## Session Summary (2026-01-28 - ws1)

### tier_a500 Sub-Chunk 10a Complete

Implemented 25 MTF + entropy + complexity features using TDD:

**Features Added:**
- **Weekly MA (3)**: weekly_ma_slope, weekly_ma_slope_acceleration, price_pct_from_weekly_ma
- **Weekly RSI (2)**: weekly_rsi_slope, weekly_rsi_slope_acceleration
- **Weekly BB (3)**: weekly_bb_position, weekly_bb_width, weekly_bb_width_slope
- **Alignment (3)**: trend_alignment_daily_weekly, rsi_alignment_daily_weekly, vol_alignment_daily_weekly
- **Entropy Extended (8)**: permutation_entropy_slope, permutation_entropy_acceleration, sample_entropy_20d, sample_entropy_slope, sample_entropy_acceleration, entropy_percentile_60d, entropy_vol_ratio, entropy_regime_score
- **Complexity (6)**: hurst_exponent_20d, hurst_exponent_slope, autocorr_lag1, autocorr_lag5, autocorr_partial_lag1, fractal_dimension_20d

**Custom Algorithms Implemented:**
- sample_entropy (m=2, r=0.2)
- hurst_exponent_rs (R/S rescaled range method)
- fractal_dimension_higuchi (Higuchi's method)

**Status:**
- 62 new tests added
- All 1608 tests pass (5 unrelated failures in threshold_sweep)
- Feature count: 420 (395 + 25)

---

## User Priorities

### ws1 (feature_generation) - Active
1. **Commit current changes** (10a)
2. **Regenerate data** - Re-run build script for 420 features
3. **Continue with Sub-Chunk 10b** - ENT Extended (~25 features)
4. Remaining: 10b, 11a, 11b (80 features to go)

### ws2 (foundation) - Queued
1. **Review v3 design** with user
2. **Implement v3 changes** to `hpo_neuralforecast.py`:
   - Change `MAE()` to `DistributionLoss('Bernoulli')`
3. **Run v3 smoke test** (3 trials)
4. **Run full HPO** (50 trials each, iTransformer first)

### ws3 (phase6c) - Queued
1. **Decide**: Run 27 supplementary trials or accept current results?
2. **If running**: ~15-30 min to implement and run
3. **If not**: Proceed to Phase 6C conclusions

---

## User Preferences (Authoritative)

### Metric Priority for Analysis (CRITICAL)
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

Show precision and recall SEPARATELY across threshold changes. The user needs to see the tradeoff curve, not a combined score.

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

## Key Insights

**Scaling laws are VIOLATED for this task:**
- Parameter scaling: 20M beats both 2M and 200M
- Feature scaling: 55 features beats 105 features consistently
- Regularization is critical: High dropout (0.5) dominates top performers
- Recall is the bottleneck: ~10% means missing 90% of opportunities

**Methodology lesson (2026-01-28):**
- Always match training objective to evaluation objective
- AUC can be misleading (measures ranking, not calibration)
- 0% recall = model not learning class separation
