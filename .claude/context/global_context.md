# Global Project Context - 2026-01-28

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-28 19:30 | tier_a500 Sub-Chunk 9b COMPLETE (395 features, 1530 tests pass) |
| ws2 | foundation | active | 2026-01-28 15:00 | METHODOLOGY CORRECTION - v1/v2 invalid, v3 design ready |
| ws3 | phase6c_hpo_analysis | paused | 2026-01-27 19:15 | HPO analysis COMPLETE - a50/a100 metrics captured, trends identified |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `ca9c4b0` feat: Add tier_a500 Sub-Chunks 7b+8a (45 VLM+TRD features)
- **Uncommitted**: 15+ files (ws1: 8b+9a+9b implementation; ws2: methodology docs)

### Test Status
- Last `make test`: 2026-01-28 19:30 - **1530 passed**, 5 failed (threshold_sweep - unrelated), 2 skipped
- All tier_a500 tests pass

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined
  - a50: features + combined (55 features)
  - a100: features + combined (105 features)
  - a200: features + combined (206 features)
  - a500: 300 features (generated, validated, registered) - **NEEDS REGENERATION** for 395 features

---

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 tier_a500]: 8/12 sub-chunks complete (395/500 features)
- [ws2 foundation]: v2 results INVALID - methodology flaw discovered, v3 ready
- [ws3 HPO Analysis]: COMPLETE - Comprehensive analysis done, supplementary trials proposed

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | MODIFIED (9b added - 395 features) |
| `tests/features/test_tier_a500.py` | ws1 | MODIFIED (497 tests) |
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

### tier_a500 Sub-Chunk 9b Complete

Implemented 25 candlestick pattern features using TDD:

**Features Added:**
- **Doji Patterns (5)**: doji_strict_indicator, doji_score, doji_type, consecutive_doji_count, doji_after_trend
- **Marubozu & Strong Candles (4)**: marubozu_indicator, marubozu_direction, marubozu_strength, consecutive_strong_candles
- **Spinning Top & Indecision (4)**: spinning_top_indicator, spinning_top_score, indecision_streak, indecision_at_extreme
- **Multi-Candle Reversal (5)**: morning_star_indicator, evening_star_indicator, three_white_soldiers, three_black_crows, harami_indicator
- **Multi-Candle Continuation (4)**: piercing_line, dark_cloud_cover, tweezer_bottom, tweezer_top
- **Pattern Context (3)**: reversal_pattern_count_5d, pattern_alignment_score, pattern_cluster_indicator

**Note:** Renamed `doji_indicator` to `doji_strict_indicator` to avoid conflict with tier_a200.

**Status:**
- 64 new tests added
- All 1530 tests pass (5 unrelated failures in threshold_sweep)
- Feature count: 395 (370 + 25)

---

## User Priorities

### ws1 (feature_generation) - Active
1. **Commit current changes** (7b + 8a + 8b + 9a + 9b)
2. **Regenerate data** - Re-run build script for 395 features
3. **Continue with Sub-Chunk 10a** - MTF Complete (~25 features)
4. Remaining: 10a, 10b, 11a, 11b (105 features to go)

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
