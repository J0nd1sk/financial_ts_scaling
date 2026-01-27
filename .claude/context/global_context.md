# Global Project Context - 2026-01-27

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-27 16:30 | tier_a500 Sub-Chunk 6a COMPLETE (24 features, 230 total) |
| ws2 | foundation | paused | 2026-01-26 12:00 | HPO script created for iTransformer/Informer, awaiting runs |
| ws3 | phase6c_hpo_data | paused | 2026-01-26 15:45 | a200 combined dataset built & validated, HPO methodology complete |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `4f9df5a` feat: Add tier_a200 build script
- **To commit this session (ws1)**:
  - `src/features/tier_a500.py` - NEW (24 features, Sub-Chunk 6a)
  - `tests/features/test_tier_a500.py` - NEW (56 tests)

### Test Status
- Last `make test`: 2026-01-27 - **1062 passed**, 2 skipped
- All tests pass (56 new tests for tier_a500)

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined ✅
  - a50: features + combined ✅
  - a100: features + combined ✅ (naming inconsistent - no `_combined` suffix)
  - a200: features ✅ + combined ✅
  - a500: IN PROGRESS (Sub-Chunk 6a complete, 11 more chunks pending)

### Latest Manifest Entries
```
SPY.features.a200: md5=01332ae16031805f0358ee7a0ca44039
SPY.dataset.a200_combined: md5=78f0f408f0943c8ca6a11901da0ce7c5
```

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 a200 Data]: ✅ COMPLETE - `SPY_dataset_a200_combined.parquet` ready for experiments
- [ws3 HPO Methodology]: ✅ COMPLETE - All 4 improvements implemented
- [ws2 HPO Ready]: Script created, need to run 50-trial HPO for iTransformer and Informer
- [ws1 tier_a500]: IN PROGRESS - 1/12 sub-chunks complete

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a500.py` | ws1 (NEW) |
| `tests/features/test_tier_a500.py` | ws1 (NEW) |
| `src/features/tier_a200.py` | ws1 (COMMITTED) |
| `scripts/validate_parquet_file.py` | ws3 |
| `experiments/architectures/hpo_neuralforecast.py` | ws2 |

---

## Session Summary (2026-01-27 - ws1)

### Work Completed
1. **Created tier_a500 module skeleton**
   - `src/features/tier_a500.py` - extends tier_a200
   - CHUNK_6A_FEATURES (24 items), A500_ADDITION_LIST, FEATURE_LIST

2. **Wrote 56 tests for Sub-Chunk 6a (TDD red phase)**
   - Feature list structure tests
   - Computation tests for all 24 features
   - Integration tests

3. **Implemented Sub-Chunk 6a (TDD green phase)**
   - 4 new SMA periods: sma_5, sma_14, sma_21, sma_63
   - 5 new EMA periods: ema_5, ema_9, ema_50, ema_100, ema_200
   - 6 MA slopes: sma_5_slope, sma_21_slope, sma_63_slope, ema_9_slope, ema_50_slope, ema_100_slope
   - 5 price distances: price_pct_from_sma_5, price_pct_from_sma_21, price_pct_from_ema_9, price_pct_from_ema_50, price_pct_from_ema_100
   - 4 MA proximities: sma_5_21_proximity, sma_21_50_proximity, sma_63_200_proximity, ema_9_50_proximity

4. **Fixed test fixture** - 400 days instead of 300 (sufficient warmup for 252-day indicators)

**Current Feature Count**: 206 (a200) + 24 (Chunk 6a) = **230 features**

---

## tier_a500 Progress

| Sub-Chunk | Ranks | Features | Status |
|-----------|-------|----------|--------|
| **6a** | 207-230 | 24 | ✅ COMPLETE |
| 6b | 231-255 | ~25 | PENDING |
| 7a | 256-278 | ~23 | PENDING |
| 7b | 279-300 | ~22 | PENDING |
| 8a | 301-323 | ~23 | PENDING |
| 8b | 324-345 | ~22 | PENDING |
| 9a | 346-370 | ~25 | PENDING |
| 9b | 371-395 | ~25 | PENDING |
| 10a | 396-420 | ~25 | PENDING |
| 10b | 421-445 | ~25 | PENDING |
| 11a | 446-472 | ~27 | PENDING |
| 11b | 473-500 | ~28 | PENDING |

**Target**: 500 features total (206 + 294 new)

---

## User Priorities

### ws1 (feature_generation) - Current Focus
1. ✅ Sub-Chunk 6a complete (24 features)
2. Continue with Sub-Chunk 6b (MA Durations/Crosses + OSC Extended)
3. TDD cycle: tests first → implementation

### ws3 (phase6c) - Queued
1. Continue HPO experiments on a100 data
2. Rename a100 file after HPO complete
3. Consider a200 experiments (data now ready)

### ws2 (foundation) - Queued
1. Run iTransformer HPO (50 trials)
2. Run Informer HPO (50 trials)
3. Decision point: AUC >= 0.70 → horizon experiments

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

### Hyperparameters (HPO-Validated, Replacing Ablation)
- **Dropout**: 0.1 (HPO found better than 0.5 for tier_a100)
- **Learning Rate**: 1e-5 (slower is better)
- **Weight Decay**: 0.001 (regularization helps)
- **Context**: 80d (unchanged)
- **Normalization**: RevIN only (unchanged)
- **Splitter**: SimpleSplitter (unchanged)

---

## Key Insight

**Probability collapse is the core issue** - models achieve decent AUC (ranking) but poor calibration (probability meaningfulness). Slowing down learning and adding regularization helps. This aligns with HPO finding lr=1e-5 and wd=0.001 optimal.
