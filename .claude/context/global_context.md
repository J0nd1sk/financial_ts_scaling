# Global Project Context - 2026-01-26

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-26 13:30 | tier_a200 ALL CHUNKS COMPLETE (uncommitted) - 106 new indicators, 206 total |
| ws2 | foundation | **HPO Ready** | 2026-01-26 12:00 | HPO script created for iTransformer/Informer, smoke test passed |
| ws3 | phase6c | **Methodology Done** | 2026-01-26 14:30 | All 4 methodology tasks complete, ready to run improved HPO experiments |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `eee3c27` feat: Add tier_a200 Chunks 3-4 with 40 new features (ranks 141-180)
- **Uncommitted**:
  - `experiments/architectures/hpo_neuralforecast.py` - NEW (ws2 HPO script)
  - `experiments/architectures/common.py` - MODIFIED (ws2 HPO utilities)
  - `src/features/tier_a200.py` - Chunks 1-5 implementation (ws1)
  - `tests/features/test_tier_a200.py` - Chunks 1-5 tests (ws1)
  - `scripts/analyze_hpo_coverage.py` - NEW (ws3 coverage analysis)
  - `scripts/analyze_hpo_results.py` - NEW (ws3 analysis pipeline)
  - `experiments/templates/hpo_template.py` - NEW (ws3 improved HPO methodology)
  - `src/evaluation/__init__.py` - NEW (ws3 module init)
  - `src/evaluation/calibration.py` - NEW (ws3 calibration classes)
  - `tests/test_calibration.py` - NEW (ws3 26 calibration tests)

### Test Status
- Last `make test`: 2026-01-26 14:30 - **970 passed**, 2 skipped
- All tests pass (944 existing + 26 new calibration tests)

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20, a50, a100 (v1) - both features-only and _combined versions
- tier_a200: Module COMPLETE (all 5 chunks, 106 additions, 206 total), no processed data yet

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 HPO 2M Done]: Best config found (d_model=96, layers=2, heads=8, lr=1e-5, dropout=0.1, wd=0.001)
- [ws3 Next]: Need 20M and 200M HPO runs
- [ws2 HPO Ready]: Script created, need to run 50-trial HPO for iTransformer and Informer
- [ws1 Complete]: tier_a200 ready to commit, can generate processed data for Phase 6C

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a200.py` | ws1 (MODIFIED - Chunks 1-5 COMPLETE) |
| `tests/features/test_tier_a200.py` | ws1 (MODIFIED - Chunks 1-5 COMPLETE) |
| `experiments/architectures/hpo_neuralforecast.py` | ws2 (NEW) |
| `experiments/architectures/common.py` | ws2 (MODIFIED) |
| `outputs/hpo/architectures/` | ws2 (NEW) |
| `experiments/phase6c_a100/*` | ws3 |
| `outputs/phase6c_a100/hpo_*` | ws3 |
| `scripts/analyze_hpo_coverage.py` | ws3 (NEW) |
| `scripts/analyze_hpo_results.py` | ws3 (NEW) |
| `experiments/templates/hpo_template.py` | ws3 (NEW) |
| `src/evaluation/` | ws3 (NEW - calibration module) |
| `tests/test_calibration.py` | ws3 (NEW) |

---

## ws2 Session Summary (2026-01-26 12:00)

### Alternative Architecture HPO Script Created

**Motivation**: Original iTransformer/Informer experiments ran with ONE configuration each - no dropout tuning, no LR exploration, only 500 steps. This is not a fair comparison to PatchTST which went through 50+ trials of HPO.

**Key insight from PatchTST**: dropout=0.5 was critical for preventing probability collapse. We never tried high dropout on alternative architectures.

**Files Created**:
1. `experiments/architectures/hpo_neuralforecast.py` (~400 lines)
   - Optuna-based HPO with TPE sampler (20 startup trials)
   - Supports both `--model itransformer` and `--model informer`
   - `--dry-run` and `--resume` flags
   - SQLite study storage, incremental result saving

2. Updated `experiments/architectures/common.py` with HPO utilities

3. Created `outputs/hpo/architectures/{itransformer,informer}/trials/`

**Bug Fixed**: NeuralForecast `val_size` must be passed to `nf.fit()`, not model constructor.

**Smoke Test**: 3 trials in 0.9 min - script works, but AUC=0 (probability collapse persists with random params).

---

## tier_a200 Completion Summary (ws1)

### All Chunks Complete
| Chunk | Ranks | Features | Categories |
|-------|-------|----------|------------|
| 1 | 101-120 | 20 | TEMA, WMA, KAMA, HMA, VWMA, derived |
| 2 | 121-140 | 20 | Duration Counters, MA Cross Recency, Proximity |
| 3 | 141-160 | 20 | BB Extension, RSI Duration, Mean Reversion, Patterns |
| 4 | 161-180 | 20 | MACD Extensions, Volume Dynamics, Calendar, Candle |
| 5 | 181-206 | 26 | Ichimoku, Donchian, Divergence, Entropy/Regime |

**Total**: 106 new features + 100 from tier_a100 = **206 features**

---

## HPO 2M Analysis Summary (ws3)

### Best Configuration Found
```
d_model=96, n_layers=2, n_heads=8, d_ff_ratio=4
learning_rate=1e-5, dropout=0.1, weight_decay=0.001
AUC-ROC: 0.7178 (vs baseline 0.7049 = +1.29%)
```

### Key Findings

1. **HPO beats baseline**: 0.7178 > 0.7049 (+1.29 AUC points)
2. **20M still better**: Reference 20M model achieves 0.7342 AUC
3. **Probability collapse**: Predictions in [0.505, 0.662] range (narrow!)
4. **Max precision ~41%**: At threshold 0.60 with only 16% recall
5. **Dropout 0.1 >> 0.5**: Contradicts previous ablation (tier_a100 features different?)
6. **Weight decay helps**: 0.001 > 0.0001 > 1e-5 > 0.0
7. **Slower LR better**: 1e-5 optimal, may benefit from even slower

### Gaps in Search Space
- 2-head NOT tested with best architecture config
- No dropout values 0.05, 0.15, 0.2 tested
- No weight decay values 0.005, 0.01 tested
- No LR values 5e-6, 2e-6 tested

---

## User Priorities (from recent sessions)

### ws2 (foundation) - Architecture HPO
1. **Run iTransformer HPO (50 trials)**
   ```bash
   caffeinate -i ./venv/bin/python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 50 2>&1 | tee outputs/hpo/architectures/itransformer/hpo.log
   ```

2. **Run Informer HPO (50 trials)**
   ```bash
   caffeinate -i ./venv/bin/python experiments/architectures/hpo_neuralforecast.py --model informer --trials 50 2>&1 | tee outputs/hpo/architectures/informer/hpo.log
   ```

3. **Decision point**: AUC >= 0.70 → horizon experiments; AUC < 0.65 → close investigation

### ws3 (phase6c) - PatchTST HPO (Methodology Complete)
**All 4 methodology tasks done. Ready for execution.**

1. **Run improved HPO** with 2-phase strategy (6 forced extremes + 44 TPE):
   ```bash
   caffeinate -i ./venv/bin/python experiments/templates/hpo_template.py \
     --budget 20M --tier a100 --horizon h1 --trials 50

   caffeinate -i ./venv/bin/python experiments/templates/hpo_template.py \
     --budget 200M --tier a100 --horizon h1 --trials 50
   ```

2. **Cross-budget validation**: Test best configs from each budget on others

3. **Apply calibration** (module ready): `from src.evaluation import PlattScaling, expected_calibration_error`

4. **Final analysis**: `./venv/bin/python scripts/analyze_hpo_results.py --all`

**Memory MCP**: HPO_Methodology_Phase6C, Calibration_Module entities stored

### ws1 (feature_generation) - Ready to Commit
1. Commit tier_a200 files (106 new features, 206 total)
2. Generate processed tier_a200 parquet files if needed for experiments

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

**Probability collapse is the core issue** - models achieve decent AUC (ranking) but poor calibration (probability meaningfulness). User notes slowing down learning and adding regularization helps. This aligns with HPO finding lr=1e-5 and wd=0.001 optimal.
