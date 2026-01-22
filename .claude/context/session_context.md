# Session Handoff - 2026-01-21 ~20:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `d2aab51` exp: 2M head count comparison (h=2, h=4, h=8) with HIGH-based 0.5% targets
- **Uncommitted changes**: none (clean working tree)
- **Ahead of origin**: 17 commits (not pushed)

### Task Status
- **Completed this session**: 2M head count experiments (h=4, h=8)
- **Status**: Complete

---

## Test Status
- Last `make test`: 2026-01-21
- Result: **471 passed**, 2 warnings
- Failing: none

---

## Completed This Session

1. **2M h=4 training script** created and trained
2. **2M h=8 training script** created and trained
3. **Threshold sweep** run on all 6 models (2M h2/h4/h8, 20M h2/h4/h8)
4. **Documentation** updated in `docs/threshold_05pct_high_experiments.md`
5. **All changes committed** (d2aab51)

---

## Key Findings This Session

### Optimal Head Count is Scale-Dependent

| Scale | Best Head Count | Head Dim (d_k) | AUC |
|-------|-----------------|----------------|-----|
| 2M (d=64) | h=8 | 8 | 0.713 |
| 20M (d=512) | h=4 | 128 | 0.712 |

- **2M scale prefers MORE heads** with SMALLER attention dimensions
- **20M scale prefers FEWER heads** with LARGER attention dimensions
- Head configuration should NOT be transferred across scales

### Best Model: 2M_h8

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.713 |
| Best Accuracy | 67.96% (@ threshold 0.45) |
| Precision @ 0.45 | 67.65% |
| Recall @ 0.45 | 56.10% |
| Trades @ 0.45 | 68 / 181 |

### Threshold Sweep: 2M_h8 at Higher Thresholds

| Threshold | Precision | Recall | Accuracy | Trades |
|-----------|-----------|--------|----------|--------|
| 0.45 | 67.65% | 56.10% | 67.96% | 68 |
| 0.50 | 69.57% | 39.02% | 64.64% | 46 |
| 0.55 | 71.88% | 28.05% | 62.43% | 32 |
| 0.60 | 70.00% | 17.07% | 59.12% | 20 |

---

## Target Definition (Confirmed)

**Task**: Predict whether tomorrow's HIGH price will be at least 0.5% above today's CLOSE.

**Formula**: `max(HIGH[t+1:t+1+horizon]) >= CLOSE[t] * 1.005`

**Rationale**: Reflects real trading — enter at close, exit when high reaches target.

---

## Pending Experiments

1. **Head Dropout Exploration** - Currently 0.0, may try 0.1-0.3 (low priority)
2. **Different Target Thresholds** - 1%, 2% instead of 0.5%
3. **Longer Horizons** - 3-day, 5-day predictions
4. **1% Threshold experiments** - 19 scripts ready (in other terminal)

---

## Files Modified This Session

- `experiments/threshold_05pct_high/train_2M_narrow_h4.py` (NEW)
- `experiments/threshold_05pct_high/train_2M_narrow_h8.py` (NEW)
- `experiments/threshold_05pct_high/sweep_thresholds.py` (updated)
- `docs/threshold_05pct_high_experiments.md` (comprehensive update)
- `outputs/threshold_05pct_high/2M_narrow_h4_*/` (NEW)
- `outputs/threshold_05pct_high/2M_narrow_h8_*/` (NEW)
- `outputs/threshold_05pct_high/threshold_sweep_results.csv` (updated)

---

## Key Decisions Made

- **Head count at 2M**: Tested h=4 and h=8 to compare with h=2 baseline. Found h=8 best.
- **Parameter matching**: All models trained with identical hyperparameters (dropout=0.5, lr=1e-4, batch=64, epochs=50, ctx=80) for fair comparison.

---

## Memory Entities Updated

- `Finding_2M_HeadCountComparison_20260121` (created): Scale-dependent head count finding
- `Plan_2M_HeadCount_Experiments_20260121` (created): Planning record for this session
- `Pending_2M_HeadCountExperiment` (updated): Marked as complete with results

**Still valid from previous sessions:**
- `Finding_2Mvs20M_InverseScaling_20260121`
- `Finding_ThresholdSweep_05pct_20260121`
- `Critical_TrainerHighPricesBug_20260121`
- `Target_Calculation_Definitive_Rule`
- `Backlog_HeadDropoutExploration`

---

## Data Versions

- Raw manifest: SPY.OHLCV.daily (verified)
- Processed manifest: SPY_dataset_a20.parquet (verified)
- Pending registrations: none

---

## Next Session Should

1. Consider longer horizons (3-day, 5-day) with the 2M_h8 architecture
2. Or try 1% threshold experiments (scripts ready)
3. Or explore head_dropout parameter (currently 0.0)
4. Eventually push commits to origin (17 ahead)

---

## Commands to Run First

```bash
source venv/bin/activate
make test
git status
make verify
```

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Document in multiple places: Memory MCP + context files + docs/
- Code comments are secondary, not primary durability

### Documentation Philosophy
- Flat docs/ structure (no subdirs except research_paper/, archive/)
- Precision in language - never reduce fidelity
- Consolidate rather than delete - preserve historical context

### Communication Standards
- Precision over brevity
- Never summarize away important details
- Evidence-based claims

### Hyperparameters (Fixed - Ablation-Validated)
Always use unless new ablation evidence supersedes:
- **Dropout**: 0.5
- **Learning Rate**: 1e-4
- **Context Length**: 80 days
- **Normalization**: RevIN only (no z-score)
- **Splitter**: SimpleSplitter (442 val samples, not ChunkSplitter's 19)
- **Metrics**: AUC, accuracy, precision, recall, pred_range (all required)

### Current Focus
- 1% threshold experiments with correct HIGH-based targets
- Phase 6A final training with corrected infrastructure
- Building valid experimental evidence
