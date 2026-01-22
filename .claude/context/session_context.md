# Session Handoff - 2026-01-22 ~01:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `8e06765` docs: head dropout ablation results - no benefit, keep at 0.0
- **Uncommitted changes**:
  - Modified: session_context.md, results.json
  - New: 11 experiment output dirs, threshold_sweep.csv, threshold_sweep_plots.png, threshold_sweep.py, test_threshold_sweep.py
- **Ahead of origin**: 2 commits (not pushed)

### Task Status
- **Completed**: Phase 6A threshold sweep analysis
- **Status**: Ready for Phase 6B/6C feature scaling

---

## Test Status
- Last `make test`: 2026-01-22
- Result: **476 passed**, 2 warnings
- Failing: none

---

## Completed This Session

1. **Session restore** from previous handoff
2. **Phase 6A experiments verified complete** (all 12 models trained)
3. **Threshold sweep script** (`scripts/threshold_sweep.py`):
   - TDD: 5 new tests in `tests/test_threshold_sweep.py`
   - Sweeps probability thresholds 0.1-0.8
   - Computes precision/recall/F1/AUC at each threshold
   - Output: `outputs/phase6a_final/threshold_sweep.csv` (96 rows)
4. **Threshold sweep visualization** (`outputs/phase6a_final/threshold_sweep_plots.png`):
   - Precision vs threshold (all models)
   - Recall vs threshold (all models)
   - Precision-Recall curves by horizon
   - AUC bar chart by budget×horizon
5. **Phase 6A analysis complete** - key findings documented

---

## Key Findings (Phase 6A Complete)

### AUC by Model
| Horizon | 2M | 20M | 200M |
|---------|-----|------|------|
| H1 | 0.707 | 0.717 | 0.724 |
| H2 | 0.640 | 0.639 | 0.642 |
| H3 | 0.621 | 0.618 | 0.631 |
| H5 | 0.609 | 0.599 | 0.613 |

### Key Conclusions
- **Parameter scaling provides minimal benefit** (200M only +1.7% over 2M)
- **Data-limited regime confirmed** - more parameters don't help
- **Shorter horizons easier** (H1 AUC ~0.72 vs H5 AUC ~0.61)
- **H5 models best calibrated** for threshold selection (pred range 0.26-0.90)
- **H1 models poorly calibrated** (predictions rarely exceed 0.5)

---

## Files Modified This Session

- `scripts/threshold_sweep.py` (NEW - 250 lines)
- `tests/test_threshold_sweep.py` (NEW - 5 tests)
- `outputs/phase6a_final/threshold_sweep.csv` (NEW - 96 rows)
- `outputs/phase6a_final/threshold_sweep_plots.png` (NEW - visualization)
- `outputs/phase6a_final/phase6a_*/results.json` (from background experiments)
- `.claude/context/session_context.md` (this file)

---

## Memory Entities Updated

- `ThresholdSweepScript_Plan_20260121` (created): Planning decision for threshold sweep implementation
- `Phase6A_ThresholdSweep_Finding_20260122` (created): Threshold sweep experimental findings
- `Phase6A_Conclusion_DataLimited_20260122` (created): Phase 6A conclusion - data-limited regime

**Still valid from previous sessions:**
- `Finding_2M_HeadCountComparison_20260121` - h=8 best at 2M scale
- `Finding_2Mvs20M_InverseScaling_20260121` - 2M comparable to 20M
- `Target_Calculation_Definitive_Rule` - HIGH-based targets

---

## Data Versions

- Raw manifest: SPY.OHLCV.daily (verified)
- Processed manifest: SPY_dataset_a20.parquet (verified)
- Pending registrations: none

---

## Next Session Should

1. **Begin Phase 6B/6C: Feature Scaling**
   - Expand from 20 indicators to 50/100/200+ features
   - No new data acquisition needed - generate from existing OHLCV
2. **Plan feature tier expansion**:
   - Tier a50: Add more MAs, oscillators, volume indicators
   - Tier a100: Add Fibonacci, Ichimoku, more RSI variants
   - Tier a200+: Comprehensive indicator library
3. **Commit Phase 6A artifacts** (threshold sweep script, results, plots)
4. **Consider**: Which horizons to focus on for feature scaling experiments

---

## Commands to Run First

```bash
source venv/bin/activate
make test
git status
# Commit new threshold sweep artifacts:
git add -A && git commit -m "feat: threshold sweep script and Phase 6A analysis"
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
- **Head dropout**: 0.0 (ablation showed no benefit)
- **Metrics**: AUC, accuracy, precision, recall, pred_range (all required)

### Current Focus
- Phase 6A complete - data-limited regime confirmed
- Next: Feature scaling (more indicators, not more parameters)
