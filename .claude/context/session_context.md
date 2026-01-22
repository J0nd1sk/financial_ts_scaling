# Session Handoff - 2026-01-21 ~18:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `c90b996` docs: session handoff - Trainer high_prices bug fixed
- **Uncommitted changes**:
  - 19 modified experiment scripts (1% threshold - high_prices added)
  - `experiments/threshold_05pct_high/sweep_thresholds.py` (updated)
  - `experiments/threshold_05pct_high/train_2M_narrow_deep.py` (new)
  - `docs/threshold_05pct_high_experiments.md` (new)
  - `outputs/threshold_05pct_high/2M_narrow_deep_threshold_05pct_HIGH/` (new)
  - `outputs/threshold_05pct_high/threshold_sweep_results.csv` (new)
- **Ahead of origin**: 15 commits (not pushed)

### Task Status
- **Completed this session**:
  - 2M narrow-deep experiment (0.5% HIGH targets)
  - Threshold sweep on 2M + 20M models
  - Full comparison analysis and documentation
- **Next**: 2M head count experiments (h=4, h=8)

---

## 🔴 CRITICAL BUG FIXED (Previous Session)

**Trainer was NOT passing `high_prices` to FinancialDataset.**

All previous threshold experiments were invalid (trained on CLOSE, evaluated on HIGH).

**Fixes applied:**
- `8235281` - Wire high_prices through Trainer
- `18bf655` - Add array length validation
- **471 tests pass**

---

## 0.5% Threshold Results (This Session)

### AUC-ROC Rankings
| Model | AUC | Params | Architecture |
|-------|-----|--------|--------------|
| 20M_h4 | 0.712 | ~20M | d=512, L=6, h=4 |
| **2M_narrow** | **0.707** | ~1.6M | d=64, L=32, h=2 |
| 20M_h8 | 0.697 | ~20M | d=512, L=6, h=8 |
| 20M_h2 | 0.629 | ~20M | d=512, L=6, h=2 |

### Best Configuration for Trading
**2M_narrow at threshold 0.45:**
- Precision: 69.5% (7/10 trades correct)
- Recall: 50% (catches half of opportunities)
- Accuracy: 67.4%
- 59 trades over test period (181 total samples)

### Key Finding: Inverse Scaling Confirmed
- 2M performs comparably to 20M with 12x fewer parameters
- 2M achieves HIGHEST accuracy (67.4%) among all models
- More parameters don't help at this data scale

### Training Hyperparameters (All Models)
- Dropout: 0.5
- Learning Rate: 1e-4
- Head Dropout: 0.0
- Epochs: 50 (early stopping on val_auc)
- Context Length: 80 days

---

## Pending Experiments

### 2M Head Count Comparison (NEXT)
- Only tested h=2 at 2M scale so far
- Need: h=4 and h=8 variants for fair comparison
- Architecture: d=64, L=32, h={4,8}
- Keep: dropout=0.5, lr=1e-4, 0.5% HIGH targets

### 1% Threshold Experiments (Other Terminal)
- 19 scripts updated with high_prices
- Ready to run after commit

---

## Test Status
- Last `make test`: 2026-01-21
- Result: **471 passed**, 2 warnings
- Failing: none

---

## Memory Entities Updated

**This session:**
- `Finding_2Mvs20M_InverseScaling_20260121` - 2M vs 20M comparison
- `Finding_ThresholdSweep_05pct_20260121` - Threshold sweep results
- `Pending_2M_HeadCountExperiment` - Next experiment to run
- `Backlog_HeadDropoutExploration` - Future experiment idea

**Still valid:**
- `Critical_TrainerHighPricesBug_20260121` - Bug and fix documentation
- `Target_Calculation_Definitive_Rule` - Canonical target definition

---

## Documentation

- `docs/threshold_05pct_high_experiments.md` - Full experimental writeup (NEW)

---

## Next Session Should

1. **Create 2M h=4 training script**: d=64, L=32, h=4
2. **Create 2M h=8 training script**: d=64, L=32, h=8
3. **Train both models** with same hyperparameters
4. **Run threshold sweep** on new models
5. **Compare all 2M variants** (h=2, h=4, h=8)
6. **Commit all changes** if experiments succeed

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status

# Then create and run 2M h=4 experiment
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

### Current Focus
- 2M head count experiments (h=4, h=8)
- Comparing architecture choices at smaller scale
- Building valid experimental evidence with correct HIGH-based targets
