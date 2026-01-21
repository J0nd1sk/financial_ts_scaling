# Session Handoff - 2026-01-21 ~16:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `c90b996` docs: session handoff - Trainer high_prices bug fixed
- **Uncommitted changes**: 19 modified experiment scripts + scripts from other session
- **Ahead of origin**: 15 commits (not pushed)

### Task Status
- **Completed**: Added `high_prices` to 19 scripts for 1% threshold re-runs
- **Other Terminal**: Running 0.5% threshold experiments
- **Next**: Commit changes, run 1% threshold experiments

---

## 🔴 CRITICAL BUG FIXED (Previous Session)

**Trainer was NOT passing `high_prices` to FinancialDataset.**

All previous threshold experiments were invalid (trained on CLOSE, evaluated on HIGH).

**Fixes applied:**
- `8235281` - Wire high_prices through Trainer
- `18bf655` - Add array length validation
- **471 tests pass**

---

## Changes Made THIS Session (1% Threshold Scripts)

Added `high_prices=df["High"].values` to 19 experiment scripts.

### Scripts Updated (19 total)

**Context Length Ablation (6):**
- `experiments/context_length_ablation/train_ctx60.py`
- `experiments/context_length_ablation/train_ctx80.py`
- `experiments/context_length_ablation/train_ctx90.py`
- `experiments/context_length_ablation/train_ctx120.py`
- `experiments/context_length_ablation/train_ctx180.py`
- `experiments/context_length_ablation/train_ctx252.py`

**Phase 6A Final - 2M (4):**
- `experiments/phase6a_final/train_2M_h{1,2,3,5}.py`

**Phase 6A Final - 20M (4):**
- `experiments/phase6a_final/train_20M_h{1,2,3,5}.py`

**Phase 6A Final - 200M (4):**
- `experiments/phase6a_final/train_200M_h{1,2,3,5}.py`

**RevIN Comparison (1):**
- `scripts/test_revin_comparison.py`

### Change Pattern (same for all 19 scripts)
1. Extract: `high_prices = df["High"].values` after loading data
2. Pass to Trainer: `high_prices=high_prices,`

---

## 0.5% Threshold Results (From Other Terminal)

### 20M Wide Architecture Comparison

| Model | Val AUC | Test AUC | Test Acc | Test Prec | Test Recall | Test F1 |
|-------|---------|----------|----------|-----------|-------------|---------|
| **h=4** | 0.606 | 0.690 | **0.646** | **0.688** | **0.402** | **0.508** |
| h=8 | 0.604 | **0.700** | 0.608 | 0.649 | 0.293 | 0.403 |
| h=2 | 0.602 | 0.630 | 0.580 | 0.552 | 0.390 | 0.457 |

**Key findings:**
- **h=4 wins** on accuracy, precision, recall, F1
- **h=8 wins** on AUC-ROC but has lowest recall
- **No probability collapse** - prediction spreads are healthy
- **Class balance ~50%** confirmed (was 29% with incorrect CLOSE-based targets)

---

## Test Status
- Last `make test`: 2026-01-21 ~16:30 UTC
- Result: **471 passed**, 2 warnings

---

## Next Session Should

1. **Commit the 19 modified scripts**
   ```bash
   git add -A
   git commit -m "fix: add high_prices to experiment scripts for correct HIGH-based targets"
   ```

2. **Run 1% threshold experiments** (user choice of order):
   - RevIN comparison (3 configs) - ~10 min each
   - Context length ablation (6 runs) - ~30 min each
   - Phase 6A final 2M/20M/200M (12 runs) - varies by model size

3. **Report metrics**: AUC, accuracy, precision, recall, F1

4. **Compare to previous (invalid) results**

---

## Memory Entities

- `Critical_TrainerHighPricesBug_20260121` - Documents the bug and fix
- `Target_Calculation_Definitive_Rule` - Canonical target definition

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
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
- Re-run 1% threshold experiments with correct HIGH-based targets
- Full metrics reporting (AUC, accuracy, precision, recall, F1)
- Compare to previous (invalid) results
