# Phase 6A Validation Exploration Tracker

**Purpose**: Track systematic exploration of validation + loss + training configurations
**Created**: 2026-01-20
**Full Plan**: `docs/phase6a_validation_exploration_plan.md`

---

## Quick Reference

**Problem**: 19-sample validation → all experiments unreliable
**Goal**: Find working config before HPO re-run
**Target**: AUC >0.60, spread >10%
**Baseline**: RF achieves AUC 0.68-0.82 (proves signal exists)

---

## Phase 1: Validation Strategy Sweep

Hold: BCE loss, val_loss stopping, sigmoid output

| ID | Strategy | Val Samples | Status | Test AUC | Spread | Notes |
|----|----------|-------------|--------|----------|--------|-------|
| V1 | Contiguous 15% | ~19 | ⏳ | | | Baseline |
| V2 | Contiguous 30% | ~38 | ⏳ | | | |
| V3 | Time-based (2021-22) | ~500 | ⏳ | | | |
| V4 | Rolling window | ~125 | ⏳ | | | |

**Winner**: _______________

---

## Phase 2: Loss Function Sweep

Hold: Best V from Phase 1, val_loss stopping

| ID | Loss | Output | Status | Test AUC | Spread | Notes |
|----|------|--------|--------|----------|--------|-------|
| L1 | BCE | Sigmoid | ⏳ | | | Baseline |
| L2 | BCE + pos_weight | Logits | ⏳ | | | |
| L3 | Focal (γ=2) | Logits | ⏳ | | | |
| L4 | SoftAUC | Sigmoid | ⏳ | | | |
| L5 | BCE + SoftAUC | Sigmoid | ⏳ | | | |

**Winner**: _______________

---

## Phase 3: Early Stopping Sweep

Hold: Best V, best L

| ID | Metric | Status | Test AUC | Spread | Notes |
|----|--------|--------|----------|--------|-------|
| E1 | val_loss | ⏳ | | | |
| E2 | val_auc | ⏳ | | | |

**Winner**: _______________

---

## Phase 4: Scale Validation

Hold: Best V, best L, best E

| ID | Budget | Status | Test AUC | Spread | Notes |
|----|--------|--------|----------|--------|-------|
| S1 | 2M | ⏳ | | | Confirm |
| S2 | 20M | ⏳ | | | Scale test |

---

## Implementation Checklist

### Code Changes Needed

- [ ] V3: Add time-based split mode to ChunkSplitter
- [ ] V4: Add rolling window mode to ChunkSplitter
- [ ] L3: Implement FocalLoss class
- [ ] L5: Implement combined BCE+SoftAUC loss
- [ ] Runner: Create experiment sweep script

### Existing (Ready to Use)

- [x] V1, V2: ChunkSplitter contiguous mode (change ratio)
- [x] L1: BCELoss
- [x] L2: BCEWithLogitsLoss (need to wire pos_weight)
- [x] L4: SoftAUCLoss
- [x] E1: val_loss early stopping
- [x] E2: val_auc early stopping (from Test 2)

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

---

## Final Configuration

After exploration complete:

```yaml
validation_strategy: ___
loss_function: ___
early_stopping_metric: ___
output_layer: ___
class_weighting: ___
```

Ready for HPO: [ ] Yes [ ] No - needs ___

---

## Memory Entities

- `Phase6A_ValidationExploration_Plan`
- `Phase6A_ValidationExploration_Results` (create after)
