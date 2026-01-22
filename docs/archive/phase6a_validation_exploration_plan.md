# Phase 6A Validation & Training Exploration Plan

> **TEMPORARY DOCUMENT** - Active experiment plan.
> Archive after exploration complete.

**Date**: 2026-01-20
**Status**: PLANNING
**Goal**: Systematically explore validation strategies, loss functions, and training configurations to find a working baseline before HPO re-run.

---

## Problem Statement

Current 2M_h1 transformer achieves AUC 0.57 with prior collapse, while Random Forest achieves AUC 0.68-0.82 on the same data. Key issues identified:

1. **Validation set too small**: 19 samples (contiguous ChunkSplitter mode)
2. **Loss function**: BCE allows prior collapse; SoftAUC untested with proper validation
3. **Early stopping**: Val_loss metric may not correlate with test AUC
4. **Architecture**: Sigmoid in forward pass, no class weighting

**Hypothesis**: With proper validation and training setup, transformers can approach RF performance.

---

## Experiment Design

### Factors to Explore

| Factor | Options | Priority |
|--------|---------|----------|
| **Validation Strategy** | 4 options | 🔴 Critical |
| **Loss Function** | 5 options | 🔴 Critical |
| **Early Stopping Metric** | 2 options | 🟡 Important |
| **Output Layer** | 2 options | 🟡 Important |
| **Class Weighting** | 2 options | 🟡 Important |

### Factor Details

#### A. Validation Strategy (4 options)

| ID | Strategy | Val Samples (est.) | Implementation |
|----|----------|-------------------|----------------|
| V1 | Current (val_ratio=0.15, contiguous) | ~19 | Baseline |
| V2 | Larger ratio (val_ratio=0.30, contiguous) | ~38 | Change ratio |
| V3 | Time-based (2021-2022 val, 2023+ test) | ~500 | Fixed date ranges |
| V4 | Rolling window (train on N years, val on next 6mo) | ~125 | New splitter mode |

#### B. Loss Function (5 options)

| ID | Loss | Description | Addresses |
|----|------|-------------|-----------|
| L1 | BCELoss | Current baseline | - |
| L2 | BCEWithLogitsLoss + pos_weight | Class weighting | Imbalance |
| L3 | Focal Loss (γ=2) | Down-weight easy examples | Imbalance |
| L4 | SoftAUCLoss | Optimize ranking directly | Prior collapse |
| L5 | BCE + SoftAUC (weighted combo) | Calibration + ranking | Both |

#### C. Early Stopping Metric (2 options)

| ID | Metric | Notes |
|----|--------|-------|
| E1 | val_loss | Current default |
| E2 | val_auc | Requires AUC computation each epoch |

#### D. Output Layer (2 options)

| ID | Output | Loss Compatibility |
|----|--------|-------------------|
| O1 | Sigmoid (probabilities) | BCELoss, SoftAUCLoss |
| O2 | Logits (raw) | BCEWithLogitsLoss, Focal |

#### E. Class Weighting (2 options)

| ID | Weighting | Notes |
|----|-----------|-------|
| W1 | None | Current default |
| W2 | pos_weight = neg/pos ratio | ~9.0 for h1 |

---

## Experiment Matrix

### Phase 1: Validation Strategy Sweep (4 experiments)

Hold constant: L1 (BCE), E1 (val_loss), O1 (sigmoid), W1 (none)
Vary: V1, V2, V3, V4

| Exp | Val Strategy | Expected Runtime | Purpose |
|-----|--------------|------------------|---------|
| 1.1 | V1 (current, 19 samples) | 5 min | Baseline |
| 1.2 | V2 (30%, ~38 samples) | 5 min | Does more samples help? |
| 1.3 | V3 (time-based, ~500 samples) | 5 min | Proper temporal split |
| 1.4 | V4 (rolling, ~125 samples) | 5 min | Rolling validation |

**Decision point**: Pick best validation strategy for Phase 2.

### Phase 2: Loss Function Sweep (5 experiments)

Hold constant: Best V from Phase 1, E1 (val_loss), appropriate O for loss
Vary: L1, L2, L3, L4, L5

| Exp | Loss | Output | Purpose |
|-----|------|--------|---------|
| 2.1 | L1 (BCE) | O1 | Baseline with good val |
| 2.2 | L2 (BCE+pos_weight) | O2 | Class weighting |
| 2.3 | L3 (Focal) | O2 | Focal loss |
| 2.4 | L4 (SoftAUC) | O1 | Ranking loss |
| 2.5 | L5 (BCE+SoftAUC) | O1 | Combo |

**Decision point**: Pick best loss function for Phase 3.

### Phase 3: Early Stopping Sweep (2 experiments)

Hold constant: Best V, best L
Vary: E1, E2

| Exp | Early Stop | Purpose |
|-----|------------|---------|
| 3.1 | E1 (val_loss) | Baseline |
| 3.2 | E2 (val_auc) | AUC-based stopping |

**Decision point**: Pick best early stopping for Phase 4.

### Phase 4: Scale Validation (2 experiments)

Test winning configuration at 20M scale.

| Exp | Budget | Purpose |
|-----|--------|---------|
| 4.1 | 2M | Confirm baseline |
| 4.2 | 20M | Does larger model help with proper training? |

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Test AUC-ROC | 0.57 | 0.62 | 0.68 (RF level) |
| Prediction Spread | 2.5% | 10% | 30% |
| Val/Test AUC Correlation | Unknown | >0.7 | >0.9 |

---

## Implementation Requirements

### New Code Needed

| Component | Effort | Priority |
|-----------|--------|----------|
| V3: Time-based splitter mode | Medium | High |
| V4: Rolling window splitter | Medium | High |
| L3: Focal Loss implementation | Low | High |
| L5: Combined loss | Low | Medium |
| Experiment runner script | Medium | High |

### Existing Code to Modify

| Component | Change | Effort |
|-----------|--------|--------|
| ChunkSplitter | Add time-based and rolling modes | Medium |
| Trainer | Support custom loss functions | Low (mostly done) |
| Trainer | Early stopping on AUC | Done (Test 2) |

---

## Experiment Runner Design

Create `experiments/validation_exploration/run_sweep.py`:

```python
# Pseudocode
CONFIGS = [
    # Phase 1: Validation sweep
    {"val_strategy": "contiguous_15", "loss": "bce", ...},
    {"val_strategy": "contiguous_30", "loss": "bce", ...},
    {"val_strategy": "time_based", "loss": "bce", ...},
    {"val_strategy": "rolling", "loss": "bce", ...},
    # Phase 2-4 configs generated after Phase 1 results
]

for config in CONFIGS:
    train_model(config)
    evaluate_on_test(config)
    log_results(config)

generate_comparison_report()
```

---

## Timeline Estimate

| Phase | Experiments | Runtime (est.) | Calendar |
|-------|-------------|----------------|----------|
| Phase 1 | 4 | 20 min | Day 1 |
| Phase 2 | 5 | 25 min | Day 1 |
| Phase 3 | 2 | 10 min | Day 1 |
| Phase 4 | 2 | 30 min | Day 1 |
| Analysis | - | 30 min | Day 1 |
| **Total** | **13** | **~2 hours** | **Day 1** |

With 2M models taking ~5 min each, entire exploration fits in one session.

---

## Decision Framework

After each phase, decide:

1. **Clear winner?** → Proceed to next phase with winner
2. **No clear winner?** → Run additional experiments or pick simplest option
3. **All options bad?** → Investigate deeper (features? architecture?)

Final decision criteria for "working baseline":
- Test AUC > 0.60
- Prediction spread > 10%
- Val/Test AUC correlation > 0.7

If we can't achieve this, escalate to architecture investigation.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Time-based split reduces training data | Accept for validation quality |
| Loss function changes require HPO | Use 2M quick experiments first |
| No configuration works | Proves architecture issue, pivot to alternatives |

---

## Deliverables

1. `experiments/validation_exploration/` - All experiment scripts
2. `outputs/validation_exploration/` - Results CSVs
3. `docs/phase6a_validation_exploration_results.md` - Analysis (TEMPORARY)
4. Decision: Best configuration for HPO re-run

---

## Memory Entities to Create

- `Phase6A_ValidationExploration_Plan`
- `Phase6A_ValidationExploration_Results` (after completion)

---

## Related Documents

- `docs/phase6a_gap_analysis.md` - Gap identification
- `.claude/context/phase6a_gap_checklist.md` - Checklist
- `docs/research_paper/notes/prior_collapse_investigation.md` - Root cause
