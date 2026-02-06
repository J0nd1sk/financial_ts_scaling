# Workstream 3 Context: a200 HPO v3
# Last Updated: 2026-02-06 14:15

## Identity
- **ID**: ws3
- **Name**: a200_hpo_v3
- **Focus**: Precision-first HPO with loss function optimization
- **Status**: active

## 🔴 CRITICAL: Metric Priority for Analysis
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

---

## Current Task
- **Working on**: a200 HPO v3 implementation
- **Status**: ✅ IMPLEMENTATION COMPLETE - ready to run experiment

---

## Progress Summary

### Completed (2026-02-01 - Post-Crash Discovery)

#### HPO v3 Implementation - DONE
1. **Created `experiments/phase6c_a200/hpo_20m_h1_a200_v3.py`**:
   - Precision-first composite objective: `precision*2 + recall*1 + auc*0.1`
   - Loss type as hyperparameter: `focal` vs `weighted_bce`
   - Conditional loss params: `focal_alpha/gamma` or `bce_pos_weight`
   - Multi-threshold metrics logging (t30, t40, t50, t60, t70)
   - 80d context length (per CLAUDE.md standard)
   - 50 trials

2. **Created `tests/test_hpo_a200_v3.py`** - 20 tests, ALL PASS:
   - TestCompositeObjective (5 tests)
   - TestSearchSpaceV3 (4 tests)
   - TestConditionalLossParams (2 tests)
   - TestMultiThresholdMetrics (2 tests)
   - TestObjectiveReturnsComposite (1 test)
   - TestScriptConfigurationV3 (4 tests)
   - TestLossFunctionInstantiation (2 tests)

#### Context Ablation Results (Previous Sessions)
- a200 @ 75d = best precision/recall (66.7%, 7.8%)
- a200 @ 80d = best AUC (0.730)
- v3 uses 80d (CLAUDE.md standard)

### Pending
1. ~~Git commit new files~~ ← NEXT
2. Run HPO experiment (~2-3 hours)
3. Analyze results

---

## Files Created/Modified

| File | Status | Type |
|------|--------|------|
| `experiments/phase6c_a200/hpo_20m_h1_a200_v3.py` | NEW | PRIMARY |
| `tests/test_hpo_a200_v3.py` | NEW | PRIMARY |
| `src/training/losses.py` | MODIFIED | SHARED (FocalLoss, WeightedBCELoss) |
| `tests/test_losses.py` | MODIFIED | SHARED |

---

## Key Technical Details

### Search Space v3
```python
SEARCH_SPACE_V3 = {
    # Architecture
    "d_model": [64, 96, 128, 160, 192],
    "n_layers": [4, 5, 6, 7, 8],
    "n_heads": [4, 8],
    "d_ff_ratio": [2, 4],
    # Training
    "learning_rate": [5e-5, 7e-5, 1e-4, 1.5e-4],
    "dropout": [0.3, 0.4, 0.5, 0.6],
    "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
    # Loss function (NEW in v3)
    "loss_type": ["focal", "weighted_bce"],
    "focal_alpha": [0.3, 0.5, 0.7, 0.9],
    "focal_gamma": [0.0, 0.5, 1.0, 2.0],
    "bce_pos_weight": [1.0, 2.0, 3.0, 5.0],
}
CONTEXT_LENGTH = 80
N_TRIALS = 50
```

### Composite Objective Function
```python
def compute_composite_score(precision, recall, auc):
    return (precision * 2.0) + (recall * 1.0) + (auc * 0.1)
```

**Weights rationale:**
- Precision: 2.0 (primary - when we say buy, be right)
- Recall: 1.0 (secondary - catch opportunities)
- AUC: 0.1 (tertiary - tie-breaking only)

---

## Key Decisions (This Session)

### v3 over v2: Precision-First + Loss Optimization
- **Decision**: v3 replaces v2's "forced extremes" approach with precision-first composite + loss function HPO
- **Rationale**: User insight: "Precision increases with stricter thresholds. Test ALL in HPO."
- **Impact**: Optimizes what we actually care about (precision) rather than just AUC

### 80d Context (not 75d)
- **Decision**: Use 80d per CLAUDE.md standard, despite 75d showing best precision in ablation
- **Rationale**: Consistency with project standards; difference is marginal

---

## Session History

### 2026-02-06 (Session Restore - Post-Crash)
- Discovered context was stale (described v2, actual state is v3)
- Verified v3 tests pass (20/20)
- Updated context files
- Preparing to commit

### 2026-02-01 (HPO v3 Implementation)
- Created hpo_20m_h1_a200_v3.py with precision-first composite + loss HPO
- Created 20 tests (all passing)
- Computer crashed during HPO run (trial 0 completed)

### 2026-01-31 (HPO v2 + Context Ablation)
- Context ablation complete (a50, a100, a200)
- Designed v2 with forced extremes (superseded by v3)

---

## Next Session Should

1. **Run HPO experiment** (~2-3 hours):
   ```bash
   ./venv/bin/python experiments/phase6c_a200/hpo_20m_h1_a200_v3.py
   ```
2. **Analyze results** - identify best loss type + params for precision
3. **Update TIER_BEST_ARCH** with findings

---

## Memory Entities (Workstream-Specific)
- **a200_75d_Best_Result**: 66.7% precision, 7.8% recall - best combo found (from ablation)
- **v3_composite_objective**: precision*2 + recall*1 + auc*0.1
