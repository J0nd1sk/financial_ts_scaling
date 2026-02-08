# Workstream 3 Context: Feature Embedding Experiments
# Last Updated: 2026-02-07 18:30

## Identity
- **ID**: ws3
- **Name**: feature_embedding_experiments
- **Focus**: Testing d_embed parameter + loss functions + advanced embeddings for precision improvement
- **Status**: Phase 1 (P1-P12) COMPLETE; Phase 2 (LF-P1 to LF-P6) COMPLETE; Phase 3 (LF-P7 to LF-P11 + AE-P1 to AE-P5) IN PROGRESS

---

## Current State Summary

### Total Experiments: 183
| Category | Count | Description |
|----------|-------|-------------|
| FE (P1-P12) | 92 | Feature embedding architecture |
| LF (LF-P1 to LF-P11) | 70 | Loss function experiments |
| AE (AE-P1 to AE-P5) | 21 | Advanced embedding architectures |

### Run Status

**Completed:**
- P1-P12: 92 FE architecture experiments
- LF-P1 to LF-P6: 48 original loss function experiments

**In Progress:**
- LF-P7: MildFocal (6 experiments) - RUNNING

**Pending (37 remaining):**
- LF-P8: AsymmetricFocal (6)
- LF-P9: EntropyRegularized (4)
- LF-P10: VarianceRegularized (4)
- LF-P11: CalibratedFocal (2)
- AE-P1: Progressive Embedding (4)
- AE-P2: Bottleneck Embedding (4)
- AE-P3: MultiHead Embedding (4)
- AE-P4: GatedResidual Embedding (4)
- AE-P5: Attention Embedding (5)

---

## Best Results (Top 10 by Precision)

| Rank | Exp ID | Tier | d_embed | Precision | Recall | AUC | Loss |
|------|--------|------|---------|-----------|--------|-----|------|
| 1 | **LF-40** | a500 | 16 | **85.7%** | 7.9% | 0.725 | LabelSmoothing e=0.20 |
| 2 | LF-48 | a200 | 64 | 60.0% | 7.9% | 0.723 | LabelSmoothing e=0.10 |
| 3 | FE-26 | a200 | 128 | 60.0% | 15.8% | 0.710 | BCE |
| 4 | LF-51 | a100 | 32 | 60.0% | 11.8% | 0.726 | MildFocal g=1.0 |
| 5 | FE-35 | a200 | 32 | 57.1% | 10.5% | 0.724 | BCE |
| 6 | FE-11 | a100 | 128 | 55.6% | 13.2% | 0.700 | BCE |
| 7 | FE-80 | a100 | 24 | 53.3% | 10.5% | 0.734 | BCE |
| 8 | FE-16 | a100 | 64 | 50.0% | 18.4% | 0.680 | BCE |
| 9 | FE-68 | a20 | None | 50.0% | 5.3% | 0.724 | BCE |
| 10 | FE-40 | a200 | 8 | 47.4% | 11.8% | 0.689 | BCE |

### Key Finding
**LF-40 achieves 85.7% precision** using LabelSmoothing (e=0.20) on a500 with d_embed=16. This is the highest precision observed across all experiments.

---

## Progress Summary

### Completed
- [2026-02-06] Feature embedding infrastructure created
- [2026-02-06] Fixed a500 data (906 → 7977 rows)
- [2026-02-07 AM] P1-P12 experiments complete (92 experiments)
- [2026-02-07 AM] LF-P1 to LF-P6 experiments complete (48 experiments)
- [2026-02-07 PM] **Phase 3 IMPLEMENTED**: 22 new LF experiments (LF-P7 to LF-P11) + 21 AE experiments (AE-P1 to AE-P5)
- [2026-02-07 PM] Created `docs/feature_embedding_experiment_tracker.md`

### Pending
- LF-P7 to LF-P11: 22 subtle loss function experiments
- AE-P1 to AE-P5: 21 advanced embedding experiments

---

## Last Session Work (2026-02-07 PM)

### Phase 3 Implementation
1. **Extended ExperimentSpec** with `embedding_type` and `embedding_params` fields
2. **Added get_embedding_layer()** helper function for creating custom embeddings
3. **Extended get_loss_function()** with 5 new loss types:
   - `mildfocal` → MildFocalLoss
   - `asymmetricfocal` → AsymmetricFocalLoss
   - `entropyreg` → EntropyRegularizedBCE
   - `variancereg` → VarianceRegularizedBCE
   - `calibratedfocal` → CalibratedFocalLoss
4. **Updated get_early_stop_metric()** for new calibration-focused losses
5. **Added embedding integration** in run_experiment() - replaces model.feature_embed when custom embedding specified
6. **Defined 22 new LF experiments** (LF-49 to LF-70):
   - LF-P7: MildFocal (6) - subtle gamma 0.5-1.0
   - LF-P8: AsymmetricFocal (6) - gamma_neg/gamma_pos ≤ 1.5
   - LF-P9: EntropyRegularized (4)
   - LF-P10: VarianceRegularized (4)
   - LF-P11: CalibratedFocal (2)
7. **Defined 21 AE experiments** (AE-01 to AE-21):
   - AE-P1: Progressive Embedding (4)
   - AE-P2: Bottleneck Embedding (4)
   - AE-P3: MultiHead Embedding (4)
   - AE-P4: GatedResidual Embedding (4)
   - AE-P5: Attention Embedding (5)
8. **Updated argparse** with new priorities
9. **Added 8 new test methods** across 2 new test classes
10. **Updated existing tests** for expanded experiment count (48→70 LF)

### Tests Status
- 25 tests in test_feature_embedding_experiments.py (all passing)
- 139 tests passing (losses + arch_grid + feature_embedding tests)

---

## Files Modified This Session

| File | Change |
|------|--------|
| `experiments/feature_embedding/run_experiments.py` | +embedding_type/params fields, +5 loss types, +43 experiments, +embedding integration |
| `tests/test_feature_embedding_experiments.py` | +8 new tests, updated existing for 70 LF experiments |
| `docs/feature_embedding_experiment_tracker.md` | NEW - comprehensive tracking document |

---

## Uncommitted Changes

All changes from this session remain uncommitted:
- Extended run_experiments.py (now 183 experiments)
- New/updated tests (25 total)
- New tracker document
- Context files

---

## Key Findings

### 1. Label Smoothing Dominates
- LF-40 (LabelSmoothing e=0.20) achieves **85.7% precision**
- Prevents overconfident predictions, improves calibration

### 2. Smaller is Better (Architecture)
- Smaller d_embed generally better
- Smaller d_model generally better
- FE-50 breakthrough: a500 + d_embed=16 + d_model=64 = 60% precision

### 3. Probability Collapse Risk
- Some experiments show predictions clustered near 0.5
- Variance/entropy regularization being tested to address this

---

## Commands to Run

```bash
# Currently running:
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P7

# After LF-P7:
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P8
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P9
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P10
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P11
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P1
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P2
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P3
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P4
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P5
```

---

## Session History

### 2026-02-07 (Session 4) - PM
- **Phase 3 IMPLEMENTED**: 22 LF + 21 AE experiments
- Extended ExperimentSpec with embedding_type/embedding_params
- Added 5 new loss function types
- Added get_embedding_layer() helper
- Created experiment tracker document
- 25 tests passing

### 2026-02-07 (Session 3) - AM
- Phase 2 Loss Function Experiments IMPLEMENTED (48 experiments)
- Extended ExperimentSpec with loss_fn/loss_params

### 2026-02-07 (Session 2)
- Analyzed P1-P5 results, FE-50 breakthrough identified
- Added 42 new experiments (P8-P12)

### 2026-02-06 (Session 1)
- Created feature embedding infrastructure
- Fixed a500 data issues

---

## Next Session Should

1. **Wait for LF-P7 to complete** (MildFocal, currently running)
2. **Run remaining priorities in order**: LF-P8 → LF-P11 → AE-P1 → AE-P5
3. **Analyze results** focusing on:
   - Do subtle loss functions improve precision?
   - Do advanced embeddings outperform simple linear?
   - Which combinations work best?
4. **Consider committing** after experiments complete

---

## Memory Entities

- `Feature_Embedding_Results`: P1-P12 complete results
- `FE50_Breakthrough`: a500 + d_embed=16 + d=64 + h=4 = 60% precision
- `LF40_Best_Result`: LabelSmoothing e=0.20 achieves 85.7% precision
- `Phase3_Experiments`: 22 LF + 21 AE experiments implemented
