# Workstream 3 Context: Phase 6C Experiments
# Last Updated: 2026-01-25 18:30

## Identity
- **ID**: ws3
- **Name**: phase6c
- **Focus**: Phase 6C feature scaling experiments (a50 tier, threshold analysis, scaling comparison)
- **Status**: active

## Current Task
- **Working on**: Scaling comparison analysis (a20 vs a50)
- **Status**: In progress - need threshold sweeps on a20 for complete comparison

---

## Progress Summary

### Completed
- 2026-01-25: Created `threshold_sweep_s2_h5.py` for s2_horizon_2m_h5_a50
- 2026-01-25: Created `threshold_sweep_comprehensive.py` covering 12 S2 models
- 2026-01-25: Full a20 vs a50 scaling comparison (24 experiments analyzed)
- 2026-01-25: Identified key scaling × feature interaction effect
- 2026-01-24: S2 horizon experiments complete (9 runs: 3 budgets × 3 horizons)
- 2026-01-24: S2 architecture & training experiments complete (14 runs)
- 2026-01-24: Runner script `run_s2_experiments.sh` fixed
- 2026-01-23: S1 baselines complete (3 budgets at H=1)

### Pending
1. **Threshold sweeps on Phase 6A (a20)** - HIGH PRIORITY
   - Need precision/recall curves for a20 to compare with a50
   - Adapt `threshold_sweep_comprehensive.py` for `outputs/phase6a_final/`
2. Test set evaluation (not just validation)
3. Document scaling findings for research paper

---

## Last Session Work (2026-01-25)

### Threshold Sweep Scripts Created
- `experiments/phase6c/threshold_sweep_s2_h5.py` - Single model sweep
- `experiments/phase6c/threshold_sweep_comprehensive.py` - All 12 S2 models

### Key Findings

**Best Production Operating Points (a50, prioritized by precision):**
| Model | Threshold | Precision | Recall | Notes |
|-------|-----------|-----------|--------|-------|
| 2M H=5 | 0.65 | **70.2%** | 50.6% | Best precision with meaningful recall |
| 20M H=5 | 0.74 | 70.6% | 45.8% | Similar precision, less recall |
| 200M H=5 | 0.74 | 71.4% | 29.9% | Highest precision, low recall |

**User Priority Clarified**: Precision > Recall > AUC for production use.

### Scaling × Feature Interaction Effect (Novel Finding)

| Horizon | a20 Best Budget | a50 Best Budget |
|---------|-----------------|-----------------|
| H1 | 200M (0.718) | **20M** (0.722) |
| H2 | 2M | 2M |
| H3 | 200M | 20M |
| H5 | 2M | 2M |

**Interpretation**: At a20 (25 features), more parameters helps. At a50 (55 features), 20M is optimal - 200M overfits the additional features.

### Complete a20 vs a50 Comparison
| Horizon | Budget | a20 AUC | a50 AUC | Delta |
|---------|--------|---------|---------|-------|
| H1 | 2M | 0.706 | 0.708 | +0.3% |
| H1 | 20M | 0.715 | **0.722** | +1.0% |
| H1 | 200M | **0.718** | 0.699 | -2.6% |
| H2 | 2M | **0.639** | **0.643** | +0.6% |
| H2 | 20M | 0.635 | 0.635 | 0.0% |
| H2 | 200M | 0.635 | 0.641 | +0.9% |
| H3 | 2M | 0.618 | 0.610 | -1.3% |
| H3 | 20M | 0.615 | **0.619** | +0.6% |
| H3 | 200M | **0.622** | 0.616 | -1.0% |
| H5 | 2M | **0.605** | **0.594** | -1.8% |
| H5 | 20M | 0.596 | 0.593 | -0.5% |
| H5 | 200M | 0.599 | 0.575 | -4.0% |

---

## Files Owned/Modified
- `experiments/phase6c/threshold_sweep_s2_h5.py` - PRIMARY (new)
- `experiments/phase6c/threshold_sweep_comprehensive.py` - PRIMARY (new)
- `outputs/phase6c/s2_horizon_2m_h5_threshold_sweep.json` - PRIMARY (new)
- `outputs/phase6c/comprehensive_threshold_sweep.json` - PRIMARY (new)
- `experiments/phase6c/s1_*.py` - PRIMARY
- `experiments/phase6c/s2_*.py` - PRIMARY (27 files)
- `outputs/phase6c/s1_*/` - PRIMARY (3 experiments)
- `outputs/phase6c/s2_*/` - PRIMARY (23 experiments)

---

## Key Decisions (Workstream-Specific)

1. **Precision prioritized over recall** (2026-01-25)
   - User preference: For binary threshold tasks, precision is #1
   - Rationale: False positives cost money in trading

2. **Threshold sweep range focused on 0.5-0.8** (2026-01-25)
   - Rationale: Production-worthy operating points are in high-precision zone

3. **threshold_1pct used for all horizon experiments** (2026-01-24)
   - Matches Phase 6A baseline for fair comparison

4. **Feature tier interaction matters** (2026-01-25)
   - At a20: 200M optimal
   - At a50: 20M optimal
   - This is a novel research finding

---

## Experiments Complete (This Phase)

### Phase 6C S1 (a50, H=1) - 3 experiments
| Experiment | Budget | AUC | Notes |
|------------|--------|-----|-------|
| s1_01_2m_h1_a50 | 2M | 0.708 | +0.3% vs a20 |
| s1_02_20m_h1_a50 | 20M | **0.722** | +1.0% vs a20, BEST |
| s1_03_200m_h1_a50 | 200M | 0.699 | -2.6% vs a20, REGRESSED |

### Phase 6C S2 Horizon (a50, H=2,3,5) - 9 experiments
| Horizon | 2M | 20M | 200M |
|---------|-----|------|------|
| H=2 | **0.643** | 0.635 | 0.641 |
| H=3 | 0.610 | **0.619** | 0.616 |
| H=5 | **0.594** | 0.593 | 0.575 |

### Phase 6C S2 Architecture/Training - 14 experiments
- Various architecture tweaks (heads, depth, width)
- Various training params (dropout, LR, weight decay)
- Results in `outputs/phase6c/s2_arch_*/` and `outputs/phase6c/s2_train_*/`

---

## Next Session Should

1. **Create threshold sweep for Phase 6A (a20) experiments**
   - Adapt `threshold_sweep_comprehensive.py` for `outputs/phase6a_final/`
   - Compare precision/recall curves between a20 and a50
   - Key question: Does a50 achieve same precision with better recall?

2. **Statistical summary for research paper**
   - Document scaling × feature interaction effect
   - Create comparison figures

3. **Consider test set evaluation**
   - Currently using validation set only
   - Test set (2025+) would give out-of-sample confirmation

---

## Memory Entities (Workstream-Specific)
- None created this session (findings documented in context file)

---

## Session History

### 2026-01-25 (Scaling Analysis)
- Created threshold_sweep_s2_h5.py and threshold_sweep_comprehensive.py
- Ran comprehensive threshold sweeps on 12 S2 models
- Best production point: 2M H=5 @ thresh 0.65 = 70% prec, 51% rec
- Analyzed a20 vs a50 scaling comparison (24 experiments)
- Key finding: Scaling × feature interaction effect confirmed
- User priority clarified: Precision > Recall > AUC

### 2026-01-24 (Debugging + Experiments)
- Fixed runner script (set -e + venv unalias issue)
- Ran all S2 experiments (~23 experiments)
- Results in outputs/phase6c/s2_*/

### 2026-01-23 (S1 Baselines)
- Completed S1 experiments (3 budgets at H=1)
- 200M regression identified (-2.6% vs a20)
