# Workstream 3 Context: Phase 6C Experiments
# Last Updated: 2026-01-25 16:45

## Identity
- **ID**: ws3
- **Name**: phase6c
- **Focus**: Phase 6C feature scaling experiments (a50 → a100 tier)
- **Status**: **S1 COMPLETE** - Baselines done, HPO next

## Current Task
- **Working on**: Phase 6C a100 HPO experiments
- **Status**: ✅ S1 baselines complete, ready for HPO

---

## Session 2026-01-25 (S1 Baselines + Analysis)

### Completed
1. ✅ Fixed broken runner scripts (`run_s1_a100.sh`, `run_hpo_a100.sh`)
   - Added `set -o pipefail`
   - Added `mkdir -p outputs/phase6c_a100`
   - Safe venv activation with `set +e`/`set -e` wrapper
   - Replaced tee pattern with direct execution + pass/fail tracking

2. ✅ Ran all 12 S1 baseline experiments (3 budgets × 4 horizons)
   - All 12 passed successfully
   - Results in `outputs/phase6c_a100/s1_*/results.json`

3. ✅ Performed threshold sweep analysis on all models

### Key Findings

#### Scaling Law Results (Preliminary)
| Horizon | 2M AUC | 20M AUC | 200M AUC | Winner |
|---------|--------|---------|----------|--------|
| H1 | **0.709** | 0.712 | 0.705 | 20M (marginal) |
| H2 | 0.632 | 0.631 | **0.636** | 200M |
| H3 | 0.609 | 0.616 | **0.632** | 200M |
| H5 | 0.583 | **0.631** | 0.613 | 20M |

**Conclusion**: No clear scaling benefit. Inverse scaling at H1. Marginal benefit at longer horizons.

#### Precision-Recall Tradeoff (H5 20M example)
| Threshold | Precision | Recall |
|-----------|-----------|--------|
| 0.712 | 100% | 3.2% |
| 0.711 | 90.9% | 4.0% |
| 0.694 | 75.0% | 22.7% |
| 0.674 | 70.0% | 46.6% |
| 0.643 (optimal F1) | 62.7% | 96.4% |

**Key insight**: High precision (90%+) means catching very few opportunities (3-7% recall).

#### Model Behavior
- 200M models are more selective (higher precision, lower recall)
- Default 0.5 threshold was suboptimal; optimal thresholds range 0.50-0.64
- Longer horizons have better PR-AUC but worse ROC-AUC
- Models trained with 105 features (100 indicators + OHLCV), not 100 as documented

---

## Files Modified This Session
- `scripts/run_s1_a100.sh` - Rewritten (fixed)
- `scripts/run_hpo_a100.sh` - Rewritten (fixed)

## Outputs Created
```
outputs/phase6c_a100/
├── s1_01_2m_h1/    (results.json, best_checkpoint.pt)
├── s1_02_20m_h1/
├── ...
└── s1_12_200m_h5/
```

---

## Next Session Should

1. **Run HPO experiments**:
   ```bash
   caffeinate ./scripts/run_hpo_a100.sh
   ```
   - 6 HPO runs (3 budgets × H1, H5)
   - 50 trials each, ~6-12 hours total

2. **Analyze HPO results**:
   - Compare tuned vs baseline performance
   - Check if dropout/lr tuning helps larger models

3. **Consider**: Whether to add threshold as HPO parameter

---

## Memory Entities (This Session)
- `Phase6C_A100_S1_Results` - Baseline experiment results
- `Phase6C_ThresholdSweep_Findings` - Precision-recall analysis
- `Phase6C_A100_RunnerScripts_Fix` - Technical fix documentation
- `Phase6C_ScalingLaw_Preliminary` - Research finding on scaling

---

## Session History

### 2026-01-25 16:45 (S1 Complete + Analysis)
- Fixed runner scripts (were completely broken)
- Ran all 12 S1 baselines successfully
- Performed threshold sweep analysis
- Key finding: No clear scaling benefit, steep precision-recall tradeoff
- Ready for HPO

### 2026-01-25 23:50 (Pipeline Creation - INCOMPLETE)
- Created full a100 experimentation pipeline
- All scripts pass syntax check but runner didn't work
- Session ended with troubleshooting needed
