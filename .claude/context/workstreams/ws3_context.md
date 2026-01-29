# Workstream 3 Context: Phase 6C HPO Analysis
# Last Updated: 2026-01-28 15:30

## Identity
- **ID**: ws3
- **Name**: phase6c_hpo_analysis
- **Focus**: HPO methodology improvement and tier HPO execution
- **Status**: active

## 🔴 CRITICAL: Metric Priority for Analysis
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **❌ NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

**Show precision and recall SEPARATELY across threshold changes to reveal the tradeoff curve.**

## Current Task
- **Working on**: Threshold sweep analysis → select top 10 models for full training
- **Status**: Threshold sweep complete, need precision-focused reformatting

---

## Progress Summary

### Completed (This Session - 2026-01-27)
- [x] **Fixed HPO template** - Changed `verbose=False` to `verbose=True` (line 327)
- [x] **Ran top model evaluation (a50)** - 9 models with real precision/recall/pred_range
- [x] **Ran top model evaluation (a100)** - 9 models with real precision/recall/pred_range
- [x] **Created comprehensive report** - `docs/hpo_comprehensive_report.md`
- [x] **Analyzed 250 HPO trials** - Identified hyperparameter trends
- [x] **Proposed 27 supplementary trials** - `docs/supplementary_hpo_proposal.md`

### HPO Results Summary

| Tier | Features | Best Budget | Best AUC | Avg Precision | Avg Recall |
|------|----------|-------------|----------|---------------|------------|
| **a50** | 55 | 20M | **0.7315** | 50.8% | 9.65% |
| **a100** | 105 | 20M | 0.7189 | 47.5% | 11.4% |

### Key Findings
1. **Scaling laws VIOLATED** - 20M beats both 2M and 200M
2. **More features HURT** - a50 (55) beats a100 (105) by 1.3% AUC
3. **Dropout 0.5 is optimal** - 56.7% of top 60 performers
4. **LR 1e-4 is optimal** - 61.7% of top 60 performers
5. **No probability collapse** - Wide prediction ranges [0.01-0.88]
6. **Recall is the bottleneck** - ~10% means missing 90% of opportunities

### Models with Best Precision (>50%)
| Model | Precision | Recall | AUC |
|-------|-----------|--------|-----|
| a50-200M-T8 | 54.5% | 7.9% | 0.7294 |
| a50-2M-T45 | 53.8% | 9.2% | 0.7300 |
| a50-20M-T5 | 53.3% | 10.5% | 0.7315 |

---

## Files Created/Modified (This Session)

### Modified
1. `experiments/templates/hpo_template.py` - **CRITICAL FIX**: `verbose=True`

### Created
2. `docs/hpo_comprehensive_report.md` - Full analysis with tables
3. `docs/supplementary_hpo_proposal.md` - 27 targeted trials proposal
4. `outputs/phase6c_a50/top_models_detailed_metrics.json` - Real metrics
5. `outputs/phase6c_a50/top_models_detailed_metrics.md` - Report
6. `outputs/phase6c_a100/top_models_detailed_metrics.json` - Real metrics
7. `outputs/phase6c_a100/top_models_detailed_metrics.md` - Report

---

## Hyperparameter Trends (from 250 Trials)

### Sweet Spots Identified
| Parameter | Optimal | Evidence |
|-----------|---------|----------|
| dropout | **0.5** | 56.7% of top 60, mean AUC 0.7150 |
| learning_rate | **1e-4** | 61.7% of top 60, mean AUC 0.7146 |
| d_model | **128** | 63.3% of top 60 |
| weight_decay | **1e-4 to 1e-3** | 82% of top 60 |
| n_layers | **2 or 6-7** | Bimodal distribution |

### Gaps Never Tested
- Dropout: 0.35, 0.40, 0.45, 0.55, 0.60
- LR: 7e-5, 8e-5, 9e-5, 1.2e-4
- Weight decay: 3e-4, 5e-4, 7e-4
- d_model: 112, 144

---

## Supplementary Trials Proposal (27 Trials)

### Phase 1: Fine-Tune Winners (15 trials)
- **A1-A6**: Dropout sweep (0.35 to 0.60)
- **B1-B5**: LR fine-tuning (7e-5 to 1.5e-4)
- **C1-C4**: Weight decay (3e-4 to 2e-3)

### Phase 2: Architecture Variants (8 trials)
- **D1-D4**: Shallow variants (2M budget)
- **E1-E4**: Deep variants (20M budget)

### Phase 3: Combined Optimization (4 trials)
- **F1-F4**: Best combinations from Phase 1+2

**Estimated time**: 15-30 minutes total

---

## Next Session Should

1. ~~**Decide**: Run supplementary trials or accept current results?~~ DONE - 2 rounds completed
2. **Reformat analysis** - Precision/recall focused (no F1, no accuracy)
3. **Select top 10 model configs** - For full training on a50 and a100
4. **Run full experiments** - With selected configs

---

## Key Commands

```bash
# Evaluate top models (already done)
./venv/bin/python scripts/evaluate_top_hpo_models.py --tier a50 --top-n 3
./venv/bin/python scripts/evaluate_top_hpo_models.py --tier a100 --top-n 3

# Supplementary HPO (proposed, not yet implemented)
./venv/bin/python scripts/run_supplementary_hpo.py --tier a50

# All tests
make test
```

---

## Session History

### 2026-01-28 15:30 (Metric Priority Correction)
- User corrected metric focus: PRECISION #1, RECALL #2, NO F1/Accuracy
- Updated CLAUDE.md, global_context.md, ws3_context.md with priority
- Created Memory MCP entity: Metric_Priority_Classification
- Need to reformat threshold sweep analysis with correct priorities
- Next: Select top 10 configs for full training (a50 + a100)

### 2026-01-28 ~14:05 (Terminal Crashed)
- Ran supplementary HPO round 2 (13 trials)
- Ran threshold sweep on 6 models
- Generated report but INCORRECTLY focused on F1 scores
- Terminal died before completing analysis

### 2026-01-27 19:15 (Comprehensive Analysis Complete)
- Fixed HPO template (`verbose=True`) - future runs will capture all metrics
- Re-trained 18 top models with `verbose=True` to get real precision/recall
- Created comprehensive report with solid tables
- Analyzed 250 HPO trials for hyperparameter trends
- Proposed 27 supplementary trials targeting gaps in search space
- Key insight: Dropout 0.5 and LR 1e-4 are clear sweet spots

### 2026-01-26 16:30 (HPO Execution Started)
- Created `docs/hpo_methodology.md`
- Started a100 HPO, then a50 HPO

### 2026-01-26 14:45 (All Improvements Complete)
- Implemented 4/4 HPO methodology improvements
- Coverage-aware sampling, cross-budget validation

---

## Memory MCP Entities
- **HPO_Methodology_Phase6C**: Two-phase strategy, findings
- **HPO_Hyperparameter_Trends**: Optimal values and gaps identified
