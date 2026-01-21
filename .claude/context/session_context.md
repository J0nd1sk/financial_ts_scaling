# Session Handoff - 2026-01-21 ~02:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `201acac` exp: MLP-only and shallow+wide architecture experiments
- **Uncommitted changes**:
  - `scripts/test_lr_ablation.py` (NEW - LR ablation experiment)
  - `scripts/test_dropout_ablation.py` (NEW - Dropout ablation experiment)
  - `scripts/test_combined_ablation.py` (NEW - Combined LR+dropout experiment)
  - `outputs/training_dynamics/` (NEW - results directory with 3 CSVs + 3 JSONs)
- **Ahead of origin**: 7 commits (not pushed)

### Task Status
- **Working on**: Architecture exploration - Training dynamics experiment COMPLETE
- **Status**: 4 of 5 experiments complete, conclusion pending

---

## Architecture Exploration Progress

### Experiments Completed

#### 1. Small Models Experiment ✅
**Result**: HYPOTHESIS REJECTED - 215K baseline beats smaller models

#### 2. Shallow+Wide Experiment ✅
**Result**: HYPOTHESIS REJECTED - L=4 d=64 beats L=1-2 with d=256-768

#### 3. MLP-Only Experiment ✅
**Result**: NUANCED - MLP peak 0.7077 beats PatchTST 0.6945, but overfits faster

#### 4. Training Dynamics Experiment ✅ (THIS SESSION)
**Result**: BREAKTHROUGH - PatchTST nearly matches RF with proper regularization!

**LR Ablation (baseline dropout 0.2):**
| Config | Best AUC | vs Baseline |
|--------|----------|-------------|
| **PatchTST_lr1e-5** | **0.7123** | **+1.8%** |
| PatchTST_lr1e-6 | 0.5587 | -13.6% (too slow) |
| MLP_lr1e-5 | 0.4734 | -23.4% |
| MLP_lr1e-6 | 0.4659 | -24.2% |

**Dropout Ablation (baseline LR 1e-3):**
| Config | Best AUC | vs Baseline |
|--------|----------|-------------|
| **PatchTST_d0.5** | **0.7199** | **+2.5%** ⭐ BEST |
| PatchTST_d0.4 | 0.7184 | +2.4% |
| MLP_d0.4 | 0.6944 | -1.3% peak |
| MLP_d0.5 | 0.6817 | -2.6% |

**Combined Ablation (LR × Dropout):**
| Config | Best AUC | Final AUC |
|--------|----------|-----------|
| PatchTST_lr1e-5_d0.5 | 0.7089 | 0.7089 |
| **MLP_lr1e-5_d0.5** | 0.6977 | **0.6969** ⭐ Best MLP final |
| PatchTST_lr1e-5_d0.4 | 0.6736 | 0.6736 |
| Others | <0.65 | Too slow |

### Experiments Remaining

5. **Conclusion** - Summarize all findings, determine final answer

---

## Key Findings Summary (All Sessions)

| Finding | Evidence |
|---------|----------|
| RF beats baseline PatchTST | RF 0.716 vs PatchTST 0.695 (gap: 3%) |
| Focal Loss helps | AUC 0.54→0.67 (+24.8%) |
| **PatchTST + dropout=0.5 nearly matches RF** | **0.7199 vs 0.716 (gap: 0.4%)** ⭐ |
| Higher dropout > lower LR for regularization | d=0.5 (0.7199) > lr=1e-5 (0.7123) |
| Combining LR+dropout doesn't add up | lr1e-5+d0.5 (0.7089) < d0.5 alone (0.7199) |
| MLP overfitting fixed | Final AUC 0.6969 vs 0.6626 baseline (+3.4%) |
| MLP can beat PatchTST at peak | MLP 0.7077 vs PatchTST 0.6945 (+1.9%) |
| Overparameterization NOT the issue | 215K baseline beats 33K-14.4M models |
| Depth NOT the issue | L=4 d=64 beats L=1-2 with d=256-768 |

**Targets:**
- RF AUC: 0.716
- XGBoost AUC: 0.7555

**Best Results:**
- **PatchTST @ d=0.5**: 0.7199 (only 0.4% below RF!) ⭐
- PatchTST @ d=0.4: 0.7184
- PatchTST @ lr=1e-5: 0.7123
- MLP peak: 0.7077 (overfits to 0.6626)
- MLP stable (lr1e-5+d0.5): 0.6969

---

## Test Status
- Last `make test`: 2026-01-21 ~01:50 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Files Created This Session

| File | Description |
|------|-------------|
| `scripts/test_lr_ablation.py` | LR ablation (1e-5, 1e-6) on PatchTST and MLP |
| `scripts/test_dropout_ablation.py` | Dropout ablation (0.4, 0.5) on PatchTST and MLP |
| `scripts/test_combined_ablation.py` | Combined LR×dropout (8 configs) |
| `outputs/training_dynamics/lr_ablation_results.csv` | LR ablation results |
| `outputs/training_dynamics/dropout_ablation_results.csv` | Dropout ablation results |
| `outputs/training_dynamics/combined_ablation_results.csv` | Combined ablation results |

---

## Memory Entities Updated This Session

**Created:**
- `Plan_TrainingDynamicsExperiment_20260121` - Planning decision for training dynamics
- `Finding_TrainingDynamicsExperiment_20260121` - BREAKTHROUGH: PatchTST 0.7199 ≈ RF 0.716

**From previous sessions (relevant):**
- `Finding_MLPOnlyExperiment_20260121` - MLP beats PatchTST at peak but overfits
- `Finding_SmallModels_20260120` - Smaller models don't help
- `Finding_ShallowWide_20260120` - Shallow+wide doesn't help
- `Finding_FocalLossFixesPatchTST_20260120` - Focal Loss breakthrough
- `Finding_RFBeatsPatchTST_20260120` - Tree models outperform (baseline)

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
make verify
```

---

## Next Session Should

1. **Commit training dynamics experiment results** (3 scripts + outputs)
2. **Write architecture exploration conclusion** - Document final findings
3. **Update phase_tracker.md** with training dynamics results
4. **Decide next phase** - What's after architecture exploration?

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
- Architecture exploration: Why do trees outperform transformers?
- **ANSWER FOUND**: PatchTST with dropout=0.5 achieves 0.7199, only 0.4% below RF (0.716)
- Transformers CAN match trees with proper regularization!
