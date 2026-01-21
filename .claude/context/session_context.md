# Session Handoff - 2026-01-20 ~21:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `a921271` docs: session handoff with context length ablation complete
- **Uncommitted changes**: ~25+ files (from multiple sessions)
- **Ahead of origin**: 18+ commits (not pushed)

### Task Status
- **Working on**: Architecture exploration planning
- **Status**: Plan document created, ready to execute experiments

---

## 🚨 MAJOR FINDING (Other Terminal)

### Random Forest BEATS PatchTST on ALL thresholds!

| Threshold | Pos Rate | PatchTST | Random Forest | Gradient Boost | Winner |
|-----------|----------|----------|---------------|----------------|--------|
| **0.5%** | 46% | 0.586 | **0.628** | 0.583 | **RF +7%** |
| **1.0%** | 20% | 0.695 | **0.716** | 0.624 | **RF +3%** |
| **2.0%** | 2% | 0.621 | 0.705 | **0.766** | **GB +23%** |

**Key insight**: Signal EXISTS - tree models find it, transformers don't.

---

## Test Status
- Last `make test`: 2026-01-20 ~21:25 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Completed This Session (Loss Function Terminal)

### 1. Expanded BCE vs weighted_05 Comparison
- 12 experiments: {BCE, weighted_05} × {h1, h3, h5} × {2M, 20M}
- **RESULT: NO CONSISTENT DIFFERENCE**
  - Mean AUC diff: -0.26% (high variance, std=1.33%)
  - 2M slightly favors weighted_05 (+0.29%)
  - 20M favors BCE (-0.80%)
- **RECOMMENDATION**: Stick with BCE (simpler, no benefit from weighted loss)
- **CONCLUSION**: Loss function is NOT the bottleneck

### 2. Architecture Exploration Plan Created
- Plan document: `docs/architecture_exploration_plan.md`
- Research question: Why do tree models outperform transformers?
- Hypothesis: Early convergence due to over-parameterization and deep composition

---

## Architecture Exploration Plan (NEXT STEPS)

**Research Question**: Why do tree-based models (XGBoost, RF) outperform PatchTST transformers?

### Experiment 1: XGBoost Baseline (~30 min)
- Same 20 features, same splits
- Establishes target AUC to beat (~0.70-0.75 expected)
- **Already confirmed by other terminal: RF gets 0.716 at 1% threshold**

### Experiment 2: Smaller Models 200K-500K params (~30 min)
- Test overparameterization hypothesis
- Configs: 200K (L=2 d=32), 500K (L=2 d=64)
- Current 2M model may be too large for ~7K training samples

### Experiment 3: Shallow + Wide (~30 min)
- L=1-2 layers with d_model=256-768
- Test if depth is the problem
- Shallow = closer to ensemble behavior

### Experiment 4: MLP-Only (1-2 hrs)
- Remove attention entirely
- Test if attention is helping or hurting
- May need new model class

### Experiment 5: Training Dynamics (if time)
- Lower LR (1e-5, 1e-6)
- Higher dropout (0.4, 0.5)
- Early stopping on AUC

**Execution Order**: XGBoost → Small models → Shallow+wide → MLP-only

**Success Criteria**: AUC > 0.70 OR conclusive evidence transformers can't match trees

---

## Files Created/Modified This Session

| File | Change |
|------|--------|
| `scripts/test_expanded_comparison.py` | NEW - 12-experiment comparison script |
| `outputs/expanded_comparison/` | NEW - BCE vs weighted_05 results |
| `docs/architecture_exploration_plan.md` | NEW - next phase plan |

**From other terminal (threshold comparison):**
- `experiments/threshold_comparison/` - threshold scripts
- `scripts/test_xgboost_thresholds.py` - XGBoost comparison
- `outputs/threshold_comparison/` - results

**Still uncommitted from previous sessions:**
- 12 HPO scripts in `experiments/phase6a/`
- `src/training/losses.py` (FocalLoss, SoftAUCLoss, WeightedSumLoss)
- Multi-objective comparison scripts and outputs

---

## Data Versions
- **Raw manifest**: SPY.parquet (8299 rows, 1993-2026)
- **Processed manifest**: SPY_dataset_a20.parquet (8100 rows)
- **Pending registrations**: none

---

## Memory Entities Updated This Session

**This session (loss function exploration):**
- `Plan_ExpandedBCEvsWeighted_20260120` - 12-experiment comparison plan
- `Finding_ExpandedBCEvsWeighted_20260120` - No consistent difference, stick with BCE
- `Plan_ArchitectureExploration_20260120` - Next phase: shallow+wide, small models, etc.
- `Hypothesis_TransformerEarlyConvergence_20260120` - Why transformers converge early

**From other terminal (to be created):**
- `Finding_RFBeatsPatchTST_20260120` - RF outperforms PatchTST on ALL thresholds
- `Finding_SignalExistsAt0.5pct_20260120` - Signal confirmed with RF AUC 0.628
- `Finding_PredictionCompression_PatchTST` - PatchTST outputs compressed to narrow range

**Relevant from previous sessions:**
- `Finding_ContextLengthAblation_20260120` - 80-day context optimal (+15.5%)
- `Finding_MultiObjectiveComparison_20260120` - Multi-objective no improvement

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
make verify
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
- Architecture exploration: Why do trees outperform transformers?
- Experiments: XGBoost baseline → Small models (200K) → Shallow+wide → MLP-only
- Goal: Find config with AUC > 0.70 or prove transformers can't match trees
