# Session Handoff - 2026-01-21 ~00:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `7e5d6ab` docs: session handoff with Focal Loss breakthrough
- **Uncommitted changes**:
  - `scripts/test_mlp_only.py` (NEW - MLP experiment script)
  - `outputs/mlp_only_experiment/` (NEW - results)
  - `outputs/shallow_wide_experiment/` (from previous session)
- **Ahead of origin**: 6 commits (not pushed)

### Task Status
- **Working on**: Architecture exploration - Training dynamics experiment is NEXT
- **Status**: 3 of 5 experiments complete, 2 remaining

---

## Architecture Exploration Progress

### Experiments Completed

#### 1. Small Models Experiment ✅
**Result**: HYPOTHESIS REJECTED - 215K baseline beats smaller models

#### 2. Shallow+Wide Experiment ✅
**Result**: HYPOTHESIS REJECTED - L=4 d=64 beats L=1-2 with d=256-768

#### 3. MLP-Only Experiment ✅ (THIS SESSION)
**Result**: NUANCED FINDING - MLP can BEAT PatchTST but overfits faster

| Model | Params | Best Val AUC | Final Val AUC |
|-------|--------|--------------|---------------|
| **MLP_h256_o64** | 185K | **0.7077** ⭐ | 0.6626 |
| MLP_h128_o32 | 72K | 0.7006 | 0.6749 |
| MLP_h512_o128 | 534K | 0.6992 | 0.6678 |
| *PatchTST baseline* | 215K | *0.6945* | - |
| *RF baseline* | - | *0.716* | - |

**Key Finding**: MLP peak AUC (0.7077) beats PatchTST (0.6945) by +1.9%!
- Attention helps GENERALIZATION, not learning capacity
- MLP overfits faster, loses gains after early stopping
- The signal IS learnable without attention

### Experiments Remaining

4. **Training dynamics** - Lower LR (1e-5, 1e-6), higher dropout (0.4, 0.5)
   - **USER CLARIFICATION**: Test BOTH transformers AND MLPs
   - Test if training stability can unlock MLP's 0.70+ potential
5. **Conclusion** - Summarize findings, determine if transformers can match trees

---

## Key Findings Summary (All Sessions)

| Finding | Evidence |
|---------|----------|
| RF beats PatchTST | RF AUC 0.716 vs PatchTST 0.695 (gap: 3%) |
| Focal Loss helps | AUC 0.54→0.67 (+24.8%) |
| MLP can beat PatchTST at peak | MLP 0.7077 vs PatchTST 0.6945 (+1.9%) |
| MLP overfits faster | Best AUC 0.7077 → Final 0.6626 |
| Attention helps generalization | PatchTST stable, MLP degrades |
| Overparameterization NOT the issue | 215K baseline beats 33K-14.4M models |
| Depth NOT the issue | L=4 d=64 beats L=1-2 with d=256-768 |

**Targets**:
- RF AUC: 0.716
- XGBoost AUC: 0.7555

**Best Results**:
- PatchTST: 0.6945 (L=4, d=64, 215K params)
- MLP peak: 0.7077 (h=256, o=64, 185K params) - but final was 0.6626

---

## Test Status
- Last `make test`: 2026-01-21 ~00:25 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Files Created This Session

| File | Description |
|------|-------------|
| `scripts/test_mlp_only.py` | MLP-only experiment (patch-based MLP without attention) |
| `outputs/mlp_only_experiment/results.json` | Experiment results |
| `outputs/mlp_only_experiment/results.csv` | Results in CSV format |

---

## Memory Entities Updated This Session

**Created:**
- `Plan_MLPOnlyExperiment_20260121` (created): Planning decision for MLP experiment
- `Finding_MLPOnlyExperiment_20260121` (created): Nuanced finding - MLP beats PatchTST at peak but overfits

**From previous sessions (relevant):**
- `Finding_SmallModels_20260120` - Smaller models don't help
- `Finding_ShallowWide_20260120` - Shallow+wide doesn't help
- `Finding_FocalLossFixesPatchTST_20260120` - Focal Loss breakthrough
- `Finding_RFBeatsPatchTST_20260120` - Tree models outperform

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

1. **Plan training dynamics experiment** - Include BOTH transformers AND MLPs
   - Lower LR (1e-5, 1e-6) to slow learning and prevent overfitting
   - Higher dropout (0.4, 0.5) for regularization
   - Test if these stabilize MLP's 0.70+ potential
2. **Run training dynamics experiment**
3. **Conclude architecture exploration** - Document final findings
4. **Commit all experiment results** (3 scripts, 3 output dirs)

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
- **USER CLARIFICATION**: Training dynamics experiment should test BOTH transformers AND MLPs
- Goal: Close 3% gap to RF or prove it's impossible
