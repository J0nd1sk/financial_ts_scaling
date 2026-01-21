# Session Handoff - 2026-01-21 ~07:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `da4abdb` exp: Dropout scaling - 20M_wide beats RF by 1.8% (AUC 0.7342)
- **Uncommitted changes**: none
- **Ahead of origin**: 9 commits (not pushed)

### Task Status
- **Working on**: Architecture scaling experiments
- **Status**: Dropout scaling complete, next: test even shallower L=2/L=4

---

## Dropout Scaling Results (This Session)

### BREAKTHROUGH: 20M_wide beats Random Forest!

| Config | d | L | Params | AUC | vs RF (0.716) |
|--------|---|---|--------|-----|---------------|
| **20M_wide** | 512 | 6 | 19M | **0.7342** | **+1.8%** ⭐ |
| 20M_balanced | 384 | 12 | 21M | 0.7282 | +1.2% |
| 20M_narrow | 256 | 32 | 25M | 0.7253 | +0.8% |
| 200M_balanced | 768 | 24 | 170M | 0.7225 | +0.6% |
| 200M_narrow | 512 | 48 | 152M | 0.7214 | +0.5% |
| 200M_wide | 1024 | 12 | 152M | 0.7204 | +0.4% |

### Key Findings
1. **WIDE > NARROW at larger scales** - opposite of 2M finding!
2. **Shallower is better at 20M**: L=6 > L=12 > L=32
3. **200M doesn't improve over 20M** - data-limited regime
4. **Some scaling IS happening** even with limited data

---

## Next Session: Test Even Shallower Architectures

User wants to test L=2 and L=4 at 20M budget:

| Config | d | L | h | ~Params | Rationale |
|--------|---|---|---|---------|-----------|
| 20M_L2 | ~640 | 2 | 8 | ~20M | Ultra-shallow |
| 20M_L3 | ~576 | 3 | 8 | ~20M | Very shallow |
| 20M_L4 | ~544 | 4 | 8 | ~20M | Matches 2M optimal depth |
| 20M_L5 | ~528 | 5 | 8 | ~20M | Between L4 and L6 |

Need to calculate exact d_model values to hit ~20M params.

---

## Test Status
- Last `make test`: 2026-01-21 ~23:00 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Files Created/Modified This Session

| File | Description |
|------|-------------|
| `scripts/test_dropout_scaling.py` | Width vs depth experiment at 20M/200M |
| `outputs/dropout_scaling/dropout_scaling_results.csv` | Results (6 configs) |
| `docs/architecture_exploration_conclusion_DRAFT.md` | Draft summary (may not retain) |

---

## Memory Entities Updated This Session

**Created:**
- `Plan_DropoutScalingExperiment_20260121` - Planning decision for dropout scaling
- `Finding_DropoutScalingExperiment_20260121` - BREAKTHROUGH: 20M_wide 0.7342 beats RF

**From previous sessions (relevant):**
- `Finding_TrainingDynamicsExperiment_20260121` - Dropout=0.5 works at 2M
- `Finding_SmallModels_20260120` - Smaller models don't help at 2M
- `Finding_ShallowWide_20260120` - At 2M, narrow-deep beats shallow-wide

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

1. **Calculate d_model for L=2,3,4,5** to hit ~20M params
2. **Run 4 additional trials** with shallower 20M architectures
3. **Analyze depth sweet spot** - is L=4-6 optimal at 20M?
4. **Then**: Plan feature/indicator expansion experiments

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
- Architecture exploration: Optimal depth/width at different scales
- **Observation**: "Some degree of scaling going on, even with this little data"
- Next: Test L=2-4 at 20M, then feature expansion experiments
