# Session Handoff - 2026-01-19 ~18:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `0c15642` docs: session handoff with normalization fix complete
- **Uncommitted changes**: 7 files (mix of both terminals' work)
  - Modified: session_context.md, phase6a_implementation_audit.md, patchtst.py, trainer.py, test_patchtst.py
  - New: phase6a_best_practices_audit.md, phase6a_research_gap_analysis.md, test_revin_comparison.py
- **Ahead of origin**: 10 commits (not pushed)

### Task Status
- **Pipeline Validation Terminal**: ChunkSplitter bug identified, SimpleSplitter approved
- **RevIN Terminal**: Working on architecture alignment (RevIN, layer count, etc.)
- **Status**: Foundation fixes in progress across both terminals

---

## CRITICAL DISCOVERY: ChunkSplitter Bug (19 Validation Samples!)

### The Problem
ChunkSplitter gives **only 19 validation samples**, not ~500 as expected!

| What We Expected | What Actually Happens |
|------------------|----------------------|
| 15% of ~8000 rows for val | 15% of 132 chunks = 19 chunks |
| ~500+ val samples | **19 val samples** (1 per chunk) |
| Reliable HPO decisions | High variance, unreliable HPO |

**Root Cause**: Design gives 1 sample per chunk. Training uses sliding window (~5700 samples) but val/test get only ONE sample per non-overlapping chunk.

**Impact**: All previous HPO decisions were based on 19 samples - unreliable!

### Approved Fix: SimpleSplitter
- Simple time-based contiguous splits
- Sliding window for ALL splits (train, val, test)
- Strict containment: sample valid only if entire span within region
- Implementation: ~50 lines, clean and verifiable

---

## Research Findings: Official PatchTST vs Ours

| Parameter | Official | Ours | Gap |
|-----------|----------|------|-----|
| **RevIN** | Always enabled | Missing | 🔴 CRITICAL |
| **n_layers** | 2-4 | 12-256 | 🔴 8-85x more |
| **context_length** | 336-512 | 60 | 🟠 5-8x shorter |
| **pos_encoding init** | std=0.02 | std=1.0 | 🟡 50x larger |
| **dropout** | 0.2 | 0.1-0.3 | OK |

Sources:
- https://github.com/yuqinie98/PatchTST
- https://context7.com/yuqinie98/patchtst
- https://openreview.net/pdf?id=cGDAkQo1C0p (RevIN paper)

---

## User Preferences for Experiments

### Data Split Preference (IMPORTANT)
- **Train**: Maximize - everything before 2023 (~7,476 samples, 90.8%)
- **Val**: 2023-2024 (~442 samples, 6.0%) - for early stopping only
- **Test**: 2025+ (~201 samples, 3.1%) - backtest on most recent data

**Rationale**: Limited training data, so maximize it. Most recent data for realistic backtesting.

### Architecture Experiments (Future - Don't Assume Official Is Best)
User explicitly stated:
> "I'm not sure reducing the layers is necessarily better, but I would like to experiment with the difference between wide and shallow (e.g., layers constrained to 2-6) and what long but more narrow."

> "How would increasing context length impact training? We don't have a long length of data right now. Maybe we just double or triple the context length?"

**Action**: After fixing foundation (splits, RevIN), run controlled experiments:
1. Wide/shallow (2-6 layers) vs narrow/deep
2. Context length: 60 → 120 → 180 (double/triple)
3. Empirical validation, not just copying official config

---

## Test Status
- Last `make test`: 2026-01-19 (start of session)
- Result: **425 passed**

---

## Completed This Session (Pipeline Terminal)
1. Session restore and context review
2. Research into official PatchTST configuration (Context7, web search)
3. **Critical**: Identified ChunkSplitter 19-sample bug
4. Designed SimpleSplitter replacement (user approved)
5. Created Memory entities for findings

## Completed by Other Terminal
1. Z-score normalization (AUC improved to 0.6488)
2. Implementation audit (docs/phase6a_implementation_audit.md)
3. Gap analysis (docs/phase6a_research_gap_analysis.md)
4. Working on RevIN implementation

---

## Next Session Priorities

### Immediate (Pipeline Terminal)
1. **Implement SimpleSplitter** with strict containment (TDD)
   - Date-based contiguous splits
   - Sliding window for ALL splits
   - Expected: Train ~7476, Val ~442, Test ~201 samples

### Coordinate with RevIN Terminal
2. Check RevIN progress
3. Review uncommitted changes
4. Merge foundation fixes

### After Foundation Fixed
5. Re-run HPO with proper splits + RevIN + normalization
6. Architecture experiments (wide/shallow vs narrow/deep)
7. Context length experiments

---

## Memory Entities Updated This Session
- `Bug_ChunkSplitter_19Samples` (created): Critical finding - val has only 19 samples
- `Solution_SimpleSplitter_Design` (created): Approved fix design
- `Research_PatchTST_Official_Config` (created): Official hyperparameters
- `Research_Architecture_Experiments_Pending` (created): User's experiment preferences

## Memory Entities from Previous Sessions
- `Bug_FeatureNormalization_Phase6A` - Root cause of distribution shift
- `Plan_ZScoreNormalization` - Completed normalization fix
- `Phase6A_DoubleSigmoidBug` - Earlier sigmoid issue
- `Phase6A_PriorCollapse_RootCause` - Prior collapse analysis

---

## Data Versions
- Raw manifest: SPY.parquet (8299 rows, 1993-2026)
- Processed manifest: SPY_dataset_c.parquet in data/processed/v1/
- Pending registrations: None

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
- Fix foundation before more experimentation
- Validate empirically, don't assume official config is best
- Maximize training data, minimize val/test
