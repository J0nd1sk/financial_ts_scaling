# Session Handoff - 2026-01-20 ~10:00 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `40ca9f9` feat: add SimpleSplitter to fix 19-sample validation bug
- **Uncommitted changes**: None (clean)
- **Ahead of origin**: 12 commits (not pushed)

### Task Status
- **Status**: SimpleSplitter + RevIN validation COMPLETE
- **All tests passing**: 443 tests

---

## Completed This Session

### 1. SimpleSplitter Implementation (TDD)
- Added `SimpleSplitter` class to `src/data/dataset.py` (~120 lines)
- Date-based contiguous splits with sliding window for ALL regions
- Strict containment: sample valid only if entire span within region
- 11 new tests in `tests/test_dataset.py`

### 2. RevIN Validation Re-run
- Updated `scripts/test_revin_comparison.py` to use SimpleSplitter
- Re-ran comparison with 442 val samples (was 19!)

---

## CRITICAL FINDING: RevIN Validation Results Reversed

Previous conclusions with 19 samples were **completely wrong**:

| Config | Old AUC (19 samples) | New AUC (442 samples) | Verdict |
|--------|---------------------|----------------------|---------|
| zscore_only | 0.716 | **0.476** | Was "best", actually worst |
| revin_only | 0.471 | **0.667** | Was "worst", actually BEST |
| zscore_revin | 0.739 | **0.515** | Combination hurts |

### Key Insight
- **RevIN alone is best** (per-instance normalization adapts to current price scale)
- **Z-score preprocessing hurts** when combined with RevIN
- Global Z-score anchors to historical stats; RevIN adapts per-sample

### Normalization Explained
| Method | Mechanism | Best for |
|--------|-----------|----------|
| Z-score | Global: `(x - train_mean) / train_std` | Stationary data |
| RevIN | Per-instance: each window normalized by its own mean/std | Non-stationary (prices trending over decades) |

---

## Test Status
- Last `make test`: 2026-01-20
- Result: **443 passed** (was 431)
- New tests: 12 SimpleSplitter tests

---

## Files Modified This Session
- `src/data/dataset.py`: +120 lines (SimpleSplitter class)
- `tests/test_dataset.py`: +180 lines (11 test methods)
- `scripts/test_revin_comparison.py`: ~10 lines (splitter swap)
- `outputs/revin_comparison/comparison_results.csv`: Updated results

---

## Memory Entities Updated This Session
- `Plan_SimpleSplitter_RevIN_Validation` (created): Planning decision
- `Finding_RevIN_Validation_442Samples` (created): Research finding with reversed conclusions

## Memory Entities from Previous Sessions
- `Bug_ChunkSplitter_19Samples` - The bug we just fixed
- `Solution_SimpleSplitter_Design` - Approved design we implemented
- `Research_PatchTST_Official_Config` - Official hyperparameters
- `Bug_FeatureNormalization_Phase6A` - Earlier normalization discovery

---

## Next Session Priorities

### Immediate
1. **Update HPO scripts** to use SimpleSplitter + RevIN only (no Z-score)
2. **Re-run HPO** with proper validation (442 samples)
3. Update phase_tracker.md with SimpleSplitter completion

### Architecture Experiments (after foundation solid)
4. Wide/shallow (2-6 layers) vs narrow/deep experiments
5. Context length experiments (60 → 120 → 180)

---

## Data Versions
- Raw manifest: SPY.parquet (8299 rows, 1993-2026)
- Processed manifest: SPY_dataset_a20.parquet (8100 rows)
- SimpleSplitter splits: Train 7277, Val 442, Test 201 samples

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

### Current Research Direction
- RevIN alone (per-instance normalization) is best approach
- No global Z-score preprocessing needed
- Validate empirically, don't assume official config is best
