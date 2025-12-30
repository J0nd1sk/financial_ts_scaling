# Documentation Consolidation Plan

**Created**: 2025-12-29
**Status**: APPROVED, ready for execution
**Priority**: High - blocks 2B HPO resumption

---

## Objective

Consolidate and update Phase 6A documentation to:
1. Reflect truth (what's actually in the repository)
2. Preserve precision and fidelity of decisions
3. Reduce fragmentation (5 stale plan files → 1 history doc)
4. Update active docs to current reality

---

## Task Breakdown

### Task 1: Create `phase6a_implementation_history.md`

**Purpose**: Archive completed implementation work with decisions preserved.

**Source files to consolidate**:
| File | Lines | Content to Extract |
|------|-------|-------------------|
| `hpo_fixes_plan.md` | 259 | 6 tasks, problem summary, rationale |
| `hpo_time_optimization_plan.md` | 533 | 6 tasks, memory findings, technical design |
| `architectural_hpo_implementation_plan.md` | 252 | 8 tasks, execution order, rollback plan |
| `hpo_supplemental_tests.md` | 176 | Why superseded, original strategy |
| `feature_pipeline_integration_issues.md` | 133 | Issues discovered, fixes applied, lessons |

**Structure**:
```markdown
# Phase 6A Implementation History

## Overview
- Timeline: Dec 11-29, 2025
- Total tasks completed: 20
- Key outcome: Architectural HPO with dynamic batch sizing

## Stage 1: Architectural HPO Implementation (Dec 11-12)
### Problem Statement
[From architectural_hpo_implementation_plan.md]
### Tasks Completed (8/8)
[Task list with key decisions]
### Files Created
- src/models/arch_grid.py
- configs/hpo/architectural_search.yaml
- tests/test_arch_grid.py

## Stage 2: HPO Fixes (Dec 13)
### Problem Statement
[From hpo_fixes_plan.md - n_heads logging, grid gaps, forced extremes]
### Tasks Completed (6/6)
[Task list]
### Key Decisions
- Added L=160, 180 to n_layers grid
- Forced extreme testing in first 6 trials

## Stage 3: HPO Time Optimization (Dec 26-29)
### Problem Statement
[From hpo_time_optimization_plan.md - memory exhaustion, no early stopping]
### Tasks Completed (6/6)
1. get_memory_safe_batch_config() - dynamic batch sizing
2. Gradient accumulation in Trainer
3. Early stopping in Trainer
4. Wire HPO to use new features (dropout search, dynamic batch)
5. Regenerate 12 HPO scripts + graceful stop
6. Integration smoke test (3 trials, 2B budget)
### Key Discoveries
- PatchTST already has dropout support (was hardcoded at 0.1)
- Memory heuristic: memory_score = (d_model² × n_layers) / 1e9
### Files Modified
- src/models/arch_grid.py (+get_memory_safe_batch_config)
- src/training/trainer.py (+accumulation_steps, +early_stopping)
- src/training/hpo.py (+dynamic batch, +dropout sampling)
- configs/hpo/architectural_search.yaml (+dropout, +early_stopping, -batch_size)

## Superseded: Supplemental Tests Plan
### Original Strategy
[From hpo_supplemental_tests.md]
- Targeted tests to fill gaps (L=64 for 2M, L=160/180 for 20M)
### Why Changed
- Decision: Full re-runs with new scripts instead of supplemental tests
- Rationale: New scripts have all fixes, cleaner than patching old runs

## Lessons Learned
### Feature Pipeline Integration (Dec 11)
[From feature_pipeline_integration_issues.md]
- vix_regime was stored as strings, caused runtime crash
- Fix: Integer encoding (0=low, 1=normal, 2=high)
- Principle: Data should be "model-ready" after processing

### Session Handoff Gap (Dec 29)
- Plan files not updated when tasks complete
- Only phase_tracker.md reflects truth
- TODO: Revise session_handoff skill to update all applicable docs
```

**Estimated length**: ~400-500 lines

---

### Task 2: Update `config_architecture.md` to v2.0

**Current version**: 1.0 (Dec 8, 2025)
**Target version**: 2.0

**Changes required**:

| Section | Current | Update To |
|---------|---------|-----------|
| Budgets in diagram | 2M, 20M, 200M | 2M, 20M, 200M, **2B** |
| Model config box | Static `patchtst_{budget}.yaml` | Dynamic via architectural HPO |
| Batch size box | Static `batch_sizes.json` | Dynamic `get_memory_safe_batch_config()` |
| HPO box | `default_search.yaml`, training params only | `architectural_search.yaml`, arch + training |
| File structure | Missing architectural_search.yaml | Add to configs/hpo/ |
| HPO Results JSON | Missing architecture field | Add architecture object |
| Phase table | "Next", "Planned", "Future" | All COMPLETE |

**Key additions**:
- Section on architectural HPO (link to architectural_hpo_design.md)
- Gradient accumulation and early stopping
- Dynamic batch sizing based on architecture
- 2B budget considerations

---

### Task 3: Update `phase6a_execution_plan.md` to v2.0

**Current version**: 1.0 (Dec 11, 2025)
**Target version**: 2.0

**Changes required**:

| Section | Current | Update To |
|---------|---------|-----------|
| Status line | (none) | **Status: Stage 4 in progress** |
| Stage 1 | Checklist unchecked | ✅ COMPLETE |
| Stage 2 | Checklist unchecked | ✅ COMPLETE (horizon variance confirmed) |
| Stage 3 | Options A/B | ✅ COMPLETE (Option A: 12 HPO runs per budget) |
| Stage 4 | Checklist unchecked | 🔄 IN PROGRESS |
| HPO Scripts table | 3 rows, some ⏳ | 12 rows, status per run |
| Estimated Runtime | Original estimates | Actual results + revised 2B estimates |
| Key Decisions | 4 items from Dec 11 | Add Dec 13, Dec 21, Dec 26-29 decisions |

**Add new sections**:
- HPO Results Summary (link to experiment_results.csv)
- Remaining Work (2B HPO, final analysis)
- Lessons Learned (link to implementation_history.md)

---

### Task 4: Delete Consolidated Files

**After Tasks 1-3 verified**, delete:
1. `docs/hpo_fixes_plan.md`
2. `docs/hpo_time_optimization_plan.md`
3. `docs/architectural_hpo_implementation_plan.md`
4. `docs/hpo_supplemental_tests.md`
5. `docs/feature_pipeline_integration_issues.md`

**Verification before delete**:
- [ ] All key decisions captured in history doc
- [ ] All lessons learned captured
- [ ] No orphaned references in other docs

---

### Task 5: Revert N_TRIALS

**File**: `experiments/phase6a/hpo_2B_h1_threshold_1pct.py`
**Line**: ~50
**Change**: `N_TRIALS = 3` → `N_TRIALS = 50`
**Reason**: Smoke test complete, ready for production

---

### Task 6: Commit All Changes

```bash
git add -A
git commit -m "docs: consolidate Phase 6A implementation history

- Create phase6a_implementation_history.md (20 tasks archived)
- Update config_architecture.md to v2.0 (add 2B, dynamic batch)
- Update phase6a_execution_plan.md to v2.0 (reflect current status)
- Delete 5 stale plan files (content preserved in history)
- Revert N_TRIALS=50 for production 2B HPO

🤖 Generated with Claude Code

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Update `project_phase_plans.md` to v3.0 (FUTURE)

**Purpose**: Master roadmap document is stale (v2.0, Nov 2025). Needs update to reflect reality.

**Current gaps**:
- Phase 5: Shows "Task 1/8 complete" — actually COMPLETE
- Phase 5.5: Shows "PROPOSED" — actually COMPLETE
- Phase 6A: Shows original 32-run design — pivoted to 12 architectural HPO runs
- Missing: 2B budget, dynamic batch sizing, architectural HPO, gradient accum, early stopping

**Changes required**:
- Mark Phase 5/5.5 COMPLETE
- Rewrite Phase 6A section to match actual design (architectural HPO, 12 runs, 1% threshold only)
- Add Phase 6A-prep section documenting infrastructure work
- Revise Phase 6B/6C/6D estimates based on lessons learned
- Update runtime estimates with actual data

**Priority**: LOW — defer until after 2B HPO complete
**Estimated effort**: 2-3 hours

---

## Execution Order

```
Task 1: Create phase6a_implementation_history.md ✅ DONE
    ↓
Task 2: Update config_architecture.md v2.0 ← NEXT
    ↓
Task 3: Update phase6a_execution_plan.md v2.0
    ↓
Task 4: Verify + Delete 5 consolidated files
    ↓
Task 5: Revert N_TRIALS = 50 ✅ DONE (collapsed into earlier commit)
    ↓
Task 6: git add -A && git commit
    ↓
Resume 2B HPO (50 trials × 3 horizons)
    ↓
Task 7: Update project_phase_plans.md v3.0 (FUTURE - after 2B HPO)
```

---

## Success Criteria

- [ ] `phase6a_implementation_history.md` contains all 20 tasks with decisions
- [ ] `config_architecture.md` v2.0 reflects current architecture
- [ ] `phase6a_execution_plan.md` v2.0 shows Stage 1-3 complete, Stage 4 in progress
- [ ] 5 stale files deleted
- [ ] `N_TRIALS = 50` in hpo_2B_h1_threshold_1pct.py
- [ ] `make test` passes (361 tests)
- [ ] All changes committed

---

## Files Affected Summary

| Action | File |
|--------|------|
| CREATE | `docs/phase6a_implementation_history.md` |
| UPDATE | `docs/config_architecture.md` |
| UPDATE | `docs/phase6a_execution_plan.md` |
| DELETE | `docs/hpo_fixes_plan.md` |
| DELETE | `docs/hpo_time_optimization_plan.md` |
| DELETE | `docs/architectural_hpo_implementation_plan.md` |
| DELETE | `docs/hpo_supplemental_tests.md` |
| DELETE | `docs/feature_pipeline_integration_issues.md` |
| MODIFY | `experiments/phase6a/hpo_2B_h1_threshold_1pct.py` |

---

*Plan Version: 1.0*
*Approved: 2025-12-29*
