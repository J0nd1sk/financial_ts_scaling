# Global Project Context - 2026-02-09

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | **COMPLETE** | 2026-01-31 | tier_a500 DONE - 500 features committed |
| ws2 | foundation | active | 2026-02-09 00:05 | Budget-aware HPO ready; helped ws3 debug |
| ws3 | feature_embedding_experiments | **BLOCKED** | 2026-02-08 | AE experiments failing - dimension mismatch |

## Shared State

### Branch & Git
- **Branch**: `main`
- **Last commit**: `7fc8680` feat: Add feature curation module, embedding experiments, and loss functions
- **Uncommitted** (17+ new files + modifications):
  - `src/data/augmentation.py` - Data augmentation transforms (NEW)
  - `src/training/coteaching.py` - Co-teaching trainer (NEW)
  - `src/training/curriculum.py` - Curriculum learning (NEW)
  - `src/training/regime.py` - Regime detection (NEW)
  - `src/models/multiscale.py` - Multi-scale temporal (NEW)
  - `src/training/contrastive.py` - Contrastive pre-training (NEW)
  - `tests/test_extended_experiments.py` - 40 tests (NEW)
  - `docs/extended_experiments.md` - Documentation (NEW)
  - 7 runner scripts (NEW)
  - `experiments/feature_embedding/run_experiments.py` - +114 experiments
  - `src/training/losses.py` - +3 noise-robust losses
  - `src/training/trainer.py` - OHLCV exclusion fix
  - `src/data/dataset.py` - OHLCV exclusion fix
  - `.claude/skills/planning_session/SKILL.md` - Subagent Execution Strategy

### Test Status
- **Extended tests**: 40/40 passing
- **Pre-existing failures**: 92 (unrelated HPO/training tests)
- **make test**: Blocked by lock file (ws3 running experiments)

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**: a20/a50/a100/a200/a500 parquet files

---

## Cross-Workstream Coordination

### ACTIVE BUG: ws3 AE Experiments Failing
- **Issue**: Dimension mismatch - models expect 100/500 features, data has 105/505
- **Partial fix applied** (by ws2 session):
  - `src/training/trainer.py` - NON_FEATURE_COLUMNS now includes OHLCV
  - `src/data/dataset.py` - EXCLUDED_COLUMNS now includes OHLCV
- **Status**: Fix verified in isolation but experiments still fail
- **Next**: ws3 needs to continue debugging data flow

### Blocking Dependencies
- ws3 blocked on AE dimension mismatch bug

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | COMPLETE |
| `experiments/feature_embedding/*` | ws3 | 297 experiments |
| `experiments/architectures/*` | ws2 | Budget-aware HPO |
| `src/data/augmentation.py` | ws3 | NEW |
| `src/training/curriculum.py` | ws3 | NEW |
| `src/training/regime.py` | ws3 | NEW |
| `src/models/multiscale.py` | ws3 | NEW |
| `src/training/contrastive.py` | ws3 | NEW |
| `src/training/trainer.py` | SHARED | OHLCV fix applied |
| `src/data/dataset.py` | SHARED | OHLCV fix applied |

---

## Session Summary (2026-02-08 Late - ws2)

### Work Done
1. **Updated planning_session skill** - Added Step 7.5 "Define Subagent Execution Strategy"
2. **Helped ws3 debug AE experiments** - Fixed OHLCV exclusion in shared files
3. **ws2 priorities not advanced** - Diverted to help ws3

### Best Results (Prior Sessions)
| Exp | Tier | Precision | Loss |
|-----|------|-----------|------|
| LF-40 | a500 | **85.7%** | LabelSmoothing |
| FE-06 | a100 | 69.2% | BCE |
| FE-50 | a500 | 60.0% | BCE |

---

## User Preferences (Authoritative)

### Development Approach
- TDD: Tests before implementation
- Planning sessions before coding
- Uses tmux for long-running experiments

### Context Durability
- Memory MCP for critical state
- `.claude/context/` for session files
- `docs/` for project documentation

### Documentation Philosophy
- Prefer consolidation over deletion
- Maintain precise historical context
- Flat `docs/` structure (no subfolders except research_paper/)

### Communication Standards
- Precision in language
- Never reduce fidelity of descriptions
- No summarizing away details

### Hyperparameters (Fixed - Ablation-Validated)
- **Dropout**: 0.5
- **Learning Rate**: 1e-4
- **Context Length**: 80 days
- **Normalization**: RevIN only
- **Splitter**: SimpleSplitter

---

## Next Session Priorities

### ws2 (foundation)
1. Run Phase 1 Forced Extremes HPO (18 configs + TPE trials)
2. Analyze results, identify top 2 budgets
3. Commit accumulated changes

### ws3 (feature_embedding_experiments)
1. **Continue debugging AE dimension mismatch** - trace data flow
2. After fix: Run AE-P1 smoke test
3. Commit all accumulated changes (17+ files)
