# Global Project Context - 2026-02-08

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | **COMPLETE** | 2026-01-31 | tier_a500 DONE - 500 features committed |
| ws2 | feature_embedding | paused | 2026-02-06 | Budget-aware HPO implemented |
| ws3 | feature_embedding_experiments | **ACTIVE** | 2026-02-08 23:45 | **114 extended experiments IMPLEMENTED** - 297 total |

## Shared State

### Branch & Git
- **Branch**: `main`
- **Last commit**: `7fc8680` feat: Add feature curation module, embedding experiments, and loss functions
- **Uncommitted** (17 new files + modifications):
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
  - `docs/feature_embedding_*.md` - Updated

### Test Status
- **Extended tests**: 40/40 passing
- **Pre-existing failures**: 92 (unrelated HPO/training tests)

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**: a20/a50/a100/a200/a500 parquet files

---

## Cross-Workstream Coordination

### Blocking Dependencies
- None currently

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | COMPLETE |
| `experiments/feature_embedding/*` | ws3 | 297 experiments |
| `src/data/augmentation.py` | ws3 | NEW |
| `src/training/curriculum.py` | ws3 | NEW |
| `src/training/regime.py` | ws3 | NEW |
| `src/models/multiscale.py` | ws3 | NEW |
| `src/training/contrastive.py` | ws3 | NEW |

---

## Session Summary (2026-02-08 - ws3)

### Extended Experiments IMPLEMENTED: 114 New Experiments

| Category | Prefix | Count | Key Module |
|----------|--------|-------|------------|
| Data Augmentation | DA | 24 | `src/data/augmentation.py` |
| Noise-Robust | NR | 18 | `src/training/losses.py`, `coteaching.py` |
| Curriculum Learning | CL | 18 | `src/training/curriculum.py` |
| Regime Detection | RD | 18 | `src/training/regime.py` |
| Multi-Scale Temporal | MS | 18 | `src/models/multiscale.py` |
| Contrastive Pre-training | CP | 18 | `src/training/contrastive.py` |

**Total experiments: 297** (183 original + 114 extended)

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

## Next Session (ws3) Should

1. **Commit all changes** - `git add -A && git commit`
2. **Run smoke test** - `--priority DA-P1`
3. **Integrate modules** - Some need trainer modifications
4. **Continue original experiments** - LF-P7 to AE-P5 pending
