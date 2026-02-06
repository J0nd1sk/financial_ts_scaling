# Global Project Context - 2026-02-06

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | **COMPLETE** | 2026-01-31 16:45 | tier_a500 DONE - 500 features committed + data regenerated |
| ws2 | foundation | **active** | 2026-02-01 10:00 | Two-phase budget-aware HPO IMPLEMENTED (18 forced configs, 76 tests pass) |
| ws3 | a200_hpo_v3 | **active** | 2026-02-06 14:15 | HPO v3 READY (precision-first + loss HPO, 20 tests pass) |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `41f1da4` feat: Add tier_a500 Sub-Chunks 11a+11b (55 ADV features, 500 total)
- **Uncommitted** (pending commit):
  - `experiments/phase6c_a200/hpo_20m_h1_a200_v3.py` - **NEW: v3 HPO script**
  - `tests/test_hpo_a200_v3.py` - **NEW: 20 tests**
  - `src/training/losses.py` - FocalLoss, WeightedBCELoss
  - `tests/test_losses.py` - loss function tests
  - `src/training/hpo_budget_extremes.py` - budget-aware HPO module (ws2)
  - `tests/test_hpo_budget_extremes.py` - 30 tests (ws2)
  - `docs/hpo_strategy_phase6.md` - strategy documentation
  - Context files, skills, various outputs

### Test Status
- **ws3 tests (make test-ws3)**: 236/236 passed
- **v3 HPO tests**: 20/20 passed
- **Full suite**: Some feature tests may have pre-existing issues

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: 25 features
  - a50: 55 features
  - a100: 105 features
  - a200: 206 features (**VERIFIED**: 212 cols, 7977 rows, 0 NaN)
  - a500: 500 features (v2 - COMPLETE)

---

## Cross-Workstream Coordination

### Blocking Dependencies
- None currently

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | **COMPLETE** |
| `experiments/architectures/common.py` | ws2 | +DATA_PATH_A200, DATA_PATHS, get_data_path() |
| `experiments/architectures/hpo_neuralforecast.py` | ws2 | +--data-tier argument |
| `experiments/phase6c_a200/hpo_20m_h1_a200_v3.py` | ws3 | Precision-first HPO |
| `src/training/losses.py` | SHARED | FocalLoss, WeightedBCELoss |

---

## Session Summary (2026-02-06 - ws3)

### Session Restore After Crash
- Computer crashed during HPO v3 run (only trial 0 completed)
- Context files were stale (described v2, actual state is v3)
- Updated ws3_context.md and global_context.md
- All v3 tests pass (20/20)

### HPO v3 Key Features
1. **Composite objective**: `precision*2 + recall*1 + auc*0.1`
2. **Loss type as hyperparameter**: `focal` vs `weighted_bce`
3. **Conditional params**: `focal_alpha/gamma` or `bce_pos_weight`
4. **Multi-threshold metrics**: t30, t40, t50, t60, t70
5. **80d context** (per CLAUDE.md)

---

## Session Summary (2026-02-01 - ws2)

### Two-Phase Budget-Aware HPO Implemented
- Created `src/training/hpo_budget_extremes.py` with 18 forced extreme configs
- 30 tests pass
- Addresses methodology gap (random params in single-trial runs)

---

## Session Summary (2026-01-31 - ws1 FINAL)

### tier_a500 COMPLETE
- Committed `41f1da4`: tier_a500 Sub-Chunks 11a+11b (55 ADV features, 500 total)
- Regenerated data: 906 rows, 500 features, v2 in manifest

---

## User Priorities

### ws1 (feature_generation) - COMPLETE
tier_a500 done. Next: push to remote if desired.

### ws2 (foundation) - Ready
1. Commit budget-aware HPO changes
2. Run validation with optimal params

### ws3 (a200_hpo_v3) - Ready
1. ~~Update stale context~~ ✅
2. Commit v3 work
3. Run 50-trial HPO (~2-3 hours)

---

## User Preferences (Authoritative)

### Metric Priority for Analysis (CRITICAL)
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments
- **Workstream-specific testing**: `make test-ws{N}` for fast iteration

### Context Durability
- Multiple places: Memory MCP + context files + docs/
- Code comments are secondary

### Documentation Philosophy
- Flat docs/ (no subdirs except research_paper/, archive/)
- Precision - never reduce fidelity
- Consolidate rather than delete

### Hyperparameters (HPO-Validated)
- **Dropout**: 0.3-0.6 (v3 search space)
- **Learning Rate**: 5e-5 to 1.5e-4
- **Weight Decay**: 1e-5 to 1e-3
- **d_model**: 64-192 depending on budget
- **n_layers**: 4-8
- **Context**: 80d (standard)
- **Normalization**: RevIN only
- **Splitter**: SimpleSplitter

---

## Key Insights

### HPO v3 Strategy (2026-02-01)
- **Precision-first composite**: Optimizes what we actually care about
- **Loss function HPO**: Focal vs weighted_bce can significantly impact precision/recall tradeoff
- **Multi-threshold logging**: Enables post-hoc analysis of precision-recall curves

### Context Ablation Results (2026-01-31)
- a200 @ 75d = 66.7% precision, 7.8% recall (best combo)
- a200 @ 80d = 0.730 AUC (best ranking)
- v3 uses 80d for consistency with CLAUDE.md

### tier_a500 Complete (2026-01-31)
- 500 features achieved across 12 sub-chunks (6a-11b)
- Data regenerated: 906 rows, 500 features, v2 in manifest
