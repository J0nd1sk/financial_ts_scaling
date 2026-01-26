# Global Project Context - 2026-01-25

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-25 21:30 | tier_a200 Chunk 1 COMPLETE (uncommitted) - 20 new MA indicators |
| ws2 | foundation | active | 2026-01-25 23:55 | Created TFM-07/08/09/10 notebooks for a50/a100 covariate experiments |
| ws3 | phase6c | **BLOCKED** | 2026-01-25 23:50 | a100 pipeline created but NOT WORKING - troubleshooting needed |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `d88d76a` feat: Add tier_a100 deep validation script
- **Uncommitted**: tier_a200 files, foundation notebooks, plus previous work

### Test Status
- Last `make test`: 2026-01-25 23:55 - **747 passed**, 2 skipped
- tier_a100 validation: 69/69 checks pass
- tier_a200 tests: 55/55 pass

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20, a50, a100 (v1) - both features-only and _combined versions
- tier_a200: Module complete, no processed data yet

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 BLOCKED]: Runner script `run_s1_a100.sh` does nothing when executed

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a200.py` | ws1 (NEW) |
| `experiments/foundation/TimesFM_*.ipynb` | ws2 |
| `experiments/phase6c_a100/*` | ws3 (UNTESTED) |
| `scripts/run_s1_a100.sh` | ws3 (NOT WORKING) |

---

## Data File Naming Convention (Important!)

Two versions of processed datasets exist:
- `SPY_dataset_a50.parquet` - Features ONLY (no OHLCV)
- `SPY_dataset_a50_combined.parquet` - OHLCV + Features

**For experiments**: Always use `_combined` versions!

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Multiple places: Memory MCP + context files + docs/
- Code comments are secondary

### Documentation Philosophy
- Flat docs/ (no subdirs except research_paper/, archive/)
- Precision - never reduce fidelity
- Consolidate rather than delete

### Hyperparameters (Fixed - Ablation-Validated)
- Dropout: 0.5, LR: 1e-4, Context: 80d, RevIN only, SimpleSplitter
