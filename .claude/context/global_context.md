# Global Project Context - 2026-01-25

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-25 15:00 | Deep validation script DONE (uncommitted) |
| ws2 | foundation | active | 2026-01-25 19:45 | Fixed JSON bug in cell-27, ready to re-run in Colab |
| ws3 | phase6c | active | 2026-01-25 18:30 | Threshold sweeps done, need a20 sweeps |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `4231cbf` fix: VIX date alignment
- **Uncommitted**: ws1 validation script + ws3 phase6c files

### Test Status
- Last `make test`: 2026-01-25 - **692 passed**, 2 skipped
- tier_a100 validation: 69/69 checks pass

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20 features (v1)

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 COMPLETE]: tier_a100 validated, ready for Phase 6C a100

### File Ownership
| Files | Owner |
|-------|-------|
| `scripts/validate_tier_a100.py` | ws1 |
| `outputs/validation/*` | ws1 |
| `experiments/phase6c/*` | ws3 |

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
