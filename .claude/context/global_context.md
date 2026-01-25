# Global Project Context - 2026-01-25

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | tier_a100 | active | 2026-01-25 | Chunk 7 complete (40/50), Chunk 8 next (final) |
| ws2 | foundation | paused | 2026-01-24 17:30 | TimesFM notebook fixed, ready for Colab |
| ws3 | phase6c | active | 2026-01-25 18:30 | Threshold sweeps done, need a20 sweeps for comparison |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `e854cb4` feat: Add comprehensive tier_a50 validation script
- **Uncommitted**: ~20 files (context system + tier_a100 + foundation experiments)

### Test Status
- Last `make test`: 2026-01-25 - **637 passed**, 2 skipped
- All tests passing

### Data Versions
- Raw manifest: SPY/DIA/QQQ OHLCV daily (v1)
- Processed manifest: SPY/DIA/QQQ features a20 (v1)
- Pending registrations: none

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 enables ws3]: tier_a100 features needed for Phase 6C a100 experiments
- ws1 at 80% complete (40/50 indicators) - Chunk 8 is final
- [ws3 needs a20 threshold sweeps]: Complete a20 vs a50 comparison blocked on a20 threshold analysis

### Shared Resources
- Both ws1 and ws3 may modify `src/features/` - coordinate commits
- Foundation work (ws2) is independent - can run in parallel

### File Ownership
| Files | Primary Owner | Shared With |
|-------|--------------|-------------|
| `src/features/tier_a100.py` | ws1 | - |
| `tests/features/test_tier_a100.py` | ws1 | - |
| `experiments/foundation/*` | ws2 | - |
| `experiments/phase6c/*` | ws3 | - |
| `outputs/phase6c/*` | ws3 | - |
| `outputs/phase6a_final/*` | ws3 (analysis) | - |

### Key Research Finding (2026-01-25)
**Scaling × Feature Interaction Effect**: At a20 features, 200M params wins. At a50 features, 20M wins. This is a novel finding for the research paper.

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

### Hyperparameters (Fixed - Ablation-Validated)
Always use unless new ablation evidence supersedes:
- **Dropout**: 0.5
- **Learning Rate**: 1e-4
- **Context Length**: 80 days
- **Normalization**: RevIN only (no z-score)
- **Splitter**: SimpleSplitter (442 val samples)
- **Metrics**: AUC, accuracy, precision, recall, pred_range (all required)

---

## Commands to Run First (Any Workstream)

```bash
source venv/bin/activate
make test
git status
make verify
```
