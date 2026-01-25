# Global Project Context - 2026-01-25

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | tier_a100 | COMPLETE | 2026-01-25 | All 8 chunks done, 100 features total, uncommitted |
| ws2 | foundation | active | 2026-01-25 09:30 | TFM-01 ran (AUC 0.364 anti-correlated), notebook needs fixing |
| ws3 | phase6c | active | 2026-01-25 18:30 | Threshold sweeps done, need a20 sweeps for comparison |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `f7dd1f9` feat: Phase 6C threshold sweeps + multi-workstream context system
- **Uncommitted**: tier_a100.py, context files, foundation notebook changes

### Test Status
- Last `make test`: 2026-01-25 09:00 - **621 passed**, 2 failed, 2 skipped
- 2 failing tests in test_tier_a100.py (Chunk 5 tests, TDD - tests ahead of implementation)

### Data Versions
- Raw manifest: SPY/DIA/QQQ OHLCV daily (v1)
- Processed manifest: SPY/DIA/QQQ features a20 (v1)
- Pending registrations: none

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 COMPLETE - unblocks ws3]: tier_a100 now has 100 features, ready for Phase 6C a100 experiments
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

### Key Research Findings

**Scaling × Feature Interaction Effect (2026-01-25)**:
At a20 features, 200M params wins. At a50 features, 20M wins. Novel finding for research paper.

**Foundation Model Anti-Correlation (2026-01-25)**:
TimesFM TFM-01 AUC = 0.364 (anti-correlated). Even inverted = 0.636, below PatchTST 0.718.

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
