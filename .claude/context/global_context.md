# Global Project Context - 2026-01-27

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-27 21:00 | tier_a500 Sub-Chunks 6a+6b+7a COMPLETE (72 features, 278 total) |
| ws2 | foundation | paused | 2026-01-26 12:00 | HPO script created for iTransformer/Informer, awaiting runs |
| ws3 | phase6c_hpo_analysis | **active** | 2026-01-27 19:15 | **HPO analysis COMPLETE** - a50/a100 metrics captured, trends identified |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `fb5eeab` feat: Add tier_a500 Sub-Chunk 7a (23 VOL features)
- **Uncommitted**: 11 modified, 12 untracked (see ws3 files below)

### Test Status
- Last `make test`: 2026-01-27 19:00 - **1178 passed**, 2 skipped
- All tests pass

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined
  - a50: features + combined (55 features)
  - a100: features + combined (105 features)
  - a200: features + combined (206 features)
  - a500: IN PROGRESS (Sub-Chunks 6a+6b+7a complete, 278 features)

---

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws1 tier_a500]: IN PROGRESS - 3/12 sub-chunks complete
- [ws3 HPO Analysis]: **COMPLETE** - Comprehensive analysis done, supplementary trials proposed

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | COMMITTED |
| `experiments/templates/hpo_template.py` | ws3 | **MODIFIED** (verbose=True fix) |
| `docs/hpo_comprehensive_report.md` | ws3 | NEW |
| `docs/supplementary_hpo_proposal.md` | ws3 | NEW |
| `scripts/evaluate_top_hpo_models.py` | ws3 | NEW |

---

## Session Summary (2026-01-27 Evening - ws3)

### Work Completed

1. **Fixed HPO Template** (`experiments/templates/hpo_template.py`)
   - Changed `verbose=False` to `verbose=True` at line 327
   - Future HPO runs will capture precision, recall, pred_range

2. **Ran Top Model Evaluations**
   - Re-trained 9 top models for a50 with `verbose=True`
   - Re-trained 9 top models for a100 with `verbose=True`
   - Captured real precision/recall/pred_range metrics

3. **Created Comprehensive Report** (`docs/hpo_comprehensive_report.md`)
   - Executive summary with solid tables
   - Scaling law analysis (VIOLATED)
   - Precision-recall tradeoff analysis
   - Probability collapse detection (none found)
   - Optimal hyperparameters summary

4. **Analyzed 250 HPO Trials** - Identified trends:
   - Dropout 0.5 optimal (56.7% of top 60)
   - LR 1e-4 optimal (61.7% of top 60)
   - d_model 128 optimal (63.3% of top 60)
   - Bimodal depth: 2 layers OR 6-7 layers

5. **Proposed Supplementary Trials** (`docs/supplementary_hpo_proposal.md`)
   - 27 targeted trials exploring gaps in search space
   - Phase 1: Fine-tune dropout/LR/WD
   - Phase 2: Architecture variants
   - Phase 3: Combined optimization

### Key HPO Results

| Tier | Best AUC | Best Precision | Avg Recall |
|------|----------|----------------|------------|
| a50 | 0.7315 (20M) | 54.5% | 9.65% |
| a100 | 0.7189 (20M) | 50.0% | 11.4% |

**Critical finding**: More features HURT - a50 beats a100 by 1.3% AUC consistently.

---

## User Priorities

### ws3 (phase6c) - Next Steps
1. **Decide**: Run 27 supplementary trials or accept current results?
2. **If running**: ~15-30 min to implement and run
3. **If not**: Proceed to Phase 6C conclusions

### ws1 (feature_generation) - Queued
1. Continue with Sub-Chunk 7b (VLM Complete ~22 features)

### ws2 (foundation) - Queued
1. Run iTransformer HPO (50 trials)
2. Run Informer HPO (50 trials)

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

### Hyperparameters (HPO-Validated)
Based on analysis of 250 trials:
- **Dropout**: 0.5 (new finding from comprehensive analysis)
- **Learning Rate**: 1e-4 (confirmed)
- **Weight Decay**: 1e-4 to 1e-3 (higher than expected)
- **d_model**: 128 for 20M budget
- **n_layers**: 2 (shallow) or 6-7 (mid-deep) - bimodal
- **Context**: 80d (unchanged)
- **Normalization**: RevIN only (unchanged)
- **Splitter**: SimpleSplitter (unchanged)

---

## Key Insight

**Scaling laws are VIOLATED for this task:**
- Parameter scaling: 20M beats both 2M and 200M
- Feature scaling: 55 features beats 105 features consistently
- Regularization is critical: High dropout (0.5) dominates top performers
- Recall is the bottleneck: ~10% means missing 90% of opportunities
