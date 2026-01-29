# Global Project Context - 2026-01-29

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-29 15:00 | tier_a500 Sub-Chunk 10b COMPLETE (445 features, 1664 tests pass) |
| ws2 | foundation | active | 2026-01-29 09:30 | v3 HPO COMPLETE - evaluation bug found, needs fix + audit |
| ws3 | phase6c_hpo_analysis | paused | 2026-01-27 19:15 | HPO analysis COMPLETE - a50/a100 metrics captured |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `6b7fb96` feat: Add tier_a500 Sub-Chunk 10a
- **Uncommitted**: `experiments/architectures/hpo_neuralforecast.py` (v3 changes)

### Test Status
- Last `make test`: 2026-01-29 ~15:00 - **1664 passed**, 5 failed (threshold_sweep - pre-existing), 2 skipped
- 56 new tests added for Sub-Chunk 10b

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: features + combined (25 features)
  - a50: features + combined (55 features)
  - a100: features + combined (105 features)
  - a200: features + combined (206 features)
  - a500: 300 features (needs regeneration for 445)

---

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws2 foundation]: v3 HPO complete but evaluation bug found - metrics invalid
- [ws1 tier_a500]: 10/12 sub-chunks complete (445/500 features)
- [ws3 HPO Analysis]: COMPLETE

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | 445 features implemented (10b done) |
| `experiments/architectures/hpo_neuralforecast.py` | ws2 | MODIFIED - v3 changes |
| `experiments/architectures/common.py` | ws2 | NEEDS FIX - evaluation bug |
| `outputs/hpo/architectures/itransformer/` | ws2 | v3 results (50 trials) |
| `outputs/hpo/architectures/informer/` | ws2 | v3 results (50 trials) |

---

## Session Summary (2026-01-29 - ws2)

### v3 HPO Implementation & Bug Discovery

**What Was Done:**
1. Implemented v3 changes to `hpo_neuralforecast.py`:
   - Changed `MSE()` to `DistributionLoss(distribution='Bernoulli')`
   - Changed target from returns to binary `threshold_target`
2. Ran smoke test (3 trials) - verified predictions in [0, 1]
3. User ran full HPO: 50 trials iTransformer, 50 trials Informer

**Critical Bug Found:**
- `evaluate_forecasting_model()` uses `return_threshold=0.01` (designed for v1/v2 regression)
- For Bernoulli outputs [0, 1], this gives 100% recall (everything > 0.01)
- All v3 precision/recall metrics are INVALID

**Actual Prediction Distribution (verified separately):**
- Only 7.5% of predictions >= 0.5
- Mean: 0.23, Median: 0.19
- Models predict mostly NEGATIVE (opposite of what HPO metrics showed)

---

## User Priorities

### ws2 (foundation) - ACTIVE - Next Session
1. **Fix evaluation bug** in `common.py`
   - Use threshold=0.5 for Bernoulli/classification outputs
2. **Re-evaluate all 100 trials** with correct threshold
3. **Comprehensive HPO audit** - verify no other issues
4. **Proper threshold sweep** - show precision/recall separately
5. **Compare to PatchTST** at same thresholds

### ws1 (feature_generation) - Queued
1. Commit current changes (10a + 10b)
2. Regenerate data for 445 features
3. Continue with Sub-Chunk 11a (ADV Part 1)

### ws3 (phase6c) - Paused
- Analysis complete, supplementary trials optional

---

## User Preferences (Authoritative)

### Metric Priority for Analysis (CRITICAL)
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

**KEY**: Precision should INCREASE with higher probability thresholds. If it doesn't, model hasn't learned meaningful discrimination.

Show precision and recall SEPARATELY across threshold changes.

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
- **Dropout**: 0.5 (high regularization critical)
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-4 to 1e-3
- **d_model**: 128 for 20M budget
- **n_layers**: 2 (shallow) or 6-7 (mid-deep) - bimodal
- **Context**: 80d
- **Normalization**: RevIN only
- **Splitter**: SimpleSplitter

---

## Key Insights

**v3 Evaluation Bug (2026-01-29):**
- HPO used threshold=0.01 (designed for regression outputs)
- For classification probabilities, this gives 100% recall (meaningless)
- Need to re-evaluate with threshold=0.5

**Scaling laws are VIOLATED for this task:**
- Parameter scaling: 20M beats both 2M and 200M
- Feature scaling: 55 features beats 105 features consistently
- Regularization is critical: High dropout (0.5) dominates

**Methodology lesson:**
- Always match training objective to evaluation objective
- Always match evaluation threshold to output type
