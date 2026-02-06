# Session Handoff - 2026-01-23 ~06:00 UTC

## Current State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `e25e680` docs: Foundation Model & Decoder Architecture Investigation plan
- **Uncommitted changes**:
  - `.claude/context/session_context.md` (this file)
  - `docs/feature_engineering_exploration.md` (from previous session)
  - `docs/indicator_catalog.md` (NEW - **v0.2 COMPLETE this session**)

### Active Work Streams
Two parallel work streams active:
1. **Architecture Terminal**: Foundation Model & Decoder Architecture Investigation
2. **Feature Terminal (THIS SESSION)**: Phase 6C Feature Engineering - **Indicator catalog v0.2 COMPLETE**

---

## Test Status
- Last `make test`: 2026-01-23 (at session restore)
- Result: **476 passed**, 2 warnings
- Failing: none
- **Note**: No code changes this session - only documentation edits

---

## Completed This Session

1. **Indicator Catalog v0.2 Revision** - ALL 12 TASKS COMPLETE
   - Task 1: Document header & revision notes updated
   - Task 2: Consolidation Patterns section added
   - Task 3: Category 1 (Moving Averages) - consolidated binaries, added accelerations
   - Task 4: Category 2 (Oscillators) - StochRSI section, signed extremes
   - Task 5: Category 3 (Volatility) - Gaussian Channel, Keltner Channel, signed distances
   - Task 6: Category 4 (Volume) - added accelerations, consolidated
   - Task 7: Category 5 (Trend) - ADX expansion, SuperTrend slope/accel
   - Task 8: Category 6-7 (S/R & Candlestick) - expanded patterns (~32 candlestick patterns)
   - Task 9: Category 8-9 (Momentum & Calendar) - removed redundant is_* binaries
   - Task 10: Category 10-12 (Entropy, Regime, MTF) - added derivatives
   - Task 11: Category 13 (SMC) - added derivatives, library reference
   - Task 12: Grand Total & Data Lookback Impact section

2. **Verification Passed**:
   - No remaining `days_above_` in feature definitions (only in consolidation notes)
   - No remaining `days_below_` in feature definitions
   - No remaining `max(0,` in feature definitions
   - No remaining binary calendar features in definitions
   - Gaussian Channel section present
   - StochRSI section present
   - Data Lookback Impact section present

### Document Stats
- Lines: 791 → 1114 (+323 lines)
- Total Features: ~2,133 → ~1,895 (**~11% reduction**)
- Net: -238 features from consolidation + additions

---

## Key Changes Applied in v0.2

### 4 Consolidation Patterns
1. **Signed Duration**: `days_above_` + `days_below_` → `days_since_cross` (signed)
2. **Signed Distance**: `value - threshold` instead of `max(0, value - threshold)`
3. **Signed Extremes**: `days_at_extreme` (+overbought, -oversold, 0 neutral)
4. **Slope + Acceleration**: Every slope has corresponding acceleration feature

### New Sections Added
- Consolidation Patterns (reference section)
- StochRSI (momentum of momentum)
- Gaussian Channel
- Keltner Channel
- Data Lookback Impact

---

## Files Modified

- `docs/indicator_catalog.md`: +323 lines, v0.1 → v0.2 complete revision

---

## Next Session Should

### Feature Engineering Terminal
1. **Review indicator catalog v0.2** - user review for any further adjustments
2. **Commit documentation** (exploration + catalog v0.2)
3. **Begin tier assignment** (a50/a100/a200/a500/a1000/a2000)
4. **Prioritize categories** - which features to implement first

### Architecture Terminal
1. Commit current changes (plan doc and documentation updates)
2. Start Task 1: Environment Setup (GluonTS, TimesFM, MPS compatibility)
3. Begin Task 2: Lag-Llama Integration

---

## Data Versions

- Raw manifest: SPY.OHLCV.daily (verified)
- Processed manifest: SPY_dataset_a20.parquet (verified)
- Pending registrations: none

---

## Memory Entities Updated

**This session:**
- No Memory MCP updates (documentation-only session)

**Still valid from previous session:**
- `Indicator_Catalog_Revision_Plan_20260123` - Plan that was executed
- `Feature_Engineering_Core_Principle_20260122` - Core principle
- `Feature_Exploration_Session2_20260122` - Feature leads
- `Foundation_Decoder_Investigation_20260122` - Architecture investigation

---

## Commands to Run First

```bash
source venv/bin/activate
make test
git status
make verify
```

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
- **Head dropout**: 0.0 (ablation showed no benefit)
- **Metrics**: AUC, accuracy, precision, recall, pred_range (all required)

### Feature Engineering Principles
- Signed features consolidate information (one feature with sign, not two separate)
- Continuous > binary for neural networks
- Every slope needs acceleration
- Neural nets learn thresholds from continuous values (no need for is_january, etc.)

### Current Focus (Two Streams)
1. **Architecture Investigation** (other terminal): Foundation models & decoder architectures
2. **Feature Engineering** (this terminal): Phase 6C - **Indicator catalog v0.2 COMPLETE, ready for review**
