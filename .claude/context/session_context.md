# Session Handoff - 2026-01-22 ~10:30 UTC

## Current State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation` (NEW - this terminal)
- **Main branch status**: `47d7481` docs: documentation cleanup and Phase 6A corrections
- **Uncommitted changes on main**:
  - `CLAUDE.md` (+Feature Engineering Principle section)
  - `docs/feature_engineering_exploration.md` (+719 lines)
- **This branch changes**: Plan doc + updates to phase_plans, history, tracker

### Active Work Streams
Two parallel work streams active:
1. **This terminal**: Foundation Model & Decoder Architecture Investigation
2. **Other terminal**: Phase 6C Feature Engineering exploration

---

## Test Status
- Last `make test`: 2026-01-22
- Result: **476 passed**, 2 warnings
- Failing: none

---

## Completed This Session (Architecture Investigation Terminal)

1. **Session restore** - Verified environment, reviewed Phase 6A conclusions
2. **Architecture research** - Investigated decoder vs encoder for time series:
   - TimesFM (Google): Decoder-only, open source, fine-tunable
   - Lag-Llama (Salesforce): Decoder-only, probabilistic, open source
   - iTransformer: Inverted attention (ICLR 2024 Spotlight)
   - TimeMixer: Pure MLP, no attention
   - LENS: Financial-specific, NO public weights
3. **Created investigation plan** - `docs/foundation_decoder_investigation_plan.md`
4. **Updated documentation**:
   - `docs/project_phase_plans.md` - Added stage section
   - `docs/project_history.md` - Added section 6.15
   - `.claude/context/phase_tracker.md` - Added stage tracking
5. **Created git branch** - `experiment/foundation-decoder-investigation`

---

## Investigation Overview (This Terminal)

### Motivation
Phase 6A found data-limited regime: 200M only +1.7% over 2M. Question: Is encoder-only PatchTST architecture the limitation?

### Models to Evaluate
| Model | Type | Priority |
|-------|------|----------|
| Lag-Llama | Decoder, Foundation | Tier 1 |
| TimesFM | Decoder, Foundation | Tier 1 |
| iTransformer | Inverted attention | Tier 2 |
| TimeMixer | MLP (no attention) | Tier 2 |

### Success Criteria
- ≥5% AUC improvement over PatchTST → Pursue this path
- Within ±2% → Return to feature scaling (Phase 6C)

---

## Feature Engineering Work (Other Terminal)

### Core Principle (User-Established)
**THE #1 GOAL**: Give the neural network the ability to discern signal from noise.
- Raw indicator values are NOISE
- Relationships and dynamics are SIGNAL

### Key User Insights Captured
1. Volume-price confluence matters SIGNIFICANTLY
2. Nested channels - Price can be in parabolic channel inside larger range-bound channel
3. Entropy ≠ volatility - Entropy = predictability, volatility = magnitude
4. Hurst 0.4-0.6 likely = range-bound, not random walk
5. Elliott Wave maps accumulation → impulse → distribution cycle
6. Self-fulfilling prophecy - Explicitly encode trader-used combinations

### Research Discoveries
- SMC Python library: github.com/joshyattridge/smart-money-concepts
- Entropy measures can detect regime instability BEFORE price shows it

---

## Memory Entities Updated

**This session (architecture):**
- `Foundation_Decoder_Investigation_20260122` - Investigation rationale and plan

**From feature engineering session:**
- `Feature_Engineering_Core_Principle_20260122` - Core principle and expanded categories
- `Feature_Exploration_Riffs_20260122` - Agent's riffs on user leads
- `Feature_Exploration_Session2_20260122` - Session 2 detailed leads and findings

**Still valid:**
- `Phase6A_Conclusion_DataLimited_20260122` - Data-limited regime confirmed
- `Target_Calculation_Definitive_Rule` - HIGH-based targets

---

## Data Versions

- Raw manifest: SPY.OHLCV.daily (verified)
- Processed manifest: SPY_dataset_a20.parquet (verified)
- Pending registrations: none

---

## Next Session Should (Architecture Terminal)

1. **Commit current changes** - Plan doc and documentation updates
2. **Start Task 1: Environment Setup**
   - Install GluonTS for Lag-Llama
   - Install TimesFM
   - Verify MPS compatibility
   - Create `src/models/foundation/` module
3. **Begin Task 2: Lag-Llama Integration** (highest priority)

---

## Commands to Run First

```bash
source venv/bin/activate
make test
git status
git branch --show-current
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

### Current Focus (Two Streams)
1. **Architecture Investigation** (this terminal): Foundation models & decoder architectures
2. **Feature Engineering** (other terminal): Phase 6C feature exploration
