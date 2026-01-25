# Financial TS Transformer Scaling Experiments

## Project Overview

Experimental research testing whether neural scaling laws apply to transformer models trained on financial time-series data. Using PatchTST architecture for clean parameter scaling isolation.

**Research Questions:**
1. Does increasing parameters (2M → 20M → 200M) improve accuracy following power law?
2. Does increasing features (20 → 2000 indicators) improve accuracy?
3. Does increasing data diversity (SPY → multi-asset → cross-asset) improve accuracy?
4. How do these scaling dimensions interact?

**Target:** Publishable findings on scaling law applicability to financial ML.

---

## Project Terminology

**Hierarchy (top to bottom):**
- **Phase**: Major project milestone (e.g., Phase 6A: Parameter Scaling)
- **Stage**: Focused work block within a phase (e.g., "HPO Time Optimization Stage")
- **Task**: Discrete deliverable with tests (e.g., "Add early stopping to Trainer")
- **Subtask**: Atomic step within a task (e.g., "Write failing test", "Implement method")

**Rules:**
- Phases are defined in `phase_tracker.md` and rarely change
- Stages may have their own temporary plan documents in `docs/`
- Stage plan documents are deleted or archived when stage completes
- Tasks are tracked in workstream context files (`workstreams/ws{N}_context.md`) and TodoWrite
- Don't conflate levels — a stage is NOT a new phase

**Example:**
```
Phase 6A: Parameter Scaling
├── Stage: HPO Time Optimization (temporary detour)
│   ├── Task 1: Memory-safe batch config ✅
│   ├── Task 2: Gradient accumulation ✅
│   └── Task 3: Early stopping ✅
│   └── Task 4: Dropout parameter ← CURRENT
├── HPO Runs (main work, resumes after stage)
└── Results Analysis
```

---

## Development Discipline Rules

### 🔴 CRITICAL: Testing

**ALWAYS run `make test` - NEVER run individual tests**

```bash
# ✅ CORRECT - ALWAYS USE THIS:
make test

# ❌ FORBIDDEN:
pytest tests/specific_test.py
pytest -k test_method_name
./venv/bin/pytest [anything]
```

- Run `make test` BEFORE any `git add`
- Run `make test` AFTER any code or dependency change
- ALL tests must pass before proceeding
- No exceptions for "quick checks"

### 🔴 CRITICAL: Approval Gates

**NEVER proceed without explicit user approval for:**

1. Any code changes
2. Any test changes
3. Creating/deleting files or directories
4. Git operations (add, commit, push)
5. Fixing errors - even errors you introduced
6. Installing or updating dependencies
7. Modifying configuration files

**Workflow:**
1. PROPOSE the change with rationale
2. WAIT for explicit "yes" / "approved" / "go ahead"
3. Only then EXECUTE

### 🔴 CRITICAL: Test-Driven Development

**Tests come BEFORE implementation:**

1. Propose what tests need to change/add
2. Wait for approval
3. Write failing tests
4. Run `make test` to confirm failure
5. Propose implementation
6. Wait for approval
7. Write minimal code to pass
8. Run `make test` to confirm all pass

### Change Size

- Each change: single logical unit
- Reviewable in isolation
- Describable in one sentence
- If it needs multiple sentences, decompose it

### Zero Technical Debt

Never:
- Add "temporary" workarounds
- Leave TODO comments
- Skip edge cases
- Add unnecessary abstractions
- Add dependencies without justification

### 🔴 CRITICAL: Documentation Organization

**ALL active documentation must be placed in `docs/` - no subfolders allowed.**

Exceptions:
- `docs/research_paper/` - evidence and notes for eventual publication
- `docs/archive/` - superseded documents preserved for historical reference

- `docs/` contains all active project documentation (plans, designs, references)
- No other subfolders (e.g., `docs/plans/`, `docs/drafts/`)
- `.claude/context/` is for session state (global_context, phase_tracker, decision_log, workstreams/)
- `.claude/context/workstreams/` contains per-workstream context files (ws1, ws2, ws3)
- `.claude/rules/` is for agent rules only
- `.claude/skills/` is for Claude Code skills only

**Rationale:** Flat structure for active docs prevents sprawl; archive preserves history without clutter.

---

## Experimental Protocol (Non-Negotiable)

### Parameter Budgets
- **2M, 20M, 200M ONLY**
- No intermediate values
- Clean isolation for scaling law tests

### Batch Size Re-Tuning
- **REQUIRED** when parameter budget changes
- **RECOMMENDED** when dataset changes
- **OPTIONAL** when feature count changes

### Architecture
- PatchTST exclusively
- One model per task
- 48 models per dataset (6 tasks × 8 timescales)

### Data Splits (Fixed)
- Training: through 2022-12-31
- Validation: 2023-01-01 to 2024-12-31
- Testing: 2025-01-01 onwards

### Dataset Matrix
- Rows A-E: Asset scaling (SPY → +DIA/QQQ → +stocks → +econ)
- Cols a-d: Quality scaling (OHLCV+indicators → +sentiment → +VIX → +trends)

### Timescales (8)
daily, 2d, 3d, 5d, weekly, 2wk, monthly, daily+multi-resolution

### Tasks (6)
direction (binary), >1%/>2%/>3%/>5% thresholds (binary each), price regression

### Hyperparameters (Fixed - Ablation-Validated)
Based on ablation studies (2026-01), always use:
- **Dropout**: 0.5 (high regularization)
- **Learning Rate**: 1e-4 (stable convergence)
- **Context Length**: 80 days (optimal from context ablation)
- **Normalization**: RevIN only (no global z-score)
- **Splitter**: SimpleSplitter (not ChunkSplitter - the latter gives only 19 val samples)

These are non-negotiable unless new ablation evidence supersedes them.

### Target Calculation (Multi-Horizon)
For horizon H days, target is TRUE if:
```
max(High[t+1], High[t+2], ..., High[t+H]) >= Close[t] * (1 + threshold)
```
This means: did the price reach the threshold at ANY point within the horizon?

### Metrics (Required for All Experiments)
Always track and report:
- AUC-ROC (discrimination)
- Accuracy (overall correctness)
- Precision (of positive predictions)
- Recall (of actual positives) - **critical: 0% recall = useless model**
- Prediction range [min, max] (detect probability collapse)

### Feature Engineering Principle (Phase 6C)

**THE #1 GOAL**: Give the neural network the ability to discern signal from noise.

**Raw indicator values are noise.** An SMA of 450 vs 150 means nothing without context.

**Relationships and dynamics are signal:**
- **Position relative to level**: % distance of price from MA (negative = below)
- **Duration**: Days price has been above/below a level (overextension signal)
- **Slope**: Rate of change of the indicator itself (trend direction/strength)
- **Acceleration**: Change in slope (momentum shifts before price)
- **Cross proximity**: % difference between two MAs (how close to crossing)
- **Recency of events**: Days since last cross (decaying influence)

**Example**: Being above the 50 MA is bullish, but less so if the 50 MA's slope is near 0 or negative. This nuance is lost with raw MA values.

**Implementation**: See `docs/feature_engineering_exploration.md` for detailed feature designs.

---

## Thermal Protocol

M4 MacBook Pro 128GB, basement cooling (50-60°F ambient)

| Temperature | Action |
|-------------|--------|
| <70°C | Normal operation |
| 70-85°C | Acceptable, monitor |
| 85-95°C | Warning, consider pause |
| >95°C | **CRITICAL STOP** immediately |

---

## RACI Matrix

| Activity | Human (Alex) | Agent |
|----------|--------------|-------|
| **Planning** |
| Experimental design | Lead, Accountable | Consult |
| Architecture decisions | Lead, Accountable | Propose |
| Task breakdown | Approve | Propose |
| **Development** |
| Writing tests | Review, Approve | Execute |
| Writing implementation | Review, Approve | Execute |
| Refactoring | Approve | Propose |
| Code review | Execute | Inform |
| Git operations | Approve | Execute |
| **Execution** |
| Data downloads | Monitor | Execute |
| Model training | Monitor thermal | Execute |
| HPO | Review strategy | Execute |

**Key:**
- **Lead** = Makes decisions, drives work
- **Execute** = Does the work
- **Approve** = Final sign-off required
- **Propose** = Suggests approach
- **Review** = Evaluates quality

---

## Git Workflow

### Commands
```bash
# Always use git add -A (never git add . or specific files)
git add -A
git commit -m "type: description"
```

### Commit Types
- `feat:` New feature
- `fix:` Bug fix
- `test:` Test changes
- `refactor:` Restructuring
- `docs:` Documentation
- `data:` Data pipeline
- `exp:` Experiment config

### Branch Strategy
```
main          # Published results only
staging       # Integration, all tests pass
feature/*     # Active work
experiment/*  # Individual experiment runs
```

---

## Data Versioning Policy

- **Manifests**: `data/raw/manifest.json` and `data/processed/manifest.json` track every dataset artifact with `{dataset, path, md5, timestamp}` (processed entries also store `version`, `tier`, `source_raw_md5s`).
- **Registration**: Use `python scripts/manage_data_versions.py register-raw ...` (or `register-processed`) immediately after writing new parquet files. Raw artifacts are immutable—new downloads get new files + manifest entries.
- **Verification**: `make verify` now runs both environment checks and `python scripts/manage_data_versions.py verify`. Handoff/restore summaries must include latest manifest entries and pending registrations.
- **Processed versions**: Increment `version` whenever processing logic, feature tiers, or source data change; capture the raw md5 list for reproducibility.

---

## Key Commands

```bash
# Environment
source venv/bin/activate
make test                    # Run ALL tests

# Data
./venv/bin/python scripts/download_ohlcv.py
./venv/bin/python scripts/calculate_indicators.py

# Training
./venv/bin/python scripts/train.py --config configs/phase1/2M-daily-direction.yaml

# Thermal monitoring
sudo powermetrics --samplers smc -i 1000 | grep -i temp
```

### 🔴 CRITICAL: Always Use venv Python

**NEVER use system Python. ALWAYS use the virtual environment.**

```bash
# ✅ CORRECT - Use venv Python:
./venv/bin/python scripts/script.py
./venv/bin/python -c "import torch; print(torch.__version__)"
source venv/bin/activate && python scripts/script.py

# ❌ FORBIDDEN - System Python lacks dependencies:
python scripts/script.py
python3 scripts/script.py
```

**Rationale:** All dependencies (PyTorch, Optuna, pandas-ta, etc.) are installed in `venv/`, not system Python. Using system Python causes `ModuleNotFoundError`.

---

## Directory Structure

```
financial_ts_scaling/
├── .claude/
│   ├── rules/              # Modular rules
│   ├── skills/             # Superpowers skills
│   └── context/            # Session state (git-tracked)
│       ├── global_context.md     # All workstreams summary
│       ├── phase_tracker.md      # Phase progress
│       ├── decision_log.md       # Architectural decisions
│       └── workstreams/          # Per-workstream context
│           ├── ws1_context.md    # Workstream 1 (e.g., tier_a100)
│           ├── ws2_context.md    # Workstream 2 (e.g., foundation)
│           └── ws3_context.md    # Workstream 3 (e.g., phase6c)
├── data/
│   ├── raw/                # Immutable downloads
│   ├── processed/          # Versioned processed data
│   └── samples/            # CSV samples for LLM review
├── src/
│   ├── data/               # Download, validate
│   ├── features/           # Indicator calculations
│   ├── models/             # PatchTST configs
│   ├── training/           # Train loops, HPO
│   └── evaluation/         # Metrics
├── tests/
├── scripts/
├── outputs/
│   ├── checkpoints/
│   ├── results/
│   └── figures/
└── docs/
```

---

## Multi-Workstream Context System

This project supports **up to 3 parallel workstreams** (terminals) working simultaneously, each with independent context management.

### Workstream Structure

| ID | Example Name | Typical Focus |
|----|--------------|---------------|
| ws1 | tier_a100 | Feature implementation |
| ws2 | foundation | Foundation model investigation |
| ws3 | phase6c | Phase 6C experiments |

**Files:**
- `global_context.md` - Summary of all workstreams, shared state, coordination notes
- `workstreams/ws{N}_context.md` - Detailed per-workstream context

### Session Handoff Protocol

At session end or ~80% context:

1. **Determine workstream** - Auto-detect or ask user
2. **Update workstream context** - Write detailed state to `workstreams/ws{N}_context.md`
3. **Update global context** - Update summary row and shared state in `global_context.md`
4. **Update phase tracker** - If progress made
5. **Report to user** - Confirm both files updated, note other workstreams

### Session Restore Protocol

At session start:

1. **Read global context** - Understand all active workstreams
2. **Show workstreams summary** - Display active workstreams table
3. **Auto-detect workstream** - From user's first message/task keywords
4. **Confirm selection** - "Detected workstream: ws1 (tier_a100). Correct?"
5. **Read workstream context** - Get detailed context for selected workstream
6. **Verify environment** - Run `make test`, check git status
7. **Show cross-workstream notes** - Blocking dependencies, shared resources
8. **Confirm priorities** - Wait for user direction before proceeding

### Cross-Workstream Coordination

- **File Ownership**: Track which workstream owns which files (PRIMARY vs SHARED)
- **Blocking Dependencies**: Document when one workstream blocks another
- **Shared Resources**: Note when workstreams share resources requiring coordination

---

## User Preferences

### Development Discipline
- TDD approach: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability (Critical)
Pending actions MUST be documented in multiple places to survive crashes:
1. **Memory MCP** — create entities for critical state
2. **`.claude/context/`** — session_context.md, phase_tracker.md, decision_log.md
3. **`docs/`** — project documentation when appropriate

Code comments are secondary, not primary durability mechanisms.

### Documentation Principles
- Prefer consolidation of `docs/` files over deletion — preserve historical context
- Maintain coherent, PRECISE history of "what we did and why"
- Flat `docs/` structure — no subdirectories except `research_paper/`
- Precision in language — never reduce fidelity of descriptions

---

## Rules Reference

See `.claude/rules/` for detailed rules:
- `global.md` - Universal rules, git workflow
- `experimental-protocol.md` - Scaling law constraints
- `testing.md` - TDD enforcement
- `development-discipline.md` - Approval gates
- `context-handoff.md` - Session continuity

---

## Tech Stack

- Python 3.12, PyTorch MPS
- Optuna HPO, W&B + MLflow tracking
- pandas-ta + TA-Lib indicators
- Parquet storage

---

*Document Version: 1.0*
*Project: financial_ts_scaling*
