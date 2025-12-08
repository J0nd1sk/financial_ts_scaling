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
- Training: through 2020
- Validation: 2021-2022
- Testing: 2023+

### Dataset Matrix
- Rows A-E: Asset scaling (SPY → +DIA/QQQ → +stocks → +econ)
- Cols a-d: Quality scaling (OHLCV+indicators → +sentiment → +VIX → +trends)

### Timescales (8)
daily, 2d, 3d, 5d, weekly, 2wk, monthly, daily+multi-resolution

### Tasks (6)
direction (binary), >1%/>2%/>3%/>5% thresholds (binary each), price regression

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

## Key Commands

```bash
# Environment
source venv/bin/activate
make test                    # Run ALL tests

# Data
python scripts/download_ohlcv.py
python scripts/calculate_indicators.py

# Training
python scripts/train.py --config configs/phase1/2M-daily-direction.yaml

# Thermal monitoring
sudo powermetrics --samplers smc -i 1000 | grep -i temp
```

---

## Directory Structure

```
financial_ts_scaling/
├── .claude/
│   ├── rules/              # Modular rules
│   ├── skills/             # Superpowers skills
│   └── context/            # Session state (git-tracked)
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

## Session Handoff Protocol

At session end or ~80% context, create handoff:

```markdown
# Session Handoff - [DATE]

## Current State
- Branch: [current branch]
- Last commit: [hash and message]
- Uncommitted changes: [list]

## Task Status
- Working on: [current task]
- Completed this session: [list]
- Blocked by: [if any]

## Test Status
- Last `make test` result: [pass/fail]
- Failing tests: [if any]

## Next Steps
1. [Immediate next task]
2. [Second priority]

## Key Decisions Made
- [Decision]: [Rationale]

## Files Modified
- [path]: [nature of change]

## Context for Next Session
[Any important context that would be lost]
```

Save to: `.claude/context/session_context.md`

---

## Session Restore Protocol

At session start:

1. Read `.claude/context/session_context.md`
2. Read `.claude/context/phase_tracker.md`
3. Run `make test` to verify environment
4. Summarize state to user
5. Confirm priorities before proceeding

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
