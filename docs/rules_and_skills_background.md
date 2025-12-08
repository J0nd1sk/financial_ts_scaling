# Rules and Skills Architecture
## Financial TS Transformer Scaling Project

---

## Purpose

LLM coding agents (Claude Code, Cursor) are powerful but undisciplined. Without explicit constraints, they:
- Make unnecessary changes
- Add complexity and technical debt
- Forget context mid-session
- Skip tests or run partial test suites
- Proceed without approval on destructive operations

This document describes the rules and skills architecture that enforces disciplined development workflow for this experimental ML research project.

---

## Core Philosophy

### 1. Tests Before Code

Every code change starts with tests:
1. **Propose** what tests need to change or be added
2. **Wait** for approval
3. **Write** the failing tests
4. **Run** `make test` to confirm they fail as expected
5. **Write** minimal implementation to pass
6. **Run** `make test` to confirm all tests pass
7. **Only then** proceed to git operations

The agent must never write implementation code before defining tests.

### 2. No Change Without Approval

The agent proposes. The human approves. The agent executes.

- No code changes without explicit approval
- No fixing errors discovered mid-task without separate approval
- No batching multiple fixes together
- No filesystem changes (create/delete files/directories) without approval
- No git operations without approval

### 3. Small, Atomic Changes

Each change should be:
- Single logical unit
- Reviewable in isolation
- Testable independently
- Reversible

Large refactors are decomposed into approved sub-tasks.

### 4. Zero Tolerance for Technical Debt

The agent must never:
- Add "temporary" workarounds
- Leave TODO comments for later
- Skip edge cases for speed
- Introduce unnecessary abstractions
- Add dependencies without justification

### 5. Context Continuity

Sessions end. Context windows fill. The agent maintains continuity via:
- **Handoff protocol**: Structured context dump at session end
- **Restore protocol**: Context reload at session start
- **State tracking**: Persistent files track phase progress and decisions

---

## File Architecture

```
financial_ts_scaling/
├── CLAUDE.md                          # Claude Code primary entry point
├── .claude/
│   └── rules/                         # Claude Code modular rules
│       ├── global.md                  # Index + universal rules
│       ├── experimental-protocol.md   # Scaling law constraints
│       ├── testing.md                 # TDD enforcement
│       ├── development-discipline.md  # Approval gates, RACI
│       └── context-handoff.md         # Session continuity
├── .cursor/
│   └── rules/                         # Cursor rules (same content, .mdc format)
│       ├── global.mdc
│       ├── experimental-protocol.mdc
│       ├── testing.mdc
│       ├── development-discipline.mdc
│       └── context-handoff.mdc
├── .claude/
│   └── skills/                        # Superpowers skills (Claude Code only)
│       ├── session_handoff/
│       │   └── SKILL.md
│       ├── session_restore/
│       │   └── SKILL.md
│       ├── planning_session/
│       │   └── SKILL.md
│       └── test_first/
│           └── SKILL.md
└── .claude/
    └── context/                       # Persistent state (git-tracked)
        ├── phase_tracker.md           # Phase progress
        ├── decision_log.md            # Architectural decisions
        └── session_context.md         # Latest handoff state
```

---

## Rules Documents

### CLAUDE.md (Root)

The comprehensive entry point for Claude Code. Contains:
- Project overview and research objectives
- Experimental constraints (non-negotiable)
- Thermal management thresholds
- RACI matrix for human-LLM collaboration
- Development workflow summary
- Session handoff template
- Key commands

This is the "constitution" - the agent reads it at session start.

### Modular Rules (.claude/rules/ and .cursor/rules/)

Split by concern for maintainability:

| File | Purpose |
|------|---------|
| `global.md` | Index of other rules, git workflow, file preservation |
| `experimental-protocol.md` | Parameter budgets (2M/20M/200M only), batch size rules, architecture constraints |
| `testing.md` | `make test` enforcement, coverage targets, test naming, TDD workflow |
| `development-discipline.md` | RACI matrix, approval gates, small change requirement, no tech debt |
| `context-handoff.md` | Handoff protocol, context file location, restore protocol |

### Format Difference

- `.claude/rules/*.md` - Markdown for Claude Code
- `.cursor/rules/*.mdc` - MDC format for Cursor (YAML frontmatter + markdown)

Content is identical; format differs for each tool's expectations.

---

## Skills (Claude Code Only)

Skills are structured prompts that enforce specific workflows. Invoked via `/skill <name>`.

### session_handoff

**Trigger**: End of session, context ~80% full, or user requests handoff

**Purpose**: Capture complete session state for next session

**Output**: Writes `.claude/context/session_context.md` containing:
- Current task and status
- Pending work items
- Files modified this session
- Git state (branch, uncommitted changes)
- Key decisions made
- Test status
- Next steps for incoming session

### session_restore

**Trigger**: Start of new session

**Purpose**: Restore context from previous session

**Process**:
1. Read `.claude/context/session_context.md`
2. Read `.claude/context/phase_tracker.md`
3. Summarize state to user
4. Confirm priorities before proceeding
5. Run `make test` to verify environment

### planning_session

**Trigger**: Before any code work begins

**Purpose**: Force explicit planning before implementation

**Output**: Structured plan covering:
- Objective (what are we trying to accomplish?)
- Success criteria (how do we know we're done?)
- Assumptions (what are we taking for granted?)
- Risks (what could go wrong?)
- Test plan (what tests change/add?)
- Approval request

Agent cannot proceed without explicit approval of the plan.

### test_first

**Trigger**: Before implementing any feature or fix

**Purpose**: Enforce TDD workflow

**Process**:
1. Define test cases for the change
2. Write test stubs (failing)
3. Run `make test` to confirm failure
4. Present to user for approval
5. Only after approval: write implementation
6. Run `make test` to confirm all pass

---

## Key Enforcement Rules

### Testing Discipline

```
🔴 CRITICAL: ALWAYS run `make test` - NEVER run individual tests

FORBIDDEN:
- pytest tests/specific_test.py
- pytest -k test_method_name
- ./venv/bin/pytest with any arguments

REQUIRED:
- make test (runs full suite)
- Run BEFORE any git operations
- Run AFTER any code change
- All tests must pass before proceeding
```

### Approval Gates

```
NEVER proceed without explicit approval for:
- Any code changes
- Any test changes
- Creating/deleting files
- Git operations (add, commit, push)
- Fixing errors (even self-introduced)
- Installing dependencies
```

### Change Size

```
Each change must be:
- Small enough to review in isolation
- Single logical unit
- Independently testable
- Described in one sentence

If a change requires multiple sentences to describe, decompose it.
```

---

## Experimental Protocol Constraints

These are non-negotiable for scientific rigor:

| Constraint | Enforcement |
|------------|-------------|
| Parameter budgets: 2M, 20M, 200M only | No intermediate values permitted |
| Batch size re-tuning | REQUIRED when parameter budget changes |
| Batch size re-tuning | RECOMMENDED when dataset changes |
| Architecture | PatchTST only for clean scaling isolation |
| One model per task | 48 models per dataset (6 tasks × 8 timescales) |
| Train/val/test splits | →2020 / 2021-22 / 2023+ (fixed) |

---

## Thermal Protocol

M4 MacBook Pro 128GB, basement cooling (50-60°F ambient):

| Temperature | Action |
|-------------|--------|
| <70°C | Normal operation |
| 70-85°C | Acceptable, monitor |
| 85-95°C | Warning, consider pause |
| >95°C | CRITICAL STOP immediately |

Agent must respect these thresholds during training runs.

---

## Workflow Summary

### Standard Development Loop

```
1. Planning session (skill or manual)
   - Define objective, tests, success criteria
   - Get approval

2. Test-first development
   - Write/modify tests
   - Confirm they fail
   - Get approval for implementation

3. Implementation
   - Minimal code to pass tests
   - Run make test
   - Fix issues (with approval for each fix)

4. Commit preparation
   - Run make test (final verification)
   - All tests pass
   - Get approval for commit

5. Git operations
   - git add -A
   - git commit with descriptive message
```

### Session Lifecycle

```
Session Start:
1. Run session_restore skill (or read context manually)
2. Verify environment: make test
3. Confirm priorities with user

Session Work:
- Follow standard development loop
- Propose → Approve → Execute

Session End:
1. Run session_handoff skill
2. Verify context file written
3. Note any uncommitted work
```

---

## Integration Notes

### Claude Code + Cursor

Both tools read their respective rules directories. Content is synchronized manually (copy .md to .mdc with frontmatter).

### Superpowers Skills

Only available in Claude Code. Cursor users must invoke the same workflows manually or through custom prompts.

### Context Files

`.claude/context/` is git-tracked. Session context persists across machines and sessions.

---

## Maintenance

### Adding Rules

1. Add to appropriate .md file in `.claude/rules/`
2. Copy to `.cursor/rules/` with .mdc extension and frontmatter
3. Update CLAUDE.md if rule is fundamental

### Modifying Skills

1. Edit SKILL.md in skill directory
2. Test by invoking skill
3. Iterate based on output quality

### Versioning

Rules are versioned with the project. Major workflow changes should be noted in decision_log.md.