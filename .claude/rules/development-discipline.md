# Development Discipline Rules

## 🔴 CRITICAL: Approval Gates

**NEVER proceed without explicit user approval.**

### Requires Approval

1. **Any code changes** - even one-line fixes
2. **Any test changes** - additions, modifications, deletions
3. **Creating files** - even "obviously needed" ones
4. **Deleting files** - no exceptions
5. **Git operations** - add, commit, push, branch, merge
6. **Fixing errors** - even errors you introduced
7. **Installing dependencies** - pip install, brew install
8. **Modifying configs** - any configuration file
9. **Formatting/linting** - no auto-formatters without approval

### Approval Workflow

```
1. PROPOSE
   - State what you want to do
   - State why
   - State what tests are affected
   - State risks

2. WAIT
   - Do not proceed on assumption
   - Do not proceed on silence
   - Explicit "yes" / "approved" / "go ahead" required

3. EXECUTE
   - Only the approved change
   - Nothing extra
   - No "while I'm here" additions
```

---

## RACI Matrix

| Activity | Human | Agent |
|----------|-------|-------|
| Experimental design | Lead | Consult |
| Architecture decisions | Lead | Propose |
| Task breakdown | Approve | Propose |
| Writing tests | Approve | Execute |
| Writing implementation | Approve | Execute |
| Refactoring | Approve | Propose |
| Code review | Execute | Inform |
| Git operations | Approve | Execute |
| Data operations | Monitor | Execute |
| Training | Monitor | Execute |

**Key:**
- **Lead**: Makes decisions, drives work
- **Approve**: Must sign off before proceeding
- **Execute**: Does the actual work
- **Propose**: Suggests approach for approval
- **Consult**: Provides input and expertise
- **Inform**: Keeps other party updated
- **Monitor**: Watches for issues

---

## Change Size Limits

### Single Logical Unit
Each change must be:
- Describable in one sentence
- Reviewable in isolation
- Testable independently
- Reversible without cascade effects

### Decomposition Required
If a change requires:
- Multiple sentences to describe
- Touching more than 3 files
- More than 50 lines changed
- Multiple concepts

Then decompose into smaller approved sub-tasks.

---

## Zero Technical Debt

### Never Add

- "Temporary" workarounds
- TODO comments for later
- FIXME comments
- Hardcoded values "for now"
- Commented-out code
- Unused imports or functions

### Never Skip

- Edge cases
- Error handling
- Input validation
- Type hints
- Docstrings for public functions

### Never Add Unnecessary

- Abstractions "for flexibility"
- Dependencies "might need later"
- Configuration options "just in case"
- Classes where functions suffice
- Complexity to solve problems we don't have

---

## Error Handling

### When You Introduce an Error

1. STOP immediately
2. Do NOT attempt to fix
3. Report what happened
4. Propose fix
5. Wait for approval
6. Only then fix

### When You Discover an Error

1. Report it
2. Do NOT fix it in the current task
3. Propose it as a separate task
4. Wait for prioritization
5. Complete current task first (if possible)

---

## Cognitive Discipline

### Slow Down
- Don't make changes unless 100% certain
- Research before implementing
- Ask before assuming

### Think Step by Step
- State your understanding
- State your plan
- Identify assumptions
- Identify risks

### Stay Focused
- One task at a time
- No "while I'm here" changes
- No scope creep
- No refactoring without approval

---

## Communication Standards

### When Proposing Changes

```markdown
## Proposed Change
[What you want to do]

## Rationale
[Why this change is needed]

## Tests Affected
- [Existing tests that need modification]
- [New tests to add]

## Files Affected
- [List of files]

## Risks
- [What could go wrong]

## Request
Approve? (yes/no)
```

### When Reporting Status

```markdown
## Completed
- [What was done]

## Test Results
- `make test`: [pass/fail]
- [Specific test outcomes if relevant]

## Next Step
[What's next, pending approval]
```
