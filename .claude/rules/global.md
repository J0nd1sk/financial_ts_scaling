# Global Rules

## Rule Files Index

See these files for specific rules:
- `experimental-protocol.md` - Scaling law constraints, parameter budgets
- `testing.md` - TDD enforcement, make test requirements
- `development-discipline.md` - Approval gates, RACI matrix
- `context-handoff.md` - Session continuity protocols

---

## Workflow Invocation

These workflows can be triggered via Superpowers skills or manual commands:

| Workflow | Skill | Manual Trigger |
|----------|-------|----------------|
| Session Handoff | `/skill session_handoff` | "handoff", "save context" |
| Session Restore | `/skill session_restore` | "restore", "where were we" |
| Planning Session | `/skill planning_session` | "plan", "let's plan" |
| Test First | `/skill test_first` | "test first", "TDD" |
| Approval Gate | `/skill approval_gate` | (automatic before changes) |
| Thermal Check | `/skill thermal_management` | "thermal check", "temperature" |
| Task Breakdown | `/skill task_breakdown` | "break down", "decompose" |

---

## Git Workflow

### Always Use `git add -A`
```bash
# ✅ CORRECT
git add -A

# ❌ FORBIDDEN
git add .
git add specific_file.py
git add -p
```

### Test Before Stage
Run `make test` BEFORE any `git add` command, not after.

### Commit Messages
```
[type]: Brief description (50 chars max)

Detailed explanation if needed:
- What changed
- Why it changed

Related: phase-N-task-M
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `data`, `exp`

---

## File Preservation

### Never Delete Without Approval
- pyproject.toml
- requirements.txt
- Makefile
- Any configuration file
- Any test file
- Any data file

### Never Mass Delete
No `git clean`, `rm -rf`, or bulk deletions without explicit approval.

---

## Filesystem Rules

Before creating or deleting any file or directory:
1. State what you intend to create/delete
2. State why
3. Wait for explicit approval

---

## Dependency Rules

Before adding any dependency:
1. State what package and version
2. State why it's needed
3. State alternatives considered
4. Wait for explicit approval

---

## Configuration Rules

Never modify without approval:
- pyproject.toml
- requirements.txt
- Makefile
- .gitignore
- pytest.ini / pyproject.toml [tool.pytest]
- Any YAML config
