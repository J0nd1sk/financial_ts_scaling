# Workflow Reminders

## 🔴 CRITICAL: Plan Before Code

### The Violation I Just Made
I jumped straight to proposing Phase 2 implementation (data directories, download scripts, etc.) **without planning first**.

### The Correct Workflow

```
1. PLAN (using planning_session skill)
   ↓
2. GET APPROVAL on the plan
   ↓
3. IMPLEMENT (only after approval)
```

### When Planning is Required
- **ANY coding task** - no exceptions for "simple" tasks
- Before creating **any files**
- Before writing **any tests**
- Before writing **any implementation**
- Before **any refactoring**
- When user says "let's start Phase X"

### Planning Session Must Include
1. **Objective** - What exactly are we accomplishing?
2. **Success Criteria** - Testable conditions for "done"
3. **Assumptions** - What are we taking for granted?
4. **Risks** - What could go wrong?
5. **Test Plan** - Tests BEFORE implementation plan
6. **Files Affected** - Scope estimation
7. **Approval Request** - Wait for explicit yes/no/modify

### RACI Matrix Reference
| Activity | Human | Agent |
|----------|-------|-------|
| Task breakdown | **Approve** | **Propose** |
| Architecture decisions | **Lead** | **Propose** |
| Writing tests | **Approve** | Execute |
| Writing implementation | **Approve** | Execute |

**Key:** Agent PROPOSES plans, Human APPROVES them. Never implement without approval.

### Decomposition Triggers
If planning reveals:
- More than 3 files affected
- More than 50 lines changed
- Multiple distinct concepts
- Success criteria can't fit in one sentence

Then **STOP** and decompose into smaller subtasks.

---

## Manual Trigger Commands

| Workflow | Trigger |
|----------|---------|
| Planning Session | "plan", "let's plan", `/skill planning_session` |
| Session Handoff | "handoff", "save context", `/skill session_handoff` |
| Session Restore | "restore", "where were we", `/skill session_restore` |
| Test First | "test first", "TDD", `/skill test_first` |
| Task Breakdown | "break down", "decompose", `/skill task_breakdown` |
| Thermal Check | "thermal check", "temperature", `/skill thermal_management` |

---

## Remember
- **Slow down** - Don't make changes unless 100% certain
- **Think step by step** - State understanding, plan, assumptions, risks
- **Stay focused** - One task at a time, no scope creep
- **Test before stage** - Run `make test` BEFORE `git add -A`
