---
name: planning_session
description: Force explicit planning before any implementation. Use before starting any coding task, feature, fix, or refactor. Produces structured plan covering objective, tests, risks, and requires approval before proceeding. Prevents undisciplined coding.
---

# Planning Session Skill

Structured planning before any implementation work.

## When to Use

- Before ANY coding task
- Before any feature implementation
- Before any bug fix
- Before any refactoring
- User says "plan", "let's plan", "planning session"
- Whenever tempted to "just start coding"

## Why This Matters

Without explicit planning:
- Tests get written after code (or skipped)
- Scope creeps during implementation
- Edge cases get missed
- Changes cascade unexpectedly

## Execution Steps

1. **Define Objective**
   
   Ask and answer:
   - What exactly are we trying to accomplish?
   - What will be different when we're done?
   - What is NOT in scope?

2. **Define Success Criteria**
   
   Specific, testable conditions:
   - How do we know we're done?
   - What tests will verify success?
   - What behavior confirms correctness?

3. **Surface Assumptions**
   
   What are we taking for granted?
   - About the existing code?
   - About data formats?
   - About dependencies?
   - About user requirements?

4. **Identify Risks**
   
   What could go wrong?
   - What might break?
   - What edge cases exist?
   - What dependencies might cause issues?
   - What if assumptions are wrong?

5. **Define Test Plan**
   
   Before writing any code:
   - What existing tests need modification?
   - What new tests are needed?
   - What assertions will each test make?
   - What edge cases need test coverage?

6. **Estimate Scope**
   
   - How many files affected?
   - Approximate lines of change?
   - Should this be decomposed?

7. **Present for Approval**
   
   Format plan and request explicit approval.

## Output Format

```markdown
## Planning Session: [Task Name]

### Objective
[Clear statement of what we're trying to accomplish]

**In Scope:**
- [item]
- [item]

**Out of Scope:**
- [item]

### Success Criteria
- [ ] [Testable criterion]
- [ ] [Testable criterion]

### Assumptions
1. [Assumption]
2. [Assumption]

### Risks
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| [risk] | [L/M/H] | [mitigation] |

### Test Plan

**Existing Tests to Modify:**
- `test_x.py::test_function`: [what changes]

**New Tests to Add:**
- `test_x.py::test_new_case`: [what it tests]
  - Assert: [specific assertion]

**Edge Cases:**
- [edge case]: tested by [test name]

### Files Affected
- `path/file.py`: [nature of changes]

### Scope Estimate
- Files: [N]
- Lines: ~[N]
- Complexity: [Low/Medium/High]

---
**Approval Required**

Proceed with this plan? (yes/no/modify)
```

## Decomposition Trigger

If during planning you find:
- More than 3 files affected
- More than 50 lines changed
- Multiple distinct concepts
- Success criteria that can't fit in one sentence

Then STOP and propose decomposition into smaller tasks.

## Critical Notes

- NEVER skip planning for "simple" tasks
- Test plan comes BEFORE implementation plan
- Wait for explicit approval
- Document scope boundaries to prevent creep
- If you can't define success criteria, you're not ready to code
