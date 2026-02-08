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

0. **Query Memory for Relevant Knowledge** (Memory MCP integration - additive)

   Before planning, search Memory for relevant lessons and patterns:

   ```
   mcp__memory__search_nodes({
     "query": "[task domain keywords: e.g., 'data pipeline', 'testing', 'feature engineering']"
   })
   ```

   Look for:
   - Lessons from similar tasks or same phase
   - Anti-patterns to avoid
   - Successful patterns to apply
   - Relevant decisions that constrain the design

   Include findings in planning considerations below.

   **Note**: Memory supplements planning, doesn't replace it. Use findings to inform risks, assumptions, and test plans.

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

7. **Define Subagent Execution Strategy** (for multi-step plans)

   Determine how implementation work will be delegated to preserve director context:

   **A. Delegation Mode Assessment:**

   | Criteria | Mode | When to Use |
   |----------|------|-------------|
   | <3 files, <50 lines, single concept | Single-agent | Director implements directly |
   | 3-10 files, independent tasks | Subagent-per-task | Spawn subagents for each task |
   | >10 files, >3 domains, complex | Wave orchestration | Multi-phase with specialized agents |

   **B. Task Delegation Table:**

   For each implementation task from the plan:

   | Task | Subagent Type | Run In | Dependencies | Director Checkpoint |
   |------|---------------|--------|--------------|---------------------|
   | [name] | general-purpose | foreground | None | Review before next |
   | [name] | Explore | background | None | Aggregate results |
   | [name] | general-purpose | foreground | After task 1 | Approve implementation |

   Subagent types: `general-purpose` (implementation), `Explore` (research), `Plan` (design)

   **C. Director vs Subagent Responsibilities:**

   Director (main context) retains:
   - Approval gates between major phases
   - Cross-task coordination decisions
   - Final verification and commit
   - Memory entity creation/updates
   - User communication

   Subagents handle:
   - File exploration and reading
   - Code writing and editing
   - Test implementation
   - Individual task verification
   - Detailed error investigation

   **D. Context Preservation:**

   Before spawning subagents:
   - [ ] Store task context in Memory MCP (if complex)
   - [ ] Update workstream context file with delegation plan
   - [ ] Define clear success criteria for each subagent task

   After subagent completion:
   - [ ] Review subagent output before proceeding
   - [ ] Update context files with results
   - [ ] Run verification commands (make test, etc.)

8. **Present for Approval**

   Format plan and request explicit approval.

9. **Store Finalized Plan in Memory** (after user approval)

   Once user approves the plan, store it in Memory MCP:

   ```
   mcp__memory__create_entities({
     "entities": [{
       "name": "[Task Name] Plan",
       "entityType": "planning_decision",
       "observations": [
         "Plan (YYYY-MM-DD, Phase N): [Objective summary]",
         "Scope: [In scope items]",
         "Test strategy: [Test plan summary]",
         "Risks identified: [Key risks and mitigations]",
         "Files affected: [N files, ~N lines, complexity level]"
       ]
     }]
   })
   ```

   This enables future sessions to learn from planning outcomes and track what worked.

## Output Format

```markdown
## Planning Session: [Task Name]

### Memory Findings (if relevant)
📚 **Relevant knowledge from previous sessions:**
- [Lesson/pattern from Memory with context]
- [Anti-pattern to avoid]
- [Decision that constrains design]

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

### Subagent Execution Strategy

**Delegation Mode:** [Single-agent | Subagent-per-task | Wave orchestration]

**Rationale:** [Why this mode - file count, complexity, independence of tasks]

**Task Delegation:**
| # | Task | Subagent | Background? | Depends On | Director Action After |
|---|------|----------|-------------|------------|----------------------|
| 1 | [task name] | general-purpose | No | - | Review output |
| 2 | [task name] | Explore | Yes | - | Read results file |
| 3 | [task name] | general-purpose | No | #1, #2 | Approve, then continue |

**Director Retains:**
- [What stays in main context - approvals, coordination, commits]

**Subagents Handle:**
- [What gets delegated - file reads, implementation, tests]

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
