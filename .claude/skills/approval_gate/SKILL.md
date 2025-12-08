---
name: approval_gate
description: Enforce the propose-wait-execute pattern for all changes. Use before ANY code change, file operation, git operation, or fix. Ensures explicit user approval before proceeding. Prevents unauthorized modifications and scope creep.
---

# Approval Gate Skill

Enforce explicit approval before any action.

## When to Use

- Before ANY code change (even one line)
- Before ANY file creation or deletion
- Before ANY git operation
- Before fixing errors (even self-introduced)
- Before installing dependencies
- Before modifying configuration
- When uncertain if action is approved

## The Pattern

```
PROPOSE → WAIT → EXECUTE

Never: EXECUTE → INFORM
Never: ASK → ASSUME SILENCE MEANS YES
```

## Execution Steps

### Step 1: Formulate Proposal

Structure your proposal:

```markdown
## Proposed: [Action Type]

**What:** [Specific action in one sentence]

**Why:** [Rationale]

**Tests Affected:**
- [Existing tests to modify]
- [New tests to add]

**Files Affected:**
- `path/file.py`: [nature of change]

**Risks:**
- [What could go wrong]

**Reversible:** [Yes/No - how to undo if needed]

---
Approve? (yes / no / modify)
```

### Step 2: Present and Wait

- Present the proposal clearly
- Do NOT proceed
- Do NOT assume
- Do NOT interpret silence as approval
- Wait for explicit response

### Step 3: Interpret Response

| Response | Action |
|----------|--------|
| "yes", "approved", "go ahead", "do it" | Proceed with exactly what was proposed |
| "no", "stop", "don't" | Abort, ask what to do instead |
| "modify", "change", "but..." | Revise proposal, re-submit |
| Silence / No response | Wait longer, then ask "Still waiting for approval" |
| Unclear | Ask for clarification |

### Step 4: Execute Exactly

If approved:
- Do exactly what was proposed
- Nothing more
- Nothing less
- No "while I'm here" additions
- No "obvious" improvements

### Step 5: Report Completion

After execution:

```markdown
## Completed

**Action:** [What was done]

**Result:** [Outcome]

**Test Status:** `make test` [pass/fail]

**Next:** [What's next, pending approval]
```

## Action Categories Requiring Approval

### Code Changes
- Any modification to source files
- Any modification to test files
- Adding/removing imports
- Changing function signatures
- Fixing bugs (even ones you introduced)

### File Operations
- Creating new files
- Deleting files
- Renaming files
- Moving files
- Creating directories

### Git Operations
- `git add` (any form)
- `git commit`
- `git push`
- `git branch`
- `git checkout`
- `git merge`
- `git rebase`

### Dependencies
- `pip install`
- Adding to requirements.txt
- Upgrading versions

### Configuration
- Any change to pyproject.toml
- Any change to Makefile
- Any change to .gitignore
- Any YAML config changes

## Anti-Patterns to Avoid

❌ "I'll just fix this quick thing"
❌ "This is obviously needed"
❌ "I'm sure you meant for me to..."
❌ "While I'm here, I'll also..."
❌ "This is too small to need approval"
❌ Proceeding after asking but before receiving response

## Escalation

If you're uncertain whether something needs approval:
- It needs approval
- When in doubt, ask

If user seems frustrated by approval requests:
- Acknowledge but maintain discipline
- "I know this seems slow, but approval gates prevent errors. Approve [X]?"

## Output Format

### Proposal

```
🔒 APPROVAL REQUIRED

## Proposed: [Action]

**What:** [description]
**Why:** [rationale]
**Files:** [list]
**Risks:** [list]

Approve? (yes/no/modify)
```

### After Approval

```
✅ APPROVED - Executing...

[action output]

✅ COMPLETE
- Action: [what was done]
- Tests: [make test result]
- Next: [pending approval]
```
