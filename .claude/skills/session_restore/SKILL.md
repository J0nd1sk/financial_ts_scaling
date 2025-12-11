---
name: session_restore
description: Restore context from previous session and verify environment. Use at the start of any new session, when user says "restore", "where were we", or "continue". Reads saved context, verifies environment, and confirms priorities before proceeding.
---

# Session Restore Skill

Restore context and verify readiness for continued work.

## When to Use

- Start of new session
- User says "restore", "session restore", "where were we", "continue"
- After any interruption or context loss
- When uncertain about current state

## Execution Steps

1. **Read Context Files**

   ```bash
   cat .claude/context/session_context.md
   cat .claude/context/phase_tracker.md
   ```

   If files don't exist, inform user and start fresh.

2. **Query Memory MCP for Relevant Knowledge** (additive - context files remain primary)

   After reading context files, retrieve Memory entities listed in the "Memory Entities Updated" section:

   ```
   # Extract entity names from session_context.md "Memory Entities Updated" section
   # Then retrieve them directly:
   mcp__memory__open_nodes({
     "names": ["Entity1", "Entity2", ...]  # exact names from context file
   })
   ```

   If no entities listed or section missing, optionally search by phase:
   ```
   mcp__memory__search_nodes({
     "query": "Phase[N]"  # e.g., "Phase5" - use underscores, match naming convention
   })
   ```

   Look for:
   - Lessons from same phase or similar tasks
   - Patterns that apply to pending work
   - Decisions that might affect current priorities
   - Anti-patterns to avoid

   Include relevant findings in summary to user. Examples:
   - "📚 Memory: Retrieved 3 entities from last session..."
   - "⚠️ Reminder: [lesson from entity observations]"
   - "✅ Pattern to apply: [successful approach from entity]"

   **Note**: This supplements (not replaces) context files. Memory provides agent-queryable knowledge; context files remain authoritative.

3. **Verify Environment**
   
   ```bash
   source venv/bin/activate
   make test
   git status
   make verify
   git branch --show-current
   ```

3. **Check for Drift**
   
   Compare current git state to saved state:
   - Same branch?
   - Any new commits since handoff?
   - Any uncommitted changes?

4. **Summarize to User**
   
   Present clear summary:
   
   ```
   ## Session Restored
   
   ### Previous Session: [date/time]
   
   **Last Working On:** [task]
   **Status:** [in progress / blocked / complete]
   
   **Git State:**
   - Branch: [branch]
   - [matches saved / drifted: describe]
   
   **Test Status:**
   - Saved: [pass/fail]
   - Current: [pass/fail from make test]
   
   **Data Manifests:**
   - Raw: [latest entry summary or "none"]
   - Processed: [latest entry summary or "none"]
   
   ### Pending Work
   1. [item from saved context]
   2. [item]
   
   ### Recommended Next Steps
   1. [from saved "Next Session Should"]
   2. [item]
   ```

5. **Confirm Priorities**
   
   Ask user:
   ```
   Priorities from last session were:
   1. [priority]
   2. [priority]
   
   Continue with these, or redirect?
   ```

6. **Wait for Confirmation**
   
   Do NOT proceed until user confirms direction.

## Output Format

```
🔄 Session Restore

📂 Context loaded from: .claude/context/session_context.md
📅 Last session: [date/time]

## Previous State
- Task: [task name]
- Status: [status]
- Branch: [branch] ✅ matches / ⚠️ changed to [new]

## Environment Check
- venv: ✅ activated
- make test: ✅ pass / ❌ [N] failing
- git status: ✅ clean / ⚠️ [N] uncommitted files
- make verify: ✅ pass / ❌ errors

## Data Manifests
- Raw: [latest dataset/file/md5 or "no entries"]
- Processed: [latest dataset/version/tier or "none"]

## Memory Retrieved
- [EntityName]: [key lesson/pattern/decision from observations]
- (or "No Memory entities listed in context file")

## Pending from Last Session
1. [item]
2. [item]

## Recommended Next
1. [priority]
2. [priority]

---
Continue with these priorities? Or redirect?
```

## If Context Files Missing

```
⚠️ No saved session context found

Files checked:
- .claude/context/session_context.md: not found
- .claude/context/phase_tracker.md: not found

Starting fresh. Current state:
- Branch: [branch]
- Tests: [make test result]
- Git: [status]

What would you like to work on?
```

## Critical Notes

- Always run `make test` during restore
- Always confirm priorities before proceeding
- Note any state drift clearly
- Do not assume saved priorities are still valid
