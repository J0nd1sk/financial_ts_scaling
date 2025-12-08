---
name: session_handoff
description: Capture complete session state for continuity. Use when ending a session, context window approaching limit (~80%), switching task areas, or user requests handoff. Creates structured context dump for next session restoration.
---

# Session Handoff Skill

Capture current session state for seamless continuation.

## When to Use

- End of session
- Context window ~80% full
- User says "handoff", "session handoff", "save context"
- Before switching to significantly different task area
- Before any extended break

## Execution Steps

1. **Gather Git State**
   ```bash
   git branch --show-current
   git log -1 --oneline
   git status --short
   ```

2. **Check Test Status**
   - When was `make test` last run?
   - Did it pass or fail?
   - Any failing tests to note?

3. **Compile Session Summary**
   - What task was in progress?
   - What was completed this session?
   - What remains pending?
   - What files were modified?

4. **Capture Decisions**
   - Any architectural decisions made?
   - Any approaches rejected and why?

5. **Write Context File**
   
   Create/update `.claude/context/session_context.md`:

   ```markdown
   # Session Handoff - [YYYY-MM-DD HH:MM]

   ## Current State

   ### Branch & Git
   - Branch: [branch name]
   - Last commit: [hash] [message]
   - Uncommitted: [files or "none"]

   ### Task Status
   - Working on: [task]
   - Status: [in progress / blocked / complete]

   ## Test Status
   - Last `make test`: [pass/fail at time]
   - Failing: [tests or "none"]

   ## Completed This Session
   1. [item]

   ## In Progress
   - [task]: [what remains]

   ## Pending
   1. [next task]

   ## Files Modified
   - `path/file.py`: [changes]

   ## Key Decisions
   - [decision]: [rationale]

   ## Context for Next Session
   [Important info that would be lost]

   ## Next Session Should
   1. [priority 1]
   2. [priority 2]

   ## Commands to Run
   ```bash
   source venv/bin/activate
   make test
   git status
   ```
   ```

6. **Update Phase Tracker** (if progress made)
   
   Update `.claude/context/phase_tracker.md` with completion status.

7. **Report to User**
   - Confirm file written
   - Summarize key state
   - Warn of any uncommitted work

## Output Format

After writing files, report:

```
✅ Session handoff complete

📍 State saved to: .claude/context/session_context.md
📊 Phase tracker: [updated/unchanged]

Summary:
- Branch: [branch]
- Task: [current task] ([status])
- Tests: [pass/fail]
- Uncommitted: [count] files

⚠️ [Any warnings about uncommitted work]
```

## Critical Notes

- Always capture uncommitted work prominently
- Include rejected approaches (valuable context)
- Note any user preferences expressed during session
- Be specific about "what remains" for in-progress tasks
