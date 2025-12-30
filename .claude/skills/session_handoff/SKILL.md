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

4. **Capture Data Version Status**
   - Latest raw manifest entry (dataset, file, md5, timestamp)
   - Latest processed manifest entry (dataset, version, tier, md5)
   - Any datasets waiting to be registered or checksums to recompute

5. **Capture Decisions**
   - Any architectural decisions made?
   - Any approaches rejected and why?

6. **Write Context File**
   
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

   ## Data Versions
   - Raw manifest: [latest dataset/file/md5 or "no entries"]
   - Processed manifest: [latest dataset/version/tier or "none"]
   - Pending registrations: [list or "none"]

   ## Memory Entities Updated
   List all Memory MCP entities created or updated this session:
   - [EntityName1] (created|updated): [brief description]
   - [EntityName2] (created|updated): [brief description]

   If none: "No Memory entities updated this session"

   ## Commands to Run
   ```bash
   source venv/bin/activate
   make test
   git status
   make verify
   ```

   ## User Preferences (Authoritative)
   [MUST include complete section - copy from previous session or reconstruct from User_Preferences_Authoritative Memory entity]
   ```

7. **Verify User Preferences Section**

   Confirm the "User Preferences (Authoritative)" section is complete with all subsections:
   - Development Approach (TDD, planning, tmux)
   - Context Durability (Memory MCP, context files, docs/)
   - Documentation Philosophy (consolidation, precision, flat structure)
   - Communication Standards (precision, no summarizing away details)

   If previous session_context.md has this section, copy it verbatim.
   If missing, query `User_Preferences_Authoritative` from Memory MCP to reconstruct.

   **NEVER reduce fidelity or summarize preferences.**

8. **Update Phase Tracker** (if progress made)

   Update `.claude/context/phase_tracker.md` with completion status.

9. **Store Lessons in Memory MCP** (additive - context files remain primary)

   Extract and store key learnings from this session. **Track all entity names for session_context.md.**

   - **For NEW entities** (first time storing this knowledge):
     ```
     mcp__memory__create_entities({
       "entities": [{
         "name": "Phase[N]_[TaskName]_[Type]",  # e.g., "Phase5_VIX_Integration_Lesson"
         "entityType": "lesson|pattern|decision",
         "observations": [
           "Lesson: [specific lesson]",
           "Phase: [current phase]",
           "Context: [when/why this applies]",
           "Session: [YYYY-MM-DD]"
         ]
       }]
     })
     ```

   - **For EXISTING entities** (adding to prior knowledge):
     ```
     mcp__memory__add_observations({
       "observations": [{
         "entityName": "[existing entity name]",
         "contents": [
           "Pattern: [new pattern discovered]",
           "Session: [YYYY-MM-DD]"
         ]
       }]
     })
     ```

   **Entity naming convention**: `Phase[N]_[Topic]_[Type]` using underscores, no spaces.
   Examples:
   - `Phase5_VIX_Volume_Handling_Decision`
   - `Phase4_TDD_Pattern`
   - `Mock_yfinance_Pattern`

   **CRITICAL**: After storing, record all entity names in session_context.md (see step 6 template).

   **Note**: This supplements (not replaces) decision_log.md. Memory MCP enables agent-queryable knowledge; decision_log.md remains the human-readable, version-controlled record.

10. **Report to User**
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
- Data manifests: [latest raw + processed summaries]
- Uncommitted: [count] files

⚠️ [Any warnings about uncommitted work]
```

## Critical Notes

- Always capture uncommitted work prominently
- Include rejected approaches (valuable context)
- Note any user preferences expressed during session
- Be specific about "what remains" for in-progress tasks
