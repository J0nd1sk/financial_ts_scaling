---
name: session_restore
description: Restore context from previous session and verify environment. Use at the start of any new session, when user says "restore", "where were we", or "continue". Reads saved context, verifies environment, and confirms priorities before proceeding. Supports multiple parallel workstreams.
---

# Session Restore Skill (Multi-Workstream)

Restore context and verify readiness for continued work across parallel workstreams.

## When to Use

- Start of new session
- User says "restore", "session restore", "where were we", "continue"
- After any interruption or context loss
- When uncertain about current state

## Workstream System

This project supports up to 3 parallel workstreams (terminals):

| ID | Current Name | Typical Focus |
|----|--------------|---------------|
| ws1 | tier_a100 | Feature implementation |
| ws2 | foundation | Foundation model investigation |
| ws3 | (available) | Phase 6C experiments, etc. |

**Files:**
- `.claude/context/global_context.md` - Summary of all workstreams
- `.claude/context/workstreams/ws{N}_context.md` - Detailed per-workstream context

## Execution Steps

### 1. Read Global Context

Read `.claude/context/global_context.md` to understand:
- Active workstreams and their status
- Shared git/test state
- Cross-workstream coordination notes

### 2. Show Active Workstreams Summary

```
 Active Workstreams:

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | tier_a100 | active | 2026-01-24 11:00 | Chunk 3 complete, Chunk 4 next |
| ws2 | foundation | paused | 2026-01-24 09:00 | TimesFM Colab ready |
```

### 3. Auto-Detect Workstream

Attempt to detect which workstream this session is for:

**Detection heuristics (in order):**
1. **User's first message/task**: Look for keywords
   - "tier_a100", "indicators", "chunk" → ws1
   - "foundation", "lag-llama", "timesfm" → ws2
   - "phase6c", "experiments", "scaling" → ws3
2. **File mentions**:
   - `tier_a100.py` → ws1
   - `experiments/foundation/*` → ws2
3. **Recent activity**: If only one workstream active in last 24h, default to it
4. **Fallback**: Ask user

### 4. Confirm Workstream Selection

```
 Detected workstream: ws1 (tier_a100)

Based on: [detection reason]

Correct? [y/n/other]
```

If user says "n" or specifies another, switch to that workstream.

### 5. Read Workstream Context

Read `.claude/context/workstreams/ws{N}_context.md` for detailed context:
- Current task and status
- Progress summary
- Key decisions
- Next session priorities

### 6. Verify Environment

```bash
source venv/bin/activate
make test
git status
make verify
git branch --show-current
```

### 7. Check for Drift

Compare current git state to saved state:
- Same branch?
- Any new commits since handoff?
- Any uncommitted changes?

### 8. Query Memory MCP (if entities listed)

If workstream context lists Memory entities:
```
mcp__memory__open_nodes({
  "names": ["Entity1", "Entity2", ...]
})
```

Include relevant findings in summary.

### 9. Show Cross-Workstream Coordination

From global context, highlight:
- Blocking dependencies that affect this workstream
- Shared resources with other workstreams
- Files owned by multiple workstreams

### 10. Summarize to User

```
 Session Restored - ws[N]: [name]

## Previous Session: [date/time]

**Current Task:** [task]
**Status:** [in progress / blocked / complete]

**Git State:**
- Branch: [branch]
- [matches saved / drifted: describe]

**Test Status:**
- Saved: [pass/fail]
- Current: [pass/fail from make test]

**Data Manifests:**
- Raw: [latest entry summary]
- Processed: [latest entry summary]

## Progress Summary
- Completed: [N items]
- Pending: [N items]

## Cross-Workstream Notes
- [ws2] foundation: [status - independent/blocking/etc.]
- Shared files: [any coordination notes]

## Recommended Next Steps (from saved context)
1. [priority 1]
2. [priority 2]

---
Continue with these priorities? Or redirect?
```

### 11. Confirm Priorities

Wait for user confirmation before proceeding.

**Do NOT proceed until user confirms direction.**

## Output Format

```
 Session Restore - ws[N]: [name]

 Context loaded:
  - Global: .claude/context/global_context.md
  - Workstream: .claude/context/workstreams/ws[N]_context.md
 Last session: [date/time]

## Workstream State
- ID: ws[N]
- Name: [name]
- Task: [task name]
- Status: [status]
- Branch: [branch]  matches /  changed

## Environment Check
- venv:  activated
- make test:  pass /  [N] failing
- git status:  clean /  [N] uncommitted files
- make verify:  pass /  errors

## Other Active Workstreams
| ID | Name | Status | Relationship |
|----|------|--------|--------------|
| ws[X] | [name] | [status] | [independent/blocks this/blocked by] |

## Pending from Last Session
1. [item]
2. [item]

## Recommended Next
1. [priority]
2. [priority]

---
Continue with these priorities? Or redirect?
```

## If Global Context Missing

```
 No global context found

Checked: .claude/context/global_context.md

 Falling back to legacy context...
```

Then try `.claude/context/session_context.md` (legacy format).

## If All Context Files Missing

```
 No saved session context found

Files checked:
- .claude/context/global_context.md: not found
- .claude/context/workstreams/: empty or not found
- .claude/context/session_context.md: not found (legacy)

Starting fresh. Current state:
- Branch: [branch]
- Tests: [make test result]
- Git: [status]

What would you like to work on?
```

## Workstream Selection Quick Reference

| User Says | Workstream |
|-----------|------------|
| "continue tier_a100" | ws1 |
| "work on indicators" | ws1 |
| "foundation models" | ws2 |
| "timesfm experiments" | ws2 |
| "phase 6c" | ws3 |
| (no specific mention) | Ask or use most recently active |

## Critical Notes

- Always run `make test` during restore
- Always confirm priorities before proceeding
- Note any state drift clearly
- Do not assume saved priorities are still valid
- Show cross-workstream coordination notes - important for parallel work
- If user's task doesn't match detected workstream, switch immediately
