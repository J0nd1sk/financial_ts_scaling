---
name: session_handoff
description: Capture complete session state for continuity. Use when ending a session, context window approaching limit (~80%), switching task areas, or user requests handoff. Creates structured context dump for next session restoration. Supports multiple parallel workstreams.
---

# Session Handoff Skill (Multi-Workstream)

Capture current session state for seamless continuation across parallel workstreams.

## When to Use

- End of session
- Context window ~80% full
- User says "handoff", "session handoff", "save context"
- Before switching to significantly different task area
- Before any extended break

## Workstream System

This project supports up to 3 parallel workstreams (terminals), each with its own context:

| ID | Current Name | Typical Focus |
|----|--------------|---------------|
| ws1 | tier_a100 | Feature implementation |
| ws2 | foundation | Foundation model investigation |
| ws3 | (available) | Phase 6C experiments, etc. |

**Files:**
- `.claude/context/global_context.md` - Summary of all workstreams
- `.claude/context/workstreams/ws{N}_context.md` - Detailed per-workstream context

## Execution Steps

### 1. Determine Workstream

If not obvious from context, ask:
```
Which workstream is this session for?
1. ws1 (tier_a100) - Feature implementation
2. ws2 (foundation) - Foundation model investigation
3. ws3 (new/other) - [specify name]
```

**Auto-detection heuristics:**
- Keywords: "tier_a100" → ws1, "foundation"/"lag-llama"/"timesfm" → ws2, "phase6c" → ws3
- Files modified: `tier_a100.py` → ws1, `experiments/foundation/*` → ws2
- If uncertain, ask user

### 2. Gather Git State

```bash
git branch --show-current
git log -1 --oneline
git status --short
```

### 3. Check Test Status

- When was `make test` last run?
- Did it pass or fail?
- Any failing tests to note?

### 4. Compile Session Summary

- What task was in progress?
- What was completed this session?
- What remains pending?
- What files were modified?

### 5. Capture Data Version Status

- Latest raw manifest entry (dataset, file, md5, timestamp)
- Latest processed manifest entry (dataset, version, tier, md5)
- Any datasets waiting to be registered

### 6. Capture Decisions

- Any workstream-specific decisions made?
- Any approaches rejected and why?

### 7. Update Workstream Context File

Create/update `.claude/context/workstreams/ws{N}_context.md`:

```markdown
# Workstream [N] Context: [Name]
# Last Updated: [YYYY-MM-DD HH:MM]

## Identity
- **ID**: ws[N]
- **Name**: [descriptive name]
- **Focus**: [brief focus description]
- **Status**: active

---

## Current Task
- **Working on**: [task]
- **Status**: [in progress / blocked / complete]

---

## Progress Summary

### Completed
[List completed items with dates]

### Pending
[List pending items]

---

## Last Session Work ([date])
[Detailed work from this session]

---

## Files Owned/Modified
- `path/to/file.py` - PRIMARY/SHARED
  - [nature of changes]

---

## Key Decisions (Workstream-Specific)
- [Decision]: [Rationale]

---

## Session History
### [date]
- [work done]

---

## Next Session Should
1. [priority 1]
2. [priority 2]

---

## Memory Entities (Workstream-Specific)
- [EntityName]: [brief description]
```

### 8. Update Global Context

Update `.claude/context/global_context.md`:

1. **Update workstream row** in Active Workstreams table:
   - Update Status (active/paused/inactive)
   - Update Last Update timestamp
   - Update Summary (brief current state)

2. **Update Shared State** if changed:
   - Branch & Git
   - Test Status
   - Data Versions

3. **Update Cross-Workstream Coordination** if relevant:
   - Blocking dependencies
   - Shared resources
   - File ownership changes

4. **Prune stale workstreams**:
   - Workstreams inactive >7 days: Remove from table (keep file)
   - Workstreams inactive >3 days: Mark as "paused"

### 9. Verify User Preferences Section

Confirm `global_context.md` contains complete "User Preferences (Authoritative)" section with all subsections:
- Development Approach (TDD, planning, tmux)
- Context Durability (Memory MCP, context files, docs/)
- Documentation Philosophy (consolidation, precision, flat structure)
- Communication Standards (precision, no summarizing away details)
- Hyperparameters (Fixed - Ablation-Validated)

**NEVER reduce fidelity or summarize preferences.**

### 10. Update Phase Tracker (if progress made)

Update `.claude/context/phase_tracker.md` with completion status.

### 11. Store Lessons in Memory MCP (optional)

For significant learnings, create/update Memory entities:
- Include workstream tag in entity name: `ws1_[Topic]_[Type]`
- Example: `ws1_DeMarker_Implementation_Decision`

Record entity names in workstream context file.

### 12. Report to User

```
 Session handoff complete (ws[N]: [name])

 State saved to:
  - .claude/context/workstreams/ws[N]_context.md
  - .claude/context/global_context.md

 Summary:
- Workstream: ws[N] ([name])
- Task: [current task] ([status])
- Tests: [pass/fail]
- Uncommitted: [count] files

 [Any warnings about uncommitted work]

 Other active workstreams:
- ws[X] ([name]): [brief status]
```

## Output Format

After writing files, report:

```
 Session handoff complete (ws[N]: [name])

 Workstream context: .claude/context/workstreams/ws[N]_context.md
 Global context: .claude/context/global_context.md
 Phase tracker: [updated/unchanged]

Summary:
- Workstream: ws[N] ([name]) - [status]
- Branch: [branch]
- Tests: [pass/fail]
- Uncommitted: [count] files

Cross-workstream notes:
- [Any blocking dependencies or shared resource notes]

 [Any warnings]
```

## Workstream Lifecycle

| Status | Criteria | Action |
|--------|----------|--------|
| active | Updated within 24h | Show in global, detailed context |
| paused | No update 3-7 days | Mark paused in global, keep file |
| inactive | No update >7 days | Remove from global table, keep file |
| archived | User archives explicitly | Move to workstreams/archive/ |

## Critical Notes

- Always capture uncommitted work prominently
- Include rejected approaches (valuable context)
- Note any user preferences expressed during session
- Be specific about "what remains" for in-progress tasks
- Update BOTH workstream file AND global summary
- Cross-workstream coordination notes are important for parallel work
