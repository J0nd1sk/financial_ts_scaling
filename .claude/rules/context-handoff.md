# Context Handoff Rules

## Purpose

Sessions end. Context windows fill. Continuity must be maintained.

This project supports **multiple parallel workstreams** (up to 3 terminals working simultaneously) with independent context management.

---

## Workstream System

### Directory Structure

```
.claude/context/
├── global_context.md           # Summary of all workstreams (minimal, pruned)
├── phase_tracker.md            # Global phase progress (unchanged)
├── decision_log.md             # Global decisions (unchanged)
└── workstreams/
    ├── ws1_context.md          # Detailed context for workstream 1
    ├── ws2_context.md          # Detailed context for workstream 2
    └── ws3_context.md          # Detailed context for workstream 3
```

### Workstream Naming

Each workstream gets a short identifier (ws1, ws2, ws3) and a descriptive name:
- `ws1: tier_a100` (feature implementation)
- `ws2: foundation` (foundation model investigation)
- `ws3: phase6c` (Phase 6C experiments)

Names are flexible and change as work evolves.

### Workstream Lifecycle

| Status | Criteria | Action |
|--------|----------|--------|
| active | Updated within 24h | Show in global, detailed context |
| paused | No update 3-7 days | Mark paused in global, keep file |
| inactive | No update >7 days | Remove from global table, keep file |
| archived | User archives explicitly | Move to workstreams/archive/ |

---

## Handoff Triggers

Initiate handoff when:
- User requests it
- Session ending
- Context window ~80% full
- Switching to different task area
- Before any extended break

---

## Handoff Protocol

### Step 1: Determine Workstream

If not obvious from conversation context, ask:
```
Which workstream is this session for?
1. ws1 (tier_a100)
2. ws2 (foundation)
3. ws3 (new/other)
```

**Auto-detection heuristics:**
- Keywords: "tier_a100" → ws1, "foundation" → ws2, "phase6c" → ws3
- Files modified: `tier_a100.py` → ws1, `experiments/foundation/*` → ws2

### Step 2: Capture Workstream State

Write to `.claude/context/workstreams/ws{N}_context.md`:

```markdown
# Workstream [N] Context: [Name]
# Last Updated: [YYYY-MM-DD HH:MM]

## Identity
- **ID**: ws[N]
- **Name**: [descriptive name]
- **Focus**: [brief focus description]
- **Status**: active

## Current Task
- **Working on**: [task]
- **Status**: [in progress / blocked / complete]

## Progress Summary

### Completed
[List completed items with dates]

### Pending
[List pending items]

## Last Session Work ([date])
[Detailed work from this session]

## Files Owned/Modified
- `path/to/file.py` - PRIMARY/SHARED
  - [nature of changes]

## Key Decisions (Workstream-Specific)
- [Decision]: [Rationale]

## Session History
### [date]
- [work done]

## Next Session Should
1. [priority 1]
2. [priority 2]

## Memory Entities (Workstream-Specific)
- [EntityName]: [brief description]
```

### Step 3: Update Global Context

Update `.claude/context/global_context.md`:

1. **Update workstream row** in Active Workstreams table
2. **Update Shared State** if changed (branch, tests, data versions)
3. **Update Cross-Workstream Coordination** if relevant
4. **Prune stale workstreams** (>7 days inactive → remove from table)

### Step 4: Data Version Snapshot

- Record latest raw manifest entry (dataset, file name, md5, timestamp)
- Record latest processed manifest entry (dataset, version, tier, md5)
- Note pending downloads or processed dataset versions that still need manifest entries

### Step 5: Update Phase Tracker

If phase progress changed, update `.claude/context/phase_tracker.md`:

```markdown
## Phase N: [Phase Name] [STATUS]
- Task A: ✅ Complete [date]
- Task B: 🔄 In Progress (50%)
- Task C: ⏸️ Pending
```

### Step 6: Verify User Preferences Section

Confirm `global_context.md` contains complete "User Preferences (Authoritative)" section with all subsections:
- Development Approach (TDD, planning, tmux)
- Context Durability (Memory MCP, context files, docs/)
- Documentation Philosophy (consolidation, precision, flat structure)
- Communication Standards (precision, no summarizing away details)
- Hyperparameters (Fixed - Ablation-Validated)

If section is missing or incomplete, reconstruct from `User_Preferences_Authoritative` Memory entity.

### Step 7: Confirm

Report to user:
- Workstream identified
- Both files updated (workstream + global)
- Summary of state
- Any uncommitted work warning
- Other active workstreams status

---

## Restore Protocol

### At Session Start

1. **Read global context**
   ```
   .claude/context/global_context.md
   ```

2. **Show active workstreams summary**

3. **Auto-detect workstream** from user's first message/task

4. **Confirm workstream selection** with user

5. **Read workstream context**
   ```
   .claude/context/workstreams/ws{N}_context.md
   ```

6. **Verify environment**
   ```bash
   source venv/bin/activate
   make test
   git status
   make verify
   ```

7. **Show cross-workstream coordination notes**

8. **Summarize to user**
   - Workstream state
   - Current task status
   - Test status
   - Other workstreams status
   - Proposed next steps

9. **Confirm priorities**
   - Do not assume previous priorities still hold
   - Ask user to confirm or redirect

10. **Only then proceed**

---

## Context File Locations

| File | Purpose | Git-tracked |
|------|---------|-------------|
| `.claude/context/global_context.md` | All workstreams summary | Yes |
| `.claude/context/workstreams/ws{N}_context.md` | Per-workstream detail | Yes |
| `.claude/context/phase_tracker.md` | Phase progress | Yes |
| `.claude/context/decision_log.md` | Architectural decisions | Yes |

**Note:** `session_context.md` is the legacy format. New sessions use the multi-workstream structure.

---

## Decision Log Format

When significant decisions are made, append to `.claude/context/decision_log.md`:

```markdown
## [YYYY-MM-DD] [Decision Title]

**Context**: [What prompted this decision]

**Decision**: [What was decided]

**Rationale**: [Why this choice]

**Alternatives Considered**:
- [Alternative 1]: [Why rejected]
- [Alternative 2]: [Why rejected]

**Implications**: [What this affects going forward]

**Workstream**: [ws1/ws2/ws3/global]
```

---

## User Commands

Users can trigger protocols with:

- `"handoff"` or `"session handoff"` → Run handoff protocol
- `"restore"` or `"session restore"` → Run restore protocol
- `"status"` → Quick summary without full restore
- `"where were we"` → Restore and summarize

---

## Handoff Checklist

Before ending session, verify:

- [ ] Workstream identified
- [ ] workstreams/ws{N}_context.md updated
- [ ] global_context.md updated
- [ ] phase_tracker.md updated (if progress made)
- [ ] decision_log.md updated (if decisions made)
- [ ] All files saved
- [ ] Git status clean (or uncommitted work noted)
- [ ] Test status noted
- [ ] Next priorities clear
- [ ] Cross-workstream coordination notes updated

---

## Cross-Workstream Coordination

### File Ownership

Maintain file ownership in `global_context.md`:
- PRIMARY: Workstream owns the file exclusively
- SHARED: Multiple workstreams may modify

### Blocking Dependencies

Document when one workstream blocks another:
- `[ws1 blocks ws3]`: tier_a100 features needed for Phase 6C experiments

### Shared Resources

Note when workstreams share resources that require coordination:
- Both ws1 and ws3 modify `src/features/` - coordinate commits
