---
name: capture_lesson
description: Store a lesson learned, pattern, or decision in Memory MCP. Use immediately after making a mistake and correcting it, discovering a successful pattern, or when user explicitly says "remember this". Supplements (not replaces) decision_log.md.
---

# Capture Lesson Skill

Store knowledge in Memory MCP for future retrieval.

## When to Use

**Automatic triggers:**
- After making a mistake and fixing it
- After code review identifies an issue
- After discovering an anti-pattern
- After successfully implementing a pattern

**User triggers:**
- User says "remember this"
- User says "add to memory"
- User says "don't forget"

**Proactive usage:**
- After completing a complex task
- When you realize something worth documenting
- Before session end (via session_handoff)

## Execution Steps

1. **Identify the Knowledge Type**

   Categorize what you're capturing:
   - `lesson`: Mistakes and how to avoid them
   - `pattern`: Successful approaches to replicate
   - `anti-pattern`: Things to avoid doing
   - `decision`: Architectural or process choices
   - `constraint`: Non-negotiable requirements

2. **Extract Key Information**

   - **Content**: Clear, actionable statement (1-2 sentences)
   - **Context**: When/where this applies
   - **Phase**: Which project phase this relates to
   - **Date**: Current session date

3. **Store in Memory MCP**

   Use appropriate format based on type:

   **For lessons learned:**
   ```
   mcp__memory__store_memory({
     "content": "Lesson: [specific lesson with actionable takeaway]",
     "metadata": {
       "type": "lesson",
       "phase": "[current phase]",
       "context": "[when this applies, what triggered it]",
       "session_date": "[YYYY-MM-DD]",
       "severity": "critical|important|helpful"
     }
   })
   ```

   **For successful patterns:**
   ```
   mcp__memory__store_memory({
     "content": "Pattern: [successful approach with specifics]",
     "metadata": {
       "type": "pattern",
       "phase": "[current phase]",
       "context": "[where to apply, what problem it solves]",
       "session_date": "[YYYY-MM-DD]",
       "tags": ["testing", "data-pipeline", etc.]
     }
   })
   ```

   **For anti-patterns:**
   ```
   mcp__memory__store_memory({
     "content": "Anti-pattern: [what to avoid and why]",
     "metadata": {
       "type": "anti-pattern",
       "phase": "[current phase]",
       "context": "[consequences, better alternatives]",
       "session_date": "[YYYY-MM-DD]",
       "severity": "critical|important|minor"
     }
   })
   ```

   **For decisions:**
   ```
   mcp__memory__store_memory({
     "content": "Decision: [what was decided]",
     "metadata": {
       "type": "decision",
       "phase": "[current phase]",
       "rationale": "[why this was chosen]",
       "alternatives": "[what was rejected]",
       "session_date": "[YYYY-MM-DD]"
     }
   })
   ```

4. **Also Update decision_log.md (if applicable)**

   For important decisions or lessons, ALSO add to decision_log.md:
   - Memory MCP is for agent-queryable knowledge
   - decision_log.md is for human-readable, version-controlled record
   - Both serve complementary purposes

5. **Confirm to User**

   Report what was stored:
   ```
   ✅ Captured in Memory MCP: [summary]
   Type: [lesson/pattern/anti-pattern/decision]
   Phase: [phase]

   [Also added to decision_log.md if applicable]
   ```

## Examples

### Example 1: Code Review Lesson

**User feedback**: "Your tests call real APIs - that's wrong, they should use mocks"

**Agent action**:
```
1. Store in Memory:
   content: "Lesson: Always mock external APIs in tests - never call real services. Tests must be fast, deterministic, and offline-capable."
   metadata: {
     type: "lesson",
     phase: "Phase 2",
     context: "Data download tests - code review identified network-dependent tests",
     session_date: "2025-12-08",
     severity: "critical",
     tags: ["testing", "mocking", "best-practice"]
   }

2. Also update decision_log.md with full context

3. Report: "✅ Captured in Memory MCP: Mock external APIs in tests (critical lesson)"
```

### Example 2: Successful Pattern

**Observation**: TDD workflow worked well for data pipeline

**Agent action**:
```
1. Store in Memory:
   content: "Pattern: TDD with RED-GREEN-REFACTOR cycle prevents bugs and ensures testability. Write tests first, confirm RED, implement minimal code, confirm GREEN."
   metadata: {
     type: "pattern",
     phase: "Phase 2",
     context: "Data pipeline implementation - 8 tests written first, all passed after implementation",
     session_date: "2025-12-08",
     tags: ["tdd", "testing", "workflow"]
   }

2. Report: "✅ Captured in Memory MCP: TDD cycle pattern for data pipeline work"
```

### Example 3: User Request

**User**: "Remember: dataset naming should be TICKER.TYPE.FREQUENCY"

**Agent action**:
```
1. Store in Memory:
   content: "Convention: Dataset naming format is {TICKER}.{TYPE}.{FREQUENCY} (e.g., SPY.OHLCV.daily). Enables future expansion while maintaining provenance."
   metadata: {
     type: "decision",
     phase: "Phase 2",
     rationale: "Hierarchical naming supports multiple data types and frequencies",
     alternatives: "Simple ticker names rejected as ambiguous",
     session_date: "2025-12-08",
     tags: ["naming", "data", "convention"]
   }

2. Also update decision_log.md (this is an architectural decision)

3. Report: "✅ Captured in Memory MCP: Dataset naming convention"
```

## Output Format

```
✅ Captured in Memory MCP: [brief summary]

Type: [lesson/pattern/anti-pattern/decision]
Phase: [phase]
Context: [when/where this applies]

[✅ Also added to decision_log.md if applicable]
```

## Critical Notes

- **Supplement, don't replace**: Memory MCP is additive, context files remain primary
- **Be specific**: Vague lessons aren't useful ("be careful" vs "mock APIs in tests")
- **Include context**: When/where does this apply? What triggered it?
- **Tag appropriately**: Use consistent tags for searchability
- **Update both when needed**: Important decisions go in both Memory and decision_log.md
- **Don't over-store**: Only capture truly useful knowledge, not trivial facts
