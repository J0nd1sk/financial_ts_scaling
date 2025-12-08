---
name: query_lessons
description: Query Memory MCP for lessons, patterns, and decisions relevant to current work. Use when starting new tasks, investigating issues, or needing to recall project-specific knowledge.
---

# Query Lessons Skill

Search Memory MCP for relevant knowledge to inform current work.

## When to Use

**Proactive usage:**
- Before starting a new task or phase
- When encountering a familiar-seeming problem
- Before making architectural decisions
- When debugging issues that might have occurred before

**Reactive usage:**
- User asks "what did we learn about X?"
- User says "check memory", "query lessons", "what do we know about"
- When unsure if a pattern/lesson exists for current situation

**Automatic triggers:**
- Beginning of planning sessions (via planning_session skill)
- During debugging/troubleshooting workflows
- Before implementing solutions to known problem types

## Execution Steps

1. **Identify Query Keywords**

   Based on current context, extract:
   - **Domain**: data pipeline, testing, feature engineering, deployment
   - **Phase**: Phase 1, Phase 2, Phase 3, etc.
   - **Concept**: mocking, fixtures, manifest, indicators, models
   - **Problem type**: bug, performance, architecture, quality

2. **Search Memory MCP**

   Use `search_nodes` with targeted query:

   ```
   mcp__memory__search_nodes({
     "query": "[domain + concept keywords]"
   })
   ```

   Examples:
   - "data pipeline testing mock" - find testing lessons for data work
   - "Phase 2 manifest automation" - find Phase 2 data versioning lessons
   - "TDD pattern" - find test-driven development patterns
   - "Memory MCP decision" - find architectural decisions about Memory

3. **Filter and Categorize Results**

   Organize findings by relevance:
   - **Critical lessons**: Must-follow rules (CRITICAL severity)
   - **Important patterns**: Proven approaches to replicate
   - **Anti-patterns**: Things to avoid
   - **Decisions**: Architectural constraints and rationale
   - **Context**: Historical information for understanding

4. **Open Specific Nodes for Details**

   If search results look relevant, use `open_nodes` to get full observations:

   ```
   mcp__memory__open_nodes({
     "names": ["Testing Best Practices", "Data Pipeline Automation"]
   })
   ```

5. **Present Findings**

   Format results for user and/or integrate into current workflow.

## Output Format

```markdown
## Memory Query Results: [query topic]

### Critical Lessons (Must Follow)
- **[Lesson name]**: [Observation summary with severity and context]
  - Phase: [phase]
  - Context: [when this applies]

### Patterns to Apply
- **[Pattern name]**: [Observation summary]
  - When to use: [context]
  - Example: [if available]

### Anti-Patterns to Avoid
- **[Anti-pattern name]**: [What to avoid and why]
  - Consequence: [what happens if you do this]

### Relevant Decisions
- **[Decision name]**: [What was decided]
  - Rationale: [why]
  - Implications: [how this constrains current work]

### Additional Context
- [Other relevant observations]

---
**[N] total entities found, [M] directly relevant**
```

## Query Strategies

### Broad Discovery
Use when exploring a new area:
```
query: "Phase 2"  # Find everything related to current phase
query: "testing"  # Find all testing-related knowledge
```

### Targeted Lookup
Use when solving specific problems:
```
query: "mock API testing"  # Specific lesson lookup
query: "manifest registration automation"  # Specific pattern lookup
```

### Relationship Exploration
After finding relevant entities, explore connections:
```
1. search_nodes for initial entities
2. read_graph to see full relationship network
3. open_nodes on related entities
```

## Integration with Other Skills

**planning_session**: Automatically queries Memory in Step 0

**session_restore**: Queries Memory for lessons from current phase

**capture_lesson**: Stores new lessons that will be found by future queries

**session_handoff**: Triggers capture_lesson for session learnings

## Examples

### Example 1: Before Data Pipeline Work

**Context**: About to implement indicator calculations

**Query**:
```
mcp__memory__search_nodes({
  "query": "data pipeline testing automation"
})
```

**Result**:
```
Found 3 entities:
- Testing Best Practices (lesson)
- Data Pipeline Automation (pattern)
- Data Versioning System (pattern)
```

**Action**: Apply lessons about mocking APIs, using fixtures, and automatic manifest registration to indicator calculation implementation.

### Example 2: Debugging Test Failures

**Context**: Tests are slow and flaky

**Query**:
```
mcp__memory__search_nodes({
  "query": "testing mock API"
})
```

**Result**:
```
CRITICAL Lesson: Always mock external APIs in tests
Context: Tests calling real yfinance causing slow, flaky, non-reproducible tests
```

**Action**: Implement mocking based on stored lesson.

### Example 3: Architectural Decision

**Context**: Deciding how to structure feature processing

**Query**:
```
mcp__memory__search_nodes({
  "query": "feature engineering architecture decision"
})
```

**Result**: Check for existing decisions and patterns before proposing new approach.

## Critical Notes

- **Query iteratively**: Start broad, then narrow based on initial results
- **Verify relevance**: Not all matches will apply to current context
- **Update if needed**: If you discover lessons are outdated or incomplete, update them
- **Combine sources**: Memory + decision_log.md + code review for complete context
- **Don't over-rely**: Memory aids decision-making but doesn't replace thinking
- **Empty results are OK**: No matches means you might be exploring new territory

## Performance Tips

- Use specific multi-word queries for better precision
- Include phase numbers when looking for recent work
- Use domain + concept combinations: "data + testing", "pipeline + automation"
- Search before creating new lessons to avoid duplication
