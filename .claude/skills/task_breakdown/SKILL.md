---
name: task_breakdown
description: Decompose large tasks into small, reviewable, independently testable subtasks. Use when a task touches more than 3 files, changes more than 50 lines, involves multiple concepts, or cannot be described in one sentence. Prevents scope creep and ensures incremental progress.
---

# Task Breakdown Skill

Decompose large work into reviewable atomic units.

## When to Use

- Task touches more than 3 files
- Task involves more than ~50 lines of changes
- Task involves multiple distinct concepts
- Task cannot be described in one sentence
- Task has multiple success criteria
- You feel uncertain about the full scope
- User says "break this down", "decompose", "smaller pieces"

## Why Decomposition Matters

Large tasks fail because:
- Scope creeps mid-implementation
- Errors compound before detection
- Reviews become superficial ("LGTM, too long to read")
- Rollback requires discarding good work with bad
- Context is lost before completion

Small tasks succeed because:
- Each piece is reviewable
- Each piece is testable
- Each piece is reversible
- Progress is visible
- Errors are isolated

## The Decomposition Process

### Step 1: Identify the Full Scope

List everything the task involves:
- Files to create
- Files to modify
- Functions to write
- Tests to add
- Dependencies needed
- Configuration changes

### Step 2: Find Natural Boundaries

Look for:
- Data vs logic vs interface
- Input validation vs processing vs output
- Setup vs core work vs cleanup
- Independent modules
- Layers (data → features → model → training)

### Step 3: Define Atomic Units

Each subtask must be:
- **Describable**: One sentence
- **Testable**: Has specific test(s)
- **Reviewable**: Can evaluate in isolation
- **Reversible**: Can undo without cascade
- **Complete**: Leaves codebase in working state

### Step 4: Order by Dependencies

```
Task A (no dependencies)
  ↓
Task B (depends on A)
  ↓
Task C (depends on B)
```

### Step 5: Present for Approval

## Output Format

```markdown
## Task Breakdown: [Original Task Name]

### Original Request
[What was asked for]

### Scope Analysis
- Files affected: [N]
- Estimated lines: ~[N]
- Concepts involved: [list]
- Decomposition required: Yes

---

### Subtasks

#### 1. [Subtask Name]
**Description:** [One sentence]
**Files:** `path/file.py`
**Tests:** `test_function_does_thing`
**Dependencies:** None
**Estimated:** ~[N] lines

#### 2. [Subtask Name]
**Description:** [One sentence]
**Files:** `path/file.py`
**Tests:** `test_other_thing`
**Dependencies:** Subtask 1
**Estimated:** ~[N] lines

#### 3. [Subtask Name]
...

---

### Execution Order
1. Subtask 1 (no deps)
2. Subtask 2 (after 1)
3. Subtask 3 (after 2)

### Checkpoints
After each subtask:
- [ ] Tests pass (`make test`)
- [ ] Code reviewed
- [ ] Can commit independently

---
**Approval Required**

Proceed with this breakdown? (yes / no / modify)
Start with Subtask 1? (yes / no)
```

## Decomposition Patterns

### Pattern: Data Pipeline Task

Original: "Build the data download and processing pipeline"

Breakdown:
1. Create download function for single ticker
2. Add validation for downloaded data
3. Create batch download for multiple tickers
4. Add parquet serialization
5. Create processing function for single file
6. Add batch processing
7. Integration test for full pipeline

### Pattern: Feature Implementation

Original: "Implement RSI indicator calculation"

Breakdown:
1. Write RSI calculation function (pure logic)
2. Add input validation
3. Add edge case handling (insufficient data)
4. Write unit tests
5. Integrate with feature pipeline
6. Add integration test

### Pattern: Model Component

Original: "Add PatchTST classification head"

Breakdown:
1. Define head architecture class
2. Add forward pass
3. Add parameter counting
4. Write unit tests for shapes
5. Integrate with base model
6. Add integration test with dummy data

## Granularity Guidelines

### Too Big (Decompose Further)
- "Implement the data pipeline"
- "Add all technical indicators"
- "Build the training loop"

### Right Size
- "Add RSI calculation function"
- "Create parquet save utility"
- "Add learning rate scheduler"

### Too Small (Can Combine)
- "Add import statement"
- "Fix typo in docstring"
- "Add single assertion to test"

## Handling Pushback

If user says "just do the whole thing":
- Acknowledge the request
- Explain the risk briefly
- Offer compromise: "I can do subtasks 1-3 together if you prefer, but let me checkpoint there before continuing"

If user approves full task anyway:
- Proceed but create internal checkpoints
- Report progress at natural boundaries
- Stop and report if issues arise

## Integration with Other Skills

After breakdown approval:
1. **For each subtask:**
   - Run `planning_session` skill
   - Run `test_first` skill
   - Run `approval_gate` skill for execution
2. **After each subtask:**
   - Run `make test`
   - Get approval before next subtask
3. **At natural breaks:**
   - Consider `session_handoff` if context heavy
