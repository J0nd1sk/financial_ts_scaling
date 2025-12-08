---
name: test_first
description: Enforce test-driven development workflow. Use before implementing any feature or fix. Ensures tests are written and failing before implementation code. Prevents implementation-first habits that skip test coverage.
---

# Test First Skill

Enforce TDD: tests before implementation, always.

## When to Use

- After planning session approves a task
- Before writing ANY implementation code
- User says "test first", "TDD", "write tests"
- Whenever tempted to "just write the code first"

## The TDD Cycle

```
RED → GREEN → REFACTOR

1. RED: Write failing test
2. GREEN: Write minimal code to pass
3. REFACTOR: Clean up while tests pass
```

## Execution Steps

### Phase 1: Define Test Cases

Before writing any test code, enumerate:

1. **Happy Path Tests**
   - What should happen with valid input?
   - What is the expected output?

2. **Edge Case Tests**
   - Empty inputs?
   - Boundary values?
   - Maximum/minimum values?

3. **Error Case Tests**
   - Invalid inputs?
   - What exceptions should be raised?
   - What error messages?

4. **Integration Tests** (if applicable)
   - How does this interact with other components?
   - What state changes occur?

### Phase 2: Write Test Stubs

Create test file/functions with descriptive names:

```python
def test_function_valid_input_returns_expected():
    """Test that function returns expected when given valid input."""
    # Arrange
    # Act  
    # Assert
    pass

def test_function_empty_input_raises_ValueError():
    """Test that function raises ValueError when input is empty."""
    pass

def test_function_boundary_value_handled_correctly():
    """Test that function handles boundary value correctly."""
    pass
```

### Phase 3: Implement Test Assertions

Fill in the test bodies:

```python
def test_calculate_rsi_valid_prices_returns_array():
    """Test that calculate_rsi returns array for valid price series."""
    # Arrange
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106])
    
    # Act
    result = calculate_rsi(prices, period=14)
    
    # Assert
    assert isinstance(result, pd.Series)
    assert len(result) == len(prices)
    assert result.iloc[-1] >= 0
    assert result.iloc[-1] <= 100
```

### Phase 4: Verify Tests Fail

```bash
make test
```

- New tests MUST fail (they have no implementation yet)
- Existing tests MUST still pass
- Failure should be for the RIGHT reason (not syntax error)

### Phase 5: Present for Approval

Show user:
- Test code written
- `make test` output showing expected failures
- Request approval to write implementation

### Phase 6: Implementation (After Approval)

Write MINIMAL code to make tests pass:
- No extra features
- No "while I'm here" additions
- No premature optimization
- Just enough to pass

### Phase 7: Verify Tests Pass

```bash
make test
```

ALL tests must pass. If any fail:
- Do NOT proceed
- Report failure
- Propose fix (with approval)

## Output Format

### After Writing Tests

```markdown
## Test First: [Feature Name]

### Tests Written

**File:** `tests/test_feature.py`

```python
[test code]
```

### Test Execution

```
$ make test
...
FAILED tests/test_feature.py::test_function_valid - NotImplementedError
FAILED tests/test_feature.py::test_function_edge - NotImplementedError
...
2 failed, 45 passed
```

✅ New tests fail as expected (no implementation yet)
✅ Existing tests still pass

---
**Approval Required**

Tests ready. Proceed to implementation? (yes/no)
```

### After Implementation

```markdown
## Implementation Complete

### Code Written

**File:** `src/module.py`

```python
[implementation code]
```

### Test Execution

```
$ make test
...
47 passed
```

✅ All tests pass

---
Ready for commit? (yes/no)
```

## Common Violations to Avoid

❌ "I'll write tests after I see if the code works"
❌ "This is too simple to need tests"
❌ "Let me just sketch the implementation first"
❌ "The tests are obvious, I'll add them at the end"

✅ Tests ALWAYS come first
✅ No exceptions for "simple" code
✅ If you can't write the test, you don't understand the requirement

## Critical Notes

- Tests define the specification
- Implementation follows specification
- If tests are hard to write, the design needs work
- NEVER run individual tests - always `make test`
