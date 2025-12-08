# Testing Rules

## 🔴 CRITICAL: Always Use `make test`

**NEVER run individual tests. ALWAYS use `make test`.**

```bash
# ✅ CORRECT - ALWAYS USE THIS:
make test

# ❌ FORBIDDEN - NEVER DO THESE:
pytest tests/specific_test.py
pytest -k test_method_name
pytest -k "pattern"
./venv/bin/pytest [anything]
python -m pytest tests/
```

**No exceptions.** Not for "quick checks." Not for debugging. Not for "just this one test."

---

## Why Full Suite Only

- Catches latent errors in seemingly unrelated code
- Detects integration issues between components
- Prevents regression of previously fixed bugs
- Data pipeline changes affect multiple components
- Full test suite pass = safe to proceed

---

## Test-Driven Development (TDD)

### Workflow

1. **Propose tests first**
   - What tests need to change?
   - What new tests are needed?
   - What should they assert?

2. **Wait for approval**

3. **Write failing tests**
   ```python
   def test_feature_does_thing():
       """Test that feature does thing when condition."""
       result = feature(input)
       assert result == expected
   ```

4. **Run `make test`**
   - Confirm new tests fail
   - Confirm existing tests still pass

5. **Propose implementation**

6. **Wait for approval**

7. **Write minimal implementation**
   - Only enough code to pass tests
   - No extra features
   - No "while I'm here" additions

8. **Run `make test`**
   - All tests must pass
   - No skipped tests

---

## Test Naming Convention

```python
def test_[function]_[scenario]_[expected_result]():
    """Test that [function] [expected_result] when [scenario]."""
```

Examples:
```python
def test_download_ohlcv_invalid_ticker_raises_ValueError():
    """Test that download_ohlcv raises ValueError when ticker is invalid."""

def test_calculate_rsi_returns_array_with_correct_shape():
    """Test that calculate_rsi returns array matching input length."""

def test_patchtst_forward_pass_output_shape_matches_prediction_length():
    """Test that PatchTST forward pass output has correct prediction length."""
```

---

## Coverage Targets

| Component | Minimum Coverage |
|-----------|------------------|
| Data pipeline | 80% |
| Feature calculation | 80% |
| Model architecture | 60% |
| Training loops | 50% |
| Utility functions | 90% |

---

## Test Execution Order

1. Make code changes
2. Run `make test`
3. Fix ALL failing tests before proceeding
4. Only after ALL tests pass, proceed with git operations
5. If tests fail, propose fixes and wait for approval

---

## Test File Organization

```
tests/
├── unit/
│   ├── test_data_download.py
│   ├── test_feature_calculation.py
│   ├── test_model_config.py
│   └── test_utils.py
├── integration/
│   ├── test_data_pipeline.py
│   ├── test_training_loop.py
│   └── test_evaluation.py
└── conftest.py
```

---

## Forbidden Practices

- Running individual tests
- Skipping tests with `@pytest.mark.skip`
- Using `pytest.mark.xfail` without approval
- Commenting out tests
- Reducing coverage to make tests pass
- Writing tests after implementation

---

## Before Proposing Any Change

Ask yourself:
1. What tests verify this behavior currently?
2. What tests need to change?
3. What new tests are needed?
4. Can I describe the test assertions before writing code?

If you cannot answer these, you're not ready to implement.
