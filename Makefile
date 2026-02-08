.PHONY: test test-ws1 test-ws2 test-ws3 test-cov lint type-check clean help

# Python and pytest from virtual environment
PYTHON := ./venv/bin/python
PYTEST := ./venv/bin/pytest

# Lock file for preventing parallel test runs
LOCKFILE := .test.lock

# Default target
help:
	@echo "Available targets:"
	@echo "  make test       - Run all tests (REQUIRED before any git operations)"
	@echo "  make test-ws1   - Run ws1 (feature_generation) tests only (~30s)"
	@echo "  make test-ws2   - Run ws2 (foundation) tests only"
	@echo "  make test-ws3   - Run ws3 (phase6c/HPO) tests only"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make verify     - Run environment + data verification"
	@echo "  make lint       - Run ruff linter"
	@echo "  make type-check - Run mypy type checker"
	@echo "  make clean      - Remove cache and build artifacts"

# Primary test target - THIS IS THE ONLY WAY TO RUN TESTS
# Uses lock file to prevent parallel runs across terminals
# Note: If tests are killed, lock file may remain - run 'make clean' to remove
test:
	@if [ -f $(LOCKFILE) ]; then \
		echo "⏳ Tests already running in another terminal. Waiting..."; \
		echo "   (If stale, run 'make clean' to remove lock file)"; \
		while [ -f $(LOCKFILE) ]; do sleep 2; done; \
	fi
	@touch $(LOCKFILE)
	@echo "Running full test suite..."
	@$(PYTEST) tests/ -v --tb=short && echo "✅ All tests passed" || (rm -f $(LOCKFILE); exit 1)
	@rm -f $(LOCKFILE)

# Workstream-specific test targets (faster feedback during development)
# Note: Full 'make test' still required before git operations
test-ws1:
	@echo "Running ws1 (feature_generation) tests..."
	@$(PYTEST) tests/features/ -v --tb=short
	@echo "✅ ws1 tests passed"

test-ws2:
	@echo "Running ws2 (foundation) tests..."
	@$(PYTEST) tests/test_evaluation.py tests/test_context_ablation_nf.py tests/test_hpo_neuralforecast.py -v --tb=short
	@echo "✅ ws2 tests passed"

test-ws3:
	@echo "Running ws3 (phase6c/HPO/feature_embedding) tests..."
	@$(PYTEST) tests/test_hpo*.py tests/test_loss*.py tests/test_feature_embedding_experiments.py -v --tb=short 2>/dev/null || $(PYTEST) tests/test_loss*.py tests/test_feature_embedding_experiments.py -v --tb=short
	@echo "✅ ws3 tests passed"

# Test with coverage
test-cov:
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

verify:
	@echo "Verifying development environment..."
	$(PYTHON) scripts/verify_environment.py
	@echo "Verifying data manifests..."
	$(PYTHON) scripts/manage_data_versions.py verify

# Linting
lint:
	./venv/bin/ruff check src/ tests/

# Type checking
type-check:
	./venv/bin/mypy src/

# Clean artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage $(LOCKFILE)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true