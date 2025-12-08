.PHONY: test test-cov lint type-check clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  make test       - Run all tests (REQUIRED before any git operations)"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make verify     - Run environment verification script"
	@echo "  make lint       - Run ruff linter"
	@echo "  make type-check - Run mypy type checker"
	@echo "  make clean      - Remove cache and build artifacts"

# Primary test target - THIS IS THE ONLY WAY TO RUN TESTS
test:
	@echo "Running full test suite..."
	pytest tests/ -v --tb=short
	@echo "✅ All tests passed"

# Test with coverage
test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

verify:
	@echo "Verifying development environment..."
	python scripts/verify_environment.py

# Linting
lint:
	ruff check src/ tests/

# Type checking
type-check:
	mypy src/

# Clean artifacts
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true