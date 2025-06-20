# =====================================================================
# Funnel Analytics Platform - Modern Development Workflow
# =====================================================================
#
# This Makefile provides a unified interface for code quality, testing,
# and development tasks. Designed for both human developers and CI/CD.
#
# Quick Start:
#   make help      - Show all available commands
#   make install   - Install dependencies
#   make check     - Format code and run all quality checks
#   make test      - Run the full test suite
# =====================================================================

.PHONY: help install install-dev clean lint format check test test-fast test-coverage docs

# Default target
help:
	@echo "🚀 Funnel Analytics Platform - Development Commands"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies (linting, testing)"
	@echo "  clean        - Remove cache files and build artifacts"
	@echo ""
	@echo "🔍 Code Quality & Formatting:"
	@echo "  lint         - Run ruff and mypy to check for errors"
	@echo "  format       - Auto-format code with ruff and black"
	@echo "  check        - Format code and run all quality checks (recommended)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test         - Run full test suite with coverage"
	@echo "  test-fast    - Run basic tests only (quick validation)"
	@echo "  test-coverage - Generate HTML coverage report"
	@echo ""
	@echo "🏗️ Development:"
	@echo "  run          - Start the Streamlit application"
	@echo "  docs         - Generate or update documentation"
	@echo ""
	@echo "💡 Recommended workflow:"
	@echo "  1. make install-dev    # One-time setup"
	@echo "  2. make check          # Before committing changes"
	@echo "  3. make test           # Validate functionality"

# =====================================================================
# Setup & Installation Commands
# =====================================================================
install:
	@echo "📦 Installing production dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	@echo "🛠️ Installing development dependencies..."
	pip install -r requirements-dev.txt
	@echo "✅ Development environment ready!"

clean:
	@echo "🧹 Cleaning cache files and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "build" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "✅ Cleanup complete!"

# =====================================================================
# Code Quality & Formatting Commands
# =====================================================================
format:
	@echo "🎨 Auto-formatting code..."
	@echo "  📋 Sorting imports and fixing auto-fixable issues with ruff..."
	ruff check . --fix --exit-zero
	@echo "  ⚫ Formatting code with black..."
	black .
	@echo "✅ Code formatting complete!"

lint:
	@echo "🔍 Running code quality checks..."
	@echo "  🚀 Running ruff linter..."
	ruff check .
	@echo "  🔎 Running mypy type checker..."
	mypy .
	@echo "✅ All quality checks passed!"

# Combined format + lint command (recommended for pre-commit)
check: format lint
	@echo "✅ Code formatting and quality checks complete!"

# =====================================================================
# Testing Commands
# =====================================================================
test:
	@echo "🧪 Running full test suite with coverage..."
	python run_tests.py --coverage
	@echo "✅ All tests completed!"

test-fast:
	@echo "⚡ Running basic tests (quick validation)..."
	python run_tests.py --basic-all
	@echo "✅ Basic tests completed!"

test-coverage:
	@echo "📊 Generating HTML coverage report..."
	python run_tests.py --coverage
	@echo "📋 Coverage report generated in htmlcov/index.html"
	@echo "💡 Open htmlcov/index.html in your browser to view detailed coverage"

# =====================================================================
# Development Commands
# =====================================================================
run:
	@echo "🚀 Starting Streamlit application..."
	@echo "💡 Application will be available at http://localhost:8501"
	streamlit run app.py

docs:
	@echo "📚 Documentation commands will be added here..."
	@echo "💡 Consider adding sphinx or mkdocs for documentation generation"

# =====================================================================
# Advanced Development Commands
# =====================================================================

# Performance testing
test-performance:
	@echo "⚡ Running performance tests..."
	python run_tests.py --benchmarks
	@echo "✅ Performance tests completed!"

# Polars-specific testing
test-polars:
	@echo "🚀 Running Polars optimization tests..."
	python run_tests.py --polars-all
	@echo "✅ Polars tests completed!"

# Integration testing
test-integration:
	@echo "🔗 Running integration tests..."
	python run_tests.py --advanced-all
	@echo "✅ Integration tests completed!"

# CI/CD workflow simulation
ci-check: install-dev clean check test
	@echo "🎯 CI/CD workflow simulation complete!"
	@echo "✅ Ready for continuous integration!"

# Pre-commit workflow (recommended before git commit)
pre-commit: check test-fast
	@echo "✅ Pre-commit checks complete - ready to commit!"

# Full validation (comprehensive quality check)
validate: clean check test test-coverage
	@echo "🎯 Full validation complete!"
	@echo "📊 Code coverage report: htmlcov/index.html"
	@echo "✅ Project is ready for production!"

# Validate synchronization between local and CI environments
validate-sync:
	@echo "🔄 Validating synchronization between local and CI environments..."
	@echo ""
	@echo "📋 Checking Python version consistency:"
	@python -c "import tomllib; config = tomllib.load(open('pyproject.toml', 'rb')); print(f'  pyproject.toml: {config[\"project\"][\"requires-python\"]}')"
	@grep -o 'python-version: \[.*\]' .github/workflows/tests.yml | head -1 | sed 's/python-version: /  workflows: /'
	@echo ""
	@echo "🔧 Testing key commands:"
	@make check > /dev/null 2>&1 && echo "  ✅ make check: works" || echo "  ❌ make check: failed"
	@ruff check . --output-format=github > /dev/null 2>&1 && echo "  ✅ GitHub format: works" || echo "  ❌ GitHub format: failed"
	@python run_tests.py --validate > /dev/null 2>&1 && echo "  ✅ test validation: works" || echo "  ❌ test validation: failed"
	@echo ""
	@echo "📊 Dependency consistency:"
	@echo "  requirements.txt: $$(wc -l < requirements.txt | tr -d ' ') packages"
	@echo "  requirements-dev.txt: $$(wc -l < requirements-dev.txt | tr -d ' ') packages"
	@echo ""
	@echo "✅ Synchronization validation complete!"
