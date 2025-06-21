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
	@echo "  format       - Auto-format code with ruff (replaces black)"
	@echo "  check        - Format code and run all quality checks (recommended)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test         - Run full test suite with coverage"
	@echo "  test-fast    - Run basic tests only (quick validation)"
	@echo "  test-coverage - Generate HTML coverage report"
	@echo ""
	@echo "🖥️ UI Testing:"
	@echo "  test-ui      - Run all UI tests with Streamlit testing framework"
	@echo "  test-ui-smoke - Run quick UI smoke tests"
	@echo ""
	@echo "🎯 Advanced Testing:"
	@echo "  test-all-categories - Run all test categories (basic, advanced, polars, ui)"
	@echo "  test-comprehensive - Run comprehensive test suite with coverage + UI"
	@echo "  test-category CATEGORY=<name> - Run specific test category"
	@echo ""
	@echo "🏗️ Development:"
	@echo "  run          - Start the Streamlit application"
	@echo "  docs         - Generate or update documentation"
	@echo ""
	@echo "🔧 Workflow Commands:"
	@echo "  pre-commit   - Quick checks before git commit (format + fast tests + UI smoke)"
	@echo "  validate     - Full validation (comprehensive quality + testing)"
	@echo "  ci-check     - Simulate CI/CD workflow"
	@echo ""
	@echo "💡 Recommended workflow:"
	@echo "  1. make install-dev    # One-time setup"
	@echo "  2. make pre-commit     # Before committing changes"
	@echo "  3. make validate       # Full project validation"
	@echo ""
	@echo "📋 Test Discovery:"
	@echo "  test-discovery - List all available tests and categories"
	@echo "  validate-sync  - Validate local/CI environment synchronization"

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
	@echo "  📋 Fixing auto-fixable issues with ruff..."
	ruff check . --fix --exit-zero
	@echo "  🎨 Formatting code with ruff..."
	ruff format .
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

# Generate test data (automatically called by other test commands)
generate-test-data:
	@echo "🔄 Ensuring test data is available..."
	@python tests/test_data_generator.py
	@echo "✅ Test data ready!"

test: generate-test-data
	@echo "🧪 Running full test suite with coverage..."
	python -m pytest tests/ --cov=app --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml:coverage.xml --junit-xml=test-results.xml -v
	@echo "✅ All tests completed!"

test-fast: generate-test-data
	@echo "⚡ Running basic tests (quick validation)..."
	python -m pytest tests/ -v --tb=short -x
	@echo "✅ Basic tests completed!"

test-coverage: generate-test-data
	@echo "📊 Generating HTML coverage report..."
	python -m pytest tests/ --cov=app --cov-report=html:htmlcov --cov-report=term-missing -v
	@echo "📋 Coverage report generated in htmlcov/index.html"
	@echo "💡 Open htmlcov/index.html in your browser to view detailed coverage"

# UI Testing Commands (using run_tests.py for comprehensive UI testing)
test-ui: generate-test-data
	@echo "🖥️ Running UI tests with Streamlit testing framework..."
	python run_tests.py --ui-all
	@echo "✅ UI tests completed!"

test-ui-smoke:
	@echo "💨 Running UI smoke tests..."
	python run_tests.py --smoke
	@echo "✅ UI smoke tests completed!"

# Advanced Testing Commands
test-all-categories: generate-test-data
	@echo "🎯 Running all test categories..."
	python run_tests.py --basic-all
	python run_tests.py --advanced-all
	python run_tests.py --polars-all
	python run_tests.py --ui-all
	@echo "✅ All test categories completed!"

# Comprehensive testing with all frameworks
test-comprehensive: generate-test-data
	@echo "🔬 Running comprehensive test suite..."
	@echo "  📋 Running pytest tests..."
	python -m pytest tests/ --cov=app --cov-report=html:htmlcov --cov-report=term-missing -v
	@echo "  🖥️ Running UI tests..."
	python run_tests.py --ui-all
	@echo "  ⚡ Running performance tests..."
	python run_tests.py --benchmarks
	@echo "  🎯 Running new comprehensive test suites..."
	make test-new-suites
	@echo "✅ Comprehensive testing completed!"

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
	python -m pytest tests/ -k "performance or benchmark" -v
	python run_tests.py --benchmarks
	@echo "✅ Performance tests completed!"

# Polars-specific testing
test-polars:
	@echo "🚀 Running Polars optimization tests..."
	python -m pytest tests/ -k "polars" -v
	python run_tests.py --polars-all
	@echo "✅ Polars tests completed!"

# Integration testing
test-integration:
	@echo "🔗 Running integration tests..."
	python -m pytest tests/ -k "integration" -v
	python run_tests.py --advanced-all
	@echo "✅ Integration tests completed!"

# Process Mining testing
test-process-mining:
	@echo "🔍 Running process mining tests..."
	python run_tests.py --process-mining-all
	@echo "✅ Process mining tests completed!"

# Fallback mechanism testing
test-fallback:
	@echo "🔄 Running fallback mechanism tests..."
	python run_tests.py --fallback-all
	@echo "✅ Fallback tests completed!"

# File Upload Testing Suite
test-file-upload:
	@echo "📁 Running file upload testing suite..."
	python -m pytest tests/test_file_upload_comprehensive.py -v
	@echo "✅ File upload tests completed!"

# Error Boundary Testing Suite
test-error-boundary:
	@echo "🚨 Running error boundary testing suite..."
	python -m pytest tests/test_error_boundary_comprehensive.py -v
	@echo "✅ Error boundary tests completed!"

# Visualization Pipeline Testing Suite
test-visualization:
	@echo "📊 Running visualization pipeline testing suite..."
	python -m pytest tests/test_visualization_pipeline_comprehensive.py -v
	@echo "✅ Visualization tests completed!"

# New Comprehensive Testing Suites
test-new-suites: test-file-upload test-error-boundary test-visualization
	@echo "🎯 All new testing suites completed!"

# CI/CD workflow simulation
ci-check: install-dev clean check test-comprehensive
	@echo "🎯 CI/CD workflow simulation complete!"
	@echo "✅ Ready for continuous integration!"

# Pre-commit workflow (recommended before git commit)
pre-commit: check test-fast test-ui-smoke
	@echo "✅ Pre-commit checks complete - ready to commit!"

# Full validation (comprehensive quality check)
validate: clean check test-comprehensive
	@echo "🎯 Full validation complete!"
	@echo "📊 Code coverage report: htmlcov/index.html"
	@echo "✅ Project is ready for production!"

# Future-proof test runner integration
test-category:
	@echo "🧪 Running specific test category..."
	@echo "Usage: make test-category CATEGORY=<category_name>"
	@echo "Available categories: basic, advanced, polars, ui, process_mining, fallback, benchmark"
	@if [ -n "$(CATEGORY)" ]; then \
		python run_tests.py --$(CATEGORY)-all; \
	else \
		echo "❌ Please specify CATEGORY variable"; \
		echo "Example: make test-category CATEGORY=ui"; \
	fi

# Test discovery and validation
test-discovery:
	@echo "🔍 Discovering and validating test structure..."
	@echo "📋 Available test files:"
	@find tests/ -name "test_*.py" -type f | sort
	@echo ""
	@echo "🎯 Test categories in run_tests.py:"
	@python run_tests.py --list
	@echo ""
	@echo "📊 Test collection validation:"
	@python -m pytest tests/ --collect-only -q | grep "collected" || echo "No tests collected"

# Validate synchronization between local and CI environments
validate-sync:
	@echo "🔄 Validating synchronization between local and CI environments..."
	@echo ""
	@echo "📋 Checking Python version consistency:"
	@python -c "import tomllib; config = tomllib.load(open('pyproject.toml', 'rb')); print(f'  pyproject.toml: {config[\"project\"][\"requires-python\"]}')"
	@grep -o 'python-version: \[.*\]' .github/workflows/tests.yml | head -1 | sed 's/python-version: /  workflows: /' 2>/dev/null || echo "  workflows: not found"
	@echo ""
	@echo "🔧 Testing key commands:"
	@make check > /dev/null 2>&1 && echo "  ✅ make check: works" || echo "  ❌ make check: failed"
	@ruff check . --output-format=github > /dev/null 2>&1 && echo "  ✅ GitHub format: works" || echo "  ❌ GitHub format: failed"
	@python -m pytest tests/ --collect-only > /dev/null 2>&1 && echo "  ✅ test validation: works" || echo "  ❌ test validation: failed"
	@python run_tests.py --list > /dev/null 2>&1 && echo "  ✅ UI test runner: works" || echo "  ❌ UI test runner: failed"
	@echo ""
	@echo "📊 Dependency consistency:"
	@echo "  requirements.txt: $$(wc -l < requirements.txt | tr -d ' ') packages"
	@echo "  requirements-dev.txt: $$(wc -l < requirements-dev.txt | tr -d ' ') packages"
	@echo ""
	@echo "✅ Synchronization validation complete!"
