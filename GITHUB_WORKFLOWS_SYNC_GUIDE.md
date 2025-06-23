# 🔄 GitHub Workflows Synchronization Guide

## 📋 Overview

This guide documents the synchronization between our local test runner (`run_tests.py`) and GitHub Actions workflows to ensure consistent CI/CD behavior.

## 🏗️ Workflow Architecture

### 1. Main Test Workflow (`tests.yml`)
**Triggers**: Push to main/develop, PRs to main
**Purpose**: Core functionality testing across multiple Python versions

```yaml
Strategy: Matrix testing on Python 3.9, 3.10, 3.11, 3.12
Jobs:
  - test: Basic test suite with coverage
  - advanced-tests: Comprehensive testing (main branch only)
  - lint: Code quality checks
  - ci-simulation: Full CI pipeline simulation
```

### 2. Pre-commit Workflow (`pre-commit.yml`)
**Triggers**: Pull requests to main/develop
**Purpose**: Fast validation for development contributions

```yaml
Jobs:
  - pre-commit: Format checking and basic tests
```

## 🔧 Command Synchronization

### Local Development Commands → GitHub Actions

| Local Command | GitHub Action | Purpose |
|---------------|---------------|---------|
| `make check` | `make check` | Format + lint validation |
| `make pre-commit` | `make pre-commit` | Pre-commit validation |
| `make ci-check` | `make ci-check` | Full CI simulation |
| `python run_tests.py --coverage` | `python run_tests.py --coverage --report` | Test execution with coverage |

### Test Categories Mapping

| Test Category | Local Command | GitHub Workflow Step |
|---------------|---------------|---------------------|
| Smoke Tests | `--smoke` | `Run smoke tests` |
| Basic Tests | `--basic-all` | `Run basic tests` |
| Polars Tests | `--polars-all` | `Run Polars tests` |
| Comprehensive | `--comprehensive-all` | `Run comprehensive tests` |
| Benchmarks | `--benchmarks` | `Run benchmarks` |

## 📊 Configuration Synchronization

### Python Version Support
```yaml
# GitHub Workflows
python-version: ["3.9", "3.10", "3.11", "3.12"]

# pyproject.toml
requires-python = ">=3.9"
target-version = "py39"
```

### Linter Configuration
```yaml
# GitHub Actions uses same commands as local development
- make check          # Format + lint
- ruff check . --output-format=github  # GitHub-specific format
```

### MyPy Configuration
```yaml
# Synchronized through pyproject.toml
[tool.mypy]
explicit_package_bases = true
no_site_packages = true
# ... other settings applied consistently
```

## 🚀 Workflow Jobs Breakdown

### Job: `test`
```yaml
Steps:
1. Setup Python matrix (3.9-3.12)
2. Install dependencies
3. Validate test environment
4. Run progressive test suite:
   - Smoke tests (quick validation)
   - Basic functionality tests
   - Polars engine tests
   - Data integrity tests
   - Full coverage report
5. Upload artifacts (coverage, test results)
```

### Job: `advanced-tests`
```yaml
Conditions: Main branch pushes only
Steps:
1. Setup Python 3.12
2. Run comprehensive test suite
3. Execute fallback detection tests
4. Generate performance benchmarks
5. Upload benchmark results
```

### Job: `lint`
```yaml
Steps:
1. Setup Python 3.12
2. Install all dependencies
3. Run unified code quality checks (make check)
4. Verify GitHub Actions format compatibility
```

### Job: `ci-simulation`
```yaml
Conditions: Main branch pushes, after test+lint success
Steps:
1. Setup Python 3.12
2. Execute full CI pipeline simulation
3. Upload comprehensive results
```

### Job: `pre-commit`
```yaml
Conditions: Pull requests only
Steps:
1. Setup Python 3.12
2. Run pre-commit workflow
3. Verify no formatting changes needed
4. Validate test runner functionality
```

## 📈 Artifact Management

### Coverage Reports
```yaml
Local: htmlcov/index.html
GitHub:
  - Codecov integration
  - HTML artifact upload
  - XML coverage for external tools
```

### Test Results
```yaml
Local: test-results.xml (JUnit format)
GitHub:
  - Per-Python-version results
  - Benchmark data (JSON)
  - CI simulation outputs
```

## 🔄 Synchronization Maintenance

### When to Update Workflows

1. **New Test Categories**: Add corresponding workflow steps
2. **Python Version Changes**: Update matrix strategy
3. **Dependency Updates**: Sync requirements files
4. **Linter Configuration**: Ensure pyproject.toml changes reflect in workflows
5. **New Make Commands**: Add to appropriate workflow jobs

### Validation Commands

```bash
# Test local-to-CI consistency
make ci-check                    # Should match GitHub CI behavior
python run_tests.py --validate   # Verify test environment
ruff check . --output-format=github  # Test GitHub format compatibility
```

## 🎯 Best Practices

### Development Workflow
```bash
1. make pre-commit     # Before creating PR
2. git commit          # Triggers pre-commit.yml
3. git push            # Triggers tests.yml
```

### Release Workflow
```bash
1. Merge to main       # Triggers full test suite
2. Advanced tests run  # Comprehensive validation
3. CI simulation       # Final verification
```

### Troubleshooting Sync Issues

```bash
# If local tests pass but GitHub fails:
1. Check Python version compatibility
2. Verify dependency versions match
3. Run: make ci-check (simulates GitHub environment)
4. Check artifact uploads for detailed logs
```

## 🔍 Monitoring & Alerts

### Success Indicators
- ✅ All matrix jobs pass
- ✅ Coverage reports generated
- ✅ No linter conflicts
- ✅ Artifacts uploaded successfully

### Failure Scenarios
- ❌ Python version incompatibility
- ❌ Dependency conflicts
- ❌ Test environment issues
- ❌ Linter configuration mismatches

## 📚 Related Documentation

- `LINTER_CONFLICTS_RESOLUTION_FINAL_REPORT.md` - Linter setup details
- `Makefile` - Local development commands
- `pyproject.toml` - Tool configurations
- `run_tests.py` - Test runner implementation

## 🎉 Conclusion

Our GitHub workflows are now fully synchronized with local development tools, ensuring:

1. **Consistency**: Same commands work locally and in CI
2. **Reliability**: Comprehensive testing across Python versions
3. **Efficiency**: Fast feedback through progressive test execution
4. **Quality**: Unified linting and formatting standards

**Status**: 🟢 **FULLY SYNCHRONIZED**
