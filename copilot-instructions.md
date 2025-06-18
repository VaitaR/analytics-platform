# 🔍 Funnel Analytics Platform - Instructions for LLM Agent

## �️ **QUICK REFERENCE INDEX**

> _Search by keywords to find what you need instantly_

**🔍 Search Keywords for Common Tasks:**

- `polars fallback pandas` → Section 9 (Known Issues)
- `performance bottleneck slow` → Section 7.2 (Performance Profiling)
- `counting method unique_users event_totals` → Section 8.1 (Domain Knowledge)
- `test funnel logic` → Section 4 (Testing) + `test_funnel_calculator_comprehensive.py`
- `visualization plotly chart` → Section 2.3 + `FunnelVisualizer class`
- `data validation schema` → Section 8.2 (Data Schema)
- `error debugging fallback` → Section 11 (Troubleshooting)
- `agent discoveries new patterns` → Section "AGENT DISCOVERIES"

**📋 Code Pattern Quick Access:**

- Adding new counting method → Section 10.1
- Performance optimization → Section 10.2
- Polars→Pandas fallback → Section 9.1
- Test template → Section 4.2
- Self-improvement updates → Section "SELF-IMPROVEMENT PROTOCOL"

---

## �🚀 0. How You, Copilot, Should Use This Document

**Your Role:** Expert Data Analytics Developer specializing in polars, streamlit, plothly, funnel analysis and performance optimization.

**Core Focus Areas:**

1. **Data Processing Excellence** - Polars optimization, large dataset handling. Try use only Polars, except problem cases.
2. **Funnel Logic Accuracy** - Complex conversion calculations, user journey analysis
3. **Performance Optimization** - Algorithm efficiency, caching, fallback mechanisms
4. **Visualization Quality** - Professional Plotly charts, interactive dashboards

### 0.1. Pre-Submission Checklist:

- [ ] **Data Accuracy:** Are funnel calculations mathematically correct?
- [ ] **Performance:** Will this scale to large datasets (1M+ events)?
- [ ] **Fallback Handling:** Graceful Polars→Pandas fallback when needed?
- [ ] **Code Quality:** Clean, well-documented, type-hinted?
- [ ] **Testing:** Unit tests for complex funnel logic?

---

## 📊 1. Project Overview

**Goal:** Enterprise-grade funnel analytics platform for analyzing user conversion journeys through event sequences.

**Core Value:** Transform raw event data into actionable funnel insights with professional visualizations.

**Key Features:**

- Multi-algorithm funnel calculation (unique_users, event_totals, unique_pairs)
- Real-time processing of large datasets (Polars optimization)
- Flexible conversion windows and re-entry modes
- ClickHouse integration for enterprise data
- Interactive Streamlit interface

**Target Users:** Data analysts, product managers, growth teams analyzing user conversion funnels.

---

## ⚡ **PROBLEM → SOLUTION MAPPING**

> _LLM Agent: Use Ctrl+F to find your exact problem_

| **Problem Description**                                  | **Keywords to Search**                                       | **Solution Location**                                                       |
| -------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| Polars calculation fails with "expression ambiguity"     | `polars fallback expression`                                 | Section 9.1 + test_polars_fallback_detection.py                             |
| Funnel results are mathematically incorrect              | `funnel logic accuracy counting`                             | Section 8.1 (Domain Knowledge)                                              |
| Performance slow on large datasets (>100k events)        | `performance optimization bottleneck`                        | Section 7.2 + get_bottleneck_analysis()                                     |
| Need to add new counting algorithm                       | `counting method template pattern`                           | Section 10.1 (Adding New Counting Method)                                   |
| Conversion rates don't match expected values             | `conversion calculation ordered unordered`                   | Section 8.1 (Funnel Orders)                                                 |
| Memory issues with million+ events                       | `memory management lazy evaluation`                          | Section 9.2 (Memory Management)                                             |
| Tests failing for edge cases                             | `test edge cases empty data`                                 | Section 4.2 (Critical Test Files)                                           |
| UI charts not rendering properly                         | `plotly visualization theme`                                 | FunnelVisualizer class methods                                              |
| Data validation errors on upload                         | `data schema validation required`                            | Section 8.2 (Data Schema)                                                   |
| Streamlit app crashing on large files                    | `streamlit caching memory clear`                             | Section 9.2 + DataSourceManager                                             |
| **Duplicate test files causing confusion**               | `duplicate test files consolidation conftest`                | **Section "AGENT DISCOVERIES" + unified run_tests.py**                      |
| **Test runner status misleading**                        | `pytest status reporting json test runner`                   | **Section "AGENT DISCOVERIES" + run_tests.py fixes**                        |
| **Time Series charts vertically stretched**              | `chart stretching vertical ui responsive height`             | **Section "AGENT DISCOVERIES" + TIME_SERIES_UI_FIXES.md**                   |
| **Inconsistent chart sizing and poor responsive design** | `universal visualization standards responsive design mobile` | **Section "AGENT DISCOVERIES" + test_universal_visualization_standards.py** |

---

## 🛠️ 2. Technology Stack

### 2.1. Core Framework

- **Streamlit 1.28+** - Web interface, caching, file uploads
  - _Rationale:_ Rapid analytics prototyping, built-in interactivity
- **Python 3.11+** - Modern language features, performance
  - _Rationale:_ Type hints, async support, dataclass features

### 2.2. Data Processing Engine

- **Polars (Primary)** - High-performance DataFrame operations
  - _Rationale:_ 10x faster than Pandas for large datasets, lazy evaluation
- **Pandas (Fallback)** - Complex operations compatibility
  - _Rationale:_ Fallback for edge cases, ecosystem compatibility
- **NumPy** - Mathematical operations, array processing

### 2.3. Visualization & UI

- **Plotly** - Professional interactive charts
  - _Rationale:_ Export-quality visuals, responsive design
- **Custom CSS** - Professional styling, accessibility
  - _Rationale:_ Enterprise-ready appearance

### 2.4. Data Sources

- **ClickHouse** - Enterprise OLAP database
  - _Rationale:_ Handles billions of events, real-time queries
- **File Upload** - CSV/Parquet support
  - _Rationale:_ Quick testing, small-scale analysis

---

## 🗺️ 3. Project Architecture & Key Files

### 3.1. Core Application Structure

```
/Users/andrew/Documents/projects/project_funnel/
├── app.py                    # Main Streamlit application
├── models.py                 # Data structures (FunnelConfig, FunnelResults)
├── path_analyzer.py          # User journey analysis engine
├── requirements.txt          # Dependencies
├── index.html               # Standalone HTML version
└── tests/                   # Comprehensive test suite
    ├── test_integration_flow.py
    ├── test_funnel_calculator_comprehensive.py
    ├──test_polars_fallback_detection.py
    └──...
```

### 3.2. Key Classes & Responsibilities

**DataSourceManager** (`app.py:150+`)

- File upload validation & processing
- ClickHouse connection management
- Sample data generation
- JSON property extraction (event/user properties)

**FunnelCalculator** (`app.py:300+`)

- Core funnel logic with multiple algorithms
- Polars optimization with Pandas fallback
- Performance monitoring & bottleneck analysis
- Conversion window handling

**PathAnalyzer** (`path_analyzer.py`)

- User journey analysis between steps
- Drop-off path identification
- Between-steps event analysis

**FunnelVisualizer** (`app.py:1500+`)

- Professional Plotly visualizations
- Dark theme, accessibility compliance
- Sankey diagrams, funnel charts, time series

---

## 🧪 4. Professional Testing Architecture

### 4.1. Unified Testing Approach

**Architecture Document:** [`tests/README.md`](tests/README.md)

**Core Principles:**

- **Professional Standards**: Enterprise-grade test patterns with proper fixtures
- **Performance Focus**: Built-in timing and memory monitoring for all tests
- **Polars-First Testing**: Test both Polars optimization and Pandas fallback
- **Data Factory Pattern**: Reusable test data generation with controlled characteristics
- **Comprehensive Validation**: Systematic edge case and boundary condition testing

### 4.2. Critical Test Files (Standardized)

- `conftest.py` - ✅ **UNIFIED** fixtures, data factories, and test utilities (single source of truth)
- `test_basic_scenarios.py` - Core happy path scenarios with @pytest.mark.basic
- `test_funnel_calculator_comprehensive.py` - Complete algorithm coverage
- `test_polars_fallback_detection.py` - Polars→Pandas fallback detection
- `test_integration_flow.py` - End-to-end workflow validation with @pytest.mark.integration
- `test_edge_cases.py` - Boundary conditions with @pytest.mark.edge_case
- `test_conversion_window.py` - Time-based logic with @pytest.mark.conversion_window
- `test_counting_methods.py` - Algorithm-specific tests with @pytest.mark.counting_method
- `test_segmentation.py` - Property filtering with @pytest.mark.segmentation
- `test_timeseries_analysis.py` - Time-based aggregation with @pytest.mark.performance
- `test_polars_engine.py` - Polars-specific functionality
- `test_polars_path_analysis.py` - Path analysis algorithms

### 4.3. Running Tests (Professional Commands)

```bash
# ✅ UPDATED: Use unified test runner
python run_tests.py                    # All tests (118 tests)
python run_tests.py --basic-all        # Basic tests (24 tests)
python run_tests.py --marker basic     # Tests with basic marker (8 tests)
python run_tests.py --marker edge_case # Edge case tests (12 tests)
python run_tests.py --smoke            # Quick smoke test
python run_tests.py --list             # List all available tests

# Performance tests with benchmarks
python run_tests.py --benchmarks
python run_tests.py --marker performance

# Polars optimization validation
python run_tests.py --polars-all
python run_tests.py --marker polars

# Integration and comprehensive tests
python run_tests.py --advanced-all
python run_tests.py --marker integration

# Coverage and parallel execution
python run_tests.py --coverage
python run_tests.py --parallel

# Direct pytest (if needed)
pytest tests/ -v --tb=short
pytest tests/ -m basic -v
pytest tests/ --cov=app --cov-report=html
```

### 4.4. Test Data Management

```python
# Professional test data generation
spec = TestDataSpec(
    total_users=10000,
    conversion_rates=[1.0, 0.7, 0.5, 0.3],
    time_spread_hours=168,
    include_noise_events=True,
    segment_distribution={'premium': 0.3, 'basic': 0.7}
)
test_data = TestDataFactory.create_funnel_data(spec, funnel_steps)

# Performance monitoring built-in
performance_monitor.time_operation("funnel_calculation",
                                  calculator.calculate_funnel_metrics,
                                  test_data, steps)
```

### 4.5. Quality Gates

- **Test Coverage**: >90% line coverage for core modules
- **Test Speed**: Full test suite completes in <60 seconds ✅ **ACHIEVED: 118 tests pass**
- **Scalability**: Tests validate performance up to 1M+ events
- **Reliability**: <1% flaky test rate, deterministic results ✅ **ACHIEVED: All tests stable**
- **Standards**: All tests follow unified patterns from `conftest.py` ✅ **ACHIEVED: Single source**
- **Architecture**: ✅ **COMPLETED: Unified testing architecture with single conftest.py and run_tests.py**

---

## 📏 5. Code Quality Standards

### 5.1. Data Processing Code

```python
# ✅ Good: Type-safe, performance-monitored, formatted with modern toolchain
@_funnel_performance_monitor('calculate_conversion')
def calculate_conversion(
    self,
    events_df: pl.DataFrame,
    steps: List[str]
) -> FunnelResults:
    """Calculate conversion rates using optimized Polars operations."""
    # Auto-formatted with Black, linted with Ruff
    return self._polars_implementation(events_df, steps)

# ❌ Bad: No fallback, no monitoring, poor formatting
def calculate_conversion(self, df, steps):
    return df.groupby('user_id').apply(lambda x: x['event_name'].isin(steps))
```

### 5.2. Code Quality Pipeline (NEW)

```python
# Automated quality enforcement with pyproject.toml configuration
[tool.ruff]
line-length = 99
select = ["E", "W", "F", "I", "B", "C4", "SIM", "UP", "N"]

[tool.black]
line-length = 99

[tool.mypy]
check_untyped_defs = true
ignore_missing_imports = true  # Flexible for data science libraries

# Quality gates enforced via pre-commit hooks
make check          # Format + lint before commits
make test          # Validate functionality
make ci-check      # Full CI/CD simulation
```

### 5.3. Performance Requirements

- **Large Dataset Support:** 1M+ events should complete in <30 seconds
- **Memory Efficiency:** Use lazy evaluation where possible
- **Graceful Degradation:** Always provide Pandas fallback

### 5.4. Funnel Logic Accuracy

- **Mathematical Correctness:** Conversion rates must be precise
- **Edge Case Handling:** Empty datasets, single users, missing events
- **Configuration Validation:** Validate funnel steps exist in data

---

## 🏗️ 6. Architectural Principles

### 6.1. Performance-First Design

- **Polars Primary:** Use Polars for all new data operations
- **Smart Fallbacks:** Detect when Polars fails, fallback gracefully
- **Lazy Evaluation:** Defer computation until required
- **Caching Strategy:** Cache preprocessed data, cleared on config change

### 6.2. Modularity & Separation

- **DataSourceManager:** Pure data I/O, no business logic
- **FunnelCalculator:** Core algorithms, no UI concerns
- **PathAnalyzer:** Specialized journey analysis
- **FunnelVisualizer:** Pure visualization, no data processing

### 6.3. Extensibility

- **Algorithm Plugins:** Easy to add new counting methods
- **Data Source Plugins:** Support new databases via DataSourceManager
- **Visualization Themes:** Configurable styling system

---

## 🔧 7. Development Workflow

### 7.1. Local Development

```bash
# Setup
pip install -r requirements.txt

# Development Dependencies (linting, formatting, type checking)
pip install -r requirements-dev.txt

# Run application
streamlit run app.py
# OR
make run

# Modern Code Quality Pipeline
make check          # Format + lint (recommended before commits)
make format         # Auto-format with ruff + black
make lint           # Check with ruff + mypy
make test           # Run full test suite

# Test specific component
python -m pytest tests/test_funnel_calculator_comprehensive.py::TestClass::test_method -v
```

### 7.2. Code Quality Standards (NEW)

```bash
# ✅ MODERN TOOLCHAIN (Fast & Comprehensive)
# Ruff: Ultra-fast linting and import sorting
# Black: Uncompromising code formatting
# Mypy: Static type checking

# Daily workflow commands
make format         # Auto-fix formatting issues
make lint          # Check for errors and type issues
make check         # Combined format + lint (pre-commit)
make test-fast     # Quick validation (~30 seconds)

# Pre-commit hooks (automatic quality gates)
pip install pre-commit
pre-commit install
# Now runs automatically on git commit

# CI/CD workflow simulation
make ci-check      # Full quality + test validation
```

### 7.3. Performance Profiling

```python
# Built-in performance monitoring
calculator = FunnelCalculator(config)
results = calculator.calculate_funnel_metrics(data, steps)

# View performance report
perf_report = calculator.get_performance_report()
bottlenecks = calculator.get_bottleneck_analysis()
```

---

## 🎯 8. Domain-Specific Knowledge

### 8.1. Funnel Analysis Concepts

**Counting Methods:**

- `UNIQUE_USERS`: Track distinct users through funnel (most common)
- `EVENT_TOTALS`: Count all events at each step (volume analysis)
- `UNIQUE_PAIRS`: Step-to-step conversion pairs (drop-off analysis)

**Funnel Orders:**

- `ORDERED`: Users must complete steps in sequence
- `UNORDERED`: Steps can be completed in any order

**Re-entry Modes:**

- `FIRST_ONLY`: Count users only on first funnel attempt
- `OPTIMIZED_REENTRY`: Allow multiple funnel attempts

### 8.2. Data Schema Requirements

```python
# Required columns for event data
events_df = pd.DataFrame({
    'user_id': str,        # Unique user identifier
    'event_name': str,     # Event type (funnel step)
    'timestamp': datetime, # When event occurred
    'event_properties': str,  # JSON string (optional)
    'user_properties': str    # JSON string (optional)
})
```

### 8.3. Performance Characteristics

- **Polars Advantages:** Large datasets (>100k events), simple aggregations
- **Polars Limitations:** Complex nested operations, certain join patterns
- **Fallback Triggers:** Schema conflicts, expression ambiguity, memory limits

---

## ⚠️ 9. Known Issues & Workarounds

### 9.1. Polars Fallback Scenarios

```python
# These combinations often trigger fallbacks:
problematic_configs = [
    (CountingMethod.UNIQUE_PAIRS, FunnelOrder.UNORDERED),
    (ReentryMode.OPTIMIZED_REENTRY, large_datasets),
]

# Always implement graceful fallback:
try:
    return self._polars_implementation(df)
except Exception as e:
    self.logger.warning(f"Falling back to Pandas: {e}")
    return self._pandas_implementation(df.to_pandas())
```

### 9.2. Memory Management

- **Large Datasets:** Use lazy evaluation, avoid collecting early
- **Polars Memory:** Monitor for OOM conditions, implement chunking
- **Streamlit Caching:** Clear cache when data changes

---

## 📝 **AGENT DISCOVERIES**

_(Auto-updated by Copilot based on task patterns)_

### Discovery Date: 2025-06-18

**Problem Pattern:** Duplicate test configuration files cause confusion and maintenance overhead
**Solution Found:** Consolidate to single source of truth for fixtures and test runners
**Code Pattern:**

```bash
# Remove duplicate conftest files, keep only the canonical one
rm tests/conftest_old.py tests/conftest_new.py
# Keep: tests/conftest.py (single source)

# Remove duplicate test runners, keep only the canonical one
rm run_tests_professional.py
# Keep: run_tests.py (comprehensive functionality)

# Fix pytest.ini markers to match actual test file markers
markers =
    basic: Basic functionality tests
    conversion_window: Conversion window tests
    counting_method: Counting method tests
    edge_case: Edge cases and boundary condition tests
    integration: Integration tests for end-to-end flows
    performance: Performance and scalability tests
    polars: Polars-specific functionality tests
```

**Add to Section:** 4 (Professional Testing Architecture)
**Search Keywords:** `duplicate test files consolidation conftest run_tests cleanup`

### Discovery Date: 2025-06-18

**Problem Pattern:** Test runner shows misleading status when pytest returns warnings in stderr
**Solution Found:** Separate pytest status reporting from generic command status
**Code Pattern:**

```python
# In run_command function - don't show immediate status for pytest
if result.returncode == 0:
    if "pytest" not in ' '.join(cmd):
        print(f"✅ {description or 'Command'} completed successfully")
else:
    if "pytest" in ' '.join(cmd):
        pass  # Let JSON report handler decide for pytest
    else:
        print(f"❌ {description or 'Command'} failed")

# In run_pytest function - show status after JSON analysis
if result['failed'] > 0:
    result['status'] = "FAILURE"
    print(f"❌ {description} failed")
else:
    result['status'] = "SUCCESS"
    print(f"✅ {description} completed successfully")
```

**Add to Section:** 4.3 (Running Tests Professional Commands)
**Search Keywords:** `pytest status reporting json test runner misleading status`

### Discovery Date: 2025-06-18

**Problem Pattern:** Time series calculations lack comprehensive UI accuracy testing and edge case coverage
**Solution Found:** Create mathematical precision test suite with boundary condition validation
**Code Pattern:**

```python
# Mathematical precision testing for time series
@pytest.mark.timeseries
def test_exact_cohort_calculation(calculator, controlled_data, funnel_steps):
    result = calculator.calculate_timeseries_metrics(data, steps, '1d')

    # Exact mathematical validation
    assert day1['started_funnel_users'] == 1000
    assert day1['completed_funnel_users'] == 300
    assert abs(day1['conversion_rate'] - 30.0) < 0.01

    # Funnel monotonicity validation
    step_counts = [row[f'{step}_users'] for step in funnel_steps]
    for j in range(1, len(step_counts)):
        assert step_counts[j] <= step_counts[j-1]

# Boundary condition testing with tolerance
def test_hourly_aggregation_accuracy(calculator, data):
    # Allow tolerance for hour boundary effects
    assert 48 <= len(result) <= 49  # Hour boundaries
    assert 350 <= total_starters <= 370  # Boundary tolerance
```

**Add to Section:** 4.2 (Critical Test Files)
**Search Keywords:** `time series mathematical precision boundary testing ui accuracy`

### Discovery Date: 2025-06-18

**Problem Pattern:** Time Series charts vertically stretched on different screens due to unbounded height calculation
**Solution Found:** Implement responsive height caps and optimized margins for dual-axis charts
**Code Pattern:**

```python
# Fixed responsive height calculation with caps
@staticmethod
def get_responsive_height(base_height: int, content_count: int = 1) -> int:
    """Calculate responsive height based on content and screen size with reasonable caps"""
    # Cap the content scaling to prevent excessive growth
    max_scaling_items = min(content_count - 1, 20)
    scaling_height = max_scaling_items * 20  # Reduced from 40 to 20 per item
    dynamic_height = base_height + scaling_height
    max_height = min(800, base_height * 1.6)  # Cap at 1.6x base or 800px max
    return max(400, min(dynamic_height, max_height))

# Optimized margins for time series dual-axis charts
margin=dict(l=60, r=60, t=80, b=100)  # Reduced by 20-25%

# Thinner range slider
rangeslider=dict(thickness=0.15)  # Reduce vertical footprint
```

**Add to Section:** 2.3 (Visualization & UI)
**Search Keywords:** `chart stretching vertical ui responsive height calculation margins range slider`

### Discovery Date: 2025-06-18

**Problem Pattern:** Inconsistent visualization sizing and poor responsive behavior across different chart types and screen sizes
**Solution Found:** Implement universal visualization standards with comprehensive testing and responsive design patterns
**Code Pattern:**

```python
# Universal visualization standards enforced across all charts
HEIGHT_STANDARDS = {
    'minimum': 350,    # Mobile usability minimum
    'maximum': 800,    # Prevent excessive stretching
    'optimal_range': (400, 600)  # Best viewing experience
}

# Responsive design patterns
layout_config = {
    'autosize': True,  # Enable responsive behavior
    'height': LayoutConfig.get_responsive_height(base_height, content_count),
    'margin': LayoutConfig.get_margins('md')  # Standardized margins
}

# Color palette accessibility
SEMANTIC = {
    'primary': '#3B82F6', 'secondary': '#6B7280',
    'success': '#10B981', 'warning': '#F59E0B', 'error': '#EF4444'
}

# Chart dimensions standardization
CHART_DIMENSIONS = {
    'small': {'width': 400, 'height': 350, 'ratio': 8/7},    # Mobile-friendly
    'medium': {'width': 600, 'height': 400, 'ratio': 3/2},   # Desktop standard
    'large': {'width': 800, 'height': 500, 'ratio': 8/5}     # Large screens
}
```

**Add to Section:** 2.3 (Visualization & UI)
**Search Keywords:** `universal visualization standards responsive design chart sizing accessibility mobile compatibility`

### Discovery Date: 2025-06-18

**Problem Pattern:** Legacy visualization tests with incomplete mock setup causing "Mock object not subscriptable" errors
**Solution Found:** Replace complex mock hierarchies with real object instances for visualization testing
**Code Pattern:**

```python
# ❌ BAD: Complex mock setup that breaks with real method calls
@pytest.fixture
def mock_visualizer(self):
    visualizer = Mock()
    visualizer.color_palette = ColorPalette()
    visualizer.create_timeseries_chart = FunnelVisualizer.create_timeseries_chart.__get__(visualizer)
    # Fails when real methods access nested mock attributes

# ✅ GOOD: Real object instance for integration testing
@pytest.fixture
def visualizer(self):
    return FunnelVisualizer(theme='dark', colorblind_friendly=False)

# Update test expectations to match actual responsive height calculation
@pytest.mark.parametrize("dataset_size,expected_height", [
    (10, 680),   # Actual: min(800, 500 + min(10-1, 20) * 20) = 680
    (30, 800),   # Actual: capped at 800px maximum
    (100, 800),  # Actual: capped at 800px maximum
])
```

**Add to Section:** 4.2 (Critical Test Files) + 9 (Known Issues)
**Search Keywords:** `mock object not subscriptable visualization tests real objects integration testing`

### Discovery Date: 2025-06-18

**Problem Pattern:** Time series conversion rates calculated incorrectly due to cohort misalignment
**Solution Found:** Implement true cohort analysis with individual conversion windows per user
**Code Pattern:**

```python
# ❌ OLD (WRONG): Cross-period misalignment
conversion_rate = completions_in_period_T / starts_in_period_T  # Mathematically incorrect

# ✅ NEW (CORRECT): True cohort analysis
# Step 1: Define cohorts by start period
cohorts_df = starter_times.with_columns([
    pl.col('start_time').dt.truncate(aggregation_period).alias('period_date')
])

# Step 2: Individual conversion windows
starters_with_deadline = starter_times.with_columns([
    (pl.col('start_time') + pl.duration(hours=conversion_window_hours)).alias('deadline')
])

# Step 3: Check completions within individual windows
step_matches = (
    starters_with_deadline
    .join(step_events, on='user_id', how='inner')
    .filter(
        (pl.col('timestamp') >= pl.col('start_time')) &
        (pl.col('timestamp') <= pl.col('deadline'))
    )
)
```

**Add to Section:** 8.1 (Domain Knowledge) + 9 (Known Issues)
**Search Keywords:** `time series cohort analysis conversion window individual periods cross-period`

**Impact:**

- ✅ Eliminates impossible >100% conversion rates
- ✅ Provides mathematically correct cohort performance metrics
- ✅ Enables accurate period-to-period comparisons
- ✅ Maintains Polars performance optimizations
