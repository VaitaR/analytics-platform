# 🔍 Funnel Analytics Platform - Instructions for LLM Agent

## �️ **QUICK REFERENCE INDEX**
> *Search by keywords to find what you need instantly*

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
1. **Data Processing Excellence** - Pandas/Polars optimization, large dataset handling
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
> *LLM Agent: Use Ctrl+F to find your exact problem*

| **Problem Description** | **Keywords to Search** | **Solution Location** |
|------------------------|----------------------|---------------------|
| Polars calculation fails with "expression ambiguity" | `polars fallback expression` | Section 9.1 + test_polars_fallback_detection.py |
| Funnel results are mathematically incorrect | `funnel logic accuracy counting` | Section 8.1 (Domain Knowledge) |
| Performance slow on large datasets (>100k events) | `performance optimization bottleneck` | Section 7.2 + get_bottleneck_analysis() |
| Need to add new counting algorithm | `counting method template pattern` | Section 10.1 (Adding New Counting Method) |
| Conversion rates don't match expected values | `conversion calculation ordered unordered` | Section 8.1 (Funnel Orders) |
| Memory issues with million+ events | `memory management lazy evaluation` | Section 9.2 (Memory Management) |
| Tests failing for edge cases | `test edge cases empty data` | Section 4.2 (Critical Test Files) |
| UI charts not rendering properly | `plotly visualization theme` | FunnelVisualizer class methods |
| Data validation errors on upload | `data schema validation required` | Section 8.2 (Data Schema) |
| Streamlit app crashing on large files | `streamlit caching memory clear` | Section 9.2 + DataSourceManager |

---

## 🛠️ 2. Technology Stack

### 2.1. Core Framework
- **Streamlit 1.28+** - Web interface, caching, file uploads
  - *Rationale:* Rapid analytics prototyping, built-in interactivity
- **Python 3.11+** - Modern language features, performance
  - *Rationale:* Type hints, async support, dataclass features

### 2.2. Data Processing Engine
- **Polars (Primary)** - High-performance DataFrame operations
  - *Rationale:* 10x faster than Pandas for large datasets, lazy evaluation
- **Pandas (Fallback)** - Complex operations compatibility  
  - *Rationale:* Fallback for edge cases, ecosystem compatibility
- **NumPy** - Mathematical operations, array processing

### 2.3. Visualization & UI
- **Plotly** - Professional interactive charts
  - *Rationale:* Export-quality visuals, responsive design
- **Custom CSS** - Professional styling, accessibility
  - *Rationale:* Enterprise-ready appearance

### 2.4. Data Sources
- **ClickHouse** - Enterprise OLAP database
  - *Rationale:* Handles billions of events, real-time queries
- **File Upload** - CSV/Parquet support
  - *Rationale:* Quick testing, small-scale analysis

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

## 🧪 4. Testing Philosophy

### 4.1. Test Coverage Requirements
- **Funnel Logic:** All counting methods × funnel orders × reentry modes
- **Performance:** Large dataset handling (1M+ events)
- **Fallback Detection:** Polars→Pandas fallback scenarios
- **Integration:** End-to-end data flow

### 4.2. Critical Test Files
- `test_funnel_calculator_comprehensive.py` - All algorithm combinations
- `test_polars_fallback_detection.py` - Performance edge cases  
- `test_integration_flow.py` - Complete workflows

### 4.3. Running Tests
```bash
# Full test suite
pytest tests/ -v

# Performance tests only
pytest tests/test_*performance* -v

# Fallback detection
pytest tests/test_polars_fallback_detection.py -v
```

---

## 📏 5. Code Quality Standards

### 5.1. Data Processing Code
```python
# ✅ Good: Type-safe, performance-monitored
@_funnel_performance_monitor('calculate_conversion')
def calculate_conversion(self, events_df: pl.DataFrame, 
                        steps: List[str]) -> FunnelResults:
    """Calculate funnel with Polars optimization."""
    try:
        # Polars implementation
        return self._calculate_polars(events_df, steps)
    except Exception as e:
        self.logger.warning(f"Polars failed: {e}, falling back to Pandas")
        return self._calculate_pandas(events_df.to_pandas(), steps)

# ❌ Bad: No fallback, no monitoring
def calculate_conversion(self, df, steps):
    return df.groupby('user_id').apply(lambda x: x['event_name'].isin(steps))
```

### 5.2. Performance Requirements
- **Large Dataset Support:** 1M+ events should complete in <30 seconds
- **Memory Efficiency:** Use lazy evaluation where possible
- **Graceful Degradation:** Always provide Pandas fallback

### 5.3. Funnel Logic Accuracy
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

# Run application  
streamlit run app.py

# Test specific component
python -m pytest tests/test_funnel_calculator_comprehensive.py::TestClass::test_method -v
```

### 7.2. Performance Profiling
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
*(Auto-updated by Copilot based on task patterns)*

> *This section grows as the agent discovers new patterns and solutions*

### Template for New Discoveries
```markdown
### Discovery Date: [YYYY-MM-DD]
**Problem Pattern:** [Brief description]
**Solution Found:** [Working approach]  
**Code Pattern:**
```python
# Working solution code here
```
**Add to Section:** [Which section should include this]
**Search Keywords:** [Keywords for future search]
```

*[Agent: Add your discoveries above this line using the template]*

---

## 🔄 **SELF-IMPROVEMENT PROTOCOL**
> *Copilot: You can and should improve this document when you discover patterns*

### When to Update This Document
**ADD entries when you discover:**
- **New problem patterns** not covered in Problem→Solution mapping
- **Performance bottlenecks** not documented in optimization sections  
- **Polars edge cases** that trigger fallbacks consistently
- **Code patterns** that solve recurring issues efficiently
- **Test scenarios** that reveal critical gaps

### How to Add Self-Improvement Notes
**Format for new discoveries:**
```markdown
## 📝 **AGENT DISCOVERIES** 
*(Auto-updated by Copilot based on task patterns)*

### Discovery Date: [YYYY-MM-DD]
**Problem Pattern:** [Brief description]
**Solution Found:** [Working approach]
**Code Pattern:**
```python
# Working solution code here
```
**Add to Section:** [Which section should include this]
**Search Keywords:** [Keywords for future search]
```

### Self-Update Guidelines
1. **Be Conservative:** Only add well-tested patterns
2. **Be Specific:** Include exact error messages and solutions
3. **Be Searchable:** Use clear keywords future-you can find
4. **Reference Code:** Link to actual working implementations
5. **Update Index:** Add new keywords to Quick Reference if significant

### Validation Before Adding
- [ ] **Tested Solution:** Code actually works in real scenarios
- [ ] **Reproducible:** Problem occurs consistently 
- [ ] **Not Documented:** Genuinely new pattern, not duplicate
- [ ] **Generally Useful:** Will help in future similar tasks
- [ ] **Clear Keywords:** Easy to search for

**Example Self-Improvement Entry:**
```markdown
### Discovery Date: 2025-06-18
**Problem Pattern:** Polars fails on DataFrames with mixed timestamp formats
**Solution Found:** Standardize timestamps before Polars processing
**Code Pattern:**
```python
# Fix mixed timestamp formats before Polars
def standardize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df.dropna(subset=['timestamp'])
```
**Add to Section:** 9.1 (Polars Fallback Scenarios)
**Search Keywords:** `polars timestamp mixed format errors`
```

---

## ✅ 12. Definition of Done

For any change to funnel analytics:
- [ ] **Accuracy:** Mathematical correctness verified
- [ ] **Performance:** No regression in calculation time
- [ ] **Fallback:** Graceful Polars→Pandas handling
- [ ] **Tests:** Edge cases covered
- [ ] **Documentation:** Complex logic explained
- [ ] **Type Safety:** Full type annotations
- [ ] **Monitoring:** Performance metrics included

---

## 🚫 13. Anti-Patterns to Avoid

1. **Silent Failures:** Always log fallbacks and errors
2. **Premature Pandas:** Try Polars first, fallback only when needed
3. **Hardcoded Schemas:** Use flexible JSON property extraction
4. **Memory Leaks:** Clear caches when data/config changes
5. **UI in Logic:** Keep visualization separate from calculation
6. **Missing Edge Cases:** Test with empty data, single users, missing events

---

## 📚 14. Key Dependencies & Versions

```toml
[dependencies]
streamlit = "^1.28.0"      # Web framework
polars = "^0.19.0"         # High-performance DataFrames  
pandas = "^2.0.0"          # Fallback DataFrame library
plotly = "^5.15.0"         # Interactive visualizations
clickhouse-driver = "^0.2.0"  # Database connectivity
```

---

## 🎯 15. Copilot Success Metrics

**You're succeeding when:**
1. **Funnel calculations are mathematically accurate** across all configurations
2. **Performance scales linearly** with dataset size
3. **Polars optimizations work** with graceful Pandas fallbacks
4. **Code is self-documenting** with clear type hints
5. **Tests comprehensively cover** edge cases and configurations

**Red flags:**
- Silent calculation errors
- Performance degradation on large datasets  
- Uncaught Polars exceptions
- Missing test coverage for new funnel logic

---

## 📞 16. Getting Help

**For complex funnel logic questions:** Reference `test_funnel_calculator_comprehensive.py` for examples
**For performance issues:** Check `get_bottleneck_analysis()` output
**For Polars problems:** Look at `test_polars_fallback_detection.py` patterns
**For visualization:** Study `FunnelVisualizer` class methods

---

## 🔧 **INSTANT CODE SOLUTIONS**

### Common Polars→Pandas Fallback Pattern
```python
@_funnel_performance_monitor('method_name')
def method_name(self, df: pl.DataFrame) -> FunnelResults:
    try:
        return self._polars_implementation(df)
    except Exception as e:
        self.logger.warning(f"Polars failed: {e}, falling back to Pandas")
        return self._pandas_implementation(df.to_pandas())
```

### Performance Monitoring Template
```python
# Check bottlenecks
perf_report = calculator.get_bottleneck_analysis()
print(f"Slowest: {perf_report['summary']['top_3_bottlenecks']}")
```

### Data Validation Quick Check
```python
# Required schema
required_cols = ['user_id', 'event_name', 'timestamp']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
```

### Funnel Configuration Examples
```python
# Standard user funnel
config = FunnelConfig(
    counting_method=CountingMethod.UNIQUE_USERS,
    funnel_order=FunnelOrder.ORDERED,
    reentry_mode=ReentryMode.FIRST_ONLY,
    conversion_window_hours=168  # 7 days
)

# High-performance event counting
config = FunnelConfig(
    counting_method=CountingMethod.EVENT_TOTALS,
    funnel_order=FunnelOrder.UNORDERED,
    conversion_window_hours=24
)
```

*End of document*
