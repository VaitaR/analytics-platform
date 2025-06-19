# Test Data Directory

This directory contains all test data files used by the Funnel Analytics Platform test suite.

## 📁 **File Organization**

### **Performance Test Data**
- `test_50k.csv` - Medium dataset with 50,000 events for performance testing
- `test_200k.csv` - Large dataset with 200,000 events for scalability testing

### **Demo & Sample Data**
- `demo_events.csv` - Demo events used in the main application interface
- `sample_events.csv` - Sample event data for testing basic scenarios
- `sample_funnel.csv` - Sample funnel data for validation

### **Analysis Test Data**
- `ab_test_data.csv` - A/B test scenario data
- `ab_test_rates.csv` - A/B test conversion rates
- `segment_data.csv` - User segmentation test data
- `time_series_data.csv` - Time-based analysis test data
- `time_to_convert_stats.csv` - Conversion timing analysis data

## 🔧 **Usage in Tests**

All test files should reference data files using relative paths from the project root:

```python
# ✅ Correct usage
df = pd.read_csv('test_data/demo_events.csv')

# ❌ Incorrect - old path
df = pd.read_csv('demo_events.csv')
```

## 📊 **Data Format Standards**

All CSV files follow the standard event schema:
```
user_id,event_name,timestamp,event_properties,user_properties
```

## 🧪 **Test Data Generation**

For programmatic test data generation, use the fixtures in `tests/conftest.py`:
- `small_linear_funnel_data` - Small dataset (100 users)
- `medium_linear_funnel_data` - Medium dataset (1000 users)
- `large_dataset` - Large dataset (10000+ users)

## 📏 **Performance Benchmarks**

| File | Events | Users | Purpose |
|------|--------|-------|---------|
| `test_50k.csv` | 50,000 | ~5,000 | Medium performance testing |
| `test_200k.csv` | 200,000 | ~20,000 | Large scale performance testing |

## 🔄 **Maintenance**

When adding new test data files:
1. Place in this `test_data/` directory
2. Update this README.md
3. Use appropriate naming convention: `test_<purpose>_<size>.csv`
4. Update any tests to use `test_data/` prefix
