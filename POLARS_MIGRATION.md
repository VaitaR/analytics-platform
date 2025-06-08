# Polars Migration Documentation

## Overview

This document outlines the incremental migration strategy from Pandas to Polars for the Professional Funnel Analytics Platform. The migration follows a careful, bridge-based approach to maintain functionality while improving performance.

## Migration Strategy

### Phase 1: Core Infrastructure (COMPLETED)

1. **Bridge System**: Added conversion utilities between Pandas and Polars DataFrames
2. **Dual Engine Support**: Modified `FunnelCalculator` to support both Pandas and Polars engines
3. **Main Entry Point**: Updated `calculate_funnel_metrics` to route between engines
4. **Fallback Mechanism**: Automatic fallback to Pandas if Polars implementation fails

### Phase 2: Core Functions (CURRENT)

1. **Preprocessing**: Implemented `_preprocess_data_polars` with Polars optimizations
2. **Unique Users Funnel**: Created `_calculate_unique_users_funnel_polars` 
3. **Conversion Detection**: Added Polars-based conversion logic

### Current Implementation Status

#### ✅ Completed
- Polars dependency added to requirements.txt
- Bridge utilities for DataFrame conversion
- Polars preprocessing implementation
- Polars unique users funnel calculation
- UI toggle for engine selection
- Performance monitoring with engine tracking

#### 🚧 In Progress
- Core funnel calculation with Polars bridges
- Comprehensive error handling and testing

#### ⏳ Next Steps (Future Iterations)
- Unique pairs funnel calculation in Polars
- Event totals funnel calculation in Polars
- Unordered funnel calculation in Polars
- Segmentation logic in Polars
- Time-to-convert analysis in Polars
- Cohort analysis in Polars
- Path analysis in Polars
- Statistical significance testing in Polars

## Code Structure

### Bridge Pattern
```python
# Entry point with automatic routing
def calculate_funnel_metrics(self, events_df: pd.DataFrame, funnel_steps: List[str]) -> FunnelResults:
    if self.use_polars:
        polars_df = self._to_polars(events_df)
        return self._calculate_funnel_metrics_polars(polars_df, funnel_steps, events_df)
    else:
        return self._calculate_funnel_metrics_pandas(events_df, funnel_steps)
```

### Polars Implementation
```python
@_funnel_performance_monitor('_calculate_unique_users_funnel_polars')
def _calculate_unique_users_funnel_polars(self, events_df: pl.DataFrame, steps: List[str]) -> FunnelResults:
    # Polars-optimized implementation
    pass
```

### Conversion Utilities
```python
def _to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
    # Handle datetime conversion explicitly
    # Convert with proper schema handling
    
def _to_pandas(self, df: pl.DataFrame) -> pd.DataFrame:
    # Convert back for compatibility
```

## Performance Benefits

### Expected Improvements
1. **Memory Efficiency**: Polars uses less memory for large datasets
2. **Parallelization**: Automatic parallel processing for operations
3. **Columnar Processing**: Optimized for analytical workloads
4. **Type Safety**: Better handling of data types and null values

### Monitoring
- Performance tracking shows engine used (Polars vs Pandas)
- Separate metrics for `calculate_funnel_metrics_polars` and `calculate_funnel_metrics_pandas`
- Eliminated duplicate monitoring of router function to avoid double-counting
- Bottleneck analysis identifies optimization opportunities
- Automatic fallback ensures reliability

### Issue Fixes

#### Data Type Compatibility (FIXED)
**Problem**: `cannot compare string with numeric type (f64)`
- **Root Cause**: Polars enforces stricter type checking than Pandas
- **Solution**: 
  - Added proper type conversion in `_to_polars()` to ensure `user_id` is string type
  - Fixed `_check_conversion_polars()` to use `pl.duration()` instead of Python `timedelta()`
  - Ensured all timestamp comparisons use compatible Polars types
- **Result**: Polars engine now handles mixed data types correctly

#### Expression Ambiguity (FIXED) 
**Problem**: `the truth value of an Expr is ambiguous`
- **Root Cause**: Using Python `and` operator with Polars expressions in list comprehensions
- **Solution**:
  - Replaced `ct > prev_time and (ct - prev_time) < conversion_window_polars` with explicit if-else logic
  - Fixed deprecated `with_row_count()` → `with_row_index()`
  - Avoided Python operators (`and`, `or`, `in`) with Polars expressions in all cases
- **Result**: Polars engine now runs without expression evaluation errors

## Python timedelta vs Polars Duration Issue Fixed

One of the key issues we encountered during the migration was handling time differences between events. 
Polars uses its own `Duration` type which has a `.nanoseconds()` method, while Python's standard `timedelta` 
objects do not have this method, causing errors when the code attempted to call `.nanoseconds()` on Python timedelta objects.

### Problem
When calculating conversion windows and time differences between events, the code was failing with:
```
'datetime.timedelta' object has no attribute 'nanoseconds'
```

This occurred primarily in the following situations:
- When handling zero or very small conversion windows
- When checking if events were within the conversion window boundary
- At the boundary condition where we need to check if an event is exactly at the window limit

### Solution
- Added a helper method `_to_nanoseconds()` that works with both Polars Duration and Python timedelta objects
- Made all time difference calculations use this helper method
- Added proper error handling with logging
- Created comprehensive tests for edge cases around conversion windows

This ensures that both engines produce consistent results and can handle all conversion window configurations.

# Fixed in commit Mon Jun 10 12:31:37 EDT 2024

## Usage

### User Control
Users can toggle between engines via the UI:
```
☑️ Use Polars Engine (experimental)
```

### Configuration
```python
# Default: Use Polars
calculator = FunnelCalculator(config, use_polars=True)

# Force Pandas
calculator = FunnelCalculator(config, use_polars=False)
```

## Migration Rules

1. **Never break existing functionality**: Always maintain Pandas fallback
2. **One function at a time**: Migrate individual methods incrementally
3. **Test thoroughly**: Compare results between engines
4. **Bridge conversions**: Use conversion points strategically
5. **Performance monitor**: Track improvements and regressions

## Testing Strategy

### Validation Approach
1. Run identical datasets through both engines
2. Compare results for accuracy
3. Measure performance improvements
4. Test edge cases and error conditions

### Test Categories
- Basic funnel calculations
- Complex segmentation scenarios
- Large dataset performance
- Error handling and fallbacks

## Known Limitations

### Current Constraints
1. Some advanced features still use Pandas (segmentation, path analysis)
2. JSON property expansion not yet implemented in Polars
3. Statistical tests still use Pandas implementations

### Compatibility
- All existing functionality preserved
- Same API interface maintained
- Same output format (FunnelResults)

## Future Roadmap

### Next Iterations
1. **Iteration 2**: Migrate unique pairs and event totals calculations
2. **Iteration 3**: Implement Polars segmentation
3. **Iteration 4**: Convert advanced analytics (cohort, path analysis)
4. **Iteration 5**: Optimize JSON property handling
5. **Iteration 6**: Full Polars migration with minimal bridges

## Testing Status

✅ **Data Type Compatibility**: Verified with mixed string/int user_id types  
✅ **Polars vs Pandas Consistency**: Results match exactly  
✅ **Error Handling**: Proper fallback to Pandas when needed  
✅ **Performance Monitoring**: Separate tracking for each engine  

### Performance Targets
- 2-5x improvement in processing speed for large datasets
- 30-50% reduction in memory usage
- Better scalability for enterprise workloads

## Next Steps

- [ ] **Phase 3**: Implement remaining functions in Polars
  - [ ] `_calculate_cohort_analysis_polars`
  - [ ] `_calculate_path_analysis_polars`
  - [ ] `_calculate_time_to_convert_polars`
- [ ] **Phase 4**: JSON property expansion optimization
- [ ] **Phase 5**: Caching optimization for Polars
- [ ] **Phase 6**: Performance benchmarking and fine-tuning 

## Recent Fixes (June 2025)

The following issues were identified and fixed in the Polars implementation:

1. **Method Name Mismatches**:
   - Changed `with_column()` to `with_columns()` to match the correct Polars API
   - Changed `sort_by()` to `sort()` to match the correct Polars API
   - Changed `list.lengths()` to `list.len()` to match the correct Polars API (previously fixed)

2. **Conversion Window Handling**:
   - Fixed the conversion window boundary handling to make it exclusive (events exactly at the boundary should not be included)
   - Added special handling for zero conversion window cases (only exact timestamp matches should be included)
   - Ensured consistent behavior between Pandas and Polars implementations

3. **Out-of-Order Events**:
   - Added a check in the Polars implementation to properly handle out-of-order events in ordered funnels
   - Implemented `_user_did_later_steps_before_current_polars` function to check for events that happen out of sequence

These fixes ensure that the Polars implementation produces identical results to the Pandas implementation, allowing for a smooth migration path while maintaining backward compatibility with existing analytics. # Fixed in commit Sun Jun  8 17:02:49 +03 2025
