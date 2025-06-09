# Fallback Patterns Report

## Summary Statistics

### Component Fallback Statistics
| Component | Fallbacks | Total Tests | Fallback Rate |
|-----------|-----------|-------------|---------------|
| path_analysis | 12 | 12 | 100.0% |
| time_to_convert | 0 | 12 | 0.0% |
| cohort_analysis | 0 | 12 | 0.0% |

### Error Type Statistics
| Error Type | Occurrences |
|------------|-------------|
| nested_object_types | 12 |
| original_order | 6 |
| cross_join_keys | 3 |

## Detailed Fallback Report by Configuration
| Funnel Order | Reentry Mode | Counting Method | Component | Fallback | Error Types |
|-------------|-------------|----------------|-----------|----------|-------------|
| ordered | first_only | event_totals | path_analysis | ✓ Yes | nested_object_types, original_order |
| ordered | first_only | event_totals | time_to_convert | ✗ No |  |
| ordered | first_only | event_totals | cohort_analysis | ✗ No |  |
| ordered | first_only | unique_pairs | path_analysis | ✓ Yes | nested_object_types, original_order |
| ordered | first_only | unique_pairs | time_to_convert | ✗ No |  |
| ordered | first_only | unique_pairs | cohort_analysis | ✗ No |  |
| ordered | first_only | unique_users | path_analysis | ✓ Yes | nested_object_types, original_order |
| ordered | first_only | unique_users | time_to_convert | ✗ No |  |
| ordered | first_only | unique_users | cohort_analysis | ✗ No |  |
| ordered | optimized_reentry | event_totals | path_analysis | ✓ Yes | nested_object_types |
| ordered | optimized_reentry | event_totals | time_to_convert | ✗ No |  |
| ordered | optimized_reentry | event_totals | cohort_analysis | ✗ No |  |
| ordered | optimized_reentry | unique_pairs | path_analysis | ✓ Yes | nested_object_types |
| ordered | optimized_reentry | unique_pairs | time_to_convert | ✗ No |  |
| ordered | optimized_reentry | unique_pairs | cohort_analysis | ✗ No |  |
| ordered | optimized_reentry | unique_users | path_analysis | ✓ Yes | nested_object_types |
| ordered | optimized_reentry | unique_users | time_to_convert | ✗ No |  |
| ordered | optimized_reentry | unique_users | cohort_analysis | ✗ No |  |
| unordered | first_only | event_totals | path_analysis | ✓ Yes | nested_object_types, original_order |
| unordered | first_only | event_totals | time_to_convert | ✗ No |  |
| unordered | first_only | event_totals | cohort_analysis | ✗ No |  |
| unordered | first_only | unique_pairs | path_analysis | ✓ Yes | nested_object_types, original_order |
| unordered | first_only | unique_pairs | time_to_convert | ✗ No |  |
| unordered | first_only | unique_pairs | cohort_analysis | ✗ No |  |
| unordered | first_only | unique_users | path_analysis | ✓ Yes | nested_object_types, original_order |
| unordered | first_only | unique_users | time_to_convert | ✗ No |  |
| unordered | first_only | unique_users | cohort_analysis | ✗ No |  |
| unordered | optimized_reentry | event_totals | path_analysis | ✓ Yes | cross_join_keys, nested_object_types |
| unordered | optimized_reentry | event_totals | time_to_convert | ✗ No |  |
| unordered | optimized_reentry | event_totals | cohort_analysis | ✗ No |  |
| unordered | optimized_reentry | unique_pairs | path_analysis | ✓ Yes | cross_join_keys, nested_object_types |
| unordered | optimized_reentry | unique_pairs | time_to_convert | ✗ No |  |
| unordered | optimized_reentry | unique_pairs | cohort_analysis | ✗ No |  |
| unordered | optimized_reentry | unique_users | path_analysis | ✓ Yes | cross_join_keys, nested_object_types |
| unordered | optimized_reentry | unique_users | time_to_convert | ✗ No |  |
| unordered | optimized_reentry | unique_users | cohort_analysis | ✗ No |  |

## Recommendations

### 1. Fix Nested Object Types Issue
This is the most common issue, occurring in all path_analysis calculations.
Recommendation: Add `strict=False` parameter to relevant Polars operations or properly handle object types.
```python
# Example fix:
df = pl.from_pandas(pandas_df, strict=False)
```

### 2. Fix Polars Join Issues
Issues with joins are causing fallbacks from standard Polars to Pandas.
Recommendation: Review join operations and ensure proper column types and join conditions.