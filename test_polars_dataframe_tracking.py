#!/usr/bin/env python3
"""
Test to track DataFrame types passed to each function in performance monitor.
This test verifies that Polars functions receive Polars DataFrames and 
identifies any unexpected conversions to Pandas.
"""

import pandas as pd
import polars as pl
import time
from datetime import datetime, timedelta
from core import FunnelCalculator
from models import FunnelConfig, CountingMethod, FunnelOrder, ReentryMode

# Dictionary to track function calls and their DataFrame types
function_calls = {}

def track_dataframe_types(original_monitor):
    """Decorator to wrap the performance monitor and track DataFrame types"""
    def wrapper(func_name):
        def decorator(func):
            def inner(self, *args, **kwargs):
                # Track the types of DataFrame arguments
                arg_types = []
                for i, arg in enumerate(args):
                    if isinstance(arg, pd.DataFrame):
                        arg_types.append(f"arg_{i}: pandas.DataFrame({len(arg)} rows)")
                    elif isinstance(arg, pl.DataFrame):
                        arg_types.append(f"arg_{i}: polars.DataFrame({len(arg)} rows)")
                    elif isinstance(arg, pl.LazyFrame):
                        arg_types.append(f"arg_{i}: polars.LazyFrame")
                    elif hasattr(arg, '__len__') and not isinstance(arg, str):
                        arg_types.append(f"arg_{i}: {type(arg).__name__}({len(arg)} items)")
                    else:
                        arg_types.append(f"arg_{i}: {type(arg).__name__}")
                
                # Store the call information
                if func_name not in function_calls:
                    function_calls[func_name] = []
                
                function_calls[func_name].append({
                    'timestamp': datetime.now(),
                    'arg_types': arg_types,
                    'function': func.__name__
                })
                
                # Call the original function
                start_time = time.time()
                result = func(self, *args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update with execution time
                function_calls[func_name][-1]['execution_time'] = execution_time
                
                return result
            return inner
        return decorator
    return wrapper

def monkey_patch_performance_monitor():
    """Monkey patch the performance monitor to track DataFrame types"""
    import core.calculator
    
    # Store the original monitor
    original_monitor = core.calculator._funnel_performance_monitor
    
    # Replace with our tracking version
    core.calculator._funnel_performance_monitor = track_dataframe_types(original_monitor)

def create_test_data():
    """Create test data for funnel analysis"""
    print("Creating test data...")
    
    events = []
    base_date = datetime(2024, 1, 1)
    
    # Create events for 100 users
    for user_id in range(100):
        # Each user starts with signup
        signup_date = base_date + timedelta(days=user_id % 30, hours=user_id % 24)
        events.append({
            'user_id': f'user_{user_id}',
            'event_name': 'signup',
            'timestamp': signup_date,
            'event_properties': '{}',
            'user_properties': '{}'
        })
        
        # 80% proceed to login
        if user_id < 80:
            login_date = signup_date + timedelta(hours=1 + (user_id % 12))
            events.append({
                'user_id': f'user_{user_id}',
                'event_name': 'login',
                'timestamp': login_date,
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            # 60% proceed to purchase
            if user_id < 60:
                purchase_date = login_date + timedelta(hours=2 + (user_id % 24))
                events.append({
                    'user_id': f'user_{user_id}',
                    'event_name': 'purchase',
                    'timestamp': purchase_date,
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # 40% complete the funnel
                if user_id < 40:
                    complete_date = purchase_date + timedelta(hours=1 + (user_id % 6))
                    events.append({
                        'user_id': f'user_{user_id}',
                        'event_name': 'complete',
                        'timestamp': complete_date,
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
    
    df = pd.DataFrame(events)
    print(f"Created {len(df)} events for {df['user_id'].nunique()} users")
    return df

def test_polars_dataframe_tracking():
    """Test to track DataFrame types in all performance monitored functions"""
    print("=== Polars DataFrame Type Tracking Test ===\n")
    
    # Monkey patch the performance monitor
    monkey_patch_performance_monitor()
    
    # Create test data
    events_df = create_test_data()
    funnel_steps = ['signup', 'login', 'purchase', 'complete']
    
    # Create Polars calculator
    config = FunnelConfig(
        counting_method=CountingMethod.UNIQUE_USERS,
        funnel_order=FunnelOrder.ORDERED,
        reentry_mode=ReentryMode.FIRST_ONLY,
        conversion_window_hours=24 * 7  # 7 days
    )
    calculator = FunnelCalculator(config, use_polars=True)
    
    print("Running funnel analysis with Polars engine...")
    start_time = time.time()
    
    # Run the analysis
    results = calculator.calculate_funnel_metrics(events_df, funnel_steps)
    
    total_time = time.time() - start_time
    print(f"Analysis completed in {total_time:.4f} seconds\n")
    
    # Analyze the function calls
    print("=== DataFrame Type Analysis ===\n")
    
    polars_functions = []
    pandas_functions = []
    mixed_functions = []
    
    for func_name, calls in function_calls.items():
        if not calls:
            continue
            
        # Analyze the types used in this function
        pandas_calls = 0
        polars_calls = 0
        
        for call in calls:
            has_pandas = any('pandas.DataFrame' in arg_type for arg_type in call['arg_types'])
            has_polars = any('polars.DataFrame' in arg_type or 'polars.LazyFrame' in arg_type for arg_type in call['arg_types'])
            
            if has_pandas:
                pandas_calls += 1
            if has_polars:
                polars_calls += 1
        
        # Categorize the function
        if polars_calls > 0 and pandas_calls == 0:
            polars_functions.append((func_name, calls))
        elif pandas_calls > 0 and polars_calls == 0:
            pandas_functions.append((func_name, calls))
        elif polars_calls > 0 and pandas_calls > 0:
            mixed_functions.append((func_name, calls))
    
    # Report results
    print(f"📊 **Function Classification:**")
    print(f"✅ Pure Polars functions: {len(polars_functions)}")
    print(f"🔄 Pure Pandas functions: {len(pandas_functions)}")
    print(f"⚠️  Mixed type functions: {len(mixed_functions)}")
    print()
    
    # Detailed analysis
    if polars_functions:
        print("✅ **Pure Polars Functions:**")
        for func_name, calls in polars_functions:
            total_time = sum(call['execution_time'] for call in calls)
            print(f"  • {func_name}: {len(calls)} calls, {total_time:.4f}s total")
        print()
    
    if pandas_functions:
        print("🔄 **Pure Pandas Functions:**")
        for func_name, calls in pandas_functions:
            total_time = sum(call['execution_time'] for call in calls)
            print(f"  • {func_name}: {len(calls)} calls, {total_time:.4f}s total")
        print()
    
    if mixed_functions:
        print("⚠️  **Mixed Type Functions (Potential Issues):**")
        for func_name, calls in mixed_functions:
            total_time = sum(call['execution_time'] for call in calls)
            print(f"  • {func_name}: {len(calls)} calls, {total_time:.4f}s total")
            
            # Show details for mixed functions
            for i, call in enumerate(calls):
                print(f"    Call {i+1}: {', '.join(call['arg_types'])}")
        print()
    
    # Check for unexpected Pandas usage in Polars functions
    suspicious_functions = []
    for func_name, calls in function_calls.items():
        # Functions that should be using Polars based on their names
        if ('_polars' in func_name or func_name.startswith('_calculate_') and 'pandas' not in func_name):
            for call in calls:
                has_pandas = any('pandas.DataFrame' in arg_type for arg_type in call['arg_types'])
                if has_pandas:
                    suspicious_functions.append((func_name, call))
    
    if suspicious_functions:
        print("🚨 **Suspicious Pandas Usage in Polars Functions:**")
        for func_name, call in suspicious_functions:
            print(f"  • {func_name}: {', '.join(call['arg_types'])}")
        print()
    
    # Performance comparison
    print("⚡ **Performance Summary:**")
    polars_total_time = sum(
        sum(call['execution_time'] for call in calls)
        for func_name, calls in polars_functions
    )
    pandas_total_time = sum(
        sum(call['execution_time'] for call in calls)
        for func_name, calls in pandas_functions
    )
    
    print(f"  • Polars functions total time: {polars_total_time:.4f}s")
    print(f"  • Pandas functions total time: {pandas_total_time:.4f}s")
    
    if polars_total_time > 0 and pandas_total_time > 0:
        ratio = pandas_total_time / polars_total_time
        print(f"  • Pandas/Polars time ratio: {ratio:.2f}x")
    print()
    
    # Expected vs actual function usage
    expected_polars_functions = [
        '_calculate_funnel_metrics_polars',
        '_preprocess_data_polars', 
        '_calculate_unique_users_funnel_polars',
        '_calculate_timeseries_metrics_polars',
        '_calculate_cohort_analysis_polars',
        '_calculate_path_analysis_polars',
        'segment_events_data_polars'
    ]
    
    actually_called_polars = [func_name for func_name, _ in polars_functions]
    
    print("🎯 **Expected vs Actual Polars Function Usage:**")
    for expected_func in expected_polars_functions:
        if expected_func in actually_called_polars:
            print(f"  ✅ {expected_func}: Called with Polars data")
        elif expected_func in [func_name for func_name, _ in pandas_functions]:
            print(f"  ❌ {expected_func}: Called with Pandas data (unexpected!)")
        elif expected_func in [func_name for func_name, _ in mixed_functions]:
            print(f"  ⚠️  {expected_func}: Called with mixed data types")
        else:
            print(f"  ❓ {expected_func}: Not called")
    print()
    
    # Final verdict
    issues_found = len(suspicious_functions) + len(mixed_functions)
    if issues_found == 0:
        print("🎉 **VERDICT: EXCELLENT!** All Polars functions are receiving Polars DataFrames")
        print("   No unexpected conversions to Pandas detected.")
    else:
        print(f"⚠️  **VERDICT: {issues_found} ISSUES FOUND**")
        print("   Some Polars functions are receiving Pandas DataFrames or mixed types.")
        print("   This may indicate unnecessary conversions affecting performance.")
    
    return {
        'polars_functions': len(polars_functions),
        'pandas_functions': len(pandas_functions), 
        'mixed_functions': len(mixed_functions),
        'suspicious_functions': len(suspicious_functions),
        'polars_time': polars_total_time,
        'pandas_time': pandas_total_time,
        'total_functions': len(function_calls)
    }

if __name__ == "__main__":
    results = test_polars_dataframe_tracking()
    
    print(f"\n📋 **Summary:**")
    print(f"Total functions monitored: {results['total_functions']}")
    print(f"Pure Polars functions: {results['polars_functions']}")
    print(f"Pure Pandas functions: {results['pandas_functions']}")
    print(f"Mixed type functions: {results['mixed_functions']}")
    print(f"Suspicious Pandas usage: {results['suspicious_functions']}")
    
    if results['suspicious_functions'] == 0 and results['mixed_functions'] == 0:
        print("\n✅ All systems optimal - pure Polars execution achieved!")
    else:
        print(f"\n⚠️  Optimization opportunities found - check mixed/suspicious functions") 