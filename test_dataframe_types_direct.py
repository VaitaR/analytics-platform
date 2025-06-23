#!/usr/bin/env python3
"""
Direct test to analyze DataFrame types through existing performance monitor.
This test uses the existing infrastructure to verify data flow.
"""

import pandas as pd
import polars as pl
import time
import logging
import io
from datetime import datetime, timedelta
from core import FunnelCalculator
from models import FunnelConfig, CountingMethod, FunnelOrder, ReentryMode

def setup_logging():
    """Setup logging to capture detailed execution info"""
    log_capture = io.StringIO()
    
    # Get the logger
    logger = logging.getLogger('core.calculator')
    logger.setLevel(logging.DEBUG)
    
    # Create handler
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return log_capture, handler

def analyze_function_execution(calculator):
    """Analyze which functions were executed and their performance"""
    if not hasattr(calculator, '_performance_metrics'):
        return {}
    
    metrics = calculator._performance_metrics
    analysis = {}
    
    for func_name, times in metrics.items():
        if times:  # Only include functions that were actually called
            analysis[func_name] = {
                'calls': len(times),
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
    
    return analysis

def create_test_data():
    """Create test data for analysis"""
    print("Creating test data...")
    
    events = []
    base_date = datetime(2024, 1, 1)
    
    # Create events for 50 users (smaller dataset for clearer analysis)
    for user_id in range(50):
        # Each user starts with signup
        signup_date = base_date + timedelta(days=user_id % 15, hours=user_id % 12)
        events.append({
            'user_id': f'user_{user_id}',
            'event_name': 'signup',
            'timestamp': signup_date,
            'event_properties': '{}',
            'user_properties': '{}'
        })
        
        # 80% proceed to login
        if user_id < 40:
            login_date = signup_date + timedelta(hours=1 + (user_id % 6))
            events.append({
                'user_id': f'user_{user_id}',
                'event_name': 'login',
                'timestamp': login_date,
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            # 60% proceed to purchase
            if user_id < 30:
                purchase_date = login_date + timedelta(hours=2 + (user_id % 12))
                events.append({
                    'user_id': f'user_{user_id}',
                    'event_name': 'purchase',
                    'timestamp': purchase_date,
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # 40% complete the funnel
                if user_id < 20:
                    complete_date = purchase_date + timedelta(hours=1 + (user_id % 3))
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

def test_dataframe_types_direct():
    """Direct test to analyze DataFrame types and execution paths"""
    print("=== Direct DataFrame Type Analysis ===\n")
    
    # Setup logging
    log_capture, handler = setup_logging()
    
    # Create test data
    events_df = create_test_data()
    funnel_steps = ['signup', 'login', 'purchase', 'complete']
    
    print("Testing Polars execution path...")
    
    # Test 1: Polars execution
    config = FunnelConfig(
        counting_method=CountingMethod.UNIQUE_USERS,
        funnel_order=FunnelOrder.ORDERED,
        reentry_mode=ReentryMode.FIRST_ONLY,
        conversion_window_hours=24 * 7  # 7 days
    )
    
    polars_calculator = FunnelCalculator(config, use_polars=True)
    
    print(f"Calculator configured with use_polars={polars_calculator.use_polars}")
    
    # Run analysis
    start_time = time.time()
    polars_results = polars_calculator.calculate_funnel_metrics(events_df, funnel_steps)
    polars_time = time.time() - start_time
    
    print(f"Polars analysis completed in {polars_time:.4f} seconds")
    
    # Analyze execution
    polars_analysis = analyze_function_execution(polars_calculator)
    
    # Get logs
    log_output = log_capture.getvalue()
    
    # Clean up logging
    logger = logging.getLogger('core.calculator')
    logger.removeHandler(handler)
    
    print("\n=== Execution Analysis ===")
    
    # Categorize functions
    polars_functions = []
    pandas_functions = []
    bridge_functions = []
    
    for func_name, stats in polars_analysis.items():
        if '_polars' in func_name:
            polars_functions.append((func_name, stats))
        elif '_pandas' in func_name:
            pandas_functions.append((func_name, stats))
        else:
            bridge_functions.append((func_name, stats))
    
    print(f"📊 **Function Execution Summary:**")
    print(f"✅ Polars functions called: {len(polars_functions)}")
    print(f"🔄 Pandas functions called: {len(pandas_functions)}")
    print(f"🌉 Bridge functions called: {len(bridge_functions)}")
    print()
    
    if polars_functions:
        print("✅ **Polars Functions Executed:**")
        for func_name, stats in polars_functions:
            print(f"  • {func_name}: {stats['calls']} calls, {stats['total_time']:.4f}s total")
        print()
    
    if pandas_functions:
        print("🔄 **Pandas Functions Executed:**")
        for func_name, stats in pandas_functions:
            print(f"  • {func_name}: {stats['calls']} calls, {stats['total_time']:.4f}s total")
        print()
        
    if bridge_functions:
        print("🌉 **Bridge/Generic Functions Executed:**")
        for func_name, stats in bridge_functions:
            print(f"  • {func_name}: {stats['calls']} calls, {stats['total_time']:.4f}s total")
        print()
    
    # Analyze logs for conversion patterns
    print("🔍 **Log Analysis for Data Type Conversions:**")
    
    conversion_patterns = [
        "Converting to Polars",
        "Converting to Pandas", 
        "falling back to pandas",
        "falling back to Pandas",
        "_to_polars",
        "_to_pandas",
        "Polars calculation failed",
        "Pandas calculation",
        "LazyFrame optimization"
    ]
    
    conversions_found = []
    for pattern in conversion_patterns:
        if pattern.lower() in log_output.lower():
            count = log_output.lower().count(pattern.lower())
            conversions_found.append((pattern, count))
    
    if conversions_found:
        print("Found conversion patterns:")
        for pattern, count in conversions_found:
            print(f"  • '{pattern}': {count} occurrences")
    else:
        print("  ✅ No explicit conversion patterns found in logs")
    print()
    
    # Check for fallback warnings
    fallback_patterns = [
        "falling back",
        "fallback",
        "failed.*pandas",
        "error.*polars"
    ]
    
    fallbacks_found = []
    for pattern in fallback_patterns:
        import re
        matches = re.findall(pattern, log_output, re.IGNORECASE)
        if matches:
            fallbacks_found.extend(matches)
    
    if fallbacks_found:
        print("⚠️  **Fallback Warnings Found:**")
        for fallback in set(fallbacks_found):
            print(f"  • {fallback}")
        print()
    else:
        print("✅ **No fallback warnings found**")
        print()
    
    # Expected vs actual function calls
    expected_polars_functions = [
        '_calculate_funnel_metrics_polars',
        '_preprocess_data_polars',
        '_calculate_unique_users_funnel_polars',
        '_calculate_cohort_analysis_polars',
        '_calculate_path_analysis_polars'
    ]
    
    called_functions = list(polars_analysis.keys())
    
    print("🎯 **Expected vs Actual Function Calls:**")
    for expected in expected_polars_functions:
        if expected in called_functions:
            stats = polars_analysis[expected]
            print(f"  ✅ {expected}: Called {stats['calls']} times ({stats['total_time']:.4f}s)")
        else:
            print(f"  ❌ {expected}: Not called")
    
    # Check for unexpected pandas calls
    unexpected_pandas = [func for func in called_functions if '_pandas' in func]
    if unexpected_pandas:
        print(f"\n⚠️  **Unexpected Pandas Function Calls:**")
        for func in unexpected_pandas:
            stats = polars_analysis[func]
            print(f"  • {func}: {stats['calls']} calls ({stats['total_time']:.4f}s)")
    else:
        print(f"\n✅ **No unexpected Pandas function calls**")
    
    print()
    
    # Performance breakdown
    total_polars_time = sum(stats['total_time'] for _, stats in polars_functions)
    total_pandas_time = sum(stats['total_time'] for _, stats in pandas_functions)
    total_bridge_time = sum(stats['total_time'] for _, stats in bridge_functions)
    total_execution_time = total_polars_time + total_pandas_time + total_bridge_time
    
    print("⚡ **Performance Breakdown:**")
    print(f"  • Total execution time: {total_execution_time:.4f}s")
    print(f"  • Polars functions: {total_polars_time:.4f}s ({total_polars_time/total_execution_time*100:.1f}%)")
    print(f"  • Pandas functions: {total_pandas_time:.4f}s ({total_pandas_time/total_execution_time*100:.1f}%)")
    print(f"  • Bridge functions: {total_bridge_time:.4f}s ({total_bridge_time/total_execution_time*100:.1f}%)")
    print()
    
    # Final verdict
    issues = len(unexpected_pandas) + len(fallbacks_found)
    
    if issues == 0 and total_polars_time > 0:
        print("🎉 **VERDICT: EXCELLENT!**")
        print("   ✅ Pure Polars execution achieved")
        print("   ✅ No unexpected Pandas fallbacks")
        print("   ✅ No data type conversion issues")
    elif total_polars_time > total_pandas_time:
        print("✅ **VERDICT: GOOD**")
        print("   ✅ Polars is primary execution engine")
        if issues > 0:
            print(f"   ⚠️  {issues} minor issues found (see details above)")
    else:
        print("⚠️  **VERDICT: NEEDS OPTIMIZATION**")
        print(f"   ❌ {issues} issues found")
        print("   ❌ Pandas usage may be higher than expected")
    
    # Show detailed log for debugging if requested
    if len(log_output) > 1000:  # Only show if there's substantial logging
        print(f"\n📋 **Detailed Log Analysis Available:**")
        print(f"   Log contains {len(log_output.splitlines())} lines")
        print(f"   Use log_output variable for detailed debugging")
    
    return {
        'polars_functions': len(polars_functions),
        'pandas_functions': len(pandas_functions),
        'bridge_functions': len(bridge_functions),
        'total_polars_time': total_polars_time,
        'total_pandas_time': total_pandas_time,
        'fallbacks_found': len(fallbacks_found),
        'execution_time': polars_time,
        'log_output': log_output
    }

if __name__ == "__main__":
    results = test_dataframe_types_direct()
    
    print(f"\n📋 **Final Summary:**")
    print(f"Polars functions executed: {results['polars_functions']}")
    print(f"Pandas functions executed: {results['pandas_functions']}")
    print(f"Bridge functions executed: {results['bridge_functions']}")
    print(f"Fallbacks detected: {results['fallbacks_found']}")
    print(f"Total execution time: {results['execution_time']:.4f}s")
    
    if results['fallbacks_found'] == 0 and results['polars_functions'] > 0:
        print("\n🚀 **System Status: OPTIMAL** - Pure Polars execution achieved!")
    elif results['polars_functions'] > results['pandas_functions']:
        print("\n✅ **System Status: GOOD** - Polars is primary engine with minor fallbacks")
    else:
        print("\n⚠️  **System Status: SUBOPTIMAL** - Check for optimization opportunities") 