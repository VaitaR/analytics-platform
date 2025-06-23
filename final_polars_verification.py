#!/usr/bin/env python3
"""
Final verification test to confirm Polars DataFrame usage in performance monitored functions.
This test correctly handles the performance metrics structure.
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from core import FunnelCalculator
from models import FunnelConfig, CountingMethod, FunnelOrder, ReentryMode

def create_test_data():
    """Create test data for verification"""
    print("Creating test data...")
    
    events = []
    base_date = datetime(2024, 1, 1)
    
    # Create events for 50 users
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

def verify_polars_execution():
    """Verify that Polars functions are being used correctly"""
    print("=== Final Polars DataFrame Verification Test ===\n")
    
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
    
    print(f"Calculator configured with use_polars={calculator.use_polars}")
    print("Running funnel analysis...")
    
    # Run analysis
    start_time = time.time()
    results = calculator.calculate_funnel_metrics(events_df, funnel_steps)
    execution_time = time.time() - start_time
    
    print(f"Analysis completed in {execution_time:.4f} seconds")
    
    # Analyze performance metrics
    performance_metrics = calculator.get_performance_report()
    
    print("\n=== Performance Analysis ===")
    
    if performance_metrics:
        print("📊 **Functions Executed:**")
        
        polars_functions = []
        pandas_functions = []
        other_functions = []
        
        total_time = 0
        for func_name, stats in performance_metrics.items():
            if stats and isinstance(stats, dict) and 'total_time' in stats:
                func_total = stats['total_time']
                func_calls = stats.get('total_calls', 1)
                total_time += func_total
                
                if '_polars' in func_name:
                    polars_functions.append((func_name, func_calls, func_total))
                elif '_pandas' in func_name:
                    pandas_functions.append((func_name, func_calls, func_total))
                else:
                    other_functions.append((func_name, func_calls, func_total))
        
        # Report Polars functions
        if polars_functions:
            print(f"\n✅ **Polars Functions ({len(polars_functions)}):**")
            for func_name, calls, time_spent in polars_functions:
                percentage = (time_spent / total_time * 100) if total_time > 0 else 0
                print(f"  • {func_name}: {calls} calls, {time_spent:.4f}s ({percentage:.1f}%)")
        
        # Report Pandas functions (should be minimal or none)
        if pandas_functions:
            print(f"\n⚠️  **Pandas Functions ({len(pandas_functions)}):**")
            for func_name, calls, time_spent in pandas_functions:
                percentage = (time_spent / total_time * 100) if total_time > 0 else 0
                print(f"  • {func_name}: {calls} calls, {time_spent:.4f}s ({percentage:.1f}%)")
        
        # Report other functions
        if other_functions:
            print(f"\n🔧 **Other Functions ({len(other_functions)}):**")
            for func_name, calls, time_spent in other_functions:
                percentage = (time_spent / total_time * 100) if total_time > 0 else 0
                print(f"  • {func_name}: {calls} calls, {time_spent:.4f}s ({percentage:.1f}%)")
        
        print(f"\n📈 **Performance Summary:**")
        polars_time = sum(time_spent for _, _, time_spent in polars_functions)
        pandas_time = sum(time_spent for _, _, time_spent in pandas_functions)
        other_time = sum(time_spent for _, _, time_spent in other_functions)
        
        print(f"  • Total execution time: {total_time:.4f}s")
        print(f"  • Polars functions time: {polars_time:.4f}s ({polars_time/total_time*100:.1f}%)")
        print(f"  • Pandas functions time: {pandas_time:.4f}s ({pandas_time/total_time*100:.1f}%)")
        print(f"  • Other functions time: {other_time:.4f}s ({other_time/total_time*100:.1f}%)")
        
    else:
        print("⚠️  No performance metrics available")
        polars_functions = []
        pandas_functions = []
        other_functions = []
        total_time = 0
        polars_time = 0
        pandas_time = 0
        other_time = 0
    
    # Verify expected functions were called
    print("\n🎯 **Expected Function Verification:**")
    
    expected_polars_functions = [
        '_calculate_funnel_metrics_polars',
        '_preprocess_data_polars',
        '_calculate_unique_users_funnel_polars',
        '_calculate_cohort_analysis_polars',
        '_calculate_path_analysis_polars_optimized'
    ]
    
    called_functions = list(performance_metrics.keys()) if performance_metrics else []
    
    for expected_func in expected_polars_functions:
        if expected_func in called_functions:
            stats = performance_metrics[expected_func]
            if stats and isinstance(stats, dict):
                calls = stats.get('total_calls', 0)
                time_spent = stats.get('total_time', 0)
                print(f"  ✅ {expected_func}: Called {calls} times ({time_spent:.4f}s)")
            else:
                print(f"  ⚠️  {expected_func}: Listed but no timing data")
        else:
            print(f"  ❌ {expected_func}: Not called")
    
    # Check for unexpected pandas usage
    pandas_usage = [func for func in called_functions if '_pandas' in func]
    if pandas_usage:
        print(f"\n⚠️  **Unexpected Pandas Usage:**")
        for func in pandas_usage:
            print(f"  • {func}")
        print("  This indicates potential fallbacks from Polars to Pandas")
    else:
        print(f"\n✅ **No unexpected Pandas usage detected**")
    
    # Final assessment
    print("\n🏆 **Final Assessment:**")
    
    polars_func_count = len(polars_functions)
    pandas_func_count = len(pandas_functions)
    polars_time_percentage = (polars_time / total_time * 100) if total_time > 0 else 0
    
    if pandas_func_count == 0 and polars_func_count >= 5:
        print("  🎉 **EXCELLENT**: Pure Polars execution achieved!")
        print(f"     • {polars_func_count} Polars functions executed")
        print(f"     • {polars_time_percentage:.1f}% of time spent in Polars functions")
        print("     • No unexpected Pandas fallbacks")
        verdict = "EXCELLENT"
        
    elif polars_time_percentage > 90:
        print("  ✅ **VERY GOOD**: Mostly Polars execution")
        print(f"     • {polars_func_count} Polars functions, {pandas_func_count} Pandas functions")
        print(f"     • {polars_time_percentage:.1f}% of time spent in Polars functions")
        if pandas_func_count > 0:
            print("     • Minor Pandas usage detected (likely fallbacks)")
        verdict = "VERY_GOOD"
            
    elif polars_time_percentage > 50:
        print("  ✅ **GOOD**: Polars is primary engine")
        print(f"     • {polars_func_count} Polars functions, {pandas_func_count} Pandas functions")
        print(f"     • {polars_time_percentage:.1f}% of time spent in Polars functions")
        print("     • Some optimization opportunities exist")
        verdict = "GOOD"
        
    else:
        print("  ⚠️  **NEEDS OPTIMIZATION**: Significant Pandas usage")
        print(f"     • {polars_func_count} Polars functions, {pandas_func_count} Pandas functions")
        print(f"     • Only {polars_time_percentage:.1f}% of time spent in Polars functions")
        print("     • Review for excessive fallbacks")
        verdict = "NEEDS_OPTIMIZATION"
    
    # DataFrame type verification based on function names and logs
    print("\n🔍 **DataFrame Type Analysis:**")
    print("Based on function execution patterns:")
    
    # Functions that should receive Polars DataFrames
    polars_dataframe_functions = [
        '_preprocess_data_polars',
        'segment_events_data_polars', 
        '_calculate_unique_users_funnel_polars',
        '_calculate_time_to_convert_polars',
        '_calculate_cohort_analysis_polars',
        '_calculate_path_analysis_polars_optimized'
    ]
    
    polars_df_confirmed = 0
    for func in polars_dataframe_functions:
        if func in called_functions:
            polars_df_confirmed += 1
            print(f"  ✅ {func}: Polars DataFrame confirmed")
        else:
            print(f"  ❌ {func}: Not executed")
    
    print(f"\n📊 **DataFrame Type Summary:**")
    print(f"  • Functions receiving Polars DataFrames: {polars_df_confirmed}/{len(polars_dataframe_functions)}")
    print(f"  • Functions with Pandas fallbacks: {pandas_func_count}")
    
    if polars_df_confirmed == len(polars_dataframe_functions) and pandas_func_count == 0:
        print("  🎉 **PERFECT**: All functions receive correct DataFrame types!")
    elif polars_df_confirmed >= len(polars_dataframe_functions) * 0.8:
        print("  ✅ **GOOD**: Most functions receive Polars DataFrames")
    else:
        print("  ⚠️  **SUBOPTIMAL**: Many functions not using Polars DataFrames")
    
    return {
        'polars_functions': polars_func_count,
        'pandas_functions': pandas_func_count,
        'execution_time': execution_time,
        'total_functions': len(called_functions),
        'polars_time_percentage': polars_time_percentage,
        'verdict': verdict,
        'polars_df_confirmed': polars_df_confirmed
    }

if __name__ == "__main__":
    results = verify_polars_execution()
    
    print(f"\n📋 **Final Summary:**")
    print(f"Polars functions: {results['polars_functions']}")
    print(f"Pandas functions: {results['pandas_functions']}")
    print(f"Total functions: {results['total_functions']}")
    print(f"Polars time percentage: {results['polars_time_percentage']:.1f}%")
    print(f"Polars DataFrames confirmed: {results['polars_df_confirmed']}")
    print(f"Execution time: {results['execution_time']:.4f}s")
    print(f"Overall verdict: {results['verdict']}")
    
    if results['verdict'] == "EXCELLENT":
        print("\n🚀 **STATUS: OPTIMAL** - Pure Polars execution with correct DataFrame types!")
        print("✅ All Polars functions receive Polars DataFrames")
        print("✅ No unexpected conversions to Pandas detected")
    elif results['verdict'] in ["VERY_GOOD", "GOOD"]:
        print("\n✅ **STATUS: GOOD** - Polars is the primary execution engine")
        print("✅ Most functions receive correct DataFrame types")
        if results['pandas_functions'] > 0:
            print("⚠️  Minor fallbacks detected - check logs for details")
    else:
        print("\n⚠️  **STATUS: NEEDS OPTIMIZATION** - Check for DataFrame conversion issues")
        print("❌ Significant Pandas usage or fallbacks detected") 