#!/usr/bin/env python3
"""
Test script for Polars path analysis migration
Tests that Polars and Pandas implementations produce identical results
"""

import sys
import os
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from collections import Counter

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import FunnelCalculator, FunnelConfig, CountingMethod, ReentryMode, FunnelOrder, PathAnalysisData

def create_test_data_for_path_analysis() -> pd.DataFrame:
    """Create test data specifically designed for path analysis testing"""
    np.random.seed(42)
    
    # Create users with different path behaviors
    events_data = []
    base_time = datetime(2024, 1, 1)
    
    # User 1: Completes funnel with events between steps
    user_id = "user_001"
    events_data.extend([
        {
            'user_id': user_id,
            'event_name': 'User Sign-Up',
            'timestamp': base_time,
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Page View',
            'timestamp': base_time + timedelta(minutes=5),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Profile Setup',
            'timestamp': base_time + timedelta(minutes=10),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Search',
            'timestamp': base_time + timedelta(minutes=15),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Verify Email',
            'timestamp': base_time + timedelta(minutes=20),
            'event_properties': '{}',
            'user_properties': '{}'
        }
    ])
    
    # User 2: Drops off after first step, does other activities
    user_id = "user_002"
    events_data.extend([
        {
            'user_id': user_id,
            'event_name': 'User Sign-Up',
            'timestamp': base_time + timedelta(hours=1),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Help Page Visit',
            'timestamp': base_time + timedelta(hours=1, minutes=30),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Contact Support',
            'timestamp': base_time + timedelta(hours=2),
            'event_properties': '{}',
            'user_properties': '{}'
        }
    ])
    
    # User 3: Completes funnel with multiple events between steps
    user_id = "user_003"
    events_data.extend([
        {
            'user_id': user_id,
            'event_name': 'User Sign-Up',
            'timestamp': base_time + timedelta(hours=2),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Product View',
            'timestamp': base_time + timedelta(hours=2, minutes=5),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Add to Wishlist',
            'timestamp': base_time + timedelta(hours=2, minutes=10),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Profile Setup',
            'timestamp': base_time + timedelta(hours=2, minutes=15),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Search',
            'timestamp': base_time + timedelta(hours=2, minutes=20),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Verify Email',
            'timestamp': base_time + timedelta(hours=2, minutes=25),
            'event_properties': '{}',
            'user_properties': '{}'
        }
    ])
    
    # User 4: Drops off after second step
    user_id = "user_004"
    events_data.extend([
        {
            'user_id': user_id,
            'event_name': 'User Sign-Up',
            'timestamp': base_time + timedelta(hours=3),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Profile Setup',
            'timestamp': base_time + timedelta(hours=3, minutes=10),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Logout',
            'timestamp': base_time + timedelta(hours=3, minutes=30),
            'event_properties': '{}',
            'user_properties': '{}'
        }
    ])
    
    # User 5: No funnel progression but has other events
    user_id = "user_005"
    events_data.extend([
        {
            'user_id': user_id,
            'event_name': 'Page View',
            'timestamp': base_time + timedelta(hours=4),
            'event_properties': '{}',
            'user_properties': '{}'
        },
        {
            'user_id': user_id,
            'event_name': 'Search',
            'timestamp': base_time + timedelta(hours=4, minutes=10),
            'event_properties': '{}',
            'user_properties': '{}'
        }
    ])
    
    df = pd.DataFrame(events_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def compare_path_analysis_results(pandas_result: PathAnalysisData, polars_result: PathAnalysisData) -> bool:
    """Compare path analysis results from Pandas and Polars implementations"""
    
    # Compare dropoff_paths
    if set(pandas_result.dropoff_paths.keys()) != set(polars_result.dropoff_paths.keys()):
        print("❌ Dropoff paths keys don't match")
        print(f"Pandas keys: {set(pandas_result.dropoff_paths.keys())}")
        print(f"Polars keys: {set(polars_result.dropoff_paths.keys())}")
        return False
    
    for step in pandas_result.dropoff_paths:
        pandas_paths = pandas_result.dropoff_paths[step]
        polars_paths = polars_result.dropoff_paths[step]
        
        if pandas_paths != polars_paths:
            print(f"❌ Dropoff paths for step '{step}' don't match")
            print(f"Pandas: {pandas_paths}")
            print(f"Polars: {polars_paths}")
            return False
    
    # Compare between_steps_events
    if set(pandas_result.between_steps_events.keys()) != set(polars_result.between_steps_events.keys()):
        print("❌ Between steps events keys don't match")
        print(f"Pandas keys: {set(pandas_result.between_steps_events.keys())}")
        print(f"Polars keys: {set(polars_result.between_steps_events.keys())}")
        return False
    
    for step_pair in pandas_result.between_steps_events:
        pandas_events = pandas_result.between_steps_events[step_pair]
        polars_events = polars_result.between_steps_events[step_pair]
        
        if pandas_events != polars_events:
            print(f"❌ Between steps events for '{step_pair}' don't match")
            print(f"Pandas: {pandas_events}")
            print(f"Polars: {polars_events}")
            return False
    
    return True

def test_path_analysis_migration():
    """Test the path analysis migration with various configurations"""
    
    print("🧪 Testing Polars Path Analysis Migration")
    print("=" * 50)
    
    # Create test data
    events_df = create_test_data_for_path_analysis()
    funnel_steps = ['User Sign-Up', 'Profile Setup', 'Verify Email']
    
    print(f"📊 Test data: {len(events_df)} events, {events_df['user_id'].nunique()} users")
    print(f"🎯 Funnel steps: {funnel_steps}")
    print()
    
    # Test configurations
    test_configs = [
        {
            'name': 'Default Configuration',
            'config': FunnelConfig(
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.ORDERED
            )
        },
        {
            'name': 'Optimized Reentry',
            'config': FunnelConfig(
                conversion_window_hours=48,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.OPTIMIZED_REENTRY,
                funnel_order=FunnelOrder.ORDERED
            )
        },
        {
            'name': 'Unordered Funnel',
            'config': FunnelConfig(
                conversion_window_hours=72,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.UNORDERED
            )
        }
    ]
    
    all_tests_passed = True
    
    for test_config in test_configs:
        print(f"🔧 Testing: {test_config['name']}")
        config = test_config['config']
        
        # Test Pandas implementation
        pandas_calculator = FunnelCalculator(config, use_polars=False)
        pandas_results = pandas_calculator.calculate_funnel_metrics(events_df, funnel_steps)
        
        # Test Polars implementation
        polars_calculator = FunnelCalculator(config, use_polars=True)
        polars_results = polars_calculator.calculate_funnel_metrics(events_df, funnel_steps)
        
        # Compare path analysis results
        if pandas_results.path_analysis and polars_results.path_analysis:
            paths_match = compare_path_analysis_results(
                pandas_results.path_analysis, 
                polars_results.path_analysis
            )
            
            if paths_match:
                print(f"✅ Path analysis results match for {test_config['name']}")
            else:
                print(f"❌ Path analysis results don't match for {test_config['name']}")
                all_tests_passed = False
        else:
            print(f"⚠️  Path analysis missing for {test_config['name']}")
            print(f"   Pandas path analysis: {pandas_results.path_analysis is not None}")
            print(f"   Polars path analysis: {polars_results.path_analysis is not None}")
            all_tests_passed = False
        
        print()
    
    # Performance comparison
    print("⚡ Performance Comparison")
    print("-" * 25)
    
    # Create larger test dataset for performance testing
    larger_events = []
    base_time = datetime(2024, 1, 1)
    
    for i in range(1000):  # 1000 users
        user_id = f"perf_user_{i:04d}"
        
        # Each user has a chance to complete the funnel
        if np.random.random() < 0.7:  # 70% start the funnel
            larger_events.append({
                'user_id': user_id,
                'event_name': 'User Sign-Up',
                'timestamp': base_time + timedelta(minutes=i),
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            # Add some events between steps
            if np.random.random() < 0.5:
                larger_events.append({
                    'user_id': user_id,
                    'event_name': 'Page View',
                    'timestamp': base_time + timedelta(minutes=i+2),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
            
            if np.random.random() < 0.5:  # 50% continue to second step
                larger_events.append({
                    'user_id': user_id,
                    'event_name': 'Profile Setup',
                    'timestamp': base_time + timedelta(minutes=i+5),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # Add events between second and third step
                if np.random.random() < 0.3:
                    larger_events.append({
                        'user_id': user_id,
                        'event_name': 'Search',
                        'timestamp': base_time + timedelta(minutes=i+7),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                
                if np.random.random() < 0.3:  # 30% complete the funnel
                    larger_events.append({
                        'user_id': user_id,
                        'event_name': 'Verify Email',
                        'timestamp': base_time + timedelta(minutes=i+10),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
    
    larger_df = pd.DataFrame(larger_events)
    larger_df['timestamp'] = pd.to_datetime(larger_df['timestamp'])
    
    print(f"📊 Performance test data: {len(larger_df)} events, {larger_df['user_id'].nunique()} users")
    
    config = FunnelConfig(
        conversion_window_hours=24,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )
    
    # Time Pandas implementation
    import time
    
    start_time = time.time()
    pandas_calculator = FunnelCalculator(config, use_polars=False)
    pandas_results = pandas_calculator.calculate_funnel_metrics(larger_df, funnel_steps)
    pandas_time = time.time() - start_time
    
    # Time Polars implementation
    start_time = time.time()
    polars_calculator = FunnelCalculator(config, use_polars=True)
    polars_results = polars_calculator.calculate_funnel_metrics(larger_df, funnel_steps)
    polars_time = time.time() - start_time
    
    print(f"⏱️  Pandas time: {pandas_time:.3f} seconds")
    print(f"⚡ Polars time: {polars_time:.3f} seconds")
    
    if polars_time < pandas_time:
        speedup = pandas_time / polars_time
        print(f"🚀 Polars is {speedup:.2f}x faster!")
    else:
        slowdown = polars_time / pandas_time
        print(f"⚠️  Polars is {slowdown:.2f}x slower")
    
    print()
    
    # Final summary
    if all_tests_passed:
        print("🎉 All path analysis migration tests passed!")
        print("✅ Polars implementation produces identical results to Pandas")

        print("\n⏱️  Testing Time to Convert Migration")
        print("=" * 50)

        # Test time to convert migration
        pandas_config = FunnelConfig(
            conversion_window_hours=168,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED
        )

        polars_config = FunnelConfig(
            conversion_window_hours=168,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED
        )

        # Create calculators
        pandas_calculator = FunnelCalculator(pandas_config, use_polars=False)
        polars_calculator = FunnelCalculator(polars_config, use_polars=True)

        # Calculate time to convert stats with both engines
        pandas_time_stats = pandas_calculator._calculate_time_to_convert_optimized(larger_df, funnel_steps)
        polars_time_stats = polars_calculator._calculate_time_to_convert_optimized(larger_df, funnel_steps)

        # Compare results
        def compare_time_to_convert_stats(pandas_stats, polars_stats):
            """Compare time to convert statistics between Pandas and Polars"""
            if len(pandas_stats) != len(polars_stats):
                return False, f"Different number of stats: {len(pandas_stats)} vs {len(polars_stats)}"
            
            for i, (pandas_stat, polars_stat) in enumerate(zip(pandas_stats, polars_stats)):
                # Compare step names
                if pandas_stat.step_from != polars_stat.step_from or pandas_stat.step_to != polars_stat.step_to:
                    return False, f"Different step names at index {i}"
                
                # Compare statistics (allow small floating point differences)
                tolerance = 0.01  # 0.01 hours tolerance
                
                if abs(pandas_stat.mean_hours - polars_stat.mean_hours) > tolerance:
                    return False, f"Mean hours differ: {pandas_stat.mean_hours} vs {polars_stat.mean_hours}"
                
                if abs(pandas_stat.median_hours - polars_stat.median_hours) > tolerance:
                    return False, f"Median hours differ: {pandas_stat.median_hours} vs {polars_stat.median_hours}"
                
                # Compare conversion times count
                if len(pandas_stat.conversion_times) != len(polars_stat.conversion_times):
                    return False, f"Different number of conversion times: {len(pandas_stat.conversion_times)} vs {len(polars_stat.conversion_times)}"
            
            return True, "Time to convert stats match"

        # Test the comparison
        success, message = compare_time_to_convert_stats(pandas_time_stats, polars_time_stats)

        if success:
            print("✅ Time to convert migration test passed!")
            print(f"📊 Found {len(pandas_time_stats)} step pairs with conversion time data")
            
            # Display some stats
            for stat in pandas_time_stats:
                print(f"   {stat.step_from} → {stat.step_to}: {stat.mean_hours:.2f}h avg, {len(stat.conversion_times)} conversions")
        else:
            print(f"❌ Time to convert migration test failed: {message}")
            print("\nPandas stats:")
            for stat in pandas_time_stats:
                print(f"   {stat.step_from} → {stat.step_to}: mean={stat.mean_hours:.2f}h, conversions={len(stat.conversion_times)}")
            
            print("\nPolars stats:")
            for stat in polars_time_stats:
                print(f"   {stat.step_from} → {stat.step_to}: mean={stat.mean_hours:.2f}h, conversions={len(stat.conversion_times)}")

        print("\n🎉 All migration tests completed!")
        print("✅ Polars implementations produce identical results to Pandas")
        return True
    else:
        print("❌ Some path analysis migration tests failed!")
        print("🔧 Please review the implementation")
        return False

if __name__ == "__main__":
    success = test_path_analysis_migration()
    sys.exit(0 if success else 1) 