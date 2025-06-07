#!/usr/bin/env python3
"""
Test suite for comparing Pandas and Polars implementations
Ensures both engines produce identical results across various scenarios
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

from app import (
    FunnelCalculator, 
    FunnelConfig, 
    CountingMethod, 
    ReentryMode, 
    FunnelOrder,
    FunnelResults,
    TimeToConvertStats
)

def create_test_dataset(scenario: str) -> Tuple[pd.DataFrame, List[str]]:
    """Create test data for different scenarios"""
    base_time = datetime(2024, 1, 1)
    events = []
    
    if scenario == "conversion_window_edge":
        # Create events exactly at conversion window boundaries
        events.extend([
            # User 1: Events exactly at conversion window boundary (24h)
            {
                'user_id': 'user_1',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=24),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # User 2: Events just inside conversion window
            {
                'user_id': 'user_2',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_2',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=23, minutes=59),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # User 3: Events just outside conversion window
            {
                'user_id': 'user_3',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_3',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=24, minutes=1),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ])
        funnel_steps = ['Step A', 'Step B']
        
    elif scenario == "reentry_patterns":
        # Test reentry patterns with multiple conversions
        events.extend([
            # User 1: Multiple valid conversions within window
            {
                'user_id': 'user_1',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'Step A',
                'timestamp': base_time + timedelta(days=2),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(days=2, hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ])
        funnel_steps = ['Step A', 'Step B']
        
    elif scenario == "mixed_data_types":
        # Test with mixed numeric and string user IDs
        events.extend([
            # String user ID
            {
                'user_id': 'user_abc',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_abc',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # Numeric user ID
            {
                'user_id': '12345',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': '12345',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ])
        funnel_steps = ['Step A', 'Step B']
    
    elif scenario == "unordered_events":
        # Test unordered funnel analysis
        events.extend([
            # User 1: Events in correct order
            {
                'user_id': 'user_1',
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # User 2: Events in reverse order
            {
                'user_id': 'user_2',
                'event_name': 'Step B',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_2',
                'event_name': 'Step A',
                'timestamp': base_time + timedelta(hours=1),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ])
        funnel_steps = ['Step A', 'Step B']
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df, funnel_steps

def compare_funnel_results(pandas_results: FunnelResults, polars_results: FunnelResults, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Compare FunnelResults from both engines with detailed reporting"""
    
    if pandas_results.steps != polars_results.steps:
        return False, f"Steps mismatch: Pandas {pandas_results.steps} vs Polars {polars_results.steps}"
    
    if len(pandas_results.users_count) != len(polars_results.users_count):
        return False, "Different number of steps in results"
    
    for i, (p_count, pl_count) in enumerate(zip(pandas_results.users_count, polars_results.users_count)):
        if p_count != pl_count:
            return False, f"User count mismatch at step {i}: Pandas {p_count} vs Polars {pl_count}"
    
    for i, (p_rate, pl_rate) in enumerate(zip(pandas_results.conversion_rates, polars_results.conversion_rates)):
        if abs(p_rate - pl_rate) > tolerance:
            return False, f"Conversion rate mismatch at step {i}: Pandas {p_rate:.2f}% vs Polars {pl_rate:.2f}%"
    
    for i, (p_drop, pl_drop) in enumerate(zip(pandas_results.drop_offs, polars_results.drop_offs)):
        if p_drop != pl_drop:
            return False, f"Drop-off count mismatch at step {i}: Pandas {p_drop} vs Polars {pl_drop}"
    
    return True, "Results match within tolerance"

def compare_time_to_convert(pandas_stats: List[TimeToConvertStats], polars_stats: List[TimeToConvertStats], tolerance: float = 1e-6) -> Tuple[bool, str]:
    """Compare time to convert statistics between engines"""
    
    if len(pandas_stats) != len(polars_stats):
        return False, f"Different number of step pairs: Pandas {len(pandas_stats)} vs Polars {len(polars_stats)}"
    
    for i, (p_stat, pl_stat) in enumerate(zip(pandas_stats, polars_stats)):
        if p_stat.step_from != pl_stat.step_from or p_stat.step_to != pl_stat.step_to:
            return False, f"Step pair mismatch at index {i}"
        
        if abs(p_stat.mean_hours - pl_stat.mean_hours) > tolerance:
            return False, f"Mean hours differ for {p_stat.step_from}->{p_stat.step_to}: Pandas {p_stat.mean_hours:.2f} vs Polars {pl_stat.mean_hours:.2f}"
        
        if abs(p_stat.median_hours - pl_stat.median_hours) > tolerance:
            return False, f"Median hours differ for {p_stat.step_from}->{p_stat.step_to}"
        
        if len(p_stat.conversion_times) != len(pl_stat.conversion_times):
            return False, f"Different number of conversion times for {p_stat.step_from}->{p_stat.step_to}"
    
    return True, "Time to convert stats match within tolerance"

@pytest.mark.parametrize("scenario", [
    "conversion_window_edge",
    "reentry_patterns",
    "mixed_data_types",
    "unordered_events"
])
def test_engine_comparison(scenario: str):
    """Test that Pandas and Polars engines produce identical results across scenarios"""
    
    # Create test data for the scenario
    events_df, funnel_steps = create_test_dataset(scenario)
    
    # Test configurations
    configs = [
        {
            'name': 'Default Config',
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
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.OPTIMIZED_REENTRY,
                funnel_order=FunnelOrder.ORDERED
            )
        }
    ]
    
    # Add unordered config for unordered scenario
    if scenario == "unordered_events":
        configs.append({
            'name': 'Unordered Funnel',
            'config': FunnelConfig(
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.UNORDERED
            )
        })
    
    for config_dict in configs:
        config = config_dict['config']
        
        # Create calculators for both engines
        pandas_calculator = FunnelCalculator(config, use_polars=False)
        polars_calculator = FunnelCalculator(config, use_polars=True)
        
        # Calculate funnel metrics
        pandas_results = pandas_calculator.calculate_funnel_metrics(events_df, funnel_steps)
        polars_results = polars_calculator.calculate_funnel_metrics(events_df, funnel_steps)
        
        # Compare results
        match, message = compare_funnel_results(pandas_results, polars_results)
        assert match, f"Scenario '{scenario}' with config '{config_dict['name']}' failed: {message}"
        
        # Compare time to convert stats if available
        if hasattr(pandas_results, 'time_to_convert') and pandas_results.time_to_convert:
            time_match, time_message = compare_time_to_convert(
                pandas_results.time_to_convert,
                polars_results.time_to_convert
            )
            assert time_match, f"Time to convert comparison failed for scenario '{scenario}': {time_message}"

@pytest.mark.parametrize("window_hours", [24, 48, 168])  # 1 day, 2 days, 1 week
def test_conversion_window_comparison(window_hours: int):
    """Test that both engines handle conversion windows identically"""
    
    # Create test data with events at window boundaries
    base_time = datetime(2024, 1, 1)
    events = []
    
    # Add events at various points around the conversion window
    for i, minutes_offset in enumerate([-1, 0, 1]):  # Just inside, exactly at, just outside
        user_id = f"user_{i+1}"
        events.extend([
            {
                'user_id': user_id,
                'event_name': 'Step A',
                'timestamp': base_time,
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': user_id,
                'event_name': 'Step B',
                'timestamp': base_time + timedelta(hours=window_hours, minutes=minutes_offset),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ])
    
    df = pd.DataFrame(events)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    funnel_steps = ['Step A', 'Step B']
    
    # Test with both engines
    config = FunnelConfig(
        conversion_window_hours=window_hours,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )
    
    pandas_calculator = FunnelCalculator(config, use_polars=False)
    polars_calculator = FunnelCalculator(config, use_polars=True)
    
    pandas_results = pandas_calculator.calculate_funnel_metrics(df, funnel_steps)
    polars_results = polars_calculator.calculate_funnel_metrics(df, funnel_steps)
    
    # Compare results
    match, message = compare_funnel_results(pandas_results, polars_results)
    assert match, f"Conversion window test failed for {window_hours} hours: {message}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 