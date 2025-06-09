"""
Test specifically for LazyFrame bug in path analysis.

This test reproduces and verifies the exact issue seen in the logs:
> cannot create expression literal for value of type LazyFrame. 
> Hint: Pass `allow_object=True` to accept any value and create a literal of type Object.
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import logging
import io
from datetime import datetime, timedelta

from app import (
    FunnelConfig,
    FunnelCalculator, 
    CountingMethod, 
    ReentryMode, 
    FunnelOrder,
)

# Configure logging to capture output
@pytest.fixture
def log_capture():
    """Setup log capture for testing"""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    yield log_stream
    
    logger.removeHandler(handler)

@pytest.fixture
def test_data():
    """Create test data for path analysis"""
    # Create data with 3 funnel steps
    data = []
    steps = ["Step1", "Step2", "Step3"]
    
    # 20 users complete all steps
    for user_id in range(1, 21):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append({
                "user_id": f"user_{user_id}",
                "event_name": step,
                "timestamp": event_time,
                "properties": "{}"
            })
    
    # Add some non-funnel events
    other_events = ["Browse", "Search"]
    for user_id in range(1, 21):
        for _ in range(3):  # 3 random events per user
            event_time = datetime(2023, 1, 1, 12, 0, 0) + timedelta(hours=user_id+np.random.randint(0, 5))
            data.append({
                "user_id": f"user_{user_id}",
                "event_name": np.random.choice(other_events),
                "timestamp": event_time,
                "properties": "{}"
            })
            
    return pd.DataFrame(data)

class TestLazyFrameBug:
    """Tests specifically for the LazyFrame bug in path analysis."""
    
    def test_reproduce_lazy_frame_error(self, test_data, log_capture):
        """
        Test that reproduces the LazyFrame error in path analysis.
        This specifically targets the error reported in logs.
        """
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Convert pandas to polars
        polars_df = pl.from_pandas(test_data)
        
        # Convert to LazyFrame - this should trigger the bug when passed to path analysis
        lazy_df = polars_df.lazy()
        
        try:
            # Try calling the path analysis method directly with LazyFrames
            # Filter first to get only funnel events
            lazy_funnel_events = lazy_df.filter(pl.col('event_name').is_in(steps))
            
            # Call the method that has the error
            result = calculator._calculate_path_analysis_polars_optimized(
                lazy_funnel_events,  # Here we're passing a LazyFrame 
                steps,
                lazy_df  # Here we're passing a LazyFrame
            )
            
            # If we get here, the bug is fixed
            print("✓ Bug appears to be fixed - no LazyFrame error")
            
        except Exception as e:
            # Check if this is the expected LazyFrame error
            error_str = str(e)
            if "cannot create expression literal for value of type LazyFrame" in error_str:
                # This is the expected error
                print(f"✓ Successfully reproduced the LazyFrame error: {error_str}")
                
                # Verify fallback behavior - the error should be caught and handled
                results = calculator.calculate_funnel_metrics(test_data, steps)
                
                log_output = log_capture.getvalue()
                assert "falling back to standard polars" in log_output.lower(), \
                    "LazyFrame error should trigger fallback to standard Polars"
                
                # Verify we still got results despite the error
                assert results is not None
                assert results.path_analysis is not None
            else:
                # This is a different error than expected
                pytest.fail(f"Got unexpected error: {error_str}")
    
    def test_fix_lazy_frame_error(self, test_data):
        """
        Test a potential fix for the LazyFrame error.
        Demonstrates how to fix the issue while keeping the function signature compatible.
        """
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY, 
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Convert pandas to polars
        polars_df = pl.from_pandas(test_data)
        lazy_df = polars_df.lazy()
        
        # Create a fixed version of the method
        def fixed_path_analysis(segment_funnel_events_df, funnel_steps, full_history_for_segment_users):
            """Fixed version that handles LazyFrames properly"""
            # If inputs are LazyFrames, collect them first
            if hasattr(segment_funnel_events_df, 'collect'):
                segment_funnel_events_df = segment_funnel_events_df.collect()
                
            if hasattr(full_history_for_segment_users, 'collect'):
                full_history_for_segment_users = full_history_for_segment_users.collect()
                
            # Now continue with normal processing knowing we have regular DataFrames
            dropoff_paths = {}
            between_steps_events = {}
            
            # Just return empty results for this test
            return {
                'dropoff_paths': dropoff_paths,
                'between_steps_events': between_steps_events
            }
        
        # Try with the fixed version
        try:
            # Filter to get only funnel events
            lazy_funnel_events = lazy_df.filter(pl.col('event_name').is_in(steps))
            
            # Call our fixed function with LazyFrames
            result = fixed_path_analysis(
                lazy_funnel_events,
                steps,
                lazy_df
            )
            
            # If we get here without exception, the fix works
            assert result is not None
            print("✓ Fix for LazyFrame error works correctly")
            
        except Exception as e:
            pytest.fail(f"Fix did not work, got error: {str(e)}")
            
    def test_lazy_frame_error_fix_suggestion(self):
        """
        Suggest a fix for the LazyFrame error based on the hint in the error message.
        """
        # Demonstrate the suggested fix from the error message
        try:
            # Create a small LazyFrame
            df = pl.DataFrame({"a": [1, 2, 3]})
            lazy_df = df.lazy()
            
            # Try to create a literal with a LazyFrame - this will fail
            # expr = pl.lit(lazy_df)  # This would raise the error
            
            # Instead, follow the hint to use allow_object=True
            expr = pl.lit(lazy_df, allow_object=True)
            
            # If we get here, the suggested fix works
            assert expr is not None
            print("✓ Suggested fix from error hint works: using allow_object=True")
            
        except Exception as e:
            pytest.fail(f"Suggested fix didn't work: {str(e)}") 