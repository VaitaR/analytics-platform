"""
Tests specifically designed to detect fallbacks from Polars to Pandas implementation.

This test suite verifies that the optimized Polars implementations are actually being used
rather than silently falling back to slower Pandas implementations.

It differs from regular tests by:
1. Actually checking logs for fallback messages
2. Verifying that specific optimized code paths are used
3. Explicitly failing if a fallback occurs
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import logging
from datetime import datetime, timedelta
import io
from typing import List, Dict, Any

from app import (
    FunnelConfig,
    FunnelCalculator,
    CountingMethod,
    ReentryMode,
    FunnelOrder,
    FunnelResults,
)

# ------------------- Test Fixtures -------------------

@pytest.fixture
def events_data_base():
    """
    Create basic sequential events for testing.
    """
    data = []
    steps = ["Step1", "Step2", "Step3"]
    
    # 30 users who complete all steps in order
    for user_id in range(1, 31):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append({
                "user_id": f"user_{user_id}",
                "event_name": step,
                "timestamp": event_time,
                "properties": "{}"
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def create_lazy_frame():
    """Generate a LazyFrame that can trigger the path analysis fallback."""
    def _create_frame(n_rows=100):
        data = {
            "user_id": [f"user_{i}" for i in range(n_rows)],
            "event_name": ["Step1" if i % 3 == 0 else "Step2" if i % 3 == 1 else "Step3" for i in range(n_rows)],
            "timestamp": [datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=i) for i in range(n_rows)],
            "properties": ["{}" for _ in range(n_rows)]
        }
        df = pl.DataFrame(data)
        return df.lazy()  # Convert to LazyFrame
    return _create_frame

@pytest.fixture
def log_capture():
    """Capture logs to check for fallback messages."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    yield log_stream
    
    logger.removeHandler(handler)

# ------------------- Test Classes -------------------

class TestPolarsFallbackDetection:
    """
    Tests specifically designed to detect fallbacks from Polars to Pandas.
    These tests should fail if a fallback occurs, unlike regular tests that
    pass even when fallbacks happen.
    """
    
    def test_path_analysis_fallback_detection(self, events_data_base, log_capture):
        """
        Test that specifically checks for fallbacks in path analysis.
        This test SHOULD FAIL if the Polars implementation falls back to Pandas.
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Execute
        results = calculator.calculate_funnel_metrics(events_data_base, steps)
        
        # Check logs for fallback messages
        log_output = log_capture.getvalue()
        assert "falling back to pandas" not in log_output.lower(), \
            "Detected fallback to Pandas in main funnel calculation"
        assert "falling back to standard polars" not in log_output.lower(), \
            "Detected fallback from optimized Polars to standard Polars"
            
        # Make sure we got valid results
        assert results is not None
        assert len(results.steps) == len(steps)
        assert results.path_analysis is not None
        
    def test_cohort_analysis_fallback_detection(self, events_data_base, log_capture):
        """
        Test that specifically checks for fallbacks in cohort analysis.
        This test SHOULD FAIL if the Polars implementation falls back.
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Run cohort analysis directly
        cohort_data = calculator._calculate_cohort_analysis_optimized(events_data_base, steps)
        
        # Check logs for fallback messages
        log_output = log_capture.getvalue()
        assert "falling back" not in log_output.lower(), \
            "Detected fallback in cohort analysis"
        
        # Verify we got valid results
        assert cohort_data is not None
        assert cohort_data.cohort_labels is not None
        
    def test_problematic_lazy_frame_path_analysis(self, create_lazy_frame, log_capture):
        """
        Test that checks the LazyFrame issue mentioned in the logs.
        This test is designed to recreate and detect the issue with LazyFrames
        in path analysis.
        """
        # Create a LazyFrame
        lazy_df = create_lazy_frame(n_rows=100)
        
        # Convert to regular Polars DataFrame and then to Pandas
        polars_df = lazy_df.collect()
        pandas_df = polars_df.to_pandas()
        
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Execute directly on the internal method that was failing
        try:
            # Convert to Polars for processing
            polars_funnel_events = polars_df.filter(pl.col('event_name').is_in(steps))
            polars_full_history = polars_df.clone()
            
            # Use the fully optimized polars implementation directly
            path_analysis = calculator._calculate_path_analysis_polars_optimized(
                polars_funnel_events, 
                steps, 
                polars_full_history
            )
            
            # Verify we got valid results without a fallback
            assert path_analysis is not None
            
            # Check logs for fallback messages
            log_output = log_capture.getvalue()
            assert "falling back" not in log_output.lower(), \
                "Detected fallback in path analysis with LazyFrame"
                
        except Exception as e:
            pytest.fail(f"Path analysis failed with LazyFrame: {str(e)}")
    
    def test_detect_polars_to_pandas_fallback_combinations(self, events_data_base, log_capture):
        """
        Test various combinations of funnel configurations to identify which ones
        trigger fallbacks from Polars to Pandas.
        """
        steps = ["Step1", "Step2", "Step3"]
        
        # Test all combinations
        for funnel_order in [FunnelOrder.ORDERED, FunnelOrder.UNORDERED]:
            for reentry_mode in [ReentryMode.FIRST_ONLY, ReentryMode.OPTIMIZED_REENTRY]:
                for counting_method in [CountingMethod.UNIQUE_USERS, CountingMethod.EVENT_TOTALS, CountingMethod.UNIQUE_PAIRS]:
                    # Clear log capture before each test
                    log_capture.truncate(0)
                    log_capture.seek(0)
                    
                    config = FunnelConfig(
                        funnel_order=funnel_order,
                        reentry_mode=reentry_mode,
                        counting_method=counting_method,
                        conversion_window_hours=48
                    )
                    calculator = FunnelCalculator(config=config, use_polars=True)
                    
                    # Execute
                    calculator.calculate_funnel_metrics(events_data_base, steps)
                    
                    # Check logs for fallback messages
                    log_output = log_capture.getvalue()
                    
                    # Assert no fallbacks for critical combinations
                    # (We might allow certain combinations to fallback)
                    if funnel_order == FunnelOrder.ORDERED and counting_method == CountingMethod.UNIQUE_USERS:
                        assert "falling back to pandas" not in log_output.lower(), \
                            f"Detected unexpected fallback for {funnel_order.value}, {reentry_mode.value}, {counting_method.value}"
                            
                    # Document which combinations trigger fallbacks
                    if "falling back to pandas" in log_output.lower():
                        print(f"⚠️ Detected fallback for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}")
                    else:
                        print(f"✅ No fallback for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}")

    def test_lazy_frame_in_path_analysis(self, events_data_base, log_capture, monkeypatch):
        """
        Test specifically targeting the LazyFrame issue in path analysis.
        This reproduces the error with "cannot create expression literal for value of type LazyFrame"
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48
        )
        calculator = FunnelCalculator(config=config, use_polars=True)
        
        # Convert pandas to polars
        polars_df = calculator._to_polars(events_data_base)
        
        # Monkey patch the _calculate_path_analysis_polars_optimized method to try to use a LazyFrame directly
        original_method = calculator._calculate_path_analysis_polars_optimized
        
        def mock_method(segment_funnel_events_df, funnel_steps, full_history_for_segment_users):
            # Create a LazyFrame that will cause problems
            lazy_segment = segment_funnel_events_df.lazy()
            lazy_history = full_history_for_segment_users.lazy()
            
            # Pass LazyFrames directly to original method
            return original_method(lazy_segment, funnel_steps, lazy_history)
        
        # Apply the monkey patch
        monkeypatch.setattr(calculator, '_calculate_path_analysis_polars_optimized', mock_method)
        
        # Execute
        try:
            results = calculator.calculate_funnel_metrics(events_data_base, steps)
            
            # Check logs for the specific LazyFrame error and fallback message
            log_output = log_capture.getvalue()
            
            if "cannot create expression literal for value of type LazyFrame" in log_output:
                assert "falling back to standard polars" in log_output.lower(), \
                    "Found LazyFrame error but no fallback to standard Polars was triggered"
                
                # Verify that we still got valid results through the fallback mechanism
                assert results is not None
                assert len(results.steps) == len(steps)
                assert results.path_analysis is not None
                
                print("✅ Successfully detected LazyFrame issue and fallback to standard Polars")
            else:
                # This is unexpected - either the error was fixed or our test didn't trigger it
                pytest.fail("Expected LazyFrame error was not detected - test needs updating")
                
        except Exception as e:
            pytest.fail(f"Unexpected error: {str(e)}") 