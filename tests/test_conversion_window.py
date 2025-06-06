"""
Test conversion window functionality.
Tests time-based filtering for funnel steps.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import json
from app import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.conversion_window
class TestConversionWindow:
    
    def test_events_within_conversion_window(self, calculator_factory, base_timestamp):
        """
        Test that users are counted when events occur within the conversion window.
        Using 1-hour conversion window.
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        # User completes both steps within 1 hour
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),  # 30 minutes later
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1  # User completes first step
        assert results.users_count[1] == 1  # User completes second step within window
        assert results.conversion_rates[1] == 100.0
    
    def test_events_outside_conversion_window(self, calculator_factory, base_timestamp):
        """
        Test that users are not counted when events occur outside the conversion window.
        Using 1-hour conversion window.
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        # User completes second step after 1.5 hours (outside window)
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=90),  # 1.5 hours later
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1  # User completes first step
        assert results.users_count[1] == 0  # User does not complete second step within window
        assert results.conversion_rates[1] == 0.0
    
    def test_events_at_conversion_window_boundary(self, calculator_factory, base_timestamp):
        """
        Test boundary condition: event occurs exactly at the conversion window limit.
        Should be included (≤ boundary).
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        # User completes second step exactly 1 hour later
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(hours=1),  # Exactly 1 hour later
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1
        assert results.users_count[1] == 0  # Should NOT be included at boundary (exclusive)
        assert results.conversion_rates[1] == 0.0
    
    def test_sequential_conversion_windows(self, calculator_factory, base_timestamp):
        """
        Test that conversion windows are calculated sequentially (B->C window starts from B, not A).
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User 1: Completes A->B within window, B->C within window
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=30),  # 30 min after A
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step C',
                'timestamp': base_timestamp + timedelta(minutes=60),  # 30 min after B
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: Completes A->B within window, but B->C outside window
            {
                'user_id': 'user_002',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=30),  # 30 min after A
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Step C',
                'timestamp': base_timestamp + timedelta(minutes=120),  # 90 min after B (outside window)
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B', 'Step C']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 2  # Both users complete Step A
        assert results.users_count[1] == 2  # Both users complete Step B within window
        assert results.users_count[2] == 1  # Only user_001 completes Step C within window from Step B
    
    def test_different_conversion_window_sizes(self, calculator_factory, base_timestamp):
        """
        Test different conversion window sizes affect results correctly.
        """
        # Test data with events at 30 min, 90 min, and 150 min intervals
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step C',
                'timestamp': base_timestamp + timedelta(minutes=90),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B', 'Step C']
        
        # Test with 1-hour window
        config_1h = FunnelConfig(conversion_window_hours=1)
        calculator_1h = calculator_factory(config_1h)
        results_1h = calculator_1h.calculate_funnel_metrics(data, steps)
        
        # Test with 2-hour window
        config_2h = FunnelConfig(conversion_window_hours=2)
        calculator_2h = calculator_factory(config_2h)
        results_2h = calculator_2h.calculate_funnel_metrics(data, steps)
        
        # With 1-hour window: A->B should work (30 min), B->C should fail (60 min)
        assert results_1h.users_count == [1, 1, 0]
        
        # With 2-hour window: Both A->B and B->C should work
        assert results_2h.users_count == [1, 1, 1]
    
    def test_multiple_events_first_within_window(self, calculator_factory, base_timestamp):
        """
        Test when user has multiple events of same type, first one is within window.
        Should count the user based on first valid occurrence.
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=30),  # First B within window
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=90),  # Second B outside window
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1
        assert results.users_count[1] == 1  # Should count based on first occurrence
    
    def test_multiple_events_first_outside_window(self, calculator_factory, base_timestamp):
        """
        Test when user has multiple events of same type, first one is outside window.
        With FIRST_ONLY mode, should not count the user.
        """
        config = FunnelConfig(
            conversion_window_hours=1,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=90),  # First B outside window
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=30),  # Second B within window but after first
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1
        assert results.users_count[1] == 0  # Should not count with FIRST_ONLY mode
    
    def test_zero_conversion_window(self, calculator_factory, base_timestamp):
        """
        Test behavior with zero conversion window (events must be simultaneous).
        """
        config = FunnelConfig(
            conversion_window_hours=0,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User 1: Events at same timestamp
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp,  # Same timestamp
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: Events 1 minute apart
            {
                'user_id': 'user_002',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=1),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 2
        assert results.users_count[1] == 1  # Only user_001 with simultaneous events
    
    def test_very_large_conversion_window(self, calculator_factory, base_timestamp):
        """
        Test with very large conversion window (should include all events).
        """
        config = FunnelConfig(
            conversion_window_hours=8760,  # 1 year
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(days=30),  # 30 days later
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 1
        assert results.users_count[1] == 1  # Should include even with 30-day gap 