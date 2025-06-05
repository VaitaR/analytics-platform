"""
Test different counting methods for funnel analysis.
Tests unique_users, event_totals, and unique_pairs counting methods.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import json
from app import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.counting_method
class TestCountingMethods:
    
    def test_unique_users_method(self, calculator_factory, base_timestamp):
        """
        Test unique_users counting method.
        Users should be counted only once per step, regardless of multiple events.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User 1: Multiple Sign Up events, single Email Verification
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=10),  # Duplicate
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: Single Sign Up, multiple Email Verification events
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=40),  # Duplicate
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 3: Only Sign Up (for comparison)
            {
                'user_id': 'user_003',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should count each user only once per step
        assert results.users_count[0] == 3  # All 3 users signed up
        assert results.users_count[1] == 2  # Only users 1 and 2 verified email
        assert results.conversion_rates[1] == 66.67  # 2/3 ≈ 66.67%
    
    def test_event_totals_method(self, calculator_factory, base_timestamp):
        """
        Test event_totals counting method.
        Should count total number of events, not unique users.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.EVENT_TOTALS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User 1: 2 Sign Up events, 1 Email Verification
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=10),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: 1 Sign Up, 3 Email Verification events
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=40),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=50),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should count total events
        assert results.users_count[0] == 3  # 2 + 1 Sign Up events
        assert results.users_count[1] == 4  # 1 + 3 Email Verification events
        assert results.conversion_rates[1] == 133.33  # 4/3 ≈ 133.33%
    
    def test_unique_pairs_method(self, calculator_factory, base_timestamp):
        """
        Test unique_pairs counting method.
        Should count step-to-step conversions, allowing users to be counted multiple times
        if they restart the funnel.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_PAIRS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User 1: Completes full sequence once
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
                'timestamp': base_timestamp + timedelta(minutes=60),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: Completes A->B but not B->C
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
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 3: Completes only A
            {
                'user_id': 'user_003',
                'event_name': 'Step A',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B', 'Step C']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # For unique pairs, conversion rates should be step-to-step
        assert results.users_count[0] == 3  # All users complete Step A
        assert results.users_count[1] == 2  # Users 1,2 complete Step B
        assert results.users_count[2] == 1  # Only user 1 completes Step C
        
        # Overall conversion rates from first step
        assert results.conversion_rates[0] == 100.0
        assert results.conversion_rates[1] == 66.67  # 2/3
        assert results.conversion_rates[2] == 33.33  # 1/3
    
    def test_counting_method_comparison_same_data(self, calculator_factory, base_timestamp):
        """
        Test that different counting methods produce different results on the same data.
        """
        # Create data with multiple events per user
        events = [
            # User 1: 2 Sign Up, 2 Email Verification
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=5),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=35),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: 1 Sign Up, 3 Email Verification
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=35),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=40),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        # Test unique users method
        config_unique = FunnelConfig(counting_method=CountingMethod.UNIQUE_USERS)
        calculator_unique = calculator_factory(config_unique)
        results_unique = calculator_unique.calculate_funnel_metrics(data, steps)
        
        # Test event totals method
        config_totals = FunnelConfig(counting_method=CountingMethod.EVENT_TOTALS)
        calculator_totals = calculator_factory(config_totals)
        results_totals = calculator_totals.calculate_funnel_metrics(data, steps)
        
        # Unique users: should count each user once
        assert results_unique.users_count == [2, 2]  # 2 users each step
        
        # Event totals: should count all events
        assert results_totals.users_count == [3, 5]  # 3 sign ups, 5 verifications
        
        # Results should be different
        assert results_unique.users_count != results_totals.users_count
    
    def test_unique_users_with_multiple_attempts(self, calculator_factory, base_timestamp):
        """
        Test unique_users method specifically with users who make multiple attempts.
        Should count user only once per step regardless of attempts.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = [
            # User attempts the same step multiple times
            {
                'user_id': 'user_persistent',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_persistent',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=10),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_persistent',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=20),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_persistent',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_persistent',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=35),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should count the user only once per step
        assert results.users_count[0] == 1
        assert results.users_count[1] == 1
        assert results.conversion_rates[1] == 100.0
    
    def test_event_totals_comprehensive(self, calculator_factory, base_timestamp):
        """
        Comprehensive test of event_totals method with various scenarios.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.EVENT_TOTALS,
            reentry_mode=ReentryMode.FIRST_ONLY
        )
        calculator = calculator_factory(config)
        
        events = []
        
        # User 1: 5 events for step 1, 3 events for step 2
        for i in range(5):
            events.append({
                'user_id': 'user_001',
                'event_name': 'Step 1',
                'timestamp': base_timestamp + timedelta(minutes=i*5),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        for i in range(3):
            events.append({
                'user_id': 'user_001',
                'event_name': 'Step 2',
                'timestamp': base_timestamp + timedelta(minutes=30 + i*5),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        # User 2: 2 events for step 1, 4 events for step 2
        for i in range(2):
            events.append({
                'user_id': 'user_002',
                'event_name': 'Step 1',
                'timestamp': base_timestamp + timedelta(minutes=i*5),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        for i in range(4):
            events.append({
                'user_id': 'user_002',
                'event_name': 'Step 2',
                'timestamp': base_timestamp + timedelta(minutes=30 + i*5),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        data = pd.DataFrame(events)
        steps = ['Step 1', 'Step 2']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should count all events
        assert results.users_count[0] == 7  # 5 + 2 events for Step 1
        assert results.users_count[1] == 7  # 3 + 4 events for Step 2
        assert results.conversion_rates[1] == 100.0  # 7/7
    
    def test_counting_method_edge_cases(self, calculator_factory, base_timestamp):
        """
        Test edge cases for counting methods.
        """
        # Test with no events for second step
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Step 1',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Step 1',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
            # No Step 2 events
        ]
        
        data = pd.DataFrame(events)
        steps = ['Step 1', 'Step 2']
        
        # Test all methods
        for method in [CountingMethod.UNIQUE_USERS, CountingMethod.EVENT_TOTALS, CountingMethod.UNIQUE_PAIRS]:
            config = FunnelConfig(counting_method=method)
            calculator = calculator_factory(config)
            results = calculator.calculate_funnel_metrics(data, steps)
            
            assert results.users_count[0] == 2  # Two Step 1 events/users
            assert results.users_count[1] == 0  # No Step 2 events/users
            assert results.conversion_rates[1] == 0.0 