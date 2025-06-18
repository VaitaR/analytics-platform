#!/usr/bin/env python3
"""
Mathematical precision tests for Time Series Analysis calculations.

This module ensures mathematical accuracy of time series calculations by:
- Testing precise cohort tracking with known data patterns
- Validating conversion rate calculations across different time periods
- Ensuring boundary condition handling (hour/day boundaries)
- Testing aggregation consistency (hourly -> daily -> weekly)
- Validating time window enforcement for conversions
- Edge case coverage for empty periods and single-user scenarios
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import FunnelCalculator, FunnelVisualizer
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.timeseries
class TestTimeSeriesMathematicalPrecision:
    """Mathematical precision tests for time series calculations."""
    
    @pytest.fixture
    def controlled_cohort_data(self):
        """Create precisely controlled cohort data for mathematical validation."""
        events = []
        base_date = datetime(2024, 1, 1, 9, 0, 0)  # Start at 9 AM sharp
        
        # Day 1: 1000 users with exact 70% -> 50% -> 30% conversion pattern
        day1_start = base_date
        for i in range(1000):
            user_id = f'day1_{i:04d}'
            
            # All 1000 users sign up
            events.append({
                'user_id': user_id,
                'event_name': 'Signup',
                'timestamp': day1_start + timedelta(minutes=i % 60),  # Spread over 1 hour
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            # Exactly 700 users (70%) verify email
            if i < 700:
                events.append({
                    'user_id': user_id,
                    'event_name': 'Verify',
                    'timestamp': day1_start + timedelta(minutes=(i % 60) + 30),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # Exactly 500 users (50% of original) complete profile
                if i < 500:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'Complete',
                        'timestamp': day1_start + timedelta(minutes=(i % 60) + 60),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                    
                    # Exactly 300 users (30% of original) make purchase
                    if i < 300:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'Purchase',
                            'timestamp': day1_start + timedelta(minutes=(i % 60) + 90),
                            'event_properties': '{}',
                            'user_properties': '{}'
                        })
        
        # Day 2: 500 users with different pattern (80% -> 60% -> 40%)
        day2_start = base_date + timedelta(days=1)
        for i in range(500):
            user_id = f'day2_{i:04d}'
            
            # All 500 users sign up
            events.append({
                'user_id': user_id,
                'event_name': 'Signup',
                'timestamp': day2_start + timedelta(minutes=i % 60),
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            # Exactly 400 users (80%) verify email
            if i < 400:
                events.append({
                    'user_id': user_id,
                    'event_name': 'Verify',
                    'timestamp': day2_start + timedelta(minutes=(i % 60) + 30),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # Exactly 300 users (60% of original) complete profile
                if i < 300:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'Complete',
                        'timestamp': day2_start + timedelta(minutes=(i % 60) + 60),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                    
                    # Exactly 200 users (40% of original) make purchase
                    if i < 200:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'Purchase',
                            'timestamp': day2_start + timedelta(minutes=(i % 60) + 90),
                            'event_properties': '{}',
                            'user_properties': '{}'
                        })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def boundary_test_data(self):
        """Create data that tests hour/day boundary handling."""
        events = []
        
        # Events right at hour boundaries
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        for hour in [0, 1, 23]:  # Test midnight, 1 AM, and 11 PM
            for minute in [0, 59]:  # Test beginning and end of hour
                timestamp = base_time + timedelta(hours=hour, minutes=minute)
                user_id = f'boundary_h{hour}_m{minute}'
                
                events.append({
                    'user_id': user_id,
                    'event_name': 'Start',
                    'timestamp': timestamp,
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                events.append({
                    'user_id': user_id,
                    'event_name': 'Middle',
                    'timestamp': timestamp + timedelta(minutes=15),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                events.append({
                    'user_id': user_id,
                    'event_name': 'End',
                    'timestamp': timestamp + timedelta(minutes=30),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def conversion_window_test_data(self):
        """Create data to test conversion window enforcement precisely."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # User 1: Completes within 1-hour window (should count)
        user1_events = [
            {'user_id': 'within_1h', 'event_name': 'Start', 'timestamp': base_time},
            {'user_id': 'within_1h', 'event_name': 'Middle', 'timestamp': base_time + timedelta(minutes=30)},
            {'user_id': 'within_1h', 'event_name': 'End', 'timestamp': base_time + timedelta(minutes=59)},
        ]
        
        # User 2: Exceeds 1-hour window (should not count as completed)
        user2_events = [
            {'user_id': 'exceeds_1h', 'event_name': 'Start', 'timestamp': base_time},
            {'user_id': 'exceeds_1h', 'event_name': 'Middle', 'timestamp': base_time + timedelta(minutes=30)},
            {'user_id': 'exceeds_1h', 'event_name': 'End', 'timestamp': base_time + timedelta(minutes=61)},
        ]
        
        # User 3: Multiple attempts, only first should count (FIRST_ONLY mode)
        user3_events = [
            {'user_id': 'multi_start', 'event_name': 'Start', 'timestamp': base_time},
            {'user_id': 'multi_start', 'event_name': 'Start', 'timestamp': base_time + timedelta(hours=2)},  # Second start
            {'user_id': 'multi_start', 'event_name': 'Middle', 'timestamp': base_time + timedelta(hours=2, minutes=15)},
            {'user_id': 'multi_start', 'event_name': 'End', 'timestamp': base_time + timedelta(hours=2, minutes=30)},
        ]
        
        for event_list in [user1_events, user2_events, user3_events]:
            for event in event_list:
                events.append({**event, 'event_properties': '{}', 'user_properties': '{}'})
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def funnel_steps_4(self):
        """4-step funnel for controlled testing."""
        return ['Signup', 'Verify', 'Complete', 'Purchase']
    
    @pytest.fixture
    def funnel_steps_3(self):
        """3-step funnel for boundary testing."""
        return ['Start', 'Middle', 'End']
    
    @pytest.fixture
    def long_window_calculator(self):
        """Calculator with long conversion window (1 week)."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
            conversion_window_hours=168  # 1 week
        )
        return FunnelCalculator(config, use_polars=True)
    
    @pytest.fixture
    def short_window_calculator(self):
        """Calculator with 1-hour conversion window."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
            conversion_window_hours=1  # 1 hour
        )
        return FunnelCalculator(config, use_polars=True)
    
    @pytest.fixture
    def multi_week_month_data(self):
        """Create data spanning 8-10 weeks across 2-3 months for weekly/monthly aggregation testing."""
        events = []
        base_date = datetime(2024, 1, 15, 9, 0, 0)  # Start mid-January
        
        # Generate data for 10 weeks (70 days) spanning Jan-Mar
        for week in range(10):
            week_start = base_date + timedelta(weeks=week)
            
            # Each week has varying number of users (50-150)
            users_this_week = 50 + (week * 10)
            
            for i in range(users_this_week):
                user_id = f'week{week:02d}_user{i:04d}'
                
                # User starts funnel at random time during the week
                start_time = week_start + timedelta(
                    days=i % 7, 
                    hours=(i * 7) % 24,
                    minutes=(i * 13) % 60
                )
                
                # All users sign up
                events.append({
                    'user_id': user_id,
                    'event_name': 'Signup',
                    'timestamp': start_time,
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # 80% verify email (within 2 hours)
                if i < users_this_week * 0.8:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'Verify',
                        'timestamp': start_time + timedelta(hours=1),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                    
                    # 60% complete profile (within 4 hours)
                    if i < users_this_week * 0.6:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'Complete',
                            'timestamp': start_time + timedelta(hours=3),
                            'event_properties': '{}',
                            'user_properties': '{}'
                        })
                        
                        # 40% make purchase (within 8 hours)
                        if i < users_this_week * 0.4:
                            events.append({
                                'user_id': user_id,
                                'event_name': 'Purchase',
                                'timestamp': start_time + timedelta(hours=6),
                                'event_properties': '{}',
                                'user_properties': '{}'
                            })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def skipped_steps_data(self):
        """Create data where users skip middle steps in the funnel."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # User 1: Complete all steps in order (A->B->C)
        user1_events = [
            {'user_id': 'complete_user', 'event_name': 'Step A', 'timestamp': base_time},
            {'user_id': 'complete_user', 'event_name': 'Step B', 'timestamp': base_time + timedelta(minutes=30)},
            {'user_id': 'complete_user', 'event_name': 'Step C', 'timestamp': base_time + timedelta(minutes=60)},
        ]
        
        # User 2: Skip middle step (A->C, missing B)
        user2_events = [
            {'user_id': 'skip_user', 'event_name': 'Step A', 'timestamp': base_time + timedelta(hours=1)},
            {'user_id': 'skip_user', 'event_name': 'Step C', 'timestamp': base_time + timedelta(hours=1, minutes=45)},
        ]
        
        # User 3: Complete all steps but out of order (A->C->B)
        user3_events = [
            {'user_id': 'out_of_order_user', 'event_name': 'Step A', 'timestamp': base_time + timedelta(hours=2)},
            {'user_id': 'out_of_order_user', 'event_name': 'Step C', 'timestamp': base_time + timedelta(hours=2, minutes=20)},
            {'user_id': 'out_of_order_user', 'event_name': 'Step B', 'timestamp': base_time + timedelta(hours=2, minutes=40)},
        ]
        
        # User 4: Another skip pattern (starts at A, jumps to C)
        user4_events = [
            {'user_id': 'another_skip_user', 'event_name': 'Step A', 'timestamp': base_time + timedelta(hours=3)},
            {'user_id': 'another_skip_user', 'event_name': 'Step C', 'timestamp': base_time + timedelta(hours=3, minutes=30)},
        ]
        
        for event_list in [user1_events, user2_events, user3_events, user4_events]:
            for event in event_list:
                events.append({**event, 'event_properties': '{}', 'user_properties': '{}'})
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def skipped_steps_funnel(self):
        """3-step funnel for skipped steps testing."""
        return ['Step A', 'Step B', 'Step C']
    
    @pytest.fixture
    def optimized_reentry_calculator(self):
        """Calculator with optimized reentry mode for comparison tests."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.OPTIMIZED_REENTRY,
            funnel_order=FunnelOrder.ORDERED,
            conversion_window_hours=168  # 1 week
        )
        return FunnelCalculator(config, use_polars=True)

    def test_exact_cohort_calculation(self, long_window_calculator, controlled_cohort_data, funnel_steps_4):
        """Test exact cohort calculation with precise known data."""
        result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1d'
        )
        
        assert len(result) == 2, f"Expected exactly 2 days, got {len(result)}"
        
        # Day 1: 1000 -> 700 -> 500 -> 300
        day1 = result.iloc[0]
        assert day1['started_funnel_users'] == 1000, f"Day 1: Expected 1000 starters, got {day1['started_funnel_users']}"
        assert day1['Signup_users'] == 1000, f"Day 1: Expected 1000 Signup users, got {day1['Signup_users']}"
        assert day1['Verify_users'] == 700, f"Day 1: Expected 700 Verify users, got {day1['Verify_users']}"
        assert day1['Complete_users'] == 500, f"Day 1: Expected 500 Complete users, got {day1['Complete_users']}"
        assert day1['Purchase_users'] == 300, f"Day 1: Expected 300 Purchase users, got {day1['Purchase_users']}"
        assert day1['completed_funnel_users'] == 300, f"Day 1: Expected 300 completers, got {day1['completed_funnel_users']}"
        assert abs(day1['conversion_rate'] - 30.0) < 0.01, f"Day 1: Expected 30% conversion, got {day1['conversion_rate']}"
        
        # Day 2: 500 -> 400 -> 300 -> 200
        day2 = result.iloc[1]
        assert day2['started_funnel_users'] == 500, f"Day 2: Expected 500 starters, got {day2['started_funnel_users']}"
        assert day2['Signup_users'] == 500, f"Day 2: Expected 500 Signup users, got {day2['Signup_users']}"
        assert day2['Verify_users'] == 400, f"Day 2: Expected 400 Verify users, got {day2['Verify_users']}"
        assert day2['Complete_users'] == 300, f"Day 2: Expected 300 Complete users, got {day2['Complete_users']}"
        assert day2['Purchase_users'] == 200, f"Day 2: Expected 200 Purchase users, got {day2['Purchase_users']}"
        assert day2['completed_funnel_users'] == 200, f"Day 2: Expected 200 completers, got {day2['completed_funnel_users']}"
        assert abs(day2['conversion_rate'] - 40.0) < 0.01, f"Day 2: Expected 40% conversion, got {day2['conversion_rate']}"
        
        # Test step-to-step conversion rates
        # Day 1 rates: 70%, 71.43% (500/700), 60% (300/500)
        assert abs(day1['Signup_to_Verify_rate'] - 70.0) < 0.01, f"Day 1: Expected 70% Signup->Verify, got {day1['Signup_to_Verify_rate']}"
        assert abs(day1['Verify_to_Complete_rate'] - 71.43) < 0.1, f"Day 1: Expected 71.43% Verify->Complete, got {day1['Verify_to_Complete_rate']}"
        assert abs(day1['Complete_to_Purchase_rate'] - 60.0) < 0.01, f"Day 1: Expected 60% Complete->Purchase, got {day1['Complete_to_Purchase_rate']}"
        
        # Day 2 rates: 80%, 75% (300/400), 66.67% (200/300)
        assert abs(day2['Signup_to_Verify_rate'] - 80.0) < 0.01, f"Day 2: Expected 80% Signup->Verify, got {day2['Signup_to_Verify_rate']}"
        assert abs(day2['Verify_to_Complete_rate'] - 75.0) < 0.01, f"Day 2: Expected 75% Verify->Complete, got {day2['Verify_to_Complete_rate']}"
        assert abs(day2['Complete_to_Purchase_rate'] - 66.67) < 0.1, f"Day 2: Expected 66.67% Complete->Purchase, got {day2['Complete_to_Purchase_rate']}"
        
        print("✅ Exact cohort calculation test passed with mathematical precision")
    
    def test_conversion_window_enforcement_precise(self, short_window_calculator, conversion_window_test_data, funnel_steps_3):
        """Test precise conversion window enforcement."""
        result = short_window_calculator.calculate_timeseries_metrics(
            conversion_window_test_data, funnel_steps_3, aggregation_period='1d'
        )
        
        assert len(result) == 1, f"Expected 1 day of data, got {len(result)}"
        
        day_result = result.iloc[0]
        
        # All 3 users started the funnel
        assert day_result['started_funnel_users'] == 3, f"Expected 3 starters, got {day_result['started_funnel_users']}"
        
        # Only 1 user (within_1h) should complete within the 1-hour window
        # - within_1h: completes in 59 minutes (valid)
        # - exceeds_1h: completes in 61 minutes (invalid)
        # - multi_start: only first start counts, but doesn't complete within window from first start
        assert day_result['completed_funnel_users'] == 1, f"Expected 1 completer, got {day_result['completed_funnel_users']}"
        
        # Conversion rate should be 1/3 = 33.33%
        expected_rate = 33.33
        assert abs(day_result['conversion_rate'] - expected_rate) < 0.1, f"Expected {expected_rate}% conversion, got {day_result['conversion_rate']}"
        
        print("✅ Conversion window enforcement precise test passed")
    
    def test_hourly_boundary_handling(self, long_window_calculator, boundary_test_data, funnel_steps_3):
        """Test handling of hour boundaries precisely."""
        result = long_window_calculator.calculate_timeseries_metrics(
            boundary_test_data, funnel_steps_3, aggregation_period='1h'
        )
        
        # Should have data for hours with events (may include boundary hours)
        assert 3 <= len(result) <= 6, f"Expected 3-6 hours of data, got {len(result)}"
        
        # Each hour with events should have the correct number of users
        total_starters = result['started_funnel_users'].sum()
        total_completers = result['completed_funnel_users'].sum()
        
        # We created 6 users (3 hours × 2 minutes each), all should complete
        assert total_starters == 6, f"Expected 6 total starters, got {total_starters}"
        assert total_completers == 6, f"Expected 6 total completers, got {total_completers}"
        
        # All periods with starters should have 100% conversion since everyone completes within same period
        for i, row in result.iterrows():
            if row['started_funnel_users'] > 0:
                assert abs(row['conversion_rate'] - 100.0) < 0.01, f"Hour {i}: Expected 100% conversion, got {row['conversion_rate']}"
        
        print("✅ Hourly boundary handling test passed")
    
    def test_aggregation_period_consistency(self, long_window_calculator, controlled_cohort_data, funnel_steps_4):
        """Test consistency across different aggregation periods."""
        # Get daily data
        daily_result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1d'
        )
        
        # Calculate total metrics
        total_daily_starters = daily_result['started_funnel_users'].sum()
        total_daily_completers = daily_result['completed_funnel_users'].sum()
        
        # Expected: 1000 + 500 = 1500 starters, 300 + 200 = 500 completers
        assert total_daily_starters == 1500, f"Expected 1500 total starters, got {total_daily_starters}"
        assert total_daily_completers == 500, f"Expected 500 total completers, got {total_daily_completers}"
        
        # Overall conversion rate should be 500/1500 = 33.33%
        overall_rate = (total_daily_completers / total_daily_starters) * 100
        assert abs(overall_rate - 33.33) < 0.1, f"Expected 33.33% overall conversion, got {overall_rate}"
        
        print("✅ Aggregation period consistency test passed")
    
    def test_empty_period_handling(self, long_window_calculator, funnel_steps_4):
        """Test handling of periods with no data."""
        # Create sparse data with gaps
        events = []
        base_date = datetime(2024, 1, 1, 10, 0, 0)
        
        # Only add data for specific hours (creating gaps)
        for hour in [0, 5, 10]:  # Hours with data
            timestamp = base_date + timedelta(hours=hour)
            user_id = f'sparse_user_{hour}'
            
            events.append({
                'user_id': user_id,
                'event_name': 'Signup',
                'timestamp': timestamp,
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            events.append({
                'user_id': user_id,
                'event_name': 'Verify',
                'timestamp': timestamp + timedelta(minutes=10),
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            events.append({
                'user_id': user_id,
                'event_name': 'Complete',
                'timestamp': timestamp + timedelta(minutes=20),
                'event_properties': '{}',
                'user_properties': '{}'
            })
            
            events.append({
                'user_id': user_id,
                'event_name': 'Purchase',
                'timestamp': timestamp + timedelta(minutes=30),
                'event_properties': '{}',
                'user_properties': '{}'
            })
        
        sparse_df = pd.DataFrame(events)
        
        result = long_window_calculator.calculate_timeseries_metrics(
            sparse_df, funnel_steps_4, aggregation_period='1h'
        )
        
        # Should have exactly 3 periods (only hours with data)
        assert len(result) == 3, f"Expected 3 periods with data, got {len(result)}"
        
        # Each period should have 1 starter and 1 completer
        for i, row in result.iterrows():
            assert row['started_funnel_users'] == 1, f"Period {i}: Expected 1 starter, got {row['started_funnel_users']}"
            assert row['completed_funnel_users'] == 1, f"Period {i}: Expected 1 completer, got {row['completed_funnel_users']}"
            assert abs(row['conversion_rate'] - 100.0) < 0.01, f"Period {i}: Expected 100% conversion, got {row['conversion_rate']}"
        
        print("✅ Empty period handling test passed")
    
    def test_single_user_edge_case(self, long_window_calculator, funnel_steps_4):
        """Test edge case with single user."""
        events = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Single user completes full funnel
        single_user_events = [
            {'user_id': 'solo_user', 'event_name': 'Signup', 'timestamp': base_time},
            {'user_id': 'solo_user', 'event_name': 'Verify', 'timestamp': base_time + timedelta(minutes=10)},
            {'user_id': 'solo_user', 'event_name': 'Complete', 'timestamp': base_time + timedelta(minutes=20)},
            {'user_id': 'solo_user', 'event_name': 'Purchase', 'timestamp': base_time + timedelta(minutes=30)},
        ]
        
        for event in single_user_events:
            events.append({**event, 'event_properties': '{}', 'user_properties': '{}'})
        
        single_user_df = pd.DataFrame(events)
        
        result = long_window_calculator.calculate_timeseries_metrics(
            single_user_df, funnel_steps_4, aggregation_period='1d'
        )
        
        assert len(result) == 1, f"Expected 1 day of data, got {len(result)}"
        
        day_result = result.iloc[0]
        assert day_result['started_funnel_users'] == 1, f"Expected 1 starter, got {day_result['started_funnel_users']}"
        assert day_result['completed_funnel_users'] == 1, f"Expected 1 completer, got {day_result['completed_funnel_users']}"
        assert abs(day_result['conversion_rate'] - 100.0) < 0.01, f"Expected 100% conversion, got {day_result['conversion_rate']}"
        
        # All step counts should be 1
        for step in funnel_steps_4:
            step_count = day_result[f'{step}_users']
            assert step_count == 1, f"Expected 1 user for {step}, got {step_count}"
        
        print("✅ Single user edge case test passed")
    
    def test_mathematical_properties_validation(self, long_window_calculator, controlled_cohort_data, funnel_steps_4):
        """Test fundamental mathematical properties of funnel calculations."""
        result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1d'
        )
        
        for i, row in result.iterrows():
            # Property 1: Completed users ≤ Started users
            assert row['completed_funnel_users'] <= row['started_funnel_users'], \
                f"Day {i}: Completers ({row['completed_funnel_users']}) > Starters ({row['started_funnel_users']})"
            
            # Property 2: Conversion rate = (Completers / Starters) * 100
            if row['started_funnel_users'] > 0:
                expected_rate = (row['completed_funnel_users'] / row['started_funnel_users']) * 100
                assert abs(row['conversion_rate'] - expected_rate) < 0.01, \
                    f"Day {i}: Conversion rate mismatch - expected {expected_rate}, got {row['conversion_rate']}"
            
            # Property 3: Funnel monotonicity (each step ≤ previous step)
            step_counts = [row[f'{step}_users'] for step in funnel_steps_4]
            for j in range(1, len(step_counts)):
                assert step_counts[j] <= step_counts[j-1], \
                    f"Day {i}: Step {j} ({step_counts[j]}) > Step {j-1} ({step_counts[j-1]}) - violates funnel monotonicity"
            
            # Property 4: All rates should be in [0, 100]
            assert 0 <= row['conversion_rate'] <= 100, \
                f"Day {i}: Conversion rate {row['conversion_rate']} outside [0, 100] range"
            
            # Property 5: Step-to-step rates consistency
            for j in range(len(funnel_steps_4) - 1):
                step_from = funnel_steps_4[j]
                step_to = funnel_steps_4[j + 1]
                rate_col = f'{step_from}_to_{step_to}_rate'
                
                if rate_col in row:
                    from_count = row[f'{step_from}_users']
                    to_count = row[f'{step_to}_users']
                    
                    if from_count > 0:
                        expected_step_rate = min((to_count / from_count) * 100, 100.0)
                        assert abs(row[rate_col] - expected_step_rate) < 0.01, \
                            f"Day {i}: {rate_col} mismatch - expected {expected_step_rate}, got {row[rate_col]}"
        
        print("✅ Mathematical properties validation test passed")


@pytest.mark.timeseries
@pytest.mark.performance  
class TestTimeSeriesPerformance:
    """Performance tests for time series calculations."""
    
    @pytest.fixture
    def performance_calculator(self):
        """Calculator optimized for performance testing."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
            conversion_window_hours=24
        )
        return FunnelCalculator(config, use_polars=True)
    
    def test_large_dataset_timeseries_performance(self, performance_calculator):
        """Test performance with large datasets."""
        import time
        
        # Generate 50K events over 30 days
        events = []
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        
        for day in range(30):
            day_start = base_date + timedelta(days=day)
            users_per_day = 500 + (day * 10)  # Growing daily volume
            
            for user_idx in range(users_per_day):
                user_id = f'perf_day{day:02d}_user{user_idx:04d}'
                
                # Signup (all users)
                events.append({
                    'user_id': user_id,
                    'event_name': 'Signup',
                    'timestamp': day_start + timedelta(minutes=user_idx % 1440),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                # 75% verify
                if user_idx < users_per_day * 0.75:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'Verify',
                        'timestamp': day_start + timedelta(minutes=(user_idx % 1440) + 30),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                    
                    # 50% complete
                    if user_idx < users_per_day * 0.50:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'Complete',
                            'timestamp': day_start + timedelta(minutes=(user_idx % 1440) + 60),
                            'event_properties': '{}',
                            'user_properties': '{}'
                        })
        
        large_df = pd.DataFrame(events)
        
        print(f"Testing performance with {len(large_df)} events, {large_df['user_id'].nunique()} users, 30 days")
        
        # Time the calculation
        start_time = time.time()
        
        result = performance_calculator.calculate_timeseries_metrics(
            large_df, ['Signup', 'Verify', 'Complete'], aggregation_period='1d'
        )
        
        calculation_time = time.time() - start_time
        
        # Validate results
        assert len(result) == 30, f"Expected 30 days, got {len(result)}"
        
        # Performance requirement: should complete in under 10 seconds for 50K+ events
        assert calculation_time < 10.0, f"Performance too slow: {calculation_time:.2f} seconds for {len(large_df)} events"
        
        # Validate some mathematical properties
        for i, row in result.iterrows():
            assert row['started_funnel_users'] > 0, f"Day {i}: Should have starters"
            assert row['completed_funnel_users'] >= 0, f"Day {i}: Should have non-negative completers"
            assert row['completed_funnel_users'] <= row['started_funnel_users'], f"Day {i}: Completers ≤ starters"
        
        print(f"✅ Large dataset performance test passed in {calculation_time:.2f} seconds")
        print(f"   Processed {len(large_df)} events, {large_df['user_id'].nunique()} users")
        print(f"   Performance: {len(large_df)/calculation_time:.0f} events/second")
    
    def test_weekly_and_monthly_aggregation(self, long_window_calculator, multi_week_month_data, funnel_steps_4):
        """Test weekly and monthly aggregation produces correct results."""
        # Test weekly aggregation
        weekly_result = long_window_calculator.calculate_timeseries_metrics(
            multi_week_month_data, funnel_steps_4, aggregation_period='1w'
        )
        
        # Should have approximately 10 weeks of data
        assert 9 <= len(weekly_result) <= 11, f"Expected ~10 weeks, got {len(weekly_result)}"
        
        # Test monthly aggregation  
        monthly_result = long_window_calculator.calculate_timeseries_metrics(
            multi_week_month_data, funnel_steps_4, aggregation_period='1mo'
        )
        
        # Should have 3 months (Jan, Feb, Mar)
        assert 2 <= len(monthly_result) <= 3, f"Expected 2-3 months, got {len(monthly_result)}"
        
        # Validate totals are consistent
        total_weekly_starters = weekly_result['started_funnel_users'].sum()
        total_weekly_completers = weekly_result['completed_funnel_users'].sum()
        
        total_monthly_starters = monthly_result['started_funnel_users'].sum()
        total_monthly_completers = monthly_result['completed_funnel_users'].sum()
        
        # Weekly and monthly totals should match (within small tolerance for boundary effects)
        starters_diff = abs(total_weekly_starters - total_monthly_starters)
        completers_diff = abs(total_weekly_completers - total_monthly_completers)
        
        assert starters_diff <= 50, f"Weekly/monthly starters mismatch: weekly={total_weekly_starters}, monthly={total_monthly_starters}"
        assert completers_diff <= 30, f"Weekly/monthly completers mismatch: weekly={total_weekly_completers}, monthly={total_monthly_completers}"
        
        # Validate mathematical properties for each period
        for i, row in weekly_result.iterrows():
            assert row['completed_funnel_users'] <= row['started_funnel_users'], \
                f"Week {i}: Completers > starters"
            assert 0 <= row['conversion_rate'] <= 100, \
                f"Week {i}: Invalid conversion rate {row['conversion_rate']}"
        
        for i, row in monthly_result.iterrows():
            assert row['completed_funnel_users'] <= row['started_funnel_users'], \
                f"Month {i}: Completers > starters"
            assert 0 <= row['conversion_rate'] <= 100, \
                f"Month {i}: Invalid conversion rate {row['conversion_rate']}"
        
        print("✅ Weekly and monthly aggregation test passed")
        print(f"   Weekly periods: {len(weekly_result)}, Monthly periods: {len(monthly_result)}")
        print(f"   Total events processed: {len(multi_week_month_data)}")

    @pytest.mark.parametrize("reentry_mode,expected_completers", [
        (ReentryMode.FIRST_ONLY, 1),     # Only within_1h completes within window from first start
        (ReentryMode.OPTIMIZED_REENTRY, 2) # both within_1h and multi_start (from second attempt)
    ])
    def test_conversion_window_enforcement_reentry_modes(self, conversion_window_test_data, funnel_steps_3, reentry_mode, expected_completers):
        """Test conversion window enforcement with different reentry modes."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=reentry_mode,
            funnel_order=FunnelOrder.ORDERED,
            conversion_window_hours=1  # 1 hour
        )
        calculator = FunnelCalculator(config, use_polars=True)
        
        result = calculator.calculate_timeseries_metrics(
            conversion_window_test_data, funnel_steps_3, aggregation_period='1d'
        )
        
        assert len(result) == 1, f"Expected 1 day of data, got {len(result)}"
        
        day_result = result.iloc[0]
        
        # All 3 users should start the funnel regardless of reentry mode
        assert day_result['started_funnel_users'] == 3, f"Expected 3 starters, got {day_result['started_funnel_users']}"
        
        # Completers depend on reentry mode
        assert day_result['completed_funnel_users'] == expected_completers, \
            f"Expected {expected_completers} completers for {reentry_mode.value}, got {day_result['completed_funnel_users']}"
        
        # Conversion rate should match expectations
        expected_rate = (expected_completers / 3) * 100
        assert abs(day_result['conversion_rate'] - expected_rate) < 0.1, \
            f"Expected {expected_rate}% conversion for {reentry_mode.value}, got {day_result['conversion_rate']}"
        
        print(f"✅ Conversion window enforcement test passed for {reentry_mode.value} mode")

    def test_skipped_step_handling(self, long_window_calculator, skipped_steps_data, skipped_steps_funnel):
        """Test handling of users who skip steps in the funnel."""
        result = long_window_calculator.calculate_timeseries_metrics(
            skipped_steps_data, skipped_steps_funnel, aggregation_period='1d'
        )
        
        assert len(result) == 1, f"Expected 1 day of data, got {len(result)}"
        
        day_result = result.iloc[0]
        
        # All 4 users started with Step A
        assert day_result['started_funnel_users'] == 4, \
            f"Expected 4 starters (all users did Step A), got {day_result['started_funnel_users']}"
        
        # Step A: All 4 users
        assert day_result['Step A_users'] == 4, \
            f"Expected 4 users at Step A, got {day_result['Step A_users']}"
        
        # Step B: Only 2 users (complete_user and out_of_order_user)
        # skip_user and another_skip_user skip Step B entirely
        assert day_result['Step B_users'] == 2, \
            f"Expected 2 users at Step B, got {day_result['Step B_users']}"
        
        # Step C: All 4 users eventually reach Step C
        # (complete_user, skip_user, out_of_order_user, another_skip_user)
        assert day_result['Step C_users'] == 4, \
            f"Expected 4 users at Step C, got {day_result['Step C_users']}"
        
        # All 4 users completed the funnel (reached final step)
        assert day_result['completed_funnel_users'] == 4, \
            f"Expected 4 completers, got {day_result['completed_funnel_users']}"
        
        # Conversion rate should be 100% (all starters completed)
        assert abs(day_result['conversion_rate'] - 100.0) < 0.01, \
            f"Expected 100% conversion, got {day_result['conversion_rate']}"
        
        # Step-to-step conversion rates validation
        # A_to_B_rate: 2/4 = 50% (only 2 out of 4 users went from A to B)
        if 'Step A_to_Step B_rate' in day_result:
            assert abs(day_result['Step A_to_Step B_rate'] - 50.0) < 0.1, \
                f"Expected 50% A->B conversion, got {day_result['Step A_to_Step B_rate']}"
        
        # B_to_C_rate: 2/2 = 100% (both users who reached B also reached C)
        if 'Step B_to_Step C_rate' in day_result:
            assert abs(day_result['Step B_to_Step C_rate'] - 100.0) < 0.1, \
                f"Expected 100% B->C conversion, got {day_result['Step B_to_Step C_rate']}"
        
        print("✅ Skipped step handling test passed")
        print(f"   Step A: {day_result['Step A_users']} users")
        print(f"   Step B: {day_result['Step B_users']} users (skipped by 2 users)")
        print(f"   Step C: {day_result['Step C_users']} users")
        print(f"   Final conversion: {day_result['conversion_rate']:.1f}%")

    def test_boundary_conditions_weekly_monthly(self, long_window_calculator, funnel_steps_4):
        """Test boundary conditions for weekly and monthly aggregation periods."""
        events = []
        
        # Create events right at week and month boundaries
        # Week boundary: Sunday -> Monday transition
        # Month boundary: Last day of month -> First day of next month
        
        boundary_times = [
            datetime(2024, 1, 28, 23, 59, 0),  # End of January (week boundary)
            datetime(2024, 1, 29, 0, 1, 0),    # Start of last week of January
            datetime(2024, 1, 31, 23, 58, 0),  # End of January (month boundary)
            datetime(2024, 2, 1, 0, 2, 0),     # Start of February (month boundary)
            datetime(2024, 2, 4, 23, 59, 0),   # End of week 5 (week boundary)
            datetime(2024, 2, 5, 0, 1, 0),     # Start of week 6 (week boundary)
        ]
        
        for i, timestamp in enumerate(boundary_times):
            user_id = f'boundary_user_{i:02d}'
            
            # Each user completes full funnel quickly (within 2 hours)
            for j, step in enumerate(funnel_steps_4):
                events.append({
                    'user_id': user_id,
                    'event_name': step,
                    'timestamp': timestamp + timedelta(minutes=j * 20),
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
        
        boundary_df = pd.DataFrame(events)
        
        # Test weekly aggregation with boundaries
        weekly_result = long_window_calculator.calculate_timeseries_metrics(
            boundary_df, funnel_steps_4, aggregation_period='1w'
        )
        
        # Test monthly aggregation with boundaries
        monthly_result = long_window_calculator.calculate_timeseries_metrics(
            boundary_df, funnel_steps_4, aggregation_period='1mo'
        )
        
        # Validate that all users are accounted for
        total_weekly_starters = weekly_result['started_funnel_users'].sum()
        total_weekly_completers = weekly_result['completed_funnel_users'].sum()
        
        total_monthly_starters = monthly_result['started_funnel_users'].sum()
        total_monthly_completers = monthly_result['completed_funnel_users'].sum()
        
        # All 6 users should be counted
        assert total_weekly_starters == 6, f"Weekly: Expected 6 starters, got {total_weekly_starters}"
        assert total_weekly_completers == 6, f"Weekly: Expected 6 completers, got {total_weekly_completers}"
        
        assert total_monthly_starters == 6, f"Monthly: Expected 6 starters, got {total_monthly_starters}"
        assert total_monthly_completers == 6, f"Monthly: Expected 6 completers, got {total_monthly_completers}"
        
        # Should have reasonable number of periods
        assert 2 <= len(weekly_result) <= 4, f"Expected 2-4 weeks, got {len(weekly_result)}"
        assert len(monthly_result) == 2, f"Expected 2 months (Jan, Feb), got {len(monthly_result)}"
        
        print("✅ Boundary conditions test passed for weekly/monthly aggregation")

    def test_aggregation_consistency_validation(self, long_window_calculator, controlled_cohort_data, funnel_steps_4):
        """Test mathematical consistency across daily, weekly, and monthly aggregations."""
        # Get results for all three aggregation periods
        daily_result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1d'
        )
        
        weekly_result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1w'
        )
        
        monthly_result = long_window_calculator.calculate_timeseries_metrics(
            controlled_cohort_data, funnel_steps_4, aggregation_period='1mo'
        )
        
        # Calculate totals for each aggregation
        daily_totals = {
            'starters': daily_result['started_funnel_users'].sum(),
            'completers': daily_result['completed_funnel_users'].sum()
        }
        
        weekly_totals = {
            'starters': weekly_result['started_funnel_users'].sum(),
            'completers': weekly_result['completed_funnel_users'].sum()
        }
        
        monthly_totals = {
            'starters': monthly_result['started_funnel_users'].sum(),
            'completers': monthly_result['completed_funnel_users'].sum()
        }
        
        # All aggregations should have same totals (controlled data spans 2 days)
        assert daily_totals['starters'] == 1500, f"Daily starters: expected 1500, got {daily_totals['starters']}"
        assert daily_totals['completers'] == 500, f"Daily completers: expected 500, got {daily_totals['completers']}"
        
        # Weekly and monthly should match daily (since data spans only 2 days)
        assert weekly_totals['starters'] == daily_totals['starters'], \
            f"Weekly/daily starters mismatch: {weekly_totals['starters']} vs {daily_totals['starters']}"
        assert weekly_totals['completers'] == daily_totals['completers'], \
            f"Weekly/daily completers mismatch: {weekly_totals['completers']} vs {daily_totals['completers']}"
        
        assert monthly_totals['starters'] == daily_totals['starters'], \
            f"Monthly/daily starters mismatch: {monthly_totals['starters']} vs {daily_totals['starters']}"
        assert monthly_totals['completers'] == daily_totals['completers'], \
            f"Monthly/daily completers mismatch: {monthly_totals['completers']} vs {daily_totals['completers']}"
        
        # Overall conversion rates should be consistent
        daily_rate = (daily_totals['completers'] / daily_totals['starters']) * 100
        weekly_rate = (weekly_totals['completers'] / weekly_totals['starters']) * 100
        monthly_rate = (monthly_totals['completers'] / monthly_totals['starters']) * 100
        
        assert abs(daily_rate - 33.33) < 0.1, f"Daily rate: expected 33.33%, got {daily_rate:.2f}%"
        assert abs(weekly_rate - daily_rate) < 0.01, f"Weekly rate differs from daily: {weekly_rate:.2f}% vs {daily_rate:.2f}%"
        assert abs(monthly_rate - daily_rate) < 0.01, f"Monthly rate differs from daily: {monthly_rate:.2f}% vs {daily_rate:.2f}%"
        
        print("✅ Aggregation consistency validation test passed")
        print(f"   Daily: {len(daily_result)} periods, Weekly: {len(weekly_result)} periods, Monthly: {len(monthly_result)} periods")
        print(f"   Consistent totals: {daily_totals['starters']} starters, {daily_totals['completers']} completers")
        print(f"   Consistent conversion rate: {daily_rate:.2f}%")

    def test_performance_with_large_weekly_dataset(self, long_window_calculator):
        """Test performance of weekly aggregation with large dataset."""
        import time
        
        # Generate 6 months of data with weekly variations
        events = []
        base_date = datetime(2024, 1, 1, 0, 0, 0)
        
        for week in range(26):  # 26 weeks = ~6 months
            week_start = base_date + timedelta(weeks=week)
            
            # Weekly volume varies (simulate seasonal patterns)
            weekly_multiplier = 1 + 0.5 * (week % 4) / 4  # 1.0 to 1.5 multiplier
            users_this_week = int(200 * weekly_multiplier)
            
            for i in range(users_this_week):
                user_id = f'perf_w{week:02d}_u{i:04d}'
                start_time = week_start + timedelta(
                    days=i % 7,
                    hours=(i * 3) % 24,
                    minutes=(i * 7) % 60
                )
                
                # Generate funnel events (80% -> 60% -> 40% conversion)
                events.append({
                    'user_id': user_id,
                    'event_name': 'Start',
                    'timestamp': start_time,
                    'event_properties': '{}',
                    'user_properties': '{}'
                })
                
                if i < users_this_week * 0.8:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'Middle',
                        'timestamp': start_time + timedelta(hours=2),
                        'event_properties': '{}',
                        'user_properties': '{}'
                    })
                    
                    if i < users_this_week * 0.6:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'End',
                            'timestamp': start_time + timedelta(hours=4),
                            'event_properties': '{}',
                            'user_properties': '{}'
                        })
        
        large_df = pd.DataFrame(events)
        
        print(f"Testing weekly aggregation performance with {len(large_df)} events, {large_df['user_id'].nunique()} users")
        
        # Time the weekly aggregation
        start_time = time.time()
        
        result = long_window_calculator.calculate_timeseries_metrics(
            large_df, ['Start', 'Middle', 'End'], aggregation_period='1w'
        )
        
        calculation_time = time.time() - start_time
        
        # Validate results
        assert len(result) == 26, f"Expected 26 weeks, got {len(result)}"
        
        total_starters = result['started_funnel_users'].sum()
        total_completers = result['completed_funnel_users'].sum()
        
        # Validate reasonable totals
        assert total_starters > 5000, f"Expected >5000 starters, got {total_starters}"
        assert total_completers > 3000, f"Expected >3000 completers, got {total_completers}"
        assert total_completers <= total_starters, "Completers should not exceed starters"
        
        # Performance requirement: should complete in under 15 seconds
        assert calculation_time < 15.0, f"Weekly aggregation too slow: {calculation_time:.2f} seconds for {len(large_df)} events"
        
        # Validate weekly patterns
        for i, row in result.iterrows():
            assert row['started_funnel_users'] > 0, f"Week {i}: Should have starters"
            assert row['completed_funnel_users'] >= 0, f"Week {i}: Non-negative completers"
            assert 0 <= row['conversion_rate'] <= 100, f"Week {i}: Valid conversion rate"
        
        print(f"✅ Large weekly dataset performance test passed in {calculation_time:.2f} seconds")
        print(f"   26 weeks processed: {total_starters} starters, {total_completers} completers")
        print(f"   Performance: {len(large_df)/calculation_time:.0f} events/second")
