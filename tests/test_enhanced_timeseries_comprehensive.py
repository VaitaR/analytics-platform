#!/usr/bin/env python3
"""
Comprehensive test for the enhanced timeseries metrics with cohort analysis fix.

This test validates that both cohort conversion metrics and daily activity metrics
are calculated correctly and attributed to the proper dates.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import FunnelCalculator
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


class TestEnhancedTimeseriesMetrics:
    """Test enhanced timeseries with both cohort and daily activity metrics."""
    
    def test_cohort_vs_daily_metrics_attribution(self):
        """
        Test that cohort conversion metrics and daily activity metrics are properly separated.
        
        Cohort metrics: Attribution based on when users started their journey
        Daily metrics: Attribution based on when events actually occurred
        """
        # Create scenario with clear cross-day conversions
        events_data = [
            # Day 1: 3 users sign up
            {
                'user_id': 'day1_user_1',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day1_user_2',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 11, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day1_user_3',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 12, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            
            # Day 2: 5 more users sign up + 2 users from day 1 convert
            {
                'user_id': 'day2_user_1',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 2, 9, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_2',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 2, 10, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_3',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 2, 11, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_4',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 2, 12, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_5',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 2, 13, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # 2 conversions from day 1 cohort happening on day 2
            {
                'user_id': 'day1_user_1',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 2, 14, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day1_user_2',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 2, 15, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            
            # Day 3: No signups, but 1 more conversion from day 1 + 2 conversions from day 2
            {
                'user_id': 'day1_user_3',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 3, 10, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_1',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 3, 11, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'day2_user_2',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 3, 12, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ]
        
        events_df = pd.DataFrame(events_data)
        funnel_steps = ['signup', 'purchase']
        
        # Use 72-hour conversion window to allow cross-day conversions
        config = FunnelConfig(
            conversion_window_hours=72,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED
        )
        calculator = FunnelCalculator(config)
        
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, '1d')
        
        # Convert to dict for easier access
        results_dict = {}
        for _, row in results.iterrows():
            date_key = row['period_date'].strftime('%Y-%m-%d')
            results_dict[date_key] = row.to_dict()
        
        # COHORT CONVERSION METRICS (attributed to cohort start date)
        
        # Day 1 cohort: 3 users started, 3 users converted (100% conversion)
        assert '2024-01-01' in results_dict
        day1_cohort = results_dict['2024-01-01']
        assert day1_cohort['started_funnel_users'] == 3, f"Day 1 cohort should have 3 starters, got {day1_cohort['started_funnel_users']}"
        assert day1_cohort['completed_funnel_users'] == 3, f"Day 1 cohort should have 3 completers, got {day1_cohort['completed_funnel_users']}"
        assert abs(day1_cohort['conversion_rate'] - 100.0) < 0.01, f"Day 1 cohort conversion should be 100%, got {day1_cohort['conversion_rate']}"
        
        # Day 2 cohort: 5 users started, 2 users converted (40% conversion)
        assert '2024-01-02' in results_dict
        day2_cohort = results_dict['2024-01-02']
        assert day2_cohort['started_funnel_users'] == 5, f"Day 2 cohort should have 5 starters, got {day2_cohort['started_funnel_users']}"
        assert day2_cohort['completed_funnel_users'] == 2, f"Day 2 cohort should have 2 completers, got {day2_cohort['completed_funnel_users']}"
        assert abs(day2_cohort['conversion_rate'] - 40.0) < 0.01, f"Day 2 cohort conversion should be 40%, got {day2_cohort['conversion_rate']}"
        
        # Day 3: No cohort (no signups)
        assert '2024-01-03' in results_dict
        day3_metrics = results_dict['2024-01-03']
        assert day3_metrics['started_funnel_users'] == 0, f"Day 3 should have 0 starters, got {day3_metrics['started_funnel_users']}"
        assert day3_metrics['completed_funnel_users'] == 0, f"Day 3 should have 0 completers, got {day3_metrics['completed_funnel_users']}"
        
        # DAILY ACTIVITY METRICS (attributed to event date)
        
        # Day 1: 3 unique users, 3 events (all signups)
        assert day1_cohort['daily_active_users'] == 3, f"Day 1 should have 3 daily active users, got {day1_cohort['daily_active_users']}"
        assert day1_cohort['daily_events_total'] == 3, f"Day 1 should have 3 daily events, got {day1_cohort['daily_events_total']}"
        
        # Day 2: 7 unique users, 7 events (5 signups + 2 purchases)
        assert day2_cohort['daily_active_users'] == 7, f"Day 2 should have 7 daily active users, got {day2_cohort['daily_active_users']}"
        assert day2_cohort['daily_events_total'] == 7, f"Day 2 should have 7 daily events, got {day2_cohort['daily_events_total']}"
        
        # Day 3: 3 unique users, 3 events (all purchases)
        assert day3_metrics['daily_active_users'] == 3, f"Day 3 should have 3 daily active users, got {day3_metrics['daily_active_users']}"
        assert day3_metrics['daily_events_total'] == 3, f"Day 3 should have 3 daily events, got {day3_metrics['daily_events_total']}"
        
        print("✅ ENHANCED TIMESERIES METRICS TEST PASSED!")
        print("  ✅ Cohort conversion metrics correctly attributed to signup dates")
        print("  ✅ Daily activity metrics correctly attributed to event dates")
    
    def test_same_day_conversions(self):
        """
        Test that same-day conversions are handled correctly in both metric types.
        """
        events_data = [
            # User signs up and converts on the same day
            {
                'user_id': 'same_day_user',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'same_day_user',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 1, 14, 0, 0),  # 4 hours later
                'event_properties': '{}',
                'user_properties': '{}'
            },
            # Another user signs up but doesn't convert
            {
                'user_id': 'non_converter',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 12, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ]
        
        events_df = pd.DataFrame(events_data)
        funnel_steps = ['signup', 'purchase']
        
        config = FunnelConfig(conversion_window_hours=24)
        calculator = FunnelCalculator(config)
        
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, '1d')
        
        assert len(results) == 1, f"Should have exactly 1 period, got {len(results)}"
        
        result = results.iloc[0]
        
        # Cohort metrics: 2 starters, 1 completer, 50% conversion
        assert result['started_funnel_users'] == 2, f"Should have 2 starters, got {result['started_funnel_users']}"
        assert result['completed_funnel_users'] == 1, f"Should have 1 completer, got {result['completed_funnel_users']}"
        assert abs(result['conversion_rate'] - 50.0) < 0.01, f"Conversion should be 50%, got {result['conversion_rate']}"
        
        # Daily activity metrics: 2 unique users, 3 events (2 signups + 1 purchase)
        assert result['daily_active_users'] == 2, f"Should have 2 daily active users, got {result['daily_active_users']}"
        assert result['daily_events_total'] == 3, f"Should have 3 daily events, got {result['daily_events_total']}"
        
        print("✅ SAME-DAY CONVERSION TEST PASSED!")
    
    def test_edge_case_no_conversions(self):
        """
        Test periods with signups but no conversions.
        """
        events_data = [
            # Day 1: 5 signups, no conversions
            {
                'user_id': f'user_{i}',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 10 + i, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            }
            for i in range(5)
        ]
        
        events_df = pd.DataFrame(events_data)
        funnel_steps = ['signup', 'purchase']
        
        config = FunnelConfig(conversion_window_hours=24)
        calculator = FunnelCalculator(config)
        
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, '1d')
        
        assert len(results) == 1, f"Should have exactly 1 period, got {len(results)}"
        
        result = results.iloc[0]
        
        # Cohort metrics: 5 starters, 0 completers, 0% conversion
        assert result['started_funnel_users'] == 5, f"Should have 5 starters, got {result['started_funnel_users']}"
        assert result['completed_funnel_users'] == 0, f"Should have 0 completers, got {result['completed_funnel_users']}"
        assert abs(result['conversion_rate'] - 0.0) < 0.01, f"Conversion should be 0%, got {result['conversion_rate']}"
        
        # Daily activity metrics: 5 unique users, 5 events (all signups)
        assert result['daily_active_users'] == 5, f"Should have 5 daily active users, got {result['daily_active_users']}"
        assert result['daily_events_total'] == 5, f"Should have 5 daily events, got {result['daily_events_total']}"
        
        print("✅ NO CONVERSIONS TEST PASSED!")
    
    def test_backwards_compatibility(self):
        """
        Test that legacy total_unique_users and total_events metrics still work.
        """
        events_data = [
            {
                'user_id': 'user_1',
                'event_name': 'signup',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            },
            {
                'user_id': 'user_1',
                'event_name': 'purchase',
                'timestamp': datetime(2024, 1, 1, 14, 0, 0),
                'event_properties': '{}',
                'user_properties': '{}'
            }
        ]
        
        events_df = pd.DataFrame(events_data)
        funnel_steps = ['signup', 'purchase']
        
        config = FunnelConfig(conversion_window_hours=24)
        calculator = FunnelCalculator(config)
        
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, '1d')
        
        result = results.iloc[0]
        
        # Legacy metrics should still exist and have the same values as new metrics
        assert 'total_unique_users' in result, "Legacy total_unique_users metric missing"
        assert 'total_events' in result, "Legacy total_events metric missing"
        assert result['total_unique_users'] == result['daily_active_users'], "Legacy and new user metrics should match"
        assert result['total_events'] == result['daily_events_total'], "Legacy and new event metrics should match"
        
        print("✅ BACKWARDS COMPATIBILITY TEST PASSED!")


if __name__ == "__main__":
    # Run all tests
    test_instance = TestEnhancedTimeseriesMetrics()
    
    print("🧪 Running Enhanced Timeseries Metrics Tests...")
    print()
    
    test_instance.test_cohort_vs_daily_metrics_attribution()
    test_instance.test_same_day_conversions()
    test_instance.test_edge_case_no_conversions()
    test_instance.test_backwards_compatibility()
    
    print()
    print("🎉 ALL ENHANCED TIMESERIES METRICS TESTS PASSED!")
    print("📊 The implementation now provides:")
    print("  - ✅ Cohort conversion metrics (attributed to signup dates)")
    print("  - ✅ Daily activity metrics (attributed to event dates)")
    print("  - ✅ Backwards compatibility with legacy metrics")
    print("  - ✅ Proper handling of cross-day conversions")
    print("  - ✅ Edge case support (no conversions, same-day conversions)")
