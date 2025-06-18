#!/usr/bin/env python3
"""
Breaking test to prove the fundamental cohort attribution error in timeseries analysis.

This test demonstrates the critical bug where conversions are attributed to when they
happen rather than to the cohort (start date) of the user who converted.

Test Case:
- User_A: Signup (Jan 1, 23:00), Purchase (Jan 2, 01:00)
- 10 other users: Signup (Jan 2), no Purchase

Current BROKEN behavior:
- Analysis for Jan 2: started=10, completed=1 (from User_A) -> 10% conversion

Correct COHORT behavior:
- Jan 1 cohort: started=1, completed=1 -> 100% conversion
- Jan 2 cohort: started=10, completed=0 -> 0% conversion
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import FunnelCalculator
from models import CountingMethod, FunnelConfig, FunnelOrder, ReentryMode


class TestCohortAttributionError:
    """Test cases that prove the cohort attribution bug in timeseries calculation."""

    def test_cross_period_conversion_attribution(self):
        """
        BREAKING TEST: Proves conversions are wrongly attributed to conversion date, not cohort date.

        This test will FAIL with current implementation and PASS after the fix.
        """
        # Create test data that exposes the attribution bug
        events_data = [
            # User_A: Signs up on Jan 1 (23:00), converts on Jan 2 (01:00)
            {
                "user_id": "user_A",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 23, 0, 0),  # Jan 1, 23:00
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "user_A",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 2, 1, 0, 0),  # Jan 2, 01:00 (2 hours later)
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]

        # Add 10 users who sign up on Jan 2 but never convert
        for i in range(10):
            events_data.append(
                {
                    "user_id": f"user_jan2_{i}",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 2, 10 + i, 0, 0),  # Jan 2, various hours
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

        events_df = pd.DataFrame(events_data)
        funnel_steps = ["signup", "purchase"]

        # Create calculator with 24-hour conversion window
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
        )
        calculator = FunnelCalculator(config)

        # Calculate daily timeseries metrics
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, "1d")

        # Convert results to dict for easier access
        results_dict = {}
        for _, row in results.iterrows():
            date_key = row["period_date"].strftime("%Y-%m-%d")
            results_dict[date_key] = row.to_dict()

        # CRITICAL ASSERTIONS: These will FAIL with current implementation

        # Jan 1 cohort: 1 user started, 1 user completed (User_A converted within 24h window)
        assert "2024-01-01" in results_dict, "Jan 1 period should exist in results"
        jan_1_data = results_dict["2024-01-01"]
        assert (
            jan_1_data["started_funnel_users"] == 1
        ), f"Jan 1 should have 1 starter, got {jan_1_data['started_funnel_users']}"
        assert (
            jan_1_data["completed_funnel_users"] == 1
        ), f"Jan 1 cohort should have 1 converter, got {jan_1_data['completed_funnel_users']}"
        assert (
            abs(jan_1_data["conversion_rate"] - 100.0) < 0.01
        ), f"Jan 1 cohort conversion should be 100%, got {jan_1_data['conversion_rate']}"

        # Jan 2 cohort: 10 users started, 0 users completed (none of the Jan 2 users converted)
        assert "2024-01-02" in results_dict, "Jan 2 period should exist in results"
        jan_2_data = results_dict["2024-01-02"]
        assert (
            jan_2_data["started_funnel_users"] == 10
        ), f"Jan 2 should have 10 starters, got {jan_2_data['started_funnel_users']}"
        assert (
            jan_2_data["completed_funnel_users"] == 0
        ), f"Jan 2 cohort should have 0 converters, got {jan_2_data['completed_funnel_users']}"
        assert (
            abs(jan_2_data["conversion_rate"] - 0.0) < 0.01
        ), f"Jan 2 cohort conversion should be 0%, got {jan_2_data['conversion_rate']}"

        print(
            "✅ COHORT ATTRIBUTION TEST PASSED: Conversions correctly attributed to cohort start dates!"
        )

    def test_multi_day_conversion_window_attribution(self):
        """
        Test proper cohort attribution across multiple days with longer conversion windows.

        This validates that even with longer conversion windows, conversions are still
        attributed to the original cohort date, not the conversion date.
        """
        events_data = []

        # Cohort 1: Jan 1 - 5 users sign up
        for i in range(5):
            events_data.append(
                {
                    "user_id": f"jan1_user_{i}",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 9 + i, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

            # 2 of them convert on Jan 3 (2 days later)
            if i < 2:
                events_data.append(
                    {
                        "user_id": f"jan1_user_{i}",
                        "event_name": "purchase",
                        "timestamp": datetime(2024, 1, 3, 10 + i, 0, 0),
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                )

        # Cohort 2: Jan 2 - 8 users sign up
        for i in range(8):
            events_data.append(
                {
                    "user_id": f"jan2_user_{i}",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 2, 9 + i, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

            # 3 of them convert on Jan 4 (2 days later)
            if i < 3:
                events_data.append(
                    {
                        "user_id": f"jan2_user_{i}",
                        "event_name": "purchase",
                        "timestamp": datetime(2024, 1, 4, 10 + i, 0, 0),
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                )

        # Cohort 3: Jan 3 - 6 users sign up, none convert
        for i in range(6):
            events_data.append(
                {
                    "user_id": f"jan3_user_{i}",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 3, 9 + i, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

        events_df = pd.DataFrame(events_data)
        funnel_steps = ["signup", "purchase"]

        # Use 72-hour conversion window to allow cross-day conversions
        config = FunnelConfig(
            conversion_window_hours=72,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
        )
        calculator = FunnelCalculator(config)

        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, "1d")

        # Convert to dict for easier assertions
        results_dict = {}
        for _, row in results.iterrows():
            date_key = row["period_date"].strftime("%Y-%m-%d")
            results_dict[date_key] = row.to_dict()

        # Validate cohort-based attribution

        # Jan 1 cohort: 5 starters, 2 converters (40% conversion)
        jan_1 = results_dict["2024-01-01"]
        assert (
            jan_1["started_funnel_users"] == 5
        ), f"Jan 1 cohort should have 5 starters, got {jan_1['started_funnel_users']}"
        assert (
            jan_1["completed_funnel_users"] == 2
        ), f"Jan 1 cohort should have 2 converters, got {jan_1['completed_funnel_users']}"
        expected_conversion_1 = 40.0
        assert (
            abs(jan_1["conversion_rate"] - expected_conversion_1) < 0.01
        ), f"Jan 1 conversion should be {expected_conversion_1}%, got {jan_1['conversion_rate']}"

        # Jan 2 cohort: 8 starters, 3 converters (37.5% conversion)
        jan_2 = results_dict["2024-01-02"]
        assert (
            jan_2["started_funnel_users"] == 8
        ), f"Jan 2 cohort should have 8 starters, got {jan_2['started_funnel_users']}"
        assert (
            jan_2["completed_funnel_users"] == 3
        ), f"Jan 2 cohort should have 3 converters, got {jan_2['completed_funnel_users']}"
        expected_conversion_2 = 37.5
        assert (
            abs(jan_2["conversion_rate"] - expected_conversion_2) < 0.01
        ), f"Jan 2 conversion should be {expected_conversion_2}%, got {jan_2['conversion_rate']}"

        # Jan 3 cohort: 6 starters, 0 converters (0% conversion)
        jan_3 = results_dict["2024-01-03"]
        assert (
            jan_3["started_funnel_users"] == 6
        ), f"Jan 3 cohort should have 6 starters, got {jan_3['started_funnel_users']}"
        assert (
            jan_3["completed_funnel_users"] == 0
        ), f"Jan 3 cohort should have 0 converters, got {jan_3['completed_funnel_users']}"
        assert (
            abs(jan_3["conversion_rate"] - 0.0) < 0.01
        ), f"Jan 3 conversion should be 0%, got {jan_3['conversion_rate']}"

        # Jan 4 should not exist as a cohort (no signups), even though conversions happened
        assert (
            "2024-01-04" not in results_dict
        ), "Jan 4 should not appear as a cohort since no signups occurred"

        print("✅ MULTI-DAY COHORT ATTRIBUTION TEST PASSED!")

    def test_edge_case_same_minute_signup_conversion(self):
        """
        Test edge case where signup and conversion happen within the same aggregation period.

        This ensures that immediate conversions are still properly attributed to cohorts.
        """
        events_data = [
            # User converts within 5 minutes of signup (same hour)
            {
                "user_id": "quick_converter",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 14, 10, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "quick_converter",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 1, 14, 15, 0),  # 5 minutes later
                "event_properties": "{}",
                "user_properties": "{}",
            },
            # User signs up but doesn't convert
            {
                "user_id": "non_converter",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 14, 30, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]

        events_df = pd.DataFrame(events_data)
        funnel_steps = ["signup", "purchase"]

        config = FunnelConfig(
            conversion_window_hours=1,  # 1 hour window
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
        )
        calculator = FunnelCalculator(config)

        # Test with hourly aggregation
        results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, "1h")

        # Should have exactly one period (14:00-15:00)
        assert len(results) == 1, f"Should have exactly 1 period, got {len(results)}"

        result = results.iloc[0]
        assert (
            result["started_funnel_users"] == 2
        ), f"Should have 2 starters, got {result['started_funnel_users']}"
        assert (
            result["completed_funnel_users"] == 1
        ), f"Should have 1 converter, got {result['completed_funnel_users']}"
        expected_conversion = 50.0
        assert (
            abs(result["conversion_rate"] - expected_conversion) < 0.01
        ), f"Conversion should be {expected_conversion}%, got {result['conversion_rate']}"

        print("✅ SAME-PERIOD CONVERSION TEST PASSED!")


if __name__ == "__main__":
    # Run the breaking tests
    test_instance = TestCohortAttributionError()

    try:
        print("🚨 Running BREAKING TEST for cohort attribution...")
        test_instance.test_cross_period_conversion_attribution()
        print("❌ UNEXPECTED: Breaking test passed! The bug might already be fixed.")
    except AssertionError as e:
        print(f"✅ EXPECTED: Breaking test failed as expected: {e}")
        print("🔧 This proves the cohort attribution bug exists and needs fixing!")

    try:
        print("\n🚨 Running multi-day cohort test...")
        test_instance.test_multi_day_conversion_window_attribution()
        print("❌ UNEXPECTED: Multi-day test passed!")
    except AssertionError as e:
        print(f"✅ EXPECTED: Multi-day test failed: {e}")

    try:
        print("\n🚨 Running same-period conversion test...")
        test_instance.test_edge_case_same_minute_signup_conversion()
        print("❌ UNEXPECTED: Same-period test passed!")
    except AssertionError as e:
        print(f"✅ EXPECTED: Same-period test failed: {e}")
