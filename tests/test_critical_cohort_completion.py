#!/usr/bin/env python3
"""
Critical breaking test to expose the exact completed_funnel_users calculation error.

This test demonstrates the specific scenario mentioned in the requirements where
completed_funnel_users might be calculated incorrectly for cross-day conversions.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import FunnelCalculator
from models import CountingMethod, FunnelConfig, FunnelOrder, ReentryMode


class TestCriticalCohortCompletionCount:
    """Critical tests to expose exact completed_funnel_users calculation errors."""

    @pytest.fixture
    def simple_cross_day_data(self):
        """
        Simple fixture with exactly the scenario mentioned in requirements:
        - User_A: Signup (1 Jan), Purchase (2 Jan)
        - User_B: Signup (1 Jan), no Purchase
        """
        return pd.DataFrame(
            [
                {
                    "user_id": "User_A",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
                {
                    "user_id": "User_B",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 14, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
                {
                    "user_id": "User_A",
                    "event_name": "purchase",
                    "timestamp": datetime(2024, 1, 2, 11, 0, 0),  # Next day conversion
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
            ]
        )

    def test_true_cohort_completion_count(self, simple_cross_day_data):
        """
        CRITICAL TEST: Verify completed_funnel_users is calculated correctly for cohorts.

        Expected behavior:
        - 2024-01-01: started_funnel_users=2, completed_funnel_users=1 (User_A converted)
        - 2024-01-02: started_funnel_users=0, completed_funnel_users=0 (no new cohort)

        Common error patterns this test catches:
        - completed_funnel_users=0 for 2024-01-01 (conversion not attributed to cohort)
        - completed_funnel_users=2 for 2024-01-01 (double counting or incorrect logic)
        - Extra period for 2024-01-02 with non-zero started_funnel_users
        """
        funnel_steps = ["signup", "purchase"]

        # Use 48-hour window to allow cross-day conversion
        config = FunnelConfig(
            conversion_window_hours=48,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
        )
        calculator = FunnelCalculator(config)

        # Calculate timeseries metrics
        results = calculator.calculate_timeseries_metrics(
            simple_cross_day_data, funnel_steps, "1d"
        )

        print("\\n🔍 CRITICAL TEST RESULTS:")
        for _, row in results.iterrows():
            date = row["period_date"].strftime("%Y-%m-%d")
            print(
                f"{date}: started={row['started_funnel_users']}, completed={row['completed_funnel_users']}, rate={row['conversion_rate']:.1f}%"
            )

        # Convert to dict for easier assertions
        results_dict = {}
        for _, row in results.iterrows():
            date_key = row["period_date"].strftime("%Y-%m-%d")
            results_dict[date_key] = row.to_dict()

        # CRITICAL ASSERTIONS

        # Jan 1: Should have exactly 2 starters (User_A and User_B)
        assert "2024-01-01" in results_dict, "2024-01-01 period must exist"
        jan_1 = results_dict["2024-01-01"]

        # This is the core test - started_funnel_users must be exactly 2
        assert jan_1["started_funnel_users"] == 2, (
            f"2024-01-01 should have exactly 2 users starting funnel, "
            f"got {jan_1['started_funnel_users']}. "
            f"This indicates cohort definition is wrong."
        )

        # This is the critical completion count test
        assert jan_1["completed_funnel_users"] == 1, (
            f"2024-01-01 cohort should have exactly 1 user completing funnel (User_A), "
            f"got {jan_1['completed_funnel_users']}. "
            f"Common errors: 0 (conversion not attributed to cohort), "
            f"2 (double counting or wrong logic)."
        )

        # Conversion rate should be exactly 50%
        expected_rate = 50.0
        assert abs(jan_1["conversion_rate"] - expected_rate) < 0.01, (
            f"2024-01-01 cohort conversion rate should be {expected_rate}% (1/2), "
            f"got {jan_1['conversion_rate']:.2f}%"
        )

        # Jan 2: Should NOT exist as a cohort (no signups on Jan 2)
        if "2024-01-02" in results_dict:
            jan_2 = results_dict["2024-01-02"]
            assert jan_2["started_funnel_users"] == 0, (
                f"2024-01-02 should have 0 users starting funnel (no signups), "
                f"got {jan_2['started_funnel_users']}"
            )
            assert jan_2["completed_funnel_users"] == 0, (
                f"2024-01-02 should have 0 users completing funnel (no cohort), "
                f"got {jan_2['completed_funnel_users']}"
            )

        print("✅ CRITICAL COHORT COMPLETION COUNT TEST PASSED!")
        print("   The completed_funnel_users calculation is mathematically correct.")

    def test_edge_case_same_day_vs_cross_day_attribution(self, simple_cross_day_data):
        """
        Test edge case: Ensure same-day conversions aren't double-counted with cross-day.
        """
        # Add a same-day conversion to the fixture data
        additional_data = pd.DataFrame(
            [
                {
                    "user_id": "User_C",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 16, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
                {
                    "user_id": "User_C",
                    "event_name": "purchase",
                    "timestamp": datetime(2024, 1, 1, 18, 0, 0),  # Same day conversion
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
            ]
        )

        combined_data = pd.concat([simple_cross_day_data, additional_data], ignore_index=True)

        config = FunnelConfig(conversion_window_hours=48)
        calculator = FunnelCalculator(config)

        results = calculator.calculate_timeseries_metrics(
            combined_data, ["signup", "purchase"], "1d"
        )

        results_dict = {}
        for _, row in results.iterrows():
            date_key = row["period_date"].strftime("%Y-%m-%d")
            results_dict[date_key] = row.to_dict()

        # Jan 1 should now have 3 starters and 2 completers (User_A cross-day + User_C same-day)
        jan_1 = results_dict["2024-01-01"]
        assert jan_1["started_funnel_users"] == 3, (
            f"Should have 3 starters, got {jan_1['started_funnel_users']}"
        )
        assert jan_1["completed_funnel_users"] == 2, (
            f"Should have 2 completers, got {jan_1['completed_funnel_users']}"
        )

        expected_rate = 66.67  # 2/3 * 100
        assert abs(jan_1["conversion_rate"] - expected_rate) < 0.1, (
            f"Conversion rate should be ~{expected_rate}%, got {jan_1['conversion_rate']:.2f}%"
        )

        print("✅ SAME-DAY vs CROSS-DAY ATTRIBUTION TEST PASSED!")

    def test_zero_conversion_cohort(self):
        """
        Test cohort with zero conversions to ensure completed_funnel_users=0 is handled correctly.
        """
        zero_conversion_data = pd.DataFrame(
            [
                {
                    "user_id": "no_convert_1",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 10, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
                {
                    "user_id": "no_convert_2",
                    "event_name": "signup",
                    "timestamp": datetime(2024, 1, 1, 12, 0, 0),
                    "event_properties": "{}",
                    "user_properties": "{}",
                },
                # No purchase events at all
            ]
        )

        config = FunnelConfig(conversion_window_hours=24)
        calculator = FunnelCalculator(config)

        results = calculator.calculate_timeseries_metrics(
            zero_conversion_data, ["signup", "purchase"], "1d"
        )

        assert len(results) == 1, f"Should have exactly 1 period, got {len(results)}"

        result = results.iloc[0]
        assert result["started_funnel_users"] == 2, (
            f"Should have 2 starters, got {result['started_funnel_users']}"
        )
        assert result["completed_funnel_users"] == 0, (
            f"Should have 0 completers, got {result['completed_funnel_users']}"
        )
        assert result["conversion_rate"] == 0.0, (
            f"Conversion rate should be 0%, got {result['conversion_rate']}"
        )

        print("✅ ZERO CONVERSION COHORT TEST PASSED!")


if __name__ == "__main__":
    # Run the critical tests
    test_instance = TestCriticalCohortCompletionCount()

    # Create test data
    simple_data = pd.DataFrame(
        [
            {
                "user_id": "User_A",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "User_B",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 14, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "User_A",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 2, 11, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]
    )

    try:
        print("🚨 Running CRITICAL cohort completion count test...")
        test_instance.test_true_cohort_completion_count(simple_data)
    except AssertionError as e:
        print(f"❌ CRITICAL ERROR DETECTED: {e}")
        print("🔧 This indicates a bug in completed_funnel_users calculation!")

    try:
        print("\\n🚨 Running edge case test...")
        test_instance.test_edge_case_same_day_vs_cross_day_attribution(simple_data)
    except AssertionError as e:
        print(f"❌ EDGE CASE ERROR: {e}")

    try:
        print("\\n🚨 Running zero conversion test...")
        test_instance.test_zero_conversion_cohort()
    except AssertionError as e:
        print(f"❌ ZERO CONVERSION ERROR: {e}")

    print("\\n✅ All critical tests completed!")
