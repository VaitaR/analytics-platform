"""
Test to verify the fix for conversion rate calculation discrepancy.
This test ensures that weighted average calculation is used instead of arithmetic mean.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app import CountingMethod, FunnelCalculator, FunnelConfig, FunnelOrder, ReentryMode


class TestConversionRateCalculationFix:
    """Test the fix for conversion rate calculation."""

    @pytest.fixture
    def unbalanced_data(self):
        """Create data with unbalanced daily distributions to test weighted average."""
        data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Day 1: 10 users start, 8 complete (80% conversion rate)
        for i in range(10):
            data.append(
                {
                    "user_id": f"day1_user_{i}",
                    "event_name": "step1",
                    "timestamp": base_time + timedelta(minutes=i * 5),
                    "event_properties": "{}",
                }
            )

            # 8 out of 10 complete
            if i < 8:
                data.append(
                    {
                        "user_id": f"day1_user_{i}",
                        "event_name": "step2",
                        "timestamp": base_time + timedelta(minutes=i * 5 + 30),
                        "event_properties": "{}",
                    }
                )

        # Day 2: 1000 users start, 10 complete (1% conversion rate)
        for i in range(1000):
            data.append(
                {
                    "user_id": f"day2_user_{i}",
                    "event_name": "step1",
                    "timestamp": base_time + timedelta(days=1, minutes=i),
                    "event_properties": "{}",
                }
            )

            # Only 10 out of 1000 complete
            if i < 10:
                data.append(
                    {
                        "user_id": f"day2_user_{i}",
                        "event_name": "step2",
                        "timestamp": base_time + timedelta(days=1, minutes=i + 30),
                        "event_properties": "{}",
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def calculator(self):
        """Create calculator with standard configuration."""
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            conversion_window_hours=24,
        )
        return FunnelCalculator(config)

    def test_weighted_vs_arithmetic_conversion_rate(self, unbalanced_data, calculator):
        """
        Test that demonstrates the difference between weighted average and arithmetic mean.

        The test data gets split across multiple days based on timestamps.
        This demonstrates why weighted average is mathematically correct for overall conversion.
        """
        steps = ["step1", "step2"]

        print("\n=== WEIGHTED VS ARITHMETIC CONVERSION RATE TEST ===")

        # Calculate daily time series
        daily_results = calculator.calculate_timeseries_metrics(unbalanced_data, steps, "1d")

        print("Daily breakdown:")
        for idx, row in daily_results.iterrows():
            print(
                f"  {row['period_date']}: {row['started_funnel_users']} started, "
                f"{row['completed_funnel_users']} completed, "
                f"{row['conversion_rate']:.2f}% conversion"
            )

        # Calculate different averages
        arithmetic_mean = daily_results["conversion_rate"].mean()

        total_started = daily_results["started_funnel_users"].sum()
        total_completed = daily_results["completed_funnel_users"].sum()
        weighted_average = (total_completed / total_started * 100) if total_started > 0 else 0

        print("\nComparison:")
        print(f"  Arithmetic mean: {arithmetic_mean:.2f}%")
        print(f"  Weighted average: {weighted_average:.2f}%")
        print(f"  Difference: {abs(arithmetic_mean - weighted_average):.2f}%")

        # The weighted average should match the overall funnel calculation
        overall_results = calculator.calculate_funnel_metrics(unbalanced_data, steps)
        overall_conversion = overall_results.conversion_rates[-1]

        print(f"  Overall funnel: {overall_conversion:.2f}%")
        print(f"  Weighted == Overall: {abs(weighted_average - overall_conversion) < 0.01}")

        # Core assertions: weighted average should match overall conversion
        assert (
            abs(weighted_average - overall_conversion) < 0.01
        ), "Weighted average should match overall conversion rate"

        # Arithmetic mean should be significantly different in unbalanced data
        assert (
            abs(arithmetic_mean - weighted_average) > 5
        ), "Arithmetic mean should be significantly different from weighted average in unbalanced data"

        # The key insight: weighted average gives correct overall conversion
        assert total_started == 1010, f"Expected 1010 total users, got {total_started}"
        assert total_completed == 18, f"Expected 18 total completions, got {total_completed}"

        # Verify weighted calculation is correct
        expected_weighted = (18 / 1010) * 100  # ~1.78%
        assert (
            abs(weighted_average - expected_weighted) < 0.01
        ), f"Weighted average calculation error: expected {expected_weighted:.2f}%, got {weighted_average:.2f}%"

    def test_balanced_data_same_result(self):
        """Test that both methods give same result when data is balanced."""
        # Create balanced data where each day has same number of starters
        data = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        for day in range(3):
            for i in range(100):  # Same 100 users each day
                data.append(
                    {
                        "user_id": f"day{day}_user_{i}",
                        "event_name": "step1",
                        "timestamp": base_time + timedelta(days=day, minutes=i),
                        "event_properties": "{}",
                    }
                )

                # 20% conversion rate each day
                if i < 20:
                    data.append(
                        {
                            "user_id": f"day{day}_user_{i}",
                            "event_name": "step2",
                            "timestamp": base_time + timedelta(days=day, minutes=i + 30),
                            "event_properties": "{}",
                        }
                    )

        df = pd.DataFrame(data)
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            conversion_window_hours=24,
        )
        calculator = FunnelCalculator(config)

        daily_results = calculator.calculate_timeseries_metrics(df, ["step1", "step2"], "1d")

        arithmetic_mean = daily_results["conversion_rate"].mean()
        total_started = daily_results["started_funnel_users"].sum()
        total_completed = daily_results["completed_funnel_users"].sum()
        weighted_average = (total_completed / total_started * 100) if total_started > 0 else 0

        print("\nBalanced data test:")
        print(f"  Arithmetic mean: {arithmetic_mean:.2f}%")
        print(f"  Weighted average: {weighted_average:.2f}%")

        # With balanced data, both should be very close
        assert (
            abs(arithmetic_mean - weighted_average) < 0.1
        ), "With balanced data, arithmetic mean and weighted average should be nearly identical"

        # Both should be close to 20%
        assert (
            abs(arithmetic_mean - 20.0) < 0.1
        ), f"Expected ~20% arithmetic mean, got {arithmetic_mean:.2f}%"
        assert (
            abs(weighted_average - 20.0) < 0.1
        ), f"Expected ~20% weighted average, got {weighted_average:.2f}%"


def test_conversion_rate_fix():
    """Standalone test function."""
    tester = TestConversionRateCalculationFix()

    # Create test data
    data = []
    base_time = datetime(2024, 1, 1, 10, 0, 0)

    # Day 1: 10 users start, 8 complete
    for i in range(10):
        data.append(
            {
                "user_id": f"day1_user_{i}",
                "event_name": "step1",
                "timestamp": base_time + timedelta(minutes=i * 5),
                "event_properties": "{}",
            }
        )
        if i < 8:
            data.append(
                {
                    "user_id": f"day1_user_{i}",
                    "event_name": "step2",
                    "timestamp": base_time + timedelta(minutes=i * 5 + 30),
                    "event_properties": "{}",
                }
            )

    # Day 2: 1000 users start, 10 complete
    for i in range(1000):
        data.append(
            {
                "user_id": f"day2_user_{i}",
                "event_name": "step1",
                "timestamp": base_time + timedelta(days=1, minutes=i),
                "event_properties": "{}",
            }
        )
        if i < 10:
            data.append(
                {
                    "user_id": f"day2_user_{i}",
                    "event_name": "step2",
                    "timestamp": base_time + timedelta(days=1, minutes=i + 30),
                    "event_properties": "{}",
                }
            )

    df = pd.DataFrame(data)

    config = FunnelConfig(
        counting_method=CountingMethod.UNIQUE_USERS,
        funnel_order=FunnelOrder.ORDERED,
        reentry_mode=ReentryMode.FIRST_ONLY,
        conversion_window_hours=24,
    )
    calculator = FunnelCalculator(config)

    tester.test_weighted_vs_arithmetic_conversion_rate(df, calculator)
