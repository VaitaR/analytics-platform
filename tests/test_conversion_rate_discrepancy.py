"""
Test to diagnose conversion rate discrepancy between timeseries and overall funnel metrics.

This test module specifically investigates why hourly timeseries conversion rates (~40%)
are significantly higher than overall funnel conversion rates (0.9%).
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from app import CountingMethod, FunnelCalculator, FunnelConfig, FunnelOrder, ReentryMode


class TestConversionRateDiscrepancy:
    """Test class to diagnose conversion rate calculation differences."""

    @pytest.fixture
    def diagnostic_data(self):
        """Create a specific dataset to test conversion rate calculations."""
        data = []
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Create a simple funnel scenario with known expected results
        # 10 users start within first hour, 4 complete the funnel
        for i in range(10):
            # All users do step1 within the first hour
            data.append(
                {
                    "user_id": f"user_{i}",
                    "event_name": "step1",
                    "timestamp": base_time + timedelta(minutes=i * 5),  # Spread across 45 minutes
                    "event_properties": "{}",
                }
            )

            # Only users 0, 2, 4, 6 complete step2 within conversion window
            if i % 2 == 0 and i < 8:
                data.append(
                    {
                        "user_id": f"user_{i}",
                        "event_name": "step2",
                        "timestamp": base_time + timedelta(minutes=i * 5 + 10),  # 10 minutes later
                        "event_properties": "{}",
                    }
                )

        # Add some users who start in the second hour
        for i in range(10, 15):
            data.append(
                {
                    "user_id": f"user_{i}",
                    "event_name": "step1",
                    "timestamp": base_time + timedelta(hours=1, minutes=(i - 10) * 10),
                    "event_properties": "{}",
                }
            )

            # Only user 10, 12 complete step2
            if i in [10, 12]:
                data.append(
                    {
                        "user_id": f"user_{i}",
                        "event_name": "step2",
                        "timestamp": base_time + timedelta(hours=1, minutes=(i - 10) * 10 + 15),
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
            conversion_window_hours=24,  # 24-hour window
        )
        return FunnelCalculator(config)

    def test_overall_vs_timeseries_conversion_rates(self, diagnostic_data, calculator):
        """
        Test to understand the discrepancy between overall and timeseries conversion rates.

        Expected behavior:
        - Overall funnel: 10 users start step1, 4+2=6 complete step2 → 60% conversion
        - Hourly timeseries should show similar aggregate conversion rate
        """
        steps = ["step1", "step2"]

        print("\n=== DIAGNOSTIC TEST: Conversion Rate Discrepancy ===")
        print("Dataset summary:")
        print(f"- Total events: {len(diagnostic_data)}")
        print(f"- Unique users: {diagnostic_data['user_id'].nunique()}")
        print(
            f"- Time range: {diagnostic_data['timestamp'].min()} to {diagnostic_data['timestamp'].max()}"
        )

        # 1. Calculate overall funnel metrics
        overall_results = calculator.calculate_funnel_metrics(diagnostic_data, steps)

        print("\n1. OVERALL FUNNEL RESULTS:")
        print(f"   Steps: {overall_results.steps}")
        print(f"   User counts: {overall_results.users_count}")
        print(f"   Conversion rates: {overall_results.conversion_rates}")

        overall_final_conversion = (
            overall_results.conversion_rates[-1] if overall_results.conversion_rates else 0
        )

        # 2. Calculate hourly timeseries metrics
        timeseries_results = calculator.calculate_timeseries_metrics(diagnostic_data, steps, "1h")

        print("\n2. HOURLY TIMESERIES RESULTS:")
        print(f"   Shape: {timeseries_results.shape}")
        if not timeseries_results.empty:
            print(f"   Columns: {list(timeseries_results.columns)}")
            print("   Data:")
            for idx, row in timeseries_results.iterrows():
                period = row.get("period_date", row.get("period", "Unknown"))
                started = row.get("started_funnel_users", row.get("started_count", "N/A"))
                completed = row.get("completed_funnel_users", row.get("completed_count", "N/A"))
                conversion_rate = row.get("conversion_rate", "N/A")
                print(
                    f"     {period}: started={started}, completed={completed}, "
                    f"conversion_rate={conversion_rate:.2f}%"
                )

        # 3. Calculate aggregate metrics from timeseries data
        if not timeseries_results.empty:
            # Use the correct column names from the actual output
            started_col = (
                "started_funnel_users"
                if "started_funnel_users" in timeseries_results.columns
                else "started_count"
            )
            completed_col = (
                "completed_funnel_users"
                if "completed_funnel_users" in timeseries_results.columns
                else "completed_count"
            )

            if (
                started_col in timeseries_results.columns
                and completed_col in timeseries_results.columns
            ):
                total_started = timeseries_results[started_col].sum()
                total_completed = timeseries_results[completed_col].sum()
                aggregate_conversion = (
                    (total_completed / total_started * 100) if total_started > 0 else 0
                )

                print("\n3. AGGREGATE FROM TIMESERIES:")
                print(f"   Total started: {total_started}")
                print(f"   Total completed: {total_completed}")
                print(f"   Aggregate conversion rate: {aggregate_conversion:.2f}%")

                # 4. Compare the methodologies
                print("\n4. COMPARISON:")
                print(f"   Overall funnel conversion: {overall_final_conversion:.2f}%")
                print(f"   Timeseries aggregate conversion: {aggregate_conversion:.2f}%")
                print(
                    f"   Difference: {abs(overall_final_conversion - aggregate_conversion):.2f}%"
                )

                # The discrepancy might be due to:
                # A) Different user cohort definitions
                # B) Different time window handling
                # C) Different reentry logic

                # Let's investigate the data in detail
                self._investigate_user_journeys(diagnostic_data, steps, calculator)

                # Check for major discrepancy
                conversion_diff = abs(overall_final_conversion - aggregate_conversion)
                if conversion_diff > 10:
                    print(
                        f"\n⚠️  WARNING: Large conversion rate discrepancy detected: {conversion_diff:.2f}%"
                    )
                    print(
                        "This suggests different calculation methodologies between overall and timeseries metrics."
                    )
            else:
                print(
                    f"\n⚠️  Could not find expected columns. Available: {list(timeseries_results.columns)}"
                )

        # Assertions to catch the issue
        assert len(overall_results.users_count) == 2, "Should have 2 steps"
        assert overall_results.users_count[0] > 0, "Should have users in first step"

        if not timeseries_results.empty:
            # The timeseries should not show dramatically different aggregate results
            # Allow some variance due to methodology differences, but flag major discrepancies
            started_col = (
                "started_funnel_users"
                if "started_funnel_users" in timeseries_results.columns
                else "started_count"
            )
            completed_col = (
                "completed_funnel_users"
                if "completed_funnel_users" in timeseries_results.columns
                else "completed_count"
            )

            if (
                started_col in timeseries_results.columns
                and completed_col in timeseries_results.columns
            ):
                total_started = timeseries_results[started_col].sum()
                total_completed = timeseries_results[completed_col].sum()
                aggregate_conversion = (
                    (total_completed / total_started * 100) if total_started > 0 else 0
                )

                # Flag if difference is > 10% (which indicates a methodology issue)
                conversion_diff = abs(overall_final_conversion - aggregate_conversion)
                if conversion_diff > 10:
                    print(
                        f"\n⚠️  WARNING: Large conversion rate discrepancy detected: {conversion_diff:.2f}%"
                    )
                    print(
                        "This suggests different calculation methodologies between overall and timeseries metrics."
                    )

    def _investigate_user_journeys(self, data, steps, calculator):
        """Investigate individual user journeys to understand the calculation differences."""
        print("\n=== USER JOURNEY INVESTIGATION ===")

        # Group events by user and analyze their journey
        for user_id in data["user_id"].unique()[:5]:  # Just first 5 users for clarity
            user_events = data[data["user_id"] == user_id].sort_values("timestamp")
            print(f"\nUser {user_id}:")
            for _, event in user_events.iterrows():
                print(f"  {event['timestamp']}: {event['event_name']}")

            # Check if this user would be counted as converted in overall funnel
            user_step1_events = user_events[user_events["event_name"] == steps[0]]
            user_step2_events = user_events[user_events["event_name"] == steps[1]]

            if not user_step1_events.empty and not user_step2_events.empty:
                first_step1 = user_step1_events.iloc[0]["timestamp"]
                first_step2 = user_step2_events.iloc[0]["timestamp"]
                time_diff = first_step2 - first_step1
                print(f"    → Conversion time: {time_diff}")
                print(f"    → Within 24h window: {time_diff < timedelta(hours=24)}")

    def test_conversion_window_impact(self, diagnostic_data, calculator):
        """Test how different conversion windows affect the discrepancy."""
        steps = ["step1", "step2"]

        print("\n=== CONVERSION WINDOW IMPACT TEST ===")

        # Test with different conversion windows
        for window_hours in [1, 6, 24, 168]:  # 1h, 6h, 24h, 1 week
            config = FunnelConfig(
                counting_method=CountingMethod.UNIQUE_USERS,
                funnel_order=FunnelOrder.ORDERED,
                reentry_mode=ReentryMode.FIRST_ONLY,
                conversion_window_hours=window_hours,
            )
            test_calculator = FunnelCalculator(config)

            # Overall funnel
            overall_results = test_calculator.calculate_funnel_metrics(diagnostic_data, steps)
            overall_conversion = (
                overall_results.conversion_rates[-1] if overall_results.conversion_rates else 0
            )

            # Timeseries
            timeseries_results = test_calculator.calculate_timeseries_metrics(
                diagnostic_data, steps, "1h"
            )

            started_col = (
                "started_funnel_users"
                if "started_funnel_users" in timeseries_results.columns
                else "started_count"
            )
            completed_col = (
                "completed_funnel_users"
                if "completed_funnel_users" in timeseries_results.columns
                else "completed_count"
            )

            if not timeseries_results.empty and started_col in timeseries_results.columns:
                total_started = timeseries_results[started_col].sum()
                total_completed = timeseries_results[completed_col].sum()
                ts_conversion = (total_completed / total_started * 100) if total_started > 0 else 0
            else:
                ts_conversion = 0

            print(
                f"Window {window_hours}h: Overall={overall_conversion:.1f}%, Timeseries={ts_conversion:.1f}%, "
                f"Diff={abs(overall_conversion - ts_conversion):.1f}%"
            )

    def test_reentry_mode_impact(self, diagnostic_data):
        """Test how reentry modes affect the discrepancy."""
        steps = ["step1", "step2"]

        print("\n=== REENTRY MODE IMPACT TEST ===")

        for reentry_mode in [ReentryMode.FIRST_ONLY, ReentryMode.OPTIMIZED_REENTRY]:
            config = FunnelConfig(
                counting_method=CountingMethod.UNIQUE_USERS,
                funnel_order=FunnelOrder.ORDERED,
                reentry_mode=reentry_mode,
                conversion_window_hours=24,
            )
            test_calculator = FunnelCalculator(config)

            # Overall funnel
            overall_results = test_calculator.calculate_funnel_metrics(diagnostic_data, steps)
            overall_conversion = (
                overall_results.conversion_rates[-1] if overall_results.conversion_rates else 0
            )

            # Timeseries
            timeseries_results = test_calculator.calculate_timeseries_metrics(
                diagnostic_data, steps, "1h"
            )

            started_col = (
                "started_funnel_users"
                if "started_funnel_users" in timeseries_results.columns
                else "started_count"
            )
            completed_col = (
                "completed_funnel_users"
                if "completed_funnel_users" in timeseries_results.columns
                else "completed_count"
            )

            if not timeseries_results.empty and started_col in timeseries_results.columns:
                total_started = timeseries_results[started_col].sum()
                total_completed = timeseries_results[completed_col].sum()
                ts_conversion = (total_completed / total_started * 100) if total_started > 0 else 0
            else:
                ts_conversion = 0

            print(
                f"{reentry_mode.value}: Overall={overall_conversion:.1f}%, Timeseries={ts_conversion:.1f}%, "
                f"Diff={abs(overall_conversion - ts_conversion):.1f}%"
            )
