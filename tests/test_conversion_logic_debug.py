"""
Debug test to understand the conversion rate calculation discrepancy.
This test specifically investigates the impossible situation where daily conversion rates
are higher than mathematically possible given the total conversion counts.
"""

from datetime import timedelta

import pandas as pd

from app import CountingMethod, FunnelCalculator, FunnelConfig, FunnelOrder, ReentryMode


class TestConversionRateLogicBug:
    """Test to identify the logic bug in conversion rate calculations."""

    def test_debug_conversion_rate_calculation(self):
        """
        Debug test to reproduce the impossible conversion rate scenario.

        Issue: User reports 0.9% overall (368/42458) but 42.7% daily conversion rate.
        This is mathematically impossible - if 42.7% were correct, there should be ~18,129 conversions.
        """
        print("\n=== CONVERSION RATE LOGIC DEBUG ===")

        # Let's load some sample data to understand the calculation
        from tests.conftest import TestDataFactory, TestDataSpec

        # Create a dataset that might trigger this issue
        spec = TestDataSpec(
            total_users=1000,
            conversion_rates=[1.0, 0.02],  # 2% actual conversion rate
            time_spread_hours=168,  # 1 week of data
            include_noise_events=False,
        )

        funnel_steps = ["step1", "step2"]
        test_data = TestDataFactory.create_funnel_data(spec, funnel_steps)

        print("Generated test data:")
        print(f"- Total events: {len(test_data)}")
        print(
            f"- Users in step1: {test_data[test_data['event_name'] == 'step1']['user_id'].nunique()}"
        )
        print(
            f"- Users in step2: {test_data[test_data['event_name'] == 'step2']['user_id'].nunique()}"
        )

        # Test with different configurations
        config = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            conversion_window_hours=168,  # 1 week window
        )
        calculator = FunnelCalculator(config, use_polars=True)

        # 1. Calculate overall funnel metrics
        overall_results = calculator.calculate_funnel_metrics(test_data, funnel_steps)

        print("\n1. OVERALL FUNNEL RESULTS:")
        print(f"   Steps: {overall_results.steps}")
        print(f"   User counts: {overall_results.users_count}")
        print(f"   Conversion rates: {overall_results.conversion_rates}")

        overall_started = overall_results.users_count[0]
        overall_completed = overall_results.users_count[1]
        overall_conversion = overall_results.conversion_rates[1]

        print(
            f"   Manual verification: {overall_completed}/{overall_started} = {(overall_completed / overall_started * 100):.2f}%"
        )

        # 2. Calculate daily timeseries metrics
        daily_results = calculator.calculate_timeseries_metrics(test_data, funnel_steps, "1d")

        print("\n2. DAILY TIMESERIES RESULTS:")
        print(f"   Shape: {daily_results.shape}")

        if not daily_results.empty:
            # Show detailed breakdown
            print(f"   Columns: {list(daily_results.columns)}")

            total_ts_started = daily_results["started_funnel_users"].sum()
            total_ts_completed = daily_results["completed_funnel_users"].sum()
            weighted_avg_conversion = daily_results["conversion_rate"].mean()

            print("\n   TIMESERIES AGGREGATION:")
            print(f"   Total started (sum): {total_ts_started}")
            print(f"   Total completed (sum): {total_ts_completed}")
            print(f"   Aggregate conversion: {(total_ts_completed / total_ts_started * 100):.2f}%")
            print(f"   Average conversion rate: {weighted_avg_conversion:.2f}%")

            # Show daily breakdown
            print("\n   DAILY BREAKDOWN:")
            for idx, row in daily_results.iterrows():
                print(
                    f"     {row['period_date']}: {row['started_funnel_users']} started, "
                    f"{row['completed_funnel_users']} completed, "
                    f"{row['conversion_rate']:.2f}% conversion"
                )

        # 3. CRITICAL CHECK: Are the totals consistent?
        print("\n3. CONSISTENCY CHECK:")

        if not daily_results.empty:
            total_ts_started = daily_results["started_funnel_users"].sum()
            total_ts_completed = daily_results["completed_funnel_users"].sum()

            print(f"   Overall funnel:  {overall_started} started, {overall_completed} completed")
            print(
                f"   Timeseries sum:  {total_ts_started} started, {total_ts_completed} completed"
            )

            started_diff = abs(overall_started - total_ts_started)
            completed_diff = abs(overall_completed - total_ts_completed)

            print(f"   Difference in started: {started_diff}")
            print(f"   Difference in completed: {completed_diff}")

            if started_diff > 0 or completed_diff > 0:
                print("\n   ⚠️  INCONSISTENCY DETECTED!")
                print("   This suggests the two methods are counting different users!")

                # Let's investigate WHY they differ
                self._investigate_counting_difference(test_data, funnel_steps, calculator)

        # 4. Test the actual user's scenario
        self._test_user_reported_scenario()

    def _investigate_counting_difference(self, test_data, funnel_steps, calculator):
        """Investigate why overall and timeseries counting might differ."""
        print("\n=== INVESTIGATING COUNTING DIFFERENCE ===")

        # Manual calculation for overall
        step1_users = set(test_data[test_data["event_name"] == funnel_steps[0]]["user_id"])
        step2_users = set(test_data[test_data["event_name"] == funnel_steps[1]]["user_id"])
        step2_who_did_step1 = step2_users.intersection(step1_users)

        print("MANUAL OVERALL CALCULATION:")
        print(f"  Users who did step1: {len(step1_users)}")
        print(f"  Users who did step2: {len(step2_users)}")
        print(f"  Users who did both: {len(step2_who_did_step1)}")
        print(f"  Manual conversion: {len(step2_who_did_step1) / len(step1_users) * 100:.2f}%")

        # Check timeseries cohort logic
        print("\nCOHORT LOGIC INVESTIGATION:")

        # Group by day and see what happens
        test_data["date"] = pd.to_datetime(test_data["timestamp"]).dt.date

        daily_breakdown = {}
        for date in test_data["date"].unique():
            day_data = test_data[test_data["date"] == date]

            # Users who STARTED on this day
            day_starters = set(day_data[day_data["event_name"] == funnel_steps[0]]["user_id"])

            # Of those starters, who completed step2 ever (within window)?
            day_completers = set()
            for user in day_starters:
                user_events = test_data[test_data["user_id"] == user].sort_values("timestamp")
                step1_events = user_events[user_events["event_name"] == funnel_steps[0]]
                step2_events = user_events[user_events["event_name"] == funnel_steps[1]]

                if not step1_events.empty and not step2_events.empty:
                    first_step1 = step1_events.iloc[0]["timestamp"]
                    first_step2 = step2_events.iloc[0]["timestamp"]

                    # Check if step2 came after step1 and within conversion window
                    if first_step2 >= first_step1:
                        time_diff = first_step2 - first_step1
                        if time_diff <= timedelta(hours=168):  # 1 week window
                            day_completers.add(user)

            daily_breakdown[date] = {
                "started": len(day_starters),
                "completed": len(day_completers),
                "conversion": (
                    len(day_completers) / len(day_starters) * 100 if day_starters else 0
                ),
            }

            print(
                f"  {date}: {len(day_starters)} started, {len(day_completers)} completed, "
                f"{len(day_completers) / len(day_starters) * 100 if day_starters else 0:.2f}%"
            )

        # Sum cohort totals
        total_cohort_started = sum(d["started"] for d in daily_breakdown.values())
        total_cohort_completed = sum(d["completed"] for d in daily_breakdown.values())

        print("\n  COHORT TOTALS:")
        print(f"  Total cohort started: {total_cohort_started}")
        print(f"  Total cohort completed: {total_cohort_completed}")
        print(
            f"  Cohort aggregate conversion: {total_cohort_completed / total_cohort_started * 100:.2f}%"
        )

        # The key insight: cohort logic might double-count or miss users!
        print("\n  POTENTIAL ISSUES:")
        print("  1. Users who start on multiple days are counted multiple times")
        print("  2. Users who start on one day but convert on another might be missed")
        print("  3. Conversion window enforcement differs between methods")

    def _test_user_reported_scenario(self):
        """Test the specific scenario reported by the user."""
        print("\n=== USER REPORTED SCENARIO TEST ===")
        print("User reports:")
        print("- Overall: 368 out of 42,458 = 0.9%")
        print("- Daily: 42.7% conversion rate")
        print("- This is impossible: 42,458 × 0.427 = 18,129 expected conversions")
        print("- But only 368 actual conversions")
        print("")
        print("POSSIBLE EXPLANATIONS:")
        print("1. Time series is using different data filtering")
        print("2. Conversion window logic differs between methods")
        print("3. Reentry mode handling creates double-counting")
        print("4. Period boundary handling causes users to be missed/duplicated")
        print("5. Different funnel step validation between methods")


def test_debug_conversion_bug():
    """Standalone test function."""
    tester = TestConversionRateLogicBug()
    tester.test_debug_conversion_rate_calculation()
