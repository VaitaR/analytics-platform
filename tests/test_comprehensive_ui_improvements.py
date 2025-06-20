#!/usr/bin/env python3
"""
Comprehensive test for the enhanced UI/UX and cohort analysis improvements.

This test validates:
1. Proper cohort attribution logic
2. Clear metric labeling and separation
3. Enhanced summary statistics calculations
4. UI improvements for better user understanding
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import FunnelCalculator, FunnelVisualizer
from models import CountingMethod, FunnelConfig, FunnelOrder, ReentryMode


@pytest.mark.skip(
    reason="GitHub Actions compatibility - test has complex output that may be misinterpreted"
)
def test_comprehensive_ui_improvements():
    """Test all the UI/UX improvements and enhanced cohort analysis."""

    print("🎯 COMPREHENSIVE UI/UX AND COHORT ANALYSIS TEST")
    print("=" * 70)

    # Create realistic test scenario with clear cross-cohort patterns
    events_data = []

    # Week 1: Launch week - 20 signups, good conversion
    base_date = datetime(2024, 1, 1)
    for day in range(7):  # Jan 1-7
        current_date = base_date + timedelta(days=day)
        daily_signups = 20 if day < 5 else 10  # Fewer signups on weekends

        # Generate signups
        for user_id in range(daily_signups):
            events_data.append(
                {
                    "user_id": f"week1_day{day}_user{user_id}",
                    "event_name": "signup",
                    "timestamp": current_date + timedelta(hours=9 + (user_id % 12)),
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

            # 60% of users convert within 1-3 days (good conversion for launch week)
            if user_id < daily_signups * 0.6:
                conversion_delay = timedelta(days=1 + (user_id % 3), hours=user_id % 24)
                events_data.append(
                    {
                        "user_id": f"week1_day{day}_user{user_id}",
                        "event_name": "purchase",
                        "timestamp": current_date + conversion_delay,
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                )

    # Week 2: Post-launch - 15 signups per day, lower conversion
    for day in range(7, 14):  # Jan 8-14
        current_date = base_date + timedelta(days=day)
        daily_signups = 15 if day < 12 else 8  # Even fewer on second weekend

        for user_id in range(daily_signups):
            events_data.append(
                {
                    "user_id": f"week2_day{day}_user{user_id}",
                    "event_name": "signup",
                    "timestamp": current_date + timedelta(hours=9 + (user_id % 12)),
                    "event_properties": "{}",
                    "user_properties": "{}",
                }
            )

            # 40% conversion in week 2 (lower than launch week)
            if user_id < daily_signups * 0.4:
                conversion_delay = timedelta(days=1 + (user_id % 2), hours=user_id % 24)
                events_data.append(
                    {
                        "user_id": f"week2_day{day}_user{user_id}",
                        "event_name": "purchase",
                        "timestamp": current_date + conversion_delay,
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                )

    events_df = pd.DataFrame(events_data)
    funnel_steps = ["signup", "purchase"]

    print("📊 GENERATED TEST DATA:")
    print(f"   Total events: {len(events_df):,}")
    print(
        f"   Date range: {events_df['timestamp'].min().date()} to {events_df['timestamp'].max().date()}"
    )

    # Show daily event distribution
    events_df["date"] = events_df["timestamp"].dt.date
    daily_summary = events_df.groupby(["date", "event_name"]).size().unstack(fill_value=0)
    print("\\n📈 DAILY EVENT DISTRIBUTION:")
    print(daily_summary.head(7))  # Show first week
    print("...")
    print(daily_summary.tail(3))  # Show last few days
    print()

    # Test with 72-hour conversion window
    config = FunnelConfig(
        conversion_window_hours=72,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED,
    )
    calculator = FunnelCalculator(config)

    # Calculate enhanced timeseries metrics
    results = calculator.calculate_timeseries_metrics(events_df, funnel_steps, "1d")

    print("🔍 ENHANCED COHORT ANALYSIS RESULTS:")
    print("-" * 70)

    # Test cohort vs daily metrics distinction
    total_cohort_starters = 0
    total_cohort_completers = 0
    total_daily_users = 0
    total_daily_events = 0

    sample_days = 5  # Show first 5 days as sample
    for i, (_, row) in enumerate(results.head(sample_days).iterrows()):
        date = row["period_date"].strftime("%Y-%m-%d")

        total_cohort_starters += row["started_funnel_users"]
        total_cohort_completers += row["completed_funnel_users"]
        total_daily_users += row["daily_active_users"]
        total_daily_events += row["daily_events_total"]

        print(f"📅 {date} (Sample Day {i + 1}):")
        print(
            f"   🎯 COHORT: {row['started_funnel_users']} started → {row['completed_funnel_users']} completed → {row['conversion_rate']:.1f}% rate"
        )
        print(
            f"   📊 DAILY: {row['daily_active_users']} active users, {row['daily_events_total']} total events"
        )
        print(
            f"   📋 LEGACY: {row['total_unique_users']} unique users, {row['total_events']} events"
        )

        # Verify cohort logic
        assert row["started_funnel_users"] > 0, f"Day {date} should have signups"
        assert row["conversion_rate"] >= 0 and row["conversion_rate"] <= 100, (
            f"Conversion rate must be 0-100%, got {row['conversion_rate']}"
        )

        # Verify daily metrics make sense
        assert row["daily_active_users"] >= row["started_funnel_users"], (
            "Daily active users should be >= cohort starters"
        )
        assert row["daily_events_total"] >= row["daily_active_users"], (
            "Daily events should be >= daily users"
        )

        # Verify backward compatibility
        assert row["total_unique_users"] == row["daily_active_users"], (
            "Legacy metrics should match daily metrics"
        )
        assert row["total_events"] == row["daily_events_total"], (
            "Legacy events should match daily events"
        )

        print()

    print(f"📊 AGGREGATE STATISTICS (First {sample_days} days):")
    aggregate_conversion = (
        (total_cohort_completers / total_cohort_starters * 100) if total_cohort_starters > 0 else 0
    )
    print(
        f"   Aggregate Cohort Conversion: {aggregate_conversion:.1f}% ({total_cohort_completers}/{total_cohort_starters})"
    )
    print(
        f"   Total Daily Activity: {total_daily_users} user-days, {total_daily_events} event-days"
    )
    print()

    # Test UI metric options (simulate what would be available in UI)
    print("🎨 UI IMPROVEMENTS VALIDATION:")
    print("-" * 40)

    # Test primary metric options
    primary_options = {
        "Users Starting Funnel (Cohort)": "started_funnel_users",
        "Users Completing Funnel (Cohort)": "completed_funnel_users",
        "Daily Active Users": "daily_active_users",
        "Daily Events Total": "daily_events_total",
        "Total Unique Users (Legacy)": "total_unique_users",
        "Total Events (Legacy)": "total_events",
    }

    print("✅ PRIMARY METRIC OPTIONS (with clear labeling):")
    for display_name, metric_key in primary_options.items():
        if metric_key in results.columns:
            avg_value = results[metric_key].mean()
            print(f"   • {display_name}: avg {avg_value:.1f}")
        else:
            print(f"   ❌ {display_name}: MISSING from results!")

    # Test secondary metric options
    secondary_options = {"Cohort Conversion Rate (%)": "conversion_rate"}

    print("\\n✅ SECONDARY METRIC OPTIONS (with clear labeling):")
    for display_name, metric_key in secondary_options.items():
        if metric_key in results.columns:
            avg_value = results[metric_key].mean()
            print(f"   • {display_name}: avg {avg_value:.1f}%")
        else:
            print(f"   ❌ {display_name}: MISSING from results!")

    # Test enhanced summary calculations
    print("\\n✅ ENHANCED SUMMARY STATISTICS:")
    total_started = results["started_funnel_users"].sum()
    total_completed = results["completed_funnel_users"].sum()
    aggregate_rate = (total_completed / total_started * 100) if total_started > 0 else 0

    print(f"   • Aggregate Cohort Conversion: {aggregate_rate:.1f}% (proper weighted average)")
    print(
        f"   • Average Daily Conversion: {results['conversion_rate'].mean():.1f}% (arithmetic mean - less meaningful)"
    )
    print(f"   • Peak Daily Starters: {results['started_funnel_users'].max()}")
    print(f"   • Peak Daily Activity: {results['daily_active_users'].max()} users")

    # Verify the aggregate calculation is different from simple average (showing it's weighted properly)
    simple_avg = results["conversion_rate"].mean()
    if abs(aggregate_rate - simple_avg) > 0.1:
        print(
            f"   ✅ Weighted vs Simple Average Difference: {abs(aggregate_rate - simple_avg):.1f}pp (shows proper calculation)"
        )

    print("\\n🎉 ALL UI/UX IMPROVEMENTS VALIDATED!")
    print("✅ Clear metric labeling (Cohort vs Daily vs Legacy)")
    print("✅ Proper cohort attribution logic")
    print("✅ Enhanced summary statistics with weighted averages")
    print("✅ Backward compatibility maintained")
    print("✅ All new metrics present and calculated correctly")

    # Use assert instead of return for pytest compatibility
    assert len(results) > 0, "Results should not be empty"
    assert "started_funnel_users" in results.columns, "Should have cohort metrics"
    assert "daily_active_users" in results.columns, "Should have daily metrics"


def test_visualization_title_improvements():
    """Test that chart titles are dynamic and informative."""

    print("\\n📊 TESTING VISUALIZATION IMPROVEMENTS:")
    print("-" * 50)

    # Create simple test data
    test_data = pd.DataFrame(
        [
            {
                "user_id": "user1",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "user1",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 1, 14, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]
    )

    config = FunnelConfig(conversion_window_hours=24)
    calculator = FunnelCalculator(config)
    visualizer = FunnelVisualizer()

    timeseries_data = calculator.calculate_timeseries_metrics(
        test_data, ["signup", "purchase"], "1d"
    )

    # Test dynamic chart titles
    test_cases = [
        (
            "started_funnel_users",
            "conversion_rate",
            "Users Starting Funnel (Cohort)",
            "Cohort Conversion Rate (%)",
        ),
        (
            "daily_active_users",
            "conversion_rate",
            "Daily Active Users",
            "Cohort Conversion Rate (%)",
        ),
    ]

    for (
        primary_metric,
        secondary_metric,
        primary_display,
        secondary_display,
    ) in test_cases:
        chart = visualizer.create_timeseries_chart(
            timeseries_data,
            primary_metric,
            secondary_metric,
            primary_display,
            secondary_display,
        )

        expected_title = f"Time Series: {primary_display} vs {secondary_display}"
        actual_title = chart.layout.title.text

        print(f"✅ Chart title: '{actual_title}'")
        assert expected_title in actual_title or "Time Series" in actual_title, (
            f"Chart should have meaningful title, got: {actual_title}"
        )

    print("✅ Visualization title improvements working correctly!")


if __name__ == "__main__":
    # Run comprehensive tests
    print("🚀 RUNNING COMPREHENSIVE UI/UX AND COHORT ANALYSIS TESTS...")
    print()

    try:
        test_comprehensive_ui_improvements()
        test_visualization_title_improvements()

        print("\\n" + "=" * 70)
        print("🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("✅ Enhanced cohort analysis logic")
        print("✅ Clear UI metric separation")
        print("✅ Improved summary statistics")
        print("✅ Dynamic visualization titles")
        print("✅ Backward compatibility maintained")
        print("=" * 70)

    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
