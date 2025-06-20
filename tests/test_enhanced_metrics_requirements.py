#!/usr/bin/env python3
"""
Better breaking test that demonstrates the real cohort attribution issue.

The issue is likely in how daily active users and total events are calculated -
these should represent the activity that happened on that date, not attributed to cohorts.
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


def test_daily_metrics_vs_cohort_metrics():
    """
    Test the difference between cohort-based conversion metrics and daily activity metrics.

    The key insight is that we need TWO different types of metrics:
    1. Cohort conversion metrics (attributed to signup date)
    2. Daily activity metrics (attributed to event date)
    """

    # Create test scenario with clear cross-day conversions
    events_data = []

    # Day 1: 5 users sign up
    for i in range(5):
        events_data.append(
            {
                "user_id": f"day1_user_{i}",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 1, 10 + i, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    # Day 2: 10 more users sign up + 2 users from day 1 convert
    for i in range(10):
        events_data.append(
            {
                "user_id": f"day2_user_{i}",
                "event_name": "signup",
                "timestamp": datetime(2024, 1, 2, 9 + i, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    # 2 users from day 1 convert on day 2
    events_data.extend(
        [
            {
                "user_id": "day1_user_0",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 2, 14, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "day1_user_1",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 2, 16, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]
    )

    # Day 3: 3 users from day 2 convert
    events_data.extend(
        [
            {
                "user_id": "day2_user_0",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 3, 11, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "day2_user_1",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 3, 13, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
            {
                "user_id": "day2_user_2",
                "event_name": "purchase",
                "timestamp": datetime(2024, 1, 3, 15, 0, 0),
                "event_properties": "{}",
                "user_properties": "{}",
            },
        ]
    )

    events_df = pd.DataFrame(events_data)

    print("📊 EVENT DISTRIBUTION:")
    print("Events by date and type:")
    events_df["date"] = events_df["timestamp"].dt.date
    summary = events_df.groupby(["date", "event_name"]).size().unstack(fill_value=0)
    print(summary)
    print()

    # Create calculator
    config = FunnelConfig(
        conversion_window_hours=72,  # 3 days
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED,
    )
    calculator = FunnelCalculator(config)

    # Get current results
    results = calculator.calculate_timeseries_metrics(
        events_df, ["signup", "purchase"], "1d"
    )

    print("🔍 CURRENT RESULTS:")
    for _, row in results.iterrows():
        date = row["period_date"].strftime("%Y-%m-%d")
        print(
            f"{date}: {row['started_funnel_users']} starters, {row['completed_funnel_users']} completers"
        )
        print(f"  Cohort conversion: {row['conversion_rate']:.1f}%")
        print(
            f"  Daily metrics: {row['total_unique_users']} unique users, {row['total_events']} events"
        )
        print()

    print("📈 WHAT WE NEED - TWO TYPES OF METRICS:")
    print()
    print("1. COHORT CONVERSION METRICS (attributed to signup date):")
    print("   Jan 1 cohort: 5 starters → 2 converters → 40% conversion")
    print("   Jan 2 cohort: 10 starters → 3 converters → 30% conversion")
    print("   Jan 3 cohort: 0 starters → 0 converters → N/A")
    print()
    print("2. DAILY ACTIVITY METRICS (attributed to event date):")
    print("   Jan 1: 5 unique users, 5 events (all signups)")
    print("   Jan 2: 12 unique users, 12 events (10 signups + 2 purchases)")
    print("   Jan 3: 3 unique users, 3 events (all purchases)")
    print()

    # Check if the current implementation gives us the new metrics we need
    expected_cohort_metrics = {
        "2024-01-01": {"starters": 5, "completers": 2, "rate": 40.0},
        "2024-01-02": {"starters": 10, "completers": 3, "rate": 30.0},
    }

    expected_daily_metrics = {
        "2024-01-01": {"users": 5, "events": 5},
        "2024-01-02": {"users": 12, "events": 12},
        "2024-01-03": {"users": 3, "events": 3},
    }

    print("🧪 TESTING FOR NEW METRICS REQUIREMENTS:")

    # Test cohort metrics
    results_dict = {}
    for _, row in results.iterrows():
        date_key = row["period_date"].strftime("%Y-%m-%d")
        results_dict[date_key] = row.to_dict()

    cohort_errors = []
    for date, expected in expected_cohort_metrics.items():
        if date in results_dict:
            actual = results_dict[date]
            if actual["started_funnel_users"] != expected["starters"]:
                cohort_errors.append(
                    f"{date}: Expected {expected['starters']} starters, got {actual['started_funnel_users']}"
                )
            if actual["completed_funnel_users"] != expected["completers"]:
                cohort_errors.append(
                    f"{date}: Expected {expected['completers']} completers, got {actual['completed_funnel_users']}"
                )

    # Check for missing daily metrics that we want to add
    daily_errors = []
    for date, expected in expected_daily_metrics.items():
        if date not in results_dict:
            daily_errors.append(
                f"{date}: Missing from results but had {expected['users']} active users"
            )
        else:
            actual = results_dict[date]
            # The issue: we should have daily_active_users and daily_events_total as NEW metrics
            if "daily_active_users" not in actual:
                daily_errors.append("Missing 'daily_active_users' metric")
            if "daily_events_total" not in actual:
                daily_errors.append("Missing 'daily_events_total' metric")

    print("❌ COHORT METRIC ISSUES:")
    if cohort_errors:
        for error in cohort_errors:
            print(f"  {error}")
    else:
        print("  ✅ Cohort metrics look correct")

    print("❌ MISSING NEW DAILY METRICS:")
    if daily_errors:
        for error in daily_errors:
            print(f"  {error}")
    else:
        print("  ✅ All required daily metrics present")

    return len(cohort_errors) > 0 or len(daily_errors) > 0


def test_current_total_users_attribution():
    """
    Test whether the current 'total_unique_users' metric is calculated correctly.

    This should show daily activity, not cohort activity.
    """
    events_data = [
        # User A: signup on Jan 1, purchase on Jan 2
        {
            "user_id": "user_A",
            "event_name": "signup",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            "event_properties": "{}",
            "user_properties": "{}",
        },
        {
            "user_id": "user_A",
            "event_name": "purchase",
            "timestamp": datetime(2024, 1, 2, 11, 0, 0),
            "event_properties": "{}",
            "user_properties": "{}",
        },
        # User B: only appears on Jan 2
        {
            "user_id": "user_B",
            "event_name": "signup",
            "timestamp": datetime(2024, 1, 2, 12, 0, 0),
            "event_properties": "{}",
            "user_properties": "{}",
        },
    ]

    events_df = pd.DataFrame(events_data)
    config = FunnelConfig(conversion_window_hours=24)
    calculator = FunnelCalculator(config)

    results = calculator.calculate_timeseries_metrics(
        events_df, ["signup", "purchase"], "1d"
    )

    print("\\n🔍 TESTING DAILY METRICS ATTRIBUTION:")
    for _, row in results.iterrows():
        date = row["period_date"].strftime("%Y-%m-%d")
        print(
            f"{date}: total_unique_users = {row['total_unique_users']}, total_events = {row['total_events']}"
        )

    # Expected daily activity:
    # Jan 1: 1 user (A), 1 event (signup)
    # Jan 2: 2 users (A, B), 2 events (purchase + signup)

    results_dict = {}
    for _, row in results.iterrows():
        date_key = row["period_date"].strftime("%Y-%m-%d")
        results_dict[date_key] = row.to_dict()

    issues = []
    if results_dict["2024-01-01"]["total_unique_users"] != 1:
        issues.append(
            f"Jan 1 should have 1 unique user, got {results_dict['2024-01-01']['total_unique_users']}"
        )
    if results_dict["2024-01-02"]["total_unique_users"] != 2:
        issues.append(
            f"Jan 2 should have 2 unique users, got {results_dict['2024-01-02']['total_unique_users']}"
        )

    if issues:
        print("❌ ISSUES WITH DAILY METRICS:")
        for issue in issues:
            print(f"  {issue}")
        return True
    print("✅ Daily metrics calculated correctly")
    return False


if __name__ == "__main__":
    print("🚨 TESTING ENHANCED METRICS REQUIREMENTS...")

    has_issues = test_daily_metrics_vs_cohort_metrics()
    has_daily_issues = test_current_total_users_attribution()

    if has_issues or has_daily_issues:
        print(
            "\\n🔧 CONCLUSION: Implementation needs enhancement to provide comprehensive metrics"
        )
        print("   - Add daily_active_users and daily_events_total metrics")
        print("   - Ensure clear separation between cohort and daily metrics")
    else:
        print("\\n✅ All metrics working as expected")
