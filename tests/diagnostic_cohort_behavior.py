#!/usr/bin/env python3
"""
Diagnostic script to examine the actual behavior of timeseries calculation
and identify where the cohort attribution bug occurs.
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


def diagnose_cohort_attribution():
    """Diagnose the actual behavior of cohort attribution in timeseries."""

    # Create test data that should expose the attribution bug clearly
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
        # Additional user who signs up on Jan 2 and converts same day
        {
            "user_id": "user_B",
            "event_name": "signup",
            "timestamp": datetime(2024, 1, 2, 10, 0, 0),  # Jan 2, 10:00
            "event_properties": "{}",
            "user_properties": "{}",
        },
        {
            "user_id": "user_B",
            "event_name": "purchase",
            "timestamp": datetime(2024, 1, 2, 15, 0, 0),  # Jan 2, 15:00
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
                "timestamp": datetime(2024, 1, 2, 11 + i % 8, 0, 0),  # Jan 2, various hours
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    events_df = pd.DataFrame(events_data)
    funnel_steps = ["signup", "purchase"]

    print("📊 EVENT DATA SUMMARY:")
    print(f"Total events: {len(events_df)}")
    print("\\nEvents by date:")
    events_df["date"] = events_df["timestamp"].dt.date
    date_summary = events_df.groupby(["date", "event_name"]).size().unstack(fill_value=0)
    print(date_summary)
    print()

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

    print("🔍 TIMESERIES RESULTS:")
    print(
        results[
            ["period_date", "started_funnel_users", "completed_funnel_users", "conversion_rate"]
        ].to_string()
    )
    print()

    # Analyze what SHOULD happen vs what IS happening
    print("📈 EXPECTED COHORT BEHAVIOR:")
    print("Jan 1 cohort: 1 signup (user_A) -> 1 conversion -> 100% rate")
    print("Jan 2 cohort: 11 signups (user_B + 10 non-converters) -> 1 conversion -> 9.09% rate")
    print()

    print("📉 POTENTIAL BUG BEHAVIOR:")
    print("Jan 1: 1 signup -> 0 conversions (user_A converts next day)")
    print("Jan 2: 11 signups -> 2 conversions (user_A + user_B both convert on Jan 2)")
    print()

    # Detailed analysis
    results_dict = {}
    for _, row in results.iterrows():
        date_key = row["period_date"].strftime("%Y-%m-%d")
        results_dict[date_key] = row.to_dict()

    print("🔍 ACTUAL RESULTS ANALYSIS:")
    for date, data in results_dict.items():
        print(
            f"{date}: {data['started_funnel_users']} starters, {data['completed_funnel_users']} completers, {data['conversion_rate']:.1f}% rate"
        )

    # Check if this reveals the bug
    if "2024-01-01" in results_dict and "2024-01-02" in results_dict:
        jan_1 = results_dict["2024-01-01"]
        jan_2 = results_dict["2024-01-02"]

        print("\\n🧐 BUG DETECTION:")

        # If working correctly (cohort-based):
        # Jan 1: 1 starter, 1 completer (user_A)
        # Jan 2: 11 starters, 1 completer (user_B)

        # If buggy (event-date based):
        # Jan 1: 1 starter, 0 completers
        # Jan 2: 11 starters, 2 completers (both user_A and user_B)

        if jan_1["completed_funnel_users"] == 0 and jan_2["completed_funnel_users"] == 2:
            print("🚨 BUG DETECTED: Conversions attributed to conversion date, not cohort date!")
            print(
                f"  Jan 1 should have 1 completer (user_A), but got {jan_1['completed_funnel_users']}"
            )
            print(
                f"  Jan 2 should have 1 completer (user_B), but got {jan_2['completed_funnel_users']}"
            )
        elif jan_1["completed_funnel_users"] == 1 and jan_2["completed_funnel_users"] == 1:
            print("✅ CORRECT: Conversions properly attributed to cohort start dates!")
        else:
            print(
                f"🤔 UNEXPECTED RESULT: Jan 1 = {jan_1['completed_funnel_users']}, Jan 2 = {jan_2['completed_funnel_users']}"
            )

    return results


def test_pandas_vs_polars():
    """Test if the bug exists in pandas vs polars implementations."""

    events_data = [
        {
            "user_id": "user_A",
            "event_name": "signup",
            "timestamp": datetime(2024, 1, 1, 23, 0, 0),
            "event_properties": "{}",
            "user_properties": "{}",
        },
        {
            "user_id": "user_A",
            "event_name": "purchase",
            "timestamp": datetime(2024, 1, 2, 1, 0, 0),
            "event_properties": "{}",
            "user_properties": "{}",
        },
    ]

    events_df = pd.DataFrame(events_data)
    funnel_steps = ["signup", "purchase"]

    config = FunnelConfig(conversion_window_hours=24)
    calculator = FunnelCalculator(config)

    print("\\n🔄 TESTING PANDAS VS POLARS IMPLEMENTATIONS:")

    # Force pandas
    try:
        pandas_result = calculator._calculate_timeseries_metrics_pandas(
            events_df, funnel_steps, "1d"
        )
        print("📊 Pandas implementation result:")
        print(
            pandas_result[
                [
                    "period_date",
                    "started_funnel_users",
                    "completed_funnel_users",
                    "conversion_rate",
                ]
            ].to_string()
        )
    except Exception as e:
        print(f"❌ Pandas implementation failed: {e}")

    # Force polars
    try:
        import polars as pl

        polars_df = pl.from_pandas(events_df)
        polars_result = calculator._calculate_timeseries_metrics_polars(
            polars_df, funnel_steps, "1d"
        )
        print("\\n📊 Polars implementation result:")
        print(
            polars_result[
                [
                    "period_date",
                    "started_funnel_users",
                    "completed_funnel_users",
                    "conversion_rate",
                ]
            ].to_string()
        )
    except Exception as e:
        print(f"❌ Polars implementation failed: {e}")


if __name__ == "__main__":
    print("🔍 DIAGNOSING COHORT ATTRIBUTION BEHAVIOR...")
    diagnose_cohort_attribution()
    test_pandas_vs_polars()
