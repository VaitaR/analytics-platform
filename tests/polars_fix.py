#!/usr/bin/env python3


import numpy as np
import pandas as pd
import polars as pl

# Test the optimized path analysis with some synthetic data
# This will verify our implementation and help debug the issues


def main():
    print("Generating test data...")

    # Generate synthetic users
    np.random.seed(42)
    user_ids = [f"user_{i}" for i in range(1000)]

    # Generate events with timestamps
    events = []
    for user_id in user_ids:
        # Add some funnel events
        timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(
            seconds=np.random.randint(0, 86400 * 30)
        )

        # Add signup event
        events.append({"user_id": user_id, "event_name": "signup", "timestamp": timestamp})

        # Add login event (90% of users)
        if np.random.random() < 0.9:
            timestamp += pd.Timedelta(minutes=np.random.randint(1, 60))
            events.append({"user_id": user_id, "event_name": "login", "timestamp": timestamp})

            # Add some events between login and purchase
            for _ in range(np.random.randint(0, 5)):
                timestamp += pd.Timedelta(minutes=np.random.randint(1, 60))
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": f"event_{np.random.randint(1, 10)}",
                        "timestamp": timestamp,
                    }
                )

            # Add purchase event (70% of logged-in users)
            if np.random.random() < 0.7:
                timestamp += pd.Timedelta(minutes=np.random.randint(1, 120))
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "purchase",
                        "timestamp": timestamp,
                    }
                )

    # Convert to DataFrame
    df = pd.DataFrame(events)

    # Create funnel steps
    funnel_steps = ["signup", "login", "purchase"]

    # Create a PlFunnelTest class to test our implementations
    class PlFunnelTest:
        def __init__(self):
            self.logger = type(
                "Logger",
                (),
                {
                    "info": lambda msg: print(f"INFO: {msg}"),
                    "warning": lambda msg: print(f"WARNING: {msg}"),
                    "error": lambda msg: print(f"ERROR: {msg}"),
                    "debug": lambda msg: None,
                },
            )

            # Mock config with conversion window
            from dataclasses import dataclass

            @dataclass
            class MockConfig:
                conversion_window_hours: int = 24

            self.config = MockConfig()

        def _to_polars(self, df):
            """Convert pandas DataFrame to polars DataFrame"""
            # Handle datetime columns explicitly
            df_copy = df.copy()

            # Handle timestamp column properly
            if "timestamp" in df_copy.columns:
                df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

            # Ensure user_id is string type to avoid mixed types
            if "user_id" in df_copy.columns:
                df_copy["user_id"] = df_copy["user_id"].astype(str)

            # Create schema for explicit type handling
            schema = {
                "timestamp": pl.Datetime,
                "user_id": pl.Utf8,
                "event_name": pl.Utf8,
            }

            # Convert to polars with schema
            return pl.from_pandas(df_copy, schema_overrides=schema)

        def test_between_steps_events(self):
            """Test that we can correctly calculate between-steps events"""
            # Convert to Polars
            pl_df = self._to_polars(df)

            # Filter to relevant events
            pl_funnel_events = pl_df.filter(pl.col("event_name").is_in(funnel_steps))

            # Create a simple manual implementation for between-steps events
            step_pairs = []
            for i in range(len(funnel_steps) - 1):
                step = funnel_steps[i]
                next_step = funnel_steps[i + 1]

                # Get users who did both steps
                step_users = (
                    pl_funnel_events.filter(pl.col("event_name") == step)["user_id"]
                    .unique()
                    .to_list()
                )
                next_step_users = (
                    pl_funnel_events.filter(pl.col("event_name") == next_step)["user_id"]
                    .unique()
                    .to_list()
                )
                common_users = list(set(step_users).intersection(set(next_step_users)))

                if not common_users:
                    continue

                # For each user, get their step timestamps
                user_data = []
                for user_id in common_users:
                    user_events = pl_funnel_events.filter(pl.col("user_id") == user_id)
                    user_step_events = user_events.filter(pl.col("event_name") == step)
                    user_next_step_events = user_events.filter(pl.col("event_name") == next_step)

                    if user_step_events.height > 0 and user_next_step_events.height > 0:
                        step_time = user_step_events.select(pl.col("timestamp").min()).item()
                        next_step_time = user_next_step_events.select(
                            pl.col("timestamp").min()
                        ).item()

                        # Only consider cases where next_step came after step
                        if next_step_time > step_time:
                            user_data.append(
                                {
                                    "user_id": user_id,
                                    "step": step,
                                    "next_step": next_step,
                                    "step_A_time": step_time,
                                    "step_B_time": next_step_time,
                                }
                            )

                # Convert to Pandas for easier handling
                user_pairs_df = pd.DataFrame(user_data)
                if len(user_pairs_df) == 0:
                    continue

                # Find events between step times for these users
                step_pair_key = f"{step} → {next_step}"
                between_event_counts = {}

                for _, row in user_pairs_df.iterrows():
                    user_id = row["user_id"]
                    step_a_time = row["step_A_time"]
                    step_b_time = row["step_B_time"]

                    # Get all events for this user between these times
                    user_all_events = df[df["user_id"] == user_id]
                    between_events = user_all_events[
                        (user_all_events["timestamp"] > step_a_time)
                        & (user_all_events["timestamp"] < step_b_time)
                        & ~user_all_events["event_name"].isin(funnel_steps)
                    ]

                    for event_name in between_events["event_name"]:
                        if event_name not in between_event_counts:
                            between_event_counts[event_name] = 0
                        between_event_counts[event_name] += 1

                # Sort and get top events
                if between_event_counts:
                    sorted_events = sorted(
                        between_event_counts.items(), key=lambda x: x[1], reverse=True
                    )
                    step_pairs.append(
                        {"step_pair": step_pair_key, "top_events": dict(sorted_events)}
                    )

            print("\nBetween-steps events found:")
            for pair in step_pairs:
                print(f"\n{pair['step_pair']}:")
                for event, count in pair["top_events"].items():
                    print(f"  - {event}: {count}")

    # Run the test
    test = PlFunnelTest()
    test.test_between_steps_events()


if __name__ == "__main__":
    main()
