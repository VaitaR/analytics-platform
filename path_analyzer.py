import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import networkx as nx
import pandas as pd
import polars as pl

# Import necessary classes from models.py
from models import (
    FunnelConfig,
    FunnelOrder,
    PathAnalysisData,
    ProcessMiningData,
    ReentryMode,
)


class PathAnalyzer:
    """
    Helper class for path analysis with optimized implementations for different funnel configurations.
    This class encapsulates the complex logic for path analysis to improve performance and maintainability.
    """

    def __init__(self, config: FunnelConfig):
        """Initialize with funnel configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Elite optimization: Enable global string cache for Polars operations
        try:
            pl.enable_string_cache()
            self.logger.debug("Polars string cache enabled in PathAnalyzer")
        except Exception as e:
            self.logger.warning(f"Could not enable Polars string cache in PathAnalyzer: {e}")

    def analyze(
        self,
        funnel_events_df: pl.DataFrame,
        full_history_df: pl.DataFrame,
        funnel_steps: list[str],
    ) -> PathAnalysisData:
        """
        Main entry point for path analysis. Handles preprocessing and delegates to specialized methods
        based on funnel configuration.

        Args:
            funnel_events_df: Polars DataFrame with funnel events
            full_history_df: Polars DataFrame with all user events
            funnel_steps: List of funnel step names in order

        Returns:
            PathAnalysisData object with dropoff paths and between-steps events
        """
        # Ensure we're working with eager DataFrames (not LazyFrames)
        if hasattr(funnel_events_df, "collect") and callable(funnel_events_df.collect):
            funnel_events_df = funnel_events_df.collect()
        if hasattr(full_history_df, "collect") and callable(full_history_df.collect):
            full_history_df = full_history_df.collect()

        # Preprocess DataFrames to handle nested object types
        funnel_events_df = self._preprocess_dataframe(funnel_events_df)
        full_history_df = self._preprocess_dataframe(full_history_df)

        # Initialize result containers
        dropoff_paths = {}
        between_steps_events = {}

        # Ensure we have the required columns
        try:
            funnel_events_df.select("user_id", "event_name", "timestamp")
            full_history_df.select("user_id", "event_name", "timestamp")
        except Exception as e:
            self.logger.error(f"Missing required columns in input DataFrames: {str(e)}")
            return PathAnalysisData({}, {})

        # Pre-calculate step user sets using Polars
        step_user_sets = {}
        for step in funnel_steps:
            step_users = set(
                funnel_events_df.filter(pl.col("event_name") == step)
                .select("user_id")
                .unique()
                .to_series()
                .to_list()
            )
            step_user_sets[step] = step_users

        # Process each step pair in the funnel
        for i, step in enumerate(funnel_steps[:-1]):
            next_step = funnel_steps[i + 1]

            # Find dropped users efficiently
            step_users = step_user_sets[step]
            next_step_users = step_user_sets[next_step]
            dropped_users = step_users - next_step_users

            # Analyze drop-off paths
            if dropped_users:
                next_events = self._analyze_dropoff_paths(
                    funnel_events_df, full_history_df, dropped_users, step
                )
                if next_events:
                    dropoff_paths[step] = dict(next_events.most_common(10))

            # Find users who converted from current_step to next_step
            conversion_pairs = self._find_conversion_pairs(
                funnel_events_df, step, next_step, funnel_steps
            )

            # Extract user IDs from conversion pairs
            if not conversion_pairs.is_empty():
                truly_converted_users = set(
                    conversion_pairs.select("user_id").to_series().to_list()
                )
            else:
                truly_converted_users = set()

            # Analyze between-steps events for converted users
            if truly_converted_users:
                between_events = self._analyze_between_steps_events(
                    funnel_events_df,
                    full_history_df,
                    truly_converted_users,
                    step,
                    next_step,
                    funnel_steps,
                    conversion_pairs,
                )
                step_pair = f"{step} → {next_step}"
                if between_events:  # Only add if non-empty
                    between_steps_events[step_pair] = dict(between_events.most_common(10))

        return PathAnalysisData(
            dropoff_paths=dropoff_paths, between_steps_events=between_steps_events
        )

    def _preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess DataFrame to handle nested object types and ensure proper column types"""
        try:
            # Handle complex data types by converting object columns to strings
            for col in df.columns:
                if col == "properties" or str(df[col].dtype).startswith("Object"):
                    df = df.with_columns([pl.col(col).cast(pl.Utf8)])

            # Ensure timestamp column has proper type
            if "timestamp" in df.columns:
                df = df.with_columns([pl.col("timestamp").cast(pl.Datetime)])

            # Ensure user_id is string type
            if "user_id" in df.columns:
                df = df.with_columns([pl.col("user_id").cast(pl.Utf8)])

            # Remove existing _original_order column if it exists and add a new one
            if "_original_order" in df.columns:
                df = df.drop("_original_order")
            df = df.with_row_index("_original_order")

            return df
        except Exception as e:
            self.logger.warning(f"Error preprocessing DataFrame: {str(e)}")
            return df

    def _analyze_dropoff_paths(
        self,
        funnel_events_df: pl.DataFrame,
        full_history_df: pl.DataFrame,
        dropped_users: set,
        step: str,
    ) -> Counter:
        """
        Analyze what events users do after dropping off from a funnel step.

        Args:
            funnel_events_df: DataFrame with funnel events
            full_history_df: DataFrame with all user events
            dropped_users: Set of user IDs who dropped off
            step: The step from which users dropped off

        Returns:
            Counter with next events and their counts
        """
        next_events = Counter()

        if not dropped_users:
            return next_events

        # Convert set to list for Polars filtering
        dropped_user_list = list(str(user_id) for user_id in dropped_users)

        # Use lazy evaluation for better query optimization
        lazy_segment_df = funnel_events_df.lazy()
        lazy_history_df = full_history_df.lazy()

        # Find the timestamp of the last step event for each dropped user
        last_step_events = (
            lazy_segment_df.filter(
                (pl.col("user_id").cast(pl.Utf8).is_in(dropped_user_list))
                & (pl.col("event_name") == step)
            )
            .group_by("user_id")
            .agg(pl.col("timestamp").max().alias("step_time"))
        )

        # Early exit if no step events found
        if last_step_events.collect().height == 0:
            return next_events

        # Find the next event after the step for each user within 7 days
        next_events_df = (
            last_step_events.join(
                lazy_history_df.filter(pl.col("user_id").cast(pl.Utf8).is_in(dropped_user_list)),
                on="user_id",
                how="inner",
            )
            .filter(
                (pl.col("timestamp") > pl.col("step_time"))
                & (pl.col("timestamp") <= pl.col("step_time") + pl.duration(days=7))
                & (pl.col("event_name") != step)
            )
            # Use window function to find first event after step for each user
            .with_columns([pl.col("timestamp").rank().over(["user_id"]).alias("event_rank")])
            .filter(pl.col("event_rank") == 1)
            .select(["user_id", "event_name"])
        )

        # Count next events
        event_counts = next_events_df.group_by("event_name").agg(pl.len().alias("count")).collect()

        # Convert to Counter format
        if event_counts.height > 0:
            next_events = Counter(
                dict(
                    zip(
                        event_counts["event_name"].to_list(),
                        event_counts["count"].to_list(),
                    )
                )
            )

        # Count users with no further activity
        users_with_events = next_events_df.select(pl.col("user_id").unique()).collect().height
        users_with_no_events = len(dropped_users) - users_with_events

        if users_with_no_events > 0:
            next_events["(no further activity)"] = users_with_no_events

        return next_events

    def _find_conversion_pairs(
        self,
        events_df: pl.DataFrame,
        step: str,
        next_step: str,
        funnel_steps: list[str],
    ) -> pl.DataFrame:
        """
        Find pairs of events representing conversions from one step to the next.
        This method handles all combinations of funnel order and reentry mode.

        Args:
            events_df: DataFrame with funnel events
            step: Current step name
            next_step: Next step name
            funnel_steps: List of all funnel steps in order

        Returns:
            DataFrame with conversion pairs (user_id, step_A_time, step_B_time)
        """
        # Filter events to only include relevant steps
        step_events = events_df.filter(pl.col("event_name").is_in([step, next_step]))

        # Extract step A and step B events separately
        step_A_df = step_events.filter(pl.col("event_name") == step).with_columns(
            [pl.col("timestamp").alias("step_A_time"), pl.lit(step).alias("step")]
        )

        step_B_df = step_events.filter(pl.col("event_name") == next_step).with_columns(
            [
                pl.col("timestamp").alias("step_B_time"),
                pl.lit(next_step).alias("step_name"),  # Used in some fallback methods
            ]
        )

        # Early exit if either step has no events
        if step_A_df.height == 0 or step_B_df.height == 0:
            return pl.DataFrame(
                {
                    "user_id": [],
                    "step": [],
                    "next_step": [],
                    "step_A_time": [],
                    "step_B_time": [],
                }
            )

        conversion_window = pl.duration(hours=self.config.conversion_window_hours)

        # Choose strategy based on funnel configuration
        if self.config.funnel_order == FunnelOrder.ORDERED:
            if self.config.reentry_mode == ReentryMode.FIRST_ONLY:
                # Get first step A for each user
                first_A = step_A_df.group_by("user_id").agg(
                    [pl.col("step_A_time").min(), pl.col("step").first()]
                )

                # For each user, find first B after A within conversion window
                conversion_pairs = (
                    first_A.join(step_B_df, on="user_id", how="inner")
                    .filter(
                        (pl.col("step_B_time") > pl.col("step_A_time"))
                        & (pl.col("step_B_time") <= pl.col("step_A_time") + conversion_window)
                    )
                    # Use window function to find earliest B for each user
                    .with_columns([pl.col("step_B_time").rank().over(["user_id"]).alias("rank")])
                    .filter(pl.col("rank") == 1)
                    .select(["user_id", "step", "step_name", "step_A_time", "step_B_time"])
                    .rename({"step_name": "next_step"})
                )

            elif self.config.reentry_mode == ReentryMode.OPTIMIZED_REENTRY:
                # Use join_asof for optimal performance with ORDERED + OPTIMIZED_REENTRY
                try:
                    # Sort both DataFrames by timestamp
                    step_A_df = step_A_df.sort(["user_id", "step_A_time"])
                    step_B_df = step_B_df.sort(["user_id", "step_B_time"])

                    # Use join_asof to find the next B event after each A event within window
                    # This avoids the "join explosion" problem
                    conversion_pairs = pl.join_asof(
                        step_A_df.select(["user_id", "step", "step_A_time"]),
                        step_B_df.select(["user_id", "step_name", "step_B_time"]).rename(
                            {"step_name": "next_step"}
                        ),
                        left_on="step_A_time",
                        right_on="step_B_time",
                        by="user_id",
                        strategy="forward",
                    ).filter(
                        # Keep only pairs within conversion window
                        (pl.col("step_B_time") > pl.col("step_A_time"))
                        & (pl.col("step_B_time") <= pl.col("step_A_time") + conversion_window)
                    )

                except Exception as e:
                    self.logger.warning(
                        f"join_asof failed: {str(e)}, falling back to optimal_step_pairs"
                    )
                    # Fall back to the optimal step pairs method
                    conversion_pairs = self._find_optimal_step_pairs(step_A_df, step_B_df)
        else:  # UNORDERED funnel
            if self.config.reentry_mode == ReentryMode.FIRST_ONLY:
                # For unordered funnels, we just need the first occurrence of each step
                first_A = step_A_df.group_by("user_id").agg(
                    [pl.col("step_A_time").min(), pl.col("step").first()]
                )

                first_B = step_B_df.group_by("user_id").agg(
                    [
                        pl.col("step_B_time").min(),
                        pl.col("step_name").first().alias("next_step"),
                    ]
                )

                # Join and filter by conversion window
                conversion_pairs = (
                    first_A.join(first_B, on="user_id", how="inner")
                    .with_columns(
                        [
                            # Calculate absolute time difference
                            (
                                (pl.col("step_B_time") - pl.col("step_A_time"))
                                .dt.total_hours()
                                .abs()
                            ).alias("time_diff_hours")
                        ]
                    )
                    .filter(pl.col("time_diff_hours") <= self.config.conversion_window_hours)
                    .drop("time_diff_hours")
                )

            else:  # UNORDERED + OPTIMIZED_REENTRY
                # For unordered with reentry, we need to find all pairs within window
                # and then group by user to find the earliest valid pair
                joined = (
                    step_A_df.select(["user_id", "step", "step_A_time"])
                    .join(
                        step_B_df.select(["user_id", "step_name", "step_B_time"]).rename(
                            {"step_name": "next_step"}
                        ),
                        on="user_id",
                        how="inner",
                    )
                    .with_columns(
                        [
                            # Calculate absolute time difference
                            (
                                (pl.col("step_B_time") - pl.col("step_A_time"))
                                .dt.total_hours()
                                .abs()
                            ).alias("time_diff_hours")
                        ]
                    )
                    .filter(pl.col("time_diff_hours") <= self.config.conversion_window_hours)
                    .drop("time_diff_hours")
                )

                # Find the earliest pair for each user (by earliest combined timestamp)
                conversion_pairs = (
                    joined.with_columns(
                        [
                            # Use the minimum of the two timestamps to determine the earliest pair
                            pl.when(pl.col("step_A_time") <= pl.col("step_B_time"))
                            .then(pl.col("step_A_time"))
                            .otherwise(pl.col("step_B_time"))
                            .alias("earliest_time")
                        ]
                    )
                    .sort(["user_id", "earliest_time"])
                    .group_by("user_id")
                    .agg(
                        [
                            pl.col("step").first(),
                            pl.col("next_step").first(),
                            pl.col("step_A_time").first(),
                            pl.col("step_B_time").first(),
                        ]
                    )
                )

        return conversion_pairs

    def _find_optimal_step_pairs(
        self, step_A_df: pl.DataFrame, step_B_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Helper function to find optimal step pairs when join_asof fails"""
        conversion_window = pl.duration(hours=self.config.conversion_window_hours)

        # Handle empty dataframes
        if step_A_df.height == 0 or step_B_df.height == 0:
            return pl.DataFrame(
                {
                    "user_id": [],
                    "step": [],
                    "next_step": [],
                    "step_A_time": [],
                    "step_B_time": [],
                }
            )

        try:
            # Ensure we have step_A_time column
            if "step_A_time" not in step_A_df.columns and "timestamp" in step_A_df.columns:
                step_A_df = step_A_df.with_columns(pl.col("timestamp").alias("step_A_time"))

            # Get step names for labels
            step_name = "Step A"
            next_step_name = "Step B"

            if "step" in step_A_df.columns and step_A_df.height > 0:
                step_name_col = step_A_df.select("step").unique()
                if step_name_col.height > 0:
                    step_name = step_name_col[0, 0]

            if "step_name" in step_B_df.columns and step_B_df.height > 0:
                next_step_name_col = step_B_df.select("step_name").unique()
                if next_step_name_col.height > 0:
                    next_step_name = next_step_name_col[0, 0]

            # Use a fully vectorized approach using only Polars expressions
            # First, create a cross join of users with their A and B times
            user_with_A_times = step_A_df.select(["user_id", "step_A_time"])

            # Ensure B times are properly named
            if "step_B_time" in step_B_df.columns:
                user_with_B_times = step_B_df.select(["user_id", "step_B_time"])
            else:
                user_with_B_times = step_B_df.select(["user_id", "timestamp"]).rename(
                    {"timestamp": "step_B_time"}
                )

            # Join both tables and filter for valid conversion pairs
            valid_conversions = (
                user_with_A_times.join(user_with_B_times, on="user_id", how="inner")
                # Use only native Polars expressions for the filter condition
                .filter(
                    (pl.col("step_B_time") > pl.col("step_A_time"))
                    & (pl.col("step_B_time") <= pl.col("step_A_time") + conversion_window)
                )
                # For each step_A_time, find the earliest valid step_B_time
                .sort(["user_id", "step_A_time", "step_B_time"])
                # Keep the first valid B time for each A time
                .group_by(["user_id", "step_A_time"])
                .agg(pl.col("step_B_time").first().alias("earliest_B_time"))
                # Keep only the first A->B pair for each user
                .sort(["user_id", "step_A_time"])
                .group_by("user_id")
                .agg(
                    [
                        pl.col("step_A_time").first(),
                        pl.col("earliest_B_time").first().alias("step_B_time"),
                    ]
                )
                # Add step names as literals
                .with_columns(
                    [
                        pl.lit(step_name).alias("step"),
                        pl.lit(next_step_name).alias("next_step"),
                    ]
                )
                # Select columns in the right order
                .select(["user_id", "step", "next_step", "step_A_time", "step_B_time"])
            )

            return valid_conversions

        except Exception as e:
            self.logger.error(f"Fully vectorized approach for finding step pairs failed: {e}")

            # Final fallback with empty DataFrame with correct structure
            return pl.DataFrame(
                {
                    "user_id": [],
                    "step": [],
                    "next_step": [],
                    "step_A_time": [],
                    "step_B_time": [],
                }
            )

    def _analyze_between_steps_events(
        self,
        funnel_events_df: pl.DataFrame,
        full_history_df: pl.DataFrame,
        converted_users: set,
        step: str,
        next_step: str,
        funnel_steps: list[str],
        conversion_pairs: pl.DataFrame,
    ) -> Counter:
        """
        Analyze events occurring between two funnel steps for converted users.

        Args:
            funnel_events_df: DataFrame with funnel events
            full_history_df: DataFrame with all user events
            converted_users: Set of user IDs who converted
            step: Current step name
            next_step: Next step name
            funnel_steps: List of all funnel steps
            conversion_pairs: DataFrame with conversion pairs from _find_conversion_pairs

        Returns:
            Counter with between-steps events and their counts
        """
        between_events = Counter()

        if not converted_users:
            return between_events

        # Convert set to list for Polars filtering
        converted_user_list = list(str(user_id) for user_id in converted_users)

        # Use lazy evaluation for better query optimization
        lazy_history_df = full_history_df.lazy()

        # Get the conversion pairs for these users
        # We already have this from _find_conversion_pairs
        if conversion_pairs.is_empty():
            return between_events

        # For each user, find events between their step_A_time and step_B_time
        between_steps_df = (
            conversion_pairs.lazy()
            .join(
                lazy_history_df.filter(pl.col("user_id").cast(pl.Utf8).is_in(converted_user_list)),
                on="user_id",
                how="inner",
            )
            .filter(
                # Events must be between step A and step B
                (pl.col("timestamp") > pl.col("step_A_time"))
                & (pl.col("timestamp") < pl.col("step_B_time"))
                &
                # Exclude the funnel steps themselves
                ~pl.col("event_name").is_in(funnel_steps)
            )
            .select(["user_id", "event_name"])
        )

        # Count events
        event_counts = (
            between_steps_df.group_by("event_name").agg(pl.len().alias("count")).collect()
        )

        # Convert to Counter format
        if event_counts.height > 0:
            between_events = Counter(
                dict(
                    zip(
                        event_counts["event_name"].to_list(),
                        event_counts["count"].to_list(),
                    )
                )
            )

        # Add special entry for users with no intermediate events
        users_with_events = between_steps_df.select(pl.col("user_id").unique()).collect().height
        users_with_no_events = len(converted_users) - users_with_events

        if users_with_no_events > 0:
            between_events["(direct conversion)"] = users_with_no_events

        return between_events

    def discover_process_mining_structure(
        self,
        events_df: Union[pd.DataFrame, pl.DataFrame],
        min_frequency: int = 10,
        include_cycles: bool = True,
        time_window_hours: Optional[int] = None,
        filter_events: Optional[list[str]] = None,
    ) -> ProcessMiningData:
        """
        Automatic process discovery from user events using advanced algorithms

        Args:
            events_df: DataFrame with events (user_id, event_name, timestamp)
            min_frequency: Minimum frequency to include transition
            include_cycles: Whether to detect cycles and loops
            time_window_hours: Optional time window for process analysis
            filter_events: Optional list of event names to filter analysis to (e.g., funnel events only)

        Returns:
            ProcessMiningData with complete process structure
        """
        # Convert to Polars for efficient processing with data type handling
        if isinstance(events_df, pd.DataFrame):
            # Clean data before conversion to avoid type conflicts
            events_clean = events_df.copy()

            # Ensure user_id is string
            events_clean["user_id"] = events_clean["user_id"].astype(str)

            # Filter out rows with None event_name
            events_clean = events_clean[events_clean["event_name"].notna()]
            events_clean["event_name"] = events_clean["event_name"].astype(str)

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(events_clean["timestamp"]):
                events_clean["timestamp"] = pd.to_datetime(events_clean["timestamp"])

            events_pl = pl.from_pandas(events_clean)
        else:
            events_pl = events_df

        # Filter by time window if specified
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            events_pl = events_pl.filter(pl.col("timestamp") >= cutoff_time)

        # Filter by specific events if specified (e.g., funnel events only)
        if filter_events:
            events_pl = events_pl.filter(pl.col("event_name").is_in(filter_events))

        # Build user journeys (optimized) - avoid dictionary conversion when possible
        journey_df = self._build_user_journeys_optimized(events_pl)

        # Discover activities and their characteristics (optimized)
        activities = self._discover_activities(events_pl, None)  # Pass None to use optimized path

        # Discover transitions between activities (optimized)
        transitions = self._discover_transitions_optimized(journey_df, min_frequency)

        # Identify process variants (optimized)
        variants = self._identify_process_variants_optimized(journey_df)

        # Find start and end activities (optimized)
        start_activities, end_activities = self._identify_start_end_activities_optimized(
            journey_df
        )

        # Detect cycles and loops if requested (use optimized method first)
        cycles = []
        if include_cycles:
            try:
                # Try optimized Polars-based cycle detection first
                cycles = self._detect_cycles_optimized(journey_df, transitions)
            except Exception as e:
                self.logger.warning(
                    f"Optimized cycle detection failed: {str(e)}, falling back to legacy method"
                )
                # Fallback to legacy method only if optimized fails
                user_journeys = self._build_user_journeys(events_pl)
                cycles = self._detect_cycles(user_journeys, transitions)

        # Calculate process statistics (optimized to work with Polars DataFrame)
        statistics = self._calculate_process_statistics_optimized(
            journey_df, activities, transitions
        )

        # Generate automatic insights
        insights = self._generate_process_insights(
            activities, transitions, cycles, variants, statistics
        )

        return ProcessMiningData(
            activities=activities,
            transitions=transitions,
            cycles=cycles,
            variants=variants,
            start_activities=start_activities,
            end_activities=end_activities,
            statistics=statistics,
            insights=insights,
        )

    def _build_user_journeys_optimized(self, events_pl: pl.DataFrame) -> pl.DataFrame:
        """Build user journeys using pure Polars for maximum performance"""
        # Sort events by user and timestamp
        sorted_events = events_pl.sort(["user_id", "timestamp"])

        # Add sequence numbers and calculate durations using window functions
        journey_df = sorted_events.with_columns(
            [
                # Add row number within each user group
                pl.int_range(pl.len()).over("user_id").alias("event_order"),
                # Calculate duration to next event (in hours)
                (pl.col("timestamp").shift(-1).over("user_id") - pl.col("timestamp"))
                .dt.total_seconds()
                .truediv(3600)
                .alias("duration_to_next"),
                # Mark start and end events
                (pl.int_range(pl.len()).over("user_id") == 0).alias("is_start"),
                (pl.int_range(pl.len()).over("user_id") == (pl.len().over("user_id") - 1)).alias(
                    "is_end"
                ),
            ]
        )

        return journey_df

    def _build_user_journeys(self, events_pl: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
        """Build user journeys from events - optimized version"""
        # Use optimized Polars implementation
        journey_df = self._build_user_journeys_optimized(events_pl)

        # Convert to dictionary format only when needed for legacy methods
        journeys = {}

        # Group by user_id and iterate
        for user_id, user_df in journey_df.group_by("user_id"):
            user_id = user_id[0] if isinstance(user_id, tuple) else user_id  # Handle group key

            journey = []
            for row in user_df.iter_rows(named=True):
                journey.append(
                    {
                        "event": row["event_name"],
                        "timestamp": row["timestamp"],
                        "order": row["event_order"],
                        "duration_to_next": row["duration_to_next"],
                        "is_start": row["is_start"],
                        "is_end": row["is_end"],
                    }
                )

            journeys[str(user_id)] = journey

        return journeys

    def _discover_activities(
        self, events_pl: pl.DataFrame, user_journeys: Optional[dict[str, list[dict[str, Any]]]] = None
    ) -> dict[str, dict[str, Any]]:
        """Discover activities and their characteristics - optimized version"""
        # Get optimized journey DataFrame
        journey_df = self._build_user_journeys_optimized(events_pl)

        # Calculate activity statistics using pure Polars
        activity_stats = journey_df.group_by("event_name").agg(
            [
                pl.len().alias("frequency"),
                pl.col("user_id").n_unique().alias("unique_users"),
                pl.col("timestamp").min().alias("first_occurrence"),
                pl.col("timestamp").max().alias("last_occurrence"),
                pl.col("duration_to_next")
                .filter(pl.col("duration_to_next").is_not_null())
                .mean()
                .alias("avg_duration"),
                pl.col("is_start").sum().alias("start_count"),
                pl.col("is_end").sum().alias("end_count"),
            ]
        )

        activities = {}
        for row in activity_stats.iter_rows(named=True):
            activity_name = row["event_name"]

            # Classify activity type
            activity_type = self._classify_activity_type(
                activity_name, row["start_count"], row["end_count"], row["frequency"]
            )

            # Calculate success rate (simplified for performance)
            success_rate = self._calculate_activity_success_rate_optimized(
                activity_name, journey_df
            )

            activities[activity_name] = {
                "frequency": row["frequency"],
                "unique_users": row["unique_users"],
                "avg_duration": row["avg_duration"] or 0,
                "is_start": row["start_count"] > 0,
                "is_end": row["end_count"] > 0,
                "activity_type": activity_type,
                "success_rate": success_rate,
                "first_occurrence": row["first_occurrence"],
                "last_occurrence": row["last_occurrence"],
            }

        return activities

    def _discover_transitions(
        self, user_journeys: dict[str, list[dict[str, Any]]], min_frequency: int
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Discover transitions between activities"""
        transition_counts = defaultdict(int)
        transition_users = defaultdict(set)
        transition_durations = defaultdict(list)

        # Count transitions across all user journeys
        for user_id, journey in user_journeys.items():
            for i in range(len(journey) - 1):
                from_event = journey[i]["event"]
                to_event = journey[i + 1]["event"]
                transition = (from_event, to_event)

                transition_counts[transition] += 1
                transition_users[transition].add(user_id)

                if journey[i]["duration_to_next"]:
                    transition_durations[transition].append(journey[i]["duration_to_next"])

        # Filter by minimum frequency and build transition data
        transitions = {}
        total_transitions = sum(transition_counts.values())

        for transition, frequency in transition_counts.items():
            if frequency >= min_frequency:
                from_event, to_event = transition
                durations = transition_durations[transition]

                transitions[transition] = {
                    "frequency": frequency,
                    "unique_users": len(transition_users[transition]),
                    "avg_duration": sum(durations) / len(durations) if durations else 0,
                    "probability": (frequency / total_transitions) * 100,
                    "transition_type": self._classify_transition_type(
                        from_event, to_event, frequency
                    ),
                }

        return transitions

    def _detect_cycles(
        self,
        user_journeys: dict[str, list[dict[str, Any]]],
        transitions: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Detect cycles and loops in user behavior using graph analysis"""
        cycles = []

        # Build directed graph from transitions
        G = nx.DiGraph()
        for (from_event, to_event), data in transitions.items():
            G.add_edge(from_event, to_event, weight=data["frequency"])

        # Find simple cycles
        try:
            simple_cycles = list(nx.simple_cycles(G))

            for cycle_path in simple_cycles:
                if len(cycle_path) <= 5:  # Focus on short cycles
                    # Calculate cycle statistics
                    cycle_frequency = self._calculate_cycle_frequency(cycle_path, user_journeys)
                    cycle_impact = self._assess_cycle_impact(cycle_path, user_journeys)

                    cycles.append(
                        {
                            "path": cycle_path,
                            "frequency": cycle_frequency,
                            "type": "loop" if len(cycle_path) == 1 else "cycle",
                            "impact": cycle_impact,
                            "avg_cycle_time": self._calculate_avg_cycle_time(
                                cycle_path, user_journeys
                            ),
                        }
                    )

        except nx.NetworkXError:
            # Handle cases where cycle detection fails
            pass

        # Sort by frequency and return top cycles
        cycles.sort(key=lambda x: x["frequency"], reverse=True)
        return cycles[:10]  # Return top 10 cycles

    def _detect_cycles_optimized(
        self,
        journey_df: pl.DataFrame,
        transitions: dict[tuple[str, str], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Optimized cycle detection using Polars operations instead of NetworkX.
        Focuses on finding the most common and impactful cycles efficiently.
        """
        cycles = []

        try:
            # Create transitions DataFrame for easier manipulation
            if not transitions:
                return cycles

            transition_data = []
            for (from_event, to_event), data in transitions.items():
                transition_data.append(
                    {
                        "from_event": from_event,
                        "to_event": to_event,
                        "frequency": data["frequency"],
                    }
                )

            transitions_df = pl.DataFrame(transition_data)

            # 1. Find self-loops (most common cycles)
            self_loops = transitions_df.filter(pl.col("from_event") == pl.col("to_event")).sort(
                "frequency", descending=True
            )

            for row in self_loops.iter_rows(named=True):
                event = row["from_event"]
                frequency = row["frequency"]

                # Calculate impact based on frequency
                impact = (
                    "negative"
                    if any(keyword in event.lower() for keyword in ["error", "fail", "timeout"])
                    else "positive"
                )

                cycles.append(
                    {
                        "path": [event],
                        "frequency": frequency,
                        "type": "loop",
                        "impact": impact,
                        "avg_cycle_time": self._estimate_avg_cycle_time_optimized(
                            journey_df, [event]
                        ),
                    }
                )

            # 2. Find 2-step cycles (A -> B -> A) using efficient joins
            two_step_cycles = (
                transitions_df.join(
                    transitions_df,
                    left_on="to_event",
                    right_on="from_event",
                    how="inner",
                    suffix="_right",
                )
                .filter(pl.col("from_event") == pl.col("to_event_right"))  # Forms a cycle
                .filter(pl.col("from_event") != pl.col("to_event"))  # Not self-loops
                .with_columns(
                    [pl.min_horizontal(["frequency", "frequency_right"]).alias("cycle_frequency")]
                )
                .select(
                    [
                        "from_event",
                        pl.col("to_event").alias("middle_event"),
                        "cycle_frequency",
                    ]
                )
                .sort("cycle_frequency", descending=True)
                .limit(5)  # Top 5 two-step cycles
            )

            for row in two_step_cycles.iter_rows(named=True):
                path = [row["from_event"], row["middle_event"]]
                frequency = row["cycle_frequency"]

                # Assess impact
                impact = (
                    "negative"
                    if any("error" in event.lower() or "fail" in event.lower() for event in path)
                    else "positive"
                )

                cycles.append(
                    {
                        "path": path,
                        "frequency": frequency,
                        "type": "cycle",
                        "impact": impact,
                        "avg_cycle_time": self._estimate_avg_cycle_time_optimized(
                            journey_df, path
                        ),
                    }
                )

            # 3. Find 3-step cycles using path extension (limited for performance)
            if len(transitions_df) < 50:  # Only for smaller datasets to avoid complexity explosion
                three_step_cycles = self._find_three_step_cycles_optimized(transitions_df)
                cycles.extend(three_step_cycles[:3])  # Top 3 three-step cycles

        except Exception as e:
            self.logger.warning(
                f"Optimized cycle detection failed: {str(e)}, falling back to simplified detection"
            )
            # Simplified fallback: just find self-loops from transitions
            for (from_event, to_event), data in transitions.items():
                if from_event == to_event:
                    cycles.append(
                        {
                            "path": [from_event],
                            "frequency": data["frequency"],
                            "type": "loop",
                            "impact": (
                                "negative" if "error" in from_event.lower() else "positive"
                            ),
                            "avg_cycle_time": 0,
                        }
                    )

        # Sort by frequency and return top cycles
        cycles.sort(key=lambda x: x["frequency"], reverse=True)
        return cycles[:10]  # Return top 10 cycles

    def _find_three_step_cycles_optimized(
        self, transitions_df: pl.DataFrame
    ) -> list[dict[str, Any]]:
        """Find 3-step cycles (A -> B -> C -> A) efficiently"""
        cycles = []

        try:
            # Join transitions to find 3-step paths
            three_step_paths = (
                transitions_df.join(
                    transitions_df,
                    left_on="to_event",
                    right_on="from_event",
                    how="inner",
                    suffix="_2",
                )
                .join(
                    transitions_df,
                    left_on="to_event_2",
                    right_on="from_event",
                    how="inner",
                    suffix="_3",
                )
                .filter(pl.col("from_event") == pl.col("to_event_3"))  # Forms a cycle
                .filter(
                    (pl.col("from_event") != pl.col("to_event"))
                    & (pl.col("to_event") != pl.col("to_event_2"))
                    & (pl.col("to_event_2") != pl.col("from_event"))
                )  # Ensure all different steps
                .with_columns(
                    [
                        pl.min_horizontal(["frequency", "frequency_2", "frequency_3"]).alias(
                            "cycle_frequency"
                        )
                    ]
                )
                .select(
                    [
                        "from_event",
                        pl.col("to_event").alias("step2"),
                        pl.col("to_event_2").alias("step3"),
                        "cycle_frequency",
                    ]
                )
                .sort("cycle_frequency", descending=True)
                .limit(3)  # Top 3 only
            )

            for row in three_step_paths.iter_rows(named=True):
                path = [row["from_event"], row["step2"], row["step3"]]
                frequency = row["cycle_frequency"]

                # Assess impact
                impact = (
                    "negative"
                    if any("error" in event.lower() or "fail" in event.lower() for event in path)
                    else "positive"
                )

                cycles.append(
                    {
                        "path": path,
                        "frequency": frequency,
                        "type": "cycle",
                        "impact": impact,
                        "avg_cycle_time": 0,  # Simplified for performance
                    }
                )

        except Exception as e:
            self.logger.warning(f"3-step cycle detection failed: {str(e)}")

        return cycles

    def _estimate_avg_cycle_time_optimized(
        self, journey_df: pl.DataFrame, cycle_path: list[str]
    ) -> float:
        """Estimate average cycle time using Polars operations"""
        try:
            if len(cycle_path) == 1:
                # Self-loop: time between consecutive occurrences of same event
                event = cycle_path[0]

                # Find consecutive occurrences of the same event for each user
                consecutive_times = (
                    journey_df.filter(pl.col("event_name") == event)
                    .sort(["user_id", "timestamp"])
                    .with_columns(
                        [pl.col("timestamp").shift(-1).over("user_id").alias("next_timestamp")]
                    )
                    .filter(pl.col("next_timestamp").is_not_null())
                    .with_columns(
                        [
                            (pl.col("next_timestamp") - pl.col("timestamp"))
                            .dt.total_hours()
                            .alias("cycle_time_hours")
                        ]
                    )
                    .filter(pl.col("cycle_time_hours") > 0)  # Positive time differences only
                )

                if consecutive_times.height > 0:
                    return consecutive_times.select(pl.col("cycle_time_hours").mean()).item() or 0

            else:
                # Multi-step cycle: simplified estimation
                # Average time between first and last event in cycle path
                first_event = cycle_path[0]
                last_event = cycle_path[-1]

                cycle_times = (
                    journey_df.filter(pl.col("event_name").is_in([first_event, last_event]))
                    .sort(["user_id", "timestamp"])
                    .group_by("user_id")
                    .agg(
                        [
                            pl.col("timestamp").min().alias("start_time"),
                            pl.col("timestamp").max().alias("end_time"),
                        ]
                    )
                    .filter(pl.col("start_time") != pl.col("end_time"))
                    .with_columns(
                        [
                            (pl.col("end_time") - pl.col("start_time"))
                            .dt.total_hours()
                            .alias("cycle_time_hours")
                        ]
                    )
                )

                if cycle_times.height > 0:
                    return cycle_times.select(pl.col("cycle_time_hours").mean()).item() or 0

        except Exception as e:
            self.logger.warning(f"Cycle time estimation failed: {str(e)}")

        return 0.0

    # Helper methods
    def _classify_activity_type(
        self, activity_name: str, start_count: int, end_count: int, frequency: int
    ) -> str:
        """Classify activity type based on patterns"""
        name_lower = activity_name.lower()

        if any(word in name_lower for word in ["login", "signup", "register", "start"]):
            return "entry"
        if any(word in name_lower for word in ["purchase", "checkout", "complete", "finish"]):
            return "conversion"
        if any(word in name_lower for word in ["error", "fail", "timeout"]):
            return "error"
        if any(word in name_lower for word in ["view", "page", "screen"]):
            return "navigation"
        if start_count > 0:
            return "entry"
        if end_count > 0:
            return "exit"
        return "process"

    def _classify_transition_type(self, from_event: str, to_event: str, frequency: int) -> str:
        """Classify transition type"""
        if from_event == to_event:
            return "loop"
        if "error" in to_event.lower():
            return "error_transition"
        if frequency > 100:
            return "main_flow"
        return "alternative_flow"

    def _calculate_activity_success_rate(
        self, activity_name: str, user_journeys: dict[str, list[dict[str, Any]]]
    ) -> float:
        """Calculate success rate for an activity"""
        success_count = 0
        total_count = 0

        for journey in user_journeys.values():
            for i, step in enumerate(journey):
                if step["event"] == activity_name:
                    total_count += 1
                    # Consider success if user continues to next steps
                    if i < len(journey) - 1:
                        success_count += 1

        return (success_count / total_count) * 100 if total_count > 0 else 0

    def _calculate_activity_success_rate_optimized(
        self, activity_name: str, journey_df: pl.DataFrame
    ) -> float:
        """Calculate activity success rate using optimized Polars operations"""
        # Define success events
        success_events = [
            "purchase",
            "conversion",
            "complete",
            "finish",
            "success",
            "checkout",
        ]

        # Count users who had this activity and later had a success event
        users_with_activity = (
            journey_df.filter(pl.col("event_name") == activity_name).select("user_id").unique()
        )

        if users_with_activity.height == 0:
            return 0.0

        # Count how many of these users eventually had a success event
        users_with_success = (
            journey_df.filter(
                pl.col("user_id").is_in(users_with_activity.get_column("user_id"))
                & pl.col("event_name").str.to_lowercase().str.contains_any(success_events)
            )
            .select("user_id")
            .unique()
            .height
        )

        return (users_with_success / users_with_activity.height) * 100

    def _discover_transitions_optimized(
        self, journey_df: pl.DataFrame, min_frequency: int
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Discover transitions using optimized Polars operations"""
        # Create transitions by joining events with their next event
        transitions_df = journey_df.with_columns(
            [
                # Get next event for each user
                pl.col("event_name").shift(-1).over("user_id").alias("next_event"),
                pl.col("timestamp").shift(-1).over("user_id").alias("next_timestamp"),
            ]
        ).filter(pl.col("next_event").is_not_null())  # Remove last events that have no next

        # Calculate transition statistics
        transition_stats = (
            transitions_df.group_by(["event_name", "next_event"])
            .agg(
                [
                    pl.len().alias("frequency"),
                    pl.col("user_id").n_unique().alias("unique_users"),
                    pl.col("duration_to_next")
                    .filter(pl.col("duration_to_next").is_not_null())
                    .mean()
                    .alias("avg_duration"),
                ]
            )
            .filter(pl.col("frequency") >= min_frequency)
        )

        # Calculate total transitions for probability calculation
        total_transitions = transition_stats.get_column("frequency").sum()

        transitions = {}
        for row in transition_stats.iter_rows(named=True):
            from_event = row["event_name"]
            to_event = row["next_event"]
            transition = (from_event, to_event)

            transitions[transition] = {
                "frequency": row["frequency"],
                "unique_users": row["unique_users"],
                "avg_duration": row["avg_duration"] or 0,
                "probability": (
                    (row["frequency"] / total_transitions) * 100 if total_transitions > 0 else 0
                ),
                "transition_type": self._classify_transition_type(
                    from_event, to_event, row["frequency"]
                ),
            }

        return transitions

    def _calculate_cycle_frequency(
        self, cycle_path: list[str], user_journeys: dict[str, list[dict[str, Any]]]
    ) -> int:
        """Calculate how often a cycle occurs"""
        frequency = 0

        for journey in user_journeys.values():
            events = [step["event"] for step in journey]
            # Look for the cycle pattern in the journey
            for i in range(len(events) - len(cycle_path) + 1):
                if events[i : i + len(cycle_path)] == cycle_path:
                    frequency += 1

        return frequency

    def _assess_cycle_impact(
        self, cycle_path: list[str], user_journeys: dict[str, list[dict[str, Any]]]
    ) -> str:
        """Assess whether a cycle has positive or negative impact"""
        # Simple heuristic: cycles involving error events are negative
        if any("error" in event.lower() or "fail" in event.lower() for event in cycle_path) or any(
            "retry" in event.lower() or "repeat" in event.lower() for event in cycle_path
        ):
            return "negative"
        return "positive"

    def _calculate_avg_cycle_time(
        self, cycle_path: list[str], user_journeys: dict[str, list[dict[str, Any]]]
    ) -> float:
        """Calculate average time to complete a cycle"""
        cycle_times = []

        for journey in user_journeys.values():
            events = [(step["event"], step.get("duration_to_next", 0)) for step in journey]

            for i in range(len(events) - len(cycle_path) + 1):
                if [event for event, _ in events[i : i + len(cycle_path)]] == cycle_path:
                    cycle_time = sum(
                        duration for _, duration in events[i : i + len(cycle_path) - 1]
                    )
                    cycle_times.append(cycle_time)

        return sum(cycle_times) / len(cycle_times) if cycle_times else 0

    def _calculate_path_success(self, journey: list[dict[str, Any]]) -> bool:
        """Calculate if a path represents a successful journey"""
        if not journey:
            return False

        last_event = journey[-1]["event"].lower()
        success_keywords = [
            "purchase",
            "complete",
            "success",
            "finish",
            "convert",
            "checkout",
        ]

        return any(keyword in last_event for keyword in success_keywords)

    def _classify_variant_type(self, path: tuple[str, ...], success_rate: float) -> str:
        """Classify variant type based on success rate and characteristics"""
        if success_rate > 80:
            return "high_success"
        if success_rate > 50:
            return "medium_success"
        if success_rate > 20:
            return "low_success"
        return "problematic"

    def _identify_process_variants_optimized(
        self, journey_df: pl.DataFrame
    ) -> list[dict[str, Any]]:
        """Identify process variants using optimized Polars operations"""
        try:
            # Create path strings for each user journey
            user_paths = (
                journey_df.sort(["user_id", "event_order"])
                .group_by("user_id")
                .agg(
                    [
                        pl.col("event_name").str.concat(" → ").alias("path_string"),
                        pl.col("duration_to_next")
                        .filter(pl.col("duration_to_next").is_not_null())
                        .sum()
                        .alias("total_duration"),
                    ]
                )
            )

            # Group by path to find variants
            path_stats = user_paths.group_by("path_string").agg(
                [
                    pl.len().alias("frequency"),
                    pl.col("total_duration").mean().alias("avg_duration"),
                ]
            )

            # Calculate success rates (simplified)
            success_events = [
                "purchase",
                "conversion",
                "complete",
                "finish",
                "success",
                "checkout",
            ]
            path_success = (
                user_paths.with_columns(
                    [
                        pl.col("path_string")
                        .str.to_lowercase()
                        .str.contains_any(success_events)
                        .alias("has_success")
                    ]
                )
                .group_by("path_string")
                .agg([pl.col("has_success").mean().mul(100).alias("success_rate")])
            )

            # Join stats with success rates
            variant_stats = (
                path_stats.join(path_success, on="path_string", how="left")
                .filter(
                    pl.col("frequency") >= max(1, user_paths.height // 10)
                )  # Dynamic threshold
                .sort("frequency", descending=True)
            )

            variants = []
            for row in variant_stats.iter_rows(named=True):
                path_steps = row["path_string"].split(" → ")
                variants.append(
                    {
                        "path": path_steps,
                        "frequency": row["frequency"],
                        "success_rate": row["success_rate"] or 0,
                        "avg_duration": row["avg_duration"] or 0,
                        "variant_type": self._classify_variant_type(
                            tuple(path_steps), row["success_rate"] or 0
                        ),
                    }
                )

            return variants[:20]  # Return top 20 variants

        except Exception as e:
            self.logger.warning(
                f"Optimized variant discovery failed: {str(e)}, returning empty list"
            )
            return []

    def _identify_start_end_activities_optimized(
        self, journey_df: pl.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Identify start and end activities using optimized Polars operations"""
        try:
            # Count start activities (first event in each user journey)
            start_counts = (
                journey_df.filter(pl.col("is_start"))
                .group_by("event_name")
                .agg(pl.len().alias("count"))
            )

            # Count end activities (last event in each user journey)
            end_counts = (
                journey_df.filter(pl.col("is_end"))
                .group_by("event_name")
                .agg(pl.len().alias("count"))
            )

            # Get total number of journeys for threshold calculation
            total_journeys = journey_df.select("user_id").n_unique()
            threshold = max(1, total_journeys * 0.05)  # 5% threshold

            # Extract activities above threshold
            start_activities = [
                row["event_name"]
                for row in start_counts.iter_rows(named=True)
                if row["count"] >= threshold
            ]

            end_activities = [
                row["event_name"]
                for row in end_counts.iter_rows(named=True)
                if row["count"] >= threshold
            ]

            return start_activities, end_activities

        except Exception as e:
            self.logger.warning(
                f"Optimized start/end activity discovery failed: {str(e)}, returning empty lists"
            )
            return [], []

    def _calculate_process_statistics_optimized(
        self,
        journey_df: pl.DataFrame,
        activities: dict[str, dict[str, Any]],
        transitions: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate overall process statistics using optimized Polars operations"""
        try:
            # Get basic counts
            total_cases = journey_df.select("user_id").n_unique()

            # Calculate average journey duration using Polars
            journey_durations = (
                journey_df.group_by("user_id")
                .agg(
                    pl.col("duration_to_next")
                    .filter(pl.col("duration_to_next").is_not_null())
                    .sum()
                    .alias("total_duration")
                )
                .select("total_duration")
                .filter(pl.col("total_duration").is_not_null())
            )

            avg_duration = journey_durations.select(pl.col("total_duration").mean()).item() or 0

            # Calculate completion rate (journeys that end with success events)
            success_events = [
                "purchase",
                "conversion",
                "complete",
                "finish",
                "success",
                "checkout",
            ]

            completed_journeys = (
                journey_df.filter(pl.col("is_end"))  # Only look at end events
                .filter(pl.col("event_name").str.to_lowercase().str.contains_any(success_events))
                .select("user_id")
                .n_unique()
            )

            completion_rate = (completed_journeys / total_cases) * 100 if total_cases > 0 else 0

            # Count unique paths
            unique_paths = (
                journey_df.sort(["user_id", "event_order"])
                .group_by("user_id")
                .agg(pl.col("event_name").str.concat(" → ").alias("path"))
                .select("path")
                .n_unique()
            )

            return {
                "total_cases": total_cases,
                "avg_duration": avg_duration,
                "completion_rate": completion_rate,
                "unique_paths": unique_paths,
                "total_activities": len(activities),
                "total_transitions": len(transitions),
            }

        except Exception as e:
            self.logger.warning(
                f"Optimized statistics calculation failed: {str(e)}, using fallback"
            )
            return {
                "total_cases": 0,
                "avg_duration": 0,
                "completion_rate": 0,
                "unique_paths": 0,
                "total_activities": len(activities),
                "total_transitions": len(transitions),
            }

    def _generate_process_insights(
        self,
        activities: dict[str, dict[str, Any]],
        transitions: dict[tuple[str, str], dict[str, Any]],
        cycles: list[dict[str, Any]],
        variants: list[dict[str, Any]],
        statistics: dict[str, float],
    ) -> list[str]:
        """Generate automatic insights about the process"""
        insights = []

        try:
            # Process complexity insight
            if statistics["unique_paths"] > statistics["total_cases"] * 0.8:
                insights.append(
                    f"🌟 High process variability: {statistics['unique_paths']:.0f} unique paths from {statistics['total_cases']:.0f} cases"
                )

            # Bottleneck detection
            bottleneck_activities = []
            for name, data in activities.items():
                if data["avg_duration"] > 24:  # More than 24 hours
                    bottleneck_activities.append((name, data["avg_duration"]))

            if bottleneck_activities:
                bottleneck_activities.sort(key=lambda x: x[1], reverse=True)
                insights.append(
                    f"🚨 Bottleneck detected: '{bottleneck_activities[0][0]}' takes {bottleneck_activities[0][1]:.1f} hours on average"
                )

            # Popular path insight
            if variants:
                top_variant = variants[0]
                insights.append(
                    f"📈 Most common path: {' → '.join(top_variant['path'][:3])}... ({top_variant['frequency']} users, {top_variant['success_rate']:.1f}% success)"
                )

            # Cycle insight
            problematic_cycles = [c for c in cycles if c.get("impact") == "negative"]
            if problematic_cycles:
                cycle = problematic_cycles[0]
                insights.append(
                    f"🔄 Problematic loop detected: {' → '.join(cycle['path'])} ({cycle['frequency']} occurrences)"
                )

            # Completion rate insight
            if statistics["completion_rate"] < 30:
                insights.append(
                    f"⚠️ Low completion rate: Only {statistics['completion_rate']:.1f}% of users complete the process"
                )
            elif statistics["completion_rate"] > 70:
                insights.append(
                    f"✅ High completion rate: {statistics['completion_rate']:.1f}% of users successfully complete the process"
                )

        except Exception as e:
            self.logger.warning(f"Insight generation failed: {str(e)}")
            insights.append("⚠️ Unable to generate insights due to data processing error")

        return insights
