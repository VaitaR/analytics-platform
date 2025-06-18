from collections import Counter
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import polars as pl
import pandas as pd
from datetime import datetime, timedelta

# Import necessary classes from models.py
from models import FunnelConfig, FunnelOrder, ReentryMode, PathAnalysisData, ProcessMiningData
import networkx as nx

class PathAnalyzer:
    """
    Helper class for path analysis with optimized implementations for different funnel configurations.
    This class encapsulates the complex logic for path analysis to improve performance and maintainability.
    """
    
    def __init__(self, config: FunnelConfig):
        """Initialize with funnel configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, funnel_events_df: pl.DataFrame, full_history_df: pl.DataFrame, funnel_steps: List[str]) -> PathAnalysisData:
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
        if hasattr(funnel_events_df, 'collect') and callable(getattr(funnel_events_df, 'collect')):
            funnel_events_df = funnel_events_df.collect()
        if hasattr(full_history_df, 'collect') and callable(getattr(full_history_df, 'collect')):
            full_history_df = full_history_df.collect()
        
        # Preprocess DataFrames to handle nested object types
        funnel_events_df = self._preprocess_dataframe(funnel_events_df)
        full_history_df = self._preprocess_dataframe(full_history_df)
        
        # Initialize result containers
        dropoff_paths = {}
        between_steps_events = {}
        
        # Ensure we have the required columns
        try:
            funnel_events_df.select('user_id', 'event_name', 'timestamp')
            full_history_df.select('user_id', 'event_name', 'timestamp')
        except Exception as e:
            self.logger.error(f"Missing required columns in input DataFrames: {str(e)}")
            return PathAnalysisData({}, {})
        
        # Pre-calculate step user sets using Polars
        step_user_sets = {}
        for step in funnel_steps:
            step_users = set(
                funnel_events_df
                .filter(pl.col('event_name') == step)
                .select('user_id')
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
                    funnel_events_df, 
                    full_history_df,
                    dropped_users, 
                    step
                )
                if next_events:
                    dropoff_paths[step] = dict(next_events.most_common(10))
            
            # Find users who converted from current_step to next_step
            conversion_pairs = self._find_conversion_pairs(
                funnel_events_df,
                step,
                next_step,
                funnel_steps
            )
            
            # Extract user IDs from conversion pairs
            if not conversion_pairs.is_empty():
                truly_converted_users = set(conversion_pairs.select('user_id').to_series().to_list())
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
                    conversion_pairs
                )
                step_pair = f"{step} → {next_step}"
                if between_events:  # Only add if non-empty
                    between_steps_events[step_pair] = dict(between_events.most_common(10))
        
        return PathAnalysisData(
            dropoff_paths=dropoff_paths,
            between_steps_events=between_steps_events
        )
    
    def _preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocess DataFrame to handle nested object types and ensure proper column types"""
        try:
            # Handle complex data types by converting object columns to strings
            for col in df.columns:
                if col == 'properties' or str(df[col].dtype).startswith('Object'):
                    df = df.with_columns([
                        pl.col(col).cast(pl.Utf8)
                    ])
            
            # Ensure timestamp column has proper type
            if 'timestamp' in df.columns:
                df = df.with_columns([
                    pl.col('timestamp').cast(pl.Datetime)
                ])
            
            # Ensure user_id is string type
            if 'user_id' in df.columns:
                df = df.with_columns([
                    pl.col('user_id').cast(pl.Utf8)
                ])
            
            # Remove existing _original_order column if it exists and add a new one
            if '_original_order' in df.columns:
                df = df.drop('_original_order')
            df = df.with_row_index("_original_order")
            
            return df
        except Exception as e:
            self.logger.warning(f"Error preprocessing DataFrame: {str(e)}")
            return df
    
    def _analyze_dropoff_paths(self, 
                              funnel_events_df: pl.DataFrame,
                              full_history_df: pl.DataFrame,
                              dropped_users: set,
                              step: str) -> Counter:
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
            lazy_segment_df
            .filter(
                (pl.col('user_id').cast(pl.Utf8).is_in(dropped_user_list)) &
                (pl.col('event_name') == step)
            )
            .group_by('user_id')
            .agg(pl.col('timestamp').max().alias('step_time'))
        )
        
        # Early exit if no step events found
        if last_step_events.collect().height == 0:
            return next_events
        
        # Find the next event after the step for each user within 7 days
        next_events_df = (
            last_step_events
            .join(
                lazy_history_df.filter(
                    pl.col('user_id').cast(pl.Utf8).is_in(dropped_user_list)
                ),
                on='user_id',
                how='inner'
            )
            .filter(
                (pl.col('timestamp') > pl.col('step_time')) &
                (pl.col('timestamp') <= pl.col('step_time') + pl.duration(days=7)) &
                (pl.col('event_name') != step)
            )
            # Use window function to find first event after step for each user
            .with_columns([
                pl.col('timestamp').rank().over(['user_id']).alias('event_rank')
            ])
            .filter(pl.col('event_rank') == 1)
            .select(['user_id', 'event_name'])
        )
        
        # Count next events
        event_counts = (
            next_events_df
            .group_by('event_name')
            .agg(pl.len().alias('count'))
            .collect()
        )
        
        # Convert to Counter format
        if event_counts.height > 0:
            next_events = Counter(dict(zip(
                event_counts['event_name'].to_list(),
                event_counts['count'].to_list()
            )))
        
        # Count users with no further activity
        users_with_events = next_events_df.select(pl.col('user_id').unique()).collect().height
        users_with_no_events = len(dropped_users) - users_with_events
        
        if users_with_no_events > 0:
            next_events['(no further activity)'] = users_with_no_events
        
        return next_events
    
    def _find_conversion_pairs(self, 
                              events_df: pl.DataFrame, 
                              step: str, 
                              next_step: str,
                              funnel_steps: List[str]) -> pl.DataFrame:
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
        step_events = events_df.filter(pl.col('event_name').is_in([step, next_step]))
        
        # Extract step A and step B events separately
        step_A_df = (
            step_events
            .filter(pl.col('event_name') == step)
            .with_columns([
                pl.col('timestamp').alias('step_A_time'),
                pl.lit(step).alias('step')
            ])
        )
            
        step_B_df = (
            step_events
            .filter(pl.col('event_name') == next_step)
            .with_columns([
                pl.col('timestamp').alias('step_B_time'),
                pl.lit(next_step).alias('step_name')  # Used in some fallback methods
            ])
        )
        
        # Early exit if either step has no events
        if step_A_df.height == 0 or step_B_df.height == 0:
            return pl.DataFrame({
                'user_id': [],
                'step': [],
                'next_step': [],
                'step_A_time': [],
                'step_B_time': []
            })
        
        conversion_window = pl.duration(hours=self.config.conversion_window_hours)
        
        # Choose strategy based on funnel configuration
        if self.config.funnel_order == FunnelOrder.ORDERED:
            if self.config.reentry_mode == ReentryMode.FIRST_ONLY:
                # Get first step A for each user
                first_A = (
                    step_A_df
                    .group_by('user_id')
                    .agg([
                        pl.col('step_A_time').min(),
                        pl.col('step').first()
                    ])
                )
                
                # For each user, find first B after A within conversion window
                conversion_pairs = (
                    first_A
                    .join(step_B_df, on='user_id', how='inner')
                    .filter(
                        (pl.col('step_B_time') > pl.col('step_A_time')) &
                        (pl.col('step_B_time') <= pl.col('step_A_time') + conversion_window)
                    )
                    # Use window function to find earliest B for each user
                    .with_columns([
                        pl.col('step_B_time').rank().over(['user_id']).alias('rank')
                    ])
                    .filter(pl.col('rank') == 1)
                    .select(['user_id', 'step', 'step_name', 'step_A_time', 'step_B_time'])
                    .rename({'step_name': 'next_step'})
                )
                
            elif self.config.reentry_mode == ReentryMode.OPTIMIZED_REENTRY:
                # Use join_asof for optimal performance with ORDERED + OPTIMIZED_REENTRY
                try:
                    # Sort both DataFrames by timestamp
                    step_A_df = step_A_df.sort(['user_id', 'step_A_time'])
                    step_B_df = step_B_df.sort(['user_id', 'step_B_time'])
                    
                    # Use join_asof to find the next B event after each A event within window
                    # This avoids the "join explosion" problem
                    conversion_pairs = pl.join_asof(
                        step_A_df.select(['user_id', 'step', 'step_A_time']),
                        step_B_df.select(['user_id', 'step_name', 'step_B_time']).rename({'step_name': 'next_step'}),
                        left_on='step_A_time',
                        right_on='step_B_time',
                        by='user_id',
                        strategy='forward'
                    ).filter(
                        # Keep only pairs within conversion window
                        (pl.col('step_B_time') > pl.col('step_A_time')) &
                        (pl.col('step_B_time') <= pl.col('step_A_time') + conversion_window)
                    )
                    
                except Exception as e:
                    self.logger.warning(f"join_asof failed: {str(e)}, falling back to optimal_step_pairs")
                    # Fall back to the optimal step pairs method
                    conversion_pairs = self._find_optimal_step_pairs(step_A_df, step_B_df)
        else:  # UNORDERED funnel
            if self.config.reentry_mode == ReentryMode.FIRST_ONLY:
                # For unordered funnels, we just need the first occurrence of each step
                first_A = (
                    step_A_df
                    .group_by('user_id')
                    .agg([
                        pl.col('step_A_time').min(),
                        pl.col('step').first()
                    ])
                )
                
                first_B = (
                    step_B_df
                    .group_by('user_id')
                    .agg([
                        pl.col('step_B_time').min(),
                        pl.col('step_name').first().alias('next_step')
                    ])
                )
                
                # Join and filter by conversion window
                conversion_pairs = (
                    first_A
                    .join(first_B, on='user_id', how='inner')
                    .with_columns([
                        # Calculate absolute time difference
                        ((pl.col('step_B_time') - pl.col('step_A_time')).dt.total_hours().abs()).alias('time_diff_hours')
                    ])
                    .filter(pl.col('time_diff_hours') <= self.config.conversion_window_hours)
                    .drop('time_diff_hours')
                )
                
            else:  # UNORDERED + OPTIMIZED_REENTRY
                # For unordered with reentry, we need to find all pairs within window
                # and then group by user to find the earliest valid pair
                joined = (
                    step_A_df.select(['user_id', 'step', 'step_A_time'])
                    .join(
                        step_B_df.select(['user_id', 'step_name', 'step_B_time']).rename({'step_name': 'next_step'}),
                        on='user_id',
                        how='inner'
                    )
                    .with_columns([
                        # Calculate absolute time difference
                        ((pl.col('step_B_time') - pl.col('step_A_time')).dt.total_hours().abs()).alias('time_diff_hours')
                    ])
                    .filter(pl.col('time_diff_hours') <= self.config.conversion_window_hours)
                    .drop('time_diff_hours')
                )
                
                # Find the earliest pair for each user (by earliest combined timestamp)
                conversion_pairs = (
                    joined
                    .with_columns([
                        # Use the minimum of the two timestamps to determine the earliest pair
                        pl.when(pl.col('step_A_time') <= pl.col('step_B_time'))
                        .then(pl.col('step_A_time'))
                        .otherwise(pl.col('step_B_time'))
                        .alias('earliest_time')
                    ])
                    .sort(['user_id', 'earliest_time'])
                    .group_by('user_id')
                    .agg([
                        pl.col('step').first(),
                        pl.col('next_step').first(),
                        pl.col('step_A_time').first(),
                        pl.col('step_B_time').first()
                    ])
                )
        
        return conversion_pairs
    
    def _find_optimal_step_pairs(self, step_A_df: pl.DataFrame, step_B_df: pl.DataFrame) -> pl.DataFrame:
        """Helper function to find optimal step pairs when join_asof fails"""
        conversion_window = pl.duration(hours=self.config.conversion_window_hours)
        
        # Handle empty dataframes
        if step_A_df.height == 0 or step_B_df.height == 0:
            return pl.DataFrame({
                'user_id': [],
                'step': [],
                'next_step': [],
                'step_A_time': [],
                'step_B_time': []
            })
        
        try:
            # Ensure we have step_A_time column
            if 'step_A_time' not in step_A_df.columns and 'timestamp' in step_A_df.columns:
                step_A_df = step_A_df.with_columns(pl.col('timestamp').alias('step_A_time'))
            
            # Get step names for labels
            step_name = "Step A"
            next_step_name = "Step B"
            
            if 'step' in step_A_df.columns and step_A_df.height > 0:
                step_name_col = step_A_df.select('step').unique()
                if step_name_col.height > 0:
                    step_name = step_name_col[0, 0]
                    
            if 'step_name' in step_B_df.columns and step_B_df.height > 0:
                next_step_name_col = step_B_df.select('step_name').unique()
                if next_step_name_col.height > 0:
                    next_step_name = next_step_name_col[0, 0]
            
            # Use a fully vectorized approach using only Polars expressions
            # First, create a cross join of users with their A and B times
            user_with_A_times = step_A_df.select(['user_id', 'step_A_time'])
            
            # Ensure B times are properly named
            if 'step_B_time' in step_B_df.columns:
                user_with_B_times = step_B_df.select(['user_id', 'step_B_time'])
            else:
                user_with_B_times = step_B_df.select(['user_id', 'timestamp']).rename({'timestamp': 'step_B_time'})
            
            # Join both tables and filter for valid conversion pairs
            valid_conversions = (
                user_with_A_times
                .join(user_with_B_times, on='user_id', how='inner')
                # Use only native Polars expressions for the filter condition
                .filter(
                    (pl.col('step_B_time') > pl.col('step_A_time')) & 
                    (pl.col('step_B_time') <= pl.col('step_A_time') + conversion_window)
                )
                # For each step_A_time, find the earliest valid step_B_time
                .sort(['user_id', 'step_A_time', 'step_B_time'])
                # Keep the first valid B time for each A time
                .group_by(['user_id', 'step_A_time'])
                .agg(pl.col('step_B_time').first().alias('earliest_B_time'))
                # Keep only the first A->B pair for each user
                .sort(['user_id', 'step_A_time'])
                .group_by('user_id')
                .agg([
                    pl.col('step_A_time').first(),
                    pl.col('earliest_B_time').first().alias('step_B_time')
                ])
                # Add step names as literals
                .with_columns([
                    pl.lit(step_name).alias('step'),
                    pl.lit(next_step_name).alias('next_step')
                ])
                # Select columns in the right order
                .select(['user_id', 'step', 'next_step', 'step_A_time', 'step_B_time'])
            )
            
            return valid_conversions
            
        except Exception as e:
            self.logger.error(f"Fully vectorized approach for finding step pairs failed: {e}")
            
            # Final fallback with empty DataFrame with correct structure
            return pl.DataFrame({
                'user_id': [],
                'step': [],
                'next_step': [],
                'step_A_time': [],
                'step_B_time': []
            })
    
    def _analyze_between_steps_events(self,
                                     funnel_events_df: pl.DataFrame,
                                     full_history_df: pl.DataFrame,
                                     converted_users: set,
                                     step: str,
                                     next_step: str,
                                     funnel_steps: List[str],
                                     conversion_pairs: pl.DataFrame) -> Counter:
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
            conversion_pairs
            .lazy()
            .join(
                lazy_history_df.filter(pl.col('user_id').cast(pl.Utf8).is_in(converted_user_list)),
                on='user_id',
                how='inner'
            )
            .filter(
                # Events must be between step A and step B
                (pl.col('timestamp') > pl.col('step_A_time')) &
                (pl.col('timestamp') < pl.col('step_B_time')) &
                # Exclude the funnel steps themselves
                ~pl.col('event_name').is_in(funnel_steps)
            )
            .select(['user_id', 'event_name'])
        )
        
        # Count events
        event_counts = (
            between_steps_df
            .group_by('event_name')
            .agg(pl.len().alias('count'))
            .collect()
        )
        
        # Convert to Counter format
        if event_counts.height > 0:
            between_events = Counter(dict(zip(
                event_counts['event_name'].to_list(),
                event_counts['count'].to_list()
            )))
        
        # Add special entry for users with no intermediate events
        users_with_events = between_steps_df.select(pl.col('user_id').unique()).collect().height
        users_with_no_events = len(converted_users) - users_with_events
        
        if users_with_no_events > 0:
            between_events['(direct conversion)'] = users_with_no_events
        
        return between_events
    
    def discover_process_mining_structure(self, events_df: Union[pd.DataFrame, pl.DataFrame], 
                                        min_frequency: int = 10,
                                        include_cycles: bool = True,
                                        time_window_hours: Optional[int] = None) -> ProcessMiningData:
        """
        Automatic process discovery from user events using advanced algorithms
        
        Args:
            events_df: DataFrame with events (user_id, event_name, timestamp)
            min_frequency: Minimum frequency to include transition
            include_cycles: Whether to detect cycles and loops
            time_window_hours: Optional time window for process analysis
            
        Returns:
            ProcessMiningData with complete process structure
        """
        # Convert to Polars for efficient processing
        if isinstance(events_df, pd.DataFrame):
            events_pl = pl.from_pandas(events_df)
        else:
            events_pl = events_df
            
        # Filter by time window if specified
        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            events_pl = events_pl.filter(pl.col('timestamp') >= cutoff_time)
        
        # Build user journeys
        user_journeys = self._build_user_journeys(events_pl)
        
        # Discover activities and their characteristics
        activities = self._discover_activities(events_pl, user_journeys)
        
        # Discover transitions between activities
        transitions = self._discover_transitions(user_journeys, min_frequency)
        
        # Detect cycles and loops if requested
        cycles = []
        if include_cycles:
            cycles = self._detect_cycles(user_journeys, transitions)
        
        # Identify process variants (common paths)
        variants = self._identify_process_variants(user_journeys)
        
        # Find start and end activities
        start_activities, end_activities = self._identify_start_end_activities(user_journeys)
        
        # Calculate process statistics
        statistics = self._calculate_process_statistics(user_journeys, activities, transitions)
        
        # Generate automatic insights
        insights = self._generate_process_insights(activities, transitions, cycles, variants, statistics)
        
        return ProcessMiningData(
            activities=activities,
            transitions=transitions,
            cycles=cycles,
            variants=variants,
            start_activities=start_activities,
            end_activities=end_activities,
            statistics=statistics,
            insights=insights
        )
    
    def _build_user_journeys(self, events_pl: pl.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Build user journeys from events"""
        journeys = {}
        
        # Group by user and sort by timestamp
        user_events = (
            events_pl
            .sort(['user_id', 'timestamp'])
            .group_by('user_id')
            .agg([
                pl.col('event_name').alias('events'),
                pl.col('timestamp').alias('timestamps')
            ])
        )
        
        for row in user_events.iter_rows(named=True):
            user_id = row['user_id']
            events = row['events']
            timestamps = row['timestamps']
            
            journey = []
            for i, (event, timestamp) in enumerate(zip(events, timestamps)):
                journey.append({
                    'event': event,
                    'timestamp': timestamp,
                    'order': i,
                    'duration_to_next': None
                })
                
                # Calculate duration to next event
                if i < len(timestamps) - 1:
                    duration = (timestamps[i + 1] - timestamp).total_seconds() / 3600  # hours
                    journey[i]['duration_to_next'] = duration
            
            journeys[user_id] = journey
        
        return journeys
    
    def _discover_activities(self, events_pl: pl.DataFrame, user_journeys: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
        """Discover activities and their characteristics"""
        activities = {}
        
        # Calculate basic activity statistics
        activity_stats = (
            events_pl
            .group_by('event_name')
            .agg([
                pl.len().alias('frequency'),
                pl.col('user_id').n_unique().alias('unique_users'),
                pl.col('timestamp').min().alias('first_occurrence'),
                pl.col('timestamp').max().alias('last_occurrence')
            ])
        )
        
        for row in activity_stats.iter_rows(named=True):
            activity_name = row['event_name']
            
            # Calculate activity characteristics
            durations = []
            is_start_activity = 0
            is_end_activity = 0
            
            for journey in user_journeys.values():
                for i, step in enumerate(journey):
                    if step['event'] == activity_name:
                        if step['duration_to_next']:
                            durations.append(step['duration_to_next'])
                        
                        # Check if this is start/end activity
                        if i == 0:
                            is_start_activity += 1
                        if i == len(journey) - 1:
                            is_end_activity += 1
            
            # Classify activity type
            activity_type = self._classify_activity_type(activity_name, is_start_activity, is_end_activity, row['frequency'])
            
            activities[activity_name] = {
                'frequency': row['frequency'],
                'unique_users': row['unique_users'],
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'is_start': is_start_activity > 0,
                'is_end': is_end_activity > 0,
                'activity_type': activity_type,
                'success_rate': self._calculate_activity_success_rate(activity_name, user_journeys),
                'first_occurrence': row['first_occurrence'],
                'last_occurrence': row['last_occurrence']
            }
        
        return activities
    
    def _discover_transitions(self, user_journeys: Dict[str, List[Dict[str, Any]]], min_frequency: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Discover transitions between activities"""
        transition_counts = defaultdict(int)
        transition_users = defaultdict(set)
        transition_durations = defaultdict(list)
        
        # Count transitions across all user journeys
        for user_id, journey in user_journeys.items():
            for i in range(len(journey) - 1):
                from_event = journey[i]['event']
                to_event = journey[i + 1]['event']
                transition = (from_event, to_event)
                
                transition_counts[transition] += 1
                transition_users[transition].add(user_id)
                
                if journey[i]['duration_to_next']:
                    transition_durations[transition].append(journey[i]['duration_to_next'])
        
        # Filter by minimum frequency and build transition data
        transitions = {}
        total_transitions = sum(transition_counts.values())
        
        for transition, frequency in transition_counts.items():
            if frequency >= min_frequency:
                from_event, to_event = transition
                durations = transition_durations[transition]
                
                transitions[transition] = {
                    'frequency': frequency,
                    'unique_users': len(transition_users[transition]),
                    'avg_duration': sum(durations) / len(durations) if durations else 0,
                    'probability': (frequency / total_transitions) * 100,
                    'transition_type': self._classify_transition_type(from_event, to_event, frequency)
                }
        
        return transitions
    
    def _detect_cycles(self, user_journeys: Dict[str, List[Dict[str, Any]]], transitions: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect cycles and loops in user behavior using graph analysis"""
        cycles = []
        
        # Build directed graph from transitions
        G = nx.DiGraph()
        for (from_event, to_event), data in transitions.items():
            G.add_edge(from_event, to_event, weight=data['frequency'])
        
        # Find simple cycles
        try:
            simple_cycles = list(nx.simple_cycles(G))
            
            for cycle_path in simple_cycles:
                if len(cycle_path) <= 5:  # Focus on short cycles
                    # Calculate cycle statistics
                    cycle_frequency = self._calculate_cycle_frequency(cycle_path, user_journeys)
                    cycle_impact = self._assess_cycle_impact(cycle_path, user_journeys)
                    
                    cycles.append({
                        'path': cycle_path,
                        'frequency': cycle_frequency,
                        'type': 'loop' if len(cycle_path) == 1 else 'cycle',
                        'impact': cycle_impact,
                        'avg_cycle_time': self._calculate_avg_cycle_time(cycle_path, user_journeys)
                    })
        
        except nx.NetworkXError:
            # Handle cases where cycle detection fails
            pass
        
        # Sort by frequency and return top cycles
        cycles.sort(key=lambda x: x['frequency'], reverse=True)
        return cycles[:10]  # Return top 10 cycles
    
    def _identify_process_variants(self, user_journeys: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify common process variants (paths)"""
        path_counts = defaultdict(int)
        path_success = defaultdict(list)
        path_durations = defaultdict(list)
        
        for user_id, journey in user_journeys.items():
            path = tuple(step['event'] for step in journey)
            path_counts[path] += 1
            
            # Calculate path success (did user complete a conversion event?)
            success = self._calculate_path_success(journey)
            path_success[path].append(success)
            
            # Calculate total path duration
            total_duration = sum(step.get('duration_to_next', 0) for step in journey[:-1])
            path_durations[path].append(total_duration)
        
        # Build variants list
        variants = []
        for path, frequency in path_counts.items():
            if frequency >= 5:  # Minimum frequency for variant
                success_rate = sum(path_success[path]) / len(path_success[path]) * 100
                avg_duration = sum(path_durations[path]) / len(path_durations[path])
                
                variants.append({
                    'path': list(path),
                    'frequency': frequency,
                    'success_rate': success_rate,
                    'avg_duration': avg_duration,
                    'variant_type': self._classify_variant_type(path, success_rate)
                })
        
        # Sort by frequency
        variants.sort(key=lambda x: x['frequency'], reverse=True)
        return variants[:20]  # Return top 20 variants
    
    def _identify_start_end_activities(self, user_journeys: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[str], List[str]]:
        """Identify start and end activities"""
        start_counts = defaultdict(int)
        end_counts = defaultdict(int)
        
        for journey in user_journeys.values():
            if journey:
                start_counts[journey[0]['event']] += 1
                end_counts[journey[-1]['event']] += 1
        
        # Get activities that appear as start/end with significant frequency
        total_journeys = len(user_journeys)
        start_threshold = max(1, total_journeys * 0.05)  # 5% threshold
        end_threshold = max(1, total_journeys * 0.05)
        
        start_activities = [event for event, count in start_counts.items() if count >= start_threshold]
        end_activities = [event for event, count in end_counts.items() if count >= end_threshold]
        
        return start_activities, end_activities
    
    def _calculate_process_statistics(self, user_journeys: Dict[str, List[Dict[str, Any]]], 
                                    activities: Dict[str, Dict[str, Any]], 
                                    transitions: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall process statistics"""
        total_cases = len(user_journeys)
        
        # Calculate average journey duration
        journey_durations = []
        for journey in user_journeys.values():
            duration = sum(step.get('duration_to_next', 0) for step in journey[:-1])
            journey_durations.append(duration)
        
        avg_duration = sum(journey_durations) / len(journey_durations) if journey_durations else 0
        
        # Calculate completion rate (journeys that end with success events)
        success_events = {'purchase', 'conversion', 'complete', 'finish', 'success'}
        completed_journeys = 0
        
        for journey in user_journeys.values():
            if journey and any(keyword in journey[-1]['event'].lower() for keyword in success_events):
                completed_journeys += 1
        
        completion_rate = (completed_journeys / total_cases) * 100 if total_cases > 0 else 0
        
        # Count unique paths
        unique_paths = len(set(tuple(step['event'] for step in journey) for journey in user_journeys.values()))
        
        return {
            'total_cases': total_cases,
            'avg_duration': avg_duration,
            'completion_rate': completion_rate,
            'unique_paths': unique_paths,
            'total_activities': len(activities),
            'total_transitions': len(transitions)
        }
    
    def _generate_process_insights(self, activities: Dict[str, Dict[str, Any]], 
                                 transitions: Dict[Tuple[str, str], Dict[str, Any]], 
                                 cycles: List[Dict[str, Any]], 
                                 variants: List[Dict[str, Any]], 
                                 statistics: Dict[str, float]) -> List[str]:
        """Generate automatic insights about the process"""
        insights = []
        
        # Process complexity insight
        if statistics['unique_paths'] > statistics['total_cases'] * 0.8:
            insights.append(f"🌟 High process variability: {statistics['unique_paths']:.0f} unique paths from {statistics['total_cases']:.0f} cases")
        
        # Bottleneck detection
        bottleneck_activities = []
        for name, data in activities.items():
            if data['avg_duration'] > 24:  # More than 24 hours
                bottleneck_activities.append((name, data['avg_duration']))
        
        if bottleneck_activities:
            bottleneck_activities.sort(key=lambda x: x[1], reverse=True)
            insights.append(f"🚨 Bottleneck detected: '{bottleneck_activities[0][0]}' takes {bottleneck_activities[0][1]:.1f} hours on average")
        
        # Popular path insight
        if variants:
            top_variant = variants[0]
            insights.append(f"📈 Most common path: {' → '.join(top_variant['path'][:3])}... ({top_variant['frequency']} users, {top_variant['success_rate']:.1f}% success)")
        
        # Cycle insight
        problematic_cycles = [c for c in cycles if c.get('impact') == 'negative']
        if problematic_cycles:
            cycle = problematic_cycles[0]
            insights.append(f"🔄 Problematic loop detected: {' → '.join(cycle['path'])} ({cycle['frequency']} occurrences)")
        
        # Completion rate insight
        if statistics['completion_rate'] < 30:
            insights.append(f"⚠️ Low completion rate: Only {statistics['completion_rate']:.1f}% of users complete the process")
        elif statistics['completion_rate'] > 70:
            insights.append(f"✅ High completion rate: {statistics['completion_rate']:.1f}% of users successfully complete the process")
        
        return insights
    
    # Helper methods
    def _classify_activity_type(self, activity_name: str, start_count: int, end_count: int, frequency: int) -> str:
        """Classify activity type based on patterns"""
        name_lower = activity_name.lower()
        
        if any(word in name_lower for word in ['login', 'signup', 'register', 'start']):
            return 'entry'
        elif any(word in name_lower for word in ['purchase', 'checkout', 'complete', 'finish']):
            return 'conversion'
        elif any(word in name_lower for word in ['error', 'fail', 'timeout']):
            return 'error'
        elif any(word in name_lower for word in ['view', 'page', 'screen']):
            return 'navigation'
        elif start_count > 0:
            return 'entry'
        elif end_count > 0:
            return 'exit'
        else:
            return 'process'
    
    def _classify_transition_type(self, from_event: str, to_event: str, frequency: int) -> str:
        """Classify transition type"""
        if from_event == to_event:
            return 'loop'
        elif 'error' in to_event.lower():
            return 'error_transition'
        elif frequency > 100:
            return 'main_flow'
        else:
            return 'alternative_flow'
    
    def _calculate_activity_success_rate(self, activity_name: str, user_journeys: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate success rate for an activity"""
        success_count = 0
        total_count = 0
        
        for journey in user_journeys.values():
            for i, step in enumerate(journey):
                if step['event'] == activity_name:
                    total_count += 1
                    # Consider success if user continues to next steps
                    if i < len(journey) - 1:
                        success_count += 1
        
        return (success_count / total_count) * 100 if total_count > 0 else 0
    
    def _calculate_cycle_frequency(self, cycle_path: List[str], user_journeys: Dict[str, List[Dict[str, Any]]]) -> int:
        """Calculate how often a cycle occurs"""
        frequency = 0
        
        for journey in user_journeys.values():
            events = [step['event'] for step in journey]
            # Look for the cycle pattern in the journey
            for i in range(len(events) - len(cycle_path) + 1):
                if events[i:i+len(cycle_path)] == cycle_path:
                    frequency += 1
        
        return frequency
    
    def _assess_cycle_impact(self, cycle_path: List[str], user_journeys: Dict[str, List[Dict[str, Any]]]) -> str:
        """Assess whether a cycle has positive or negative impact"""
        # Simple heuristic: cycles involving error events are negative
        if any('error' in event.lower() or 'fail' in event.lower() for event in cycle_path):
            return 'negative'
        # Cycles involving retry or repeat actions might be negative
        elif any('retry' in event.lower() or 'repeat' in event.lower() for event in cycle_path):
            return 'negative'
        else:
            return 'positive'
    
    def _calculate_avg_cycle_time(self, cycle_path: List[str], user_journeys: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate average time to complete a cycle"""
        cycle_times = []
        
        for journey in user_journeys.values():
            events = [(step['event'], step.get('duration_to_next', 0)) for step in journey]
            
            for i in range(len(events) - len(cycle_path) + 1):
                if [event for event, _ in events[i:i+len(cycle_path)]] == cycle_path:
                    cycle_time = sum(duration for _, duration in events[i:i+len(cycle_path)-1])
                    cycle_times.append(cycle_time)
        
        return sum(cycle_times) / len(cycle_times) if cycle_times else 0
    
    def _calculate_path_success(self, journey: List[Dict[str, Any]]) -> bool:
        """Calculate if a path represents a successful journey"""
        if not journey:
            return False
        
        last_event = journey[-1]['event'].lower()
        success_keywords = ['purchase', 'complete', 'success', 'finish', 'convert', 'checkout']
        
        return any(keyword in last_event for keyword in success_keywords)
    
    def _classify_variant_type(self, path: Tuple[str, ...], success_rate: float) -> str:
        """Classify variant type based on success rate and characteristics"""
        if success_rate > 80:
            return 'high_success'
        elif success_rate > 50:
            return 'medium_success'
        elif success_rate > 20:
            return 'low_success'
        else:
            return 'problematic'