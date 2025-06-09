import pandas as pd
import polars as pl
import numpy as np
import time
import random
from collections import Counter
from datetime import datetime, timedelta
from app import FunnelCalculator, FunnelConfig, CountingMethod, ReentryMode, FunnelOrder

def generate_test_data(num_users=1000, events_per_user=20):
    """Generate synthetic test data for benchmarking"""
    user_ids = [f"user_{i}" for i in range(num_users)]
    event_names = ["Step A", "Step B", "Step C", "Other Event 1", "Other Event 2", "Other Event 3"]
    
    data = []
    now = datetime.now()
    
    for user_id in user_ids:
        # Ensure each user has the funnel steps
        data.append({
            "user_id": user_id,
            "event_name": "Step A",
            "timestamp": now - timedelta(hours=random.randint(100, 200))
        })
        
        # 80% of users complete step B
        if random.random() < 0.8:
            step_a_time = data[-1]["timestamp"]
            data.append({
                "user_id": user_id,
                "event_name": "Step B",
                "timestamp": step_a_time + timedelta(hours=random.randint(1, 48))
            })
        
        # 50% of users complete step C
        if random.random() < 0.5:
            # Get the latest event time for this user
            latest_time = max([e["timestamp"] for e in data if e["user_id"] == user_id])
            data.append({
                "user_id": user_id,
                "event_name": "Step C",
                "timestamp": latest_time + timedelta(hours=random.randint(1, 48))
            })
        
        # Add random events
        for _ in range(events_per_user):
            # Get the latest event time for this user
            user_events = [e for e in data if e["user_id"] == user_id]
            if user_events:
                latest_time = max([e["timestamp"] for e in user_events])
                time_offset = timedelta(hours=random.randint(-24, 72))  # Some events before, some after
            else:
                latest_time = now
                time_offset = timedelta(hours=random.randint(-100, 0))
                
            data.append({
                "user_id": user_id,
                "event_name": random.choice(event_names[3:]),  # Other events
                "timestamp": latest_time + time_offset
            })
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")
    
    return df

def benchmark_path_analysis():
    """Benchmark path analysis functions"""
    print("Generating test data...")
    df = generate_test_data(num_users=5000, events_per_user=30)
    print(f"Generated dataset with {len(df)} events for {df['user_id'].nunique()} users")
    
    # Create calculator instances
    config = FunnelConfig(
        conversion_window_hours=72,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )
    
    calculator = FunnelCalculator(config, use_polars=True)
    funnel_steps = ["Step A", "Step B", "Step C"]
    
    # Convert to Polars DataFrame
    polars_df = pl.from_pandas(df)
    
    # Run original and optimized functions for analyze_dropoff_paths
    print("\nBenchmarking _analyze_dropoff_paths functions...")
    
    # Setup for dropoff paths test
    step_user_sets = {}
    for step in funnel_steps:
        step_users = set(
            polars_df
            .filter(pl.col('event_name') == step)
            .select('user_id')
            .unique()
            .to_series()
            .to_list()
        )
        step_user_sets[step] = step_users
    
    step = "Step A"
    next_step = "Step B"
    step_users = step_user_sets[step]
    next_step_users = step_user_sets[next_step]
    dropped_users = step_users - next_step_users
    
    # Benchmark original function
    start_time = time.time()
    original_result = calculator._analyze_dropoff_paths_polars(
        polars_df, 
        polars_df,
        dropped_users, 
        step
    )
    original_time = time.time() - start_time
    print(f"Original _analyze_dropoff_paths_polars: {original_time:.6f} seconds")
    
    # Benchmark optimized function
    start_time = time.time()
    optimized_result = calculator._analyze_dropoff_paths_polars_optimized(
        polars_df, 
        polars_df,
        dropped_users, 
        step
    )
    optimized_time = time.time() - start_time
    print(f"Optimized _analyze_dropoff_paths_polars_optimized: {optimized_time:.6f} seconds")
    
    # Calculate improvement
    improvement = (original_time - optimized_time) / original_time * 100
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"Improvement: {improvement:.2f}%, Speedup: {speedup:.2f}x")
    
    # Verify results are equivalent
    print(f"Results equivalent: {dict(original_result) == dict(optimized_result)}")
    
    # Run original and optimized functions for analyze_between_steps
    print("\nBenchmarking _analyze_between_steps functions...")
    
    # Find converted users (users who did both step A and step B)
    converted_users = step_users.intersection(next_step_users)
    
    # Benchmark original function
    start_time = time.time()
    original_result = calculator._analyze_between_steps_polars(
        polars_df,
        polars_df,
        converted_users, 
        step, 
        next_step, 
        funnel_steps
    )
    original_time = time.time() - start_time
    print(f"Original _analyze_between_steps_polars: {original_time:.6f} seconds")
    
    # Benchmark optimized function
    start_time = time.time()
    optimized_result = calculator._analyze_between_steps_polars_optimized(
        polars_df,
        polars_df,
        converted_users, 
        step, 
        next_step, 
        funnel_steps
    )
    optimized_time = time.time() - start_time
    print(f"Optimized _analyze_between_steps_polars_optimized: {optimized_time:.6f} seconds")
    
    # Calculate improvement
    improvement = (original_time - optimized_time) / original_time * 100
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    print(f"Improvement: {improvement:.2f}%, Speedup: {speedup:.2f}x")
    
    # Verify results are equivalent
    print(f"Results equivalent: {dict(original_result) == dict(optimized_result)}")

if __name__ == "__main__":
    benchmark_path_analysis() 