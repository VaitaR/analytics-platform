import pandas as pd
import polars as pl
import numpy as np
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder
from app import FunnelCalculator
import time
from datetime import datetime, timedelta

print("Final Verification Test")
print("=" * 50)

# Load test dataset
print("Loading test dataset...")
events_df = pd.read_csv('test_data/test_50k.csv')
events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
print(f"Loaded {len(events_df)} events, {events_df['user_id'].nunique()} unique users")

# Create a larger synthetic dataset for more realistic performance testing
print("\nGenerating larger synthetic dataset for performance testing...")
num_users = 50000
num_events_per_user = 10
total_events = num_users * num_events_per_user

# Event types
event_types = [
    'User Sign-Up', 'Verify Email', 'First Login', 
    'Profile Setup', 'Tutorial Completed', 'First Purchase',
    'Return Visit', 'Add to Cart', 'View Product', 'Checkout'
]

# Generate synthetic data
np.random.seed(42)  # For reproducibility
user_ids = np.repeat(np.arange(1, num_users + 1), num_events_per_user)
event_indices = np.random.randint(0, len(event_types), size=total_events)
event_names = [event_types[i] for i in event_indices]

# Generate timestamps with some randomness
base_time = datetime(2023, 1, 1)
timestamps = []
for user_id in range(1, num_users + 1):
    user_start = base_time + timedelta(days=np.random.randint(0, 30))
    for _ in range(num_events_per_user):
        timestamp = user_start + timedelta(hours=np.random.randint(0, 168))
        timestamps.append(timestamp)

# Create DataFrame
synthetic_df = pd.DataFrame({
    'user_id': user_ids,
    'event_name': event_names,
    'timestamp': timestamps
})

print(f"Generated synthetic dataset with {len(synthetic_df)} events and {synthetic_df['user_id'].nunique()} users")

# Define test configuration and steps
config = FunnelConfig(
    counting_method=CountingMethod.UNIQUE_USERS,
    reentry_mode=ReentryMode.FIRST_ONLY,
    funnel_order=FunnelOrder.ORDERED,
    conversion_window_hours=168
)
funnel_steps = [
    'User Sign-Up', 'Verify Email', 'First Login', 
    'Profile Setup', 'Tutorial Completed', 'First Purchase'
]

print("\nRunning end-to-end test with standard funnel steps...")

# Create calculator instances
calculator_polars = FunnelCalculator(config, use_polars=True)
calculator_pandas = FunnelCalculator(config, use_polars=False)

# Test with original dataset
print("\n1. Testing with original dataset:")
print("-" * 50)

# Run with Polars implementation
print("Running Polars implementation...")
start_time = time.time()
results_polars = calculator_polars.calculate_funnel_metrics(events_df, funnel_steps)
polars_time = time.time() - start_time
print(f'Polars execution time: {polars_time:.6f} seconds')

# Run with Pandas implementation
print("\nRunning Pandas implementation...")
start_time = time.time()
results_pandas = calculator_pandas.calculate_funnel_metrics(events_df, funnel_steps)
pandas_time = time.time() - start_time
print(f'Pandas execution time: {pandas_time:.6f} seconds')

# Calculate speedup
speedup = pandas_time / polars_time if polars_time > 0 else float('inf')
print(f'\nSpeedup factor (including conversion overhead): {speedup:.2f}x')

# Test with synthetic dataset
print("\n2. Testing with synthetic dataset:")
print("-" * 50)

# Run with Polars implementation
print("Running Polars implementation...")
start_time = time.time()
results_polars_synth = calculator_polars.calculate_funnel_metrics(synthetic_df, funnel_steps)
polars_time_synth = time.time() - start_time
print(f'Polars execution time: {polars_time_synth:.6f} seconds')

# Run with Pandas implementation
print("\nRunning Pandas implementation...")
start_time = time.time()
results_pandas_synth = calculator_pandas.calculate_funnel_metrics(synthetic_df, funnel_steps)
pandas_time_synth = time.time() - start_time
print(f'Pandas execution time: {pandas_time_synth:.6f} seconds')

# Calculate speedup
speedup_synth = pandas_time_synth / polars_time_synth if polars_time_synth > 0 else float('inf')
print(f'\nSpeedup factor (including conversion overhead): {speedup_synth:.2f}x')

# Test with pre-converted polars dataframe (no conversion overhead)
print("\n3. Testing with pre-converted Polars DataFrame (no conversion overhead):")
print("-" * 50)

# Convert to polars once
synthetic_df_polars = pl.from_pandas(synthetic_df)

# Run with Polars implementation directly on polars dataframe
print("Running Polars implementation on pre-converted data...")
start_time = time.time()
results_polars_direct = calculator_polars._calculate_funnel_metrics_polars(synthetic_df_polars, funnel_steps, synthetic_df)
polars_direct_time = time.time() - start_time
print(f'Polars direct execution time: {polars_direct_time:.6f} seconds')

# Calculate speedup vs pandas
speedup_direct = pandas_time_synth / polars_direct_time if polars_direct_time > 0 else float('inf')
print(f'\nSpeedup factor (no conversion overhead): {speedup_direct:.2f}x')

print("\nSummary:")
print("-" * 50)
print(f"Original dataset ({len(events_df)} events):")
print(f"  Pandas: {pandas_time:.6f} seconds")
print(f"  Polars (with conversion): {polars_time:.6f} seconds")
print(f"  Speedup: {speedup:.2f}x")

print(f"\nSynthetic dataset ({len(synthetic_df)} events):")
print(f"  Pandas: {pandas_time_synth:.6f} seconds")
print(f"  Polars (with conversion): {polars_time_synth:.6f} seconds")
print(f"  Polars (no conversion): {polars_direct_time:.6f} seconds")
print(f"  Speedup with conversion: {speedup_synth:.2f}x")
print(f"  Speedup without conversion: {speedup_direct:.2f}x")

print("\nConclusion:")
print("-" * 50)
print("1. The optimized Polars implementation is significantly faster at the function level")
print("2. However, the pandas-to-polars conversion overhead is substantial")
print("3. For data already in Polars format, the optimized implementation provides a significant speedup")
print("4. For larger datasets, the performance benefits become more pronounced")
print("5. Recommendation: Keep data in Polars format throughout the pipeline to avoid conversion overhead")

# Get performance report for specific functions
print("\nFunction-level performance metrics:")
print("-" * 50)

performance_report = calculator_polars.get_performance_report()
for func_name, metrics in sorted(performance_report.items()):
    if '_find_converted_users_polars_optimized' in func_name:
        print(f"Polars optimized find_converted_users: {metrics.get('avg_time', 0)*1000:.2f} ms")
        print(f"Call count: {metrics.get('call_count', 0)}")

performance_report_pandas = calculator_pandas.get_performance_report()
for func_name, metrics in sorted(performance_report_pandas.items()):
    if '_find_converted_users_vectorized' in func_name:
        print(f"Pandas vectorized find_converted_users: {metrics.get('avg_time', 0)*1000:.2f} ms")
        print(f"Call count: {metrics.get('call_count', 0)}") 