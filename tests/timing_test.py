import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import CountingMethod, FunnelCalculator, FunnelConfig, FunnelOrder, ReentryMode

# Load test dataset
print("Loading test dataset...")
events_df = pd.read_csv("test_data/test_50k.csv")
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
print(f"Loaded {len(events_df)} events, {events_df['user_id'].nunique()} unique users")

# Convert to polars once
print("\nConverting to polars format...")
events_df_polars = pl.from_pandas(events_df)

# Define test configuration and steps
config = FunnelConfig(
    counting_method=CountingMethod.UNIQUE_USERS,
    reentry_mode=ReentryMode.FIRST_ONLY,
    funnel_order=FunnelOrder.ORDERED,
    conversion_window_hours=168,
)

funnel_steps = [
    "User Sign-Up",
    "Verify Email",
    "First Login",
    "Profile Setup",
    "Tutorial Completed",
    "First Purchase",
]

print("\nRunning performance test...")
print("=" * 50)

# Create calculator instances
calculator_pandas = FunnelCalculator(config, use_polars=False)
calculator_polars = FunnelCalculator(config, use_polars=True)

print("\nWarm-up runs...")
# Warm-up to avoid initialization overhead
_ = calculator_pandas.calculate_funnel_metrics(events_df, funnel_steps)
calculator_pandas.clear_cache()

# For polars, use pre-converted dataframe directly in the polars function to avoid conversion overhead
# This bypasses the automatic conversion in calculate_funnel_metrics
_ = calculator_polars._calculate_funnel_metrics_polars(events_df_polars, funnel_steps, events_df)
calculator_polars.clear_cache()

# Run a more comprehensive test with more iterations
NUM_RUNS = 10
print(f"\nDetailed performance test with {NUM_RUNS} iterations...")
print("-" * 65)

# Test pandas implementation
print("Testing Pandas _calculate_unique_users_funnel_optimized...")
pandas_times = []
for i in range(NUM_RUNS):
    start_time = time.time()
    users_funnel = calculator_pandas._calculate_unique_users_funnel_optimized(
        events_df, funnel_steps
    )
    pandas_time = time.time() - start_time
    pandas_times.append(pandas_time)
    print(f"  Run {i+1}: {pandas_time:.6f} seconds")

avg_pandas_time = sum(pandas_times) / len(pandas_times)
pandas_std = np.std(pandas_times)
print(f"Average Pandas time: {avg_pandas_time:.6f} seconds (std: {pandas_std:.6f})\n")

# Test polars implementation
print("Testing Polars _calculate_unique_users_funnel_polars (with pre-converted df)...")
polars_times = []
for i in range(NUM_RUNS):
    start_time = time.time()
    users_funnel = calculator_polars._calculate_unique_users_funnel_polars(
        events_df_polars, funnel_steps
    )
    polars_time = time.time() - start_time
    polars_times.append(polars_time)
    print(f"  Run {i+1}: {polars_time:.6f} seconds")

avg_polars_time = sum(polars_times) / len(polars_times)
polars_std = np.std(polars_times)
print(f"Average Polars time: {avg_polars_time:.6f} seconds (std: {polars_std:.6f})\n")

# Conversion overhead measurement
print("Measuring pandas-to-polars conversion overhead...")
conversion_times = []
for i in range(NUM_RUNS):
    start_time = time.time()
    _ = pl.from_pandas(events_df)
    conversion_time = time.time() - start_time
    conversion_times.append(conversion_time)
    print(f"  Run {i+1}: {conversion_time:.6f} seconds")

avg_conversion_time = sum(conversion_times) / len(conversion_times)
conversion_std = np.std(conversion_times)
print(f"Average conversion time: {avg_conversion_time:.6f} seconds (std: {conversion_std:.6f})\n")

# Test with full pipeline including conversion
print("Testing complete pipeline with conversion...")
complete_polars_times = []
for i in range(NUM_RUNS):
    start_time = time.time()
    # Simulate the full pipeline including conversion
    temp_polars_df = pl.from_pandas(events_df)
    result = calculator_polars._calculate_unique_users_funnel_polars(temp_polars_df, funnel_steps)
    total_time = time.time() - start_time
    complete_polars_times.append(total_time)
    print(f"  Run {i+1}: {total_time:.6f} seconds")

avg_complete_time = sum(complete_polars_times) / len(complete_polars_times)
complete_std = np.std(complete_polars_times)
print(
    f"Average complete pipeline time: {avg_complete_time:.6f} seconds (std: {complete_std:.6f})\n"
)

print("\nSummary of Results:")
print("=" * 65)
print("Function execution only (without conversion overhead):")
print(f"  Pandas: {avg_pandas_time:.6f} ± {pandas_std:.6f} seconds")
print(f"  Polars: {avg_polars_time:.6f} ± {polars_std:.6f} seconds")

speedup = avg_pandas_time / avg_polars_time if avg_polars_time > 0 else float("inf")
improvement = (avg_pandas_time - avg_polars_time) / avg_pandas_time * 100
print(f"  Speedup factor: {speedup:.2f}x")
print(f"  Performance improvement: {improvement:.1f}%")

print(
    f"\nPandas-to-Polars conversion overhead: {avg_conversion_time:.6f} ± {conversion_std:.6f} seconds"
)

print("\nComplete pipeline (with conversion):")
print(f"  Pandas: {avg_pandas_time:.6f} seconds")
print(f"  Polars: {avg_complete_time:.6f} seconds")

complete_speedup = avg_pandas_time / avg_complete_time if avg_complete_time > 0 else float("inf")
complete_improvement = (avg_pandas_time - avg_complete_time) / avg_pandas_time * 100
print(f"  Speedup factor: {complete_speedup:.2f}x")
print(f"  Performance: {complete_improvement:.1f}%")

print(
    f"\nConversion overhead as % of Polars execution: {avg_conversion_time/avg_polars_time*100:.1f}%"
)

print("\nConclusion:")
print("-" * 65)
if speedup > 1:
    print(f"The optimized Polars implementation is {speedup:.2f}x faster than Pandas")
    print("when measuring actual function execution time without conversion overhead.")
    print("\nHowever, the pandas-to-polars conversion overhead is significant,")
    print(
        f"taking {avg_conversion_time/avg_polars_time:.1f}x longer than the function execution itself."
    )

    if complete_speedup < 1:
        print("\nWhen including conversion overhead in a complete pipeline,")
        print(f"Pandas is actually {1/complete_speedup:.2f}x faster overall.")
        print("\nRecommendation: Use Polars only when data is already in Polars format")
        print("or when the calculation time dominates the conversion overhead.")
    else:
        print("\nEven with conversion overhead, Polars is still faster overall.")
else:
    print("The Pandas implementation is faster than Polars, even without conversion overhead.")
