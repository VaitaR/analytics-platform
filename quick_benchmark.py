"""
Quick performance test to identify process mining bottlenecks
"""
import pandas as pd
import polars as pl
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from path_analyzer import PathAnalyzer
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder

def generate_small_test_data(num_users=100, events_per_user=10):
    """Generate small test dataset"""
    import random
    from datetime import datetime, timedelta
    
    events = []
    event_names = ["login", "view_product", "add_to_cart", "checkout", "purchase"]
    now = datetime.now()
    
    for user_id in range(num_users):
        for i in range(events_per_user):
            events.append({
                "user_id": f"user_{user_id}",
                "event_name": random.choice(event_names),
                "timestamp": now - timedelta(hours=random.randint(1, 168))
            })
    
    df = pd.DataFrame(events)
    return df.sort_values(["user_id", "timestamp"])

def quick_benchmark():
    """Quick benchmark to identify bottlenecks"""
    print("=== QUICK PROCESS MINING BENCHMARK ===")
    
    # Create analyzer
    config = FunnelConfig(
        conversion_window_hours=72,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )
    analyzer = PathAnalyzer(config)
    
    # Test with small dataset
    print("\n1. Testing with small dataset (100 users, ~1K events)...")
    df = generate_small_test_data(100, 10)
    total_events = len(df)
    print(f"   Generated {total_events:,} events")
    
    # Test individual components
    print("\n2. Testing individual process mining components...")
    
    # Test journey building (optimized)
    print("   Testing _build_user_journeys_optimized...")
    start_time = time.time()
    journey_df = analyzer._build_user_journeys_optimized(pl.from_pandas(df))
    journey_time = time.time() - start_time
    print(f"   ✅ Journey building: {journey_time:.3f}s ({total_events/journey_time:,.0f} events/sec)")
    
    # Test activity discovery (optimized)
    print("   Testing _discover_activities (optimized)...")
    start_time = time.time()
    activities = analyzer._discover_activities(pl.from_pandas(df), None)
    activity_time = time.time() - start_time
    print(f"   ✅ Activity discovery: {activity_time:.3f}s, found {len(activities)} activities")
    
    # Test transition discovery (optimized)
    print("   Testing _discover_transitions_optimized...")
    start_time = time.time()
    transitions = analyzer._discover_transitions_optimized(journey_df, min_frequency=1)
    transition_time = time.time() - start_time
    print(f"   ✅ Transition discovery: {transition_time:.3f}s, found {len(transitions)} transitions")
    
    # Test variant discovery (optimized)
    print("   Testing _identify_process_variants_optimized...")
    start_time = time.time()
    variants = analyzer._identify_process_variants_optimized(journey_df)
    variant_time = time.time() - start_time
    print(f"   ✅ Variant discovery: {variant_time:.3f}s, found {len(variants)} variants")
    
    # Test start/end activities (optimized)
    print("   Testing _identify_start_end_activities_optimized...")
    start_time = time.time()
    start_activities, end_activities = analyzer._identify_start_end_activities_optimized(journey_df)
    start_end_time = time.time() - start_time
    print(f"   ✅ Start/end discovery: {start_end_time:.3f}s, {len(start_activities)} start, {len(end_activities)} end")
    
    # Test statistics (optimized)
    print("   Testing _calculate_process_statistics_optimized...")
    start_time = time.time()
    statistics = analyzer._calculate_process_statistics_optimized(journey_df, activities, transitions)
    stats_time = time.time() - start_time
    print(f"   ✅ Statistics calculation: {stats_time:.3f}s")
    
    # Calculate optimized total (without cycles)
    optimized_total = journey_time + activity_time + transition_time + variant_time + start_end_time + stats_time
    print(f"\n   📊 Total optimized components: {optimized_total:.3f}s")
    
    # Test cycle detection (legacy - bottleneck)
    print("\n   Testing cycle detection (legacy method)...")
    start_time = time.time()
    try:
        # Need legacy user_journeys format for cycle detection
        user_journeys = analyzer._build_user_journeys(pl.from_pandas(df))
        cycles = analyzer._detect_cycles(user_journeys, transitions)
        cycle_time = time.time() - start_time
        print(f"   ⚠️  Cycle detection: {cycle_time:.3f}s, found {len(cycles)} cycles")
        
        # Calculate overhead
        overhead_pct = (cycle_time / optimized_total) * 100
        print(f"   🚨 Cycle detection overhead: {overhead_pct:.1f}% of optimized time")
        
    except Exception as e:
        cycle_time = time.time() - start_time
        print(f"   ❌ Cycle detection failed: {e} (took {cycle_time:.3f}s)")
    
    # Test full process discovery (with and without cycles)
    print("\n3. Testing full process discovery...")
    
    print("   Without cycles:")
    start_time = time.time()
    result_no_cycles = analyzer.discover_process_mining_structure(df, min_frequency=1, include_cycles=False)
    time_no_cycles = time.time() - start_time
    print(f"   ✅ No cycles: {time_no_cycles:.3f}s ({total_events/time_no_cycles:,.0f} events/sec)")
    
    print("   With cycles:")
    start_time = time.time()
    result_with_cycles = analyzer.discover_process_mining_structure(df, min_frequency=1, include_cycles=True)
    time_with_cycles = time.time() - start_time
    print(f"   ⚠️  With cycles: {time_with_cycles:.3f}s ({total_events/time_with_cycles:,.0f} events/sec)")
    
    # Performance analysis
    print("\n4. Performance Analysis:")
    cycle_overhead = time_with_cycles - time_no_cycles
    overhead_pct = (cycle_overhead / time_no_cycles) * 100
    print(f"   • Cycle detection adds {cycle_overhead:.3f}s ({overhead_pct:.1f}% overhead)")
    
    # Performance targets
    target_events_per_sec = 10000
    if total_events/time_no_cycles >= target_events_per_sec:
        print(f"   ✅ Performance target met (no cycles): {total_events/time_no_cycles:,.0f} >= {target_events_per_sec:,} events/sec")
    else:
        print(f"   ❌ Performance target missed (no cycles): {total_events/time_no_cycles:,.0f} < {target_events_per_sec:,} events/sec")
    
    if total_events/time_with_cycles >= target_events_per_sec:
        print(f"   ✅ Performance target met (with cycles): {total_events/time_with_cycles:,.0f} >= {target_events_per_sec:,} events/sec")
    else:
        print(f"   ❌ Performance target missed (with cycles): {total_events/time_with_cycles:,.0f} < {target_events_per_sec:,} events/sec")
    
    # Recommendations
    print("\n5. Recommendations:")
    if overhead_pct > 200:
        print("   🚨 CRITICAL: Cycle detection is a major bottleneck - consider optimizing or making optional")
    elif overhead_pct > 100:
        print("   ⚠️  WARNING: Cycle detection significantly impacts performance - optimization recommended")
    elif overhead_pct > 50:
        print("   💡 NOTICE: Cycle detection has moderate impact - optimization could help")
    else:
        print("   ✅ GOOD: Cycle detection overhead is acceptable")
    
    print(f"\n=== BENCHMARK COMPLETE ===")

if __name__ == "__main__":
    quick_benchmark()
