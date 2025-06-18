"""
Benchmark script for process mining performance optimization
"""
import pandas as pd
import polars as pl
import numpy as np
import time
import random
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from path_analyzer import PathAnalyzer
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder

def generate_process_mining_data(num_users=1000, events_per_user=20):
    """Generate synthetic event data for process mining benchmarking"""
    user_ids = [f"user_{i}" for i in range(num_users)]
    event_names = [
        "login", "view_product", "add_to_cart", "view_cart", "checkout_start",
        "payment_info", "purchase", "logout", "browse_category", "search",
        "view_reviews", "compare_products", "wishlist_add", "error_page",
        "timeout", "back_button", "refresh_page", "contact_support"
    ]
    
    data = []
    now = datetime.now()
    
    for user_id in user_ids:
        # Simulate realistic user journeys
        journey_length = random.randint(5, events_per_user)
        current_time = now - timedelta(hours=random.randint(1, 168))  # Last week
        
        # Start with login for most users
        if random.random() < 0.8:
            data.append({
                "user_id": user_id,
                "event_name": "login",
                "timestamp": current_time
            })
            current_time += timedelta(minutes=random.randint(1, 30))
        
        # Generate realistic event sequences
        for _ in range(journey_length):
            # Weight events based on realistic probabilities
            if random.random() < 0.3:
                event = random.choice(["view_product", "browse_category", "search"])
            elif random.random() < 0.1:
                event = random.choice(["error_page", "timeout", "back_button"])
            elif random.random() < 0.05:
                event = "purchase"  # Low conversion rate
            else:
                event = random.choice(event_names)
            
            data.append({
                "user_id": user_id,
                "event_name": event,
                "timestamp": current_time
            })
            
            # Add realistic time gaps
            current_time += timedelta(
                minutes=random.randint(1, 60),
                seconds=random.randint(0, 59)
            )
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(data)
    df = df.sort_values(["user_id", "timestamp"])
    
    return df

def benchmark_process_mining():
    """Benchmark process mining performance with different dataset sizes"""
    # Create a basic config for PathAnalyzer
    config = FunnelConfig(
        conversion_window_hours=72,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )
    analyzer = PathAnalyzer(config)
    
    # Test different data sizes
    test_sizes = [
        (500, 15),    # Small: ~7.5K events
        (1000, 20),   # Medium: ~20K events  
        (2500, 25),   # Large: ~62.5K events
        (5000, 30),   # Extra Large: ~150K events
    ]
    
    results = []
    
    for num_users, events_per_user in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {num_users} users, ~{events_per_user} events per user")
        print(f"{'='*60}")
        
        # Generate test data
        print("Generating test data...")
        start_time = time.time()
        df = generate_process_mining_data(num_users, events_per_user)
        data_gen_time = time.time() - start_time
        
        total_events = len(df)
        unique_users = df['user_id'].nunique()
        print(f"Generated {total_events:,} events for {unique_users:,} users in {data_gen_time:.2f}s")
        
        # Test with cycles disabled (should be fast)
        print("\nTesting process mining discovery (cycles disabled)...")
        start_time = time.time()
        result_no_cycles = analyzer.discover_process_mining_structure(
            df, 
            min_frequency=5,
            include_cycles=False
        )
        time_no_cycles = time.time() - start_time
        
        print(f"  Time: {time_no_cycles:.3f}s")
        print(f"  Activities: {len(result_no_cycles.activities)}")
        print(f"  Transitions: {len(result_no_cycles.transitions)}")
        print(f"  Variants: {len(result_no_cycles.variants)}")
        print(f"  Events/sec: {total_events/time_no_cycles:,.0f}")
        
        # Test with cycles enabled (bottleneck)
        print("\nTesting process mining discovery (cycles enabled)...")
        start_time = time.time()
        result_with_cycles = analyzer.discover_process_mining_structure(
            df, 
            min_frequency=5,
            include_cycles=True
        )
        time_with_cycles = time.time() - start_time
        
        print(f"  Time: {time_with_cycles:.3f}s")
        print(f"  Activities: {len(result_with_cycles.activities)}")
        print(f"  Transitions: {len(result_with_cycles.transitions)}")
        print(f"  Variants: {len(result_with_cycles.variants)}")
        print(f"  Cycles: {len(result_with_cycles.cycles)}")
        print(f"  Events/sec: {total_events/time_with_cycles:,.0f}")
        
        # Calculate cycle detection overhead
        cycle_overhead = time_with_cycles - time_no_cycles
        overhead_pct = (cycle_overhead / time_no_cycles) * 100
        print(f"  Cycle detection overhead: {cycle_overhead:.3f}s ({overhead_pct:.1f}%)")
        
        results.append({
            'users': num_users,
            'events_per_user': events_per_user,
            'total_events': total_events,
            'time_no_cycles': time_no_cycles,
            'time_with_cycles': time_with_cycles,
            'cycle_overhead': cycle_overhead,
            'events_per_sec_no_cycles': total_events/time_no_cycles,
            'events_per_sec_with_cycles': total_events/time_with_cycles
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Size':<10} {'Events':<8} {'No Cycles':<12} {'With Cycles':<12} {'Overhead':<10}")
    print(f"{'-'*60}")
    
    for r in results:
        size_label = f"{r['users']}u"
        events_label = f"{r['total_events']:,}"
        no_cycles_label = f"{r['time_no_cycles']:.2f}s"
        with_cycles_label = f"{r['time_with_cycles']:.2f}s"
        overhead_label = f"{r['cycle_overhead']:.2f}s"
        
        print(f"{size_label:<10} {events_label:<8} {no_cycles_label:<12} {with_cycles_label:<12} {overhead_label:<10}")
    
    # Performance targets
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Check if we meet performance targets
    target_events_per_sec = 10000  # Target: process 10K events per second
    
    for r in results:
        print(f"\n{r['users']} users ({r['total_events']:,} events):")
        
        if r['events_per_sec_no_cycles'] >= target_events_per_sec:
            print(f"  ✅ No cycles: {r['events_per_sec_no_cycles']:,.0f} events/sec (meets target)")
        else:
            print(f"  ❌ No cycles: {r['events_per_sec_no_cycles']:,.0f} events/sec (below target)")
        
        if r['events_per_sec_with_cycles'] >= target_events_per_sec:
            print(f"  ✅ With cycles: {r['events_per_sec_with_cycles']:,.0f} events/sec (meets target)")
        else:
            print(f"  ❌ With cycles: {r['events_per_sec_with_cycles']:,.0f} events/sec (below target)")
            
        # Identify bottleneck severity
        overhead_pct = (r['cycle_overhead'] / r['time_no_cycles']) * 100
        if overhead_pct > 100:
            print(f"  🚨 Severe bottleneck: Cycle detection adds {overhead_pct:.0f}% overhead")
        elif overhead_pct > 50:
            print(f"  ⚠️  Moderate bottleneck: Cycle detection adds {overhead_pct:.0f}% overhead")
        else:
            print(f"  ✅ Acceptable: Cycle detection adds {overhead_pct:.0f}% overhead")

if __name__ == "__main__":
    benchmark_process_mining()
