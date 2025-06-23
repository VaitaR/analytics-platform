"""
Test process mining with larger datasets to verify scalability
"""

import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import CountingMethod, FunnelConfig, FunnelOrder, ReentryMode
from path_analyzer import PathAnalyzer


def generate_large_test_data(num_users=2000, events_per_user=15):
    """Generate larger test dataset"""
    import random
    from datetime import datetime, timedelta

    events = []
    event_names = [
        "login",
        "view_homepage",
        "search",
        "view_product",
        "add_to_cart",
        "view_cart",
        "checkout_start",
        "payment_info",
        "purchase",
        "logout",
        "browse_category",
        "view_reviews",
        "compare_products",
        "wishlist_add",
        "error_page",
        "timeout",
        "back_button",
        "refresh_page",
        "contact_support",
    ]
    now = datetime.now()

    for user_id in range(num_users):
        # Generate realistic user journey
        journey_length = random.randint(5, events_per_user)
        current_time = now - timedelta(hours=random.randint(1, 168))

        # Most users start with login
        if random.random() < 0.8:
            events.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": "login",
                    "timestamp": current_time,
                }
            )
            current_time += timedelta(minutes=random.randint(1, 10))

        # Generate journey events
        for _ in range(journey_length):
            # Weighted event selection for realistic patterns
            if random.random() < 0.4:
                event = random.choice(["view_product", "search", "browse_category"])
            elif random.random() < 0.15:
                event = random.choice(["add_to_cart", "view_cart", "checkout_start"])
            elif random.random() < 0.05:
                event = "purchase"  # 5% conversion rate
            elif random.random() < 0.1:
                event = random.choice(["error_page", "timeout", "back_button"])
            else:
                event = random.choice(event_names)

            events.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": event,
                    "timestamp": current_time,
                }
            )

            # Realistic time gaps
            current_time += timedelta(minutes=random.randint(1, 30), seconds=random.randint(0, 59))

    df = pd.DataFrame(events)
    return df.sort_values(["user_id", "timestamp"])


def scalability_test():
    """Test process mining performance with increasing dataset sizes"""
    print("=== PROCESS MINING SCALABILITY TEST ===")

    config = FunnelConfig(
        conversion_window_hours=72,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED,
    )
    analyzer = PathAnalyzer(config)

    # Test different sizes
    test_sizes = [
        (500, 10),  # ~5K events
        (1000, 15),  # ~15K events
        (2000, 20),  # ~40K events
        (3000, 25),  # ~75K events
    ]

    print(
        f"\n{'Dataset':<12} {'Events':<8} {'No Cycles':<12} {'With Cycles':<12} {'Overhead':<10} {'Status':<10}"
    )
    print("-" * 75)

    results = []

    for num_users, events_per_user in test_sizes:
        print(f"{num_users}u x {events_per_user}e", end="  ")

        # Generate data
        df = generate_large_test_data(num_users, events_per_user)
        total_events = len(df)
        print(f"{total_events:>6,}", end="   ")

        # Test without cycles
        start_time = time.time()
        analyzer.discover_process_mining_structure(
            df, min_frequency=10, include_cycles=False
        )
        time_no_cycles = time.time() - start_time
        print(f"{time_no_cycles:>8.2f}s", end="     ")

        # Test with cycles
        start_time = time.time()
        analyzer.discover_process_mining_structure(
            df, min_frequency=10, include_cycles=True
        )
        time_with_cycles = time.time() - start_time
        print(f"{time_with_cycles:>8.2f}s", end="     ")

        # Calculate overhead
        overhead = ((time_with_cycles - time_no_cycles) / time_no_cycles) * 100
        print(f"{overhead:>6.1f}%", end="    ")

        # Check performance
        events_per_sec = total_events / time_with_cycles
        if events_per_sec >= 10000:
            print("✅ FAST")
        elif events_per_sec >= 5000:
            print("⚡ GOOD")
        elif events_per_sec >= 1000:
            print("⚠️  SLOW")
        else:
            print("❌ TOO SLOW")

        results.append(
            {
                "users": num_users,
                "events": total_events,
                "time_no_cycles": time_no_cycles,
                "time_with_cycles": time_with_cycles,
                "events_per_sec": events_per_sec,
                "overhead": overhead,
            }
        )

    # Summary analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    print("Target: >10,000 events/sec for production use")
    print("Acceptable: >5,000 events/sec for most use cases")

    for result in results:
        if result["events_per_sec"] >= 10000:
            status = "✅ PRODUCTION READY"
        elif result["events_per_sec"] >= 5000:
            status = "⚡ ACCEPTABLE"
        elif result["events_per_sec"] >= 1000:
            status = "⚠️  NEEDS OPTIMIZATION"
        else:
            status = "❌ REQUIRES MAJOR OPTIMIZATION"

        print(
            f"{result['users']:>4} users ({result['events']:>5,} events): {result['events_per_sec']:>6,.0f} events/sec - {status}"
        )

    # Check if scaling is linear
    if len(results) >= 2:
        # Compare first and last results
        first = results[0]
        last = results[-1]

        size_ratio = last["events"] / first["events"]
        time_ratio = last["time_with_cycles"] / first["time_with_cycles"]

        print("\n=== SCALABILITY ANALYSIS ===")
        print(f"Dataset size increased by {size_ratio:.1f}x")
        print(f"Processing time increased by {time_ratio:.1f}x")

        if time_ratio <= size_ratio * 1.2:  # Within 20% of linear
            print("✅ EXCELLENT: Near-linear scaling")
        elif time_ratio <= size_ratio * 1.5:  # Within 50% of linear
            print("⚡ GOOD: Acceptable scaling")
        elif time_ratio <= size_ratio * 2.0:  # Within 100% of linear
            print("⚠️  MODERATE: Some performance degradation with size")
        else:
            print("❌ POOR: Significant performance degradation with size")


if __name__ == "__main__":
    scalability_test()
