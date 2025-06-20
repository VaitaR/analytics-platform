import json

import numpy as np
import pandas as pd

# Let's create sample data for our funnel analysis application

# Generate demo events
demo_events = [
    {
        "name": "User Sign-Up",
        "category": "Authentication",
        "description": "User creates new account",
        "frequency": "high",
    },
    {
        "name": "Verify Email",
        "category": "Authentication",
        "description": "Email verification completed",
        "frequency": "high",
    },
    {
        "name": "First Login",
        "category": "Authentication",
        "description": "User logs in for first time",
        "frequency": "high",
    },
    {
        "name": "Profile Setup",
        "category": "Onboarding",
        "description": "User completes profile information",
        "frequency": "medium",
    },
    {
        "name": "Tutorial Completed",
        "category": "Onboarding",
        "description": "User finishes onboarding tutorial",
        "frequency": "medium",
    },
    {
        "name": "Add to Cart",
        "category": "E-commerce",
        "description": "Item added to shopping cart",
        "frequency": "high",
    },
    {
        "name": "Checkout Started",
        "category": "E-commerce",
        "description": "User begins checkout process",
        "frequency": "high",
    },
    {
        "name": "Payment Info Entered",
        "category": "E-commerce",
        "description": "Payment information provided",
        "frequency": "medium",
    },
    {
        "name": "Payment Completed",
        "category": "E-commerce",
        "description": "Transaction successfully processed",
        "frequency": "high",
    },
    {
        "name": "First Purchase",
        "category": "E-commerce",
        "description": "User makes first purchase",
        "frequency": "medium",
    },
    {
        "name": "Product View",
        "category": "Engagement",
        "description": "User views product details",
        "frequency": "high",
    },
    {
        "name": "Search Performed",
        "category": "Engagement",
        "description": "User searches for products",
        "frequency": "high",
    },
    {
        "name": "Wishlist Added",
        "category": "Engagement",
        "description": "Item added to wishlist",
        "frequency": "low",
    },
    {
        "name": "Review Submitted",
        "category": "Engagement",
        "description": "User submits product review",
        "frequency": "low",
    },
    {
        "name": "Share Product",
        "category": "Social",
        "description": "User shares product on social media",
        "frequency": "low",
    },
    {
        "name": "Invite Friend",
        "category": "Social",
        "description": "User invites friend to platform",
        "frequency": "low",
    },
    {
        "name": "App Downloaded",
        "category": "Mobile",
        "description": "Mobile app downloaded",
        "frequency": "medium",
    },
    {
        "name": "Push Notification Enabled",
        "category": "Mobile",
        "description": "User enables push notifications",
        "frequency": "medium",
    },
]

# Create a DataFrame from the events
df_events = pd.DataFrame(demo_events)
print("Demo Events:")
print(df_events.head())

# Sample funnel data for visualization
sample_funnel_data = {
    "User Sign-Up": {"users": 10000, "conversion_rate": 100},
    "Verify Email": {"users": 7500, "conversion_rate": 75},
    "First Login": {"users": 6000, "conversion_rate": 60},
    "Profile Setup": {"users": 4500, "conversion_rate": 45},
    "Tutorial Completed": {"users": 3600, "conversion_rate": 36},
}

# Convert to DataFrame for visualization
df_funnel = pd.DataFrame(
    {
        "step": list(sample_funnel_data.keys()),
        "users": [data["users"] for data in sample_funnel_data.values()],
        "conversion_rate": [data["conversion_rate"] for data in sample_funnel_data.values()],
    }
)
print("\nSample Funnel Data:")
print(df_funnel)

# Time series data for conversion over time
dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
steps = [
    "User Sign-Up",
    "Verify Email",
    "First Login",
    "Profile Setup",
    "Tutorial Completed",
]

# Generate random time series data
np.random.seed(42)
time_series_data = {}
for i, step in enumerate(steps):
    # Create a base value that decreases for each step in the funnel
    base_value = 1000 * (0.8**i)
    # Add some random fluctuation
    values = np.random.normal(base_value, base_value * 0.1, size=len(dates))
    # Ensure values are positive and have a slight trend
    values = np.maximum(values + np.linspace(0, base_value * 0.2, len(dates)), 0)
    time_series_data[step] = values.round().astype(int)

# Convert to DataFrame
df_time_series = pd.DataFrame(time_series_data, index=dates)

print("\nTime Series Data (first 5 rows):")
print(df_time_series.head())

# Sample segment data (mobile vs desktop)
segments = ["Mobile", "Desktop", "Tablet"]
segment_data = {}

for segment in segments:
    segment_data[segment] = {}
    # Different starting values for each segment
    base_value = 1000 if segment == "Mobile" else (800 if segment == "Desktop" else 400)
    for i, step in enumerate(steps):
        # Each step drops by a percentage, different for each segment
        drop_rate = 0.75 if segment == "Mobile" else (0.85 if segment == "Desktop" else 0.70)
        segment_data[segment][step] = int(base_value * (drop_rate**i))

# Convert to DataFrame
df_segments = pd.DataFrame(segment_data)
df_segments.index = steps
print("\nSegment Data:")
print(df_segments)

# Properties for filtering
properties = [
    {"name": "platform", "type": "string", "values": ["mobile", "desktop", "tablet"]},
    {"name": "country", "type": "string", "values": ["US", "UK", "CA", "AU", "DE"]},
    {"name": "user_type", "type": "string", "values": ["new", "returning", "premium"]},
    {
        "name": "campaign",
        "type": "string",
        "values": ["organic", "paid_search", "social", "email"],
    },
    {
        "name": "device_type",
        "type": "string",
        "values": ["iOS", "Android", "Windows", "Mac"],
    },
]

# Sample data for time-to-convert distribution
time_to_convert = {
    "Verify Email": np.random.exponential(scale=0.5, size=1000) * 60,  # in minutes
    "First Login": np.random.exponential(scale=12, size=1000) * 60,  # in minutes
    "Profile Setup": np.random.exponential(scale=24, size=1000) * 60,  # in minutes
    "First Purchase": np.random.exponential(scale=72, size=1000) * 60,  # in minutes
}

# Convert to DataFrame
df_time_to_convert = pd.DataFrame(time_to_convert)
print("\nTime to Convert Statistics (in minutes):")
print(df_time_to_convert.describe())

# Sample data for A/B test comparison
ab_test_data = {
    "Variant A": {
        "User Sign-Up": 5000,
        "Verify Email": 4000,
        "First Login": 3400,
        "Profile Setup": 2800,
        "First Purchase": 2100,
    },
    "Variant B": {
        "User Sign-Up": 5000,
        "Verify Email": 4200,
        "First Login": 3650,
        "Profile Setup": 3100,
        "First Purchase": 2400,
    },
}

# Convert to DataFrame
df_ab_test = pd.DataFrame(ab_test_data)
print("\nA/B Test Data:")
print(df_ab_test)

# Calculate conversion rates for A/B test
df_ab_test_rates = df_ab_test.div(df_ab_test.iloc[0]) * 100
print("\nA/B Test Conversion Rates (%):")
print(df_ab_test_rates)

# Export to CSV files
df_events.to_csv("test_data/demo_events.csv", index=False)
df_funnel.to_csv("test_data/sample_funnel.csv", index=False)
df_time_series.to_csv("test_data/time_series_data.csv")
df_segments.to_csv("test_data/segment_data.csv")
df_time_to_convert.describe().to_csv("test_data/time_to_convert_stats.csv")
df_ab_test.to_csv("test_data/ab_test_data.csv")
df_ab_test_rates.to_csv("test_data/ab_test_rates.csv")

# Create a comprehensive data package for the application
data_package = {
    "demo_events": demo_events,
    "sample_funnel_data": sample_funnel_data,
    "time_series_data": df_time_series.reset_index().to_dict("records"),
    "segment_data": segment_data,
    "properties": properties,
    "time_to_convert": {k: v.tolist() for k, v in time_to_convert.items()},
    "ab_test_data": ab_test_data,
}

# Export to JSON
with open("funnel_analysis_data.json", "w") as f:
    json.dump(data_package, f, indent=2, default=str)

print("\nData files successfully exported!")
