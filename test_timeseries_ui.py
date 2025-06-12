import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Test data for Time Series Analysis UI debugging
st.title("🧪 Time Series Analysis UI Test")

# Generate test funnel data
@st.cache_data
def generate_test_data():
    funnel_steps = ['view_product', 'add_to_cart', 'checkout', 'purchase']
    events = []
    base_date = datetime(2024, 1, 1)
    
    np.random.seed(42)
    
    for day in range(10):
        current_date = base_date + timedelta(days=day)
        daily_users = 50 + np.random.randint(-10, 11)
        
        for i in range(daily_users):
            user_id = f'user_{day}_{i}'
            
            # Step 1: View product (100%)
            events.append({
                'user_id': user_id,
                'event_name': 'view_product',
                'timestamp': current_date + timedelta(minutes=np.random.randint(0, 1440)),
                'event_properties': {}
            })
            
            # Step 2: Add to cart (40%)
            if np.random.random() < 0.40:
                events.append({
                    'user_id': user_id,
                    'event_name': 'add_to_cart', 
                    'timestamp': current_date + timedelta(minutes=np.random.randint(5, 1440)),
                    'event_properties': {}
                })
                
                # Step 3: Checkout (60% of cart users)
                if np.random.random() < 0.60:
                    events.append({
                        'user_id': user_id,
                        'event_name': 'checkout',
                        'timestamp': current_date + timedelta(minutes=np.random.randint(30, 1440)),
                        'event_properties': {}
                    })
                    
                    # Step 4: Purchase (80% of checkout users)
                    if np.random.random() < 0.80:
                        events.append({
                            'user_id': user_id,
                            'event_name': 'purchase',
                            'timestamp': current_date + timedelta(minutes=np.random.randint(60, 1440)),
                            'event_properties': {}
                        })
    
    return pd.DataFrame(events), funnel_steps

df, steps = generate_test_data()

st.info(f"Generated {len(df):,} events for {df['user_id'].nunique():,} users with {len(steps)} funnel steps")

# Import and test the calculator
try:
    from app import FunnelCalculator, FunnelVisualizer
    
    config = {'use_polars': True}
    calc = FunnelCalculator(config)
    viz = FunnelVisualizer()
    
    # Test different aggregations
    st.subheader("📊 Aggregation Tests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox("Aggregation Period", ["daily", "hourly", "weekly"])
    
    with col2:
        show_details = st.checkbox("Show Details")
    
    # Calculate metrics
    result = calc.calculate_timeseries_metrics(df, steps, period)
    
    if not result.empty:
        st.success(f"✅ {period.capitalize()} aggregation successful: {len(result)} periods")
        
        if show_details:
            st.write("**Available Columns:**")
            st.write(list(result.columns))
            
            st.write("**Sample Data:**")
            st.dataframe(result.head())
        
        # Test secondary metric options
        st.subheader("🎯 Secondary Metric Options")
        
        # Build options like in the real UI
        secondary_options = {"Overall Conversion Rate": "conversion_rate"}
        
        for i in range(len(steps)-1):
            step_from = steps[i]
            step_to = steps[i+1]
            display_name = f"{step_from} → {step_to} Conversion"
            metric_name = f"{step_from}_to_{step_to}_rate"
            secondary_options[display_name] = metric_name
        
        st.write("**Available Secondary Metrics:**")
        for display_name, metric_name in secondary_options.items():
            available = "✅" if metric_name in result.columns else "❌"
            st.write(f"{available} {display_name} (`{metric_name}`)")
        
        # Test chart creation
        primary_metric = "view_product_users"
        secondary_metric = "conversion_rate"
        
        if primary_metric in result.columns and secondary_metric in result.columns:
            st.subheader("📈 Chart Test")
            chart = viz.create_timeseries_chart(result, primary_metric, secondary_metric)
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.error(f"Missing metrics: primary={primary_metric in result.columns}, secondary={secondary_metric in result.columns}")
    
    else:
        st.error("❌ No timeseries data returned")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.exception(e)
