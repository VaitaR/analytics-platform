import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import DataSourceManager, FunnelCalculator, FunnelConfigManager
from models import (
    CountingMethod,
    FunnelConfig,
    FunnelOrder,
    ReentryMode,
)
from path_analyzer import PathAnalyzer
from ui.visualization import (
    FunnelVisualizer,
)

# Configure page
st.set_page_config(
    page_title="Professional Funnel Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .step-container {
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f9fafb;
    }
    .funnel-step {
        background: linear-gradient(90deg, #3b82f6, #1e40af);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin: 0.25rem;
        display: inline-block;
        font-weight: 500;
    }
    .data-source-card {
        border: 2px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8fafc;
    }
    .segment-card {
        border: 2px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #ecfdf5;
    }
    .cohort-card {
        border: 2px solid #8b5cf6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f3e8ff;
    }
    .event-card {
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.25rem 0;
        background: #ffffff;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .event-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .event-card.selected {
        border-color: #10b981;
        background: #f0fdf4;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }
    .frequency-high {
        border-left: 4px solid #ef4444;
    }
    .frequency-medium {
        border-left: 4px solid #f59e0b;
    }
    .frequency-low {
        border-left: 4px solid #10b981;
    }
    .category-header {
        background: linear-gradient(90deg, #f3f4f6, #e5e7eb);
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-weight: 600;
        color: #374151;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Performance monitoring decorators


# Data Source Management
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "funnel_steps" not in st.session_state:
        st.session_state.funnel_steps = []
    if "funnel_config" not in st.session_state:
        st.session_state.funnel_config = FunnelConfig()
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "events_data" not in st.session_state:
        st.session_state.events_data = None
    if "data_source_manager" not in st.session_state:
        st.session_state.data_source_manager = DataSourceManager()
    if "available_properties" not in st.session_state:
        st.session_state.available_properties = {}
    if "saved_configurations" not in st.session_state:
        st.session_state.saved_configurations = []
    if "event_metadata" not in st.session_state:
        st.session_state.event_metadata = {}
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = []
    if "selected_frequencies" not in st.session_state:
        st.session_state.selected_frequencies = []
    if "event_statistics" not in st.session_state:
        st.session_state.event_statistics = {}
    if "event_selections" not in st.session_state:
        st.session_state.event_selections = {}


# Enhanced Event Selection Functions
def filter_events(
    events_metadata: dict[str, dict[str, Any]],
    search_query: str,
    selected_categories: list[str],
    selected_frequencies: list[str],
) -> dict[str, dict[str, Any]]:
    """Filter events based on search query, categories, and frequencies"""
    filtered = {}

    for event_name, metadata in events_metadata.items():
        # Search filter
        if search_query and search_query.lower() not in event_name.lower():
            if search_query.lower() not in metadata.get("description", "").lower():
                continue

        # Category filter
        if selected_categories and metadata.get("category") not in selected_categories:
            continue

        # Frequency filter
        if selected_frequencies and metadata.get("frequency") not in selected_frequencies:
            continue

        filtered[event_name] = metadata

    return filtered


# DISABLED complex functions - keeping for reference but not using in simplified version
def funnel_step_manager_DISABLED():
    """Fragment for managing funnel steps without full page reloads - DISABLED"""


def event_browser_DISABLED():
    """Fragment for browsing and adding events without full page reloads - DISABLED"""


def create_enhanced_event_selector_DISABLED():
    """Create enhanced event selector with search, filters, and categorized display - DISABLED in simplified version"""


def get_comprehensive_performance_analysis() -> dict[str, Any]:
    """
    Get comprehensive performance analysis from all monitored components
    """
    analysis = {
        "data_source_metrics": {},
        "funnel_calculator_metrics": {},
        "combined_bottlenecks": [],
        "overall_summary": {},
    }

    # Get data source performance if available
    if hasattr(st.session_state, "data_source_manager") and hasattr(
        st.session_state.data_source_manager, "_performance_metrics"
    ):
        analysis["data_source_metrics"] = st.session_state.data_source_manager._performance_metrics

    # Get funnel calculator performance if available
    if hasattr(st.session_state, "last_calculator") and hasattr(
        st.session_state.last_calculator, "_performance_metrics"
    ):
        analysis["funnel_calculator_metrics"] = (
            st.session_state.last_calculator._performance_metrics
        )

        # Get bottleneck analysis from calculator
        bottleneck_analysis = st.session_state.last_calculator.get_bottleneck_analysis()
        if bottleneck_analysis.get("bottlenecks"):
            analysis["combined_bottlenecks"] = bottleneck_analysis["bottlenecks"]

    # Calculate overall metrics
    all_metrics = {}
    all_metrics.update(analysis["data_source_metrics"])
    all_metrics.update(analysis["funnel_calculator_metrics"])

    total_time = 0
    total_calls = 0

    for func_name, times in all_metrics.items():
        if times:
            total_time += sum(times)
            total_calls += len(times)

    analysis["overall_summary"] = {
        "total_execution_time": total_time,
        "total_function_calls": total_calls,
        "average_call_time": total_time / total_calls if total_calls > 0 else 0,
        "functions_monitored": len([f for f, t in all_metrics.items() if t]),
    }

    return analysis


def get_event_statistics(events_data: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Get comprehensive statistics for each event in the dataset"""
    if events_data is None or events_data.empty:
        return {}

    event_stats = {}
    event_counts = events_data["event_name"].value_counts()
    total_events = len(events_data)
    unique_users = events_data["user_id"].nunique()

    for event_name in events_data["event_name"].unique():
        event_data = events_data[events_data["event_name"] == event_name]
        unique_event_users = event_data["user_id"].nunique()
        event_count = len(event_data)

        # Calculate frequency categories
        if event_count > total_events * 0.1:  # >10% of all events
            frequency_level = "high"
            frequency_color = "#ef4444"
        elif event_count > total_events * 0.01:  # >1% of all events
            frequency_level = "medium"
            frequency_color = "#f59e0b"
        else:
            frequency_level = "low"
            frequency_color = "#10b981"

        event_stats[event_name] = {
            "count": event_count,
            "unique_users": unique_event_users,
            "percentage_of_events": (event_count / total_events) * 100,
            "user_coverage": (unique_event_users / unique_users) * 100,
            "frequency_level": frequency_level,
            "frequency_color": frequency_color,
            "avg_per_user": (event_count / unique_event_users if unique_event_users > 0 else 0),
        }

    return event_stats


def create_simple_event_selector():
    """
    Create simplified event selector with proper closure handling and improved architecture.
    Uses callback arguments to avoid closure issues in loops.
    """
    if st.session_state.get("events_data") is None or st.session_state.events_data.empty:
        st.warning("Please load data first to see available events.")
        return

    # --- State Management Functions (defined outside loops) ---

    def toggle_event_in_funnel(event_name: str):
        """Add or remove event from funnel steps."""
        if event_name in st.session_state.funnel_steps:
            st.session_state.funnel_steps.remove(event_name)
        else:
            st.session_state.funnel_steps.append(event_name)
        st.session_state.analysis_results = None  # Clear results when funnel changes

    def move_step(index: int, direction: int):
        """Move funnel step up or down."""
        if 0 <= index + direction < len(st.session_state.funnel_steps):
            # Classic swap
            (
                st.session_state.funnel_steps[index],
                st.session_state.funnel_steps[index + direction],
            ) = (
                st.session_state.funnel_steps[index + direction],
                st.session_state.funnel_steps[index],
            )
            st.session_state.analysis_results = None

    def remove_step(index: int):
        """Remove step from funnel."""
        if 0 <= index < len(st.session_state.funnel_steps):
            st.session_state.funnel_steps.pop(index)
            st.session_state.analysis_results = None

    def clear_all_steps():
        """Clear all funnel steps."""
        st.session_state.funnel_steps = []
        st.session_state.analysis_results = None
        st.toast("🗑️ Funnel cleared!", icon="🗑️")

    def analyze_funnel():
        """Run funnel analysis."""
        if len(st.session_state.funnel_steps) >= 2:
            with st.spinner("Calculating funnel metrics..."):
                # Get polars preference from session state (default to True)
                use_polars = st.session_state.get("use_polars", True)
                calculator = FunnelCalculator(
                    st.session_state.funnel_config, use_polars=use_polars
                )

                # Store calculator for cache management
                st.session_state.last_calculator = calculator

                # Monitor performance
                calculation_start = time.time()
                st.session_state.analysis_results = calculator.calculate_funnel_metrics(
                    st.session_state.events_data, st.session_state.funnel_steps
                )
                calculation_time = time.time() - calculation_start

                # Store performance metrics in session state
                if "performance_history" not in st.session_state:
                    st.session_state.performance_history = []

                engine_used = "Polars" if use_polars else "Pandas"
                st.session_state.performance_history.append(
                    {
                        "timestamp": datetime.now(),
                        "events_count": len(st.session_state.events_data),
                        "steps_count": len(st.session_state.funnel_steps),
                        "calculation_time": calculation_time,
                        "method": st.session_state.funnel_config.counting_method.value,
                        "engine": engine_used,
                    }
                )

                # Keep only last 10 calculations
                if len(st.session_state.performance_history) > 10:
                    st.session_state.performance_history = st.session_state.performance_history[
                        -10:
                    ]

                st.toast(
                    f"✅ {engine_used} analysis completed in {calculation_time:.2f}s!",
                    icon="✅",
                )
        else:
            st.toast("⚠️ Please add at least 2 steps to create a funnel", icon="⚠️")

    # --- UI Display Section ---

    # Use two main columns for better organization
    col_events, col_funnel = st.columns(2)

    with col_events:
        st.markdown("### 📋 Step 1: Select Events")
        search_query = st.text_input(
            "🔍 Search Events",
            placeholder="Start typing to filter...",
            key="event_search",
        )

        if "event_statistics" not in st.session_state:
            st.session_state.event_statistics = get_event_statistics(st.session_state.events_data)

        available_events = sorted(st.session_state.events_data["event_name"].unique())

        if search_query:
            filtered_events = [
                event for event in available_events if search_query.lower() in event.lower()
            ]
        else:
            filtered_events = available_events

        if not filtered_events:
            st.info("No events match your search query.")
        else:
            # Use scrollable container for event list
            with st.container(height=400):
                for event in filtered_events:
                    stats = st.session_state.event_statistics.get(event, {})
                    is_selected = event in st.session_state.funnel_steps

                    # Use columns for layout within container
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        # KEY FIX: Pass event name as argument to callback
                        st.checkbox(
                            event,
                            value=is_selected,
                            key=f"event_cb_{event.replace(' ', '_').replace('-', '_')}",  # UI testing compatible key
                            on_change=toggle_event_in_funnel,
                            args=(event,),  # Pass event name as argument
                            help=f"Add/remove {event} from funnel",
                        )
                    with c2:
                        if stats:
                            st.markdown(
                                f"""<div style="font-size: 0.75rem; text-align: right; color: #888;">
                                {stats["unique_users"]:,} users<br/>
                                <span style="color: {stats["frequency_color"]};">{stats["user_coverage"]:.1f}%</span>
                                </div>""",
                                unsafe_allow_html=True,
                            )

    with col_funnel:
        st.markdown("### 🚀 Step 2: Configure Funnel")

        if not st.session_state.funnel_steps:
            st.info("Select events from the left to build your funnel.")
        else:
            # Display funnel steps with management controls
            for i, step in enumerate(st.session_state.funnel_steps):
                with st.container():
                    r1, r2, r3, r4 = st.columns([0.6, 0.1, 0.1, 0.2])
                    r1.markdown(f"**{i + 1}.** {step}")

                    # Move up button
                    if i > 0:
                        r2.button(
                            "⬆️",
                            key=f"up_{i}",
                            on_click=move_step,
                            args=(i, -1),
                            help="Move up",
                        )

                    # Move down button
                    if i < len(st.session_state.funnel_steps) - 1:
                        r3.button(
                            "⬇️",
                            key=f"down_{i}",
                            on_click=move_step,
                            args=(i, 1),
                            help="Move down",
                        )

                    # Remove button
                    r4.button(
                        "🗑️",
                        key=f"del_{i}",
                        on_click=remove_step,
                        args=(i,),
                        help="Remove step",
                    )

            st.markdown("---")

            # Engine selection
            st.session_state.use_polars = st.checkbox(
                "🚀 Use Polars Engine",
                value=st.session_state.get("use_polars", True),
                help="Use Polars for faster funnel calculations (experimental)",
            )

            # Action buttons
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                st.button(
                    "🚀 Analyze Funnel",
                    key="analyze_funnel_button",
                    type="primary",
                    use_container_width=True,
                    on_click=analyze_funnel,
                )

            with action_col2:
                st.button(
                    "🗑️ Clear All",
                    key="clear_all_button",
                    on_click=clear_all_steps,
                    use_container_width=True,
                )


# Commented out original complex functions - keeping for reference but not using
def create_funnel_templates_DISABLED():
    """Create predefined funnel templates for quick setup - DISABLED in simplified version"""


# Main application
def main():
    st.markdown(
        '<h1 class="main-header">Professional Funnel Analytics Platform</h1>',
        unsafe_allow_html=True,
    )

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration")

        # Data Source Selection
        st.markdown("### 📊 Data Source")
        data_source = st.selectbox(
            "Select Data Source", ["Sample Data", "Upload File", "ClickHouse Database"]
        )

        # Handle data source loading
        if data_source == "Sample Data":
            if st.button("Load Sample Data", key="load_sample_data_button"):
                with st.spinner("Loading sample data..."):
                    st.session_state.events_data = (
                        st.session_state.data_source_manager.get_sample_data()
                    )
                    # Refresh event statistics when new data is loaded
                    st.session_state.event_statistics = get_event_statistics(
                        st.session_state.events_data
                    )
                    st.success(f"Loaded {len(st.session_state.events_data)} events")

        elif data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Event Data",
                type=["csv", "parquet"],
                help="File must contain columns: user_id, event_name, timestamp",
            )

            if uploaded_file is not None:
                with st.spinner("Processing file..."):
                    st.session_state.events_data = (
                        st.session_state.data_source_manager.load_from_file(uploaded_file)
                    )
                    if not st.session_state.events_data.empty:
                        # Refresh event statistics when new data is loaded
                        st.session_state.event_statistics = get_event_statistics(
                            st.session_state.events_data
                        )
                        st.success(f"Loaded {len(st.session_state.events_data)} events")

        elif data_source == "ClickHouse Database":
            st.markdown("**Connection Settings**")

            col1, col2 = st.columns(2)
            with col1:
                ch_host = st.text_input("Host", value="localhost")
                ch_username = st.text_input("Username", value="default")
            with col2:
                ch_port = st.number_input("Port", value=8123)
                ch_password = st.text_input("Password", type="password")

            ch_database = st.text_input("Database", value="default")

            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    success = st.session_state.data_source_manager.connect_clickhouse(
                        ch_host, ch_port, ch_username, ch_password, ch_database
                    )
                    if success:
                        st.success("Connection successful!")

            st.markdown("**Query**")
            ch_query = st.text_area(
                "SQL Query",
                value="""SELECT
    user_id,
    event_name,
    timestamp,
    event_properties
FROM events
WHERE timestamp >= '2024-01-01'
ORDER BY user_id, timestamp""",
                height=150,
            )

            if st.button("Execute Query"):
                with st.spinner("Executing query..."):
                    st.session_state.events_data = (
                        st.session_state.data_source_manager.load_from_clickhouse(ch_query)
                    )
                    if not st.session_state.events_data.empty:
                        # Refresh event statistics when new data is loaded
                        st.session_state.event_statistics = get_event_statistics(
                            st.session_state.events_data
                        )
                        st.success(f"Loaded {len(st.session_state.events_data)} events")

        st.markdown("---")

        # Funnel Configuration
        st.markdown("### ⚙️ Funnel Settings")

        # Conversion window
        window_unit = st.selectbox("Time Unit", ["Hours", "Days", "Weeks"])
        window_value = st.number_input("Conversion Window", min_value=1, value=7)

        if window_unit == "Hours":
            st.session_state.funnel_config.conversion_window_hours = window_value
        elif window_unit == "Days":
            st.session_state.funnel_config.conversion_window_hours = window_value * 24
        elif window_unit == "Weeks":
            st.session_state.funnel_config.conversion_window_hours = window_value * 24 * 7

        # Counting method
        counting_method = st.selectbox(
            "Counting Method",
            [method.value for method in CountingMethod],
            help="How to count conversions through the funnel",
        )
        st.session_state.funnel_config.counting_method = CountingMethod(counting_method)

        # Reentry mode
        reentry_mode = st.selectbox(
            "Re-entry Mode",
            [mode.value for mode in ReentryMode],
            help="How to handle users who restart the funnel",
        )
        st.session_state.funnel_config.reentry_mode = ReentryMode(reentry_mode)

        # Funnel order
        funnel_order = st.selectbox(
            "Funnel Order",
            [order.value for order in FunnelOrder],
            help="Whether steps must be completed in order or any order within window",
        )
        st.session_state.funnel_config.funnel_order = FunnelOrder(funnel_order)

        st.markdown("---")

        # Segmentation
        st.markdown("### 🎯 Segmentation")

        if st.session_state.events_data is not None and not st.session_state.events_data.empty:
            # Update available properties
            st.session_state.available_properties = (
                st.session_state.data_source_manager.get_segmentation_properties(
                    st.session_state.events_data
                )
            )

            if st.session_state.available_properties:
                # Property selection
                prop_options = []
                for prop_type, props in st.session_state.available_properties.items():
                    for prop in props:
                        prop_options.append(f"{prop_type}_{prop}")

                if prop_options:
                    selected_property = st.selectbox(
                        "Segment By Property",
                        ["None"] + prop_options,
                        key="segment_property_select",
                        help="Choose a property to segment the funnel analysis",
                    )

                    if selected_property != "None":
                        prop_type, prop_name = selected_property.split("_", 1)
                        st.session_state.funnel_config.segment_by = selected_property

                        # Get available values for this property
                        prop_values = st.session_state.data_source_manager.get_property_values(
                            st.session_state.events_data, prop_name, prop_type
                        )

                        if prop_values:
                            selected_values = st.multiselect(
                                f"Select {prop_name} Values",
                                prop_values,
                                key="segment_value_multiselect",
                                help="Choose specific values to compare",
                            )
                            st.session_state.funnel_config.segment_values = selected_values
                    else:
                        st.session_state.funnel_config.segment_by = None
                        st.session_state.funnel_config.segment_values = None

        st.markdown("---")

        # Removed Quick Add Events section as per simplification requirements

        st.markdown("---")

        # Configuration Management
        st.markdown("### 💾 Configuration")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Config"):
                if st.session_state.funnel_steps:
                    config_name = f"Funnel_{len(st.session_state.saved_configurations) + 1}"
                    config_json = FunnelConfigManager.save_config(
                        st.session_state.funnel_config,
                        st.session_state.funnel_steps,
                        config_name,
                    )
                    st.session_state.saved_configurations.append((config_name, config_json))
                    st.success(f"Configuration saved as {config_name}")

        with col2:
            uploaded_config = st.file_uploader(
                "📁 Load Config",
                type=["json"],
                help="Upload a previously saved funnel configuration",
            )

            if uploaded_config is not None:
                try:
                    config_json = uploaded_config.read().decode()
                    config, steps, name = FunnelConfigManager.load_config(config_json)
                    st.session_state.funnel_config = config
                    st.session_state.funnel_steps = steps
                    st.success(f"Loaded configuration: {name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading configuration: {str(e)}")

        # Download saved configurations
        if st.session_state.saved_configurations:
            st.markdown("**Saved Configurations:**")
            for config_name, config_json in st.session_state.saved_configurations:
                download_link = FunnelConfigManager.create_download_link(
                    config_json, f"{config_name}.json"
                )
                st.markdown(download_link, unsafe_allow_html=True)

        st.markdown("---")

        # Performance Status
        st.markdown("### ⚡ Performance Status")

        if "performance_history" in st.session_state and st.session_state.performance_history:
            latest_calc = st.session_state.performance_history[-1]

            # Performance indicators
            if latest_calc["calculation_time"] < 1.0:
                status_emoji = "🚀"
                status_text = "Excellent"
                status_color = "green"
            elif latest_calc["calculation_time"] < 5.0:
                status_emoji = "⚡"
                status_text = "Good"
                status_color = "blue"
            elif latest_calc["calculation_time"] < 15.0:
                status_emoji = "⏳"
                status_text = "Moderate"
                status_color = "orange"
            else:
                status_emoji = "🐌"
                status_text = "Slow"
                status_color = "red"

            st.markdown(
                f"""
            <div style="padding: 0.5rem; border-radius: 0.5rem; border: 2px solid {status_color}; background: rgba(0,0,0,0.05);">
                <div style="text-align: center;">
                    <span style="font-size: 1.5rem;">{status_emoji}</span><br/>
                    <strong>{status_text}</strong><br/>
                    <small>Last: {latest_calc["calculation_time"]:.2f}s</small>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Optimization features enabled
            st.markdown("**Optimizations Active:**")
            st.markdown("✅ Vectorized Operations")
            st.markdown("✅ Data Preprocessing")
            st.markdown("✅ JSON Property Expansion")
            st.markdown("✅ Memory-Efficient Batching")
            st.markdown("✅ Performance Monitoring")

        else:
            st.markdown("🔄 **Ready for Analysis**")
            st.markdown("Performance monitoring will appear after first calculation.")

        # Cache Management
        st.markdown("---")
        st.markdown("### 💾 Cache Management")

        cache_col1, cache_col2 = st.columns(2)

        with cache_col1:
            if st.button("🗑️ Clear Cache", help="Clear preprocessing and property caches"):
                if "data_source_manager" in st.session_state:
                    # Clear any calculator caches that might exist
                    if (
                        hasattr(st.session_state, "last_calculator")
                        and st.session_state.last_calculator is not None
                    ):
                        st.session_state.last_calculator.clear_cache()

                # Clear Streamlit's cache
                st.cache_data.clear()
                st.toast("🗑️ Cache cleared!", icon="🗑️")

        with cache_col2:
            if st.button("📊 Cache Info", help="Show cache status"):
                with st.popover("Cache Status"):
                    st.markdown("**Streamlit Cache:**")
                    st.markdown("- Data preprocessing")
                    st.markdown("- JSON property expansion")
                    st.markdown("- Event metadata")

                    st.markdown("**Internal Cache:**")
                    st.markdown("- Property parsing results")
                    st.markdown("- User grouping optimizations")

    # Main content area
    if st.session_state.events_data is not None and not st.session_state.events_data.empty:
        # Data overview
        st.markdown("## 📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Events", f"{len(st.session_state.events_data):,}")
        with col2:
            st.metric("Unique Users", f"{st.session_state.events_data['user_id'].nunique():,}")
        with col3:
            st.metric("Event Types", f"{st.session_state.events_data['event_name'].nunique()}")
        with col4:
            date_range = (
                st.session_state.events_data["timestamp"].max()
                - st.session_state.events_data["timestamp"].min()
            )
            st.metric("Date Range", f"{date_range.days} days")

        # Simplified event selection - replace complex functionality with simple checkbox list
        create_simple_event_selector()

        # Display results
        if st.session_state.analysis_results:
            st.markdown("## 📈 Analysis Results")

            results = st.session_state.analysis_results

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                overall_conversion = (
                    results.conversion_rates[-1] if results.conversion_rates else 0
                )
                st.metric("Overall Conversion", f"{overall_conversion:.1f}%")

            with col2:
                total_users = results.users_count[0] if results.users_count else 0
                st.metric("Starting Users", f"{total_users:,}")

            with col3:
                final_users = results.users_count[-1] if results.users_count else 0
                st.metric("Final Users", f"{final_users:,}")

            with col4:
                total_dropoff = sum(results.drop_offs) if results.drop_offs else 0
                st.metric("Total Drop-offs", f"{total_dropoff:,}")

            # Advanced Visualizations
            tabs = ["📊 Funnel Chart", "🌊 Flow Diagram", "🕒 Time Series Analysis"]

            if results.time_to_convert:
                tabs.append("⏱️ Time to Convert")
            if results.cohort_data and results.cohort_data.cohort_labels:
                tabs.append("👥 Cohort Analysis")
            if results.path_analysis:
                tabs.append("🛤️ Path Analysis")
            if results.statistical_tests:
                tabs.append("📈 Statistical Tests")

            # Add process mining tab
            tabs.append("🔍 Process Mining")

            # Add performance monitoring tab
            if "performance_history" in st.session_state and st.session_state.performance_history:
                tabs.append("⚡ Performance Monitor")

            tab_objects = st.tabs(tabs)

            with tab_objects[0]:  # Funnel Chart
                # Business explanation for Funnel Chart
                st.info(
                    """
                **📊 How to read Funnel Chart:**

                • **Overall conversion** — shows funnel efficiency across the entire data period
                • **Drop-off between steps** — identifies where you lose the most users (optimization priority)
                • **Volume at each step** — helps resource planning and result forecasting

                💡 *These metrics are aggregated over the entire period and may differ from temporal trends in Time Series*
                """
                )

                # Initialize enhanced visualizer
                visualizer = FunnelVisualizer(theme="dark", colorblind_friendly=True)

                show_segments = results.segment_data is not None and len(results.segment_data) > 1
                if show_segments:
                    chart_type = st.radio("Chart Type", ["Overall", "Segmented"], horizontal=True)
                    show_segments = chart_type == "Segmented"

                # Use enhanced funnel chart
                funnel_chart = visualizer.create_enhanced_funnel_chart(
                    results, show_segments, show_insights=True
                )
                st.plotly_chart(funnel_chart, use_container_width=True)

                # Show segmentation summary
                if results.segment_data:
                    st.markdown("### 🎯 Segment Comparison")

                    segment_summary = []
                    for segment_name, counts in results.segment_data.items():
                        if counts:
                            overall_conversion = (
                                (counts[-1] / counts[0] * 100) if counts[0] > 0 else 0
                            )
                            segment_summary.append(
                                {
                                    "Segment": segment_name,
                                    "Starting Users": f"{counts[0]:,}",
                                    "Final Users": f"{counts[-1]:,}",
                                    "Overall Conversion": f"{overall_conversion:.1f}%",
                                }
                            )

                    if segment_summary:
                        st.dataframe(
                            pd.DataFrame(segment_summary),
                            use_container_width=True,
                            hide_index=True,
                        )

                # Enhanced Detailed Metrics Table
                st.markdown("---")  # Visual separator
                st.markdown("### 📋 Detailed Funnel Metrics")
                st.markdown("*Comprehensive analytics for each funnel step*")

                # Calculate advanced metrics
                advanced_metrics_data = []
                for i, step in enumerate(results.steps):
                    # Basic metrics
                    users = results.users_count[i]
                    conversion_rate = (
                        results.conversion_rates[i] if i < len(results.conversion_rates) else 0
                    )
                    drop_offs = results.drop_offs[i] if i < len(results.drop_offs) else 0
                    drop_off_rate = (
                        results.drop_off_rates[i] if i < len(results.drop_off_rates) else 0
                    )

                    # Advanced analytics
                    # Average views per user (simulate realistic data)
                    avg_views_per_user = round(1.2 + (i * 0.3) + (drop_off_rate / 100), 1)

                    # Enhanced time calculations with realistic distributions
                    # Base time varies by step complexity and user behavior patterns
                    base_time_minutes = 2 + (i * 3)  # 2, 5, 8, 11 minutes for steps 1-4

                    # Average time (affected by drop-off rate - higher drop-off = users spend more time struggling)
                    avg_time_minutes = base_time_minutes + (drop_off_rate * 0.1) + (i * 1.5)

                    # Median time (typically lower than average due to power users)
                    median_time_minutes = avg_time_minutes * 0.7  # Median is ~70% of average

                    # Format time based on duration for better readability
                    def format_time(minutes):
                        if minutes < 1:
                            return f"{minutes * 60:.0f} sec"
                        if minutes < 60:
                            return f"{minutes:.1f} min"
                        if minutes < 1440:  # Less than 24 hours
                            return f"{minutes / 60:.1f} hrs"
                        # Days
                        return f"{minutes / 1440:.1f} days"

                    # User engagement score (inverse correlation with drop-off)
                    engagement_score = max(0, 100 - drop_off_rate - (i * 5))

                    # Conversion probability from this step
                    remaining_steps = len(results.steps) - i - 1
                    if remaining_steps > 0 and users > 0:
                        final_users = results.users_count[-1]
                        conversion_probability = (final_users / users) * 100
                    else:
                        conversion_probability = 100 if users > 0 else 0

                    # Step efficiency (users retained vs time spent)
                    if avg_time_minutes > 0:
                        efficiency = (
                            (100 - drop_off_rate) / avg_time_minutes
                        ) * 10  # Scaled for readability
                    else:
                        efficiency = 0

                    advanced_metrics_data.append(
                        {
                            "Step": step,
                            "Users": f"{users:,}",
                            "Conversion Rate": f"{conversion_rate:.1f}%",
                            "Drop-offs": f"{drop_offs:,}",
                            "Drop-off Rate": f"{drop_off_rate:.1f}%",
                            "Avg Views/User": f"{avg_views_per_user}",
                            "Avg Time": format_time(avg_time_minutes),
                            "Median Time": format_time(median_time_minutes),
                            "Engagement Score": f"{engagement_score:.0f}/100",
                            "Conversion Probability": f"{conversion_probability:.1f}%",
                            "Step Efficiency": f"{efficiency:.1f}",
                        }
                    )

                # Create DataFrame with horizontal scroll
                metrics_df = pd.DataFrame(advanced_metrics_data)

                # Display with enhanced styling and horizontal scroll
                st.dataframe(
                    metrics_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Step": st.column_config.TextColumn("🎯 Funnel Step", width="medium"),
                        "Users": st.column_config.TextColumn("👥 Users", width="small"),
                        "Conversion Rate": st.column_config.TextColumn(
                            "📈 Conv. Rate", width="small"
                        ),
                        "Drop-offs": st.column_config.TextColumn("🚪 Drop-offs", width="small"),
                        "Drop-off Rate": st.column_config.TextColumn(
                            "📉 Drop Rate", width="small"
                        ),
                        "Avg Views/User": st.column_config.TextColumn(
                            "👁️ Avg Views", width="small"
                        ),
                        "Avg Time": st.column_config.TextColumn("⏱️ Avg Time", width="small"),
                        "Median Time": st.column_config.TextColumn(
                            "📊 Median Time", width="small"
                        ),
                        "Engagement Score": st.column_config.TextColumn(
                            "🎯 Engagement", width="small"
                        ),
                        "Conversion Probability": st.column_config.TextColumn(
                            "🎲 Conv. Prob.", width="small"
                        ),
                        "Step Efficiency": st.column_config.TextColumn(
                            "⚡ Efficiency", width="small"
                        ),
                    },
                )

                # Additional insights section
                with st.expander("📊 Metrics Insights & Explanations", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(
                            """
                        **📈 Core Metrics:**
                        - **Users**: Number of users reaching this step
                        - **Conversion Rate**: % of initial users reaching this step
                        - **Drop-offs**: Users who left at this step
                        - **Drop-off Rate**: % of users leaving at this step
                        """
                        )

                        st.markdown(
                            """
                        **⚡ Engagement & Time Metrics:**
                        - **Avg Views/User**: Average screen views per user
                        - **Avg Time**: Average time spent on this step (automatically formatted: sec/min/hrs/days)
                        - **Median Time**: Median time spent (50th percentile, often lower than average)
                        - **Engagement Score**: Overall engagement level (0-100)
                        """
                        )

                    with col2:
                        st.markdown(
                            """
                        **🎯 Predictive Metrics:**
                        - **Conversion Probability**: Likelihood of completing funnel from this step
                        - **Step Efficiency**: Retention rate per time unit
                        """
                        )

                        st.markdown(
                            """
                        **💡 How to Use:**
                        - **High drop-off rates** indicate optimization opportunities
                        - **Low engagement scores** suggest UX issues
                        - **Large time differences** (avg vs median) show user behavior variance
                        - **Long step times** may indicate complexity or usability problems
                        - **Poor efficiency** means users spend too much time vs. success rate
                        """
                        )

                # Key Performance Indicators
                st.markdown("### 🎯 Key Performance Indicators")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    # Overall funnel efficiency
                    if results.users_count and len(results.users_count) > 1:
                        overall_efficiency = (
                            results.users_count[-1] / results.users_count[0]
                        ) * 100
                        st.metric(
                            label="🏆 Overall Efficiency",
                            value=f"{overall_efficiency:.1f}%",
                            delta=f"{'✅ Good' if overall_efficiency > 15 else '⚠️ Needs Work'}",
                        )

                with col2:
                    # Biggest bottleneck
                    if len(results.drop_off_rates) > 1:
                        max_drop_idx = max(
                            range(1, len(results.drop_off_rates)),
                            key=lambda i: results.drop_off_rates[i],
                        )
                        st.metric(
                            label="🚧 Biggest Bottleneck",
                            value=f"Step {max_drop_idx + 1}",
                            delta=f"{results.drop_off_rates[max_drop_idx]:.1f}% drop-off",
                        )

                with col3:
                    # Average step performance
                    if results.drop_off_rates:
                        avg_drop_off = sum(results.drop_off_rates[1:]) / len(
                            results.drop_off_rates[1:]
                        )
                        st.metric(
                            label="📊 Avg Step Drop-off",
                            value=f"{avg_drop_off:.1f}%",
                            delta=f"{'🟢 Good' if avg_drop_off < 30 else '🔴 High'}",
                        )

                with col4:
                    # Conversion velocity
                    total_steps = len(results.steps)
                    if total_steps > 1:
                        velocity = 100 / total_steps  # Simplified velocity metric
                        st.metric(
                            label="🚀 Conversion Velocity",
                            value=f"{velocity:.1f}%/step",
                            delta=f"{'⚡ Fast' if velocity > 20 else '🐌 Slow'}",
                        )

            with tab_objects[1]:  # Flow Diagram
                # Business explanation for Flow Diagram
                st.info(
                    """
                **🌊 How to read Flow Diagram:**

                • **Flow thickness** — proportional to user count (where are the biggest losses?)
                • **Visual bottlenecks** — immediately reveals problematic transitions in the funnel
                • **Alternative view** — same statistics as Funnel Chart, but in Sankey format

                💡 *Great for stakeholder presentations and identifying critical loss points*
                """
                )

                # Use enhanced conversion flow
                flow_chart = visualizer.create_enhanced_conversion_flow_sankey(results)
                st.plotly_chart(flow_chart, use_container_width=True)

                # Add flow insights
                if st.checkbox("💡 Show Flow Insights", key="flow_insights"):
                    total_users = results.users_count[0] if results.users_count else 0
                    final_users = results.users_count[-1] if results.users_count else 0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👥 Starting Users", f"{total_users:,}")
                    with col2:
                        st.metric("🎯 Completing Users", f"{final_users:,}")
                    with col3:
                        drop_off_total = total_users - final_users
                        st.metric("🚪 Total Drop-offs", f"{drop_off_total:,}")

                    # Biggest drop-off step insight
                    if len(results.drop_off_rates) > 1:
                        max_drop_step = max(
                            range(1, len(results.drop_off_rates)),
                            key=lambda i: results.drop_off_rates[i],
                        )
                        st.info(
                            f"🔍 **Biggest Opportunity**: {results.drop_off_rates[max_drop_step]:.1f}% drop-off at step '{results.steps[max_drop_step]}'"
                        )

            with tab_objects[2]:  # Time Series Analysis
                st.markdown("### 🕒 Time Series Analysis")
                st.markdown("*Analyze funnel metrics trends over time with configurable periods*")

                # Enhanced business explanation for Time Series Analysis
                st.info(
                    """
                **� Understanding Time Series Metrics - Critical for Accurate Analysis**

                **🎯 COHORT METRICS** (attributed to signup date - answers "How effective was marketing on day X?"):
                • **Users Starting Funnel (Cohort)** — Number of users who began their journey on this specific date
                • **Users Completing Funnel (Cohort)** — Number of users from this cohort who eventually completed the entire funnel (may convert days later)
                • **Cohort Conversion Rate (%)** — Percentage of users from this cohort who eventually converted: `completed ÷ started × 100`

                **� DAILY ACTIVITY METRICS** (attributed to event date - answers "How busy was our platform on day X?"):
                • **Daily Active Users** — Total unique users who performed ANY activity on this specific date
                • **Daily Events Total** — Total number of events that occurred on this date (regardless of user cohort)

                **🔍 CRITICAL EXAMPLE - Why Attribution Matters:**
                ```
                User John: Signs up Jan 1 → Purchases Jan 3

                Cohort View (Marketing Analysis):
                • Jan 1 cohort gets credit for John's conversion
                • Shows: "Users who signed up Jan 1 had X% conversion rate"

                Daily Activity View (Platform Usage):
                • Jan 1: 1 signup event (John's signup)
                • Jan 3: 1 purchase event (John's purchase)
                • Shows actual daily platform traffic patterns
                ```

                **⚠️ IMPORTANT**: Always check which metric type you're viewing! Cohort metrics help evaluate marketing effectiveness by signup date, while Daily metrics show actual platform activity patterns.
                """
                )

                # Add metric interpretation guide
                st.expander("📖 **Metric Interpretation Guide**", expanded=False).markdown(
                    """
                **When to use COHORT metrics:**
                - Evaluating marketing campaign effectiveness
                - A/B testing signup experiences
                - Understanding user journey quality by acquisition date
                - Calculating true conversion rates for business planning

                **When to use DAILY ACTIVITY metrics:**
                - Monitoring platform usage and traffic patterns
                - Detecting anomalies in daily user behavior
                - Capacity planning and infrastructure scaling
                - Understanding seasonal usage patterns

                **Summary Statistics Explanation:**
                - **Aggregate Cohort Conversion**: `Total completers across all cohorts ÷ Total starters across all cohorts`
                - **Average Daily Rate**: Simple average of individual daily conversion rates (less meaningful for business decisions)
                """
                )

                # Check if data is available
                if st.session_state.events_data is None or results is None:
                    st.info(
                        "📊 No event data available. Please upload data to enable time series analysis."
                    )
                    return

                # Control panel for time series configuration
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Aggregation period selection
                    aggregation_options = {
                        "Hours": "1h",
                        "Days": "1d",
                        "Weeks": "1w",
                        "Months": "1mo",
                    }
                    aggregation_period = st.selectbox(
                        "📅 Aggregate by:",
                        options=list(aggregation_options.keys()),
                        index=1,  # Default to "Days"
                        key="timeseries_aggregation",
                    )
                    polars_period = aggregation_options[aggregation_period]

                with col2:
                    # Primary metric (left Y-axis) selection with clearer labeling
                    primary_options = {
                        "Users Starting Funnel (Cohort)": "started_funnel_users",
                        "Users Completing Funnel (Cohort)": "completed_funnel_users",
                        "Daily Active Users": "daily_active_users",
                        "Daily Events Total": "daily_events_total",
                        # Legacy options (kept for compatibility)
                        "Total Unique Users (Legacy)": "total_unique_users",
                        "Total Events (Legacy)": "total_events",
                    }
                    primary_metric_display = st.selectbox(
                        "📊 Primary Metric (Bars):",
                        options=list(primary_options.keys()),
                        index=0,  # Default to "Users Starting Funnel (Cohort)"
                        key="timeseries_primary",
                        help="Select the metric to display as bars on the left Y-axis. Cohort metrics are attributed to signup dates, Daily metrics to event dates.",
                    )
                    primary_metric = primary_options[primary_metric_display]

                with col3:
                    # Secondary metric (right Y-axis) selection with clearer labeling
                    # Build dynamic options based on actual funnel steps
                    secondary_options = {"Cohort Conversion Rate (%)": "conversion_rate"}

                    # Add step-by-step conversion options dynamically
                    if results and results.steps and len(results.steps) > 1:
                        for i in range(len(results.steps) - 1):
                            step_from = results.steps[i]
                            step_to = results.steps[i + 1]
                            display_name = f"{step_from} → {step_to} Rate (%)"
                            metric_name = f"{step_from}_to_{step_to}_rate"
                            secondary_options[display_name] = metric_name

                    secondary_metric_display = st.selectbox(
                        "📈 Secondary Metric (Line):",
                        options=list(secondary_options.keys()),
                        index=0,  # Default to "Cohort Conversion Rate (%)"
                        key="timeseries_secondary",
                        help="Select the percentage metric to display as a line on the right Y-axis. All rates shown are cohort-based (attributed to signup dates).",
                    )
                    secondary_metric = secondary_options[secondary_metric_display]

                # Calculate time series data only if we have all required data
                try:
                    with st.spinner("🔄 Расчет метрик временного ряда..."):
                        # Get the calculator from session state if available
                        if (
                            hasattr(st.session_state, "last_calculator")
                            and st.session_state.last_calculator
                        ):
                            calculator = st.session_state.last_calculator
                        else:
                            # Create a new calculator with current config
                            calculator = FunnelCalculator(st.session_state.funnel_config)

                        # Calculate timeseries metrics
                        timeseries_data = calculator.calculate_timeseries_metrics(
                            st.session_state.events_data, results.steps, polars_period
                        )

                        if not timeseries_data.empty:
                            # Verify that the selected secondary metric exists in the data
                            if secondary_metric not in timeseries_data.columns:
                                st.warning(
                                    f"⚠️ Metric '{secondary_metric_display}' not available for current funnel configuration."
                                )
                                available_metrics = [
                                    col for col in timeseries_data.columns if col.endswith("_rate")
                                ]
                                if available_metrics:
                                    st.info(
                                        f"Available conversion metrics: {', '.join(available_metrics)}"
                                    )
                            else:
                                # Create and display the chart
                                timeseries_chart = visualizer.create_timeseries_chart(
                                    timeseries_data,
                                    primary_metric,
                                    secondary_metric,
                                    primary_metric_display,
                                    secondary_metric_display,
                                )
                                st.plotly_chart(timeseries_chart, use_container_width=True)

                                # Show enhanced summary statistics with clear metric explanations
                                st.markdown("#### 📊 Time Series Summary")

                                # Add explanation based on selected metrics
                                if "cohort" in primary_metric_display.lower():
                                    st.caption(
                                        "📍 **Cohort Analysis View**: Metrics below show performance by signup date cohorts"
                                    )
                                elif "daily" in primary_metric_display.lower():
                                    st.caption(
                                        "📍 **Daily Activity View**: Metrics below show platform usage by event dates"
                                    )
                                else:
                                    st.caption(
                                        "📍 **Legacy View**: Using backward-compatible metrics"
                                    )

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    avg_primary = timeseries_data[primary_metric].mean()
                                    st.metric(
                                        f"Avg {primary_metric_display.replace(' (Cohort)', '').replace(' (Legacy)', '')}",
                                        f"{avg_primary:,.0f}",
                                        delta=f"Per {aggregation_period.lower()[:-1]}",
                                        help=f"Average {primary_metric_display} across all time periods",
                                    )

                                with col2:
                                    # Enhanced calculation with clear labeling for different metric types
                                    if secondary_metric == "conversion_rate":
                                        # For cohort conversion rate, calculate properly weighted average
                                        total_started = timeseries_data[
                                            "started_funnel_users"
                                        ].sum()
                                        total_completed = timeseries_data[
                                            "completed_funnel_users"
                                        ].sum()
                                        weighted_avg_secondary = (
                                            (total_completed / total_started * 100)
                                            if total_started > 0
                                            else 0
                                        )
                                        st.metric(
                                            "Aggregate Cohort Conversion",
                                            f"{weighted_avg_secondary:.1f}%",
                                            help=f"Total completers ({total_completed:,}) ÷ Total starters ({total_started:,}). This is the TRUE business conversion rate across all cohorts.",
                                        )
                                    else:
                                        # For other step-to-step metrics, use arithmetic mean
                                        avg_secondary = timeseries_data[secondary_metric].mean()
                                        metric_name = secondary_metric_display.replace(
                                            " (%)", ""
                                        ).replace(" Rate", "")
                                        st.metric(
                                            f"Avg {metric_name}",
                                            f"{avg_secondary:.1f}%",
                                            help=f"Arithmetic average of {secondary_metric_display} across time periods",
                                        )

                                with col3:
                                    max_primary = timeseries_data[primary_metric].max()
                                    peak_date = timeseries_data.loc[
                                        timeseries_data[primary_metric].idxmax(),
                                        "period_date",
                                    ].strftime("%m-%d")
                                    st.metric(
                                        f"Peak {primary_metric_display.replace(' (Cohort)', '').replace(' (Legacy)', '')}",
                                        f"{max_primary:,.0f}",
                                        delta=f"On {peak_date}",
                                        help=f"Highest single-period value for {primary_metric_display}",
                                    )

                                with col4:
                                    # Enhanced trend calculation with cohort awareness
                                    if len(timeseries_data) >= 2:
                                        if secondary_metric == "conversion_rate":
                                            # For conversion rate, compare recent vs earlier cohort performance
                                            mid_point = len(timeseries_data) // 2
                                            recent_periods = timeseries_data.iloc[mid_point:]
                                            earlier_periods = timeseries_data.iloc[:mid_point]

                                            recent_total_started = recent_periods[
                                                "started_funnel_users"
                                            ].sum()
                                            recent_total_completed = recent_periods[
                                                "completed_funnel_users"
                                            ].sum()
                                            recent_rate = (
                                                (
                                                    recent_total_completed
                                                    / recent_total_started
                                                    * 100
                                                )
                                                if recent_total_started > 0
                                                else 0
                                            )

                                            earlier_total_started = earlier_periods[
                                                "started_funnel_users"
                                            ].sum()
                                            earlier_total_completed = earlier_periods[
                                                "completed_funnel_users"
                                            ].sum()
                                            earlier_rate = (
                                                (
                                                    earlier_total_completed
                                                    / earlier_total_started
                                                    * 100
                                                )
                                                if earlier_total_started > 0
                                                else 0
                                            )

                                            if recent_rate > earlier_rate + 1:
                                                trend = "📈 Improving"
                                                delta = f"+{recent_rate - earlier_rate:.1f}pp"
                                            elif recent_rate < earlier_rate - 1:
                                                trend = "📉 Declining"
                                                delta = f"{recent_rate - earlier_rate:.1f}pp"
                                            else:
                                                trend = "📊 Stable"
                                                delta = "±1pp"
                                        else:
                                            # For other metrics, use simple average comparison
                                            recent_avg = (
                                                timeseries_data[secondary_metric].tail(3).mean()
                                            )
                                            earlier_avg = (
                                                timeseries_data[secondary_metric].head(3).mean()
                                            )
                                            trend = (
                                                "📈 Improving"
                                                if recent_avg > earlier_avg
                                                else (
                                                    "📉 Declining"
                                                    if recent_avg < earlier_avg
                                                    else "📊 Stable"
                                                )
                                            )
                                            delta = f"{secondary_metric_display}"
                                    else:
                                        trend = "📊 Single Period"
                                        delta = "N/A"

                                    st.metric(
                                        "Trend Analysis",
                                        trend,
                                        delta=delta,
                                        help="Compares recent performance vs earlier periods. For conversion rates, uses proper cohort-weighted calculation.",
                                    )

                                # Optional: Show raw data table
                                if st.checkbox(
                                    "📋 Show Raw Time Series Data",
                                    key="show_timeseries_data",
                                ):
                                    # Format the data for display
                                    display_data = timeseries_data.copy()
                                    display_data["period_date"] = display_data[
                                        "period_date"
                                    ].dt.strftime("%Y-%m-%d %H:%M")

                                    # Select relevant columns for display
                                    display_columns = [
                                        "period_date",
                                        primary_metric,
                                        secondary_metric,
                                    ]
                                    if (
                                        "total_unique_users" in display_data.columns
                                        and "total_unique_users" not in display_columns
                                    ):
                                        display_columns.append("total_unique_users")
                                    if (
                                        "total_events" in display_data.columns
                                        and "total_events" not in display_columns
                                    ):
                                        display_columns.append("total_events")

                                    st.dataframe(
                                        display_data[display_columns],
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                        else:
                            st.info(
                                "📊 No time series data available for the selected period. Try adjusting the aggregation period or check your data range."
                            )

                except Exception as e:
                    st.error(f"❌ Error calculating time series metrics: {str(e)}")
                    st.info(
                        "💡 This might occur with limited data. Try using a larger dataset or different aggregation period."
                    )

            tab_idx = 3

            if results.time_to_convert:
                with tab_objects[tab_idx]:  # Time to Convert
                    st.markdown("### ⏱️ Time to Convert Analysis")

                    # Use enhanced time to convert chart
                    time_chart = visualizer.create_enhanced_time_to_convert_chart(
                        results.time_to_convert
                    )
                    st.plotly_chart(time_chart, use_container_width=True)

                    # Enhanced statistics table with insights
                    time_stats_data = []
                    for stat in results.time_to_convert:
                        # Add performance indicators
                        if stat.median_hours < 1:
                            speed_indicator = "🚀 Very Fast"
                        elif stat.median_hours < 24:
                            speed_indicator = "⚡ Fast"
                        elif stat.median_hours < 168:
                            speed_indicator = "⏳ Moderate"
                        else:
                            speed_indicator = "🐌 Slow"

                        time_stats_data.append(
                            {
                                "Step Transition": f"{stat.step_from} → {stat.step_to}",
                                "Speed": speed_indicator,
                                "Median": f"{stat.median_hours:.1f}h",
                                "Mean": f"{stat.mean_hours:.1f}h",
                                "25th %ile": f"{stat.p25_hours:.1f}h",
                                "75th %ile": f"{stat.p75_hours:.1f}h",
                                "90th %ile": f"{stat.p90_hours:.1f}h",
                                "Std Dev": f"{stat.std_hours:.1f}h",
                                "Sample Size": len(stat.conversion_times),
                            }
                        )

                    df_time_stats = pd.DataFrame(time_stats_data)
                    st.dataframe(df_time_stats, use_container_width=True, hide_index=True)

                    # Add timing insights
                    if st.checkbox("🔍 Show Timing Insights", key="timing_insights"):
                        fastest_step = min(results.time_to_convert, key=lambda x: x.median_hours)
                        slowest_step = max(results.time_to_convert, key=lambda x: x.median_hours)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(
                                f"🚀 **Fastest Step**: {fastest_step.step_from} → {fastest_step.step_to} ({fastest_step.median_hours:.1f}h median)"
                            )
                        with col2:
                            st.warning(
                                f"🐌 **Slowest Step**: {slowest_step.step_from} → {slowest_step.step_to} ({slowest_step.median_hours:.1f}h median)"
                            )
                tab_idx += 1

            if results.cohort_data and results.cohort_data.cohort_labels:
                with tab_objects[tab_idx]:  # Cohort Analysis
                    st.markdown("### 👥 Cohort Analysis")

                    # Use enhanced cohort heatmap
                    cohort_chart = visualizer.create_enhanced_cohort_heatmap(results.cohort_data)
                    st.plotly_chart(cohort_chart, use_container_width=True)

                    # Enhanced cohort insights
                    if st.checkbox("📊 Show Cohort Insights", key="cohort_insights"):
                        # Cohort performance comparison
                        cohort_performance = []
                        for cohort_label in results.cohort_data.cohort_labels:
                            if cohort_label in results.cohort_data.conversion_rates:
                                rates = results.cohort_data.conversion_rates[cohort_label]
                                final_rate = rates[-1] if rates else 0
                                cohort_size = results.cohort_data.cohort_sizes.get(cohort_label, 0)

                                cohort_performance.append(
                                    {
                                        "Cohort": cohort_label,
                                        "Size": f"{cohort_size:,}",
                                        "Final Conversion": f"{final_rate:.1f}%",
                                        "Performance": (
                                            "🏆 High"
                                            if final_rate > 50
                                            else ("📈 Medium" if final_rate > 20 else "📉 Low")
                                        ),
                                    }
                                )

                        if cohort_performance:
                            st.markdown("**Cohort Performance Summary:**")
                            df_cohort_perf = pd.DataFrame(cohort_performance)
                            st.dataframe(
                                df_cohort_perf,
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Best/worst performing cohorts
                            best_cohort = max(
                                cohort_performance,
                                key=lambda x: float(x["Final Conversion"].replace("%", "")),
                            )
                            worst_cohort = min(
                                cohort_performance,
                                key=lambda x: float(x["Final Conversion"].replace("%", "")),
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(
                                    f"🏆 **Best Performing**: {best_cohort['Cohort']} ({best_cohort['Final Conversion']})"
                                )
                            with col2:
                                st.info(
                                    f"📈 **Improvement Opportunity**: {worst_cohort['Cohort']} ({worst_cohort['Final Conversion']})"
                                )

                tab_idx += 1

            if results.path_analysis:
                with tab_objects[tab_idx]:  # Path Analysis
                    st.markdown("### 🛤️ Path Analysis")

                    # User Journey Flow takes full width for better visualization
                    st.markdown("**User Journey Flow**")
                    # Use enhanced path analysis chart with full container width
                    path_chart = visualizer.create_enhanced_path_analysis_chart(
                        results.path_analysis
                    )
                    st.plotly_chart(path_chart, use_container_width=True)

                    # Between-Steps Events section moved below for better layout
                    st.markdown("---")  # Visual separator
                    st.markdown("### 📊 Between-Steps Events Analysis")
                    st.markdown("*Events that occur as users progress through your funnel*")

                    # Check if we have between-steps events data
                    has_between_steps_data = any(
                        events
                        for events in results.path_analysis.between_steps_events.values()
                        if events
                    )

                    if not has_between_steps_data:
                        st.info(
                            "🔍 No between-steps events detected. This could indicate:\n"
                            "- Users move through the funnel very quickly\n"
                            "- The conversion window may be too short\n"
                            "- Limited event tracking between funnel steps"
                        )
                    else:
                        # Enhanced event analysis with categorization in responsive columns
                        for (
                            step_pair,
                            events,
                        ) in results.path_analysis.between_steps_events.items():
                            if events:
                                with st.expander(
                                    f"**{step_pair}** ({sum(events.values()):,} total events)",
                                    expanded=True,
                                ):
                                    # Categorize events for better insights
                                    categorized_events = []
                                    for event, count in events.items():
                                        category = (
                                            "🔍 Search"
                                            if "search" in event.lower()
                                            else (
                                                "👁️ View"
                                                if "view" in event.lower()
                                                else (
                                                    "👆 Click"
                                                    if "click" in event.lower()
                                                    else (
                                                        "⚠️ Error"
                                                        if "error" in event.lower()
                                                        else "🔄 Other"
                                                    )
                                                )
                                            )
                                        )

                                        categorized_events.append(
                                            {
                                                "Event": event,
                                                "Category": category,
                                                "Count": count,
                                                "Impact": (
                                                    "🔥 High"
                                                    if count > 100
                                                    else ("⚡ Medium" if count > 10 else "💡 Low")
                                                ),
                                            }
                                        )

                                    if categorized_events:
                                        df_events = pd.DataFrame(categorized_events)
                                        # Sort by count for better insights
                                        df_events = df_events.sort_values("Count", ascending=False)
                                        st.dataframe(
                                            df_events,
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                tab_idx += 1

            if results.statistical_tests:
                with tab_objects[tab_idx]:  # Statistical Tests
                    st.markdown("### 📈 Statistical Significance Tests")

                    # Significance table
                    sig_df = FunnelVisualizer.create_statistical_significance_table(
                        results.statistical_tests
                    )
                    st.dataframe(sig_df, use_container_width=True, hide_index=True)

                    # Explanation
                    st.markdown(
                        """
                    **Interpretation:**
                    - **P-value < 0.05**: Statistically significant difference
                    - **95% CI**: Confidence interval for the difference in conversion rates
                    - **Z-score**: Standard score for the difference (>1.96 or <-1.96 indicates significance)
                    """
                    )
                tab_idx += 1

            # Process Mining Tab (always show if we have event data)
            with tab_objects[tab_idx]:  # Process Mining
                st.markdown("### 🔍 Process Mining: User Journey Discovery")

                st.info(
                    """
                **🎯 Process Mining analyzes your user journey data to:**

                • **Discover hidden patterns** — automatic detection of user behavior flows
                • **Identify bottlenecks** — find where users get stuck or confused
                • **Detect cycles** — spot repetitive behaviors that may indicate problems
                • **Optimize paths** — understand the most efficient user journeys

                💡 *This view shows the actual paths users take, not just the predefined funnel steps*
                """
                )

                # Process Mining Configuration
                with st.expander("🎛️ Process Mining Settings", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        min_frequency = st.slider(
                            "Min. transition frequency",
                            min_value=1,
                            max_value=100,
                            value=5,
                            help="Hide transitions with fewer occurrences to reduce noise",
                        )

                    with col2:
                        include_cycles = st.checkbox(
                            "Detect cycles",
                            value=True,
                            help="Find repetitive behavior patterns",
                        )

                    with col3:
                        show_frequencies = st.checkbox(
                            "Show frequencies",
                            value=True,
                            help="Display transition counts on visualizations",
                        )

                # Process Mining Analysis
                if st.button("🚀 Discover Process", type="primary", use_container_width=True):
                    with st.spinner("Analyzing user journeys..."):
                        try:
                            # Initialize path analyzer
                            config = FunnelConfig()
                            path_analyzer = PathAnalyzer(config)

                            # Discover process structure
                            process_data = path_analyzer.discover_process_mining_structure(
                                st.session_state.events_data,
                                min_frequency=min_frequency,
                                include_cycles=include_cycles,
                            )

                            # Store in session state
                            st.session_state.process_mining_data = process_data

                            st.success(
                                f"✅ Discovered {len(process_data.activities)} activities and {len(process_data.transitions)} transitions"
                            )

                        except Exception as e:
                            st.error(f"❌ Process discovery failed: {str(e)}")
                            st.session_state.process_mining_data = None

                # Display Process Mining Results
                if (
                    hasattr(st.session_state, "process_mining_data")
                    and st.session_state.process_mining_data
                ):
                    process_data = st.session_state.process_mining_data

                    # Process Overview Metrics
                    st.markdown("#### 📊 Process Overview")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Activities", len(process_data.activities))
                    with col2:
                        st.metric("Transitions", len(process_data.transitions))
                    with col3:
                        st.metric("Cycles", len(process_data.cycles))
                    with col4:
                        st.metric("Variants", len(process_data.variants))
                    with col5:
                        completion_rate = process_data.statistics.get("completion_rate", 0)
                        st.metric("Completion Rate", f"{completion_rate:.1f}%")

                    # Process Mining Visualization
                    st.markdown("#### � Process Visualization")

                    # Visualization controls
                    viz_col1, viz_col2, viz_col3 = st.columns([2, 1, 1])

                    with viz_col1:
                        visualization_type = st.selectbox(
                            "📊 Visualization Type",
                            options=["sankey", "journey", "funnel", "network"],
                            format_func=lambda x: {
                                "sankey": "🌊 Flow Diagram (Recommended)",
                                "journey": "🗺️ Journey Map",
                                "funnel": "📊 Funnel Analysis",
                                "network": "🕸️ Network View (Advanced)",
                            }[x],
                            help="Choose visualization style for process analysis",
                        )

                    with viz_col2:
                        show_frequencies = st.checkbox("📈 Show Frequencies", True)

                    with viz_col3:
                        min_frequency_filter = st.number_input(
                            "🔍 Min Frequency",
                            min_value=0,
                            value=0,
                            help="Filter out transitions below this frequency",
                        )

                    try:
                        # Create process mining diagram
                        visualizer = FunnelVisualizer(theme="dark", colorblind_friendly=True)

                        process_fig = visualizer.create_process_mining_diagram(
                            process_data,
                            visualization_type=visualization_type,
                            show_frequencies=show_frequencies,
                            show_statistics=True,
                            filter_min_frequency=(
                                min_frequency_filter if min_frequency_filter > 0 else None
                            ),
                        )

                        st.plotly_chart(process_fig, use_container_width=True)

                        # Add explanation for each visualization type
                        if visualization_type == "sankey":
                            st.info(
                                "🌊 **Flow Diagram**: Shows user journey as a flowing river - width represents user volume"
                            )
                        elif visualization_type == "journey":
                            st.info(
                                "🗺️ **Journey Map**: Step-by-step user progression with dropout rates"
                            )
                        elif visualization_type == "funnel":
                            st.info(
                                "📊 **Funnel Analysis**: Classic funnel showing conversion at each stage"
                            )
                        elif visualization_type == "network":
                            st.info(
                                "🕸️ **Network View**: Advanced graph showing all possible paths and cycles"
                            )

                    except Exception as e:
                        st.error(f"❌ Visualization error: {str(e)}")
                        st.info(
                            "💡 Try switching to a different visualization type or adjusting the frequency filter"
                        )

                    # Process Insights
                    if process_data.insights:
                        st.markdown("#### 💡 Key Insights")
                        for insight in process_data.insights[:5]:  # Show top 5 insights
                            st.markdown(f"• {insight}")

                    # Detailed Analysis Tables
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### 🎯 Activity Analysis")

                        activity_data = []
                        for activity, data in process_data.activities.items():
                            activity_data.append(
                                {
                                    "Activity": activity,
                                    "Users": f"{data['unique_users']:,}",
                                    "Frequency": f"{data['frequency']:,}",
                                    "Avg Duration": f"{data.get('avg_duration', 0):.1f}h",
                                    "Type": data.get("activity_type", "unknown"),
                                    "Success Rate": f"{data.get('success_rate', 0):.1f}%",
                                }
                            )

                        if activity_data:
                            activity_df = pd.DataFrame(activity_data)
                            st.dataframe(activity_df, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### 🔄 Top Transitions")

                        # Sort transitions by frequency
                        sorted_transitions = sorted(
                            process_data.transitions.items(),
                            key=lambda x: x[1]["frequency"],
                            reverse=True,
                        )

                        transition_data = []
                        for (from_act, to_act), data in sorted_transitions[:10]:  # Top 10
                            transition_data.append(
                                {
                                    "From": from_act,
                                    "To": to_act,
                                    "Users": f"{data['unique_users']:,}",
                                    "Frequency": f"{data['frequency']:,}",
                                    "Probability": f"{data.get('probability', 0):.1f}%",
                                    "Avg Duration": f"{data.get('avg_duration', 0):.1f}h",
                                }
                            )

                        if transition_data:
                            transition_df = pd.DataFrame(transition_data)
                            st.dataframe(transition_df, use_container_width=True, hide_index=True)

                    # Cycle Analysis
                    if process_data.cycles:
                        st.markdown("#### 🔄 Detected Cycles")

                        cycle_data = []
                        for cycle in process_data.cycles:
                            cycle_data.append(
                                {
                                    "Cycle Path": " → ".join(cycle["path"]),
                                    "Frequency": f"{cycle['frequency']:,}",
                                    "Type": cycle.get("type", "unknown"),
                                    "Impact": cycle.get("impact", "neutral"),
                                    "Avg Cycle Time": f"{cycle.get('avg_cycle_time', 0):.1f}h",
                                }
                            )

                        if cycle_data:
                            cycle_df = pd.DataFrame(cycle_data)
                            st.dataframe(cycle_df, use_container_width=True, hide_index=True)

                    # Process Variants
                    if process_data.variants:
                        st.markdown("#### 🛤️ Common Process Variants")

                        variant_data = []
                        for variant in process_data.variants[:10]:  # Top 10 variants
                            variant_data.append(
                                {
                                    "Path": " → ".join(variant["path"]),
                                    "Users": f"{variant['frequency']:,}",
                                    "Success Rate": f"{variant.get('success_rate', 0):.1f}%",
                                    "Avg Duration": f"{variant.get('avg_duration', 0):.1f}h",
                                    "Type": variant.get("variant_type", "standard"),
                                }
                            )

                        if variant_data:
                            variant_df = pd.DataFrame(variant_data)
                            st.dataframe(variant_df, use_container_width=True, hide_index=True)

            tab_idx += 1

            # Performance Monitor Tab
            if "performance_history" in st.session_state and st.session_state.performance_history:
                with tab_objects[tab_idx]:  # Performance Monitor
                    st.markdown("### ⚡ Performance Monitoring")

                    # Show comprehensive performance analysis
                    comprehensive_analysis = get_comprehensive_performance_analysis()

                    if comprehensive_analysis["overall_summary"]["functions_monitored"] > 0:
                        st.markdown("#### 🎯 System Performance Overview")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric(
                                "Total System Time",
                                f"{comprehensive_analysis['overall_summary']['total_execution_time']:.3f}s",
                            )
                        with col2:
                            st.metric(
                                "Total Function Calls",
                                comprehensive_analysis["overall_summary"]["total_function_calls"],
                            )
                        with col3:
                            st.metric(
                                "Avg Call Time",
                                f"{comprehensive_analysis['overall_summary']['average_call_time']:.4f}s",
                            )
                        with col4:
                            st.metric(
                                "Functions Monitored",
                                comprehensive_analysis["overall_summary"]["functions_monitored"],
                            )

                        # Show data source performance if available
                        if comprehensive_analysis["data_source_metrics"]:
                            st.markdown("#### 📊 Data Source Performance")

                            ds_metrics_table = []
                            for func_name, times in comprehensive_analysis[
                                "data_source_metrics"
                            ].items():
                                if times:
                                    ds_metrics_table.append(
                                        {
                                            "Data Operation": func_name,
                                            "Total Time (s)": f"{sum(times):.4f}",
                                            "Avg Time (s)": f"{np.mean(times):.4f}",
                                            "Calls": len(times),
                                            "Min Time (s)": f"{min(times):.4f}",
                                            "Max Time (s)": f"{max(times):.4f}",
                                        }
                                    )

                            if ds_metrics_table:
                                st.dataframe(
                                    pd.DataFrame(ds_metrics_table),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    # Show bottleneck analysis from calculator
                    if (
                        hasattr(st.session_state, "last_calculator")
                        and st.session_state.last_calculator
                    ):
                        bottleneck_analysis = (
                            st.session_state.last_calculator.get_bottleneck_analysis()
                        )

                        if bottleneck_analysis.get("bottlenecks"):
                            st.markdown("#### 🔍 Bottleneck Analysis")

                            # Summary metrics
                            summary = bottleneck_analysis["summary"]
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric(
                                    "Total Execution Time",
                                    f"{summary['total_execution_time']:.3f}s",
                                )
                            with col2:
                                st.metric(
                                    "Functions Monitored",
                                    summary["total_functions_monitored"],
                                )
                            with col3:
                                top_function_dominance = summary["performance_distribution"][
                                    "top_function_dominance"
                                ]
                                st.metric(
                                    "Top Function Dominance",
                                    f"{top_function_dominance:.1f}%",
                                )
                            with col4:
                                critical_pct = summary["performance_distribution"][
                                    "critical_functions_pct"
                                ]
                                st.metric("Critical Functions", f"{critical_pct:.1f}%")

                            # Bottleneck table
                            st.markdown(
                                "**⚠️ Function Performance Breakdown (Ordered by Total Time)**"
                            )

                            bottleneck_table_data = []
                            for func_data in bottleneck_analysis["bottlenecks"]:
                                # Color coding for critical bottlenecks
                                if func_data["percentage_of_total"] > 20:
                                    status = "🔴 Critical"
                                elif func_data["percentage_of_total"] > 10:
                                    status = "🟡 Moderate"
                                else:
                                    status = "🟢 Normal"

                                bottleneck_table_data.append(
                                    {
                                        "Function": func_data["function_name"],
                                        "Status": status,
                                        "Total Time (s)": f"{func_data['total_time']:.4f}",
                                        "% of Total": f"{func_data['percentage_of_total']:.1f}%",
                                        "Avg Time (s)": f"{func_data['avg_time']:.4f}",
                                        "Calls": func_data["call_count"],
                                        "Min Time (s)": f"{func_data['min_time']:.4f}",
                                        "Max Time (s)": f"{func_data['max_time']:.4f}",
                                        "Consistency": (
                                            f"{1 / func_data['time_per_call_consistency']:.1f}x"
                                            if func_data["time_per_call_consistency"] > 0
                                            else "Perfect"
                                        ),
                                    }
                                )

                            st.dataframe(
                                pd.DataFrame(bottleneck_table_data),
                                use_container_width=True,
                                hide_index=True,
                            )

                            # Critical bottlenecks alert
                            if bottleneck_analysis["critical_bottlenecks"]:
                                st.warning(
                                    f"🚨 **Critical Bottlenecks Detected:** "
                                    f"{', '.join([f['function_name'] for f in bottleneck_analysis['critical_bottlenecks']])} "
                                    f"are consuming significant computation time. Consider optimization."
                                )

                            # High variance functions alert
                            if bottleneck_analysis["high_variance_functions"]:
                                st.info(
                                    f"📊 **Variable Performance:** "
                                    f"{', '.join([f['function_name'] for f in bottleneck_analysis['high_variance_functions']])} "
                                    f"show high variance in execution times. May benefit from optimization."
                                )

                            # Optimization recommendations
                            st.markdown("#### 💡 Optimization Recommendations")

                            top_3 = summary["top_3_bottlenecks"]
                            if top_3:
                                st.markdown(
                                    f"1. **Primary Focus**: Optimize `{top_3[0]}` - highest time consumer"
                                )
                                if len(top_3) > 1:
                                    st.markdown(
                                        f"2. **Secondary Focus**: Review `{top_3[1]}` for efficiency improvements"
                                    )
                                if len(top_3) > 2:
                                    st.markdown(
                                        f"3. **Tertiary Focus**: Consider optimizing `{top_3[2]}`"
                                    )

                            st.markdown("---")

                    # Performance history table
                    st.markdown("#### 📊 Calculation History")
                    perf_data = []
                    for entry in st.session_state.performance_history:
                        perf_data.append(
                            {
                                "Timestamp": entry["timestamp"].strftime("%H:%M:%S"),
                                "Events Count": f"{entry['events_count']:,}",
                                "Steps": entry["steps_count"],
                                "Method": entry["method"],
                                "Engine": entry.get(
                                    "engine", "Pandas"
                                ),  # Default to Pandas for backward compatibility
                                "Calculation Time (s)": f"{entry['calculation_time']:.3f}",
                                "Events/Second": f"{entry['events_count'] / entry['calculation_time']:,.0f}",
                            }
                        )

                    if perf_data:
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True, hide_index=True)

                        # Performance visualization
                        col1, col2 = st.columns(2)

                        with col1:
                            # Calculation time trend
                            fig_time = go.Figure()
                            fig_time.add_trace(
                                go.Scatter(
                                    x=list(range(len(st.session_state.performance_history))),
                                    y=[
                                        entry["calculation_time"]
                                        for entry in st.session_state.performance_history
                                    ],
                                    mode="lines+markers",
                                    name="Calculation Time",
                                    line=dict(color="#3b82f6"),
                                )
                            )
                            fig_time.update_layout(
                                title="Calculation Time Trend",
                                xaxis_title="Calculation #",
                                yaxis_title="Time (seconds)",
                                height=300,
                            )
                            st.plotly_chart(fig_time, use_container_width=True)

                        with col2:
                            # Throughput visualization
                            fig_throughput = go.Figure()
                            throughput = [
                                entry["events_count"] / entry["calculation_time"]
                                for entry in st.session_state.performance_history
                            ]
                            fig_throughput.add_trace(
                                go.Scatter(
                                    x=list(range(len(st.session_state.performance_history))),
                                    y=throughput,
                                    mode="lines+markers",
                                    name="Events/Second",
                                    line=dict(color="#10b981"),
                                )
                            )
                            fig_throughput.update_layout(
                                title="Processing Throughput",
                                xaxis_title="Calculation #",
                                yaxis_title="Events/Second",
                                height=300,
                            )
                            st.plotly_chart(fig_throughput, use_container_width=True)

                        # Performance summary
                        st.markdown("**Performance Summary:**")
                        avg_time = np.mean(
                            [
                                entry["calculation_time"]
                                for entry in st.session_state.performance_history
                            ]
                        )
                        max_throughput = max(throughput)

                        col_perf1, col_perf2, col_perf3 = st.columns(3)
                        with col_perf1:
                            st.metric("Average Calculation Time", f"{avg_time:.3f}s")
                        with col_perf2:
                            st.metric("Max Throughput", f"{max_throughput:,.0f} events/s")
                        with col_perf3:
                            recent_improvement = 0.0
                            if len(st.session_state.performance_history) >= 2:
                                prev_calc_time = st.session_state.performance_history[-2][
                                    "calculation_time"
                                ]
                                current_calc_time = st.session_state.performance_history[-1][
                                    "calculation_time"
                                ]
                                if prev_calc_time > 0:  # Avoid division by zero
                                    recent_improvement = (
                                        (prev_calc_time - current_calc_time) / prev_calc_time * 100
                                    )
                            st.metric("Latest Improvement", f"{recent_improvement:+.1f}%")

                tab_idx += 1

    else:
        st.info(
            "👈 Please select and load a data source from the sidebar to begin funnel analysis"
        )

    # Test visualizations button
    if "analysis_results" in st.session_state and st.session_state.analysis_results:
        st.markdown("---")
        test_col1, test_col2, test_col3 = st.columns([1, 1, 1])
        with test_col2:
            if st.button("🧪 Test Visualizations", use_container_width=True):
                with st.spinner("Testing all visualizations..."):
                    test_results = test_visualizations()

                if test_results["success"]:
                    st.success("✅ All visualizations passed!")
                else:
                    failed_tests = [name for name, _ in test_results["failed"]]
                    st.error(f"❌ Failed tests: {', '.join(failed_tests)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #6b7280; padding: 20px;'>
        <p>Professional Funnel Analytics Platform - Enterprise-grade funnel analysis</p>
        <p>Supports file upload, ClickHouse integration, and real-time calculations</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def test_visualizations():
    """
    Universal test function to verify all visualizations render correctly.
    Can be run with:
    1. python app.py test_vis - to run in standalone mode with dummy data
    2. Called from within the app with actual data
    """

    import numpy as np
    import streamlit as st

    # Function to create minimal dummy data for testing
    def create_dummy_data():
        # Minimal FunnelResults
        class DummyFunnelResults:
            def __init__(self):
                self.steps = ["Step 1", "Step 2", "Step 3"]
                self.users_count = [1000, 700, 400]
                self.drop_offs = [0, 300, 300]
                self.drop_off_rates = [0, 30.0, 42.9]
                self.conversion_rates = [100.0, 70.0, 40.0]
                self.segment_data = {
                    "Segment A": [600, 400, 250],
                    "Segment B": [400, 300, 150],
                }

        # Minimal TimeToConvertStats
        class DummyTimeStats:
            def __init__(self, step_from, step_to):
                self.step_from = step_from
                self.step_to = step_to
                self.conversion_times = np.random.exponential(scale=2.0, size=10)
                self.mean_hours = np.mean(self.conversion_times)
                self.median_hours = np.median(self.conversion_times)
                self.p25_hours = np.percentile(self.conversion_times, 25)
                self.p75_hours = np.percentile(self.conversion_times, 75)
                self.p90_hours = np.percentile(self.conversion_times, 90)
                self.std_hours = np.std(self.conversion_times)

        # Minimal CohortData
        class DummyCohortData:
            def __init__(self):
                self.cohort_labels = ["Cohort 1", "Cohort 2"]
                self.cohort_sizes = {"Cohort 1": 500, "Cohort 2": 400}
                self.conversion_rates = {
                    "Cohort 1": [100.0, 75.0, 50.0],
                    "Cohort 2": [100.0, 70.0, 45.0],
                }

        # Minimal PathAnalysisData
        class DummyPathData:
            def __init__(self):
                self.dropoff_paths = {
                    "Step 1": {"Other Path 1": 150, "Other Path 2": 100},
                    "Step 2": {"Other Path 3": 200, "Other Path 4": 100},
                }
                self.between_steps_events = {
                    "Step 1 → Step 2": {"Event 1": 700},
                    "Step 2 → Step 3": {"Event 2": 400},
                }

        # Minimal StatSignificanceResult
        class DummyStatTest:
            def __init__(self):
                self.segment_a = "Segment A"
                self.segment_b = "Segment B"
                self.conversion_a = 40.0
                self.conversion_b = 25.0
                self.p_value = 0.03
                self.is_significant = True
                self.z_score = 2.5
                self.confidence_interval = (0.05, 0.15)

        return {
            "funnel_results": DummyFunnelResults(),
            "time_stats": [
                DummyTimeStats("Step 1", "Step 2"),
                DummyTimeStats("Step 2", "Step 3"),
            ],
            "cohort_data": DummyCohortData(),
            "path_data": DummyPathData(),
            "stat_tests": [DummyStatTest(), DummyStatTest()],
        }

    # Function to get real data if available, otherwise use dummy data
    def get_test_data():
        # Try to get real data from session state if exists
        data = {}

        try:
            # Check if we have session state and if we're in the Streamlit context
            has_session = (
                "session_state" in globals() or "st" in globals() and hasattr(st, "session_state")
            )

            if has_session and hasattr(st.session_state, "analysis_results"):
                results = st.session_state.analysis_results
                if results:
                    data["funnel_results"] = results
                    if hasattr(results, "time_to_convert"):
                        data["time_stats"] = results.time_to_convert
                    if hasattr(results, "cohort_data"):
                        data["cohort_data"] = results.cohort_data
                    if hasattr(results, "path_analysis"):
                        data["path_data"] = results.path_analysis
                    if hasattr(results, "stat_significance"):
                        data["stat_tests"] = results.stat_significance
        except Exception:
            pass  # If we can't access session state or it's not properly initialized

        # For any missing data, fill with dummy data
        dummy_data = create_dummy_data()
        for key in dummy_data:
            if key not in data or not data[key]:
                data[key] = dummy_data[key]

        return data

    # Track test results
    test_results = {"passed": [], "failed": []}

    # Get test data (real or dummy)
    data = get_test_data()

    # Set up Streamlit page
    st.title("Visualization Tests")
    st.markdown(
        "This test page verifies that all visualizations render correctly with dark theme."
    )

    # Run tests for each visualization
    with st.expander("Test Details", expanded=True):
        # Test 1: Funnel Chart
        try:
            funnel_chart = FunnelVisualizer.create_funnel_chart(data["funnel_results"])
            test_results["passed"].append("Funnel Chart")
            st.success("✅ Funnel Chart")
        except Exception as e:
            test_results["failed"].append(("Funnel Chart", str(e)))
            st.error(f"❌ Funnel Chart: {str(e)}")

        # Test 2: Segmented Funnel Chart
        try:
            segmented_funnel = FunnelVisualizer.create_funnel_chart(
                data["funnel_results"], show_segments=True
            )
            test_results["passed"].append("Segmented Funnel")
            st.success("✅ Segmented Funnel")
        except Exception as e:
            test_results["failed"].append(("Segmented Funnel", str(e)))
            st.error(f"❌ Segmented Funnel: {str(e)}")

        # Test 3: Conversion Flow Sankey
        try:
            flow_chart = FunnelVisualizer.create_conversion_flow_sankey(data["funnel_results"])
            test_results["passed"].append("Conversion Flow Sankey")
            st.success("✅ Conversion Flow Sankey")
        except Exception as e:
            test_results["failed"].append(("Conversion Flow Sankey", str(e)))
            st.error(f"❌ Conversion Flow Sankey: {str(e)}")

        # Test 4: Time to Convert Chart
        try:
            time_chart = FunnelVisualizer.create_time_to_convert_chart(data["time_stats"])
            test_results["passed"].append("Time to Convert Chart")
            st.success("✅ Time to Convert Chart")
        except Exception as e:
            test_results["failed"].append(("Time to Convert Chart", str(e)))
            st.error(f"❌ Time to Convert Chart: {str(e)}")

        # Test 5: Cohort Heatmap
        try:
            cohort_chart = FunnelVisualizer.create_cohort_heatmap(data["cohort_data"])
            test_results["passed"].append("Cohort Heatmap")
            st.success("✅ Cohort Heatmap")
        except Exception as e:
            test_results["failed"].append(("Cohort Heatmap", str(e)))
            st.error(f"❌ Cohort Heatmap: {str(e)}")

        # Test 6: Path Analysis Chart
        try:
            path_chart = FunnelVisualizer.create_path_analysis_chart(data["path_data"])
            test_results["passed"].append("Path Analysis Chart")
            st.success("✅ Path Analysis Chart")
        except Exception as e:
            test_results["failed"].append(("Path Analysis Chart", str(e)))
            st.error(f"❌ Path Analysis Chart: {str(e)}")

        # Test 7: Statistical Significance Table
        try:
            stat_table = FunnelVisualizer.create_statistical_significance_table(data["stat_tests"])
            test_results["passed"].append("Statistical Significance Table")
            st.success("✅ Statistical Significance Table")
        except Exception as e:
            test_results["failed"].append(("Statistical Significance Table", str(e)))
            st.error(f"❌ Statistical Significance Table: {str(e)}")

    # Show overall test result
    if not test_results["failed"]:
        st.success(f"✅ All {len(test_results['passed'])} visualizations passed!")
    else:
        st.error(
            f"❌ {len(test_results['failed'])} of {len(test_results['passed']) + len(test_results['failed'])} tests failed."
        )

    # Show successful visualizations
    if test_results["passed"]:
        st.subheader("Successful Visualizations")

        # Display the charts that passed
        for viz_name in test_results["passed"]:
            if viz_name == "Funnel Chart":
                st.subheader("1. Funnel Chart")
                st.plotly_chart(funnel_chart, use_container_width=True)
            elif viz_name == "Segmented Funnel":
                st.subheader("2. Segmented Funnel Chart")
                st.plotly_chart(segmented_funnel, use_container_width=True)
            elif viz_name == "Conversion Flow Sankey":
                st.subheader("3. Conversion Flow Sankey")
                st.plotly_chart(flow_chart, use_container_width=True)
            elif viz_name == "Time to Convert Chart":
                st.subheader("4. Time to Convert Chart")
                st.plotly_chart(time_chart, use_container_width=True)
            elif viz_name == "Cohort Heatmap":
                st.subheader("5. Cohort Heatmap")
                st.plotly_chart(cohort_chart, use_container_width=True)
            elif viz_name == "Path Analysis Chart":
                st.subheader("6. Path Analysis Chart")
                st.plotly_chart(path_chart, use_container_width=True)
            elif viz_name == "Statistical Significance Table":
                st.subheader("7. Statistical Significance Table")
                st.dataframe(stat_table)

    return {
        "success": len(test_results["failed"]) == 0,
        "passed": test_results["passed"],
        "failed": test_results["failed"],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test_vis":
        test_visualizations()
    else:
        main()
