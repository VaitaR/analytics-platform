"""
CSS Styles Module for Funnel Analytics Platform UI
==============================================

This module contains all CSS styles for the Streamlit application.
Extracted from the main app.py file to improve maintainability and separation of concerns.

Usage:
    from ui.utils.css_styles import apply_custom_css
    apply_custom_css()
"""

import streamlit as st


def get_custom_css() -> str:
    """
    Get the custom CSS styles for the application.

    Returns:
        str: CSS styles as a string
    """
    return """
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
"""


def apply_custom_css() -> None:
    """
    Apply custom CSS styles to the Streamlit application.

    This function should be called once at the beginning of the application
    to ensure all custom styles are applied.
    """
    st.markdown(get_custom_css(), unsafe_allow_html=True)


# CSS utility functions for dynamic styling
def get_card_class(card_type: str) -> str:
    """
    Get the appropriate CSS class for different card types.

    Args:
        card_type: Type of card ('metric', 'step', 'data-source', 'segment', 'cohort', 'event')

    Returns:
        str: CSS class name
    """
    card_classes = {
        "metric": "metric-card",
        "step": "step-container",
        "data-source": "data-source-card",
        "segment": "segment-card",
        "cohort": "cohort-card",
        "event": "event-card",
    }
    return card_classes.get(card_type, "event-card")


def get_frequency_class(frequency_level: str) -> str:
    """
    Get the appropriate CSS class for frequency indicators.

    Args:
        frequency_level: Level of frequency ('high', 'medium', 'low')

    Returns:
        str: CSS class name
    """
    frequency_classes = {
        "high": "frequency-high",
        "medium": "frequency-medium",
        "low": "frequency-low",
    }
    return frequency_classes.get(frequency_level, "frequency-low")
