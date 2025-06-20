"""
UI Components Module for Funnel Analytics Platform
================================================

This module contains reusable UI components for the funnel analytics application.

Functions:
    create_simple_event_selector: Create simplified event selector interface
    filter_events: Filter events based on search and categories
    get_event_statistics: Get comprehensive event statistics
    create_funnel_step_display: Create visual funnel step display
"""

from ui.components.funnel_builder import (
    calculate_funnel,
    clear_funnel,
    create_funnel_step_display,
    create_simple_event_selector,
    filter_events,
    get_event_statistics,
    move_step,
    remove_step,
    toggle_event_in_funnel,
)

__all__ = [
    "create_simple_event_selector",
    "filter_events",
    "get_event_statistics",
    "create_funnel_step_display",
    "toggle_event_in_funnel",
    "move_step",
    "remove_step",
    "clear_funnel",
    "calculate_funnel",
]
