"""
Layout Configuration for Funnel Analytics Platform
===============================================

This module contains layout-related configurations including responsive design,
spacing, and chart dimensions. Extracted from app.py for better maintainability.

Classes:
    LayoutConfig: 8px grid system and responsive layout configuration

Usage:
    from ui.visualization.layout import LayoutConfig
    layout = LayoutConfig()
    height = layout.get_responsive_height(500, 10)
"""


class LayoutConfig:
    """8px grid system and responsive layout configuration"""

    # 8px grid system
    SPACING = {
        "xs": 8,  # 0.5rem
        "sm": 16,  # 1rem
        "md": 24,  # 1.5rem
        "lg": 32,  # 2rem
        "xl": 48,  # 3rem
        "2xl": 64,  # 4rem
        "3xl": 96,  # 6rem
    }

    # Responsive breakpoints
    BREAKPOINTS = {"mobile": 640, "tablet": 768, "desktop": 1024, "wide": 1280}

    # Chart dimensions and aspect ratios
    CHART_DIMENSIONS = {
        "small": {
            "width": 400,
            "height": 350,
            "ratio": 8 / 7,
        },  # Mobile-friendly, meets 350px minimum
        "medium": {"width": 600, "height": 400, "ratio": 3 / 2},  # Standard desktop
        "large": {"width": 800, "height": 500, "ratio": 8 / 5},  # Large desktop
        "wide": {"width": 1200, "height": 600, "ratio": 2 / 1},  # Ultra-wide displays
    }

    @staticmethod
    def get_responsive_height(base_height: int, content_count: int = 1) -> int:
        """Calculate responsive height based on content and screen size with reasonable caps"""
        # Ensure minimum height for usability
        min_height = 400

        # Cap the content scaling to prevent excessive growth
        # Only allow scaling up to 20 items worth of growth
        max_scaling_items = min(content_count - 1, 20)
        scaling_height = max_scaling_items * 20  # Reduced from 40 to 20 per item

        dynamic_height = base_height + scaling_height

        # Set reasonable maximum height limits
        max_height = min(800, base_height * 1.6)  # Cap at 1.6x base or 800px max

        # Apply all constraints
        final_height = max(min_height, min(dynamic_height, max_height))

        return int(final_height)

    @staticmethod
    def get_margins(size: str = "md") -> dict[str, int]:
        """Get standard margins for charts"""
        base = LayoutConfig.SPACING[size]
        return {
            "l": base * 2,  # Left margin for y-axis labels
            "r": base,  # Right margin
            "t": base * 2,  # Top margin for title
            "b": base,  # Bottom margin
        }
