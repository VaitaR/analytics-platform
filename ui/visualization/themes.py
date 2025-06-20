"""
Theme System for Funnel Analytics Platform
=========================================

This module contains all theme-related classes including color palettes, typography,
and interaction patterns. Extracted from the main app.py file to improve maintainability.

Classes:
    ColorPalette: WCAG 2.1 AA compliant color system
    TypographySystem: Responsive typography system
    InteractionPatterns: Consistent interaction patterns and animations

Usage:
    from ui.visualization.themes import ColorPalette, TypographySystem
    colors = ColorPalette()
    typography = TypographySystem()
"""

from typing import Any, Optional


class ColorPalette:
    """WCAG 2.1 AA compliant color palette with colorblind-friendly options"""

    # Primary semantic colors with accessibility compliance
    SEMANTIC = {
        "primary": "#3B82F6",  # Blue - primary brand color
        "secondary": "#6B7280",  # Gray - secondary brand color
        "success": "#10B981",  # Green - 4.5:1 contrast ratio
        "warning": "#F59E0B",  # Amber - 4.5:1 contrast ratio
        "error": "#EF4444",  # Red - 4.5:1 contrast ratio
        "info": "#3B82F6",  # Blue - 4.5:1 contrast ratio
        "neutral": "#6B7280",  # Gray - 4.5:1 contrast ratio
    }

    # Colorblind-friendly palette (Viridis-inspired)
    COLORBLIND_FRIENDLY = [
        "#440154",  # Dark purple
        "#31688E",  # Steel blue
        "#35B779",  # Teal green
        "#FDE725",  # Bright yellow
        "#B83A7E",  # Magenta
        "#1F968B",  # Cyan
        "#73D055",  # Light green
        "#DCE319",  # Yellow-green
    ]

    # High-contrast dark mode palette
    DARK_MODE = {
        "background": "#0F172A",  # Slate-900
        "surface": "#1E293B",  # Slate-800
        "surface_light": "#334155",  # Slate-700
        "text_primary": "#F8FAFC",  # Slate-50
        "text_secondary": "#E2E8F0",  # Slate-200
        "text_muted": "#94A3B8",  # Slate-400
        "border": "#475569",  # Slate-600
        "grid": "rgba(148, 163, 184, 0.2)",  # Subtle grid lines
    }

    # Gradient variations for depth
    GRADIENTS = {
        "primary": ["#3B82F6", "#1E40AF", "#1E3A8A"],
        "success": ["#10B981", "#059669", "#047857"],
        "warning": ["#F59E0B", "#D97706", "#B45309"],
        "error": ["#EF4444", "#DC2626", "#B91C1C"],
    }

    @staticmethod
    def get_color_with_opacity(color: str, opacity: float) -> str:
        """Convert hex color to rgba with specified opacity"""
        # If color is already in rgba format, extract rgb values and apply new opacity
        if color.startswith("rgba("):
            # Extract rgb values from rgba string
            import re

            rgba_match = re.match(r"rgba\((\d+),\s*(\d+),\s*(\d+),\s*[\d.]+\)", color)
            if rgba_match:
                r, g, b = rgba_match.groups()
                return f"rgba({r}, {g}, {b}, {opacity})"

        # Handle hex colors
        if color.startswith("#"):
            color = color[1:]

        # Ensure we have a valid hex color
        if len(color) == 6:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {opacity})"

        # Fallback - return original color if parsing fails
        return color

    @staticmethod
    def get_colorblind_scale(n_colors: int) -> list[str]:
        """Get n colors from colorblind-friendly palette"""
        if n_colors <= len(ColorPalette.COLORBLIND_FRIENDLY):
            return ColorPalette.COLORBLIND_FRIENDLY[:n_colors]
        # Repeat colors if needed
        return (
            ColorPalette.COLORBLIND_FRIENDLY
            * ((n_colors // len(ColorPalette.COLORBLIND_FRIENDLY)) + 1)
        )[:n_colors]


class TypographySystem:
    """Responsive typography system with proper hierarchy"""

    # Typography scale (rem units)
    SCALE = {
        "xs": 12,  # 0.75rem
        "sm": 14,  # 0.875rem
        "base": 16,  # 1rem
        "lg": 18,  # 1.125rem
        "xl": 20,  # 1.25rem
        "2xl": 24,  # 1.5rem
        "3xl": 30,  # 1.875rem
        "4xl": 36,  # 2.25rem
    }

    # Font weights
    WEIGHTS = {
        "light": 300,
        "normal": 400,
        "medium": 500,
        "semibold": 600,
        "bold": 700,
        "extrabold": 800,
    }

    # Line heights for optimal readability
    LINE_HEIGHTS = {"tight": 1.25, "normal": 1.5, "relaxed": 1.625, "loose": 2.0}

    @staticmethod
    def get_font_config(
        size: str = "base",
        weight: str = "normal",
        line_height: str = "normal",
        color: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get complete font configuration"""
        config = {
            "size": TypographySystem.SCALE[size],
            "weight": TypographySystem.WEIGHTS[weight],
            "family": '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        }

        if color:
            config["color"] = color

        return config


class InteractionPatterns:
    """Consistent interaction patterns and animations"""

    # Animation durations (milliseconds)
    TRANSITIONS = {"fast": 150, "normal": 300, "slow": 500}

    # Hover states
    HOVER_EFFECTS = {"scale": 1.05, "opacity_change": 0.8, "border_width": 2}

    @staticmethod
    def get_hover_template(
        title: str, value_formatter: str = "%{y}", extra_info: Optional[str] = None
    ) -> str:
        """Generate consistent hover templates"""
        template = f"<b>{title}</b><br>"
        template += f"Value: {value_formatter}<br>"

        if extra_info:
            template += f"{extra_info}<br>"

        template += "<extra></extra>"
        return template

    @staticmethod
    def get_animation_config(duration: str = "normal") -> dict[str, Any]:
        """Get animation configuration"""
        return {
            "transition": {
                "duration": InteractionPatterns.TRANSITIONS[duration],
                "easing": "cubic-bezier(0.4, 0, 0.2, 1)",  # Smooth easing
            }
        }
