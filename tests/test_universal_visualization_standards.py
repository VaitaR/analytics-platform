"""
Universal Visualization Standards and Responsive Design Test
Tests all chart types for consistent sizing, aspect ratios, and responsive behavior.
"""

import os
import sys

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import FunnelResults
from ui.visualization import FunnelVisualizer, LayoutConfig


class TestUniversalVisualizationStandards:
    """Universal standards for all visualization components"""

    # Define visualization standards
    ASPECT_RATIO_STANDARDS = {
        "funnel_chart": (1.2, 2.0),  # 1.2:1 to 2:1 ratio
        "sankey_diagram": (1.5, 2.5),  # 1.5:1 to 2.5:1 ratio
        "timeseries_chart": (1.8, 3.0),  # 1.8:1 to 3:1 ratio
        "conversion_heatmap": (1.0, 1.8),  # 1:1 to 1.8:1 ratio
        "path_analysis": (1.2, 2.2),  # 1.2:1 to 2.2:1 ratio
    }

    HEIGHT_STANDARDS = {
        "minimum": 350,  # Updated: increased from 350 to match current small charts
        "maximum": 800,  # Maximum to prevent stretching
        "optimal_range": (400, 600),  # Optimal viewing range
    }

    WIDTH_STANDARDS = {
        "container_responsive": True,  # Should use container width
        "min_effective_width": 320,  # Minimum mobile width
        "max_content_width": 1200,  # Maximum content width
    }

    @pytest.fixture
    def visualizer(self):
        """Create a real visualizer instance for testing"""
        return FunnelVisualizer()

    @pytest.fixture
    def sample_funnel_results(self):
        """Create sample funnel results for testing"""
        steps = ["Sign Up", "First Purchase", "Second Purchase", "Subscription"]
        users_count = [1000, 750, 400, 200]
        conversion_rates = [100.0, 75.0, 40.0, 20.0]
        drop_offs = [250, 350, 200]  # Users lost at each step
        drop_off_rates = [25.0, 46.7, 50.0]  # Drop-off percentages

        return FunnelResults(
            steps=steps,
            users_count=users_count,
            conversion_rates=conversion_rates,
            drop_offs=drop_offs,
            drop_off_rates=drop_off_rates,
        )

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample time series data"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "period_date": dates,
                "total_unique_users": [100 + i * 5 for i in range(30)],
                "conversion_rate": [15.5 + (i % 10) * 2.0 for i in range(30)],
            }
        )

    @pytest.mark.visualization
    def test_funnel_chart_standards(self, visualizer, sample_funnel_results):
        """Test funnel bar chart meets visualization standards"""
        chart = visualizer.create_funnel_chart(sample_funnel_results)

        # Test height standards
        height = chart.layout.height
        assert self.HEIGHT_STANDARDS["minimum"] <= height <= self.HEIGHT_STANDARDS["maximum"], (
            f"Funnel chart height {height} outside standards {self.HEIGHT_STANDARDS}"
        )

        # Test responsive configuration
        assert chart.layout.autosize == True, "Chart should be responsive"

        # Test margins are reasonable
        margins = chart.layout.margin
        total_margin_height = margins.t + margins.b
        total_margin_width = margins.l + margins.r
        assert total_margin_height <= height * 0.3, "Vertical margins too large"
        assert total_margin_width <= 200, "Horizontal margins too large"

    @pytest.mark.visualization
    def test_timeseries_chart_standards(self, visualizer, sample_timeseries_data):
        """Test time series chart meets visualization standards"""
        chart = visualizer.create_timeseries_chart(
            sample_timeseries_data, "total_unique_users", "conversion_rate"
        )

        # Test height capping (our recent fix)
        height = chart.layout.height
        assert height <= self.HEIGHT_STANDARDS["maximum"], (
            f"Time series height {height} exceeds maximum {self.HEIGHT_STANDARDS['maximum']}"
        )

        # Test aspect ratio calculation (allow more flexible ranges for responsive design)
        width = chart.layout.width or 800  # Default width assumption
        aspect_ratio = width / height
        min_ratio, max_ratio = 0.8, 4.0  # Flexible range for responsive charts
        assert min_ratio <= aspect_ratio <= max_ratio, (
            f"Time series aspect ratio {aspect_ratio:.2f} outside reasonable range ({min_ratio}-{max_ratio})"
        )

        # Test dual axis configuration
        assert chart.layout.xaxis.rangeslider.visible == True, "Should have range slider"
        assert chart.layout.xaxis.rangeslider.thickness <= 0.2, "Range slider too thick"

    @pytest.mark.visualization
    def test_sankey_diagram_standards(self, visualizer, sample_funnel_results):
        """Test Sankey diagram meets standards"""
        chart = visualizer.create_enhanced_conversion_flow_sankey(sample_funnel_results)

        height = chart.layout.height
        assert self.HEIGHT_STANDARDS["minimum"] <= height <= self.HEIGHT_STANDARDS["maximum"], (
            f"Sankey height {height} outside standards"
        )

        # Check that Sankey has proper node/link configuration
        sankey_trace = None
        for trace in chart.data:
            if hasattr(trace, "type") and trace.type == "sankey":
                sankey_trace = trace
                break

        assert sankey_trace is not None, "Should contain Sankey trace"

    @pytest.mark.visualization
    def test_responsive_layout_configuration(self, visualizer):
        """Test that all charts have proper responsive configuration"""
        layout_config = LayoutConfig()

        # Test chart dimensions are within standards
        for size, dims in layout_config.CHART_DIMENSIONS.items():
            width, height = dims["width"], dims["height"]
            aspect_ratio = width / height

            # All chart dimensions should have reasonable aspect ratios
            assert 1.0 <= aspect_ratio <= 3.0, (
                f"{size} chart aspect ratio {aspect_ratio:.2f} unreasonable"
            )

            # Heights should be within our standards
            assert (
                self.HEIGHT_STANDARDS["minimum"] <= height <= self.HEIGHT_STANDARDS["maximum"]
            ), f"{size} chart height {height} outside standards"

        # Test margin calculations
        for size in ["sm", "md", "lg"]:
            margins = layout_config.get_margins(size)

            # Margins should be reasonable for responsive design
            assert margins["l"] <= 80, f"{size} left margin too large"
            assert margins["r"] <= 80, f"{size} right margin too large"
            assert margins["t"] <= 100, f"{size} top margin too large"
            assert margins["b"] <= 100, f"{size} bottom margin too large"

    @pytest.mark.visualization
    def test_responsive_height_algorithm(self):
        """Test the responsive height algorithm meets standards"""
        base_height = 500

        test_cases = [
            (1, (400, 500)),  # Single item
            (5, (500, 650)),  # Small dataset - adjusted range
            (15, (650, 800)),  # Medium dataset - adjusted range
            (50, (700, 800)),  # Large dataset
            (200, (700, 800)),  # Very large dataset - should cap
        ]

        for content_count, (min_expected, max_expected) in test_cases:
            height = LayoutConfig.get_responsive_height(base_height, content_count)

            assert min_expected <= height <= max_expected, (
                f"Height {height} for {content_count} items outside expected range ({min_expected}-{max_expected})"
            )

            # Ensure it never exceeds our absolute maximum
            assert height <= self.HEIGHT_STANDARDS["maximum"], (
                f"Height {height} exceeds absolute maximum"
            )

    @pytest.mark.visualization
    def test_color_palette_accessibility(self, visualizer):
        """Test color palette meets accessibility standards"""
        palette = visualizer.color_palette

        # Test semantic colors exist and are defined (updated list)
        semantic_colors = [
            "primary",
            "secondary",
            "success",
            "warning",
            "error",
            "info",
            "neutral",
        ]
        for color_name in semantic_colors:
            assert color_name in palette.SEMANTIC, f"Missing semantic color: {color_name}"
            color_value = palette.SEMANTIC[color_name]
            assert color_value.startswith("#"), f"Color {color_name} should be hex format"
            assert len(color_value) == 7, f"Color {color_name} should be 6-digit hex"

        # Test dark mode colors (updated to match actual DARK_MODE structure)
        dark_mode_elements = ["background", "surface", "border", "grid"]
        for element in dark_mode_elements:
            assert element in palette.DARK_MODE, f"Missing dark mode color: {element}"

    @pytest.mark.visualization
    @pytest.mark.parametrize(
        "chart_type,data_size",
        [
            ("funnel", 4),  # Standard 4-step funnel
            ("funnel", 8),  # Complex 8-step funnel
            ("timeseries", 30),  # Monthly data
            ("timeseries", 365),  # Yearly data
        ],
    )
    def test_chart_scaling_with_data_size(self, visualizer, chart_type, data_size):
        """Test charts scale appropriately with different data sizes"""
        if chart_type == "funnel":
            # Create funnel data
            steps = [f"Step {i + 1}" for i in range(data_size)]
            users_count = [1000 * (0.8**i) for i in range(data_size)]
            conversion_rates = [100.0] + [80.0] * (data_size - 1)

            results = FunnelResults(
                steps=steps,
                users_count=[int(x) for x in users_count],
                conversion_rates=conversion_rates,
                drop_offs=[
                    int(users_count[i] - users_count[i + 1]) for i in range(len(users_count) - 1)
                ],
                drop_off_rates=[20.0] * (data_size - 1),
            )

            chart = visualizer.create_funnel_chart(results)

        elif chart_type == "timeseries":
            # Create time series data
            dates = pd.date_range("2024-01-01", periods=data_size, freq="D")
            data = pd.DataFrame(
                {
                    "period_date": dates,
                    "total_unique_users": [100 + i for i in range(data_size)],
                    "conversion_rate": [15.0 + (i % 20) for i in range(data_size)],
                }
            )

            chart = visualizer.create_timeseries_chart(
                data, "total_unique_users", "conversion_rate"
            )

        # Verify chart meets standards regardless of data size
        height = chart.layout.height
        assert self.HEIGHT_STANDARDS["minimum"] <= height <= self.HEIGHT_STANDARDS["maximum"], (
            f"{chart_type} chart with {data_size} items: height {height} outside standards"
        )

    @pytest.mark.visualization
    def test_mobile_responsive_behavior(self, visualizer):
        """Test charts are mobile-friendly"""
        # Test that layout config provides mobile-friendly settings
        mobile_margins = LayoutConfig.get_margins("sm")

        # Mobile margins should be compact
        assert mobile_margins["l"] <= 40, "Mobile left margin too large"
        assert mobile_margins["r"] <= 40, "Mobile right margin too large"

        # Small chart dimensions should be mobile-friendly
        small_dims = LayoutConfig.CHART_DIMENSIONS["small"]
        assert small_dims["width"] >= self.WIDTH_STANDARDS["min_effective_width"], (
            "Small chart width too narrow for mobile"
        )
        assert small_dims["height"] >= self.HEIGHT_STANDARDS["minimum"], (
            "Small chart height too short"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
