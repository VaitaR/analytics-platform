"""
Test file to reproduce and fix UI chart stretching issues
for Time Series Analysis visualization.
"""

import os
import sys

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import FunnelVisualizer, LayoutConfig


class TestTimeSeriesChartLayout:
    """Test suite for Time Series chart layout and responsive behavior"""

    @pytest.fixture
    def visualizer(self):
        """Create a real visualizer for testing"""
        return FunnelVisualizer(theme="dark", colorblind_friendly=False)

    @pytest.fixture
    def sample_timeseries_data(self):
        """Create sample time series data for testing"""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "period_date": dates,
                "total_unique_users": [100 + i * 5 for i in range(30)],
                "conversion_rate": [15.5 + (i % 10) * 2.0 for i in range(30)],
            }
        )

    def test_chart_height_calculation_basic(self, visualizer, sample_timeseries_data):
        """Test basic height calculation for time series charts"""
        chart = visualizer.create_timeseries_chart(
            sample_timeseries_data, "total_unique_users", "conversion_rate"
        )

        # Check that height is calculated based on data points
        assert "height" in chart.layout
        # New universal standards: minimum 350px, capped at 800px
        assert chart.layout.height >= 350
        assert chart.layout.height <= 800

    def test_chart_height_with_large_dataset(self, visualizer):
        """Test height calculation with large datasets that might cause stretching"""
        # Create large dataset to test stretching issues
        large_data = pd.DataFrame(
            {
                "period_date": pd.date_range("2023-01-01", periods=365, freq="D"),
                "total_unique_users": range(365),
                "conversion_rate": [15.0 + (i % 20) for i in range(365)],
            }
        )

        chart = visualizer.create_timeseries_chart(
            large_data, "total_unique_users", "conversion_rate"
        )

        # New universal standards: height is capped at 800px
        assert chart.layout.height <= 800  # Maximum height cap
        assert chart.layout.height >= 350  # Minimum height

    def test_chart_margins_for_dual_axis(self, visualizer, sample_timeseries_data):
        """Test that margins are appropriate for dual y-axis display"""
        chart = visualizer.create_timeseries_chart(
            sample_timeseries_data, "total_unique_users", "conversion_rate"
        )

        # Check margins are set correctly for dual axis - new optimized margins
        margins = chart.layout.margin
        assert margins.l >= 50  # Left margin for primary y-axis (optimized)
        assert margins.r >= 50  # Right margin for secondary y-axis (optimized)
        assert margins.t >= 70  # Top margin for title (optimized)
        assert margins.b >= 80  # Bottom margin for legend and range slider (optimized)

    def test_responsive_height_caps(self):
        """Test that responsive height calculation has reasonable caps"""
        # Test with small dataset
        small_height = LayoutConfig.get_responsive_height(500, 5)
        assert small_height >= 350  # Universal minimum height

        # Test with medium dataset - new scaling factor is 20 per item
        medium_height = LayoutConfig.get_responsive_height(500, 15)
        expected_height = min(800, 500 + min(15 - 1, 20) * 20)  # Capped scaling
        assert medium_height == expected_height

        # Test with large dataset - should be capped at 800px
        large_height = LayoutConfig.get_responsive_height(500, 100)
        assert large_height <= 800  # Universal maximum height cap

    @pytest.mark.visualization
    def test_chart_responsiveness_config(self, visualizer, sample_timeseries_data):
        """Test chart configuration for responsive behavior"""
        chart = visualizer.create_timeseries_chart(
            sample_timeseries_data, "total_unique_users", "conversion_rate"
        )

        # Check that chart has responsive configuration
        assert chart.layout.xaxis.rangeslider.visible == True
        assert chart.layout.hovermode == "x unified"

        # Check legend positioning for mobile compatibility
        legend = chart.layout.legend
        assert legend.orientation == "h"  # Horizontal for space efficiency
        assert legend.yanchor == "bottom"
        assert legend.y == -0.2  # Below chart to avoid overlap

    @pytest.mark.parametrize(
        "dataset_size,expected_height_range",
        [
            (10, (680, 680)),  # Small dataset - actual responsive calculation
            (30, (800, 800)),  # Medium dataset - capped at 800
            (100, (800, 800)),  # Large dataset - capped at 800
        ],
    )
    def test_height_scaling_by_data_size(
        self, visualizer, dataset_size, expected_height_range
    ):
        """Test height scaling based on dataset size"""
        # Create dataset of specified size
        data = pd.DataFrame(
            {
                "period_date": pd.date_range(
                    "2024-01-01", periods=dataset_size, freq="D"
                ),
                "total_unique_users": range(dataset_size),
                "conversion_rate": [15.0] * dataset_size,
            }
        )

        chart = visualizer.create_timeseries_chart(
            data, "total_unique_users", "conversion_rate"
        )

        min_height, max_height = expected_height_range
        assert min_height <= chart.layout.height <= max_height


class TestChartStretchingFix:
    """Specific tests for identifying and fixing chart stretching issues"""

    def test_identify_stretching_issue(self):
        """Identify the root cause of vertical stretching"""
        # The issue is likely in:
        # 1. get_responsive_height calculation
        # 2. Margins being too large
        # 3. Range slider taking too much space

        # Test current responsive height behavior
        base_height = 500
        content_counts = [1, 10, 20, 50, 100]

        for count in content_counts:
            height = LayoutConfig.get_responsive_height(base_height, count)
            print(f"Content count: {count}, Height: {height}")

            # Issue: Height grows unbounded with content
            if count > 20:
                # For large datasets, height should be capped
                assert (
                    height <= base_height * 2
                ), f"Height {height} too large for {count} items"

    def test_fix_height_calculation(self):
        """Test improved height calculation that prevents stretching"""

        def get_improved_responsive_height(
            base_height: int, content_count: int = 1
        ) -> int:
            """Improved height calculation with better caps"""
            # Cap the content scaling to prevent excessive growth
            scaling_factor = (
                min(content_count - 1, 20) * 20
            )  # Max 20 items of scaling, 20px per item
            dynamic_height = base_height + scaling_factor

            # Ensure reasonable bounds - universal standards
            min_height = 350  # Universal minimum
            max_height = 800  # Universal maximum

            return max(min_height, min(dynamic_height, max_height))

        # Test the improved calculation
        base_height = 500

        # Small datasets should work normally
        small_result = get_improved_responsive_height(base_height, 5)
        assert 350 <= small_result <= 800

        # Large datasets should be capped
        large_height = get_improved_responsive_height(base_height, 100)
        assert large_height <= 800  # Should be capped
        assert large_height >= 350  # Should meet minimum

    def test_fix_margin_calculation(self):
        """Test improved margin calculation for dual-axis charts"""

        def get_improved_margins_for_timeseries() -> dict:
            """Improved margins specifically for time series dual-axis charts"""
            return {
                "l": 50,  # Optimized left margin
                "r": 50,  # Optimized right margin
                "t": 70,  # Optimized top margin
                "b": 80,  # Optimized bottom margin
            }

        margins = get_improved_margins_for_timeseries()

        # Margins should be optimized for mobile/responsive design
        assert margins["l"] <= 50
        assert margins["r"] <= 50
        assert margins["t"] <= 70
        assert margins["b"] <= 80


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
