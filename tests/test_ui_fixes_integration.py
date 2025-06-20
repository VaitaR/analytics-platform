"""
Simple integration test to verify Time Series chart UI fixes.
Tests the actual functionality without complex mocking.
"""

import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import LayoutConfig


class TestTimeSeriesUIFixesIntegration:
    """Integration tests for Time Series UI fixes"""

    def test_height_calculation_is_capped(self):
        """Test that height calculation has reasonable caps"""
        # Test small dataset
        small_height = LayoutConfig.get_responsive_height(500, 5)
        assert 400 <= small_height <= 600

        # Test medium dataset
        medium_height = LayoutConfig.get_responsive_height(500, 15)
        assert 400 <= medium_height <= 800

        # Test large dataset - should be capped
        large_height = LayoutConfig.get_responsive_height(500, 50)
        assert large_height <= 800, f"Height {large_height} should be capped at 800px"

        # Test very large dataset - should still be capped
        very_large_height = LayoutConfig.get_responsive_height(500, 200)
        assert (
            very_large_height <= 800
        ), f"Height {very_large_height} should be capped at 800px"

    def test_height_minimum_enforced(self):
        """Test that minimum height is enforced"""
        # Even tiny datasets should have minimum height
        tiny_height = LayoutConfig.get_responsive_height(300, 1)
        assert tiny_height >= 400, f"Height {tiny_height} should meet minimum of 400px"

    def test_height_scaling_is_reasonable(self):
        """Test that height scaling is reasonable and predictable"""
        base_height = 500

        # Test progressive scaling
        heights = []
        for count in [1, 5, 10, 20, 50, 100]:
            height = LayoutConfig.get_responsive_height(base_height, count)
            heights.append((count, height))

        print("\nHeight scaling test results:")
        for count, height in heights:
            print(f"Items: {count:3d} -> Height: {height:3d}px")

        # Check that scaling is reasonable
        assert heights[0][1] == 500  # Base case
        assert heights[1][1] > heights[0][1]  # Should increase
        assert heights[-1][1] <= 800  # Should be capped

        # Check that very large datasets don't cause excessive growth
        assert (
            heights[-1][1] == heights[-2][1]
        ), "Very large datasets should hit the cap"

    def test_performance_with_different_dataset_sizes(self):
        """Test that height calculation is fast for any dataset size"""
        import time

        dataset_sizes = [10, 100, 1000, 10000]

        for size in dataset_sizes:
            start_time = time.time()
            height = LayoutConfig.get_responsive_height(500, size)
            end_time = time.time()

            # Should be instant
            assert (
                end_time - start_time
            ) < 0.001, f"Height calculation too slow for {size} items"
            assert (
                400 <= height <= 800
            ), f"Height {height} out of range for {size} items"


class TestUIResponsiveDesign:
    """Test responsive design improvements"""

    def test_margin_calculations(self):
        """Test that margin calculations are reasonable for different screen sizes"""
        margins_sm = LayoutConfig.get_margins("sm")
        margins_md = LayoutConfig.get_margins("md")
        margins_lg = LayoutConfig.get_margins("lg")

        print("\nMargin test results:")
        print(f"Small: {margins_sm}")
        print(f"Medium: {margins_md}")
        print(f"Large: {margins_lg}")

        # All margins should be reasonable for mobile/responsive design
        for size, margins in [
            ("sm", margins_sm),
            ("md", margins_md),
            ("lg", margins_lg),
        ]:
            assert margins["l"] <= 80, f"{size} left margin too large: {margins['l']}"
            assert margins["r"] <= 80, f"{size} right margin too large: {margins['r']}"
            assert margins["t"] <= 100, f"{size} top margin too large: {margins['t']}"
            assert margins["b"] <= 80, f"{size} bottom margin too large: {margins['b']}"

    def test_chart_dimensions_are_reasonable(self):
        """Test that chart dimensions are appropriate for different screen sizes"""
        dimensions = LayoutConfig.CHART_DIMENSIONS

        print("\nChart dimensions test results:")
        for size, dims in dimensions.items():
            print(
                f"{size}: {dims['width']}x{dims['height']} (ratio: {dims['ratio']:.2f})"
            )

            # Check that dimensions are reasonable
            assert (
                300 <= dims["width"] <= 1400
            ), f"{size} width {dims['width']} unreasonable"
            assert (
                200 <= dims["height"] <= 800
            ), f"{size} height {dims['height']} unreasonable"

            # Check aspect ratios are reasonable (not too tall/wide)
            assert (
                1.0 <= dims["ratio"] <= 3.0
            ), f"{size} aspect ratio {dims['ratio']} unreasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
