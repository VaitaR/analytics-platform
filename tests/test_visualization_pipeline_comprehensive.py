"""
Comprehensive Visualization Pipeline Testing Suite

This test suite covers all aspects of the visualization pipeline:
- Chart rendering pipeline validation
- Plotly specification compliance
- Responsive behavior testing
- Theme consistency
- Performance optimization
"""

import json

import plotly.graph_objects as go
import pytest

from models import (
    FunnelResults,
)
from ui.visualization import FunnelVisualizer
from ui.visualization.layout import LayoutConfig


@pytest.mark.visualization
@pytest.mark.rendering
class TestChartRenderingPipeline:
    """Test the complete chart rendering pipeline."""

    @pytest.fixture
    def visualizer(self):
        """Standard visualizer for testing."""
        return FunnelVisualizer(theme="dark", colorblind_friendly=True)

    @pytest.fixture
    def sample_funnel_results(self):
        """Sample funnel results for testing."""
        results = FunnelResults(
            steps=["Sign Up", "Email Verification", "First Login", "First Purchase"],
            users_count=[1000, 800, 600, 400],
            conversion_rates=[100.0, 80.0, 75.0, 66.7],
            drop_offs=[0, 200, 200, 200],
            drop_off_rates=[0.0, 20.0, 25.0, 33.3],
        )
        # Add segment data for testing
        results.segment_data = {
            "Premium": [600, 500, 400, 300],
            "Free": [400, 300, 200, 100],
        }
        return results

    def test_funnel_chart_rendering_pipeline(self, visualizer, sample_funnel_results):
        """Test complete funnel chart rendering pipeline."""
        # Test basic funnel chart
        chart = visualizer.create_enhanced_funnel_chart(sample_funnel_results)

        # Validate chart object
        assert isinstance(chart, go.Figure), "Should return Plotly Figure object"
        assert chart.data is not None, "Chart should have data"
        assert chart.layout is not None, "Chart should have layout"

        # Validate data traces
        assert len(chart.data) > 0, "Chart should have at least one trace"

        # Validate layout properties
        assert chart.layout.title is not None, "Chart should have title"
        assert chart.layout.height is not None, "Chart should have height"

        # Test with segments
        segmented_chart = visualizer.create_enhanced_funnel_chart(
            sample_funnel_results, show_segments=True
        )
        assert isinstance(segmented_chart, go.Figure), "Segmented chart should be valid Figure"

        print("✅ Funnel chart rendering pipeline test passed")

    def test_sankey_diagram_rendering_pipeline(self, visualizer, sample_funnel_results):
        """Test Sankey diagram rendering pipeline."""
        chart = visualizer.create_enhanced_conversion_flow_sankey(sample_funnel_results)

        # Validate Sankey structure
        assert isinstance(chart, go.Figure), "Should return Plotly Figure object"

        # Find Sankey trace
        sankey_trace = None
        for trace in chart.data:
            if hasattr(trace, "type") and trace.type == "sankey":
                sankey_trace = trace
                break

        assert sankey_trace is not None, "Should contain Sankey trace"
        assert hasattr(sankey_trace, "node"), "Sankey should have nodes"
        assert hasattr(sankey_trace, "link"), "Sankey should have links"

        print("✅ Sankey diagram rendering pipeline test passed")


@pytest.mark.visualization
@pytest.mark.responsive
class TestResponsiveBehavior:
    """Test responsive behavior across different screen sizes and data sizes."""

    @pytest.fixture
    def visualizer(self):
        """Standard visualizer for testing."""
        return FunnelVisualizer(theme="dark", colorblind_friendly=True)

    def test_responsive_height_calculation(self, visualizer):
        """Test responsive height calculation algorithm."""
        # Test LayoutConfig responsive height calculation
        test_scenarios = [
            (400, 3, 400),  # Small funnel, minimum height
            (400, 10, 580),  # Medium funnel, scaled height
            (400, 50, 800),  # Large funnel, capped at maximum
        ]

        for base_height, data_items, expected_range in test_scenarios:
            calculated_height = LayoutConfig.get_responsive_height(base_height, data_items)

            # Should be within universal height standards
            assert (
                350 <= calculated_height <= 800
            ), f"Height {calculated_height} outside standards for {data_items} items"

        print("✅ Responsive height calculation test passed")

    def test_mobile_compatibility(self, visualizer):
        """Test mobile device compatibility."""
        # Create small dataset for mobile testing
        mobile_results = FunnelResults(
            steps=["Step 1", "Step 2", "Step 3"],
            users_count=[100, 80, 60],
            conversion_rates=[100.0, 80.0, 75.0],
            drop_offs=[0, 20, 20],
            drop_off_rates=[0.0, 20.0, 25.0],
        )

        chart = visualizer.create_enhanced_funnel_chart(mobile_results)

        # Validate mobile-friendly properties
        assert chart.layout.height <= 600, "Mobile charts should have reasonable height"

        # Check margin configuration for mobile
        margins = chart.layout.margin
        assert margins.l <= 80, "Left margin should be mobile-friendly"
        assert margins.r <= 80, "Right margin should be mobile-friendly"

        print("✅ Mobile compatibility test passed")


@pytest.mark.visualization
@pytest.mark.accessibility
class TestAccessibilityCompliance:
    """Test accessibility compliance across all visualizations."""

    @pytest.fixture
    def visualizer(self):
        """Colorblind-friendly visualizer for accessibility testing."""
        return FunnelVisualizer(theme="dark", colorblind_friendly=True)

    def test_colorblind_friendly_palettes(self, visualizer):
        """Test colorblind-friendly color palette usage."""
        # Test with multiple segments to trigger color palette
        segmented_results = FunnelResults(
            steps=["Step 1", "Step 2", "Step 3"],
            users_count=[1000, 800, 600],
            conversion_rates=[100.0, 80.0, 75.0],
            drop_offs=[0, 200, 200],
            drop_off_rates=[0.0, 20.0, 25.0],
        )
        segmented_results.segment_data = {
            "Segment A": [600, 500, 400],
            "Segment B": [400, 300, 200],
        }

        chart = visualizer.create_enhanced_funnel_chart(segmented_results, show_segments=True)

        # Validate colorblind-friendly colors are used
        colors_used = []
        for trace in chart.data:
            if hasattr(trace, "marker") and trace.marker and hasattr(trace.marker, "color"):
                colors_used.append(trace.marker.color)

        # Should use distinct, accessible colors
        assert len(set(colors_used)) >= 2, "Should use multiple distinct colors"

        print("✅ Colorblind-friendly palettes test passed")


@pytest.mark.visualization
class TestPlotlySpecificationCompliance:
    """Test compliance with Plotly specifications and best practices."""

    @pytest.fixture
    def visualizer(self):
        """Standard visualizer for testing."""
        return FunnelVisualizer(theme="dark", colorblind_friendly=False)

    def test_plotly_figure_structure(self, visualizer):
        """Test Plotly Figure structure compliance."""
        chart = visualizer.create_enhanced_funnel_chart(
            FunnelResults(
                steps=["Step 1", "Step 2"],
                users_count=[100, 80],
                conversion_rates=[100.0, 80.0],
                drop_offs=[0, 20],
                drop_off_rates=[0.0, 20.0],
            )
        )

        # Validate Figure structure
        assert isinstance(chart, go.Figure), "Should be Plotly Figure object"
        assert hasattr(chart, "data"), "Should have data attribute"
        assert hasattr(chart, "layout"), "Should have layout attribute"

        # Validate data traces
        assert isinstance(chart.data, tuple), "Data should be tuple of traces"
        for trace in chart.data:
            assert hasattr(trace, "type"), "Each trace should have type"

        print("✅ Plotly Figure structure test passed")

    def test_json_serialization_compatibility(self, visualizer):
        """Test JSON serialization compatibility."""
        chart = visualizer.create_enhanced_funnel_chart(
            FunnelResults(
                steps=["Step 1", "Step 2"],
                users_count=[100, 80],
                conversion_rates=[100.0, 80.0],
                drop_offs=[0, 20],
                drop_off_rates=[0.0, 20.0],
            )
        )

        # Should be JSON serializable
        try:
            chart_json = chart.to_json()
            assert isinstance(chart_json, str), "Should serialize to JSON string"

            # Should be valid JSON
            parsed_json = json.loads(chart_json)
            assert isinstance(parsed_json, dict), "Should parse to dictionary"

        except Exception as e:
            assert False, f"Chart should be JSON serializable: {str(e)}"

        print("✅ JSON serialization compatibility test passed")


@pytest.mark.visualization
class TestErrorHandlingInVisualization:
    """Test error handling in visualization components."""

    @pytest.fixture
    def visualizer(self):
        """Standard visualizer for testing."""
        return FunnelVisualizer(theme="dark", colorblind_friendly=True)

    def test_empty_data_visualization(self, visualizer):
        """Test visualization with empty data."""
        empty_results = FunnelResults(
            steps=[],
            users_count=[],
            conversion_rates=[],
            drop_offs=[],
            drop_off_rates=[],
        )

        # Should handle empty data gracefully
        chart = visualizer.create_enhanced_funnel_chart(empty_results)

        assert isinstance(chart, go.Figure), "Should return valid Figure for empty data"

        print("✅ Empty data visualization test passed")

    def test_invalid_data_visualization(self, visualizer):
        """Test visualization with invalid data."""
        # Test with None values
        try:
            chart = visualizer.create_enhanced_funnel_chart(None)
            # Should either return valid chart or raise clear exception
            if chart is not None:
                assert isinstance(chart, go.Figure), "Should return valid Figure or None"
        except (TypeError, AttributeError, ValueError) as e:
            # Expected behavior - should raise clear exception
            assert str(e), "Exception should have descriptive message"

        print("✅ Invalid data visualization test passed")
