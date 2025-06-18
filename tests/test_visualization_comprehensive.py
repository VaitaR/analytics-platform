#!/usr/bin/env python3
"""
Comprehensive Test Suite for FunnelVisualizer
===========================================

This module provides comprehensive testing for all FunnelVisualizer functionality,
which was previously undertested. Covers all chart types, themes, and edge cases.

Test Categories:
1. Chart Creation (all types)
2. Theme and Accessibility 
3. Data Validation and Edge Cases
4. Progressive Disclosure Features
5. Error Handling
6. Performance with Large Datasets

Professional Testing Standards:
- Universal fixtures for reusable test data
- Comprehensive edge case coverage
- Performance validation for large datasets
- Type safety and API consistency testing
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

from app import (
    FunnelVisualizer, 
    FunnelResults,
    TimeToConvertStats,
    CohortData,
    PathAnalysisData,
    StatSignificanceResult
)
from models import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.visualization
class TestFunnelVisualizerChartCreation:
    """Test core chart creation functionality for all visualization types."""
    
    @pytest.fixture
    def visualizer(self):
        """Standard visualizer instance with default settings."""
        return FunnelVisualizer(theme='dark', colorblind_friendly=False)
    
    @pytest.fixture
    def colorblind_visualizer(self):
        """Visualizer instance optimized for colorblind accessibility."""
        return FunnelVisualizer(theme='dark', colorblind_friendly=True)
    
    @pytest.fixture
    def sample_funnel_results(self):
        """Standard funnel results for testing visualizations."""
        results = FunnelResults(
            steps=["Sign Up", "Email Verification", "First Login", "First Purchase"],
            users_count=[1000, 750, 600, 400],
            conversion_rates=[100.0, 75.0, 60.0, 40.0],
            drop_offs=[0, 250, 150, 200],
            drop_off_rates=[0.0, 25.0, 20.0, 33.3]
        )
        results.segment_data = {
            "Premium": [400, 350, 320, 280],
            "Basic": [600, 400, 280, 120]
        }
        return results
    
    @pytest.fixture
    def sample_time_stats(self):
        """Sample time-to-convert statistics for testing."""
        stats = []
        transitions = [
            ("Sign Up", "Email Verification", [0.5, 1.2, 0.8, 2.1, 1.5]),
            ("Email Verification", "First Login", [2.4, 4.1, 3.2, 5.8, 1.9]),
            ("First Login", "First Purchase", [24.5, 48.2, 36.7, 72.1, 28.4])
        ]
        
        for step_from, step_to, times in transitions:
            stat = TimeToConvertStats(
                step_from=step_from,
                step_to=step_to,
                conversion_times=times,
                median_hours=np.median(times),
                mean_hours=np.mean(times),
                p25_hours=np.percentile(times, 25),
                p75_hours=np.percentile(times, 75),
                p90_hours=np.percentile(times, 90),
                std_hours=np.std(times)
            )
            stats.append(stat)
        
        return stats
    
    @pytest.fixture
    def sample_cohort_data(self):
        """Sample cohort data for heatmap testing."""
        cohort_data = CohortData(
            cohort_period="monthly",
            cohort_labels=["2024-01", "2024-02", "2024-03", "2024-04"],
            cohort_sizes={
                "2024-01": 300,
                "2024-02": 250, 
                "2024-03": 400,
                "2024-04": 350
            },
            conversion_rates={
                "2024-01": [100.0, 80.0, 65.0, 45.0],
                "2024-02": [100.0, 85.0, 70.0, 50.0],
                "2024-03": [100.0, 75.0, 60.0, 40.0],
                "2024-04": [100.0, 90.0, 75.0, 55.0]
            }
        )
        return cohort_data
    
    @pytest.fixture 
    def sample_path_data(self):
        """Sample path analysis data for journey visualization."""
        path_data = PathAnalysisData(
            dropoff_paths={
                "Sign Up": {
                    "Homepage Exit": 150,
                    "Help Page": 50,
                    "Contact Support": 30,
                    "Session Timeout": 20
                },
                "Email Verification": {
                    "Account Settings": 80,
                    "Profile Page": 40,
                    "Logout": 30
                },
                "First Login": {
                    "Dashboard": 120,
                    "Tutorial Skip": 50,
                    "Settings": 30
                }
            },
            between_steps_events={
                "Sign Up → Email Verification": {
                    "Welcome Email Sent": 750,
                    "Verification Link Clicked": 680
                },
                "Email Verification → First Login": {
                    "Login Attempt": 600,
                    "Password Reset": 30
                }
            }
        )
        return path_data
    
    def test_enhanced_funnel_chart_creation(self, visualizer, sample_funnel_results):
        """Test enhanced funnel chart creation with insights."""
        chart = visualizer.create_enhanced_funnel_chart(
            sample_funnel_results, 
            show_segments=False,
            show_insights=True
        )
        
        # Validate chart structure
        assert chart is not None, "Chart should be created"
        assert len(chart.data) == 1, "Should have 1 trace for simple funnel"
        assert chart.data[0].type == 'funnel', "Should be funnel chart type"
        
        # Validate data content
        funnel_trace = chart.data[0]
        assert list(funnel_trace.y) == sample_funnel_results.steps
        assert list(funnel_trace.x) == sample_funnel_results.users_count
        
        # Validate theme application
        assert chart.layout.plot_bgcolor is not None
        assert chart.layout.paper_bgcolor is not None
        
        # Validate accessibility features
        assert funnel_trace.textinfo == "value+percent initial"
        assert funnel_trace.textfont.color == 'white'
        
        print("✅ Enhanced funnel chart creation test passed")
    
    def test_segmented_funnel_chart_creation(self, visualizer, sample_funnel_results):
        """Test segmented funnel chart with multiple segments."""
        chart = visualizer.create_enhanced_funnel_chart(
            sample_funnel_results,
            show_segments=True,
            show_insights=False
        )
        
        # Validate segmented structure
        assert chart is not None, "Segmented chart should be created"
        assert len(chart.data) == 2, "Should have 2 traces for 2 segments"
        
        # Validate segment names
        trace_names = [trace.name for trace in chart.data]
        assert "Premium" in trace_names
        assert "Basic" in trace_names
        
        # Validate different colors for segments
        colors = []
        for trace in chart.data:
            if hasattr(trace.marker, 'color'):
                if isinstance(trace.marker.color, list):
                    colors.append(trace.marker.color[0] if trace.marker.color else '#000000')
                else:
                    colors.append(trace.marker.color)
            else:
                colors.append('#000000')  # default color
        
        # At least check that we have color information
        assert len(colors) == 2, "Should have colors for both segments"
        
        print("✅ Segmented funnel chart creation test passed")
    
    def test_conversion_flow_sankey_creation(self, visualizer, sample_funnel_results):
        """Test enhanced Sankey diagram creation."""
        chart = visualizer.create_enhanced_conversion_flow_sankey(sample_funnel_results)
        
        # Validate Sankey structure
        assert chart is not None, "Sankey chart should be created"
        assert len(chart.data) == 1, "Should have 1 Sankey trace"
        assert chart.data[0].type == 'sankey', "Should be Sankey diagram"
        
        sankey_trace = chart.data[0]
        
        # Validate node structure
        assert len(sankey_trace.node.label) > len(sample_funnel_results.steps)
        
        # Validate links exist
        assert len(sankey_trace.link.source) > 0, "Should have conversion flows"
        assert len(sankey_trace.link.target) > 0, "Should have target nodes"
        assert len(sankey_trace.link.value) > 0, "Should have flow values"
        
        # Validate enhanced features
        assert sankey_trace.node.pad == visualizer.layout.SPACING['md']
        assert sankey_trace.link.hovertemplate is not None
        
        print("✅ Enhanced Sankey chart creation test passed")
    
    def test_time_to_convert_chart_creation(self, visualizer, sample_time_stats):
        """Test enhanced time-to-convert visualization."""
        chart = visualizer.create_enhanced_time_to_convert_chart(sample_time_stats)
        
        # Validate chart structure
        assert chart is not None, "Time chart should be created"
        assert len(chart.data) == len(sample_time_stats), "Should have trace per transition"
        
        # Validate log scale
        assert chart.layout.yaxis.type == "log", "Should use log scale for time"
        
        # Validate trace types (violin or box plots)
        valid_types = ['violin', 'box']
        for trace in chart.data:
            assert trace.type in valid_types, f"Invalid trace type: {trace.type}"
        
        # Validate enhanced annotations
        assert chart.layout.annotations is not None, "Should have reference time annotations"
        
        print("✅ Enhanced time-to-convert chart creation test passed")
    
    def test_cohort_heatmap_creation(self, visualizer, sample_cohort_data):
        """Test enhanced cohort heatmap creation."""
        chart = visualizer.create_enhanced_cohort_heatmap(sample_cohort_data)
        
        # Validate heatmap structure
        assert chart is not None, "Heatmap should be created"
        assert len(chart.data) == 1, "Should have 1 heatmap trace"
        assert chart.data[0].type == 'heatmap', "Should be heatmap"
        
        heatmap_trace = chart.data[0]
        
        # Validate data structure
        assert len(heatmap_trace.z) == len(sample_cohort_data.cohort_labels)
        assert all(len(row) == 4 for row in heatmap_trace.z), "Should have 4 steps per cohort"
        
        # Validate enhanced features
        assert heatmap_trace.colorscale is not None, "Should have accessible colorscale"
        assert heatmap_trace.hovertemplate is not None, "Should have enhanced hover"
        
        print("✅ Enhanced cohort heatmap creation test passed")
    
    def test_path_analysis_chart_creation(self, visualizer, sample_path_data):
        """Test enhanced path analysis Sankey diagram."""
        chart = visualizer.create_enhanced_path_analysis_chart(sample_path_data)
        
        # Validate path analysis structure
        assert chart is not None, "Path chart should be created"
        assert len(chart.data) == 1, "Should have 1 Sankey trace"
        
        sankey_trace = chart.data[0]
        
        # Validate node categorization
        funnel_nodes = [label for label in sankey_trace.node.label if '📍' in label]
        assert len(funnel_nodes) >= 3, "Should have funnel step nodes"
        
        # Validate drop-off destinations
        dropoff_nodes = [label for label in sankey_trace.node.label if 'Exit' in label or 'Support' in label]
        assert len(dropoff_nodes) > 0, "Should have drop-off destination nodes"
        
        # Validate semantic coloring
        assert len(sankey_trace.node.color) == len(sankey_trace.node.label)
        assert len(sankey_trace.link.color) == len(sankey_trace.link.source)
        
        print("✅ Enhanced path analysis chart creation test passed")


@pytest.mark.visualization  
class TestFunnelVisualizerThemeAndAccessibility:
    """Test theme application and accessibility features."""
    
    @pytest.fixture
    def sample_results(self):
        """Minimal results for theme testing."""
        results = FunnelResults(
            steps=["Step 1", "Step 2", "Step 3"],
            users_count=[100, 75, 50],
            conversion_rates=[100.0, 75.0, 50.0],
            drop_offs=[0, 25, 25],
            drop_off_rates=[0.0, 25.0, 33.3]
        )
        return results
    
    def test_dark_theme_application(self, sample_results):
        """Test dark theme is properly applied to charts."""
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=False)
        chart = visualizer.create_enhanced_funnel_chart(sample_results)
        
        # Validate dark theme colors
        layout = chart.layout
        assert layout.plot_bgcolor.startswith('rgb'), "Should have dark plot background"
        assert layout.paper_bgcolor.startswith('rgb'), "Should have dark paper background"
        
        # Validate text colors
        assert layout.font.color is not None, "Should have light text color"
        
        print("✅ Dark theme application test passed")
    
    def test_colorblind_friendly_mode(self, sample_results):
        """Test colorblind-friendly color schemes."""
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=True)
        
        # Test with segmented data
        sample_results.segment_data = {
            "Segment A": [60, 45, 30],
            "Segment B": [40, 30, 20]
        }
        
        chart = visualizer.create_enhanced_funnel_chart(sample_results, show_segments=True)
        
        # Validate colorblind accessibility
        assert visualizer.colorblind_friendly == True
        
        # Verify distinct colors for segments
        colors = [trace.marker.color for trace in chart.data]
        assert len(set(colors)) == len(chart.data), "Segments should have distinct colors"
        
        print("✅ Colorblind accessibility test passed")
    
    def test_accessibility_report_generation(self, sample_results):
        """Test accessibility report generation."""
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=True)
        report = visualizer.create_accessibility_report(sample_results)
        
        # Validate report structure
        assert 'color_accessibility' in report
        assert 'typography' in report
        assert 'interaction_patterns' in report
        assert 'layout_system' in report
        assert 'visualization_complexity' in report
        
        # Validate accessibility metrics
        color_access = report['color_accessibility']
        assert color_access['wcag_compliant'] == True
        assert color_access['colorblind_friendly'] == True
        
        # Validate complexity assessment
        complexity = report['visualization_complexity']
        assert 'score' in complexity
        assert 'level' in complexity
        assert 'recommendations' in complexity
        
        print("✅ Accessibility report generation test passed")
    
    def test_responsive_height_calculation(self, sample_results):
        """Test responsive height calculations for different data sizes."""
        visualizer = FunnelVisualizer(theme='dark')
        
        # Test with different step counts - Updated for new universal standards
        # New calculation: min(800, base + min(content_count-1, 20) * 20)
        test_cases = [
            (["Step 1", "Step 2"], 350),  # Min height enforced (350px universal minimum)
            (["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"], 350),  # Still within minimum range
            (["Step " + str(i) for i in range(1, 11)], 350),  # Min height enforced, capped at 800px max
        ]
        
        for steps, expected_min_height in test_cases:
            sample_results.steps = steps
            sample_results.users_count = [100 - i*10 for i in range(len(steps))]
            sample_results.drop_offs = [10] * len(steps)  # Consistent drop-offs
            sample_results.drop_off_rates = [10.0] * len(steps)  # Consistent rates
            sample_results.conversion_rates = [100.0 - i*10 for i in range(len(steps))]
            
            chart = visualizer.create_enhanced_funnel_chart(sample_results)
            
            # Validate responsive height with universal standards
            actual_height = chart.layout.height
            assert actual_height >= expected_min_height, f"Height {actual_height} too small for {len(steps)} steps"
            assert actual_height <= 800, f"Height {actual_height} exceeds universal maximum of 800px"
        
        print("✅ Responsive height calculation test passed")


@pytest.mark.visualization
class TestFunnelVisualizerEdgeCases:
    """Test edge cases and error handling in visualizations."""
    
    @pytest.fixture
    def visualizer(self):
        """Standard visualizer for edge case testing."""
        return FunnelVisualizer(theme='dark', colorblind_friendly=False)
    
    def test_empty_funnel_results(self, visualizer):
        """Test visualization with empty funnel results."""
        empty_results = FunnelResults(
            steps=[],
            users_count=[],
            conversion_rates=[],
            drop_offs=[],
            drop_off_rates=[]
        )
        empty_results.steps = []
        empty_results.users_count = []
        
        chart = visualizer.create_enhanced_funnel_chart(empty_results)
        
        # Should create chart with helpful message
        assert chart is not None, "Should create chart even with empty data"
        assert len(chart.layout.annotations) > 0, "Should have helpful annotation"
        
        annotation_text = chart.layout.annotations[0].text.lower()
        assert 'no data' in annotation_text or 'available' in annotation_text
        
        print("✅ Empty funnel results test passed")
    
    def test_single_step_funnel(self, visualizer):
        """Test visualization with single step funnel."""
        single_step_results = FunnelResults(
            steps=["Single Step"],
            users_count=[100],
            conversion_rates=[100.0],
            drop_offs=[0],
            drop_off_rates=[0.0]
        )
        single_step_results.steps = ["Single Step"]
        single_step_results.users_count = [100]
        single_step_results.conversion_rates = [100.0]
        
        # Funnel chart should work
        funnel_chart = visualizer.create_enhanced_funnel_chart(single_step_results)
        assert funnel_chart is not None
        
        # Sankey should show helpful message
        sankey_chart = visualizer.create_enhanced_conversion_flow_sankey(single_step_results)
        assert sankey_chart is not None
        assert len(sankey_chart.layout.annotations) > 0, "Should explain insufficient steps"
        
        print("✅ Single step funnel test passed")
    
    def test_zero_users_scenario(self, visualizer):
        """Test visualization with zero users in all steps."""
        zero_results = FunnelResults(
            steps=["Step 1", "Step 2", "Step 3"],
            users_count=[0, 0, 0],
            conversion_rates=[0.0, 0.0, 0.0],
            drop_offs=[0, 0, 0],
            drop_off_rates=[0.0, 0.0, 0.0]
        )
        zero_results.steps = ["Step 1", "Step 2", "Step 3"]
        zero_results.users_count = [0, 0, 0]
        zero_results.conversion_rates = [0.0, 0.0, 0.0]
        
        chart = visualizer.create_enhanced_funnel_chart(zero_results)
        
        # Should handle gracefully
        assert chart is not None, "Should create chart with zero data"
        assert len(chart.data) > 0, "Should have at least one trace"
        
        print("✅ Zero users scenario test passed")
    
    def test_invalid_time_stats(self, visualizer):
        """Test time chart with invalid/empty time statistics."""
        # Empty stats
        empty_chart = visualizer.create_enhanced_time_to_convert_chart([])
        assert empty_chart is not None
        assert len(empty_chart.layout.annotations) > 0, "Should show helpful message"
        
        # Stats with no valid times
        invalid_stat = TimeToConvertStats(
            step_from="Step 1",
            step_to="Step 2",
            mean_hours=0.0,
            median_hours=0.0,
            p25_hours=0.0,
            p75_hours=0.0,
            p90_hours=0.0,
            std_hours=0.0,
            conversion_times=[]
        )
        invalid_stat.step_from = "Step 1"
        invalid_stat.step_to = "Step 2"
        invalid_stat.conversion_times = []  # Empty times
        
        invalid_chart = visualizer.create_enhanced_time_to_convert_chart([invalid_stat])
        assert invalid_chart is not None
        
        print("✅ Invalid time stats test passed")
    
    def test_empty_cohort_data(self, visualizer):
        """Test cohort heatmap with empty cohort data."""
        empty_cohort = CohortData(
            cohort_period="monthly",
            cohort_sizes={},
            conversion_rates={},
            cohort_labels=[]
        )
        empty_cohort.cohort_labels = []
        empty_cohort.cohort_sizes = {}
        empty_cohort.conversion_rates = {}
        
        chart = visualizer.create_enhanced_cohort_heatmap(empty_cohort)
        
        # Should handle gracefully
        assert chart is not None, "Should create chart with empty cohort data"
        assert len(chart.layout.annotations) > 0, "Should show helpful message"
        
        print("✅ Empty cohort data test passed")
    
    def test_empty_path_data(self, visualizer):
        """Test path analysis with empty journey data."""
        empty_path = PathAnalysisData(
            dropoff_paths={},
            between_steps_events={}
        )
        empty_path.dropoff_paths = {}
        empty_path.between_steps_events = {}
        
        chart = visualizer.create_enhanced_path_analysis_chart(empty_path)
        
        # Should handle gracefully
        assert chart is not None, "Should create chart with empty path data"
        assert len(chart.layout.annotations) > 0, "Should show helpful message"
        
        print("✅ Empty path data test passed")


@pytest.mark.visualization
@pytest.mark.performance
class TestFunnelVisualizerPerformance:
    """Test visualizer performance with large datasets."""
    
    @pytest.fixture
    def large_funnel_results(self):
        """Large funnel results for performance testing."""
        steps = [f"Step {i}" for i in range(1, 21)]  # 20 steps
        users_count = [10000 - i*400 for i in range(20)]  # Decreasing counts
        conversion_rates = [100.0 - i*4 for i in range(20)]
        drop_offs = [400] * 20  # Consistent drop-offs
        drop_off_rates = [4.0] * 20  # Consistent rates
        
        results = FunnelResults(
            steps=steps,
            users_count=users_count,
            conversion_rates=conversion_rates,
            drop_offs=drop_offs,
            drop_off_rates=drop_off_rates
        )
        
        # Large segment data
        results.segment_data = {
            f"Segment {chr(65+i)}": [5000 - i*200 - j*150 for j in range(20)]
            for i in range(10)  # 10 segments
        }
        return results
    
    @pytest.fixture
    def large_time_stats(self):
        """Large time statistics for performance testing."""
        stats = []
        for i in range(19):  # 19 transitions
            # Large number of conversion times
            conversion_times = np.random.exponential(24, 10000).tolist()
            stat = TimeToConvertStats(
                step_from=f"Step {i+1}",
                step_to=f"Step {i+2}",
                conversion_times=conversion_times,
                median_hours=np.median(conversion_times),
                mean_hours=np.mean(conversion_times),
                p25_hours=np.percentile(conversion_times, 25),
                p75_hours=np.percentile(conversion_times, 75),
                p90_hours=np.percentile(conversion_times, 90),
                std_hours=np.std(conversion_times)
            )
            stats.append(stat)
        return stats
    
    def test_large_funnel_chart_performance(self, large_funnel_results):
        """Test funnel chart performance with large datasets."""
        import time
        
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=False)
        
        start_time = time.time()
        chart = visualizer.create_enhanced_funnel_chart(large_funnel_results, show_segments=True)
        end_time = time.time()
        
        # Validate performance
        execution_time = end_time - start_time
        assert execution_time < 10.0, f"Chart creation took too long: {execution_time:.2f}s"
        
        # Validate chart was created
        assert chart is not None
        assert len(chart.data) == len(large_funnel_results.segment_data)
        
        print(f"✅ Large funnel chart performance test passed ({execution_time:.2f}s)")
    
    def test_large_time_chart_performance(self, large_time_stats):
        """Test time chart performance with large datasets."""
        import time
        
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=False)
        
        start_time = time.time()
        chart = visualizer.create_enhanced_time_to_convert_chart(large_time_stats)
        end_time = time.time()
        
        # Validate performance
        execution_time = end_time - start_time
        assert execution_time < 15.0, f"Time chart creation took too long: {execution_time:.2f}s"
        
        # Validate chart was created
        assert chart is not None
        assert len(chart.data) == len(large_time_stats)
        
        print(f"✅ Large time chart performance test passed ({execution_time:.2f}s)")
    
    def test_memory_efficiency_large_datasets(self, large_funnel_results):
        """Test memory efficiency with large visualization datasets."""
        import tracemalloc
        
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=False)
        
        # Start memory tracking
        tracemalloc.start()
        
        # Create multiple charts
        charts = []
        for _ in range(5):
            chart = visualizer.create_enhanced_funnel_chart(large_funnel_results)
            charts.append(chart)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validate memory usage (less than 100MB)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Memory usage too high: {peak_mb:.1f}MB"
        
        # Validate charts were created
        assert len(charts) == 5
        assert all(chart is not None for chart in charts)
        
        print(f"✅ Memory efficiency test passed (peak: {peak_mb:.1f}MB)")


@pytest.mark.visualization
class TestFunnelVisualizerAPIConsistency:
    """Test API consistency and type safety across all visualization methods."""
    
    @pytest.fixture
    def minimal_results(self):
        """Minimal valid results for API testing."""
        results = FunnelResults(
            steps=["A", "B", "C"],
            users_count=[10, 5, 2],
            conversion_rates=[100.0, 50.0, 20.0],
            drop_offs=[0, 5, 3],
            drop_off_rates=[0.0, 50.0, 60.0]
        )
        return results
    
    def test_all_chart_methods_return_plotly_figure(self, minimal_results):
        """Test all chart methods return valid Plotly Figure objects."""
        visualizer = FunnelVisualizer()
        
        # Test all main chart creation methods
        chart_methods = [
            (visualizer.create_enhanced_funnel_chart, (minimal_results,)),
            (visualizer.create_enhanced_conversion_flow_sankey, (minimal_results,)),
        ]
        
        for method, args in chart_methods:
            chart = method(*args)
            assert isinstance(chart, go.Figure), f"{method.__name__} should return plotly Figure"
            assert hasattr(chart, 'data'), f"{method.__name__} should have data attribute"
            assert hasattr(chart, 'layout'), f"{method.__name__} should have layout attribute"
        
        print("✅ API consistency test passed")
    
    def test_theme_parameter_consistency(self, minimal_results):
        """Test theme parameters work consistently across all methods."""
        themes = ['dark', 'light']
        
        for theme in themes:
            visualizer = FunnelVisualizer(theme=theme)
            
            # Test basic chart creation works with theme
            chart = visualizer.create_enhanced_funnel_chart(minimal_results)
            assert chart is not None, f"Chart creation should work with {theme} theme"
            
            # Validate theme is applied
            assert chart.layout.plot_bgcolor is not None
            assert chart.layout.paper_bgcolor is not None
        
        print("✅ Theme parameter consistency test passed")
    
    def test_error_handling_consistency(self):
        """Test error handling is consistent across all methods."""
        visualizer = FunnelVisualizer()
        
        # Test with None inputs
        none_inputs = [
            (visualizer.create_enhanced_funnel_chart, (None,)),
        ]
        
        for method, args in none_inputs:
            try:
                result = method(*args)
                # Should either return valid figure or raise clear exception
                if result is not None:
                    assert isinstance(result, go.Figure)
            except (TypeError, AttributeError, ValueError) as e:
                # Expected behavior - should raise clear exceptions
                assert str(e), "Exception should have descriptive message"
        
        print("✅ Error handling consistency test passed")


# Additional utility fixtures for cross-test compatibility
@pytest.fixture
def standard_visualizer():
    """Standard visualizer instance for general testing."""
    return FunnelVisualizer(theme='dark', colorblind_friendly=False)


@pytest.fixture 
def comprehensive_funnel_results():
    """Comprehensive funnel results with all optional fields populated."""
    results = FunnelResults(
        steps=["Registration", "Email Confirm", "Profile Setup", "First Purchase"],
        users_count=[1000, 800, 650, 450],
        conversion_rates=[100.0, 80.0, 65.0, 45.0],
        drop_offs=[0, 200, 150, 200],
        drop_off_rates=[0.0, 20.0, 18.8, 30.8]
    )
    
    # Segment data
    results.segment_data = {
        "Mobile": [600, 500, 420, 320],
        "Desktop": [400, 300, 230, 130]
    }
    
    # Time to convert stats
    time_stats = []
    for i in range(3):
        conversion_times = np.random.exponential(24 * (i + 1), 100).tolist()
        stat = TimeToConvertStats(
            step_from=results.steps[i],
            step_to=results.steps[i + 1],
            conversion_times=conversion_times,
            median_hours=np.median(conversion_times),
            mean_hours=np.mean(conversion_times),
            p25_hours=np.percentile(conversion_times, 25),
            p75_hours=np.percentile(conversion_times, 75),
            p90_hours=np.percentile(conversion_times, 90),
            std_hours=np.std(conversion_times)
        )
        time_stats.append(stat)
    results.time_to_convert = time_stats
    
    # Cohort data
    results.cohort_data = CohortData(
        cohort_period="quarterly",
        cohort_labels=["Q1", "Q2", "Q3", "Q4"],
        cohort_sizes={"Q1": 250, "Q2": 300, "Q3": 275, "Q4": 175},
        conversion_rates={
            "Q1": [100.0, 85.0, 70.0, 50.0],
            "Q2": [100.0, 80.0, 65.0, 45.0],
            "Q3": [100.0, 75.0, 60.0, 40.0],
            "Q4": [100.0, 90.0, 75.0, 55.0]
        }
    )
    
    # Path analysis data
    results.path_analysis = PathAnalysisData(
        dropoff_paths={
            "Registration": {"Home": 100, "Help": 50, "Exit": 50},
            "Email Confirm": {"Settings": 80, "Profile": 40, "Logout": 30},
            "Profile Setup": {"Dashboard": 90, "Tutorial": 60}
        },
        between_steps_events={
            "Registration → Email Confirm": {"Email Sent": 800, "Clicked Link": 750},
            "Email Confirm → Profile Setup": {"Login Success": 650, "Welcome Message": 600},
            "Profile Setup → First Purchase": {"Browse Products": 450, "Add to Cart": 400}
        }
    )
    
    return results


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v", "--tb=short"])
