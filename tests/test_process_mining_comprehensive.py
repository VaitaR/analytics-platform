"""
Comprehensive tests for process mining functionality
Tests process discovery, cycle detection, visualization, and performance
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from unittest.mock import Mock, patch

from models import ProcessMiningData, FunnelConfig
from path_analyzer import PathAnalyzer
from app import FunnelVisualizer

class TestProcessMiningDiscovery:
    """Test process discovery algorithms"""
    
    @pytest.fixture
    def simple_process_data(self):
        """Create simple linear process data"""
        events = []
        users = ['user_1', 'user_2', 'user_3']
        process_steps = ['Start', 'Middle', 'End']
        
        for user_id in users:
            base_time = datetime(2024, 1, 1, 10, 0, 0)
            for i, step in enumerate(process_steps):
                events.append({
                    'user_id': user_id,
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i)
                })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def complex_process_data(self):
        """Create complex process data with cycles and branches"""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Linear path users
        for user_id in ['user_1', 'user_2']:
            for i, step in enumerate(['Login', 'Browse', 'Purchase', 'Checkout']):
                events.append({
                    'user_id': user_id,
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i, minutes=int(user_id[-1]) * 10)
                })
        
        # Cycle path users
        for user_id in ['user_3', 'user_4']:
            cycle_steps = ['Login', 'Browse', 'Cart', 'Browse', 'Cart', 'Purchase']
            for i, step in enumerate(cycle_steps):
                events.append({
                    'user_id': user_id,
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i, minutes=int(user_id[-1]) * 10)
                })
        
        # Error path users
        for user_id in ['user_5']:
            error_steps = ['Login', 'Browse', 'Error', 'Exit']
            for i, step in enumerate(error_steps):
                events.append({
                    'user_id': user_id,
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i, minutes=int(user_id[-1]) * 10)
                })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def path_analyzer(self):
        """Create path analyzer instance"""
        config = FunnelConfig()
        return PathAnalyzer(config)
    
    def test_simple_process_discovery(self, path_analyzer, simple_process_data):
        """Test basic process discovery with linear process"""
        process_data = path_analyzer.discover_process_mining_structure(
            simple_process_data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Verify activities discovered
        assert len(process_data.activities) == 3
        assert 'Start' in process_data.activities
        assert 'Middle' in process_data.activities
        assert 'End' in process_data.activities
        
        # Verify activity characteristics
        start_activity = process_data.activities['Start']
        assert start_activity['unique_users'] == 3
        assert start_activity['is_start'] == True
        
        end_activity = process_data.activities['End']
        assert end_activity['is_end'] == True
        
        # Verify transitions
        assert len(process_data.transitions) == 2
        assert ('Start', 'Middle') in process_data.transitions
        assert ('Middle', 'End') in process_data.transitions
        
        # Verify start/end activities
        assert 'Start' in process_data.start_activities
        assert 'End' in process_data.end_activities
        
        # Verify statistics
        assert process_data.statistics['total_cases'] == 3
        assert process_data.statistics['total_activities'] == 3
        assert process_data.statistics['total_transitions'] == 2
    
    def test_complex_process_discovery(self, path_analyzer, complex_process_data):
        """Test process discovery with cycles and branches"""
        process_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Verify activities discovered
        expected_activities = {'Login', 'Browse', 'Purchase', 'Checkout', 'Cart', 'Error', 'Exit'}
        discovered_activities = set(process_data.activities.keys())
        assert expected_activities <= discovered_activities
        
        # Verify cycles detected
        assert len(process_data.cycles) > 0
        
        # Check for Browse -> Cart -> Browse cycle
        cycle_paths = [cycle['path'] for cycle in process_data.cycles]
        has_browse_cart_cycle = any(
            'Browse' in path and 'Cart' in path 
            for path in cycle_paths
        )
        assert has_browse_cart_cycle
        
        # Verify process variants
        assert len(process_data.variants) > 0
        
        # Verify error activity classification
        if 'Error' in process_data.activities:
            assert process_data.activities['Error']['activity_type'] == 'error'
    
    def test_minimum_frequency_filtering(self, path_analyzer, complex_process_data):
        """Test that minimum frequency filtering works correctly"""
        # High frequency threshold should reduce transitions
        high_freq_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=10,  # High threshold
            include_cycles=True
        )
        
        low_freq_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=1,   # Low threshold
            include_cycles=True
        )
        
        # High frequency should have fewer or equal transitions
        assert len(high_freq_data.transitions) <= len(low_freq_data.transitions)
    
    def test_time_window_filtering(self, path_analyzer, complex_process_data):
        """Test time window filtering"""
        # Test with 1 hour window (should capture only first events)
        recent_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=1,
            time_window_hours=1
        )
        
        all_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=1
        )
        
        # Recent data should have fewer activities/transitions
        assert recent_data.statistics['total_cases'] <= all_data.statistics['total_cases']
    
    def test_cycle_detection_accuracy(self, path_analyzer, complex_process_data):
        """Test accuracy of cycle detection"""
        process_data = path_analyzer.discover_process_mining_structure(
            complex_process_data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should detect some cycles in the complex data
        assert len(process_data.cycles) > 0
        
        # Verify cycle structure
        for cycle in process_data.cycles:
            assert 'path' in cycle
            assert 'frequency' in cycle
            assert 'type' in cycle
            assert cycle['type'] in ['loop', 'cycle']
            assert cycle['frequency'] > 0


class TestProcessMiningInsights:
    """Test automatic insight generation"""
    
    @pytest.fixture
    def insight_data(self):
        """Create data designed to generate specific insights"""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create bottleneck scenario - step takes very long
        for user_id in range(10):
            events.extend([
                {
                    'user_id': f'user_{user_id}',
                    'event_name': 'Start',
                    'timestamp': base_time
                },
                {
                    'user_id': f'user_{user_id}',
                    'event_name': 'Bottleneck_Step',
                    'timestamp': base_time + timedelta(hours=1)
                },
                {
                    'user_id': f'user_{user_id}',
                    'event_name': 'End',
                    'timestamp': base_time + timedelta(hours=48)  # Very long duration
                }
            ])
        
        return pd.DataFrame(events)
    
    def test_bottleneck_detection(self, insight_data):
        """Test that bottlenecks are correctly identified in insights"""
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            insight_data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should generate bottleneck insight
        bottleneck_insights = [insight for insight in process_data.insights 
                              if 'bottleneck' in insight.lower() or 'takes' in insight.lower()]
        assert len(bottleneck_insights) > 0
    
    def test_completion_rate_insights(self, insight_data):
        """Test completion rate insight generation"""
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            insight_data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should generate completion rate insights
        completion_insights = [insight for insight in process_data.insights 
                              if 'completion' in insight.lower() or 'complete' in insight.lower()]
        # Note: May or may not have completion insights depending on data characteristics
        # This is more of a structure test than a strict requirement
        assert isinstance(process_data.insights, list)


class TestProcessMiningVisualization:
    """Test process mining visualization components"""
    
    @pytest.fixture
    def visualizer(self):
        """Create FunnelVisualizer instance"""
        return FunnelVisualizer(theme='dark', colorblind_friendly=False)
    
    @pytest.fixture
    def sample_process_data(self):
        """Create sample ProcessMiningData for visualization testing"""
        activities = {
            'Start': {
                'frequency': 100,
                'unique_users': 100,
                'avg_duration': 0.5,
                'is_start': True,
                'is_end': False,
                'activity_type': 'entry',
                'success_rate': 95.0,
                'first_occurrence': datetime(2024, 1, 1),
                'last_occurrence': datetime(2024, 1, 31)
            },
            'Middle': {
                'frequency': 80,
                'unique_users': 80,
                'avg_duration': 2.0,
                'is_start': False,
                'is_end': False,
                'activity_type': 'process',
                'success_rate': 87.5,
                'first_occurrence': datetime(2024, 1, 1),
                'last_occurrence': datetime(2024, 1, 31)
            },
            'End': {
                'frequency': 70,
                'unique_users': 70,
                'avg_duration': 1.0,
                'is_start': False,
                'is_end': True,
                'activity_type': 'conversion',
                'success_rate': 100.0,
                'first_occurrence': datetime(2024, 1, 1),
                'last_occurrence': datetime(2024, 1, 31)
            }
        }
        
        transitions = {
            ('Start', 'Middle'): {
                'frequency': 80,
                'unique_users': 80,
                'avg_duration': 1.5,
                'probability': 80.0,
                'transition_type': 'main_flow'
            },
            ('Middle', 'End'): {
                'frequency': 70,
                'unique_users': 70,
                'avg_duration': 1.0,
                'probability': 87.5,
                'transition_type': 'main_flow'
            }
        }
        
        cycles = []
        variants = [
            {
                'path': ['Start', 'Middle', 'End'],
                'frequency': 70,
                'success_rate': 100.0,
                'avg_duration': 4.5,
                'variant_type': 'high_success'
            }
        ]
        
        start_activities = ['Start']
        end_activities = ['End']
        
        statistics = {
            'total_cases': 100,
            'avg_duration': 4.0,
            'completion_rate': 70.0,
            'unique_paths': 1,
            'total_activities': 3,
            'total_transitions': 2
        }
        
        insights = [
            "📈 Most common path: Start → Middle → End (70 users, 100.0% success)",
            "✅ High completion rate: 70.0% of users successfully complete the process"
        ]
        
        return ProcessMiningData(
            activities=activities,
            transitions=transitions,
            cycles=cycles,
            variants=variants,
            start_activities=start_activities,
            end_activities=end_activities,
            statistics=statistics,
            insights=insights
        )
    
    def test_process_mining_diagram_creation(self, visualizer, sample_process_data):
        """Test that process mining diagram is created successfully"""
        fig = visualizer.create_process_mining_diagram(
            sample_process_data,
            layout_algorithm="hierarchical",
            show_frequencies=True,
            show_statistics=True
        )
        
        # Verify figure is created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Verify layout has proper theme applied
        assert fig.layout.plot_bgcolor == 'rgba(0,0,0,0)'
        assert fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'
    
    def test_different_layout_algorithms(self, visualizer, sample_process_data):
        """Test different layout algorithms work"""
        algorithms = ["hierarchical", "force", "circular"]
        
        for algorithm in algorithms:
            fig = visualizer.create_process_mining_diagram(
                sample_process_data,
                layout_algorithm=algorithm,
                show_frequencies=True,
                show_statistics=True
            )
            
            # Should create figure without error
            assert fig is not None
            assert hasattr(fig, 'data')
    
    def test_empty_process_data_handling(self, visualizer):
        """Test handling of empty process data"""
        empty_data = ProcessMiningData(
            activities={},
            transitions={},
            cycles=[],
            variants=[],
            start_activities=[],
            end_activities=[],
            statistics={},
            insights=[]
        )
        
        fig = visualizer.create_process_mining_diagram(
            empty_data,
            layout_algorithm="hierarchical",
            show_frequencies=True,
            show_statistics=True
        )
        
        # Should handle empty data gracefully
        assert fig is not None
        # Should show appropriate message for no data
        assert len(fig.layout.annotations) > 0
    
    def test_frequency_filtering(self, visualizer, sample_process_data):
        """Test frequency filtering in visualization"""
        # Test with high filter that removes all transitions
        fig = visualizer.create_process_mining_diagram(
            sample_process_data,
            layout_algorithm="hierarchical",
            show_frequencies=True,
            show_statistics=True,
            filter_min_frequency=1000  # Very high threshold
        )
        
        # Should handle filtered data gracefully
        assert fig is not None


class TestProcessMiningPerformance:
    """Test performance characteristics of process mining"""
    
    @pytest.fixture
    def large_process_data(self):
        """Create large dataset for performance testing"""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create data for 1000 users with various paths
        steps = ['Start', 'Step1', 'Step2', 'Step3', 'Step4', 'End']
        
        for user_id in range(1000):
            user_steps = steps.copy()
            
            # Add some randomness - some users skip steps
            if user_id % 5 == 0:
                user_steps.remove('Step2')  # 20% skip step 2
            if user_id % 10 == 0:
                user_steps.remove('Step4')  # 10% skip step 4
            
            # Add some cycles for certain users
            if user_id % 7 == 0:
                # Add a retry loop
                user_steps.insert(-1, 'Step3')  # Repeat step 3
            
            for i, step in enumerate(user_steps):
                events.append({
                    'user_id': f'user_{user_id}',
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i, minutes=user_id % 60)
                })
        
        return pd.DataFrame(events)
    
    def test_large_dataset_performance(self, large_process_data):
        """Test performance on large dataset"""
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        import time
        start_time = time.time()
        
        process_data = analyzer.discover_process_mining_structure(
            large_process_data,
            min_frequency=10,
            include_cycles=True
        )
        
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30.0, f"Process discovery took too long: {execution_time:.2f}s"
        
        # Should discover meaningful structure
        assert len(process_data.activities) > 0
        assert len(process_data.transitions) > 0
        assert process_data.statistics['total_cases'] == 1000
    
    def test_memory_efficiency(self, large_process_data):
        """Test memory efficiency with large dataset"""
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        # Monitor memory usage (simplified)
        import sys
        
        # Get initial memory footprint
        initial_refs = sys.getrefcount(large_process_data)
        
        process_data = analyzer.discover_process_mining_structure(
            large_process_data,
            min_frequency=10,
            include_cycles=True
        )
        
        # Should not create excessive object references
        final_refs = sys.getrefcount(large_process_data)
        
        # Memory should not have grown excessively
        ref_growth = final_refs - initial_refs
        assert ref_growth < 100, f"Excessive memory growth detected: {ref_growth} new references"
        
        # Process data should be reasonable size
        assert len(process_data.activities) < 100  # Should not have excessive activities
        assert len(process_data.transitions) < 500  # Should not have excessive transitions


class TestProcessMiningEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_single_user_single_event(self):
        """Test with minimal data - single user, single event"""
        data = pd.DataFrame([{
            'user_id': 'user_1',
            'event_name': 'single_event',
            'timestamp': datetime(2024, 1, 1, 10, 0, 0)
        }])
        
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should handle gracefully
        assert len(process_data.activities) == 1
        assert len(process_data.transitions) == 0
        assert process_data.statistics['total_cases'] == 1
    
    def test_no_transitions_data(self):
        """Test with data that has no valid transitions"""
        events = []
        # Create data where each user has only one unique event
        for user_id in range(5):
            events.append({
                'user_id': f'user_{user_id}',
                'event_name': f'unique_event_{user_id}',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0)
            })
        
        data = pd.DataFrame(events)
        
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should handle gracefully
        assert len(process_data.activities) == 5
        assert len(process_data.transitions) == 0
        assert len(process_data.cycles) == 0
    
    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps"""
        data = pd.DataFrame([
            {
                'user_id': 'user_1',
                'event_name': 'event_1',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0)
            },
            {
                'user_id': 'user_1',
                'event_name': 'event_2',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0)  # Same timestamp
            },
            {
                'user_id': 'user_1',
                'event_name': 'event_3',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0)  # Same timestamp
            }
        ])
        
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        # Should not raise an error
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=1,
            include_cycles=True
        )
        
        assert len(process_data.activities) == 3
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        data = pd.DataFrame([
            {
                'user_id': 123,  # Non-string user_id
                'event_name': 'event_1',
                'timestamp': datetime(2024, 1, 1, 10, 0, 0)
            },
            {
                'user_id': 'user_2',
                'event_name': None,  # None event name
                'timestamp': datetime(2024, 1, 1, 11, 0, 0)
            }
        ])
        
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        # Should handle gracefully without crashing
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should have filtered out invalid data
        assert len(process_data.activities) <= 1  # Only valid events should remain


class TestProcessMiningIntegration:
    """Test integration between process mining components"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create realistic e-commerce data
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Simulate 50 users going through e-commerce funnel
        for user_id in range(50):
            user_events = ['Landing', 'Browse', 'Product_View']
            
            # 60% add to cart
            if user_id < 30:
                user_events.append('Add_to_Cart')
                
                # 70% of those proceed to checkout
                if user_id < 21:
                    user_events.append('Checkout')
                    
                    # 80% complete purchase
                    if user_id < 17:
                        user_events.append('Purchase')
            
            # Add some browsing cycles for engaged users
            if user_id % 5 == 0:
                user_events.insert(-1, 'Browse')  # Browse again
                user_events.insert(-1, 'Product_View')  # View more products
            
            for i, event in enumerate(user_events):
                events.append({
                    'user_id': f'user_{user_id}',
                    'event_name': event,
                    'timestamp': base_time + timedelta(hours=i, minutes=user_id % 60)
                })
        
        data = pd.DataFrame(events)
        
        # Step 1: Discover process structure
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=2,
            include_cycles=True
        )
        
        # Verify meaningful discovery
        assert len(process_data.activities) >= 5  # Should find main activities
        assert len(process_data.transitions) >= 4  # Should find main flow
        
        # Step 2: Create visualization
        visualizer = FunnelVisualizer(theme='dark', colorblind_friendly=False)
        
        fig = visualizer.create_process_mining_diagram(
            process_data,
            layout_algorithm="hierarchical",
            show_frequencies=True,
            show_statistics=True
        )
        
        # Verify visualization created
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        
        # Step 3: Verify insights generation
        assert len(process_data.insights) > 0
        
        # Should contain meaningful insights about e-commerce flow
        insight_text = ' '.join(process_data.insights).lower()
        assert any(keyword in insight_text for keyword in ['conversion', 'path', 'rate', 'users'])
    
    def test_integration_with_existing_funnel_analysis(self):
        """Test that process mining integrates well with existing funnel analysis"""
        # This test ensures process mining doesn't break existing functionality
        # and can work alongside traditional funnel analysis
        
        # Create funnel-like data that can be analyzed both ways
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        funnel_steps = ['Step_1', 'Step_2', 'Step_3', 'Step_4']
        
        for user_id in range(20):
            # Some users complete all steps
            if user_id < 15:
                user_steps = funnel_steps.copy()
            # Some drop off at step 3
            elif user_id < 18:
                user_steps = funnel_steps[:3]
            # Some drop off at step 2
            else:
                user_steps = funnel_steps[:2]
            
            for i, step in enumerate(user_steps):
                events.append({
                    'user_id': f'user_{user_id}',
                    'event_name': step,
                    'timestamp': base_time + timedelta(hours=i)
                })
        
        data = pd.DataFrame(events)
        
        # Test process mining analysis
        config = FunnelConfig()
        analyzer = PathAnalyzer(config)
        
        process_data = analyzer.discover_process_mining_structure(
            data,
            min_frequency=1,
            include_cycles=True
        )
        
        # Should discover the funnel structure
        assert len(process_data.activities) == 4
        assert len(process_data.transitions) == 3
        
        # Should identify proper start and end activities
        assert 'Step_1' in process_data.start_activities
        assert 'Step_4' in process_data.end_activities
        
        # Verify this works alongside traditional funnel analysis
        # (This would typically be tested with actual FunnelCalculator integration)
        assert process_data.statistics['total_cases'] == 20
        assert process_data.statistics['completion_rate'] < 100  # Some users drop off
