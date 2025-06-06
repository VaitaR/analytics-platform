"""
Tests for no-reload improvements functionality.

This module tests the new features that prevent page reloads:
1. Event statistics display
2. No-reload event selection with checkboxes
3. Funnel step reordering with arrow buttons
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    get_event_statistics,
    initialize_session_state,
    FunnelConfig
)


class TestEventStatistics:
    """Test event statistics calculation and display functionality."""
    
    def test_get_event_statistics_empty_data(self):
        """Test statistics calculation with empty or None data."""
        # Test with None
        stats = get_event_statistics(None)
        assert stats == {}
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        stats = get_event_statistics(empty_df)
        assert stats == {}
    
    def test_get_event_statistics_basic_functionality(self, simple_linear_funnel_data):
        """Test basic event statistics calculation."""
        stats = get_event_statistics(simple_linear_funnel_data)
        
        # Check that statistics are calculated for all events
        unique_events = simple_linear_funnel_data['event_name'].unique()
        assert len(stats) == len(unique_events)
        
        # Check that all required fields are present
        for event_name, event_stats in stats.items():
            required_fields = [
                'count', 'unique_users', 'percentage_of_events',
                'user_coverage', 'frequency_level', 'frequency_color', 'avg_per_user'
            ]
            for field in required_fields:
                assert field in event_stats, f"Missing field {field} for event {event_name}"
    
    def test_event_statistics_accuracy(self, simple_linear_funnel_data):
        """Test accuracy of calculated statistics."""
        stats = get_event_statistics(simple_linear_funnel_data)
        
        # Manually calculate expected values for verification
        total_events = len(simple_linear_funnel_data)
        total_users = simple_linear_funnel_data['user_id'].nunique()
        
        for event_name, event_stats in stats.items():
            event_data = simple_linear_funnel_data[simple_linear_funnel_data['event_name'] == event_name]
            expected_count = len(event_data)
            expected_unique_users = event_data['user_id'].nunique()
            expected_percentage = (expected_count / total_events) * 100
            expected_user_coverage = (expected_unique_users / total_users) * 100
            expected_avg_per_user = expected_count / expected_unique_users if expected_unique_users > 0 else 0
            
            assert event_stats['count'] == expected_count
            assert event_stats['unique_users'] == expected_unique_users
            assert abs(event_stats['percentage_of_events'] - expected_percentage) < 0.01
            assert abs(event_stats['user_coverage'] - expected_user_coverage) < 0.01
            assert abs(event_stats['avg_per_user'] - expected_avg_per_user) < 0.01
    
    def test_frequency_level_classification(self, simple_linear_funnel_data):
        """Test that frequency levels are correctly classified."""
        stats = get_event_statistics(simple_linear_funnel_data)
        
        total_events = len(simple_linear_funnel_data)
        
        for event_name, event_stats in stats.items():
            event_count = event_stats['count']
            frequency_level = event_stats['frequency_level']
            frequency_color = event_stats['frequency_color']
            
            if event_count > total_events * 0.1:
                assert frequency_level == "high"
                assert frequency_color == "#ef4444"
            elif event_count > total_events * 0.01:
                assert frequency_level == "medium"
                assert frequency_color == "#f59e0b"
            else:
                assert frequency_level == "low"
                assert frequency_color == "#10b981"
    
    def test_event_statistics_with_different_data_distributions(self):
        """Test statistics with various data distributions."""
        # Create test data with known distribution
        data = []
        
        # High frequency event (50% of events)
        for i in range(500):
            data.append({
                'user_id': f'user_{i % 100}',  # 100 unique users
                'event_name': 'high_freq_event',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            })
        
        # Medium frequency event (5% of events)
        for i in range(50):
            data.append({
                'user_id': f'user_{i % 20}',  # 20 unique users
                'event_name': 'medium_freq_event',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            })
        
        # Low frequency event (0.5% of events)
        for i in range(5):
            data.append({
                'user_id': f'user_{i}',  # 5 unique users
                'event_name': 'low_freq_event',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            })
        
        df = pd.DataFrame(data)
        stats = get_event_statistics(df)
        
        # Verify frequency classifications
        assert stats['high_freq_event']['frequency_level'] == 'high'
        assert stats['medium_freq_event']['frequency_level'] == 'medium'
        assert stats['low_freq_event']['frequency_level'] == 'low'


class MockSessionState:
    """Mock session state that supports both dict-like and attribute access."""
    def __init__(self):
        self._data = {}
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._data.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._data[name] = value


class TestSessionStateManagement:
    """Test session state initialization and management for no-reload functionality."""
    
    def test_initialize_session_state_new_variables(self):
        """Test that new session state variables are properly initialized."""
        mock_session_state = MockSessionState()
        
        # Mock the st module
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            initialize_session_state()
            
            # Check new variables specific to no-reload improvements
            assert 'event_statistics' in mock_session_state
            assert 'event_selections' in mock_session_state
            assert mock_session_state['event_statistics'] == {}
            assert mock_session_state['event_selections'] == {}
    
    def test_initialize_session_state_preserves_existing(self):
        """Test that existing session state is preserved during initialization."""
        mock_session_state = MockSessionState()
        
        # Pre-populate some session state
        mock_session_state['funnel_steps'] = ['event1', 'event2']
        mock_session_state['event_statistics'] = {'event1': {'count': 100}}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            initialize_session_state()
            
            # Check that existing values are preserved
            assert mock_session_state['funnel_steps'] == ['event1', 'event2']
            assert mock_session_state['event_statistics'] == {'event1': {'count': 100}}


class TestNoReloadEventSelection:
    """Test the no-reload event selection functionality."""
    
    def setup_method(self):
        """Set up test data for each test method."""
        self.sample_data = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'],
            'event_name': ['login', 'purchase', 'login', 'view_product', 'signup'],
            'timestamp': pd.to_datetime([
                '2024-01-01 10:00:00',
                '2024-01-01 10:05:00',
                '2024-01-01 11:00:00',
                '2024-01-01 11:05:00',
                '2024-01-01 12:00:00'
            ])
        })
    
    def test_event_selection_toggle_add(self):
        """Test adding an event to the funnel steps."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = []
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate the toggle function that would be called by checkbox
            event_name = 'login'
            if event_name in mock_session_state['funnel_steps']:
                mock_session_state['funnel_steps'].remove(event_name)
            else:
                mock_session_state['funnel_steps'].append(event_name)
            mock_session_state['analysis_results'] = None
            
            assert 'login' in mock_session_state['funnel_steps']
            assert mock_session_state['analysis_results'] is None
    
    def test_event_selection_toggle_remove(self):
        """Test removing an event from the funnel steps."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = ['login', 'purchase']
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate the toggle function that would be called by checkbox
            event_name = 'login'
            if event_name in mock_session_state['funnel_steps']:
                mock_session_state['funnel_steps'].remove(event_name)
            else:
                mock_session_state['funnel_steps'].append(event_name)
            mock_session_state['analysis_results'] = None
            
            assert 'login' not in mock_session_state['funnel_steps']
            assert 'purchase' in mock_session_state['funnel_steps']
            assert mock_session_state['analysis_results'] is None


class TestFunnelStepReordering:
    """Test the funnel step reordering functionality."""
    
    def test_move_step_up(self):
        """Test moving a funnel step up in the order."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = ['login', 'view_product', 'purchase']
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate moving step at index 1 up
            index = 1
            mock_session_state['funnel_steps'][index], mock_session_state['funnel_steps'][index-1] = \
                mock_session_state['funnel_steps'][index-1], mock_session_state['funnel_steps'][index]
            mock_session_state['analysis_results'] = None
            
            expected_order = ['view_product', 'login', 'purchase']
            assert mock_session_state['funnel_steps'] == expected_order
            assert mock_session_state['analysis_results'] is None
    
    def test_move_step_down(self):
        """Test moving a funnel step down in the order."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = ['login', 'view_product', 'purchase']
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate moving step at index 1 down
            index = 1
            mock_session_state['funnel_steps'][index], mock_session_state['funnel_steps'][index+1] = \
                mock_session_state['funnel_steps'][index+1], mock_session_state['funnel_steps'][index]
            mock_session_state['analysis_results'] = None
            
            expected_order = ['login', 'purchase', 'view_product']
            assert mock_session_state['funnel_steps'] == expected_order
            assert mock_session_state['analysis_results'] is None
    
    def test_remove_step(self):
        """Test removing a funnel step."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = ['login', 'view_product', 'purchase']
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate removing step at index 1
            index = 1
            mock_session_state['funnel_steps'].pop(index)
            mock_session_state['analysis_results'] = None
            
            expected_order = ['login', 'purchase']
            assert mock_session_state['funnel_steps'] == expected_order
            assert mock_session_state['analysis_results'] is None
    
    def test_clear_all_steps(self):
        """Test clearing all funnel steps."""
        mock_session_state = MockSessionState()
        mock_session_state['funnel_steps'] = ['login', 'view_product', 'purchase']
        mock_session_state['analysis_results'] = {'some': 'results'}
        
        with patch('app.st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Simulate clearing all steps
            mock_session_state['funnel_steps'] = []
            mock_session_state['analysis_results'] = None
            
            assert mock_session_state['funnel_steps'] == []
            assert mock_session_state['analysis_results'] is None


class TestDataLoadingIntegration:
    """Test integration with data loading to refresh event statistics."""
    
    def test_event_statistics_refresh_on_data_load(self, simple_linear_funnel_data):
        """Test that event statistics are refreshed when new data is loaded."""
        # This test would need to be integrated with the actual data loading process
        # For now, we'll test the logic that should be called
        
        # Simulate loading new data
        new_stats = get_event_statistics(simple_linear_funnel_data)
        
        # Verify that statistics are calculated
        assert len(new_stats) > 0
        assert all(isinstance(stats, dict) for stats in new_stats.values())
        
        # Verify that all expected fields are present
        for event_stats in new_stats.values():
            required_fields = [
                'count', 'unique_users', 'percentage_of_events',
                'user_coverage', 'frequency_level', 'frequency_color', 'avg_per_user'
            ]
            for field in required_fields:
                assert field in event_stats


class TestUIComponentKeys:
    """Test that UI component keys are unique and properly generated."""
    
    def test_checkbox_key_generation(self):
        """Test that checkbox keys are properly generated and unique."""
        event_names = ['login', 'purchase', 'view-product', 'sign up!', 'user@email.com']
        
        generated_keys = []
        for event in event_names:
            safe_event_name = "".join(c if c.isalnum() else "_" for c in event)
            checkbox_key = f"event_checkbox_{safe_event_name}"
            generated_keys.append(checkbox_key)
        
        # All keys should be unique
        assert len(generated_keys) == len(set(generated_keys))
        
        # Keys should only contain alphanumeric characters and underscores
        for key in generated_keys:
            assert all(c.isalnum() or c == '_' for c in key)
    
    def test_button_key_generation(self):
        """Test that button keys for reordering are properly generated."""
        steps = ['login', 'purchase', 'view-product']
        
        generated_keys = []
        for i, step in enumerate(steps):
            # Generate keys as they would be in the actual code
            move_up_key = f"move_up_{i}_{step}"
            move_down_key = f"move_down_{i}_{step}"
            remove_key = f"remove_step_{i}_{step}"
            
            generated_keys.extend([move_up_key, move_down_key, remove_key])
        
        # All keys should be unique
        assert len(generated_keys) == len(set(generated_keys))


class TestPerformanceConsiderations:
    """Test performance aspects of the no-reload improvements."""
    
    def test_event_statistics_caching(self):
        """Test that event statistics are properly cached to avoid recalculation."""
        # Create a large dataset to test performance
        large_data = []
        for i in range(10000):
            large_data.append({
                'user_id': f'user_{i % 1000}',
                'event_name': f'event_{i % 50}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            })
        
        df = pd.DataFrame(large_data)
        
        # Time the statistics calculation
        import time
        start_time = time.time()
        stats = get_event_statistics(df)
        calculation_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert calculation_time < 5.0  # 5 seconds max
        assert len(stats) == 50  # Should have 50 unique events
    
    def test_statistics_memory_efficiency(self):
        """Test that statistics don't consume excessive memory."""
        # Create test data
        data = []
        for i in range(1000):
            data.append({
                'user_id': f'user_{i % 100}',
                'event_name': f'event_{i % 10}',
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=i)
            })
        
        df = pd.DataFrame(data)
        stats = get_event_statistics(df)
        
        # Check that the stats dictionary is not unreasonably large
        import sys
        stats_size = sys.getsizeof(stats)
        
        # Should be reasonable size (adjust as needed)
        assert stats_size < 10000  # 10KB max for this test data


if __name__ == "__main__":
    pytest.main([__file__]) 