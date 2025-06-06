#!/usr/bin/env python3
"""
Tests for the simplified UI functionality

This module tests the new simplified event selection interface to ensure
it works correctly and maintains all core functionality.
"""

import pytest
import pandas as pd
import json
from datetime import datetime, timedelta
from app import DataSourceManager, FunnelCalculator, FunnelConfig


class TestSimplifiedUI:
    """Test cases for the simplified UI functionality"""
    
    @pytest.fixture
    def sample_events_data(self):
        """Create sample events data for testing"""
        base_timestamp = datetime(2024, 1, 1, 10, 0)
        events = []
        
        # Create events for 100 users through a simple funnel
        for i in range(100):
            # All users sign up
            events.append({
                'user_id': f'user_{i:03d}',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=i),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
            
            # 80% verify email
            if i < 80:
                events.append({
                    'user_id': f'user_{i:03d}',
                    'event_name': 'Email Verification',
                    'timestamp': base_timestamp + timedelta(minutes=i + 30),
                    'event_properties': json.dumps({}),
                    'user_properties': json.dumps({})
                })
            
            # 60% complete first login
            if i < 60:
                events.append({
                    'user_id': f'user_{i:03d}',
                    'event_name': 'First Login',
                    'timestamp': base_timestamp + timedelta(minutes=i + 60),
                    'event_properties': json.dumps({}),
                    'user_properties': json.dumps({})
                })
        
        return pd.DataFrame(events)
    
    @pytest.fixture
    def data_source_manager(self):
        """Create a DataSourceManager instance"""
        return DataSourceManager()
    
    def test_get_unique_events_from_data(self, data_source_manager, sample_events_data):
        """Test that we can extract all unique events from data sources"""
        # Get unique events (this simulates what the simplified UI does)
        unique_events = sorted(sample_events_data['event_name'].unique())
        
        expected_events = ['Email Verification', 'First Login', 'Sign Up']
        assert unique_events == expected_events
        
        # Verify counts
        assert len(unique_events) == 3
        assert 'Sign Up' in unique_events
        assert 'Email Verification' in unique_events
        assert 'First Login' in unique_events
    
    def test_search_filtering_functionality(self, sample_events_data):
        """Test that search filtering works correctly"""
        all_events = sorted(sample_events_data['event_name'].unique())
        
        # Test search for "Sign"
        search_query = "sign"
        filtered_events = [event for event in all_events 
                          if search_query.lower() in event.lower()]
        assert filtered_events == ['Sign Up']
        
        # Test search for "Login"
        search_query = "login"
        filtered_events = [event for event in all_events 
                          if search_query.lower() in event.lower()]
        assert filtered_events == ['First Login']
        
        # Test search for "Email"
        search_query = "email"
        filtered_events = [event for event in all_events 
                          if search_query.lower() in event.lower()]
        assert filtered_events == ['Email Verification']
        
        # Test case insensitive search
        search_query = "VERIFICATION"
        filtered_events = [event for event in all_events 
                          if search_query.lower() in event.lower()]
        assert filtered_events == ['Email Verification']
    
    def test_funnel_calculation_with_selected_events(self, sample_events_data):
        """Test that funnel calculation works with events selected via simplified UI"""
        # Simulate user selecting events via checkboxes
        selected_events = ['Sign Up', 'Email Verification', 'First Login']
        
        # Calculate funnel metrics
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Verify results
        assert len(results.steps) == 3
        assert results.steps == selected_events
        assert results.users_count[0] == 100  # All users sign up
        assert results.users_count[1] == 80   # 80% verify email
        assert results.users_count[2] == 60   # 60% complete first login
        
        # Verify conversion rates
        assert results.conversion_rates[0] == 100.0  # First step is always 100%
        assert results.conversion_rates[1] == 80.0   # 80/100
        assert results.conversion_rates[2] == 60.0   # 60/100
    
    def test_partial_event_selection(self, sample_events_data):
        """Test funnel calculation with only some events selected"""
        # Simulate user selecting only first two events
        selected_events = ['Sign Up', 'Email Verification']
        
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Verify results
        assert len(results.steps) == 2
        assert results.steps == selected_events
        assert results.users_count[0] == 100  # All users sign up
        assert results.users_count[1] == 80   # 80% verify email
        
        # Verify conversion rates
        assert results.conversion_rates[0] == 100.0  # First step is always 100%
        assert results.conversion_rates[1] == 80.0   # 80/100
    
    def test_single_event_selection(self, sample_events_data):
        """Test that single event selection is handled correctly"""
        # Simulate user selecting only one event
        selected_events = ['Sign Up']
        
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Single step should return empty results (funnel needs at least 2 steps)
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
    
    def test_no_events_selected(self, sample_events_data):
        """Test that no events selected is handled correctly"""
        # Simulate user selecting no events
        selected_events = []
        
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Empty selection should return empty results
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
    
    def test_events_not_in_data_selected(self, sample_events_data):
        """Test that selecting events not in data is handled correctly"""
        # Simulate user selecting events that don't exist in data
        selected_events = ['Nonexistent Event 1', 'Nonexistent Event 2']
        
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Should return zero counts for non-existent events
        assert len(results.steps) == 2
        assert results.steps == selected_events
        assert results.users_count == [0, 0]
        assert results.conversion_rates[0] == 100.0  # First step is always 100% of its own count
        assert results.conversion_rates[1] == 0.0    # But if count is 0, then 0%
    
    def test_mixed_existing_and_nonexisting_events(self, sample_events_data):
        """Test selecting a mix of existing and non-existing events"""
        # Mix of existing and non-existing events
        selected_events = ['Sign Up', 'Nonexistent Event', 'Email Verification']
        
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_events_data, selected_events)
        
        # Should handle mixed selection correctly
        assert len(results.steps) == 3
        assert results.steps == selected_events
        assert results.users_count[0] == 100  # Sign Up exists
        assert results.users_count[1] == 0    # Nonexistent Event
        assert results.users_count[2] == 0    # Can't progress through nonexistent step
    
    def test_data_source_validation_still_works(self, data_source_manager):
        """Test that data source validation functionality is preserved"""
        # Test with valid data
        valid_data = pd.DataFrame({
            'user_id': ['user_001', 'user_002'],
            'event_name': ['Sign Up', 'Login'],
            'timestamp': [datetime.now(), datetime.now()],
            'event_properties': ['{}', '{}'],
            'user_properties': ['{}', '{}']
        })
        
        is_valid, message = data_source_manager.validate_event_data(valid_data)
        assert is_valid
        assert "valid" in message.lower()
        
        # Test with invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'user_id': ['user_001'],
            'event_name': ['Sign Up']
            # Missing timestamp, event_properties, user_properties
        })
        
        is_valid, message = data_source_manager.validate_event_data(invalid_data)
        assert not is_valid
        assert "missing" in message.lower() 