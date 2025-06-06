"""
Pytest fixtures for funnel analytics testing system.
Provides reusable test data and configuration objects.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

# Import the classes we need to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    FunnelCalculator, 
    FunnelConfig, 
    CountingMethod, 
    ReentryMode, 
    FunnelOrder,
    DataSourceManager
)


@pytest.fixture
def base_timestamp():
    """Base timestamp for all test data."""
    return datetime(2024, 1, 1, 10, 0, 0)


@pytest.fixture
def default_config():
    """Default funnel configuration for testing."""
    return FunnelConfig(
        conversion_window_hours=24,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED
    )


@pytest.fixture
def simple_linear_funnel_data(base_timestamp):
    """
    Simple linear funnel test data:
    - 1000 users start (Sign Up)
    - 800 users complete Email Verification  
    - 600 users complete First Login
    - 400 users complete First Purchase
    """
    events = []
    
    # All users complete signup
    for i in range(1000):
        events.append({
            'user_id': f'user_{i:04d}',
            'event_name': 'Sign Up',
            'timestamp': base_timestamp + timedelta(minutes=i//10),
            'event_properties': json.dumps({'source': 'organic'}),
            'user_properties': json.dumps({'segment': 'new'})
        })
    
    # 800 users verify email (within 2 hours)
    for i in range(800):
        events.append({
            'user_id': f'user_{i:04d}',
            'event_name': 'Email Verification',
            'timestamp': base_timestamp + timedelta(minutes=i//10 + 30),
            'event_properties': json.dumps({'source': 'organic'}),
            'user_properties': json.dumps({'segment': 'new'})
        })
    
    # 600 users complete first login (within 4 hours)
    for i in range(600):
        events.append({
            'user_id': f'user_{i:04d}',
            'event_name': 'First Login',
            'timestamp': base_timestamp + timedelta(minutes=i//10 + 120),
            'event_properties': json.dumps({'source': 'organic'}),
            'user_properties': json.dumps({'segment': 'new'})
        })
    
    # 400 users make first purchase (within 8 hours)
    for i in range(400):
        events.append({
            'user_id': f'user_{i:04d}',
            'event_name': 'First Purchase',
            'timestamp': base_timestamp + timedelta(minutes=i//10 + 240),
            'event_properties': json.dumps({'source': 'organic'}),
            'user_properties': json.dumps({'segment': 'new'})
        })
    
    return pd.DataFrame(events)


@pytest.fixture
def conversion_window_test_data(base_timestamp):
    """
    Test data specifically for conversion window testing.
    Users complete events at different time intervals.
    """
    events = []
    
    # User 1: All events within 1 hour (should convert)
    user1_events = [
        ('Sign Up', 0),
        ('Email Verification', 30),  # 30 minutes later
        ('First Login', 45),         # 45 minutes later
    ]
    
    for event_name, minutes_offset in user1_events:
        events.append({
            'user_id': 'user_001',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    # User 2: Events beyond conversion window (should not convert)
    user2_events = [
        ('Sign Up', 0),
        ('Email Verification', 90),   # 1.5 hours later (beyond 1h window)
        ('First Login', 150),         # 2.5 hours later
    ]
    
    for event_name, minutes_offset in user2_events:
        events.append({
            'user_id': 'user_002',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    # User 3: On the boundary (exactly at window edge)
    user3_events = [
        ('Sign Up', 0),
        ('Email Verification', 60),   # Exactly 1 hour later
        ('First Login', 120),         # 2 hours later
    ]
    
    for event_name, minutes_offset in user3_events:
        events.append({
            'user_id': 'user_003',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    return pd.DataFrame(events)


@pytest.fixture
def reentry_test_data(base_timestamp):
    """
    Test data for reentry mode testing.
    Users restart the funnel multiple times.
    """
    events = []
    
    # User 1: Completes funnel, then restarts
    user1_events = [
        ('Sign Up', 0),
        ('Email Verification', 30),
        ('First Login', 60),
        ('Sign Up', 120),        # Restarts funnel
        ('Email Verification', 150),
        ('First Login', 180),
    ]
    
    for event_name, minutes_offset in user1_events:
        events.append({
            'user_id': 'user_001',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    # User 2: Partial completion, then restart
    user2_events = [
        ('Sign Up', 0),
        ('Email Verification', 30),
        ('Sign Up', 90),         # Restarts without completing
        ('Email Verification', 120),
        ('First Login', 150),
    ]
    
    for event_name, minutes_offset in user2_events:
        events.append({
            'user_id': 'user_002',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    return pd.DataFrame(events)


@pytest.fixture
def segmentation_test_data(base_timestamp):
    """
    Test data for segmentation testing.
    Users with different properties and segments.
    """
    events = []
    
    # Mobile users
    mobile_users = ['user_001', 'user_002', 'user_003']
    for user_id in mobile_users:
        user_events = [
            ('Sign Up', 0),
            ('Email Verification', 30),
            ('First Login', 60),
        ]
        
        for event_name, minutes_offset in user_events:
            events.append({
                'user_id': user_id,
                'event_name': event_name,
                'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
                'event_properties': json.dumps({'platform': 'mobile', 'utm_source': 'google'}),
                'user_properties': json.dumps({'subscription': 'free', 'country': 'US'})
            })
    
    # Desktop users
    desktop_users = ['user_004', 'user_005']
    for user_id in desktop_users:
        user_events = [
            ('Sign Up', 0),
            ('Email Verification', 45),
        ]
        
        for event_name, minutes_offset in user_events:
            events.append({
                'user_id': user_id,
                'event_name': event_name,
                'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
                'event_properties': json.dumps({'platform': 'desktop', 'utm_source': 'facebook'}),
                'user_properties': json.dumps({'subscription': 'premium', 'country': 'UK'})
            })
    
    return pd.DataFrame(events)


@pytest.fixture
def edge_case_test_data(base_timestamp):
    """
    Test data for edge cases and boundary conditions.
    """
    events = []
    
    # User with same timestamp events
    same_time_events = [
        ('Sign Up', 0),
        ('Email Verification', 0),  # Same timestamp
        ('First Login', 0),         # Same timestamp
    ]
    
    for event_name, minutes_offset in same_time_events:
        events.append({
            'user_id': 'user_same_time',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    # User with out-of-order events
    out_of_order_events = [
        ('Sign Up', 0),
        ('First Login', 30),        # Skip Email Verification
        ('Email Verification', 60), # Do Email Verification after First Login
    ]
    
    for event_name, minutes_offset in out_of_order_events:
        events.append({
            'user_id': 'user_out_of_order',
            'event_name': event_name,
            'timestamp': base_timestamp + timedelta(minutes=minutes_offset),
            'event_properties': json.dumps({}),
            'user_properties': json.dumps({})
        })
    
    # User with missing data
    events.append({
        'user_id': None,  # Missing user_id
        'event_name': 'Sign Up',
        'timestamp': base_timestamp,
        'event_properties': json.dumps({}),
        'user_properties': json.dumps({})
    })
    
    events.append({
        'user_id': 'user_missing_event',
        'event_name': None,  # Missing event_name
        'timestamp': base_timestamp,
        'event_properties': json.dumps({}),
        'user_properties': json.dumps({})
    })
    
    return pd.DataFrame(events)


@pytest.fixture
def empty_data():
    """Empty DataFrame for testing edge cases."""
    return pd.DataFrame(columns=['user_id', 'event_name', 'timestamp', 'event_properties', 'user_properties'])


@pytest.fixture
def standard_funnel_steps():
    """Standard funnel steps for most tests."""
    return ['Sign Up', 'Email Verification', 'First Login', 'First Purchase']


@pytest.fixture
def calculator_factory():
    """Factory function to create FunnelCalculator instances with different configs."""
    def _create_calculator(config=None):
        if config is None:
            config = FunnelConfig()
        return FunnelCalculator(config)
    return _create_calculator


@pytest.fixture
def data_source_manager():
    """DataSourceManager instance for testing."""
    return DataSourceManager()


# Expected results fixtures for verification
@pytest.fixture
def expected_simple_linear_results():
    """Expected results for simple linear funnel."""
    return {
        'steps': ['Sign Up', 'Email Verification', 'First Login', 'First Purchase'],
        'users_count': [1000, 800, 600, 400],
        'conversion_rates': [100.0, 80.0, 60.0, 40.0],
        'drop_offs': [0, 200, 200, 200],
        'drop_off_rates': [0.0, 20.0, 25.0, 33.33]  # Approximate drop-off rates
    }


@pytest.fixture
def expected_conversion_window_results():
    """Expected results for conversion window test with 1-hour window."""
    return {
        'within_window_users': 1,    # Only user_001 should convert
        'outside_window_users': 0,   # user_002 should not convert
        'boundary_users': 1,         # user_003 at boundary should convert (≤ boundary)
    }


# Performance test data fixtures
@pytest.fixture
def large_dataset(base_timestamp):
    """Large dataset for performance testing."""
    import numpy as np
    np.random.seed(42)
    
    events = []
    n_users = 5000  # Reduced for faster test execution
    
    for i in range(n_users):
        # Simulate different completion rates
        if i < 8000:  # 80% complete step 1
            events.append({
                'user_id': f'user_{i:05d}',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=np.random.randint(0, 1440)),
                'event_properties': json.dumps({'platform': np.random.choice(['mobile', 'desktop'])}),
                'user_properties': json.dumps({'country': np.random.choice(['US', 'UK', 'DE'])})
            })
            
            if i < 6000:  # 60% complete step 2
                events.append({
                    'user_id': f'user_{i:05d}',
                    'event_name': 'Email Verification',
                    'timestamp': base_timestamp + timedelta(minutes=np.random.randint(30, 1470)),
                    'event_properties': json.dumps({'platform': np.random.choice(['mobile', 'desktop'])}),
                    'user_properties': json.dumps({'country': np.random.choice(['US', 'UK', 'DE'])})
                })
                
                if i < 4000:  # 40% complete step 3
                    events.append({
                        'user_id': f'user_{i:05d}',
                        'event_name': 'First Login',
                        'timestamp': base_timestamp + timedelta(minutes=np.random.randint(60, 1500)),
                        'event_properties': json.dumps({'platform': np.random.choice(['mobile', 'desktop'])}),
                        'user_properties': json.dumps({'country': np.random.choice(['US', 'UK', 'DE'])})
                    })
    
    return pd.DataFrame(events) 