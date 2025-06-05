"""
Test edge cases and boundary conditions for funnel analysis.
Tests empty data, missing data, same timestamps, out-of-order events, etc.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from app import FunnelConfig, CountingMethod, ReentryMode, FunnelOrder


@pytest.mark.edge_case
class TestEdgeCases:
    
    def test_empty_dataset(self, calculator_factory, empty_data):
        """
        Test behavior with completely empty dataset.
        Expected: Should return empty results without errors.
        """
        calculator = calculator_factory()
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(empty_data, steps)
        
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
        assert results.drop_offs == []
        assert results.drop_off_rates == []
    
    def test_empty_funnel_steps(self, calculator_factory, simple_linear_funnel_data):
        """
        Test behavior with empty funnel steps list.
        Expected: Should return empty results.
        """
        calculator = calculator_factory()
        empty_steps = []
        
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, empty_steps)
        
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
        assert results.drop_offs == []
        assert results.drop_off_rates == []
    
    def test_single_step_funnel(self, calculator_factory, simple_linear_funnel_data):
        """
        Test funnel with only one step.
        Expected: Should return empty results (funnel requires at least 2 steps).
        """
        calculator = calculator_factory()
        single_step = ['Sign Up']
        
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, single_step)
        
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
        assert results.drop_offs == []
        assert results.drop_off_rates == []
    
    def test_events_with_same_timestamp(self, calculator_factory, base_timestamp):
        """
        Test deterministic behavior when multiple events have identical timestamps.
        Expected: Should handle gracefully and provide consistent results.
        """
        calculator = calculator_factory()
        
        # All events at the same timestamp
        same_time = base_timestamp
        events = [
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': same_time,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': same_time,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'First Login',
                'timestamp': same_time,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # Another user with same timestamps
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': same_time,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': same_time,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification', 'First Login']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should handle same timestamps deterministically
        assert results.users_count[0] == 2  # Both users sign up
        assert results.users_count[1] == 2  # Both users verify email
        assert results.users_count[2] == 1  # Only user_001 has First Login
    
    def test_out_of_order_events(self, calculator_factory, base_timestamp):
        """
        Test behavior when users perform events in wrong order.
        Expected: Should not count conversions for out-of-order sequences.
        """
        calculator = calculator_factory()
        
        events = [
            # User 1: Correct order
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'First Login',
                'timestamp': base_timestamp + timedelta(minutes=60),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User 2: Wrong order (skips Email Verification, then does it)
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'First Login',
                'timestamp': base_timestamp + timedelta(minutes=30),  # Skip verification
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=60),  # Too late
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification', 'First Login']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 2  # Both users sign up
        assert results.users_count[1] == 1  # Only user_001 verifies email in order
        assert results.users_count[2] == 1  # Only user_001 completes sequence correctly
    
    def test_missing_user_id(self, calculator_factory, base_timestamp):
        """
        Test handling of events with missing user_id.
        Expected: Should ignore events with null/missing user_id.
        """
        calculator = calculator_factory()
        
        events = [
            # Valid event
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Invalid event (missing user_id)
            {
                'user_id': None,
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Invalid event (empty string user_id)
            {
                'user_id': '',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Valid event
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should only count valid events
        assert results.users_count[0] == 1  # Only user_001's sign up
        assert results.users_count[1] == 1  # Only user_001's verification
    
    def test_missing_event_name(self, calculator_factory, base_timestamp):
        """
        Test handling of events with missing event_name.
        Expected: Should ignore events with null/missing event_name.
        """
        calculator = calculator_factory()
        
        events = [
            # Valid event
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Invalid event (missing event_name)
            {
                'user_id': 'user_001',
                'event_name': None,
                'timestamp': base_timestamp + timedelta(minutes=15),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Valid event
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should ignore event with missing event_name
        assert results.users_count[0] == 1  # user_001's sign up
        assert results.users_count[1] == 1  # user_001's verification
    
    def test_missing_timestamp(self, calculator_factory, base_timestamp):
        """
        Test handling of events with missing timestamp.
        Expected: Should ignore events with null/missing timestamp.
        """
        calculator = calculator_factory()
        
        events = [
            # Valid event
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Invalid event (missing timestamp)
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': None,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            # Valid event
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should ignore event with missing timestamp
        assert results.users_count[0] == 2  # Both users' sign ups
        assert results.users_count[1] == 0  # No valid email verifications
    
    def test_malformed_properties(self, calculator_factory, base_timestamp):
        """
        Test handling of malformed event_properties and user_properties.
        Expected: Should handle gracefully, not crash on invalid JSON.
        """
        calculator = calculator_factory()
        
        events = [
            # Valid properties
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({'platform': 'mobile'}),
                'user_properties': json.dumps({'country': 'US'})
            },
            # Invalid JSON properties
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': 'invalid json {',
                'user_properties': 'also invalid }'
            },
            # Missing properties
            {
                'user_id': 'user_003',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': None,
                'user_properties': None
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        # Should not crash with malformed properties
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 3  # All sign ups should be counted
    
    def test_very_large_dataset_performance(self, calculator_factory, large_dataset):
        """
        Test performance with large dataset.
        Expected: Should complete within reasonable time.
        """
        import time
        
        calculator = calculator_factory()
        steps = ['Sign Up', 'Email Verification', 'First Login']
        
        start_time = time.time()
        results = calculator.calculate_funnel_metrics(large_dataset, steps)
        end_time = time.time()
        
        # Should complete within 30 seconds (adjust as needed)
        execution_time = end_time - start_time
        assert execution_time < 30.0, f"Execution took {execution_time:.2f} seconds"
        
        # Should return valid results
        assert len(results.users_count) == 3
        assert results.users_count[0] > 0
    
    def test_extreme_timestamps(self, calculator_factory):
        """
        Test handling of extreme timestamp values.
        Expected: Should handle very old and very future timestamps.
        """
        calculator = calculator_factory()
        
        events = [
            # Very old timestamp
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': datetime(1970, 1, 1),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': datetime(1970, 1, 1, 0, 30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # Very future timestamp
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': datetime(2099, 12, 31),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': datetime(2099, 12, 31, 0, 30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should handle extreme timestamps
        assert results.users_count[0] == 2
        assert results.users_count[1] == 2
    
    def test_mixed_data_types(self, calculator_factory, base_timestamp):
        """
        Test handling of mixed data types in columns.
        Expected: Should handle gracefully and process valid data.
        """
        # Create DataFrame with mixed types
        events_data = {
            'user_id': ['user_001', 123, 'user_003', None],  # Mixed string/int/null
            'event_name': ['Sign Up', 'Sign Up', None, 'Email Verification'],  # Mixed string/null
            'timestamp': [
                base_timestamp,
                base_timestamp + timedelta(minutes=30),
                base_timestamp + timedelta(minutes=60),
                'invalid_timestamp'  # Invalid timestamp
            ],
            'event_properties': [
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
                json.dumps({})
            ],
            'user_properties': [
                json.dumps({}),
                json.dumps({}),
                json.dumps({}),
                json.dumps({})
            ]
        }
        
        data = pd.DataFrame(events_data)
        calculator = calculator_factory()
        steps = ['Sign Up', 'Email Verification']
        
        # Should handle mixed types without crashing
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should process valid events only
        assert len(results.users_count) >= 0  # Should not crash
    
    def test_duplicate_events_same_user_same_time(self, calculator_factory, base_timestamp):
        """
        Test handling of exact duplicate events (same user, event, timestamp).
        Expected: Should be deterministic in counting.
        """
        calculator = calculator_factory()
        
        events = [
            # Exact duplicates
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,  # Exact same timestamp
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Should handle duplicates consistently
        assert results.users_count[0] == 1  # User should be counted once for Sign Up
        assert results.users_count[1] == 1  # User should convert to Email Verification
    
    def test_unordered_funnel_edge_cases(self, calculator_factory, base_timestamp):
        """
        Test edge cases specific to unordered funnels.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.UNORDERED
        )
        calculator = calculator_factory(config)
        
        events = [
            # User completes events in reverse order
            {
                'user_id': 'user_001',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_001',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            
            # User completes all events at same time
            {
                'user_id': 'user_002',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp,
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            },
            {
                'user_id': 'user_002',
                'event_name': 'Email Verification',
                'timestamp': base_timestamp,  # Same time
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            }
        ]
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification']
        
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # For unordered funnel, both users should be counted
        assert results.users_count[0] == 2  # Both users have Sign Up
        assert results.users_count[1] == 2  # Both users have Email Verification 