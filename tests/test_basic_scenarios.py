"""
Test basic scenarios (happy path) for funnel analysis.
Tests linear funnels, zero conversion, and 100% conversion cases.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import json


@pytest.mark.basic
class TestBasicScenarios:
    
    def test_linear_funnel_calculation(self, calculator_factory, simple_linear_funnel_data, standard_funnel_steps):
        """
        Test standard linear funnel where users progressively drop off at each step.
        Expected: 1000 -> 800 -> 600 -> 400 users
        """
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, standard_funnel_steps)
        
        # Verify basic structure
        assert results.steps == standard_funnel_steps
        assert len(results.users_count) == 4
        assert len(results.conversion_rates) == 4
        assert len(results.drop_offs) == 4
        assert len(results.drop_off_rates) == 4
        
        # Verify user counts
        assert results.users_count[0] == 1000  # Sign Up
        assert results.users_count[1] == 800   # Email Verification
        assert results.users_count[2] == 600   # First Login  
        assert results.users_count[3] == 400   # First Purchase
        
        # Verify conversion rates (from first step)
        assert results.conversion_rates[0] == 100.0  # First step always 100%
        assert results.conversion_rates[1] == 80.0   # 800/1000
        assert results.conversion_rates[2] == 60.0   # 600/1000
        assert results.conversion_rates[3] == 40.0   # 400/1000
        
        # Verify drop-offs
        assert results.drop_offs[0] == 0    # No drop-off at first step
        assert results.drop_offs[1] == 200  # 1000 - 800
        assert results.drop_offs[2] == 200  # 800 - 600
        assert results.drop_offs[3] == 200  # 600 - 400
        
        # Verify drop-off rates (step-to-step)
        assert results.drop_off_rates[0] == 0.0
        assert abs(results.drop_off_rates[1] - 20.0) < 0.1  # 200/1000
        assert abs(results.drop_off_rates[2] - 25.0) < 0.1  # 200/800  
        assert abs(results.drop_off_rates[3] - 33.33) < 0.1 # 200/600
    
    def test_zero_conversion_scenario(self, calculator_factory, base_timestamp):
        """
        Test scenario where no users complete beyond the first step.
        Expected: All conversion rates after first step should be 0%.
        """
        # Create data where only first step is completed
        events = []
        for i in range(100):
            events.append({
                'user_id': f'user_{i:03d}',
                'event_name': 'Sign Up',
                'timestamp': base_timestamp + timedelta(minutes=i),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        data = pd.DataFrame(events)
        steps = ['Sign Up', 'Email Verification', 'First Login']
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Verify counts
        assert results.users_count[0] == 100  # All users complete first step
        assert results.users_count[1] == 0    # No users complete second step
        assert results.users_count[2] == 0    # No users complete third step
        
        # Verify conversion rates
        assert results.conversion_rates[0] == 100.0
        assert results.conversion_rates[1] == 0.0
        assert results.conversion_rates[2] == 0.0
        
        # Verify drop-offs
        assert results.drop_offs[0] == 0
        assert results.drop_offs[1] == 100  # All users drop off
        assert results.drop_offs[2] == 0    # No one to drop off
    
    def test_perfect_conversion_scenario(self, calculator_factory, base_timestamp):
        """
        Test scenario where all users complete the entire funnel.
        Expected: 100% conversion rate for all steps.
        """
        events = []
        steps = ['Sign Up', 'Email Verification', 'First Login']
        
        # All 50 users complete all steps
        for i in range(50):
            for j, step in enumerate(steps):
                events.append({
                    'user_id': f'user_{i:03d}',
                    'event_name': step,
                    'timestamp': base_timestamp + timedelta(minutes=i + j * 10),
                    'event_properties': json.dumps({}),
                    'user_properties': json.dumps({})
                })
        
        data = pd.DataFrame(events)
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(data, steps)
        
        # Verify all users complete all steps
        assert results.users_count[0] == 50
        assert results.users_count[1] == 50
        assert results.users_count[2] == 50
        
        # Verify 100% conversion at all steps
        assert results.conversion_rates[0] == 100.0
        assert results.conversion_rates[1] == 100.0
        assert results.conversion_rates[2] == 100.0
        
        # Verify no drop-offs
        assert results.drop_offs[0] == 0
        assert results.drop_offs[1] == 0
        assert results.drop_offs[2] == 0
        
        assert results.drop_off_rates[0] == 0.0
        assert results.drop_off_rates[1] == 0.0
        assert results.drop_off_rates[2] == 0.0
    
    def test_single_step_funnel(self, calculator_factory, simple_linear_funnel_data):
        """
        Test funnel with only one step.
        Expected: Should handle gracefully and return appropriate metrics.
        """
        calculator = calculator_factory()
        single_step = ['Sign Up']
        
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, single_step)
        
        # Single step should return empty results (funnel needs at least 2 steps)
        assert results.steps == []
        assert results.users_count == []
        assert results.conversion_rates == []
        assert results.drop_offs == []
        assert results.drop_off_rates == []
    
    def test_two_step_minimal_funnel(self, calculator_factory, base_timestamp):
        """
        Test minimal valid funnel with exactly two steps.
        Expected: Should calculate conversion from step 1 to step 2.
        """
        events = []
        
        # 10 users complete first step
        for i in range(10):
            events.append({
                'user_id': f'user_{i:02d}',
                'event_name': 'Step A',
                'timestamp': base_timestamp + timedelta(minutes=i),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        # 6 users complete second step
        for i in range(6):
            events.append({
                'user_id': f'user_{i:02d}',
                'event_name': 'Step B',
                'timestamp': base_timestamp + timedelta(minutes=i + 30),
                'event_properties': json.dumps({}),
                'user_properties': json.dumps({})
            })
        
        data = pd.DataFrame(events)
        steps = ['Step A', 'Step B']
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(data, steps)
        
        assert results.users_count[0] == 10
        assert results.users_count[1] == 6
        assert results.conversion_rates[0] == 100.0
        assert results.conversion_rates[1] == 60.0  # 6/10
        assert results.drop_offs[1] == 4  # 10 - 6
        assert results.drop_off_rates[1] == 40.0  # 4/10
    
    def test_funnel_with_noise_events(self, calculator_factory, simple_linear_funnel_data, standard_funnel_steps):
        """
        Test that events not in the funnel definition are correctly ignored.
        Expected: Noise events should not affect funnel calculations.
        """
        # Add noise events to the data
        noise_events = []
        for i in range(100):
            noise_events.extend([
                {
                    'user_id': f'user_{i:04d}',
                    'event_name': 'Page View',
                    'timestamp': simple_linear_funnel_data.iloc[0]['timestamp'] + timedelta(minutes=i),
                    'event_properties': json.dumps({}),
                    'user_properties': json.dumps({})
                },
                {
                    'user_id': f'user_{i:04d}',
                    'event_name': 'Button Click',
                    'timestamp': simple_linear_funnel_data.iloc[0]['timestamp'] + timedelta(minutes=i + 15),
                    'event_properties': json.dumps({}),
                    'user_properties': json.dumps({})
                }
            ])
        
        # Combine original data with noise
        noise_df = pd.DataFrame(noise_events)
        combined_data = pd.concat([simple_linear_funnel_data, noise_df], ignore_index=True)
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(combined_data, standard_funnel_steps)
        
        # Results should be identical to original test
        assert results.users_count[0] == 1000
        assert results.users_count[1] == 800
        assert results.users_count[2] == 600
        assert results.users_count[3] == 400
    
    def test_funnel_steps_not_in_data(self, calculator_factory, simple_linear_funnel_data):
        """
        Test behavior when funnel steps don't exist in the data.
        Expected: Should return zero counts for non-existent steps.
        """
        # Use steps that don't exist in the data
        non_existent_steps = ['Nonexistent Step 1', 'Nonexistent Step 2']
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, non_existent_steps)
        
        # Should return zero counts
        assert results.users_count[0] == 0
        assert results.users_count[1] == 0
        assert results.conversion_rates[0] == 100.0  # First step is always 100% of its own count
        assert results.conversion_rates[1] == 0.0    # But if count is 0, then 0%
    
    def test_partial_funnel_steps_in_data(self, calculator_factory, simple_linear_funnel_data):
        """
        Test behavior when only some funnel steps exist in the data.
        Expected: Should calculate for existing steps, zero for non-existing.
        """
        # Mix of existing and non-existing steps
        mixed_steps = ['Sign Up', 'Nonexistent Step', 'Email Verification']
        
        calculator = calculator_factory()
        results = calculator.calculate_funnel_metrics(simple_linear_funnel_data, mixed_steps)
        
        # First step should have normal count
        assert results.users_count[0] == 1000
        # Second step (non-existent) should have zero
        assert results.users_count[1] == 0
        # Third step should have zero (no users can progress through non-existent step)
        assert results.users_count[2] == 0 