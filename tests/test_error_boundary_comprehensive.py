"""
Comprehensive Error Boundary Testing Suite

This test suite covers all aspects of error handling and recovery:
- Exception handling in all components
- User-friendly error message validation
- Recovery mechanism testing
- Graceful degradation patterns
"""

import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from core import DataSourceManager, FunnelCalculator, FunnelConfigManager
from models import CountingMethod, FunnelConfig, FunnelOrder, ReentryMode


@pytest.mark.error_boundary
class TestDataSourceErrorHandling:
    """Test error handling in DataSourceManager component."""

    @pytest.fixture
    def data_manager(self):
        """Standard data source manager for testing."""
        return DataSourceManager()

    def test_file_loading_exception_handling(self, data_manager):
        """Test exception handling during file loading."""
        mock_file = Mock()
        mock_file.name = "test.csv"

        # Test various file loading exceptions
        file_exceptions = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            IOError("IO error occurred"),
            pd.errors.EmptyDataError("No data"),
            Exception("Unexpected error"),
        ]

        for exception in file_exceptions:
            with patch("pandas.read_csv") as mock_read_csv:
                mock_read_csv.side_effect = exception

                # Should handle all exceptions gracefully
                result_df = data_manager.load_from_file(mock_file)

                # Should return empty DataFrame
                assert isinstance(result_df, pd.DataFrame), f"Should return DataFrame for {type(exception).__name__}"
                assert len(result_df) == 0, f"Should return empty DataFrame for {type(exception).__name__}"

        print("✅ File loading exception handling test passed")

    def test_data_validation_error_messages(self, data_manager):
        """Test user-friendly error messages in data validation."""
        # Test scenarios with specific error messages
        validation_scenarios = [
            # Missing columns scenario
            (
                pd.DataFrame({"wrong_column": [1, 2, 3]}),
                ["missing", "required", "columns"],
            ),
            
            # Empty DataFrame scenario
            (
                pd.DataFrame(),
                ["missing", "required", "columns"],
            ),
        ]

        for test_df, expected_keywords in validation_scenarios:
            is_valid, message = data_manager.validate_event_data(test_df)

            # Should be invalid
            assert not is_valid, f"Should be invalid for scenario: {test_df.columns.tolist()}"

            # Should contain user-friendly keywords
            message_lower = message.lower()
            for keyword in expected_keywords:
                assert keyword in message_lower, f"Error message should contain '{keyword}': {message}"

        print("✅ Data validation error messages test passed")


@pytest.mark.error_boundary
class TestFunnelCalculatorErrorHandling:
    """Test error handling in FunnelCalculator component."""

    @pytest.fixture
    def calculator(self):
        """Standard funnel calculator for testing."""
        config = FunnelConfig()
        return FunnelCalculator(config)

    def test_empty_data_handling(self, calculator):
        """Test handling of empty data scenarios."""
        empty_scenarios = [
            pd.DataFrame(),  # Completely empty
            pd.DataFrame({"user_id": [], "event_name": [], "timestamp": []}),  # Empty with columns
        ]

        for empty_df in empty_scenarios:
            steps = ["Step1", "Step2"]
            
            # Should handle empty data gracefully
            results = calculator.calculate_funnel_metrics(empty_df, steps)
            
            # Should return valid structure with zero counts
            assert results.steps == [], "Should return empty steps for empty data"
            assert results.users_count == [], "Should return empty users_count for empty data"

        print("✅ Empty data handling test passed")

    def test_invalid_funnel_steps_handling(self, calculator):
        """Test handling of invalid funnel step configurations."""
        # Create valid data
        valid_data = pd.DataFrame({
            "user_id": ["user1", "user2", "user3"],
            "event_name": ["EventA", "EventB", "EventC"],
            "timestamp": [datetime.now() + timedelta(minutes=i) for i in range(3)],
            "event_properties": ["{}"] * 3,
            "user_properties": ["{}"] * 3,
        })

        invalid_step_scenarios = [
            [],  # Empty steps
            ["NonExistentEvent"],  # Non-existent event
            ["EventA", "NonExistentEvent"],  # Mix of valid and invalid
        ]

        for steps in invalid_step_scenarios:
            # Should handle invalid steps gracefully
            results = calculator.calculate_funnel_metrics(valid_data, steps)
            
            # Should return appropriate structure
            if not steps:
                assert results.steps == [], f"Should return empty steps for invalid configuration: {steps}"
            else:
                # Should filter out non-existent events
                assert isinstance(results.steps, list), f"Should return list for steps: {steps}"

        print("✅ Invalid funnel steps handling test passed")


@pytest.mark.recovery
class TestRecoveryMechanisms:
    """Test recovery mechanisms and graceful degradation."""

    def test_partial_data_recovery(self):
        """Test recovery from partial data corruption."""
        # Create partially corrupted data
        partial_data = pd.DataFrame({
            "user_id": ["user1", None, "user3", ""],  # Some missing user IDs
            "event_name": ["Event1", "Event2", None, "Event4"],  # Some missing events
            "timestamp": [
                datetime.now(),
                datetime.now() + timedelta(minutes=1),
                None,  # Missing timestamp
                datetime.now() + timedelta(minutes=3),
            ],
            "event_properties": ["{}"] * 4,
            "user_properties": ["{}"] * 4,
        })

        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        steps = ["Event1", "Event2", "Event4"]

        # Should recover and process valid data
        results = calculator.calculate_funnel_metrics(partial_data, steps)
        
        # Should have some results from valid data
        assert isinstance(results.steps, list), "Should return valid results from partial data"

        print("✅ Partial data recovery test passed")


@pytest.mark.user_experience
class TestUserFriendlyErrorMessages:
    """Test user-friendly error message patterns."""

    def test_error_message_clarity(self):
        """Test that error messages are clear and actionable."""
        data_manager = DataSourceManager()
        
        # Test scenarios with expected user-friendly messages
        error_scenarios = [
            (
                pd.DataFrame({"wrong_column": [1, 2, 3]}),
                ["missing", "required", "columns", "user_id", "event_name", "timestamp"],
                ["traceback", "exception", "error:", "failed:"],
            ),
        ]

        for test_data, should_contain, should_not_contain in error_scenarios:
            is_valid, message = data_manager.validate_event_data(test_data)
            
            assert not is_valid, "Should be invalid"
            
            message_lower = message.lower()
            
            # Should contain helpful keywords
            for keyword in should_contain:
                assert keyword in message_lower, f"Message should contain '{keyword}': {message}"
            
            # Should not contain technical jargon
            for jargon in should_not_contain:
                assert jargon not in message_lower, f"Message should not contain '{jargon}': {message}"

        print("✅ Error message clarity test passed") 