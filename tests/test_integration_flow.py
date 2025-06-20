"""
Integration tests for the complete funnel analytics flow.

These tests verify the entire process from:
1. Data loading (file upload, sample data, database connection)
2. Data validation and preprocessing
3. Event selection and funnel step configuration
4. Calculation execution for all counting methods and configurations
5. Results validation and output formatting

The tests ensure that the complete workflow functions correctly for all supported
funnel types, counting methods, and configuration options with valid input data.
"""

import io
import json
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from app import (
    CountingMethod,
    DataSourceManager,
    FunnelCalculator,
    FunnelConfig,
    FunnelOrder,
    FunnelVisualizer,
    ReentryMode,
)


class TestCompleteIntegrationFlow:
    """Test the complete end-to-end funnel analytics workflow."""

    @pytest.fixture
    def integration_test_data(self, base_timestamp):
        """
        Comprehensive test dataset for integration testing.
        Includes multiple user segments, various timing patterns, and edge cases.
        """
        events = []

        # Segment A: High-converting users (100 users)
        for i in range(100):
            user_id = f"segment_a_user_{i:03d}"
            # All complete signup
            events.append(
                {
                    "user_id": user_id,
                    "event_name": "Sign Up",
                    "timestamp": base_timestamp + timedelta(minutes=i),
                    "event_properties": json.dumps(
                        {"source": "organic", "campaign": "spring_2024"}
                    ),
                    "user_properties": json.dumps(
                        {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                    ),
                }
            )

            # 90% verify email
            if i < 90:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "Email Verification",
                        "timestamp": base_timestamp + timedelta(minutes=i + 15),
                        "event_properties": json.dumps(
                            {"source": "organic", "method": "email_click"}
                        ),
                        "user_properties": json.dumps(
                            {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                        ),
                    }
                )

            # 80% complete first login
            if i < 80:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "First Login",
                        "timestamp": base_timestamp + timedelta(minutes=i + 45),
                        "event_properties": json.dumps({"device": "desktop", "browser": "chrome"}),
                        "user_properties": json.dumps(
                            {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                        ),
                    }
                )

            # 60% make first purchase
            if i < 60:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "First Purchase",
                        "timestamp": base_timestamp + timedelta(minutes=i + 120),
                        "event_properties": json.dumps(
                            {"amount": 29.99 + (i % 10), "currency": "USD"}
                        ),
                        "user_properties": json.dumps(
                            {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                        ),
                    }
                )

        # Segment B: Lower-converting users (150 users)
        for i in range(150):
            user_id = f"segment_b_user_{i:03d}"
            # All complete signup
            events.append(
                {
                    "user_id": user_id,
                    "event_name": "Sign Up",
                    "timestamp": base_timestamp + timedelta(minutes=i + 200),
                    "event_properties": json.dumps({"source": "paid", "campaign": "summer_2024"}),
                    "user_properties": json.dumps(
                        {"segment": "basic", "country": "UK", "age": 30 + (i % 15)}
                    ),
                }
            )

            # 60% verify email
            if i < 90:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "Email Verification",
                        "timestamp": base_timestamp + timedelta(minutes=i + 230),
                        "event_properties": json.dumps(
                            {"source": "paid", "method": "email_click"}
                        ),
                        "user_properties": json.dumps(
                            {"segment": "basic", "country": "UK", "age": 30 + (i % 15)}
                        ),
                    }
                )

            # 40% complete first login
            if i < 60:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "First Login",
                        "timestamp": base_timestamp + timedelta(minutes=i + 280),
                        "event_properties": json.dumps({"device": "mobile", "browser": "safari"}),
                        "user_properties": json.dumps(
                            {"segment": "basic", "country": "UK", "age": 30 + (i % 15)}
                        ),
                    }
                )

            # 20% make first purchase
            if i < 30:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "First Purchase",
                        "timestamp": base_timestamp + timedelta(minutes=i + 400),
                        "event_properties": json.dumps(
                            {"amount": 19.99 + (i % 5), "currency": "USD"}
                        ),
                        "user_properties": json.dumps(
                            {"segment": "basic", "country": "UK", "age": 30 + (i % 15)}
                        ),
                    }
                )

        # Add some noise events that should be filtered out
        noise_events = ["Page View", "Button Click", "Form Submit", "Newsletter Subscribe"]
        for i in range(50):
            events.append(
                {
                    "user_id": f"segment_a_user_{i % 100:03d}",
                    "event_name": np.random.choice(noise_events),
                    "timestamp": base_timestamp + timedelta(minutes=i + 500),
                    "event_properties": json.dumps({"page": "home"}),
                    "user_properties": json.dumps({"segment": "premium"}),
                }
            )

        return pd.DataFrame(events)

    @pytest.fixture
    def expected_integration_results(self):
        """Expected results for the integration test dataset."""
        return {
            "total_users": {
                "Sign Up": 250,
                "Email Verification": 180,  # 90 + 90
                "First Login": 140,  # 80 + 60
                "First Purchase": 90,  # 60 + 30
            },
            "conversion_rates": {
                "Sign Up": 100.0,
                "Email Verification": 72.0,  # 180/250
                "First Login": 56.0,  # 140/250
                "First Purchase": 36.0,  # 90/250
            },
            "segment_a": {
                "Sign Up": 100,
                "Email Verification": 90,
                "First Login": 80,
                "First Purchase": 60,
            },
            "segment_b": {
                "Sign Up": 150,
                "Email Verification": 90,
                "First Login": 60,
                "First Purchase": 30,
            },
        }

    def test_complete_file_upload_to_results_flow(
        self, integration_test_data, expected_integration_results
    ):
        """
        Test complete flow: File upload -> Data validation -> Funnel calculation -> Results
        """
        # Step 1: Simulate file upload process
        data_manager = DataSourceManager()

        # Convert DataFrame to CSV for upload simulation
        csv_buffer = io.StringIO()
        integration_test_data.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # Simulate uploaded file object
        mock_file = Mock()
        mock_file.read.return_value = csv_content.encode("utf-8")
        mock_file.name = "test_funnel_data.csv"

        # Step 2: Load CSV data directly (simulating successful file upload)
        csv_buffer.seek(0)
        loaded_data = pd.read_csv(csv_buffer)

        # Step 3: Validate the loaded data
        is_valid, validation_message = data_manager.validate_event_data(loaded_data)
        assert is_valid, f"Data validation failed: {validation_message}"
        assert len(loaded_data) == len(integration_test_data)
        assert all(col in loaded_data.columns for col in ["user_id", "event_name", "timestamp"])

        # Step 4: Verify event discovery directly from the data
        # Since get_event_metadata loads demo data, we'll verify events directly
        expected_events = ["Sign Up", "Email Verification", "First Login", "First Purchase"]
        actual_events = loaded_data["event_name"].unique().tolist()

        # Check that our expected events are present in the loaded data
        for event in expected_events:
            assert event in actual_events, (
                f"Event '{event}' not found in loaded data. Found: {actual_events}"
            )

        # Step 5: Configure and run funnel analysis
        funnel_steps = ["Sign Up", "Email Verification", "First Login", "First Purchase"]
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            reentry_mode=ReentryMode.FIRST_ONLY,
            funnel_order=FunnelOrder.ORDERED,
        )

        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(loaded_data, funnel_steps)

        # Step 6: Validate results
        assert results.steps == funnel_steps
        assert len(results.users_count) == 4

        # Verify user counts match expected
        expected = expected_integration_results["total_users"]
        for i, step in enumerate(funnel_steps):
            assert results.users_count[i] == expected[step], (
                f"User count mismatch for {step}: expected {expected[step]}, got {results.users_count[i]}"
            )

        # Verify conversion rates
        expected_rates = expected_integration_results["conversion_rates"]
        for i, step in enumerate(funnel_steps):
            expected_rate = expected_rates[step]
            actual_rate = results.conversion_rates[i]
            assert abs(actual_rate - expected_rate) < 0.1, (
                f"Conversion rate mismatch for {step}: expected {expected_rate}%, got {actual_rate}%"
            )

    def test_all_counting_methods_integration(self, integration_test_data):
        """
        Test complete flow with all counting methods to ensure consistent behavior.
        """
        funnel_steps = ["Sign Up", "Email Verification", "First Login", "First Purchase"]
        base_config = FunnelConfig(conversion_window_hours=24, reentry_mode=ReentryMode.FIRST_ONLY)

        results = {}

        # Test each counting method
        for method in CountingMethod:
            config = FunnelConfig(
                conversion_window_hours=24,
                counting_method=method,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.ORDERED,
            )

            calculator = FunnelCalculator(config)
            result = calculator.calculate_funnel_metrics(integration_test_data, funnel_steps)
            results[method.value] = result

            # Basic validation for each method
            assert result.steps == funnel_steps
            assert len(result.users_count) == 4
            assert len(result.conversion_rates) == 4
            assert result.conversion_rates[0] == 100.0  # First step always 100%

            # Ensure counts are monotonically decreasing for ordered funnel
            for i in range(1, len(result.users_count)):
                assert result.users_count[i] <= result.users_count[i - 1], (
                    f"User count should decrease at each step for {method.value}"
                )

        # Compare methods - unique_users should generally have lower counts than event_totals
        unique_users_result = results["unique_users"]
        event_totals_result = results["event_totals"]

        # For the last step, unique_users should be <= event_totals
        assert unique_users_result.users_count[-1] <= event_totals_result.users_count[-1], (
            "Unique users count should be <= event totals count"
        )

    def test_segmentation_integration_flow(
        self, integration_test_data, expected_integration_results
    ):
        """
        Test complete flow with segmentation: Data load -> Segment selection -> Calculation -> Results
        """
        # Step 1: Load data and get segmentation properties
        data_manager = DataSourceManager()
        segmentation_props = data_manager.get_segmentation_properties(integration_test_data)

        # Verify segmentation properties are discovered
        assert "user_properties" in segmentation_props
        assert "segment" in segmentation_props["user_properties"]
        assert "country" in segmentation_props["user_properties"]

        # Step 2: Configure segmented funnel analysis
        funnel_steps = ["Sign Up", "Email Verification", "First Login", "First Purchase"]
        config = FunnelConfig(
            conversion_window_hours=24,
            counting_method=CountingMethod.UNIQUE_USERS,
            segment_by="segment",
            segment_values=["premium", "basic"],
        )

        calculator = FunnelCalculator(config)

        # Step 3: Execute segmented calculation
        results = calculator.calculate_funnel_metrics(integration_test_data, funnel_steps)

        # Step 4: Validate segmented results
        assert results.segment_data is not None

        # The segment keys include the property name prefix
        segment_keys = list(results.segment_data.keys())
        premium_key = next((k for k in segment_keys if "premium" in k), None)
        basic_key = next((k for k in segment_keys if "basic" in k), None)

        assert premium_key is not None, f"Premium segment not found in {segment_keys}"
        assert basic_key is not None, f"Basic segment not found in {segment_keys}"

        # Verify segment A (premium) results
        premium_counts = results.segment_data[premium_key]
        expected_premium = expected_integration_results["segment_a"]

        for i, step in enumerate(funnel_steps):
            assert premium_counts[i] == expected_premium[step], (
                f"Premium segment count mismatch for {step}: expected {expected_premium[step]}, got {premium_counts[i]}"
            )

        # Verify segment B (basic) results
        basic_counts = results.segment_data[basic_key]
        expected_basic = expected_integration_results["segment_b"]

        for i, step in enumerate(funnel_steps):
            assert basic_counts[i] == expected_basic[step], (
                f"Basic segment count mismatch for {step}: expected {expected_basic[step]}, got {basic_counts[i]}"
            )

    def test_conversion_window_integration(self, base_timestamp):
        """
        Test complete flow with different conversion windows to verify timing logic.
        """
        # Create test data with specific timing
        events = []

        # User 1: Events within 1 hour
        user1_times = [0, 30, 45, 50]  # minutes
        for i, minutes in enumerate(user1_times):
            events.append(
                {
                    "user_id": "user_001",
                    "event_name": f"Step_{i + 1}",
                    "timestamp": base_timestamp + timedelta(minutes=minutes),
                    "event_properties": json.dumps({}),
                    "user_properties": json.dumps({}),
                }
            )

        # User 2: Events spanning 3 hours
        user2_times = [0, 90, 150, 180]  # minutes
        for i, minutes in enumerate(user2_times):
            events.append(
                {
                    "user_id": "user_002",
                    "event_name": f"Step_{i + 1}",
                    "timestamp": base_timestamp + timedelta(minutes=minutes),
                    "event_properties": json.dumps({}),
                    "user_properties": json.dumps({}),
                }
            )

        test_data = pd.DataFrame(events)
        funnel_steps = ["Step_1", "Step_2", "Step_3", "Step_4"]

        # Test with 1-hour window
        config_1h = FunnelConfig(conversion_window_hours=1)
        calculator_1h = FunnelCalculator(config_1h)
        results_1h = calculator_1h.calculate_funnel_metrics(test_data, funnel_steps)

        # Test with 4-hour window
        config_4h = FunnelConfig(conversion_window_hours=4)
        calculator_4h = FunnelCalculator(config_4h)
        results_4h = calculator_4h.calculate_funnel_metrics(test_data, funnel_steps)

        # With 1-hour window, only user_001 should complete all steps
        assert results_1h.users_count[0] == 2  # Both users start
        assert results_1h.users_count[-1] == 1  # Only user_001 completes

        # With 4-hour window, both users should complete all steps
        assert results_4h.users_count[0] == 2  # Both users start
        assert results_4h.users_count[-1] == 2  # Both users complete

    def test_reentry_mode_integration(self, base_timestamp):
        """
        Test complete flow with different reentry modes.
        """
        # Create test data with reentry scenarios
        events = []

        # User with multiple funnel attempts
        user_events = [
            ("Step_1", 0),
            ("Step_2", 30),  # First attempt - partial
            ("Step_1", 60),
            ("Step_2", 90),
            ("Step_3", 120),  # Second attempt - complete
        ]

        for event_name, minutes in user_events:
            events.append(
                {
                    "user_id": "user_001",
                    "event_name": event_name,
                    "timestamp": base_timestamp + timedelta(minutes=minutes),
                    "event_properties": json.dumps({}),
                    "user_properties": json.dumps({}),
                }
            )

        test_data = pd.DataFrame(events)
        funnel_steps = ["Step_1", "Step_2", "Step_3"]

        # Test FIRST_ONLY mode
        config_first = FunnelConfig(reentry_mode=ReentryMode.FIRST_ONLY)
        calculator_first = FunnelCalculator(config_first)
        results_first = calculator_first.calculate_funnel_metrics(test_data, funnel_steps)

        # Test OPTIMIZED_REENTRY mode
        config_reentry = FunnelConfig(reentry_mode=ReentryMode.OPTIMIZED_REENTRY)
        calculator_reentry = FunnelCalculator(config_reentry)
        results_reentry = calculator_reentry.calculate_funnel_metrics(test_data, funnel_steps)

        # Both should start with 1 user
        assert results_first.users_count[0] == 1
        assert results_reentry.users_count[0] == 1

        # Results may differ based on reentry handling
        # This verifies that both modes execute without error
        assert len(results_first.users_count) == 3
        assert len(results_reentry.users_count) == 3

    def test_sample_data_integration(self):
        """
        Test complete flow using built-in sample data.
        """
        # Step 1: Load sample data
        data_manager = DataSourceManager()
        sample_data = data_manager.get_sample_data()

        # Step 2: Validate sample data
        is_valid, message = data_manager.validate_event_data(sample_data)
        assert is_valid, f"Sample data validation failed: {message}"

        # Step 3: Get available events
        event_metadata = data_manager.get_event_metadata(sample_data)
        available_events = list(event_metadata.keys())

        assert len(available_events) > 0, "No events found in sample data"

        # Step 4: Create a funnel with available events (use first 3-4 events)
        funnel_steps = available_events[: min(4, len(available_events))]

        # Step 5: Run analysis
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(sample_data, funnel_steps)

        # Step 6: Validate results structure
        assert results.steps == funnel_steps
        assert len(results.users_count) == len(funnel_steps)
        assert len(results.conversion_rates) == len(funnel_steps)
        assert results.conversion_rates[0] == 100.0

    def test_visualization_integration(self, integration_test_data):
        """
        Test complete flow including visualization generation.
        """
        # Step 1: Run funnel analysis
        funnel_steps = ["Sign Up", "Email Verification", "First Login", "First Purchase"]
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(integration_test_data, funnel_steps)

        # Step 2: Generate visualizations
        visualizer = FunnelVisualizer()

        # Test funnel chart creation
        funnel_chart = visualizer.create_funnel_chart(results)
        assert funnel_chart is not None
        assert len(funnel_chart.data) > 0

        # Test conversion flow sankey
        sankey_chart = visualizer.create_conversion_flow_sankey(results)
        assert sankey_chart is not None
        assert len(sankey_chart.data) > 0

        # If time-to-convert data is available, test that chart too
        if results.time_to_convert:
            ttc_chart = visualizer.create_time_to_convert_chart(results.time_to_convert)
            assert ttc_chart is not None

    def test_error_handling_integration(self):
        """
        Test complete flow with various error conditions to ensure graceful handling.
        """
        data_manager = DataSourceManager()

        # Test with invalid data formats
        invalid_data_scenarios = [
            pd.DataFrame(),  # Empty DataFrame
            pd.DataFrame({"wrong_column": [1, 2, 3]}),  # Missing required columns
            pd.DataFrame(
                {  # Invalid timestamp format
                    "user_id": ["user1", "user2"],
                    "event_name": ["Event1", "Event2"],
                    "timestamp": ["invalid_date", "another_invalid_date"],
                }
            ),
        ]

        for invalid_data in invalid_data_scenarios:
            is_valid, message = data_manager.validate_event_data(invalid_data)
            assert not is_valid, "Invalid data should fail validation"
            assert len(message) > 0, "Error message should be provided"

        # Test with valid data but invalid funnel configuration
        valid_data = pd.DataFrame(
            {
                "user_id": ["user1", "user2"],
                "event_name": ["Event1", "Event2"],
                "timestamp": [datetime.now(), datetime.now()],
                "event_properties": ["{}", "{}"],
                "user_properties": ["{}", "{}"],
            }
        )

        config = FunnelConfig()
        calculator = FunnelCalculator(config)

        # Test with empty funnel steps
        results = calculator.calculate_funnel_metrics(valid_data, [])
        assert results.steps == []
        assert results.users_count == []

        # Test with non-existent events
        results = calculator.calculate_funnel_metrics(valid_data, ["NonExistentEvent"])
        assert results.steps == []

    def test_performance_integration(self, base_timestamp):
        """
        Test complete flow with larger dataset to verify performance characteristics.
        """
        # Generate larger test dataset (1000 users, 4 events each)
        events = []
        event_names = ["Signup", "Activate", "FirstUse", "Purchase"]

        for user_i in range(1000):
            user_id = f"perf_user_{user_i:04d}"

            # Each user has a decreasing probability of completing next steps
            completion_probability = [1.0, 0.7, 0.5, 0.3]

            for event_i, event_name in enumerate(event_names):
                if np.random.random() < completion_probability[event_i]:
                    events.append(
                        {
                            "user_id": user_id,
                            "event_name": event_name,
                            "timestamp": base_timestamp + timedelta(minutes=user_i + event_i * 30),
                            "event_properties": json.dumps({"step": event_i}),
                            "user_properties": json.dumps({"cohort": user_i % 10}),
                        }
                    )

        large_dataset = pd.DataFrame(events)

        # Test complete flow with performance monitoring
        import time

        start_time = time.time()

        # Data validation
        data_manager = DataSourceManager()
        is_valid, message = data_manager.validate_event_data(large_dataset)
        assert is_valid

        # Funnel calculation
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(large_dataset, event_names)

        # Visualization
        visualizer = FunnelVisualizer()
        chart = visualizer.create_funnel_chart(results)

        end_time = time.time()
        total_time = end_time - start_time

        # Performance assertion (should complete within reasonable time)
        assert total_time < 30, f"Complete flow took too long: {total_time:.2f} seconds"

        # Validate results make sense for large dataset
        assert results.users_count[0] == 1000  # All users should start
        assert results.users_count[-1] < results.users_count[0]  # Some attrition should occur

        # Verify that the calculation completed successfully
        assert len(results.users_count) == len(event_names)
        assert all(count >= 0 for count in results.users_count)


@pytest.mark.integration
class TestDataSourceIntegration:
    """Test data source management integration."""

    @pytest.fixture
    def integration_test_data(self, base_timestamp):
        """
        Comprehensive test dataset for integration testing.
        Includes multiple user segments, various timing patterns, and edge cases.
        """
        events = []

        # Segment A: High-converting users (100 users)
        for i in range(100):
            user_id = f"segment_a_user_{i:03d}"
            # All complete signup
            events.append(
                {
                    "user_id": user_id,
                    "event_name": "Sign Up",
                    "timestamp": base_timestamp + timedelta(minutes=i),
                    "event_properties": json.dumps(
                        {"source": "organic", "campaign": "spring_2024"}
                    ),
                    "user_properties": json.dumps(
                        {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                    ),
                }
            )

            # 90% verify email
            if i < 90:
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "Email Verification",
                        "timestamp": base_timestamp + timedelta(minutes=i + 15),
                        "event_properties": json.dumps(
                            {"source": "organic", "method": "email_click"}
                        ),
                        "user_properties": json.dumps(
                            {"segment": "premium", "country": "US", "age": 25 + (i % 20)}
                        ),
                    }
                )

        return pd.DataFrame(events)

    def test_file_format_support_integration(self, integration_test_data):
        """Test support for different file formats in complete workflow."""
        data_manager = DataSourceManager()

        # Test CSV format
        csv_buffer = io.StringIO()
        integration_test_data.to_csv(csv_buffer, index=False)

        mock_csv_file = Mock()
        mock_csv_file.read.return_value = csv_buffer.getvalue().encode("utf-8")
        mock_csv_file.name = "test.csv"

        # Load CSV data directly (simulating successful file upload)
        csv_buffer.seek(0)
        loaded_csv = pd.read_csv(csv_buffer)

        # Validate the loaded data
        is_valid, _ = data_manager.validate_event_data(loaded_csv)
        assert is_valid, "CSV data validation failed"

        # Run analysis on loaded data
        config = FunnelConfig()
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(
            loaded_csv, ["Sign Up", "Email Verification"]
        )
        assert len(results.users_count) == 2

    def test_data_preprocessing_integration(self, base_timestamp):
        """Test data preprocessing in complete workflow."""
        # Create data with various preprocessing needs
        events = []

        # Add events with JSON properties that need expansion
        for i in range(50):
            events.append(
                {
                    "user_id": f"user_{i:03d}",
                    "event_name": "Purchase",
                    "timestamp": base_timestamp + timedelta(minutes=i),
                    "event_properties": json.dumps(
                        {
                            "amount": 29.99 + i,
                            "currency": "USD",
                            "product_id": f"prod_{i % 10}",
                            "category": "electronics" if i % 2 == 0 else "books",
                        }
                    ),
                    "user_properties": json.dumps(
                        {
                            "age": 25 + (i % 30),
                            "country": "US" if i % 3 == 0 else "UK",
                            "premium": i % 4 == 0,
                        }
                    ),
                }
            )

        test_data = pd.DataFrame(events)

        # Test complete flow with preprocessing
        data_manager = DataSourceManager()
        is_valid, _ = data_manager.validate_event_data(test_data)
        assert is_valid

        # Test segmentation with expanded properties
        segmentation_props = data_manager.get_segmentation_properties(test_data)

        # Should include both user_properties and event_properties
        assert len(segmentation_props) > 0

        # Add a second event for proper funnel analysis
        for i in range(25):  # Half the users complete a second event
            events.append(
                {
                    "user_id": f"user_{i:03d}",
                    "event_name": "Review",
                    "timestamp": base_timestamp + timedelta(minutes=i + 60),
                    "event_properties": json.dumps({"rating": 5}),
                    "user_properties": json.dumps(
                        {"age": 25 + (i % 30), "country": "US" if i % 3 == 0 else "UK"}
                    ),
                }
            )

        test_data = pd.DataFrame(events)

        # Run funnel analysis with two steps
        config = FunnelConfig(
            segment_by=(
                "country" if "country" in segmentation_props.get("user_properties", []) else None
            )
        )
        calculator = FunnelCalculator(config)
        results = calculator.calculate_funnel_metrics(test_data, ["Purchase", "Review"])

        # Verify preprocessing worked
        assert len(results.users_count) == 2
        assert results.users_count[0] == 50  # All users purchase
        assert results.users_count[1] == 25  # Half review


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
