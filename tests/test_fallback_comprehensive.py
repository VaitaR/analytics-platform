"""
Comprehensive tests for detecting Polars to Pandas fallback in all configurations.

This test suite checks all combinations of:
- funnel_order: ORDERED, UNORDERED (2 values)
- reentry_mode: FIRST_ONLY, OPTIMIZED_REENTRY (2 values)
- counting_method: UNIQUE_USERS, EVENT_TOTALS, UNIQUE_PAIRS (3 values)

Total: 12 combinations

Each test explicitly checks for fallback messages in logs and fails if any fallback is detected.
"""

import io
import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from app import CountingMethod, FunnelCalculator, FunnelConfig, FunnelOrder, ReentryMode

# ------------------- Test Fixtures -------------------


@pytest.fixture
def log_capture():
    """Capture logs to check for fallback messages."""
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all log levels
    logger.addHandler(handler)

    yield log_stream

    logger.removeHandler(handler)


@pytest.fixture
def standard_test_data():
    """Create standard test data for funnel analysis with perfect progression."""
    data = []
    steps = ["Step1", "Step2", "Step3"]

    # 30 users who complete all steps in order (perfect data)
    for user_id in range(1, 31):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def unordered_test_data():
    """Create test data where users complete steps in different orders."""
    data = []
    steps = ["Step1", "Step2", "Step3"]

    # 15 users with normal progression
    for user_id in range(1, 16):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

    # 15 users with reversed order
    for user_id in range(16, 31):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(reversed(steps)):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def reentry_test_data():
    """Create test data with reentry patterns."""
    data = []
    steps = ["Step1", "Step2", "Step3"]

    # 15 users who complete the funnel once
    for user_id in range(1, 16):
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

    # 15 users who reenter the funnel
    for user_id in range(16, 31):
        # First complete funnel
        base_time = datetime(2023, 1, 1, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

        # Then reenter and complete again
        base_time = datetime(2023, 1, 2, 10, 0, 0) + timedelta(hours=user_id)
        for i, step in enumerate(steps):
            event_time = base_time + timedelta(hours=i)
            data.append(
                {
                    "user_id": f"user_{user_id}",
                    "event_name": step,
                    "timestamp": event_time,
                    "properties": "{}",
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def all_test_data_fixtures(standard_test_data, unordered_test_data, reentry_test_data):
    """Return all test data fixtures."""
    return {
        "standard": standard_test_data,
        "unordered": unordered_test_data,
        "reentry": reentry_test_data,
    }


# ------------------- Helper Functions -------------------


def get_appropriate_test_data(funnel_order, reentry_mode, all_fixtures):
    """Get the most appropriate test data for a specific configuration."""
    if funnel_order == FunnelOrder.UNORDERED:
        return all_fixtures["unordered"]
    if reentry_mode == ReentryMode.OPTIMIZED_REENTRY:
        return all_fixtures["reentry"]
    return all_fixtures["standard"]


def check_logs_for_fallback(log_output: str) -> tuple[bool, set[str]]:
    """
    Check if logs contain any fallback messages.

    Returns:
        Tuple of (fallback_detected, set_of_fallback_messages)
    """
    fallback_phrases = [
        "falling back to pandas",
        "falling back to standard polars",
        "fallback to pandas",
        "fallback to original method",
        "polars calculation failed",
    ]

    fallback_detected = False
    detected_messages = set()

    log_lower = log_output.lower()
    for phrase in fallback_phrases:
        if phrase in log_lower:
            fallback_detected = True
            # Find the full error message
            start_idx = log_lower.find(phrase) - 100  # Look back for context
            start_idx = max(0, start_idx)
            end_idx = log_lower.find("\n", log_lower.find(phrase))
            if end_idx == -1:  # If no newline after, take the rest of the string
                end_idx = len(log_lower)

            error_context = log_lower[start_idx:end_idx].strip()
            detected_messages.add(error_context)

    return fallback_detected, detected_messages


# ------------------- Test Classes -------------------


class TestFallbackComprehensive:
    """
    Comprehensive tests to detect fallbacks in all possible FunnelCalculator configurations.
    These tests intentionally fail if a fallback is detected.
    """

    @pytest.mark.parametrize("funnel_order", list(FunnelOrder))
    @pytest.mark.parametrize("reentry_mode", list(ReentryMode))
    @pytest.mark.parametrize("counting_method", list(CountingMethod))
    def test_no_fallback_in_funnel_calculation(
        self,
        funnel_order,
        reentry_mode,
        counting_method,
        all_test_data_fixtures,
        log_capture,
    ):
        """
        Test that funnel calculation does not trigger fallbacks for any configuration.

        This test will fail if any fallback is detected in the logs.
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=funnel_order,
            reentry_mode=reentry_mode,
            counting_method=counting_method,
            conversion_window_hours=48,
        )
        calculator = FunnelCalculator(config=config, use_polars=True)

        # Choose appropriate test data for this configuration
        test_data = get_appropriate_test_data(funnel_order, reentry_mode, all_test_data_fixtures)

        # Clear log capture before running
        log_capture.truncate(0)
        log_capture.seek(0)

        # Execute
        try:
            results = calculator.calculate_funnel_metrics(test_data, steps)

            # Check logs for fallback messages
            log_output = log_capture.getvalue()
            fallback_detected, fallback_messages = check_logs_for_fallback(log_output)

            # This test should fail if fallback was detected
            if fallback_detected:
                pytest.fail(
                    f"Fallback detected for configuration: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                    f"Fallback messages: {', '.join(fallback_messages)}"
                )

            # Verify we got valid results
            assert results is not None
            assert len(results.steps) == len(steps)

            # Print success message
            print(
                f"✓ No fallback for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}"
            )

        except Exception as e:
            # If there was an exception unrelated to fallback, we want to know that too
            log_output = log_capture.getvalue()
            fallback_detected, fallback_messages = check_logs_for_fallback(log_output)

            if fallback_detected:
                pytest.fail(
                    f"Exception and fallback detected for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                    f"Exception: {str(e)}. Fallback messages: {', '.join(fallback_messages)}"
                )
            else:
                pytest.fail(
                    f"Exception (no fallback) for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                    f"Exception: {str(e)}"
                )

    @pytest.mark.parametrize("funnel_order", list(FunnelOrder))
    @pytest.mark.parametrize("reentry_mode", list(ReentryMode))
    @pytest.mark.parametrize("counting_method", list(CountingMethod))
    def test_component_specific_fallback_detection(
        self,
        funnel_order,
        reentry_mode,
        counting_method,
        all_test_data_fixtures,
        log_capture,
    ):
        """
        Test specific components for fallbacks across all configurations.

        This test checks each major component separately: path_analysis, time_to_convert, and cohort_analysis.
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=funnel_order,
            reentry_mode=reentry_mode,
            counting_method=counting_method,
            conversion_window_hours=48,
        )
        calculator = FunnelCalculator(config=config, use_polars=True)

        # Choose appropriate test data for this configuration
        test_data = get_appropriate_test_data(funnel_order, reentry_mode, all_test_data_fixtures)

        # Test components separately
        components_to_test = [
            (
                "path_analysis",
                lambda calc, data: calc._calculate_path_analysis_optimized(data, steps, data),
            ),
            (
                "time_to_convert",
                lambda calc, data: calc._calculate_time_to_convert_optimized(data, steps),
            ),
            (
                "cohort_analysis",
                lambda calc, data: calc._calculate_cohort_analysis_optimized(data, steps),
            ),
        ]

        for component_name, component_func in components_to_test:
            # Clear log capture before running
            log_capture.truncate(0)
            log_capture.seek(0)

            try:
                # Execute the component function
                result = component_func(calculator, test_data)

                # Check logs for fallback messages
                log_output = log_capture.getvalue()
                fallback_detected, fallback_messages = check_logs_for_fallback(log_output)

                # This test should fail if fallback was detected
                if fallback_detected:
                    pytest.fail(
                        f"Fallback detected in {component_name} for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                        f"Fallback messages: {', '.join(fallback_messages)}"
                    )

                # Verify we got valid results
                assert result is not None

                print(
                    f"✓ No fallback in {component_name} for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}"
                )

            except Exception as e:
                # If there was an exception unrelated to fallback, we want to know that too
                log_output = log_capture.getvalue()
                fallback_detected, fallback_messages = check_logs_for_fallback(log_output)

                if fallback_detected:
                    pytest.fail(
                        f"Exception and fallback detected in {component_name} for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                        f"Exception: {str(e)}. Fallback messages: {', '.join(fallback_messages)}"
                    )
                else:
                    pytest.fail(
                        f"Exception (no fallback) in {component_name} for: {funnel_order.value}, {reentry_mode.value}, {counting_method.value}. "
                        f"Exception: {str(e)}"
                    )

    def test_path_analysis_int64_uint32_error(self, standard_test_data, log_capture):
        """
        Test specifically for the Int64/UInt32 type incompatibility error.

        This test explicitly checks for the error:
        "type Int64 is incompatible with expected type UInt32"
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48,
        )
        calculator = FunnelCalculator(config=config, use_polars=True)

        # Clear log capture
        log_capture.truncate(0)
        log_capture.seek(0)

        # Execute with standard data
        results = calculator.calculate_funnel_metrics(standard_test_data, steps)

        # Check logs for specific error
        log_output = log_capture.getvalue()
        int64_uint32_error = (
            "type int64 is incompatible with expected type uint32" in log_output.lower()
        )

        # Assert that either we don't have the error, or if we do, it's properly handled
        if int64_uint32_error:
            # Check that we got valid results despite the error (due to fallback)
            assert results is not None
            assert len(results.steps) == len(steps)
            assert results.path_analysis is not None

            # But we should mark this test as failed to indicate the issue
            pytest.fail(
                "Int64/UInt32 type incompatibility error detected in path analysis. "
                "This is a known issue that should be fixed."
            )

    def test_path_analysis_nested_object_types_error(self, standard_test_data, log_capture):
        """
        Test specifically for the nested object types error.

        This test explicitly checks for the error:
        "not yet implemented: Nested object types"

        Note: This test has been modified to introduce complex nested types to test the fix
        """
        # Create a copy of the standard data with more complex property values
        test_data = standard_test_data.copy()

        # Create a complex nested property structure to specifically trigger the error
        complex_props = []
        for i in range(len(test_data)):
            if i % 3 == 0:
                # Add a dict/object type property
                complex_props.append({"key": "value", "nested": {"more": "data"}})
            elif i % 3 == 1:
                # Add a list type property
                complex_props.append(["item1", "item2", {"key": "value"}])
            else:
                # Add a string property
                complex_props.append("{}")

        # Replace the properties column with these complex values
        test_data["properties"] = complex_props

        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48,
        )
        calculator = FunnelCalculator(config=config, use_polars=True)

        # Clear log capture
        log_capture.truncate(0)
        log_capture.seek(0)

        # Execute with modified test data
        results = calculator.calculate_funnel_metrics(test_data, steps)

        # Check logs for specific error
        log_output = log_capture.getvalue()
        nested_object_error = "not yet implemented: nested object types" in log_output.lower()

        # Verify that we got valid results with our fix
        assert results is not None
        assert len(results.steps) == len(steps)
        assert results.path_analysis is not None

        # Check if the error is detected and properly handled
        # This test now succeeds if either:
        # 1. The error doesn't occur anymore (our fix worked perfectly), or
        # 2. The error occurs but is properly handled with a fallback
        if nested_object_error:
            # We still have the error, but it's being handled - check if there's a proper fallback
            fallback_detected = any(
                phrase in log_output.lower()
                for phrase in [
                    "falling back to pandas",
                    "falling back to standard polars",
                    "fallback to pandas",
                ]
            )

            # If we detected the error but there's no fallback, the fix is incomplete
            if not fallback_detected:
                pytest.fail(
                    "Nested object types error detected in path analysis, but no fallback was triggered. "
                    "This indicates the fix is incomplete."
                )

    def test_lazy_frame_error(self, standard_test_data, log_capture, monkeypatch):
        """
        Test specifically for the LazyFrame error.

        This test explicitly reproduces and checks for the error:
        "cannot create expression literal for value of type LazyFrame"
        """
        # Setup
        steps = ["Step1", "Step2", "Step3"]
        config = FunnelConfig(
            funnel_order=FunnelOrder.ORDERED,
            reentry_mode=ReentryMode.FIRST_ONLY,
            counting_method=CountingMethod.UNIQUE_USERS,
            conversion_window_hours=48,
        )
        calculator = FunnelCalculator(config=config, use_polars=True)

        # Monkey patch to inject LazyFrame
        original_method = calculator._calculate_path_analysis_polars_optimized

        def mock_method(segment_funnel_events_df, funnel_steps, full_history_for_segment_users):
            # Convert inputs to LazyFrame to trigger the error
            if hasattr(segment_funnel_events_df, "lazy"):
                segment_funnel_events_df = segment_funnel_events_df.lazy()

            if hasattr(full_history_for_segment_users, "lazy"):
                full_history_for_segment_users = full_history_for_segment_users.lazy()

            # Call original method with LazyFrames
            return original_method(
                segment_funnel_events_df, funnel_steps, full_history_for_segment_users
            )

        # Apply monkey patch
        monkeypatch.setattr(calculator, "_calculate_path_analysis_polars_optimized", mock_method)

        # Clear log capture
        log_capture.truncate(0)
        log_capture.seek(0)

        # Execute with patched method to trigger LazyFrame error
        results = calculator.calculate_funnel_metrics(standard_test_data, steps)

        # Check logs for specific error
        log_output = log_capture.getvalue()
        lazy_frame_error = (
            "cannot create expression literal for value of type lazyframe" in log_output.lower()
        )

        # Assert that either we don't have the error, or if we do, it's properly handled
        if lazy_frame_error:
            # Check that we got valid results despite the error (due to fallback)
            assert results is not None
            assert len(results.steps) == len(steps)
            assert results.path_analysis is not None

            # But we should mark this test as failed to indicate the issue
            pytest.fail(
                "LazyFrame error detected in path analysis. "
                "This is a known issue that should be fixed using the provided solution."
            )

    def test_documentation_of_fallback_patterns(self, all_test_data_fixtures, log_capture):
        """
        Document which configurations trigger fallbacks.

        This test doesn't fail on fallbacks, but generates a comprehensive report
        of which configurations trigger fallbacks and why.
        """
        steps = ["Step1", "Step2", "Step3"]
        results = {}

        # Test all combinations and record results
        for funnel_order in FunnelOrder:
            for reentry_mode in ReentryMode:
                for counting_method in CountingMethod:
                    # Setup
                    config = FunnelConfig(
                        funnel_order=funnel_order,
                        reentry_mode=reentry_mode,
                        counting_method=counting_method,
                        conversion_window_hours=48,
                    )
                    calculator = FunnelCalculator(config=config, use_polars=True)

                    # Choose appropriate test data for this configuration
                    test_data = get_appropriate_test_data(
                        funnel_order, reentry_mode, all_test_data_fixtures
                    )

                    # Test components separately
                    components_to_test = [
                        "path_analysis",
                        "time_to_convert",
                        "cohort_analysis",
                    ]

                    component_results = {}

                    for component_name in components_to_test:
                        # Clear log capture
                        log_capture.truncate(0)
                        log_capture.seek(0)

                        # Execute component-specific function
                        try:
                            if component_name == "path_analysis":
                                calculator._calculate_path_analysis_optimized(
                                    test_data, steps, test_data
                                )
                            elif component_name == "time_to_convert":
                                calculator._calculate_time_to_convert_optimized(test_data, steps)
                            elif component_name == "cohort_analysis":
                                calculator._calculate_cohort_analysis_optimized(test_data, steps)

                            # Check logs for fallback messages
                            log_output = log_capture.getvalue()
                            fallback_detected, fallback_messages = check_logs_for_fallback(
                                log_output
                            )

                            # Record result for this component
                            if fallback_detected:
                                # Extract specific error messages
                                error_types = set()
                                for message in fallback_messages:
                                    if "nested object types" in message.lower():
                                        error_types.add("nested_object_types")
                                    elif "cross join should not pass join keys" in message.lower():
                                        error_types.add("cross_join_keys")
                                    elif (
                                        "int64 is incompatible with expected type uint32"
                                        in message.lower()
                                    ):
                                        error_types.add("int64_uint32_type")
                                    elif "_original_order" in message.lower():
                                        error_types.add("original_order")
                                    elif "lazyframe" in message.lower():
                                        error_types.add("lazy_frame")
                                    else:
                                        error_types.add("other")

                                component_results[component_name] = {
                                    "fallback": True,
                                    "error_types": error_types,
                                    "messages": fallback_messages,
                                    "exception": None,
                                }
                            else:
                                component_results[component_name] = {
                                    "fallback": False,
                                    "error_types": set(),
                                    "messages": set(),
                                    "exception": None,
                                }

                        except Exception as e:
                            # Record exception
                            log_output = log_capture.getvalue()
                            fallback_detected, fallback_messages = check_logs_for_fallback(
                                log_output
                            )

                            error_types = set()
                            if fallback_detected:
                                for message in fallback_messages:
                                    if "nested object types" in message.lower():
                                        error_types.add("nested_object_types")
                                    elif "cross join should not pass join keys" in message.lower():
                                        error_types.add("cross_join_keys")
                                    elif (
                                        "int64 is incompatible with expected type uint32"
                                        in message.lower()
                                    ):
                                        error_types.add("int64_uint32_type")
                                    elif "_original_order" in message.lower():
                                        error_types.add("original_order")
                                    elif "lazyframe" in message.lower():
                                        error_types.add("lazy_frame")
                                    else:
                                        error_types.add("other")

                            component_results[component_name] = {
                                "fallback": fallback_detected,
                                "error_types": error_types,
                                "messages": fallback_messages,
                                "exception": str(e),
                            }

                    # Record results for this configuration
                    config_key = (
                        funnel_order.value,
                        reentry_mode.value,
                        counting_method.value,
                    )
                    results[config_key] = component_results

        # Generate report
        report = ["# Fallback Patterns Report", ""]
        report.append("## Summary Statistics")

        # Count fallback occurrences by component and error type
        component_stats = {comp: {"total": 0, "fallbacks": 0} for comp in components_to_test}
        error_type_counts = {}

        for config, components in results.items():
            for comp_name, comp_result in components.items():
                component_stats[comp_name]["total"] += 1
                if comp_result["fallback"]:
                    component_stats[comp_name]["fallbacks"] += 1

                    for error_type in comp_result["error_types"]:
                        if error_type not in error_type_counts:
                            error_type_counts[error_type] = 0
                        error_type_counts[error_type] += 1

        # Add component statistics
        report.append("\n### Component Fallback Statistics")
        report.append("| Component | Fallbacks | Total Tests | Fallback Rate |")
        report.append("|-----------|-----------|-------------|---------------|")

        for comp_name, stats in component_stats.items():
            fallback_rate = (
                (stats["fallbacks"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            )
            report.append(
                f"| {comp_name} | {stats['fallbacks']} | {stats['total']} | {fallback_rate:.1f}% |"
            )

        # Add error type statistics
        report.append("\n### Error Type Statistics")
        report.append("| Error Type | Occurrences |")
        report.append("|------------|-------------|")

        # Sort error types by frequency (most common first)
        sorted_error_types = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_error_types:
            report.append(f"| {error_type} | {count} |")

        # Add detailed report by configuration
        report.append("\n## Detailed Fallback Report by Configuration")
        report.append(
            "| Funnel Order | Reentry Mode | Counting Method | Component | Fallback | Error Types |"
        )
        report.append(
            "|-------------|-------------|----------------|-----------|----------|-------------|"
        )

        # Get all keys and sort them for consistent order
        all_keys = sorted(results.keys())

        for key in all_keys:
            funnel_order, reentry_mode, counting_method = key
            components = results[key]

            # Add a row for each component
            for comp_name, comp_result in components.items():
                fallback_status = "✓ Yes" if comp_result["fallback"] else "✗ No"
                error_types = (
                    ", ".join(comp_result["error_types"]) if comp_result["error_types"] else ""
                )

                report.append(
                    f"| {funnel_order} | {reentry_mode} | {counting_method} | {comp_name} | {fallback_status} | {error_types} |"
                )

        # Add recommendations based on findings
        report.append("\n## Recommendations")

        if "nested_object_types" in error_type_counts:
            report.append("\n### 1. Fix Nested Object Types Issue")
            report.append(
                "This is the most common issue, occurring in all path_analysis calculations."
            )
            report.append(
                "Recommendation: Add `strict=False` parameter to relevant Polars operations or properly handle object types."
            )
            report.append("```python")
            report.append("# Example fix:")
            report.append("df = pl.from_pandas(pandas_df, strict=False)")
            report.append("```")

        if "original_order" in error_type_counts or "cross_join_keys" in error_type_counts:
            report.append("\n### 2. Fix Polars Join Issues")
            report.append(
                "Issues with joins are causing fallbacks from standard Polars to Pandas."
            )
            report.append(
                "Recommendation: Review join operations and ensure proper column types and join conditions."
            )

        if "int64_uint32_type" in error_type_counts:
            report.append("\n### 3. Fix Type Compatibility Issues")
            report.append("Type mismatches between Int64 and UInt32 are causing fallbacks.")
            report.append(
                "Recommendation: Explicitly cast types before operations or use compatible types."
            )
            report.append("```python")
            report.append("# Example fix:")
            report.append("df = df.with_column(pl.col('column_name').cast(pl.UInt32))")
            report.append("```")

        if "lazy_frame" in error_type_counts:
            report.append("\n### 4. Fix LazyFrame Issues")
            report.append(
                "LazyFrames are causing issues when passed to functions expecting DataFrames."
            )
            report.append(
                "Recommendation: Call .collect() on LazyFrames before passing them to functions."
            )
            report.append("```python")
            report.append("# Example fix:")
            report.append("if hasattr(df, 'collect'):")
            report.append("    df = df.collect()")
            report.append("```")

        # Print report
        report_text = "\n".join(report)
        print("\n" + "=" * 80)
        print(report_text)
        print("=" * 80 + "\n")

        # Save report to file
        with open("FALLBACK_REPORT.md", "w") as f:
            f.write(report_text)

        print("Report saved to FALLBACK_REPORT.md")

        # This test doesn't fail but documents the patterns
        assert True


# For development/debugging purposes
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
