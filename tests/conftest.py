"""
Professional Test Configuration and Fixtures for Funnel Analytics Platform
===========================================================================

This module provides a unified, professional approach to testing the funnel analytics platform.
Follows enterprise testing standards with proper fixtures, data factories, and utilities.

Key Principles:
1. **Reusable Fixtures**: Centralized test data and configuration
2. **Performance Testing**: Built-in timing and memory monitoring
3. **Polars-First**: Test both Polars and Pandas implementations
4. **Edge Case Coverage**: Systematic testing of boundary conditions
5. **Professional Standards**: Clear naming, documentation, type hints

Test Architecture:
- conftest.py: This file - central fixtures and utilities
- test_core_*.py: Core funnel calculation logic
- test_performance_*.py: Performance and scalability tests
- test_integration_*.py: End-to-end integration tests
- test_edge_cases_*.py: Boundary and error condition tests
"""

import json
import logging
import os

# Import system under test
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    CountingMethod,
    FunnelCalculator,
    FunnelConfig,
    FunnelOrder,
    FunnelResults,
    ReentryMode,
)

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# =============================================================================
# PROFESSIONAL TEST DATA FACTORIES
# =============================================================================


@dataclass
class TestDataSpec:
    """Specification for generating test data with controlled characteristics."""

    total_users: int = 1000
    conversion_rates: list[float] = None  # [0.8, 0.6, 0.4] means 80%, 60%, 40% convert
    time_spread_hours: int = 24
    base_timestamp: datetime = None
    include_noise_events: bool = False
    noise_event_ratio: float = 0.1
    include_properties: bool = True
    segment_distribution: dict[str, float] = None  # {'premium': 0.3, 'basic': 0.7}

    def __post_init__(self):
        if self.base_timestamp is None:
            self.base_timestamp = datetime(2024, 1, 1, 10, 0, 0)
        if self.conversion_rates is None:
            self.conversion_rates = [1.0, 0.8, 0.6, 0.4]  # 4-step funnel
        if self.segment_distribution is None:
            self.segment_distribution = {"segment_a": 0.5, "segment_b": 0.5}


class TestDataFactory:
    """Professional test data factory with controlled data generation."""

    @staticmethod
    def create_funnel_data(spec: TestDataSpec, steps: list[str]) -> pd.DataFrame:
        """
        Create realistic funnel test data with precise control over conversion rates.

        Args:
            spec: Data generation specification
            steps: List of funnel step names

        Returns:
            DataFrame with event data matching the specification
        """
        events = []
        step_count = len(steps)

        # Validate conversion rates
        if len(spec.conversion_rates) != step_count:
            raise ValueError(
                f"Conversion rates length ({len(spec.conversion_rates)}) must match steps length ({step_count})"
            )

        # Calculate cumulative users for each step
        users_per_step = []
        for i, rate in enumerate(spec.conversion_rates):
            if i == 0:
                users_per_step.append(int(spec.total_users * rate))
            else:
                users_per_step.append(int(users_per_step[0] * rate))

        # Generate segment assignments
        segment_keys = list(spec.segment_distribution.keys())
        segment_probs = list(spec.segment_distribution.values())
        user_segments = np.random.choice(segment_keys, size=spec.total_users, p=segment_probs)

        # Generate events for each step
        for step_idx, step_name in enumerate(steps):
            users_for_step = users_per_step[step_idx]

            for user_idx in range(users_for_step):
                user_id = f"user_{user_idx:05d}"

                # Calculate realistic timing
                base_time = spec.base_timestamp + timedelta(
                    hours=user_idx * spec.time_spread_hours / spec.total_users
                )
                event_time = base_time + timedelta(
                    minutes=step_idx * 30 + np.random.randint(0, 30)
                )

                # Create event properties
                event_props = {}
                user_props = {"segment": user_segments[user_idx], "user_type": "test_user"}

                if spec.include_properties:
                    event_props.update(
                        {
                            "source": np.random.choice(["organic", "paid", "referral"]),
                            "device": np.random.choice(["mobile", "desktop", "tablet"]),
                            "step_order": step_idx,
                        }
                    )

                events.append(
                    {
                        "user_id": user_id,
                        "event_name": step_name,
                        "timestamp": event_time,
                        "event_properties": (
                            json.dumps(event_props) if spec.include_properties else "{}"
                        ),
                        "user_properties": (
                            json.dumps(user_props) if spec.include_properties else "{}"
                        ),
                    }
                )

        # Add noise events if requested
        if spec.include_noise_events:
            noise_events = TestDataFactory._generate_noise_events(
                spec, users_per_step[0], spec.noise_event_ratio
            )
            events.extend(noise_events)

        df = pd.DataFrame(events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def _generate_noise_events(spec: TestDataSpec, max_users: int, ratio: float) -> list[dict]:
        """Generate noise events that shouldn't affect funnel calculations."""
        noise_events = []
        noise_event_names = ["Page View", "Click", "Scroll", "Hover", "Session Start"]
        num_noise_events = int(max_users * ratio * len(noise_event_names))

        for i in range(num_noise_events):
            user_idx = np.random.randint(0, max_users)
            event_time = spec.base_timestamp + timedelta(
                hours=np.random.randint(0, spec.time_spread_hours),
                minutes=np.random.randint(0, 60),
            )

            noise_events.append(
                {
                    "user_id": f"user_{user_idx:05d}",
                    "event_name": np.random.choice(noise_event_names),
                    "timestamp": event_time,
                    "event_properties": json.dumps({"noise": True}),
                    "user_properties": json.dumps({"user_type": "test_user"}),
                }
            )

        return noise_events

    @staticmethod
    def create_edge_case_data(case_type: str) -> pd.DataFrame:
        """Create specific edge case datasets for testing boundary conditions."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        if case_type == "empty":
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "event_name",
                    "timestamp",
                    "event_properties",
                    "user_properties",
                ]
            )

        if case_type == "single_user":
            return pd.DataFrame(
                [
                    {
                        "user_id": "single_user",
                        "event_name": "Single Event",
                        "timestamp": base_time,
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                ]
            )

        if case_type == "duplicate_events":
            # Same user, same event, same timestamp
            event = {
                "user_id": "dup_user",
                "event_name": "Duplicate Event",
                "timestamp": base_time,
                "event_properties": "{}",
                "user_properties": "{}",
            }
            return pd.DataFrame([event, event, event])

        if case_type == "out_of_order":
            # Events with timestamps in wrong order
            events = []
            for i, (event, offset) in enumerate(
                [
                    ("Step 3", 0),  # Should be last
                    ("Step 1", 1),  # Should be first
                    ("Step 2", 2),  # Should be middle
                ]
            ):
                events.append(
                    {
                        "user_id": "order_user",
                        "event_name": event,
                        "timestamp": base_time + timedelta(hours=offset),
                        "event_properties": "{}",
                        "user_properties": "{}",
                    }
                )
            return pd.DataFrame(events)

        if case_type == "missing_columns":
            return pd.DataFrame(
                [
                    {
                        "user_id": "incomplete_user",
                        "event_name": "Incomplete Event",
                        # Missing timestamp, properties
                    }
                ]
            )

        if case_type == "invalid_json":
            return pd.DataFrame(
                [
                    {
                        "user_id": "json_user",
                        "event_name": "JSON Event",
                        "timestamp": base_time,
                        "event_properties": "invalid json string",
                        "user_properties": '{"valid": "json"}',
                    }
                ]
            )

        raise ValueError(f"Unknown edge case type: {case_type}")


# =============================================================================
# PERFORMANCE MONITORING UTILITIES
# =============================================================================


class PerformanceMonitor:
    """Professional performance monitoring for tests."""

    def __init__(self):
        self.timings = {}
        self.memory_usage = {}

    def time_operation(self, operation_name: str, func, *args, **kwargs):
        """Time an operation and store the result."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        self.timings[operation_name] = end_time - start_time
        return result

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.timings:
            return {"message": "No performance data collected"}

        return {
            "timings": self.timings,
            "total_time": sum(self.timings.values()),
            "slowest_operation": max(self.timings.items(), key=lambda x: x[1]),
            "fastest_operation": min(self.timings.items(), key=lambda x: x[1]),
            "average_time": np.mean(list(self.timings.values())),
        }


# =============================================================================
# STANDARD TEST FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def base_timestamp():
    """Standard base timestamp for all tests."""
    return datetime(2024, 1, 1, 10, 0, 0)


@pytest.fixture(scope="session")
def standard_funnel_steps():
    """Standard 4-step funnel for consistent testing."""
    return ["Sign Up", "Email Verification", "First Login", "First Purchase"]


@pytest.fixture(scope="session")
def simple_funnel_steps():
    """Simple 3-step funnel for quick tests."""
    return ["Step 1", "Step 2", "Step 3"]


@pytest.fixture
def default_config():
    """Default funnel configuration optimized for testing."""
    return FunnelConfig(
        conversion_window_hours=24,
        counting_method=CountingMethod.UNIQUE_USERS,
        reentry_mode=ReentryMode.FIRST_ONLY,
        funnel_order=FunnelOrder.ORDERED,
    )


@pytest.fixture
def all_config_combinations():
    """All valid combinations of funnel configuration for comprehensive testing."""
    configs = []
    for counting_method in CountingMethod:
        for reentry_mode in ReentryMode:
            for funnel_order in FunnelOrder:
                configs.append(
                    FunnelConfig(
                        conversion_window_hours=24,
                        counting_method=counting_method,
                        reentry_mode=reentry_mode,
                        funnel_order=funnel_order,
                    )
                )
    return configs


@pytest.fixture
def calculator_factory():
    """Factory for creating FunnelCalculator instances with different configurations."""

    def _create_calculator(
        config: FunnelConfig = None, use_polars: bool = True
    ) -> FunnelCalculator:
        if config is None:
            config = FunnelConfig(
                conversion_window_hours=24,
                counting_method=CountingMethod.UNIQUE_USERS,
                reentry_mode=ReentryMode.FIRST_ONLY,
                funnel_order=FunnelOrder.ORDERED,
            )
        return FunnelCalculator(config, use_polars=use_polars)

    return _create_calculator


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture for timing tests."""
    return PerformanceMonitor()


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================


@pytest.fixture
def small_linear_funnel_data(standard_funnel_steps):
    """Small linear funnel dataset for quick tests (100 users)."""
    spec = TestDataSpec(
        total_users=100,
        conversion_rates=[1.0, 0.8, 0.6, 0.4],
        time_spread_hours=2,
        include_noise_events=False,
    )
    return TestDataFactory.create_funnel_data(spec, standard_funnel_steps)


@pytest.fixture
def medium_linear_funnel_data(standard_funnel_steps):
    """Medium linear funnel dataset for standard tests (1000 users)."""
    spec = TestDataSpec(
        total_users=1000,
        conversion_rates=[1.0, 0.8, 0.6, 0.4],
        time_spread_hours=24,
        include_noise_events=True,
        noise_event_ratio=0.1,
    )
    return TestDataFactory.create_funnel_data(spec, standard_funnel_steps)


@pytest.fixture
def large_funnel_data(standard_funnel_steps):
    """Large funnel dataset for performance testing (10000 users)."""
    spec = TestDataSpec(
        total_users=10000,
        conversion_rates=[1.0, 0.7, 0.5, 0.3],
        time_spread_hours=168,  # 1 week
        include_noise_events=True,
        noise_event_ratio=0.2,
    )
    return TestDataFactory.create_funnel_data(spec, standard_funnel_steps)


@pytest.fixture
def perfect_conversion_data(simple_funnel_steps):
    """Perfect 100% conversion funnel for boundary testing."""
    spec = TestDataSpec(
        total_users=50,
        conversion_rates=[1.0, 1.0, 1.0],
        time_spread_hours=1,
        include_noise_events=False,
    )
    return TestDataFactory.create_funnel_data(spec, simple_funnel_steps)


@pytest.fixture
def zero_conversion_data(simple_funnel_steps, base_timestamp):
    """Zero conversion funnel - all users only complete first step."""
    events = []
    for i in range(100):
        events.append(
            {
                "user_id": f"user_{i:03d}",
                "event_name": simple_funnel_steps[0],  # Only first step
                "timestamp": base_timestamp + timedelta(minutes=i),
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@pytest.fixture
def segmented_funnel_data(standard_funnel_steps):
    """Funnel data with clear segmentation for testing segment analysis."""
    spec = TestDataSpec(
        total_users=500,
        conversion_rates=[1.0, 0.8, 0.6, 0.4],
        segment_distribution={"premium": 0.3, "basic": 0.7},
        include_properties=True,
    )
    return TestDataFactory.create_funnel_data(spec, standard_funnel_steps)


@pytest.fixture
def conversion_window_test_data(simple_funnel_steps, base_timestamp):
    """Data specifically designed to test conversion window logic."""
    events = []

    # User 1: Completes funnel within window (should convert)
    for i, step in enumerate(simple_funnel_steps):
        events.append(
            {
                "user_id": "within_window_user",
                "event_name": step,
                "timestamp": base_timestamp + timedelta(hours=i * 2),  # 2 hours between steps
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    # User 2: Takes too long between steps (should not convert)
    for i, step in enumerate(simple_funnel_steps):
        events.append(
            {
                "user_id": "outside_window_user",
                "event_name": step,
                "timestamp": base_timestamp + timedelta(hours=i * 30),  # 30 hours between steps
                "event_properties": "{}",
                "user_properties": "{}",
            }
        )

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# =============================================================================
# EDGE CASE FIXTURES
# =============================================================================


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame with correct schema."""
    return TestDataFactory.create_edge_case_data("empty")


@pytest.fixture
def single_user_data():
    """Single user, single event."""
    return TestDataFactory.create_edge_case_data("single_user")


@pytest.fixture
def duplicate_events_data():
    """Multiple identical events for same user."""
    return TestDataFactory.create_edge_case_data("duplicate_events")


@pytest.fixture
def out_of_order_events_data():
    """Events with timestamps in wrong chronological order."""
    return TestDataFactory.create_edge_case_data("out_of_order")


@pytest.fixture
def invalid_json_data():
    """Data with malformed JSON in properties fields."""
    return TestDataFactory.create_edge_case_data("invalid_json")


@pytest.fixture
def simple_linear_funnel_data(base_timestamp):
    """
    Simple linear funnel test data for backward compatibility:
    - 1000 users start (Sign Up)
    - 800 users complete Email Verification
    - 600 users complete First Login
    - 400 users complete First Purchase
    """
    events = []

    # All users complete signup
    for i in range(1000):
        events.append(
            {
                "user_id": f"user_{i:04d}",
                "event_name": "Sign Up",
                "timestamp": base_timestamp + timedelta(minutes=i // 10),
                "event_properties": json.dumps({"source": "organic"}),
                "user_properties": json.dumps({"segment": "new"}),
            }
        )

    # 800 users verify email (within 2 hours)
    for i in range(800):
        events.append(
            {
                "user_id": f"user_{i:04d}",
                "event_name": "Email Verification",
                "timestamp": base_timestamp + timedelta(minutes=i // 10 + 30),
                "event_properties": json.dumps({"source": "organic"}),
                "user_properties": json.dumps({"segment": "new"}),
            }
        )

    # 600 users complete first login (within 4 hours)
    for i in range(600):
        events.append(
            {
                "user_id": f"user_{i:04d}",
                "event_name": "First Login",
                "timestamp": base_timestamp + timedelta(minutes=i // 10 + 120),
                "event_properties": json.dumps({"source": "organic"}),
                "user_properties": json.dumps({"segment": "new"}),
            }
        )

    # 400 users make first purchase (within 8 hours)
    for i in range(400):
        events.append(
            {
                "user_id": f"user_{i:04d}",
                "event_name": "First Purchase",
                "timestamp": base_timestamp + timedelta(minutes=i // 10 + 240),
                "event_properties": json.dumps({"source": "organic"}),
                "user_properties": json.dumps({"segment": "new"}),
            }
        )

    return pd.DataFrame(events)


# =============================================================================
# POLARS-SPECIFIC FIXTURES
# =============================================================================


@pytest.fixture
def polars_test_data(medium_linear_funnel_data):
    """Convert test data to Polars format for Polars-specific tests."""
    return pl.from_pandas(medium_linear_funnel_data)


@pytest.fixture
def both_formats_data(medium_linear_funnel_data):
    """Provide data in both Pandas and Polars formats for comparison tests."""
    pandas_data = medium_linear_funnel_data
    polars_data = pl.from_pandas(pandas_data)
    return {"pandas": pandas_data, "polars": polars_data}


# =============================================================================
# UTILITY FUNCTIONS FOR TESTS
# =============================================================================


def assert_funnel_results_valid(results: FunnelResults, expected_steps: list[str]):
    """Comprehensive validation of FunnelResults structure and data."""
    # Structure validation
    assert results.steps == expected_steps
    assert len(results.users_count) == len(expected_steps)
    assert len(results.conversion_rates) == len(expected_steps)
    assert len(results.drop_offs) == len(expected_steps)
    assert len(results.drop_off_rates) == len(expected_steps)

    # Data consistency validation
    assert all(count >= 0 for count in results.users_count)
    assert all(0 <= rate <= 100 for rate in results.conversion_rates)
    assert all(drop >= 0 for drop in results.drop_offs)
    assert all(0 <= rate <= 100 for rate in results.drop_off_rates)

    # Logical consistency
    assert results.conversion_rates[0] == 100.0  # First step always 100%
    assert results.drop_offs[0] == 0  # No drop-off at first step

    # User counts should be monotonically decreasing (or equal)
    for i in range(1, len(results.users_count)):
        assert results.users_count[i] <= results.users_count[i - 1], (
            f"User count increased from step {i - 1} to {i}: {results.users_count[i - 1]} -> {results.users_count[i]}"
        )


def assert_results_approximately_equal(
    result1: FunnelResults, result2: FunnelResults, tolerance: float = 0.01
):
    """Assert that two FunnelResults are approximately equal within tolerance."""
    assert result1.steps == result2.steps

    # Allow small differences due to floating point precision
    assert len(result1.users_count) == len(result2.users_count)
    for i, (count1, count2) in enumerate(zip(result1.users_count, result2.users_count)):
        assert abs(count1 - count2) <= tolerance * max(count1, count2, 1), (
            f"User counts differ at step {i}: {count1} vs {count2}"
        )

    for i, (rate1, rate2) in enumerate(zip(result1.conversion_rates, result2.conversion_rates)):
        assert abs(rate1 - rate2) <= tolerance * 100, (
            f"Conversion rates differ at step {i}: {rate1}% vs {rate2}%"
        )


def log_test_performance(test_name: str, duration: float, data_size: int):
    """Log performance metrics for test analysis."""
    logger.info(
        f"PERFORMANCE: {test_name} took {duration:.3f}s for {data_size} events "
        f"({data_size / duration:.0f} events/sec)"
    )


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers for organized test execution."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for end-to-end flows")
    config.addinivalue_line("markers", "performance: Performance and scalability tests")
    config.addinivalue_line("markers", "edge_case: Edge cases and boundary condition tests")
    config.addinivalue_line("markers", "polars: Polars-specific functionality tests")
    config.addinivalue_line("markers", "fallback: Polars to Pandas fallback detection tests")
    config.addinivalue_line("markers", "slow: Tests that take longer than 5 seconds")


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests and organize test execution."""
    for item in items:
        # Mark performance tests as slow
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Mark large data tests as slow
        if "large" in item.name.lower():
            item.add_marker(pytest.mark.slow)
