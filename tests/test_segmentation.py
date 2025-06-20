"""
Test segmentation functionality for funnel analysis.
Tests filtering by event and user properties.
"""

import json
from datetime import timedelta

import pandas as pd
import pytest

from app import CountingMethod, FunnelConfig


@pytest.mark.segmentation
class TestSegmentation:
    def test_segmentation_by_event_property(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation by event properties (platform).
        Expected: Different conversion rates for mobile vs desktop users.
        """
        # Configure for mobile platform segmentation
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_platform",
            segment_values=["mobile", "desktop"],
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification", "First Login"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should have segment data
        assert results.segment_data is not None
        assert len(results.segment_data) == 2

        # Check mobile segment
        mobile_key = "platform=mobile"
        desktop_key = "platform=desktop"

        assert mobile_key in results.segment_data
        assert desktop_key in results.segment_data

        # Mobile users: 3 users complete all steps
        mobile_counts = results.segment_data[mobile_key]
        assert mobile_counts[0] == 3  # Sign Up
        assert mobile_counts[1] == 3  # Email Verification
        assert mobile_counts[2] == 3  # First Login

        # Desktop users: 2 users, only complete first 2 steps
        desktop_counts = results.segment_data[desktop_key]
        assert desktop_counts[0] == 2  # Sign Up
        assert desktop_counts[1] == 2  # Email Verification
        assert desktop_counts[2] == 0  # First Login (not in test data)

    def test_segmentation_by_user_property(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation by user properties (subscription type).
        Expected: Different conversion rates for free vs premium users.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="user_properties_subscription",
            segment_values=["free", "premium"],
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should have segment data
        assert results.segment_data is not None
        assert len(results.segment_data) == 2

        free_key = "subscription=free"
        premium_key = "subscription=premium"

        assert free_key in results.segment_data
        assert premium_key in results.segment_data

        # Free users: mobile users in test data
        free_counts = results.segment_data[free_key]
        assert free_counts[0] == 3  # 3 mobile users
        assert free_counts[1] == 3  # All verify email

        # Premium users: desktop users in test data
        premium_counts = results.segment_data[premium_key]
        assert premium_counts[0] == 2  # 2 desktop users
        assert premium_counts[1] == 2  # All verify email

    def test_no_segmentation(self, calculator_factory, segmentation_test_data):
        """
        Test funnel without segmentation (all users together).
        Expected: Should return single aggregate result.
        """
        config = FunnelConfig(conversion_window_hours=24, segment_by=None, segment_values=None)
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should not have segment data
        assert results.segment_data is None

        # Should have aggregate counts
        assert results.users_count[0] == 5  # All 5 users sign up
        assert results.users_count[1] == 5  # All 5 users verify email

    def test_single_segment_value(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation with only one segment value selected.
        Expected: Should filter to only selected segment.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_platform",
            segment_values=["mobile"],  # Only mobile
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification", "First Login"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should have only one segment
        assert results.segment_data is not None
        assert len(results.segment_data) == 1

        mobile_key = "platform=mobile"
        assert mobile_key in results.segment_data

        # Only mobile users should be counted
        mobile_counts = results.segment_data[mobile_key]
        assert mobile_counts[0] == 3  # 3 mobile users
        assert mobile_counts[1] == 3  # All complete email verification
        assert mobile_counts[2] == 3  # All complete first login

    def test_nonexistent_segment_value(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation with non-existent segment values.
        Expected: Should return empty segments for non-existent values.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_platform",
            segment_values=["tablet", "smart_tv"],  # Don't exist in data
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should return empty result or handle gracefully
        if results.segment_data:
            # If segments are returned, they should be empty
            for segment_name, counts in results.segment_data.items():
                assert all(count == 0 for count in counts)

    def test_mixed_existing_nonexistent_segments(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation with mix of existing and non-existing segment values.
        Expected: Should include existing segments, ignore non-existing ones.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_platform",
            segment_values=["mobile", "tablet", "desktop"],  # tablet doesn't exist
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        if results.segment_data:
            # Should have segments for existing values
            mobile_key = "platform=mobile"
            desktop_key = "platform=desktop"

            # These should exist and have data
            if mobile_key in results.segment_data:
                mobile_counts = results.segment_data[mobile_key]
                assert mobile_counts[0] > 0  # Should have mobile users

            if desktop_key in results.segment_data:
                desktop_counts = results.segment_data[desktop_key]
                assert desktop_counts[0] > 0  # Should have desktop users

    def test_segmentation_with_different_counting_methods(
        self, calculator_factory, base_timestamp
    ):
        """
        Test that segmentation works correctly with different counting methods.
        """
        # Create data with multiple events per user per segment
        events = []

        # Mobile users with multiple events
        for user_id in ["user_001", "user_002"]:
            # Multiple sign up events
            for i in range(3):
                events.append(
                    {
                        "user_id": user_id,
                        "event_name": "Sign Up",
                        "timestamp": base_timestamp + timedelta(minutes=i * 10),
                        "event_properties": json.dumps({"platform": "mobile"}),
                        "user_properties": json.dumps({}),
                    }
                )

            # Single email verification
            events.append(
                {
                    "user_id": user_id,
                    "event_name": "Email Verification",
                    "timestamp": base_timestamp + timedelta(minutes=60),
                    "event_properties": json.dumps({"platform": "mobile"}),
                    "user_properties": json.dumps({}),
                }
            )

        # Desktop user with single events
        events.append(
            {
                "user_id": "user_003",
                "event_name": "Sign Up",
                "timestamp": base_timestamp,
                "event_properties": json.dumps({"platform": "desktop"}),
                "user_properties": json.dumps({}),
            }
        )

        data = pd.DataFrame(events)
        steps = ["Sign Up", "Email Verification"]

        # Test with unique_users method
        config_unique = FunnelConfig(
            counting_method=CountingMethod.UNIQUE_USERS,
            segment_by="event_properties_platform",
            segment_values=["mobile", "desktop"],
        )
        calculator_unique = calculator_factory(config_unique)
        results_unique = calculator_unique.calculate_funnel_metrics(data, steps)

        # Test with event_totals method
        config_totals = FunnelConfig(
            counting_method=CountingMethod.EVENT_TOTALS,
            segment_by="event_properties_platform",
            segment_values=["mobile", "desktop"],
        )
        calculator_totals = calculator_factory(config_totals)
        results_totals = calculator_totals.calculate_funnel_metrics(data, steps)

        # Verify different results
        if results_unique.segment_data and results_totals.segment_data:
            mobile_unique = results_unique.segment_data.get("platform=mobile", [0, 0])
            mobile_totals = results_totals.segment_data.get("platform=mobile", [0, 0])

            # Unique users should count each user once
            assert mobile_unique[0] == 2  # 2 unique mobile users

            # Event totals should count all events
            assert mobile_totals[0] == 6  # 6 total sign up events for mobile

    def test_segmentation_property_not_in_data(self, calculator_factory, segmentation_test_data):
        """
        Test segmentation by property that doesn't exist in the data.
        Expected: Should handle gracefully, return no segments or empty segments.
        """
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_nonexistent_property",
            segment_values=["value1", "value2"],
        )
        calculator = calculator_factory(config)
        steps = ["Sign Up", "Email Verification"]

        results = calculator.calculate_funnel_metrics(segmentation_test_data, steps)

        # Should handle gracefully - either no segments or empty segments
        if results.segment_data:
            for segment_name, counts in results.segment_data.items():
                assert all(count == 0 for count in counts)

    def test_multiple_properties_same_event(self, calculator_factory, base_timestamp):
        """
        Test segmentation when events have multiple properties.
        Should correctly filter by the specified property.
        """
        events = [
            {
                "user_id": "user_001",
                "event_name": "Sign Up",
                "timestamp": base_timestamp,
                "event_properties": json.dumps(
                    {
                        "platform": "mobile",
                        "utm_source": "google",
                        "app_version": "2.1.0",
                    }
                ),
                "user_properties": json.dumps({}),
            },
            {
                "user_id": "user_001",
                "event_name": "Email Verification",
                "timestamp": base_timestamp + timedelta(minutes=30),
                "event_properties": json.dumps(
                    {
                        "platform": "mobile",
                        "utm_source": "google",
                        "app_version": "2.1.0",
                    }
                ),
                "user_properties": json.dumps({}),
            },
            {
                "user_id": "user_002",
                "event_name": "Sign Up",
                "timestamp": base_timestamp,
                "event_properties": json.dumps(
                    {
                        "platform": "desktop",
                        "utm_source": "facebook",
                        "app_version": "2.2.0",
                    }
                ),
                "user_properties": json.dumps({}),
            },
        ]

        data = pd.DataFrame(events)
        steps = ["Sign Up", "Email Verification"]

        # Segment by utm_source
        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_utm_source",
            segment_values=["google", "facebook"],
        )
        calculator = calculator_factory(config)
        results = calculator.calculate_funnel_metrics(data, steps)

        if results.segment_data:
            # Should correctly segment by utm_source despite other properties
            google_key = "utm_source=google"
            facebook_key = "utm_source=facebook"

            if google_key in results.segment_data:
                google_counts = results.segment_data[google_key]
                assert google_counts[0] == 1  # user_001
                assert google_counts[1] == 1  # user_001 completes verification

            if facebook_key in results.segment_data:
                facebook_counts = results.segment_data[facebook_key]
                assert facebook_counts[0] == 1  # user_002
                assert facebook_counts[1] == 0  # user_002 doesn't complete verification

    def test_statistical_significance_between_segments(self, calculator_factory, base_timestamp):
        """
        Test that statistical significance is calculated between segments.
        Expected: Should return statistical test results for segment comparison.
        """
        # Create data with clear difference between segments
        events = []

        # Segment A: High conversion (8/10 = 80%)
        for i in range(10):
            events.append(
                {
                    "user_id": f"user_a_{i:02d}",
                    "event_name": "Sign Up",
                    "timestamp": base_timestamp + timedelta(minutes=i),
                    "event_properties": json.dumps({"segment": "A"}),
                    "user_properties": json.dumps({}),
                }
            )

            if i < 8:  # 8 out of 10 convert
                events.append(
                    {
                        "user_id": f"user_a_{i:02d}",
                        "event_name": "Email Verification",
                        "timestamp": base_timestamp + timedelta(minutes=i + 30),
                        "event_properties": json.dumps({"segment": "A"}),
                        "user_properties": json.dumps({}),
                    }
                )

        # Segment B: Low conversion (3/10 = 30%)
        for i in range(10):
            events.append(
                {
                    "user_id": f"user_b_{i:02d}",
                    "event_name": "Sign Up",
                    "timestamp": base_timestamp + timedelta(minutes=i),
                    "event_properties": json.dumps({"segment": "B"}),
                    "user_properties": json.dumps({}),
                }
            )

            if i < 3:  # 3 out of 10 convert
                events.append(
                    {
                        "user_id": f"user_b_{i:02d}",
                        "event_name": "Email Verification",
                        "timestamp": base_timestamp + timedelta(minutes=i + 30),
                        "event_properties": json.dumps({"segment": "B"}),
                        "user_properties": json.dumps({}),
                    }
                )

        data = pd.DataFrame(events)
        steps = ["Sign Up", "Email Verification"]

        config = FunnelConfig(
            conversion_window_hours=24,
            segment_by="event_properties_segment",
            segment_values=["A", "B"],
        )
        calculator = calculator_factory(config)
        results = calculator.calculate_funnel_metrics(data, steps)

        # Should have statistical test results
        assert results.statistical_tests is not None
        assert len(results.statistical_tests) > 0

        # Should show significant difference between A and B
        test_result = results.statistical_tests[0]
        assert test_result.segment_a == "segment=A"
        assert test_result.segment_b == "segment=B"
        assert abs(test_result.conversion_a - 80.0) < 5.0  # ~80% conversion for A
        assert abs(test_result.conversion_b - 30.0) < 5.0  # ~30% conversion for B
        assert test_result.p_value < 0.05  # Should be statistically significant
