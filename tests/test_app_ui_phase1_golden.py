"""
Golden E2E Tests for Phase 1: Test-Driven State Separation

These tests serve as the "golden standard" that must pass before, during, and after
the Phase 1 refactoring. They validate the complete user journey and ensure that
our internal refactoring doesn't break the user-visible behavior.

Key Principles:
1. Test user workflows, not implementation details
2. Use stable UI keys for reliable test automation
3. Assert on both UI state and session state
4. Cover the complete "happy path" and key edge cases
"""

import time

from streamlit.testing.v1 import AppTest

APP_FILE = "app.py"
DEFAULT_TIMEOUT = 20  # Increased from 10 to handle slower module loading in Python 3.12


def wait_for_condition(at, condition_func, max_retries=10, sleep_time=0.5, timeout=15):
    """
    Robust waiting mechanism for test conditions.

    Args:
        at: AppTest instance
        condition_func: Function that returns True when condition is met
        max_retries: Maximum number of retries
        sleep_time: Time to sleep between retries
        timeout: Timeout for each at.run() call
    """
    for attempt in range(max_retries):
        try:
            if condition_func():
                return True
        except (AttributeError, KeyError, TypeError):
            # Condition not ready yet, continue waiting
            pass

        if attempt < max_retries - 1:
            time.sleep(sleep_time)
            at.run(timeout=timeout)

    return False


class TestPhase1GoldenStandard:
    """
    Golden tests that must pass throughout Phase 1 refactoring.
    These tests define the contract for user-visible behavior.
    """

    def test_full_successful_analysis_flow(self):
        """
        Golden Test: Complete "happy path" user journey.

        This test validates the full workflow from data loading to analysis results.
        It must pass before, during, and after Phase 1 refactoring.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Step 1: Load sample data with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data to be fully loaded and processed
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            max_retries=15,
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Events data should be loaded within timeout"

        # Verify data was loaded (check session state)
        assert at.session_state.events_data is not None, "Events data should be loaded"
        assert len(at.session_state.events_data) > 0, "Events data should not be empty"

        # Step 2: Build funnel by selecting events
        # Get available events from the loaded data
        available_events = sorted(at.session_state.events_data["event_name"].unique())

        # Select at least 3 events for a meaningful funnel
        selected_events = available_events[:3]

        for event in selected_events:
            checkbox_key = f"event_cb_{event.replace(' ', '_').replace('-', '_')}"
            at.checkbox(key=checkbox_key).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for funnel steps to be updated
        funnel_built = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 3, timeout=DEFAULT_TIMEOUT
        )
        assert funnel_built, "Funnel should be built within timeout"

        # Verify funnel steps were added to session state
        assert len(at.session_state.funnel_steps) == 3, "Should have 3 funnel steps"
        assert at.session_state.funnel_steps == selected_events, (
            "Steps should match selected events"
        )

        # Step 3: Run analysis with robust waiting
        at.button(key="analyze_funnel_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for analysis to complete
        analysis_complete = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "analysis_results")
                and at.session_state.analysis_results is not None
            ),
            max_retries=20,  # Analysis can take longer
            timeout=DEFAULT_TIMEOUT,
        )
        assert analysis_complete, "Analysis should complete within timeout"

        # Verify analysis results were generated
        assert at.session_state.analysis_results is not None, (
            "Analysis results should be generated"
        )
        assert hasattr(at.session_state.analysis_results, "steps"), "Results should have steps"
        assert len(at.session_state.analysis_results.steps) == 3, "Results should have 3 steps"

        # Verify no exceptions occurred
        assert not at.exception, "No exceptions should occur during analysis"

    def test_state_reset_on_clear_all(self):
        """
        Golden Test: Clear All functionality resets state properly.

        This test ensures that the Clear All button properly resets the funnel state
        and that the UI reflects this change correctly.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Load data and build funnel with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data loading
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Data should be loaded"

        available_events = sorted(at.session_state.events_data["event_name"].unique())
        first_event = available_events[0]
        checkbox_key = f"event_cb_{first_event.replace(' ', '_').replace('-', '_')}"

        at.checkbox(key=checkbox_key).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for step to be added
        step_added = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 1, timeout=DEFAULT_TIMEOUT
        )
        assert step_added, "Step should be added"

        # Verify step was added
        assert len(at.session_state.funnel_steps) == 1, "Should have 1 funnel step"
        assert first_event in at.session_state.funnel_steps, "Event should be in funnel steps"

        # Clear all steps
        at.button(key="clear_all_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for state to be cleared
        state_cleared = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 0, timeout=DEFAULT_TIMEOUT
        )
        assert state_cleared, "State should be cleared"

        # Verify state was cleared
        assert len(at.session_state.funnel_steps) == 0, "Funnel steps should be empty after clear"
        assert at.session_state.analysis_results is None, "Analysis results should be cleared"

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during clear"

    def test_event_selection_state_management(self):
        """
        Golden Test: Event selection properly manages session state.

        This test verifies that event selection/deselection properly updates
        the session state and that the UI reflects these changes.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Load data with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data loading
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Data should be loaded"

        available_events = sorted(at.session_state.events_data["event_name"].unique())
        test_events = available_events[:2]

        # Select first event
        checkbox_key_1 = f"event_cb_{test_events[0].replace(' ', '_').replace('-', '_')}"
        at.checkbox(key=checkbox_key_1).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for first event selection
        first_selected = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 1, timeout=DEFAULT_TIMEOUT
        )
        assert first_selected, "First event should be selected"

        assert len(at.session_state.funnel_steps) == 1, "Should have 1 step after first selection"
        assert test_events[0] in at.session_state.funnel_steps, "First event should be selected"

        # Select second event
        checkbox_key_2 = f"event_cb_{test_events[1].replace(' ', '_').replace('-', '_')}"
        at.checkbox(key=checkbox_key_2).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for second event selection
        second_selected = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 2, timeout=DEFAULT_TIMEOUT
        )
        assert second_selected, "Second event should be selected"

        assert len(at.session_state.funnel_steps) == 2, (
            "Should have 2 steps after second selection"
        )
        assert test_events[1] in at.session_state.funnel_steps, "Second event should be selected"

        # Deselect first event
        at.checkbox(key=checkbox_key_1).uncheck().run(timeout=DEFAULT_TIMEOUT)

        # Wait for first event deselection
        first_deselected = wait_for_condition(
            at,
            lambda: (
                len(at.session_state.funnel_steps) == 1
                and test_events[0] not in at.session_state.funnel_steps
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        assert first_deselected, "First event should be deselected"

        assert len(at.session_state.funnel_steps) == 1, "Should have 1 step after deselection"
        assert test_events[0] not in at.session_state.funnel_steps, (
            "First event should be deselected"
        )
        assert test_events[1] in at.session_state.funnel_steps, (
            "Second event should still be selected"
        )

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during event selection"

    def test_data_loading_initializes_session_state(self):
        """
        Golden Test: Data loading properly initializes all required session state.

        This test ensures that loading data sets up all the necessary session state
        variables that other components depend on.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Verify initial state
        assert at.session_state.events_data is None, "Events data should be None initially"
        assert len(at.session_state.funnel_steps) == 0, "Funnel steps should be empty initially"

        # Load data with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data loading to complete
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            max_retries=15,
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Events data should be loaded within timeout"

        # Verify data loading initialized required state
        assert at.session_state.events_data is not None, "Events data should be loaded"
        assert len(at.session_state.events_data) > 0, "Events data should not be empty"

        # Verify required columns exist
        required_columns = ["user_id", "event_name", "timestamp"]
        for col in required_columns:
            assert col in at.session_state.events_data.columns, f"Column {col} should exist"

        # Wait for event statistics to be generated
        stats_generated = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "event_statistics")
                and len(at.session_state.event_statistics) > 0
            ),
            max_retries=10,
            timeout=DEFAULT_TIMEOUT,
        )
        assert stats_generated, "Event statistics should be generated within timeout"

        # Verify event statistics were generated
        assert hasattr(at.session_state, "event_statistics"), (
            "Event statistics should be generated"
        )

        # Verify funnel config exists
        assert hasattr(at.session_state, "funnel_config"), "Funnel config should exist"

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during data loading"

    def test_analysis_requires_minimum_steps(self):
        """
        Golden Test: Analysis correctly handles insufficient funnel steps.

        This test ensures that the analysis function properly validates
        that at least 2 steps are required for funnel analysis.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Load data with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data loading
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Data should be loaded"

        # Note: analyze_funnel_button only appears when there are funnel steps
        # So we can't test clicking it with 0 steps - the button won't exist

        # Add one step
        available_events = sorted(at.session_state.events_data["event_name"].unique())
        first_event = available_events[0]
        checkbox_key = f"event_cb_{first_event.replace(' ', '_').replace('-', '_')}"
        at.checkbox(key=checkbox_key).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for step to be added
        step_added = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 1, timeout=DEFAULT_TIMEOUT
        )
        assert step_added, "Step should be added"

        # Now the analyze button should appear, try to analyze with 1 step
        at.button(key="analyze_funnel_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait a moment for any potential analysis attempt
        time.sleep(1)
        at.run(timeout=DEFAULT_TIMEOUT)

        # Should not generate analysis results with insufficient steps (< 2)
        assert at.session_state.analysis_results is None, "Should not generate results with 1 step"

        # Add second step
        second_event = available_events[1]
        checkbox_key_2 = f"event_cb_{second_event.replace(' ', '_').replace('-', '_')}"
        at.checkbox(key=checkbox_key_2).check().run(timeout=DEFAULT_TIMEOUT)

        # Wait for second step
        second_step_added = wait_for_condition(
            at, lambda: len(at.session_state.funnel_steps) == 2, timeout=DEFAULT_TIMEOUT
        )
        assert second_step_added, "Second step should be added"

        # Now analysis should work with 2+ steps
        at.button(key="analyze_funnel_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for analysis to complete
        analysis_complete = wait_for_condition(
            at,
            lambda: at.session_state.analysis_results is not None,
            max_retries=20,
            timeout=DEFAULT_TIMEOUT,
        )
        assert analysis_complete, "Analysis should complete with 2+ steps"

        # Should generate analysis results with 2+ steps
        assert at.session_state.analysis_results is not None, (
            "Should generate results with 2+ steps"
        )

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during validation"

    def test_session_state_persistence_across_interactions(self):
        """
        Golden Test: Session state persists correctly across multiple interactions.

        This test ensures that session state maintains consistency across
        multiple user interactions and UI updates.
        """
        at = AppTest.from_file(APP_FILE, default_timeout=DEFAULT_TIMEOUT).run()

        # Load data with robust waiting
        at.button(key="load_sample_data_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for data loading
        data_loaded = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "events_data")
                and at.session_state.events_data is not None
                and len(at.session_state.events_data) > 0
            ),
            timeout=DEFAULT_TIMEOUT,
        )
        assert data_loaded, "Data should be loaded"

        original_data_length = len(at.session_state.events_data)

        # Build funnel with robust waiting
        available_events = sorted(at.session_state.events_data["event_name"].unique())
        selected_events = available_events[:3]

        for i, event in enumerate(selected_events):
            checkbox_key = f"event_cb_{event.replace(' ', '_').replace('-', '_')}"
            at.checkbox(key=checkbox_key).check().run(timeout=DEFAULT_TIMEOUT)

            # Wait for each step to be added
            step_count = i + 1
            step_added = wait_for_condition(
                at,
                lambda: len(at.session_state.funnel_steps) == step_count,
                timeout=DEFAULT_TIMEOUT,
            )
            assert step_added, f"Step {step_count} should be added"

        # Run analysis with robust waiting
        at.button(key="analyze_funnel_button").click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for analysis to complete
        analysis_complete = wait_for_condition(
            at,
            lambda: at.session_state.analysis_results is not None,
            max_retries=20,
            timeout=DEFAULT_TIMEOUT,
        )
        assert analysis_complete, "Analysis should complete"

        # Verify all state is still consistent
        assert len(at.session_state.events_data) == original_data_length, "Data should persist"
        assert len(at.session_state.funnel_steps) == 3, "Funnel steps should persist"
        assert at.session_state.analysis_results is not None, "Analysis results should persist"

        # Add another step if available
        if len(available_events) > 3:
            fourth_event = available_events[3]
            checkbox_key_4 = f"event_cb_{fourth_event.replace(' ', '_').replace('-', '_')}"
            at.checkbox(key=checkbox_key_4).check().run(timeout=DEFAULT_TIMEOUT)

            # Wait for fourth step and results clearing
            fourth_step_added = wait_for_condition(
                at,
                lambda: (
                    len(at.session_state.funnel_steps) == 4
                    and at.session_state.analysis_results is None
                ),
                timeout=DEFAULT_TIMEOUT,
            )
            assert fourth_step_added, "Fourth step should be added and results cleared"

            # Verify analysis results were cleared (as expected when funnel changes)
            assert at.session_state.analysis_results is None, (
                "Results should clear when funnel changes"
            )
            assert len(at.session_state.funnel_steps) == 4, "Should have 4 steps now"

            # Data should still persist
            assert len(at.session_state.events_data) == original_data_length, (
                "Data should still persist"
            )

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during state persistence test"
