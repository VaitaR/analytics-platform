"""
Golden Test Suite for Phase 1 Refactoring - UI Tests

This test suite validates that the core user journey remains functional
before, during, and after Phase 1 refactoring. These tests must pass
at all times to ensure no regression in critical functionality.

Key Test Coverage:
- Complete user flow from data loading to analysis results
- Session state management and persistence
- Event selection and funnel building
- Analysis execution and results display
- Error handling and edge cases

Test Strategy:
- Uses robust waiting mechanisms for async operations
- Validates both UI state and session state
- Includes performance and timeout considerations
- Focuses on user-facing functionality over implementation details
"""

import time

from streamlit.testing.v1 import AppTest

# Test configuration
APP_FILE = "app.py"
DEFAULT_TIMEOUT = 30  # Increased timeout for complex operations


def wait_for_condition(at, condition_func, max_retries=10, sleep_time=0.5, timeout=15):
    """
    Wait for a condition to become true with timeout.

    Args:
        at: AppTest instance
        condition_func: Function that returns True when condition is met
        max_retries: Maximum number of retries
        sleep_time: Time to sleep between retries
        timeout: Maximum total time to wait

    Returns:
        bool: True if condition was met, False if timeout
    """
    start_time = time.time()
    retries = 0

    while retries < max_retries and (time.time() - start_time) < timeout:
        try:
            if condition_func():
                return True
        except (AttributeError, KeyError, IndexError):
            # Expected during state transitions
            pass

        time.sleep(sleep_time)
        retries += 1
        at.run(timeout=timeout)  # Refresh the app state

    return False


class TestPhase1GoldenStandard:
    """Golden standard tests that must pass throughout Phase 1 refactoring."""

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

        # Step 3: Run analysis using the new form submit button
        # In the new architecture, we need to find the form and submit it
        # Look for forms in the app
        forms = [element for element in at.main if hasattr(element, 'form_id')]
        assert len(forms) > 0, "Should have at least one form for funnel configuration"

        # Find the funnel configuration form
        funnel_form = None
        for form in forms:
            if form.form_id == "funnel_config_form":
                funnel_form = form
                break

        assert funnel_form is not None, "Should find funnel configuration form"

        # Submit the form by finding the submit button
        # The form submit button should be accessible through the form
        submit_buttons = [btn for btn in at.button if "Run Funnel Analysis" in btn.label]
        assert len(submit_buttons) > 0, "Should find the Run Funnel Analysis button"

        submit_buttons[0].click().run(timeout=DEFAULT_TIMEOUT)

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

        # Now try to analyze with 1 step using the new form submit approach
        # With only 1 step, the form should still be available but analysis should show warning
        submit_buttons = [btn for btn in at.button if "Run Funnel Analysis" in btn.label]
        if len(submit_buttons) > 0:
            submit_buttons[0].click().run(timeout=DEFAULT_TIMEOUT)

            # The analysis should not create results with only 1 step
            # Instead, it should show a warning toast
            time.sleep(1)  # Give time for toast to appear

            # Verify that analysis results are not created with insufficient steps
            assert (
                not hasattr(at.session_state, "analysis_results")
                or at.session_state.analysis_results is None
            ), "Analysis results should not be created with only 1 step"

        # Verify no exceptions occurred (warnings are OK)
        assert not at.exception, "No exceptions should occur"

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

        # Run analysis using the new form submit approach
        submit_buttons = [btn for btn in at.button if "Run Funnel Analysis" in btn.label]
        assert len(submit_buttons) > 0, "Should find the Run Funnel Analysis button"

        submit_buttons[0].click().run(timeout=DEFAULT_TIMEOUT)

        # Wait for analysis to complete
        analysis_complete = wait_for_condition(
            at,
            lambda: (
                hasattr(at.session_state, "analysis_results")
                and at.session_state.analysis_results is not None
            ),
            max_retries=20,
            timeout=DEFAULT_TIMEOUT,
        )
        assert analysis_complete, "Analysis should complete within timeout"

        # Verify session state persistence after analysis
        assert len(at.session_state.events_data) == original_data_length, (
            "Events data should remain unchanged after analysis"
        )
        assert len(at.session_state.funnel_steps) == 3, (
            "Funnel steps should remain unchanged after analysis"
        )
        assert at.session_state.funnel_steps == selected_events, (
            "Funnel steps should maintain order after analysis"
        )
        assert at.session_state.analysis_results is not None, (
            "Analysis results should be persisted"
        )

        # Verify no exceptions
        assert not at.exception, "No exceptions should occur during state persistence test"
