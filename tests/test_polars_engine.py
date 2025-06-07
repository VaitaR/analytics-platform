import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from app import FunnelCalculator, FunnelConfig, TimeToConvertStats, PathAnalysisData, FunnelResults

@pytest.fixture
def sample_events_df():
    """Return a sample DataFrame for testing."""
    data = {
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'event_name': ['A', 'B', 'C', 'A', 'B', 'A', 'B', 'D', 'C'],
        'timestamp': pd.to_datetime([
            '2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00',
            '2023-01-01 10:01:00', '2023-01-01 10:06:00',
            '2023-01-01 10:02:00', '2023-01-01 10:03:00', '2023-01-01 10:04:00', '2023-01-01 10:12:00'
        ])
    }
    return pd.DataFrame(data)

@patch('app.FunnelCalculator._calculate_funnel_metrics_pandas')
def test_polars_engine_no_pandas_fallback(mock_pandas_calculator, sample_events_df):
    """
    Tests that when use_polars is True, the polars calculation engine is used
    and it does not fall back to the pandas implementation.
    """
    config = FunnelConfig()
    funnel_steps = ['A', 'B', 'C']

    # We need to mock the polars implementation to return a valid FunnelResults object
    # to avoid it failing and legitimately calling the pandas fallback.
    # The goal is to check the dispatch logic, not the implementation of polars itself.
    mock_polars_result = FunnelResults(
        steps=funnel_steps,
        users_count=[3, 2, 1],
        conversion_rates=[100.0, 66.7, 50.0],
        drop_offs=[1, 1, 1],
        drop_off_rates=[33.3, 50.0, 0.0]
    )

    with patch('app.FunnelCalculator._calculate_funnel_metrics_polars', return_value=mock_polars_result) as mock_polars_method:
        funnel_calculator = FunnelCalculator(config=config, use_polars=True)
        funnel_calculator.calculate_funnel_metrics(sample_events_df, funnel_steps)

        mock_polars_method.assert_called_once()
        mock_pandas_calculator.assert_not_called()

@patch('app.FunnelCalculator._calculate_funnel_metrics_polars')
def test_polars_engine_fallback_on_error(mock_polars_calculator, sample_events_df):
    """
    Tests that if the polars calculation engine raises an exception,
    it correctly falls back to the pandas implementation.
    """
    config = FunnelConfig()
    funnel_steps = ['A', 'B', 'C']

    # Simulate an error in the polars implementation
    mock_polars_calculator.side_effect = Exception("Polars engine failure")

    # The pandas implementation will be called, so we need to mock its return value
    mock_pandas_result = FunnelResults(
        steps=funnel_steps,
        users_count=[3, 2, 1],
        conversion_rates=[100.0, 66.7, 50.0],
        drop_offs=[1, 1, 1],
        drop_off_rates=[33.3, 50.0, 0.0]
    )

    with patch('app.FunnelCalculator._calculate_funnel_metrics_pandas', return_value=mock_pandas_result) as mock_pandas_method:
        funnel_calculator = FunnelCalculator(config=config, use_polars=True)
        funnel_calculator.calculate_funnel_metrics(sample_events_df, funnel_steps)

        mock_polars_calculator.assert_called_once()
        mock_pandas_method.assert_called_once() 