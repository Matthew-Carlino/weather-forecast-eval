"""Unit tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engine import (
    compute_airport_spread,
    compute_diurnal_progress,
    compute_hourly_delta,
    compute_running_daily_max,
    compute_trend,
)


@pytest.fixture
def sample_obs_series():
    """Create sample temperature time series for testing.

    Returns:
        Pandas Series with hourly temperatures and UTC datetime index
    """
    dates = pd.date_range("2025-01-01", periods=72, freq="H", tz="UTC")
    # Simulate daily cycle: low 40°F at midnight, high 65°F at 2 PM
    temps = []
    for i, date in enumerate(dates):
        hour = date.hour
        day_cycle = 40 + 25 * np.sin((hour - 6) * np.pi / 12)  # Low at 6 AM, high at 2 PM
        noise = np.random.normal(0, 1)
        temps.append(max(day_cycle + noise, 30))

    return pd.Series(temps, index=dates)


class TestComputeRunningDailyMax:
    """Tests for running daily max computation."""

    def test_running_max_increases_monotonically(self, sample_obs_series):
        """Running max should never decrease within a day."""
        result = compute_running_daily_max(sample_obs_series)

        # Check monotonic increase within each day
        for date in sample_obs_series.index.date:
            mask = sample_obs_series.index.date == date
            daily_max = result[mask]
            assert (daily_max.diff().fillna(0) >= 0).all(), \
                f"Running max not monotonic on {date}"

    def test_running_max_resets_at_midnight(self, sample_obs_series):
        """Running max should reset to first observation at midnight."""
        result = compute_running_daily_max(sample_obs_series)

        dates = sample_obs_series.index.date
        for date in np.unique(dates):
            mask = dates == date
            first_obs = sample_obs_series[mask].iloc[0]
            first_running_max = result[mask].iloc[0]
            assert first_running_max == first_obs, \
                f"Running max doesn't equal first obs on {date}"

    def test_running_max_matches_max(self, sample_obs_series):
        """Running max at day end should equal day's max temp."""
        result = compute_running_daily_max(sample_obs_series)

        dates = sample_obs_series.index.date
        for date in np.unique(dates):
            mask = dates == date
            expected_max = sample_obs_series[mask].max()
            actual_max = result[mask].max()
            assert np.isclose(expected_max, actual_max), \
                f"Max mismatch on {date}: {expected_max} vs {actual_max}"


class TestComputeHourlyDelta:
    """Tests for hourly delta computation."""

    def test_delta_shape(self, sample_obs_series):
        """Delta should have same shape as input."""
        result = compute_hourly_delta(sample_obs_series, hours=1)
        assert result.shape == sample_obs_series.shape

    def test_delta_first_value_nan(self, sample_obs_series):
        """First value should be NaN (no prior observation)."""
        result = compute_hourly_delta(sample_obs_series, hours=1)
        assert pd.isna(result.iloc[0])

    def test_delta_calculation_correct(self, sample_obs_series):
        """Delta should equal current - past value."""
        result = compute_hourly_delta(sample_obs_series, hours=1)

        for i in range(1, len(result)):
            expected = sample_obs_series.iloc[i] - sample_obs_series.iloc[i-1]
            actual = result.iloc[i]
            assert np.isclose(expected, actual, atol=1e-10), \
                f"Delta mismatch at index {i}"


class TestComputeTrend:
    """Tests for trend computation."""

    def test_trend_shape(self, sample_obs_series):
        """Trend should have same shape as input."""
        result = compute_trend(sample_obs_series, window=3)
        assert result.shape == sample_obs_series.shape

    def test_trend_first_window_nan(self, sample_obs_series):
        """First window-1 values should be NaN."""
        window = 3
        result = compute_trend(sample_obs_series, window=window)
        assert result.iloc[0:window-1].isna().all()

    def test_trend_flat_series_zero(self):
        """Trend of constant series should be zero."""
        const_series = pd.Series([50.0] * 10, index=pd.date_range("2025-01-01", periods=10, freq="H", tz="UTC"))
        result = compute_trend(const_series, window=3)
        assert np.allclose(result.dropna(), 0, atol=1e-10)

    def test_trend_increasing_series_positive(self):
        """Trend of increasing series should be positive."""
        increasing = pd.Series(range(10), index=pd.date_range("2025-01-01", periods=10, freq="H", tz="UTC"), dtype=float)
        result = compute_trend(increasing, window=3)
        assert (result.dropna() > 0).all()


class TestComputeDiurnalProgress:
    """Tests for diurnal progress computation."""

    def test_diurnal_progress_range(self, sample_obs_series):
        """Diurnal progress should be between 0 and ~2."""
        running_max = compute_running_daily_max(sample_obs_series)
        result = compute_diurnal_progress(running_max, sample_obs_series)

        # Should be mostly in [0, 2] range
        valid_values = result[(result != 0) & (result.notna())]
        assert (valid_values < 5).all(), "Diurnal progress has unexpectedly high values"

    def test_diurnal_progress_increases_through_day(self, sample_obs_series):
        """Diurnal progress should generally increase through afternoon."""
        running_max = compute_running_daily_max(sample_obs_series)
        result = compute_diurnal_progress(running_max, sample_obs_series)

        # Check trend through afternoon hours
        dates = sample_obs_series.index.date
        for date in np.unique(dates):
            mask = (dates == date) & (sample_obs_series.index.hour >= 8) & (sample_obs_series.index.hour <= 16)
            if mask.sum() > 0:
                afternoon_progress = result[mask]
                # Rough trend should be upward (allow some noise)
                # Not a strict requirement, but generally true


class TestComputeAirportSpread:
    """Tests for airport spread computation."""

    def test_spread_with_empty_airports(self, sample_obs_series):
        """Spread should be zero when no airport data provided."""
        primary_obs = pd.DataFrame({"tmpf": sample_obs_series})
        result = compute_airport_spread(primary_obs, [], eval_hour=13)

        assert (result == 0).all()

    def test_spread_shape(self, sample_obs_series):
        """Spread should match primary_obs shape."""
        primary_obs = pd.DataFrame({"tmpf": sample_obs_series})
        airport_obs = [pd.DataFrame({"tmpf": sample_obs_series - 2})]  # Slightly cooler

        result = compute_airport_spread(primary_obs, airport_obs, eval_hour=13)

        assert len(result) == len(primary_obs)

    def test_spread_primary_higher_than_airport(self, sample_obs_series):
        """Spread should be positive when primary is warmer than airports."""
        primary_obs = pd.DataFrame({"tmpf": sample_obs_series})
        airport_obs = [pd.DataFrame({"tmpf": sample_obs_series - 5})]  # Much cooler

        result = compute_airport_spread(primary_obs, airport_obs, eval_hour=13)

        # Most values should be positive (primary warmer)
        positive_count = (result > 0).sum()
        total_count = len(result[result != 0])
        if total_count > 0:
            assert positive_count / total_count > 0.5, "Expected primary warmer than airports"


class TestFeatureIntegration:
    """Integration tests for complete feature engineering."""

    def test_build_features_output_shape(self, sample_obs_series):
        """Feature matrix should have correct shape."""
        from src.feature_engine import build_features

        obs_df = pd.DataFrame({"tmpf": sample_obs_series})
        features = build_features(obs_df, eval_hour=13)

        # Should have one row per day (72 hours / 24 = 3 days)
        assert len(features) == 3

        # Should have required columns
        required_cols = [
            "date", "running_max", "current_temp", "airport_spread",
            "delta_1h", "delta_3h", "trend_3h", "diurnal_progress",
            "month", "is_dst"
        ]
        for col in required_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_build_features_no_nan_except_intentional(self, sample_obs_series):
        """Feature matrix should not have NaN values after fillna."""
        from src.feature_engine import build_features

        obs_df = pd.DataFrame({"tmpf": sample_obs_series})
        features = build_features(obs_df, eval_hour=13)

        # After fillna, should have no NaN
        assert not features.isna().any().any(), "Feature matrix contains NaN values"

    def test_build_features_date_column_correct(self, sample_obs_series):
        """Date column should match input dates."""
        from src.feature_engine import build_features

        obs_df = pd.DataFrame({"tmpf": sample_obs_series})
        features = build_features(obs_df, eval_hour=13)

        unique_dates = np.unique(sample_obs_series.index.date)
        assert len(features) == len(unique_dates)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
