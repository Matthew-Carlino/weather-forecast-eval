"""Feature engineering from raw METAR observations."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# Expected diurnal range by month (in Fahrenheit)
# Conservative mid-range values for typical US locations
EXPECTED_DIURNAL_RANGE_BY_MONTH = {
    1: 10.0,   # January
    2: 11.0,   # February
    3: 12.0,   # March
    4: 13.5,   # April
    5: 15.0,   # May
    6: 15.5,   # June
    7: 15.0,   # July
    8: 14.5,   # August
    9: 14.0,   # September
    10: 12.5,  # October
    11: 11.0,  # November
    12: 10.0,  # December
}


def compute_running_daily_max(obs_series: pd.Series) -> pd.Series:
    """Compute running daily maximum temperature.

    For each observation time, computes the maximum temperature observed
    so far during that calendar day (midnight to midnight UTC).

    Args:
        obs_series: Pandas Series with datetime index and temperature values (°F)

    Returns:
        Series with same index, containing running daily max at each time
    """
    running_max = obs_series.groupby(obs_series.index.date).cummax()
    return running_max


def compute_hourly_delta(obs_series: pd.Series, hours: int = 1) -> pd.Series:
    """Compute temperature change over specified hour window.

    Args:
        obs_series: Pandas Series with datetime index and temperature values (°F)
        hours: Number of hours to look back (default 1)

    Returns:
        Series with temperature delta (current - past)
    """
    return obs_series - obs_series.shift(hours)


def compute_trend(obs_series: pd.Series, window: int = 3) -> pd.Series:
    """Compute linear trend over rolling window.

    Uses least-squares regression on last N observations to estimate
    temperature change rate (°F per hour).

    Args:
        obs_series: Pandas Series with datetime index and temperature values (°F)
        window: Number of observations to use for trend (default 3 = 3 hours)

    Returns:
        Series with slope of linear regression (°F/hour)
    """
    trends = []
    for i in range(len(obs_series)):
        if i < window - 1:
            trends.append(np.nan)
        else:
            subset = obs_series.iloc[i - window + 1 : i + 1].values
            x = np.arange(len(subset))
            try:
                slope, _, _, _, _ = linregress(x, subset)
                trends.append(slope)
            except Exception:
                trends.append(np.nan)

    return pd.Series(trends, index=obs_series.index)


def compute_diurnal_progress(
    running_max: pd.Series,
    obs_series: pd.Series,
    expected_diurnal: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute progress through expected daily temperature range.

    Ratio of temperature gained so far to expected full diurnal range.
    Values > 1 indicate we've exceeded typical daily range (afternoon heating ongoing).

    Args:
        running_max: Running daily maximum temperature (°F)
        obs_series: Current temperature observations (°F)
        expected_diurnal: Expected full diurnal range (°F). If None, uses typical
                         by-month values.

    Returns:
        Series with diurnal_progress values (dimensionless ratio)
    """
    if expected_diurnal is None:
        months = obs_series.index.month
        expected_diurnal = pd.Series(
            [EXPECTED_DIURNAL_RANGE_BY_MONTH[m] for m in months],
            index=obs_series.index,
        )

    # Avoid division by zero
    expected_diurnal = expected_diurnal.replace(0, np.nan)

    # Progress = (running_max - low_temp) / expected_range
    # Use min of obs so far as proxy for low
    daily_min = obs_series.groupby(obs_series.index.date).cummin()
    temp_gained = running_max - daily_min

    progress = temp_gained / expected_diurnal

    return progress.fillna(0.0)


def compute_airport_spread(
    primary_obs: pd.DataFrame,
    airport_obs: List[pd.DataFrame],
    eval_hour: int = 13,
) -> pd.Series:
    """Compute spread between primary station and nearby airports.

    Measures the temperature difference between the primary observation
    station (e.g., KNYC) and the maximum of nearby airports at a given hour.
    Useful for detecting local heating or cold effects.

    Args:
        primary_obs: DataFrame with datetime index and 'tmpf' column for primary station
        airport_obs: List of DataFrames for nearby airports, each with 'tmpf' column
        eval_hour: Hour of day to evaluate spread (UTC)

    Returns:
        Series with airport_spread values (primary_max - airport_max in °F)
    """
    if not airport_obs:
        return pd.Series(0.0, index=primary_obs.index)

    # Filter to eval hour for each airport
    airport_hourly_maxes = []
    for airport_df in airport_obs:
        hourly = airport_df[airport_df.index.hour == eval_hour]
        if len(hourly) > 0:
            daily_max = hourly.groupby(hourly.index.date)["tmpf"].max()
            airport_hourly_maxes.append(daily_max)

    if not airport_hourly_maxes:
        return pd.Series(0.0, index=primary_obs.index)

    # Concatenate all airports and take max per day
    airport_max = pd.concat(airport_hourly_maxes, axis=1).max(axis=1)

    # Primary max at eval hour
    primary_hourly = primary_obs[primary_obs.index.hour == eval_hour]
    primary_max = primary_hourly.groupby(primary_hourly.index.date)["tmpf"].max()

    # Align to original index by date
    spread = pd.Series(index=primary_obs.index, dtype=float)
    for date in spread.index.date:
        if date in primary_max.index and date in airport_max.index:
            spread[spread.index.date == date] = (
                primary_max[date] - airport_max[date]
            )
        else:
            spread[spread.index.date == date] = 0.0

    return spread.fillna(0.0)


def build_features(
    obs_df: pd.DataFrame,
    eval_hour: int = 13,
    primary_station: str = "primary",
    airport_stations: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build feature matrix from raw METAR observations.

    Constructs daily feature vectors at a specified evaluation hour (default 1 PM UTC).
    Each row corresponds to one calendar day.

    Args:
        obs_df: METAR observation DataFrame with:
            - index: datetime (UTC)
            - 'tmpf': temperature in Fahrenheit
            - 'station': station identifier (optional)
        eval_hour: Hour of day (UTC) to evaluate features
        primary_station: Name of primary station column value (unused if single station)
        airport_stations: List of airport station names for spread calculation

    Returns:
        DataFrame with one row per day containing:
            - date: Calendar date
            - running_max: Max temperature observed so far today
            - current_temp: Temperature at eval_hour
            - airport_max: Max of nearby airports at eval_hour
            - airport_spread: primary_max - airport_max
            - delta_1h: Temperature change over last hour
            - delta_3h: Temperature change over last 3 hours
            - trend_3h: Linear trend slope over last 3 hours (°F/hour)
            - diurnal_progress: Progress through expected daily range
            - month: Calendar month (1-12)
            - is_dst: Whether daylight saving time is active
    """
    # Ensure datetime index is UTC-aware
    if obs_df.index.tz is None:
        obs_df.index = obs_df.index.tz_localize("UTC")

    if "tmpf" not in obs_df.columns:
        raise ValueError("obs_df must contain 'tmpf' column")

    # Extract temperature series
    temps = obs_df["tmpf"]

    # Compute base features
    running_max = compute_running_daily_max(temps)
    delta_1h = compute_hourly_delta(temps, hours=1)
    delta_3h = compute_hourly_delta(temps, hours=3)
    trend_3h = compute_trend(temps, window=3)
    diurnal_progress = compute_diurnal_progress(running_max, temps)

    # Filter to eval hour
    eval_mask = obs_df.index.hour == eval_hour
    eval_df = obs_df[eval_mask].copy()
    eval_df["running_max"] = running_max[eval_mask]
    eval_df["delta_1h"] = delta_1h[eval_mask]
    eval_df["delta_3h"] = delta_3h[eval_mask]
    eval_df["trend_3h"] = trend_3h[eval_mask]
    eval_df["diurnal_progress"] = diurnal_progress[eval_mask]

    # Build result DataFrame
    result = pd.DataFrame()
    result["date"] = eval_df.index.date
    result["current_temp"] = eval_df["tmpf"].values
    result["running_max"] = eval_df["running_max"].values
    result["delta_1h"] = eval_df["delta_1h"].values
    result["delta_3h"] = eval_df["delta_3h"].values
    result["trend_3h"] = eval_df["trend_3h"].values
    result["diurnal_progress"] = eval_df["diurnal_progress"].values

    # Airport spread (if multiple stations provided)
    if "station" in obs_df.columns and airport_stations:
        # Split observations by station
        try:
            primary_obs = obs_df[obs_df["station"] == primary_station]
            airport_dfs = [
                obs_df[obs_df["station"] == station].copy()
                for station in airport_stations
            ]
            spread = compute_airport_spread(primary_obs, airport_dfs, eval_hour)
            result["airport_spread"] = spread.values
        except Exception as e:
            logger.warning(f"Could not compute airport spread: {e}")
            result["airport_spread"] = 0.0
    else:
        result["airport_spread"] = 0.0

    # Calendar features
    result["month"] = eval_df.index.month.values
    result["is_dst"] = eval_df.index.dayofweek < 5  # Placeholder; proper DST would use pytz

    # Fill NaN values
    result = result.fillna(method="bfill").fillna(method="ffill").fillna(0)

    logger.info(f"Built feature matrix with {len(result)} days")

    return result


def get_feature_names() -> List[str]:
    """Get list of feature column names in standard order.

    Returns:
        List of feature column names
    """
    return [
        "running_max",
        "current_temp",
        "airport_spread",
        "delta_1h",
        "delta_3h",
        "trend_3h",
        "diurnal_progress",
        "month",
        "is_dst",
    ]
