"""Data fetching and caching for METAR observations and weather forecasts."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Station code mappings: station name -> IEM code (NOT ICAO)
IEM_STATION_MAP = {
    "NYC": "NYC",      # Central Park (hourly only)
    "EWR": "EWR",      # Newark
    "LGA": "LGA",      # LaGuardia
    "JFK": "JFK",      # JFK
    "CHI": "ORD",      # Chicago O'Hare
    "MIA": "MIA",      # Miami
    "LAX": "LAX",      # Los Angeles
    "DEN": "DEN",      # Denver
}

OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


class METARFetcher:
    """Fetches hourly METAR observations from Iowa State University archive.

    The Iowa State archive provides free access to Automated Surface Observing
    System (ASOS) records going back several decades. This class handles data
    retrieval, caching, and basic validation.

    Attributes:
        station: IEM station code (e.g., "NYC", "EWR")
        cache_dir: Directory for caching CSV files
        base_url: Iowa State IEM API base URL
    """

    def __init__(
        self,
        station: str,
        cache_dir: str = "data/metar_cache",
    ):
        """Initialize METAR fetcher.

        Args:
            station: Station identifier (NYC, EWR, LGA, JFK, ORD, MIA, LAX, DEN)
            cache_dir: Local directory for caching downloaded CSVs

        Raises:
            ValueError: If station not recognized
        """
        if station not in IEM_STATION_MAP:
            raise ValueError(
                f"Unknown station: {station}. "
                f"Supported: {list(IEM_STATION_MAP.keys())}"
            )

        self.station = station
        self.iem_code = IEM_STATION_MAP[station]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

    def _get_cache_path(self, start_date: str, end_date: str) -> Path:
        """Get cache file path for a date range.

        Args:
            start_date: YYYY-MM-DD format
            end_date: YYYY-MM-DD format

        Returns:
            Path to cache CSV file
        """
        return self.cache_dir / f"{self.station}_{start_date}_{end_date}.csv"

    def fetch(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch hourly METAR data for station and date range.

        Attempts to load from local cache first. If not cached or
        force_refresh=True, downloads from Iowa State IEM.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with columns:
                - valid: UTC timestamp (datetime)
                - tmpf: Temperature in Fahrenheit (float)
                - dwpf: Dew point (float)
                - drct: Wind direction (int)
                - sknt: Wind speed in knots (int)
                - p01i: Precipitation (float)
                - alti: Altimeter (float)

        Raises:
            requests.RequestException: If API request fails
            ValueError: If response is empty
        """
        cache_path = self._get_cache_path(start_date, end_date)

        # Try to load from cache
        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading {self.station} from cache: {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["valid"])
            return df

        # Fetch from Iowa State
        logger.info(
            f"Fetching {self.station} observations from {start_date} to {end_date}"
        )

        params = {
            "station": self.iem_code,
            "data": "tmpf,dwpf,drct,sknt,p01i,alti",
            "tz": "UTC",
            "format": "csv",
            "ts": f"{start_date}T00:00",
            "te": f"{end_date}T23:59",
        }

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        lines = response.text.strip().split("\n")
        if len(lines) < 2:
            raise ValueError(
                f"Empty response from IEM for {self.station} "
                f"({start_date} to {end_date})"
            )

        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

        if "valid" not in df.columns:
            raise ValueError("Unexpected response format from IEM")

        df["valid"] = pd.to_datetime(df["valid"], utc=True)

        # Drop columns with all NaN
        df = df.dropna(how="all", axis=1)

        logger.info(f"Downloaded {len(df)} observations for {self.station}")

        # Cache result
        df.to_csv(cache_path, index=False)
        logger.debug(f"Cached to {cache_path}")

        return df


class ForecastFetcher:
    """Fetches temperature forecasts from Open-Meteo API.

    Open-Meteo provides free access to multiple weather models (ICON, GFS, ECMWF)
    with historical data available. This class retrieves daily max temperatures
    for a given location.

    Attributes:
        latitude: Location latitude
        longitude: Location longitude
        cache_dir: Directory for caching responses
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        cache_dir: str = "data/forecast_cache",
    ):
        """Initialize forecast fetcher.

        Args:
            latitude: Location latitude (e.g., 40.78 for NYC)
            longitude: Location longitude (e.g., -73.97 for NYC)
            cache_dir: Local directory for caching responses
        """
        self.latitude = latitude
        self.longitude = longitude
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, model: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for a model and date range.

        Args:
            model: Model name (icon, gfs, ecmwf)
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD

        Returns:
            Path to cache JSON file
        """
        return (
            self.cache_dir /
            f"{model}_{self.latitude}_{self.longitude}_{start_date}_{end_date}.json"
        )

    def _fetch_model(
        self,
        model: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch temperature forecast from Open-Meteo for a specific model.

        Args:
            model: "icon", "gfs", or "ecmwf"
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with columns:
                - date: Date (YYYY-MM-DD string)
                - temperature_2m_max: Daily max temperature in °C
                - model: Model name

        Raises:
            requests.RequestException: If API request fails
        """
        cache_path = self._get_cache_path(model, start_date, end_date)

        # Try cache first
        if cache_path.exists() and not force_refresh:
            logger.debug(f"Loading {model} forecast from cache")
            df = pd.read_json(cache_path)
            return df

        logger.info(f"Fetching {model} forecast for {start_date} to {end_date}")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "models": model,
        }

        response = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "daily" not in data or "time" not in data["daily"]:
            raise ValueError(f"Unexpected response format from Open-Meteo for {model}")

        df = pd.DataFrame({
            "date": data["daily"]["time"],
            "temperature_2m_max": data["daily"]["temperature_2m_max"],
            "model": model,
        })

        # Cache result
        df.to_json(cache_path)
        logger.debug(f"Cached {model} to {cache_path}")

        return df

    def fetch_icon(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch ICON model forecast (DWD).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with date and temperature_2m_max in °C
        """
        return self._fetch_model("icon", start_date, end_date, force_refresh)

    def fetch_gfs(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch GFS model forecast (NOAA).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with date and temperature_2m_max in °C
        """
        return self._fetch_model("gfs", start_date, end_date, force_refresh)

    def fetch_ecmwf(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch ECMWF model forecast (European Centre).

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, re-download even if cached

        Returns:
            DataFrame with date and temperature_2m_max in °C
        """
        return self._fetch_model("ecmwf", start_date, end_date, force_refresh)
