# Weather Forecast Evaluation Framework

A Python framework for evaluating temperature forecasts at the station level using METAR observations and numerical weather prediction models.

## Motivation

Predicting high temperatures is harder than it seems. NWS stations (like the Central Park KNYC sensor) are continuous instruments that often read 2-5°F warmer than nearby airport METAR measurements due to sensor placement, time-of-observation effects, and urban heat. Meanwhile, global weather models (ICON, GFS, ECMWF) produce 2m temperature forecasts, but these don't directly match either NWS station observations or airport METAR max values.

This creates a feature engineering challenge: **given hourly METAR observations and model forecasts, how accurately can we predict categorical temperature outcomes?**

This framework tackles that problem using walk-forward XGBoost classification with carefully engineered features from raw observations.

## Approach

1. **Data Collection**: Pulls hourly METAR observations from Iowa State University's archive and numerical forecasts from Open-Meteo
2. **Feature Engineering**: Builds a feature matrix from raw observations:
   - Running daily maximum temperature
   - Hourly temperature delta and trends
   - Diurnal cycle progress
   - Multi-station airport spreads (comparing nearby airports to primary station)
   - Calendar and seasonal features
3. **Walk-Forward Backtesting**: Proper temporal validation (no future peeking) using expanding training windows
4. **Classification**: Predicts categorical temperature outcomes (e.g., which bracket will today's high fall into?)
5. **Evaluation**: Accuracy by class, calibration curves, feature importance

## Key Results

Example: predicting New York City (KNYC) afternoon temperature bracket at 1 PM ET

- **Accuracy**: 60-75% depending on season
- **Best performing features**: Running daily max, airport spread, time-of-day, month
- **Worst accuracy month**: January (winter is harder)
- **Calibration**: Model confidence correlates well with actual accuracy (r > 0.7)

## Quick Start

### Installation

```bash
git clone https://github.com/Matthew-Carlino/weather-forecast-eval.git
cd weather-forecast-eval
pip install -r requirements.txt
```

### Run a Backtest

```bash
python scripts/run_backtest.py --station NYC --start 2025-01-01 --end 2025-12-31 --eval-hour 13
```

This will:
- Fetch hourly METAR observations for NYC from Iowa State
- Fetch model forecasts from Open-Meteo
- Engineer features at 1 PM ET each day
- Run walk-forward XGBoost classification
- Print accuracy tables and save plots to `outputs/`

### Use in Code

```python
from src.data_fetcher import METARFetcher, ForecastFetcher
from src.feature_engine import build_features
from src.model import OMOClassifier
import pandas as pd

# Fetch data
metar_fetcher = METARFetcher(station="NYC", cache_dir="data/")
obs_df = metar_fetcher.fetch(start_date="2025-01-01", end_date="2025-12-31")

forecast_fetcher = ForecastFetcher(latitude=40.78, longitude=-73.97)
forecasts = forecast_fetcher.fetch_icon(start_date="2025-01-01", end_date="2025-12-31")

# Build features
features = build_features(
    obs_df=obs_df,
    stations=["NYC", "EWR", "LGA"],  # Primary + nearby airports
    eval_hour=13
)

# Train and evaluate
clf = OMOClassifier()
results = clf.walk_forward_backtest(features, min_train=100, retrain_every=10)
print(f"Accuracy: {results['accuracy']:.1%}")
```

### Jupyter Walkthrough

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

## Project Structure

```
weather-forecast-eval/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py           # METAR and forecast data collection
│   ├── feature_engine.py         # Feature engineering from raw observations
│   ├── model.py                  # Walk-forward XGBoost classifier
│   └── visualization.py          # Plotting and results rendering
├── scripts/
│   └── run_backtest.py          # CLI entry point for backtests
├── tests/
│   └── test_features.py         # Unit tests for feature engineering
├── notebooks/
│   └── walkthrough.ipynb        # Interactive demo notebook
└── data/                         # Local cache for METAR and forecasts
```

## Data Sources

- **METAR Observations**: Iowa State University [Automated Surface Observing System (ASOS)](https://mesonet.agron.iastate.edu/request/download.phtml?ts=202501010000&te=202501020000)
- **Temperature Forecasts**: [Open-Meteo](https://open-meteo.com/en/docs) — free, no API key required
  - ICON (DWD)
  - GFS (NOAA)
  - ECMWF (European Centre)

## Design Principles

- **Pure Functions**: Feature engineering functions are side-effect-free and testable
- **Type Hints**: All public functions include type annotations for clarity
- **Proper Validation**: Temporal train/test separation with no future peeking
- **Reproducibility**: Deterministic seeds, explicit feature definitions
- **Error Handling**: Graceful fallback for missing data, logging of issues

## Limitations

- Model forecasts (ICON/GFS/ECMWF) predict generic 2m temperature, not station-specific observations
- Station closures and maintenance create gaps in historical records
- Seasonal accuracy varies widely (winter < summer)
- Accuracy degrades with longer lead times (D-1 forecasts less accurate than D-0)

## Contributing

This is a portfolio project. Feel free to fork and extend it for your own analysis.

## License

MIT

---

**Built by**: Matthew Carlino
**Contact**: Available upon request
**Last updated**: March 2026
