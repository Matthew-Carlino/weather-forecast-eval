# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/weather-forecast-eval.git
cd weather-forecast-eval

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run Your First Backtest

```bash
# Basic usage (NYC, 1 month, 1 PM eval)
python scripts/run_backtest.py

# Custom station and date range
python scripts/run_backtest.py --station EWR --start 2025-06-01 --end 2025-08-31 --eval-hour 14

# Force re-download (don't use cache)
python scripts/run_backtest.py --force-refresh
```

The script will:
1. Download METAR data from Iowa State (or load from cache)
2. Engineer features at your specified hour
3. Run walk-forward XGBoost backtest
4. Print accuracy metrics
5. Save plots to `outputs/`

## Use in Your Code

```python
import pandas as pd
from src.data_fetcher import METARFetcher
from src.feature_engine import build_features, get_feature_names
from src.model import OMOClassifier

# 1. Fetch data
fetcher = METARFetcher(station="NYC")
obs_df = fetcher.fetch("2025-01-01", "2025-12-31")
obs_df.index = pd.to_datetime(obs_df["valid"], utc=True)

# 2. Build features
features = build_features(obs_df, eval_hour=13)

# 3. Create and train model
clf = OMOClassifier()

# For demo, create synthetic targets
targets = (features["running_max"] > 65).astype(int)

# 4. Run walk-forward backtest
results = clf.walk_forward_backtest(
    features[["date"] + get_feature_names()],
    targets,
    min_train=50,
    retrain_every=5
)

print(f"Accuracy: {results['accuracy']:.1%}")

# 5. Get feature importance
importance = clf.feature_importance()
print(importance.head(10))
```

## Interactive Notebook

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

The notebook provides step-by-step walkthrough with visualizations.

## Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_features.py::TestComputeRunningDailyMax -v

# With coverage
pytest tests/ --cov=src
```

## Project Structure

```
weather-forecast-eval/
├── README.md              # Project overview
├── QUICKSTART.md          # This file
├── ARCHITECTURE.md        # Design decisions
├── requirements.txt       # Dependencies
├── .gitignore
│
├── src/
│   ├── data_fetcher.py    # Fetch METAR & forecasts
│   ├── feature_engine.py  # Build features from raw obs
│   ├── model.py           # XGBoost classifier + backtest
│   └── visualization.py   # Plotting utilities
│
├── scripts/
│   └── run_backtest.py    # CLI entry point
│
├── tests/
│   └── test_features.py   # Unit tests
│
├── notebooks/
│   └── walkthrough.ipynb  # Interactive demo
│
└── data/
    ├── metar_cache/       # Cached METAR CSVs
    └── forecast_cache/    # Cached forecasts
```

## Common Tasks

### Fetch Data for a Different Station

Supported stations: NYC, EWR, LGA, JFK, CHI, MIA, LAX, DEN

```python
fetcher = METARFetcher(station="LAX")
obs_df = fetcher.fetch("2025-01-01", "2025-06-30")
```

### Use a Different Evaluation Hour

Evaluation hour is UTC (0-23). For example:
- Hour 13 = 1 PM UTC (8 AM EST / 7 AM PST)
- Hour 14 = 2 PM UTC (9 AM EST / 6 AM PST)
- Hour 21 = 9 PM UTC (4 PM EST / 1 PM PST)

```python
features = build_features(obs_df, eval_hour=21)  # 9 PM UTC
```

### Integrate Forecast Models

The framework supports Open-Meteo forecasts:

```python
from src.data_fetcher import ForecastFetcher

# NYC coordinates: 40.78°N, 73.97°W
fetcher = ForecastFetcher(latitude=40.78, longitude=-73.97)

icon = fetcher.fetch_icon("2025-01-01", "2025-06-30")
gfs = fetcher.fetch_gfs("2025-01-01", "2025-06-30")
ecmwf = fetcher.fetch_ecmwf("2025-01-01", "2025-06-30")
```

### Create Custom Targets

Instead of synthetic targets, use your own categorical labels:

```python
# Example: temperature brackets
def cli_to_bracket(cli_temp):
    if cli_temp < 50:
        return 0
    elif cli_temp < 60:
        return 1
    elif cli_temp < 70:
        return 2
    else:
        return 3

targets = cli_data.apply(cli_to_bracket)
```

### Save Results to CSV

```python
results_df = pd.DataFrame(results['daily_results'])
results_df.to_csv("backtest_results.csv", index=False)
```

### Generate Custom Plots

```python
from src.visualization import plot_accuracy_by_month, plot_feature_importance

# Plot accuracy by month
plot_accuracy_by_month(
    results_df,
    figsize=(12, 6),
    save_path="accuracy_by_month.png"
)

# Plot feature importance
feature_importance = clf.feature_importance()
plot_feature_importance(
    feature_importance,
    top_n=15,
    save_path="features.png"
)
```

## Troubleshooting

### "No observations downloaded"
- Check station code is valid (NYC, EWR, LGA, etc.)
- Verify date range doesn't exceed available data
- Iowa State archive goes back to ~1990s for most stations
- Check internet connection

### "Empty response from IEM"
- Date range may be invalid or in future
- Try a shorter date range to isolate the issue
- The Iowa State API can be slow; give it 30+ seconds

### ImportError for xgboost or other packages
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

### Model accuracy is low (< 30%)
- This is expected on a 4-class problem with synthetic targets
- With real temperature data, accuracy should be 50-70%
- Feature engineering quality directly impacts model performance

## Next Steps

1. **Understand the data**: Run the Jupyter notebook to see feature examples
2. **Integrate real targets**: Replace synthetic targets with actual CLI temperatures
3. **Add forecast models**: Combine METAR features with ICON/GFS/ECMWF predictions
4. **Optimize hyperparameters**: Tune XGBoost params for your use case
5. **Deploy**: Use `model.joblib` to save trained model for serving

## Documentation

- **ARCHITECTURE.md**: Design decisions and module descriptions
- **src/model.py**: Walk-forward backtest methodology explained in docstrings
- **src/feature_engine.py**: Each feature function documented with examples

## Contributing

This is a portfolio project. Feel free to:
- Fork and extend for your own use cases
- Experiment with different models (RF, GBDT, neural networks)
- Add more weather stations
- Integrate additional data sources

## Questions?

See ARCHITECTURE.md for detailed design documentation and design patterns.

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Author**: Matthew Carlino
