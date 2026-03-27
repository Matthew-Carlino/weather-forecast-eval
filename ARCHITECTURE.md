# Architecture & Design Document

## Project Overview

**weather-forecast-eval** is a production-grade Python framework for evaluating temperature forecasts using station-level METAR observations and numerical weather prediction models.

The project demonstrates:
- Clean data engineering practices (multiple API sources, caching, error handling)
- Proper machine learning methodology (walk-forward backtesting, no future peeking)
- Feature engineering from raw meteorological data
- Professional Python code quality (type hints, docstrings, testing)

## Module Structure

### `src/data_fetcher.py` — Data Acquisition

**Purpose**: Fetch and cache METAR observations and forecasts

**Classes**:
- `METARFetcher`: Pulls hourly observations from Iowa State University ASOS archive
  - Handles timezone conversion (all UTC internally)
  - Implements local CSV caching to avoid re-downloading
  - Graceful error handling for missing data or API failures
  
- `ForecastFetcher`: Pulls numerical forecasts from Open-Meteo API
  - Supports multiple models: ICON, GFS, ECMWF
  - Returns daily maximum temperature in °C
  - JSON-based caching with date range keys

**Design decisions**:
- All datetimes are UTC-aware (pytz aware datetime objects)
- CSV caching by date range to allow incremental updates
- Lazy fetching — only downloads when data not in cache or `force_refresh=True`

### `src/feature_engine.py` — Feature Engineering

**Purpose**: Transform raw observations into interpretable features for ML

**Key functions**:
- `compute_running_daily_max()`: Running maximum during each calendar day
- `compute_hourly_delta()`: Temperature change over specified hours
- `compute_trend()`: Linear regression slope of last N hours
- `compute_diurnal_progress()`: Fraction of expected daily range achieved
- `compute_airport_spread()`: Difference between primary station and nearby airports
- `build_features()`: Complete pipeline returning feature DataFrame

**Design philosophy**:
- All functions are pure (no side effects, deterministic, testable)
- Handles multi-station inputs for computing relative spreads
- Expected diurnal ranges vary by month (lookup table)
- No NaN values after final feature matrix (uses forward/backward fill + zeros)

### `src/model.py` — Walk-Forward XGBoost Classifier

**Purpose**: Implement proper temporal validation for time-series forecasting

**Classes**:
- `OMOClassifier`: XGBoost classifier for categorical temperature prediction
  - Predicts 4 classes (remaining afternoon uplift: 0, 1, 2, or 3+ °F)
  - `walk_forward_backtest()`: Expanding window with no future peeking
  - Retrain periodically (every N days) as new data arrives
  - Returns predictions + confidence scores

**Walk-forward methodology**:
```
For each test date t:
  1. Train on all data strictly before t
  2. Make prediction for t
  3. Record actual outcome
  4. Move to t+1
```

This prevents look-ahead bias and simulates realistic deployment.

**XGBoost parameters** (tuned for temperature prediction):
- n_estimators: 150 trees
- max_depth: 4 (shallow for stability)
- learning_rate: 0.08 (conservative)
- subsample: 0.8 (reduce overfitting)
- min_child_weight: 5

### `src/visualization.py` — Plotting & Analysis

**Purpose**: Clean matplotlib/seaborn plots for results presentation

**Functions**:
- `plot_accuracy_by_month()`: Monthly accuracy comparison
- `plot_feature_importance()`: Top features ranked by gain
- `plot_confusion_matrix()`: Per-class prediction errors
- `plot_accuracy_over_time()`: Rolling accuracy time series
- `plot_calibration_curve()`: Predicted prob vs actual accuracy
- `plot_prediction_confidence_distribution()`: Histogram of confidence

**Design**:
- Consistent color scheme (seaborn husl palette)
- All plots high-DPI (150 dpi) and publication-ready
- Optional `save_path` parameter for automated report generation

## Data Flow

```
                       ┌─────────────────────┐
                       │  METAR Observations │
                       │  (Iowa State IEM)   │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  METARFetcher       │
                       │  - Download         │
                       │  - Cache CSV        │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  build_features()   │
                       │  - Running max      │
                       │  - Trends           │
                       │  - Spreads          │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Feature Matrix     │
                       │  (N days × M feats) │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Target Labels    │      │ OMOClassifier    │
          │ (actual outcomes)│      │ - Train          │
          └──────────────────┘      │ - Predict        │
                    │               │ - Backtest       │
                    │               └────────┬─────────┘
                    │                        │
                    └────────────┬───────────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │  Results DataFrame  │
                       │  - Predictions      │
                       │  - Confidence       │
                       │  - Accuracy metrics │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Visualizations     │
                       │  - Accuracy plots   │
                       │  - Feature import   │
                       │  - Confusion matrix │
                       └─────────────────────┘
```

## Key Design Patterns

### 1. Pure Functions for Feature Engineering
All feature functions are stateless and deterministic. This makes them:
- Easy to test (unit tests don't require mocking)
- Composable (can be applied in any order)
- Parallelizable (no side effects)

### 2. Proper Temporal Validation
Walk-forward backtesting prevents look-ahead bias. Never train on test data.

### 3. Type Hints Throughout
Every function signature includes type hints for clarity and IDE support.

### 4. Google-Style Docstrings
Docstrings include:
- One-line summary
- Extended description of purpose
- Args with types and descriptions
- Returns with types
- Raises for exception cases

### 5. Logging over Print
Library code uses Python `logging` module (not print statements), allowing users to control verbosity.

## Testing

Unit tests in `tests/test_features.py` cover:
- Running max computation
- Hourly delta calculation
- Trend estimation
- Feature engineering pipeline
- Shape and type correctness

Run tests with:
```bash
pytest tests/test_features.py -v
```

## Deployment Considerations

### Production Checklist
- [ ] Cache directory writable and has sufficient disk space
- [ ] API keys stored in environment variables (not in code)
- [ ] Logging configured appropriately
- [ ] Error handling for network failures
- [ ] Data quality monitoring (missing values, outliers)
- [ ] Model performance tracking over time
- [ ] Automated retraining pipeline

### Scaling
For high-volume usage:
1. Use streaming data fetchers (fetch incremental updates, not full history)
2. Implement model persistence (pickle/joblib) to avoid retraining
3. Add Redis/memcached for feature caching
4. Consider parallel feature computation across days

## Future Extensions

1. **Ensemble Methods**: Combine multiple model predictions
2. **Neural Networks**: LSTM for temporal dependencies
3. **Feature Store**: Centralized feature management and versioning
4. **Multi-location Support**: Extend to dozens of weather stations
5. **API Wrapper**: REST API for production serving
6. **Monitoring Dashboard**: Track model performance metrics in real-time

## Performance Benchmarks

On a typical machine (MacBook Pro 2021):
- Fetching 6 months METAR data: ~2-3 seconds (cached: <100ms)
- Feature engineering: ~50ms for 182 days
- Walk-forward backtest (180 test days): ~5-10 seconds
- Generating plots: ~2-3 seconds per plot

## Data Sources & Attribution

- **METAR Observations**: Iowa State University - Automated Surface Observing System (ASOS) Archive
- **Weather Forecasts**: Open-Meteo (free, no API key required)
  - ICON model: Deutscher Wetterdienst (DWD)
  - GFS model: NOAA
  - ECMWF: European Centre for Medium-Range Weather Forecasts

## Code Quality Standards

All code adheres to:
- **PEP 8**: Python style guide
- **Type hints**: Every function signature
- **Docstrings**: Google format on all public functions
- **No star imports**: Explicit imports only
- **f-strings**: Not .format() or %
- **Logging**: Not print() in library code
- **Constants**: UPPERCASE at module top
