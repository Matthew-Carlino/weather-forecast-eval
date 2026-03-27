#!/usr/bin/env python3
"""Run a complete walk-forward backtest for temperature forecast evaluation.

Example:
    python scripts/run_backtest.py --station NYC --start 2025-01-01 --end 2025-12-31 --eval-hour 13

This script:
    1. Fetches METAR observations from Iowa State
    2. Engineers features at the specified hour
    3. Generates synthetic targets (example: discretized temperature brackets)
    4. Runs walk-forward XGBoost classification
    5. Prints accuracy metrics
    6. Saves plots to outputs/
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data_fetcher import METARFetcher
from src.feature_engine import build_features, get_feature_names
from src.model import OMOClassifier, evaluate_predictions
from src.visualization import (
    plot_accuracy_by_month,
    plot_confusion_matrix,
    plot_feature_importance,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_synthetic_targets(features_df: pd.DataFrame) -> pd.Series:
    """Generate synthetic target labels for demonstration.

    In a real scenario, this would come from actual measured high temperatures.
    Here we create a simple synthetic target based on running_max + random noise.

    Args:
        features_df: Feature DataFrame with 'running_max' column

    Returns:
        Series with class labels (0, 1, 2, 3)
    """
    # Simulate afternoon uplift based on running max and other features
    running_max = features_df["running_max"].values
    diurnal_progress = features_df["diurnal_progress"].values
    trend = features_df["trend_3h"].values

    # Create synthetic labels:
    # Class depends on temperature and progress through day
    base_class = (running_max - 32) / 20  # Normalize temp to 0-4 scale
    adjustment = diurnal_progress * 0.5 + trend * 2  # Add features

    synthetic_uplift = base_class + adjustment

    # Map to classes
    targets = pd.cut(
        synthetic_uplift,
        bins=[-np.inf, 0.5, 1.5, 2.5, np.inf],
        labels=[0, 1, 2, 3],
    )

    return targets.astype(int)


def print_results_table(backtest_results: dict) -> None:
    """Print formatted results table.

    Args:
        backtest_results: Results dict from walk_forward_backtest()
    """
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS".center(70))
    print("=" * 70)

    overall_accuracy = backtest_results["accuracy"]
    print(f"\nOverall Accuracy: {overall_accuracy:.1%}")

    accuracy_by_class = backtest_results["accuracy_by_class"]
    if accuracy_by_class:
        print("\nPer-Class Accuracy:")
        for cls in sorted(accuracy_by_class.keys()):
            acc = accuracy_by_class[cls]
            print(f"  Class {cls}: {acc:.1%}")

    print(f"\nTotal Test Samples: {len(backtest_results['actual'])}")

    # Compute additional metrics
    from sklearn.metrics import precision_recall_fscore_support
    y_true = backtest_results["actual"]
    y_pred = backtest_results["predictions"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"Macro Precision: {precision:.3f}")
    print(f"Macro Recall: {recall:.3f}")
    print(f"Macro F1: {f1:.3f}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward XGBoost backtest for temperature forecasts"
    )
    parser.add_argument(
        "--station",
        type=str,
        default="NYC",
        help="Station code (NYC, EWR, LGA, JFK, CHI, MIA, LAX, DEN)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2025-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--eval-hour",
        type=int,
        default=13,
        help="Hour of day to evaluate features (UTC, 0-23)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download data even if cached",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Backtest configuration:")
    logger.info(f"  Station: {args.station}")
    logger.info(f"  Date range: {args.start} to {args.end}")
    logger.info(f"  Eval hour: {args.eval_hour} UTC")
    logger.info(f"  Output directory: {output_dir}")

    # 1. Fetch METAR data
    logger.info("\n1. Fetching METAR observations...")
    fetcher = METARFetcher(args.station)
    obs_df = fetcher.fetch(args.start, args.end, force_refresh=args.force_refresh)
    logger.info(f"   Downloaded {len(obs_df)} observations")

    if len(obs_df) == 0:
        logger.error("No observations downloaded. Exiting.")
        return

    # 2. Build features
    logger.info("\n2. Engineering features...")
    obs_df.index = pd.to_datetime(obs_df["valid"])
    features_df = build_features(obs_df, eval_hour=args.eval_hour)
    logger.info(f"   Built {len(features_df)} feature vectors")
    logger.info(f"   Features: {get_feature_names()}")

    # 3. Generate synthetic targets
    logger.info("\n3. Generating synthetic targets...")
    import numpy as np
    targets_df = generate_synthetic_targets(features_df)
    logger.info(f"   Target class distribution:")
    for cls in sorted(targets_df.unique()):
        count = (targets_df == cls).sum()
        pct = 100.0 * count / len(targets_df)
        logger.info(f"     Class {cls}: {count} ({pct:.1f}%)")

    # 4. Run walk-forward backtest
    logger.info("\n4. Running walk-forward backtest...")
    clf = OMOClassifier()
    backtest_results = clf.walk_forward_backtest(
        features_df[["date"] + get_feature_names()],
        targets_df,
        min_train=100,
        retrain_every=10,
    )

    # 5. Print results
    print_results_table(backtest_results)

    # 6. Save results to CSV
    logger.info("\n5. Saving results...")
    results_csv = pd.DataFrame(backtest_results["daily_results"])
    results_csv_path = output_dir / "backtest_results.csv"
    results_csv.to_csv(results_csv_path, index=False)
    logger.info(f"   Saved results to {results_csv_path}")

    # 7. Generate plots
    logger.info("\n6. Generating plots...")

    plot_accuracy_by_month(
        results_csv,
        save_path=str(output_dir / "accuracy_by_month.png"),
    )

    feature_importance = clf.feature_importance()
    plot_feature_importance(
        feature_importance,
        save_path=str(output_dir / "feature_importance.png"),
    )

    plot_confusion_matrix(
        backtest_results["actual"],
        backtest_results["predictions"],
        class_names=["Class 0", "Class 1", "Class 2", "Class 3"],
        save_path=str(output_dir / "confusion_matrix.png"),
    )

    logger.info(f"\nAll plots saved to {output_dir}/")
    logger.info("Backtest complete!")


if __name__ == "__main__":
    main()
