"""Walk-forward XGBoost classification for temperature forecasts."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# XGBoost hyperparameters optimized for temperature classification
XGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": 42,
    "verbosity": 0,
}


class OMOClassifier:
    """XGBoost classifier for predicting afternoon temperature uplift.

    Predicts the "Overnight to Mid-afternoon" (OMO) temperature class based on
    current observations. The OMO class represents how much additional warming
    is expected during the afternoon:
        - Class 0: ≤0.5°F additional
        - Class 1: 0.5-1.5°F additional
        - Class 2: 1.5-2.5°F additional
        - Class 3: >2.5°F additional

    This is useful for predicting final daily high temperatures.

    Attributes:
        model: Underlying XGBoost model
        classes_: Array of class labels
        feature_names_: List of feature column names
    """

    def __init__(self, params: Optional[Dict] = None):
        """Initialize the classifier.

        Args:
            params: XGBoost parameters dict. If None, uses defaults.
        """
        self.params = params if params is not None else XGB_PARAMS.copy()
        self.model = None
        self.classes_ = None
        self.feature_names_ = None
        self.label_encoder_ = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost classifier.

        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
            y: Target class labels (0, 1, 2, or 3)

        Raises:
            ValueError: If X or y have incompatible shapes
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y have mismatched lengths: {len(X)} vs {len(y)}"
            )

        self.feature_names_ = X.columns.tolist()

        # Encode labels if not already numeric
        if not np.issubdtype(y.dtype, np.integer):
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
        else:
            y_encoded = y.values
            self.classes_ = np.unique(y_encoded)

        logger.info(
            f"Training XGBoost with {len(X)} samples, "
            f"{X.shape[1]} features, {len(self.classes_)} classes"
        )

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, y_encoded)

        # Log class distribution
        unique, counts = np.unique(y_encoded, return_counts=True)
        for class_idx, count in zip(unique, counts):
            pct = 100.0 * count / len(y_encoded)
            logger.debug(f"  Class {class_idx}: {count} samples ({pct:.1f}%)")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and confidence scores.

        Args:
            X: Feature DataFrame with shape (n_samples, n_features)

        Returns:
            Tuple of:
                - predictions: Class labels
                - confidence: Max predicted probability for each sample

        Raises:
            RuntimeError: If model has not been trained
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call train() first.")

        # Get predictions and probabilities
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        # Confidence is max probability
        confidence = np.max(y_proba, axis=1)

        return y_pred, confidence

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - importance: Gain-based importance score
                - importance_pct: Importance as percentage of total
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")

        importances = self.model.feature_importances_
        feature_names = self.feature_names_

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        df["importance_pct"] = 100.0 * df["importance"] / df["importance"].sum()
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def walk_forward_backtest(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        min_train: int = 100,
        retrain_every: int = 10,
    ) -> Dict:
        """Run walk-forward backtest with expanding training window.

        Proper temporal validation: for each test date, trains only on data
        strictly before that date (no future peeking).

        Args:
            features_df: DataFrame with features, must have 'date' column
            targets_df: DataFrame with target classes, same length as features_df
            min_train: Minimum training samples before making predictions
            retrain_every: Retrain model every N test days

        Returns:
            Dict with keys:
                - predictions: Array of predicted classes
                - confidence: Array of confidence scores
                - actual: Array of actual target classes
                - dates: Array of test dates
                - accuracy: Overall accuracy
                - accuracy_by_class: Dict of per-class accuracy
                - daily_results: List of dicts with date-level results
                - training_sizes: List of training set sizes used

        Raises:
            ValueError: If features_df or targets_df is empty
        """
        if len(features_df) == 0 or len(targets_df) == 0:
            raise ValueError("Empty input DataFrames")

        if len(features_df) != len(targets_df):
            raise ValueError("features_df and targets_df must have same length")

        if "date" not in features_df.columns:
            raise ValueError("features_df must contain 'date' column")

        dates = features_df["date"].values
        X_cols = [c for c in features_df.columns if c != "date"]
        X = features_df[X_cols].values
        y = targets_df.values.flatten()

        # Sort by date
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        X = X[sort_idx]
        y = y[sort_idx]

        predictions = []
        confidence_scores = []
        actual = []
        test_dates = []
        training_sizes = []
        daily_results = []

        last_retrain_idx = 0

        for test_idx in range(min_train, len(dates)):
            # Training data: everything before test date
            train_X = X[:test_idx]
            train_y = y[:test_idx]

            # Only retrain if enough steps have passed
            if test_idx - last_retrain_idx >= retrain_every or self.model is None:
                logger.debug(
                    f"Retraining at index {test_idx} "
                    f"(training size: {len(train_X)})"
                )
                self.train(
                    pd.DataFrame(train_X, columns=X_cols),
                    pd.Series(train_y),
                )
                last_retrain_idx = test_idx

            # Test on current date
            test_X = X[test_idx : test_idx + 1]
            pred, conf = self.predict(
                pd.DataFrame(test_X, columns=X_cols)
            )

            predictions.append(pred[0])
            confidence_scores.append(conf[0])
            actual.append(y[test_idx])
            test_dates.append(dates[test_idx])
            training_sizes.append(len(train_X))

            daily_results.append({
                "date": dates[test_idx],
                "prediction": pred[0],
                "confidence": conf[0],
                "actual": y[test_idx],
                "training_size": len(train_X),
                "correct": pred[0] == y[test_idx],
            })

        predictions = np.array(predictions)
        confidence_scores = np.array(confidence_scores)
        actual = np.array(actual)

        # Compute accuracy
        accuracy = accuracy_score(actual, predictions)

        # Per-class accuracy
        unique_classes = np.unique(actual)
        accuracy_by_class = {}
        for cls in unique_classes:
            mask = actual == cls
            if mask.sum() > 0:
                cls_accuracy = accuracy_score(actual[mask], predictions[mask])
                accuracy_by_class[int(cls)] = cls_accuracy

        logger.info(
            f"Walk-forward backtest complete: {accuracy:.1%} accuracy "
            f"({len(actual)} test samples)"
        )

        return {
            "predictions": predictions,
            "confidence": confidence_scores,
            "actual": actual,
            "dates": test_dates,
            "accuracy": accuracy,
            "accuracy_by_class": accuracy_by_class,
            "daily_results": daily_results,
            "training_sizes": training_sizes,
        }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate classifier predictions.

    Args:
        y_true: Actual class labels
        y_pred: Predicted class labels
        y_proba: Predicted probabilities (n_samples, n_classes), optional

    Returns:
        Dict with evaluation metrics:
            - accuracy: Overall accuracy
            - precision: Per-class precision
            - recall: Per-class recall
            - f1: Per-class F1 score
            - confusion_matrix: Confusion matrix (dense)
    """
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": conf_matrix,
    }
