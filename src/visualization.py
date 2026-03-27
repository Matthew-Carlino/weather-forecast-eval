"""Plotting and visualization utilities for forecast evaluation."""

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
COLORS = sns.color_palette("husl", 4)


def plot_accuracy_by_month(
    results_df: pd.DataFrame,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy by calendar month.

    Args:
        results_df: DataFrame with 'date' and 'correct' columns
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df["month"] = results_df["date"].dt.month
    results_df["month_name"] = results_df["date"].dt.strftime("%B")

    monthly_acc = (
        results_df.groupby(["month", "month_name"])["correct"]
        .agg(["sum", "count"])
        .reset_index()
    )
    monthly_acc["accuracy"] = monthly_acc["sum"] / monthly_acc["count"]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        monthly_acc["month_name"],
        monthly_acc["accuracy"],
        color=COLORS[0],
        alpha=0.7,
        edgecolor="black",
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_title("Classification Accuracy by Month", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_feature_importance(
    feature_importances: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot feature importance rankings.

    Args:
        feature_importances: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    top_features = feature_importances.head(top_n).copy()

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        range(len(top_features)),
        top_features["importance"],
        color=COLORS[1],
        alpha=0.7,
        edgecolor="black",
    )

    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importance", fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    # Add value labels
    for i, (bar, val, pct) in enumerate(
        zip(bars, top_features["importance"], top_features["importance_pct"])
    ):
        ax.text(val, i, f" {pct:.1f}%", va="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (8, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot calibration curve (predicted probability vs actual accuracy).

    Args:
        y_true: True class labels
        y_proba: Predicted probabilities (confidence scores)
        n_bins: Number of bins for grouping predictions
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    # Bin predictions by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            accuracy = (y_true[mask] == 1).mean()  # Assumes binary classification
            bin_accs.append(accuracy)
            bin_confs.append(bin_centers[i])
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Actual calibration
    ax.scatter(
        bin_confs,
        bin_accs,
        s=[c * 2 for c in bin_counts],
        alpha=0.6,
        color=COLORS[2],
        edgecolor="black",
        label="Observed",
    )

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Accuracy", fontsize=12)
    ax.set_title("Calibration Curve", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 7),
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        class_names: Names for classes (e.g., ["Class 0", "Class 1", ...])
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_accuracy_over_time(
    results_df: pd.DataFrame,
    window: int = 30,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot rolling accuracy over time.

    Args:
        results_df: DataFrame with 'date' and 'correct' columns
        window: Window size for rolling average (days)
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["date"])
    results_df = results_df.sort_values("date")
    results_df = results_df.set_index("date")

    rolling_acc = results_df["correct"].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rolling_acc.index, rolling_acc.values, color=COLORS[3], linewidth=2)
    ax.fill_between(rolling_acc.index, rolling_acc.values, alpha=0.3, color=COLORS[3])

    ax.set_ylabel(f"Rolling Accuracy ({window}-day window)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Classification Accuracy Over Time", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()


def plot_prediction_confidence_distribution(
    confidence_scores: np.ndarray,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """Plot histogram of predicted confidence scores.

    Args:
        confidence_scores: Array of confidence values [0, 1]
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        confidence_scores,
        bins=30,
        color=COLORS[0],
        alpha=0.7,
        edgecolor="black",
    )

    ax.axvline(
        confidence_scores.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {confidence_scores.mean():.3f}",
    )

    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Prediction Confidence", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()
