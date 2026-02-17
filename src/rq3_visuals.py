"""
RQ3 Visualization module for Profit Erosion Capstone Project.

Provides plotting functions for predictive modeling results:
- ROC curves for model comparison
- Feature importance bar charts
- Confusion matrices
- Precision-recall curves
"""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

from src.config import REPORTS_DIR


def _safe_tight_layout():
    """Apply tight_layout with warning suppression for small figures."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Tight layout.*")
        plt.tight_layout()


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ROC curves for all models on the same axes.

    Args:
        results: Output from train_and_evaluate().
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, res in results.items():
        fpr, tpr, _ = res["roc_curve"]
        auc = res["test_auc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("RQ3: ROC Curves — Predicting High Profit Erosion Customers")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot horizontal bar charts of feature importance per model.

    Args:
        importance_df: DataFrame with columns: feature, model, importance.
        figsize: Figure size. Auto-calculated if None.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    models = importance_df["model"].unique()
    n_models = len(models)

    if figsize is None:
        figsize = (6 * n_models, max(5, len(importance_df["feature"].unique()) * 0.4))

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        model_data = (
            importance_df[importance_df["model"] == model_name]
            .sort_values("importance", ascending=True)
        )
        ax.barh(model_data["feature"], model_data["importance"], color="steelblue")
        ax.set_title(model_name)
        ax.set_xlabel("Importance")

    fig.suptitle("RQ3: Feature Importance by Model (Post-Hoc)", y=1.02)
    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot confusion matrices side-by-side for each model.

    Args:
        results: Output from train_and_evaluate().
        figsize: Figure size. Auto-calculated if None.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    n_models = len(results)
    if figsize is None:
        figsize = (5 * n_models, 4)

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    labels = ["Low Erosion", "High Erosion"]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax,
        )
        ax.set_title(f"{name}\n(AUC={res['test_auc']:.3f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("RQ3: Confusion Matrices", y=1.02)
    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_precision_recall_curves(
    results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot precision-recall curves for all models.

    Args:
        results: Output from train_and_evaluate(). Each entry must contain
            'y_proba' (predicted probabilities) and the y_test used for evaluation.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=figsize)

    for name, res in results.items():
        y_proba = res["y_proba"]
        y_pred = res["y_pred"]
        # Reconstruct y_test from confusion matrix dimensions
        # PR curve requires y_test — store it in results during train_and_evaluate
        if "y_test" in res:
            precision, recall, _ = precision_recall_curve(res["y_test"], y_proba)
            ax.plot(recall, precision, label=f"{name} (F1={res['f1']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("RQ3: Precision-Recall Curves")
    ax.legend(loc="lower left")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
