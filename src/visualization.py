"""
Visualization module for the Profit Erosion E-commerce Capstone Project.

This module provides plotting functions for EDA and analysis reporting.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, Tuple

from src.config import FIGURES_DIR, MIN_ROWS_THRESHOLD


def _safe_tight_layout():
    """Apply tight_layout with warning suppression for small figures."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Tight layout.*")
        plt.tight_layout()


def set_plot_style():
    """Set consistent plotting style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11


def plot_status_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the distribution of order item statuses.

    Args:
        df: DataFrame with item_status column.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    status_counts = df["item_status"].value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(status_counts.index, status_counts.values, color="steelblue", edgecolor="black")

    ax.set_xlabel("Item Status")
    ax.set_ylabel("Count")
    ax.set_title("Order Item Status Distribution")

    # Add value labels on bars
    for bar, val in zip(bars, status_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_return_rate_by_category(
    df: pd.DataFrame,
    top_n: int = 15,
    min_rows: int = MIN_ROWS_THRESHOLD,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot return rates by product category.

    Args:
        df: DataFrame with category and is_returned_item columns.
        top_n: Number of top categories to display.
        min_rows: Minimum sample size for inclusion.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    ret_by_cat = (
        df.groupby("category")
        .agg(
            item_rows=("order_item_id", "size"),
            returned_items=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned_items"] / x["item_rows"])
        .query("item_rows >= @min_rows")
        .sort_values("return_rate", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(ret_by_cat.index, ret_by_cat["return_rate"] * 100, color="coral", edgecolor="black")

    ax.set_xlabel("Return Rate (%)")
    ax.set_ylabel("Category")
    ax.set_title(f"Top {top_n} Categories by Return Rate (min {min_rows} items)")
    ax.invert_yaxis()

    # Add percentage labels
    for bar, val in zip(bars, ret_by_cat["return_rate"]):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_margin_distribution(
    df: pd.DataFrame,
    returned_only: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the distribution of item margins.

    Args:
        df: DataFrame with item_margin column.
        returned_only: If True, only plot margins for returned items.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if returned_only:
        data = df.loc[df["is_returned_item"] == 1, "item_margin"].dropna()
        title = "Margin Distribution - Returned Items Only"
    else:
        data = df["item_margin"].dropna()
        title = "Margin Distribution - All Items"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    axes[0].hist(data, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Item Margin ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(title)
    axes[0].axvline(data.median(), color="red", linestyle="--", label=f"Median: ${data.median():.2f}")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(data, vert=True)
    axes[1].set_ylabel("Item Margin ($)")
    axes[1].set_title("Margin Box Plot")

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_margin_loss_by_category(
    df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot total margin loss by category for returned items.

    Args:
        df: DataFrame with category, is_returned_item, and item_margin columns.
        top_n: Number of top categories to display.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    margin_loss = (
        df.loc[df["is_returned_item"] == 1]
        .groupby("category")
        .agg(
            returned_items=("order_item_id", "count"),
            total_lost_margin=("item_margin", "sum"),
        )
        .sort_values("total_lost_margin", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(margin_loss.index, margin_loss["total_lost_margin"], color="indianred", edgecolor="black")

    ax.set_xlabel("Total Lost Margin ($)")
    ax.set_ylabel("Category")
    ax.set_title(f"Top {top_n} Categories by Margin Loss from Returns")
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, margin_loss["total_lost_margin"]):
        ax.text(
            bar.get_width() + 500,
            bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_customer_margin_exposure(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot customers with highest margin exposure from returns.

    Args:
        df: DataFrame with user_id, is_returned_item, and item_margin columns.
        top_n: Number of top customers to display.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    customer_exposure = (
        df.loc[df["is_returned_item"] == 1]
        .groupby("user_id")
        .agg(
            return_events=("order_item_id", "count"),
            total_lost_margin=("item_margin", "sum"),
        )
        .sort_values("total_lost_margin", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(customer_exposure))
    bars = ax.bar(x, customer_exposure["total_lost_margin"], color="darkorange", edgecolor="black")

    ax.set_xlabel("Customer (User ID)")
    ax.set_ylabel("Total Lost Margin ($)")
    ax.set_title(f"Top {top_n} Customers by Margin Exposure from Returns")
    ax.set_xticks(x)
    ax.set_xticklabels(customer_exposure.index, rotation=45, ha="right")

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_return_rate_heatmap(
    df: pd.DataFrame,
    row_col: str = "category",
    col_col: str = "traffic_source",
    min_rows: int = 100,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap of return rates across two dimensions.

    Args:
        df: DataFrame with required columns.
        row_col: Column for heatmap rows.
        col_col: Column for heatmap columns.
        min_rows: Minimum sample size for cell inclusion.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    pivot = (
        df.groupby([row_col, col_col])
        .agg(
            item_rows=("order_item_id", "size"),
            returned_items=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned_items"] / x["item_rows"])
        .reset_index()
    )

    # Filter low sample sizes
    pivot = pivot[pivot["item_rows"] >= min_rows]

    # Create pivot table
    heatmap_data = pivot.pivot(index=row_col, columns=col_col, values="return_rate")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        heatmap_data * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Return Rate (%)"},
        ax=ax,
    )

    ax.set_title(f"Return Rate (%) by {row_col} and {col_col}")
    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
