"""
Visualization module for the Profit Erosion E-commerce Capstone Project.

This module provides plotting functions for EDA and analysis reporting.
"""

import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    bars = ax.bar(
        status_counts.index, status_counts.values, color="steelblue", edgecolor="black"
    )

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
            item_rows=("order_id", "size"),
            returned_items=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned_items"] / x["item_rows"])
        .query("item_rows >= @min_rows")
        .sort_values("return_rate", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        ret_by_cat.index,
        ret_by_cat["return_rate"] * 100,
        color="coral",
        edgecolor="black",
    )

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
    axes[0].axvline(
        data.median(),
        color="red",
        linestyle="--",
        label=f"Median: ${data.median():.2f}",
    )
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
            returned_items=("order_id", "count"),
            total_lost_margin=("item_margin", "sum"),
        )
        .sort_values("total_lost_margin", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        margin_loss.index,
        margin_loss["total_lost_margin"],
        color="indianred",
        edgecolor="black",
    )

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
            return_events=("order_id", "count"),
            total_lost_margin=("item_margin", "sum"),
        )
        .sort_values("total_lost_margin", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(customer_exposure))
    bars = ax.bar(
        x, customer_exposure["total_lost_margin"], color="darkorange", edgecolor="black"
    )

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
            item_rows=("order_id", "size"),
            returned_items=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned_items"] / x["item_rows"])
        .reset_index()
    )

    # Filter low sample sizes
    pivot = pivot[pivot["item_rows"] >= min_rows]

    # Create pivot table
    heatmap_data = pivot.pivot(index=row_col, columns=col_col, values="return_rate")
    # Convert pd.NA to np.nan for compatibility with seaborn
    heatmap_data = heatmap_data.astype("float64")

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


def plot_price_margin_returned_by_status_country(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Create a 3x2 grid of bar charts showing cost, price, and margin metrics for RETURNED items by country.

    Creates 6 bar charts arranged as:
    - Row 1: avg_cost, total_cost
    - Row 2: avg_sale_price, total_sale_price
    - Row 3: avg_margin, total_margin
    - Row 4: item_count (volume)

    **Important:** Input DataFrame should contain ONLY returned items (already filtered).
    The DataFrame is expected to be the output of calculate_price_margin_by_status_country()
    which automatically filters for status='Returned'.

    Args:
        df: DataFrame with aggregated metrics for returned items by country
            (output from calculate_price_margin_by_status_country).
            Already contains only returned items - no filtering applied in this function.
        save_path: Optional path to save figures.

    Returns:
        None (displays plots directly)
    """
    # Reset index to convert multi-index into columns
    df_plot = df.reset_index()

    if df_plot.empty:
        print("No data found in input DataFrame")
        return

    # Create 4x2 grid (7 charts total, last one is item_count)
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(
        "Cost, Price, Margin, and Volume Analysis for RETURNED Items by Country",
        fontsize=16,
        fontweight="bold",
        y=0.998,
    )

    # Helper function to create bar charts
    def create_bar_chart(ax, data, metric_name, ylabel_text):
        # For returned items only, create simple bar chart by country
        data_grouped = data.groupby("country")[metric_name].first()
        data_grouped.plot(
            kind="bar", ax=ax, color="steelblue", edgecolor="black", width=0.7
        )
        ax.set_title(
            metric_name.replace("_", " ").title(), fontsize=12, fontweight="bold"
        )
        ax.set_ylabel(ylabel_text, fontsize=11)
        ax.set_xlabel("Country", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    # Row 1: Cost metrics
    create_bar_chart(axes[0, 0], df_plot, "avg_cost", "Average Cost ($)")
    create_bar_chart(axes[0, 1], df_plot, "total_cost", "Total Cost ($)")

    # Row 2: Price metrics
    create_bar_chart(axes[1, 0], df_plot, "avg_sale_price", "Average Sale Price ($)")
    create_bar_chart(axes[1, 1], df_plot, "total_sale_price", "Total Sale Price ($)")

    # Row 3: Margin metrics
    create_bar_chart(axes[2, 0], df_plot, "avg_margin", "Average Margin ($)")
    create_bar_chart(axes[2, 1], df_plot, "total_margin", "Total Margin ($)")

    # Row 4: Item Count (volume)
    create_bar_chart(axes[3, 0], df_plot, "item_count", "Item Count (Volume)")
    axes[3, 1].axis("off")  # Hide the last subplot

    _safe_tight_layout()

    if save_path:
        fig.savefig(f"{save_path}_metrics_grid.png", dpi=150, bbox_inches="tight")
