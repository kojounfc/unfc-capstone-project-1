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

from pathlib import Path
from typing import Sequence

from src.config import FIGURES_DIR, MIN_ROWS_THRESHOLD

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],  # force DejaVu first/only
})


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


def _validate_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    """
    Validate required columns exist before plotting.

    Args:
        df: Input DataFrame.
        required: List/sequence of required column names.
        context: Name of calling function for error messaging.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)[:50]}..."
        )


def _ensure_parent_dir(out_path: Path) -> None:
    """
    Ensure output directory exists.

    Args:
        out_path: File path where the figure will be saved.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_missingness_overview(
    df: pd.DataFrame,
    *,
    out_path: Path,
    top_n: int = 30,
    figsize: Tuple[int, int] = (12, 8),
) -> Path:
    """
    Plot missingness rate (%) for the top-N columns with missing values.

    Args:
        df: Input DataFrame.
        out_path: Output PNG path.
        top_n: Number of columns to display (sorted by missingness desc).
        figsize: Figure size.

    Returns:
        Saved figure path.
    """
    set_plot_style()
    _ensure_parent_dir(out_path)

    miss_pct = df.isna().mean().sort_values(ascending=False) * 100
    miss_pct = miss_pct[miss_pct > 0].head(top_n)

    if miss_pct.empty:
        # still save an "empty" diagnostic plot for traceability
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No missing values detected.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(miss_pct) + 2)))
    ax.barh(miss_pct.index[::-1], miss_pct.values[::-1], edgecolor="black", alpha=0.85)

    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Missingness Overview (Top Columns)")
    ax.grid(axis="x", alpha=0.25, linestyle=":")

    _safe_tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_numeric_distributions_grid(
    df: pd.DataFrame,
    *,
    numeric_cols: Sequence[str],
    out_path: Path,
    sample_n: int = 80_000,
    bins: int = 50,
) -> Path:
    """
    Plot histograms for multiple numeric columns in a single grid.

    Args:
        df: Input DataFrame.
        numeric_cols: Numeric columns to plot.
        out_path: Output PNG path.
        sample_n: Maximum number of rows to sample for speed.
        bins: Histogram bins.

    Returns:
        Saved figure path.
    """
    set_plot_style()
    _ensure_parent_dir(out_path)
    _validate_columns(df, list(numeric_cols), context="plot_numeric_distributions_grid")

    d = df[list(numeric_cols)].copy()
    if len(d) > sample_n:
        d = d.sample(sample_n, random_state=42)

    cols = list(numeric_cols)
    if not cols:
        raise ValueError("numeric_cols is empty. Provide at least one numeric column.")

    n = len(cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        x = pd.to_numeric(d[col], errors="coerce").dropna()
        axes[i].hist(x, bins=bins, edgecolor="black", alpha=0.8)
        axes[i].set_title(col)
        axes[i].grid(alpha=0.2, linestyle=":")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Numeric Distributions (Sampled)", y=0.995, fontweight="bold")
    _safe_tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_binary_target_balance(
    df: pd.DataFrame,
    *,
    target_col: str,
    out_path: Path,
    figsize: Tuple[int, int] = (8, 5),
) -> Path:
    """
    Plot class balance for a binary target column (e.g., is_returned_item).

    Args:
        df: Input DataFrame.
        target_col: Binary target column.
        out_path: Output PNG path.
        figsize: Figure size.

    Returns:
        Saved figure path.
    """
    set_plot_style()
    _ensure_parent_dir(out_path)
    _validate_columns(df, [target_col], context="plot_binary_target_balance")

    counts = df[target_col].value_counts(dropna=False).sort_index()
    labels = [str(i) for i in counts.index]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts.values, edgecolor="black", alpha=0.85)

    ax.set_xlabel(target_col)
    ax.set_ylabel("Rows")
    ax.set_title(f"Target Balance: {target_col}")

    for b, v in zip(bars, counts.values):
        pct = (v / total * 100) if total else 0
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,} ({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    _safe_tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


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


# ============================================================================
# RQ2 CONCENTRATION & SEGMENTATION VISUALIZATIONS
# ============================================================================


def plot_feature_concentration_ranking(
    concentration_df: pd.DataFrame,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create horizontal bar chart of feature concentration ranking.
    
    Args:
        concentration_df: Output from rank_features_by_concentration()
                         with columns: feature, gini_coefficient, p_value, etc.
        figsize: Figure size tuple. If None, auto-sized based on number of features.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    # Auto-size figure based on number of features
    if figsize is None:
        height = max(8, len(concentration_df) * 0.4)
        figsize = (14, height)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    plot_data = concentration_df.copy()
    plot_data['significant'] = plot_data['p_value'] < 0.05
    
    # Color mapping
    colors = plot_data['significant'].map({True: '#d62728', False: '#7f7f7f'})
    
    # Create bars
    bars = ax.barh(
        y=plot_data['feature'][::-1],
        width=plot_data['gini_coefficient'][::-1],
        color=colors[::-1],
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
    )
    
    # Add value labels
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        # Gini value
        ax.text(
            row['gini_coefficient'] + 0.01,
            i,
            f"{row['gini_coefficient']:.3f}",
            va='center',
            fontsize=9,
            fontweight='bold',
        )
        
        # P-value (inside bar if there's room)
        p_val_text = f"p={row['p_value']:.4f}" if row['p_value'] >= 0.001 else "p<0.001"
        text_color = 'white' if row['gini_coefficient'] > 0.15 else 'black'
        ax.text(
            0.01,
            i,
            p_val_text,
            va='center',
            fontsize=8,
            style='italic',
            color=text_color,
        )
    
    # Add concentration level reference lines
    ax.axvline(x=0.3, color='green', linestyle=':', alpha=0.5, linewidth=2, label='Moderate (0.3)')
    ax.axvline(x=0.5, color='orange', linestyle=':', alpha=0.5, linewidth=2, label='High (0.5)')
    ax.axvline(x=0.7, color='red', linestyle=':', alpha=0.5, linewidth=2, label='Extreme (0.7)')
    
    # Customize plot
    ax.set_xlabel('Gini Coefficient (Concentration)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature Concentration Ranking\n'
        '(Higher Gini = More Concentrated Among Fewer Customers)',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.set_xlim(0, max(plot_data['gini_coefficient']) * 1.2)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Statistically Significant (p < 0.05)'),
        Patch(facecolor='#7f7f7f', alpha=0.7, edgecolor='black', label='Not Significant (p ≥ 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=10)
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gini_vs_pareto_scatter(
    concentration_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create scatter plot of Gini coefficient vs Top 20% share.
    
    Args:
        concentration_df: Output from rank_features_by_concentration()
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate significant and non-significant
    sig = concentration_df[concentration_df['p_value'] < 0.05]
    non_sig = concentration_df[concentration_df['p_value'] >= 0.05]
    
    # Plot points
    ax.scatter(
        sig['gini_coefficient'],
        sig['top_20_pct_share'],
        s=150,
        c='#d62728',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5,
        label='Significant (p < 0.05)',
        zorder=3,
    )
    
    ax.scatter(
        non_sig['gini_coefficient'],
        non_sig['top_20_pct_share'],
        s=150,
        c='#7f7f7f',
        alpha=0.5,
        edgecolors='black',
        linewidth=1,
        label='Not Significant (p ≥ 0.05)',
        zorder=2,
    )
    
    # Add labels
    for idx, row in concentration_df.iterrows():
        ax.annotate(
            row['feature'],
            (row['gini_coefficient'], row['top_20_pct_share']),
            xytext=(8, 5),
            textcoords='offset points',
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7),
        )
    
    # Reference lines
    ax.axhline(y=80, color='blue', linestyle='--', alpha=0.4, linewidth=2, label='80/20 Rule (80%)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.4, linewidth=2, label='High Concentration (0.5)')
    
    # Quadrant shading (high priority = top-right)
    y_max = ax.get_ylim()[1]
    x_max = ax.get_xlim()[1]
    ax.axhspan(80, y_max, xmin=0.5/x_max, alpha=0.05, color='red', zorder=1)
    ax.text(
        0.7, 85, 
        'HIGH PRIORITY\nTargeted Intervention',
        fontsize=11,
        fontweight='bold',
        color='darkred',
        alpha=0.6,
        ha='center',
    )
    
    # Customize
    ax.set_xlabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Top 20% Customer Share (%)', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature Concentration: Gini vs Pareto Share\n'
        '(Top-Right Quadrant = Highest Priority for Targeted Intervention)',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pareto_curve(
    pareto_df: pd.DataFrame,
    gini: float,
    figsize: Tuple[int, int] = (12, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create Pareto curve showing cumulative concentration.
    
    Args:
        pareto_df: Output from compute_pareto_table() with customer_share and value_share columns
        gini: Gini coefficient value
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(
        pareto_df['customer_share'] * 100,
        pareto_df['value_share'] * 100,
        linewidth=2.5,
        color='#1f77b4',
        label='Actual Distribution'
    )
    
    # Add 80/20 reference
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='80% Threshold')
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='20% Threshold')
    
    # Mark the 20% point
    idx_20 = (pareto_df['customer_share'] - 0.20).abs().idxmin()
    share_at_20 = pareto_df.loc[idx_20, 'value_share'] * 100
    
    ax.plot(20, share_at_20, 'ro', markersize=12, zorder=5)
    ax.annotate(
        f'Top 20% = {share_at_20:.1f}%',
        xy=(20, share_at_20),
        xytext=(35, share_at_20 - 10),
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', lw=1.5, color='red')
    )
    
    ax.set_xlabel('Cumulative % of Customers (Ranked by Erosion)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative % of Profit Erosion', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Pareto Curve: Profit Erosion Concentration\n'
        f'(Gini = {gini:.3f}, Top 20% = {share_at_20:.1f}%)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_lorenz_curve(
    lorenz_df: pd.DataFrame,
    gini: float,
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create Lorenz curve showing inequality distribution.
    
    Args:
        lorenz_df: Output from lorenz_curve_points() with population_share and value_share columns
        gini: Gini coefficient value
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual distribution
    ax.plot(
        lorenz_df['population_share'],
        lorenz_df['value_share'],
        linewidth=2.5,
        color='#d62728',
        label=f'Actual Distribution (Gini={gini:.3f})'
    )
    
    # Plot line of equality
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Equality')
    
    # Shade area between curves (Gini visualization)
    ax.fill_between(
        lorenz_df['population_share'],
        lorenz_df['population_share'],
        lorenz_df['value_share'],
        alpha=0.3,
        color='red',
        label='Inequality Area (∝ Gini)'
    )
    
    ax.set_xlabel('Cumulative Share of Customers', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Share of Profit Erosion', fontsize=12, fontweight='bold')
    ax.set_title(
        'Lorenz Curve: Profit Erosion Inequality\n'
        '(Greater bow = higher inequality)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_clustering_diagnostics(
    elbow_df: pd.DataFrame,
    silhouette_df: pd.DataFrame,
    optimal_k: int,
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create side-by-side elbow and silhouette diagnostic plots.
    
    Args:
        elbow_df: DataFrame with columns k, inertia
        silhouette_df: DataFrame with columns k, silhouette
        optimal_k: The selected optimal k value
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    ax1.plot(
        elbow_df['k'],
        elbow_df['inertia'],
        marker='o',
        linewidth=2,
        markersize=8,
        color='#1f77b4'
    )
    ax1.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=11, fontweight='bold')
    ax1.set_title('Elbow Method: Inertia vs k', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xticks(elbow_df['k'])
    
    # Silhouette plot
    ax2.plot(
        silhouette_df['k'],
        silhouette_df['silhouette'],
        marker='o',
        linewidth=2,
        markersize=8,
        color='#2ca02c'
    )
    
    # Highlight optimal k
    optimal_row = silhouette_df[silhouette_df['k'] == optimal_k].iloc[0]
    ax2.plot(
        optimal_k,
        optimal_row['silhouette'],
        'ro',
        markersize=15,
        zorder=5,
        label=f'Optimal k={optimal_k}'
    )
    
    ax2.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax2.set_title('Silhouette Method: Score vs k', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xticks(silhouette_df['k'])
    ax2.legend(loc='best', fontsize=10)
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_erosion_comparison(
    cluster_summary_df: pd.DataFrame,
    optimal_k: int,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create bar chart comparing mean profit erosion by cluster.

    This function is intentionally tolerant to alternative summary schemas.
    It supports both:
      - summarize_clusters() output: cluster_id, Count, Mean_Erosion
      - SSL-style output: cluster_id, customers, mean_profit_erosion

    Args:
        cluster_summary_df: Cluster summary DataFrame.
        optimal_k: The k value used for clustering.
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    df = cluster_summary_df.copy()

    # ------------------------------------------------------------------
    # Schema normalization (backward compatible)
    # ------------------------------------------------------------------
    col_map = {}

    # cluster id
    if "cluster_id" not in df.columns and "cluster" in df.columns:
        col_map["cluster"] = "cluster_id"

    # mean erosion (bar height)
    if "Mean_Erosion" not in df.columns:
        if "mean_profit_erosion" in df.columns:
            col_map["mean_profit_erosion"] = "Mean_Erosion"
        elif "mean_erosion" in df.columns:
            col_map["mean_erosion"] = "Mean_Erosion"
        elif "avg_erosion" in df.columns:
            col_map["avg_erosion"] = "Mean_Erosion"

    # count (bar label)
    if "Count" not in df.columns:
        if "customers" in df.columns:
            col_map["customers"] = "Count"
        elif "count" in df.columns:
            col_map["count"] = "Count"
        elif "n_customers" in df.columns:
            col_map["n_customers"] = "Count"

    if col_map:
        df = df.rename(columns=col_map)

    required = ["cluster_id", "Mean_Erosion", "Count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "plot_cluster_erosion_comparison requires columns "
            f"{required}. Missing: {missing}. Available: {list(cluster_summary_df.columns)}"
        )

    # ensure numeric
    df["Mean_Erosion"] = pd.to_numeric(df["Mean_Erosion"], errors="coerce").fillna(0.0)
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0).astype(int)

    # stable ordering
    df = df.sort_values("cluster_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set3(range(len(df)))
    bars = ax.bar(
        df["cluster_id"].astype(str),
        df["Mean_Erosion"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add value labels on top
    for bar, value in zip(bars, df["Mean_Erosion"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"${value:,.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Add count labels inside bars
    for bar, count in zip(bars, df["Count"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 0.5,
            f"n={count:,}",
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
            color="white",
            fontweight="bold",
        )

    ax.set_xlabel("Cluster ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Profit Erosion ($)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Average Profit Erosion by Customer Segment (k={optimal_k})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    _safe_tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig



def plot_clustering_feature_importance(
    feature_importance_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create horizontal bar chart of feature importance for clustering.
    
    Shows which features best separate the discovered clusters, with
    features ordered from highest to lowest F-statistic (most important at top).
    
    Args:
        feature_importance_df: Output from analyze_feature_importance_for_clustering()
                               with columns: feature, f_statistic, p_value, significant
        figsize: Figure size tuple.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by significance
    colors = feature_importance_df['significant'].map({
        True: '#2ecc71',   # Green for significant
        False: '#e74c3c'   # Red for not significant
    })
    
    # Reverse order so highest F-statistic appears at top
    reversed_df = feature_importance_df[::-1].reset_index(drop=True)
    reversed_colors = colors[::-1].reset_index(drop=True)
    
    # Create horizontal bar chart
    bars = ax.barh(
        reversed_df['feature'],
        reversed_df['f_statistic'],
        color=reversed_colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Add p-value labels
    for i, (idx, row) in enumerate(reversed_df.iterrows()):
        p_text = f"p<0.001" if row['p_value'] < 0.001 else f"p={row['p_value']:.3f}"
        # Dynamic offset based on plot width
        offset = reversed_df['f_statistic'].max() * 0.02
        ax.text(
            row['f_statistic'] + offset,
            i,
            p_text,
            va='center',
            fontsize=9,
            style='italic'
        )
    
    # Labels and title
    ax.set_xlabel('F-Statistic (Cluster Separation Power)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(
        'Feature Importance for Cluster Separation\n'
        '(Higher F-statistic = Better differentiation between clusters)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, edgecolor='black', 
              label='Significant (p < 0.05)'),
        Patch(facecolor='#e74c3c', alpha=0.7, edgecolor='black', 
              label='Not Significant (p ≥ 0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    _safe_tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# ============================================================================
# RQ1 VISUALS
# Notes:
# - These functions were migrated from `rq1_visuals.py` so that RQ1 visuals live
#   in a single module (`visualization.py`).
# - All figures are saved to the provided `out_path`. The notebook is responsible
#   for ensuring `out_path` is inside figures/rq1/.
# - Each plot calls `set_plot_style()` and `_safe_tight_layout()` for consistency.
# ============================================================================

from pathlib import Path
from typing import Sequence, List


def _rq1_validate_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    """Validate that required columns exist in a DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)[:40]}..."
        )


def _rq1_ensure_parent_dir(out_path: Path) -> None:
    """Create the parent directory for an output path if it does not exist."""
    out_path.parent.mkdir(parents=True, exist_ok=True)


def _rq1_safe_float_series(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to floats (coercing errors to NaN)."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def _rq1_format_currency(x: float) -> str:
    """Format a numeric value into a compact currency label for annotations."""
    if np.isnan(x):
        return "n/a"
    absx = abs(x)
    if absx >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"


def plot_top_groups_total_erosion(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str = "total_profit_erosion",
    top_n: int = 15,
    out_path: Path,
    title: Optional[str] = None,
    annotate_top_k: int = 10,
) -> Path:
    """Plot top groups ranked by total profit erosion (horizontal bar chart).

    Purpose
    -------
    Identifies which product-level groups (category, brand, department) contribute
    the greatest absolute financial impact from returns.

    Inputs
    ------
    Expects an aggregated table with at least:
    - group_col (e.g., category/brand/department)
    - value_col (default: total_profit_erosion)

    Output
    ------
    Saves a PNG image to `out_path` and returns the same path.

    Notes
    -----
    Annotation is limited to `annotate_top_k` bars to keep the figure readable.
    """
    set_plot_style()
    _rq1_validate_columns(df, [group_col, value_col], context="plot_top_groups_total_erosion")
    _rq1_ensure_parent_dir(out_path)

    df2 = df[[group_col, value_col]].copy()
    df2[value_col] = _rq1_safe_float_series(df2[value_col])
    df2 = df2.dropna(subset=[value_col])

    df_top = df2.sort_values(value_col, ascending=False).head(top_n).copy()
    df_top = df_top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(df_top[group_col].astype(str), df_top[value_col])

    ax.set_xlabel("Total Profit Erosion")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.set_title(title or f"Top {top_n} {group_col.replace('_', ' ').title()} by Total Profit Erosion")

    maxv = float(df_top[value_col].max()) if not df_top.empty else 0.0
    ax.set_xlim(0, maxv * 1.12 if maxv > 0 else 1.0)

    k = min(annotate_top_k, len(df_top))
    if k > 0:
        for bar in bars[-k:]:
            v = float(bar.get_width())
            ax.text(
                v + (maxv * 0.01 if maxv > 0 else 0.01),
                bar.get_y() + bar.get_height() / 2,
                _rq1_format_currency(v),
                va="center",
                ha="left",
            )

    _safe_tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_return_rate_vs_mean_erosion(
    df: pd.DataFrame,
    *,
    x_col: str = "return_rate",
    y_col: str = "avg_profit_erosion",
    label_col: str = "category",
    size_col: Optional[str] = "returned_items",
    out_path: Path,
    title: str = "Return Rate vs Mean Profit Erosion per Return",
    annotate_top_k: int = 10,
) -> Path:
    """Plot return rate vs mean erosion per return (bubble scatter).

    Purpose
    -------
    Helps interpret total erosion by separating drivers into:
    - frequency (return rate)
    - severity (mean profit erosion per return)

    Inputs
    ------
    Expects group-level columns:
    - x_col (return_rate)
    - y_col (avg_profit_erosion)
    - label_col (group label)
    - optional size_col (returned_items)

    Output
    ------
    Saves a PNG image to `out_path` and returns the same path.

    Notes
    -----
    Labels only the top `annotate_top_k` groups by total_profit_erosion if present,
    otherwise by y_col.
    """
    set_plot_style()
    req = [x_col, y_col, label_col]
    if size_col is not None:
        req.append(size_col)

    _rq1_validate_columns(df, req, context="plot_return_rate_vs_mean_erosion")
    _rq1_ensure_parent_dir(out_path)

    d = df[req].copy()
    d[x_col] = _rq1_safe_float_series(d[x_col])
    d[y_col] = _rq1_safe_float_series(d[y_col])
    if size_col is not None:
        d[size_col] = _rq1_safe_float_series(d[size_col]).clip(lower=0)

    d = d.dropna(subset=[x_col, y_col])

    if size_col is None or size_col not in d.columns or d[size_col].max() <= 0:
        sizes = np.full(len(d), 140.0)
    else:
        s_raw = d[size_col].to_numpy(dtype=float)
        s_norm = s_raw / (np.max(s_raw) if np.max(s_raw) > 0 else 1.0)
        sizes = s_norm * 1200 + 60

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(d[x_col], d[y_col], s=sizes, alpha=0.65)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)

    rank_col = "total_profit_erosion" if "total_profit_erosion" in df.columns else y_col
    if rank_col in df.columns:
        d_rank = df[[label_col, rank_col]].copy()
        d_rank[rank_col] = _rq1_safe_float_series(d_rank[rank_col])
        top_labels = (
            d_rank.dropna()
            .sort_values(rank_col, ascending=False)
            .head(min(annotate_top_k, len(d_rank)))[label_col]
            .astype(str)
            .tolist()
        )
    else:
        top_labels = (
            d.sort_values(y_col, ascending=False)
            .head(min(annotate_top_k, len(d)))[label_col]
            .astype(str)
            .tolist()
        )

    d[label_col] = d[label_col].astype(str)
    for _, r in d.iterrows():
        if str(r[label_col]) in top_labels:
            ax.annotate(
                str(r[label_col]),
                (float(r[x_col]), float(r[y_col])),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

    _safe_tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_severity_vs_volume_decomposition(
    df: pd.DataFrame,
    *,
    group_col: str,
    returned_items_col: str = "returned_items",
    avg_erosion_col: str = "avg_profit_erosion",
    total_erosion_col: str = "total_profit_erosion",
    out_path: Path,
    title: str = "Severity vs Volume Decomposition (Total Erosion = Volume × Severity)",
    annotate_top_k: int = 10,
) -> Path:
    """Plot the volume-by-severity decomposition of total erosion.

    Identity
    --------
    total_profit_erosion = returned_items * avg_profit_erosion

    Inputs
    ------
    Expects group-level columns:
    - returned_items_col
    - avg_erosion_col
    - total_erosion_col

    Output
    ------
    Saves a PNG image to `out_path` and returns the same path.
    """
    set_plot_style()
    _rq1_validate_columns(
        df,
        [group_col, returned_items_col, avg_erosion_col, total_erosion_col],
        context="plot_severity_vs_volume_decomposition",
    )
    _rq1_ensure_parent_dir(out_path)

    df2 = df[[group_col, returned_items_col, avg_erosion_col, total_erosion_col]].copy()
    df2[returned_items_col] = _rq1_safe_float_series(df2[returned_items_col]).clip(lower=0)
    df2[avg_erosion_col] = _rq1_safe_float_series(df2[avg_erosion_col]).clip(lower=0)
    df2[total_erosion_col] = _rq1_safe_float_series(df2[total_erosion_col]).clip(lower=0)
    df2 = df2.dropna()

    s_raw = df2[total_erosion_col]
    s = (s_raw / (s_raw.max() if s_raw.max() > 0 else 1.0)) * 1400 + 40

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(df2[returned_items_col], df2[avg_erosion_col], s=s, alpha=0.60)

    ax.set_xlabel("Returned Items (Volume)")
    ax.set_ylabel("Average Profit Erosion per Return (Severity)")
    ax.set_title(title)

    k = min(annotate_top_k, len(df2))
    top = df2.sort_values(total_erosion_col, ascending=False).head(k)
    for _, r in top.iterrows():
        ax.annotate(
            str(r[group_col]),
            (float(r[returned_items_col]), float(r[avg_erosion_col])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    _safe_tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_profit_erosion_distribution_log(
    returned_df: pd.DataFrame,
    *,
    value_col: str = "profit_erosion",
    out_path: Path,
    title: str = "Distribution of Profit Erosion for Returned Items (Log Scale)",
    bins: int = 60,
) -> Path:
    """Plot the distribution of item-level profit erosion on a log-scaled x-axis.

    Purpose
    -------
    Demonstrates skewness and heavy tails in item-level erosion, providing a visual
    justification for non-parametric tests in RQ1.

    Input
    -----
    Expects an item-level returned-items dataset with `value_col`.

    Output
    ------
    Saves a PNG image to `out_path` and returns the same path.

    Raises
    ------
    ValueError if no positive values exist (required for log scaling).
    """
    set_plot_style()
    _rq1_validate_columns(returned_df, [value_col], context="plot_profit_erosion_distribution_log")
    _rq1_ensure_parent_dir(out_path)

    x = _rq1_safe_float_series(returned_df[value_col]).dropna()
    x = x[x > 0]
    if x.empty:
        raise ValueError("No positive values available for log-scale plot.")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.hist(x, bins=bins, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(f"{value_col} (log scale)")
    ax.set_ylabel("Count of returned items")
    ax.set_title(title)

    _safe_tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bootstrap_ci_mean_by_group(
    returned_df: pd.DataFrame,
    *,
    group_col: str = "category",
    value_col: str = "profit_erosion",
    out_path: Path,
    title: str = "Bootstrap 95% CI for Mean Profit Erosion (Top Groups)",
    n_boot: int = 800,
    min_group_size: int = 30,
    top_n_plot: int = 15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Path]:
    """Compute and plot bootstrap 95% confidence intervals for mean profit erosion.

    Purpose
    -------
    Adds uncertainty quantification to RQ1 by estimating confidence intervals for
    group mean erosion via bootstrap resampling.

    Input
    -----
    Item-level returned-items dataset with:
    - group_col (e.g., category, brand)
    - value_col (profit_erosion)

    Output
    ------
    Returns a DataFrame of all eligible groups (including CI bounds) and saves a PNG
    figure to `out_path`.

    Raises
    ------
    ValueError if no groups meet `min_group_size`.
    """
    set_plot_style()
    _rq1_validate_columns(returned_df, [group_col, value_col], context="plot_bootstrap_ci_mean_by_group")
    _rq1_ensure_parent_dir(out_path)

    df = returned_df[[group_col, value_col]].dropna().copy()
    df[value_col] = _rq1_safe_float_series(df[value_col])

    counts = df[group_col].value_counts(dropna=False)
    keep = counts[counts >= min_group_size].index
    df = df[df[group_col].isin(keep)].copy()
    if df.empty:
        raise ValueError(f"No groups meet min_group_size={min_group_size}.")

    rng = np.random.default_rng(random_state)

    rows = []
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].to_numpy()
        n = len(vals)

        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            boot_means[i] = float(np.mean(sample))

        mean_hat = float(np.mean(vals))
        ci_low = float(np.quantile(boot_means, 0.025))
        ci_high = float(np.quantile(boot_means, 0.975))

        rows.append(
            {
                group_col: g,
                "n_returned_items": int(n),
                "mean_profit_erosion": mean_hat,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "ci_width": ci_high - ci_low,
            }
        )

    ci_df = pd.DataFrame(rows).sort_values("mean_profit_erosion", ascending=False).reset_index(drop=True)

    plot_df = ci_df.head(top_n_plot).sort_values("mean_profit_erosion", ascending=True).copy()
    y = np.arange(len(plot_df))
    means = plot_df["mean_profit_erosion"].to_numpy()
    err_low = means - plot_df["ci_low_95"].to_numpy()
    err_high = plot_df["ci_high_95"].to_numpy() - means

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.errorbar(means, y, xerr=[err_low, err_high], fmt="o", capsize=4, linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[group_col].astype(str))
    ax.set_xlabel("Mean Profit Erosion per Returned Item")
    ax.set_title(title)

    for yi, mean, n in zip(y, means, plot_df["n_returned_items"].to_numpy()):
        ax.text(mean, yi, f"  n={int(n)}", va="center", ha="left", fontsize=9)

    _safe_tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return ci_df, out_path
