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
    sns.set_theme(style="whitegrid")  # use theme, not set_style

    # Force font AFTER seaborn sets theme (seaborn can override rcParams)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.figsize": (10, 6),
        "font.size": 11,
    })



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
