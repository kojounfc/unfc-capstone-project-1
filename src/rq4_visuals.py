"""
RQ4 Visualization Module

This module contains visualization functions for RQ4 econometric regression analysis.
It generates publication-ready figures for profit erosion analysis.

Functions:
    - plot_target_distribution: EDA plots of target variable distribution
    - plot_coefficient_forest: Forest plot of regression coefficients with CIs
    - plot_residual_diagnostics: Four-panel residual diagnostic plots
    - plot_qq_comparison: Q-Q plot comparison (linear vs log models)
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# Visualization constants
HISTOGRAM_BINS = 50
HISTOGRAM_ALPHA_OPAQUE = 0.7
GRID_ALPHA = 0.3
SCATTER_ALPHA = 0.5
SCATTER_POINT_SIZE = 10
PLOT_LINE_WIDTH = 1


def plot_target_distribution(customers: pd.DataFrame, figures_dir: Path):
    """Create EDA plots of target variable distribution.

    Generates histogram of profit erosion (linear and log-transformed scales).

    Args:
        customers: Customer data with 'total_profit_erosion' column
        figures_dir: Directory to save figure

    Returns:
        matplotlib.figure.Figure: Figure object (displays in Jupyter)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(
        customers["total_profit_erosion"], bins=HISTOGRAM_BINS, edgecolor="black", alpha=HISTOGRAM_ALPHA_OPAQUE
    )
    axes[0].set_xlabel("Total Profit Erosion ($)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Profit Erosion Among Returners")
    axes[0].grid(True, alpha=GRID_ALPHA)

    # Log-scale histogram
    axes[1].hist(
        np.log(customers["total_profit_erosion"]),
        bins=HISTOGRAM_BINS,
        edgecolor="black",
        alpha=HISTOGRAM_ALPHA_OPAQUE,
        color="orange",
    )
    axes[1].set_xlabel("Log(Total Profit Erosion)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Log-Transformed Distribution")
    axes[1].grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "rq4_target_distribution.png", dpi=300, bbox_inches="tight"
    )

    logger.info("[OK] Target distribution plot saved")
    return fig


def plot_coefficient_forest(coef_table: pd.DataFrame, figures_dir: Path):
    """Create forest plot of regression coefficients with 95% confidence intervals.

    Displays all coefficients sorted by magnitude with error bars representing
    95% confidence intervals. Red dots indicate statistically significant
    coefficients (p < 0.05), blue dots indicate non-significant.

    Args:
        coef_table: Coefficient table with columns:
            - 'Feature': Feature name
            - 'Coefficient': Regression coefficient
            - 'p-value': P-value for significance test
            - '95% CI Lower': Lower bound of 95% CI
            - '95% CI Upper': Upper bound of 95% CI
        figures_dir: Directory to save figure

    Returns:
        matplotlib.figure.Figure: Figure object (displays in Jupyter)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    coef_plot = coef_table[coef_table["Feature"] != "const"].copy()
    coef_plot = coef_plot.sort_values("Coefficient")

    y_pos = np.arange(len(coef_plot))
    ax.errorbar(
        coef_plot["Coefficient"],
        y_pos,
        xerr=[
            coef_plot["Coefficient"] - coef_plot["95% CI Lower"],
            coef_plot["95% CI Upper"] - coef_plot["Coefficient"],
        ],
        fmt="o",
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2,
    )

    # Color significant vs non-significant
    for i, (coef, p) in enumerate(zip(coef_plot["Coefficient"], coef_plot["p-value"])):
        color = "red" if p < 0.05 else "blue"
        ax.plot(coef, i, "o", markersize=8, color=color)

    # Vertical line at zero
    ax.axvline(x=0, color="black", linestyle="--", linewidth=PLOT_LINE_WIDTH, alpha=SCATTER_ALPHA)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_plot["Feature"])
    ax.set_xlabel("Coefficient (95% CI)", fontsize=12)
    ax.set_title(
        "OLS Coefficients with 95% Confidence Intervals\n"
        + "(Red = Significant at p<0.05, Blue = Not Significant)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(figures_dir / "rq4_coefficient_plot.png", dpi=300, bbox_inches="tight")
    logger.info("[OK] Coefficient forest plot saved")
    return fig



def plot_residual_diagnostics(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    fitted_values: pd.Series,
    residuals: pd.Series,
    figures_dir: Path,
):
    """Create four-panel residual diagnostic plot.

    Examines residual patterns to assess OLS assumptions:
    - Residuals vs Fitted: Checks for linearity and heteroscedasticity
    - Q-Q Plot: Checks for normality
    - Histogram: Shows distribution of residuals
    - Scale-Location: Checks for homoscedasticity

    Args:
        results: Fitted regression model results (statsmodels RegressionResults)
        fitted_values: Fitted values from regression
        residuals: Residuals from regression
        figures_dir: Directory to save figure

    Returns:
        matplotlib.figure.Figure: Figure object (displays in Jupyter)
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(fitted_values, residuals, alpha=SCATTER_ALPHA, s=SCATTER_POINT_SIZE)
    axes[0, 0].axhline(y=0, color="r", linestyle="--", linewidth=1)
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted Values")
    axes[0, 0].grid(True, alpha=GRID_ALPHA)

    # 2. Q-Q Plot
    sp_stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot (Normal Distribution)")
    axes[0, 1].grid(True, alpha=GRID_ALPHA)

    # 3. Histogram of residuals
    axes[1, 0].hist(residuals, bins=HISTOGRAM_BINS, edgecolor="black", alpha=HISTOGRAM_ALPHA_OPAQUE, density=True)
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Distribution of Residuals")
    axes[1, 0].grid(True, alpha=GRID_ALPHA)

    # 4. Scale-Location Plot
    residuals_std = residuals.std()
    if residuals_std == 0:
        raise ValueError("Residuals have zero variance and cannot be standardized.")
    standardized_residuals = residuals / residuals_std
    axes[1, 1].scatter(
        fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=SCATTER_ALPHA, s=SCATTER_POINT_SIZE
    )
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("√|Standardized Residuals|")
    axes[1, 1].set_title("Scale-Location Plot")
    axes[1, 1].grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "rq4_residual_diagnostics.png", dpi=300, bbox_inches="tight"
    )

    logger.info("[OK] Residual diagnostics plot saved")
    return fig


def plot_qq_comparison(
    residuals: pd.Series,
    residuals_log: pd.Series,
    jb_stat: float,
    jb_stat_log: float,
    figures_dir: Path,
):
    """Create side-by-side Q-Q plot comparison of linear vs log-transformed models.

    Visualizes improvement in residual normality achieved through log transformation.
    Jarque-Bera test improvement is prominently displayed in titles.

    Args:
        residuals: Residuals from linear model
        residuals_log: Residuals from log-transformed model
        jb_stat: Jarque-Bera statistic for linear model
        jb_stat_log: Jarque-Bera statistic for log model
        figures_dir: Directory to save figure

    Returns:
        matplotlib.figure.Figure: Figure object (displays in Jupyter)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear model Q-Q
    sp_stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title(
        "Linear Model: Q-Q Plot\n(Jarque-Bera = {:.0f})".format(jb_stat),
        fontweight="bold",
    )
    axes[0].grid(True, alpha=GRID_ALPHA)

    # Log model Q-Q
    sp_stats.probplot(residuals_log, dist="norm", plot=axes[1])
    improvement_factor = jb_stat / jb_stat_log
    axes[1].set_title(
        "Log Model: Q-Q Plot\n(Jarque-Bera = {:.0f}, {:.1f}x improvement)".format(
            jb_stat_log, improvement_factor
        ),
        fontweight="bold",
    )
    axes[1].grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    plt.savefig(
        figures_dir / "rq4_qq_plot_comparison.png", dpi=300, bbox_inches="tight"
    )

    logger.info("[OK] Q-Q plot comparison saved")
    logger.info("\nJarque-Bera Test Results:")
    logger.info(f"  Linear Model: {jb_stat:.2f} (p < 0.0001)")
    logger.info(f"  Log Model:    {jb_stat_log:.2f} (p < 0.0001)")
    logger.info(f"  Improvement:  {improvement_factor:.1f}x reduction")
    return fig
