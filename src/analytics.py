"""
Analytics module for the Profit Erosion E-commerce Capstone Project.

This module provides analytical functions for:
- Return rate calculations by product attributes (category, brand)
- Product-level feature engineering
- Temporal feature engineering
- Customer segmentation by return behavior
- Feature quality validation

Note: Renamed from modeling.py to better reflect purpose.
ML models will be added in a separate modeling.py module later.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import MIN_ROWS_THRESHOLD
from src.feature_engineering import engineer_customer_behavioral_features


def calculate_return_rates_by_group(
    df: pd.DataFrame,
    group_cols: list,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame:
    """
    Calculate return rates aggregated by specified grouping columns.

    Args:
        df: DataFrame with order item data.
        group_cols: List of columns to group by.
        min_rows: Minimum sample size for inclusion.

    Returns:
        DataFrame with return rate statistics per group.
    """
    result = (
        df.groupby(group_cols)
        .agg(
            item_rows=("order_id", "size"),
            returned_items=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned_items"] / x["item_rows"])
        .query("item_rows >= @min_rows")
        .sort_values("return_rate", ascending=False)
    )
    return result


def calculate_margin_loss_by_group(
    df: pd.DataFrame,
    group_cols: list,
) -> pd.DataFrame:
    """
    Calculate margin loss from returns aggregated by specified grouping columns.

    Args:
        df: DataFrame with order item data including margin calculations.
        group_cols: List of columns to group by.

    Returns:
        DataFrame with margin loss statistics per group.
    """
    result = (
        df.loc[df["is_returned_item"] == 1]
        .groupby(group_cols)
        .agg(
            returned_items=("order_id", "count"),
            total_lost_sales=("sale_price", "sum"),
            total_lost_margin=("item_margin", "sum"),
            median_margin_per_return=("item_margin", "median"),
            avg_margin_per_return=("item_margin", "mean"),
        )
        .sort_values("total_lost_margin", ascending=False)
    )
    return result




def calculate_customer_margin_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate margin exposure from returns at customer level.

    Args:
        df: DataFrame with order item data.

    Returns:
        DataFrame with customer-level margin exposure metrics.
    """
    exposure = (
        df.loc[df["is_returned_item"] == 1]
        .groupby("user_id")
        .agg(
            return_events=("order_id", "count"),
            total_lost_margin=("item_margin", "sum"),
            total_lost_sales=("sale_price", "sum"),
            median_margin_per_return=("item_margin", "median"),
            max_single_return_margin=("item_margin", "max"),
        )
        .sort_values("total_lost_margin", ascending=False)
    )
    return exposure


def segment_customers_by_return_behavior(
    df: pd.DataFrame,
    return_rate_thresholds: Tuple[float, float] = (0.05, 0.15),
) -> pd.DataFrame:
    """
    Segment customers based on return behavior patterns.

    Args:
        df: DataFrame with order item data.
        return_rate_thresholds: Tuple of (low, high) thresholds for segmentation.

    Returns:
        Customer profile DataFrame with segment labels.
    """
    profile = engineer_customer_behavioral_features(df)

    low_thresh, high_thresh = return_rate_thresholds

    conditions = [
        profile["customer_return_rate"] == 0,
        profile["customer_return_rate"] <= low_thresh,
        profile["customer_return_rate"] <= high_thresh,
        profile["customer_return_rate"] > high_thresh,
    ]

    labels = ["no_returns", "low_returner", "moderate_returner", "high_returner"]

    profile["return_segment"] = np.select(conditions, labels, default="unknown")

    return profile


def calculate_price_margin_returned_by_country(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate price, margin, and cost metrics for RETURNED items aggregated by country.

    Filters to show only items with item_status='Returned', then aggregates metrics
    by country to analyze the economic impact of returns across geographical regions.

    Args:
        df: DataFrame with order item data including item_status, cost, sale_price,
            item_margin, and country columns.

    Returns:
        DataFrame with aggregated metrics per country for returned items only.
        Returns empty DataFrame if no returned items found.
    """
    # Filter for returned items only
    returned_df = df.loc[df["item_status"] == "Returned"].copy()

    if returned_df.empty:
        return pd.DataFrame()

    result = (
        returned_df.groupby("country")
        .agg(
            item_count=("order_id", "size"),
            avg_cost=("cost", "mean"),
            total_cost=("cost", "sum"),
            avg_sale_price=("sale_price", "mean"),
            total_sale_price=("sale_price", "sum"),
            avg_margin=("item_margin", "mean"),
            total_margin=("item_margin", "sum"),
            median_margin=("item_margin", "median"),
            min_margin=("item_margin", "min"),
            max_margin=("item_margin", "max"),
        )
        .round(2)
        .sort_values("total_margin", ascending=False)
    )
    return result


# =============================================================================
# US06 Task 3: Product-Level Features
# =============================================================================


def calculate_category_return_rates(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame:
    """
    Calculate return rates aggregated by product category.

    Args:
        df: DataFrame with order item data including category and is_returned_item.
        min_rows: Minimum sample size for inclusion (default from config).

    Returns:
        DataFrame with one row per category containing:
        - item_count: total items in category
        - returned_items: number of returned items
        - return_rate: returned_items / item_count
    """
    return calculate_return_rates_by_group(df, ["category"], min_rows)


def calculate_brand_return_rates(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame:
    """
    Calculate return rates aggregated by brand.

    Args:
        df: DataFrame with order item data including brand and is_returned_item.
        min_rows: Minimum sample size for inclusion (default from config).

    Returns:
        DataFrame with one row per brand containing:
        - item_count: total items for brand
        - returned_items: number of returned items
        - return_rate: returned_items / item_count
    """
    return calculate_return_rates_by_group(df, ["brand"], min_rows)


def engineer_product_level_features(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame:
    """
    Add product-level aggregated features to item-level data.

    This function enriches item-level data with aggregated product features
    useful for analysis and modeling.

    Args:
        df: Item-level DataFrame with category, brand, sale_price columns.
        min_rows: Minimum sample size for calculating group return rates.

    Returns:
        DataFrame with additional columns:
        - category_return_rate: return rate for item's category
        - brand_return_rate: return rate for item's brand
        - price_tier: 'low', 'medium', or 'high' based on sale_price quantiles

    Example:
        >>> df = engineer_product_level_features(df)
        >>> high_risk = df[df["category_return_rate"] > 0.15]
    """
    df = df.copy()

    # Calculate category return rates
    if "category" in df.columns and "is_returned_item" in df.columns:
        category_rates = calculate_category_return_rates(df, min_rows)
        if not category_rates.empty:
            # Map return rate back to item level
            df["category_return_rate"] = df["category"].map(
                category_rates["return_rate"].to_dict()
            )
        else:
            df["category_return_rate"] = np.nan

    # Calculate brand return rates
    if "brand" in df.columns and "is_returned_item" in df.columns:
        brand_rates = calculate_brand_return_rates(df, min_rows)
        if not brand_rates.empty:
            df["brand_return_rate"] = df["brand"].map(
                brand_rates["return_rate"].to_dict()
            )
        else:
            df["brand_return_rate"] = np.nan

    # Create price tiers based on quantiles
    if "sale_price" in df.columns:
        df["price_tier"] = pd.qcut(
            df["sale_price"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )

    return df


# =============================================================================
# US06 Task 4: Temporal Features
# =============================================================================


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features for analysis.

    This function extracts temporal patterns from order dates that may be
    useful for understanding return behavior patterns.

    Args:
        df: Item-level DataFrame with order_created_at, item_delivered_at columns.
            Optional: item_returned_at for return timing analysis.

    Returns:
        DataFrame with additional columns:
        - order_day_of_week: 0-6 (Monday=0)
        - order_month: 1-12
        - order_quarter: 1-4
        - order_year: year
        - days_to_delivery: days from order creation to delivery
        - days_to_return: days from delivery to return (NaN if not returned)
        - is_weekend_order: boolean
        - season: 'winter', 'spring', 'summer', 'fall'

    Example:
        >>> df = engineer_temporal_features(df)
        >>> weekend_returns = df[df["is_weekend_order"] == True]
    """
    df = df.copy()

    # Ensure datetime types
    if "order_created_at" in df.columns:
        df["order_created_at"] = pd.to_datetime(df["order_created_at"])

        # Extract date components
        df["order_day_of_week"] = df["order_created_at"].dt.dayofweek
        df["order_month"] = df["order_created_at"].dt.month
        df["order_quarter"] = df["order_created_at"].dt.quarter
        df["order_year"] = df["order_created_at"].dt.year

        # Weekend indicator
        df["is_weekend_order"] = df["order_day_of_week"].isin([5, 6])

        # Season mapping (Northern Hemisphere)
        season_map = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "fall", 10: "fall", 11: "fall",
        }
        df["season"] = df["order_month"].map(season_map)

    # Days to delivery
    if "order_created_at" in df.columns and "item_delivered_at" in df.columns:
        df["item_delivered_at"] = pd.to_datetime(df["item_delivered_at"])
        df["days_to_delivery"] = (
            df["item_delivered_at"] - df["order_created_at"]
        ).dt.days

    # Days to return (only for returned items)
    if "item_delivered_at" in df.columns and "item_returned_at" in df.columns:
        df["item_returned_at"] = pd.to_datetime(df["item_returned_at"])
        df["days_to_return"] = (
            df["item_returned_at"] - df["item_delivered_at"]
        ).dt.days

    return df


# =============================================================================
# US06 Task 6: Feature Quality Validation
# =============================================================================


def validate_feature_quality(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    vif_threshold: float = 5.0,
) -> Dict[str, Any]:
    """
    Validate engineered features for quality issues.

    This function performs comprehensive quality checks on engineered features
    to ensure they are suitable for downstream analysis and modeling.

    Args:
        df: DataFrame with engineered features.
        feature_cols: List of columns to validate. If None, uses all numeric columns.
        vif_threshold: Threshold for flagging multicollinearity (default 5.0).
            VIF > 5 suggests moderate multicollinearity, VIF > 10 is severe.

    Returns:
        Dict with validation report containing:
        - missing_values: {col: count} for columns with missing values
        - missing_pct: {col: pct} percentage missing for each column
        - distribution_stats: {col: {mean, std, min, max, skew, kurtosis}}
        - correlation_matrix: correlation matrix for numeric features
        - high_correlations: list of column pairs with |corr| > 0.8
        - constant_columns: columns with zero variance
        - total_rows: number of rows in dataset
        - total_features: number of features analyzed

    Example:
        >>> report = validate_feature_quality(df)
        >>> print(f"Missing values: {report['missing_values']}")
        >>> print(f"High correlations: {report['high_correlations']}")
    """
    # Select features to validate
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    numeric_df = df[feature_cols].select_dtypes(include=[np.number])

    report: Dict[str, Any] = {
        "total_rows": len(df),
        "total_features": len(feature_cols),
    }

    # Missing values analysis
    missing_counts = df[feature_cols].isna().sum()
    report["missing_values"] = missing_counts[missing_counts > 0].to_dict()
    report["missing_pct"] = {
        col: (count / len(df)) * 100
        for col, count in report["missing_values"].items()
    }

    # Distribution statistics
    distribution_stats = {}
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()
        if len(col_data) > 0:
            distribution_stats[col] = {
                "mean": col_data.mean(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "median": col_data.median(),
                "skew": col_data.skew(),
                "kurtosis": col_data.kurtosis(),
            }
    report["distribution_stats"] = distribution_stats

    # Correlation matrix
    if len(numeric_df.columns) > 1:
        report["correlation_matrix"] = numeric_df.corr()

        # Find high correlations (|corr| > 0.8)
        high_correlations = []
        corr_matrix = report["correlation_matrix"]
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle only
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.8:
                        high_correlations.append({
                            "col1": col1,
                            "col2": col2,
                            "correlation": corr_val,
                        })
        report["high_correlations"] = high_correlations
    else:
        report["correlation_matrix"] = None
        report["high_correlations"] = []

    # Constant columns (zero variance)
    variances = numeric_df.var()
    report["constant_columns"] = variances[variances == 0].index.tolist()

    # Columns with very low variance (potential issues)
    report["low_variance_columns"] = variances[
        (variances > 0) & (variances < 0.01)
    ].index.tolist()

    return report


def generate_feature_quality_report(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a human-readable feature quality report.

    This function calls validate_feature_quality() and formats the results
    into a readable text report suitable for documentation.

    Args:
        df: DataFrame with engineered features.
        feature_cols: List of columns to validate. If None, uses all numeric columns.
        output_path: Optional path to save report. If None, just returns string.

    Returns:
        String containing formatted quality report.

    Example:
        >>> report = generate_feature_quality_report(df)
        >>> print(report)
        >>> # Or save to file
        >>> generate_feature_quality_report(df, output_path="reports/quality_report.txt")
    """
    validation = validate_feature_quality(df, feature_cols)

    lines = [
        "=" * 60,
        "FEATURE QUALITY VALIDATION REPORT",
        "=" * 60,
        "",
        f"Dataset Size: {validation['total_rows']:,} rows",
        f"Features Analyzed: {validation['total_features']} columns",
        "",
        "-" * 40,
        "MISSING VALUES",
        "-" * 40,
    ]

    if validation["missing_values"]:
        for col, count in validation["missing_values"].items():
            pct = validation["missing_pct"][col]
            lines.append(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        lines.append("  No missing values found.")

    lines.extend([
        "",
        "-" * 40,
        "DISTRIBUTION SUMMARY",
        "-" * 40,
    ])

    if validation["distribution_stats"]:
        for col, stats in list(validation["distribution_stats"].items())[:10]:
            lines.append(f"  {col}:")
            lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            lines.append(f"    Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            lines.append(f"    Skew: {stats['skew']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")
        if len(validation["distribution_stats"]) > 10:
            lines.append(f"  ... and {len(validation['distribution_stats']) - 10} more columns")

    lines.extend([
        "",
        "-" * 40,
        "HIGH CORRELATIONS (|r| > 0.8)",
        "-" * 40,
    ])

    if validation["high_correlations"]:
        for item in validation["high_correlations"]:
            lines.append(
                f"  {item['col1']} <-> {item['col2']}: {item['correlation']:.4f}"
            )
    else:
        lines.append("  No high correlations found.")

    lines.extend([
        "",
        "-" * 40,
        "POTENTIAL ISSUES",
        "-" * 40,
    ])

    if validation["constant_columns"]:
        lines.append(f"  Constant columns (zero variance): {validation['constant_columns']}")
    if validation["low_variance_columns"]:
        lines.append(f"  Low variance columns: {validation['low_variance_columns']}")
    if not validation["constant_columns"] and not validation["low_variance_columns"]:
        lines.append("  No variance issues detected.")

    lines.extend([
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report_text)

    return report_text
