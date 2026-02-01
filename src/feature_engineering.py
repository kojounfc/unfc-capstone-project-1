"""
Feature engineering module for the Profit Erosion E-commerce Capstone Project.

This module provides feature engineering functions for:
- Return-related flags and indicators
- Margin calculations
- Profit erosion metrics (margin reversal + processing costs)
- Customer behavioral features (RFM-style metrics)
- Target variables for predictive modeling

For the TheLook e-commerce dataset from BigQuery.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


def engineer_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create return-related flags and indicators.

    Args:
        df: Merged DataFrame.

    Returns:
        DataFrame with additional return feature columns.
    """
    df = df.copy()

    # Return flags
    df["is_returned_item"] = (df["item_status"].str.lower() == "returned").astype(int)
    df["is_returned_order"] = (df["order_status"].str.lower() == "returned").astype(int)

    return df


def calculate_margins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate margin and discount metrics for profit erosion analysis.

    Args:
        df: Merged DataFrame with price and cost columns.

    Returns:
        DataFrame with margin and discount columns added.
    """
    df = df.copy()

    # Ensure numeric types
    df["retail_price"] = pd.to_numeric(df["retail_price"], errors="coerce")
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    # Discount metrics
    df["discount_amount"] = df["retail_price"] - df["sale_price"]
    df["discount_pct"] = np.where(
        df["retail_price"] > 0,
        df["discount_amount"] / df["retail_price"],
        np.nan,
    )

    # Margin metrics
    df["item_margin"] = df["sale_price"] - df["cost"]
    df["item_margin_pct"] = np.where(
        df["sale_price"] > 0,
        df["item_margin"] / df["sale_price"],
        np.nan,
    )

    return df


# =============================================================================
# US06 Task 1: Profit Erosion Metrics
# =============================================================================
# See docs/PROCESSING_COST_METHODOLOGY.md for complete analysis and rationale.
#
# Design Decision: Functions receive pre-filtered returned items only
# (where is_returned_item == 1) for efficiency and alignment with RQs.
# =============================================================================

# Base processing cost components ($12 total per return)
# Based on industry benchmarks: Rogers & Tibben-Lembke (2001), Guide & Van Wassenhove (2009)
DEFAULT_COST_COMPONENTS: Dict[str, float] = {
    "customer_care": 4.0,   # Phone/email support time for return request processing
    "inspection": 2.5,      # Quality assessment upon receipt in returns center
    "restocking": 3.0,      # Shelving in warehouse, inventory system updates
    "logistics": 2.5,       # Return label generation, carrier coordination, administrative processing
}

# Category tier multipliers based on margin-at-risk analysis of 18,208 returned items
# Margin CV = 59.4% across categories justifies tiered approach
# See docs/PROCESSING_COST_METHODOLOGY.md Section 5 for complete mapping
CATEGORY_TIER_MULTIPLIERS: Dict[str, float] = {
    # Premium Tier (1.3x) - High-margin items, avg margin $52.25
    "Outerwear & Coats": 1.3,
    "Suits & Sport Coats": 1.3,
    "Blazers & Jackets": 1.3,
    "Jeans": 1.3,
    "Dresses": 1.3,
    "Suits": 1.3,
    "Sweaters": 1.3,
    "Pants": 1.3,
    # Moderate Tier (1.15x) - Mid-margin items, avg margin $27.49
    "Skirts": 1.15,
    "Active": 1.15,
    "Swim": 1.15,
    "Maternity": 1.15,
    "Sleep & Lounge": 1.15,
    "Accessories": 1.15,
    "Pants & Capris": 1.15,
    "Fashion Hoodies & Sweatshirts": 1.15,
    "Shorts": 1.15,
    # Standard Tier (1.0x) - Lower-margin items, avg margin $15.25
    "Plus": 1.0,
    "Tops & Tees": 1.0,
    "Intimates": 1.0,
    "Underwear": 1.0,
    "Leggings": 1.0,
    "Socks & Hosiery": 1.0,
    "Socks": 1.0,
    "Jumpsuits & Rompers": 1.0,
    "Clothing Sets": 1.0,
}

# Default multiplier for categories not in the mapping
DEFAULT_CATEGORY_MULTIPLIER: float = 1.0


def calculate_profit_erosion(
    df: pd.DataFrame,
    cost_components: Optional[Dict[str, float]] = None,
    category_multipliers: Optional[Dict[str, float]] = None,
    use_category_tiers: bool = True,
) -> pd.DataFrame:
    """
    Calculate profit erosion metrics for returned items.

    Profit erosion = margin_reversal + process_cost
    Where process_cost = base_cost × category_multiplier (if use_category_tiers=True)

    Args:
        df: DataFrame containing ONLY returned items (pre-filtered where
            is_returned_item == 1). Must have item_margin and category columns.
        cost_components: Base cost breakdown. Default $12 total:
            - customer_care: $4.00
            - inspection: $2.50
            - restocking: $3.00
            - logistics: $2.50
        category_multipliers: Category-to-multiplier mapping. Default uses
            CATEGORY_TIER_MULTIPLIERS (Premium=1.3x, Moderate=1.15x, Standard=1.0x).
        use_category_tiers: If True, apply category multipliers to base cost.
            If False, use flat base cost for all categories.

    Returns:
        DataFrame with columns added:
        - margin_reversal: item_margin (the margin lost on this return)
        - process_cost: base_cost × category_multiplier
        - profit_erosion: margin_reversal + process_cost

    Example:
        >>> returned_df = df[df["is_returned_item"] == 1].copy()
        >>> returned_df = calculate_profit_erosion(returned_df)
    """
    df = df.copy()

    if cost_components is None:
        cost_components = DEFAULT_COST_COMPONENTS

    if category_multipliers is None:
        category_multipliers = CATEGORY_TIER_MULTIPLIERS

    base_cost = sum(cost_components.values())

    # Margin reversal: the margin lost on each returned item
    df["margin_reversal"] = df["item_margin"]

    # Processing cost: base_cost × category_multiplier
    if use_category_tiers and "category" in df.columns:
        # Normalize category to title case for tier lookup
        # (CATEGORY_TIER_MULTIPLIERS uses title case keys like "Outerwear & Coats")
        df["_category_normalized"] = df["category"].str.title()
        df["category_multiplier"] = df["_category_normalized"].map(
            category_multipliers
        ).fillna(DEFAULT_CATEGORY_MULTIPLIER)
        df["process_cost"] = base_cost * df["category_multiplier"]
        df = df.drop(columns=["_category_normalized", "category_multiplier"])
    else:
        df["process_cost"] = base_cost

    # Total profit erosion per item
    df["profit_erosion"] = df["margin_reversal"] + df["process_cost"]

    return df


def aggregate_profit_erosion_by_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate profit erosion metrics to order level.

    Args:
        df: DataFrame of returned items with profit erosion columns
            (output from calculate_profit_erosion).

    Returns:
        DataFrame with one row per order containing:
        - user_id: Customer identifier
        - returned_items: Number of returned items in this order
        - total_margin_reversal: Sum of margin_reversal
        - total_process_cost: Sum of process_cost
        - total_profit_erosion: Sum of profit_erosion
        - avg_margin_per_return: Average margin per returned item
    """
    result = (
        df.groupby("order_id")
        .agg(
            user_id=("user_id", "first"),
            returned_items=("order_item_id", "count"),
            total_margin_reversal=("margin_reversal", "sum"),
            total_process_cost=("process_cost", "sum"),
            total_profit_erosion=("profit_erosion", "sum"),
            total_sales=("sale_price", "sum"),
            total_margin=("item_margin", "sum"),
        )
        .assign(
            avg_margin_per_return=lambda x: x["total_margin"] / x["returned_items"],
        )
        .reset_index()
    )
    return result


def aggregate_profit_erosion_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate profit erosion metrics to customer level.

    Args:
        df: DataFrame of returned items with profit erosion columns
            (output from calculate_profit_erosion).

    Returns:
        DataFrame with one row per customer containing:
        - total_orders: Number of unique orders with returns
        - returned_items: Total returned items
        - total_margin_reversal: Sum of margin_reversal
        - total_process_cost: Sum of process_cost
        - total_profit_erosion: Sum of profit_erosion
        - total_sales: Sum of sale_price (for returned items)
        - total_margin: Sum of item_margin (margin lost)
        - avg_erosion_per_return: Average erosion per returned item
    """
    result = (
        df.groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),
            returned_items=("order_item_id", "count"),
            total_margin_reversal=("margin_reversal", "sum"),
            total_process_cost=("process_cost", "sum"),
            total_profit_erosion=("profit_erosion", "sum"),
            total_sales=("sale_price", "sum"),
            total_margin=("item_margin", "sum"),
        )
        .assign(
            avg_erosion_per_return=lambda x: x["total_profit_erosion"] / x["returned_items"],
        )
        .reset_index()
    )
    return result


def summarize_profit_erosion(
    df: pd.DataFrame,
    cost_components: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Generate summary statistics for profit erosion analysis.

    Args:
        df: DataFrame of returned items with profit erosion columns
            (output from calculate_profit_erosion).
        cost_components: Base cost breakdown for reference (default $12 total).

    Returns:
        Dictionary with profit erosion summary metrics:
        - total_returned: Number of returned items
        - total_margin_reversal: Sum of margins lost on returns
        - avg_margin_per_return: Average margin per returned item
        - median_margin_per_return: Median margin per returned item
        - total_process_costs: Sum of processing costs
        - total_profit_erosion: Total economic loss (margin + processing)
        - max_single_margin_loss: Largest single margin loss
        - avg_erosion_per_return: Average profit erosion per return
    """
    if cost_components is None:
        cost_components = DEFAULT_COST_COMPONENTS

    total_returned = len(df)

    # Handle edge case of empty DataFrame
    if total_returned == 0:
        return {
            "total_returned": 0,
            "total_margin_reversal": 0.0,
            "avg_margin_per_return": 0.0,
            "median_margin_per_return": 0.0,
            "total_process_costs": 0.0,
            "total_profit_erosion": 0.0,
            "max_single_margin_loss": 0.0,
            "avg_erosion_per_return": 0.0,
        }

    # Use pre-calculated columns if available, otherwise calculate
    if "margin_reversal" in df.columns:
        total_margin_reversal = df["margin_reversal"].sum()
    else:
        total_margin_reversal = df["item_margin"].sum()

    if "process_cost" in df.columns:
        total_process_costs = df["process_cost"].sum()
    else:
        base_cost = sum(cost_components.values())
        total_process_costs = total_returned * base_cost

    if "profit_erosion" in df.columns:
        total_profit_erosion = df["profit_erosion"].sum()
    else:
        total_profit_erosion = total_margin_reversal + total_process_costs

    summary = {
        "total_returned": total_returned,
        "total_margin_reversal": total_margin_reversal,
        "avg_margin_per_return": df["item_margin"].mean(),
        "median_margin_per_return": df["item_margin"].median(),
        "total_process_costs": total_process_costs,
        "total_profit_erosion": total_profit_erosion,
        "max_single_margin_loss": df["item_margin"].max(),
        "avg_erosion_per_return": total_profit_erosion / total_returned,
    }

    return summary


# =============================================================================
# US06 Task 2 & 4: Customer Behavioral & Temporal Features
# =============================================================================


def engineer_customer_behavioral_features(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Create RFM-style (Recency, Frequency, Monetary - a classic customer segmentation framework from marketing analytics) customer behavioral features.

    This function takes item-level data and aggregates to customer level,
    creating features useful for customer segmentation and predictive modeling.

    Args:
        df: Item-level DataFrame with order and customer data.
            Required columns: user_id, order_id, order_item_id, sale_price,
            item_margin, is_returned_item, order_created_at, user_created_at
        reference_date: Date for calculating recency metrics.
            Defaults to max order_created_at in the data for reproducibility.

    Returns:
        DataFrame with one row per customer containing:
        - order_frequency: total unique orders per customer
        - return_frequency: total return events per customer
        - customer_return_rate: return_frequency / total_items
        - avg_basket_size: avg items per order
        - avg_order_value: avg total order value
        - customer_tenure_days: days since user_created_at
        - purchase_recency_days: days since last order
        - total_items: total items purchased
        - total_sales: sum of sale_price
        - total_margin: sum of item_margin
        - avg_item_price: average sale_price per item
        - avg_item_margin: average item_margin per item

    Example:
        >>> customer_features = engineer_customer_behavioral_features(df)
        >>> high_returners = customer_features[customer_features["customer_return_rate"] > 0.2]
    """
    # Set reference date for recency calculations
    if reference_date is None:
        reference_date = pd.to_datetime(df["order_created_at"]).max()
    else:
        reference_date = pd.to_datetime(reference_date)

    # Ensure datetime columns are properly typed
    df = df.copy()
    df["order_created_at"] = pd.to_datetime(df["order_created_at"])
    df["user_created_at"] = pd.to_datetime(df["user_created_at"])

    # Calculate order-level metrics first (for avg_order_value)
    order_totals = (
        df.groupby(["user_id", "order_id"])
        .agg(
            order_total=("sale_price", "sum"),
            order_items=("order_item_id", "count"),
        )
        .reset_index()
    )

    order_level = (
        order_totals.groupby("user_id")
        .agg(
            avg_order_value=("order_total", "mean"),
        )
    )

    # Calculate customer-level metrics from item-level data
    customer_metrics = (
        df.groupby("user_id")
        .agg(
            # Frequency metrics
            total_items=("order_item_id", "count"),
            order_frequency=("order_id", "nunique"),
            return_frequency=("is_returned_item", "sum"),
            # Value metrics
            total_sales=("sale_price", "sum"),
            total_margin=("item_margin", "sum"),
            avg_item_price=("sale_price", "mean"),
            avg_item_margin=("item_margin", "mean"),
            # Temporal metrics (will be transformed)
            last_order_date=("order_created_at", "max"),
            first_order_date=("order_created_at", "min"),
            user_created_at=("user_created_at", "first"),
        )
    )

    # Calculate derived metrics
    customer_metrics["customer_return_rate"] = (
        customer_metrics["return_frequency"] / customer_metrics["total_items"]
    )
    customer_metrics["avg_basket_size"] = (
        customer_metrics["total_items"] / customer_metrics["order_frequency"]
    )

    # Calculate temporal features
    customer_metrics["purchase_recency_days"] = (
        (reference_date - customer_metrics["last_order_date"]).dt.days
    )
    customer_metrics["customer_tenure_days"] = (
        (reference_date - customer_metrics["user_created_at"]).dt.days
    )

    # Join order-level metrics
    customer_metrics = customer_metrics.join(order_level)

    # Drop intermediate date columns
    customer_metrics = customer_metrics.drop(
        columns=["last_order_date", "first_order_date", "user_created_at"]
    )

    return customer_metrics.reset_index()


# =============================================================================
# Save Feature-Engineered Dataset
# =============================================================================


def save_feature_engineered_dataset(
    df: pd.DataFrame,
    filename: str = "feature_engineered_dataset",
    output_dir: Optional[Path] = None,
    save_csv: bool = True,
    save_parquet: bool = True,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save the feature-engineered dataset to the processed data folder.

    Saves the DataFrame containing all engineered features to both parquet
    and CSV formats for downstream analysis and modeling.

    Args:
        df: Feature-engineered DataFrame to save.
        filename: Base filename without extension. Default "feature_engineered_dataset".
        output_dir: Output directory. Defaults to PROCESSED_DATA_DIR from config.
        save_csv: Whether to save CSV format. Default True.
        save_parquet: Whether to save parquet format. Default True.

    Returns:
        Tuple of (parquet_path, csv_path). Path is None if that format was not saved.

    Example:
        >>> # After running full feature engineering pipeline
        >>> df = engineer_return_features(merged_df)
        >>> df = calculate_margins(df)
        >>> parquet_path, csv_path = save_feature_engineered_dataset(df)
        >>> print(f"Saved to {parquet_path}")
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = None
    csv_path = None

    if save_parquet:
        parquet_path = output_dir / f"{filename}.parquet"
        df.to_parquet(parquet_path, index=False)

    if save_csv:
        csv_path = output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


# =============================================================================
# US06 Task 5: Target Variables for Predictive Modeling
# =============================================================================


def create_profit_erosion_targets(
    customer_df: pd.DataFrame,
    high_erosion_percentile: float = 0.75,
    erosion_column: str = "total_profit_erosion",
) -> pd.DataFrame:
    """
    Create target variables for predictive modeling of profit erosion.

    This function takes customer-level data (output from
    aggregate_profit_erosion_by_customer) and creates target variables for:
    - Binary classification (RQ3): Predict high vs. low erosion customers
    - Regression (RQ4): Predict continuous profit erosion amount

    Threshold Selection Methodology:
    - Default uses 75th percentile to identify "high erosion" customers
    - This threshold balances class imbalance (75/25 split) while capturing
      the most impactful customers for intervention
    - Percentile-based thresholds adapt to data distribution and are
      reproducible across different datasets

    Args:
        customer_df: Customer-level DataFrame with aggregated profit erosion.
            Must contain the column specified by erosion_column.
            Typically output from aggregate_profit_erosion_by_customer().
        high_erosion_percentile: Percentile threshold for binary classification.
            Default 0.75 (75th percentile). Customers above this percentile
            are flagged as "high erosion".
        erosion_column: Name of column containing profit erosion values.
            Default "total_profit_erosion".

    Returns:
        DataFrame with additional columns:
        - is_high_erosion_customer: Binary flag (1 if erosion > percentile threshold)
        - profit_erosion_quartile: 1-4 quartile assignment (4 = highest erosion)
        - erosion_percentile_rank: Percentile rank (0-100) of each customer

    Example:
        >>> returned_df = df[df["is_returned_item"] == 1].copy()
        >>> returned_df = calculate_profit_erosion(returned_df)
        >>> customer_erosion = aggregate_profit_erosion_by_customer(returned_df)
        >>> customer_targets = create_profit_erosion_targets(customer_erosion)
        >>> high_risk = customer_targets[customer_targets["is_high_erosion_customer"] == 1]

    Raises:
        ValueError: If erosion_column is not in the DataFrame.
        ValueError: If high_erosion_percentile is not between 0 and 1.
    """
    if erosion_column not in customer_df.columns:
        raise ValueError(
            f"Column '{erosion_column}' not found in DataFrame. "
            f"Available columns: {list(customer_df.columns)}"
        )

    if not 0 <= high_erosion_percentile <= 1:
        raise ValueError(
            f"high_erosion_percentile must be between 0 and 1, got {high_erosion_percentile}"
        )

    df = customer_df.copy()

    # Calculate percentile rank (0-100) for each customer
    df["erosion_percentile_rank"] = df[erosion_column].rank(pct=True) * 100

    # Binary classification target: 1 if above threshold percentile
    threshold_value = df[erosion_column].quantile(high_erosion_percentile)
    df["is_high_erosion_customer"] = (df[erosion_column] > threshold_value).astype(int)

    # Quartile assignment (1 = lowest erosion, 4 = highest erosion)
    # Handle edge case where duplicates reduce number of bins
    try:
        quartiles = pd.qcut(
            df[erosion_column],
            q=4,
            labels=False,
            duplicates="drop",
        )
        # Map to 1-4 labels (qcut returns 0-indexed, and may have fewer bins)
        n_bins = quartiles.nunique()
        if n_bins == 4:
            df["profit_erosion_quartile"] = quartiles + 1
        else:
            # Fewer than 4 unique bins due to duplicates - map available bins to 1-4
            df["profit_erosion_quartile"] = quartiles.map(
                {i: i + 1 for i in range(n_bins)}
            )
    except ValueError:
        # Fallback: use percentile rank to assign quartiles
        df["profit_erosion_quartile"] = pd.cut(
            df["erosion_percentile_rank"],
            bins=[0, 25, 50, 75, 100],
            labels=[1, 2, 3, 4],
            include_lowest=True,
        )

    df["profit_erosion_quartile"] = df["profit_erosion_quartile"].astype(int)

    return df
