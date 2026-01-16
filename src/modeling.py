"""
Modeling module for the Profit Erosion E-commerce Capstone Project.

This module provides analytical functions for profit erosion modeling,
customer segmentation, and return behavior analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from src.config import MIN_ROWS_THRESHOLD


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
            item_rows=("order_item_id", "size"),
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
            returned_items=("order_item_id", "count"),
            total_lost_sales=("sale_price", "sum"),
            total_lost_margin=("item_margin", "sum"),
            median_margin_per_return=("item_margin", "median"),
            avg_margin_per_return=("item_margin", "mean"),
        )
        .sort_values("total_lost_margin", ascending=False)
    )
    return result


def build_customer_behavior_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level behavioral profile for return analysis.

    Args:
        df: DataFrame with order item data.

    Returns:
        DataFrame with one row per customer containing behavioral metrics.
    """
    profile = (
        df.groupby("user_id")
        .agg(
            total_items=("order_item_id", "count"),
            total_orders=("order_id", "nunique"),
            return_events=("is_returned_item", "sum"),
            total_sales=("sale_price", "sum"),
            total_margin=("item_margin", "sum"),
            avg_item_price=("sale_price", "mean"),
            avg_item_margin=("item_margin", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            delivered_items=("item_delivered_at", lambda x: x.notna().sum()),
        )
        .assign(
            return_rate=lambda x: x["return_events"] / x["total_items"],
            items_per_order=lambda x: x["total_items"] / x["total_orders"],
        )
    )
    return profile


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
            return_events=("order_item_id", "count"),
            total_lost_margin=("item_margin", "sum"),
            total_lost_sales=("sale_price", "sum"),
            median_margin_per_return=("item_margin", "median"),
            max_single_return_margin=("item_margin", "max"),
        )
        .sort_values("total_lost_margin", ascending=False)
    )
    return exposure


def estimate_return_process_cost(
    df: pd.DataFrame,
    cost_per_return: float = 15.0,
    cost_components: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Estimate return processing costs and total profit erosion.

    This function models the operational cost of processing returns,
    combining margin reversal with estimated handling costs.

    Args:
        df: DataFrame with order item data.
        cost_per_return: Flat cost per return event (default $15).
        cost_components: Optional dict with granular cost breakdown:
            - customer_care: Cost of customer service handling
            - inspection: Cost of product inspection
            - restocking: Cost of restocking/shelving
            - logistics: Reverse logistics cost

    Returns:
        DataFrame with profit erosion estimates per returned item.
    """
    if cost_components is None:
        cost_components = {
            "customer_care": 5.0,
            "inspection": 3.0,
            "restocking": 4.0,
            "logistics": 3.0,
        }

    returned_df = df.loc[df["is_returned_item"] == 1].copy()

    # Calculate process cost
    total_process_cost = sum(cost_components.values())
    returned_df["process_cost"] = total_process_cost

    # Add component breakdown
    for component, cost in cost_components.items():
        returned_df[f"cost_{component}"] = cost

    # Calculate total profit erosion
    returned_df["total_profit_erosion"] = (
        returned_df["item_margin"] + returned_df["process_cost"]
    )

    return returned_df


def summarize_profit_erosion(
    df: pd.DataFrame,
    cost_per_return: float = 15.0,
) -> Dict[str, float]:
    """
    Generate summary statistics for profit erosion analysis.

    Args:
        df: DataFrame with order item data.
        cost_per_return: Estimated cost per return event.

    Returns:
        Dictionary with profit erosion summary metrics.
    """
    returned = df.loc[df["is_returned_item"] == 1]
    total_items = len(df)
    total_returned = len(returned)

    summary = {
        "total_items": total_items,
        "total_returned": total_returned,
        "return_rate_pct": (total_returned / total_items) * 100,
        "total_margin_reversal": returned["item_margin"].sum(),
        "avg_margin_per_return": returned["item_margin"].mean(),
        "median_margin_per_return": returned["item_margin"].median(),
        "estimated_process_costs": total_returned * cost_per_return,
        "total_profit_erosion": (
            returned["item_margin"].sum() + (total_returned * cost_per_return)
        ),
        "max_single_margin_loss": returned["item_margin"].max(),
        "pct_margin_lost_to_returns": (
            returned["item_margin"].sum() / df["item_margin"].sum()
        ) * 100,
    }

    return summary


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
    profile = build_customer_behavior_profile(df)

    low_thresh, high_thresh = return_rate_thresholds

    conditions = [
        profile["return_rate"] == 0,
        profile["return_rate"] <= low_thresh,
        profile["return_rate"] <= high_thresh,
        profile["return_rate"] > high_thresh,
    ]

    labels = ["no_returns", "low_returner", "moderate_returner", "high_returner"]

    profile["return_segment"] = np.select(conditions, labels, default="unknown")

    return profile
