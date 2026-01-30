"""
Feature engineering module for the Profit Erosion E-commerce Capstone Project.

This module provides feature engineering functions for creating return-related
flags and margin calculations for the TheLook e-commerce dataset.
"""

import numpy as np
import pandas as pd


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
