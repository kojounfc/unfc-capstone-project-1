"""
Data processing module for the Profit Erosion E-commerce Capstone Project.

This module handles data loading, cleaning, merging, and feature engineering
for the TheLook e-commerce dataset from BigQuery.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_FILES,
    DATETIME_COLS,
    NUMERIC_COLS,
    STRING_COLS,
)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by stripping whitespace and removing BOM characters.

    Args:
        df: Input DataFrame with potentially messy column names.

    Returns:
        DataFrame with cleaned column names.
    """
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )
    return df


def load_raw_data(
    raw_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV files from the data/raw directory.

    Args:
        raw_dir: Optional path to raw data directory. Defaults to config RAW_DATA_DIR.

    Returns:
        Tuple of (order_items, orders, products, users) DataFrames.

    Raises:
        FileNotFoundError: If any required CSV file is missing.
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    order_items = pd.read_csv(
        raw_dir / "order_items.csv",
        parse_dates=["created_at", "shipped_at", "delivered_at", "returned_at"],
    )
    orders = pd.read_csv(
        raw_dir / "orders.csv",
        parse_dates=["created_at", "shipped_at", "delivered_at", "returned_at"],
    )
    products = pd.read_csv(raw_dir / "products.csv")
    users = pd.read_csv(
        raw_dir / "users.csv",
        parse_dates=["created_at"],
        low_memory=False,
    )

    # Clean column names
    order_items = clean_columns(order_items)
    orders = clean_columns(orders)
    products = clean_columns(products)
    users = clean_columns(users)

    return order_items, orders, products, users


def merge_datasets(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the four source tables into a single order-item grain DataFrame.

    Args:
        order_items: Order items table.
        orders: Orders table.
        products: Products table.
        users: Users table.

    Returns:
        Merged DataFrame at the order-item grain level.
    """
    # Rename overlapping columns in orders
    orders_renamed = orders.rename(columns={
        "created_at": "order_created_at",
        "shipped_at": "order_shipped_at",
        "delivered_at": "order_delivered_at",
        "returned_at": "order_returned_at",
        "status": "order_status",
        "gender": "order_gender",
    })

    # Rename users.id to avoid collision
    users_renamed = users.rename(columns={
        "id": "user_dim_id",
        "created_at": "user_created_at",
        "gender": "user_gender",
    })

    # Rename products.id to avoid collision
    products_renamed = products.rename(columns={"id": "product_dim_id"})

    # Rename order_items columns
    order_items_renamed = order_items.rename(columns={
        "id": "order_item_id",
        "created_at": "item_created_at",
        "shipped_at": "item_shipped_at",
        "delivered_at": "item_delivered_at",
        "returned_at": "item_returned_at",
        "status": "item_status",
    })

    # Merge at order-item grain
    df = (
        order_items_renamed
        .merge(
            orders_renamed[[
                "order_id", "user_id", "order_status",
                "order_created_at", "order_shipped_at", "order_delivered_at",
                "order_returned_at", "num_of_item", "order_gender",
            ]],
            on=["order_id", "user_id"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            products_renamed,
            left_on="product_id",
            right_on="product_dim_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            users_renamed,
            left_on="user_id",
            right_on="user_dim_id",
            how="left",
            validate="many_to_one",
        )
    )

    return df


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


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types for consistent processing and parquet serialization.

    Args:
        df: DataFrame with potentially mixed types.

    Returns:
        DataFrame with standardized data types.
    """
    df = df.copy()

    # Force known mixed-type columns to string
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Convert timestamps safely
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_analysis_dataset(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Execute the full data pipeline: load, merge, engineer features, and save.

    Args:
        raw_dir: Optional path to raw data directory.
        output_dir: Optional path to output directory.
        save_output: Whether to save the output files.

    Returns:
        Fully processed DataFrame ready for analysis.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    # Load raw data
    order_items, orders, products, users = load_raw_data(raw_dir)

    # Merge datasets
    df = merge_datasets(order_items, orders, products, users)

    # Engineer features
    df = engineer_return_features(df)
    df = calculate_margins(df)

    # Standardize data types
    df = standardize_dtypes(df)

    # Save output
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "returns_eda_v1.parquet", index=False, compression="snappy")
        df.to_csv(output_dir / "returns_eda_v1.csv", index=False)

    return df


def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the processed parquet file for analysis.

    Args:
        file_path: Optional path to parquet file. Defaults to config PROCESSED_PARQUET.

    Returns:
        Processed DataFrame.

    Raises:
        FileNotFoundError: If the parquet file does not exist.
    """
    if file_path is None:
        from src.config import PROCESSED_PARQUET
        file_path = PROCESSED_PARQUET

    return pd.read_parquet(file_path)


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, any]:
    """
    Generate a data quality summary report.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dictionary containing quality metrics.
    """
    return {
        "total_rows": len(df),
        "unique_order_items": df["order_item_id"].nunique(),
        "unique_orders": df["order_id"].nunique(),
        "unique_users": df["user_id"].nunique(),
        "missing_product_pct": df["product_dim_id"].isna().mean() * 100,
        "missing_user_pct": df["user_dim_id"].isna().mean() * 100,
        "return_rate": df["is_returned_item"].mean() * 100,
        "columns": list(df.columns),
    }
