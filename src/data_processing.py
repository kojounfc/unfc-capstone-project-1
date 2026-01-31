"""
Data processing module for the Profit Erosion E-commerce Capstone Project.

This module handles data loading, cleaning, merging, and type standardization
for the TheLook e-commerce dataset from BigQuery.

NOTE: This module outputs RAW merged data. For feature engineering
(return flags, margins, profit erosion), use src.feature_engineering.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.config import (
    DATETIME_COLS,
    NUMERIC_COLS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
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
        df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    )
    return df


def load_raw_data(
    raw_dir: Optional[Path] = None,
    order_items_cols: Optional[list] = None,
    orders_cols: Optional[list] = None,
    products_cols: Optional[list] = None,
    users_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV files from the data/raw directory with optional column selection.

    Args:
        raw_dir: Optional path to raw data directory. Defaults to config RAW_DATA_DIR.
        order_items_cols: Optional list of columns to load from order_items.csv. If None, loads all columns.
        orders_cols: Optional list of columns to load from orders.csv. If None, loads all columns.
        products_cols: Optional list of columns to load from products.csv. If None, loads all columns.
        users_cols: Optional list of columns to load from users.csv. If None, loads all columns.

    Returns:
        Tuple of (order_items, orders, products, users) DataFrames.

    Raises:
        FileNotFoundError: If any required CSV file is missing.
    """
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR

    order_items = pd.read_csv(
        raw_dir / "order_items.csv",
        usecols=order_items_cols,
        parse_dates=["shipped_at", "delivered_at", "returned_at"],
    )
    orders = pd.read_csv(
        raw_dir / "orders.csv",
        usecols=orders_cols,
        parse_dates=["shipped_at", "delivered_at", "returned_at"],
    )
    products = pd.read_csv(
        raw_dir / "products.csv",
        usecols=products_cols,
    )
    users = pd.read_csv(
        raw_dir / "users.csv",
        usecols=users_cols,
        parse_dates=["created_at"],
        low_memory=False,
    )

    # Clean column names
    order_items = clean_columns(order_items)
    orders = clean_columns(orders)
    products = clean_columns(products)
    users = clean_columns(users)

    return order_items, orders, products, users


def remove_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[list] = None,
    columns_to_keep: Optional[list] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove unnecessary columns from a DataFrame.

    Can work in two modes:
    1. Remove specific columns (columns_to_drop)
    2. Keep only specific columns (columns_to_keep) - removes all others

    Args:
        df: Input DataFrame.
        columns_to_drop: List of column names to remove. If None and columns_to_keep is None, returns df unchanged.
        columns_to_keep: List of column names to keep. If provided, all other columns are dropped.
                        Takes precedence over columns_to_drop.

    Returns:
        Tuple of (DataFrame with columns removed, report dictionary with removal stats).
    """
    df = df.copy()
    report = {}
    initial_columns = df.columns.tolist()
    initial_count = len(df.columns)

    if columns_to_keep is not None:
        # Keep only specified columns
        existing_cols_to_keep = [col for col in columns_to_keep if col in df.columns]
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        df = df[existing_cols_to_keep]
        report["mode"] = "keep_mode"
        report["requested_columns_to_keep"] = columns_to_keep
        report["columns_actually_kept"] = existing_cols_to_keep
        report["missing_columns"] = missing_cols
    elif columns_to_drop is not None:
        # Drop specified columns
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        non_existing_cols = [col for col in columns_to_drop if col not in df.columns]
        df = df.drop(columns=existing_cols_to_drop, errors="ignore")
        report["mode"] = "drop_mode"
        report["requested_columns_to_drop"] = columns_to_drop
        report["columns_actually_dropped"] = existing_cols_to_drop
        report["non_existing_columns"] = non_existing_cols
    else:
        report["mode"] = "no_action"
        report["message"] = "No columns specified for removal"

    # Summary statistics
    final_count = len(df.columns)
    report["initial_column_count"] = initial_count
    report["final_column_count"] = final_count
    report["columns_removed"] = initial_count - final_count
    report["remaining_columns"] = df.columns.tolist()

    return df, report


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

    orders_renamed = orders.rename(
        columns={
            "shipped_at": "order_shipped_at",
            "delivered_at": "order_delivered_at",
            "returned_at": "order_returned_at",
            "status": "order_status",
        }
    )

    # Rename users.id to avoid collision
    users_renamed = users.rename(
        columns={
            "created_at": "user_created_at",
            "id": "user_id",
            "gender": "user_gender",
        }
    )

    # Rename order_items columns
    # Drop user_id from order_items since we'll get it from orders merge
    order_items_renamed = order_items.drop(columns=["user_id"], errors="ignore").rename(
        columns={
            "id": "order_item_id",
            "shipped_at": "item_shipped_at",
            "delivered_at": "item_delivered_at",
            "returned_at": "item_returned_at",
            "status": "item_status",
        }
    )

    products_renamed = products.rename(columns={"id": "product_id"})

    # Merge at order-item grain
    # First merge order_items with orders on order_id to get user_id and order metadata
    df = order_items_renamed.merge(
        orders_renamed[
            [
                "order_id",
                "user_id",
                "order_status",
                "order_shipped_at",
                "order_delivered_at",
                "order_returned_at",
                "num_of_item",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # Merge with products on product_id
    df = df.merge(
        products_renamed,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Merge with users on user_id (obtained from orders merge)
    df = df.merge(
        users_renamed,
        on="user_id",
        how="left",
        validate="many_to_one",
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

    # Convert all ID columns to string (prevents accidental numeric operations)
    id_columns = [col for col in df.columns if "id" in col.lower()]
    for col in id_columns:
        df[col] = df[col].astype("string")

    # Force known mixed-type columns to string
    for col in STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Convert timestamps safely
    for col in DATETIME_COLS:
        if col in df.columns:
            # Handle both formats: with and without microseconds
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
    Execute the data pipeline: load, merge, standardize types, and save.

    NOTE: This function outputs RAW merged data. Feature engineering
    (return flags, margins, profit erosion) should be applied separately
    using functions from src.feature_engineering.

    Args:
        raw_dir: Optional path to raw data directory.
        output_dir: Optional path to output directory.
        save_output: Whether to save the output files.

    Returns:
        Merged DataFrame with standardized types, ready for feature engineering.
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    # Load raw data
    order_items, orders, products, users = load_raw_data(raw_dir)

    # Merge datasets
    df = merge_datasets(order_items, orders, products, users)

    # Standardize data types
    df = standardize_dtypes(df)

    # Save output (raw merged data only - no feature engineering)
    if save_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            output_dir / "returns_eda_v1.parquet", index=False, compression="snappy"
        )
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
