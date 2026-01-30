"""
Data cleaning module for the Profit Erosion E-commerce Capstone Project.

This module provides deep data cleaning functions to handle outliers, missing values,
duplicates, inconsistencies, and data quality issues in the returns dataset.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


def detect_and_handle_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None, action: str = "remove"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect and handle duplicate records in the dataset.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplicate detection. If None, uses all columns.
        action: 'remove' to drop duplicates or 'flag' to add a flag column.

    Returns:
        Tuple of (cleaned DataFrame, report dictionary with duplicate stats).
    """
    df = df.copy()
    report = {}

    # Detect duplicates
    dup_mask = df.duplicated(subset=subset, keep=False)
    num_duplicates = dup_mask.sum()
    num_unique_duplicates = df[dup_mask].drop_duplicates(subset=subset).shape[0]

    report["total_duplicates_found"] = num_duplicates
    report["unique_duplicate_groups"] = num_unique_duplicates

    if action == "remove":
        df = df.drop_duplicates(subset=subset, keep="first")
        report["action"] = "removed"
        report["rows_removed"] = num_duplicates
    elif action == "flag":
        df["is_duplicate"] = dup_mask.astype(int)
        report["action"] = "flagged"

    return df, report


def handle_missing_values(
    df: pd.DataFrame, strategy: str = "report"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze and handle missing values in the dataset.

    Args:
        df: Input DataFrame.
        strategy: 'report' (only report), 'drop' (drop rows with any missing),
                 'fill_numeric' (fill numeric with median), 'fill_categorical' (fill with mode).

    Returns:
        Tuple of (cleaned DataFrame, report dictionary with missing value stats).
    """
    df = df.copy()
    report = {}

    # Calculate missing value statistics
    missing_stats = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isnull().sum().values,
            "missing_pct": (df.isnull().sum().values / len(df) * 100).round(2),
        }
    )
    missing_stats = missing_stats[missing_stats["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    report["missing_summary"] = missing_stats.to_dict("records")
    report["columns_with_missing"] = missing_stats["column"].tolist()
    report["total_missing_cells"] = df.isnull().sum().sum()

    if strategy == "drop":
        initial_rows = len(df)
        df = df.dropna()
        report["action"] = "dropped"
        report["rows_removed"] = initial_rows - len(df)
    elif strategy == "fill_numeric":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        report["action"] = "filled_numeric_with_median"
    elif strategy == "fill_categorical":
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        report["action"] = "filled_categorical_with_mode"

    return df, report


def detect_outliers_iqr(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    multiplier: float = 1.5,
    action: str = "flag",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Detect outliers using Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame.
        numeric_cols: Columns to check for outliers. If None, checks all numeric columns.
        multiplier: IQR multiplier (default 1.5 for standard outliers, 3.0 for extreme).
        action: 'flag' to add outlier flag column, or 'remove' to drop outliers.

    Returns:
        Tuple of (DataFrame, report dictionary with outlier stats).
        When action='flag', adds 'is_outlier', 'outlier_columns', and 'outlier_values' columns.
    """
    df = df.copy()
    report = {}

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude flag columns (is_*) and derived columns (margin*, discount*) from outlier detection
        # ID columns are already string type, so they're excluded automatically
        numeric_cols = [
            col
            for col in numeric_cols
            if not (
                col.startswith("is_")
                or "margin" in col.lower()
                or "discount" in col.lower()
            )
        ]

    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    outlier_counts = {}
    outlier_details_by_row = {idx: {"columns": [], "values": {}} for idx in df.index}

    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask[col] = col_outliers
            outlier_counts[col] = col_outliers.sum()

            # Store outlier column and value for each row
            for idx in df.index[col_outliers]:
                outlier_details_by_row[idx]["columns"].append(col)
                outlier_details_by_row[idx]["values"][col] = df.loc[idx, col]

    # Overall outlier flag (any column flagged as outlier)
    is_outlier = outlier_mask.any(axis=1)
    report["total_outlier_rows"] = is_outlier.sum()
    report["outlier_counts_by_column"] = outlier_counts
    report["columns_with_outliers"] = [
        col for col, count in outlier_counts.items() if count > 0
    ]

    if action == "flag":
        df["is_outlier"] = is_outlier.astype(int)
        # Add columns showing which columns are outliers and their values
        df["outlier_columns"] = [
            (
                ";".join(outlier_details_by_row[idx]["columns"])
                if outlier_details_by_row[idx]["columns"]
                else ""
            )
            for idx in df.index
        ]
        df["outlier_values"] = [
            (
                ";".join(
                    [
                        f"{col}={val}"
                        for col, val in outlier_details_by_row[idx]["values"].items()
                    ]
                )
                if outlier_details_by_row[idx]["values"]
                else ""
            )
            for idx in df.index
        ]
        report["action"] = "flagged"
    elif action == "remove":
        initial_rows = len(df)
        df = df[~is_outlier]
        report["action"] = "removed"
        report["rows_removed"] = initial_rows - len(df)

    return df, report


def validate_price_consistency(
    df: pd.DataFrame, action: str = "flag"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate price consistency (retail >= sale price >= cost >= 0).

    Args:
        df: Input DataFrame with price columns.
        action: 'flag' to add inconsistency flag, 'remove' to drop inconsistent rows.

    Returns:
        Tuple of (DataFrame, report dictionary with consistency stats).
    """
    df = df.copy()
    report = {}

    issues = pd.DataFrame(
        False,
        index=df.index,
        columns=[
            "negative_price",
            "sale_exceeds_retail",
            "cost_exceeds_sale",
            "cost_negative",
        ],
    )

    # Check for negative prices
    if "sale_price" in df.columns:
        issues["negative_price"] = df["sale_price"] < 0
    if "cost" in df.columns:
        issues["cost_negative"] = df["cost"] < 0

    # Check for illogical price relationships
    if "retail_price" in df.columns and "sale_price" in df.columns:
        issues["sale_exceeds_retail"] = df["sale_price"] > df["retail_price"]

    if "sale_price" in df.columns and "cost" in df.columns:
        issues["cost_exceeds_sale"] = df["cost"] > df["sale_price"]

    has_issue = issues.any(axis=1)
    report["total_inconsistent_rows"] = has_issue.sum()
    report["issue_breakdown"] = issues.sum().to_dict()

    if action == "flag":
        df["has_price_inconsistency"] = has_issue.astype(int)
        report["action"] = "flagged"
    elif action == "remove":
        initial_rows = len(df)
        df = df[~has_issue]
        report["action"] = "removed"
        report["rows_removed"] = initial_rows - len(df)

    return df, report


def validate_status_consistency(
    df: pd.DataFrame, action: str = "flag"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate logical consistency in status flags (e.g., item_returned requires item_delivered).

    Args:
        df: Input DataFrame with status columns.
        action: 'flag' to add inconsistency flag, 'remove' to drop inconsistent rows.

    Returns:
        Tuple of (DataFrame, report dictionary with status consistency stats).
    """
    df = df.copy()
    report = {}

    issues = pd.DataFrame(
        False,
        index=df.index,
        columns=[
            "returned_not_delivered",
            "returned_not_shipped",
            "item_order_status_mismatch",
        ],
    )

    # Returned items should have been delivered
    if "is_returned_item" in df.columns and "item_status" in df.columns:
        is_returned = df["is_returned_item"] == 1
        issues["returned_not_delivered"] = is_returned & (
            df["item_status"].str.lower() != "returned"
        )

    # Delivered items should have been shipped
    if "item_delivered_at" in df.columns and "item_shipped_at" in df.columns:
        issues["returned_not_shipped"] = (
            df["item_delivered_at"].notna() & df["item_shipped_at"].isna()
        )

    # Item status should match order status on returns
    if "is_returned_item" in df.columns and "is_returned_order" in df.columns:
        issues["item_order_status_mismatch"] = (df["is_returned_item"] == 1) & (
            df["is_returned_order"] == 0
        )

    has_issue = issues.any(axis=1)
    report["total_inconsistent_rows"] = has_issue.sum()
    report["issue_breakdown"] = issues.sum().to_dict()

    if action == "flag":
        df["has_status_inconsistency"] = has_issue.astype(int)
        report["action"] = "flagged"
    elif action == "remove":
        initial_rows = len(df)
        df = df[~has_issue]
        report["action"] = "removed"
        report["rows_removed"] = initial_rows - len(df)

    return df, report


def validate_temporal_consistency(
    df: pd.DataFrame, action: str = "flag"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate temporal consistency (shipped < delivered < returned).

    Args:
        df: Input DataFrame with timestamp columns.
        action: 'flag' to add inconsistency flag, 'remove' to drop inconsistent rows.

    Returns:
        Tuple of (DataFrame, report dictionary with temporal consistency stats).
    """
    df = df.copy()
    report = {}

    issues = pd.DataFrame(
        False,
        index=df.index,
        columns=[
            "delivered_before_shipped",
            "returned_before_delivered",
        ],
    )

    # Delivered should be after shipped
    if "item_shipped_at" in df.columns and "item_delivered_at" in df.columns:
        mask = df["item_shipped_at"].notna() & df["item_delivered_at"].notna()
        issues.loc[mask, "delivered_before_shipped"] = (
            df.loc[mask, "item_delivered_at"] < df.loc[mask, "item_shipped_at"]
        )

    # Returned should be after delivered
    if "item_delivered_at" in df.columns and "item_returned_at" in df.columns:
        mask = df["item_delivered_at"].notna() & df["item_returned_at"].notna()
        issues.loc[mask, "returned_before_delivered"] = (
            df.loc[mask, "item_returned_at"] < df.loc[mask, "item_delivered_at"]
        )

    has_issue = issues.any(axis=1)
    report["total_inconsistent_rows"] = has_issue.sum()
    report["issue_breakdown"] = issues.sum().to_dict()

    if action == "flag":
        df["has_temporal_inconsistency"] = has_issue.astype(int)
        report["action"] = "flagged"
    elif action == "remove":
        initial_rows = len(df)
        df = df[~has_issue]
        report["action"] = "removed"
        report["rows_removed"] = initial_rows - len(df)

    return df, report


def clean_categorical_values(
    df: pd.DataFrame,
    cat_cols: Optional[List[str]] = None,
    lowercase: bool = True,
    strip_whitespace: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean categorical columns by standardizing case and removing whitespace.

    Args:
        df: Input DataFrame.
        cat_cols: Columns to clean. If None, cleans object/string columns.
        lowercase: Convert to lowercase.
        strip_whitespace: Strip leading/trailing whitespace.

    Returns:
        Tuple of (cleaned DataFrame, report dictionary).
    """
    df = df.copy()
    report = {}

    if cat_cols is None:
        cat_cols = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

    report["columns_cleaned"] = []
    report["value_replacements"] = {}

    for col in cat_cols:
        if col in df.columns:
            original_nunique = df[col].nunique()

            if strip_whitespace:
                df[col] = df[col].str.strip()
            if lowercase:
                df[col] = df[col].str.lower()

            new_nunique = df[col].nunique()
            if original_nunique != new_nunique:
                report["value_replacements"][col] = {
                    "before": original_nunique,
                    "after": new_nunique,
                    "reduced_by": original_nunique - new_nunique,
                }
            report["columns_cleaned"].append(col)

    return df, report


def remove_low_variance_columns(
    df: pd.DataFrame, variance_threshold: float = 0.01
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove columns with very low variance (near-constant values).

    Args:
        df: Input DataFrame.
        variance_threshold: Variance threshold below which columns are removed.

    Returns:
        Tuple of (cleaned DataFrame, report dictionary).
    """
    df = df.copy()
    report = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    removed_cols = []

    for col in numeric_cols:
        if df[col].std() < variance_threshold:
            removed_cols.append(col)
            df = df.drop(columns=[col])

    report["columns_removed"] = removed_cols
    report["num_columns_removed"] = len(removed_cols)

    return df, report


def perform_deep_clean(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    handle_missing: str = "report",
    detect_outliers: bool = True,
    validate_prices: bool = True,
    validate_status: bool = True,
    validate_temporal: bool = True,
    clean_categories: bool = True,
    remove_low_variance: bool = False,
    outlier_action: str = "flag",
    price_action: str = "flag",
    status_action: str = "flag",
    temporal_action: str = "flag",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Perform comprehensive deep cleaning of the dataset using all validation functions.

    Args:
        df: Input DataFrame.
        remove_duplicates: Whether to check for duplicates.
        handle_missing: Strategy for missing values ('report', 'drop', 'fill_numeric', 'fill_categorical').
        detect_outliers: Whether to detect outliers.
        validate_prices: Whether to validate price consistency.
        validate_status: Whether to validate status consistency.
        validate_temporal: Whether to validate temporal consistency.
        clean_categories: Whether to clean categorical values.
        remove_low_variance: Whether to remove low-variance columns.
        outlier_action: Action for outliers ('flag' or 'remove').
        price_action: Action for price issues ('flag' or 'remove').
        status_action: Action for status issues ('flag' or 'remove').
        temporal_action: Action for temporal issues ('flag' or 'remove').

    Returns:
        Tuple of (cleaned DataFrame, comprehensive report dictionary).
    """
    df = df.copy()
    cleaning_report = {}

    # Step 1: Remove duplicates
    if remove_duplicates:
        df, dup_report = detect_and_handle_duplicates(df, action="remove")
        cleaning_report["duplicates"] = dup_report

    # Step 2: Handle missing values
    if handle_missing != "report":
        df, missing_report = handle_missing_values(df, strategy=handle_missing)
        cleaning_report["missing_values"] = missing_report
    else:
        _, missing_report = handle_missing_values(df, strategy="report")
        cleaning_report["missing_values"] = missing_report

    # Step 3: Detect outliers
    if detect_outliers:
        df, outlier_report = detect_outliers_iqr(df, action=outlier_action)
        cleaning_report["outliers"] = outlier_report

    # Step 4: Validate prices
    if validate_prices:
        df, price_report = validate_price_consistency(df, action=price_action)
        cleaning_report["price_consistency"] = price_report

    # Step 5: Validate status consistency
    if validate_status:
        df, status_report = validate_status_consistency(df, action=status_action)
        cleaning_report["status_consistency"] = status_report

    # Step 6: Validate temporal consistency
    if validate_temporal:
        df, temporal_report = validate_temporal_consistency(df, action=temporal_action)
        cleaning_report["temporal_consistency"] = temporal_report

    # Step 7: Clean categorical values
    if clean_categories:
        df, cat_report = clean_categorical_values(df)
        cleaning_report["categorical_cleaning"] = cat_report

    # Step 8: Remove low variance columns
    if remove_low_variance:
        df, variance_report = remove_low_variance_columns(df)
        cleaning_report["low_variance_removal"] = variance_report

    # Final summary
    cleaning_report["summary"] = {
        "initial_rows": len(df) if remove_duplicates else len(df),
        "final_rows": len(df),
        "rows_removed": (
            cleaning_report.get("duplicates", {}).get("rows_removed", 0)
            + cleaning_report.get("missing_values", {}).get("rows_removed", 0)
            + cleaning_report.get("outliers", {}).get("rows_removed", 0)
            + cleaning_report.get("price_consistency", {}).get("rows_removed", 0)
            + cleaning_report.get("status_consistency", {}).get("rows_removed", 0)
            + cleaning_report.get("temporal_consistency", {}).get("rows_removed", 0)
        ),
    }

    return df, cleaning_report


def save_cleaned_dataset(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Save the dataset with validation flags, creating separate review dataset for flagged records.

    Args:
        df: DataFrame with validation flags (is_outlier, outlier_columns, outlier_values,
            has_price_inconsistency, has_status_inconsistency, has_temporal_inconsistency).
        output_dir: Optional path to output directory.

    Returns:
        DataFrame without validation flag columns (all records kept)
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define validation flag columns to remove from analysis dataset
    validation_flag_cols = [
        "is_outlier",
        "outlier_columns",
        "outlier_values",
        "has_price_inconsistency",
        "has_status_inconsistency",
        "has_temporal_inconsistency",
    ]

    # Identify which validation columns exist in the dataframe
    existing_validation_cols = [
        col for col in validation_flag_cols if col in df.columns
    ]

    # Identify rows with any validation flag = 1
    has_any_error = pd.Series(False, index=df.index)
    for col in existing_validation_cols:
        has_any_error = has_any_error | (df[col] == 1)

    # Create main dataset: ALL rows, but without validation flag columns
    clean_df = df.drop(columns=existing_validation_cols, errors="ignore")

    # Create error dataset: only flagged rows with order_item_id and validation details for review
    error_df = df[has_any_error].copy()
    # Ensure order_item_id is included in error dataset if it exists
    error_cols_to_keep = []
    if "order_item_id" in error_df.columns:
        error_cols_to_keep.append("order_item_id")
    error_cols_to_keep.extend(existing_validation_cols)
    error_df = error_df[error_cols_to_keep]

    # Save main dataset for analysis (all records)
    clean_df.to_parquet(
        output_dir / "returns_eda_v1.parquet", index=False, compression="snappy"
    )
    clean_df.to_csv(output_dir / "returns_eda_v1.csv", index=False)

    # Save error dataset for review (flagged records only)
    error_df.to_csv(output_dir / "data_to_review.csv", index=False)

    return clean_df
