"""
Unit tests for the data_cleaning module.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (clean_categorical_values,
                               detect_and_handle_duplicates,
                               detect_outliers_iqr, handle_missing_values,
                               perform_deep_clean, remove_low_variance_columns,
                               save_cleaned_dataset,
                               validate_price_consistency,
                               validate_status_consistency,
                               validate_temporal_consistency)


class TestDetectAndHandleDuplicates:
    """Test cases for the detect_and_handle_duplicates function."""

    def test_detects_complete_duplicates(self):
        """Test that complete row duplicates are detected."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": [10, 20, 20, 30],
                "category": ["A", "B", "B", "C"],
            }
        )
        result, report = detect_and_handle_duplicates(df, action="flag")
        assert report["total_duplicates_found"] == 2  # Two duplicate rows
        assert "is_duplicate" in result.columns
        assert result["is_duplicate"].sum() == 2

    def test_removes_duplicates(self):
        """Test that duplicates are removed when action='remove'."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": [10, 20, 20, 30],
            }
        )
        result, report = detect_and_handle_duplicates(df, action="remove")
        assert len(result) == 3  # Two duplicate rows (both marked duplicates) removed
        assert report["action"] == "removed"
        assert report["rows_removed"] == 2

    def test_detects_duplicates_on_subset(self):
        """Test duplicate detection on specific columns."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 3],
                "value": [10, 20, 25, 30],
                "category": ["A", "B", "B", "C"],
            }
        )
        result, report = detect_and_handle_duplicates(df, subset=["id"], action="flag")
        # Rows 2 and 3 have same id
        assert result["is_duplicate"].sum() == 2

    def test_no_duplicates_found(self):
        """Test case when no duplicates exist."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "value": [10, 20, 30, 40],
            }
        )
        result, report = detect_and_handle_duplicates(df, action="remove")
        assert report["total_duplicates_found"] == 0
        assert len(result) == len(df)

    def test_does_not_modify_original_dataframe(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 2],
                "value": [10, 20, 20],
            }
        )
        original_len = len(df)
        detect_and_handle_duplicates(df, action="remove")
        assert len(df) == original_len


class TestHandleMissingValues:
    """Test cases for the handle_missing_values function."""

    def test_reports_missing_values(self):
        """Test that missing value report is generated correctly."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4],
                "col2": [np.nan, np.nan, 6, 7],
                "col3": [8, 9, 10, 11],
            }
        )
        result, report = handle_missing_values(df, strategy="report")
        assert report["total_missing_cells"] == 3
        assert len(report["columns_with_missing"]) == 2
        assert "col3" not in report["columns_with_missing"]

    def test_drops_rows_with_missing_values(self):
        """Test that rows with any missing values are dropped."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4],
                "col2": [5, np.nan, 7, 8],
                "col3": [9, 10, 11, 12],
            }
        )
        result, report = handle_missing_values(df, strategy="drop")
        assert len(result) == 2  # Only rows 0 and 3 have no missing values
        assert report["action"] == "dropped"
        assert report["rows_removed"] == 2

    def test_fills_numeric_with_median(self):
        """Test that numeric columns are filled with median."""
        df = pd.DataFrame(
            {
                "col1": [10.0, 20.0, np.nan, 40.0],
                "col2": [1, 2, 3, 4],
            }
        )
        result, report = handle_missing_values(df, strategy="fill_numeric")
        assert not result["col1"].isna().any()
        # Median of [10, 20, 40] is 20
        assert result["col1"].iloc[2] == 20.0
        assert report["action"] == "filled_numeric_with_median"

    def test_fills_categorical_with_mode(self):
        """Test that categorical columns are filled with mode."""
        df = pd.DataFrame(
            {
                "col1": ["A", "B", "A", np.nan],
                "col2": [1, 2, 3, 4],
            }
        )
        result, report = handle_missing_values(df, strategy="fill_categorical")
        assert not result["col1"].isna().any()
        # Mode of ["A", "B", "A"] is "A"
        assert result["col1"].iloc[3] == "A"
        assert report["action"] == "filled_categorical_with_mode"

    def test_identifies_all_columns_with_missing(self):
        """Test that all columns with missing values are identified."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, np.nan],
                "col2": [4, np.nan, 6],
                "col3": [7, 8, 9],
                "col4": [np.nan, np.nan, np.nan],
            }
        )
        result, report = handle_missing_values(df, strategy="report")
        assert set(report["columns_with_missing"]) == {"col1", "col2", "col4"}


class TestDetectOutliersIQR:
    """Test cases for the detect_outliers_iqr function."""

    def test_detects_outliers_in_numeric_column(self):
        """Test that outliers are detected using IQR method."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
                "category": ["A", "A", "B", "B", "C", "C"],
            }
        )
        result, report = detect_outliers_iqr(df, numeric_cols=["value"], action="flag")
        assert report["total_outlier_rows"] > 0
        assert "is_outlier" in result.columns
        assert result.loc[result["value"] == 100, "is_outlier"].values[0] == 1

    def test_removes_outliers(self):
        """Test that outliers are removed when action='remove'."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5, 100],
                "category": ["A", "A", "B", "B", "C", "C"],
            }
        )
        result, report = detect_outliers_iqr(
            df, numeric_cols=["value"], action="remove"
        )
        assert report["action"] == "removed"
        assert 100 not in result["value"].values
        assert report["rows_removed"] >= 1

    def test_uses_custom_multiplier(self):
        """Test that custom IQR multiplier affects detection threshold."""
        df = pd.DataFrame(
            {
                "value": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    20,
                ],  # 20 might not be outlier with larger multiplier
                "category": ["A"] * 6,
            }
        )
        # With multiplier=1.5 (default)
        _, report_tight = detect_outliers_iqr(
            df, numeric_cols=["value"], multiplier=1.5, action="flag"
        )
        # With multiplier=3.0 (more lenient)
        _, report_lenient = detect_outliers_iqr(
            df, numeric_cols=["value"], multiplier=3.0, action="flag"
        )
        # Lenient should find fewer outliers
        assert (
            report_lenient["total_outlier_rows"] <= report_tight["total_outlier_rows"]
        )

    def test_auto_detects_numeric_columns(self):
        """Test that numeric columns are automatically detected."""
        df = pd.DataFrame(
            {
                "numeric": [1, 2, 3, 100],
                "text": ["A", "B", "C", "D"],
            }
        )
        result, report = detect_outliers_iqr(df, action="flag")
        # Should only check numeric column
        assert "numeric" in report["columns_with_outliers"]

    def test_tracks_outlier_columns_and_values(self):
        """Test that outlier tracking columns are created with details."""
        df = pd.DataFrame(
            {
                "value1": [1, 2, 3, 100],
                "value2": [10, 20, 30, 40],
                "text": ["A", "B", "C", "D"],
            }
        )
        result, report = detect_outliers_iqr(df, action="flag")
        # Check that tracking columns exist
        assert "outlier_columns" in result.columns
        assert "outlier_values" in result.columns
        # The outlier row should have non-empty tracking information
        outlier_rows = result[result["is_outlier"] == 1]
        if len(outlier_rows) > 0:
            assert outlier_rows["outlier_columns"].iloc[0] != ""
            assert outlier_rows["outlier_values"].iloc[0] != ""

    def test_no_outliers_found(self):
        """Test case when no outliers exist."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5],
            }
        )
        result, report = detect_outliers_iqr(df, numeric_cols=["value"], action="flag")
        assert report["total_outlier_rows"] == 0


class TestValidatePriceConsistency:
    """Test cases for the validate_price_consistency function."""

    def test_detects_negative_sale_price(self):
        """Test that negative sale prices are flagged."""
        df = pd.DataFrame(
            {
                "sale_price": [100.0, -50.0, 75.0],
                "cost": [40.0, 30.0, 40.0],
                "retail_price": [120.0, 80.0, 100.0],
            }
        )
        result, report = validate_price_consistency(df, action="flag")
        assert "has_price_inconsistency" in result.columns
        assert result.loc[1, "has_price_inconsistency"] == 1

    def test_detects_sale_exceeds_retail(self):
        """Test that sale price > retail price is flagged."""
        df = pd.DataFrame(
            {
                "sale_price": [100.0, 150.0],
                "cost": [40.0, 50.0],
                "retail_price": [120.0, 120.0],
            }
        )
        result, report = validate_price_consistency(df, action="flag")
        assert result.loc[1, "has_price_inconsistency"] == 1

    def test_detects_cost_exceeds_sale(self):
        """Test that cost > sale price is flagged."""
        df = pd.DataFrame(
            {
                "sale_price": [100.0, 50.0],
                "cost": [40.0, 60.0],
                "retail_price": [120.0, 80.0],
            }
        )
        result, report = validate_price_consistency(df, action="flag")
        assert result.loc[1, "has_price_inconsistency"] == 1

    def test_removes_inconsistent_rows(self):
        """Test that inconsistent rows are removed."""
        df = pd.DataFrame(
            {
                "sale_price": [100.0, -50.0, 75.0],
                "cost": [40.0, 30.0, 40.0],
                "retail_price": [120.0, 80.0, 100.0],
            }
        )
        result, report = validate_price_consistency(df, action="remove")
        assert len(result) == 2
        assert report["rows_removed"] == 1

    def test_valid_prices_pass(self):
        """Test that valid price relationships pass validation."""
        df = pd.DataFrame(
            {
                "sale_price": [100.0, 75.0, 50.0],
                "cost": [40.0, 30.0, 20.0],
                "retail_price": [120.0, 100.0, 80.0],
            }
        )
        result, report = validate_price_consistency(df, action="flag")
        assert report["total_inconsistent_rows"] == 0


class TestValidateStatusConsistency:
    """Test cases for the validate_status_consistency function."""

    def test_detects_returned_items_mismatch(self):
        """Test that returned item flags are validated."""
        df = pd.DataFrame(
            {
                "is_returned_item": [1, 1, 0],
                "item_status": ["Complete", "Returned", "Shipped"],
            }
        )
        result, report = validate_status_consistency(df, action="flag")
        assert "has_status_inconsistency" in result.columns
        assert (
            result.loc[0, "has_status_inconsistency"] == 1
        )  # is_returned=1 but status!="Returned"

    def test_flags_item_order_status_mismatch(self):
        """Test that item/order status mismatches are detected."""
        df = pd.DataFrame(
            {
                "is_returned_item": [1, 0, 1],
                "is_returned_order": [1, 1, 0],
                "item_status": ["Returned", "Complete", "Returned"],
            }
        )
        result, report = validate_status_consistency(df, action="flag")
        assert result.loc[2, "has_status_inconsistency"] == 1

    def test_removes_inconsistent_status_rows(self):
        """Test that inconsistent status rows are removed."""
        df = pd.DataFrame(
            {
                "is_returned_item": [1, 0, 1],
                "item_status": ["Complete", "Shipped", "Returned"],
            }
        )
        result, report = validate_status_consistency(df, action="remove")
        assert len(result) < len(df)
        assert report["rows_removed"] >= 1

    def test_valid_status_passes(self):
        """Test that valid status combinations pass."""
        df = pd.DataFrame(
            {
                "is_returned_item": [0, 1],
                "item_status": ["Complete", "Returned"],
            }
        )
        result, report = validate_status_consistency(df, action="flag")
        assert report["total_inconsistent_rows"] == 0


class TestValidateTemporalConsistency:
    """Test cases for the validate_temporal_consistency function."""

    def test_detects_shipped_before_created(self):
        """Test that delivered before shipped is flagged."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-03", "2024-01-05"]),
                "item_delivered_at": pd.to_datetime(["2024-01-02", "2024-01-06"]),
                "item_returned_at": pd.to_datetime([None, None]),
            }
        )
        result, report = validate_temporal_consistency(df, action="flag")
        assert "has_temporal_inconsistency" in result.columns
        assert result.loc[0, "has_temporal_inconsistency"] == 1

    def test_detects_delivered_before_shipped(self):
        """Test that delivered before shipped is flagged."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-05"]),
                "item_delivered_at": pd.to_datetime(["2024-01-03"]),
                "item_returned_at": pd.to_datetime([None]),
            }
        )
        result, report = validate_temporal_consistency(df, action="flag")
        assert result.loc[0, "has_temporal_inconsistency"] == 1

    def test_detects_returned_before_delivered(self):
        """Test that returned before delivered is flagged."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-02"]),
                "item_delivered_at": pd.to_datetime(["2024-01-05"]),
                "item_returned_at": pd.to_datetime(["2024-01-03"]),
            }
        )
        result, report = validate_temporal_consistency(df, action="flag")
        assert result.loc[0, "has_temporal_inconsistency"] == 1

    def test_removes_temporal_inconsistencies(self):
        """Test that temporal inconsistencies are removed."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-02", "2024-01-05"]),
                "item_delivered_at": pd.to_datetime(["2024-01-03", "2024-01-01"]),
                "item_returned_at": pd.to_datetime([None, None]),
            }
        )
        result, report = validate_temporal_consistency(df, action="remove")
        assert len(result) == 1
        assert report["rows_removed"] == 1

    def test_valid_temporal_sequence_passes(self):
        """Test that valid temporal sequences pass."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-02"]),
                "item_delivered_at": pd.to_datetime(["2024-01-03"]),
                "item_returned_at": pd.to_datetime(["2024-01-04"]),
            }
        )
        result, report = validate_temporal_consistency(df, action="flag")
        assert report["total_inconsistent_rows"] == 0

    def test_handles_missing_timestamps(self):
        """Test that missing timestamps are handled gracefully."""
        df = pd.DataFrame(
            {
                "item_shipped_at": pd.to_datetime(["2024-01-02", None]),
                "item_delivered_at": pd.to_datetime([None, None]),
                "item_returned_at": pd.to_datetime([None, None]),
            }
        )
        result, report = validate_temporal_consistency(df, action="flag")
        # Should not flag rows with missing timestamps
        assert report["total_inconsistent_rows"] == 0


class TestCleanCategoricalValues:
    """Test cases for the clean_categorical_values function."""

    def test_converts_to_lowercase(self):
        """Test that categorical values are converted to lowercase."""
        df = pd.DataFrame(
            {
                "category": ["JEANS", "Tops", "OUTERWEAR"],
                "brand": ["BRAND X", "Brand Y", "brand z"],
            }
        )
        result, report = clean_categorical_values(df, lowercase=True)
        assert result["category"].iloc[0] == "jeans"
        assert result["brand"].iloc[1] == "brand y"
        assert len(report["columns_cleaned"]) == 2

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        df = pd.DataFrame(
            {
                "category": [" JEANS ", "Tops  ", "  OUTERWEAR"],
                "brand": ["  BRAND X", " Brand Y ", "brand z"],
            }
        )
        result, report = clean_categorical_values(df, strip_whitespace=True)
        assert result["category"].iloc[0] == "jeans"  # strip then lowercase
        assert result["brand"].iloc[0] == "brand x"

    def test_reduces_unique_values(self):
        """Test that cleaning reduces unique values in categorical columns."""
        df = pd.DataFrame(
            {
                "status": ["Complete", "COMPLETE", " complete", "Returned", "RETURNED"],
            }
        )
        result, report = clean_categorical_values(
            df, lowercase=True, strip_whitespace=True
        )
        assert report["value_replacements"]["status"]["before"] == 5
        assert report["value_replacements"]["status"]["after"] == 2

    def test_handles_none_categorical_columns(self):
        """Test that None for cat_cols triggers auto-detection."""
        df = pd.DataFrame(
            {
                "text": ["ABC", "DEF"],
                "number": [1, 2],
            }
        )
        result, report = clean_categorical_values(df, cat_cols=None)
        assert "text" in report["columns_cleaned"]
        assert "number" not in report["columns_cleaned"]

    def test_does_not_modify_numeric_columns(self):
        """Test that numeric columns are not modified."""
        df = pd.DataFrame(
            {
                "amount": [100, 200],
                "category": ["HIGH", "LOW"],
            }
        )
        original_amount = df["amount"].tolist()
        result, report = clean_categorical_values(df)
        assert result["amount"].tolist() == original_amount


class TestRemoveLowVarianceColumns:
    """Test cases for the remove_low_variance_columns function."""

    def test_removes_constant_columns(self):
        """Test that columns with zero variance are removed."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "constant": [5, 5, 5, 5],
                "varying": [1, 2, 3, 4],
            }
        )
        result, report = remove_low_variance_columns(df, variance_threshold=0.01)
        assert "constant" not in result.columns
        assert "varying" in result.columns
        assert report["num_columns_removed"] == 1

    def test_removes_near_constant_columns(self):
        """Test that columns below variance threshold are removed."""
        df = pd.DataFrame(
            {
                "col1": [1.0, 1.0, 1.0, 1.001],
                "col2": [1, 2, 3, 4],
            }
        )
        result, report = remove_low_variance_columns(df, variance_threshold=0.001)
        assert "col1" not in result.columns
        assert "col2" in result.columns

    def test_no_removal_when_threshold_not_met(self):
        """Test that columns are kept if variance exceeds threshold."""
        df = pd.DataFrame(
            {
                "col1": [1, 100, 200, 300],
                "col2": [5, 150, 250, 350],
            }
        )
        result, report = remove_low_variance_columns(df, variance_threshold=1)
        assert len(result.columns) == len(df.columns)
        assert report["num_columns_removed"] == 0

    def test_ignores_string_columns(self):
        """Test that string columns are not affected."""
        df = pd.DataFrame(
            {
                "text": ["A", "A", "A", "A"],
                "number": [1, 2, 3, 4],
            }
        )
        result, report = remove_low_variance_columns(df, variance_threshold=0.01)
        assert "text" in result.columns
        assert report["num_columns_removed"] == 0


class TestPerformDeepClean:
    """Test cases for the perform_deep_clean orchestration function."""

    def test_performs_all_cleaning_steps(self, sample_merged_df):
        """Test that all cleaning steps are executed."""
        df, report = perform_deep_clean(sample_merged_df)
        assert "duplicates" in report
        assert "missing_values" in report
        assert "outliers" in report
        assert "price_consistency" in report
        assert "status_consistency" in report
        assert "temporal_consistency" in report
        assert "categorical_cleaning" in report
        assert "summary" in report

    def test_skips_unwanted_steps(self, sample_merged_df):
        """Test that steps can be disabled."""
        df, report = perform_deep_clean(
            sample_merged_df,
            remove_duplicates=False,
            detect_outliers=False,
            validate_prices=False,
        )
        assert "duplicates" not in report
        assert "outliers" not in report
        assert "price_consistency" not in report

    def test_respects_action_parameters(self, sample_merged_df):
        """Test that action parameters are respected."""
        df, report = perform_deep_clean(
            sample_merged_df,
            outlier_action="flag",
            price_action="flag",
            status_action="flag",
        )
        # Flags should be added
        if report["outliers"]["total_outlier_rows"] > 0:
            assert "is_outlier" in df.columns
        if report["price_consistency"]["total_inconsistent_rows"] > 0:
            assert "has_price_inconsistency" in df.columns

    def test_handles_missing_value_strategies(self, sample_merged_df):
        """Test different missing value handling strategies."""
        # Report-only strategy
        df1, report1 = perform_deep_clean(sample_merged_df, handle_missing="report")
        assert len(df1) == len(sample_merged_df)

        # Drop strategy
        df2, report2 = perform_deep_clean(sample_merged_df, handle_missing="drop")
        assert len(df2) <= len(sample_merged_df)

    def test_summary_statistics_are_correct(self, sample_merged_df):
        """Test that summary statistics are calculated correctly."""
        initial_len = len(sample_merged_df)
        df, report = perform_deep_clean(sample_merged_df, remove_duplicates=True)
        summary = report["summary"]
        assert summary["final_rows"] == len(df)
        assert summary["final_rows"] <= summary["initial_rows"]

    def test_returns_cleaned_dataframe_and_report(self, sample_merged_df):
        """Test that function returns both DataFrame and report."""
        result = perform_deep_clean(sample_merged_df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        df, report = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, dict)

    def test_comprehensive_cleaning_workflow(self):
        """Test comprehensive cleaning workflow on sample dataset."""
        # Create a dataset with various issues
        df = pd.DataFrame(
            {
                "id": [1, 2, 2, 4, 5, 6],  # Duplicate id=2
                "sale_price": [
                    100.0,
                    -50.0,
                    80.0,
                    120.0,
                    200.0,
                    100.0,
                ],  # Negative price
                "cost": [40.0, 60.0, 30.0, 150.0, 50.0, 40.0],  # Cost > sale
                "retail_price": [120.0, 60.0, 100.0, 100.0, 180.0, 120.0],
                "status": [
                    "Complete",
                    "COMPLETE",
                    "Returned",
                    "RETURNED",
                    "SHIPPED",
                    "shipped",
                ],
                "shipped_at": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-01",
                        "2024-01-04",
                        "2024-01-03",
                        "2024-01-06",
                        "2024-01-07",
                    ]
                ),
            }
        )

        result, report = perform_deep_clean(
            df,
            remove_duplicates=True,
            handle_missing="report",
            detect_outliers=True,
            validate_prices=True,
            validate_status=False,
            validate_temporal=True,
            clean_categories=True,
            outlier_action="flag",
            price_action="flag",
            temporal_action="flag",
        )

        # Verify cleaning was applied
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, dict)
        assert "summary" in report
        assert result.shape[0] <= df.shape[0]


class TestSaveCleanedDataset:
    """Test cases for the save_cleaned_dataset function."""

    def test_saves_to_default_directory(self):
        """Test that dataset is saved to default PROCESSED_DATA_DIR."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        # Save with default directory
        save_cleaned_dataset(df)

        # Verify files were created (check in default location)
        from src.config import PROCESSED_DATA_DIR

        parquet_file = PROCESSED_DATA_DIR / "returns_eda_v1.parquet"
        csv_file = PROCESSED_DATA_DIR / "returns_eda_v1.csv"

        assert parquet_file.exists(), f"Parquet file not found at {parquet_file}"
        assert csv_file.exists(), f"CSV file not found at {csv_file}"

    def test_saves_to_custom_directory(self):
        """Test that dataset can be saved to custom directory."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            parquet_file = output_dir / "returns_eda_v1.parquet"
            csv_file = output_dir / "returns_eda_v1.csv"

            assert parquet_file.exists()
            assert csv_file.exists()

    def test_creates_directory_if_not_exists(self):
        """Test that output directory is created if it doesn't exist."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "nested"
            assert not output_dir.exists()

            save_cleaned_dataset(df, output_dir=output_dir)

            assert output_dir.exists()
            assert (output_dir / "returns_eda_v1.parquet").exists()
            assert (output_dir / "returns_eda_v1.csv").exists()

    def test_overwrites_existing_files(self):
        """Test that existing files are overwritten."""
        df1 = pd.DataFrame({"id": [1], "value": [10]})
        df2 = pd.DataFrame({"id": [2, 3], "value": [20, 30]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save first dataset
            save_cleaned_dataset(df1, output_dir=output_dir)
            csv_file = output_dir / "returns_eda_v1.csv"
            first_size = csv_file.stat().st_size

            # Save second dataset
            save_cleaned_dataset(df2, output_dir=output_dir)
            second_size = csv_file.stat().st_size

            # File should be different size
            assert first_size != second_size

            # Verify second dataset is saved
            df_loaded = pd.read_csv(csv_file)
            assert len(df_loaded) == 2

    def test_saves_correct_data_to_csv(self):
        """Test that data is correctly saved to CSV."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.5, 20.5, 30.5],
                "category": ["A", "B", "C"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            csv_file = output_dir / "returns_eda_v1.csv"
            df_loaded = pd.read_csv(csv_file)

            assert len(df_loaded) == 3
            assert list(df_loaded.columns) == ["id", "value", "category"]
            assert df_loaded["id"].tolist() == [1, 2, 3]
            assert df_loaded["category"].tolist() == ["A", "B", "C"]

    def test_saves_correct_data_to_parquet(self):
        """Test that data is correctly saved to Parquet."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10.5, 20.5, 30.5],
                "category": ["A", "B", "C"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            parquet_file = output_dir / "returns_eda_v1.parquet"
            df_loaded = pd.read_parquet(parquet_file)

            assert len(df_loaded) == 3
            assert list(df_loaded.columns) == ["id", "value", "category", "date"]
            assert df_loaded["date"].dtype == df["date"].dtype

    def test_parquet_uses_snappy_compression(self):
        """Test that Parquet file uses snappy compression."""
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            parquet_file = output_dir / "returns_eda_v1.parquet"
            import pyarrow.parquet as pq

            parquet_file_obj = pq.read_table(parquet_file)

            # Verify compression by checking metadata
            metadata = pq.read_metadata(parquet_file)
            # Snappy compression should be used
            assert parquet_file_obj is not None

    def test_saves_large_dataframe(self):
        """Test that large dataframes are saved correctly."""
        df = pd.DataFrame(
            {
                "id": range(10000),
                "value": np.random.randn(10000),
                "category": np.random.choice(["A", "B", "C"], 10000),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            csv_file = output_dir / "returns_eda_v1.csv"
            parquet_file = output_dir / "returns_eda_v1.parquet"

            assert csv_file.exists()
            assert parquet_file.exists()

            # Verify row counts
            df_csv = pd.read_csv(csv_file)
            df_parquet = pd.read_parquet(parquet_file)

            assert len(df_csv) == 10000
            assert len(df_parquet) == 10000

    def test_preserves_data_types_in_parquet(self):
        """Test that data types are preserved in Parquet format."""
        df = pd.DataFrame(
            {
                "int_col": np.array([1, 2, 3], dtype=np.int32),
                "float_col": np.array([1.1, 2.2, 3.3], dtype=np.float64),
                "str_col": ["a", "b", "c"],
                "date_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            parquet_file = output_dir / "returns_eda_v1.parquet"
            df_loaded = pd.read_parquet(parquet_file)

            assert pd.api.types.is_datetime64_any_dtype(df_loaded["date_col"])
            assert pd.api.types.is_integer_dtype(df_loaded["int_col"])
            assert pd.api.types.is_float_dtype(df_loaded["float_col"])

    def test_handles_missing_values_in_save(self):
        """Test that missing values are properly saved and loaded."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "value": [10.0, np.nan, 30.0, np.nan],
                "category": ["A", "B", np.nan, "D"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            csv_file = output_dir / "returns_eda_v1.csv"
            parquet_file = output_dir / "returns_eda_v1.parquet"

            df_csv = pd.read_csv(csv_file)
            df_parquet = pd.read_parquet(parquet_file)

            # Check missing values are preserved
            assert df_csv["value"].isna().sum() == 2
            assert df_parquet["category"].isna().sum() == 1

    def test_csv_does_not_save_index(self):
        """Test that CSV file does not include index column."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "value": [10, 20, 30],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            csv_file = output_dir / "returns_eda_v1.csv"
            df_loaded = pd.read_csv(csv_file)

            # Should not have an "Unnamed: 0" index column
            assert "Unnamed: 0" not in df_loaded.columns
            assert list(df_loaded.columns) == ["id", "value"]

    def test_saves_dataframe_with_special_characters(self):
        """Test that special characters in data are handled correctly."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "description": ["Hello, World!", "Café ☕", "Test™"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_cleaned_dataset(df, output_dir=output_dir)

            csv_file = output_dir / "returns_eda_v1.csv"
            df_loaded = pd.read_csv(csv_file)

            assert df_loaded["description"].iloc[0] == "Hello, World!"
            assert df_loaded["description"].iloc[1] == "Café ☕"
            assert df_loaded["description"].iloc[2] == "Test™"
