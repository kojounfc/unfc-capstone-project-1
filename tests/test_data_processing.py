"""
Unit tests for the data_processing module.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    clean_columns,
    engineer_return_features,
    calculate_margins,
    merge_datasets,
    get_data_quality_report,
)


class TestCleanColumns:
    """Test cases for the clean_columns function."""

    def test_strips_whitespace_from_column_names(self):
        """Test that whitespace is stripped from column names."""
        df = pd.DataFrame({" col1 ": [1], "col2  ": [2], "  col3": [3]})
        result = clean_columns(df)
        assert list(result.columns) == ["col1", "col2", "col3"]

    def test_removes_bom_characters(self):
        """Test that BOM characters are removed from column names."""
        df = pd.DataFrame({"\ufeffcol1": [1], "col2": [2]})
        result = clean_columns(df)
        assert list(result.columns) == ["col1", "col2"]

    def test_does_not_modify_original_dataframe(self):
        """Test that the original DataFrame is not modified."""
        df = pd.DataFrame({" col1 ": [1]})
        original_cols = list(df.columns)
        clean_columns(df)
        assert list(df.columns) == original_cols


class TestEngineerReturnFeatures:
    """Test cases for the engineer_return_features function."""

    def test_creates_is_returned_item_flag(self):
        """Test that is_returned_item flag is correctly created."""
        df = pd.DataFrame({
            "item_status": ["Complete", "Returned", "Shipped", "returned", "RETURNED"],
            "order_status": ["Complete", "Complete", "Complete", "Complete", "Complete"],
        })
        result = engineer_return_features(df)
        assert "is_returned_item" in result.columns
        assert list(result["is_returned_item"]) == [0, 1, 0, 1, 1]

    def test_creates_is_returned_order_flag(self):
        """Test that is_returned_order flag is correctly created."""
        df = pd.DataFrame({
            "item_status": ["Complete", "Complete"],
            "order_status": ["Complete", "Returned"],
        })
        result = engineer_return_features(df)
        assert "is_returned_order" in result.columns
        assert list(result["is_returned_order"]) == [0, 1]

    def test_does_not_modify_original_dataframe(self):
        """Test that the original DataFrame is not modified."""
        df = pd.DataFrame({
            "item_status": ["Complete", "Returned"],
            "order_status": ["Complete", "Complete"],
        })
        original_cols = list(df.columns)
        engineer_return_features(df)
        assert list(df.columns) == original_cols


class TestCalculateMargins:
    """Test cases for the calculate_margins function."""

    def test_calculates_item_margin(self):
        """Test that item_margin is correctly calculated."""
        df = pd.DataFrame({
            "sale_price": [100.0, 50.0],
            "cost": [40.0, 30.0],
            "retail_price": [120.0, 60.0],
        })
        result = calculate_margins(df)
        assert list(result["item_margin"]) == [60.0, 20.0]

    def test_calculates_discount_amount(self):
        """Test that discount_amount is correctly calculated."""
        df = pd.DataFrame({
            "sale_price": [100.0, 50.0],
            "cost": [40.0, 30.0],
            "retail_price": [120.0, 60.0],
        })
        result = calculate_margins(df)
        assert list(result["discount_amount"]) == [20.0, 10.0]

    def test_handles_zero_retail_price(self):
        """Test that zero retail price results in NaN discount_pct."""
        df = pd.DataFrame({
            "sale_price": [100.0],
            "cost": [40.0],
            "retail_price": [0.0],
        })
        result = calculate_margins(df)
        assert np.isnan(result["discount_pct"].iloc[0])

    def test_handles_zero_sale_price(self):
        """Test that zero sale price results in NaN margin_pct."""
        df = pd.DataFrame({
            "sale_price": [0.0],
            "cost": [40.0],
            "retail_price": [100.0],
        })
        result = calculate_margins(df)
        assert np.isnan(result["item_margin_pct"].iloc[0])

    def test_coerces_non_numeric_values(self):
        """Test that non-numeric values are coerced to NaN."""
        df = pd.DataFrame({
            "sale_price": ["invalid", 50.0],
            "cost": [40.0, "invalid"],
            "retail_price": [120.0, 60.0],
        })
        result = calculate_margins(df)
        assert np.isnan(result["item_margin"].iloc[0])
        assert np.isnan(result["item_margin"].iloc[1])


class TestMergeDatasets:
    """Test cases for the merge_datasets function."""

    def test_merge_produces_correct_row_count(
        self, sample_order_items, sample_orders, sample_products, sample_users
    ):
        """Test that merge produces expected number of rows."""
        result = merge_datasets(
            sample_order_items, sample_orders, sample_products, sample_users
        )
        # Should have same number of rows as order_items (grain level)
        assert len(result) == len(sample_order_items)

    def test_merge_renames_columns_correctly(
        self, sample_order_items, sample_orders, sample_products, sample_users
    ):
        """Test that columns are renamed to avoid collisions."""
        result = merge_datasets(
            sample_order_items, sample_orders, sample_products, sample_users
        )
        assert "order_item_id" in result.columns
        assert "item_status" in result.columns
        assert "order_status" in result.columns
        assert "user_gender" in result.columns

    def test_merge_preserves_order_item_grain(
        self, sample_order_items, sample_orders, sample_products, sample_users
    ):
        """Test that merge maintains order-item level granularity."""
        result = merge_datasets(
            sample_order_items, sample_orders, sample_products, sample_users
        )
        assert result["order_item_id"].nunique() == len(sample_order_items)


class TestGetDataQualityReport:
    """Test cases for the get_data_quality_report function."""

    def test_returns_expected_keys(self, sample_merged_df):
        """Test that report contains expected keys."""
        report = get_data_quality_report(sample_merged_df)
        expected_keys = [
            "total_rows",
            "unique_order_items",
            "unique_orders",
            "unique_users",
            "return_rate",
            "columns",
        ]
        for key in expected_keys:
            assert key in report

    def test_calculates_correct_return_rate(self, sample_merged_df):
        """Test that return rate is correctly calculated."""
        report = get_data_quality_report(sample_merged_df)
        # 1 returned item out of 5 = 20%
        assert report["return_rate"] == 20.0
