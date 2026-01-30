"""
Unit tests for the data_processing module.
"""

import pandas as pd
import pytest

from src.data_processing import clean_columns, merge_datasets


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
