"""
Unit tests for the feature_engineering module.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import calculate_margins, engineer_return_features


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
