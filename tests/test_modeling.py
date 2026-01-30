"""
Unit tests for the modeling module.
"""

import numpy as np
import pandas as pd
import pytest

from src.modeling import (build_customer_behavior_profile,
                          calculate_customer_margin_exposure,
                          calculate_margin_loss_by_group,
                          calculate_price_margin_returned_by_country,
                          calculate_return_rates_by_group,
                          segment_customers_by_return_behavior,
                          summarize_profit_erosion)


class TestCalculateReturnRatesByGroup:
    """Test cases for calculate_return_rates_by_group function."""

    def test_calculates_return_rate_by_single_column(self, sample_merged_df):
        """Test return rate calculation grouped by one column."""
        result = calculate_return_rates_by_group(
            sample_merged_df, ["category"], min_rows=1
        )
        assert "return_rate" in result.columns
        assert "item_rows" in result.columns
        assert "returned_items" in result.columns

    def test_filters_by_minimum_rows(self, sample_merged_df):
        """Test that groups below min_rows are filtered out."""
        result = calculate_return_rates_by_group(
            sample_merged_df, ["category"], min_rows=3
        )
        # Only Jeans (2) and Tops & Tees (2) exist, neither has 3+
        assert len(result) == 0

    def test_sorts_by_return_rate_descending(self, sample_merged_df):
        """Test that results are sorted by return rate descending."""
        result = calculate_return_rates_by_group(
            sample_merged_df, ["category"], min_rows=1
        )
        rates = result["return_rate"].tolist()
        assert rates == sorted(rates, reverse=True)


class TestCalculateMarginLossByGroup:
    """Test cases for calculate_margin_loss_by_group function."""

    def test_only_includes_returned_items(self, sample_merged_df):
        """Test that only returned items are included."""
        result = calculate_margin_loss_by_group(sample_merged_df, ["category"])
        total_returned = result["returned_items"].sum()
        expected_returned = sample_merged_df["is_returned_item"].sum()
        assert total_returned == expected_returned

    def test_calculates_total_lost_margin(self, sample_merged_df):
        """Test that total lost margin is correctly calculated."""
        result = calculate_margin_loss_by_group(sample_merged_df, ["category"])
        returned_df = sample_merged_df[sample_merged_df["is_returned_item"] == 1]
        expected_total = returned_df["item_margin"].sum()
        assert result["total_lost_margin"].sum() == expected_total


class TestBuildCustomerBehaviorProfile:
    """Test cases for build_customer_behavior_profile function."""

    def test_creates_one_row_per_customer(self, sample_merged_df):
        """Test that profile has one row per unique customer."""
        result = build_customer_behavior_profile(sample_merged_df)
        expected_customers = sample_merged_df["user_id"].nunique()
        assert len(result) == expected_customers

    def test_calculates_return_rate_per_customer(self, sample_merged_df):
        """Test that return rate is calculated per customer."""
        result = build_customer_behavior_profile(sample_merged_df)
        assert "return_rate" in result.columns
        # User 1001 has 1 return out of 2 items = 0.5
        assert result.loc[1001, "return_rate"] == 0.5

    def test_calculates_items_per_order(self, sample_merged_df):
        """Test that items per order is calculated."""
        result = build_customer_behavior_profile(sample_merged_df)
        assert "items_per_order" in result.columns


class TestCalculateCustomerMarginExposure:
    """Test cases for calculate_customer_margin_exposure function."""

    def test_only_includes_customers_with_returns(self, sample_merged_df):
        """Test that only customers with returns are included."""
        result = calculate_customer_margin_exposure(sample_merged_df)
        # Only user 1001 has a return
        assert len(result) == 1
        assert 1001 in result.index

    def test_calculates_total_lost_margin(self, sample_merged_df):
        """Test that total lost margin is correct per customer."""
        result = calculate_customer_margin_exposure(sample_merged_df)
        # User 1001 returned item with margin 45.0
        assert result.loc[1001, "total_lost_margin"] == 45.0


class TestSummarizeProfitErosion:
    """Test cases for summarize_profit_erosion function."""

    def test_returns_expected_keys(self, sample_merged_df):
        """Test that summary contains all expected keys."""
        result = summarize_profit_erosion(sample_merged_df)
        expected_keys = [
            "total_items",
            "total_returned",
            "return_rate_pct",
            "total_margin_reversal",
            "avg_margin_per_return",
            "estimated_process_costs",
            "total_profit_erosion",
        ]
        for key in expected_keys:
            assert key in result

    def test_calculates_correct_return_rate(self, sample_merged_df):
        """Test that return rate percentage is correct."""
        result = summarize_profit_erosion(sample_merged_df)
        # 1 return out of 5 items = 20%
        assert result["return_rate_pct"] == 20.0

    def test_includes_process_costs(self, sample_merged_df):
        """Test that process costs are included in total erosion."""
        cost_per_return = 15.0
        result = summarize_profit_erosion(
            sample_merged_df, cost_per_return=cost_per_return
        )
        # 1 return * $15 = $15
        assert result["estimated_process_costs"] == 15.0
        # Total erosion = margin reversal + process costs
        expected_erosion = result["total_margin_reversal"] + 15.0
        assert result["total_profit_erosion"] == expected_erosion


class TestSegmentCustomersByReturnBehavior:
    """Test cases for segment_customers_by_return_behavior function."""

    def test_creates_segment_column(self, sample_merged_df):
        """Test that segment column is created."""
        result = segment_customers_by_return_behavior(sample_merged_df)
        assert "return_segment" in result.columns

    def test_assigns_correct_segments(self, sample_merged_df):
        """Test that segments are correctly assigned."""
        result = segment_customers_by_return_behavior(
            sample_merged_df, return_rate_thresholds=(0.1, 0.3)
        )
        # User 1001: 50% return rate -> high_returner
        # User 1002: 0% return rate -> no_returns
        # User 1003: 0% return rate -> no_returns
        assert result.loc[1001, "return_segment"] == "high_returner"
        assert result.loc[1002, "return_segment"] == "no_returns"
        assert result.loc[1003, "return_segment"] == "no_returns"

    def test_all_customers_have_segment(self, sample_merged_df):
        """Test that all customers are assigned a segment."""
        result = segment_customers_by_return_behavior(sample_merged_df)
        assert result["return_segment"].notna().all()


class TestCalculatePriceMarginReturnedByCountry:
    """Test cases for calculate_price_margin_returned_by_country function."""

    def test_filters_to_returned_items_only(self, sample_merged_df):
        """Test that function filters for returned items only."""
        result = calculate_price_margin_returned_by_country(sample_merged_df)

        # All rows in sample_merged_df should have at least one returned item
        # Based on sample_merged_df fixture, returned items should be present
        assert not result.empty

    def test_groups_by_country(self, sample_merged_df):
        """Test that results are grouped by country."""
        result = calculate_price_margin_returned_by_country(sample_merged_df)

        # Index should be country
        assert result.index.name == "country"

    def test_returns_required_columns(self, sample_merged_df):
        """Test that all required metric columns are present."""
        result = calculate_price_margin_returned_by_country(sample_merged_df)

        expected_columns = [
            "item_count",
            "avg_cost",
            "total_cost",
            "avg_sale_price",
            "total_sale_price",
            "avg_margin",
            "total_margin",
            "median_margin",
            "min_margin",
            "max_margin",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_total_margin_descending(self, sample_merged_df):
        """Test that results are sorted by total_margin in descending order."""
        result = calculate_price_margin_returned_by_country(sample_merged_df)

        if not result.empty:
            # Check that total_margin values are in descending order
            margins = result["total_margin"].values
            assert all(margins[i] >= margins[i + 1] for i in range(len(margins) - 1))

    def test_numeric_values_rounded_to_two_decimals(self, sample_merged_df):
        """Test that numeric values are rounded to 2 decimal places."""
        result = calculate_price_margin_returned_by_country(sample_merged_df)

        # Skip item_count (integer column)
        numeric_cols = [
            "avg_cost",
            "total_cost",
            "avg_sale_price",
            "total_sale_price",
            "avg_margin",
            "total_margin",
            "median_margin",
            "min_margin",
            "max_margin",
        ]

        for col in numeric_cols:
            if not result[col].empty:
                # Check that values don't have more than 2 decimal places
                for val in result[col]:
                    if pd.notna(val):
                        rounded = round(val, 2)
                        assert (
                            val == rounded
                        ), f"Value {val} not rounded to 2 decimals in column {col}"

    def test_returns_empty_dataframe_when_no_returned_items(self):
        """Test that empty DataFrame is returned when no returned items exist."""
        # Create a DataFrame with no returned items
        df = pd.DataFrame(
            {
                "order_id": [1, 2],
                "item_status": ["Delivered", "Delivered"],
                "cost": [10.0, 15.0],
                "sale_price": [20.0, 25.0],
                "item_margin": [10.0, 10.0],
                "country": ["US", "US"],
            }
        )

        result = calculate_price_margin_returned_by_country(df)
        assert result.empty

    def test_single_country_aggregation(self):
        """Test aggregation with single country."""
        df = pd.DataFrame(
            {
                "order_id": [1, 2, 3],
                "item_status": ["Returned", "Returned", "Returned"],
                "cost": [10.0, 15.0, 20.0],
                "sale_price": [20.0, 25.0, 30.0],
                "item_margin": [10.0, 10.0, 10.0],
                "country": ["US", "US", "US"],
            }
        )

        result = calculate_price_margin_returned_by_country(df)

        assert len(result) == 1
        assert result.loc["US", "item_count"] == 3
        assert result.loc["US", "total_cost"] == 45.0
        assert result.loc["US", "avg_cost"] == 15.0

    def test_multiple_countries_aggregation(self):
        """Test aggregation with multiple countries."""
        df = pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4],
                "item_status": ["Returned", "Returned", "Returned", "Returned"],
                "cost": [10.0, 15.0, 20.0, 25.0],
                "sale_price": [20.0, 25.0, 30.0, 35.0],
                "item_margin": [10.0, 10.0, 10.0, 10.0],
                "country": ["US", "US", "CA", "CA"],
            }
        )

        result = calculate_price_margin_returned_by_country(df)

        assert len(result) == 2
        assert "US" in result.index
        assert "CA" in result.index
        assert result.loc["US", "item_count"] == 2
        assert result.loc["CA", "item_count"] == 2

    def test_excludes_non_returned_items_with_mixed_status(self):
        """Test that non-returned items are excluded even in mixed-status DataFrame."""
        df = pd.DataFrame(
            {
                "order_id": [1, 2, 3, 4],
                "item_status": ["Returned", "Delivered", "Returned", "Delivered"],
                "cost": [10.0, 15.0, 20.0, 25.0],
                "sale_price": [20.0, 25.0, 30.0, 35.0],
                "item_margin": [10.0, 10.0, 10.0, 10.0],
                "country": ["US", "US", "US", "US"],
            }
        )

        result = calculate_price_margin_returned_by_country(df)

        # Should only count returned items (1 and 3)
        assert result.loc["US", "item_count"] == 2
        assert result.loc["US", "total_cost"] == 30.0  # 10 + 20
