"""
Unit tests for the modeling module.
"""
import numpy as np
import pandas as pd
import pytest

from src.modeling import (
    calculate_return_rates_by_group,
    calculate_margin_loss_by_group,
    build_customer_behavior_profile,
    calculate_customer_margin_exposure,
    summarize_profit_erosion,
    segment_customers_by_return_behavior,
)


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
        result = summarize_profit_erosion(sample_merged_df, cost_per_return=cost_per_return)
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
