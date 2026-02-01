"""
Unit tests for the feature_engineering module.

Tests are organized by function, with separate classes for:
- engineer_return_features: Creates return indicator flags
- calculate_margins: Calculates margin and discount metrics
- calculate_profit_erosion: Calculates profit erosion for returned items
- aggregate_profit_erosion_by_order: Aggregates erosion to order level
- aggregate_profit_erosion_by_customer: Aggregates erosion to customer level
- summarize_profit_erosion: Generates summary statistics

Note: Profit erosion functions receive PRE-FILTERED returned items only
(where is_returned_item == 1) for efficiency and alignment with RQs.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    CATEGORY_TIER_MULTIPLIERS,
    DEFAULT_CATEGORY_MULTIPLIER,
    DEFAULT_COST_COMPONENTS,
    aggregate_profit_erosion_by_customer,
    aggregate_profit_erosion_by_order,
    calculate_margins,
    calculate_profit_erosion,
    create_profit_erosion_targets,
    engineer_customer_behavioral_features,
    engineer_return_features,
    save_feature_engineered_dataset,
    summarize_profit_erosion,
)


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


# =============================================================================
# Task 1: Profit Erosion Tests
# =============================================================================
# Note: These functions receive PRE-FILTERED returned items only.
# The caller is responsible for filtering: df[df["is_returned_item"] == 1]
# =============================================================================


class TestCalculateProfitErosion:
    """Test cases for the calculate_profit_erosion function.

    Note: Function expects pre-filtered returned items only.
    """

    @pytest.fixture
    def returned_items_df(self):
        """Create sample DataFrame of RETURNED items only (pre-filtered)."""
        return pd.DataFrame({
            "order_item_id": [1, 2, 3, 4],
            "item_margin": [60.0, 20.0, 40.0, 30.0],
            "category": ["Jeans", "Tops & Tees", "Outerwear & Coats", "Socks"],
        })

    def test_creates_margin_reversal_column(self, returned_items_df):
        """Test that margin_reversal column is created."""
        result = calculate_profit_erosion(returned_items_df)
        assert "margin_reversal" in result.columns

    def test_margin_reversal_equals_item_margin(self, returned_items_df):
        """Test margin_reversal equals item_margin for all returned items."""
        result = calculate_profit_erosion(returned_items_df)
        assert list(result["margin_reversal"]) == [60.0, 20.0, 40.0, 30.0]

    def test_creates_process_cost_column(self, returned_items_df):
        """Test that process_cost column is created."""
        result = calculate_profit_erosion(returned_items_df)
        assert "process_cost" in result.columns

    def test_process_cost_uses_default_base_cost(self, returned_items_df):
        """Test process_cost uses default $12 base cost with category multipliers."""
        result = calculate_profit_erosion(returned_items_df, use_category_tiers=False)
        expected_cost = sum(DEFAULT_COST_COMPONENTS.values())  # $12
        assert all(result["process_cost"] == expected_cost)

    def test_process_cost_applies_category_multipliers(self, returned_items_df):
        """Test process_cost applies category tier multipliers."""
        result = calculate_profit_erosion(returned_items_df, use_category_tiers=True)
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())  # $12

        # Jeans = Premium (1.3x)
        assert result["process_cost"].iloc[0] == base_cost * 1.3
        # Tops & Tees = Standard (1.0x)
        assert result["process_cost"].iloc[1] == base_cost * 1.0
        # Outerwear & Coats = Premium (1.3x)
        assert result["process_cost"].iloc[2] == base_cost * 1.3
        # Socks = Standard (1.0x)
        assert result["process_cost"].iloc[3] == base_cost * 1.0

    def test_process_cost_uses_custom_components(self, returned_items_df):
        """Test process_cost uses custom components when provided."""
        custom_costs = {"handling": 10.0, "shipping": 5.0}
        result = calculate_profit_erosion(
            returned_items_df,
            cost_components=custom_costs,
            use_category_tiers=False,
        )
        assert all(result["process_cost"] == 15.0)

    def test_process_cost_uses_custom_multipliers(self, returned_items_df):
        """Test process_cost uses custom category multipliers when provided."""
        custom_multipliers = {"Jeans": 2.0, "Tops & Tees": 0.5}
        result = calculate_profit_erosion(
            returned_items_df,
            category_multipliers=custom_multipliers,
            use_category_tiers=True,
        )
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())

        # Jeans = 2.0x (custom)
        assert result["process_cost"].iloc[0] == base_cost * 2.0
        # Tops & Tees = 0.5x (custom)
        assert result["process_cost"].iloc[1] == base_cost * 0.5
        # Outerwear & Coats = 1.0x (default, not in custom)
        assert result["process_cost"].iloc[2] == base_cost * DEFAULT_CATEGORY_MULTIPLIER
        # Socks = 1.0x (default, not in custom)
        assert result["process_cost"].iloc[3] == base_cost * DEFAULT_CATEGORY_MULTIPLIER

    def test_creates_profit_erosion_column(self, returned_items_df):
        """Test that profit_erosion column is created."""
        result = calculate_profit_erosion(returned_items_df)
        assert "profit_erosion" in result.columns

    def test_profit_erosion_equals_margin_plus_cost(self, returned_items_df):
        """Test profit_erosion = margin_reversal + process_cost."""
        result = calculate_profit_erosion(returned_items_df, use_category_tiers=False)
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())

        for i in range(len(returned_items_df)):
            expected = returned_items_df["item_margin"].iloc[i] + base_cost
            assert result["profit_erosion"].iloc[i] == expected

    def test_does_not_modify_original_dataframe(self, returned_items_df):
        """Test that the original DataFrame is not modified."""
        original_cols = list(returned_items_df.columns)
        calculate_profit_erosion(returned_items_df)
        assert list(returned_items_df.columns) == original_cols

    def test_handles_unknown_category(self):
        """Test that unknown categories use default multiplier."""
        df = pd.DataFrame({
            "order_item_id": [1],
            "item_margin": [50.0],
            "category": ["Unknown Category"],
        })
        result = calculate_profit_erosion(df, use_category_tiers=True)
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())
        assert result["process_cost"].iloc[0] == base_cost * DEFAULT_CATEGORY_MULTIPLIER

    def test_handles_missing_category_column(self):
        """Test that missing category column falls back to flat rate."""
        df = pd.DataFrame({
            "order_item_id": [1, 2],
            "item_margin": [50.0, 30.0],
        })
        result = calculate_profit_erosion(df, use_category_tiers=True)
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())
        assert all(result["process_cost"] == base_cost)


class TestCategoryTierMultipliers:
    """Test cases for the CATEGORY_TIER_MULTIPLIERS constant."""

    def test_premium_tier_categories_have_correct_multiplier(self):
        """Test that premium tier categories have 1.3x multiplier."""
        premium_categories = [
            "Outerwear & Coats", "Suits & Sport Coats", "Blazers & Jackets",
            "Jeans", "Dresses", "Suits", "Sweaters", "Pants",
        ]
        for cat in premium_categories:
            assert CATEGORY_TIER_MULTIPLIERS.get(cat) == 1.3, f"{cat} should be 1.3x"

    def test_moderate_tier_categories_have_correct_multiplier(self):
        """Test that moderate tier categories have 1.15x multiplier."""
        moderate_categories = [
            "Skirts", "Active", "Swim", "Maternity", "Sleep & Lounge",
            "Accessories", "Pants & Capris", "Fashion Hoodies & Sweatshirts", "Shorts",
        ]
        for cat in moderate_categories:
            assert CATEGORY_TIER_MULTIPLIERS.get(cat) == 1.15, f"{cat} should be 1.15x"

    def test_standard_tier_categories_have_correct_multiplier(self):
        """Test that standard tier categories have 1.0x multiplier."""
        standard_categories = [
            "Plus", "Tops & Tees", "Intimates", "Underwear", "Leggings",
            "Socks & Hosiery", "Socks", "Jumpsuits & Rompers", "Clothing Sets",
        ]
        for cat in standard_categories:
            assert CATEGORY_TIER_MULTIPLIERS.get(cat) == 1.0, f"{cat} should be 1.0x"

    def test_default_multiplier_is_one(self):
        """Test that default category multiplier is 1.0."""
        assert DEFAULT_CATEGORY_MULTIPLIER == 1.0


class TestAggregateProfitErosionByOrder:
    """Test cases for the aggregate_profit_erosion_by_order function.

    Note: Function expects pre-filtered returned items only.
    """

    @pytest.fixture
    def returned_items_df(self):
        """Create sample DataFrame of RETURNED items only (pre-filtered)."""
        return pd.DataFrame({
            "order_id": [1, 2, 2],
            "order_item_id": [1, 3, 4],
            "user_id": [100, 200, 200],
            "sale_price": [100.0, 80.0, 60.0],
            "item_margin": [60.0, 40.0, 30.0],
            "margin_reversal": [60.0, 40.0, 30.0],
            "process_cost": [15.6, 15.6, 12.0],  # Different category costs
            "profit_erosion": [75.6, 55.6, 42.0],
        })

    def test_produces_one_row_per_order(self, returned_items_df):
        """Test that result has one row per order."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        assert len(result) == 2

    def test_aggregates_returned_items(self, returned_items_df):
        """Test returned_items is correctly aggregated."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["returned_items"] == 1
        assert order2["returned_items"] == 2

    def test_aggregates_total_profit_erosion(self, returned_items_df):
        """Test total_profit_erosion is correctly summed."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["total_profit_erosion"] == 75.6
        assert abs(order2["total_profit_erosion"] - 97.6) < 0.01  # 55.6 + 42.0

    def test_aggregates_total_margin_reversal(self, returned_items_df):
        """Test total_margin_reversal is correctly summed."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["total_margin_reversal"] == 60.0
        assert order2["total_margin_reversal"] == 70.0  # 40 + 30

    def test_aggregates_total_process_cost(self, returned_items_df):
        """Test total_process_cost is correctly summed."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["total_process_cost"] == 15.6
        assert abs(order2["total_process_cost"] - 27.6) < 0.01  # 15.6 + 12.0

    def test_includes_user_id(self, returned_items_df):
        """Test user_id is included in aggregation."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        assert "user_id" in result.columns
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["user_id"] == 100
        assert order2["user_id"] == 200

    def test_calculates_avg_margin_per_return(self, returned_items_df):
        """Test avg_margin_per_return is correctly calculated."""
        result = aggregate_profit_erosion_by_order(returned_items_df)
        order1 = result[result["order_id"] == 1].iloc[0]
        order2 = result[result["order_id"] == 2].iloc[0]
        assert order1["avg_margin_per_return"] == 60.0  # 60 / 1
        assert order2["avg_margin_per_return"] == 35.0  # 70 / 2


class TestAggregateProfitErosionByCustomer:
    """Test cases for the aggregate_profit_erosion_by_customer function.

    Note: Function expects pre-filtered returned items only.
    """

    @pytest.fixture
    def returned_items_df(self):
        """Create sample DataFrame of RETURNED items only (pre-filtered)."""
        return pd.DataFrame({
            "order_id": [1, 2, 3],
            "order_item_id": [1, 3, 5],
            "user_id": [100, 100, 200],
            "sale_price": [100.0, 80.0, 40.0],
            "item_margin": [60.0, 40.0, 20.0],
            "margin_reversal": [60.0, 40.0, 20.0],
            "process_cost": [15.6, 12.0, 12.0],
            "profit_erosion": [75.6, 52.0, 32.0],
        })

    def test_produces_one_row_per_customer(self, returned_items_df):
        """Test that result has one row per customer."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        assert len(result) == 2

    def test_aggregates_total_orders(self, returned_items_df):
        """Test total_orders counts unique orders with returns."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["total_orders"] == 2  # Orders 1 and 2
        assert cust200["total_orders"] == 1  # Order 3

    def test_aggregates_returned_items(self, returned_items_df):
        """Test returned_items is correctly aggregated."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["returned_items"] == 2
        assert cust200["returned_items"] == 1

    def test_aggregates_total_profit_erosion(self, returned_items_df):
        """Test total_profit_erosion is correctly summed."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert abs(cust100["total_profit_erosion"] - 127.6) < 0.01  # 75.6 + 52.0
        assert cust200["total_profit_erosion"] == 32.0

    def test_aggregates_total_margin_reversal(self, returned_items_df):
        """Test total_margin_reversal is correctly summed."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["total_margin_reversal"] == 100.0  # 60 + 40
        assert cust200["total_margin_reversal"] == 20.0

    def test_aggregates_total_process_cost(self, returned_items_df):
        """Test total_process_cost is correctly summed."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert abs(cust100["total_process_cost"] - 27.6) < 0.01  # 15.6 + 12.0
        assert cust200["total_process_cost"] == 12.0

    def test_calculates_avg_erosion_per_return(self, returned_items_df):
        """Test avg_erosion_per_return is correctly calculated."""
        result = aggregate_profit_erosion_by_customer(returned_items_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert abs(cust100["avg_erosion_per_return"] - 63.8) < 0.01  # 127.6 / 2
        assert cust200["avg_erosion_per_return"] == 32.0  # 32 / 1


class TestSummarizeProfitErosion:
    """Test cases for the summarize_profit_erosion function.

    Note: Function expects pre-filtered returned items only.
    """

    @pytest.fixture
    def returned_items_df(self):
        """Create sample DataFrame of RETURNED items only (pre-filtered)."""
        return pd.DataFrame({
            "item_margin": [60.0, 40.0],
            "margin_reversal": [60.0, 40.0],
            "process_cost": [15.6, 12.0],
            "profit_erosion": [75.6, 52.0],
        })

    def test_returns_dict_with_expected_keys(self, returned_items_df):
        """Test that summary returns all expected keys."""
        result = summarize_profit_erosion(returned_items_df)
        expected_keys = {
            "total_returned",
            "total_margin_reversal",
            "avg_margin_per_return",
            "median_margin_per_return",
            "total_process_costs",
            "total_profit_erosion",
            "max_single_margin_loss",
            "avg_erosion_per_return",
        }
        assert set(result.keys()) == expected_keys

    def test_calculates_total_returned(self, returned_items_df):
        """Test that total_returned counts all rows."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["total_returned"] == 2

    def test_calculates_margin_reversal(self, returned_items_df):
        """Test margin reversal sum is correct."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["total_margin_reversal"] == 100.0  # 60 + 40

    def test_calculates_process_costs_from_column(self, returned_items_df):
        """Test process costs are summed from process_cost column."""
        result = summarize_profit_erosion(returned_items_df)
        assert abs(result["total_process_costs"] - 27.6) < 0.01  # 15.6 + 12.0

    def test_calculates_total_profit_erosion_from_column(self, returned_items_df):
        """Test total profit erosion is summed from profit_erosion column."""
        result = summarize_profit_erosion(returned_items_df)
        assert abs(result["total_profit_erosion"] - 127.6) < 0.01  # 75.6 + 52.0

    def test_calculates_avg_margin_per_return(self, returned_items_df):
        """Test avg_margin_per_return is correctly calculated."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["avg_margin_per_return"] == 50.0  # (60 + 40) / 2

    def test_calculates_median_margin_per_return(self, returned_items_df):
        """Test median_margin_per_return is correctly calculated."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["median_margin_per_return"] == 50.0  # median of [60, 40]

    def test_calculates_max_single_margin_loss(self, returned_items_df):
        """Test max_single_margin_loss is correctly calculated."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["max_single_margin_loss"] == 60.0

    def test_calculates_avg_erosion_per_return(self, returned_items_df):
        """Test avg_erosion_per_return is correctly calculated."""
        result = summarize_profit_erosion(returned_items_df)
        assert abs(result["avg_erosion_per_return"] - 63.8) < 0.01  # 127.6 / 2

    def test_handles_empty_dataframe(self):
        """Test handling of empty DataFrame (no returned items)."""
        df = pd.DataFrame({
            "item_margin": pd.Series([], dtype=float),
        })
        result = summarize_profit_erosion(df)
        assert result["total_returned"] == 0
        assert result["total_margin_reversal"] == 0.0
        assert result["total_profit_erosion"] == 0.0
        assert result["avg_erosion_per_return"] == 0.0

    def test_calculates_without_precomputed_columns(self):
        """Test calculation when erosion columns are not precomputed."""
        df = pd.DataFrame({
            "item_margin": [60.0, 40.0, 30.0],
        })
        result = summarize_profit_erosion(df)
        base_cost = sum(DEFAULT_COST_COMPONENTS.values())  # $12
        expected_margin = 130.0  # 60 + 40 + 30
        expected_process = 3 * base_cost  # 3 returns * $12
        expected_total = expected_margin + expected_process

        assert result["total_returned"] == 3
        assert result["total_margin_reversal"] == expected_margin
        assert result["total_process_costs"] == expected_process
        assert result["total_profit_erosion"] == expected_total

    def test_uses_custom_cost_components(self):
        """Test that custom cost components are used when columns missing."""
        df = pd.DataFrame({
            "item_margin": [60.0, 40.0],
        })
        custom_costs = {"handling": 20.0}
        result = summarize_profit_erosion(df, cost_components=custom_costs)
        assert result["total_process_costs"] == 40.0  # 2 returns * $20


# =============================================================================
# Task 2: Customer Behavioral Features Tests
# =============================================================================


class TestEngineerCustomerBehavioralFeatures:
    """Test cases for the engineer_customer_behavioral_features function."""

    @pytest.fixture
    def sample_item_df(self):
        """Create sample item-level DataFrame with customer data."""
        return pd.DataFrame({
            "user_id": [100, 100, 100, 100, 200, 200],
            "order_id": [1, 1, 2, 2, 3, 3],
            "order_item_id": [1, 2, 3, 4, 5, 6],
            "sale_price": [50.0, 30.0, 80.0, 20.0, 100.0, 60.0],
            "item_margin": [25.0, 15.0, 40.0, 10.0, 50.0, 30.0],
            "is_returned_item": [1, 0, 0, 1, 0, 0],
            "order_created_at": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-15", "2024-01-15",
                "2024-01-10", "2024-01-10",
            ]),
            "user_created_at": pd.to_datetime([
                "2023-06-01", "2023-06-01", "2023-06-01", "2023-06-01",
                "2023-12-01", "2023-12-01",
            ]),
        })

    def test_produces_one_row_per_customer(self, sample_item_df):
        """Test that result has one row per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        assert len(result) == 2

    def test_includes_user_id_column(self, sample_item_df):
        """Test that user_id is included in the result."""
        result = engineer_customer_behavioral_features(sample_item_df)
        assert "user_id" in result.columns
        assert set(result["user_id"]) == {100, 200}

    def test_calculates_order_frequency(self, sample_item_df):
        """Test order_frequency counts unique orders per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["order_frequency"] == 2  # Orders 1 and 2
        assert cust200["order_frequency"] == 1  # Order 3

    def test_calculates_return_frequency(self, sample_item_df):
        """Test return_frequency counts return events per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["return_frequency"] == 2  # 2 returned items
        assert cust200["return_frequency"] == 0  # No returns

    def test_calculates_customer_return_rate(self, sample_item_df):
        """Test customer_return_rate is return_frequency / total_items."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["customer_return_rate"] == 0.5  # 2/4 items
        assert cust200["customer_return_rate"] == 0.0  # 0/2 items

    def test_calculates_avg_basket_size(self, sample_item_df):
        """Test avg_basket_size is total_items / order_frequency."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["avg_basket_size"] == 2.0  # 4 items / 2 orders
        assert cust200["avg_basket_size"] == 2.0  # 2 items / 1 order

    def test_calculates_avg_order_value(self, sample_item_df):
        """Test avg_order_value is mean of order totals."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        # Cust 100: Order 1 = 80, Order 2 = 100, avg = 90
        assert cust100["avg_order_value"] == 90.0
        # Cust 200: Order 3 = 160
        assert cust200["avg_order_value"] == 160.0

    def test_calculates_total_items(self, sample_item_df):
        """Test total_items counts items per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["total_items"] == 4
        assert cust200["total_items"] == 2

    def test_calculates_total_sales(self, sample_item_df):
        """Test total_sales sums sale_price per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["total_sales"] == 180.0  # 50+30+80+20
        assert cust200["total_sales"] == 160.0  # 100+60

    def test_calculates_total_margin(self, sample_item_df):
        """Test total_margin sums item_margin per customer."""
        result = engineer_customer_behavioral_features(sample_item_df)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        assert cust100["total_margin"] == 90.0  # 25+15+40+10
        assert cust200["total_margin"] == 80.0  # 50+30

    def test_calculates_purchase_recency_days(self, sample_item_df):
        """Test purchase_recency_days from reference date."""
        ref_date = pd.Timestamp("2024-01-20")
        result = engineer_customer_behavioral_features(sample_item_df, reference_date=ref_date)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        # Cust 100: Last order 2024-01-15, recency = 5 days
        assert cust100["purchase_recency_days"] == 5
        # Cust 200: Last order 2024-01-10, recency = 10 days
        assert cust200["purchase_recency_days"] == 10

    def test_calculates_customer_tenure_days(self, sample_item_df):
        """Test customer_tenure_days from user_created_at."""
        ref_date = pd.Timestamp("2024-01-20")
        result = engineer_customer_behavioral_features(sample_item_df, reference_date=ref_date)
        cust100 = result[result["user_id"] == 100].iloc[0]
        cust200 = result[result["user_id"] == 200].iloc[0]
        # Cust 100: Created 2023-06-01, tenure = 233 days
        assert cust100["customer_tenure_days"] == 233
        # Cust 200: Created 2023-12-01, tenure = 50 days
        assert cust200["customer_tenure_days"] == 50

    def test_uses_max_date_as_default_reference(self, sample_item_df):
        """Test that max order date is used when reference_date not specified."""
        result = engineer_customer_behavioral_features(sample_item_df)
        # Max date in data is 2024-01-15
        cust100 = result[result["user_id"] == 100].iloc[0]
        # Cust 100's last order is 2024-01-15, so recency = 0
        assert cust100["purchase_recency_days"] == 0

    def test_does_not_modify_original_dataframe(self, sample_item_df):
        """Test that the original DataFrame is not modified."""
        original_cols = list(sample_item_df.columns)
        engineer_customer_behavioral_features(sample_item_df)
        assert list(sample_item_df.columns) == original_cols

    def test_expected_output_columns(self, sample_item_df):
        """Test that all expected columns are present in output."""
        result = engineer_customer_behavioral_features(sample_item_df)
        expected_cols = {
            "user_id",
            "order_frequency",
            "return_frequency",
            "customer_return_rate",
            "avg_basket_size",
            "avg_order_value",
            "customer_tenure_days",
            "purchase_recency_days",
            "total_items",
            "total_sales",
            "total_margin",
            "avg_item_price",
            "avg_item_margin",
        }
        assert expected_cols.issubset(set(result.columns))


# =============================================================================
# Save Feature-Engineered Dataset Tests
# =============================================================================


class TestSaveFeatureEngineeredDataset:
    """Test cases for the save_feature_engineered_dataset function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample feature-engineered DataFrame."""
        return pd.DataFrame({
            "order_item_id": [1, 2, 3],
            "sale_price": [50.0, 75.0, 100.0],
            "item_margin": [25.0, 35.0, 50.0],
            "is_returned_item": [0, 1, 0],
        })

    def test_saves_parquet_file(self, sample_df, tmp_path):
        """Test that parquet file is saved correctly."""
        parquet_path, _ = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path, save_csv=False
        )
        assert parquet_path is not None
        assert parquet_path.exists()
        assert parquet_path.suffix == ".parquet"

    def test_saves_csv_file(self, sample_df, tmp_path):
        """Test that CSV file is saved correctly."""
        _, csv_path = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path, save_parquet=False
        )
        assert csv_path is not None
        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

    def test_saves_both_formats(self, sample_df, tmp_path):
        """Test that both parquet and CSV files are saved."""
        parquet_path, csv_path = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path
        )
        assert parquet_path is not None and parquet_path.exists()
        assert csv_path is not None and csv_path.exists()

    def test_parquet_contents_match(self, sample_df, tmp_path):
        """Test that parquet file contains correct data."""
        parquet_path, _ = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path, save_csv=False
        )
        loaded_df = pd.read_parquet(parquet_path)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_csv_contents_match(self, sample_df, tmp_path):
        """Test that CSV file contains correct data."""
        _, csv_path = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path, save_parquet=False
        )
        loaded_df = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

    def test_creates_output_directory(self, sample_df, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_subdir" / "another_level"
        parquet_path, csv_path = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=new_dir
        )
        assert new_dir.exists()
        assert parquet_path.exists()
        assert csv_path.exists()

    def test_returns_none_when_format_disabled(self, sample_df, tmp_path):
        """Test that None is returned when a format is disabled."""
        parquet_path, csv_path = save_feature_engineered_dataset(
            sample_df, filename="test_data", output_dir=tmp_path,
            save_parquet=False, save_csv=True
        )
        assert parquet_path is None
        assert csv_path is not None

        parquet_path2, csv_path2 = save_feature_engineered_dataset(
            sample_df, filename="test_data2", output_dir=tmp_path,
            save_parquet=True, save_csv=False
        )
        assert parquet_path2 is not None
        assert csv_path2 is None

    def test_uses_custom_filename(self, sample_df, tmp_path):
        """Test that custom filename is used correctly."""
        parquet_path, csv_path = save_feature_engineered_dataset(
            sample_df, filename="my_custom_dataset", output_dir=tmp_path
        )
        assert parquet_path.name == "my_custom_dataset.parquet"
        assert csv_path.name == "my_custom_dataset.csv"


# =============================================================================
# Task 5: Target Variables for Predictive Modeling Tests
# =============================================================================


class TestCreateProfitErosionTargets:
    """Test cases for the create_profit_erosion_targets function."""

    @pytest.fixture
    def customer_erosion_df(self):
        """Create sample customer-level profit erosion DataFrame."""
        return pd.DataFrame({
            "user_id": [100, 101, 102, 103, 104, 105, 106, 107],
            "total_profit_erosion": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "returned_items": [1, 2, 3, 4, 5, 6, 7, 8],
        })

    def test_creates_is_high_erosion_customer_column(self, customer_erosion_df):
        """Test that is_high_erosion_customer column is created."""
        result = create_profit_erosion_targets(customer_erosion_df)
        assert "is_high_erosion_customer" in result.columns

    def test_creates_profit_erosion_quartile_column(self, customer_erosion_df):
        """Test that profit_erosion_quartile column is created."""
        result = create_profit_erosion_targets(customer_erosion_df)
        assert "profit_erosion_quartile" in result.columns

    def test_creates_erosion_percentile_rank_column(self, customer_erosion_df):
        """Test that erosion_percentile_rank column is created."""
        result = create_profit_erosion_targets(customer_erosion_df)
        assert "erosion_percentile_rank" in result.columns

    def test_binary_flag_uses_75th_percentile_by_default(self, customer_erosion_df):
        """Test that default threshold is 75th percentile."""
        result = create_profit_erosion_targets(customer_erosion_df)
        # With 8 values [10,20,30,40,50,60,70,80], 75th percentile is 62.5
        # Only values > 62.5 (i.e., 70, 80) should be flagged
        high_erosion = result[result["is_high_erosion_customer"] == 1]
        assert len(high_erosion) == 2
        assert set(high_erosion["user_id"]) == {106, 107}

    def test_binary_flag_uses_custom_percentile(self, customer_erosion_df):
        """Test that custom percentile threshold is respected."""
        result = create_profit_erosion_targets(
            customer_erosion_df, high_erosion_percentile=0.5
        )
        # 50th percentile of [10,20,30,40,50,60,70,80] is 45
        # Values > 45 (i.e., 50, 60, 70, 80) should be flagged
        high_erosion = result[result["is_high_erosion_customer"] == 1]
        assert len(high_erosion) == 4
        assert set(high_erosion["user_id"]) == {104, 105, 106, 107}

    def test_quartiles_assigned_correctly(self, customer_erosion_df):
        """Test that quartile assignments are correct."""
        result = create_profit_erosion_targets(customer_erosion_df)
        # Quartile 1 (lowest): 10, 20
        # Quartile 2: 30, 40
        # Quartile 3: 50, 60
        # Quartile 4 (highest): 70, 80
        q1 = result[result["profit_erosion_quartile"] == 1]["user_id"].tolist()
        q4 = result[result["profit_erosion_quartile"] == 4]["user_id"].tolist()
        assert 100 in q1 and 101 in q1  # Lowest erosion
        assert 106 in q4 and 107 in q4  # Highest erosion

    def test_percentile_rank_ranges_0_to_100(self, customer_erosion_df):
        """Test that percentile ranks are between 0 and 100."""
        result = create_profit_erosion_targets(customer_erosion_df)
        assert result["erosion_percentile_rank"].min() >= 0
        assert result["erosion_percentile_rank"].max() <= 100

    def test_does_not_modify_original_dataframe(self, customer_erosion_df):
        """Test that the original DataFrame is not modified."""
        original_cols = list(customer_erosion_df.columns)
        create_profit_erosion_targets(customer_erosion_df)
        assert list(customer_erosion_df.columns) == original_cols

    def test_raises_error_for_missing_column(self, customer_erosion_df):
        """Test that ValueError is raised if erosion column is missing."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            create_profit_erosion_targets(
                customer_erosion_df, erosion_column="nonexistent"
            )

    def test_raises_error_for_invalid_percentile(self, customer_erosion_df):
        """Test that ValueError is raised for invalid percentile values."""
        with pytest.raises(ValueError, match="high_erosion_percentile must be between"):
            create_profit_erosion_targets(
                customer_erosion_df, high_erosion_percentile=1.5
            )
        with pytest.raises(ValueError, match="high_erosion_percentile must be between"):
            create_profit_erosion_targets(
                customer_erosion_df, high_erosion_percentile=-0.1
            )

    def test_uses_custom_erosion_column(self):
        """Test that custom erosion column name is used."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4],
            "custom_erosion": [10.0, 20.0, 30.0, 40.0],
        })
        result = create_profit_erosion_targets(df, erosion_column="custom_erosion")
        # Should use custom_erosion for calculations, not fail
        assert "is_high_erosion_customer" in result.columns
        assert "profit_erosion_quartile" in result.columns

    def test_handles_edge_case_few_unique_values(self):
        """Test handling when there are few unique erosion values."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4],
            "total_profit_erosion": [10.0, 10.0, 10.0, 40.0],
        })
        # Should not raise an error, quartiles handle duplicates
        result = create_profit_erosion_targets(df)
        assert "is_high_erosion_customer" in result.columns
