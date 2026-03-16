"""
Unit tests for the analytics module.

Note: Renamed from test_modeling.py to match the module rename.
"""

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    calculate_brand_return_rates,
    calculate_category_return_rates,
    calculate_customer_margin_exposure,
    calculate_margin_loss_by_group,
    calculate_price_margin_returned_by_country,
    calculate_return_rates_by_group,
    engineer_product_level_features,
    engineer_temporal_features,
    generate_feature_quality_report,
    segment_customers_by_return_behavior,
    validate_feature_quality,
)
from src.feature_engineering import summarize_profit_erosion


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
    """Test cases for summarize_profit_erosion function.

    Note: This function now lives in feature_engineering.py and expects
    pre-filtered returned items only. The function signature changed from
    the old analytics.py version.
    """

    @pytest.fixture
    def returned_items_df(self, sample_merged_df):
        """Filter sample_merged_df to returned items only."""
        return sample_merged_df[sample_merged_df["is_returned_item"] == 1].copy()

    def test_returns_expected_keys(self, returned_items_df):
        """Test that summary contains all expected keys."""
        result = summarize_profit_erosion(returned_items_df)
        expected_keys = [
            "total_returned",
            "total_margin_reversal",
            "avg_margin_per_return",
            "median_margin_per_return",
            "total_process_costs",
            "total_profit_erosion",
            "max_single_margin_loss",
            "avg_erosion_per_return",
        ]
        for key in expected_keys:
            assert key in result

    def test_calculates_total_returned(self, returned_items_df):
        """Test that total returned count is correct."""
        result = summarize_profit_erosion(returned_items_df)
        assert result["total_returned"] == len(returned_items_df)

    def test_includes_process_costs(self, returned_items_df):
        """Test that process costs are included in total erosion."""
        # Default cost components sum to $12
        result = summarize_profit_erosion(returned_items_df)
        # Total erosion = margin reversal + process costs
        expected_erosion = result["total_margin_reversal"] + result["total_process_costs"]
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
        user_1001 = result[result["user_id"] == 1001].iloc[0]
        user_1002 = result[result["user_id"] == 1002].iloc[0]
        user_1003 = result[result["user_id"] == 1003].iloc[0]
        assert user_1001["return_segment"] == "high_returner"
        assert user_1002["return_segment"] == "no_returns"
        assert user_1003["return_segment"] == "no_returns"

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


# =============================================================================
# Task 3: Product-Level Features Tests
# =============================================================================


class TestCalculateCategoryReturnRates:
    """Test cases for calculate_category_return_rates function."""

    def test_calculates_return_rate_by_category(self, sample_merged_df):
        """Test return rate calculation by category."""
        result = calculate_category_return_rates(sample_merged_df, min_rows=1)
        assert "return_rate" in result.columns
        assert "category" in result.index.names or result.index.name == "category"

    def test_uses_calculate_return_rates_by_group(self, sample_merged_df):
        """Test that it delegates to calculate_return_rates_by_group."""
        result = calculate_category_return_rates(sample_merged_df, min_rows=1)
        expected = calculate_return_rates_by_group(
            sample_merged_df, ["category"], min_rows=1
        )
        pd.testing.assert_frame_equal(result, expected)


class TestCalculateBrandReturnRates:
    """Test cases for calculate_brand_return_rates function."""

    def test_calculates_return_rate_by_brand(self):
        """Test return rate calculation by brand."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3, 4],
            "brand": ["Nike", "Nike", "Adidas", "Adidas"],
            "is_returned_item": [1, 0, 1, 1],
        })
        result = calculate_brand_return_rates(df, min_rows=1)
        assert "return_rate" in result.columns
        # Nike: 1/2 = 0.5, Adidas: 2/2 = 1.0
        assert result.loc["Adidas", "return_rate"] == 1.0
        assert result.loc["Nike", "return_rate"] == 0.5


class TestEngineerProductLevelFeatures:
    """Test cases for engineer_product_level_features function."""

    @pytest.fixture
    def product_df(self):
        """Create sample DataFrame for product feature testing."""
        return pd.DataFrame({
            "order_id": [1, 2, 3, 4, 5, 6],
            "category": ["Jeans", "Jeans", "Tops", "Tops", "Jeans", "Tops"],
            "brand": ["Nike", "Nike", "Adidas", "Adidas", "Nike", "Adidas"],
            "sale_price": [100.0, 50.0, 30.0, 80.0, 120.0, 40.0],
            "is_returned_item": [1, 0, 1, 0, 0, 0],
        })

    def test_creates_category_return_rate_column(self, product_df):
        """Test that category_return_rate column is created."""
        result = engineer_product_level_features(product_df, min_rows=1)
        assert "category_return_rate" in result.columns

    def test_creates_brand_return_rate_column(self, product_df):
        """Test that brand_return_rate column is created."""
        result = engineer_product_level_features(product_df, min_rows=1)
        assert "brand_return_rate" in result.columns

    def test_creates_price_tier_column(self, product_df):
        """Test that price_tier column is created."""
        result = engineer_product_level_features(product_df, min_rows=1)
        assert "price_tier" in result.columns
        assert set(result["price_tier"].dropna().unique()).issubset(
            {"low", "medium", "high"}
        )

    def test_maps_category_return_rate_correctly(self, product_df):
        """Test that category return rate is mapped to items."""
        result = engineer_product_level_features(product_df, min_rows=1)
        # Jeans: 1 return / 3 items = 0.333...
        jeans_rate = result[result["category"] == "Jeans"]["category_return_rate"].iloc[0]
        assert abs(jeans_rate - 1/3) < 0.01
        # Tops: 1 return / 3 items = 0.333...
        tops_rate = result[result["category"] == "Tops"]["category_return_rate"].iloc[0]
        assert abs(tops_rate - 1/3) < 0.01

    def test_does_not_modify_original_dataframe(self, product_df):
        """Test that the original DataFrame is not modified."""
        original_cols = list(product_df.columns)
        engineer_product_level_features(product_df, min_rows=1)
        assert list(product_df.columns) == original_cols


# =============================================================================
# Task 4: Temporal Features Tests
# =============================================================================


class TestEngineerTemporalFeatures:
    """Test cases for engineer_temporal_features function."""

    @pytest.fixture
    def temporal_df(self):
        """Create sample DataFrame for temporal feature testing."""
        return pd.DataFrame({
            "order_id": [1, 2, 3, 4],
            "order_created_at": pd.to_datetime([
                "2024-01-15",  # Monday, January (Winter)
                "2024-03-23",  # Saturday, March (Spring)
                "2024-07-04",  # Thursday, July (Summer)
                "2024-10-31",  # Thursday, October (Fall)
            ]),
            "item_delivered_at": pd.to_datetime([
                "2024-01-20",  # 5 days later
                "2024-03-28",  # 5 days later
                "2024-07-10",  # 6 days later
                "2024-11-05",  # 5 days later
            ]),
            "item_returned_at": pd.to_datetime([
                "2024-01-25",  # 5 days after delivery
                pd.NaT,        # Not returned
                "2024-07-15",  # 5 days after delivery
                pd.NaT,        # Not returned
            ]),
        })

    def test_creates_day_of_week_column(self, temporal_df):
        """Test that order_day_of_week column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "order_day_of_week" in result.columns
        # 2024-01-15 is Monday (0)
        assert result["order_day_of_week"].iloc[0] == 0
        # 2024-03-23 is Saturday (5)
        assert result["order_day_of_week"].iloc[1] == 5

    def test_creates_month_column(self, temporal_df):
        """Test that order_month column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "order_month" in result.columns
        assert list(result["order_month"]) == [1, 3, 7, 10]

    def test_creates_quarter_column(self, temporal_df):
        """Test that order_quarter column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "order_quarter" in result.columns
        assert list(result["order_quarter"]) == [1, 1, 3, 4]

    def test_creates_year_column(self, temporal_df):
        """Test that order_year column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "order_year" in result.columns
        assert all(result["order_year"] == 2024)

    def test_creates_is_weekend_order_column(self, temporal_df):
        """Test that is_weekend_order column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "is_weekend_order" in result.columns
        # Only the Saturday order should be weekend
        assert list(result["is_weekend_order"]) == [False, True, False, False]

    def test_creates_season_column(self, temporal_df):
        """Test that season column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "season" in result.columns
        assert list(result["season"]) == ["winter", "spring", "summer", "fall"]

    def test_creates_days_to_delivery_column(self, temporal_df):
        """Test that days_to_delivery column is created."""
        result = engineer_temporal_features(temporal_df)
        assert "days_to_delivery" in result.columns
        assert list(result["days_to_delivery"]) == [5, 5, 6, 5]

    def test_creates_days_to_return_column(self, temporal_df):
        """Test that days_to_return column is created for returned items."""
        result = engineer_temporal_features(temporal_df)
        assert "days_to_return" in result.columns
        # First item: 5 days after delivery
        assert result["days_to_return"].iloc[0] == 5
        # Second item: not returned, should be NaN
        assert pd.isna(result["days_to_return"].iloc[1])

    def test_does_not_modify_original_dataframe(self, temporal_df):
        """Test that the original DataFrame is not modified."""
        original_cols = list(temporal_df.columns)
        engineer_temporal_features(temporal_df)
        assert list(temporal_df.columns) == original_cols

    def test_handles_missing_date_columns_gracefully(self):
        """Test that function handles missing date columns gracefully."""
        df = pd.DataFrame({
            "order_id": [1, 2],
            "sale_price": [100.0, 200.0],
        })
        result = engineer_temporal_features(df)
        # Should return DataFrame without temporal columns if dates missing
        assert "order_day_of_week" not in result.columns


# =============================================================================
# Task 6: Feature Quality Validation Tests
# =============================================================================


class TestValidateFeatureQuality:
    """Test cases for validate_feature_quality function."""

    @pytest.fixture
    def quality_df(self):
        """Create sample DataFrame for quality validation testing."""
        return pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "feature_c": [1.0, np.nan, 3.0, np.nan, 5.0],  # Has missing values
            "feature_d": [1.0, 1.0, 1.0, 1.0, 1.0],  # Constant (zero variance)
            "category": ["A", "B", "C", "D", "E"],  # Non-numeric
        })

    def test_returns_dict_with_expected_keys(self, quality_df):
        """Test that validation returns dict with expected keys."""
        result = validate_feature_quality(quality_df)
        expected_keys = [
            "total_rows",
            "total_features",
            "missing_values",
            "missing_pct",
            "distribution_stats",
            "correlation_matrix",
            "high_correlations",
            "constant_columns",
            "low_variance_columns",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_counts_missing_values(self, quality_df):
        """Test that missing values are correctly counted."""
        result = validate_feature_quality(quality_df)
        assert "feature_c" in result["missing_values"]
        assert result["missing_values"]["feature_c"] == 2

    def test_calculates_missing_percentage(self, quality_df):
        """Test that missing percentage is correctly calculated."""
        result = validate_feature_quality(quality_df)
        assert "feature_c" in result["missing_pct"]
        assert result["missing_pct"]["feature_c"] == 40.0  # 2/5 = 40%

    def test_calculates_distribution_stats(self, quality_df):
        """Test that distribution statistics are calculated."""
        result = validate_feature_quality(quality_df)
        assert "feature_a" in result["distribution_stats"]
        stats = result["distribution_stats"]["feature_a"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["mean"] == 3.0  # Mean of [1,2,3,4,5]

    def test_identifies_constant_columns(self, quality_df):
        """Test that constant columns are identified."""
        result = validate_feature_quality(quality_df)
        assert "feature_d" in result["constant_columns"]

    def test_identifies_high_correlations(self):
        """Test that high correlations are identified."""
        # Create DataFrame with highly correlated features
        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfectly correlated with a
            "feature_c": [5.0, 4.0, 3.0, 2.0, 1.0],   # Negatively correlated with a
        })
        result = validate_feature_quality(df)
        assert len(result["high_correlations"]) > 0
        # Check that feature_a and feature_b correlation is found
        found_ab = any(
            (item["col1"] == "feature_a" and item["col2"] == "feature_b") or
            (item["col1"] == "feature_b" and item["col2"] == "feature_a")
            for item in result["high_correlations"]
        )
        assert found_ab

    def test_uses_custom_feature_cols(self, quality_df):
        """Test that custom feature_cols parameter is used."""
        result = validate_feature_quality(quality_df, feature_cols=["feature_a"])
        assert result["total_features"] == 1
        # Should only have stats for feature_a
        assert "feature_a" in result["distribution_stats"]
        assert "feature_b" not in result["distribution_stats"]

    def test_counts_total_rows(self, quality_df):
        """Test that total rows are correctly counted."""
        result = validate_feature_quality(quality_df)
        assert result["total_rows"] == 5

    def test_excludes_non_numeric_from_correlation(self, quality_df):
        """Test that non-numeric columns are excluded from correlation."""
        result = validate_feature_quality(quality_df)
        # category column should not appear in correlation matrix
        if result["correlation_matrix"] is not None:
            assert "category" not in result["correlation_matrix"].columns


class TestGenerateFeatureQualityReport:
    """Test cases for generate_feature_quality_report function."""

    @pytest.fixture
    def report_df(self):
        """Create sample DataFrame for report generation testing."""
        return pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [10.0, 20.0, np.nan, 40.0, 50.0],
        })

    def test_returns_string(self, report_df):
        """Test that function returns a string."""
        result = generate_feature_quality_report(report_df)
        assert isinstance(result, str)

    def test_includes_header(self, report_df):
        """Test that report includes header."""
        result = generate_feature_quality_report(report_df)
        assert "FEATURE QUALITY VALIDATION REPORT" in result

    def test_includes_dataset_size(self, report_df):
        """Test that report includes dataset size."""
        result = generate_feature_quality_report(report_df)
        assert "5 rows" in result

    def test_includes_missing_values_section(self, report_df):
        """Test that report includes missing values section."""
        result = generate_feature_quality_report(report_df)
        assert "MISSING VALUES" in result
        assert "feature_b" in result

    def test_includes_distribution_section(self, report_df):
        """Test that report includes distribution section."""
        result = generate_feature_quality_report(report_df)
        assert "DISTRIBUTION SUMMARY" in result

    def test_saves_to_file_when_path_provided(self, report_df, tmp_path):
        """Test that report is saved to file when path provided."""
        output_file = tmp_path / "quality_report.txt"
        result = generate_feature_quality_report(report_df, output_path=str(output_file))
        assert output_file.exists()
        with open(output_file) as f:
            saved_content = f.read()
        assert saved_content == result