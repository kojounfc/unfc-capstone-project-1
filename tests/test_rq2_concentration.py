import pandas as pd
import pytest

from src.rq2_concentration import (
    analyze_feature_concentration,
    bootstrap_gini_p_value,
    compute_pareto_table,
    concentration_comparison,
    filter_significant_features,
    get_business_summary,
    gini_coefficient,
    lorenz_curve_points,
    rank_features_by_concentration,
    summarize_concentration_findings,
    top_n_customer_impact,
    top_x_customer_share_of_value,
)


class TestRQ2Concentration:
    """Test cases for RQ2 concentration metrics."""

    def test_pareto_table_basic_shares_end_at_one(self):
        df = pd.DataFrame(
            {"user_id": ["a", "b", "c", "d"], "total_profit_erosion": [80, 10, 5, 5]}
        )
        t = compute_pareto_table(df)
        assert len(t) == 4
        assert t["total_profit_erosion"].iloc[0] == 80
        assert t["customer_share"].iloc[-1] == 1.0
        assert abs(t["value_share"].iloc[-1] - 1.0) < 1e-9

    def test_lorenz_includes_endpoints(self):
        df = pd.DataFrame({"total_profit_erosion": [1, 2, 3]})
        pts = lorenz_curve_points(df)
        assert pts.iloc[0]["population_share"] == 0.0
        assert pts.iloc[0]["value_share"] == 0.0
        assert pts.iloc[-1]["population_share"] == 1.0
        assert abs(pts.iloc[-1]["value_share"] - 1.0) < 1e-9

    def test_gini_equal_is_near_zero_and_unequal_is_higher(self):
        df_equal = pd.DataFrame({"total_profit_erosion": [10, 10, 10, 10]})
        g_equal = gini_coefficient(df_equal)
        assert 0.0 <= g_equal <= 0.01

        df_unequal = pd.DataFrame({"total_profit_erosion": [100, 0, 0, 0]})
        g_unequal = gini_coefficient(df_unequal)
        assert 0.0 <= g_unequal <= 1.0
        assert g_unequal > g_equal

    def test_top_x_share_computes_reasonable_value(self):
        df = pd.DataFrame(
            {"user_id": ["a", "b", "c", "d"], "total_profit_erosion": [80, 10, 5, 5]}
        )
        # Top 25% (1 customer out of 4) contributes 80 / 100 = 0.8
        share = top_x_customer_share_of_value(df, x=0.25)
        assert abs(share - 0.8) < 1e-9

    def test_top_x_share_validates_x(self):
        df = pd.DataFrame({"user_id": ["a"], "total_profit_erosion": [1.0]})
        with pytest.raises(ValueError):
            top_x_customer_share_of_value(df, x=0.0)
        with pytest.raises(ValueError):
            top_x_customer_share_of_value(df, x=1.1)

    def test_top_n_customer_impact_returns_expected_metrics(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "total_profit_erosion": [80.0, 10.0, 5.0, 5.0],
            }
        )
        out = top_n_customer_impact(df, n=2)

        assert out["count"] == 2
        assert out["absolute_loss"] == 90.0
        assert out["percentage_of_total"] == 90.0

    def test_top_n_customer_impact_handles_zero_total(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c"],
                "total_profit_erosion": [0.0, 0.0, 0.0],
            }
        )
        out = top_n_customer_impact(df, n=2)

        assert out["count"] == 2
        assert out["absolute_loss"] == 0.0
        assert out["percentage_of_total"] == 0.0

    def test_get_business_summary_high_concentration(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "total_profit_erosion": [100.0, 0.0, 0.0, 0.0],
            }
        )
        out = get_business_summary(df)

        assert out["gini_index"] > 0.5
        assert out["concentration_level"] in {"High", "Extreme"}
        assert out["pareto_ratio"].startswith("20% of customers = ")
        assert out["recommendation"] == "Targeted Policy"

    def test_bootstrap_gini_p_value_detects_non_uniform_concentration(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "total_profit_erosion": [100.0, 0.0, 0.0, 0.0],
            }
        )
        out = bootstrap_gini_p_value(df, n_bootstrap=100, random_state=7)

        assert out["observed_gini"] > 0.5
        assert out["null_mean_gini"] == 0.0
        assert 0.0 <= out["p_value"] <= 1.0

    def test_concentration_comparison_returns_both_ginis(self):
        df = pd.DataFrame(
            {
                "total_profit_erosion": [80.0, 10.0, 5.0, 5.0],
                "total_sales": [300.0, 250.0, 200.0, 150.0],
            }
        )

        out = concentration_comparison(df)
        assert set(out.keys()) == {"gini_erosion", "gini_baseline"}
        assert out["gini_erosion"] > out["gini_baseline"]

    def test_get_business_summary_moderate_concentration(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "total_profit_erosion": [25.0, 25.0, 25.0, 25.0],
            }
        )
        out = get_business_summary(df)

        assert out["gini_index"] == 0.0
        assert out["concentration_level"] == "Moderate"
        assert out["pareto_ratio"].startswith("20% of customers = ")
        assert out["recommendation"] == "Broad Policy"

    # -----------------------------
    # New coverage for untested API
    # -----------------------------

    def test_analyze_feature_concentration_handles_all_zero_or_too_few(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c"],
                "feature_x": [0.0, 0.0, 0.0],
            }
        )
        out = analyze_feature_concentration(
            df, feature_col="feature_x", n_bootstrap=25, random_state=1
        )

        assert out["feature"] == "feature_x"
        assert out["gini_coefficient"] == 0.0
        assert out["p_value"] == 1.0
        assert out["concentration_level"] == "N/A"
        assert out["n_customers"] == 0
        assert out["top_20_pct_share"] == 0.0

        df_one_nonzero = pd.DataFrame(
            {"user_id": ["a", "b", "c"], "feature_x": [10.0, 0.0, 0.0]}
        )
        out2 = analyze_feature_concentration(
            df_one_nonzero, feature_col="feature_x", n_bootstrap=25, random_state=1
        )
        assert out2["concentration_level"] == "N/A"
        assert out2["n_customers"] == 0

    def test_analyze_feature_concentration_computes_level_and_counts(self):
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d", "e"],
                # 3 non-zero customers -> should be analyzed
                "feature_skew": [100.0, 1.0, 1.0, 0.0, 0.0],
            }
        )

        out = analyze_feature_concentration(
            df,
            feature_col="feature_skew",
            n_bootstrap=50,
            random_state=7,
        )

        assert out["feature"] == "feature_skew"
        assert 0.0 <= out["gini_coefficient"] <= 1.0
        assert 0.0 <= out["p_value"] <= 1.0
        assert out["concentration_level"] in {"Low", "Moderate", "High", "Extreme"}
        assert out["n_customers"] == 3
        # stored as percent (0-100)
        assert 0.0 <= out["top_20_pct_share"] <= 100.0

    def test_rank_features_by_concentration_defaults_to_numeric_columns_and_sorts(self):
        df = pd.DataFrame(
            {
                "user_id": [f"u{i}" for i in range(1, 11)],
                # Uniform (low concentration)
                "feat_uniform": [1.0] * 10,
                # Skewed (higher concentration), at least min_customers non-zero after cleaning
                "feat_skew": [100.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                # Non-numeric column should be ignored
                "category": ["x"] * 10,
            }
        )

        ranking = rank_features_by_concentration(
            df,
            feature_cols=None,
            n_bootstrap=50,
            random_state=9,
            min_customers=3,
        )

        assert list(ranking.columns) == [
            "feature",
            "gini_coefficient",
            "concentration_pct",
            "p_value",
            "concentration_level",
            "n_customers",
            "top_20_pct_share",
        ]
        # Should include only the 2 numeric features (excluding user_id)
        assert set(ranking["feature"]) == {"feat_uniform", "feat_skew"}

        # Sorted by gini desc
        assert ranking["gini_coefficient"].is_monotonic_decreasing
        assert ranking.iloc[0]["feature"] == "feat_skew"
        assert ranking.iloc[-1]["feature"] == "feat_uniform"

        # concentration_pct is gini * 100
        assert (
            abs(
                ranking.iloc[0]["concentration_pct"]
                - ranking.iloc[0]["gini_coefficient"] * 100
            )
            < 1e-9
        )

    def test_rank_features_by_concentration_validates_feature_cols_exist(self):
        df = pd.DataFrame({"user_id": ["a", "b"], "x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            rank_features_by_concentration(df, feature_cols=["x", "missing_col"])

    def test_rank_features_by_concentration_returns_empty_when_no_features_or_no_results(
        self,
    ):
        df = pd.DataFrame({"user_id": ["a", "b", "c"], "category": ["x", "y", "z"]})

        # No numeric features -> empty result with expected columns
        ranking = rank_features_by_concentration(df, feature_cols=None)
        assert ranking.empty
        assert "gini_coefficient" in ranking.columns

        # Numeric feature, but min_customers too high -> filtered out
        df2 = pd.DataFrame({"user_id": ["a", "b", "c"], "feat": [10.0, 0.0, 0.0]})
        ranking2 = rank_features_by_concentration(
            df2, feature_cols=["feat"], min_customers=10
        )
        assert ranking2.empty

    def test_filter_significant_features_filters_on_alpha(self):
        ranking_df = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "gini_coefficient": [0.8, 0.4, 0.2],
                "concentration_pct": [80.0, 40.0, 20.0],
                "p_value": [0.01, 0.049, 0.2],
                "concentration_level": ["Extreme", "Moderate", "Low"],
                "n_customers": [100, 100, 100],
                "top_20_pct_share": [90.0, 60.0, 30.0],
            }
        )

        filtered = filter_significant_features(ranking_df, alpha=0.05)
        assert list(filtered["feature"]) == ["a", "b"]

        filtered_strict = filter_significant_features(ranking_df, alpha=0.02)
        assert list(filtered_strict["feature"]) == ["a"]

    def test_summarize_concentration_findings_empty_and_non_empty(self):
        empty_summary = summarize_concentration_findings(pd.DataFrame())
        assert empty_summary["n_features_analyzed"] == 0
        assert empty_summary["top_features"] == []
        assert empty_summary["avg_gini"] == 0.0
        assert empty_summary["n_significant"] == 0
        assert empty_summary["n_extreme"] == 0
        assert empty_summary["n_high"] == 0

        ranking_df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3", "f4"],
                "gini_coefficient": [0.8, 0.6, 0.4, 0.1],
                "concentration_pct": [80.0, 60.0, 40.0, 10.0],
                "p_value": [0.01, 0.2, 0.03, 0.5],
                "concentration_level": ["Extreme", "High", "Moderate", "Low"],
                "n_customers": [100, 100, 100, 100],
                "top_20_pct_share": [95.0, 85.0, 60.0, 25.0],
            }
        )

        summary = summarize_concentration_findings(ranking_df, top_n=2)

        assert summary["n_features_analyzed"] == 4
        assert len(summary["top_features"]) == 2
        assert summary["top_features"][0]["feature"] == "f1"
        assert 0.0 <= summary["avg_gini"] <= 1.0

        # Counts by level
        assert summary["n_extreme"] == 1
        assert summary["n_high"] == 1
        assert summary["n_moderate"] == 1
        assert summary["n_low"] == 1

        # Significant = p < 0.05 -> f1, f3
        assert summary["n_significant"] == 2
