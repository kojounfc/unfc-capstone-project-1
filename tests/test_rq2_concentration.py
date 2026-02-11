import pandas as pd
import pytest

from src.rq2_concentration import (
    bootstrap_gini_p_value,
    compute_pareto_table,
    concentration_comparison,
    get_business_summary,
    gini_coefficient,
    lorenz_curve_points,
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
