"""
Unit tests for RQ3 External Validation pipeline (SSL).

Uses small synthetic SSL-like fixtures (CI-safe, no real data).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.rq3_validation import (
    build_validation_summary,
    create_ssl_targets,
    engineer_ssl_account_features,
    load_ssl_data,
    validate_directional_predictions,
    validate_feature_patterns,
)


@pytest.fixture
def synthetic_ssl_df():
    """Create a synthetic SSL-like DataFrame mimicking return order lines.

    Includes Sales_Type column with ~30% RETURN and ~70% ORDER lines,
    matching the real SSL dataset distribution.
    """
    rng = np.random.RandomState(42)
    n_lines = 300
    n_accounts = 30
    n_orders = 60

    accounts = rng.choice(range(1000, 1000 + n_accounts), n_lines)
    orders = rng.choice(range(5000, 5000 + n_orders), n_lines)

    booked_dates = pd.date_range("2024-01-01", periods=n_lines, freq="2h")

    # ~30% RETURN (actual returns), ~70% ORDER (no-charge replacements)
    sales_type = rng.choice(["RETURN", "ORDER"], n_lines, p=[0.3, 0.7])

    # RETURN lines have negative CreditReturn Sales; ORDER lines ≈ 0
    credit_sales = np.where(
        sales_type == "RETURN",
        rng.uniform(-500, -10, n_lines),
        rng.uniform(-1, 1, n_lines),  # near zero for replacements
    )
    # RETURN lines have negative qty; ORDER lines positive
    ordered_qty = np.where(
        sales_type == "RETURN",
        -rng.randint(1, 10, n_lines),
        rng.randint(1, 10, n_lines),
    )

    df = pd.DataFrame({
        "Bill To Act #": accounts,
        "Order Number": orders,
        "Order Line ID": np.arange(1, n_lines + 1),
        "Booked Date": rng.choice(booked_dates, n_lines),
        "Billed Date": rng.choice(booked_dates, n_lines),
        "Reference Booked Date": rng.choice(booked_dates, n_lines),
        "Reference Sale Amount": rng.uniform(10, 500, n_lines),
        "CreditReturn Sales": credit_sales,
        "Ordered Qty": ordered_qty,
        "Product Cost": rng.uniform(5, 200, n_lines),
        "Gross Profit": rng.uniform(-200, 100, n_lines),
        "gross_financial_loss": rng.uniform(0, 500, n_lines),
        "total_loss": rng.uniform(0, 600, n_lines),
        "total_return_cogs": rng.uniform(0, 300, n_lines),
        "estimated_labor_cost": rng.uniform(0, 50, n_lines),
        "Lines Per Order": rng.randint(1, 10, n_lines),
        "Sales_Type": sales_type,
        "Return_Type": rng.choice(
            ["Credit Only", "No-Charge Replacement", "FC Return",
             "Vendor Return", "Unauthorized Return"],
            n_lines,
        ),
    })

    return df


@pytest.fixture
def ssl_account_df(synthetic_ssl_df):
    """Create account-level features from synthetic SSL data."""
    return engineer_ssl_account_features(synthetic_ssl_df)


@pytest.fixture
def ssl_account_with_target(ssl_account_df):
    """Account-level features with target variable."""
    return create_ssl_targets(ssl_account_df)


@pytest.fixture
def thelook_screening_report():
    """Mock TheLook screening report for pattern comparison."""
    features = [
        "order_frequency", "return_frequency", "customer_return_rate",
        "avg_basket_size", "avg_order_value", "customer_tenure_days",
        "purchase_recency_days", "total_items", "total_sales",
        "total_margin", "avg_item_price", "avg_item_margin",
    ]
    return pd.DataFrame({
        "feature": features,
        "final_status": [
            "pass", "pass", "pass", "fail", "pass", "pass",
            "fail", "pass", "pass", "pass", "pass", "pass",
        ],
    })


@pytest.fixture
def mock_thelook_model():
    """Create a simple trained model for directional validation."""
    rng = np.random.RandomState(42)
    # Train a simple RF on synthetic data matching TheLook feature space
    feature_names = [
        "order_frequency", "return_frequency", "customer_return_rate",
        "avg_basket_size", "avg_order_value", "customer_tenure_days",
        "purchase_recency_days", "total_items", "total_sales",
        "total_margin", "avg_item_price", "avg_item_margin",
    ]
    n = 200
    X = pd.DataFrame(
        rng.uniform(0, 100, (n, len(feature_names))),
        columns=feature_names,
    )
    y = (X["return_frequency"] > 50).astype(int)

    model = RandomForestClassifier(
        n_estimators=10, max_depth=3, random_state=42,
    )
    model.fit(X, y)
    return model, feature_names


class TestLoadSslData:
    """Tests for load_ssl_data()."""

    def test_loads_dataframe(self, tmp_path, synthetic_ssl_df):
        filepath = tmp_path / "test_ssl.csv"
        synthetic_ssl_df.to_csv(filepath, index=False)
        result = load_ssl_data(filepath=str(filepath))
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_date_columns_parsed(self, tmp_path, synthetic_ssl_df):
        filepath = tmp_path / "test_ssl.csv"
        synthetic_ssl_df.to_csv(filepath, index=False)
        result = load_ssl_data(filepath=str(filepath))
        assert pd.api.types.is_datetime64_any_dtype(result["Booked Date"])

    def test_expected_columns_present(self, tmp_path, synthetic_ssl_df):
        filepath = tmp_path / "test_ssl.csv"
        synthetic_ssl_df.to_csv(filepath, index=False)
        result = load_ssl_data(filepath=str(filepath))
        assert "Bill To Act #" in result.columns
        assert "Order Number" in result.columns
        assert "Return_Type" in result.columns

    def test_drops_missing_account(self, tmp_path, synthetic_ssl_df):
        df = synthetic_ssl_df.copy()
        df.loc[0, "Bill To Act #"] = np.nan
        filepath = tmp_path / "test_ssl.csv"
        df.to_csv(filepath, index=False)
        result = load_ssl_data(filepath=str(filepath))
        assert len(result) == len(df) - 1


class TestEngineerSslAccountFeatures:
    """Tests for engineer_ssl_account_features()."""

    def test_one_row_per_account(self, synthetic_ssl_df, ssl_account_df):
        n_accounts = synthetic_ssl_df["Bill To Act #"].nunique()
        assert len(ssl_account_df) == n_accounts

    def test_expected_feature_columns(self, ssl_account_df):
        expected = [
            "order_frequency", "return_frequency", "customer_return_rate",
            "avg_basket_size", "avg_order_value", "total_items",
            "total_sales", "total_margin", "avg_item_price", "avg_item_margin",
            "customer_tenure_days", "purchase_recency_days",
        ]
        for col in expected:
            assert col in ssl_account_df.columns, f"Missing column: {col}"

    def test_has_account_id(self, ssl_account_df):
        assert "account_id" in ssl_account_df.columns

    def test_has_total_loss(self, ssl_account_df):
        assert "total_loss" in ssl_account_df.columns

    def test_order_frequency_positive(self, ssl_account_df):
        assert (ssl_account_df["order_frequency"] > 0).all()

    def test_return_frequency_non_negative(self, ssl_account_df):
        assert (ssl_account_df["return_frequency"] >= 0).all()

    def test_return_frequency_leq_total_items(self, ssl_account_df):
        # return_frequency counts only RETURN lines, total_items counts all lines
        assert (ssl_account_df["return_frequency"] <= ssl_account_df["total_items"]).all()

    def test_customer_return_rate_has_variance(self, ssl_account_df):
        # With ~30% RETURN / ~70% ORDER split, rate should NOT be 1.0 for all
        assert ssl_account_df["customer_return_rate"].std() > 0

    def test_customer_return_rate_in_range(self, ssl_account_df):
        assert (ssl_account_df["customer_return_rate"] >= 0).all()
        assert (ssl_account_df["customer_return_rate"] <= 1).all()


class TestAvgItemPriceComputation:
    """Tests for avg_item_price computation in engineer_ssl_account_features().

    Validates the S5 strategy: Reference Sale Amount from ALL lines as
    primary source, with CreditReturn Sales fallback on RETURN lines.
    """

    def test_uses_reference_sale_amount(self, synthetic_ssl_df):
        """When Reference Sale Amount is available, it should be used
        for ALL line types (RETURN + ORDER), not just CreditReturn Sales."""
        acct = engineer_ssl_account_features(synthetic_ssl_df)
        # With RefSale available on all lines, coverage should be high
        # (every account has at least one line with positive RefSale)
        coverage = acct["avg_item_price"].notna().mean()
        assert coverage > 0.9, (
            f"Expected >90% coverage with RefSale on all lines, got {coverage:.1%}"
        )

    def test_includes_order_lines(self, synthetic_ssl_df):
        """ORDER-only accounts should have non-null avg_item_price when
        Reference Sale Amount is available on ORDER lines."""
        # Create a fixture with only ORDER lines for some accounts
        df = synthetic_ssl_df.copy()
        # Make first 5 accounts ORDER-only
        order_only_accounts = df["Bill To Act #"].unique()[:5]
        df.loc[
            df["Bill To Act #"].isin(order_only_accounts), "Sales_Type"
        ] = "ORDER"

        acct = engineer_ssl_account_features(df)
        order_only_rows = acct[acct["account_id"].isin(order_only_accounts)]
        # These accounts have no RETURN lines, but RefSale should provide price
        assert order_only_rows["avg_item_price"].notna().all(), (
            "ORDER-only accounts should have avg_item_price from RefSale"
        )

    def test_fallback_to_credit_sales(self, synthetic_ssl_df):
        """When Reference Sale Amount is null, RETURN lines should fall
        back to |CreditReturn Sales / Ordered Qty|."""
        df = synthetic_ssl_df.copy()
        # Null out RefSale for all lines
        df["Reference Sale Amount"] = np.nan
        acct = engineer_ssl_account_features(df)
        # Only RETURN-line accounts should have avg_item_price
        return_accounts = set(
            df.loc[df["Sales_Type"] == "RETURN", "Bill To Act #"].unique()
        )
        for _, row in acct.iterrows():
            if row["account_id"] in return_accounts:
                # Should have a value from CreditReturn fallback
                assert pd.notna(row["avg_item_price"]), (
                    f"Account {row['account_id']} has RETURN lines but null avg_item_price"
                )
            else:
                # ORDER-only with no RefSale → should be NaN
                assert pd.isna(row["avg_item_price"]), (
                    f"Account {row['account_id']} is ORDER-only with no RefSale "
                    "but has non-null avg_item_price"
                )

    def test_fallback_when_refsale_all_null(self, synthetic_ssl_df):
        """When Reference Sale Amount is all null, avg_item_price should
        fall back to RETURN-only CreditReturn logic (reduced coverage)."""
        df = synthetic_ssl_df.copy()
        df["Reference Sale Amount"] = np.nan
        acct = engineer_ssl_account_features(df)
        # Should still produce avg_item_price from CreditReturn Sales
        assert "avg_item_price" in acct.columns
        # Coverage limited to RETURN-line accounts only
        return_accounts = set(
            df.loc[df["Sales_Type"] == "RETURN", "Bill To Act #"].unique()
        )
        non_null = acct.loc[acct["avg_item_price"].notna(), "account_id"]
        assert set(non_null).issubset(return_accounts)

    def test_coverage_improves_with_refsale(self, synthetic_ssl_df):
        """Coverage should be higher with valid Reference Sale Amount
        than when it is null (forcing CreditReturn fallback only)."""
        # With RefSale (normal fixture)
        acct_with = engineer_ssl_account_features(synthetic_ssl_df)
        coverage_with = acct_with["avg_item_price"].notna().mean()

        # Without RefSale (null out → forces fallback to CreditReturn)
        df_null_ref = synthetic_ssl_df.copy()
        df_null_ref["Reference Sale Amount"] = np.nan
        acct_without = engineer_ssl_account_features(df_null_ref)
        coverage_without = acct_without["avg_item_price"].notna().mean()

        assert coverage_with >= coverage_without, (
            f"Coverage with RefSale ({coverage_with:.1%}) should be >= "
            f"without ({coverage_without:.1%})"
        )


class TestCreateSslTargets:
    """Tests for create_ssl_targets()."""

    def test_target_column_created(self, ssl_account_with_target):
        assert "is_high_loss_account" in ssl_account_with_target.columns

    def test_target_is_binary(self, ssl_account_with_target):
        assert set(ssl_account_with_target["is_high_loss_account"].unique()).issubset({0, 1})

    def test_75th_percentile_logic(self, ssl_account_df):
        result = create_ssl_targets(ssl_account_df, percentile=75.0)
        threshold = np.percentile(ssl_account_df["total_loss"].dropna(), 75.0)
        expected_high = (ssl_account_df["total_loss"] >= threshold).astype(int)
        np.testing.assert_array_equal(
            result["is_high_loss_account"].values, expected_high.values
        )

    def test_approximately_25_pct_high(self, ssl_account_with_target):
        pct_high = ssl_account_with_target["is_high_loss_account"].mean()
        # Should be approximately 25% (with small sample tolerance)
        assert 0.10 <= pct_high <= 0.50

    def test_custom_percentile(self, ssl_account_df):
        result = create_ssl_targets(ssl_account_df, percentile=50.0)
        pct_high = result["is_high_loss_account"].mean()
        assert 0.30 <= pct_high <= 0.70


class TestValidateFeaturePatterns:
    """Tests for validate_feature_patterns()."""

    def test_returns_dataframe(
        self, ssl_account_with_target, thelook_screening_report
    ):
        result = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(
        self, ssl_account_with_target, thelook_screening_report
    ):
        result = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        expected = {
            "feature", "thelook_status", "ssl_status",
            "both_pass", "both_fail", "agreement",
        }
        assert set(result.columns) == expected

    def test_comparison_has_features(
        self, ssl_account_with_target, thelook_screening_report
    ):
        result = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        assert len(result) > 0

    def test_agreement_is_boolean(
        self, ssl_account_with_target, thelook_screening_report
    ):
        result = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        assert result["agreement"].dtype == bool

    def test_both_pass_is_boolean(
        self, ssl_account_with_target, thelook_screening_report
    ):
        result = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        assert result["both_pass"].dtype == bool


class TestValidateDirectionalPredictions:
    """Tests for validate_directional_predictions()."""

    def test_returns_dict(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        result = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        assert isinstance(result, dict)

    def test_expected_keys(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        result = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        expected_keys = {
            "directional_accuracy", "rank_correlation", "rank_pvalue",
            "predicted_high_pct", "actual_high_pct",
            "confusion_at_directional", "n_accounts",
            "n_features_available", "n_features_missing",
            "missing_features", "predictions_df",
        }
        assert expected_keys.issubset(result.keys())

    def test_directional_accuracy_in_range(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        result = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        assert 0.0 <= result["directional_accuracy"] <= 1.0

    def test_rank_correlation_in_range(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        result = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        assert -1.0 <= result["rank_correlation"] <= 1.0

    def test_predictions_df_correct_length(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        result = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        assert len(result["predictions_df"]) == len(ssl_account_with_target)

    def test_handles_missing_features(
        self, ssl_account_with_target, mock_thelook_model
    ):
        model, features = mock_thelook_model
        # Add a feature the SSL data doesn't have
        extended_features = features + ["nonexistent_feature"]
        # Retrain model with the extended feature set
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame(
            rng.uniform(0, 100, (n, len(extended_features))),
            columns=extended_features,
        )
        y = (X["return_frequency"] > 50).astype(int)
        new_model = RandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42,
        )
        new_model.fit(X, y)

        result = validate_directional_predictions(
            ssl_account_with_target, new_model, extended_features
        )
        assert result["n_features_missing"] == 1
        assert "nonexistent_feature" in result["missing_features"]


class TestBuildValidationSummary:
    """Tests for build_validation_summary()."""

    @pytest.fixture
    def pattern_and_directional(
        self, ssl_account_with_target, thelook_screening_report, mock_thelook_model
    ):
        pattern = validate_feature_patterns(
            ssl_account_with_target, thelook_screening_report
        )
        model, features = mock_thelook_model
        directional = validate_directional_predictions(
            ssl_account_with_target, model, features
        )
        return pattern, directional

    def test_returns_dataframe(self, pattern_and_directional):
        pattern, directional = pattern_and_directional
        result = build_validation_summary(pattern, directional)
        assert isinstance(result, pd.DataFrame)

    def test_expected_metrics(self, pattern_and_directional):
        pattern, directional = pattern_and_directional
        result = build_validation_summary(pattern, directional)
        metrics = set(result["metric"].values)
        expected = {
            "pattern_features_compared",
            "pattern_agreement_count",
            "pattern_agreement_pct",
            "directional_accuracy",
            "directional_rank_correlation",
            "ssl_accounts_evaluated",
        }
        assert expected.issubset(metrics)

    def test_has_metric_and_value_columns(self, pattern_and_directional):
        pattern, directional = pattern_and_directional
        result = build_validation_summary(pattern, directional)
        assert "metric" in result.columns
        assert "value" in result.columns
