"""
Unit tests for RQ3 Predictive Modeling pipeline.

Uses small synthetic fixtures (CI-safe, no real data).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.rq3_modeling import (
    build_comparison_table,
    build_model_configs,
    get_feature_importance,
    prepare_modeling_data,
    screen_features,
    test_hypothesis as run_hypothesis_test,
    train_and_evaluate,
)
from src.config import RQ3_CANDIDATE_FEATURES, RQ3_LEAKAGE_COLUMNS, RQ3_TARGET


# --- Minimal configs for fast CI tests ---
FAST_MODEL_CONFIGS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            solver="saga", max_iter=5000, class_weight="balanced", random_state=42,
        ),
        "param_grid": {"C": [1.0], "penalty": ["l2"]},
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced", random_state=42,
        ),
        "param_grid": {"n_estimators": [50], "max_depth": [5]},
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {"n_estimators": [50], "max_depth": [3]},
    },
}


@pytest.fixture
def synthetic_customer_df():
    """Create a synthetic customer-level DataFrame with all required columns."""
    rng = np.random.RandomState(42)
    n = 200

    # Create target with known class split
    target = np.concatenate([np.ones(50, dtype=int), np.zeros(150, dtype=int)])

    # Create features with some signal correlated to target
    return_freq = rng.randint(0, 10, n) + target * 3
    order_freq = rng.randint(1, 20, n)

    df = pd.DataFrame({
        "order_frequency": order_freq,
        "return_frequency": return_freq,
        "customer_return_rate": return_freq / (order_freq * 2 + 1),
        "avg_basket_size": rng.uniform(1, 5, n),
        "avg_order_value": rng.uniform(20, 200, n),
        "customer_tenure_days": rng.randint(30, 1000, n),
        "purchase_recency_days": rng.randint(0, 365, n),
        "total_items": rng.randint(1, 50, n),
        "total_sales": rng.uniform(50, 5000, n),
        "total_margin": rng.uniform(10, 2000, n),
        "avg_item_price": rng.uniform(10, 200, n),
        "avg_item_margin": rng.uniform(5, 100, n),
        # Target
        "is_high_erosion_customer": target,
        # Leakage columns (should be dropped)
        "total_profit_erosion": rng.uniform(0, 5000, n),
        "total_margin_reversal": rng.uniform(0, 3000, n),
        "total_process_cost": rng.uniform(0, 500, n),
        "profit_erosion_quartile": rng.choice([1, 2, 3, 4], n),
        "erosion_percentile_rank": rng.uniform(0, 100, n),
        "user_id": np.arange(1, n + 1),
    })

    return df


@pytest.fixture
def train_test_data(synthetic_customer_df):
    """Prepare split data from synthetic customer DataFrame."""
    return prepare_modeling_data(synthetic_customer_df)


@pytest.fixture
def screened_data(train_test_data):
    """Run feature screening on training data."""
    X_train, X_test, y_train, y_test = train_test_data
    surviving_features, screening_report = screen_features(X_train, y_train)
    return surviving_features, screening_report, X_train, X_test, y_train, y_test


class TestPrepareModelingData:
    """Tests for prepare_modeling_data()."""

    def test_returns_four_elements(self, synthetic_customer_df):
        result = prepare_modeling_data(synthetic_customer_df)
        assert len(result) == 4

    def test_split_sizes(self, synthetic_customer_df):
        X_train, X_test, y_train, y_test = prepare_modeling_data(synthetic_customer_df)
        total = len(X_train) + len(X_test)
        assert total == 200
        assert len(X_test) == pytest.approx(200 * 0.20, abs=2)

    def test_all_candidates_present(self, train_test_data):
        X_train, X_test, _, _ = train_test_data
        for feat in RQ3_CANDIDATE_FEATURES:
            assert feat in X_train.columns
            assert feat in X_test.columns

    def test_leakage_columns_excluded(self, train_test_data):
        X_train, X_test, _, _ = train_test_data
        for col in RQ3_LEAKAGE_COLUMNS:
            assert col not in X_train.columns
            assert col not in X_test.columns

    def test_stratification_preserved(self, synthetic_customer_df):
        X_train, X_test, y_train, y_test = prepare_modeling_data(synthetic_customer_df)
        original_rate = synthetic_customer_df[RQ3_TARGET].mean()
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert train_rate == pytest.approx(original_rate, abs=0.05)
        assert test_rate == pytest.approx(original_rate, abs=0.05)

    def test_no_missing_values(self, train_test_data):
        X_train, X_test, _, _ = train_test_data
        assert X_train.isna().sum().sum() == 0
        assert X_test.isna().sum().sum() == 0

    def test_missing_feature_raises(self, synthetic_customer_df):
        df = synthetic_customer_df.drop(columns=["order_frequency"])
        with pytest.raises(ValueError, match="Missing candidate features"):
            prepare_modeling_data(df)

    def test_missing_target_raises(self, synthetic_customer_df):
        df = synthetic_customer_df.drop(columns=[RQ3_TARGET])
        with pytest.raises(ValueError, match="Target column"):
            prepare_modeling_data(df)


class TestScreenFeatures:
    """Tests for screen_features()."""

    def test_returns_tuple(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        result = screen_features(X_train, y_train)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_surviving_features_subset_of_candidates(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        surviving, _ = screen_features(X_train, y_train)
        assert all(f in RQ3_CANDIDATE_FEATURES for f in surviving)

    def test_screening_report_has_all_candidates(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        _, report = screen_features(X_train, y_train)
        assert set(report["feature"]) == set(RQ3_CANDIDATE_FEATURES)

    def test_screening_report_columns(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        _, report = screen_features(X_train, y_train)
        expected_cols = [
            "feature", "variance", "variance_pass",
            "correlation_pass", "univariate_corr", "univariate_pvalue",
            "bonferroni_alpha", "univariate_pass", "final_status",
        ]
        for col in expected_cols:
            assert col in report.columns, f"Missing column: {col}"

    def test_final_status_values(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        _, report = screen_features(X_train, y_train)
        assert set(report["final_status"].unique()).issubset({"pass", "fail"})

    def test_variance_gate_drops_constant(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        X_train = X_train.copy()
        X_train["constant_feat"] = 1.0
        surviving, report = screen_features(X_train, y_train)
        assert "constant_feat" not in surviving
        const_row = report[report["feature"] == "constant_feat"]
        assert bool(const_row["variance_pass"].values[0]) is False

    def test_correlation_gate_drops_redundant(self, train_test_data):
        X_train, _, y_train, _ = train_test_data
        X_train = X_train.copy()
        X_train["duplicate_feat"] = X_train["order_frequency"] + np.random.normal(0, 0.001, len(X_train))
        surviving, report = screen_features(X_train, y_train, correlation_threshold=0.85)
        dup_row = report[report["feature"] == "duplicate_feat"]
        orig_row = report[report["feature"] == "order_frequency"]
        dropped = (
            dup_row["final_status"].values[0] == "fail"
            or orig_row["final_status"].values[0] == "fail"
        )
        assert dropped


class TestBuildModelConfigs:
    """Tests for build_model_configs()."""

    def test_returns_three_models(self):
        configs = build_model_configs()
        assert len(configs) == 3

    def test_expected_model_names(self):
        configs = build_model_configs()
        expected = {"Logistic Regression", "Random Forest", "Gradient Boosting"}
        assert set(configs.keys()) == expected

    def test_each_has_estimator_and_params(self):
        configs = build_model_configs()
        for name, config in configs.items():
            assert "estimator" in config, f"{name} missing estimator"
            assert "param_grid" in config, f"{name} missing param_grid"
            assert isinstance(config["param_grid"], dict)


class TestTrainAndEvaluate:
    """Tests for train_and_evaluate()."""

    @pytest.fixture
    def model_results(self, screened_data):
        surviving, _, X_train, X_test, y_train, y_test = screened_data
        X_train_s = X_train[surviving]
        X_test_s = X_test[surviving]
        return train_and_evaluate(
            X_train_s, X_test_s, y_train, y_test,
            model_configs=FAST_MODEL_CONFIGS,
        )

    def test_returns_all_models(self, model_results):
        assert len(model_results) == 3

    def test_auc_in_valid_range(self, model_results):
        for name, res in model_results.items():
            assert 0 <= res["cv_auc"] <= 1, f"{name} cv_auc out of range"
            assert 0 <= res["test_auc"] <= 1, f"{name} test_auc out of range"

    def test_result_keys(self, model_results):
        expected_keys = {
            "best_estimator", "best_params", "cv_auc", "test_auc",
            "y_pred", "y_proba", "precision", "recall", "f1",
            "accuracy", "confusion_matrix", "roc_curve",
        }
        for name, res in model_results.items():
            assert expected_keys.issubset(res.keys()), f"{name} missing keys"

    def test_predictions_correct_length(self, model_results, screened_data):
        _, _, _, X_test, _, y_test = screened_data
        for name, res in model_results.items():
            assert len(res["y_pred"]) == len(y_test)
            assert len(res["y_proba"]) == len(y_test)


class TestGetFeatureImportance:
    """Tests for get_feature_importance()."""

    @pytest.fixture
    def importance_data(self, screened_data):
        surviving, _, X_train, X_test, y_train, y_test = screened_data
        X_train_s = X_train[surviving]
        X_test_s = X_test[surviving]
        configs = {
            "Logistic Regression": FAST_MODEL_CONFIGS["Logistic Regression"],
            "Random Forest": FAST_MODEL_CONFIGS["Random Forest"],
        }
        results = train_and_evaluate(
            X_train_s, X_test_s, y_train, y_test, model_configs=configs,
        )
        return get_feature_importance(results, surviving), surviving

    def test_returns_dataframe(self, importance_data):
        df, _ = importance_data
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, importance_data):
        df, _ = importance_data
        assert set(df.columns) == {"feature", "model", "importance"}

    def test_only_surviving_features(self, importance_data):
        df, surviving = importance_data
        assert set(df["feature"].unique()).issubset(set(surviving))


class TestBuildComparisonTable:
    """Tests for build_comparison_table()."""

    @pytest.fixture
    def mock_results(self):
        return {
            "Model A": {
                "cv_auc": 0.85, "test_auc": 0.82,
                "precision": 0.78, "recall": 0.72, "f1": 0.75, "accuracy": 0.80,
            },
            "Model B": {
                "cv_auc": 0.65, "test_auc": 0.60,
                "precision": 0.55, "recall": 0.50, "f1": 0.52, "accuracy": 0.58,
            },
        }

    def test_expected_columns(self, mock_results):
        table = build_comparison_table(mock_results)
        expected = {"model", "cv_auc", "test_auc", "precision", "recall", "f1", "accuracy", "meets_threshold"}
        assert set(table.columns) == expected

    def test_meets_threshold_flag(self, mock_results):
        table = build_comparison_table(mock_results, auc_threshold=0.70)
        model_a = table[table["model"] == "Model A"]
        model_b = table[table["model"] == "Model B"]
        assert bool(model_a["meets_threshold"].values[0]) is True
        assert bool(model_b["meets_threshold"].values[0]) is False

    def test_sorted_by_test_auc(self, mock_results):
        table = build_comparison_table(mock_results)
        aucs = table["test_auc"].tolist()
        assert aucs == sorted(aucs, reverse=True)


class TestHypothesisTest:
    """Tests for run_hypothesis_test() (test_hypothesis in rq3_modeling)."""

    @pytest.fixture
    def results_above_threshold(self):
        return {
            "Model A": {"test_auc": 0.85},
            "Model B": {"test_auc": 0.72},
        }

    @pytest.fixture
    def results_below_threshold(self):
        return {
            "Model A": {"test_auc": 0.65},
            "Model B": {"test_auc": 0.60},
        }

    def test_reject_null_when_above(self, results_above_threshold):
        result = run_hypothesis_test(results_above_threshold, auc_threshold=0.70)
        assert result["reject_null"] is True
        assert result["best_model"] == "Model A"
        assert result["best_auc"] == 0.85

    def test_fail_to_reject_when_below(self, results_below_threshold):
        result = run_hypothesis_test(results_below_threshold, auc_threshold=0.70)
        assert result["reject_null"] is False

    def test_result_keys(self, results_above_threshold):
        result = run_hypothesis_test(results_above_threshold)
        expected_keys = {"best_model", "best_auc", "threshold", "reject_null", "conclusion"}
        assert set(result.keys()) == expected_keys

    def test_conclusion_string(self, results_above_threshold):
        result = run_hypothesis_test(results_above_threshold)
        assert isinstance(result["conclusion"], str)
        assert len(result["conclusion"]) > 0
