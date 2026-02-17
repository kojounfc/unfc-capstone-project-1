"""
Unit tests for RQ3 Sensitivity Analysis pipeline.

Uses small synthetic fixtures (CI-safe, no real data).
Same pattern as test_rq3_modeling.py.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.rq3_sensitivity import (
    _scale_cost_components,
    compute_label_stability,
    run_cost_sensitivity,
    run_threshold_sensitivity,
)
from src.feature_engineering import DEFAULT_COST_COMPONENTS


# Minimal RF config for fast CI tests
FAST_SENSITIVITY_CONFIGS = {
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced", random_state=42, n_estimators=10,
        ),
        "param_grid": {"n_estimators": [10], "max_depth": [3]},
    },
}


@pytest.fixture
def synthetic_returned_items():
    """Synthetic item-level returned items DataFrame.

    Mimics the structure of returned items from feature_engineered_dataset
    with columns needed by calculate_profit_erosion().
    """
    rng = np.random.RandomState(42)
    n = 300
    n_users = 80

    categories = rng.choice(
        ["Tops & Tees", "Jeans", "Accessories", "Outerwear & Coats", "Active"],
        n,
    )
    user_ids = rng.choice(range(1, n_users + 1), n)
    order_ids = rng.choice(range(1000, 1200), n)

    df = pd.DataFrame({
        "user_id": user_ids,
        "order_id": order_ids,
        "order_item_id": np.arange(1, n + 1),
        "sale_price": rng.uniform(10, 200, n),
        "cost": rng.uniform(5, 100, n),
        "category": categories,
    })
    df["item_margin"] = df["sale_price"] - df["cost"]

    return df


@pytest.fixture
def synthetic_behavioral_df():
    """Synthetic customer-level behavioral features DataFrame.

    Mimics output of engineer_customer_behavioral_features().
    """
    rng = np.random.RandomState(42)
    n_users = 80

    # Create target with known signal
    return_freq = rng.randint(0, 10, n_users)
    order_freq = rng.randint(1, 20, n_users)

    df = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "order_frequency": order_freq,
        "return_frequency": return_freq,
        "customer_return_rate": return_freq / (order_freq * 2 + 1),
        "avg_basket_size": rng.uniform(1, 5, n_users),
        "avg_order_value": rng.uniform(20, 200, n_users),
        "customer_tenure_days": rng.randint(30, 1000, n_users),
        "purchase_recency_days": rng.randint(0, 365, n_users),
        "total_items": rng.randint(1, 50, n_users),
        "total_sales": rng.uniform(50, 5000, n_users),
        "total_margin": rng.uniform(10, 2000, n_users),
        "avg_item_price": rng.uniform(10, 200, n_users),
        "avg_item_margin": rng.uniform(5, 100, n_users),
    })

    return df


@pytest.fixture
def synthetic_customer_targets_df(synthetic_behavioral_df, synthetic_returned_items):
    """Customer-level DataFrame with behavioral features and total_profit_erosion.

    Mimics the merged data used for threshold sensitivity.
    """
    from src.feature_engineering import (
        aggregate_profit_erosion_by_customer,
        calculate_profit_erosion,
    )

    erosion_df = calculate_profit_erosion(synthetic_returned_items)
    customer_erosion = aggregate_profit_erosion_by_customer(erosion_df)

    merged = synthetic_behavioral_df.merge(
        customer_erosion[["user_id", "total_profit_erosion"]],
        on="user_id",
        how="inner",
    )
    return merged


class TestScaleCostComponents:
    """Tests for _scale_cost_components()."""

    def test_sums_to_target(self):
        scaled = _scale_cost_components(DEFAULT_COST_COMPONENTS, 18.0)
        assert abs(sum(scaled.values()) - 18.0) < 0.01

    def test_proportions_preserved(self):
        original_total = sum(DEFAULT_COST_COMPONENTS.values())
        scaled = _scale_cost_components(DEFAULT_COST_COMPONENTS, 18.0)
        for key in DEFAULT_COST_COMPONENTS:
            original_pct = DEFAULT_COST_COMPONENTS[key] / original_total
            scaled_pct = scaled[key] / sum(scaled.values())
            assert abs(original_pct - scaled_pct) < 0.01

    def test_same_keys(self):
        scaled = _scale_cost_components(DEFAULT_COST_COMPONENTS, 8.0)
        assert set(scaled.keys()) == set(DEFAULT_COST_COMPONENTS.keys())

    def test_identity_at_baseline(self):
        baseline_total = sum(DEFAULT_COST_COMPONENTS.values())
        scaled = _scale_cost_components(DEFAULT_COST_COMPONENTS, baseline_total)
        for key in DEFAULT_COST_COMPONENTS:
            assert abs(scaled[key] - DEFAULT_COST_COMPONENTS[key]) < 0.01

    def test_zero_total_returns_zeros(self):
        zero_components = {k: 0.0 for k in DEFAULT_COST_COMPONENTS}
        scaled = _scale_cost_components(zero_components, 10.0)
        assert all(v == 0.0 for v in scaled.values())


class TestRunCostSensitivity:
    """Tests for run_cost_sensitivity()."""

    def test_returns_tuple(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        result = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=[10.0, 12.0],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_summary_row_count(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        costs = [8.0, 12.0, 18.0]
        summary_df, _ = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=costs,
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        assert len(summary_df) == len(costs)

    def test_summary_expected_columns(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        summary_df, _ = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=[12.0],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        expected = {
            "base_cost", "best_auc", "f1", "precision", "recall",
            "threshold_value", "n_positive", "n_total", "positive_rate",
            "n_surviving_features", "surviving_features",
        }
        assert expected.issubset(set(summary_df.columns))

    def test_auc_in_valid_range(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        summary_df, _ = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=[12.0],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        valid = summary_df["best_auc"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_labels_dict_keys_match_costs(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        costs = [10.0, 12.0]
        _, labels_dict = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=costs,
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        assert set(labels_dict.keys()) == set(costs)

    def test_labels_are_binary(
        self, synthetic_returned_items, synthetic_behavioral_df
    ):
        _, labels_dict = run_cost_sensitivity(
            synthetic_returned_items,
            synthetic_behavioral_df,
            base_costs=[12.0],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        for labels in labels_dict.values():
            assert set(labels.unique()).issubset({0, 1})


class TestRunThresholdSensitivity:
    """Tests for run_threshold_sensitivity()."""

    def test_returns_dataframe(self, synthetic_customer_targets_df):
        result = run_threshold_sensitivity(
            synthetic_customer_targets_df,
            thresholds=[0.50, 0.75],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches_thresholds(self, synthetic_customer_targets_df):
        thresholds = [0.50, 0.70, 0.90]
        result = run_threshold_sensitivity(
            synthetic_customer_targets_df,
            thresholds=thresholds,
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        assert len(result) == len(thresholds)

    def test_expected_columns(self, synthetic_customer_targets_df):
        result = run_threshold_sensitivity(
            synthetic_customer_targets_df,
            thresholds=[0.75],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        expected = {
            "threshold", "best_auc", "f1", "precision", "recall",
            "positive_rate", "n_positive", "n_total",
            "n_surviving_features", "surviving_features",
        }
        assert expected.issubset(set(result.columns))

    def test_positive_rate_decreases_with_threshold(
        self, synthetic_customer_targets_df
    ):
        thresholds = [0.50, 0.75, 0.90]
        result = run_threshold_sensitivity(
            synthetic_customer_targets_df,
            thresholds=thresholds,
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        rates = result.sort_values("threshold")["positive_rate"].values
        # Positive rate should generally decrease as threshold increases
        # (allowing tolerance for ties at boundary)
        assert rates[0] >= rates[-1]

    def test_auc_in_valid_range(self, synthetic_customer_targets_df):
        result = run_threshold_sensitivity(
            synthetic_customer_targets_df,
            thresholds=[0.75],
            model_configs=FAST_SENSITIVITY_CONFIGS,
        )
        valid = result["best_auc"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


class TestComputeLabelStability:
    """Tests for compute_label_stability()."""

    @pytest.fixture
    def sample_labels_dict(self):
        """Labels dict with known properties for stability testing."""
        idx = pd.RangeIndex(100)
        rng = np.random.RandomState(42)

        baseline = pd.Series(
            np.concatenate([np.ones(25), np.zeros(75)]).astype(int),
            index=idx,
        )
        # Identical to baseline
        same = baseline.copy()
        # Slightly different
        shifted = baseline.copy()
        shifted.iloc[0] = 0  # flip one positive to negative
        shifted.iloc[90] = 1  # flip one negative to positive
        # Completely different
        opposite = (1 - baseline)

        return {
            10.0: same,
            12.0: baseline,
            14.0: shifted,
            99.0: opposite,
        }

    def test_returns_dataframe(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        assert isinstance(result, pd.DataFrame)

    def test_row_count(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        assert len(result) == len(sample_labels_dict)

    def test_expected_columns(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        expected = {
            "scenario", "jaccard_similarity", "flip_rate",
            "n_flagged", "n_flagged_baseline",
        }
        assert set(result.columns) == expected

    def test_self_comparison_jaccard_one(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        baseline_row = result[result["scenario"] == 12.0]
        assert baseline_row["jaccard_similarity"].values[0] == pytest.approx(1.0)

    def test_self_comparison_flip_rate_zero(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        baseline_row = result[result["scenario"] == 12.0]
        assert baseline_row["flip_rate"].values[0] == pytest.approx(0.0)

    def test_identical_labels_jaccard_one(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        same_row = result[result["scenario"] == 10.0]
        assert same_row["jaccard_similarity"].values[0] == pytest.approx(1.0)

    def test_shifted_labels_flip_rate(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        shifted_row = result[result["scenario"] == 14.0]
        # 2 flips out of 100
        assert shifted_row["flip_rate"].values[0] == pytest.approx(0.02)

    def test_opposite_labels_jaccard_zero(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        opposite_row = result[result["scenario"] == 99.0]
        assert opposite_row["jaccard_similarity"].values[0] == pytest.approx(0.0)

    def test_missing_baseline_key_raises(self, sample_labels_dict):
        with pytest.raises(KeyError, match="not found"):
            compute_label_stability(sample_labels_dict, baseline_key=999.0)

    def test_jaccard_in_range(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        assert (result["jaccard_similarity"] >= 0).all()
        assert (result["jaccard_similarity"] <= 1).all()

    def test_flip_rate_in_range(self, sample_labels_dict):
        result = compute_label_stability(sample_labels_dict, baseline_key=12.0)
        assert (result["flip_rate"] >= 0).all()
        assert (result["flip_rate"] <= 1).all()
