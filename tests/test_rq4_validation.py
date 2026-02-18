"""
Comprehensive test suite for RQ4 External Validation module.

Tests cover:
- SSL data loading and aggregation
- SSL account-level feature engineering
- Continuous regression target creation (profit erosion)
- Coefficient alignment validation (Level 1)
- Effect size generalization validation (Level 2)
- Validation summary generation
- Full validation pipeline orchestration

Tests use synthetic data fixtures to avoid SSL data dependencies in CI.
Integration tests check actual SSL data but skip gracefully if unavailable.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import statsmodels.api as sm
from scipy import stats

# Import RQ4 validation functions
try:
    from rq4_validation import (
        build_validation_summary,
        create_ssl_regression_target,
        engineer_ssl_regression_features,
        run_full_rq4_validation,
        validate_coefficient_alignment,
        validate_directional_effect_sizes,
    )
except ImportError as e:
    pytest.skip(
        f"rq4_validation not yet fully implemented: {e}", allow_module_level=True
    )


# ============================================================================
# FIXTURES: Synthetic SSL Account Data
# ============================================================================


@pytest.fixture
def synthetic_ssl_accounts():
    """
    Generate synthetic SSL account-level dataset.

    Returns:
        pd.DataFrame: SSL accounts with engineered features
    """
    np.random.seed(42)
    n_accounts = 500

    df = pd.DataFrame(
        {
            # Account identifier
            "Bill To Act #": [f"ACC{i:05d}" for i in range(n_accounts)],
            # Hypothesis predictors
            "return_frequency": np.random.poisson(lam=2, size=n_accounts),
            "avg_basket_size": np.random.uniform(1, 20, size=n_accounts),
            "purchase_recency_days": np.random.gamma(
                shape=2, scale=50, size=n_accounts
            ),
            # Behavioral controls
            "order_frequency": np.random.poisson(lam=4, size=n_accounts),
            "avg_order_value": np.random.uniform(50, 500, size=n_accounts),
            "customer_tenure_days": np.random.gamma(
                shape=2, scale=200, size=n_accounts
            ),
            "customer_return_rate": np.random.uniform(0, 1, size=n_accounts),
            # Target - SSL profit erosion
            "total_loss": np.random.gamma(shape=2, scale=500, size=n_accounts),
        }
    )

    return df


@pytest.fixture
def synthetic_regression_results():
    """
    Generate synthetic fitted regression results (TheLook baseline).

    Returns:
        tuple: (RegressionResults, coefficient_table DataFrames)
    """
    np.random.seed(42)
    n = 300

    # Create X variables (predictors, standardized)
    X = pd.DataFrame(
        {
            "const": np.ones(n),
            "return_frequency": np.random.normal(0, 1, n),
            "avg_basket_size": np.random.normal(0, 1, n),
            "purchase_recency_days": np.random.normal(0, 1, n),
            "order_frequency": np.random.normal(0, 1, n),
            "avg_order_value": np.random.normal(0, 1, n),
            "customer_tenure_days": np.random.normal(0, 1, n),
            "customer_return_rate": np.random.normal(0, 1, n),
        }
    )

    # Create Y (target, standardized)
    true_coefs = np.array([0.5, 0.66, -0.37, 0.003, 0.7, 0.002, 0.002, -0.005])
    y = X @ true_coefs + np.random.normal(0, 0.1, n)

    # Fit OLS
    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC3")

    return results


# ============================================================================
# TEST CLASS: TestCreateSSLRegressionTarget
# ============================================================================


class TestCreateSSLRegressionTarget:
    """Tests for create_ssl_regression_target() function."""

    def test_returns_dataframe(self, synthetic_ssl_accounts):
        """Test that function returns modified DataFrame."""
        result = create_ssl_regression_target(
            synthetic_ssl_accounts.copy(), loss_column="total_loss"
        )
        assert isinstance(result, pd.DataFrame)

    def test_creates_target_column(self, synthetic_ssl_accounts):
        """Test that target column is created."""
        result = create_ssl_regression_target(
            synthetic_ssl_accounts.copy(), loss_column="total_loss"
        )
        assert "total_profit_erosion_ssl" in result.columns

    def test_target_column_numeric(self, synthetic_ssl_accounts):
        """Test that target column is numeric."""
        result = create_ssl_regression_target(
            synthetic_ssl_accounts.copy(), loss_column="total_loss"
        )
        assert pd.api.types.is_numeric_dtype(result["total_profit_erosion_ssl"])

    def test_target_all_positive(self, synthetic_ssl_accounts):
        """Test that all target values are positive."""
        result = create_ssl_regression_target(
            synthetic_ssl_accounts.copy(), loss_column="total_loss"
        )
        assert (result["total_profit_erosion_ssl"] > 0).all()

    def test_target_statistics_reasonable(self, synthetic_ssl_accounts):
        """Test that target has reasonable statistics."""
        result = create_ssl_regression_target(
            synthetic_ssl_accounts.copy(), loss_column="total_loss"
        )
        target = result["total_profit_erosion_ssl"]

        # Check mean > 0 and std > 0
        assert target.mean() > 0
        assert target.std() > 0

        # Check range is reasonable
        assert target.min() >= 0


# ============================================================================
# TEST CLASS: TestEngineerSSLRegressionFeatures
# ============================================================================


class TestEngineerSSLRegressionFeatures:
    """Tests for engineer_ssl_regression_features() function."""

    def test_returns_dataframe(self, synthetic_ssl_accounts):
        """Test that function returns DataFrame."""
        result = engineer_ssl_regression_features(synthetic_ssl_accounts.copy())
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_features(self, synthetic_ssl_accounts):
        """Test that original features are preserved."""
        result = engineer_ssl_regression_features(synthetic_ssl_accounts.copy())

        original_features = [
            "return_frequency",
            "avg_basket_size",
            "purchase_recency_days",
            "order_frequency",
            "avg_order_value",
            "customer_tenure_days",
            "customer_return_rate",
        ]
        for feat in original_features:
            assert feat in result.columns

    def test_no_missing_values(self, synthetic_ssl_accounts):
        """Test that function handles missing values appropriately."""
        df = synthetic_ssl_accounts.copy()
        df.loc[0:5, "avg_basket_size"] = np.nan  # Add some missing values

        result = engineer_ssl_regression_features(df)

        # Check that result doesn't have unexpected NaNs in key features
        key_features = ["return_frequency", "order_frequency"]
        for feat in key_features:
            assert result[feat].notna().sum() > 0


# ============================================================================
# TEST CLASS: TestValidateCoefficientAlignment
# ============================================================================


class TestValidateCoefficientAlignment:
    """Tests for validate_coefficient_alignment() function."""

    def test_returns_dataframe(self, synthetic_regression_results):
        """Test that function returns DataFrame."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results  # Use same for testing

        comparison = validate_coefficient_alignment(
            thelook_results,
            ssl_results,
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        assert isinstance(comparison, pd.DataFrame)

    def test_has_required_columns(self, synthetic_regression_results):
        """Test that comparison DataFrame has all required columns."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        comparison = validate_coefficient_alignment(
            thelook_results,
            ssl_results,
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        required_cols = [
            "feature",
            "thelook_coefficient",
            "ssl_coefficient",
            "thelook_p_value",
            "ssl_p_value",
            "direction_aligned",
            "significance_agreement",
        ]
        for col in required_cols:
            assert col in comparison.columns

    def test_direction_aligned_is_boolean(self, synthetic_regression_results):
        """Test that direction_aligned column is boolean."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        comparison = validate_coefficient_alignment(
            thelook_results,
            ssl_results,
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        assert comparison["direction_aligned"].dtype == bool

    def test_alignment_values_reasonable(self, synthetic_regression_results):
        """Test that alignment metrics are in reasonable ranges."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        comparison = validate_coefficient_alignment(
            thelook_results,
            ssl_results,
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        # P-values should be in [0, 1]
        assert (comparison["thelook_p_value"] >= 0).all()
        assert (comparison["thelook_p_value"] <= 1).all()
        assert (comparison["ssl_p_value"] >= 0).all()
        assert (comparison["ssl_p_value"] <= 1).all()


# ============================================================================
# TEST CLASS: TestValidateDirectionalEffectSizes
# ============================================================================


class TestValidateDirectionalEffectSizes:
    """Tests for validate_directional_effect_sizes() function."""

    def test_returns_dict(self, synthetic_ssl_accounts, synthetic_regression_results):
        """Test that function returns dictionary."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        result = validate_directional_effect_sizes(
            synthetic_ssl_accounts,
            thelook_results,
            ssl_results,
            thelook_features=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
                "order_frequency",
                "avg_order_value",
                "customer_tenure_days",
            ],
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        assert isinstance(result, dict)

    def test_has_required_keys(
        self, synthetic_ssl_accounts, synthetic_regression_results
    ):
        """Test that result dictionary has required keys."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        result = validate_directional_effect_sizes(
            synthetic_ssl_accounts,
            thelook_results,
            ssl_results,
            thelook_features=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
                "order_frequency",
                "avg_order_value",
                "customer_tenure_days",
            ],
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        required_keys = [
            "effect_size_comparison",
            "model_fit_comparison",
            "generalization_score",
            "n_accounts_ssl",
        ]
        for key in required_keys:
            assert key in result

    def test_generalization_score_in_range(
        self, synthetic_ssl_accounts, synthetic_regression_results
    ):
        """Test that generalization score is in reasonable range."""
        thelook_results = synthetic_regression_results
        ssl_results = synthetic_regression_results

        result = validate_directional_effect_sizes(
            synthetic_ssl_accounts,
            thelook_results,
            ssl_results,
            thelook_features=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
                "order_frequency",
                "avg_order_value",
                "customer_tenure_days",
            ],
            hypothesis_predictors=[
                "return_frequency",
                "avg_basket_size",
                "purchase_recency_days",
            ],
        )

        gen_score = result["generalization_score"]
        assert 0 <= gen_score <= 1


# ============================================================================
# TEST CLASS: TestBuildValidationSummary
# ============================================================================


class TestBuildValidationSummary:
    """Tests for build_validation_summary() function."""

    def test_returns_dataframe(self, synthetic_regression_results):
        """Test that function returns DataFrame."""
        # Create dummy coefficient comparison and effect size result
        coef_comparison = pd.DataFrame(
            {
                "feature": ["return_frequency", "avg_basket_size"],
                "thelook_coefficient": [0.66, -0.37],
                "ssl_coefficient": [0.62, -0.35],
                "thelook_p_value": [0.001, 0.002],
                "ssl_p_value": [0.001, 0.003],
                "thelook_significant": [True, True],
                "ssl_significant": [True, True],
                "direction_aligned": [True, True],
                "significance_agreement": [True, True],
            }
        )

        effect_size_result = {
            "generalization_score": 0.45,
            "n_accounts_ssl": 500,
            "effect_size_comparison": pd.DataFrame(),
            "model_fit_comparison": {
                "thelook_r_squared": 0.45,
                "ssl_r_squared": 0.42,
                "r_squared_ratio": 0.93,
                "thelook_n_obs": 8547,
                "ssl_n_obs": 500,
                "thelook_aic": 5124.35,
                "ssl_aic": 5287.42,
            },
        }

        summary = build_validation_summary(coef_comparison, effect_size_result)
        assert isinstance(summary, pd.DataFrame)

    def test_has_metric_and_value_columns(self, synthetic_regression_results):
        """Test that summary has metric and value columns."""
        coef_comparison = pd.DataFrame(
            {
                "feature": ["return_frequency"],
                "thelook_coefficient": [0.66],
                "ssl_coefficient": [0.62],
                "thelook_p_value": [0.001],
                "ssl_p_value": [0.001],
                "thelook_significant": [True],
                "ssl_significant": [True],
                "direction_aligned": [True],
                "significance_agreement": [True],
            }
        )

        effect_size_result = {
            "generalization_score": 0.5,
            "n_accounts_ssl": 500,
            "effect_size_comparison": pd.DataFrame(),
            "model_fit_comparison": {
                "thelook_r_squared": 0.45,
                "ssl_r_squared": 0.42,
                "r_squared_ratio": 0.93,
                "thelook_n_obs": 8547,
                "ssl_n_obs": 500,
                "thelook_aic": 5124.35,
                "ssl_aic": 5287.42,
            },
        }

        summary = build_validation_summary(coef_comparison, effect_size_result)

        assert "metric" in summary.columns
        assert "value" in summary.columns

    def test_summary_not_empty(self):
        """Test that summary generates meaningful metrics."""
        coef_comparison = pd.DataFrame(
            {
                "feature": ["return_frequency", "avg_basket_size"],
                "thelook_coefficient": [0.66, -0.37],
                "ssl_coefficient": [0.62, -0.35],
                "thelook_p_value": [0.001, 0.002],
                "ssl_p_value": [0.001, 0.003],
                "thelook_significant": [True, True],
                "ssl_significant": [True, False],
                "direction_aligned": [True, True],
                "significance_agreement": [True, False],
            }
        )

        effect_size_result = {
            "generalization_score": 0.33,
            "n_accounts_ssl": 500,
            "effect_size_comparison": pd.DataFrame(),
            "model_fit_comparison": {
                "thelook_r_squared": 0.45,
                "ssl_r_squared": 0.42,
                "r_squared_ratio": 0.93,
                "thelook_n_obs": 8547,
                "ssl_n_obs": 500,
                "thelook_aic": 5124.35,
                "ssl_aic": 5287.42,
            },
        }

        summary = build_validation_summary(coef_comparison, effect_size_result)

        assert len(summary) > 0


# ============================================================================
# Integration Tests (requires real data)
# ============================================================================


@pytest.mark.integration
class TestRQ4ValidationIntegration:
    """Integration tests using actual RQ4 data files."""

    @pytest.fixture(scope="class")
    def real_data_available(self):
        """Check if real data files are available."""
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "processed"

        ssl_file = data_dir / "returns_eda_v1.csv"
        customer_file = data_dir / "customer_profit_erosion_targets.csv"

        return ssl_file.exists() and customer_file.exists()

    def test_full_validation_pipeline(self, real_data_available):
        """Test complete validation pipeline if data available."""
        if not real_data_available:
            pytest.skip("Real data files not available")

        from pathlib import Path

        from rq3_validation import engineer_ssl_account_features, load_ssl_data
        from rq4_econometrics import fit_ols_robust, load_rq4_data

        # Load data
        thelook_df = load_rq4_data()
        ssl_raw = load_ssl_data()
        ssl_accounts = engineer_ssl_account_features(ssl_raw)
        ssl_accounts = create_ssl_regression_target(ssl_accounts)

        # Check that accounts were created
        assert len(ssl_accounts) > 0
        assert "total_profit_erosion_ssl" in ssl_accounts.columns


# ============================================================================
# Test Helper Functions
# ============================================================================


def _check_ssl_data_available() -> bool:
    """
    Check if SSL data files are available for integration tests.

    Returns:
        bool: True if SSL data exists with valid schema
    """
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"

    ssl_file = data_dir / "returns_eda_v1.csv"

    if not ssl_file.exists():
        return False

    try:
        df = pd.read_csv(ssl_file, nrows=10)
        return len(df) > 0
    except Exception:
        return False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
