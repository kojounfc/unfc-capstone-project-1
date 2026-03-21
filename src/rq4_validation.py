"""
RQ4 External Validation module for Profit Erosion Capstone Project.

Validates OLS behavioral model against School Specialty LLC (SSL) data.
Two validation levels:
    Level 1 (Coefficient Validation): Do the same behavioral features matter in both datasets?
    Level 2 (Effect Size Validation): Does the TheLook regression model generalize to SSL accounts?

The SSL dataset contains return-related order lines aggregated to account level,
with features engineered to match TheLook's behavioral dimensions.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import (
    RANDOM_STATE,
    RQ4_ALPHA,
    RQ4_BEHAVIORAL_CONTROLS,
    RQ4_HYPOTHESIS_PREDICTORS,
    RQ4_TARGET_COL,
    SSL_RETURNS_CSV,
)
from src.rq3_validation import engineer_ssl_account_features, load_ssl_data
from src.rq4_econometrics import (
    extract_coefficient_table,
    fit_ols_robust,
    prepare_regression_data,
)

logger = logging.getLogger(__name__)


def create_ssl_regression_target(
    account_df: pd.DataFrame,
    loss_column: str = "total_loss",
) -> pd.DataFrame:
    """
    Create continuous regression target for SSL accounts using total_loss.

    Unlike RQ3 which creates binary classification, RQ4 uses continuous
    profit erosion equivalent. For SSL data, this is the total_loss aggregated
    from return-related transactions per account.

    Args:
        account_df: Account-level DataFrame from engineer_ssl_account_features().
        loss_column: Column containing total loss per account.

    Returns:
        Account DataFrame with 'total_profit_erosion_ssl' column added.
    """
    df = account_df.copy()

    # Rename for clarity
    if loss_column in df.columns:
        df["total_profit_erosion_ssl"] = df[loss_column]
    else:
        raise ValueError(f"Loss column '{loss_column}' not found in SSL data")

    n_accounts = len(df)
    mean_loss = df["total_profit_erosion_ssl"].mean()
    std_loss = df["total_profit_erosion_ssl"].std()

    logger.info(
        "SSL target created: %d accounts, mean erosion=%.2f, std=%.2f",
        n_accounts,
        mean_loss,
        std_loss,
    )

    return df


def engineer_ssl_regression_features(
    ssl_account_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ensure SSL account features match RQ4 regression requirements.

    Validates that all required features are present and handles demographics.
    For RQ4, the key features are:
    - Hypothesis predictors: return_frequency, avg_basket_size, purchase_recency_days
    - Behavioral controls: order_frequency, avg_order_value, customer_tenure_days,
                          customer_return_rate

    Args:
        ssl_account_df: Account-level DataFrame from engineer_ssl_account_features().

    Returns:
        SSL DataFrame with validated features for regression.
    """
    df = ssl_account_df.copy()

    # Define required features
    hypothesis_predictors = RQ4_HYPOTHESIS_PREDICTORS
    behavioral_controls = RQ4_BEHAVIORAL_CONTROLS

    required_features = hypothesis_predictors + behavioral_controls
    available_features = [f for f in required_features if f in df.columns]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        logger.warning(
            "Missing %d required features in SSL data: %s",
            len(missing_features),
            missing_features,
        )

    # Fill missing numeric features with 0 (conservative assumption)
    for feat in missing_features:
        if feat in required_features:
            logger.info("Creating placeholder for missing feature: %s", feat)
            df[feat] = 0.0

    # Add demographic placeholders (not available in SSL data)
    categorical_placeholders = [
        "user_gender",
        "traffic_source",
        "dominant_return_category",
    ]
    for cat_feat in categorical_placeholders:
        if cat_feat not in df.columns:
            df[cat_feat] = "unknown"
            logger.info("Added placeholder for demographic feature: %s", cat_feat)

    logger.info(
        "SSL regression features validated: %d/%d available",
        len(available_features),
        len(required_features),
    )

    return df


def validate_coefficient_alignment(
    thelook_results: sm.regression.linear_model.RegressionResultsWrapper,
    ssl_results: sm.regression.linear_model.RegressionResultsWrapper,
    hypothesis_predictors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Level 1 Coefficient Validation: Compare coefficient estimates, standard errors,
    and significance between TheLook and SSL datasets.

    Args:
        thelook_results: Fitted OLS results from TheLook data (rq4_econometrics.py).
        ssl_results: Fitted OLS results from SSL data.
        hypothesis_predictors: Feature names to compare (defaults to RQ4 hypothesis predictors).

    Returns:
        Comparison DataFrame with coefficients, p-values, effect size alignment from both datasets.
    """
    if hypothesis_predictors is None:
        hypothesis_predictors = RQ4_HYPOTHESIS_PREDICTORS

    # Extract coefficients from both models
    tl_coef_table = extract_coefficient_table(thelook_results)
    ssl_coef_table = extract_coefficient_table(ssl_results)

    comparison_rows = []

    for feat in hypothesis_predictors:
        # Find feature in TheLook results
        tl_row = tl_coef_table[tl_coef_table["feature"] == feat]
        ssl_row = ssl_coef_table[ssl_coef_table["feature"] == feat]

        if len(tl_row) == 0 or len(ssl_row) == 0:
            logger.warning("Feature %s not found in one or both model results", feat)
            continue

        tl_coef = tl_row["coefficient"].values[0]
        tl_se = tl_row["std_error"].values[0]
        tl_pval = tl_row["p_value"].values[0]
        tl_sig = tl_pval < 0.05

        ssl_coef = ssl_row["coefficient"].values[0]
        ssl_se = ssl_row["std_error"].values[0]
        ssl_pval = ssl_row["p_value"].values[0]
        ssl_sig = ssl_pval < 0.05

        # Directional alignment — only meaningful when TheLook coefficient is
        # statistically significant; if TheLook's p ~ 1.0, the direction is
        # unreliable and should not count as aligned regardless of SSL.
        same_sign = (tl_coef > 0 and ssl_coef > 0) or (tl_coef < 0 and ssl_coef < 0)
        direction_aligned = same_sign and tl_sig

        # Significance agreement
        sig_agreement = tl_sig == ssl_sig

        # Effect size comparison (coefficient ratio)
        coef_ratio = ssl_coef / tl_coef if tl_coef != 0 else np.nan

        comparison_rows.append(
            {
                "feature": feat,
                "thelook_coefficient": round(tl_coef, 4),
                "ssl_coefficient": round(ssl_coef, 4),
                "thelook_pct_effect": round((np.exp(tl_coef) - 1) * 100, 2),
                "ssl_pct_effect": round((np.exp(ssl_coef) - 1) * 100, 2),
                "thelook_p_value": round(tl_pval, 4),
                "ssl_p_value": round(ssl_pval, 4),
                "thelook_significant": tl_sig,
                "ssl_significant": ssl_sig,
                "direction_aligned": direction_aligned,
                "significance_agreement": sig_agreement,
                "coefficient_ratio": (
                    round(coef_ratio, 2) if not np.isnan(coef_ratio) else np.nan
                ),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    n_aligned = comparison_df["direction_aligned"].sum()
    n_sig_agree = comparison_df["significance_agreement"].sum()

    logger.info(
        "Coefficient alignment: %d/%d direction aligned, %d/%d significance agreement",
        n_aligned,
        len(comparison_df),
        n_sig_agree,
        len(comparison_df),
    )

    return comparison_df


def validate_directional_effect_sizes(
    ssl_account_df: pd.DataFrame,
    thelook_results: sm.regression.linear_model.RegressionResultsWrapper,
    ssl_results: sm.regression.linear_model.RegressionResultsWrapper,
    thelook_features: List[str],
    hypothesis_predictors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Level 2 Effect Size Validation: Compare standardized effect sizes between datasets
    and check if behavioral associations generalize to SSL accounts.

    Args:
        ssl_account_df: SSL account-level data with features.
        thelook_results: Fitted OLS results from TheLook data.
        ssl_results: Fitted OLS results from SSL data.
        thelook_features: Feature names the TheLook model was trained on.
        hypothesis_predictors: Features to validate (defaults to RQ4 hypothesis predictors).

    Returns:
        Dict with effect size comparisons and generalization metrics.
    """
    if hypothesis_predictors is None:
        hypothesis_predictors = RQ4_HYPOTHESIS_PREDICTORS

    tl_coef_table = extract_coefficient_table(thelook_results)
    ssl_coef_table = extract_coefficient_table(ssl_results)

    # Compare model fit
    model_fit_comparison = {
        "thelook_r_squared": round(thelook_results.rsquared, 4),
        "ssl_r_squared": round(ssl_results.rsquared, 4),
        "r_squared_ratio": (
            round(ssl_results.rsquared / thelook_results.rsquared, 2)
            if thelook_results.rsquared > 0
            else np.nan
        ),
        "thelook_n_obs": thelook_results.nobs,
        "ssl_n_obs": ssl_results.nobs,
        "thelook_aic": round(thelook_results.aic, 2),
        "ssl_aic": round(ssl_results.aic, 2),
    }

    # Effect size validation for hypothesis predictors
    effect_sizes = []

    for feat in hypothesis_predictors:
        tl_row = tl_coef_table[tl_coef_table["feature"] == feat]
        ssl_row = ssl_coef_table[ssl_coef_table["feature"] == feat]

        if len(tl_row) == 0 or len(ssl_row) == 0:
            continue

        tl_coef = tl_row["coefficient"].values[0]
        tl_std_err = tl_row["std_error"].values[0]
        ssl_coef = ssl_row["coefficient"].values[0]
        ssl_std_err = ssl_row["std_error"].values[0]

        # Standardized effect sizes (Cohen's d-like)
        tl_effect = tl_coef / tl_std_err if tl_std_err > 0 else 0
        ssl_effect = ssl_coef / ssl_std_err if ssl_std_err > 0 else 0

        # Effect size ratio
        effect_ratio = ssl_effect / tl_effect if tl_effect != 0 else np.nan

        effect_sizes.append(
            {
                "feature": feat,
                "thelook_effect_size": round(tl_effect, 3),
                "ssl_effect_size": round(ssl_effect, 3),
                "effect_size_ratio": round(effect_ratio, 2),
            }
        )

    effect_sizes_df = pd.DataFrame(effect_sizes)

    # Generalization assessment
    all_hypothesis_sig_in_thelook = (
        tl_coef_table[tl_coef_table["feature"].isin(hypothesis_predictors)]["p_value"]
        < 0.05
    ).all()

    all_hypothesis_sig_in_ssl = (
        ssl_coef_table[ssl_coef_table["feature"].isin(hypothesis_predictors)]["p_value"]
        < 0.05
    ).all()

    generalization_score = (
        effect_sizes_df["effect_size_ratio"].abs() > 0.5
    ).sum() / len(effect_sizes_df)

    result = {
        "model_fit_comparison": model_fit_comparison,
        "effect_size_comparison": effect_sizes_df,
        "hypothesis_predictors_thelook_sig": bool(all_hypothesis_sig_in_thelook),
        "hypothesis_predictors_ssl_sig": bool(all_hypothesis_sig_in_ssl),
        "generalization_score": round(generalization_score, 2),
        "n_accounts_ssl": len(ssl_account_df),
    }

    logger.info(
        "Effect size validation: TL R²=%.4f, SSL R²=%.4f, generalization_score=%.2f",
        model_fit_comparison["thelook_r_squared"],
        model_fit_comparison["ssl_r_squared"],
        generalization_score,
    )

    return result


def build_validation_summary(
    coefficient_comparison: pd.DataFrame,
    effect_size_result: Dict[str, Any],
    diagnostics_log: Optional[Dict] = None,
    jb_linear_stat: Optional[float] = None,
    vif_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Summarize coefficient alignment, effect size validation, and model diagnostics.

    Args:
        coefficient_comparison: Output from validate_coefficient_alignment().
        effect_size_result: Output from validate_directional_effect_sizes().
        diagnostics_log: Optional output from run_diagnostics() on the log-linear model.
        jb_linear_stat: Optional Jarque-Bera statistic from the linear model (for ratio).
        vif_df: Optional VIF DataFrame from calculate_vif() — max VIF is extracted.

    Returns:
        Summary DataFrame with key validation metrics and interpretations.
    """
    n_features = len(coefficient_comparison)
    n_direction_aligned = coefficient_comparison["direction_aligned"].sum()
    n_sig_agreement = coefficient_comparison["significance_agreement"].sum()
    n_both_sig = (
        coefficient_comparison["thelook_significant"]
        & coefficient_comparison["ssl_significant"]
    ).sum()

    model_fit = effect_size_result["model_fit_comparison"]

    rows = [
        {"metric": "n_hypothesis_predictors", "value": n_features},
        {
            "metric": "direction_aligned_count",
            "value": n_direction_aligned,
        },
        {
            "metric": "direction_aligned_pct",
            "value": (
                round(n_direction_aligned / n_features * 100, 1)
                if n_features > 0
                else 0
            ),
        },
        {
            "metric": "significance_agreement_count",
            "value": n_sig_agreement,
        },
        {
            "metric": "significance_agreement_pct",
            "value": (
                round(n_sig_agreement / n_features * 100, 1) if n_features > 0 else 0
            ),
        },
        {
            "metric": "both_datasets_significant",
            "value": n_both_sig,
        },
        {
            "metric": "thelook_r_squared",
            "value": model_fit["thelook_r_squared"],
        },
        {
            "metric": "ssl_r_squared",
            "value": model_fit["ssl_r_squared"],
        },
        {
            "metric": "r_squared_ratio_ssl_to_thelook",
            "value": model_fit["r_squared_ratio"],
        },
        {
            "metric": "generalization_score",
            "value": effect_size_result["generalization_score"],
        },
        {
            "metric": "ssl_accounts_validated",
            "value": effect_size_result["n_accounts_ssl"],
        },
        {
            "metric": "thelook_nobs",
            "value": model_fit["thelook_n_obs"],
        },
    ]

    if diagnostics_log is not None and jb_linear_stat is not None:
        jb_log_stat = diagnostics_log["jarque_bera"]["statistic"]
        rows += [
            {"metric": "jb_linear_model", "value": jb_linear_stat},
            {"metric": "jb_log_linear_model", "value": jb_log_stat},
            {
                "metric": "jb_improvement_ratio",
                "value": round(jb_linear_stat / jb_log_stat, 1) if jb_log_stat else None,
            },
            {"metric": "bp_statistic", "value": diagnostics_log["breusch_pagan"]["statistic"]},
            {"metric": "bp_pvalue", "value": diagnostics_log["breusch_pagan"]["pvalue"]},
            {"metric": "durbin_watson_statistic", "value": diagnostics_log["durbin_watson"]},
        ]

    if vif_df is not None:
        rows += [{"metric": "max_vif", "value": float(vif_df["VIF"].max())}]

    summary_df = pd.DataFrame(rows)

    logger.info(
        "Validation summary: direction_aligned=%.1f%%, sig_agreement=%.1f%%, "
        "both_significant=%d, generalization_score=%.2f",
        n_direction_aligned / n_features * 100 if n_features > 0 else 0,
        n_sig_agreement / n_features * 100 if n_features > 0 else 0,
        n_both_sig,
        effect_size_result["generalization_score"],
    )

    return summary_df


def run_full_rq4_validation(
    thelook_results: sm.regression.linear_model.RegressionResultsWrapper,
    thelook_data: pd.DataFrame,
    ssl_filepath: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run complete RQ4 external validation pipeline on SSL data.

    This is the main entry point for RQ4 validation that orchestrates:
    1. Load and engineer SSL data
    2. Create regression targets and features
    3. Fit OLS regression on SSL data
    4. Validate coefficient alignment (Level 1)
    5. Validate effect size generalization (Level 2)
    6. Build comprehensive validation summary

    Args:
        thelook_results: Fitted OLS results from TheLook data (rq4_econometrics.py).
        thelook_data: Original regression-ready TheLook data used for fitting.
        ssl_filepath: Path to SSL CSV file (defaults to config SSL_RETURNS_CSV).

    Returns:
        Dict with full validation results including:
        - coefficient_comparison: Level 1 validation results
        - effect_size_result: Level 2 validation results
        - ssl_regression_results: OLS regression results on SSL data
        - validation_summary: Aggregated metrics
    """
    logger.info("Starting RQ4 external validation pipeline on SSL data")

    # Step 1: Load and engineer SSL data
    ssl_df = load_ssl_data(ssl_filepath)
    ssl_account_df = engineer_ssl_account_features(ssl_df)
    logger.info("SSL data engineered: %d accounts", len(ssl_account_df))

    # Step 2: Create targets and features
    ssl_account_df = create_ssl_regression_target(ssl_account_df)
    ssl_account_df = engineer_ssl_regression_features(ssl_account_df)
    logger.info("SSL regression features validated")

    # Step 3: Prepare data for regression (matching TheLook data preparation)
    ssl_regression_data = prepare_regression_data(
        ssl_account_df,
        target_col="total_profit_erosion_ssl",
        numeric_features=RQ4_HYPOTHESIS_PREDICTORS + RQ4_BEHAVIORAL_CONTROLS,
        categorical_features=[
            "user_gender",
            "traffic_source",
            "dominant_return_category",
        ],
        exclude_features=[],
    )
    logger.info("SSL regression data prepared: %d rows", len(ssl_regression_data))

    # Step 4: Fit OLS regression on SSL data
    ssl_results = fit_ols_robust(ssl_regression_data, "total_profit_erosion_ssl")
    logger.info(
        "SSL OLS regression fitted: R²=%.4f, F-stat=%.2f (p<%.2e)",
        ssl_results.rsquared,
        ssl_results.fvalue,
        ssl_results.f_pvalue,
    )

    # Step 5: Level 1 - Coefficient Validation
    coefficient_comparison = validate_coefficient_alignment(
        thelook_results, ssl_results, RQ4_HYPOTHESIS_PREDICTORS
    )

    # Step 6: Level 2 - Effect Size Validation
    effect_size_result = validate_directional_effect_sizes(
        ssl_account_df,
        thelook_results,
        ssl_results,
        RQ4_HYPOTHESIS_PREDICTORS + RQ4_BEHAVIORAL_CONTROLS,
        RQ4_HYPOTHESIS_PREDICTORS,
    )

    # Step 7: Build summary
    validation_summary = build_validation_summary(
        coefficient_comparison, effect_size_result
    )

    return {
        "coefficient_comparison": coefficient_comparison,
        "effect_size_result": effect_size_result,
        "ssl_regression_results": ssl_results,
        "ssl_account_data": ssl_account_df,
        "ssl_regression_data": ssl_regression_data,
        "validation_summary": validation_summary,
    }
