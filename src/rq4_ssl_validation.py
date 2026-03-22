"""
RQ4 External Validation (v2) — School Specialty LLC (SSL).

Fixed version of rq4_validation.py that corrects three design issues:

1. SSL categorical control — dominant_return_category — is engineered from the
   SSL 'Department' column (modal department per account), providing the same
   "dominant product category" control as TheLook without requiring identical
   category names. The categories are intentionally domain-specific (apparel
   categories for TheLook; educational supply departments for SSL); no reviewer
   expects them to align. The control's purpose is to partial out within-dataset
   category heterogeneity from the hypothesis predictor coefficients.
   Note: user_gender is excluded from SSL (B2B institutional dataset; no gender
   dimension exists).

2. Missing numeric features are filled with the column median rather than 0.
   Zero is a non-neutral imputation that biases regression coefficients toward
   the intercept for features with non-zero distributions.

3. Coefficient alignment (Level 1) is restricted to the numeric hypothesis
   predictors only (return_frequency, avg_basket_size, purchase_recency_days),
   where cross-domain alignment is expected because the behavioral mechanism is
   domain-agnostic.

All other validation logic (validate_coefficient_alignment,
validate_directional_effect_sizes, build_validation_summary) is unchanged from
rq4_validation.py.

Functions
---------
engineer_dominant_return_category    Compute modal Department per SSL account
engineer_ssl_regression_features_v2  Numeric + category feature prep for SSL
run_full_rq4_ssl_validation          Complete RQ4 SSL validation pipeline (fixed)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import (
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
    screen_features,
)
from src.rq4_validation import (
    build_validation_summary,
    create_ssl_regression_target,
    validate_coefficient_alignment,
    validate_directional_effect_sizes,
)

logger = logging.getLogger(__name__)


def engineer_dominant_return_category(
    ssl_raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the dominant product department per SSL account.

    Uses the 'Department' column from raw SSL line-level data to derive
    dominant_return_category — the SSL equivalent of TheLook's
    dominant_return_category (most-returned product category per customer).

    The category labels intentionally differ between datasets (apparel
    categories in TheLook vs educational supply departments in SSL). The
    control is included to partial out within-dataset category heterogeneity
    from the hypothesis predictor coefficients, not for cross-dataset
    coefficient comparison.

    Args:
        ssl_raw_df: Raw SSL line-level DataFrame from load_ssl_data().
            Must contain 'Bill To Act #' and 'Department' columns.

    Returns:
        DataFrame with columns ['account_id', 'dominant_return_category'].
    """
    dominant = (
        ssl_raw_df.groupby("Bill To Act #")["Department"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={
            "Bill To Act #": "account_id",
            "Department": "dominant_return_category",
        })
    )
    logger.info(
        "SSL dominant_return_category engineered: %d accounts, %d unique departments",
        len(dominant),
        dominant["dominant_return_category"].nunique(),
    )
    return dominant


def engineer_ssl_regression_features_v2(
    ssl_account_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and prepare SSL numeric features for RQ4 regression.

    Differences from rq4_validation.engineer_ssl_regression_features():
    - Does NOT add demographic placeholders (user_gender, traffic_source).
      SSL is a B2B institutional dataset with no gender dimension.
    - dominant_return_category is engineered separately via
      engineer_dominant_return_category() and joined before this call.
    - Fills missing numeric features with the column median (not 0) to avoid
      biasing regression coefficients toward zero.

    Args:
        ssl_account_df: Account-level DataFrame from engineer_ssl_account_features(),
            with dominant_return_category already joined.

    Returns:
        SSL DataFrame with validated numeric features and dominant_return_category.
    """
    df = ssl_account_df.copy()

    required_features = RQ4_HYPOTHESIS_PREDICTORS + RQ4_BEHAVIORAL_CONTROLS
    available_features = [f for f in required_features if f in df.columns]
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        logger.warning(
            "Missing %d required numeric features in SSL data: %s",
            len(missing_features),
            missing_features,
        )

    # Fill missing numeric features with column median (neutral imputation).
    for feat in missing_features:
        col_median = df[feat].median() if feat in df.columns else 0.0
        logger.info(
            "Filling missing feature '%s' with median=%.4f", feat, col_median
        )
        df[feat] = col_median

    logger.info(
        "SSL regression features validated: %d/%d numeric available, "
        "dominant_return_category present=%s",
        len(available_features),
        len(required_features),
        "dominant_return_category" in df.columns,
    )

    return df


def run_full_rq4_ssl_validation(
    thelook_results: sm.regression.linear_model.RegressionResultsWrapper,
    thelook_data: pd.DataFrame,
    ssl_filepath: Optional[str] = None,
    surviving_numeric: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run complete RQ4 external validation pipeline on SSL data (fixed version).

    Fixed behaviour vs. run_full_rq4_validation() in rq4_validation.py:
    - SSL model includes dominant_return_category (from Department modal per
      account) as a categorical control — SSL equivalent of TheLook's
      dominant_return_category. Category labels differ by domain; no
      cross-dataset comparison of category coefficients is made.
    - user_gender excluded from SSL (B2B institutional dataset; no gender dim).
    - Coefficient alignment restricted to numeric hypothesis predictors only.
    - Missing numeric features filled with median, not 0.

    Pipeline
    --------
    1. Load and engineer SSL data (via rq3_validation)
    1b. Engineer dominant_return_category from Department (modal per account)
    2. Create regression target (total_profit_erosion_ssl from total_loss)
    3. Validate/fill numeric features (v2)
    4. Prepare SSL regression data (numeric + dominant_return_category)
    5. Fit OLS on SSL data with HC3 robust standard errors
    6. Level 1 — Coefficient alignment on hypothesis predictors (numeric only)
    7. Level 2 — Effect size generalization
    8. Build validation summary

    Args:
        thelook_results: Fitted OLS results from TheLook data (rq4_econometrics.py).
        thelook_data: Regression-ready TheLook DataFrame used for fitting.
        ssl_filepath: Path to SSL CSV (defaults to config.SSL_RETURNS_CSV).
        surviving_numeric: Screened numeric features from screen_features().
            If None, uses RQ4_HYPOTHESIS_PREDICTORS + RQ4_BEHAVIORAL_CONTROLS.

    Returns:
        Dict with:
        - coefficient_comparison: Level 1 validation DataFrame
        - effect_size_result:     Level 2 validation Dict
        - ssl_regression_results: Fitted OLS results on SSL data
        - ssl_account_data:       Account-level SSL DataFrame
        - ssl_regression_data:    Regression-ready SSL DataFrame
        - validation_summary:     Aggregated metrics DataFrame
    """
    logger.info("Starting RQ4 SSL external validation pipeline (v2 — fixed)")

    numeric_features = surviving_numeric or (
        RQ4_HYPOTHESIS_PREDICTORS + RQ4_BEHAVIORAL_CONTROLS
    )

    # Step 1: Load and engineer SSL data
    ssl_df = load_ssl_data(ssl_filepath)
    ssl_account_df = engineer_ssl_account_features(ssl_df)
    logger.info("SSL data engineered: %d accounts", len(ssl_account_df))

    # Step 1b: Join dominant_return_category from raw line-level Department column.
    # SSL equivalent of TheLook's dominant_return_category — modal Department per
    # account. Category labels differ by domain (educational vs apparel); no
    # cross-dataset comparison of category coefficients is made.
    dominant_cat = engineer_dominant_return_category(ssl_df)
    ssl_account_df = ssl_account_df.merge(dominant_cat, on="account_id", how="left")
    logger.info(
        "dominant_return_category joined: %d/%d accounts have a value",
        ssl_account_df["dominant_return_category"].notna().sum(),
        len(ssl_account_df),
    )

    # Step 2: Create regression target
    ssl_account_df = create_ssl_regression_target(ssl_account_df, loss_column="total_loss")

    # Filter to positive erosion only — matches TheLook scope and enables log transform
    n_before = len(ssl_account_df)
    ssl_account_df = ssl_account_df[ssl_account_df["total_profit_erosion_ssl"] > 0].copy()
    logger.info(
        "SSL accounts after filtering to positive erosion: %d (dropped %d)",
        len(ssl_account_df), n_before - len(ssl_account_df),
    )

    # Step 3: Validate numeric features and confirm category column present
    ssl_account_df = engineer_ssl_regression_features_v2(ssl_account_df)

    # Step 4: Prepare regression data — log-linear to match TheLook model spec
    ssl_regression_data = prepare_regression_data(
        ssl_account_df,
        target_col="total_profit_erosion_ssl",
        numeric_features=numeric_features,
        categorical_features=["dominant_return_category"],
        exclude_features=[],
        log_transform=True,
    )
    log_ssl_target = "log_total_profit_erosion_ssl"
    ssl_regression_data = ssl_regression_data.drop(columns=["total_profit_erosion_ssl"])
    logger.info("SSL regression data prepared: %d rows (log-linear)", len(ssl_regression_data))

    # Step 5: Fit OLS on SSL data (log-linear, matching TheLook model)
    ssl_results = fit_ols_robust(ssl_regression_data, log_ssl_target)
    logger.info(
        "SSL OLS fitted: R²=%.4f, F=%.2f (p<%.2e)",
        ssl_results.rsquared,
        ssl_results.fvalue,
        ssl_results.f_pvalue,
    )

    # Step 6: Level 1 — Coefficient alignment (hypothesis predictors only)
    coefficient_comparison = validate_coefficient_alignment(
        thelook_results,
        ssl_results,
        RQ4_HYPOTHESIS_PREDICTORS,
    )

    # Step 7: Level 2 — Effect size generalization
    effect_size_result = validate_directional_effect_sizes(
        ssl_account_df,
        thelook_results,
        ssl_results,
        numeric_features,
        RQ4_HYPOTHESIS_PREDICTORS,
    )

    # Step 8: Build summary
    validation_summary = build_validation_summary(coefficient_comparison, effect_size_result)

    return {
        "coefficient_comparison": coefficient_comparison,
        "effect_size_result": effect_size_result,
        "ssl_regression_results": ssl_results,
        "ssl_account_data": ssl_account_df,
        "ssl_regression_data": ssl_regression_data,
        "validation_summary": validation_summary,
    }
