"""
RQ3 External Validation module for Profit Erosion Capstone Project.

Validates TheLook predictive model against School Specialty LLC (SSL) data.
Two validation levels:
    Level 1 (Pattern): Do the same features matter in both datasets?
    Level 2 (Directional): Does the TheLook model generalize to SSL accounts?

Validates TheLook predictive model against School Specialty LLC (SSL) data.
SSL data contains return-related order lines with two Sales_Type values:
    - RETURN: Actual return of goods (credit/refund issued, negative qty)
    - ORDER: No-charge replacement shipments (CreditReturn Sales ≈ 0, positive qty)
Feature engineering distinguishes these to produce accurate mappings.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RANDOM_STATE, SSL_RETURNS_CSV
from src.rq3_modeling import screen_features

logger = logging.getLogger(__name__)


def load_ssl_data(
    filepath: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load SSL returns CSV and perform basic cleaning.

    The SSL dataset contains return-related order lines with two Sales_Type
    values: RETURN (actual returns) and ORDER (no-charge replacements).

    Args:
        filepath: Path to SSL CSV file. Defaults to config SSL_RETURNS_CSV.

    Returns:
        Cleaned SSL DataFrame with parsed dates.
    """
    if filepath is None:
        filepath = SSL_RETURNS_CSV

    df = pd.read_csv(filepath)
    logger.info("Loaded SSL data: %d rows, %d columns", len(df), len(df.columns))

    # Parse date columns
    date_cols = ["Booked Date", "Billed Date", "Reference Booked Date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Drop rows with missing account identifier
    initial_len = len(df)
    df = df.dropna(subset=["Bill To Act #"])
    if len(df) < initial_len:
        logger.info(
            "Dropped %d rows with missing 'Bill To Act #'", initial_len - len(df)
        )

    logger.info(
        "SSL data after cleaning: %d rows, %d unique accounts",
        len(df),
        df["Bill To Act #"].nunique(),
    )

    return df


def engineer_ssl_account_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate SSL order lines to account level, producing features analogous
    to TheLook's 12 candidate features.

    The SSL dataset contains two Sales_Type values:
        - RETURN: Actual return of goods (credit/refund, negative qty)
        - ORDER: No-charge replacement shipments (CreditReturn Sales ≈ 0)

    Feature mappings distinguish these:
        order_frequency      = unique Order Number count (all lines)
        return_frequency     = count of actual RETURN lines only
        customer_return_rate = RETURN lines / total lines per account
        avg_basket_size      = mean Lines Per Order (all lines)
        avg_order_value      = mean Reference Sale Amount per order (all lines)
        total_items          = total lines per account (all types)
        total_sales          = sum of Reference Sale Amount (all lines)
        total_margin         = sum of gross_financial_loss (all lines)
        avg_item_price       = mean |CreditReturn Sales / Ordered Qty| (RETURN lines only)
        avg_item_margin      = mean gross_financial_loss per line (all lines)
        customer_tenure_days = date range of Booked Date per account
        purchase_recency_days = days from last Booked Date to max date in dataset

    Args:
        df: Cleaned SSL DataFrame from load_ssl_data().

    Returns:
        Account-level DataFrame with one row per account.
    """
    df = df.copy()
    account_col = "Bill To Act #"
    max_date = df["Booked Date"].max()

    # Compute per-line item price using Reference Sale Amount (original sale
    # price) from ALL line types — semantically closest to TheLook's
    # avg_item_price.  ORDER lines carry Reference Sale Amount (price of the
    # item being replaced) but have CreditReturn Sales ≈ $0, so CreditReturn
    # Sales is only used as fallback on RETURN lines where Reference Sale
    # Amount is unavailable.  This increases account-level coverage from
    # 37.7% to ~90%.
    is_return = (
        df["Sales_Type"] == "RETURN"
        if "Sales_Type" in df.columns
        else pd.Series(True, index=df.index)
    )
    has_qty = df["Ordered Qty"].abs() > 0

    if "Reference Sale Amount" in df.columns:
        has_ref = df["Reference Sale Amount"].notna() & (
            df["Reference Sale Amount"] > 0
        )
        df["_item_price"] = np.where(
            has_qty & has_ref,
            df["Reference Sale Amount"] / df["Ordered Qty"].abs(),
            np.where(
                is_return & has_qty,
                df["CreditReturn Sales"].abs() / df["Ordered Qty"].abs(),
                np.nan,
            ),
        )
    else:
        # Fallback for DataFrames without Reference Sale Amount column
        df["_item_price"] = np.where(
            is_return & has_qty,
            df["CreditReturn Sales"].abs() / df["Ordered Qty"].abs(),
            np.nan,
        )

    # Count actual RETURN lines per account
    df["_is_return"] = is_return.astype(int)

    # Order-level aggregation (for avg_order_value)
    order_agg = (
        df.groupby([account_col, "Order Number"])
        .agg(order_ref_sale=("Reference Sale Amount", "sum"))
        .reset_index()
    )
    avg_order_value = (
        order_agg.groupby(account_col)["order_ref_sale"]
        .mean()
        .rename("avg_order_value")
    )

    # Account-level aggregation
    account_features = df.groupby(account_col).agg(
        order_frequency=("Order Number", "nunique"),
        return_frequency=("_is_return", "sum"),
        avg_basket_size=("Lines Per Order", "mean"),
        total_items=("Order Line ID", "count"),
        total_sales=("Reference Sale Amount", "sum"),
        total_margin=("gross_financial_loss", "sum"),
        avg_item_price=("_item_price", "mean"),
        avg_item_margin=("gross_financial_loss", "mean"),
        _min_date=("Booked Date", "min"),
        _max_date=("Booked Date", "max"),
        # Keep total_loss for target construction
        total_loss=("total_loss", "sum"),
    )

    # customer_return_rate: actual RETURN lines / total lines
    # Produces meaningful variance (not always 1.0)
    account_features["customer_return_rate"] = (
        account_features["return_frequency"] / account_features["total_items"]
    )

    # Tenure: days between first and last booked date
    account_features["customer_tenure_days"] = (
        (account_features["_max_date"] - account_features["_min_date"]).dt.days
    )

    # Recency: days from last booked date to dataset max date
    account_features["purchase_recency_days"] = (
        (max_date - account_features["_max_date"]).dt.days
    )

    # Join avg_order_value
    account_features = account_features.join(avg_order_value)

    # Drop temp columns
    account_features = account_features.drop(columns=["_min_date", "_max_date"])

    # Reset index to make account a column
    account_features = account_features.reset_index()
    account_features = account_features.rename(columns={account_col: "account_id"})

    logger.info(
        "Engineered SSL account features: %d accounts, %d columns",
        len(account_features),
        len(account_features.columns),
    )

    return account_features


def create_ssl_targets(
    account_df: pd.DataFrame,
    loss_column: str = "total_loss",
    percentile: float = 75.0,
) -> pd.DataFrame:
    """
    Create binary target variable for SSL accounts using the 75th percentile
    of total_loss, mirroring TheLook methodology.

    Args:
        account_df: Account-level DataFrame from engineer_ssl_account_features().
        loss_column: Column to threshold on.
        percentile: Percentile for high-loss classification.

    Returns:
        Account DataFrame with 'is_high_loss_account' column added.
    """
    df = account_df.copy()
    threshold = np.percentile(df[loss_column].dropna(), percentile)
    df["is_high_loss_account"] = (df[loss_column] >= threshold).astype(int)

    n_high = df["is_high_loss_account"].sum()
    logger.info(
        "SSL target: threshold=%.2f (%.0fth pct), %d/%d (%.1f%%) high-loss accounts",
        threshold,
        percentile,
        n_high,
        len(df),
        n_high / len(df) * 100,
    )

    return df


def validate_feature_patterns(
    ssl_account_df: pd.DataFrame,
    thelook_screening: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = "is_high_loss_account",
) -> pd.DataFrame:
    """
    Level 1 Pattern Validation: Run the same 3-gate feature screening on SSL
    data independently and compare against TheLook screening results.

    Args:
        ssl_account_df: SSL account-level data with features and target.
        thelook_screening: TheLook screening report from screen_features().
        feature_columns: Feature columns to screen. Defaults to TheLook candidates
            that exist in SSL data.
        target_column: SSL target column name.

    Returns:
        Comparison DataFrame showing which features survived in each dataset.
    """
    from src.config import RQ3_CANDIDATE_FEATURES

    if feature_columns is None:
        feature_columns = [
            f for f in RQ3_CANDIDATE_FEATURES if f in ssl_account_df.columns
        ]

    # Prepare SSL features for screening
    X_ssl = ssl_account_df[feature_columns].copy()
    y_ssl = ssl_account_df[target_column].copy()

    # Impute missing values
    for col in X_ssl.columns:
        if X_ssl[col].isna().any():
            X_ssl[col] = X_ssl[col].fillna(X_ssl[col].median())

    # Run screening on SSL data
    ssl_surviving, ssl_report = screen_features(X_ssl, y_ssl)

    # Build comparison
    thelook_status = dict(
        zip(thelook_screening["feature"], thelook_screening["final_status"])
    )

    comparison_rows = []
    for feat in feature_columns:
        tl_status = thelook_status.get(feat, "not_available")
        ssl_row = ssl_report[ssl_report["feature"] == feat]
        ssl_status = ssl_row["final_status"].values[0] if len(ssl_row) > 0 else "not_available"

        comparison_rows.append({
            "feature": feat,
            "thelook_status": tl_status,
            "ssl_status": ssl_status,
            "both_pass": tl_status == "pass" and ssl_status == "pass",
            "both_fail": tl_status == "fail" and ssl_status == "fail",
            "agreement": tl_status == ssl_status,
        })

    comparison_df = pd.DataFrame(comparison_rows)

    n_agree = comparison_df["agreement"].sum()
    n_both_pass = comparison_df["both_pass"].sum()
    logger.info(
        "Pattern validation: %d/%d features agree, %d pass in both datasets",
        n_agree,
        len(comparison_df),
        n_both_pass,
    )

    return comparison_df


def validate_directional_predictions(
    ssl_account_df: pd.DataFrame,
    thelook_model: Any,
    thelook_features: List[str],
    scaler: Optional[Any] = None,
    target_column: str = "is_high_loss_account",
    loss_column: str = "total_loss",
) -> Dict[str, Any]:
    """
    Level 2 Directional Validation: Apply the TheLook-trained model to SSL
    data and check directional alignment between predictions and actual loss.

    Args:
        ssl_account_df: SSL account-level data with features and target.
        thelook_model: Trained model from TheLook pipeline.
        thelook_features: Feature names the model was trained on.
        scaler: Optional StandardScaler if model requires scaled input.
        target_column: SSL target column name.
        loss_column: Actual loss column for directional comparison.

    Returns:
        Dict with directional alignment metrics:
        - directional_accuracy: % of accounts where prediction direction matches actual
        - rank_correlation: Spearman correlation between predicted probability and actual loss
        - predicted_high_pct: % of accounts predicted as high-risk
        - actual_high_pct: % of accounts actually high-loss
        - confusion_at_directional: confusion matrix (predicted vs actual binary)
        - predictions_df: DataFrame with account-level predictions
    """
    # Identify available features
    available_features = [f for f in thelook_features if f in ssl_account_df.columns]
    missing_features = [f for f in thelook_features if f not in ssl_account_df.columns]

    if missing_features:
        logger.warning(
            "Missing %d features in SSL data: %s. Imputing with zeros.",
            len(missing_features),
            missing_features,
        )

    # Prepare feature matrix (matching TheLook feature order)
    X_ssl = pd.DataFrame(index=ssl_account_df.index)
    for feat in thelook_features:
        if feat in ssl_account_df.columns:
            X_ssl[feat] = ssl_account_df[feat].fillna(
                ssl_account_df[feat].median()
            )
        else:
            X_ssl[feat] = 0.0

    # Scale if needed
    if scaler is not None:
        X_ssl = pd.DataFrame(
            scaler.transform(X_ssl),
            columns=X_ssl.columns,
            index=X_ssl.index,
        )

    # Predict
    y_proba = thelook_model.predict_proba(X_ssl)[:, 1]
    y_pred = thelook_model.predict(X_ssl)

    y_actual = ssl_account_df[target_column].values
    actual_loss = ssl_account_df[loss_column].values

    # Directional accuracy (predicted binary vs actual binary)
    directional_accuracy = np.mean(y_pred == y_actual)

    # Rank correlation between predicted probability and actual loss
    rank_corr, rank_pvalue = stats.spearmanr(y_proba, actual_loss)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix as cm_func
    conf_matrix = cm_func(y_actual, y_pred)

    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        "account_id": ssl_account_df["account_id"].values,
        "actual_loss": actual_loss,
        "actual_high_loss": y_actual,
        "predicted_high_risk": y_pred,
        "predicted_probability": y_proba,
    })

    result = {
        "directional_accuracy": directional_accuracy,
        "rank_correlation": rank_corr,
        "rank_pvalue": rank_pvalue,
        "predicted_high_pct": np.mean(y_pred) * 100,
        "actual_high_pct": np.mean(y_actual) * 100,
        "confusion_at_directional": conf_matrix,
        "n_accounts": len(ssl_account_df),
        "n_features_available": len(available_features),
        "n_features_missing": len(missing_features),
        "missing_features": missing_features,
        "predictions_df": predictions_df,
    }

    logger.info(
        "Directional validation: accuracy=%.3f, rank_corr=%.3f (p=%.2e), "
        "%d/%d features available",
        directional_accuracy,
        rank_corr,
        rank_pvalue,
        len(available_features),
        len(thelook_features),
    )

    return result


def build_validation_summary(
    pattern_comparison: pd.DataFrame,
    directional_result: Dict[str, Any],
) -> pd.DataFrame:
    """
    Summarize pattern alignment and directional validation metrics.

    Args:
        pattern_comparison: Output from validate_feature_patterns().
        directional_result: Output from validate_directional_predictions().

    Returns:
        Summary DataFrame with key validation metrics.
    """
    n_features = len(pattern_comparison)
    n_agree = pattern_comparison["agreement"].sum()
    n_both_pass = pattern_comparison["both_pass"].sum()
    n_both_fail = pattern_comparison["both_fail"].sum()

    rows = [
        {"metric": "pattern_features_compared", "value": n_features},
        {"metric": "pattern_agreement_count", "value": n_agree},
        {
            "metric": "pattern_agreement_pct",
            "value": round(n_agree / n_features * 100, 1) if n_features > 0 else 0,
        },
        {"metric": "pattern_both_pass", "value": n_both_pass},
        {"metric": "pattern_both_fail", "value": n_both_fail},
        {
            "metric": "directional_accuracy",
            "value": round(directional_result["directional_accuracy"], 4),
        },
        {
            "metric": "directional_rank_correlation",
            "value": round(directional_result["rank_correlation"], 4),
        },
        {
            "metric": "directional_rank_pvalue",
            "value": directional_result["rank_pvalue"],
        },
        {"metric": "ssl_accounts_evaluated", "value": directional_result["n_accounts"]},
        {
            "metric": "features_available",
            "value": directional_result["n_features_available"],
        },
        {
            "metric": "features_missing",
            "value": directional_result["n_features_missing"],
        },
        {
            "metric": "predicted_high_risk_pct",
            "value": round(directional_result["predicted_high_pct"], 1),
        },
        {
            "metric": "actual_high_loss_pct",
            "value": round(directional_result["actual_high_pct"], 1),
        },
    ]

    summary_df = pd.DataFrame(rows)

    logger.info(
        "Validation summary: pattern agreement=%.1f%%, "
        "directional accuracy=%.3f, rank correlation=%.3f",
        n_agree / n_features * 100 if n_features > 0 else 0,
        directional_result["directional_accuracy"],
        directional_result["rank_correlation"],
    )

    return summary_df
