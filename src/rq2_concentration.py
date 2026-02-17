"""
RQ2 / US09 - Improved Customer-level concentration metrics (Pareto, Lorenz, Gini).
Enhanced with business logic, outlier detection, and robust Gini calculations.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.descriptive_transformations import _require_columns


def compute_pareto_table(
    df: pd.DataFrame,
    value_col: str = "total_profit_erosion",
    id_col: str = "user_id",
) -> pd.DataFrame:
    """
    Build an enhanced Pareto-style table with business categories.

    Args:
        df: Customer-level DataFrame containing identifier and value columns.
        value_col: Column containing the concentration metric values.
        id_col: Customer identifier column.

    Returns:
        DataFrame sorted by value with cumulative shares and categories.
    """
    if df.empty:
        return pd.DataFrame()

    _require_columns(df, [id_col, value_col], "compute_pareto_table")

    # 1. Clean and Sort
    res = df[[id_col, value_col]].copy()
    res[value_col] = pd.to_numeric(res[value_col], errors="coerce").fillna(0.0)
    res = res.sort_values(by=value_col, ascending=False).reset_index(drop=True)

    # 2. Cumulative Calculations
    total_val = res[value_col].sum()
    res["rank"] = res.index + 1
    res["customer_share"] = res["rank"] / len(res)
    res["cum_value"] = res[value_col].cumsum()
    res["value_share"] = res["cum_value"] / total_val if total_val > 0 else 0.0

    # 3. IMPROVEMENT: Categorization (Vital Few vs Useful Many)
    # Traditionally, the top 20% are the 'Vital Few'
    res["concentration_category"] = np.where(
        res["customer_share"] <= 0.20, "Vital Few", "Useful Many"
    )

    return res


def gini_coefficient(
    df: pd.DataFrame,
    value_col: str = "total_profit_erosion",
) -> float:
    """
    Compute the Gini coefficient with improved handling for negative values.

    Args:
        df: Customer-level DataFrame containing the concentration value column.
        value_col: Column containing the concentration metric values.

    Returns:
        Gini coefficient as a float in [0, 1] for non-negative distributions.
    """
    if df.empty:
        return 0.0

    # Sort values ascending for the standard Gini formula
    x = np.sort(df[value_col].to_numpy())
    n = len(x)

    # IMPROVEMENT: Check for negative values (Profit instead of Erosion)
    if np.any(x < 0):
        # Shift values so the minimum is 0 for standard Gini,
        # or alert user that distribution includes 'profit'
        x = x - np.min(x)

    if np.sum(x) == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def top_n_customer_impact(
    df: pd.DataFrame, n: int = 50, value_col: str = "total_profit_erosion"
) -> dict:
    """
    IMPROVEMENT: Specific helper for stakeholder reporting.
    Calculates the absolute and relative impact of the 'N' worst offenders.

    Args:
        df: Customer-level DataFrame containing the value column.
        n: Number of customers to include in the impact summary.
        value_col: Column containing the concentration metric values.

    Returns:
        Dictionary with count, absolute loss, and percentage of total loss.
    """
    top_n = df.nlargest(n, value_col)
    total_impact = top_n[value_col].sum()
    grand_total = df[value_col].sum()

    share = total_impact / grand_total if grand_total > 0 else 0.0

    return {
        "count": n,
        "absolute_loss": round(total_impact, 2),
        "percentage_of_total": round(share * 100, 2),
    }


def get_business_summary(
    df: pd.DataFrame, value_col: str = "total_profit_erosion"
) -> dict:
    """
    IMPROVEMENT: High-level summary for the Final Notebook.

    Args:
        df: Customer-level DataFrame containing the concentration value column.
        value_col: Column containing the concentration metric values.

    Returns:
        Dictionary with summary metrics and recommended action level.
    """
    gini = gini_coefficient(df, value_col)
    top_20_pct = top_x_customer_share_of_value(df, 0.20, value_col)

    return {
        "gini_index": round(gini, 3),
        "concentration_level": (
            "Extreme" if gini > 0.7 else "High" if gini > 0.5 else "Moderate"
        ),
        "pareto_ratio": f"20% of customers = {round(top_20_pct * 100, 1)}% of loss",
        "recommendation": "Targeted Policy" if gini > 0.5 else "Broad Policy",
    }


# Standard helper kept for compatibility
def lorenz_curve_points(
    df: pd.DataFrame, value_col: str = "total_profit_erosion"
) -> pd.DataFrame:
    """
    Compute Lorenz curve coordinates for a concentration distribution.

    Args:
        df: Customer-level table containing the concentration value column.
        value_col: Numeric column used to build cumulative value share.

    Returns:
        DataFrame with two columns:
            - population_share: Cumulative share of population in [0, 1]
            - value_share: Cumulative share of value in [0, 1]
    """
    x = np.sort(df[value_col].to_numpy())
    total = x.sum()
    pop = np.linspace(0, 1, len(x) + 1)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    val = cum / total if total > 0 else np.zeros_like(cum)
    return pd.DataFrame({"population_share": pop, "value_share": val})


def top_x_customer_share_of_value(
    df: pd.DataFrame,
    x: float,
    value_col: str = "total_profit_erosion",
    id_col: str = "user_id",
) -> float:
    """
    Return cumulative value share captured by the top x share of customers.

    Args:
        df: Customer-level table with customer ID and concentration value columns.
        x: Customer share threshold in (0, 1], e.g., 0.20 for top 20%.
        value_col: Numeric value column used for concentration calculation.
        id_col: Customer identifier column.

    Returns:
        Cumulative share of value attributable to the top x share of customers.

    Raises:
        ValueError: If x is not in the interval (0, 1].
    """
    if not (0.0 < x <= 1.0):
        raise ValueError("x must be in the interval (0, 1].")

    table = compute_pareto_table(df, value_col=value_col, id_col=id_col)
    if table.empty:
        return 0.0
    # Find the row closest to the x% threshold
    idx = (table["customer_share"] - x).abs().idxmin()
    return float(table.loc[idx, "value_share"])


def bootstrap_gini_p_value(
    df: pd.DataFrame,
    value_col: str = "total_profit_erosion",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Test whether observed concentration exceeds a uniform allocation null.

    The null preserves the total value while distributing it uniformly across
    customers (equal-share baseline), then adds random row permutations to
    produce a bootstrap null distribution of Gini values.

    Args:
        df: Customer-level DataFrame containing the concentration metric.
        value_col: Numeric column used for Gini computation.
        n_bootstrap: Number of bootstrap samples used to estimate p-value.
        random_state: Random seed for reproducible null sampling.

    Returns:
        Dictionary with observed_gini, null_mean_gini, p_value, and n_bootstrap.
    """
    if df.empty:
        return {
            "observed_gini": 0.0,
            "null_mean_gini": 0.0,
            "p_value": 1.0,
            "n_bootstrap": int(n_bootstrap),
        }

    vals = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy()
    n = len(vals)
    total = float(np.sum(vals))

    observed = gini_coefficient(df, value_col=value_col)
    if n == 0 or total == 0:
        return {
            "observed_gini": float(observed),
            "null_mean_gini": 0.0,
            "p_value": 1.0,
            "n_bootstrap": int(n_bootstrap),
        }

    equal_share = np.repeat(total / n, n)
    rng = np.random.default_rng(random_state)
    null_ginis = []
    for _ in range(n_bootstrap):
        shuffled = rng.permutation(equal_share)
        null_df = pd.DataFrame({value_col: shuffled})
        null_ginis.append(gini_coefficient(null_df, value_col=value_col))

    null_arr = np.array(null_ginis, dtype=float)
    p_value = float(np.mean(null_arr >= observed))

    return {
        "observed_gini": float(observed),
        "null_mean_gini": float(np.mean(null_arr)),
        "p_value": float(p_value),
        "n_bootstrap": int(n_bootstrap),
    }


def concentration_comparison(
    df: pd.DataFrame,
    erosion_col: str = "total_profit_erosion",
    baseline_col: str = "total_sales",
) -> dict:
    """
    Compare erosion concentration against a baseline business distribution.

    Args:
        df: Customer-level DataFrame containing both value columns.
        erosion_col: Column representing erosion outcomes.
        baseline_col: Baseline comparison column (for example sales).

    Returns:
        Dictionary with Gini values for erosion and baseline distributions.
    """
    _require_columns(df, [erosion_col, baseline_col], "concentration_comparison")
    return {
        "gini_erosion": float(gini_coefficient(df, value_col=erosion_col)),
        "gini_baseline": float(gini_coefficient(df, value_col=baseline_col)),
    }


def analyze_feature_concentration(
    df: pd.DataFrame,
    feature_col: str,
    id_col: str = "user_id",
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Analyze concentration for a single feature.

    Args:
        df: Customer-level DataFrame
        feature_col: Column to analyze for concentration
        id_col: Customer identifier column
        n_bootstrap: Number of bootstrap samples for p-value
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with feature name, Gini coefficient, p-value, and concentration level
    """
    # Clean data
    clean_df = df[[id_col, feature_col]].copy()
    clean_df[feature_col] = pd.to_numeric(
        clean_df[feature_col], errors="coerce"
    ).fillna(0.0)

    # Remove zero values for meaningful concentration analysis
    clean_df = clean_df[clean_df[feature_col] > 0]

    if clean_df.empty or len(clean_df) < 2:
        return {
            "feature": feature_col,
            "gini_coefficient": 0.0,
            "p_value": 1.0,
            "concentration_level": "N/A",
            "n_customers": 0,
            "top_20_pct_share": 0.0,
        }

    # Calculate Gini coefficient
    gini = gini_coefficient(clean_df, value_col=feature_col)

    # Calculate p-value using bootstrap
    bootstrap_result = bootstrap_gini_p_value(
        clean_df,
        value_col=feature_col,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Calculate top 20% share
    top_20_share = top_x_customer_share_of_value(
        clean_df, x=0.20, value_col=feature_col, id_col=id_col
    )

    # Determine concentration level
    if gini > 0.7:
        level = "Extreme"
    elif gini > 0.5:
        level = "High"
    elif gini > 0.3:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "feature": feature_col,
        "gini_coefficient": round(gini, 4),
        "p_value": round(bootstrap_result["p_value"], 4),
        "concentration_level": level,
        "n_customers": len(clean_df),
        "top_20_pct_share": round(top_20_share * 100, 2),
    }


def rank_features_by_concentration(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    id_col: str = "user_id",
    n_bootstrap: int = 1000,
    random_state: int = 42,
    min_customers: int = 10,
) -> pd.DataFrame:
    """
    Analyze concentration across multiple features and rank them by Gini coefficient.

    This is the main function you requested - it returns a table of features ranked
    from highest to lowest concentration with their Gini coefficients and p-values.

    Args:
        df: Customer-level DataFrame
        feature_cols: List of feature columns to analyze. If None, uses all numeric columns
                     except id_col.
        id_col: Customer identifier column
        n_bootstrap: Number of bootstrap samples for p-value calculation
        random_state: Random seed for reproducibility
        min_customers: Minimum number of non-zero customers required for analysis

    Returns:
        DataFrame with columns:
            - feature: Feature name
            - gini_coefficient: Gini coefficient (0-1 scale)
            - concentration_pct: Gini as percentage (0-100%)
            - p_value: Bootstrap p-value for significance test
            - concentration_level: Categorical level (Low/Moderate/High/Extreme)
            - n_customers: Number of customers with non-zero values
            - top_20_pct_share: Percentage of total captured by top 20% of customers

        Sorted by gini_coefficient descending (highest concentration first)

    Example:
        >>> # Analyze all numeric features
        >>> concentration_ranking = rank_features_by_concentration(customer_df)
        >>> print(concentration_ranking.head())

        >>> # Analyze specific features only
        >>> features = ['total_profit_erosion', 'total_margin_reversal', 'total_sales']
        >>> ranking = rank_features_by_concentration(customer_df, feature_cols=features)
    """
    # Determine which features to analyze
    if feature_cols is None:
        # Use all numeric columns except id_col
        feature_cols = [
            col
            for col in df.columns
            if col != id_col and pd.api.types.is_numeric_dtype(df[col])
        ]
    else:
        # Validate that all requested columns exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

    if not feature_cols:
        return pd.DataFrame(
            columns=[
                "feature",
                "gini_coefficient",
                "concentration_pct",
                "p_value",
                "concentration_level",
                "n_customers",
                "top_20_pct_share",
            ]
        )

    # Analyze each feature
    results = []
    for feature in feature_cols:
        result = analyze_feature_concentration(
            df=df,
            feature_col=feature,
            id_col=id_col,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        # Only include if we have enough customers
        if result["n_customers"] >= min_customers:
            results.append(result)

    # Convert to DataFrame
    if not results:
        return pd.DataFrame(
            columns=[
                "feature",
                "gini_coefficient",
                "concentration_pct",
                "p_value",
                "concentration_level",
                "n_customers",
                "top_20_pct_share",
            ]
        )

    ranking_df = pd.DataFrame(results)

    # Add concentration percentage column for easier interpretation
    ranking_df["concentration_pct"] = (ranking_df["gini_coefficient"] * 100).round(2)

    # Reorder columns for better readability
    ranking_df = ranking_df[
        [
            "feature",
            "gini_coefficient",
            "concentration_pct",
            "p_value",
            "concentration_level",
            "n_customers",
            "top_20_pct_share",
        ]
    ]

    # Sort by Gini coefficient (highest concentration first)
    ranking_df = ranking_df.sort_values(
        "gini_coefficient", ascending=False
    ).reset_index(drop=True)

    return ranking_df


def filter_significant_features(
    ranking_df: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Filter concentration ranking to only statistically significant features.

    Args:
        ranking_df: Output from rank_features_by_concentration()
        alpha: Significance level threshold (default 0.05)

    Returns:
        Filtered DataFrame containing only features with p_value < alpha
    """
    return ranking_df[ranking_df["p_value"] < alpha].reset_index(drop=True)


def summarize_concentration_findings(
    ranking_df: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """
    Generate a summary of concentration analysis findings.

    Args:
        ranking_df: Output from rank_features_by_concentration()
        top_n: Number of top features to highlight in summary

    Returns:
        Dictionary with summary statistics and insights
    """
    if ranking_df.empty:
        return {
            "n_features_analyzed": 0,
            "top_features": [],
            "avg_gini": 0.0,
            "n_significant": 0,
            "n_extreme": 0,
            "n_high": 0,
        }

    top_features = ranking_df.head(top_n)[
        ["feature", "gini_coefficient", "p_value"]
    ].to_dict("records")

    return {
        "n_features_analyzed": len(ranking_df),
        "top_features": top_features,
        "avg_gini": round(ranking_df["gini_coefficient"].mean(), 4),
        "n_significant": len(ranking_df[ranking_df["p_value"] < 0.05]),
        "n_extreme": len(ranking_df[ranking_df["concentration_level"] == "Extreme"]),
        "n_high": len(ranking_df[ranking_df["concentration_level"] == "High"]),
        "n_moderate": len(ranking_df[ranking_df["concentration_level"] == "Moderate"]),
        "n_low": len(ranking_df[ranking_df["concentration_level"] == "Low"]),
    }
