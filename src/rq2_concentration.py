"""
RQ2 / US09 - Improved Customer-level concentration metrics (Pareto, Lorenz, Gini).
Enhanced with business logic, outlier detection, and robust Gini calculations.
"""

from __future__ import annotations

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
