"""
src/descriptive_transformations.py

US07 Task #57: Product-level profit erosion metrics for RQ1.

Key rule:
- calculate_profit_erosion() is designed to receive *returned items only*
  (is_returned_item == 1). :contentReference[oaicite:3]{index=3}

This module:
- uses analytics.calculate_return_rates_by_group() for return-rate context
  (item_rows, returned_items, return_rate). :contentReference[oaicite:4]{index=4}
- filters returned items
- calls feature_engineering.calculate_profit_erosion()
- aggregates profit erosion by category/brand/department
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.analytics import calculate_return_rates_by_group
from src.feature_engineering import calculate_profit_erosion


def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _aggregate_profit_erosion(returned_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Aggregate profit erosion metrics on returned items only.
    Assumes calculate_profit_erosion() already added:
      - margin_reversal
      - process_cost
      - profit_erosion
    """
    _require_columns(
        returned_df,
        required=[group_col, "order_id", "profit_erosion", "margin_reversal", "process_cost"],
        context=f"_aggregate_profit_erosion:{group_col}",
    )

    out = (
        returned_df.groupby(group_col)
        .agg(
            returned_item_rows=("order_id", "size"),
            total_profit_erosion=("profit_erosion", "sum"),
            mean_profit_erosion_per_return=("profit_erosion", "mean"),
            median_profit_erosion_per_return=("profit_erosion", "median"),
            var_profit_erosion_per_return=("profit_erosion", "var"),
            total_margin_reversal=("margin_reversal", "sum"),
            total_process_cost=("process_cost", "sum"),
        )
        .reset_index()
        .sort_values("total_profit_erosion", ascending=False)
    )
    return out


def build_product_profit_erosion_metrics(
    df: pd.DataFrame,
    min_rows: int = 200,
    use_category_tiers: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    US07 #57: Produce tables of product-level erosion metrics to support RQ1.

    Returns dict:
      - by_category
      - by_brand
      - by_department

    Each output includes:
      - return-rate context: item_rows, returned_items, return_rate
      - erosion metrics (returned items only): total_profit_erosion, etc.
    """
    _require_columns(
        df,
        required=[
            "order_id",
            "is_returned_item",
            "item_margin",
            "category",
            "brand",
            "department",
        ],
        context="build_product_profit_erosion_metrics",
    )

    def _build(group_col: str) -> pd.DataFrame:
        # A) Return-rate context on ALL items (existing function)
        rr = calculate_return_rates_by_group(df, [group_col], min_rows=min_rows).reset_index()

        # B) Returned items only (required for profit erosion function)
        returned = df[df["is_returned_item"] == 1].copy()

        if returned.empty:
            # Return rate context still valid; erosion columns empty
            for col in [
                "returned_item_rows",
                "total_profit_erosion",
                "mean_profit_erosion_per_return",
                "median_profit_erosion_per_return",
                "var_profit_erosion_per_return",
                "total_margin_reversal",
                "total_process_cost",
            ]:
                rr[col] = pd.NA
            return rr

        # C) Add profit erosion columns (existing function)
        returned = calculate_profit_erosion(returned, use_category_tiers=use_category_tiers)

        # D) Aggregate erosion
        eros = _aggregate_profit_erosion(returned, group_col)

        # E) Merge (keep all groups passing min_rows)
        out = rr.merge(eros, on=group_col, how="left")

        # Sort by biggest economic impact (RQ1)
        out = out.sort_values("total_profit_erosion", ascending=False, na_position="last")
        return out

    return {
        "by_category": _build("category"),
        "by_brand": _build("brand"),
        "by_department": _build("department"),
    }