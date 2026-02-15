"""
transformations/model_ready_views.py

US07 #60: Prepare analysis-ready datasets for downstream modeling (RQ2, RQ3, RQ4).

Key rules:
- Inputs are item-level denormalized data (from US06 output).
- This module does NOT do feature selection. Each RQ will do feature importance / pruning.
- This module DOES:
    (1) ensure required base columns exist (return flag, item margin, profit erosion),
    (2) build stable modeling datasets with consistent grain,
    (3) remove obvious leakage columns,
    (4) write ONLY to data/processed/ (parquet).

Outputs (parquet):
- data/processed/rq2/rq2_customer_segmentation_base.parquet
- data/processed/rq3/rq3_item_return_classification_base.parquet
- data/processed/rq4/rq4_returned_item_profit_erosion_base.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_MIN_ROWS_GROUP = 200


@dataclass(frozen=True)
class OutputPaths:
    """Centralized paths to enforce 'write only to data/processed'."""
    base_processed_dir: Path

    @property
    def rq2_dir(self) -> Path:
        return self.base_processed_dir 

    @property
    def rq3_dir(self) -> Path:
        return self.base_processed_dir

    @property
    def rq4_dir(self) -> Path:
        return self.base_processed_dir

    @property
    def rq2_customer_base(self) -> Path:
        return self.rq2_dir / "rq2_customer_segmentation_base.parquet"

    @property
    def rq3_item_classification_base(self) -> Path:
        return self.rq3_dir / "rq3_item_return_classification_base.parquet"

    @property
    def rq4_returned_regression_base(self) -> Path:
        return self.rq4_dir / "rq4_returned_item_profit_erosion_base.parquet"


def default_processed_dir() -> Path:
    """
    Resolve default processed directory.

    Repo convention:
      data/processed/
    """
    return Path("data") / "processed"


# -----------------------------
# Validation helpers
# -----------------------------

def _require_columns(df: pd.DataFrame, required: List[str], context: str) -> None:
    """Raise a clear error if df is missing required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _ensure_dir(p: Path) -> None:
    """Create parent directory if needed (deterministic)."""
    p.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Base derivations (no feature selection)
# -----------------------------

def ensure_return_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `is_returned_item` exists.

    Logic (robust to different schemas):
    - If is_returned_item already exists, keep as-is.
    - Else derive it from item_returned_at OR order_returned_at (if either exists).
      Returned = not-null timestamp.
    """
    out = df.copy()

    if "is_returned_item" in out.columns:
        # Normalize to 0/1 ints
        out["is_returned_item"] = out["is_returned_item"].fillna(0).astype(int)
        return out

    candidates = []
    if "item_returned_at" in out.columns:
        candidates.append(out["item_returned_at"].notna())
    if "order_returned_at" in out.columns:
        candidates.append(out["order_returned_at"].notna())

    if not candidates:
        raise ValueError(
            "[ensure_return_flag] No return indicator found. "
            "Expected `is_returned_item` OR one of [`item_returned_at`, `order_returned_at`]."
        )

    out["is_returned_item"] = np.any(np.vstack([c.values for c in candidates]), axis=0).astype(int)
    return out


def ensure_item_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `item_margin` exists.

    Definition:
      item_margin = sale_price - cost

    Notes:
    - This is not “feature engineering” in the modeling sense.
      It’s a core accounting field required by downstream erosion calculations.
    """
    out = df.copy()

    if "item_margin" in out.columns:
        return out

    _require_columns(out, ["sale_price", "cost"], context="ensure_item_margin")
    out["item_margin"] = out["sale_price"] - out["cost"]
    return out


def ensure_profit_erosion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure `profit_erosion` exists (item-level target for RQ4).

    Conservative baseline (deterministic):
      profit_erosion = item_margin (returned items), else 0

    Why:
    - Your pipeline may compute a richer profit_erosion elsewhere.
    - But US07#60 must be able to build the modeling base even if upstream didn't persist it.
    - If you already have profit_erosion, we do NOT overwrite it.
    """
    out = df.copy()

    if "profit_erosion" in out.columns:
        return out

    out = ensure_return_flag(out)
    out = ensure_item_margin(out)

    out["profit_erosion"] = np.where(out["is_returned_item"] == 1, out["item_margin"], 0.0)
    return out


def drop_leakage_columns_item_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop obvious leakage columns for return prediction (RQ3).

    Leakage = columns that directly encode the outcome or occur after the outcome.
    We drop them if present.
    """
    out = df.copy()

    leakage_cols = [
        "item_returned_at",
        "order_returned_at",
        "item_status",        # can sometimes encode returned/refunded
        "order_status",       # can sometimes encode returned/refunded
    ]

    existing = [c for c in leakage_cols if c in out.columns]
    if existing:
        out = out.drop(columns=existing)

    return out


# -----------------------------
# US07 #60 outputs
# -----------------------------

def build_rq2_customer_segmentation_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RQ2 base dataset (customer grain).

    Grain:
      - 1 row per user_id

    Goal:
      - Provide stable segmentation base with customer attributes + return/erosion summaries.
      - RQ2 will later choose features (feature importance / selection happens in RQ2 code).

    Output columns (minimum):
      user_id
      total_items
      returned_items
      return_rate
      total_profit_erosion
      avg_profit_erosion_per_return
      (plus customer demographic columns when present)
    """
    out = df.copy()
    out = ensure_return_flag(out)
    out = ensure_profit_erosion(out)

    _require_columns(out, ["user_id", "order_item_id"], context="build_rq2_customer_segmentation_base")

    # Optional customer attributes if available (keep them; do not force)
    optional_customer_cols = [
        "age",
        "user_gender",
        "country",
        "state",
        "city",
        "traffic_source",
    ]
    keep_customer_cols = [c for c in optional_customer_cols if c in out.columns]

    grp = out.groupby("user_id", dropna=False)

    base = grp.agg(
        total_items=("order_item_id", "count"),
        returned_items=("is_returned_item", "sum"),
        total_profit_erosion=("profit_erosion", "sum"),
    ).reset_index()

    base["return_rate"] = np.where(base["total_items"] > 0, base["returned_items"] / base["total_items"], 0.0)

    # avg erosion per returned item (avoid divide-by-zero)
    base["avg_profit_erosion_per_return"] = np.where(
        base["returned_items"] > 0,
        base["total_profit_erosion"] / base["returned_items"],
        0.0,
    )

    if keep_customer_cols:
        # attach 1st observed demographic values per user (stable and deterministic)
        demo = grp[keep_customer_cols].first().reset_index()
        base = base.merge(demo, on="user_id", how="left")

    # Basic sanity
    base = base.sort_values(["total_profit_erosion", "returned_items"], ascending=False)

    return base


def build_rq3_item_return_classification_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RQ3 base dataset (item grain) for return prediction.

    Grain:
      - 1 row per order_item_id

    Target:
      - is_returned_item (0/1)

    Rules:
      - Keep most engineered features (RQ3 will prune later).
      - Drop leakage columns (returned timestamps/status).

    Required:
      order_item_id, user_id, is_returned_item
    """
    out = df.copy()
    out = ensure_return_flag(out)

    _require_columns(out, ["order_item_id", "user_id", "is_returned_item"], context="build_rq3_item_return_classification_base")

    out = drop_leakage_columns_item_level(out)

    # Defensive: ensure target is int
    out["is_returned_item"] = out["is_returned_item"].fillna(0).astype(int)

    return out


def build_rq4_returned_item_profit_erosion_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RQ4 base dataset (returned items only) for profit erosion regression.

    Grain:
      - 1 row per returned order_item_id

    Target:
      - profit_erosion (numeric)

    Notes:
    - We filter returned items only so the target has signal.
    - RQ4 will later perform feature selection / importance.
    """
    out = df.copy()
    out = ensure_return_flag(out)
    out = ensure_profit_erosion(out)

    _require_columns(out, ["order_item_id", "profit_erosion", "is_returned_item"], context="build_rq4_returned_item_profit_erosion_base")

    out = out[out["is_returned_item"] == 1].copy()

    # Remove return timestamps (they can leak post-event handling)
    for c in ["item_returned_at", "order_returned_at"]:
        if c in out.columns:
            out = out.drop(columns=[c])

    # Basic cleanup
    out = out.dropna(subset=["profit_erosion"])

    return out


def write_us07_task_60_outputs(
    df: pd.DataFrame,
    processed_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Build and write all US07 #60 datasets to data/processed.

    Returns:
      dict of logical output names -> file paths
    """
    base_dir = processed_dir if processed_dir is not None else default_processed_dir()
    paths = OutputPaths(base_processed_dir=base_dir)

    # Build datasets
    rq2 = build_rq2_customer_segmentation_base(df)
    rq3 = build_rq3_item_return_classification_base(df)
    rq4 = build_rq4_returned_item_profit_erosion_base(df)

    # Write (parquet)
    _ensure_dir(paths.rq2_customer_base)
    _ensure_dir(paths.rq3_item_classification_base)
    _ensure_dir(paths.rq4_returned_regression_base)

    rq2.to_parquet(paths.rq2_customer_base, index=False)
    rq3.to_parquet(paths.rq3_item_classification_base, index=False)
    rq4.to_parquet(paths.rq4_returned_regression_base, index=False)

    return {
        "rq2_customer_segmentation_base": paths.rq2_customer_base,
        "rq3_item_return_classification_base": paths.rq3_item_classification_base,
        "rq4_returned_item_profit_erosion_base": paths.rq4_returned_regression_base,
    }
