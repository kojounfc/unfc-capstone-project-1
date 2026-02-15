"""
tests/test_model_ready_views.py

Unit tests for:
  src/model_ready_views.py

Focus:
- Deterministic construction of RQ2/RQ3/RQ4 base datasets
- Schema guarantees (required columns)
- Leakage drops for RQ3
- Correct writing to a provided processed_dir (tmp_path)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.model_ready_views as mrv


def _make_item_level_df() -> pd.DataFrame:
    """
    Create a tiny denormalized item-level dataset that mimics your pipeline columns.

    Includes:
    - order_item_id, user_id
    - sale_price, cost (for item_margin)
    - item_returned_at (for deriving is_returned_item)
    - some customer attrs for optional RQ2 merge
    """
    return pd.DataFrame(
        {
            "order_item_id": [1, 2, 3, 4, 5, 6],
            "order_id": [10, 10, 11, 12, 12, 13],
            "user_id": [100, 100, 101, 102, 102, 103],
            "sale_price": [100.0, 50.0, 120.0, 80.0, 40.0, 60.0],
            "cost": [60.0, 30.0, 70.0, 50.0, 15.0, 10.0],
            # Returned items: 1, 3, 5 (three returned)
            "item_returned_at": [
                "2025-01-01",
                None,
                "2025-01-02",
                None,
                "2025-01-03",
                None,
            ],
            # leakage-ish columns (to confirm drops in RQ3 base)
            "item_status": ["Returned", "Shipped", "Returned", "Delivered", "Returned", "Shipped"],
            "order_status": ["Returned", "Complete", "Returned", "Complete", "Returned", "Complete"],
            # optional customer attrs
            "age": [30, 30, 40, 35, 35, 50],
            "user_gender": ["F", "F", "M", "F", "F", "M"],
            "country": ["CA", "CA", "CA", "CA", "CA", "CA"],
            "traffic_source": ["Search", "Search", "Email", "Ads", "Ads", "Email"],
        }
    )


def test_ensure_return_flag_derives_from_item_returned_at() -> None:
    df = _make_item_level_df()
    out = mrv.ensure_return_flag(df)

    assert "is_returned_item" in out.columns
    assert out["is_returned_item"].dtype.kind in ("i", "u")  # int/uint

    # expected: rows 0,2,4 are returned -> 1; others -> 0
    expected = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    np.testing.assert_array_equal(out["is_returned_item"].values, expected)


def test_ensure_item_margin_creates_column() -> None:
    df = _make_item_level_df()
    out = mrv.ensure_item_margin(df)

    assert "item_margin" in out.columns
    # sale_price - cost
    np.testing.assert_allclose(out["item_margin"].values, (df["sale_price"] - df["cost"]).values)


def test_ensure_profit_erosion_creates_column_and_zero_for_non_returns() -> None:
    df = _make_item_level_df()
    out = mrv.ensure_profit_erosion(df)

    assert "profit_erosion" in out.columns

    # For returned items profit_erosion == item_margin; else 0
    out = mrv.ensure_item_margin(out)
    expected = np.where(out["is_returned_item"].values == 1, out["item_margin"].values, 0.0)
    np.testing.assert_allclose(out["profit_erosion"].values, expected)


def test_build_rq2_customer_segmentation_base_grain_and_columns() -> None:
    df = _make_item_level_df()
    rq2 = mrv.build_rq2_customer_segmentation_base(df)

    # Grain: one row per user_id
    assert rq2["user_id"].is_unique
    assert set(["user_id", "total_items", "returned_items", "return_rate", "total_profit_erosion", "avg_profit_erosion_per_return"]).issubset(
        rq2.columns
    )

    # There are 4 users in the synthetic dataset
    assert len(rq2) == 4

    # Quick sanity: user 100 has 2 items, 1 returned
    row100 = rq2.loc[rq2["user_id"] == 100].iloc[0]
    assert row100["total_items"] == 2
    assert row100["returned_items"] == 1
    assert row100["return_rate"] == pytest.approx(0.5)


def test_build_rq3_item_return_classification_base_drops_leakage_cols() -> None:
    df = _make_item_level_df()
    rq3 = mrv.build_rq3_item_return_classification_base(df)

    assert set(["order_item_id", "user_id", "is_returned_item"]).issubset(rq3.columns)

    # leakage cols should be dropped if present
    assert "item_returned_at" not in rq3.columns
    assert "order_returned_at" not in rq3.columns  # not present originally, but ensure not added
    assert "item_status" not in rq3.columns
    assert "order_status" not in rq3.columns

    # should still have same number of rows (item grain)
    assert len(rq3) == len(df)


def test_build_rq4_returned_item_profit_erosion_base_filters_returned_only() -> None:
    df = _make_item_level_df()
    rq4 = mrv.build_rq4_returned_item_profit_erosion_base(df)

    assert set(["order_item_id", "profit_erosion", "is_returned_item"]).issubset(rq4.columns)

    # Only returned items
    assert (rq4["is_returned_item"] == 1).all()
    assert len(rq4) == 3  # returned: 3 rows in synthetic data

    # profit_erosion should be non-null
    assert rq4["profit_erosion"].notna().all()


def test_write_us07_task_60_outputs_writes_three_parquets(tmp_path: Path) -> None:
    df = _make_item_level_df()

    outputs = mrv.write_us07_task_60_outputs(df, processed_dir=tmp_path)

    assert set(outputs.keys()) == {
        "rq2_customer_segmentation_base",
        "rq3_item_return_classification_base",
        "rq4_returned_item_profit_erosion_base",
    }

    # Files exist
    for p in outputs.values():
        assert Path(p).exists()
        assert str(p).endswith(".parquet")

    # Read-back sanity
    rq2 = pd.read_parquet(outputs["rq2_customer_segmentation_base"])
    rq3 = pd.read_parquet(outputs["rq3_item_return_classification_base"])
    rq4 = pd.read_parquet(outputs["rq4_returned_item_profit_erosion_base"])

    assert len(rq2) == 4
    assert len(rq3) == 6
    assert len(rq4) == 3
