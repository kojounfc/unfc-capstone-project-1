import pandas as pd
import pytest

import src.data_processing as dp
from src.feature_engineering import engineer_return_features, calculate_margins
from src.descriptive_transformations import (
    build_product_profit_erosion_metrics,
    build_product_return_behavior_metrics,
    build_customer_profit_erosion_summaries,
    build_product_modeling_dataset,
    build_customer_modeling_dataset,
)

@pytest.fixture
def analysis_df():
    """
    Synthetic 'analysis dataset' at order-item grain.

    Goal: Make tests CI-safe by removing dependency on data/raw/*.csv.
    Also make fixture resilient to schema expectations inside engineer_return_features()
    by including both item-level and order-level status + common timestamp variants.

    This fixture is intentionally "wide" to avoid KeyError when feature code evolves.
    """
    n = 240  # bigger to ensure min_rows=50 works across multiple group levels

    # create repeated order ids (multiple items per order)
    order_ids = [f"o{i//3}" for i in range(n)]  # ~80 orders
    user_ids = [f"u{i%25}" for i in range(n)]  # 25 users
    product_ids = [f"p{i%40}" for i in range(n)]  # 40 products

    # item-level return pattern: 1 in 10 returned
    item_status = ["returned" if i % 10 == 0 else "complete" for i in range(n)]
    returned_at = [pd.Timestamp("2025-01-10") if s == "returned" else pd.NaT for s in item_status]

    # order-level return pattern: mark an order returned if ANY item in that order is returned
    # build a mapping from order_id -> order_status
    order_has_return = {}
    for oid, s in zip(order_ids, item_status):
        order_has_return[oid] = order_has_return.get(oid, False) or (s == "returned")
    order_status = ["returned" if order_has_return[oid] else "complete" for oid in order_ids]

    # timestamps (include common variants used across pipelines)
    created_at = pd.Timestamp("2025-01-01")
    shipped_at = pd.Timestamp("2025-01-03")
    delivered_at = pd.Timestamp("2025-01-06")

    df = pd.DataFrame({
        # identifiers
        "order_id": order_ids,
        "user_id": user_ids,
        "product_id": product_ids,

        # product attributes (grouping dims)
        "category": ["cat_a" if i < n * 0.65 else "cat_b" for i in range(n)],
        "brand": ["brand_x" if i % 2 == 0 else "brand_y" for i in range(n)],
        "department": ["dept_1" if i % 3 else "dept_2" for i in range(n)],

        # pricing (needed by calculate_margins)
        "sale_price": [100.0 + (i % 10) for i in range(n)],
        "cost": [60.0 + (i % 5) for i in range(n)],
        "retail_price": [120.0 + (i % 10) for i in range(n)],

        # item-level return signals (required by your function)
        "item_status": item_status,
        "returned_at": returned_at,

        # order-level return signal (your error shows this is required)
        "order_status": order_status,

        # timestamps (common names)
        "created_at": [created_at for _ in range(n)],
        "shipped_at": [shipped_at for _ in range(n)],
        "delivered_at": [delivered_at for _ in range(n)],

        # additional common variants (guards against schema differences)
        "order_created_at": [created_at for _ in range(n)],
        "order_shipped_at": [shipped_at for _ in range(n)],
        "order_delivered_at": [delivered_at for _ in range(n)],
    })

    return df


@pytest.fixture(autouse=True)
def patch_build_analysis_dataset(monkeypatch, analysis_df):
    """
    Patch dp.build_analysis_dataset() so tests do not depend on files on disk.
    """
    monkeypatch.setattr(dp, "build_analysis_dataset", lambda *args, **kwargs: analysis_df.copy())


def test_product_profit_erosion_metrics_runs():
    df = dp.build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    views = build_product_profit_erosion_metrics(df, min_rows=50)

    assert isinstance(views, dict)
    assert set(views.keys()) == {"by_category", "by_brand", "by_department"}

    for _, table in views.items():
        assert isinstance(table, pd.DataFrame)
        assert "item_rows" in table.columns
        assert "returned_items" in table.columns
        assert "return_rate" in table.columns
        assert "total_profit_erosion" in table.columns


def test_product_return_behavior_metrics_runs():
    df = dp.build_analysis_dataset()
    df = engineer_return_features(df)

    rb = build_product_return_behavior_metrics(df, min_rows=50)

    assert isinstance(rb, dict)
    assert "by_category" in rb
    assert "by_brand" in rb

    for key in ["by_category", "by_brand"]:
        table = rb[key]
        assert isinstance(table, pd.DataFrame)
        assert "item_rows" in table.columns
        assert "returned_items" in table.columns
        assert "return_rate" in table.columns


def test_customer_profit_erosion_summaries_runs():
    df = dp.build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    out = build_customer_profit_erosion_summaries(df, min_returns=1)

    assert isinstance(out, pd.DataFrame)
    assert "user_id" in out.columns
    assert "return_rows" in out.columns
    assert "total_profit_erosion" in out.columns
    assert "avg_profit_erosion_per_return" in out.columns


def test_product_modeling_dataset():
    df = dp.build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    prod_erosion = build_product_profit_erosion_metrics(df, min_rows=50)
    prod_returns = build_product_return_behavior_metrics(df, min_rows=50)

    assert "by_category" in prod_erosion
    assert "by_category" in prod_returns

    out = build_product_modeling_dataset(
        prod_erosion,
        prod_returns,
        level="by_category",
    )

    assert "total_profit_erosion" in out.columns
    assert "return_rate" in out.columns
    assert "category" in out.columns
    assert len(out) > 0


def test_customer_modeling_dataset():
    df = dp.build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    cust = build_customer_profit_erosion_summaries(df)

    out = build_customer_modeling_dataset(cust)

    assert "user_id" in out.columns
    assert "total_profit_erosion" in out.columns