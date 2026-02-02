import pandas as pd
import pytest

from src.data_processing import build_analysis_dataset
from src.feature_engineering import engineer_return_features, calculate_margins
from src.descriptive_transformations import (
    build_product_profit_erosion_metrics,
    build_product_return_behavior_metrics,
    build_customer_profit_erosion_summaries,
    build_product_modeling_dataset,
    build_customer_modeling_dataset,
)

def test_product_profit_erosion_metrics_runs():
    df = build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    views = build_product_profit_erosion_metrics(df, min_rows=50)

    assert isinstance(views, dict)
    assert set(views.keys()) == {"by_category", "by_brand", "by_department"}

    for _, table in views.items():
        assert isinstance(table, pd.DataFrame)
        # return-rate context columns from analytics.calculate_return_rates_by_group
        assert "item_rows" in table.columns
        assert "returned_items" in table.columns
        assert "return_rate" in table.columns
        # erosion summary columns
        assert "total_profit_erosion" in table.columns


def test_product_return_behavior_metrics_runs():
    df = build_analysis_dataset()
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
    df = build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    out = build_customer_profit_erosion_summaries(df, min_returns=1)

    assert isinstance(out, pd.DataFrame)
    assert "user_id" in out.columns
    assert "return_rows" in out.columns
    assert "total_profit_erosion" in out.columns
    assert "avg_profit_erosion_per_return" in out.columns

def test_product_modeling_dataset():
    df = build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    prod_erosion = build_product_profit_erosion_metrics(df, min_rows=50)
    prod_returns = build_product_return_behavior_metrics(df, min_rows=50)

    # Validate expected keys exist BEFORE calling the join
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
    df = build_analysis_dataset()
    df = engineer_return_features(df)
    df = calculate_margins(df)

    cust = build_customer_profit_erosion_summaries(df)

    out = build_customer_modeling_dataset(cust)

    assert "user_id" in out.columns
    assert "total_profit_erosion" in out.columns