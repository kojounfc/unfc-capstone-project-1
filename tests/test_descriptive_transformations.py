import pandas as pd

from src.data_processing import build_analysis_dataset
from src.feature_engineering import engineer_return_features, calculate_margins
from src.descriptive_transformations import build_product_profit_erosion_metrics


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