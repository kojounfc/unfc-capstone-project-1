\
from pathlib import Path

import pandas as pd

from src import rq1_run


def test_run_rq1_writes_expected_artifacts(tmp_path):
    # Build a minimal dataset with required columns
    base_df = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2],
            "product_id": [10, 11, 10, 12],
            "category": ["CatA", "CatB", "CatA", "CatC"],
            "brand": ["Brand1", "Brand2", "Brand1", "Brand3"],
            "department": ["Dept1", "Dept1", "Dept1", "Dept2"],
            "sale_price": [100.0, 50.0, 100.0, 20.0],
            "cost": [60.0, 30.0, 60.0, 10.0],
            "is_returned_item": [1, 0, 1, 0],
        }
    )

    def build_analysis_dataset_fn():
        return base_df.copy()

    def engineer_return_features_fn(df):
        # already present, but keep contract
        return df

    def calculate_margins_fn(df):
        df = df.copy()
        df["item_margin"] = df["sale_price"] - df["cost"]
        return df

    def calculate_profit_erosion_fn(df, use_category_tiers=True):
        df = df.copy()
        # A toy "profit_erosion" calculation: margin reversal + fixed cost
        df["profit_erosion"] = (df["sale_price"] - df["cost"]) + 5.0
        return df

    def build_product_profit_erosion_metrics_fn(df, min_rows=200, use_category_tiers=True):
        # Return the dict contract used by rq1_run
        by_category = (
            df.groupby("category", as_index=False)
            .agg(total_profit_erosion=("sale_price", "sum"))
        )
        by_brand = (
            df.groupby("brand", as_index=False)
            .agg(total_profit_erosion=("sale_price", "sum"))
        )
        by_department = (
            df.groupby("department", as_index=False)
            .agg(total_profit_erosion=("sale_price", "sum"))
        )
        return {"by_category": by_category, "by_brand": by_brand, "by_department": by_department}

    def build_product_return_behavior_metrics_fn(df, min_rows=200):
        by_category = df.groupby("category", as_index=False).agg(return_rate=("is_returned_item", "mean"))
        by_brand = df.groupby("brand", as_index=False).agg(return_rate=("is_returned_item", "mean"))
        return {"by_category": by_category, "by_brand": by_brand}

    out_dir = tmp_path / "data" / "processed" / "rq1"

    summary = rq1_run.run_rq1(
        out_dir=out_dir,
        build_analysis_dataset_fn=build_analysis_dataset_fn,
        engineer_return_features_fn=engineer_return_features_fn,
        calculate_margins_fn=calculate_margins_fn,
        calculate_profit_erosion_fn=calculate_profit_erosion_fn,
        build_product_profit_erosion_metrics_fn=build_product_profit_erosion_metrics_fn,
        build_product_return_behavior_metrics_fn=build_product_return_behavior_metrics_fn,
    )

    assert summary["status"] == "ok"
    artifacts = summary["artifacts"]

    # Files must exist
    assert Path(artifacts["returned_items_parquet"]).exists()
    assert Path(artifacts["erosion_by_category_csv"]).exists()
    assert Path(artifacts["return_rates_by_brand_csv"]).exists()
