from pathlib import Path

from src.data_processing import build_analysis_dataset
from src.feature_engineering import (
    engineer_return_features,
    calculate_margins,
    calculate_profit_erosion,
)
from src.descriptive_transformations import (
    build_product_profit_erosion_metrics,
    build_product_return_behavior_metrics,
)

OUT_DIR = Path("reports/rq1")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # STEP 1 (Template): Build dataset
    df = build_analysis_dataset()

    # STEP 2: Engineer return features (creates is_returned_item)
    df = engineer_return_features(df)

    # STEP 3: Calculate margins (creates item_margin)
    df = calculate_margins(df)

    # STEP 4 (Template): Filter to returned items BEFORE profit erosion calc
    returned_df = df[df["is_returned_item"] == 1].copy()

    # STEP 5 (Template): Calculate profit erosion on returned items only
    returned_df = calculate_profit_erosion(
        returned_df,
        use_category_tiers=True
    )

    # Save returned-item-level dataset for hypothesis tests
    returned_df.to_parquet(OUT_DIR / "rq1_returned_items.parquet", index=False)

    # STEP 6 (Reporting): Use US07 descriptive tables (do not recreate)
    # IMPORTANT: These US07 functions should receive the FULL df (all items)
    # so return-rate denominators include non-returned items.
    erosion_views = build_product_profit_erosion_metrics(
        df,
        min_rows=200,
        use_category_tiers=True
    )
    return_views = build_product_return_behavior_metrics(
        df,
        min_rows=200
    )

    # Export reporting tables
    erosion_views["by_category"].to_csv(OUT_DIR / "rq1_erosion_by_category.csv", index=False)
    erosion_views["by_brand"].to_csv(OUT_DIR / "rq1_erosion_by_brand.csv", index=False)
    erosion_views["by_department"].to_csv(OUT_DIR / "rq1_erosion_by_department.csv", index=False)

    return_views["by_category"].to_csv(OUT_DIR / "rq1_return_rates_by_category.csv", index=False)
    return_views["by_brand"].to_csv(OUT_DIR / "rq1_return_rates_by_brand.csv", index=False)

    print("RQ1 data + tables written to reports/rq1/")
    print("Next: run stats with python -m src.rq1.rq1_stats")


if __name__ == "__main__":
    main()