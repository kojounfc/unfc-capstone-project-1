"""
RQ1 runner: orchestration of customer/product-level return + profit erosion analysis.

This module intentionally focuses on *wiring* together already-validated upstream pipeline
components (data processing + feature engineering + descriptive transformations) to produce
RQ1 artifacts in a deterministic, CI-safe way.

Key outputs:
- Returned item dataset (parquet) used for hypothesis tests
- Descriptive erosion/return tables (csv/parquet)

Figures are handled separately in `rq1_visuals.py` and must be saved under figures/rq1/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd


def _default_processed_dir() -> Path:
    """
    Resolve the default RQ1 processed output directory.

    Uses `src.config.PROCESSED_DATA_DIR` when available; otherwise falls back to
    `data/processed` relative to the current working directory.

    Returns
    -------
    Path
        Directory path where RQ1 processed artifacts should be written.
    """
    try:
        # Expected to exist in the project repo (used in rq2 as well)
        from src.config import PROCESSED_DATA_DIR  # type: ignore

        return Path(PROCESSED_DATA_DIR) / "rq1"
    except Exception:
        return Path("data") / "processed" / "rq1"


@dataclass(frozen=True)
class RQ1Artifacts:
    """
    Container for key RQ1 output artifact paths.
    """

    out_dir: Path
    returned_items_parquet: Path
    erosion_by_category_csv: Path
    erosion_by_brand_csv: Path
    erosion_by_department_csv: Path
    return_rates_by_category_csv: Path
    return_rates_by_brand_csv: Path
    stats_summary_csv: Path


def run_rq1(
    *,
    out_dir: Optional[Path] = None,
    min_rows: int = 200,
    use_category_tiers: bool = True,
    build_analysis_dataset_fn: Optional[Callable[[], pd.DataFrame]] = None,
    engineer_return_features_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    calculate_margins_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    calculate_profit_erosion_fn: Optional[Callable[..., pd.DataFrame]] = None,
    build_product_profit_erosion_metrics_fn: Optional[Callable[..., Dict[str, pd.DataFrame]]] = None,
    build_product_return_behavior_metrics_fn: Optional[Callable[..., Dict[str, pd.DataFrame]]] = None,
) -> Dict[str, Any]:
    """
    Execute the full RQ1 pipeline and write version-controlled artifacts.

    Parameters
    ----------
    out_dir:
        Output directory for RQ1 processed artifacts. If None, a default path is resolved.
    min_rows:
        Minimum rows threshold passed into descriptive table builders (to reduce noise).
    use_category_tiers:
        Whether to apply category-tier handling when calculating profit erosion.
    *_fn:
        Optional dependency-injection hooks used primarily for unit testing. When not
        provided, imports from the project's pipeline modules are used.

    Returns
    -------
    Dict[str, Any]
        A summary dictionary including record counts and the artifact paths written.
    """
    out_dir = Path(out_dir) if out_dir is not None else _default_processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dependencies (default to project implementations)
    if build_analysis_dataset_fn is None:
        from src.data_processing import build_analysis_dataset as build_analysis_dataset_fn  # type: ignore
    if engineer_return_features_fn is None:
        from src.feature_engineering import engineer_return_features as engineer_return_features_fn  # type: ignore
    if calculate_margins_fn is None:
        from src.feature_engineering import calculate_margins as calculate_margins_fn  # type: ignore
    if calculate_profit_erosion_fn is None:
        from src.feature_engineering import calculate_profit_erosion as calculate_profit_erosion_fn  # type: ignore
    if build_product_profit_erosion_metrics_fn is None:
        from src.descriptive_transformations import (  # type: ignore
            build_product_profit_erosion_metrics as build_product_profit_erosion_metrics_fn,
        )
    if build_product_return_behavior_metrics_fn is None:
        from src.descriptive_transformations import (  # type: ignore
            build_product_return_behavior_metrics as build_product_return_behavior_metrics_fn,
        )

    # STEP 1: Build base analysis dataset
    df = build_analysis_dataset_fn()

    # STEP 2: Engineer return features (expects to create `is_returned_item`)
    df = engineer_return_features_fn(df)

    # STEP 3: Calculate margins (expects to create `item_margin`)
    df = calculate_margins_fn(df)

    # Guardrails
    required_cols = {"is_returned_item"}
    missing = sorted([c for c in required_cols if c not in df.columns])
    if missing:
        raise ValueError(f"RQ1 pipeline missing required columns after feature engineering: {missing}")

    # STEP 4: Filter returned items BEFORE profit erosion calc
    returned_df = df[df["is_returned_item"] == 1].copy()
    if returned_df.empty:
        raise ValueError("No returned items found (is_returned_item==1). RQ1 cannot proceed.")

    # STEP 5: Calculate profit erosion on returned items only
    returned_df = calculate_profit_erosion_fn(returned_df, use_category_tiers=use_category_tiers)

    # Persist returned-item-level dataset for hypothesis tests
    returned_items_parquet = out_dir / "rq1_returned_items.parquet"
    returned_df.to_parquet(returned_items_parquet, index=False)

    # STEP 6: Descriptive tables (pass FULL df so denominators include non-returned items)
    erosion_views = build_product_profit_erosion_metrics_fn(
        df,
        min_rows=min_rows,
        use_category_tiers=use_category_tiers,
    )
    return_views = build_product_return_behavior_metrics_fn(df, min_rows=min_rows)

    # Export reporting tables (CSV)
    erosion_by_category_csv = out_dir / "rq1_erosion_by_category.csv"
    erosion_by_brand_csv = out_dir / "rq1_erosion_by_brand.csv"
    erosion_by_department_csv = out_dir / "rq1_erosion_by_department.csv"
    return_rates_by_category_csv = out_dir / "rq1_return_rates_by_category.csv"
    return_rates_by_brand_csv = out_dir / "rq1_return_rates_by_brand.csv"

    erosion_views["by_category"].to_csv(erosion_by_category_csv, index=False)
    erosion_views["by_brand"].to_csv(erosion_by_brand_csv, index=False)
    erosion_views["by_department"].to_csv(erosion_by_department_csv, index=False)

    return_views["by_category"].to_csv(return_rates_by_category_csv, index=False)
    return_views["by_brand"].to_csv(return_rates_by_brand_csv, index=False)

    # Create an artifacts manifest path for downstream (stats/visuals) tooling
    stats_summary_csv = out_dir / "rq1_statistical_tests_summary.csv"  # produced by rq1_stats

    artifacts = RQ1Artifacts(
        out_dir=out_dir,
        returned_items_parquet=returned_items_parquet,
        erosion_by_category_csv=erosion_by_category_csv,
        erosion_by_brand_csv=erosion_by_brand_csv,
        erosion_by_department_csv=erosion_by_department_csv,
        return_rates_by_category_csv=return_rates_by_category_csv,
        return_rates_by_brand_csv=return_rates_by_brand_csv,
        stats_summary_csv=stats_summary_csv,
    )

    return {
        "status": "ok",
        "out_dir": str(out_dir),
        "n_rows_full": int(len(df)),
        "n_rows_returned": int(len(returned_df)),
        "artifacts": {k: str(v) for k, v in artifacts.__dict__.items()},
    }


def main() -> None:
    """
    CLI entrypoint for running RQ1 artifacts generation.

    Notes
    -----
    This writes artifacts to the default processed directory:
    - data/processed/rq1/ (fallback)
    - or src.config.PROCESSED_DATA_DIR / rq1 (preferred)

    Examples
    --------
    python -m src.rq1_run
    """
    summary = run_rq1()
    print("RQ1 artifacts written.")
    print(summary)


if __name__ == "__main__":
    main()
