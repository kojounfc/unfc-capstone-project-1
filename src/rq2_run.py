"""
RQ2 / US09 Runner - Profit erosion concentration + customer segmentation.
OPTIMIZED VERSION with performance improvements for large datasets.

This runner:
- Loads the processed item-level dataset: returns_eda_v1.parquet
- Builds customer_behavior (engineer_customer_behavioral_features)
- Builds customer_erosion
  (engineer_return_features -> calculate_margins -> filter returns ->
   calculate_profit_erosion -> aggregate_profit_erosion_by_customer)
- Uses src.rq2_concentration utilities:
    - compute_pareto_table
    - lorenz_curve_points
    - gini_coefficient
    - top_x_customer_share_of_value
- Uses src.rq2_segmentation utilities:
    - build_customer_segmentation_table
    - select_numeric_features
    - standardize_features
    - kmeans_fit_predict
    - summarize_clusters
    - elbow_inertia_over_k
    - silhouette_over_k

PERFORMANCE OPTIMIZATIONS:
- Reduced k range (2-8 instead of 2-10) for faster computation
- Sampling for silhouette computation on large datasets (>10k samples)
- Progress indicators for long-running operations

Outputs (CI-safe) written to:
- data/processed/rq2/
and optional plots to:
- figures/rq2/

Usage:
  python -m src.rq2_run
  python -m src.rq2_run --k 4
  python -m src.rq2_run --k 5 --top-x 0.2
  python -m src.rq2_run --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # CI-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR
from src.data_processing import load_processed_data
from src.descriptive_transformations import _require_columns
from src.feature_engineering import (
    aggregate_profit_erosion_by_customer,
    calculate_margins,
    calculate_profit_erosion,
    engineer_customer_behavioral_features,
    engineer_return_features,
    save_feature_engineered_dataset,
)
from src.rq2_concentration import (
    bootstrap_gini_p_value,
    concentration_comparison,
    compute_pareto_table,
    gini_coefficient,
    lorenz_curve_points,
    top_x_customer_share_of_value,
)
from src.rq2_segmentation import (
    build_customer_segmentation_table,
    elbow_inertia_over_k,
    kmeans_fit_predict,
    select_numeric_features,
    silhouette_over_k,
    standardize_features,
    summarize_clusters,
)
from src.visualization import _safe_tight_layout, set_plot_style

RQ2_OUT_DIR = PROCESSED_DATA_DIR / "rq2"
RQ2_FIG_DIR = FIGURES_DIR / "rq2"


@dataclass(frozen=True)
class RQ2Summary:
    """Structured summary of key RQ2 concentration and clustering outputs."""

    customers: int
    total_profit_erosion: float
    gini: float
    top_x: float
    top_x_share_of_erosion: float
    k_used: int


def _plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    marker: Optional[str] = None,
    add_equality_line: bool = False,
) -> None:
    """Render and save a standard line plot with consistent style."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    fig, ax = plt.subplots()

    line_kwargs = {"marker": marker} if marker else {}
    ax.plot(df[x_col], df[y_col], **line_kwargs)
    if add_equality_line:
        ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    _safe_tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_customer_erosion(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer_erosion table using feature_engineering functions.
    
    This function handles both raw and processed data:
    - If data already has 'is_returned_item', uses it directly (processed data)
    - If data has 'item_status', engineers return features first (raw data)
    
    Returned-items only are passed into calculate_profit_erosion (by design).
    
    Args:
        item_df: Item-level DataFrame (raw or processed)
    
    Returns:
        Customer-level erosion table sorted by total_profit_erosion descending
    """
    df = item_df.copy()

    # Check if we need to engineer return features
    if 'is_returned_item' not in df.columns:
        # Raw data path - requires item_status and order_status
        _require_columns(
            df,
            required=[
                "user_id",
                "order_id",
                "order_item_id",
                "item_status",
                "order_status",
                "sale_price",
                "retail_price",
                "cost",
            ],
            context="build_customer_erosion (raw data)",
        )
        df = engineer_return_features(df)
        df = calculate_margins(df)
    else:
        # Processed data path - requires basic columns
        _require_columns(
            df,
            required=[
                "user_id",
                "order_id",
                "order_item_id",
                "is_returned_item",
                "sale_price",
                "retail_price",
                "cost",
            ],
            context="build_customer_erosion (processed data)",
        )
        # Ensure margins exist
        if 'item_margin' not in df.columns:
            df = calculate_margins(df)

    returned = df[df["is_returned_item"] == 1].copy()

    # Profit erosion (uses category tiers if category exists;
    # function handles missing category)
    returned = calculate_profit_erosion(returned, use_category_tiers=True)

    customer_erosion = aggregate_profit_erosion_by_customer(returned)
    customer_erosion = customer_erosion.sort_values(
        "total_profit_erosion", ascending=False
    ).reset_index(drop=True)

    # Drop columns that overlap with customer_behavior to avoid merge collision
    overlap = ["total_sales", "total_margin"]
    customer_erosion = customer_erosion.drop(
        columns=[c for c in overlap if c in customer_erosion.columns]
    )

    return customer_erosion


def run_rq2(
    input_parquet: Optional[Path] = None,
    out_dir: Path = RQ2_OUT_DIR,
    k: Optional[int] = None,
    k_min: int = 2,
    k_max: int = 8,  # OPTIMIZED: Reduced from 10 to 8
    top_x: float = 0.2,
    make_plots: bool = True,
) -> RQ2Summary:
    """
    Run the full RQ2 concentration and segmentation pipeline.
    
    PERFORMANCE OPTIMIZATIONS:
    - Reduced k_max default from 10 to 8 (faster, rarely need k>8 for interpretability)
    - Sampling for silhouette computation on large datasets (>10k samples)
    - Progress indicators for long-running operations
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load processed item-level dataset
    print("Loading processed data...", file=sys.stderr)
    item_df = load_processed_data(input_parquet)
    print(f"✓ Loaded {len(item_df):,} item records", file=sys.stderr)

    # Ensure required engineered columns exist for behavior features
    item_df = engineer_return_features(item_df)
    item_df = calculate_margins(item_df)

    # 2) Customer behavior features
    # feature_engineering expects order_created_at for recency logic
    # Use a safe proxy timestamp when order_created_at is missing.
    if "order_created_at" not in item_df.columns:
        for c in ["item_created_at", "item_shipped_at", "item_delivered_at"]:
            if c in item_df.columns:
                item_df["order_created_at"] = item_df[c]
                break
        else:
            item_df["order_created_at"] = pd.Timestamp.today()

    print("Building customer behavioral features...", file=sys.stderr)
    customer_behavior = engineer_customer_behavioral_features(item_df)
    print(f"✓ Created features for {len(customer_behavior):,} customers", file=sys.stderr)

    # 3) Customer erosion table
    print("Building customer erosion metrics...", file=sys.stderr)
    customer_erosion = build_customer_erosion(item_df)
    print(f"✓ Computed erosion for {len(customer_erosion):,} customers with returns", file=sys.stderr)

    save_feature_engineered_dataset(
        customer_erosion,
        filename="customer_erosion",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )

    # 4) Concentration metrics
    print("Computing concentration metrics...", file=sys.stderr)
    pareto = compute_pareto_table(customer_erosion, value_col="total_profit_erosion")
    lorenz = lorenz_curve_points(customer_erosion, value_col="total_profit_erosion")
    gini = gini_coefficient(customer_erosion, value_col="total_profit_erosion")
    top_share = top_x_customer_share_of_value(
        customer_erosion, x=top_x, value_col="total_profit_erosion"
    )
    print(f"✓ Gini coefficient: {gini:.4f}, Top {top_x:.0%} share: {top_share:.1%}", file=sys.stderr)

    save_feature_engineered_dataset(
        pareto,
        filename="pareto_table",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )
    save_feature_engineered_dataset(
        lorenz,
        filename="lorenz_points",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )

    # 5) Build segmentation table (behavior + erosion)
    print("Building segmentation table...", file=sys.stderr)
    seg_table = build_customer_segmentation_table(customer_behavior, customer_erosion)

    # 6) Select + standardize features, cluster, diagnostics
    print("Selecting features (excluding leakage)...", file=sys.stderr)
    X_df, used_cols = select_numeric_features(
        seg_table,
        id_col="user_id",
        feature_cols=None,
        exclude_leakage_features=True,
    )
    print(f"✓ Selected {len(used_cols)} behavioral features", file=sys.stderr)
    
    X_scaled = standardize_features(X_df)

    # OPTIMIZATION: Use sampling for large datasets
    use_sampling = len(X_scaled) > 10000
    if use_sampling:
        print(f"Large dataset detected ({len(X_scaled):,} samples)", file=sys.stderr)
        print("Using sampling (5,000 samples) for silhouette computation...", file=sys.stderr)
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_scaled), size=min(5000, len(X_scaled)), replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled

    selected_k = k
    if selected_k is None:
        print(f"Auto-selecting k using silhouette score (k range: {k_min}-{k_max})...", file=sys.stderr)
        k_list_sil = list(range(max(2, k_min), k_max + 1))
        silhouette_for_selection = silhouette_over_k(
            X_sample, k_list=k_list_sil, random_state=42
        )
        selected_k = int(
            silhouette_for_selection.sort_values(
                ["silhouette", "k"], ascending=[False, True]
            )
            .iloc[0]["k"]
        )
        print(f"✓ Selected k={selected_k}", file=sys.stderr)

    print(f"Applying K-Means clustering (k={selected_k})...", file=sys.stderr)
    labels = kmeans_fit_predict(X_scaled, k=selected_k, random_state=42)
    clustered = seg_table.copy()
    clustered["cluster_id"] = labels.astype(int)
    print(f"✓ Clustering complete", file=sys.stderr)

    save_feature_engineered_dataset(
        clustered,
        filename="clustered_customers",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )

    # Cluster summary (US09 requirement #89: segment-level profit erosion compared)
    cluster_summary = summarize_clusters(
        clustered,
        value_col="total_profit_erosion",
        cluster_col="cluster_id",
    )

    save_feature_engineered_dataset(
        cluster_summary,
        filename="cluster_summary",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )

    # Diagnostics
    print("Computing clustering diagnostics...", file=sys.stderr)
    k_list_elbow = list(range(1, k_max + 1))  # OPTIMIZED: Uses k_max=8
    elbow_df = elbow_inertia_over_k(X_scaled, k_list=k_list_elbow, random_state=42)

    k_list_sil = list(range(max(2, k_min), k_max + 1))
    # OPTIMIZATION: Use sampled data for silhouette diagnostics too
    silhouette_df = silhouette_over_k(X_sample, k_list=k_list_sil, random_state=42)
    print(f"✓ Diagnostics complete", file=sys.stderr)

    save_feature_engineered_dataset(
        elbow_df,
        filename="elbow_inertia",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )
    save_feature_engineered_dataset(
        silhouette_df,
        filename="silhouette_scores",
        output_dir=out_dir,
        save_parquet=True,
        save_csv=True,
    )

    # Save metadata (features used)
    meta = {
        "k_used": int(selected_k),
        "k_selection_method": (
            "silhouette_argmax_tiebreak_lowest_k" if k is None else "user_provided"
        ),
        "feature_columns_used": used_cols,
        "feature_policy": "behavioral_non_leakage_only",
        "top_x": float(top_x),
        "gini": float(gini),
        "top_x_share_of_erosion": float(top_share),
        "performance_optimizations": {
            "k_max_reduced": True,
            "k_max_value": k_max,
            "sampling_used": use_sampling,
            "sample_size": 5000 if use_sampling else len(X_scaled),
            "total_customers": len(X_scaled),
        },
    }

    # Concentration inference + context
    print("Running bootstrap significance test...", file=sys.stderr)
    gini_bootstrap = bootstrap_gini_p_value(
        customer_erosion,
        value_col="total_profit_erosion",
        n_bootstrap=1000,
        random_state=42,
    )

    concentration_comparison_metrics = {}
    if "total_sales" in seg_table.columns:
        concentration_comparison_metrics = concentration_comparison(
            seg_table,
            erosion_col="total_profit_erosion",
            baseline_col="total_sales",
        )

    meta["gini_bootstrap_test"] = gini_bootstrap
    if concentration_comparison_metrics:
        meta["concentration_comparison"] = concentration_comparison_metrics
        
    (out_dir / "rq2_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Optional plots
    if make_plots:
        print("Generating plots...", file=sys.stderr)
        _plot_line(
            pareto,
            x_col="customer_share",
            y_col="value_share",
            xlabel="Cumulative share of customers",
            ylabel="Cumulative share of profit erosion",
            title="Pareto Curve: Profit Erosion Concentration",
            out_path=RQ2_FIG_DIR / "pareto_curve.png",
        )
        _plot_line(
            lorenz,
            x_col="population_share",
            y_col="value_share",
            xlabel="Cumulative share of customers",
            ylabel="Cumulative share of profit erosion",
            title="Lorenz Curve: Profit Erosion Distribution",
            out_path=RQ2_FIG_DIR / "lorenz_curve.png",
            add_equality_line=True,
        )
        _plot_line(
            elbow_df,
            x_col="k",
            y_col="inertia",
            xlabel="k",
            ylabel="Inertia",
            title="Elbow Diagnostic (Inertia vs k)",
            out_path=RQ2_FIG_DIR / "elbow_inertia.png",
            marker="o",
        )
        _plot_line(
            silhouette_df,
            x_col="k",
            y_col="silhouette",
            xlabel="k",
            ylabel="Silhouette score",
            title="Silhouette Diagnostic (Score vs k)",
            out_path=RQ2_FIG_DIR / "silhouette_scores.png",
            marker="o",
        )
        print(f"✓ Plots saved to {RQ2_FIG_DIR}", file=sys.stderr)

    # Summary
    total_erosion = float(
        pd.to_numeric(customer_erosion["total_profit_erosion"], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    summary = RQ2Summary(
        customers=int(len(customer_erosion)),
        total_profit_erosion=float(total_erosion),
        gini=float(gini),
        top_x=float(top_x),
        top_x_share_of_erosion=float(top_share),
        k_used=int(selected_k),
    )

    (out_dir / "rq2_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )

    print("✓ RQ2 analysis complete", file=sys.stderr)
    return summary


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the RQ2 runner."""
    p = argparse.ArgumentParser(
        description="Run RQ2 concentration + segmentation pipeline (optimized)."
    )
    p.add_argument(
        "--input-parquet",
        type=str,
        default=None,
        help=(
            "Optional path to processed parquet; default uses "
            "config PROCESSED_PARQUET."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(RQ2_OUT_DIR),
        help="Output directory (default: data/processed/rq2).",
    )
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="KMeans clusters (default: auto-select via silhouette).",
    )
    p.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Min k for silhouette diagnostic (default 2).",
    )
    p.add_argument(
        "--k-max", 
        type=int, 
        default=8,  # OPTIMIZED: Changed from 10 to 8
        help="Max k for diagnostics (default 8, optimized for performance)."
    )
    p.add_argument(
        "--top-x",
        type=float,
        default=0.2,
        help="Top x fraction for concentration metric (default 0.2).",
    )
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return p.parse_args()


def main() -> None:
    """CLI entrypoint for running RQ2 end-to-end."""
    args = _parse_args()
    summary = run_rq2(
        input_parquet=Path(args.input_parquet) if args.input_parquet else None,
        out_dir=Path(args.out_dir),
        k=args.k,
        k_min=args.k_min,
        k_max=args.k_max,
        top_x=args.top_x,
        make_plots=not args.no_plots,
    )

    print("RQ2 complete.")
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
