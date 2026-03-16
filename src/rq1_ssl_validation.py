# src/rq1_ssl_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RQ1SSLMapping:
    # Engineered SSL fields
    ssl_returns_col: str = "Returns"
    ssl_loss_col: str = "total_loss"

    # Category is built by concatenating three SSL fields:
    #   {Pillar} - {Major Market Cat} - {Department}
    ssl_pillar_col: str = "Pillar"
    ssl_major_market_cat_col: str = "Major Market Cat"
    ssl_department_col: str = "Department"

    # Separator used when concatenating the three fields
    category_sep: str = "-"

    # Canonical RQ1 fields
    canonical_return_flag: str = "is_returned_item"
    canonical_profit_erosion: str = "profit_erosion"
    canonical_category: str = "category"


def _coerce_bool_to_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    s = series.astype(str).str.strip().str.upper()
    return s.isin(["TRUE", "T", "1", "YES", "Y"]).astype(int)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _build_category_label(df: pd.DataFrame, m: RQ1SSLMapping) -> pd.Series:
    """
    Concatenate Pillar + Major Market Cat + Department with the configured
    separator to form a single category label.

    Example: "STEM" + "-" + "Science" + "-" + "Physics" → "STEM-Science-Physics"

    Missing values in any component are filled with "Unknown" before
    concatenation so no row produces a NaN label.
    """
    pillar = df[m.ssl_pillar_col].fillna("Unknown").astype(str).str.strip()
    major = df[m.ssl_major_market_cat_col].fillna("Unknown").astype(str).str.strip()
    dept = df[m.ssl_department_col].fillna("Unknown").astype(str).str.strip()
    return pillar + m.category_sep + major + m.category_sep + dept


def validate_ssl_columns(df: pd.DataFrame, m: RQ1SSLMapping) -> None:
    required = [
        m.ssl_returns_col,
        m.ssl_loss_col,
        m.ssl_pillar_col,
        m.ssl_major_market_cat_col,
        m.ssl_department_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Engineered SSL dataset is missing required columns for RQ1 mapping:\n"
            f"{missing}\n\n"
            "Fix preprocessing or update RQ1SSLMapping.\n\n"
            "Note: category is built from Pillar + Major Market Cat + Department."
        )


def build_rq1_ssl_canonical_base(
    ssl_engineered_df: pd.DataFrame,
    mapping: Optional[RQ1SSLMapping] = None,
    *,
    enforce_positive_erosion: bool = True,
) -> pd.DataFrame:
    m = mapping or RQ1SSLMapping()
    validate_ssl_columns(ssl_engineered_df, m)

    df = ssl_engineered_df.copy()

    df[m.canonical_return_flag] = _coerce_bool_to_int(df[m.ssl_returns_col])
    df[m.canonical_profit_erosion] = _safe_numeric(df[m.ssl_loss_col])
    df = df.dropna(subset=[m.canonical_profit_erosion]).copy()

    if enforce_positive_erosion:
        df[m.canonical_profit_erosion] = df[m.canonical_profit_erosion].abs()

    # Category: concatenation of Pillar + Major Market Cat + Department
    df[m.canonical_category] = _build_category_label(df, m)

    return df


def _group_metrics(
    base_df: pd.DataFrame,
    group_col: str,
    *,
    return_flag_col: str = "is_returned_item",
    erosion_col: str = "profit_erosion",
) -> pd.DataFrame:
    # return rate on all lines
    rr = (
        base_df.groupby(group_col)[return_flag_col]
        .mean()
        .reset_index()
        .rename(columns={return_flag_col: "return_rate"})
    )

    # erosion on returned lines only
    ret = base_df[base_df[return_flag_col] == 1].copy()
    if ret.empty:
        out = rr.copy()
        out["total_profit_erosion"] = 0.0
        out["returned_items"] = 0
        out["mean_profit_erosion_per_return"] = np.nan
        out["avg_profit_erosion"] = np.nan
        return out

    erosion = (
        ret.groupby(group_col)[erosion_col]
        .agg(
            total_profit_erosion="sum",
            returned_items="count",
            mean_profit_erosion_per_return="mean",
        )
        .reset_index()
    )

    out = erosion.merge(rr, on=group_col, how="left")
    out["avg_profit_erosion"] = out["mean_profit_erosion_per_return"]  # compatibility alias
    out = out.sort_values("total_profit_erosion", ascending=False).reset_index(drop=True)
    return out


def build_rq1_ssl_group_artifacts(base_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build category-level group artifacts only.

    Brand and department are not validated — category alone is the
    validation dimension, constructed from Pillar + Major Market Cat + Department.
    """
    return {
        "by_category": _group_metrics(base_df, "category"),
    }


def save_rq1_ssl_artifacts(
    base_df: pd.DataFrame,
    artifacts: Dict[str, pd.DataFrame],
    out_dir: Path,
    *,
    csv_dir: Optional[Path] = None,
    prefix: str = "rq1_ssl"
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if csv_dir is None:
        csv_dir = out_dir
    csv_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    base_path = out_dir / f"{prefix}_base_canonical.parquet"
    base_df.to_parquet(base_path, index=False)
    paths["base"] = base_path

    returned_df = base_df[base_df["is_returned_item"] == 1].copy()
    returned_path = out_dir / f"{prefix}_returned_items.parquet"
    returned_df.to_parquet(returned_path, index=False)
    paths["returned"] = returned_path

    for k, df in artifacts.items():
        pq = out_dir / f"{prefix}_{k}.parquet"
        csv = csv_dir / f"{prefix}_{k}.csv"
        df.to_parquet(pq, index=False)
        df.to_csv(csv, index=False)
        paths[f"{k}_parquet"] = pq
        paths[f"{k}_csv"] = csv

    return paths


def build_and_save_rq1_ssl_dataset(
    ssl_engineered_df: pd.DataFrame,
    out_dir: Path,
    mapping: Optional[RQ1SSLMapping] = None,
    *,
    csv_dir: Optional[Path] = None,
    prefix: str = "rq1_ssl"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Path]]:
    base = build_rq1_ssl_canonical_base(ssl_engineered_df, mapping=mapping)
    artifacts = build_rq1_ssl_group_artifacts(base)
    paths = save_rq1_ssl_artifacts(base, artifacts, out_dir=out_dir, csv_dir=csv_dir, prefix=prefix)
    return base, artifacts, paths
