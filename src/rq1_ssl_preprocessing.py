# src/rq1_ssl_preprocessing.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RQ1SSLPreprocessConfig:
    """
    RQ1 SSL Preprocessing (Raw -> Engineered)

    Input:
      data/raw/SSL_Returns_df_yoy.csv

    Output:
      data/processed/rq1_ssl/rq1_ssl_engineered.parquet
      (optional) data/processed/rq1_ssl/rq1_ssl_engineered.csv

    This preprocessing is intentionally minimal. It only ensures the raw SSL export
    has the required fields for the RQ1 validation pipeline:
      - Returns (boolean)
      - total_loss (numeric magnitude proxy for profit erosion)
      - basic date parsing (if present)

    The RQ1 validation pipeline will then map:
      profit_erosion := total_loss
      category      := Pillar + "-" + Major Market Cat + "-" + Department
                        (concatenated in rq1_ssl_validation.py)

    The columns Pillar, Major Market Cat, and Department are passed through
    untouched by this preprocessing step and combined downstream.
    """

    # IO
    raw_path: Path = Path("data/raw/SSL_Returns_df_yoy.csv")
    out_dir: Path = Path("data/processed/rq1_ssl")
    out_stem: str = "rq1_ssl_engineered"

    # If present in raw, we will coerce. If missing, infer.
    returns_col: str = "Returns"

    # If present in raw, we will use. If missing, compute with fallback.
    total_loss_col: str = "total_loss"

    # Candidate columns to use as loss proxy if total_loss missing
    loss_candidates: Tuple[str, ...] = (
        "total_loss",
        "gross_financial_loss",
        "Gross Financial Loss",
        "gross_financial_loss_amt",
        "total_return_cogs",
        "Total Return COGS",
        "gross_financial_loss_total",
        "Gross Profit",
        "Product Cost",
        "CreditReturn Sales",
        "Credit Return Sales",
    )

    # Date columns we attempt to parse if present
    date_cols: Tuple[str, ...] = ("Booked Date", "Billed Date", "Reference Booked Date", "Prev_Return_Date")

    # Used to infer Returns if Returns column missing
    ordered_qty_col: str = "Ordered Qty"
    billed_qty_col: str = "Billed Qty"
    credit_sales_col: str = "CreditReturn Sales"


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.upper()
    return s.isin(["TRUE", "T", "1", "YES", "Y"])


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _infer_returns_flag(df: pd.DataFrame, cfg: RQ1SSLPreprocessConfig) -> pd.Series:
    """
    Conservative inference for Returns if raw does not contain a Returns flag:
      - if Billed Qty < 0 OR Ordered Qty < 0 OR CreditReturn Sales < 0 -> Returns=True
    """
    flags = pd.Series(False, index=df.index)

    if cfg.billed_qty_col in df.columns:
        flags = flags | (_to_numeric(df[cfg.billed_qty_col]) < 0)

    if cfg.ordered_qty_col in df.columns:
        flags = flags | (_to_numeric(df[cfg.ordered_qty_col]) < 0)

    if cfg.credit_sales_col in df.columns:
        flags = flags | (_to_numeric(df[cfg.credit_sales_col]) < 0)

    return flags


def _compute_total_loss(df: pd.DataFrame, cfg: RQ1SSLPreprocessConfig) -> Tuple[pd.Series, str]:
    """
    Build total_loss from the first available candidate column.
    Always uses abs() to represent magnitude.

    Returns:
      (series, source_description)
    """
    for col in cfg.loss_candidates:
        if col in df.columns:
            s = _to_numeric(df[col]).abs()
            return s, f"abs({col})"

    return pd.Series(np.nan, index=df.index), "missing (no loss candidates found)"


def preprocess_rq1_ssl_raw(raw_df: pd.DataFrame, cfg: Optional[RQ1SSLPreprocessConfig] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cfg = cfg or RQ1SSLPreprocessConfig()
    df = raw_df.copy()

    meta: Dict[str, str] = {}

    # Parse dates (safe)
    for c in cfg.date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            meta[f"parsed_date:{c}"] = "to_datetime(errors='coerce')"

    # Returns
    if cfg.returns_col in df.columns:
        df["Returns"] = _coerce_bool(df[cfg.returns_col])
        meta["Returns"] = f"coerced from {cfg.returns_col}"
    else:
        df["Returns"] = _infer_returns_flag(df, cfg)
        meta["Returns"] = "inferred from qty/sales sign (fallback)"

    # total_loss
    if cfg.total_loss_col in df.columns:
        df["total_loss"] = _to_numeric(df[cfg.total_loss_col]).abs()
        meta["total_loss"] = f"abs({cfg.total_loss_col})"
    else:
        df["total_loss"], source = _compute_total_loss(df, cfg)
        meta["total_loss"] = source

    # Drop missing total_loss
    before = len(df)
    df = df.dropna(subset=["total_loss"]).copy()
    meta["dropped_rows_missing_total_loss"] = str(before - len(df))

    return df, meta


def build_and_save_rq1_ssl_engineered(
    project_root: Path,
    cfg: Optional[RQ1SSLPreprocessConfig] = None,
    *,
    save_csv: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str], Path]:
    """
    One-call:
      - reads raw SSL csv from data/raw
      - preprocess minimal required fields
      - saves engineered parquet (and optional csv)
    """
    cfg = cfg or RQ1SSLPreprocessConfig()

    raw_path = project_root / cfg.raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw SSL file not found: {raw_path}")

    raw_df = pd.read_csv(raw_path)
    engineered_df, meta = preprocess_rq1_ssl_raw(raw_df, cfg)

    out_dir = project_root / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / f"{cfg.out_stem}.parquet"
    engineered_df.to_parquet(out_parquet, index=False)
    meta["saved_parquet"] = str(out_parquet)

    if save_csv:
        out_csv = out_dir / f"{cfg.out_stem}.csv"
        engineered_df.to_csv(out_csv, index=False)
        meta["saved_csv"] = str(out_csv)

    return engineered_df, meta, out_parquet