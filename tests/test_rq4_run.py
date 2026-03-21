"""
Tests for src/rq4_run.py — RQ4 pipeline runner.

Uses synthetic customer data to exercise the run_rq4() pipeline without
touching real data files. SSL validation is always skipped (no SSL file
present in CI).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rq4_run import RQ4Artifacts, run_rq4  # type: ignore


# ---------------------------------------------------------------------------
# Synthetic customer dataset fixture
# ---------------------------------------------------------------------------

def _make_customer_df(n: int = 300) -> pd.DataFrame:
    """Build a minimal customer DataFrame that passes rq4_econometrics assertions."""
    rng = np.random.default_rng(42)
    base = rng.exponential(scale=50, size=n)
    df = pd.DataFrame(
        {
            "total_profit_erosion": base + 1.0,          # strictly positive
            "return_frequency": rng.integers(1, 10, size=n).astype(float),
            "avg_order_value": rng.uniform(20, 200, size=n),
            "avg_basket_size": rng.uniform(1, 10, size=n),
            "customer_return_rate": rng.uniform(0.1, 1.0, size=n),
            "purchase_recency_days": rng.integers(1, 365, size=n).astype(float),
            "customer_tenure_days": rng.integers(30, 1000, size=n).astype(float),
            "avg_item_margin": rng.uniform(5, 80, size=n),
            "user_gender": rng.choice(["M", "F"], size=n),
            "traffic_source": rng.choice(["Search", "Email", "Organic"], size=n),
            "dominant_return_category": rng.choice(
                ["Jeans", "Tops & Tees", "Outerwear", "Dresses"], size=n
            ),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRQ4Artifacts:
    def test_dataclass_defaults(self, tmp_path):
        art = RQ4Artifacts(
            reports_dir=tmp_path,
            figures_dir=tmp_path,
            thelook_coefficients_csv=tmp_path / "coef.csv",
            thelook_coefficients_linear_csv=tmp_path / "coef_lin.csv",
            validation_summary_csv=tmp_path / "val.csv",
        )
        assert art.ssl_validated is False
        assert art.ssl_coefficient_alignment_csv is None
        assert art.ssl_coefficients_csv is None
        assert art.ssl_effect_size_csv is None
        assert isinstance(art.diagnostics, dict)


class TestRunRQ4:
    def test_returns_rq4artifacts_instance(self, tmp_path):
        """run_rq4() must return an RQ4Artifacts instance."""
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),  # explicit skip
            )

        assert isinstance(art, RQ4Artifacts)

    def test_coefficient_csv_written(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        assert art.thelook_coefficients_csv.exists(), "Log-linear coefficient CSV not written"

    def test_linear_coefficient_csv_written(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        assert art.thelook_coefficients_linear_csv.exists(), "Linear coefficient CSV not written"

    def test_coefficient_csv_has_expected_columns(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        df = pd.read_csv(art.thelook_coefficients_csv)
        expected = {"feature", "coefficient", "std_error", "t_stat", "p_value", "ci_lower", "ci_upper"}
        assert expected.issubset(set(df.columns))

    def test_diagnostics_populated(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        assert "jb_linear" in art.diagnostics
        assert "jb_log" in art.diagnostics
        assert "max_vif" in art.diagnostics
        # VIF must be finite and positive
        assert np.isfinite(art.diagnostics["max_vif"])
        assert art.diagnostics["max_vif"] > 0

    def test_ssl_skipped_when_empty_path(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        assert art.ssl_validated is False
        assert art.ssl_coefficient_alignment_csv is None

    def test_ssl_skipped_when_file_missing(self, tmp_path):
        customer_df = _make_customer_df()

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=tmp_path / "figures",
                ssl_filepath=tmp_path / "nonexistent_ssl.csv",
            )

        assert art.ssl_validated is False

    def test_reports_dir_created(self, tmp_path):
        customer_df = _make_customer_df()
        rpt = tmp_path / "deep" / "nested" / "rq4"

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=rpt,
                figures_dir=tmp_path / "figures",
                ssl_filepath=Path(""),
            )

        assert rpt.exists()

    def test_figures_dir_created(self, tmp_path):
        customer_df = _make_customer_df()
        fig = tmp_path / "deep" / "nested" / "figs"

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            run_rq4(
                reports_dir=tmp_path / "reports",
                figures_dir=fig,
                ssl_filepath=Path(""),
            )

        assert fig.exists()

    def test_artifact_paths_match_reports_dir(self, tmp_path):
        customer_df = _make_customer_df()
        rpt = tmp_path / "rpt"

        with patch("src.rq4_run.load_rq4_data", return_value=customer_df):
            art = run_rq4(
                reports_dir=rpt,
                figures_dir=tmp_path / "figs",
                ssl_filepath=Path(""),
            )

        assert art.thelook_coefficients_csv.parent == rpt
        assert art.thelook_coefficients_linear_csv.parent == rpt
