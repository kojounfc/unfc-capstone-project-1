"""
RQ4 runner: end-to-end log-linear OLS pipeline for behavioral associations
with profit erosion, including SSL external validation.

Follows the same pattern as rq1_run.py and rq2_run.py.

Key outputs (reports/rq4/):
- rq4_thelook_coefficients.csv        Full OLS coefficient table (log-linear)
- rq4_thelook_coefficients_linear.csv Linear OLS coefficient table (robustness)
- rq4_ssl_coefficient_alignment.csv   3-predictor direction alignment
- rq4_ssl_coefficients.csv            SSL OLS coefficient table
- rq4_ssl_effect_size_comparison.csv  Effect size comparison table
- rq4_validation_summary.csv          Validation summary incl. diagnostics

Key outputs (figures/rq4/):
- rq4_target_distribution.png
- rq4_coefficient_plot.png
- rq4_residual_diagnostics.png
- rq4_qq_plot_comparison.png

SSL validation is skipped gracefully when SSL_RETURNS_CSV is not found.

Usage
-----
python -m src.rq4_run
"""

from __future__ import annotations

import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from statsmodels.stats.stattools import jarque_bera as _jb_test

from src.config import (  # type: ignore
    RQ4_TARGET_COL,
    RQ4_ALPHA,
    RQ4_COLLINEARITY_THRESHOLD,
    REPORTS_DIR,
    FIGURES_DIR,
)
from src.rq4_econometrics import (  # type: ignore
    load_rq4_data,
    screen_features,
    prepare_regression_data,
    fit_ols_robust,
    extract_coefficient_table,
    calculate_vif,
    run_diagnostics,
)
from src.rq4_visuals import (  # type: ignore
    plot_target_distribution,
    plot_coefficient_forest,
    plot_residual_diagnostics,
    plot_qq_comparison,
)
from src.rq4_ssl_validation import run_full_rq4_ssl_validation  # type: ignore
from src.rq4_validation import build_validation_summary  # type: ignore

logger = logging.getLogger(__name__)


def _ssl_path() -> Optional[Path]:
    try:
        from src.config import SSL_RETURNS_CSV  # type: ignore
        p = Path(SSL_RETURNS_CSV)
        return p if p.exists() else None
    except Exception:
        return None


@dataclass
class RQ4Artifacts:
    """Container for key RQ4 output artifact paths."""
    reports_dir: Path
    figures_dir: Path
    thelook_coefficients_csv: Path
    thelook_coefficients_linear_csv: Path
    validation_summary_csv: Path
    ssl_validated: bool = False
    ssl_coefficient_alignment_csv: Optional[Path] = None
    ssl_coefficients_csv: Optional[Path] = None
    ssl_effect_size_csv: Optional[Path] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _run_thelook(
    thelook_df: pd.DataFrame,
    rpt: Path,
    fig: Path,
) -> tuple:
    """Fit OLS, run diagnostics, save figures and coefficient CSVs. Returns (results_log, results_lin, reg_log, surviving_numeric, jb_lin, jb_log, diagnostics_log, vif_df)."""
    screen = screen_features(
        thelook_df,
        target_col=RQ4_TARGET_COL,
        alpha=RQ4_ALPHA,
        collinearity_threshold=RQ4_COLLINEARITY_THRESHOLD,
    )
    surviving_numeric = screen["surviving_numeric"]
    surviving_categorical = screen["surviving_categorical"]
    logger.info("Surviving numeric: %s", surviving_numeric)

    reg_data = prepare_regression_data(
        thelook_df,
        target_col=RQ4_TARGET_COL,
        numeric_features=surviving_numeric,
        categorical_features=surviving_categorical,
        log_transform=True,
    )
    log_target = f"log_{RQ4_TARGET_COL}"
    reg_log = reg_data.drop(columns=[RQ4_TARGET_COL])
    reg_lin = reg_data.drop(columns=[log_target])

    results_log = fit_ols_robust(reg_log, log_target)
    results_lin = fit_ols_robust(reg_lin, RQ4_TARGET_COL)
    logger.info("Log-linear R² = %.4f", results_log.rsquared)

    diagnostics_log = run_diagnostics(results_log)
    vif_df = calculate_vif(reg_log, target_col=log_target)
    jb_lin, _, _, _ = _jb_test(results_lin.resid)
    jb_log, _, _, _ = _jb_test(results_log.resid)

    # Figures
    fig_tgt = plot_target_distribution(thelook_df, fig)
    plt.close(fig_tgt)

    coef_log_full = extract_coefficient_table(results_log)
    coef_for_plot = coef_log_full.rename(columns={
        "feature": "Feature",
        "coefficient": "Coefficient",
        "p_value": "p-value",
        "ci_lower": "95% CI Lower",
        "ci_upper": "95% CI Upper",
    })
    fig_coef = plot_coefficient_forest(coef_for_plot, fig)
    plt.close(fig_coef)

    fig_resid = plot_residual_diagnostics(
        results_log, results_log.fittedvalues, results_log.resid, fig
    )
    plt.close(fig_resid)

    fig_qq = plot_qq_comparison(
        results_lin.resid, results_log.resid,
        float(jb_lin), float(jb_log), fig
    )
    plt.close(fig_qq)

    # Coefficient CSVs
    coef_log = extract_coefficient_table(results_log)
    coef_lin = extract_coefficient_table(results_lin)
    coef_log.to_csv(rpt / "rq4_thelook_coefficients.csv", index=False)
    coef_lin.to_csv(rpt / "rq4_thelook_coefficients_linear.csv", index=False)
    logger.info("Saved coefficient tables (%d rows)", len(coef_log))

    return results_log, results_lin, reg_log, surviving_numeric, float(jb_lin), float(jb_log), diagnostics_log, vif_df


def _run_ssl(
    results_log: Any,
    reg_log: pd.DataFrame,
    surviving_numeric: list,
    jb_lin: float,
    diagnostics_log: Any,
    vif_df: pd.DataFrame,
    rpt: Path,
    ssl_filepath: Path,
) -> bool:
    """Run SSL external validation and write artifacts. Returns True on success."""
    try:
        ssl_result = run_full_rq4_ssl_validation(
            thelook_results=results_log,
            thelook_data=reg_log,
            ssl_filepath=str(ssl_filepath),
            surviving_numeric=surviving_numeric,
        )

        coef_comparison = ssl_result["coefficient_comparison"]
        effect_size_result = ssl_result["effect_size_result"]
        ssl_coef_table = extract_coefficient_table(ssl_result["ssl_regression_results"])

        validation_summary = build_validation_summary(
            coef_comparison,
            effect_size_result,
            diagnostics_log=diagnostics_log,
            jb_linear_stat=jb_lin,
            vif_df=vif_df,
        )

        coef_comparison.to_csv(rpt / "rq4_ssl_coefficient_alignment.csv", index=False)
        ssl_coef_table.to_csv(rpt / "rq4_ssl_coefficients.csv", index=False)
        effect_size_result["effect_size_comparison"].to_csv(
            rpt / "rq4_ssl_effect_size_comparison.csv", index=False
        )
        validation_summary.to_csv(rpt / "rq4_validation_summary.csv", index=False)

        logger.info(
            "SSL validation complete — direction_aligned=%d/3, R²=%.4f",
            int(coef_comparison["direction_aligned"].sum()),
            ssl_result["ssl_regression_results"].rsquared,
        )
        return True
    except Exception as exc:
        logger.warning("SSL validation failed (non-fatal): %s", exc)
        return False


def run_rq4(
    *,
    reports_dir: Optional[Path] = None,
    figures_dir: Optional[Path] = None,
    ssl_filepath: Optional[Path] = None,
) -> RQ4Artifacts:
    """
    Run the full RQ4 pipeline and write all artifacts.

    Parameters
    ----------
    reports_dir : Path, optional
        Directory for CSV report artifacts. Defaults to reports/rq4/.
    figures_dir : Path, optional
        Directory for figure artifacts. Defaults to figures/rq4/.
    ssl_filepath : Path, optional
        Path to SSL CSV. If None, auto-detects from config.SSL_RETURNS_CSV.
        Pass Path('') to explicitly skip SSL validation.

    Returns
    -------
    RQ4Artifacts dataclass with paths to all written artifacts.
    """
    rpt = reports_dir or Path(REPORTS_DIR) / "rq4"
    fig = figures_dir or Path(FIGURES_DIR) / "rq4"
    rpt.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    logger.info("Loading RQ4 data")
    thelook_df = load_rq4_data()
    logger.info("Loaded %d customers", len(thelook_df))

    results_log, results_lin, reg_log, surviving_numeric, jb_lin, jb_log, diagnostics_log, vif_df = _run_thelook(
        thelook_df, rpt, fig
    )

    artifacts = RQ4Artifacts(
        reports_dir=rpt,
        figures_dir=fig,
        thelook_coefficients_csv=rpt / "rq4_thelook_coefficients.csv",
        thelook_coefficients_linear_csv=rpt / "rq4_thelook_coefficients_linear.csv",
        validation_summary_csv=rpt / "rq4_validation_summary.csv",
        diagnostics={"jb_linear": jb_lin, "jb_log": jb_log, "max_vif": float(vif_df["VIF"].max())},
    )

    _ssl = ssl_filepath if ssl_filepath is not None else _ssl_path()

    if _ssl and Path(_ssl).exists():
        logger.info("Running SSL external validation from %s", _ssl)
        ok = _run_ssl(
            results_log, reg_log, surviving_numeric,
            jb_lin, diagnostics_log, vif_df, rpt, _ssl
        )
        if ok:
            artifacts.ssl_validated = True
            artifacts.ssl_coefficient_alignment_csv = rpt / "rq4_ssl_coefficient_alignment.csv"
            artifacts.ssl_coefficients_csv = rpt / "rq4_ssl_coefficients.csv"
            artifacts.ssl_effect_size_csv = rpt / "rq4_ssl_effect_size_comparison.csv"
    else:
        logger.info(
            "SSL file not found — skipping external validation. "
            "Place file at config.SSL_RETURNS_CSV to enable."
        )

    return artifacts


def main() -> None:
    """
    CLI entry point for RQ4 artifact generation.

    Examples
    --------
    python -m src.rq4_run
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    artifacts = run_rq4()
    print("RQ4 artifacts written.")
    print(f"  Reports : {artifacts.reports_dir}")
    print(f"  Figures : {artifacts.figures_dir}")
    print(f"  TheLook coefficients : {artifacts.thelook_coefficients_csv}")
    print(f"  Validation summary   : {artifacts.validation_summary_csv}")
    if artifacts.ssl_validated:
        print(f"  SSL alignment        : {artifacts.ssl_coefficient_alignment_csv}")
    else:
        print("  SSL validation       : skipped (file not found)")


if __name__ == "__main__":
    main()
