"""
Core artifact pipeline — runs all Research Questions in sequence.

Executes RQ1 → RQ2 → RQ3 → RQ4, writing artifacts to reports/ and figures/.

SCOPE
-----
This runner covers the *core* analytical outputs for each RQ:
  RQ1  Descriptive statistics, Kruskal-Wallis tests, erosion figures
  RQ2  K-Means clustering, Gini/Lorenz/Pareto concentration analysis
  RQ3  ML classification (RF/GB/LR), rule-based baseline, ablation study
  RQ4  Log-linear OLS regression + SSL coefficient alignment (if SSL file present)

NOT covered here (requires the master notebook):
  - RQ3 sensitivity analysis (processing cost $8–$18; threshold 50th–90th pct)
  - RQ3 SSL external validation (Level 1 pattern + Level 2 directional accuracy)
  - RQ2 SSL clustering validation
  - Full ETL pipeline from raw BigQuery exports
  → Run notebooks/profit_erosion_analysis.ipynb for complete robustness checks.

Usage
-----
python main.py                  # run full pipeline
python main.py --skip-ssl       # skip RQ4 SSL validation (when SSL file absent)
python main.py --rq RQ1 RQ3     # run only selected RQs

Notes
-----
- Requires the Conda/venv environment with all project dependencies (statsmodels,
  scikit-learn, etc.).  Run from the project root: python main.py
- SSL validation (RQ4) requires data/raw/SSL_Returns_df_yoy.csv (not tracked in git).
  It is skipped gracefully when the file is absent.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("main")


def _hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _run_rq1() -> bool:
    logger.info("=== RQ1: Category / Brand Profit Erosion Analysis ===")
    try:
        from src.rq1_run import run_rq1  # type: ignore
        summary = run_rq1()
        logger.info("RQ1 complete — status: %s", summary.get("status"))
        return True
    except Exception as exc:
        logger.error("RQ1 FAILED: %s", exc)
        return False


def _run_rq2() -> bool:
    logger.info("=== RQ2: Customer Segmentation & Concentration Analysis ===")
    try:
        from src.rq2_run import run_rq2  # type: ignore
        run_rq2()
        logger.info("RQ2 complete")
        return True
    except Exception as exc:
        logger.error("RQ2 FAILED: %s", exc)
        return False


def _run_rq3() -> bool:
    logger.info("=== RQ3: Predictive Modeling (RF / GB / LR + Rule-Based + Ablation) ===")
    try:
        from src.rq3_modeling import main as rq3_main  # type: ignore
        rq3_main()
        logger.info("RQ3 complete")
        return True
    except Exception as exc:
        logger.error("RQ3 FAILED: %s", exc)
        return False


def _run_rq4(skip_ssl: bool = False) -> bool:
    logger.info("=== RQ4: Log-Linear OLS Behavioral Association Analysis ===")
    try:
        from src.rq4_run import run_rq4  # type: ignore

        ssl_filepath = Path("") if skip_ssl else None  # None → auto-detect
        artifacts = run_rq4(ssl_filepath=ssl_filepath)

        logger.info("RQ4 complete — reports: %s", artifacts.reports_dir)
        if artifacts.ssl_validated:
            logger.info("  SSL validation: direction_aligned artifacts written")
        else:
            logger.info("  SSL validation: skipped (file not found or --skip-ssl)")
        return True
    except Exception as exc:
        logger.exception("RQ4 FAILED")
        return False


_RQ_RUNNERS = {
    "RQ1": _run_rq1,
    "RQ2": _run_rq2,
    "RQ3": _run_rq3,
    "RQ4": _run_rq4,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full profit-erosion analytics pipeline (RQ1–RQ4)."
    )
    parser.add_argument(
        "--rq",
        nargs="+",
        choices=list(_RQ_RUNNERS),
        default=list(_RQ_RUNNERS),
        metavar="RQ",
        help="Which research questions to run (default: all). E.g. --rq RQ1 RQ4",
    )
    parser.add_argument(
        "--skip-ssl",
        action="store_true",
        default=False,
        help="Skip SSL external validation (RQ4). Useful when SSL data file is absent.",
    )
    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("Profit Erosion Analytics Pipeline — full artifact generation")
    logger.info("=" * 60)
    logger.info("SCOPE: This runner generates core research artifacts only.")
    logger.info("  RQ1 — descriptive statistics, Kruskal-Wallis tests, erosion figures")
    logger.info("  RQ2 — K-Means clustering, Gini/Lorenz/Pareto concentration analysis")
    logger.info("  RQ3 — ML classification (RF/GB/LR), rule-based baseline, ablation study")
    logger.info("  RQ4 — log-linear OLS regression, SSL external validation")
    logger.info("NOT COVERED by this runner (requires the master notebook):")
    logger.info("  - RQ3 sensitivity analysis (cost $8-$18, threshold 50th-90th pct)")
    logger.info("  - RQ3 SSL external validation (Level 1 pattern + Level 2 directional)")
    logger.info("  - RQ2 SSL clustering validation")
    logger.info("  - Full end-to-end data pipeline (ETL from raw BigQuery exports)")
    logger.info("  For complete robustness checks, run: notebooks/profit_erosion_analysis.ipynb")
    logger.info("SSL note: RQ4 SSL validation requires data/raw/SSL_Returns_df_yoy.csv (not tracked).")
    logger.info("  Use --skip-ssl to suppress the warning when the file is absent.")
    logger.info("=" * 60)

    pipeline_start = time.time()
    results: dict[str, bool] = {}

    for rq in args.rq:
        t0 = time.time()
        runner = _RQ_RUNNERS[rq]
        # Pass skip_ssl only to RQ4
        if rq == "RQ4":
            ok = runner(skip_ssl=args.skip_ssl)  # type: ignore[call-arg]
        else:
            ok = runner()
        elapsed = time.time() - t0
        results[rq] = ok
        status = "OK" if ok else "FAILED"
        logger.info("%s finished in %s — %s", rq, _hms(elapsed), status)

    total = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"Pipeline complete — total time: {_hms(total)}")
    print("-" * 60)
    for rq, ok in results.items():
        icon = "[OK]" if ok else "[FAILED]"
        print(f"  {icon}  {rq}")
    print("=" * 60)

    failed = [rq for rq, ok in results.items() if not ok]
    if failed:
        logger.error("The following RQs failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
