"""
RQ1 statistical testing utilities.

This module runs group-level hypothesis tests (by category, brand, etc.) to assess whether
profit erosion differs materially across groups.

Testing logic:
- If per-group Shapiro-Wilk indicates normality, use one-way ANOVA + eta-squared.
- Otherwise, use Kruskal–Wallis + epsilon-squared.

Outputs are designed to be CI-safe and deterministic given the same input parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:  # pragma: no cover
    pairwise_tukeyhsd = None

try:
    import scikit_posthocs as sp
except Exception:  # pragma: no cover
    sp = None

ALPHA = 0.05
EFFECT_THRESHOLD = 0.06  # template success criterion (can be adjusted per rubric)


def _default_processed_dir() -> Path:
    """
    Resolve the default RQ1 processed output directory.

    Uses `src.config.PROCESSED_DATA_DIR` when available; otherwise falls back to
    `data/processed/rq1` relative to the current working directory.
    """
    try:
        from src.config import PROCESSED_DATA_DIR  # type: ignore

        return Path(PROCESSED_DATA_DIR) / "rq1"
    except Exception:
        return Path("data") / "processed" / "rq1"


def _eta_squared(groups: list[np.ndarray], all_values: np.ndarray) -> float:
    """
    Compute eta-squared effect size for one-way ANOVA.

    Parameters
    ----------
    groups:
        List of arrays containing the values for each group.
    all_values:
        Concatenated vector of all values across groups.

    Returns
    -------
    float
        Eta-squared (proportion of variance explained).
    """
    overall_mean = float(np.mean(all_values))
    ss_between = sum(len(g) * (float(np.mean(g)) - overall_mean) ** 2 for g in groups)
    ss_total = float(np.sum((all_values - overall_mean) ** 2))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _epsilon_squared(H: float, n: int, k: int) -> float:
    """
    Compute epsilon-squared effect size for Kruskal–Wallis.

    Parameters
    ----------
    H:
        Kruskal–Wallis H statistic.
    n:
        Total number of observations.
    k:
        Number of groups.

    Returns
    -------
    float
        Epsilon-squared effect size.
    """
    denom = (n - k)
    return float((H - k + 1) / denom) if denom > 0 else 0.0


def _normality_ok(df: pd.DataFrame, group_col: str, value_col: str) -> bool:
    """
    Check Shapiro-Wilk normality per group.

    If ANY group fails normality (p < ALPHA) or has insufficient samples (<3),
    we treat the overall factor as non-normal and use Kruskal–Wallis.

    Returns
    -------
    bool
        True if all groups pass normality, otherwise False.
    """
    for _, s in df.groupby(group_col)[value_col]:
        vals = s.dropna().values
        if len(vals) < 3:
            return False
        p = float(stats.shapiro(vals)[1])
        if p < ALPHA:
            return False
    return True


def _posthoc_tukey(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc test after ANOVA.

    Requires statsmodels. Raises ImportError if unavailable.
    """
    if pairwise_tukeyhsd is None:  # pragma: no cover
        raise ImportError("statsmodels is required for Tukey posthoc. pip install statsmodels")
    res = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=ALPHA)
    return pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])


def _posthoc_dunn(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Run Dunn post-hoc test after Kruskal–Wallis with Bonferroni correction.

    Requires scikit-posthocs. Raises ImportError if unavailable.
    """
    if sp is None:  # pragma: no cover
        raise ImportError("scikit-posthocs is required for Dunn posthoc. pip install scikit-posthocs")
    pvals = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust="bonferroni")
    pvals.index.name = "group_a"
    pvals.columns.name = "group_b"
    return pvals.reset_index().melt(id_vars="group_a", var_name="group_b", value_name="p_adj")


def run_factor(
    df: pd.DataFrame,
    group_col: str,
    *,
    value_col: str = "profit_erosion",
    min_group_size: int = 5,
    run_posthoc: bool = True,
    max_groups_posthoc: int = 12,
    max_rows_per_group_posthoc: int = 2000,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Run a statistical group comparison test for a single categorical factor
    (e.g., product category or brand) against a numeric outcome variable.

    This function evaluates whether the distribution of the outcome variable
    differs significantly across groups defined by `group_col`. It automatically
    selects an appropriate statistical test based on distributional assumptions:

    - If normality assumptions are satisfied, a one-way ANOVA is applied,
      followed by Tukey HSD post-hoc comparisons (when enabled).
    - If normality assumptions are violated, a Kruskal–Wallis test is applied,
      followed by Dunn’s post-hoc comparisons (when enabled).

    To ensure CI-safety and practical runtimes on real-world datasets, guardrails
    are applied to post-hoc testing:
    - Post-hoc tests may be disabled entirely via `run_posthoc=False`.
    - Dunn post-hoc comparisons are limited to the largest groups only.
    - Optional per-group sampling is applied to prevent excessive computation.

    The function is deterministic given the same input dataset and parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the grouping column (`group_col`)
        and the numeric outcome column (`value_col`).
    group_col : str
        Name of the categorical column defining comparison groups
        (e.g., 'category', 'brand').
    value_col : str, default 'profit_erosion'
        Name of the numeric outcome column to be tested.
    min_group_size : int, default 5
        Minimum number of observations required per group. Groups with fewer
        observations are excluded from the analysis.
    run_posthoc : bool, default True
        Whether to run post-hoc pairwise comparisons after the global test.
    max_groups_posthoc : int, default 12
        Maximum number of groups included in Dunn post-hoc testing.
        The largest groups by sample size are selected.
    max_rows_per_group_posthoc : int, default 2000
        Maximum number of observations per group used in Dunn post-hoc testing.
        Larger groups are downsampled deterministically.

    Returns
    -------
    (summary_dict, posthoc_df) : Tuple[Dict[str, object], pd.DataFrame]
        summary_dict:
            Dictionary containing:
            - factor
            - test_used
            - p_value
            - effect_size
            - effect_metric
            - reject_h0
            - meets_effect_threshold
            - success_criteria_met
            - n_groups
            - n_rows
            - posthoc_ran
            - posthoc_note
        posthoc_df:
            DataFrame of post-hoc pairwise comparisons (Tukey HSD or Dunn test).
            Empty if post-hoc testing is not applicable or disabled.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If `group_col` is None or not a non-empty string.
    KeyError
        If required columns are missing from the dataframe.
    """

    # ------------------------------------------------------------------
    # Input validation (fail fast with explicit errors)
    # ------------------------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"run_factor(): df must be a pandas DataFrame. Got: {type(df)}"
        )

    if group_col is None:
        raise ValueError(
            "run_factor(): group_col is None. Expected a column name such as "
            "'category' or 'brand'."
        )

    if not isinstance(group_col, str) or not group_col.strip():
        raise ValueError(
            f"run_factor(): group_col must be a non-empty string. Got: {group_col!r}"
        )

    if group_col not in df.columns:
        raise KeyError(
            f"run_factor(): missing grouping column {group_col!r}. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    if value_col is None or value_col not in df.columns:
        raise KeyError(
            f"run_factor(): missing value column {value_col!r}. "
            f"Available columns: {list(df.columns)[:30]}..."
        )

    # ------------------------------------------------------------------
    # Filter groups by minimum size
    # ------------------------------------------------------------------
    counts = df[group_col].value_counts(dropna=False)
    keep = counts[counts >= min_group_size].index
    df2 = df[df[group_col].isin(keep)].copy()

    if df2[group_col].nunique() < 2:
        return (
            {
                "factor": group_col,
                "test_used": "n/a",
                "p_value": np.nan,
                "effect_size": np.nan,
                "effect_metric": "n/a",
                "reject_h0": False,
                "meets_effect_threshold": False,
                "success_criteria_met": False,
                "n_groups": int(df2[group_col].nunique()),
                "n_rows": int(len(df2)),
                "posthoc_ran": False,
                "posthoc_note": "Not enough groups after filtering.",
            },
            pd.DataFrame(),
        )

    # ------------------------------------------------------------------
    # Prepare grouped values
    # ------------------------------------------------------------------
    grouped = df2.groupby(group_col)[value_col].apply(
        lambda s: s.dropna().values
    )
    groups = [np.asarray(v) for v in grouped.values]
    all_values = np.concatenate(groups)

    # ------------------------------------------------------------------
    # Select statistical test based on normality
    # ------------------------------------------------------------------
    normal = _normality_ok(df2, group_col, value_col)

    posthoc_df = pd.DataFrame()
    posthoc_ran = False
    posthoc_note = ""

    if normal:
        stat, p = stats.f_oneway(*groups)
        effect = _eta_squared(groups, all_values)
        test_used = "anova"
        effect_metric = "eta_squared"

        if run_posthoc:
            posthoc_df = _posthoc_tukey(df2, group_col, value_col)
            posthoc_ran = True
        else:
            posthoc_note = "Post-hoc testing disabled by configuration."
    else:
        stat, p = stats.kruskal(*groups)
        effect = _epsilon_squared(
            float(stat), n=len(all_values), k=len(groups)
        )
        test_used = "kruskal"
        effect_metric = "epsilon_squared"

        if run_posthoc:
            group_sizes = df2[group_col].value_counts()
            selected_groups = list(group_sizes.head(max_groups_posthoc).index)

            df_posthoc = df2[df2[group_col].isin(selected_groups)].copy()

            if max_rows_per_group_posthoc is not None:
                df_posthoc = (
                    df_posthoc.groupby(group_col, group_keys=False)
                    .apply(
                        lambda g: g.sample(
                            n=min(len(g), max_rows_per_group_posthoc),
                            random_state=42,
                        )
                    )
                )

            posthoc_df = _posthoc_dunn(df_posthoc, group_col, value_col)
            posthoc_ran = True
            posthoc_note = (
                f"Dunn post-hoc ran on top {len(selected_groups)} groups "
                f"with up to {max_rows_per_group_posthoc} rows per group."
            )
        else:
            posthoc_note = "Post-hoc testing disabled by configuration."

    # ------------------------------------------------------------------
    # Assemble results
    # ------------------------------------------------------------------
    reject = bool(p < ALPHA)
    effect_ok = bool(effect >= EFFECT_THRESHOLD)
    success = bool(reject and effect_ok)

    summary = {
        "factor": group_col,
        "test_used": test_used,
        "p_value": float(p),
        "effect_size": float(effect),
        "effect_metric": effect_metric,
        "reject_h0": reject,
        "meets_effect_threshold": effect_ok,
        "success_criteria_met": success,
        "n_groups": int(df2[group_col].nunique()),
        "n_rows": int(len(df2)),
        "posthoc_ran": posthoc_ran,
        "posthoc_note": posthoc_note,
    }

    return summary, posthoc_df




def main() -> None:
    """
    CLI entrypoint for running RQ1 hypothesis tests.

    Expects:
    - data/processed/rq1/rq1_returned_items.parquet (fallback)
      or src.config.PROCESSED_DATA_DIR/rq1/rq1_returned_items.parquet

    Writes:
    - rq1_statistical_tests_summary.csv
    - rq1_posthoc_category.csv (if available)
    - rq1_posthoc_brand.csv (if available)
    """
    out_dir = _default_processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(out_dir / "rq1_returned_items.parquet")

    required = ["profit_erosion", "category", "brand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in returned-items parquet: {missing}")

    summaries = []
    for factor in ["category", "brand"]:
        summary, posthoc = run_factor(df, factor, value_col="profit_erosion", min_group_size=5)
        summaries.append(summary)
        if not posthoc.empty:
            posthoc.to_csv(out_dir / f"rq1_posthoc_{factor}.csv", index=False)

    pd.DataFrame(summaries).to_csv(out_dir / "rq1_statistical_tests_summary.csv", index=False)
    print("RQ1 stats complete. Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
