# src/rq1/rq1_stats.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:
    pairwise_tukeyhsd = None

try:
    import scikit_posthocs as sp
except Exception:
    sp = None

OUT_DIR = Path("reports/rq1")
ALPHA = 0.05
EFFECT_THRESHOLD = 0.06  # template success criterion


def _eta_squared(groups, all_values):
    overall_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups)
    ss_total = sum((x - overall_mean) ** 2 for x in all_values)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _epsilon_squared(H, n, k):
    denom = (n - k)
    return float((H - k + 1) / denom) if denom > 0 else 0.0


def _normality_ok(df, group_col, value_col):
    # Shapiro per group; if ANY group non-normal => use Kruskal
    for g, s in df.groupby(group_col)[value_col]:
        vals = s.dropna().values
        if len(vals) < 3:
            return False
        p = stats.shapiro(vals)[1]
        if p < ALPHA:
            return False
    return True


def _posthoc_tukey(df, group_col, value_col):
    if pairwise_tukeyhsd is None:
        raise ImportError("statsmodels is required for Tukey posthoc. pip install statsmodels")
    res = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=ALPHA)
    out = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return out


def _posthoc_dunn(df, group_col, value_col):
    if sp is None:
        raise ImportError("scikit-posthocs is required for Dunn posthoc. pip install scikit-posthocs")
    pvals = sp.posthoc_dunn(df, val_col=value_col, group_col=group_col, p_adjust="bonferroni")
    pvals.index.name = "group_a"
    pvals.columns.name = "group_b"
    return pvals.reset_index().melt(id_vars="group_a", var_name="group_b", value_name="p_adj")


def run_factor(df, group_col, value_col="profit_erosion", min_group_size=5):
    # drop tiny groups
    counts = df[group_col].value_counts()
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
            },
            pd.DataFrame(),
        )

    grouped = df2.groupby(group_col)[value_col].apply(lambda s: s.dropna().values)
    groups = list(grouped.values)
    all_values = np.concatenate(groups)

    normal = _normality_ok(df2, group_col, value_col)

    if normal:
        stat, p = stats.f_oneway(*groups)
        effect = _eta_squared(groups, all_values)
        test_used = "anova"
        effect_metric = "eta_squared"
        posthoc = _posthoc_tukey(df2, group_col, value_col)
    else:
        stat, p = stats.kruskal(*groups)
        effect = _epsilon_squared(stat, n=len(all_values), k=len(groups))
        test_used = "kruskal"
        effect_metric = "epsilon_squared"
        posthoc = _posthoc_dunn(df2, group_col, value_col)

    reject = p < ALPHA
    effect_ok = effect >= EFFECT_THRESHOLD
    success = bool(reject and effect_ok)

    summary = {
        "factor": group_col,
        "test_used": test_used,
        "p_value": float(p),
        "effect_size": float(effect),
        "effect_metric": effect_metric,
        "reject_h0": bool(reject),
        "meets_effect_threshold": bool(effect_ok),
        "success_criteria_met": bool(success),
        "n_groups": int(df2[group_col].nunique()),
        "n_rows": int(len(df2)),
    }

    return summary, posthoc


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(OUT_DIR / "rq1_returned_items.parquet")

    # Template factors: category + brand
    required = ["profit_erosion", "category", "brand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in returned_df: {missing}")

    summaries = []
    posthoc_outputs = {}

    for factor in ["category", "brand"]:
        summary, posthoc = run_factor(df, factor, value_col="profit_erosion", min_group_size=5)
        summaries.append(summary)
        posthoc_outputs[factor] = posthoc

        # export posthoc
        if not posthoc.empty:
            posthoc.to_csv(OUT_DIR / f"rq1_posthoc_{factor}.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "rq1_statistical_tests_summary.csv", index=False)

    print("RQ1 stats complete. Outputs:")
    print("- reports/rq1/rq1_statistical_tests_summary.csv")
    print("- reports/rq1/rq1_posthoc_category.csv (if available)")
    print("- reports/rq1/rq1_posthoc_brand.csv (if available)")


if __name__ == "__main__":
    main()