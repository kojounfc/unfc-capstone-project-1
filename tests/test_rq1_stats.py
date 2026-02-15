\
import numpy as np
import pandas as pd
import pytest

from src import rq1_stats


def test_run_factor_returns_na_when_insufficient_groups():
    df = pd.DataFrame({"category": ["A", "A", "A"], "profit_erosion": [1.0, 2.0, 3.0]})
    summary, posthoc = rq1_stats.run_factor(df, "category", value_col="profit_erosion", min_group_size=2)

    assert summary["test_used"] == "n/a"
    assert summary["reject_h0"] is False
    assert posthoc.empty


def test_run_factor_anova_path_via_monkeypatch(monkeypatch):
    # Force normality=True so we test ANOVA branch deterministically
    monkeypatch.setattr(rq1_stats, "_normality_ok", lambda *args, **kwargs: True)

    # Also stub Tukey to avoid optional dependency
    monkeypatch.setattr(rq1_stats, "_posthoc_tukey", lambda *args, **kwargs: pd.DataFrame({"ok": [1]}))

    df = pd.DataFrame(
        {
            "brand": ["A"] * 10 + ["B"] * 10,
            "profit_erosion": np.concatenate([np.random.default_rng(0).normal(10, 1, 10),
                                             np.random.default_rng(1).normal(12, 1, 10)]),
        }
    )
    summary, posthoc = rq1_stats.run_factor(df, "brand", value_col="profit_erosion", min_group_size=5)

    assert summary["test_used"] == "anova"
    assert summary["effect_metric"] == "eta_squared"
    assert isinstance(summary["p_value"], float)
    assert "ok" in posthoc.columns


def test_run_factor_kruskal_path_via_monkeypatch(monkeypatch):
    # Force normality=False so we test Kruskal branch deterministically
    monkeypatch.setattr(rq1_stats, "_normality_ok", lambda *args, **kwargs: False)

    # Stub Dunn to avoid optional dependency
    monkeypatch.setattr(rq1_stats, "_posthoc_dunn", lambda *args, **kwargs: pd.DataFrame({"ok": [1]}))

    df = pd.DataFrame(
        {
            "category": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
            "profit_erosion": np.concatenate(
                [
                    np.random.default_rng(0).lognormal(mean=2.0, sigma=0.2, size=8),
                    np.random.default_rng(1).lognormal(mean=2.2, sigma=0.2, size=8),
                    np.random.default_rng(2).lognormal(mean=2.4, sigma=0.2, size=8),
                ]
            ),
        }
    )
    summary, posthoc = rq1_stats.run_factor(df, "category", value_col="profit_erosion", min_group_size=5)

    assert summary["test_used"] == "kruskal"
    assert summary["effect_metric"] == "epsilon_squared"
    assert isinstance(summary["p_value"], float)
    assert "ok" in posthoc.columns
