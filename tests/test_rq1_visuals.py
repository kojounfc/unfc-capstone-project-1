from pathlib import Path
import pandas as pd
import numpy as np

from src import rq1_visuals


# ------------------------------------------------------------
# Helper: basic synthetic dataset
# ------------------------------------------------------------

def _sample_group_df():
    return pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "total_profit_erosion": [100.0, 50.0, 10.0],
            "returned_items": [20, 10, 5],
            "avg_profit_erosion": [5.0, 5.0, 2.0],
            "return_rate": [0.2, 0.1, 0.05],
        }
    )


def _sample_item_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "category": np.random.choice(["A", "B"], 100),
            "brand": np.random.choice(["X", "Y"], 100),
            "profit_erosion": np.random.exponential(scale=10, size=100),
        }
    )


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def test_plot_top_groups_saves_file(tmp_path):
    out_path = tmp_path / "figures" / "rq1" / "fig_test.png"
    df = _sample_group_df()

    saved = rq1_visuals.plot_top_groups_total_erosion(
        df,
        group_col="category",
        value_col="total_profit_erosion",
        out_path=out_path,
        title="Test Plot",
        top_n=2,
    )

    assert saved.exists()
    assert saved.suffix == ".png"


def test_plot_return_rate_vs_mean_saves_file(tmp_path):
    out_path = tmp_path / "figures" / "rq1" / "scatter.png"
    df = _sample_group_df()

    saved = rq1_visuals.plot_return_rate_vs_mean_erosion(
        df,
        out_path=out_path,
    )

    assert saved.exists()
    assert saved.suffix == ".png"


def test_plot_severity_vs_volume_saves_file(tmp_path):
    out_path = tmp_path / "figures" / "rq1" / "decomposition.png"
    df = _sample_group_df()

    saved = rq1_visuals.plot_severity_vs_volume_decomposition(
        df,
        group_col="category",
        returned_items_col="returned_items",
        avg_erosion_col="avg_profit_erosion",
        total_erosion_col="total_profit_erosion",
        out_path=out_path,
    )

    assert saved.exists()
    assert saved.suffix == ".png"


def test_plot_distribution_log_saves_file(tmp_path):
    out_path = tmp_path / "figures" / "rq1" / "distribution.png"
    df = _sample_item_df()

    saved = rq1_visuals.plot_profit_erosion_distribution_log(
        df,
        value_col="profit_erosion",
        out_path=out_path,
    )

    assert saved.exists()
    assert saved.suffix == ".png"


def test_plot_bootstrap_ci_returns_dataframe_and_saves_file(tmp_path):
    out_path = tmp_path / "figures" / "rq1" / "ci.png"
    df = _sample_item_df()

    ci_df, saved = rq1_visuals.plot_bootstrap_ci_mean_by_group(
        df,
        group_col="category",
        value_col="profit_erosion",
        out_path=out_path,
        n_boot=50,  # small for fast test
    )

    assert isinstance(ci_df, pd.DataFrame)
    assert saved.exists()
    assert saved.suffix == ".png"
