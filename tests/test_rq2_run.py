import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import rq2_run


class TestRQ2Run:
    """Test cases for the RQ2 runner module."""

    def test_build_customer_erosion_filters_returns_and_sorts(self):
        item_df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "order_id": ["o1", "o1", "o2"],
                "order_item_id": ["oi1", "oi2", "oi3"],
                "item_status": ["Returned", "Complete", "Returned"],
                "order_status": ["Complete", "Complete", "Complete"],
                "sale_price": [100.0, 50.0, 200.0],
                "retail_price": [120.0, 60.0, 230.0],
                "cost": [40.0, 20.0, 100.0],
            }
        )

        out = rq2_run.build_customer_erosion(item_df)

        assert list(out["user_id"]) == ["u2", "u1"]
        assert len(out) == 2
        assert out["total_profit_erosion"].is_monotonic_decreasing
        assert (out["total_profit_erosion"] > 0).all()

    def test_build_customer_erosion_raw_path_engineers_features_and_drops_overlap(
        self, monkeypatch
    ):
        """Covers raw-data branch in build_customer_erosion."""
        calls = {"require_ctx": None, "engineer_called": 0, "margins_called": 0}

        def _mock_require_columns(df, required, context):
            calls["require_ctx"] = context
            for c in required:
                assert c in df.columns

        def _mock_engineer_return_features(df):
            calls["engineer_called"] += 1
            out = df.copy()
            out["is_returned_item"] = (out["item_status"] == "Returned").astype(int)
            return out

        def _mock_calculate_margins(df):
            calls["margins_called"] += 1
            out = df.copy()
            out["item_margin"] = out["sale_price"] - out["cost"]
            return out

        def _mock_calculate_profit_erosion(df, use_category_tiers=True):
            out = df.copy()
            out["profit_erosion"] = (out["retail_price"] - out["sale_price"]).clip(
                lower=0.0
            )
            return out

        def _mock_aggregate_profit_erosion_by_customer(df):
            agg = (
                df.groupby("user_id", as_index=False)["profit_erosion"]
                .sum()
                .rename(columns={"profit_erosion": "total_profit_erosion"})
            )
            # include overlap columns to ensure they get dropped by build_customer_erosion
            agg["total_sales"] = 999.0
            agg["total_margin"] = 888.0
            return agg

        monkeypatch.setattr(rq2_run, "_require_columns", _mock_require_columns)
        monkeypatch.setattr(
            rq2_run, "engineer_return_features", _mock_engineer_return_features
        )
        monkeypatch.setattr(rq2_run, "calculate_margins", _mock_calculate_margins)
        monkeypatch.setattr(
            rq2_run, "calculate_profit_erosion", _mock_calculate_profit_erosion
        )
        monkeypatch.setattr(
            rq2_run,
            "aggregate_profit_erosion_by_customer",
            _mock_aggregate_profit_erosion_by_customer,
        )

        item_df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u3"],
                "order_id": ["o1", "o1", "o2", "o3"],
                "order_item_id": ["oi1", "oi2", "oi3", "oi4"],
                "item_status": ["Returned", "Complete", "Returned", "Complete"],
                "order_status": ["Complete", "Complete", "Complete", "Complete"],
                "sale_price": [100.0, 50.0, 200.0, 10.0],
                "retail_price": [130.0, 60.0, 210.0, 15.0],
                "cost": [40.0, 20.0, 100.0, 5.0],
            }
        )

        out = rq2_run.build_customer_erosion(item_df)

        assert calls["require_ctx"] == "build_customer_erosion (raw data)"
        assert calls["engineer_called"] == 1
        assert calls["margins_called"] == 1

        # Only users with returned items remain (u1 and u2)
        assert set(out["user_id"]) == {"u1", "u2"}
        assert out["total_profit_erosion"].is_monotonic_decreasing

        # overlap columns dropped
        assert "total_sales" not in out.columns
        assert "total_margin" not in out.columns

    def test_build_customer_erosion_processed_path_skips_engineering_and_respects_item_margin(
        self, monkeypatch
    ):
        """Covers processed-data branch in build_customer_erosion."""
        calls = {"require_ctx": None, "engineer_called": 0, "margins_called": 0}

        def _mock_require_columns(df, required, context):
            calls["require_ctx"] = context
            for c in required:
                assert c in df.columns

        def _mock_engineer_return_features(df):
            calls["engineer_called"] += 1
            return df

        def _mock_calculate_margins(df):
            calls["margins_called"] += 1
            out = df.copy()
            out["item_margin"] = out["sale_price"] - out["cost"]
            return out

        def _mock_calculate_profit_erosion(df, use_category_tiers=True):
            out = df.copy()
            out["profit_erosion"] = (out["retail_price"] - out["sale_price"]).clip(
                lower=0.0
            )
            return out

        def _mock_aggregate_profit_erosion_by_customer(df):
            return (
                df.groupby("user_id", as_index=False)["profit_erosion"]
                .sum()
                .rename(columns={"profit_erosion": "total_profit_erosion"})
            )

        monkeypatch.setattr(rq2_run, "_require_columns", _mock_require_columns)
        monkeypatch.setattr(
            rq2_run, "engineer_return_features", _mock_engineer_return_features
        )
        monkeypatch.setattr(rq2_run, "calculate_margins", _mock_calculate_margins)
        monkeypatch.setattr(
            rq2_run, "calculate_profit_erosion", _mock_calculate_profit_erosion
        )
        monkeypatch.setattr(
            rq2_run,
            "aggregate_profit_erosion_by_customer",
            _mock_aggregate_profit_erosion_by_customer,
        )

        # Case A: item_margin missing => calculate_margins must run
        processed_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u2"],
                "order_id": ["o1", "o2", "o2"],
                "order_item_id": ["oi1", "oi2", "oi3"],
                "is_returned_item": [1, 0, 1],
                "sale_price": [100.0, 50.0, 120.0],
                "retail_price": [130.0, 60.0, 150.0],
                "cost": [40.0, 20.0, 60.0],
            }
        )

        out_a = rq2_run.build_customer_erosion(processed_df)

        assert calls["require_ctx"] == "build_customer_erosion (processed data)"
        assert calls["engineer_called"] == 0
        assert calls["margins_called"] == 1
        assert set(out_a["user_id"]) == {"u1", "u2"}

        # Case B: item_margin already present => calculate_margins should NOT run again
        calls["margins_called"] = 0
        processed_with_margin = processed_df.copy()
        processed_with_margin["item_margin"] = (
            processed_with_margin["sale_price"] - processed_with_margin["cost"]
        )

        out_b = rq2_run.build_customer_erosion(processed_with_margin)
        assert calls["margins_called"] == 0
        assert set(out_b["user_id"]) == {"u1", "u2"}

    def test_run_rq2_generates_plots_when_enabled_and_k_is_user_provided(
        self, monkeypatch, tmp_path
    ):
        """Covers make_plots=True branch and user-provided k branch in run_rq2."""
        item_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "order_id": ["o1", "o2"],
                "order_item_id": ["oi1", "oi2"],
                "item_status": ["Returned", "Returned"],
                "order_status": ["Complete", "Complete"],
                "sale_price": [100.0, 80.0],
                "retail_price": [120.0, 90.0],
                "cost": [40.0, 30.0],
                "item_created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        customer_behavior = pd.DataFrame(
            {"user_id": ["u1", "u2"], "order_frequency": [2, 1]}
        )
        customer_erosion = pd.DataFrame(
            {"user_id": ["u1", "u2"], "total_profit_erosion": [70.0, 30.0]}
        )
        pareto = pd.DataFrame({"customer_share": [0.5, 1.0], "value_share": [0.7, 1.0]})
        lorenz = pd.DataFrame(
            {"population_share": [0.0, 0.5, 1.0], "value_share": [0.0, 0.3, 1.0]}
        )
        seg_table = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "order_frequency": [2, 1],
                "total_sales": [200.0, 150.0],
                "total_profit_erosion": [70.0, 30.0],
            }
        )
        X_df = pd.DataFrame(
            {"order_frequency": [2.0, 1.0], "total_sales": [200.0, 150.0]}
        )
        X_scaled = np.array([[1.0, 1.0], [-1.0, -1.0]])
        elbow_df = pd.DataFrame({"k": [1, 2], "inertia": [2.0, 0.0]})
        silhouette_df = pd.DataFrame({"k": [2], "silhouette": [0.5]})
        cluster_summary = pd.DataFrame(
            {
                "cluster_id": [0, 1],
                "Count": [1, 1],
                "Total_Erosion": [70.0, 30.0],
                "Mean_Erosion": [70.0, 30.0],
                "Median_Erosion": [70.0, 30.0],
            }
        )

        # Mocks for pipeline steps (deterministic and fast)
        monkeypatch.setattr(rq2_run, "load_processed_data", lambda _: item_df.copy())
        monkeypatch.setattr(rq2_run, "engineer_return_features", lambda df: df.copy())
        monkeypatch.setattr(rq2_run, "calculate_margins", lambda df: df.copy())
        monkeypatch.setattr(
            rq2_run,
            "engineer_customer_behavioral_features",
            lambda df: customer_behavior.copy(),
        )
        monkeypatch.setattr(
            rq2_run, "build_customer_erosion", lambda df: customer_erosion.copy()
        )
        monkeypatch.setattr(
            rq2_run, "compute_pareto_table", lambda *_a, **_k: pareto.copy()
        )
        monkeypatch.setattr(
            rq2_run, "lorenz_curve_points", lambda *_a, **_k: lorenz.copy()
        )
        monkeypatch.setattr(rq2_run, "gini_coefficient", lambda *_a, **_k: 0.42)
        monkeypatch.setattr(
            rq2_run, "top_x_customer_share_of_value", lambda *_a, **_k: 0.7
        )
        monkeypatch.setattr(
            rq2_run,
            "bootstrap_gini_p_value",
            lambda *_a, **_k: {
                "observed_gini": 0.42,
                "null_mean_gini": 0.0,
                "p_value": 0.0,
                "n_bootstrap": 10,
            },
        )
        monkeypatch.setattr(
            rq2_run,
            "concentration_comparison",
            lambda *_a, **_k: {"gini_erosion": 0.42, "gini_baseline": 0.2},
        )
        monkeypatch.setattr(
            rq2_run,
            "build_customer_segmentation_table",
            lambda *_a, **_k: seg_table.copy(),
        )
        monkeypatch.setattr(
            rq2_run,
            "select_numeric_features",
            lambda *_a, **_k: (X_df.copy(), ["order_frequency", "total_sales"]),
        )
        monkeypatch.setattr(rq2_run, "standardize_features", lambda _x: X_scaled.copy())
        monkeypatch.setattr(
            rq2_run, "kmeans_fit_predict", lambda *_a, **_k: np.array([0, 1])
        )
        monkeypatch.setattr(
            rq2_run, "summarize_clusters", lambda *_a, **_k: cluster_summary.copy()
        )
        monkeypatch.setattr(
            rq2_run, "elbow_inertia_over_k", lambda *_a, **_k: elbow_df.copy()
        )
        monkeypatch.setattr(
            rq2_run, "silhouette_over_k", lambda *_a, **_k: silhouette_df.copy()
        )
        monkeypatch.setattr(
            rq2_run, "save_feature_engineered_dataset", lambda *_a, **_k: None
        )

        # Capture plot requests
        plot_names = []

        def _capture_plot(*_args, **kwargs):
            plot_names.append(Path(kwargs["out_path"]).name)

        monkeypatch.setattr(rq2_run, "_plot_line", _capture_plot)

        out_dir = tmp_path / "rq2"
        summary = rq2_run.run_rq2(
            out_dir=out_dir,
            k=2,
            k_min=2,
            k_max=2,
            top_x=0.2,
            make_plots=True,
        )

        assert summary.k_used == 2

        meta = json.loads((out_dir / "rq2_metadata.json").read_text(encoding="utf-8"))
        assert meta["k_selection_method"] == "user_provided"

        assert set(plot_names) == {
            "pareto_curve.png",
            "lorenz_curve.png",
            "elbow_inertia.png",
            "silhouette_scores.png",
        }

    def test_run_rq2_writes_summary_and_metadata(self, monkeypatch, tmp_path):
        item_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "order_id": ["o1", "o2"],
                "order_item_id": ["oi1", "oi2"],
                "item_status": ["Returned", "Complete"],
                "order_status": ["Complete", "Complete"],
                "sale_price": [100.0, 80.0],
                "retail_price": [120.0, 90.0],
                "cost": [40.0, 30.0],
                "item_created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            }
        )
        customer_behavior = pd.DataFrame(
            {"user_id": ["u1", "u2"], "order_frequency": [2, 1]}
        )
        customer_erosion = pd.DataFrame(
            {"user_id": ["u1", "u2"], "total_profit_erosion": [70.0, 30.0]}
        )
        pareto = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "total_profit_erosion": [70.0, 30.0],
                "rank": [1, 2],
                "customer_share": [0.5, 1.0],
                "cum_value": [70.0, 100.0],
                "value_share": [0.7, 1.0],
                "concentration_category": ["Useful Many", "Useful Many"],
            }
        )
        lorenz = pd.DataFrame(
            {"population_share": [0.0, 0.5, 1.0], "value_share": [0.0, 0.3, 1.0]}
        )
        seg_table = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "order_frequency": [2, 1],
                "total_sales": [200.0, 150.0],
                "total_profit_erosion": [70.0, 30.0],
            }
        )
        X_df = pd.DataFrame(
            {"order_frequency": [2.0, 1.0], "total_sales": [200.0, 150.0]}
        )
        X_scaled = np.array([[1.0, 1.0], [-1.0, -1.0]])
        elbow_df = pd.DataFrame({"k": [1, 2], "inertia": [2.0, 0.0]})
        silhouette_df = pd.DataFrame({"k": [2], "silhouette": [0.5]})

        # Mock cluster summary output
        cluster_summary = pd.DataFrame(
            {
                "cluster_id": [0, 1],
                "Count": [1, 1],
                "Total_Erosion": [70.0, 30.0],
                "Mean_Erosion": [70.0, 30.0],
                "Median_Erosion": [70.0, 30.0],
            }
        )

        saved_filenames = []

        monkeypatch.setattr(rq2_run, "load_processed_data", lambda _: item_df.copy())
        monkeypatch.setattr(
            rq2_run,
            "engineer_customer_behavioral_features",
            lambda df: customer_behavior,
        )
        monkeypatch.setattr(
            rq2_run, "build_customer_erosion", lambda df: customer_erosion.copy()
        )
        monkeypatch.setattr(
            rq2_run, "compute_pareto_table", lambda *_args, **_kw: pareto
        )
        monkeypatch.setattr(
            rq2_run, "lorenz_curve_points", lambda *_args, **_kw: lorenz
        )
        monkeypatch.setattr(rq2_run, "gini_coefficient", lambda *_args, **_kw: 0.42)
        monkeypatch.setattr(
            rq2_run, "top_x_customer_share_of_value", lambda *_args, **_kw: 0.7
        )
        monkeypatch.setattr(
            rq2_run,
            "bootstrap_gini_p_value",
            lambda *_args, **_kw: {
                "observed_gini": 0.42,
                "null_mean_gini": 0.0,
                "p_value": 0.0,
                "n_bootstrap": 1000,
            },
        )
        monkeypatch.setattr(
            rq2_run,
            "concentration_comparison",
            lambda *_args, **_kw: {"gini_erosion": 0.42, "gini_baseline": 0.2},
        )
        monkeypatch.setattr(
            rq2_run,
            "build_customer_segmentation_table",
            lambda *_args, **_kw: seg_table.copy(),
        )

        def _mock_select_numeric_features(*_args, **_kw):
            assert _kw.get("exclude_leakage_features") is True
            return X_df.copy(), ["order_frequency", "total_sales"]

        monkeypatch.setattr(
            rq2_run, "select_numeric_features", _mock_select_numeric_features
        )
        monkeypatch.setattr(rq2_run, "standardize_features", lambda _x: X_scaled.copy())
        monkeypatch.setattr(
            rq2_run, "kmeans_fit_predict", lambda *_args, **_kw: np.array([0, 1])
        )
        # Mock summarize_clusters (new function)
        monkeypatch.setattr(
            rq2_run, "summarize_clusters", lambda *_args, **_kw: cluster_summary.copy()
        )
        monkeypatch.setattr(
            rq2_run, "elbow_inertia_over_k", lambda *_args, **_kw: elbow_df.copy()
        )
        monkeypatch.setattr(
            rq2_run, "silhouette_over_k", lambda *_args, **_kw: silhouette_df.copy()
        )
        monkeypatch.setattr(
            rq2_run,
            "save_feature_engineered_dataset",
            lambda df, filename, output_dir, save_parquet, save_csv: (
                saved_filenames.append(filename)
            ),
        )
        monkeypatch.setattr(
            rq2_run,
            "_plot_line",
            lambda *_args, **_kw: (_ for _ in ()).throw(
                AssertionError("_plot_line should not be called when make_plots=False")
            ),
        )

        out_dir = tmp_path / "rq2"
        summary = rq2_run.run_rq2(
            out_dir=out_dir,
            k=None,
            k_min=2,
            k_max=2,
            top_x=0.2,
            make_plots=False,
        )

        assert summary.customers == 2
        assert summary.total_profit_erosion == 100.0
        assert summary.gini == 0.42
        assert summary.top_x == 0.2
        assert summary.top_x_share_of_erosion == 0.7
        assert summary.k_used == 2

        # Updated: now expects cluster_summary to be saved
        assert set(saved_filenames) == {
            "customer_erosion",
            "pareto_table",
            "lorenz_points",
            "clustered_customers",
            "cluster_summary",  # <- NEW: Added this
            "elbow_inertia",
            "silhouette_scores",
        }

        metadata_path = out_dir / "rq2_metadata.json"
        summary_path = out_dir / "rq2_summary.json"
        assert metadata_path.exists()
        assert summary_path.exists()

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
        assert metadata["feature_columns_used"] == ["order_frequency", "total_sales"]
        assert metadata["feature_policy"] == "behavioral_non_leakage_only"
        assert metadata["gini_bootstrap_test"]["observed_gini"] == 0.42
        assert metadata["concentration_comparison"]["gini_baseline"] == 0.2
        assert metadata["k_used"] == 2
        assert metadata["k_selection_method"] == "silhouette_argmax_tiebreak_lowest_k"
        assert summary_json["customers"] == 2
        assert summary_json["k_used"] == 2

    def test_parse_args_reads_cli_options(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--k",
                "5",
                "--k-min",
                "3",
                "--k-max",
                "8",
                "--top-x",
                "0.25",
                "--out-dir",
                "custom-out",
                "--no-plots",
            ],
        )
        args = rq2_run._parse_args()

        assert args.k == 5
        assert args.k_min == 3
        assert args.k_max == 8
        assert args.top_x == 0.25
        assert args.out_dir == "custom-out"
        assert args.no_plots is True

    def test_parse_args_defaults_to_auto_k(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["prog"])
        args = rq2_run._parse_args()
        assert args.k is None

    def test_main_prints_summary(self, monkeypatch, capsys):
        monkeypatch.setattr(
            rq2_run,
            "_parse_args",
            lambda: argparse.Namespace(
                input_parquet=None,
                out_dir="unused",
                k=None,
                k_min=2,
                k_max=10,
                top_x=0.2,
                no_plots=True,
            ),
        )
        monkeypatch.setattr(
            rq2_run,
            "run_rq2",
            lambda **_kw: rq2_run.RQ2Summary(
                customers=1,
                total_profit_erosion=10.0,
                gini=0.1,
                top_x=0.2,
                top_x_share_of_erosion=0.5,
                k_used=2,
            ),
        )

        rq2_run.main()
        out = capsys.readouterr().out

        assert "RQ2 complete." in out
        assert '"customers": 1' in out
