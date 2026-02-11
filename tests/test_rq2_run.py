import argparse
import json

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
        X_df = pd.DataFrame({"order_frequency": [2.0, 1.0], "total_sales": [200.0, 150.0]})
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

        monkeypatch.setattr(rq2_run, "select_numeric_features", _mock_select_numeric_features)
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
