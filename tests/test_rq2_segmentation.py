import numpy as np
import pandas as pd
import pytest

from src.rq2_segmentation import (
    DEFAULT_SEGMENTATION_FEATURES,
    build_customer_segmentation_table,
    clustering_metrics_over_k,
    combined_diagnostics,
    compute_clustering_quality_metrics,
    elbow_inertia_over_k,
    kmeans_fit_predict,
    select_numeric_features,
    silhouette_over_k,
    standardize_features,
    summarize_clusters,
    validate_clustering_matrix,
)


class TestRQ2Segmentation:
    """Test cases for RQ2 customer segmentation utilities."""

    def test_build_customer_segmentation_table_left_join_and_fill_zeros(self):
        behavior = pd.DataFrame(
            {
                "user_id": ["100", "200", "300"],
                "order_frequency": [2, 1, 3],
                "customer_return_rate": [0.5, 0.0, 0.25],
            }
        )
        erosion = pd.DataFrame(
            {
                "user_id": ["100", "300"],
                "total_profit_erosion": [127.6, 50.0],
                "returned_items": [2, 1],
            }
        )

        out = build_customer_segmentation_table(behavior, erosion)
        assert len(out) == 3
        # Customer 200 had no erosion row => filled with 0
        cust200 = out[out["user_id"] == "200"].iloc[0]
        assert cust200["total_profit_erosion"] == 0.0
        assert cust200["returned_items"] == 0.0

    def test_build_customer_segmentation_table_rejects_overlapping_columns(self):
        behavior = pd.DataFrame(
            {
                "user_id": ["100", "200"],
                "order_frequency": [2, 1],
                "total_sales": [100.0, 200.0],
            }
        )
        erosion = pd.DataFrame(
            {
                "user_id": ["100", "200"],
                "total_sales": [10.0, 20.0],
                "total_profit_erosion": [4.0, 5.0],
            }
        )

        with pytest.raises(ValueError, match="overlap"):
            build_customer_segmentation_table(behavior, erosion)

    def test_build_customer_segmentation_table_has_no_merge_suffix_columns(self):
        behavior = pd.DataFrame(
            {
                "user_id": ["100", "200", "300"],
                "order_frequency": [2, 1, 3],
            }
        )
        erosion = pd.DataFrame(
            {
                "user_id": ["100", "300"],
                "total_profit_erosion": [127.6, 50.0],
                "returned_items": [2, 1],
            }
        )

        out = build_customer_segmentation_table(behavior, erosion)
        assert not any(col.endswith("_x") or col.endswith("_y") for col in out.columns)

    def test_select_numeric_features_default_excludes_id(self):
        df = pd.DataFrame(
            {
                "user_id": ["1", "2"],
                "order_frequency": [2, 1],
                "customer_return_rate": [0.5, 0.0],
                "total_profit_erosion": [20.0, 10.0],
                "segment": ["A", "B"],  # non-numeric should be ignored
            }
        )
        X, cols = select_numeric_features(df)
        assert "user_id" not in cols
        assert set(cols) == {"order_frequency", "customer_return_rate"}
        assert X.shape == (2, 2)

    def test_select_numeric_features_prefers_behavioral_default_set(self):
        df = pd.DataFrame(
            {
                "user_id": ["1", "2"],
                "total_items_purchased": [3, 4],
                "avg_order_value": [50.0, 75.0],
                "customer_return_rate": [0.2, 0.1],
                "total_sales": [150.0, 300.0],
                "total_profit_erosion": [25.0, 10.0],
            }
        )
        _, cols = select_numeric_features(df)
        assert cols == [
            c for c in DEFAULT_SEGMENTATION_FEATURES if c in df.columns
        ]
        assert "total_profit_erosion" not in cols

    def test_select_numeric_features_rejects_explicit_leakage_columns(self):
        df = pd.DataFrame(
            {
                "user_id": ["1", "2"],
                "order_frequency": [1, 2],
                "total_profit_erosion": [30.0, 40.0],
            }
        )

        with pytest.raises(ValueError, match="leakage"):
            select_numeric_features(
                df,
                feature_cols=["order_frequency", "total_profit_erosion"],
                exclude_leakage_features=True,
            )

    def test_select_numeric_features_excludes_pattern_leakage_columns(self):
        df = pd.DataFrame(
            {
                "user_id": ["1", "2"],
                "order_frequency": [1, 2],
                "erosion_score": [0.1, 0.2],
                "is_high_erosion_customer": [0, 1],
            }
        )
        X, cols = select_numeric_features(df)
        assert cols == ["order_frequency"]
        assert list(X.columns) == ["order_frequency"]

    def test_standardize_features_shape_and_values_finite(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        Xs = standardize_features(X)
        assert Xs.shape == (3, 2)
        assert np.isfinite(Xs).all()

    def test_validate_clustering_matrix_raises_on_non_finite_values(self):
        X = pd.DataFrame({"a": [1.0, np.inf], "b": [2.0, 3.0]})
        with pytest.raises(ValueError, match="NaN or infinite"):
            validate_clustering_matrix(X)

    def test_kmeans_fit_predict_is_deterministic_with_random_state(self):
        X = pd.DataFrame({"a": [0.0, 0.1, 10.0, 10.1], "b": [0.0, 0.2, 9.9, 10.2]})
        Xs = standardize_features(X)
        labels1 = kmeans_fit_predict(Xs, k=2, random_state=42)
        labels2 = kmeans_fit_predict(Xs, k=2, random_state=42)
        assert np.array_equal(labels1, labels2)
        assert set(labels1) == {0, 1}

    def test_elbow_inertia_over_k_returns_expected_columns(self):
        X = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
        Xs = standardize_features(X)
        out = elbow_inertia_over_k(Xs, k_list=[1, 2, 3], random_state=42)
        assert list(out.columns) == ["k", "inertia"]
        assert len(out) == 3

    def test_silhouette_over_k_returns_expected_columns(self):
        X = pd.DataFrame({"a": [0.0, 0.1, 10.0, 10.1], "b": [0.0, 0.2, 9.9, 10.2]})
        Xs = standardize_features(X)
        out = silhouette_over_k(Xs, k_list=[2, 3], random_state=42)
        assert list(out.columns) == ["k", "silhouette"]
        assert len(out) == 2

    def test_combined_diagnostics_returns_expected_shapes(self):
        X = pd.DataFrame({"a": [0.0, 0.1, 10.0, 10.1], "b": [0.0, 0.2, 9.9, 10.2]})
        Xs = standardize_features(X)
        elbow_df, silhouette_df = combined_diagnostics(
            Xs, k_list=[1, 2, 3], random_state=42
        )

        assert list(elbow_df.columns) == ["k", "inertia"]
        assert list(silhouette_df.columns) == ["k", "silhouette"]
        assert len(elbow_df) == 3
        assert len(silhouette_df) == 2
        assert silhouette_df["k"].tolist() == [2, 3]

    def test_combined_diagnostics_validates_k_values(self):
        X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
        Xs = standardize_features(X)

        with pytest.raises(ValueError):
            combined_diagnostics(Xs, k_list=[0, 1], random_state=42)

    def test_clustering_metrics_over_k_returns_expected_columns(self):
        X = pd.DataFrame({"a": [0.0, 0.2, 10.0, 10.2], "b": [0.0, 0.1, 9.9, 10.1]})
        Xs = standardize_features(X)

        out = clustering_metrics_over_k(Xs, k_list=[1, 2, 3], random_state=42)

        assert list(out.columns) == [
            "k",
            "inertia",
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        ]
        assert len(out) == 3

        k1 = out[out["k"] == 1].iloc[0]
        assert np.isnan(k1["silhouette"])
        assert np.isnan(k1["calinski_harabasz"])
        assert np.isnan(k1["davies_bouldin"])

        k2 = out[out["k"] == 2].iloc[0]
        assert np.isfinite(k2["silhouette"])
        assert np.isfinite(k2["calinski_harabasz"])
        assert np.isfinite(k2["davies_bouldin"])

    def test_clustering_metrics_over_k_validates_k_values(self):
        X = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
        Xs = standardize_features(X)

        with pytest.raises(ValueError):
            clustering_metrics_over_k(Xs, k_list=[0, 1], random_state=42)

    def test_compute_clustering_quality_metrics_for_valid_labels(self):
        X = pd.DataFrame({"a": [0.0, 0.2, 10.0, 10.2], "b": [0.0, 0.1, 9.9, 10.1]})
        Xs = standardize_features(X)
        labels = kmeans_fit_predict(Xs, k=2, random_state=42)

        metrics = compute_clustering_quality_metrics(Xs, labels)

        assert set(metrics.keys()) == {
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
        }
        assert np.isfinite(metrics["silhouette"])
        assert np.isfinite(metrics["calinski_harabasz"])
        assert np.isfinite(metrics["davies_bouldin"])

    def test_compute_clustering_quality_metrics_single_cluster_returns_nan(self):
        X = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
        Xs = standardize_features(X)
        labels = np.zeros(Xs.shape[0], dtype=int)

        metrics = compute_clustering_quality_metrics(Xs, labels)

        assert np.isnan(metrics["silhouette"])
        assert np.isnan(metrics["calinski_harabasz"])
        assert np.isnan(metrics["davies_bouldin"])

    def test_summarize_clusters_basic_aggregation(self):
        """Test that summarize_clusters correctly aggregates by cluster."""
        clustered_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3", "u4"],
                "cluster_id": [0, 0, 1, 1],
                "total_profit_erosion": [100.0, 80.0, 50.0, 30.0],
            }
        )

        summary = summarize_clusters(clustered_df)

        assert len(summary) == 2
        assert list(summary.columns) == [
            "cluster_id",
            "Count",
            "Total_Erosion",
            "Mean_Erosion",
            "Median_Erosion",
        ]

        # Cluster 0: 2 customers with erosion [100, 80]
        cluster_0 = summary[summary["cluster_id"] == 0].iloc[0]
        assert cluster_0["Count"] == 2
        assert cluster_0["Total_Erosion"] == 180.0
        assert cluster_0["Mean_Erosion"] == 90.0
        assert cluster_0["Median_Erosion"] == 90.0

        # Cluster 1: 2 customers with erosion [50, 30]
        cluster_1 = summary[summary["cluster_id"] == 1].iloc[0]
        assert cluster_1["Count"] == 2
        assert cluster_1["Total_Erosion"] == 80.0
        assert cluster_1["Mean_Erosion"] == 40.0
        assert cluster_1["Median_Erosion"] == 40.0

    def test_summarize_clusters_custom_column_names(self):
        """Test that summarize_clusters works with custom column names."""
        clustered_df = pd.DataFrame(
            {
                "customer_id": ["u1", "u2", "u3"],
                "segment": [0, 0, 1],
                "erosion_value": [100.0, 50.0, 25.0],
            }
        )

        summary = summarize_clusters(
            clustered_df,
            value_col="erosion_value",
            cluster_col="segment",
        )

        assert len(summary) == 2
        assert "segment" in summary.columns
        assert summary.loc[0, "Total_Erosion"] == 150.0
        assert summary.loc[1, "Total_Erosion"] == 25.0

    def test_summarize_clusters_single_cluster(self):
        """Test that summarize_clusters works with a single cluster."""
        clustered_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3"],
                "cluster_id": [0, 0, 0],
                "total_profit_erosion": [100.0, 50.0, 30.0],
            }
        )

        summary = summarize_clusters(clustered_df)

        assert len(summary) == 1
        assert summary.loc[0, "Count"] == 3
        assert summary.loc[0, "Total_Erosion"] == 180.0
        assert summary.loc[0, "Mean_Erosion"] == 60.0

    def test_summarize_clusters_uneven_distribution(self):
        """Test summarize_clusters with clusters of different sizes."""
        clustered_df = pd.DataFrame(
            {
                "user_id": ["u1", "u2", "u3", "u4", "u5"],
                "cluster_id": [0, 1, 1, 1, 1],
                "total_profit_erosion": [200.0, 40.0, 30.0, 20.0, 10.0],
            }
        )

        summary = summarize_clusters(clustered_df)

        # Cluster 0: 1 customer
        cluster_0 = summary[summary["cluster_id"] == 0].iloc[0]
        assert cluster_0["Count"] == 1
        assert cluster_0["Mean_Erosion"] == 200.0

        # Cluster 1: 4 customers
        cluster_1 = summary[summary["cluster_id"] == 1].iloc[0]
        assert cluster_1["Count"] == 4
        assert cluster_1["Total_Erosion"] == 100.0
        assert cluster_1["Mean_Erosion"] == 25.0
        assert cluster_1["Median_Erosion"] == 25.0
