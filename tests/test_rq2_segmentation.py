import numpy as np
import pandas as pd
import pytest

from src.rq2_segmentation import (
    DEFAULT_SEGMENTATION_FEATURES,
    analyze_feature_importance_for_clustering,
    build_customer_segmentation_table,
    clustering_metrics_over_k,
    combined_diagnostics,
    compute_clustering_quality_metrics,
    elbow_inertia_over_k,
    kmeans_fit_predict,
    screen_clustering_features,
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
        assert cols == [c for c in DEFAULT_SEGMENTATION_FEATURES if c in df.columns]
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

    # Tests for screen_clustering_features
    def test_screen_clustering_features_drops_low_variance_features(self):
        """Test that features with variance below threshold are dropped."""
        X = pd.DataFrame(
            {
                "high_variance": [1.0, 100.0, 50.0, 75.0],
                "zero_variance": [5.0, 5.0, 5.0, 5.0],
                "low_variance": [10.0, 10.1, 10.0, 10.1],
                "medium_variance": [10.0, 20.0, 30.0, 40.0],
            }
        )

        surviving, report = screen_clustering_features(
            X, variance_threshold=0.01, verbose=False
        )

        # zero_variance should be dropped (variance = 0)
        assert "zero_variance" not in surviving
        assert "high_variance" in surviving
        assert "medium_variance" in surviving

        # Check report structure
        assert len(report) == 4
        assert "variance" in report.columns
        assert "variance_pass" in report.columns
        assert "final_status" in report.columns

    def test_screen_clustering_features_drops_highly_correlated_features(self):
        """Test that highly correlated features are dropped."""
        # FIXED: Create features with NON-MONOTONIC data to avoid perfect Spearman correlation
        X = pd.DataFrame(
            {
                "feature_a": [1.0, 4.0, 2.0, 3.0],  # Not monotonic
                "feature_b": [
                    2.0,
                    8.0,
                    4.0,
                    6.0,
                ],  # 2x feature_a (still highly correlated)
                "feature_c": [50.0, 10.0, 30.0, 20.0],  # Different pattern
            }
        )

        surviving, report = screen_clustering_features(
            X, correlation_threshold=0.85, verbose=False
        )

        # One of feature_a or feature_b should be dropped
        # feature_c should survive (different pattern)
        assert "feature_c" in surviving
        # Either feature_a or feature_b should survive (lower variance gets dropped)
        assert ("feature_a" in surviving) or ("feature_b" in surviving)
        # But not both (they are highly correlated)
        assert not (("feature_a" in surviving) and ("feature_b" in surviving))

    def test_screen_clustering_features_returns_expected_report_columns(self):
        """Test that the screening report has all expected columns."""
        X = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [10.0, 20.0, 30.0],
                "feat3": [5.0, 5.0, 5.0],  # zero variance
            }
        )

        surviving, report = screen_clustering_features(X, verbose=False)

        expected_columns = [
            "feature",
            "variance",
            "variance_pass",
            "correlation_pass",
            "highly_correlated_with",
            "final_status",
        ]

        for col in expected_columns:
            assert col in report.columns

    def test_screen_clustering_features_with_single_feature(self):
        """Test behavior when only one feature remains after variance filtering."""
        X = pd.DataFrame(
            {
                "good_feature": [1.0, 2.0, 3.0, 4.0],
                "zero_var_1": [5.0, 5.0, 5.0, 5.0],
                "zero_var_2": [10.0, 10.0, 10.0, 10.0],
            }
        )

        surviving, report = screen_clustering_features(
            X, variance_threshold=0.01, verbose=False
        )

        # Only one feature should survive
        assert len(surviving) == 1
        assert "good_feature" in surviving

        # correlation_pass should be True for the surviving feature
        good_feat_row = report[report["feature"] == "good_feature"].iloc[0]
        assert good_feat_row["correlation_pass"] is True

    def test_screen_clustering_features_all_features_pass(self):
        """Test case where all features pass both gates."""
        # FIXED: Use longer arrays with truly independent patterns
        X = pd.DataFrame(
            {
                "feat_a": [1.0, 10.0, 5.0, 15.0, 3.0, 12.0],
                "feat_b": [100.0, 50.0, 200.0, 25.0, 150.0, 75.0],
                "feat_c": [0.5, 3.5, 1.5, 2.5, 3.0, 1.0],
            }
        )

        surviving, report = screen_clustering_features(
            X, variance_threshold=0.01, correlation_threshold=0.85, verbose=False
        )

        # All features should survive (different patterns, high variance)
        assert len(surviving) == 3
        assert set(surviving) == {"feat_a", "feat_b", "feat_c"}

        # All should show final_status = 'pass'
        assert (report["final_status"] == "pass").all()

    def test_screen_clustering_features_verbose_output(self, capsys):
        """Test that verbose=True produces output."""
        X = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [5.0, 5.0, 5.0],  # zero variance
            }
        )

        surviving, report = screen_clustering_features(X, verbose=True)

        captured = capsys.readouterr()
        assert "GATE 1: Variance Threshold" in captured.out
        assert "GATE 2: Correlation Threshold" in captured.out
        assert "SCREENING COMPLETE" in captured.out

    def test_screen_clustering_features_custom_thresholds(self):
        """Test that custom thresholds work correctly."""
        X = pd.DataFrame(
            {
                "feat1": [1.0, 1.1, 1.2, 1.3],  # low variance (~0.0167)
                "feat2": [1.0, 10.0, 50.0, 100.0],  # high variance
            }
        )

        # With strict variance threshold
        surviving_strict, _ = screen_clustering_features(
            X, variance_threshold=0.02, verbose=False  # FIXED: Use 0.02 to drop feat1
        )

        # feat1 should be dropped with threshold 0.02 (variance ~0.0167 < 0.02)
        assert len(surviving_strict) == 1
        assert "feat2" in surviving_strict

        # With relaxed variance threshold
        surviving_relaxed, _ = screen_clustering_features(
            X, variance_threshold=0.001, verbose=False
        )

        # Both features should survive with threshold 0.001
        assert len(surviving_relaxed) == 2

    def test_screen_clustering_features_correlation_with_three_features(self):
        """Test correlation detection with three features where two are correlated."""
        # FIXED: Create proper non-monotonic data
        X = pd.DataFrame(
            {
                "independent": [1.0, 10.0, 5.0, 15.0],  # Different pattern
                "corr_a": [2.0, 6.0, 4.0, 8.0],  # Correlated with corr_b
                "corr_b": [20.0, 60.0, 40.0, 80.0],  # 10x corr_a pattern
            }
        )

        surviving, report = screen_clustering_features(
            X, correlation_threshold=0.85, verbose=False
        )

        # Only one of corr_a or corr_b should survive (they're correlated)
        # independent may or may not survive depending on its correlation
        assert len(surviving) == 2

        # Check that the report indicates correlation
        corr_report = report[report["feature"].isin(["corr_a", "corr_b"])]
        # At least one should have correlation_pass = False
        assert any(corr_report["correlation_pass"] == False)

    def test_screen_clustering_features_report_structure(self):
        """Test that the report DataFrame has correct structure and data types."""
        X = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0],
                "feat2": [10.0, 20.0, 30.0, 40.0],
                "feat3": [5.0, 5.0, 5.0, 5.0],
            }
        )

        surviving, report = screen_clustering_features(X, verbose=False)

        # Check DataFrame structure
        assert isinstance(report, pd.DataFrame)
        assert len(report) == 3  # One row per feature

        # Check data types
        assert report["variance"].dtype == np.float64
        assert report["variance_pass"].dtype == bool
        assert pd.api.types.is_string_dtype(report["final_status"])

        # Check final_status values are valid
        assert report["final_status"].isin(["pass", "fail"]).all()

    def test_screen_clustering_features_integration_with_actual_data(self):
        """Integration test with realistic customer segmentation features."""
        # Simulate realistic customer features
        np.random.seed(42)
        n_customers = 100

        X = pd.DataFrame(
            {
                "total_sales": np.random.lognormal(5, 2, n_customers),
                "order_frequency": np.random.poisson(3, n_customers),
                "avg_order_value": np.random.lognormal(4, 1, n_customers),
                "customer_tenure_days": np.random.uniform(30, 1000, n_customers),
                "return_rate": np.random.beta(2, 10, n_customers),
                # Add a constant feature (should be dropped)
                "constant_feature": np.ones(n_customers) * 100,
                # Add a highly correlated feature (should be dropped)
                "total_sales_duplicate": None,
            }
        )

        # Make total_sales_duplicate highly correlated with total_sales
        X["total_sales_duplicate"] = X["total_sales"] * 1.1 + np.random.normal(
            0, 0.01, n_customers
        )

        surviving, report = screen_clustering_features(
            X, variance_threshold=0.01, correlation_threshold=0.85, verbose=False
        )

        # constant_feature should be dropped (zero variance)
        assert "constant_feature" not in surviving

        # One of total_sales or total_sales_duplicate should be dropped
        assert not (
            ("total_sales" in surviving) and ("total_sales_duplicate" in surviving)
        )

        # Verify we have a reasonable number of features left
        assert 3 <= len(surviving) <= 6

    # Tests for analyze_feature_importance_for_clustering
    def test_analyze_feature_importance_returns_expected_columns(self):
        """Test that the function returns all expected columns."""
        # Create data with clear cluster separation
        X = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 10.0, 11.0],
                "feature_b": [100.0, 101.0, 200.0, 201.0],
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        expected_columns = [
            "feature",
            "f_statistic",
            "p_value",
            "eta_squared",
            "importance_score",
            "significant",
        ]

        for col in expected_columns:
            assert col in result.columns

    def test_analyze_feature_importance_sorted_by_importance(self):
        """Test that results are sorted by importance score (descending)."""
        # Create features with different separation power
        X = pd.DataFrame(
            {
                "high_separation": [1.0, 2.0, 100.0, 101.0],  # Clear separation
                "low_separation": [10.0, 11.0, 12.0, 13.0],  # Little separation
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Should be sorted descending by importance_score
        assert result["importance_score"].is_monotonic_decreasing
        # high_separation should be first (higher F-statistic)
        assert result.iloc[0]["feature"] == "high_separation"

    def test_analyze_feature_importance_well_separated_clusters(self):
        """Test with well-separated clusters (should have high F-statistics)."""
        X = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 100.0, 101.0, 102.0],
                "feature_2": [10.0, 11.0, 12.0, 200.0, 201.0, 202.0],
            }
        )
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Well-separated clusters should have high F-statistics
        assert (result["f_statistic"] > 10.0).all()
        # Should have significant p-values
        assert (result["p_value"] < 0.05).all()
        assert (result["significant"] == True).all()

    def test_analyze_feature_importance_poorly_separated_clusters(self):
        """Test with poorly separated clusters (should have low F-statistics)."""
        # Create overlapping clusters
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature_1": np.random.normal(10, 5, 20),
                "feature_2": np.random.normal(50, 5, 20),
            }
        )
        # Random cluster assignments (no real structure)
        cluster_labels = np.random.randint(0, 2, 20)

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Poorly separated clusters typically have lower F-statistics
        # (though this depends on random assignments)
        assert result["f_statistic"].notna().all()
        assert result["p_value"].notna().all()

    def test_analyze_feature_importance_single_feature(self):
        """Test with a single feature."""
        X = pd.DataFrame({"only_feature": [1.0, 2.0, 10.0, 11.0]})
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        assert len(result) == 1
        assert result.iloc[0]["feature"] == "only_feature"
        assert result.iloc[0]["f_statistic"] > 0

    def test_analyze_feature_importance_three_clusters(self):
        """Test with three clusters instead of two."""
        X = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 50.0, 51.0, 100.0, 101.0],
                "feature_b": [10.0, 11.0, 60.0, 61.0, 110.0, 111.0],
            }
        )
        cluster_labels = np.array([0, 0, 1, 1, 2, 2])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        assert len(result) == 2
        # Should work with 3 clusters
        assert result["f_statistic"].notna().all()
        assert result["p_value"].notna().all()

    def test_analyze_feature_importance_eta_squared_range(self):
        """Test that eta_squared is in valid range [0, 1]."""
        X = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
                "feature_2": [5.0, 6.0, 7.0, 50.0, 51.0, 52.0],
            }
        )
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # eta_squared should be between 0 and 1
        assert (result["eta_squared"] >= 0).all()
        assert (result["eta_squared"] <= 1).all()

    @pytest.mark.filterwarnings("ignore:.*constant.*:UserWarning")
    def test_analyze_feature_importance_identical_feature_values_in_clusters(self):
        """Test with features that have identical values within each cluster.
        
        constant_per_cluster: [5, 5, 5, 10, 10, 10]
        - Cluster 0: all 5.0 (no variance within cluster)
        - Cluster 1: all 10.0 (no variance within cluster)
        - Between clusters: perfect separation (5 vs 10)
        - Result: May produce inf F-stat (perfect separation, zero within-cluster variance)
        """
        X = pd.DataFrame(
            {
                "constant_per_cluster": [5.0, 5.0, 5.0, 10.0, 10.0, 10.0],
                "varying": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        cluster_labels = np.array([0, 0, 0, 1, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Should have result for both features
        assert len(result) == 2
        assert set(result["feature"]) == {"constant_per_cluster", "varying"}

        # Get F-statistics
        varying_row = result[result["feature"] == "varying"].iloc[0]
        const_row = result[result["feature"] == "constant_per_cluster"].iloc[0]
        
        # Varying should have a finite, positive F-statistic
        assert np.isfinite(varying_row["f_statistic"])
        assert varying_row["f_statistic"] > 0
        
        # constant_per_cluster has perfect separation (5 vs 10) with zero within-cluster variance
        # This can produce inf (which is mathematically correct - perfect discrimination)
        # Just verify it's not NaN (which would indicate a calculation error)
        assert not np.isnan(const_row["f_statistic"])
        # Should be marked as significant
        assert const_row["significant"] == True

    def test_analyze_feature_importance_importance_score_equals_f_statistic(self):
        """Test that importance_score equals f_statistic."""
        X = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 10.0, 11.0],
                "feature_b": [5.0, 6.0, 50.0, 51.0],
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # importance_score should equal f_statistic
        assert np.allclose(result["importance_score"], result["f_statistic"])

    def test_analyze_feature_importance_all_features_significant(self):
        """Test significant flag for well-separated features."""
        # Create very clear separation
        X = pd.DataFrame(
            {
                "feature_1": [0.0, 1.0, 1000.0, 1001.0],
                "feature_2": [10.0, 11.0, 2000.0, 2001.0],
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # All features should be significant (p < 0.05)
        assert (result["significant"] == True).all()
        assert (result["p_value"] < 0.05).all()

    def test_analyze_feature_importance_single_cluster_returns_error(self):
        """Test behavior when all samples belong to same cluster."""
        X = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [10.0, 20.0, 30.0, 40.0],
            }
        )
        cluster_labels = np.array([0, 0, 0, 0])  # All same cluster

        # scipy.stats.f_oneway requires at least 2 groups
        with pytest.raises(TypeError, match="least two samples"):
            analyze_feature_importance_for_clustering(X, cluster_labels)

    @pytest.mark.filterwarnings("ignore:.*constant.*:UserWarning")
    def test_analyze_feature_importance_zero_variance_feature(self):
        """Test with a feature that has zero variance across ALL samples.
        
        constant: [5, 5, 5, 5]
        - All values identical across all samples
        - No variance within OR between clusters
        - Result: NaN F-stat (undefined - cannot discriminate)
        """
        X = pd.DataFrame(
            {
                "constant": [5.0, 5.0, 5.0, 5.0],  # Zero variance everywhere
                "varying": [1.0, 2.0, 10.0, 11.0],  # Has variance
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Should handle zero variance gracefully
        assert len(result) == 2
        assert set(result["feature"]) == {"constant", "varying"}
        
        # Get the results
        const_row = result[result["feature"] == "constant"].iloc[0]
        varying_row = result[result["feature"] == "varying"].iloc[0]
        
        # Constant feature: all identical values → F-stat is NaN
        assert np.isnan(const_row["f_statistic"])
        # eta_squared should be 0 (no separation possible)
        assert const_row["eta_squared"] == 0.0
        
        # Varying feature should have a finite, positive F-statistic
        assert np.isfinite(varying_row["f_statistic"])
        assert varying_row["f_statistic"] > 0

    def test_analyze_feature_importance_feature_names_preserved(self):
        """Test that feature names are correctly preserved."""
        feature_names = ["sales", "margin", "frequency", "recency"]
        X = pd.DataFrame({name: np.random.randn(10) for name in feature_names})
        cluster_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # All feature names should be present
        assert set(result["feature"]) == set(feature_names)

    def test_analyze_feature_importance_integration_with_actual_clustering(self):
        """Integration test with actual K-means clustering."""
        # Create data with clear cluster structure
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature_1": np.concatenate(
                    [np.random.normal(0, 1, 25), np.random.normal(10, 1, 25)]
                ),
                "feature_2": np.concatenate(
                    [np.random.normal(0, 1, 25), np.random.normal(10, 1, 25)]
                ),
                "noise_feature": np.random.normal(5, 1, 50),  # No cluster structure
            }
        )

        # Cluster the data
        X_scaled = standardize_features(X)
        cluster_labels = kmeans_fit_predict(X_scaled, k=2, random_state=42)

        # Analyze feature importance
        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # feature_1 and feature_2 should be more important than noise_feature
        assert len(result) == 3
        # Top features should be feature_1 or feature_2
        assert result.iloc[0]["feature"] in ["feature_1", "feature_2"]
        assert result.iloc[1]["feature"] in ["feature_1", "feature_2"]

    def test_analyze_feature_importance_return_type(self):
        """Test that function returns a pandas DataFrame."""
        X = pd.DataFrame({"feature": [1.0, 2.0, 10.0, 11.0]})
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        assert isinstance(result, pd.DataFrame)

    def test_analyze_feature_importance_index_reset(self):
        """Test that the returned DataFrame has reset index."""
        X = pd.DataFrame(
            {
                "feat_a": [1.0, 2.0, 10.0, 11.0],
                "feat_b": [5.0, 6.0, 50.0, 51.0],
            }
        )
        cluster_labels = np.array([0, 0, 1, 1])

        result = analyze_feature_importance_for_clustering(X, cluster_labels)

        # Index should be 0, 1, 2, ... (not feature names or other values)
        assert (result.index == range(len(result))).all()
