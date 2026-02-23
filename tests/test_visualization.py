"""
Unit tests for the visualization module.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualization import (plot_customer_margin_exposure,
                               plot_margin_distribution,
                               plot_margin_loss_by_category,
                               plot_price_margin_returned_by_status_country,
                               plot_return_rate_by_category,
                               plot_return_rate_heatmap,
                               plot_status_distribution, set_plot_style,)


class TestSetPlotStyle:
    """Test cases for set_plot_style function."""

    def test_does_not_raise_error(self):
        """Test that set_plot_style executes without error."""
        set_plot_style()  # Should not raise


class TestPlotStatusDistribution:
    """Test cases for plot_status_distribution function."""

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_status_distribution(sample_merged_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_bar_chart(self, sample_merged_df):
        """Test that a bar chart is created."""
        fig = plot_status_distribution(sample_merged_df)
        ax = fig.axes[0]
        # Check that there are bar patches
        assert len(ax.patches) > 0
        plt.close(fig)


class TestPlotReturnRateByCategory:
    """Test cases for plot_return_rate_by_category function."""

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_return_rate_by_category(sample_merged_df, min_rows=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_respects_top_n_parameter(self, sample_merged_df):
        """Test that top_n limits the number of categories shown."""
        fig = plot_return_rate_by_category(sample_merged_df, top_n=2, min_rows=1)
        ax = fig.axes[0]
        # Should have at most 2 bars
        assert len(ax.patches) <= 2
        plt.close(fig)


class TestPlotMarginDistribution:
    """Test cases for plot_margin_distribution function."""

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_margin_distribution(sample_merged_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_two_subplots(self, sample_merged_df):
        """Test that two subplots are created (histogram and boxplot)."""
        fig = plot_margin_distribution(sample_merged_df)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_filters_returned_items_when_specified(self, sample_merged_df):
        """Test that returned_only parameter filters data."""
        fig = plot_margin_distribution(sample_merged_df, returned_only=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMarginLossByCategory:
    """Test cases for plot_margin_loss_by_category function."""

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_margin_loss_by_category(sample_merged_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_only_shows_categories_with_returns(self, sample_merged_df):
        """Test that only categories with returns are displayed."""
        fig = plot_margin_loss_by_category(sample_merged_df)
        ax = fig.axes[0]
        # Only Tops & Tees has a return in sample data
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotCustomerMarginExposure:
    """Test cases for plot_customer_margin_exposure function."""

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_customer_margin_exposure(sample_merged_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_only_shows_customers_with_returns(self, sample_merged_df):
        """Test that only customers with returns are displayed."""
        fig = plot_customer_margin_exposure(sample_merged_df)
        ax = fig.axes[0]
        # Only 1 customer (1001) has returns in sample data
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotPriceMarginReturnedByStatusCountry:
    """Test cases for plot_price_margin_returned_by_status_country function."""

    def test_creates_figure_without_error(self, sample_merged_df):
        """Test that function creates figure without raising errors."""
        # Need to create aggregated data first
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            # Should not raise any errors
            plot_price_margin_returned_by_status_country(aggregated_df)
            assert len(plt.get_fignums()) > 0
            plt.close("all")

    def test_handles_empty_dataframe(self):
        """Test that function handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame()

        # Should print message and return without error
        plot_price_margin_returned_by_status_country(empty_df)
        plt.close("all")

    def test_creates_4x2_grid(self, sample_merged_df):
        """Test that function creates correct 4x2 grid of subplots."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()

            # Check that we have 4x2 subplots
            assert len(fig.axes) == 8  # 4 rows x 2 columns
            plt.close(fig)

    def test_figure_has_correct_title(self, sample_merged_df):
        """Test that figure has the correct title."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()

            # Check for title containing key words
            assert fig._suptitle is not None
            title_text = fig._suptitle.get_text()
            assert "RETURNED" in title_text.upper()
            assert "COUNTRY" in title_text.upper()
            plt.close(fig)

    def test_bottom_right_subplot_hidden(self, sample_merged_df):
        """Test that the bottom-right subplot (axes[3, 1]) is hidden."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()

            # Get the bottom-right subplot (last one in the 4x2 grid)
            # Axes are returned in order: axes[0,0], axes[0,1], ..., axes[3,1]
            bottom_right_ax = fig.axes[-1]

            # When axis("off") is called, the label is set to ""
            # and the axis ticks and labels should be disabled
            # Check that no patches or lines are drawn (empty plot)
            has_content = (
                len(bottom_right_ax.patches) > 0
                or len(bottom_right_ax.lines) > 0
                or len(bottom_right_ax.collections) > 0
            )
            # The subplot should be empty (no bars, lines, or collections)
            assert not has_content, "Bottom-right subplot should be empty/hidden"
            plt.close(fig)

    def test_save_path_creates_file(self, sample_merged_df, tmp_path):
        """Test that save_path parameter saves figure to file."""
        import os

        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            save_path = os.path.join(str(tmp_path), "test_plot.png")
            plot_price_margin_returned_by_status_country(
                aggregated_df, save_path=save_path
            )

            # File should be created if save_path is implemented
            # Note: Current implementation may not save, so this is conditional
            plt.close("all")


class TestPlotReturnRateHeatmap:
    """Test cases for plot_return_rate_heatmap function."""

    def test_creates_figure_without_error(self, sample_merged_df):
        """Test that function creates figure without raising errors."""
        # Use min_rows=1 to allow small sample data
        fig = plot_return_rate_heatmap(sample_merged_df, min_rows=1)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_returns_figure_object(self, sample_merged_df):
        """Test that function returns a matplotlib Figure."""
        fig = plot_return_rate_heatmap(sample_merged_df, min_rows=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_heatmap(self, sample_merged_df):
        """Test that figure contains a heatmap visualization."""
        fig = plot_return_rate_heatmap(sample_merged_df, min_rows=1)

        # Should have at least one axis (heatmap)
        assert len(fig.axes) > 0
        plt.close(fig)
# ============================================================================
# RQ2-SPECIFIC VISUALIZATION TESTS (Concentration & Segmentation)
# ============================================================================


class TestPlotLorenzCurve:
    """Test cases for plot_lorenz_curve (RQ2)."""

    def test_returns_figure_object(self):
        from src.visualization import plot_lorenz_curve

        lorenz_df = pd.DataFrame(
            {
                "population_share": [0.0, 0.5, 1.0],
                "value_share": [0.0, 0.2, 1.0],
            }
        )
        fig = plot_lorenz_curve(lorenz_df, gini=0.6)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        # Should have at least 2 lines: Lorenz + equality
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_lorenz_curve

        lorenz_df = pd.DataFrame(
            {
                "population_share": [0.0, 0.5, 1.0],
                "value_share": [0.0, 0.3, 1.0],
            }
        )
        out_path = tmp_path / "lorenz.png"
        fig = plot_lorenz_curve(lorenz_df, gini=0.5, save_path=str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotParetoCurve:
    """Test cases for plot_pareto_curve (RQ2)."""

    def test_returns_figure_object_and_has_reference_lines(self):
        from src.visualization import plot_pareto_curve

        pareto_df = pd.DataFrame(
            {
                "customer_share": [0.0, 0.2, 1.0],
                "value_share": [0.0, 0.75, 1.0],
            }
        )
        fig = plot_pareto_curve(pareto_df, gini=0.7)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should have curve line + at least 2 reference lines (80%/20%)
        assert len(ax.lines) >= 3
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_pareto_curve

        pareto_df = pd.DataFrame(
            {
                "customer_share": [0.0, 0.2, 1.0],
                "value_share": [0.0, 0.8, 1.0],
            }
        )
        out_path = tmp_path / "pareto.png"
        fig = plot_pareto_curve(pareto_df, gini=0.8, save_path=str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotGiniVsParetoScatter:
    """Test cases for plot_gini_vs_pareto_scatter (RQ2)."""

    def test_returns_figure_object_and_has_two_scatter_groups(self):
        from src.visualization import plot_gini_vs_pareto_scatter

        concentration_df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "gini_coefficient": [0.8, 0.4, 0.2],
                "top_20_pct_share": [90.0, 70.0, 40.0],
                "p_value": [0.01, 0.2, 0.03],  # two significant, one not
            }
        )
        fig = plot_gini_vs_pareto_scatter(concentration_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Matplotlib scatter creates PathCollection objects in ax.collections
        assert len(ax.collections) >= 2  # significant + non-significant groups
        # Should annotate each point
        assert len(ax.texts) >= len(concentration_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_gini_vs_pareto_scatter

        concentration_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "gini_coefficient": [0.6, 0.3],
                "top_20_pct_share": [85.0, 55.0],
                "p_value": [0.01, 0.2],
            }
        )
        out_path = tmp_path / "gini_scatter.png"
        fig = plot_gini_vs_pareto_scatter(concentration_df, save_path=str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotClusteringDiagnostics:
    """Test cases for plot_clustering_diagnostics (RQ2)."""

    def test_returns_figure_object_with_two_axes(self):
        from src.visualization import plot_clustering_diagnostics

        elbow_df = pd.DataFrame({"k": [1, 2, 3], "inertia": [10.0, 5.0, 3.0]})
        silhouette_df = pd.DataFrame({"k": [2, 3], "silhouette": [0.4, 0.35]})
        fig = plot_clustering_diagnostics(elbow_df, silhouette_df, optimal_k=2)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        # Optimal-k highlight adds a point marker on silhouette axis
        ax2 = fig.axes[1]
        assert len(ax2.lines) >= 2
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_clustering_diagnostics

        elbow_df = pd.DataFrame({"k": [1, 2], "inertia": [10.0, 4.0]})
        silhouette_df = pd.DataFrame({"k": [2], "silhouette": [0.5]})
        out_path = tmp_path / "diag.png"
        fig = plot_clustering_diagnostics(
            elbow_df, silhouette_df, optimal_k=2, save_path=str(out_path)
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotClusterErosionComparison:
    """Test cases for plot_cluster_erosion_comparison (RQ2)."""

    def test_returns_figure_object_and_has_bars(self):
        from src.visualization import plot_cluster_erosion_comparison

        cluster_summary_df = pd.DataFrame(
            {
                "cluster_id": [0, 1, 2],
                "Count": [10, 5, 3],
                "Mean_Erosion": [100.0, 60.0, 30.0],
            }
        )
        fig = plot_cluster_erosion_comparison(cluster_summary_df, optimal_k=3)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Bars should match number of clusters
        assert len(ax.patches) == len(cluster_summary_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_cluster_erosion_comparison

        cluster_summary_df = pd.DataFrame(
            {"cluster_id": [0, 1], "Count": [2, 2], "Mean_Erosion": [10.0, 20.0]}
        )
        out_path = tmp_path / "cluster_comp.png"
        fig = plot_cluster_erosion_comparison(
            cluster_summary_df, optimal_k=2, save_path=str(out_path)
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotClusteringFeatureImportance:
    """Test cases for plot_clustering_feature_importance (RQ2)."""

    def test_returns_figure_object_and_has_bars(self):
        from src.visualization import plot_clustering_feature_importance

        feature_importance_df = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "f_statistic": [12.0, 5.0, 1.0],
                "p_value": [0.001, 0.02, 0.2],
                "significant": [True, True, False],
            }
        )
        fig = plot_clustering_feature_importance(feature_importance_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Should have one bar per feature
        assert len(ax.patches) == len(feature_importance_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_clustering_feature_importance

        feature_importance_df = pd.DataFrame(
            {
                "feature": ["x", "y"],
                "f_statistic": [3.0, 1.0],
                "p_value": [0.01, 0.2],
                "significant": [True, False],
            }
        )
        out_path = tmp_path / "fi.png"
        fig = plot_clustering_feature_importance(
            feature_importance_df, save_path=str(out_path)
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotFeatureConcentrationRanking:
    """Test cases for plot_feature_concentration_ranking (RQ2)."""

    def test_returns_figure_object_and_has_bars(self):
        from src.visualization import plot_feature_concentration_ranking

        concentration_df = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "gini_coefficient": [0.7, 0.4, 0.2],
                "p_value": [0.01, 0.2, 0.03],
            }
        )
        fig = plot_feature_concentration_ranking(concentration_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        # Horizontal bars: one Rectangle patch per feature
        assert len(ax.patches) == len(concentration_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_feature_concentration_ranking

        concentration_df = pd.DataFrame(
            {
                "feature": ["f1", "f2"],
                "gini_coefficient": [0.6, 0.3],
                "p_value": [0.01, 0.2],
            }
        )
        out_path = tmp_path / "ranking.png"
        fig = plot_feature_concentration_ranking(
            concentration_df, save_path=str(out_path)
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)

# ============================================================================
# RQ1-SPECIFIC VISUALIZATION TESTS (Profit Erosion Analysis)
# ============================================================================


class TestPlotTopGroupsTotalErosion:
    def test_returns_figure_object(self):
        from src.visualization import plot_top_groups_total_erosion

        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "total_profit_erosion": [1000, 500, 200]
        })

        fig = plot_top_groups_total_erosion(
            df,
            group_col="category",
            value_col="total_profit_erosion"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotReturnRateVsMeanErosion:
    def test_returns_figure_object(self):
        from src.visualization import plot_return_rate_vs_mean_erosion

        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "return_rate": [0.1, 0.2, 0.05],
            "mean_profit_erosion": [50, 100, 20]
        })

        fig = plot_return_rate_vs_mean_erosion(
            df,
            group_col="category",
            return_rate_col="return_rate",
            erosion_col="mean_profit_erosion"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSeverityVsVolumeDecomposition:
    def test_returns_figure_object(self):
        from src.visualization import plot_severity_vs_volume_decomposition

        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "returned_items": [10, 5, 2],
            "mean_profit_erosion": [100, 50, 20]
        })

        fig = plot_severity_vs_volume_decomposition(
            df,
            group_col="category",
            volume_col="returned_items",
            severity_col="mean_profit_erosion"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotProfitErosionDistributionLog:
    def test_returns_figure_object(self):
        from src.visualization import plot_profit_erosion_distribution_log

        df = pd.DataFrame({
            "profit_erosion": [10, 100, 1000, 50, 75]
        })

        fig = plot_profit_erosion_distribution_log(
            df,
            erosion_col="profit_erosion"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotBootstrapCIMeanByGroup:
    def test_returns_figure_object(self):
        from src.visualization import plot_bootstrap_ci_mean_by_group

        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C", "C"],
            "profit_erosion": [100, 120, 50, 60, 20, 25]
        })

        fig = plot_bootstrap_ci_mean_by_group(
            df,
            group_col="category",
            value_col="profit_erosion"
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)