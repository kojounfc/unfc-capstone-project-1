"""
Unit tests for the visualization module.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualization import (
    # style / legacy eda
    set_plot_style,
    plot_status_distribution,
    plot_return_rate_by_category,
    plot_return_rate_heatmap,
    plot_margin_distribution,
    plot_margin_loss_by_category,
    plot_customer_margin_exposure,
    plot_price_margin_returned_by_status_country,
)


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
        # Only Tops & Tees has a return in sample data (per fixture assumption)
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
        # Only 1 customer (1001) has returns in sample data (per fixture assumption)
        assert len(ax.patches) == 1
        plt.close(fig)


class TestPlotPriceMarginReturnedByStatusCountry:
    """Test cases for plot_price_margin_returned_by_status_country function."""

    def test_creates_figure_without_error(self, sample_merged_df):
        """Test that function creates figure without raising errors."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            assert len(plt.get_fignums()) > 0
            plt.close("all")

    def test_handles_empty_dataframe(self):
        """Test that function handles empty DataFrame gracefully."""
        empty_df = pd.DataFrame()
        plot_price_margin_returned_by_status_country(empty_df)
        plt.close("all")

    def test_creates_4x2_grid(self, sample_merged_df):
        """Test that function creates correct 4x2 grid of subplots."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()
            assert len(fig.axes) == 8  # 4 rows x 2 columns
            plt.close(fig)

    def test_figure_has_correct_title(self, sample_merged_df):
        """Test that figure has the correct title."""
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()
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
            bottom_right_ax = fig.axes[-1]
            has_content = (
                len(bottom_right_ax.patches) > 0
                or len(bottom_right_ax.lines) > 0
                or len(bottom_right_ax.collections) > 0
            )
            assert not has_content, "Bottom-right subplot should be empty/hidden"
            plt.close(fig)

    def test_save_path_creates_file(self, sample_merged_df, tmp_path):
        """
        Test that save_path parameter saves figure to file.

        NOTE: plot_price_margin_returned_by_status_country() saves as:
          f"{save_path}_metrics_grid.png"
        """
        from src.analytics import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            base = tmp_path / "country_grid"
            plot_price_margin_returned_by_status_country(
                aggregated_df, save_path=str(base)
            )
            expected = tmp_path / "country_grid_metrics_grid.png"
            assert expected.exists()
            assert expected.stat().st_size > 0
            plt.close("all")


class TestPlotReturnRateHeatmap:
    """Test cases for plot_return_rate_heatmap function."""

    def test_creates_figure_without_error(self, sample_merged_df):
        """Test that function creates figure without raising errors."""
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
            {"population_share": [0.0, 0.5, 1.0], "value_share": [0.0, 0.2, 1.0]}
        )
        fig = plot_lorenz_curve(lorenz_df, gini=0.6)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_lorenz_curve

        lorenz_df = pd.DataFrame(
            {"population_share": [0.0, 0.5, 1.0], "value_share": [0.0, 0.3, 1.0]}
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
            {"customer_share": [0.0, 0.2, 1.0], "value_share": [0.0, 0.75, 1.0]}
        )
        fig = plot_pareto_curve(pareto_df, gini=0.7)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.lines) >= 3
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_pareto_curve

        pareto_df = pd.DataFrame(
            {"customer_share": [0.0, 0.2, 1.0], "value_share": [0.0, 0.8, 1.0]}
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
                "p_value": [0.01, 0.2, 0.03],
            }
        )
        fig = plot_gini_vs_pareto_scatter(concentration_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.collections) >= 2
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
            {"cluster_id": [0, 1, 2], "Count": [10, 5, 3], "Mean_Erosion": [100.0, 60.0, 30.0]}
        )
        fig = plot_cluster_erosion_comparison(cluster_summary_df, optimal_k=3)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
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
        assert len(ax.patches) == len(feature_importance_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_clustering_feature_importance

        feature_importance_df = pd.DataFrame(
            {"feature": ["x", "y"], "f_statistic": [3.0, 1.0], "p_value": [0.01, 0.2], "significant": [True, False]}
        )
        out_path = tmp_path / "fi.png"
        fig = plot_clustering_feature_importance(feature_importance_df, save_path=str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


class TestPlotFeatureConcentrationRanking:
    """Test cases for plot_feature_concentration_ranking (RQ2)."""

    def test_returns_figure_object_and_has_bars(self):
        from src.visualization import plot_feature_concentration_ranking

        concentration_df = pd.DataFrame(
            {"feature": ["f1", "f2", "f3"], "gini_coefficient": [0.7, 0.4, 0.2], "p_value": [0.01, 0.2, 0.03]}
        )
        fig = plot_feature_concentration_ranking(concentration_df)
        assert isinstance(fig, plt.Figure)
        ax = fig.axes[0]
        assert len(ax.patches) == len(concentration_df)
        plt.close(fig)

    def test_save_path_creates_file(self, tmp_path):
        from src.visualization import plot_feature_concentration_ranking

        concentration_df = pd.DataFrame({"feature": ["f1", "f2"], "gini_coefficient": [0.6, 0.3], "p_value": [0.01, 0.2]})
        out_path = tmp_path / "ranking.png"
        fig = plot_feature_concentration_ranking(concentration_df, save_path=str(out_path))
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        plt.close(fig)


# ============================================================================
# RQ1-SPECIFIC VISUALIZATION TESTS (Profit Erosion Analysis)
# IMPORTANT:
# RQ1 plot helpers now SAVE figures to disk and return Path (or (df, Path)).
# ============================================================================


class TestPlotTopGroupsTotalErosion:
    def test_saves_png_and_returns_path(self, tmp_path):
        from src.visualization import plot_top_groups_total_erosion

        df = pd.DataFrame(
            {"category": ["A", "B", "C"], "total_profit_erosion": [1000, 500, 200]}
        )

        out_path = tmp_path / "rq1_top_groups_total_erosion.png"
        result_path = plot_top_groups_total_erosion(
            df,
            group_col="category",
            value_col="total_profit_erosion",
            out_path=out_path,
        )

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestPlotReturnRateVsMeanErosion:
    def test_saves_png_and_returns_path(self, tmp_path):
        from src.visualization import plot_return_rate_vs_mean_erosion

        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "return_rate": [0.10, 0.20, 0.05],
                "avg_profit_erosion": [50.0, 100.0, 20.0],
                "returned_items": [10, 5, 2],
                "total_profit_erosion": [500, 500, 40],
            }
        )

        out_path = tmp_path / "rq1_return_rate_vs_mean.png"
        result_path = plot_return_rate_vs_mean_erosion(
            df,
            x_col="return_rate",
            y_col="avg_profit_erosion",
            label_col="category",
            size_col="returned_items",
            out_path=out_path,
        )

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestPlotSeverityVsVolumeDecomposition:
    def test_saves_png_and_returns_path(self, tmp_path):
        from src.visualization import plot_severity_vs_volume_decomposition

        df = pd.DataFrame(
            {
                "category": ["A", "B", "C"],
                "returned_items": [10, 5, 2],
                "avg_profit_erosion": [100.0, 50.0, 20.0],
                "total_profit_erosion": [1000.0, 250.0, 40.0],
            }
        )

        out_path = tmp_path / "rq1_severity_volume.png"
        result_path = plot_severity_vs_volume_decomposition(
            df,
            group_col="category",
            returned_items_col="returned_items",
            avg_erosion_col="avg_profit_erosion",
            total_erosion_col="total_profit_erosion",
            out_path=out_path,
        )

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestPlotProfitErosionDistributionLog:
    def test_saves_png_and_returns_path(self, tmp_path):
        from src.visualization import plot_profit_erosion_distribution_log

        returned_df = pd.DataFrame({"profit_erosion": [10, 100, 1000, 50, 75]})

        out_path = tmp_path / "rq1_erosion_dist_log.png"
        result_path = plot_profit_erosion_distribution_log(
            returned_df,
            value_col="profit_erosion",
            out_path=out_path,
        )

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestPlotBootstrapCIMeanByGroup:
    def test_returns_ci_df_and_saves_png(self, tmp_path):
        from src.visualization import plot_bootstrap_ci_mean_by_group

        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C"],
                "profit_erosion": [100, 120, 50, 60, 20, 25],
            }
        )

        out_path = tmp_path / "rq1_bootstrap_ci.png"
        ci_df, fig_path = plot_bootstrap_ci_mean_by_group(
            df,
            group_col="category",
            value_col="profit_erosion",
            out_path=out_path,
            n_boot=50,          # keep unit tests fast
            min_group_size=2,   # allow tiny fixture groups
            top_n_plot=3,
        )

        assert isinstance(ci_df, pd.DataFrame)
        assert fig_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0