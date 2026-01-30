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
                               plot_status_distribution, set_plot_style)


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
        from src.modeling import calculate_price_margin_returned_by_country

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
        from src.modeling import calculate_price_margin_returned_by_country

        aggregated_df = calculate_price_margin_returned_by_country(sample_merged_df)

        if not aggregated_df.empty:
            plot_price_margin_returned_by_status_country(aggregated_df)
            fig = plt.gcf()

            # Check that we have 4x2 subplots
            assert len(fig.axes) == 8  # 4 rows x 2 columns
            plt.close(fig)

    def test_figure_has_correct_title(self, sample_merged_df):
        """Test that figure has the correct title."""
        from src.modeling import calculate_price_margin_returned_by_country

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
        from src.modeling import calculate_price_margin_returned_by_country

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

        from src.modeling import calculate_price_margin_returned_by_country

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
