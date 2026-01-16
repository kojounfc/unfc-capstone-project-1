"""
Unit tests for the visualization module.
"""
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualization import (
    set_plot_style,
    plot_status_distribution,
    plot_return_rate_by_category,
    plot_margin_distribution,
    plot_margin_loss_by_category,
    plot_customer_margin_exposure,
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
