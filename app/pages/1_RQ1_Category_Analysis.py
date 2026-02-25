"""
RQ1: Profit Erosion Differences Across Product Categories & Brands

Method: Descriptive Analysis + Kruskal-Wallis + Post-hoc Dunn
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="RQ1 – Category Analysis",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).parent.parent.parent
FIGURES_RQ1 = ROOT / "figures" / "rq1"
PROCESSED_RQ1 = ROOT / "data" / "processed" / "rq1"
PROCESSED = ROOT / "data" / "processed"

# ── Load data ─────────────────────────────────────────────────────────────────
_cat_df = None
_ci_df = None

_cat_path = PROCESSED / "us07_product_profit_erosion_by_category.parquet"
if _cat_path.exists():
    _cat_df = pd.read_parquet(_cat_path)

_ci_path = PROCESSED_RQ1 / "rq1_bootstrap_ci_category_mean.parquet"
if _ci_path.exists():
    _ci_df = pd.read_parquet(_ci_path)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 RQ1: Profit Erosion by Category & Brand")
st.markdown(
    """
**Research Question**: Do profit erosion levels differ significantly across product categories and brands?

**Method**: Kruskal-Wallis H-test (non-parametric ANOVA) with post-hoc Dunn pairwise comparisons
and bootstrap 95% confidence intervals on group means.
"""
)
st.divider()

# ── Statistical test summary ──────────────────────────────────────────────────
st.header("Statistical Test Summary")

stats_path = PROCESSED_RQ1 / "rq1_statistical_tests_summary.parquet"
if stats_path.exists():
    stats_df = pd.read_parquet(stats_path)
    display_cols = [
        "factor", "test_used", "p_value", "effect_size",
        "effect_metric", "reject_h0", "success_criteria_met", "n_groups", "n_rows",
    ]
    cols_present = [c for c in display_cols if c in stats_df.columns]
    styled = stats_df[cols_present].copy()
    styled["reject_h0"] = styled["reject_h0"].map({True: "✅ Yes", False: "❌ No"})
    styled["success_criteria_met"] = styled["success_criteria_met"].map(
        {True: "✅ Yes", False: "❌ No"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "Kruskal-Wallis tests whether profit erosion distributions differ significantly across groups. "
        "Effect size η² ≥ 0.06 (medium) confirms practical significance beyond statistical."
    )
else:
    st.warning("Statistical tests summary not found. Run the master notebook first.")

st.divider()

# ── Static figures (Fig 1, 2, 5, 6) ──────────────────────────────────────────
st.header("Visualizations")

static_figures = [
    ("fig1_top_categories_total_erosion.png", "Fig 1: Top Categories — Total Profit Erosion"),
    ("fig2_top_brands_total_erosion.png", "Fig 2: Top Brands — Total Profit Erosion"),
    ("fig5_severity_vs_volume_category.png", "Fig 5: Severity vs Volume by Category"),
    ("fig6_profit_erosion_distribution_log.png", "Fig 6: Profit Erosion Distribution (Log Scale)"),
]

for i in range(0, len(static_figures), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(static_figures):
            fname, label = static_figures[i + j]
            fig_path = FIGURES_RQ1 / fname
            with col:
                st.subheader(label)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fname}")

st.divider()

# ── Fig 3: Interactive scatter — Return Rate vs Mean Erosion ──────────────────
st.subheader("Fig 3: Return Rate vs Mean Erosion by Category (Interactive)")
if _cat_df is not None:
    size_col = next(
        (c for c in ["returned_items", "returned_item_rows", "return_count"]
         if c in _cat_df.columns),
        None,
    )
    x_col = "return_rate" if "return_rate" in _cat_df.columns else None
    y_col = next(
        (c for c in ["mean_profit_erosion_per_return", "mean_profit_erosion"]
         if c in _cat_df.columns),
        None,
    )

    if x_col and y_col and "category" in _cat_df.columns:
        scatter_kwargs = dict(
            data_frame=_cat_df,
            x=x_col,
            y=y_col,
            hover_name="category",
            color="total_profit_erosion",
            color_continuous_scale="Reds",
            title="Return Rate vs Mean Profit Erosion per Return — by Category",
            labels={
                x_col: "Return Rate",
                y_col: "Mean Erosion per Return ($)",
                "total_profit_erosion": "Total Erosion ($)",
            },
        )
        if size_col:
            scatter_kwargs["size"] = size_col
            scatter_kwargs["size_max"] = 30
        fig3 = px.scatter(**scatter_kwargs)
        fig3.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Return Rate: %{x:.1%}<br>"
                "Mean Erosion: $%{y:,.2f}<br>"
                "<extra></extra>"
            )
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(
            "Bubble size = number of returned items. Color intensity = total profit erosion. "
            "High-volume return categories do not always produce the highest per-return erosion."
        )
    else:
        fig_path = FIGURES_RQ1 / "fig3_return_rate_vs_mean_erosion_category.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.warning("Category data columns not found.")
else:
    fig_path = FIGURES_RQ1 / "fig3_return_rate_vs_mean_erosion_category.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.warning("Category parquet not found. Run the master notebook.")

st.divider()

# ── Fig 7: Interactive CI error bar chart ─────────────────────────────────────
st.subheader("Fig 7: Bootstrap 95% CI — Mean Profit Erosion by Category (Interactive)")
if _ci_df is not None and "mean_profit_erosion" in _ci_df.columns:
    cat_col = next(
        (c for c in ["category", "factor", "group"] if c in _ci_df.columns), None
    )
    if cat_col:
        ci_plot = _ci_df.copy().sort_values("mean_profit_erosion", ascending=True)
        err_plus = (
            (ci_plot["ci_high_95"] - ci_plot["mean_profit_erosion"]).tolist()
            if "ci_high_95" in ci_plot.columns else None
        )
        err_minus = (
            (ci_plot["mean_profit_erosion"] - ci_plot["ci_low_95"]).tolist()
            if "ci_low_95" in ci_plot.columns else None
        )
        fig7 = px.bar(
            ci_plot,
            x="mean_profit_erosion",
            y=cat_col,
            orientation="h",
            error_x=err_plus,
            error_x_minus=err_minus,
            title="Bootstrap 95% CI — Mean Profit Erosion per Return by Category",
            labels={
                "mean_profit_erosion": "Mean Profit Erosion per Return ($)",
                cat_col: "Category",
            },
            color="mean_profit_erosion",
            color_continuous_scale="Blues",
        )
        fig7.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Mean Erosion: $%{x:,.2f}<extra></extra>"
            )
        )
        st.plotly_chart(fig7, use_container_width=True)
        st.caption(
            "Error bars show 95% bootstrap confidence intervals around the group mean. "
            "Non-overlapping CIs confirm that category mean differences are statistically stable."
        )
    else:
        fig_path = FIGURES_RQ1 / "fig7_bootstrap_ci_category_mean.png"
        if fig_path.exists():
            st.image(str(fig_path), use_container_width=True)
else:
    fig_path = FIGURES_RQ1 / "fig7_bootstrap_ci_category_mean.png"
    if fig_path.exists():
        st.image(str(fig_path), use_container_width=True)
    else:
        st.warning("Bootstrap CI parquet not found. Run the master notebook.")

st.divider()

# ── Key findings (data-driven) ────────────────────────────────────────────────
st.header("Key Findings")

_stats_path = PROCESSED_RQ1 / "rq1_statistical_tests_summary.parquet"

if _cat_df is not None and _stats_path.exists():
    _stats = pd.read_parquet(_stats_path)

    _top5 = _cat_df.nlargest(5, "total_profit_erosion")["category"].tolist()
    _top5_str = ", ".join(_top5)

    _cat_row = (
        _stats[_stats["factor"] == "category"].iloc[0]
        if "category" in _stats["factor"].values else None
    )
    _brand_row = (
        _stats[_stats["factor"] == "brand"].iloc[0]
        if "brand" in _stats["factor"].values else None
    )

    _cat_p = f"{_cat_row['p_value']:.2e}" if _cat_row is not None else "N/A"
    _cat_eff = (
        f"{_cat_row['effect_size']:.3f} ({_cat_row['effect_metric']})"
        if _cat_row is not None else "N/A"
    )
    _cat_h0 = "Rejected" if (_cat_row is not None and _cat_row["reject_h0"]) else "Not Rejected"

    _brand_p = f"{_brand_row['p_value']:.2e}" if _brand_row is not None else "N/A"
    _brand_eff = (
        f"{_brand_row['effect_size']:.3f} ({_brand_row['effect_metric']})"
        if _brand_row is not None else "N/A"
    )
    _brand_h0 = (
        "Rejected" if (_brand_row is not None and _brand_row["reject_h0"]) else "Not Rejected"
    )

    st.markdown(
        f"""
- **Kruskal-Wallis (category)**: p = {_cat_p}, effect size = {_cat_eff} — H₀ **{_cat_h0}**
- **Kruskal-Wallis (brand)**: p = {_brand_p}, effect size = {_brand_eff} — H₀ **{_brand_h0}**
- **Top 5 categories by total profit erosion**: {_top5_str}
- **Return rate vs mean erosion** (Fig 3) shows that high-volume return categories do not always
  produce the highest per-return erosion — severity and volume are distinct risk dimensions
- **Bootstrap CIs** (Fig 7) confirm category mean differences are stable and not driven by
  small-sample noise
"""
    )
else:
    st.info("Key findings require processed data files. Run the master notebook first.")
