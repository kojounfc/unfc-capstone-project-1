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


# ── CSS: inline hover tooltip for all section/figure headers ──────────────────
st.markdown(
    """
    <style>
    .rq1-tip-title {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .rq1-tip-title h2 {
        margin: 0; padding: 0;
        font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em;
    }
    .rq1-tip-title h3 {
        margin: 0; padding: 0;
        font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em;
    }
    .rq1-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
        margin-left: 10px;
        flex-shrink: 0;
    }
    .rq1-tip-icon { font-size: 0.9rem; color: #888; user-select: none; }
    .rq1-tip-box {
        visibility: hidden;
        opacity: 0;
        width: 380px;
        background-color: rgba(28, 28, 44, 0.97);
        color: #e4e4f0;
        text-align: left;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.95rem;
        line-height: 1.65;
        position: absolute;
        z-index: 9999;
        bottom: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%);
        transition: opacity 0.2s ease;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
        pointer-events: none;
        white-space: normal;
    }
    .rq1-tip-box::after {
        content: "";
        position: absolute;
        top: 100%; left: 50%; margin-left: -6px;
        border: 6px solid transparent;
        border-top-color: rgba(28, 28, 44, 0.97);
    }
    .rq1-tip:hover .rq1-tip-box { visibility: visible; opacity: 1; }
    @media (max-width: 768px) {
        .rq1-tip-box { width: 260px; font-size: 0.85rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # app/pages/ → app/ → project root
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

# ── Executive Summary (always visible, above KPI cards) ──────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0f2440 0%, #1a3660 100%);
        border-left: 5px solid #e63946;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 8px;
    ">
        <p style="
            color: #f0c040;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0 0 10px 0;
        ">Executive Summary — Key Findings &amp; Implications</p>
        <p style="color: #e8eaf0; font-size: 1.0rem; line-height: 1.75; margin: 0;">
            <strong style="color: #ffffff;">Pipeline Demonstration — Distributional Output (Synthetic Dataset)</strong><br>
            On TheLook, the analytical pipeline surfaces statistically significant differences in profit erosion
            across product categories (ε² = 0.454, p &lt; 0.001) and brands (ε² = 0.442, p &lt; 0.001).
            Total financial loss is driven by two independent forces:
            <em>return frequency</em> (volume) and <em>cost per return</em> (severity) —
            illustrating the pipeline's ability to decompose erosion by product grouping.
            <strong style="color: #f0c040;">Decision: Reject H₀.</strong>
            Figures reflect the synthetic TheLook dataset; SSL directional validation confirms
            that category- and brand-level differences in profit erosion generalise in direction
            to real-world operational data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border: 0; border-top: 1px solid rgba(49,51,63,0.3); margin: 20px 0 24px 0;'>",
    unsafe_allow_html=True,
)

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


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ov, tab_cat, tab_brand, tab_decomp, tab_val, tab_conc = st.tabs([
    "📋 Overview",
    "📊 Category",
    "🏷️ Brand",
    "🔍 Decomposition",
    "🌐 Validation",
    "🎯 Conclusion",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:

    # ── 3-Panel Logic Chain ───────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            "<div style='background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;'>"
            "<span style='font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;'>"
            "STEP 1 — WHY NON-PARAMETRIC?</span></div>",
            unsafe_allow_html=True,
        )
        err_minus = (
            (ci_plot["mean_profit_erosion"] - ci_plot["ci_low_95"]).tolist()
            if "ci_low_95" in ci_plot.columns else None
        )
        _tip_header("Top Categories by Total Erosion", "fig_top_cat")
        st.caption("A small number of categories drive the majority of total erosion — the 80/20 pattern applies here.")
        fig1_path = FIGURES_RQ1 / "fig1_top_categories_total_erosion.png"
        if fig1_path.exists():
            st.image(str(fig1_path), use_container_width=True)
        else:
            if _cat_df is not None and {"category", "total_profit_erosion"}.issubset(_cat_df.columns):
                tmp = _cat_df.nlargest(10, "total_profit_erosion").copy()
                fig = px.bar(
                    tmp.sort_values("total_profit_erosion", ascending=True),
                    x="total_profit_erosion",
                    y="category",
                    orientation="h",
                    title="Top Categories by Total Profit Erosion",
                    labels={"total_profit_erosion": "Total Profit Erosion ($)", "category": "Category"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Category analysis results are not yet available.")

    with col_c:
        st.markdown(
            "<div style='background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;'>"
            "<span style='font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;'>"
            "STEP 3 — WHAT TO DO ABOUT IT?</span></div>",
            unsafe_allow_html=True,
        )
        fig7.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>Mean Erosion: $%{x:,.2f}<extra></extra>"
            )
        )
    else:
        st.warning("Statistical test results are not yet available.")

    st.divider()

    # Missing department figure (now included)
    st.header("Supporting View: Departments")
    _tip_header("Top Departments — Total Profit Erosion", "fig_top_dept")
    st.caption("Descriptive summary only — department-level differences are not formally tested in RQ1.")

    fig_dept_path = FIGURES_RQ1 / "fig4_top_departments_total_erosion.png"
    if fig_dept_path.exists():
        st.image(str(fig_dept_path), use_container_width=True)
    else:
        if _dept_df is not None and "total_profit_erosion" in _dept_df.columns:
            dept_col = "department" if "department" in _dept_df.columns else ("Department" if "Department" in _dept_df.columns else None)
            if dept_col:
                tmp = _dept_df.copy().sort_values("total_profit_erosion", ascending=False).head(12)
                fig = px.bar(
                    tmp.sort_values("total_profit_erosion", ascending=True),
                    x="total_profit_erosion",
                    y=dept_col,
                    orientation="h",
                    title="Top Departments by Total Profit Erosion",
                    labels={"total_profit_erosion": "Total Profit Erosion ($)", dept_col: "Department"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Department column not found in department parquet.")
        else:
            st.info("Department data not available yet.")

    st.divider()

    # ── Top Post-Hoc Pairs (elevated — no expander needed) ────────────────────
    _tip_header("Top 5 Most Significantly Different Category Pairs", "posthoc_cat", level=3)
    st.caption(
        "Dunn's test with Bonferroni correction — the five category pairs with the "
        "largest statistically significant differences in mean profit erosion (p_adj < 0.05)."
    )
    _ph_cat_path_ov = PROCESSED_RQ1 / "rq1_posthoc_category.csv"
    if _ph_cat_path_ov.exists():
        _ph_top = pd.read_csv(_ph_cat_path_ov)
        _ph_top = _ph_top.loc[:, ~_ph_top.columns.str.startswith("Unnamed")]
        if "p_adj" in _ph_top.columns:
            _ph_sig = _ph_top[_ph_top["p_adj"] < 0.05].sort_values("p_adj").head(5).copy()
            _ph_sig["p_adj"] = _ph_sig["p_adj"].map(lambda x: f"{x:.2e}")
            _ph_sig = _ph_sig.rename(columns={"group_a": "Category A", "group_b": "Category B", "p_adj": "p (Bonferroni)"})
            st.dataframe(_ph_sig, use_container_width=True, hide_index=True)
            st.caption(
                f"{len(_ph_top[_ph_top['p_adj'] < 0.05])} of {len(_ph_top)} category pairs "
                "are significantly different at α = 0.05 after Bonferroni correction."
            )
        else:
            st.info("p_adj column not found in post-hoc file.")
    else:
        st.info("Post-hoc category results not yet available.")

    st.divider()

    # Post-hoc and CI: keep but de-bloat via expanders
    st.header("Detailed Results (for review)")

    with st.expander("Post-hoc comparisons (Category & Brand)", expanded=False):
        st.caption(
            "Error bars show 95% bootstrap confidence intervals around the group mean. "
            "Non-overlapping CIs confirm that category mean differences are statistically stable."
        )

        ph_col_cat, ph_col_brand = st.columns(2)

        with ph_col_cat:
            _tip_header("Category Post-Hoc (Top pairs)", "posthoc_cat")
            ph_cat_path = PROCESSED_RQ1 / "rq1_posthoc_category.csv"
            if ph_cat_path.exists():
                ph_cat = pd.read_csv(ph_cat_path)
                ph_cat = ph_cat.loc[:, ~ph_cat.columns.str.startswith("Unnamed")]
                if "p_adj" in ph_cat.columns:
                    top10 = ph_cat.sort_values("p_adj", ascending=True).head(10).copy()
                    top10["p_adj"] = top10["p_adj"].map(lambda x: f"{x:.4f}")
                    st.dataframe(top10, use_container_width=True, hide_index=True)

                    with st.expander("Show full category post-hoc table", expanded=False):
                        full = ph_cat.copy()
                        full["p_adj"] = full["p_adj"].map(lambda x: f"{x:.6f}")
                        st.dataframe(full, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(ph_cat, use_container_width=True, hide_index=True)
            else:
                st.info("Post-hoc category file not found.")

        with ph_col_brand:
            _tip_header("Brand Post-Hoc (Top pairs)", "posthoc_brand")
            ph_brand_path = PROCESSED_RQ1 / "rq1_posthoc_brand.csv"
            if ph_brand_path.exists():
                ph_brand = pd.read_csv(ph_brand_path)
                ph_brand = ph_brand.loc[:, ~ph_brand.columns.str.startswith("Unnamed")]
                if "p_adj" in ph_brand.columns:
                    top10 = ph_brand.sort_values("p_adj", ascending=True).head(10).copy()
                    top10["p_adj"] = top10["p_adj"].map(lambda x: f"{x:.4f}")
                    st.dataframe(top10, use_container_width=True, hide_index=True)

                    with st.expander("Show full brand post-hoc table", expanded=False):
                        full = ph_brand.copy()
                        full["p_adj"] = full["p_adj"].map(lambda x: f"{x:.6f}")
                        st.dataframe(full, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(ph_brand, use_container_width=True, hide_index=True)
            else:
                st.info("Post-hoc brand file not found.")

    with st.expander("Bootstrap confidence intervals (Category means)", expanded=False):
        _tip_header("Bootstrap 95% CI — Category Means", "fig_ci", level=2)
        if _ci_df is not None and "mean_profit_erosion" in _ci_df.columns:
            cat_col = next(
                (c for c in _ci_df.columns if c.lower() in ("category", "product_category")),
                _ci_df.columns[0],
            )
            ci_sorted = (
                _ci_df.sort_values("mean_profit_erosion", ascending=False)
                .head(8)
                .copy()
            )
            ci_sorted["err_plus"] = ci_sorted["ci_high_95"] - ci_sorted["mean_profit_erosion"]
            ci_sorted["err_minus"] = ci_sorted["mean_profit_erosion"] - ci_sorted["ci_low_95"]
            ci_plot = ci_sorted.sort_values("mean_profit_erosion", ascending=True)
            ci_fig = px.scatter(
                ci_plot,
                x="mean_profit_erosion",
                y=cat_col,
                error_x="err_plus",
                error_x_minus="err_minus",
                title="Bootstrap 95% CI — Mean Profit Erosion per Return (Top 8 Categories)",
                labels={"mean_profit_erosion": "Mean Profit Erosion per Return ($)", cat_col: ""},
            )
            ci_fig.update_traces(marker=dict(size=10, color="#e63946"))
            ci_fig.update_layout(height=400, margin=dict(l=0, r=20, t=44, b=20))
            st.plotly_chart(ci_fig, use_container_width=True)
            st.caption(
                "Sorted by mean erosion (highest first). Error bars show 95% bootstrap confidence intervals. "
                "Non-overlapping intervals indicate materially different category-level erosion."
            )

            with st.expander("Show CI table", expanded=False):
                ci_display = _ci_df.copy()
                for col in ["mean_profit_erosion", "ci_low_95", "ci_high_95", "ci_width"]:
                    if col in ci_display.columns:
                        ci_display[col] = ci_display[col].map(lambda x: f"${x:,.2f}")
                st.dataframe(ci_display, use_container_width=True, hide_index=True)
        else:
            fig7_path = FIGURES_RQ1 / "fig7_bootstrap_ci_category_mean.png"
            if fig7_path.exists():
                st.image(str(fig7_path), use_container_width=True)
            else:
                st.info("Bootstrap confidence interval data is not yet available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CATEGORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_cat:
    _tip_header("Top Categories — Total Profit Erosion", "fig_top_cat", level=2)
    fig1_path = FIGURES_RQ1 / "fig1_top_categories_total_erosion.png"
    if fig1_path.exists():
        st.image(str(fig1_path), use_container_width=True)

    st.divider()

    _tip_header("Return Rate vs Mean Erosion by Category (Interactive)", "key_findings", level=2)
    if _cat_df is not None:
        size_col = next((c for c in ["returned_items", "returned_item_rows", "return_count"] if c in _cat_df.columns), None)
        x_col = "return_rate" if "return_rate" in _cat_df.columns else None
        y_col = next((c for c in ["mean_profit_erosion_per_return", "mean_profit_erosion"] if c in _cat_df.columns), None)

        if x_col and y_col and "category" in _cat_df.columns:
            scatter_kwargs = dict(
                data_frame=_cat_df,
                x=x_col,
                y=y_col,
                hover_name="category",
                color="total_profit_erosion" if "total_profit_erosion" in _cat_df.columns else None,
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
            fig = px.scatter(**scatter_kwargs)
            fig.update_traces(
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Return Rate: %{x:.1%}<br>"
                    "Mean Erosion: $%{y:,.2f}<br>"
                    "<extra></extra>"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Return frequency and loss per return represent distinct risk dimensions. "
                "Categories in the upper-right quadrant are structurally high-risk, while others require "
                "targeted interventions depending on whether risk is driven by severity or volume."
            )
        else:
            fig3_path = FIGURES_RQ1 / "fig3_return_rate_vs_mean_erosion_category.png"
            if fig3_path.exists():
                st.image(str(fig3_path), use_container_width=True)
            else:
                st.info("Required columns not found for interactive scatter.")
    else:
        st.info("Category data is not yet available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BRAND
# ─────────────────────────────────────────────────────────────────────────────
with tab_brand:
    _tip_header("Top Brands — Total Profit Erosion", "fig_top_brand", level=2)
    fig2_path = FIGURES_RQ1 / "fig2_top_brands_total_erosion.png"
    if fig2_path.exists():
        st.image(str(fig2_path), use_container_width=True)
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
        """
TheLook is a synthetic dataset. To strengthen external validity, the RQ1 workflow was replicated on
a real-world returns dataset (SSL — School Specialty, Inc., 2025).

**Objective:** Confirm that profit erosion differs across product groupings under operational conditions.
The same non-parametric workflow, grouping structure, and erosion definition were applied to the SSL dataset.

| Test | Dataset | p-value | Decision |
|---|---|---|---|
| Kruskal–Wallis (Category) | SSL | < 0.001 | ✅ Reject H₀ |
| Kruskal–Wallis (Brand / Supplier) | SSL | < 0.001 | ✅ Reject H₀ |

**Directional Validation:** The replication on SSL data confirms that the pipeline's core detection —
profit erosion differs across product groupings — generalises in direction to real-world operational data.
This is directional validation of framework utility, not parameter transferability.
Specific coefficient magnitudes from TheLook reflect the synthetic dataset and should not be applied directly.
"""
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
with tab_conc:
    st.header("Conclusion — Pipeline Output")

    # ── Dark-gradient callout ─────────────────────────────────────────────────
    if _cat_df is not None and "total_profit_erosion" in _cat_df.columns:
        _conc_total_erosion = float(_cat_df["total_profit_erosion"].sum())
        _conc_total_returns = (
            int(_cat_df["returned_items"].sum())
            if "returned_items" in _cat_df.columns else 0
        )
        _conc_mean_erosion = (
            _conc_total_erosion / _conc_total_returns
            if _conc_total_returns > 0 else 0.0
        )
        _conc_top_cat = (
            _cat_df.nlargest(1, "total_profit_erosion")["category"].iloc[0]
            if "category" in _cat_df.columns else "N/A"
        )
        _conc_n_cats = len(_cat_df)
        st.markdown(
            f"""
    <div style="background:linear-gradient(135deg,#0f2440 0%,#1a3660 100%);
                border-left:5px solid #e63946; border-radius:10px;
                padding:20px 26px; margin:0 0 16px 0;">
        <p style="color:#ef9a9a;font-size:0.75rem;font-weight:700;
                  letter-spacing:0.12em;text-transform:uppercase;margin:0 0 8px 0;">
            Pipeline Demonstration — Category-Level Output (Synthetic Dataset)
        </p>
        <p style="color:#ffffff;font-size:1.05rem;font-weight:700;margin:0 0 6px 0;">
            On TheLook, the pipeline surfaces {_conc_total_returns:,} returned items across
            {_conc_n_cats} categories, with total modelled profit erosion of
            USD&nbsp;{_conc_total_erosion:,.2f} (mean USD&nbsp;{_conc_mean_erosion:,.2f} per return).
        </p>
        <p style="color:#fce4ec;font-size:0.9rem;line-height:1.65;margin:0;">
            Highest-risk category by total erosion: <strong>{_conc_top_cat}</strong>.
            This illustrates how the framework decomposes profit erosion by product grouping —
            distinguishing high-severity categories from high-volume categories.
            Figures reflect the synthetic dataset; SSL directional validation confirms the
            pattern generalises in direction to real-world operational data.
        </p>
    </div>
    """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Category erosion data not available — callout requires processed parquet files.")

    st.divider()

    # ── Hypothesis Decision Table ─────────────────────────────────────────────
    st.subheader("Hypothesis Decisions")

    # Try to pull effect sizes dynamically from _stats_df
    _cat_es = _brand_es = "0.454"
    _brand_es_val = "0.442"
    if _stats_df is not None and "effect_size" in _stats_df.columns:
        _factor_col = [c for c in _stats_df.columns if "factor" in c.lower()]
        if _factor_col:
            _cat_row = _stats_df[_stats_df[_factor_col[0]].str.lower().str.contains("categ", na=False)]
            _brd_row = _stats_df[_stats_df[_factor_col[0]].str.lower().str.contains("brand", na=False)]
            if not _cat_row.empty:
                _cat_es = f"{float(_cat_row['effect_size'].iloc[0]):.3f}"
            if not _brd_row.empty:
                _brand_es_val = f"{float(_brd_row['effect_size'].iloc[0]):.3f}"

    st.markdown(
        f"""
| Hypothesis | Dataset | Test | Decision |
|---|---|---|---|
| **H₀₁**: Profit erosion equal across categories | TheLook | Kruskal–Wallis, ε² = {_cat_es}, p < 0.001 | ✅ **REJECT H₀** |
| **H₀₁**: Profit erosion equal across brands | TheLook | Kruskal–Wallis, ε² = {_brand_es_val}, p < 0.001 | ✅ **REJECT H₀** |
| Directional validation (categories) | SSL | Kruskal–Wallis, p < 0.001 | ✅ Pattern confirmed |
| Directional validation (brands/suppliers) | SSL | Kruskal–Wallis, p < 0.001 | ✅ Pattern confirmed |
"""
    )

    st.divider()

    # ── Top-3 Category Pipeline Output Panel ─────────────────────────────────
    st.subheader("Top Categories Identified by the Pipeline")

    if _cat_df is not None and "category" in _cat_df.columns:
        _top3 = _cat_df.nlargest(3, "total_profit_erosion").reset_index(drop=True)
        _panel_cols = st.columns(3)
        _panel_colors = [
            ("#FFEBEE", "#C62828", "#B71C1C"),
            ("#FFF3E0", "#E65100", "#BF360C"),
            ("#FCE4EC", "#AD1457", "#880E4F"),
        ]
        for _i, (_col, (_bg, _header, _text)) in enumerate(
            zip(_panel_cols, _panel_colors)
        ):
            if _i < len(_top3):
                _row = _top3.iloc[_i]
                _cat_name = _row["category"]
                _cat_erosion = float(_row["total_profit_erosion"])
                _cat_returns = (
                    int(_row["returned_items"])
                    if "returned_items" in _row.index else 0
                )
                _cat_mean = (
                    float(_row["mean_profit_erosion_per_return"])
                    if "mean_profit_erosion_per_return" in _row.index
                    else (_cat_erosion / _cat_returns if _cat_returns > 0 else 0.0)
                )
                with _col:
                    st.markdown(
                        f"""
<div style="background:{_bg};border-left:4px solid {_header};
            padding:16px;border-radius:6px;">
<h4 style="margin:0 0 8px 0;color:{_header};">#{_i + 1} — {_cat_name}</h4>
<p style="margin:0;font-size:13px;color:{_text};">
<b>Total erosion: USD&nbsp;{_cat_erosion:,.2f}</b><br>
Returns: {_cat_returns:,} | Mean: USD&nbsp;{_cat_mean:,.2f}/return<br><br>
Highest-erosion category group identified by the pipeline on TheLook
(synthetic dataset — directional indicator only).
</p></div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Category data not available.")

    st.divider()
    st.subheader("RQ1 Summary")
    st.markdown(
        f"""
| Finding | Result |
|---|---|
| **H₀₁ (categories): Reject?** | ✅ Yes — Kruskal–Wallis ε² = {_cat_es}, p < 0.001 |
| **H₀₁ (brands): Reject?** | ✅ Yes — Kruskal–Wallis ε² = {_brand_es_val}, p < 0.001 |
| **Erosion decomposition** | Two independent drivers: return frequency (volume) and cost per return (severity) |
| **Highest-risk category** | {_conc_top_cat if _cat_df is not None else "N/A"} (highest total modelled profit erosion) |
| **External validation (SSL)** | ✅ Directional patterns confirmed — profit erosion differs across categories and brands |
| **Framework contribution** | Pipeline decomposes erosion by product grouping; figures are from synthetic dataset |
"""
    )

st.caption(
    "DAMO-699-4 · University of Niagara Falls, Canada · Winter 2026 · "
    "RQ1 — Category & Brand Erosion Analysis"
)
