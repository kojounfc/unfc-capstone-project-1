"""
Streamlit App — RQ1 Dashboard (Presentation-First, Graduate Tone)

RQ1: Profit Erosion Differences Across Product Categories & Brands
Method: Descriptive Analysis + Kruskal-Wallis + Post-hoc Dunn + Bootstrap CIs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RQ1 – Profit Erosion Dashboard",
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]  # app/pages/ → app/ → project root
FIGURES_RQ1 = ROOT / "figures" / "rq1"
PROCESSED_RQ1 = ROOT / "data" / "processed" / "rq1"
PROCESSED = ROOT / "data" / "processed"


# ── Tooltips (aligned to rq1_technical_documentation_corrected_latest_results.md) ──
_TOOLTIPS = {
    # KPI cards
    "kpi_total_erosion": (
        "**Total Profit Erosion:** Total financial loss from all returned items in the dataset. "
        "It combines lost margin and processing costs, and reflects the overall economic impact of returns."
    ),
    "kpi_returns": (
        "**Total Returns Analyzed:** Number of returned order-items included in the analysis. "
        "Only returned items are included when calculating profit erosion."
    ),
    "kpi_mean_erosion": (
        "**Mean Erosion per Return:** Average financial loss per returned item. "
        "This highlights categories where each return event is especially costly."
    ),
    "kpi_top_cat": (
        "**Highest-Risk Category:** Category with the largest total profit erosion. "
        "This shows where overall financial risk is most concentrated."
    ),

    # Statistical summaries (latest results)
    "stat_category": (
        "**What this test evaluates:** Whether profit erosion differs across product categories "
        "using the Kruskal–Wallis non-parametric test. Effect size (ε²) indicates magnitude of difference."
    ),
    "stat_brand": (
         "**What this test evaluates:** Whether profit erosion differs across brands "
         "using the Kruskal–Wallis non-parametric test. Effect size (ε²) indicates magnitude of difference."
    ),

    # Post-hoc
    "posthoc_cat": (
         "**What this table shows:** Pairwise category comparisons following a significant "
         "Kruskal–Wallis result, adjusted for multiple testing using Dunn's procedure."
    ),
    "posthoc_brand": (
        "**What this table shows:** Pairwise brand comparisons following a significant "
        "Kruskal–Wallis result, adjusted for multiple testing using Dunn's procedure."
    ),

    # Figures
    "fig_top_cat": (
        "**What this chart shows:** Total profit erosion aggregated by product category. "
        "Bars are ordered by magnitude, highlighting which categories contribute most to overall financial loss."
    ),
    "fig_top_brand": (
        "**What this chart shows:** Total profit erosion aggregated by brand. "
        "This visual compares the relative contribution of brands to overall return-related financial loss."
    ),
    "fig_top_dept": (
        "**What this chart shows:** Total profit erosion aggregated at the department level. "
        "This is a descriptive organizational view and is not part of the formal hypothesis testing."
    ),
    "fig_dist": (
        "**What this chart shows:** Distribution of profit erosion per returned item on a log scale. "
        "The long right tail indicates the presence of extreme loss values, motivating the use of non-parametric tests."
    ),
    "fig_sev_vol": (
        "**What this chart shows:** Each point represents a category positioned by return frequency (volume) "
        "and average erosion per return (severity). Bubble size reflects total erosion. "
        "This separates structural volume effects from per-unit loss effects."
    ),
    "fig_ci": (
        "**What this chart shows:** Bootstrap 95% confidence intervals for mean profit erosion per return "
        "across categories. Error bars reflect sampling variability around the mean estimate."
    ),

    # Validation
    "ssl_validation": (
        "**What this section shows:** Replication of the category-level non-parametric workflow "
        "on real-world SSL returns data. Category is constructed as Pillar-MajorMarketCat-Department. "
        "This evaluates whether the direction of category-level differences persists outside the synthetic dataset."
    ),
}


def _tip_header(label: str, tooltip_key: str, level: int = 3) -> None:
    """Render a section/figure header with an inline CSS hover tooltip."""
    raw = _TOOLTIPS[tooltip_key]
    parts = raw.split("**")
    tip_html = "".join(
        f"<strong>{p}</strong>" if i % 2 == 1 else p
        for i, p in enumerate(parts)
    )
    st.markdown(
        f'<div class="rq1-tip-title">'
        f'<h{level}>{label}</h{level}>'
        f'<span class="rq1-tip">'
        f'<span class="rq1-tip-icon">ℹ️</span>'
        f'<span class="rq1-tip-box">{tip_html}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _safe_read_parquet(path: Path):
    """Read parquet with a clean error if parquet engines are missing."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.error(
            f"Could not read parquet file: {path.name}. "
            "Make sure your environment has a parquet engine installed (pyarrow or fastparquet)."
        )
        st.caption(f"Details: {e}")
        return None


def _plain_tooltip(key: str) -> str:
    """Plain-text tooltip for Streamlit help="" fields."""
    return _TOOLTIPS[key].replace("**", "")


# ── Load processed data ───────────────────────────────────────────────────────
_cat_df = None
_brand_df = None
_dept_df = None
_ci_df = None
_stats_df = None

cat_path = PROCESSED_RQ1 / "rq1_erosion_by_category.csv"
brand_path = PROCESSED_RQ1 / "rq1_erosion_by_brand.csv"
dept_path = PROCESSED_RQ1 / "rq1_erosion_by_department.csv"
ci_path = PROCESSED_RQ1 / "rq1_bootstrap_ci_category_mean.parquet"
stats_path = PROCESSED_RQ1 / "rq1_statistical_tests_summary.parquet"

if cat_path.exists():
    _cat_df = pd.read_csv(cat_path)

if brand_path.exists():
    _brand_df = pd.read_csv(brand_path)

if dept_path.exists():
    _dept_df = pd.read_csv(dept_path)

if ci_path.exists():
    _ci_df = _safe_read_parquet(ci_path)

if stats_path.exists():
    _stats_df = _safe_read_parquet(stats_path)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 RQ1: Profit Erosion by Category & Brand")
st.markdown(
    """
<p><strong>Research Question (RQ1):</strong> Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?</p>
<div style="margin-left: 1.5rem;">
<p><strong>Null Hypothesis (H₀):</strong> Mean profit erosion associated with returned items is equal across product categories and brands.</p>
<p><strong>Alternative Hypothesis (H₁):</strong> Mean profit erosion associated with returned items differs significantly across product categories and/or brands.</p>
</div>

**Method**: Kruskal–Wallis test with post-hoc Dunn comparisons and bootstrap confidence intervals.
""",
    unsafe_allow_html=True,
)
st.divider()


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
        ">Pipeline Output — Distributional Summary (Synthetic Dataset)</p>
        <p style="color: #e8eaf0; font-size: 1.0rem; line-height: 1.75; margin: 0;">
            <strong style="color: #ffffff;">Pipeline Demonstration — Distributional Output (Synthetic Dataset)</strong><br>
            On TheLook, the analytical pipeline surfaces statistically significant differences in profit erosion
            across product categories (ε² = 0.454, p &lt; 0.001) and brands (ε² = 0.442, p &lt; 0.001).
            Total financial loss is driven by two independent forces:
            <em>return frequency</em> (volume) and <em>cost per return</em> (severity) —
            illustrating the pipeline's ability to decompose erosion by product grouping.
            <strong style="color: #f0c040;">Decision: Reject H₀.</strong>
            Figures reflect the synthetic TheLook dataset; SSL directional validation confirms that
            category-level differences in profit erosion generalise in direction to real-world operational data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border: 0; border-top: 1px solid rgba(49,51,63,0.3); margin: 20px 0 24px 0;'>",
    unsafe_allow_html=True,
)


# ── KPI Cards ────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

if _cat_df is not None and "total_profit_erosion" in _cat_df.columns:
    total_erosion = float(_cat_df["total_profit_erosion"].sum())
    total_returns = int(_cat_df["returned_items"].sum()) if "returned_items" in _cat_df.columns else 0
    mean_erosion = total_erosion / total_returns if total_returns > 0 else 0.0
    top_cat = (
        _cat_df.nlargest(1, "total_profit_erosion")["category"].iloc[0]
        if "category" in _cat_df.columns else "N/A"
    )

    kpi1.metric("Total Profit Erosion", f"${total_erosion:,.0f}", help=_plain_tooltip("kpi_total_erosion"))
    kpi2.metric("Total Returns Analyzed", f"{total_returns:,}", help=_plain_tooltip("kpi_returns"))
    kpi3.metric("Mean Erosion per Return", f"${mean_erosion:,.2f}", help=_plain_tooltip("kpi_mean_erosion"))
    kpi4.metric("Highest-Risk Category (by total profit erosion)", top_cat, help=_plain_tooltip("kpi_top_cat"))
else:
    kpi1.metric("Total Profit Erosion", "N/A")
    kpi2.metric("Total Returns Analyzed", "N/A")
    kpi3.metric("Mean Erosion per Return", "N/A")
    kpi4.metric("Highest-Risk Category (by total)", "N/A")

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
        _tip_header("Loss Shape (Heavy-Tail Distribution)", "fig_dist")
        st.caption("Most returns are moderate losses — but a long tail of very large losses skews the data, making standard t-tests unreliable.")
        fig6_path = FIGURES_RQ1 / "fig6_profit_erosion_distribution_log.png"
        if fig6_path.exists():
            st.image(str(fig6_path), width='stretch')
        else:
            st.info("Figure not found: fig6_profit_erosion_distribution_log.png")

    with col_b:
        st.markdown(
            "<div style='background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;'>"
            "<span style='font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;'>"
            "STEP 2 — WHERE DOES LOSS CONCENTRATE?</span></div>",
            unsafe_allow_html=True,
        )
        _tip_header("Top Categories by Total Erosion", "fig_top_cat")
        st.caption("A small number of categories drive the majority of total erosion — the 80/20 pattern applies here.")
        fig1_path = FIGURES_RQ1 / "fig1_top_categories_total_erosion.png"
        if fig1_path.exists():
            st.image(str(fig1_path), width='stretch')
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
                st.info("Category data not available yet. Run the master notebook.")

    with col_c:
        st.markdown(
            "<div style='background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;'>"
            "<span style='font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;'>"
            "STEP 3 — WHAT TO DO ABOUT IT?</span></div>",
            unsafe_allow_html=True,
        )
        _tip_header("Severity × Volume — Two Levers", "fig_sev_vol")
        st.caption("Categories in the top-right quadrant are high on both dimensions — the priority targets. Others need different interventions.")
        fig5_path = FIGURES_RQ1 / "fig5_severity_vs_volume_category.png"
        if fig5_path.exists():
            st.image(str(fig5_path), width='stretch')
        else:
            st.info("Figure not found: fig5_severity_vs_volume_category.png")

    st.divider()

    # Statistical headline
    st.header("Statistical Evidence")
    if _stats_df is not None and "factor" in _stats_df.columns:
        cat_row = _stats_df[_stats_df["factor"] == "category"].iloc[0] if "category" in _stats_df["factor"].values else None
        brand_row = _stats_df[_stats_df["factor"] == "brand"].iloc[0] if "brand" in _stats_df["factor"].values else None

        _tip_header("Category-Level — Kruskal-Wallis", "stat_category")
        c1, c2, c3, c4 = st.columns(4)
        if cat_row is not None:
            _p_cat = cat_row['p_value']
            c1.metric("p-value", "< 0.001" if _p_cat < 0.001 else f"{_p_cat:.2e}",
                      help=f"Exact: {_p_cat:.2e}")
            c2.metric("Effect Size (ε²)", f"{cat_row['effect_size']:.3f}")
            c3.metric("Groups Tested", str(int(cat_row["n_groups"])))
            c4.metric("H₀ Decision", "✅ Rejected" if bool(cat_row["reject_h0"]) else "❌ Not Rejected")

        st.divider()

        _tip_header("Brand-Level — Kruskal-Wallis", "stat_brand")
        b1, b2, b3, b4 = st.columns(4)
        if brand_row is not None:
            _p_brand = brand_row['p_value']
            b1.metric("p-value", "< 0.001" if _p_brand < 0.001 else f"{_p_brand:.2e}",
                      help=f"Exact: {_p_brand:.2e}")
            b2.metric("Effect Size (ε²)", f"{brand_row['effect_size']:.3f}")
            b3.metric("Groups Tested", str(int(brand_row["n_groups"])))
            b4.metric("H₀ Decision", "✅ Rejected" if bool(brand_row["reject_h0"]) else "❌ Not Rejected")

        st.markdown(
            "<p style='color:#999; font-size:0.75rem; margin-top:4px;'>"
            "Effect size: ε² ≥ 0.06 = medium; ≥ 0.14 = large (Tomczak &amp; Tomczak, 2014). "
            "Both tests reject H₀."
            "</p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Statistical tests summary not found. Run the master notebook first.")

    st.divider()

    # Missing department figure (now included)
    st.header("Supporting View: Departments")
    _tip_header("Top Departments — Total Profit Erosion", "fig_top_dept")
    st.caption("Descriptive summary only — department-level differences are not formally tested in RQ1.")

    fig_dept_path = FIGURES_RQ1 / "fig4_top_departments_total_erosion.png"
    if fig_dept_path.exists():
        st.image(str(fig_dept_path), width='stretch')
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

    # Post-hoc and CI: keep but de-bloat via expanders
    st.header("Detailed Results (for review)")

    with st.expander("Post-hoc comparisons (Category & Brand)", expanded=False):
        st.caption(
            "Dunn's test with Bonferroni correction. Showing the most significant pairs first. "
            "Use the full tables only when you need detail."
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
                    st.dataframe(top10, width='stretch', hide_index=True)

                    with st.expander("Show full category post-hoc table", expanded=False):
                        full = ph_cat.copy()
                        full["p_adj"] = full["p_adj"].map(lambda x: f"{x:.6f}")
                        st.dataframe(full, width='stretch', hide_index=True)
                else:
                    st.dataframe(ph_cat, width='stretch', hide_index=True)
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
                    st.dataframe(top10, width='stretch', hide_index=True)

                    with st.expander("Show full brand post-hoc table", expanded=False):
                        full = ph_brand.copy()
                        full["p_adj"] = full["p_adj"].map(lambda x: f"{x:.6f}")
                        st.dataframe(full, width='stretch', hide_index=True)
                else:
                    st.dataframe(ph_brand, width='stretch', hide_index=True)
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
                st.dataframe(ci_display, width='stretch', hide_index=True)
        else:
            fig7_path = FIGURES_RQ1 / "fig7_bootstrap_ci_category_mean.png"
            if fig7_path.exists():
                st.image(str(fig7_path), width='stretch')
            else:
                st.info("Bootstrap CI data not found. Run the master notebook first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CATEGORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_cat:
    _tip_header("Top Categories — Total Profit Erosion", "fig_top_cat", level=2)
    fig1_path = FIGURES_RQ1 / "fig1_top_categories_total_erosion.png"
    if fig1_path.exists():
        st.image(str(fig1_path), width='stretch')

    st.divider()

    _tip_header("Return Rate vs Mean Erosion by Category (Interactive)", "fig_sev_vol", level=2)
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
                "Each point represents a category. The x-axis shows return rate, "
                "the y-axis shows mean erosion per return, and bubble size reflects total erosion."
            )
        else:
            fig3_path = FIGURES_RQ1 / "fig3_return_rate_vs_mean_erosion_category.png"
            if fig3_path.exists():
                st.image(str(fig3_path), width='stretch')
            else:
                st.info("Required columns not found for interactive scatter.")
    else:
        st.info("Category data not found. Run the master notebook first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BRAND
# ─────────────────────────────────────────────────────────────────────────────
with tab_brand:
    _tip_header("Top Brands — Total Profit Erosion", "fig_top_brand", level=2)
    fig2_path = FIGURES_RQ1 / "fig2_top_brands_total_erosion.png"
    if fig2_path.exists():
        st.image(str(fig2_path), width='stretch')
    else:
        if _brand_df is not None and {"brand", "total_profit_erosion"}.issubset(_brand_df.columns):
            tmp = _brand_df.nlargest(10, "total_profit_erosion").copy()
            fig = px.bar(
                tmp.sort_values("total_profit_erosion", ascending=True),
                x="total_profit_erosion",
                y="brand",
                orientation="h",
                title="Top Brands by Total Profit Erosion",
                labels={"total_profit_erosion": "Total Profit Erosion ($)", "brand": "Brand"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brand data not found yet.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
with tab_decomp:
    col_l, col_r = st.columns(2)
    with col_l:
        _tip_header("Severity vs Volume by Category", "fig_sev_vol")
        fig5_path = FIGURES_RQ1 / "fig5_severity_vs_volume_category.png"
        if fig5_path.exists():
            st.image(str(fig5_path), width='stretch')
        else:
            st.info("Figure not found: fig5_severity_vs_volume_category.png")

    with col_r:
        _tip_header("Profit Erosion Distribution (Log Scale)", "fig_dist")
        fig6_path = FIGURES_RQ1 / "fig6_profit_erosion_distribution_log.png"
        if fig6_path.exists():
            st.image(str(fig6_path), width='stretch')
        else:
            st.info("Figure not found: fig6_profit_erosion_distribution_log.png")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_val:
    _tip_header("External Validation — School Specialty LLC (SSL)", "ssl_validation", level=2)
    st.markdown(
        """
TheLook is a synthetic dataset. To strengthen external validity, the RQ1 category-level workflow was
replicated on a real-world returns dataset (SSL — School Specialty, Inc., 2025).

**Objective:** Confirm that profit erosion differs across product categories under operational conditions.

**SSL Category definition:**
Category is constructed by concatenating three SSL fields:
> `Pillar` + `-` + `Major Market Cat` + `-` + `Department`
> Example: `STEM-Science-Physics`

The same non-parametric workflow and erosion definition were applied. Brand and department are
**not** validated independently — only the composite category dimension is tested.

| Test | Dataset | Groups | p-value | Decision |
|---|---|---|---|---|
| Kruskal–Wallis (Category) | SSL | Pillar-MajorMarketCat-Dept | < 0.001 | ✅ Reject H₀ |

**Conclusion:** The replication on SSL data supports the RQ1 category-level finding — profit erosion
differs significantly across product categories under operational conditions. This strengthens
confidence that category-level differences are not artifacts of the synthetic dataset.
"""
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCLUSION
# ═════════════════════════════════════════════════════════════════════════════
with tab_conc:
    st.header("Conclusion — Pipeline Output")

    # ── Dark-gradient callout ─────────────────────────────────────────────────────────
    _conc_top_cat = "N/A"
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
            On TheLook, {_conc_total_returns:,} returned items are analysed across
            {_conc_n_cats} categories. Estimated total profit erosion is
            USD&nbsp;{_conc_total_erosion:,.2f} (mean USD&nbsp;{_conc_mean_erosion:,.2f} per returned item),
            combining margin reversal and category-tier processing costs.
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

    # ── Hypothesis Decision Table ──────────────────────────────────────────────────────────
    st.subheader("Hypothesis Decisions")

    _cat_es = "0.454"
    _brand_es_val = "0.442"
    if _stats_df is not None and "effect_size" in _stats_df.columns:
        _factor_col = [c for c in _stats_df.columns if "factor" in c.lower()]
        if _factor_col:
            _cat_row_c = _stats_df[
                _stats_df[_factor_col[0]].str.lower().str.contains("categ", na=False)
            ]
            _brd_row_c = _stats_df[
                _stats_df[_factor_col[0]].str.lower().str.contains("brand", na=False)
            ]
            if not _cat_row_c.empty:
                _cat_es = f"{float(_cat_row_c['effect_size'].iloc[0]):.3f}"
            if not _brd_row_c.empty:
                _brand_es_val = f"{float(_brd_row_c['effect_size'].iloc[0]):.3f}"

    st.markdown(
        f"""
| Hypothesis | Dataset | Test | Decision |
|---|---|---|---|
| **H₀₁**: Profit erosion equal across categories | TheLook | Kruskal–Wallis, ε² = {_cat_es}, p < 0.001 | ✅ **REJECT H₀** |
| **H₀₁**: Profit erosion equal across brands | TheLook | Kruskal–Wallis, ε² = {_brand_es_val}, p < 0.001 | ✅ **REJECT H₀** |
| Directional validation (categories only) | SSL | Kruskal–Wallis, p < 0.001 | ✅ Pattern confirmed — category level |
"""
    )
    st.caption(
        "SSL external validation covers category-level only "
        "(composite Pillar + MajorMarketCat + Dept). "
        "Brand and department are not independently validated on SSL data."
    )

    st.divider()

    # ── Top-3 Category Pipeline Output Panel ─────────────────────────────────────────────
    st.subheader("Top Categories Identified by the Pipeline")

    if _cat_df is not None and "category" in _cat_df.columns:
        _top3 = _cat_df.nlargest(3, "total_profit_erosion").reset_index(drop=True)
        _panel_cols = st.columns(3)
        _panel_colors = [
            ("#C62828", "#e57373"),
            ("#E65100", "#ffb74d"),
            ("#AD1457", "#f48fb1"),
        ]
        for _i, (_col, (_border, _accent)) in enumerate(zip(_panel_cols, _panel_colors)):
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
<div style="background:linear-gradient(135deg,{_border}55,{_border}33);
            border-left:4px solid {_border}; border-radius:6px; padding:16px;">
<h4 style="margin:0 0 8px 0;color:#ffffff;">#{_i + 1} — {_cat_name}</h4>
<p style="margin:0;font-size:13px;color:#f0f0f0;">
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
| **Highest-risk category** | {_conc_top_cat} (highest total modelled profit erosion) |
| **External validation (SSL)** | ✅ Directional patterns confirmed — profit erosion differs across categories (category level only) |
| **Framework contribution** | Pipeline decomposes erosion by product grouping; figures are from synthetic dataset |
"""
    )

st.caption(
    "DAMO-699-4 · University of Niagara Falls, Canada · Winter 2026 · "
    "RQ1 — Category & Brand Erosion Analysis"
)
