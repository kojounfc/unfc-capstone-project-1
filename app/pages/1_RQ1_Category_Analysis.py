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
        "**Category-Level Kruskal-Wallis:** Tests whether profit erosion differs across product categories. "
        "With p = 2.63 × 10⁻³³ and ε² = 0.454 (large), category differences are statistically significant "
        "and meaningful in practice."
    ),
    "stat_brand": (
        "**Brand-Level Kruskal-Wallis:** Tests whether profit erosion differs across brands. "
        "With p = 1.08 × 10⁻⁴ and ε² = 0.442 (large), brand differences are statistically significant "
        "and meaningful in practice."
    ),

    # Post-hoc
    "posthoc_cat": (
        "**Category Post-Hoc (Dunn + Bonferroni):** Shows which category pairs differ significantly. "
        "This helps explain which groups drive the overall Kruskal-Wallis result."
    ),
    "posthoc_brand": (
        "**Brand Post-Hoc (Dunn + Bonferroni):** Shows which brand pairs differ significantly. "
        "This helps identify which brands are associated with higher loss when returns happen."
    ),

    # Key findings
    "key_findings": (
        "**Key Takeaway:** Profit erosion is not evenly distributed. "
        "Category and brand both matter, and the differences are large enough to influence business decisions."
    ),

    # Figures
    "fig_top_cat": (
        "**Interpretation:** Total loss is concentrated in a small number of categories. "
        "This suggests you get the biggest impact by focusing mitigation efforts on these categories first."
    ),
    "fig_top_brand": (
        "**Interpretation:** Total loss is concentrated in a small number of brands. "
        "This suggests supplier/brand differences play an important role in return-related financial risk."
    ),
    "fig_top_dept": (
        "**Interpretation:** Some departments contribute more total loss than others. "
        "This helps prioritize monitoring and mitigation at a higher organizational level (beyond categories/brands)."
    ),
    "fig_dist": (
        "**Interpretation:** Most returns generate moderate losses, but a small number generate very large losses. "
        "This heavy tail is why we use non-parametric tests instead of relying on normality."
    ),
    "fig_sev_vol": (
        "**Interpretation:** Total loss comes from two sources: (1) how often returns happen (volume) "
        "and (2) how costly each return is (severity). Different categories can be high-risk for different reasons."
    ),
    "fig_ci": (
        "**Interpretation:** Bootstrap confidence intervals show the uncertainty around category averages. "
        "Limited overlap increases confidence that category differences are reliable."
    ),

    # Validation
    "ssl_validation": (
        "**External Validation (SSL):** Running the same workflow on real-world SSL returns data "
        "supports the same conclusion: profit erosion differs across product groupings. "
        "This strengthens confidence that the findings are not dataset-specific."
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

cat_path = PROCESSED_RQ1 / "rq1_product_profit_erosion_by_category.parquet"
brand_path = PROCESSED_RQ1 / "rq1_product_profit_erosion_by_brand.parquet"
dept_path = PROCESSED_RQ1 / "rq1_product_profit_erosion_by_department.parquet"
ci_path = PROCESSED_RQ1 / "rq1_bootstrap_ci_category_mean.parquet"
stats_path = PROCESSED_RQ1 / "rq1_statistical_tests_summary.parquet"

if cat_path.exists():
    _cat_df = _safe_read_parquet(cat_path)

if brand_path.exists():
    _brand_df = _safe_read_parquet(brand_path)

if dept_path.exists():
    _dept_df = _safe_read_parquet(dept_path)

if ci_path.exists():
    _ci_df = _safe_read_parquet(ci_path)

if stats_path.exists():
    _stats_df = _safe_read_parquet(stats_path)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 RQ1: Profit Erosion by Category & Brand")
st.markdown(
    """
**Research Question**: Do profit erosion levels differ significantly across product categories and brands?

**Method**: Kruskal–Wallis test with post-hoc Dunn comparisons and bootstrap confidence intervals.
"""
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
        ">Executive Summary — Key Findings &amp; Implications</p>
        <p style="color: #e8eaf0; font-size: 1.0rem; line-height: 1.75; margin: 0;">
            <strong style="color: #ffffff;">Returns are not equally expensive.</strong>
            Profit erosion differs significantly across product categories and brands,
            and these differences are statistically significant and substantively meaningful in practice.
            Total financial loss is driven by two independent forces:
            <em>return frequency</em> (volume) and <em>cost per return</em> (severity).
            Because categories can be high-risk for different reasons,
            mitigation strategies should be targeted rather than uniform.
            <strong style="color: #f0c040;">Strategic implication:</strong>
            high-severity categories require per-unit quality and fit improvements,
            while high-volume categories require systemic process improvements
            such as policy design and demand planning.
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
tab_ov, tab_cat, tab_brand, tab_decomp, tab_val = st.tabs([
    "📋 Overview",
    "📊 Category",
    "🏷️ Brand",
    "🔍 Decomposition",
    "🌐 Validation",
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
            st.image(str(fig6_path), use_container_width=True)
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
            st.image(str(fig5_path), use_container_width=True)
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
                st.info("Bootstrap CI data not found. Run the master notebook first.")


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
        st.info("Category data not found. Run the master notebook first.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — BRAND
# ─────────────────────────────────────────────────────────────────────────────
with tab_brand:
    _tip_header("Top Brands — Total Profit Erosion", "fig_top_brand", level=2)
    fig2_path = FIGURES_RQ1 / "fig2_top_brands_total_erosion.png"
    if fig2_path.exists():
        st.image(str(fig2_path), use_container_width=True)
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
            st.image(str(fig5_path), use_container_width=True)
        else:
            st.info("Figure not found: fig5_severity_vs_volume_category.png")

    with col_r:
        _tip_header("Profit Erosion Distribution (Log Scale)", "fig_dist")
        fig6_path = FIGURES_RQ1 / "fig6_profit_erosion_distribution_log.png"
        if fig6_path.exists():
            st.image(str(fig6_path), use_container_width=True)
        else:
            st.info("Figure not found: fig6_profit_erosion_distribution_log.png")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_val:
    _tip_header("External Validation — School Specialty LLC (SSL)", "ssl_validation", level=2)
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

**Conclusion:** The replication on SSL data supports the RQ1 findings — profit erosion differs
significantly across product groupings under operational conditions. This strengthens confidence
that category- and brand-level differences are not artifacts of the synthetic dataset.
"""
    )
