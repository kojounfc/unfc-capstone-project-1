"""
RQ2: Customer Behavioral Segments with Differential Profit Erosion
Structured identically to RQ1 — CSS hover tooltips, executive summary banner, tab layout.

Tabs:
  📋 Overview     — KPIs, hypothesis results, integrated story
  📈 Concentration — Pareto, Lorenz, feature concentration ranking
  👥 Segmentation  — Cluster profiles, diagnostics, feature importance, explorer
  🌐 Validation   — External SSL pattern validation
  🎯 Conclusion   — Integrated interpretation & action plan
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RQ2 – Customer Segments",
    page_icon="👥",
    layout="wide",
)

# ── CSS: same hover tooltip system as RQ1 ─────────────────────────────────────
st.markdown("""
<style>
.rq2-tip-title {
    display: flex; align-items: center; margin-bottom: 0.4rem;
}
.rq2-tip-title h2 { margin:0; padding:0; font-size:1.5rem; font-weight:700; letter-spacing:-0.01em; }
.rq2-tip-title h3 { margin:0; padding:0; font-size:1.35rem; font-weight:600; letter-spacing:-0.01em; }
.rq2-tip {
    position: relative; display: inline-flex; align-items: center;
    cursor: help; margin-left: 10px; flex-shrink: 0;
}
.rq2-tip-icon { font-size: 0.9rem; color: #888; user-select: none; }
.rq2-tip-box {
    visibility: hidden; opacity: 0; width: 380px;
    background-color: rgba(28, 28, 44, 0.97); color: #e4e4f0;
    text-align: left; border-radius: 8px; padding: 14px 18px;
    font-size: 0.95rem; line-height: 1.65;
    position: absolute; z-index: 9999;
    bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%);
    transition: opacity 0.2s ease;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    pointer-events: none; white-space: normal;
}
.rq2-tip-box::after {
    content: ""; position: absolute; top: 100%; left: 50%; margin-left: -6px;
    border: 6px solid transparent; border-top-color: rgba(28,28,44,0.97);
}
.rq2-tip:hover .rq2-tip-box { visibility: visible; opacity: 1; }
.step-badge {
    background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;
    font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
FIGURES_RQ2   = ROOT / "figures" / "rq2"
PROCESSED_RQ2 = ROOT / "data" / "processed" / "rq2"

# ── Tooltips ──────────────────────────────────────────────────────────────────
_TOOLTIPS = {
    "kpi_customers": (
        "**Customers with Returns:** Total unique customers with at least one returned item "
        "in the analysis population. Only customers with profit erosion > 0 are included."
    ),
    "kpi_erosion": (
        "**Total Profit Erosion:** Sum of margin reversal + processing cost across all "
        "returned items for the returning-customer population."
    ),
    "kpi_gini": (
        "**Gini Coefficient:** Measures inequality of profit erosion distribution. "
        "0 = perfectly equal (every customer causes the same loss). "
        "1 = one customer causes everything. "
        "A moderate Gini means the problem is uneven but not extreme."
    ),
    "kpi_top20": (
        "**Top 20% Share:** The top 20% of returning customers (ranked by total erosion) "
        "account for this share of all profit erosion. "
        "The Pareto principle in action — a small subset drives a disproportionate share of losses."
    ),
    "kpi_h01": (
        "**Concentration Finding (descriptive):** Gini coefficient = 0.409 with bootstrap "
        "p < 0.0001 confirms that profit erosion is significantly unequal across customers. "
        "This is the foundation that motivates the segmentation in H₀₂ — if erosion were uniform, "
        "segmentation would have no practical value."
    ),
    "kpi_h02": (
        "**H₀₂ Result (RQ2 primary hypothesis):** H₀₂ states that customer segments identified "
        "through clustering do not differ significantly in mean profit erosion. "
        "Both ANOVA (F=1,479.64, p < 0.0001) and Kruskal-Wallis (H=893.49, p < 0.0001) reject H₀₂. "
        "η² = 0.112 means cluster membership explains 11.2% of erosion variance — a medium effect."
    ),
    "fig_pareto": (
        "**Pareto Curve:** Ranks customers highest to lowest by erosion and shows cumulative share "
        "of total losses. A curve bowing above the 45 degree diagonal means erosion is unequal. "
        "The dashed crosshairs mark the 20% customer / top-20%-share crossover point."
    ),
    "fig_lorenz": (
        "**Lorenz Curve:** The standard tool for visualising inequality. "
        "The dashed diagonal = perfect equality. The curve bows below it — "
        "the further the bow, the more concentrated the problem. "
        "The shaded area between the two lines is proportional to the Gini coefficient."
    ),
    "fig_feat_conc": (
        "**Feature Concentration Ranking:** Each behavioral feature is scored by its own Gini "
        "coefficient — how unequally erosion is distributed when customers are ranked by that feature. "
        "purchase_recency_days tops the list (Gini=0.528), making recency-based alerting the most "
        "precise early-warning signal available."
    ),
    "fig_gini_pareto": (
        "**Gini vs Pareto Cross-Validation:** Two independent concentration measures plotted "
        "against each other for every feature. The tight linear relationship confirms both "
        "measures agree — the ranking is not an artefact of metric choice. "
        "Features in the top-right quadrant are highest priority for targeting."
    ),
    "fig_cluster_erosion": (
        "**Cluster Erosion Comparison:** Side-by-side comparison of mean and median profit erosion "
        "for each K-Means cluster. Cluster 0 (avg $95.51) is 1.80x higher than Cluster 1 ($53.07). "
        "Despite smaller headcount, it drives nearly equal total dollars."
    ),
    "fig_diagnostics": (
        "**Clustering Diagnostics:** Silhouette score peaks at k=2 (score=0.284) — the global "
        "maximum across k=2 to 8. The elbow curve confirms the same inflection. "
        "k=2 is not a simplification — it is the statistically optimal answer."
    ),
    "fig_feat_imp": (
        "**Feature Importance (ANOVA F-statistic):** After clustering, one-way ANOVA on each "
        "feature measures how strongly it discriminates between the two clusters. "
        "order_frequency dominates (F=12,486, eta squared=0.514) — frequent buyers return more and cost more."
    ),
    "fig_explorer": (
        "**Cluster Explorer:** Each point is one customer coloured by cluster assignment. "
        "Clear separation confirms the two archetypes are behaviorally distinct. "
        "Teal = Cluster 0 (high-erosion). Orange = Cluster 1 (lower-erosion)."
    ),
    "ssl_validation": (
        "**External Validation (SSL):** Since RQ2 uses unsupervised clustering, we use Level 1 "
        "Pattern Validation — testing whether the same behavioral features that discriminate "
        "high-loss customers in TheLook also do so in the external SSL dataset. "
        "Agreement >= 50% confirms cross-domain validity."
    ),
    "conclusion": (
        "**Integrated Interpretation:** Concentration identifies WHO to target (top 20%). "
        "Segmentation reveals HOW to differentiate intervention (Cluster 0 vs Cluster 1). "
        "The optimal targeting zone is the top 20% of Cluster 0 — smallest group, highest impact."
    ),
}


def _tip_header(label: str, tooltip_key: str, level: int = 3) -> None:
    """Render a section header with an inline CSS hover tooltip — identical to RQ1."""
    raw = _TOOLTIPS[tooltip_key]
    parts = raw.split("**")
    tip_html = "".join(
        f"<strong>{p}</strong>" if i % 2 == 1 else p
        for i, p in enumerate(parts)
    )
    st.markdown(
        f'<div class="rq2-tip-title">'
        f'<h{level}>{label}</h{level}>'
        f'<span class="rq2-tip">'
        f'<span class="rq2-tip-icon">ℹ️</span>'
        f'<span class="rq2-tip-box">{tip_html}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _plain_tip(key: str) -> str:
    return _TOOLTIPS[key].replace("**", "")


def _safe_parquet(path: Path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ── Load data ─────────────────────────────────────────────────────────────────
_conc       = {}
_valid      = {}
_cs_df      = None
_pareto_df  = None
_cluster_df = None
_feat_conc  = None
_feat_imp   = None
_elbow_df   = None
_sil_df     = None
_lorenz_df  = None

for path, key in [
    (PROCESSED_RQ2 / "profit_erosion_concentration_metrics.json", "conc"),
    (PROCESSED_RQ2 / "rq2_validation_results.json",               "valid"),
]:
    if path.exists():
        with open(path) as f:
            if key == "conc": _conc  = json.load(f)
            else:             _valid = json.load(f)

for path, attr in [
    (PROCESSED_RQ2 / "cluster_summary.parquet",               "_cs_df"),
    (PROCESSED_RQ2 / "pareto_table.parquet",                  "_pareto_df"),
    (PROCESSED_RQ2 / "clustered_customers.parquet",           "_cluster_df"),
    (PROCESSED_RQ2 / "feature_concentration_ranking.parquet", "_feat_conc"),
    (PROCESSED_RQ2 / "clustering_feature_importance.parquet", "_feat_imp"),
    (PROCESSED_RQ2 / "elbow_df.parquet",                      "_elbow_df"),
    (PROCESSED_RQ2 / "silhouette_df.parquet",                 "_sil_df"),
]:
    if Path(path).exists():
        globals()[attr] = _safe_parquet(path)

for csv_path, attr in [
    (PROCESSED_RQ2 / "clustering_feature_importance.csv", "_feat_imp"),
    (PROCESSED_RQ2 / "feature_concentration_ranking.csv", "_feat_conc"),
    (PROCESSED_RQ2 / "lorenz_curve_points.csv",           "_lorenz_df"),
]:
    if globals().get(attr) is None and Path(csv_path).exists():
        globals()[attr] = pd.read_csv(csv_path)

# Derived scalars
_gini      = _conc.get("gini_coefficient", None)
_top20_raw = _conc.get("top_20_pct_share", None)
_top20     = (_top20_raw * 100 if isinstance(_top20_raw, float) and _top20_raw < 1 else _top20_raw)
_boot      = _conc.get("bootstrap_test", {})
_top50     = _conc.get("top_50_impact",  {})

# Shared chart theme
CHART_H = 400
LAYOUT  = dict(height=CHART_H, margin=dict(t=36, b=40, l=10, r=10),
               plot_bgcolor="white", paper_bgcolor="white",
               font=dict(family="Inter, sans-serif", size=12))

# ── Page header ───────────────────────────────────────────────────────────────
st.title("👥 RQ2: Customer Behavioral Segments & Profit Erosion")
st.markdown("""
**Research Question (RQ2)**: Can unsupervised learning identify distinct customer behavioral
segments, and do these segments differ significantly in profit erosion intensity?

**H₀₂ (Null):** Customer segments identified through clustering algorithms do not differ
significantly in mean profit erosion from returns.

**H₁₂ (Alternative):** Customer segments identified through clustering algorithms exhibit
statistically significant differences in mean profit erosion from returns.

**Method**: K-Means clustering · ANOVA + Kruskal-Wallis · Gini / Lorenz concentration analysis ·
External pattern validation (SSL)
""")
st.divider()

# ── Executive Summary Banner ──────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #0f2440 0%, #1a3660 100%);
    border-left: 5px solid #00897B;
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 8px;
">
    <p style="color:#f0c040; font-size:0.78rem; font-weight:700;
              letter-spacing:0.12em; text-transform:uppercase; margin:0 0 10px 0;">
        Executive Summary — Key Findings &amp; Implications
    </p>
    <p style="color:#e8eaf0; font-size:1.0rem; line-height:1.75; margin:0;">
        <strong style="color:#ffffff;">Profit erosion is both concentrated and segmented.</strong>
        The top 20% of returning customers drive nearly half of all losses (Gini = 0.409, p &lt; 0.0001),
        and K-Means clustering reveals two statistically distinct archetypes
        (ANOVA F = 1,479.64, p &lt; 0.0001, &eta;&sup2; = 0.112).
        These two findings are complementary &mdash; concentration tells you
        <em>who</em> to target, segmentation tells you <em>how</em> to intervene differently.
        <strong style="color:#f0c040;">Strategic implication:</strong>
        Focus Cluster 0 (frequent buyers, avg $95.51 loss) on personalised fit guidance and loyalty
        retention, while applying lighter-touch policy guardrails to Cluster 1 (avg $53.07).
        The highest-leverage signal is <strong style="color:#80cbc4;">purchase_recency_days</strong>
        (Gini = 0.528) &mdash; the most concentrated behavioral feature.
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown(
    "<hr style='border:0; border-top:1px solid rgba(49,51,63,0.3); margin:20px 0 24px 0;'>",
    unsafe_allow_html=True,
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Top 20% Share",
          f"{_top20:.1f}%" if isinstance(_top20, float) else "N/A",
          "of total profit erosion", help=_plain_tip("kpi_top20"))
k2.metric("H₀₂ Result", "Rejected", "p < 0.0001", help=_plain_tip("kpi_h02"))
k3.metric("Customer Segments", "2", "K-Means clusters",
          help="K-Means with k=2 — statistically optimal (silhouette peak at k=2).")
k4.metric("Gini Coefficient",
          f"{_gini:.3f}" if isinstance(_gini, float) else "N/A",
          "Moderate concentration", help=_plain_tip("kpi_gini"))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_conc, tab_seg, tab_val, tab_end = st.tabs([
    "📋 Overview",
    "📈 Concentration",
    "👥 Segmentation",
    "🌐 Validation",
    "🎯 Conclusion",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ov:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown('<div class="step-badge">STEP 1 — IS THE PROBLEM CONCENTRATED?</div>',
                    unsafe_allow_html=True)
        _tip_header("Lorenz Curve — Inequality", "fig_lorenz")
        st.caption("The curve bows below the equality diagonal — a small subset drives a "
                   "disproportionate share of total profit erosion.")
        if _lorenz_df is not None:
            pop_col = next((c for c in ["cumulative_population","cum_population","customer_share","x"]
                            if c in _lorenz_df.columns), None)
            val_col = next((c for c in ["cumulative_value","cum_value","value_share","y"]
                            if c in _lorenz_df.columns), None)
            if pop_col and val_col:
                lsrc = _lorenz_df.sort_values(pop_col)
                scale = 100 if lsrc[pop_col].max() <= 1.0 else 1
                lx = np.concatenate([[0], lsrc[pop_col].values * scale])
                ly = np.concatenate([[0], lsrc[val_col].values * scale])
                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines",
                    line=dict(dash="dash", color="#999", width=1),
                    hoverinfo="skip", showlegend=False))
                fig_ml.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                    line=dict(color="#E65100", width=2), fill="tonexty",
                    fillcolor="rgba(230,81,0,0.10)", showlegend=False,
                    hovertemplate="Bottom %{x:.1f}%<br>→ %{y:.1f}% of erosion<extra></extra>"))
                fig_ml.update_layout(height=300, margin=dict(t=10,b=40,l=50,r=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_title="Cumulative % of Customers",
                    yaxis_title="Cumulative % of Erosion")
                st.plotly_chart(fig_ml, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "lorenz_curve.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)

    with col_b:
        st.markdown('<div class="step-badge">STEP 2 — WHO ARE THE HIGH-RISK CUSTOMERS?</div>',
                    unsafe_allow_html=True)
        _tip_header("Two Customer Archetypes (k=2)", "fig_cluster_erosion")
        st.caption("K-Means reveals two distinct groups. Cluster 0 costs 1.80x more per customer "
                   "— a clear priority target.")
        if _cs_df is not None:
            id_col = next((c for c in ["cluster_id","Cluster","cluster"] if c in _cs_df.columns), None)
            if id_col and "Mean_Erosion" in _cs_df.columns:
                cs = _cs_df.copy()
                cs["Label"] = "Cluster " + cs[id_col].astype(str)
                fig_mc = px.bar(cs, x="Label", y="Mean_Erosion", color="Label",
                    color_discrete_map={"Cluster 0":"#00897B","Cluster 1":"#E64A19"},
                    text=cs["Mean_Erosion"].map("${:,.2f}".format),
                    labels={"Mean_Erosion":"Mean Erosion ($)","Label":""})
                fig_mc.update_traces(textposition="outside", cliponaxis=False)
                fig_mc.update_layout(height=300, margin=dict(t=50,b=40,l=40,r=10),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                    yaxis=dict(range=[0, cs["Mean_Erosion"].max() * 1.30]))
                st.plotly_chart(fig_mc, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "cluster_erosion_comparison.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)

    with col_c:
        st.markdown('<div class="step-badge">STEP 3 — WHAT DRIVES THEM?</div>',
                    unsafe_allow_html=True)
        _tip_header("Top Concentrated Features", "fig_feat_conc")
        st.caption("purchase_recency_days (Gini=0.528) is the most concentrated signal — "
                   "recently active customers drive a disproportionate share of losses.")
        if _feat_conc is not None:
            fc_top = _feat_conc.sort_values("gini_coefficient", ascending=False).head(8)
            fc_plot = fc_top.sort_values("gini_coefficient", ascending=True)
            fig_mf = px.bar(fc_plot, x="gini_coefficient", y="feature", orientation="h",
                color="gini_coefficient", color_continuous_scale="Reds",
                text=fc_plot["gini_coefficient"].map("{:.3f}".format),
                labels={"gini_coefficient":"Gini","feature":""})
            fig_mf.update_traces(textposition="outside", cliponaxis=False)
            fig_mf.update_layout(height=300, margin=dict(t=10,b=40,l=10,r=60),
                plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False,
                xaxis=dict(range=[0, fc_plot["gini_coefficient"].max() * 1.35]))
            st.plotly_chart(fig_mf, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "feature_concentration_ranking.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)

    st.divider()
    st.header("Statistical Evidence")

    h1, h2 = st.columns(2)
    with h1:
        st.markdown("#### Concentration Analysis (descriptive foundation for H₀₂)")
        st.caption("Gini coefficient and bootstrap test confirm erosion is significantly unequal — "
                   "motivating the segmentation hypothesis.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Gini Coefficient",
                  f"{_gini:.3f}" if isinstance(_gini, float) else "N/A",
                  help=_plain_tip("kpi_gini"))
        _boot_p = _boot.get("p_value", None)
        _boot_p_str = (
            "< 0.0001" if _boot_p is None or (isinstance(_boot_p, float) and _boot_p < 0.0001)
            else f"{_boot_p:.4f}"
        )
        m2.metric("Bootstrap p-value", _boot_p_str,
                  help="p-value from 1,000-resample bootstrap test. Confirms concentration is not due to chance.")
        m3.metric("Finding", "p < 0.0001", help=_plain_tip("kpi_h01"))

    with h2:
        st.markdown("#### H₀₂ — Segments do not differ in mean profit erosion")
        st.caption("Primary RQ2 hypothesis test. Both parametric and non-parametric tests applied.")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("ANOVA F-stat", "1,479.64",
            help="One-way ANOVA F-statistic comparing mean erosion across clusters.")
        n2.metric("KW H-stat", "893.49",
            help="Kruskal-Wallis H — non-parametric equivalent, robust to non-normality.")
        n3.metric("η² effect size", "0.112",
            help="11.2% of erosion variance explained by cluster membership (medium effect).")
        n4.metric("H₀₂ Decision", "Rejected", help=_plain_tip("kpi_h02"))

    st.markdown(
        "<p style='color:#999; font-size:0.75rem; margin-top:4px;'>"
        "H₀₂ rejected at p &lt; 0.0001 with independent parametric (ANOVA) and non-parametric "
        "(Kruskal-Wallis) tests. Effect size η² ≥ 0.06 = medium (Tomczak &amp; Tomczak, 2014)."
        "</p>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_conc:

    st.divider()
    col_p, col_l = st.columns(2)

    with col_p:
        _tip_header("Pareto Curve — How Concentrated is the Loss?", "fig_pareto")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(f"""
- Curve bowing **above** the 45° diagonal = erosion is unequal
- Dashed crosshairs = **20% customer / {_top20:.1f}% erosion** crossover
- Top 20% of returning customers cause **{_top20:.1f}%** of all profit erosion
""" if isinstance(_top20, float) else "Ranks customers by erosion — shows cumulative share of losses.")
        if _pareto_df is not None and "customer_share" in _pareto_df.columns:
            pdf = _pareto_df.sort_values("customer_share")
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(
                x=pdf["customer_share"]*100, y=pdf["value_share"]*100,
                mode="lines", name="Cumulative Erosion Share",
                line=dict(color="#C62828", width=2.5),
                fill="tozeroy", fillcolor="rgba(198,40,40,0.08)",
                hovertemplate="Top %{x:.1f}% of customers<br>→ %{y:.1f}% of total erosion<extra></extra>"))
            fig_p.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Perfect Equality",
                line=dict(dash="dash", color="#999", width=1), hoverinfo="skip"))
            if isinstance(_top20, float):
                fig_p.add_vline(x=20, line_dash="dot", line_color="#555", line_width=1.2)
                fig_p.add_hline(y=_top20, line_dash="dot", line_color="#555", line_width=1.2,
                    annotation_text=f"  {_top20:.1f}% of erosion",
                    annotation_position="bottom right", annotation_font_size=11)
            fig_p.update_layout(**LAYOUT,
                xaxis_title="Cumulative % of Customers (ranked by erosion)",
                yaxis_title="Cumulative % of Total Profit Erosion",
                legend=dict(x=0.05, y=0.95))
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "pareto_curve.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)
            else:
                st.warning("Pareto data not found.")

    with col_l:
        _tip_header(
            f"Lorenz Curve — Inequality (Gini = {_gini:.3f})" if isinstance(_gini, float) else "Lorenz Curve",
            "fig_lorenz")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(f"""
- **Dashed diagonal** = perfect equality
- **Orange curve** bows *below* the diagonal — further bow = more concentrated
- **Shaded area** is proportional to Gini = **{_gini:.3f}** (moderate)
- Targeting the high-erosion tail addresses nearly half the total problem
""" if isinstance(_gini, float) else "Further bow below diagonal = more unequal distribution.")
        if _pareto_df is not None and "customer_share" in _pareto_df.columns:
            if _lorenz_df is not None:
                pop_col = next((c for c in ["cumulative_population","cum_population","customer_share","x"]
                                if c in _lorenz_df.columns), None)
                val_col = next((c for c in ["cumulative_value","cum_value","value_share","y"]
                                if c in _lorenz_df.columns), None)
                if pop_col and val_col:
                    lsrc  = _lorenz_df.sort_values(pop_col)
                    scale = 100 if lsrc[pop_col].max() <= 1.0 else 1
                    lx = np.concatenate([[0], lsrc[pop_col].values * scale])
                    ly = np.concatenate([[0], lsrc[val_col].values * scale])
                else:
                    pdf_r = _pareto_df.sort_values("customer_share")
                    lx = np.concatenate([[0], (1 - pdf_r["customer_share"].values[::-1]) * 100])
                    ly = np.concatenate([[0], (1 - pdf_r["value_share"].values[::-1]) * 100])
            else:
                pdf_r = _pareto_df.sort_values("customer_share")
                lx = np.concatenate([[0], (1 - pdf_r["customer_share"].values[::-1]) * 100])
                ly = np.concatenate([[0], (1 - pdf_r["value_share"].values[::-1]) * 100])

            fig_l = go.Figure()
            fig_l.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Perfect Equality",
                line=dict(dash="dash", color="#999", width=1.5), hoverinfo="skip"))
            fig_l.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                name=f"Lorenz Curve (Gini={_gini:.3f})" if isinstance(_gini, float) else "Lorenz Curve",
                line=dict(color="#E65100", width=2.5),
                fill="tonexty", fillcolor="rgba(230,81,0,0.12)",
                hovertemplate="Bottom %{x:.1f}% of customers<br>→ %{y:.1f}% of total erosion<extra></extra>"))
            fig_l.update_layout(**LAYOUT,
                xaxis_title="Cumulative % of Customers",
                yaxis_title="Cumulative % of Profit Erosion",
                legend=dict(x=0.05, y=0.95))
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "lorenz_curve.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)
            else:
                st.warning("Lorenz data not found.")

    st.divider()

    col_fc, col_gp = st.columns(2)

    with col_fc:
        _tip_header("Feature Concentration Ranking", "fig_feat_conc")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown("""
Each behavioral feature scored by Gini — how unequally erosion is distributed when customers
are ranked by that feature.

| Level | Gini range |
|---|---|
| Low | < 0.3 |
| Moderate | 0.3 – 0.6 |
| High | > 0.6 |

**purchase_recency_days tops the list (Gini=0.528)**
""")
        if _feat_conc is not None:
            fc = _feat_conc.sort_values("gini_coefficient", ascending=True).copy()
            color_col = "concentration_level" if "concentration_level" in fc.columns else None
            fig_rank = px.bar(fc, x="gini_coefficient", y="feature", orientation="h",
                color=color_col if color_col else "gini_coefficient",
                color_discrete_map={"High":"#C62828","Moderate":"#EF6C00","Low":"#1565C0"},
                color_continuous_scale="Reds",
                text=fc["gini_coefficient"].map("{:.3f}".format),
                labels={"gini_coefficient":"Gini Coefficient","feature":"Feature","concentration_level":"Level"},
                title="Feature Concentration Ranking (Gini Coefficient)")
            fig_rank.update_traces(textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>Gini: %{x:.3f}<extra></extra>")
            rank_layout = {**LAYOUT, "height": max(CHART_H, len(fc)*40)}
            fig_rank.update_layout(**rank_layout,
                yaxis=dict(categoryorder="total ascending"),
                xaxis=dict(range=[0, fc["gini_coefficient"].max()*1.30]),
                coloraxis_showscale=False, showlegend=bool(color_col))
            st.plotly_chart(fig_rank, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "feature_concentration_ranking.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)
            else:
                st.warning("Feature concentration data not found.")

    with col_gp:
        _tip_header("Gini vs Pareto Cross-Validation", "fig_gini_pareto")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown("""
Two independent concentration measures plotted against each other.
The tight linear relationship confirms both measures agree.
Features in the **top-right quadrant** are highest priority for targeting.
""")
        if _feat_conc is not None and "top_20_pct_share" in _feat_conc.columns:
            fc2 = _feat_conc.copy()
            if fc2["top_20_pct_share"].max() <= 1.0:
                fc2["top_20_pct_share"] = fc2["top_20_pct_share"] * 100
            color_col2 = "concentration_level" if "concentration_level" in fc2.columns else None
            fig_cross = px.scatter(fc2, x="gini_coefficient", y="top_20_pct_share", text="feature",
                color=color_col2 if color_col2 else "gini_coefficient",
                color_discrete_map={"High":"#C62828","Moderate":"#EF6C00","Low":"#1565C0"},
                color_continuous_scale="Reds",
                labels={"gini_coefficient":"Gini Coefficient","top_20_pct_share":"Top 20% Share (%)","concentration_level":"Level"},
                title="Gini vs Pareto — Do Both Measures Agree?")
            fig_cross.update_traces(textposition="top center", marker=dict(size=10),
                hovertemplate="<b>%{text}</b><br>Gini: %{x:.3f}<br>Top 20%%: %{y:.1f}%%<extra></extra>")
            fig_cross.update_layout(**LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_cross, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "concentration_gini_vs_pareto.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)
            else:
                st.info("Gini vs Pareto figure not found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_seg:

    st.markdown("""
**RQ2 Primary Hypothesis Test**

> **H₀₂ (Null):** Customer segments identified through clustering algorithms do not differ
> significantly in mean profit erosion from returns.
>
> **H₁₂ (Alternative):** Customer segments identified through clustering algorithms exhibit
> statistically significant differences in mean profit erosion from returns.

K-Means clustering on 8 screened behavioral features (highly correlated features with r > 0.85 removed).
""")
    st.success(
        "H₀₂ REJECTED — ANOVA: F = 1,479.64, p < 0.0001 · "
        "Kruskal-Wallis: H = 893.49, p < 0.0001 · η² = 0.112 (medium effect). "
        "Clusters differ significantly in mean profit erosion — H₁₂ is supported.",
        icon="✅",
    )

    col_l, col_r = st.columns(2)

    with col_l:
        _tip_header("Cluster Erosion Comparison", "fig_cluster_erosion")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown("""
| Cluster | Size | Avg Erosion | Insight |
|---------|------|-------------|---------|
| **Cluster 0** | n ≈ 4,302 | $95.51 | 1.80x higher — priority intervention target |
| **Cluster 1** | n ≈ 7,488 | $53.07 | Manageable with lighter-touch responses |
""")
        if _cs_df is not None:
            id_col = next((c for c in ["cluster_id","Cluster","cluster"] if c in _cs_df.columns), None)
            melt_cols = [c for c in ["Mean_Erosion","Median_Erosion"] if c in _cs_df.columns]
            if id_col and melt_cols:
                melted = _cs_df[[id_col]+melt_cols].melt(id_vars=id_col, var_name="Metric", value_name="Value")
                melted["Cluster"] = "Cluster " + melted[id_col].astype(str)
                fig_cs = px.bar(melted, x="Cluster", y="Value", color="Metric", barmode="group",
                    color_discrete_map={"Mean_Erosion":"#C62828","Median_Erosion":"#EF9A9A"},
                    text=melted["Value"].map("${:,.2f}".format),
                    labels={"Value":"Profit Erosion ($)","Metric":""},
                    title="Mean vs Median Profit Erosion by Cluster")
                fig_cs.update_traces(textposition="outside", cliponaxis=False,
                    hovertemplate="<b>%{x}</b><br>%{data.name}: $%{y:,.2f}<extra></extra>")
                cs_layout = {**LAYOUT, "height": CHART_H + 40,
                             "margin": dict(t=60, b=40, l=10, r=10)}
                fig_cs.update_layout(**cs_layout,
                    yaxis_title="Profit Erosion ($)",
                    yaxis=dict(range=[0, melted["Value"].max() * 1.30]),
                    legend=dict(orientation="h", y=1.05, x=0))
                st.plotly_chart(fig_cs, use_container_width=True)

    with col_r:
        _tip_header("Why k = 2? Diagnostics Confirm the Choice", "fig_diagnostics")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown("""
| Method | Result |
|--------|--------|
| **Silhouette** | Peaks at k=2 (score=0.284) — global max across k=2 to 8 |
| **Elbow (inertia)** | Inflection point at k=2 |

**k=2 is not a simplification — it is the statistically optimal answer.**
""")
        if _elbow_df is not None and _sil_df is not None:
            fig_diag = make_subplots(rows=1, cols=2,
                subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])
            fig_diag.add_trace(go.Scatter(x=_elbow_df["k"], y=_elbow_df["inertia"],
                mode="lines+markers", name="Inertia",
                line=dict(color="#1565C0", width=2), marker=dict(size=7),
                hovertemplate="k=%{x}<br>Inertia: %{y:,.0f}<extra></extra>"), row=1, col=1)
            fig_diag.add_trace(go.Scatter(x=_sil_df["k"], y=_sil_df["silhouette"],
                mode="lines+markers", name="Silhouette",
                line=dict(color="#C62828", width=2), marker=dict(size=7),
                hovertemplate="k=%{x}<br>Silhouette: %{y:.4f}<extra></extra>"), row=1, col=2)
            fig_diag.add_vline(x=2, line_dash="dot", line_color="#EF6C00",
                line_width=1.5, annotation_text="k=2", row=1, col=1)
            fig_diag.add_vline(x=2, line_dash="dot", line_color="#EF6C00",
                line_width=1.5, annotation_text="k=2 (peak)", row=1, col=2)
            fig_diag.update_xaxes(title_text="Number of Clusters (k)", dtick=1)
            fig_diag.update_layout(**LAYOUT, showlegend=False)
            st.plotly_chart(fig_diag, use_container_width=True)
        else:
            fp = FIGURES_RQ2 / "clustering_diagnostics.png"
            if fp.exists():
                st.image(str(fp), use_container_width=True)
            else:
                st.info("Diagnostics data not found. Run the master notebook.")

    st.divider()

    _tip_header("Feature Importance — What Drives Cluster Separation?", "fig_feat_imp")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown("""
One-way ANOVA F-statistic for each feature after clustering.

| Metric | Meaning |
|--------|---------|
| **F-statistic** | Between-cluster / within-cluster variance. Higher = stronger discriminator. |
| **eta squared** | Proportion of variance explained by cluster membership. |

**order_frequency dominates (F=12,486, eta squared=0.514)**
""")
    if _feat_imp is not None:
        fi = _feat_imp.copy()
        f_col  = next((c for c in ["f_statistic","F_statistic","f_stat"] if c in fi.columns), None)
        e_col  = next((c for c in ["eta_squared","eta2","effect_size"] if c in fi.columns), None)
        ft_col = next((c for c in ["feature","Feature"] if c in fi.columns), None)
        if f_col and ft_col:
            fi = fi.sort_values(f_col, ascending=True)
            fig_fi = px.bar(fi, x=f_col, y=ft_col, orientation="h",
                color=fi[e_col] if e_col else fi[f_col], color_continuous_scale="Reds",
                custom_data=[fi[e_col].values] if e_col else None,
                text=fi[f_col].map("{:,.0f}".format),
                labels={f_col:"F-statistic (ANOVA)", ft_col:"Feature"},
                title="Feature Importance for Cluster Separation (ANOVA F-statistic)")
            eta_hover = "<br>eta squared: %{customdata[0]:.4f}" if e_col else ""
            fig_fi.update_traces(textposition="outside", cliponaxis=False,
                hovertemplate=f"<b>%{{y}}</b><br>F: %{{x:,.0f}}{eta_hover}<extra></extra>")
            fi_layout = {**LAYOUT, "height": max(CHART_H, len(fi)*40)}
            fig_fi.update_layout(**fi_layout,
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(title="eta sq.", len=0.6))
            st.plotly_chart(fig_fi, use_container_width=True)
    else:
        fp = FIGURES_RQ2 / "clustering_feature_importance.png"
        if fp.exists():
            st.image(str(fp), use_container_width=True)
        else:
            st.info("Feature importance data not found.")

    st.divider()

    _tip_header("Customer Cluster Explorer", "fig_explorer")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown("""
Each point is one customer coloured by K-Means cluster assignment.
Clear separation confirms the two archetypes are behaviorally distinct.

- **Teal = Cluster 0** — high-erosion, frequent buyers ($95.51 avg)
- **Orange = Cluster 1** — lower-erosion, occasional buyers ($53.07 avg)
""")
    if _cluster_df is not None:
        x_opts = [c for c in ["return_frequency","customer_return_rate","avg_order_value",
                               "total_items","avg_basket_size","customer_tenure_days"]
                  if c in _cluster_df.columns]
        y_opts = [c for c in ["total_profit_erosion","avg_erosion_per_return",
                               "return_frequency","avg_order_value"]
                  if c in _cluster_df.columns]
        cx, cy = st.columns(2)
        xc = cx.selectbox("X axis", x_opts, index=0)
        yc = cy.selectbox("Y axis", y_opts, index=0)
        pdf = _cluster_df.copy()
        pdf["Cluster"] = ("Cluster " + pdf["cluster_id"].astype(str)
                          if "cluster_id" in pdf.columns else "Unknown")
        sz  = ("avg_order_value" if "avg_order_value" in pdf.columns
               and xc != "avg_order_value" and yc != "avg_order_value" else None)
        hov = {c: True for c in ["return_frequency","customer_return_rate",
                                  "total_profit_erosion","avg_order_value"]
               if c in pdf.columns and c not in (xc, yc)}
        kw  = dict(data_frame=pdf, x=xc, y=yc, color="Cluster", opacity=0.55,
                   hover_data=hov,
                   title=f"Clusters: {xc.replace('_',' ').title()} vs {yc.replace('_',' ').title()}",
                   labels={xc: xc.replace("_"," ").title(), yc: yc.replace("_"," ").title()},
                   color_discrete_map={"Cluster 0":"#00897B","Cluster 1":"#E64A19"})
        if sz:
            kw["size"] = sz
            kw["size_max"] = 12
        fig_sc = px.scatter(**kw)
        fig_sc.update_layout(**LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Clustered customers parquet not found. Run the master notebook.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_val:

    _tip_header("External Validation — School Specialty LLC (SSL)", "ssl_validation", level=2)
    st.markdown("""
TheLook is a synthetic dataset. To strengthen external validity, the same behavioral features
were tested on a real-world returns dataset (SSL — School Specialty, Inc., 2025).

Since RQ2 uses **unsupervised clustering**, direct model transfer is not possible.
We use **Level 1 Pattern Validation**: do the same features that discriminate high-loss customers
in TheLook also do so in SSL?

**Success criterion:** Agreement rate >= 50% — patterns generalise across datasets.
""")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown("""
**Why Level 1 only?** K-Means cluster IDs have no shared meaning across datasets.
Instead we ask: do the same features pass 3-gate screening in both datasets?

| Status | Meaning |
|--------|---------|
| **Both Pass** | Informative in BOTH datasets — strongest cross-domain signal |
| **Disagree** | Passes in one, fails in the other — domain-specific |
| **Both Fail** | Uninformative in both — not useful for targeting |
""")

    if _valid:
        pv         = _valid.get("pattern_validation", {})
        n_feat     = pv.get("n_features_tested", "N/A")
        n_agree    = pv.get("n_agreement",        "N/A")
        n_bp       = pv.get("n_both_pass",        "N/A")
        n_bf       = pv.get("n_both_fail",        "N/A")
        ag_rate    = pv.get("agreement_rate",     None)
        passed     = pv.get("validation_passed",  None)
        gen_f      = pv.get("generalizing_features",    [])
        dom_f      = pv.get("domain_specific_features", [])
        n_disagree = (int(n_feat) - int(n_agree)) if (
            str(n_feat).isdigit() and str(n_agree).isdigit()) else "N/A"

        cv1, cv2, cv3, cv4, cv5 = st.columns(5)
        cv1.metric("Features Tested", str(n_feat),
            help="Total RQ2 behavioral features submitted to cross-dataset screening.")
        cv2.metric("Both Pass", str(n_bp),
            help="Informative in BOTH TheLook and SSL. Strongest cross-domain signals.")
        cv3.metric("Both Fail", str(n_bf),
            help="Uninformative in both datasets.")
        cv4.metric("Disagree", str(n_disagree),
            help="Pass in one dataset, fail in the other. Domain-specific.")
        cv5.metric("Agreement Rate",
            f"{ag_rate:.1%}" if isinstance(ag_rate, float) else "N/A",
            ">= 50% = validated" if isinstance(ag_rate, float) else "",
            help=f"(Both Pass + Both Fail) / Total = ({n_bp} + {n_bf}) / {n_feat}.")

        if passed is True:
            st.success(
                f"VALIDATION PASSED — {n_bp} of {n_feat} features discriminate high-loss "
                f"customers in BOTH TheLook and SSL (agreement rate {ag_rate:.1%}, threshold >= 50%). "
                "Segmentation strategy has cross-domain validity.",
                icon="✅",
            )
        elif passed is False:
            st.warning(
                f"VALIDATION INCONCLUSIVE — Agreement {ag_rate:.1%} below 50% threshold.",
                icon="⚠️",
            )

        pv_csv = PROCESSED_RQ2 / "rq2_pattern_validation.csv"
        if pv_csv.exists():
            pv_df    = pd.read_csv(pv_csv)
            feat_col = next((c for c in ["feature","Feature"] if c in pv_df.columns), None)
            if feat_col:
                pv_df["Status"] = pv_df.apply(
                    lambda r: "Both Pass" if r.get("both_pass")
                    else "Both Fail" if r.get("both_fail")
                    else "Disagree", axis=1)
                order_map = {"Both Pass": 0, "Disagree": 1, "Both Fail": 2}
                pv_df["_sort"] = pv_df["Status"].map(order_map)
                pv_df = pv_df.sort_values("_sort")
                fig_val = px.bar(pv_df, y=feat_col, x=[1]*len(pv_df), color="Status",
                    orientation="h",
                    color_discrete_map={"Both Pass":"#2E7D32","Disagree":"#EF6C00","Both Fail":"#B0BEC5"},
                    title="Feature Screening Agreement — TheLook vs SSL",
                    labels={feat_col:"","x":"","color":""},
                    text="Status")
                fig_val.update_traces(textposition="inside", insidetextanchor="middle")
                fig_val.update_layout(**LAYOUT,
                    height=max(300, len(pv_df)*42),
                    xaxis=dict(visible=False), yaxis=dict(title=""),
                    showlegend=True, legend=dict(orientation="h", y=1.06, x=0), bargap=0.3)
                st.plotly_chart(fig_val, use_container_width=True)

                with st.expander("ℹ️ How to read this chart", expanded=False):
                    st.markdown(f"""
The 6 agreeing features in the notebook = **{n_bp} Both Pass** + **{n_bf} Both Fail**.
The meaningful cross-domain signals are the **{n_bp} Both Pass** features only.
Both Fail means the feature is uninformative in both datasets — it still "agrees" on uselessness.
""")

        if gen_f or dom_f:
            col_gen, col_dom = st.columns(2)
            with col_gen:
                st.subheader("Generalising Features")
                st.caption("Informative in both TheLook and SSL — use confidently for cross-domain targeting.")
                for f in gen_f:
                    st.markdown(f"- `{f}`")
            with col_dom:
                st.subheader("Domain-Specific Features")
                st.caption("Different patterns across datasets — may require recalibration.")
                for f in dom_f:
                    st.markdown(f"- `{f}`")
    else:
        st.info(
            "Validation results not found (`rq2_validation_results.json`). "
            "Run Phase 4 of the master notebook — requires the SSL dataset.",
            icon="ℹ️",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_end:

    _tip_header("Integrated Interpretation & Recommendations", "conclusion", level=2)
    st.markdown("""
Concentration identifies **WHO** to target; segmentation reveals **HOW** to differentiate the intervention.
Together they provide two complementary targeting dimensions.
""")

    if _conc and _cs_df is not None:
        _high_mean = _cs_df["Mean_Erosion"].max()
        _low_mean  = _cs_df["Mean_Erosion"].min()
        _ratio     = _high_mean / _low_mean if _low_mean > 0 else float("inf")

        st.subheader("Hypothesis Test Result")
        st.markdown(f"""
| | |
|---|---|
| **Research Question** | Can unsupervised learning identify distinct customer behavioral segments, and do these segments differ significantly in profit erosion intensity? |
| **H₀₂ (Null)** | Customer segments identified through clustering algorithms do not differ significantly in mean profit erosion from returns. |
| **H₁₂ (Alternative)** | Customer segments identified through clustering algorithms exhibit statistically significant differences in mean profit erosion from returns. |
| **Test** | One-way ANOVA + Kruskal-Wallis (dual test for robustness) |
| **Result** | **H₀₂ REJECTED** — ANOVA F = 1,479.64, p < 0.0001 · KW H = 893.49, p < 0.0001 · η² = 0.112 |
| **Conclusion** | H₁₂ supported — clusters differ significantly in mean profit erosion. Gini = {_gini:.3f} (concentration confirmed by bootstrap p < 0.0001). |
""")

        st.divider()

        col_i1, col_i2, col_i3 = st.columns(3)
        col_i1.metric("WHO to Target", "Top 20% of customers",
            f"-> {_top20:.1f}% of total erosion" if isinstance(_top20, float) else "",
            help="Ranked by purchase_recency_days (Gini=0.528) — highest-concentration feature.")
        col_i2.metric("HOW — Priority Segment", "Cluster 0",
            f"${_high_mean:,.2f} avg vs ${_low_mean:,.2f}",
            help=f"Cluster 0 is {_ratio:.2f}x higher erosion per customer than Cluster 1.")
        col_i3.metric("Primary Driver", "order_frequency", "F = 12,486, eta squared = 0.514",
            help="Single feature most strongly separating the two clusters.")

        id_col = next((c for c in ["cluster_id","Cluster","cluster"] if c in _cs_df.columns), None)
        if id_col and "Total_Erosion" in _cs_df.columns:
            fig_pie = px.pie(_cs_df,
                names="Cluster " + _cs_df[id_col].astype(str),
                values="Total_Erosion",
                color_discrete_sequence=["#00897B","#E64A19"],
                title="Share of Total Profit Erosion by Cluster",
                hole=0.45)
            fig_pie.update_traces(
                hovertemplate="<b>%{label}</b><br>Total: $%{value:,.2f}<br>Share: %{percent}<extra></extra>")
            pie_layout = {**LAYOUT, "height": 320}
            fig_pie.update_layout(**pie_layout)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.subheader("Strategic Action Plan")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown("""
<div style="background:#E0F2F1;border-left:4px solid #00897B;padding:16px;border-radius:6px;">
<h4 style="margin:0 0 8px 0;color:#00695C;">P1 — Cluster 0</h4>
<p style="margin:0;font-size:13px;color:#004D40;">
<b>4,302 customers · $95.51 avg · Frequent buyers</b><br><br>
Personalised fit/size guidance + loyalty incentives to redirect returns into exchanges.
</p></div>""", unsafe_allow_html=True)
        with col_p2:
            st.markdown("""
<div style="background:#FFF3E0;border-left:4px solid #E65100;padding:16px;border-radius:6px;">
<h4 style="margin:0 0 8px 0;color:#E65100;">P2 — Top 20% by Recency</h4>
<p style="margin:0;font-size:13px;color:#BF360C;">
<b>Ranked by purchase_recency_days (Gini 0.528)</b><br><br>
Real-time return alerts + proactive outreach before the next return is processed.
</p></div>""", unsafe_allow_html=True)
        with col_p3:
            st.markdown("""
<div style="background:#F3E5F5;border-left:4px solid #7B1FA2;padding:16px;border-radius:6px;">
<h4 style="margin:0 0 8px 0;color:#6A1B9A;">P3 — Cluster 1</h4>
<p style="margin:0;font-size:13px;color:#4A148C;">
<b>7,488 customers · $53.07 avg</b><br><br>
Improved product content + light policy guardrails to reduce avoidable returns at scale.
</p></div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("RQ2 Summary")
        top50_pct = _top50.get("percentage_of_total", None)
        top50_str = f"{top50_pct:.1f}%" if isinstance(top50_pct, (int, float)) else "N/A"
        val_str   = ("Passed — patterns generalise to SSL"
                     if _valid.get("pattern_validation", {}).get("validation_passed")
                     else "Not available or inconclusive")
        st.markdown(f"""
| Finding | Result |
|---|---|
| **H₀₂: Segments do not differ in mean profit erosion?** | ✅ REJECTED — ANOVA F = 1,479.64, p < 0.0001, η² = 0.112 |
| **Concentration (Gini, bootstrap)** | Gini = {_gini:.3f}, p < 0.0001 — erosion significantly unequal |
| **Top 20% customer erosion share** | {_top20:.1f}% |
| **Top 50 customers** | {top50_str} of total erosion |
| **High- vs low-erosion cluster ratio** | {_ratio:.2f}× (${_high_mean:,.2f} vs ${_low_mean:,.2f} avg) |
| **Primary cluster driver** | order_frequency (F = 12,486, η² = 0.514) |
| **Highest-concentration feature** | purchase_recency_days (Gini = 0.528) |
| **External validation (SSL)** | {val_str} |
""")
    else:
        st.info("Conclusion requires processed data files. Run the master notebook first.")

st.caption(
    "DAMO-699-4 · University of Niagara Falls, Canada · Winter 2026 · "
    "RQ2 — Concentration & Segmentation Analysis"
)
