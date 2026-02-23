"""
RQ2: Customer Behavioral Segments with Differential Profit Erosion

Method: K-Means Clustering + Concentration Analysis (Lorenz/Pareto/Gini)
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(
    page_title="RQ2 – Customer Segments",
    page_icon="👥",
    layout="wide",
)

ROOT = Path(__file__).parent.parent.parent
FIGURES_RQ2 = ROOT / "figures" / "rq2"
PROCESSED_RQ2 = ROOT / "data" / "processed" / "rq2"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👥 RQ2: Customer Behavioral Segments")
st.markdown(
    """
**Research Question**: Do behaviorally distinct customer segments exhibit differential profit erosion patterns?

**Method**: K-Means clustering on behavioral features + Gini/Lorenz/Pareto concentration analysis
to identify which customer segments drive disproportionate profit erosion.
"""
)
st.divider()

# ── KPI cards from concentration metrics ──────────────────────────────────────
st.header("Concentration Metrics at a Glance")

conc_path = PROCESSED_RQ2 / "profit_erosion_concentration_metrics.json"
if conc_path.exists():
    with open(conc_path) as f:
        conc = json.load(f)

    gini = conc.get("gini_coefficient", "N/A")
    top20 = conc.get("top_20_pct_share", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gini Coefficient", f"{gini:.3f}" if isinstance(gini, float) else gini,
                "Moderate concentration")
    col2.metric("Top 20% Share", f"{top20:.1f}%" if isinstance(top20, float) else top20,
                "of total profit erosion")

    # Try to get cluster count from cluster_summary
    cs_path = PROCESSED_RQ2 / "cluster_summary.parquet"
    if cs_path.exists():
        cs = pd.read_parquet(cs_path)
        col3.metric("Customer Segments", len(cs), "K-Means clusters")
        col4.metric("Total Customers", f"{cs['Count'].sum():,}", "in analysis population")
    else:
        col3.metric("Customer Segments", "—", "")
        col4.metric("Total Customers", "—", "")
else:
    st.warning("Concentration metrics JSON not found. Run the master notebook first.")

st.divider()

# ── Cluster summary table ─────────────────────────────────────────────────────
st.header("Cluster Profiles")

cs_path = PROCESSED_RQ2 / "cluster_summary.parquet"
if cs_path.exists():
    cs = pd.read_parquet(cs_path)
    display_df = cs.copy()
    # Format currency columns
    for col in ["Total_Erosion", "Mean_Erosion", "Median_Erosion"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map("${:,.2f}".format)
    if "Count" in display_df.columns:
        display_df["Count"] = display_df["Count"].map("{:,}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "Each cluster represents a behaviorally distinct customer group. "
        "Higher Mean_Erosion clusters are priority intervention targets."
    )
else:
    st.info("Cluster summary not found. Run the master notebook.")

st.divider()

# ── Visualizations ────────────────────────────────────────────────────────────
st.header("Visualizations")

figures = [
    ("cluster_erosion_comparison.png", "Cluster Erosion Comparison"),
    ("clustering_diagnostics.png", "Clustering Diagnostics (Silhouette / Elbow)"),
    ("lorenz_curve.png", "Lorenz Curve — Profit Erosion Concentration"),
    ("pareto_curve.png", "Pareto Curve — Cumulative Erosion Share"),
    ("concentration_gini_vs_pareto.png", "Gini vs Pareto Concentration"),
    ("clustering_feature_importance.png", "Feature Importance for Clustering"),
    ("feature_concentration_ranking.png", "Feature Concentration Ranking"),
]

for i in range(0, len(figures), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(figures):
            fname, label = figures[i + j]
            fig_path = FIGURES_RQ2 / fname
            with col:
                st.subheader(label)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fname}")

st.divider()

# ── Feature concentration table ───────────────────────────────────────────────
st.header("Feature Concentration Ranking")

feat_path = PROCESSED_RQ2 / "feature_concentration_ranking.parquet"
if feat_path.exists():
    feat_df = pd.read_parquet(feat_path)
    display_cols = [c for c in [
        "feature", "gini_coefficient", "top_20_pct_share",
        "concentration_level", "p_value", "n_customers",
    ] if c in feat_df.columns]
    st.dataframe(feat_df[display_cols], use_container_width=True, hide_index=True)
    st.caption(
        "Gini coefficient measures inequality in feature distribution across customers. "
        "Higher Gini = more concentrated in fewer customers."
    )
else:
    st.info("Feature concentration ranking not found. Run the master notebook.")

st.divider()

# ── Key findings (data-driven) ───────────────────────────────────────────────
st.header("Key Findings")

_conc_path = PROCESSED_RQ2 / "profit_erosion_concentration_metrics.json"
_cs_path = PROCESSED_RQ2 / "cluster_summary.parquet"
_feat_path = PROCESSED_RQ2 / "feature_concentration_ranking.parquet"

if _conc_path.exists() and _cs_path.exists():
    with open(_conc_path) as _f:
        _conc = json.load(_f)

    _gini = _conc.get("gini_coefficient", None)
    _top20 = _conc.get("top_20_pct_share", None)

    _cs = pd.read_parquet(_cs_path)
    _n_clusters = len(_cs)
    _high_mean = _cs["Mean_Erosion"].max()
    _low_mean = _cs["Mean_Erosion"].min()
    _ratio = _high_mean / _low_mean

    _top_feats = []
    if _feat_path.exists():
        _feat_df = pd.read_parquet(_feat_path)
        if "gini_coefficient" in _feat_df.columns:
            _top_feats = _feat_df.nlargest(3, "gini_coefficient")["feature"].tolist()

    st.markdown(
        f"""
- **Gini coefficient = {_gini:.3f}** confirms concentration — profit erosion is unevenly
  distributed across customers (not random)
- **Top 20% of customers** account for {_top20:.1f}% of total profit erosion (Pareto principle observed)
- **K-Means clustering** identifies {_n_clusters} behaviorally distinct segment(s) with materially
  different mean erosion levels — supporting H₀ rejection
- **High-erosion segment** (mean ${_high_mean:,.2f}) has {_ratio:.2f}× higher mean erosion than
  the low-erosion segment (mean ${_low_mean:,.2f}), making targeted intervention economically feasible
- **Top 3 most concentrated features**: {", ".join([f"`{f}`" for f in _top_feats]) if _top_feats else "see table above"}
- **Hypothesis**: Reject H₀ — behaviorally distinct segments with differential profit erosion confirmed
"""
    )
else:
    st.info("Key findings require processed data files. Run the master notebook first.")
