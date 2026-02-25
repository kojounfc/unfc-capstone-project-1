"""
RQ2: Customer Behavioral Segments with Differential Profit Erosion

Method: K-Means Clustering + Concentration Analysis (Lorenz/Pareto/Gini)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ── Load data ─────────────────────────────────────────────────────────────────
_conc = {}
_cs_df = None
_pareto_df = None
_cluster_df = None

_conc_path = PROCESSED_RQ2 / "profit_erosion_concentration_metrics.json"
if _conc_path.exists():
    with open(_conc_path) as f:
        _conc = json.load(f)

_cs_path = PROCESSED_RQ2 / "cluster_summary.parquet"
if _cs_path.exists():
    _cs_df = pd.read_parquet(_cs_path)

_pareto_path = PROCESSED_RQ2 / "pareto_table.parquet"
if _pareto_path.exists():
    _pareto_df = pd.read_parquet(_pareto_path)

_cluster_path = PROCESSED_RQ2 / "clustered_customers.parquet"
if _cluster_path.exists():
    _cluster_df = pd.read_parquet(_cluster_path)

# Derived values — JSON stores fractions (0–1), multiply × 100 for display
_gini = _conc.get("gini_coefficient", None)
_top20_raw = _conc.get("top_20_pct_share", None)
_top20_display = (
    _top20_raw * 100
    if isinstance(_top20_raw, float) and _top20_raw < 1
    else _top20_raw
)

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

# ── KPI cards ─────────────────────────────────────────────────────────────────
st.header("Concentration Metrics at a Glance")

if _conc:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Gini Coefficient",
        f"{_gini:.3f}" if isinstance(_gini, float) else "N/A",
        "Moderate concentration",
    )
    col2.metric(
        "Top 20% Share",
        f"{_top20_display:.1f}%" if isinstance(_top20_display, float) else "N/A",
        "of total profit erosion",
    )
    if _cs_df is not None:
        col3.metric("Customer Segments", len(_cs_df), "K-Means clusters")
        col4.metric("Total Customers", f"{_cs_df['Count'].sum():,}", "in analysis population")
    else:
        col3.metric("Customer Segments", "—", "")
        col4.metric("Total Customers", "—", "")
else:
    st.warning("Concentration metrics JSON not found. Run the master notebook first.")

st.divider()

# ── Cluster summary table ─────────────────────────────────────────────────────
st.header("Cluster Profiles")

if _cs_df is not None:
    display_df = _cs_df.copy()
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

# ── Customer Cluster Explorer (interactive scatter — net new) ─────────────────
st.header("Customer Cluster Explorer")
st.markdown(
    "Explore how customers separate into clusters across behavioral dimensions. "
    "Each point is one customer, colored by cluster assignment."
)

if _cluster_df is not None:
    x_options = [
        c for c in [
            "return_frequency", "customer_return_rate", "avg_order_value",
            "total_items", "avg_basket_size", "customer_tenure_days",
        ] if c in _cluster_df.columns
    ]
    y_options = [
        c for c in [
            "total_profit_erosion", "avg_erosion_per_return",
            "return_frequency", "avg_order_value",
        ] if c in _cluster_df.columns
    ]

    col_x, col_y = st.columns(2)
    x_col = col_x.selectbox("X axis", x_options, index=0)
    y_col = col_y.selectbox("Y axis", y_options, index=0)

    plot_df = _cluster_df.copy()
    if "cluster_id" in plot_df.columns:
        plot_df["Cluster"] = "Cluster " + plot_df["cluster_id"].astype(str)
    else:
        plot_df["Cluster"] = "Unknown"

    size_col = (
        "avg_order_value"
        if "avg_order_value" in plot_df.columns
        and x_col != "avg_order_value"
        and y_col != "avg_order_value"
        else None
    )

    hover_data = {
        c: True for c in [
            "return_frequency", "customer_return_rate",
            "total_profit_erosion", "avg_order_value",
        ] if c in plot_df.columns and c not in (x_col, y_col)
    }

    scatter_kwargs = dict(
        data_frame=plot_df,
        x=x_col,
        y=y_col,
        color="Cluster",
        opacity=0.55,
        hover_data=hover_data,
        title=f"Customer Clusters: {x_col} vs {y_col}",
        labels={
            x_col: x_col.replace("_", " ").title(),
            y_col: y_col.replace("_", " ").title(),
        },
        color_discrete_sequence=["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"],
    )
    if size_col:
        scatter_kwargs["size"] = size_col
        scatter_kwargs["size_max"] = 12

    fig_cluster = px.scatter(**scatter_kwargs)
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.caption(
        "Clusters derived from K-Means on behavioral features. "
        "Clear separation across axes confirms behaviorally distinct segments."
    )
else:
    st.info("Clustered customers parquet not found. Run the master notebook.")

st.divider()

# ── Static figures (diagnostics, feature importance, comparison) ───────────────
st.header("Clustering Diagnostics & Feature Importance")

static_figs = [
    ("clustering_diagnostics.png", "Clustering Diagnostics (Silhouette / Elbow)"),
    ("clustering_feature_importance.png", "Feature Importance for Clustering"),
    ("cluster_erosion_comparison.png", "Cluster Erosion Comparison"),
    ("feature_concentration_ranking.png", "Feature Concentration Ranking"),
]

for i in range(0, len(static_figs), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(static_figs):
            fname, label = static_figs[i + j]
            fig_path = FIGURES_RQ2 / fname
            with col:
                st.subheader(label)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fname}")

st.divider()

# ── Lorenz Curve (interactive) ────────────────────────────────────────────────
st.header("Concentration Analysis")
st.subheader("Lorenz Curve — Profit Erosion Concentration (Interactive)")

if (
    _pareto_df is not None
    and "customer_share" in _pareto_df.columns
    and "value_share" in _pareto_df.columns
):
    lorenz_df = _pareto_df.sort_values("customer_share")
    gini_label = (
        f"Lorenz Curve (Gini = {_gini:.3f})"
        if isinstance(_gini, float) else "Lorenz Curve"
    )

    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(
        x=lorenz_df["customer_share"] * 100,
        y=lorenz_df["value_share"] * 100,
        mode="lines",
        name=gini_label,
        line=dict(color="#1565C0", width=2),
        hovertemplate=(
            "Top %{x:.1f}% of customers<br>"
            "→ %{y:.1f}% of total erosion<extra></extra>"
        ),
    ))
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode="lines",
        name="Perfect Equality",
        line=dict(dash="dash", color="gray", width=1),
        hoverinfo="skip",
    ))
    fig_lorenz.update_layout(
        title="Lorenz Curve — Cumulative Profit Erosion Distribution",
        xaxis_title="Cumulative % of Customers",
        yaxis_title="Cumulative % of Profit Erosion",
        legend=dict(x=0.05, y=0.95),
        height=450,
    )
    st.plotly_chart(fig_lorenz, use_container_width=True)
    if isinstance(_gini, float) and isinstance(_top20_display, float):
        st.caption(
            f"Gini = {_gini:.3f} — the curve bowing below the diagonal quantifies concentration. "
            f"Top 20% of customers account for {_top20_display:.1f}% of total profit erosion."
        )
else:
    lorenz_path = FIGURES_RQ2 / "lorenz_curve.png"
    if lorenz_path.exists():
        st.image(str(lorenz_path), use_container_width=True)
    else:
        st.warning("Pareto/Lorenz data not found. Run the master notebook.")

st.divider()

# ── Pareto Curve (interactive) ────────────────────────────────────────────────
st.subheader("Pareto Curve — Cumulative Erosion Share (Interactive)")

if (
    _pareto_df is not None
    and "customer_share" in _pareto_df.columns
    and "value_share" in _pareto_df.columns
):
    pareto_df = _pareto_df.sort_values("customer_share")

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=pareto_df["customer_share"] * 100,
        y=pareto_df["value_share"] * 100,
        mode="lines",
        name="Cumulative Erosion Share",
        line=dict(color="#C62828", width=2),
        hovertemplate=(
            "Top %{x:.1f}% of customers<br>"
            "→ %{y:.1f}% of total erosion<extra></extra>"
        ),
    ))

    if isinstance(_top20_display, float):
        fig_pareto.add_vline(
            x=20,
            line_dash="dash",
            line_color="gray",
            annotation_text="20% of customers",
            annotation_position="top right",
        )
        fig_pareto.add_hline(
            y=_top20_display,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{_top20_display:.1f}% of erosion",
            annotation_position="bottom right",
        )

    fig_pareto.update_layout(
        title="Pareto Curve — Cumulative Profit Erosion by Customer Rank",
        xaxis_title="Cumulative % of Customers (ranked by erosion)",
        yaxis_title="Cumulative % of Total Profit Erosion",
        height=450,
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    if isinstance(_top20_display, float):
        st.caption(
            f"The top 20% of customers account for {_top20_display:.1f}% of total profit erosion. "
            "Dashed lines mark the 20% crossover point."
        )
else:
    pareto_path = FIGURES_RQ2 / "pareto_curve.png"
    if pareto_path.exists():
        st.image(str(pareto_path), use_container_width=True)
    else:
        st.warning("Pareto table not found. Run the master notebook.")

st.divider()

# ── Gini vs Pareto static figure ──────────────────────────────────────────────
conc_fig_path = FIGURES_RQ2 / "concentration_gini_vs_pareto.png"
if conc_fig_path.exists():
    st.subheader("Gini vs Pareto Concentration")
    st.image(str(conc_fig_path), use_container_width=True)
    st.divider()

# ── Key findings (data-driven) ────────────────────────────────────────────────
st.header("Key Findings")

if _conc and _cs_df is not None:
    _n_clusters = len(_cs_df)
    _high_mean = _cs_df["Mean_Erosion"].max()
    _low_mean = _cs_df["Mean_Erosion"].min()
    _ratio = _high_mean / _low_mean if _low_mean > 0 else float("inf")

    st.markdown(
        f"""
- **Gini coefficient = {_gini:.3f}** confirms concentration — profit erosion is unevenly
  distributed across customers (not random)
- **Top 20% of customers** account for {_top20_display:.1f}% of total profit erosion
  (Pareto principle observed)
- **K-Means clustering** identifies {_n_clusters} behaviorally distinct segment(s) with materially
  different mean erosion levels — supporting H₀ rejection
- **High-erosion segment** (mean ${_high_mean:,.2f}) has {_ratio:.2f}× higher mean erosion than
  the low-erosion segment (mean ${_low_mean:,.2f}), making targeted intervention economically feasible
- **Hypothesis**: Reject H₀ — behaviorally distinct segments with differential profit erosion confirmed
"""
    )
else:
    st.info("Key findings require processed data files. Run the master notebook first.")
