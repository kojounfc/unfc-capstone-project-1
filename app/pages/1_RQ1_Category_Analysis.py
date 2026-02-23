"""
RQ1: Profit Erosion Differences Across Product Categories & Brands

Method: Descriptive Analysis + Kruskal-Wallis + Post-hoc Dunn
"""

import streamlit as st
import pandas as pd
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

# ── Figures ───────────────────────────────────────────────────────────────────
st.header("Visualizations")

figures = [
    ("fig1_top_categories_total_erosion.png", "Fig 1: Top Categories — Total Profit Erosion"),
    ("fig2_top_brands_total_erosion.png", "Fig 2: Top Brands — Total Profit Erosion"),
    ("fig3_return_rate_vs_mean_erosion_category.png", "Fig 3: Return Rate vs Mean Erosion by Category"),
    ("fig4_top_departments_total_erosion.png", "Fig 4: Top Departments — Total Profit Erosion"),
    ("fig5_severity_vs_volume_category.png", "Fig 5: Severity vs Volume by Category"),
    ("fig6_profit_erosion_distribution_log.png", "Fig 6: Profit Erosion Distribution (Log Scale)"),
    ("fig7_bootstrap_ci_category_mean.png", "Fig 7: Bootstrap 95% CI — Category Mean Erosion"),
]

# Display in 2-column grid
for i in range(0, len(figures), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(figures):
            fname, label = figures[i + j]
            fig_path = FIGURES_RQ1 / fname
            with col:
                st.subheader(label)
                if fig_path.exists():
                    st.image(str(fig_path), use_container_width=True)
                else:
                    st.warning(f"Figure not found: {fname}")

st.divider()

# ── Category summary table ────────────────────────────────────────────────────
st.header("Category-Level Profit Erosion Summary")

cat_path = PROCESSED / "us07_product_profit_erosion_by_category.parquet"
if cat_path.exists():
    cat_df = pd.read_parquet(cat_path)
    display_cols = [c for c in [
        "category", "total_profit_erosion", "mean_profit_erosion",
        "return_count", "return_rate", "total_margin_reversal", "total_process_cost",
    ] if c in cat_df.columns]
    if display_cols:
        show_df = cat_df[display_cols].copy()
        # Format currency columns
        for col in show_df.columns:
            if "erosion" in col or "margin" in col or "cost" in col:
                if show_df[col].dtype in ["float64", "float32"]:
                    show_df[col] = show_df[col].map("${:,.2f}".format)
        st.dataframe(show_df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(cat_df.head(20), use_container_width=True, hide_index=True)
else:
    st.info("Category summary parquet not found at data/processed/. Run the master notebook.")

st.divider()

# ── Brand summary table ───────────────────────────────────────────────────────
st.header("Brand-Level Profit Erosion Summary")

brand_path = PROCESSED / "us07_product_profit_erosion_by_brand.parquet"
if brand_path.exists():
    brand_df = pd.read_parquet(brand_path)
    display_cols = [c for c in [
        "brand", "total_profit_erosion", "mean_profit_erosion",
        "return_count", "return_rate",
    ] if c in brand_df.columns]
    show_df = brand_df[display_cols].head(20).copy() if display_cols else brand_df.head(20)
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.caption("Showing top 20 brands by total profit erosion.")
else:
    st.info("Brand summary parquet not found. Run the master notebook.")

st.divider()

# ── Key findings (data-driven) ───────────────────────────────────────────────
st.header("Key Findings")

_cat_path = PROCESSED / "us07_product_profit_erosion_by_category.parquet"
_stats_path = PROCESSED_RQ1 / "rq1_statistical_tests_summary.parquet"

if _cat_path.exists() and _stats_path.exists():
    _cat = pd.read_parquet(_cat_path)
    _stats = pd.read_parquet(_stats_path)

    _top5 = _cat.nlargest(5, "total_profit_erosion")["category"].tolist()
    _top5_str = ", ".join(_top5)

    _cat_row = _stats[_stats["factor"] == "category"].iloc[0] if "category" in _stats["factor"].values else None
    _brand_row = _stats[_stats["factor"] == "brand"].iloc[0] if "brand" in _stats["factor"].values else None

    _cat_p = f"{_cat_row['p_value']:.2e}" if _cat_row is not None else "N/A"
    _cat_eff = f"{_cat_row['effect_size']:.3f} ({_cat_row['effect_metric']})" if _cat_row is not None else "N/A"
    _cat_h0 = "Rejected" if (_cat_row is not None and _cat_row["reject_h0"]) else "Not Rejected"

    _brand_p = f"{_brand_row['p_value']:.2e}" if _brand_row is not None else "N/A"
    _brand_eff = f"{_brand_row['effect_size']:.3f} ({_brand_row['effect_metric']})" if _brand_row is not None else "N/A"
    _brand_h0 = "Rejected" if (_brand_row is not None and _brand_row["reject_h0"]) else "Not Rejected"

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
