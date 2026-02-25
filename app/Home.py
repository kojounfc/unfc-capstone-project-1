"""
Home.py — Landing page for the Profit Erosion Analytics Dashboard.

Analyzing Profit Erosion from Product Returns in E-Commerce:
A Multi-Method Analytics Framework

University of Niagara Falls Canada — DAMO-699-4 Capstone (Winter 2026)
"""

import sys
import streamlit as st
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profit Erosion Analytics",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
# Add both project root (for 'from src.config import ...' inside modules)
# and src/ (for direct imports) so feature_engineering.py resolves correctly
for _p in [str(ROOT), str(ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load source constants from feature_engineering ───────────────────────────
try:
    from feature_engineering import DEFAULT_COST_COMPONENTS, CATEGORY_TIER_MULTIPLIERS
    from config import AUC_THRESHOLD
    _src_loaded = True
except Exception:
    _src_loaded = False
    DEFAULT_COST_COMPONENTS = {}
    CATEGORY_TIER_MULTIPLIERS = {}
    AUC_THRESHOLD = 0.70

# ── Load result data ──────────────────────────────────────────────────────────
PROCESSED = ROOT / "data" / "processed"
REPORTS_RQ3 = ROOT / "reports" / "rq3"
REPORTS_RQ4 = ROOT / "reports" / "rq4"

_rq1_top_cat = None
_rq2_n_clusters = None
_rq3_champion_name = None
_rq3_champion_auc = None
_rq4_r2 = None
_rq4_top_feat = None
_rq4_top_pct = None
_ssl_accounts_rq3 = None

_cat_path = PROCESSED / "us07_product_profit_erosion_by_category.parquet"
if _cat_path.exists():
    _cat_df = pd.read_parquet(_cat_path)
    _rq1_top_cat = _cat_df.nlargest(1, "total_profit_erosion").iloc[0]["category"]

_rq1_top_brand = None
_brand_path = PROCESSED / "us07_product_profit_erosion_by_brand.parquet"
if _brand_path.exists():
    _brand_df = pd.read_parquet(_brand_path)
    _rq1_top_brand = _brand_df.nlargest(1, "total_profit_erosion").iloc[0]["brand"]

_cs_path = PROCESSED / "rq2" / "cluster_summary.parquet"
if _cs_path.exists():
    _cs_df = pd.read_parquet(_cs_path)
    _rq2_n_clusters = len(_cs_df)

_comp_path = REPORTS_RQ3 / "rq3_model_comparison.csv"
if _comp_path.exists():
    _comp_df = pd.read_csv(_comp_path)
    _champ_row = _comp_df.loc[_comp_df["test_auc"].idxmax()]
    _rq3_champion_name = _champ_row["model"]
    _rq3_champion_auc = float(_champ_row["test_auc"])

_val3_path = REPORTS_RQ3 / "rq3_validation_summary.csv"
if _val3_path.exists():
    _val3 = pd.read_csv(_val3_path)
    _val3_dict = dict(zip(_val3["metric"], _val3["value"]))
    _ssl_accounts_rq3 = int(float(_val3_dict.get("ssl_accounts_evaluated", 0)))

_val4_path = REPORTS_RQ4 / "rq4v2_validation_summary.csv"
_align4_path = REPORTS_RQ4 / "rq4v2_ssl_coefficient_alignment.csv"
_coef4_path = REPORTS_RQ4 / "rq4v2_thelook_coefficients.csv"
if _val4_path.exists():
    _val4 = pd.read_csv(_val4_path)
    _val4_dict = dict(zip(_val4.iloc[:, 0], _val4.iloc[:, 1]))
    _rq4_r2 = float(_val4_dict.get("thelook_r_squared", float("nan")))
if _coef4_path.exists() and _align4_path.exists():
    _coef4 = pd.read_csv(_coef4_path)
    _align4 = pd.read_csv(_align4_path)
    _behav = _coef4[
        ~_coef4["feature"].str.startswith("dominant_return_category")
        & (_coef4["feature"] != "const")
    ]
    if not _behav.empty:
        _top = _behav.loc[_behav["coefficient"].abs().idxmax()]
        _rq4_top_feat = _top["feature"]
        _align_top = _align4[_align4["feature"] == _rq4_top_feat]
        _rq4_top_pct = float(_align_top["thelook_pct_effect"].iloc[0]) if not _align_top.empty else None

# ── Cost model derived values ─────────────────────────────────────────────────
_base_cost = sum(DEFAULT_COST_COMPONENTS.values()) if DEFAULT_COST_COMPONENTS else None
_base_str = f"${_base_cost:.2f}" if _base_cost else "N/A"

_tiers = defaultdict(list)
for cat, mult in CATEGORY_TIER_MULTIPLIERS.items():
    _tiers[mult].append(cat)

_high_erosion_pct = 75  # project default — 75th percentile threshold

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📉 Profit Erosion Analytics Dashboard")
st.markdown(
    """
**Analyzing Profit Erosion from Product Returns in E-Commerce:
A Multi-Method Analytics Framework**

*University of Niagara Falls Canada — DAMO-699-4 Capstone (Winter 2026)*
"""
)
st.divider()

# ── Problem statement ─────────────────────────────────────────────────────────
st.header("Problem Statement")
st.markdown(
    f"""
Product returns are **economic reversal events** that directly erode realized revenue and margin.
This project reframes returns beyond operational metrics to quantify:

- **Margin reversal** — the margin lost on returned items (sale price − cost)
- **Incremental processing costs** — customer service, inspection, restocking, logistics
  ({_base_str} base per return, category-tier adjusted)

> **Core formula:**  `Profit Erosion = Margin Reversal + Processing Cost`
"""
)

# ── Project at a Glance KPIs ──────────────────────────────────────────────────
st.header("Project at a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Data Source", "TheLook E-Commerce", "Google BigQuery")
col2.metric(
    "External Validation",
    "School Specialty LLC",
    f"{_ssl_accounts_rq3:,} accounts" if _ssl_accounts_rq3 else "B2B retailer",
)
col3.metric(
    "Processing Cost Base",
    f"{_base_str} / return",
    "Category tier adjusted",
)
col4.metric(
    "High-Erosion Threshold",
    f"{_high_erosion_pct}th Percentile",
    f"Top {100 - _high_erosion_pct}% customers",
)

st.divider()

# ── Research questions overview (data-driven Key Results) ─────────────────────
st.header("Research Questions")

if _rq1_top_cat and _rq1_top_brand:
    _rq1_result = f"{_rq1_top_cat} (category) · {_rq1_top_brand} (brand) — highest erosion"
elif _rq1_top_cat:
    _rq1_result = f"{_rq1_top_cat} — highest total erosion category"
else:
    _rq1_result = "See RQ1 page"
_rq2_result = (
    f"{_rq2_n_clusters} behaviorally distinct customer segment(s) identified"
    if _rq2_n_clusters else "See RQ2 page"
)
_rq3_result = (
    f"AUC = {_rq3_champion_auc:.4f} ({_rq3_champion_name} champion)"
    if _rq3_champion_auc else "See RQ3 page"
)
_rq4_result = (
    f"R² = {_rq4_r2:.4f} | {_rq4_top_feat}: {_rq4_top_pct:+.1f}% effect"
    if _rq4_r2 and _rq4_top_feat and _rq4_top_pct else "See RQ4 page"
)

rq_df = pd.DataFrame({
    "RQ": ["RQ1", "RQ2", "RQ3", "RQ4"],
    "Focus": [
        "Profit erosion differences across product categories & brands",
        "Customer behavioral segments with differential profit erosion",
        f"Predict high-erosion customers (target AUC > {AUC_THRESHOLD})",
        "Marginal associations between behaviors and profit erosion",
    ],
    "Method": [
        "Descriptive Analysis + Kruskal-Wallis",
        "K-Means Clustering + Concentration Analysis",
        "ML Classification (RF, GB, LR)",
        "Log-Linear OLS Regression",
    ],
    "Key Result": [_rq1_result, _rq2_result, _rq3_result, _rq4_result],
})
st.dataframe(rq_df, use_container_width=True, hide_index=True)

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
st.header("Navigation")
st.markdown(
    """
Use the **sidebar** to navigate between research questions:

| Page | Content |
|------|---------|
| **RQ1: Category Analysis** | Category & brand profit erosion rankings, statistical tests, bootstrap CIs |
| **RQ2: Customer Segments** | Cluster profiles, Lorenz/Pareto concentration curves, segment diagnostics |
| **RQ3: Predictive Model** | ROC curves, feature importance, SSL external validation, sensitivity analysis |
| **RQ4: Behavioral Associations** | OLS coefficient forest plot, residual diagnostics, SSL alignment |
"""
)

st.divider()

# ── Cost model (data-driven from feature_engineering.py) ─────────────────────
st.header("Cost Model")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Processing Cost Components")
    if DEFAULT_COST_COMPONENTS:
        _component_labels = {
            "customer_care": ("Customer Care", "Phone/email support for return request processing"),
            "inspection": ("Inspection", "Quality assessment upon receipt in returns center"),
            "restocking": ("Restocking", "Shelving in warehouse, inventory system updates"),
            "logistics": ("Logistics", "Return label, carrier coordination, admin processing"),
        }
        _rows = []
        for key, amount in DEFAULT_COST_COMPONENTS.items():
            label, rationale = _component_labels.get(key, (key.replace("_", " ").title(), ""))
            _rows.append({"Component": label, "Amount": f"${amount:.2f}", "Rationale": rationale})
        _rows.append({
            "Component": "Total Base",
            "Amount": _base_str,
            "Rationale": "Conservative mid-range ($10–$25 literature range)",
        })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Cost components not loaded. Check src/feature_engineering.py.")

with col_b:
    st.subheader("Category Tier Multipliers")
    if _tiers:
        _tier_rows = []
        for mult in sorted(_tiers.keys(), reverse=True):
            cats = sorted(_tiers[mult])
            effective = _base_cost * mult if _base_cost else None
            _tier_rows.append({
                "Multiplier": f"{mult}×",
                "Effective Cost": f"${effective:.2f}" if effective else "N/A",
                "Categories": ", ".join(cats),
            })
        st.dataframe(pd.DataFrame(_tier_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Tier multipliers justified by Margin CV = 59.4% across categories "
            "(exceeds 15% threshold for tiered treatment)."
        )
    else:
        st.warning("Tier multipliers not loaded. Check src/feature_engineering.py.")

st.divider()

# ── Team ──────────────────────────────────────────────────────────────────────
st.header("Team")
st.dataframe(
    pd.DataFrame({
        "Name": [
            "Mario Zamudio",
            "Joseph Kojo Foli",
            "Avinash Brandon Maharaj",
            "Roberto San Miguel",
        ],
        "Student ID": ["NF1002499", "NF1007842", "NF1002706", "NF1001332"],
        "Primary RQ": ["RQ4", "RQ3 & RQ4", "RQ2", "RQ1"],
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Data: bigquery-public-data.thelook_ecommerce | "
    "External validation: School Specialty LLC (SSL) 2024–2025"
)
