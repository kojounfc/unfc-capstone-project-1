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

# ── CSS: hover tooltip system (mirrors RQ pages) ─────────────────────────────
st.markdown(
    """
    <style>
    .home-tip-title {
        display: flex; align-items: center; margin-bottom: 0.4rem;
    }
    .home-tip-title h2 { margin:0; padding:0; font-size:1.5rem; font-weight:700; letter-spacing:-0.01em; }
    .home-tip-title h3 { margin:0; padding:0; font-size:1.35rem; font-weight:600; letter-spacing:-0.01em; }
    .home-tip {
        position: relative; display: inline-flex; align-items: center;
        cursor: help; margin-left: 10px; flex-shrink: 0;
    }
    .home-tip-icon { font-size: 0.9rem; color: #888; user-select: none; }
    .home-tip-box {
        visibility: hidden; opacity: 0; width: 380px;
        background-color: rgba(28,28,44,0.97); color: #e4e4f0;
        text-align: left; border-radius: 8px; padding: 14px 18px;
        font-size: 0.95rem; line-height: 1.65;
        position: absolute; z-index: 9999;
        bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%);
        transition: opacity 0.2s ease;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
        pointer-events: none; white-space: normal;
    }
    .home-tip-box::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -6px;
        border: 6px solid transparent; border-top-color: rgba(28,28,44,0.97);
    }
    .home-tip:hover .home-tip-box { visibility: visible; opacity: 1; }
    .home-step-badge {
        background: #f0f4ff; border-radius: 6px; padding: 8px 14px; margin-bottom: 8px;
        font-size: 0.75rem; font-weight: 700; color: #2c5282; letter-spacing: 0.08em;
    }
    @media (max-width: 768px) {
        .home-tip-box { width: 260px; font-size: 0.85rem; }
        .home-step-badge { font-size: 0.68rem; padding: 6px 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Project root ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
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

_cat_path = PROCESSED / "rq1" / "rq1_product_profit_erosion_by_category.parquet"
if _cat_path.exists():
    _cat_df = pd.read_parquet(_cat_path)
    _rq1_top_cat = _cat_df.nlargest(1, "total_profit_erosion").iloc[0]["category"]

_rq1_top_brand = None
_brand_path = PROCESSED / "rq1" / "rq1_product_profit_erosion_by_brand.parquet"
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

_val4_path = REPORTS_RQ4 / "rq4_validation_summary.csv"
_align4_path = REPORTS_RQ4 / "rq4_ssl_coefficient_alignment.csv"
_coef4_path = REPORTS_RQ4 / "rq4_thelook_coefficients.csv"
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

_high_erosion_pct = 75

# ── Tooltips ──────────────────────────────────────────────────────────────────
_TOOLTIPS = {
    "kpi_datasource": (
        "**Primary Data Source:** bigquery-public-data.thelook_ecommerce (Google BigQuery). "
        "Synthetic e-commerce dataset covering order_items, orders, products, and users tables. "
        "Used for all four research questions as the primary analysis population."
    ),
    "kpi_validation": (
        "**External Validation Dataset:** School Specialty LLC (SSL) — a real-world B2B "
        "educational supplies retailer. ~234K return order lines, ~16.7K accounts, 2024–2025. "
        "Used for directional validation of RQ1–RQ4 findings to test cross-domain generalizability."
    ),
    "kpi_cost": (
        "**Processing Cost Base Rate:** $12.00 per return — conservative mid-range of the "
        "$10–$25 literature range. Components: Customer Care ($4.00), Inspection ($2.50), "
        "Restocking ($3.00), Logistics ($2.50). Adjusted by category tier multipliers "
        "(1.0× to 1.3×) based on margin CV analysis."
    ),
    "kpi_threshold": (
        "**High-Erosion Classification Threshold:** RQ3 binary target uses the 75th percentile "
        "of total_profit_erosion — the top 25% of returning customers are labelled high-risk. "
        "Rationale: standard quartile-based segmentation; operationally feasible intervention scope."
    ),
}


def _tip_header(label: str, tooltip_key: str, level: int = 3) -> None:
    """Render a section header with an inline CSS hover tooltip — mirrors RQ pages."""
    raw = _TOOLTIPS[tooltip_key]
    parts = raw.split("**")
    tip_html = "".join(
        f"<strong>{p}</strong>" if i % 2 == 1 else p
        for i, p in enumerate(parts)
    )
    st.markdown(
        f'<div class="home-tip-title">'
        f'<h{level}>{label}</h{level}>'
        f'<span class="home-tip">'
        f'<span class="home-tip-icon">ℹ️</span>'
        f'<span class="home-tip-box">{tip_html}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _plain_tip(key: str) -> str:
    return _TOOLTIPS[key].replace("**", "")


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

# ── Executive Summary Banner ──────────────────────────────────────────────────
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
        ">Project Overview — Research Framework &amp; Key Results</p>
        <p style="color: #e8eaf0; font-size: 1.0rem; line-height: 1.75; margin: 0;">
            <strong style="color: #ffffff;">Product returns are economic reversal events
            that directly erode realized revenue and margin.</strong>
            This project quantifies profit erosion across four research lenses —
            category risk (RQ1), customer segmentation (RQ2),
            predictive targeting (RQ3), and behavioral econometrics (RQ4) —
            using TheLook E-Commerce (BigQuery) as primary data and
            School Specialty LLC (B2B, ~13,600 accounts) for external validation.
            <strong style="color: #f0c040;">All four null hypotheses are rejected</strong>
            — confirming that return-driven profit erosion is predictable, segmentable,
            and materially linked to customer behavioral patterns.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border: 0; border-top: 1px solid rgba(49,51,63,0.3); margin: 20px 0 24px 0;'>",
    unsafe_allow_html=True,
)

# ── KPI Cards (above tabs — consistent with RQ3/RQ4 pattern) ─────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Data Source",
    "TheLook E-Commerce",
    "Google BigQuery",
    help=_plain_tip("kpi_datasource"),
)
col2.metric(
    "External Validation",
    "School Specialty LLC",
    f"{_ssl_accounts_rq3:,} accounts" if _ssl_accounts_rq3 else "B2B retailer",
    help=_plain_tip("kpi_validation"),
)
col3.metric(
    "Processing Cost Base",
    f"{_base_str} / return",
    "Category tier adjusted",
    help=_plain_tip("kpi_cost"),
)
col4.metric(
    "High-Risk Customers",
    f"Top {100 - _high_erosion_pct}%",
    f"Above {_high_erosion_pct}th percentile of erosion",
    help=_plain_tip("kpi_threshold"),
)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ov, tab_cost, tab_team = st.tabs(["📋 Overview", "⚙️ Cost Model", "👥 Team"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ov:

    # ── 3-Panel Logic Chain ───────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            '<div class="home-step-badge">STEP 1 — THE PROBLEM</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### The Economic Reframe")
        st.caption(
            "Returns are traditionally tracked as operational events — counts and rates. "
            "This project reframes them as economic reversal events that destroy realized margin."
        )
        st.markdown(
            f"""
**Core formula:**
```
Profit Erosion = Margin Reversal + Processing Cost
```
- **Margin Reversal** — sale price − cost (margin lost on the returned item)
- **Processing Cost** — {_base_str} base per return, category-tier adjusted

The combined metric captures both the financial loss on the item itself
and the operational cost of handling the return.
"""
        )

    with col_b:
        st.markdown(
            '<div class="home-step-badge">STEP 2 — THE DATA</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Two Complementary Datasets")
        st.caption(
            "TheLook provides the analytical foundation. SSL provides a real-world "
            "B2B stress-test to validate that findings generalize beyond synthetic data."
        )
        st.markdown(
            """
**Primary — TheLook E-Commerce (BigQuery)**

| Table | Content |
|-------|---------|
| `order_items` | Item-level transactions |
| `orders` | Order-level information |
| `products` | Catalog with cost & pricing |
| `users` | Customer demographics |

**External Validation — School Specialty LLC (SSL)**

~234K return order lines · ~16.7K accounts · 2024–2025 · B2B retailer
"""
        )

    with col_c:
        st.markdown(
            '<div class="home-step-badge">STEP 3 — THE FRAMEWORK</div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### Four Research Lenses")
        st.caption(
            "Each RQ addresses a distinct analytical layer — from descriptive patterns "
            "to predictive models to econometric associations."
        )
        st.markdown(
            f"""
| RQ | Lens | Method |
|----|------|--------|
| **RQ1** | Category risk | Kruskal-Wallis |
| **RQ2** | Customer segments | K-Means + Gini |
| **RQ3** | Predictive targeting | RF / GB / LR (AUC target >{AUC_THRESHOLD}) |
| **RQ4** | Behavioral drivers | Log-Linear OLS |

All four RQs include SSL external validation.
Use the **sidebar** to navigate to each RQ page.
"""
        )

    st.divider()

    # ── Research Questions Results Table ─────────────────────────────────────
    st.subheader("Research Questions — Key Results")

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
        "Decision": ["✅ Reject H₀", "✅ Reject H₀₂", "✅ Reject H₀₃", "✅ Reject H₀₄"],
        "Key Result": [_rq1_result, _rq2_result, _rq3_result, _rq4_result],
    })
    st.dataframe(rq_df, width='stretch', hide_index=True)

    st.divider()

    # ── Navigation Guide ──────────────────────────────────────────────────────
    st.subheader("Navigation")
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_cost:
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
            st.dataframe(pd.DataFrame(_rows), width='stretch', hide_index=True)
        else:
            st.warning("Cost components unavailable. Using default values.")

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
            st.dataframe(pd.DataFrame(_tier_rows), width='stretch', hide_index=True)
            st.caption(
                "Tier multipliers justified by Margin CV = 59.4% across categories "
                "(exceeds 15% threshold for tiered treatment)."
            )
        else:
            st.warning("Tier multiplier data unavailable.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab_team:
    st.subheader("Project Team")
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
        width='stretch',
        hide_index=True,
    )
    st.caption(
        "Data: bigquery-public-data.thelook_ecommerce | "
        "External validation: School Specialty LLC (SSL) 2024–2025 | "
        "DAMO-699-4 · University of Niagara Falls Canada · Winter 2026"
    )
