"""
EDA Page — TheLook E-Commerce Dataset (Presentation-First, Graduate Tone)

Exploratory Data Analysis overview for the full dataset.
This page is a placeholder; charts are generated from processed data.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA — TheLook",
    page_icon="🔍",
    layout="wide",
)


# ── CSS: inline hover tooltip for all section headers ─────────────────────────
st.markdown(
    """
    <style>
    .eda-tip-title {
        display: flex;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .eda-tip-title h2 {
        margin: 0; padding: 0;
        font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em;
    }
    .eda-tip-title h3 {
        margin: 0; padding: 0;
        font-size: 1.35rem; font-weight: 600; letter-spacing: -0.01em;
    }
    .eda-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        cursor: help;
        margin-left: 10px;
        flex-shrink: 0;
    }
    .eda-tip-icon { font-size: 0.9rem; color: #888; user-select: none; }
    .eda-tip-box {
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
    .eda-tip-box::after {
        content: "";
        position: absolute;
        top: 100%; left: 50%; margin-left: -6px;
        border: 6px solid transparent;
        border-top-color: rgba(28, 28, 44, 0.97);
    }
    .eda-tip:hover .eda-tip-box { visibility: visible; opacity: 1; }
    .eda-step-badge {
        background: #f0f4ff; border-radius: 6px; padding: 8px 14px; margin-bottom: 8px;
        font-size: 0.75rem; font-weight: 700; color: #2c5282; letter-spacing: 0.08em;
    }
    @media (max-width: 768px) {
        .eda-tip-box { width: 260px; font-size: 0.85rem; }
        .eda-step-badge { font-size: 0.68rem; padding: 6px 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PARQUET = PROCESSED / "returns_eda_v1.parquet"
FEATURE_PARQUET = PROCESSED / "feature_engineered_dataset.parquet"


# ── Tooltips ──────────────────────────────────────────────────────────────────
_TOOLTIPS = {
    "dataset_overview": (
        "**Dataset Source:** BigQuery public dataset `bigquery-public-data.thelook_ecommerce`. "
        "Four tables were merged at the order-item grain: order_items, orders, products, and users. "
        "The analytical unit is one row per order-item, not per order."
    ),
    "schema": (
        "**Schema & Data Types:** Displays all columns and their inferred types after the merge and "
        "standardization pipeline. Key columns include sale_price, cost, item_status, category, brand, "
        "and engineered features like profit_erosion."
    ),
    "status_dist": (
        "**Item Status Distribution:** Order items progress through statuses: Shipped, Complete, "
        "Returned, Cancelled, Processing. Only items with status == 'Returned' are included in "
        "profit erosion calculations. The share of returned items validates the scope and relevance of the analysis."
    ),
    "return_by_cat": (
        "**Return Rate by Category:** Return rate = returned items / total items per category. "
        "Categories with fewer than 200 items are excluded for stability. "
        "The processing cost model applies tiered multipliers (1.0×–1.3×) based on category risk."
    ),
    "margin_dist": (
        "**Item Margin Distribution:** item_margin = sale_price − cost. "
        "For returned items, the margin is reversed and added to the processing cost to form profit_erosion. "
        "The heavy left tail in returned items illustrates why average-based analysis alone is insufficient."
    ),
    "geo_dist": (
        "**Geographic Distribution:** Return rates were compared across all countries in the dataset. "
        "The coefficient of variation (CV) was 3.58%, below the 10% threshold for meaningful geographic tiering. "
        "No geographic multipliers were applied to the processing cost model as a result."
    ),
    "data_quality": (
        "**Data Quality Checks:** The cleaning pipeline checks for duplicates, missing values, "
        "price inconsistencies, status logic errors, and temporal ordering violations. "
        "Flagged rows are logged for audit and remediation."
    ),
    "feature_engineering": (
        "**Engineered Features:** Key derived columns support all four research questions. "
        "profit_erosion = margin_reversal + process_cost is the central outcome variable. "
        "is_high_erosion_customer flags customers at or above the 75th percentile of total erosion."
    ),
    "step_data": (
        "**What is TheLook?** A synthetic but realistic e-commerce dataset published by Google on "
        "BigQuery. It covers ~125K customers, ~500K order items, 29 product categories, and 10 "
        "countries — large enough to support statistical inference but clean enough to validate methods."
    ),
    "step_scope": (
        "**Why returns only?** Only 10.1% of order-items are returned (18,208 / 180,908), but they account for 100% of "
        "profit erosion in this study. The EDA scopes every chart and metric to this subset so that "
        "the analytical story is consistent with the research questions downstream."
    ),
    "step_features": (
        "**Why engineer features?** Raw transaction records do not contain return rates, margins, or "
        "erosion amounts — those must be computed. Feature engineering translates raw prices and "
        "statuses into the outcome measures (profit_erosion) and predictors (return_frequency, "
        "avg_order_value, etc.) used by all four research questions."
    ),
}


def _plain_tip(tooltip_key: str) -> str:
    """Return raw tooltip text for use in st.metric(help=...)."""
    return _TOOLTIPS[tooltip_key]


def _tip_header(label: str, tooltip_key: str, level: int = 2) -> None:
    """Render a section header with an inline CSS hover tooltip."""
    raw = _TOOLTIPS[tooltip_key]
    parts = raw.split("**")
    tip_html = "".join(
        f"<strong>{p}</strong>" if i % 2 == 1 else p
        for i, p in enumerate(parts)
    )
    st.markdown(
        f'<div class="eda-tip-title">'
        f'<h{level}>{label}</h{level}>'
        f'<span class="eda-tip">'
        f'<span class="eda-tip-icon">ℹ️</span>'
        f'<span class="eda-tip-box">{tip_html}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔍 Exploratory Data Analysis — TheLook E-Commerce")

# ── Executive Summary Banner ───────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0f2440 0%, #1a3660 100%);
        border-left: 5px solid #7B1FA2;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 8px;
    ">
        <p style="color:#f0c040; font-size:0.78rem; font-weight:700;
                  letter-spacing:0.12em; text-transform:uppercase; margin:0 0 10px 0;">
            Dataset Overview — Scope &amp; Structure
        </p>
        <p style="color:#e8eaf0; font-size:1.0rem; line-height:1.75; margin:0;">
            <strong style="color:#ffffff;">TheLook is a synthetic e-commerce dataset
            purpose-built for analytics.</strong>
            Four BigQuery tables — <code>order_items</code>, <code>orders</code>,
            <code>products</code>, and <code>users</code> — are merged at the
            <strong>order-item grain</strong>: one analytical row per item purchased.
            The dataset spans ~125K customers, 29 product categories, and 10 countries.
            <strong style="color:#f0c040;">Only returned items drive profit erosion</strong>
            — 10.1% of all order-items — making return behaviour the central
            analytical lens for all four research questions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── 3-Panel Step-Badge Logic Chain ────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        '<div class="eda-step-badge">STEP 1 — WHAT IS THE DATA?</div>',
        unsafe_allow_html=True,
    )
    _tip_header("TheLook Dataset — 4 Tables, 1 Grain", "step_data", level=3)
    st.caption(
        "Four BigQuery tables merged at the order-item grain. "
        "~125K customers · 29 categories · 10 countries."
    )
    st.markdown(
        """
| Table | Rows (approx) | Key columns |
|-------|---------------|-------------|
| `order_items` | 500K+ | `sale_price`, `status`, `product_id` |
| `orders` | 125K+ | `user_id`, timestamps |
| `products` | 30K+ | `cost`, `category`, `brand` |
| `users` | 100K+ | `age`, `gender`, `country` |
"""
    )

with col_b:
    st.markdown(
        '<div class="eda-step-badge">STEP 2 — WHAT IS IN SCOPE?</div>',
        unsafe_allow_html=True,
    )
    _tip_header("Returns = 10.1% of Items, 100% of Erosion", "step_scope", level=3)
    st.caption(
        "Every profit erosion metric is scoped to returned items only. "
        "Non-returned items serve as the baseline."
    )
    st.markdown(
        """
- **Returned items**: `item_status == 'Returned'`
- **Return rate**: 10.1% of all order-item rows (18,208 / 180,908)
- **Profit erosion**: `margin_reversal + process_cost`
- **Processing cost**: USD 12 base × category multiplier (1.0–1.3×)
- **Geographic tiers**: none — return rate CV = 3.58% < 10% threshold
"""
    )

with col_c:
    st.markdown(
        '<div class="eda-step-badge">STEP 3 — WHAT IS ENGINEERED?</div>',
        unsafe_allow_html=True,
    )
    _tip_header("9 Derived Features Feed All 4 RQs", "step_features", level=3)
    st.caption(
        "Feature engineering translates raw prices and statuses into outcome "
        "measures and predictors used by RQ1–RQ4."
    )
    st.markdown(
        """
- `item_margin` = `sale_price − cost`
- `margin_reversal` = item_margin (returned items only)
- `process_cost` = USD 12 × tier multiplier
- `profit_erosion` = `margin_reversal + process_cost`
- `total_profit_erosion` — customer-level sum
- `is_high_erosion_customer` — top 25% flag (RQ3 target)
- `return_frequency`, `avg_order_value` etc. — RQ3/RQ4 predictors
"""
    )

st.divider()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def _load():
    if FEATURE_PARQUET.exists():
        return pd.read_parquet(FEATURE_PARQUET)
    if PARQUET.exists():
        return pd.read_parquet(PARQUET)
    return None


df = _load()

if df is None:
    st.warning("Processed dataset not available. Please ensure the analysis pipeline has been run.")
    st.stop()

# ── Section 1: Dataset Overview ───────────────────────────────────────────────
_tip_header("1. Dataset Overview", "dataset_overview")

total_rows = len(df)
n_orders = df["order_id"].nunique() if "order_id" in df.columns else None
n_customers = df["user_id"].nunique() if "user_id" in df.columns else None
n_products = df["product_id"].nunique() if "product_id" in df.columns else None
n_returned = int(df["is_returned_item"].sum()) if "is_returned_item" in df.columns else None
return_rate = n_returned / total_rows if n_returned is not None else None

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Order-Item Rows", f"{total_rows:,}")
c2.metric("Unique Orders", f"{n_orders:,}" if n_orders else "—")
c3.metric("Unique Customers", f"{n_customers:,}" if n_customers else "—")
c4.metric("Unique Products", f"{n_products:,}" if n_products else "—")
c5.metric(
    "Overall Return Rate",
    f"{return_rate:.1%}" if return_rate is not None else "—",
    help="Returned items / total order-item rows",
)

st.divider()

# ── Section 2: Schema ─────────────────────────────────────────────────────────
_tip_header("2. Schema & Data Types", "schema")
with st.expander("View column list", expanded=False):
    schema_df = pd.DataFrame(
        {"column": df.columns, "dtype": df.dtypes.astype(str).values}
    )
    st.dataframe(schema_df, width='stretch', hide_index=True)

st.divider()

# ── Section 3: Item Status Distribution ───────────────────────────────────────
_tip_header("3. Item Status Distribution", "status_dist")
st.markdown(
    "Order items progress through statuses: `Shipped`, `Complete`, `Returned`, `Cancelled`, `Processing`."
)

if "item_status" in df.columns:
    status_counts = df["item_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    status_counts["Share (%)"] = (status_counts["Count"] / total_rows * 100).round(2)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        import plotly.express as px
        fig = px.bar(
            status_counts,
            x="Status",
            y="Count",
            text="Share (%)",
            color="Status",
            title="Order Item Status Counts",
            labels={"Count": "Number of Items"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_table:
        st.dataframe(status_counts, width='stretch', hide_index=True)
else:
    st.info("Column `item_status` not found in dataset.")

st.divider()

# ── Section 4: Return Rate by Category ────────────────────────────────────────
_tip_header("4. Return Rate by Product Category", "return_by_cat")
st.markdown(
    "Categories vary widely in return rates. "
    "The processing cost model applies tiered multipliers (1.0×–1.3×) based on category."
)

if "category" in df.columns and "is_returned_item" in df.columns:
    cat_agg = (
        df.groupby("category")
        .agg(
            total_items=("order_id", "size"),
            returned=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned"] / x["total_items"])
        .query("total_items >= 200")
        .sort_values("return_rate", ascending=False)
        .reset_index()
    )
    cat_agg["return_rate_pct"] = (cat_agg["return_rate"] * 100).round(2)

    import plotly.express as px
    fig = px.bar(
        cat_agg,
        x="return_rate_pct",
        y="category",
        orientation="h",
        color="return_rate_pct",
        color_continuous_scale="Reds",
        title="Return Rate (%) by Product Category (min 200 items)",
        labels={"return_rate_pct": "Return Rate (%)", "category": "Category"},
        text="return_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        height=max(400, len(cat_agg) * 28),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Columns `category` or `is_returned_item` not found.")

st.divider()

# ── Section 5: Margin Distribution ───────────────────────────────────────────
_tip_header("5. Item Margin Distribution", "margin_dist")
st.markdown(
    "`item_margin = sale_price − cost`. For returned items, this becomes the **margin reversal** "
    "component of profit erosion."
)

if "item_margin" in df.columns:
    import plotly.express as px

    col_all, col_ret = st.columns(2)
    with col_all:
        fig_all = px.histogram(
            df,
            x="item_margin",
            nbins=60,
            title="All Items — Margin Distribution",
            labels={"item_margin": "Item Margin (USD)"},
            color_discrete_sequence=["steelblue"],
        )
        fig_all.update_layout(height=350)
        st.plotly_chart(fig_all, use_container_width=True)

    if "is_returned_item" in df.columns:
        returned_df = df[df["is_returned_item"] == 1]
        with col_ret:
            fig_ret = px.histogram(
                returned_df,
                x="item_margin",
                nbins=60,
                title="Returned Items Only — Margin Reversal",
                labels={"item_margin": "Margin Reversal (USD)"},
                color_discrete_sequence=["coral"],
            )
            fig_ret.update_layout(height=350)
            st.plotly_chart(fig_ret, use_container_width=True)

    median_all = df["item_margin"].median()
    median_ret = returned_df["item_margin"].median() if "is_returned_item" in df.columns else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Median Margin — All Items", f"USD {median_all:.2f}")
    if median_ret is not None:
        c2.metric("Median Margin — Returned", f"USD {median_ret:.2f}")
    if "profit_erosion" in df.columns:
        median_erosion = df.loc[df["is_returned_item"] == 1, "profit_erosion"].median()
        c3.metric(
            "Median Profit Erosion per Return",
            f"USD {median_erosion:.2f}",
            help="Margin reversal + processing cost",
        )
else:
    st.info("Column `item_margin` not found. Run feature engineering first.")

st.divider()

# ── Section 6: Geographic Distribution ───────────────────────────────────────
_tip_header("6. Geographic Distribution", "geo_dist")
st.markdown(
    "Return rates show low variance across countries (CV = 3.58% < 10% threshold), "
    "so no geographic tiers were applied to the processing cost model."
)

if "country" in df.columns and "is_returned_item" in df.columns:
    geo_agg = (
        df.groupby("country")
        .agg(
            total_items=("order_id", "size"),
            returned=("is_returned_item", "sum"),
        )
        .assign(return_rate=lambda x: x["returned"] / x["total_items"])
        .sort_values("total_items", ascending=False)
        .reset_index()
    )
    geo_agg["return_rate_pct"] = (geo_agg["return_rate"] * 100).round(2)

    import plotly.express as px
    fig = px.bar(
        geo_agg,
        x="country",
        y="return_rate_pct",
        color="total_items",
        color_continuous_scale="Blues",
        title="Return Rate by Country (color = order volume)",
        labels={"return_rate_pct": "Return Rate (%)", "country": "Country", "total_items": "Items"},
        text="return_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Columns `country` or `is_returned_item` not found.")

st.divider()

# ── Section 7: Data Quality Summary ──────────────────────────────────────────
_tip_header("7. Data Quality Summary", "data_quality")
st.markdown(
    "The cleaning pipeline checks for duplicates, missing values, "
    "price inconsistencies, status logic errors, and temporal ordering violations."
)

quality_checks = {
    "Duplicate rows": "Detected and removed (keep first)",
    "Missing values": "Reported; numeric imputed with median for ML (RQ3 only)",
    "Price inconsistencies": "Flagged: sale > retail, cost > sale, negative prices",
    "Status consistency": "Flagged: returned-not-delivered, item/order status mismatch",
    "Temporal consistency": "Flagged: delivered-before-shipped, returned-before-delivered",
    "Categorical cleaning": "Stripped whitespace; case-normalised for grouping",
}

st.table(pd.DataFrame({"Check": quality_checks.keys(), "Action": quality_checks.values()}))

review_path = PROCESSED / "data_to_review.csv"
if review_path.exists():
    review_df = pd.read_csv(review_path)
    st.caption(f"Flagged records for review: {len(review_df):,} rows")

st.divider()

# ── Section 8: Feature Engineering Overview ───────────────────────────────────
_tip_header("8. Engineered Features — Quick Reference", "feature_engineering")
st.markdown("Key derived columns computed from the raw transaction data:")

features_table = pd.DataFrame(
    {
        "Feature": [
            "is_returned_item",
            "item_margin",
            "item_margin_pct",
            "discount_amount",
            "margin_reversal",
            "process_cost",
            "profit_erosion",
            "total_profit_erosion",
            "is_high_erosion_customer",
        ],
        "Level": [
            "Item",
            "Item",
            "Item",
            "Item",
            "Item (returned only)",
            "Item (returned only)",
            "Item (returned only)",
            "Customer",
            "Customer",
        ],
        "Formula / Definition": [
            "1 if item_status == 'Returned', else 0",
            "sale_price − cost",
            "(sale_price − cost) / sale_price",
            "retail_price − sale_price",
            "item_margin (for returned items)",
            "USD 12 base × category tier multiplier (1.0 / 1.15 / 1.3)",
            "margin_reversal + process_cost",
            "SUM(profit_erosion) per customer",
            "1 if total_profit_erosion ≥ 75th percentile",
        ],
    }
)
st.dataframe(features_table, width='stretch', hide_index=True)

st.caption(
    "Full feature definitions and processing cost methodology are documented "
    "in the project technical documentation."
)

# ── Section 9: Return Rate Trend (2019–2026) ──────────────────────────────────
st.divider()
_tip_header("9. Return Rate Trend — 2019 to 2026", "dataset_overview")
st.caption(
    "Monthly return rate (returned items ÷ total items) aggregated from the full dataset. "
    "The 3-month rolling average smooths short-term noise to reveal the structural trend."
)

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _df_trend = pd.read_parquet(FEATURE_PARQUET, columns=["order_created_at", "is_returned_item"])
    _df_trend["order_created_at"] = pd.to_datetime(_df_trend["order_created_at"], errors="coerce")
    _df_trend = _df_trend.dropna(subset=["order_created_at"])
    _df_trend["month"] = _df_trend["order_created_at"].dt.to_period("M").dt.to_timestamp()

    _monthly = (
        _df_trend.groupby("month")["is_returned_item"]
        .agg(returned="sum", total="count")
        .reset_index()
    )
    _monthly["return_rate"] = _monthly["returned"] / _monthly["total"] * 100
    _monthly["rolling_avg"] = _monthly["return_rate"].rolling(3, min_periods=1).mean()
    _monthly = _monthly.sort_values("month")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=_monthly["month"], y=_monthly["return_rate"],
        mode="lines",
        name="Monthly Return Rate",
        line=dict(color="#90caf9", width=1.2),
        opacity=0.65,
        hovertemplate="<b>%{x|%b %Y}</b><br>Return rate: %{y:.2f}%<extra></extra>",
    ))
    fig_trend.add_trace(go.Scatter(
        x=_monthly["month"], y=_monthly["rolling_avg"],
        mode="lines",
        name="3-Month Rolling Average",
        line=dict(color="#1565C0", width=2.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>3-mo avg: %{y:.2f}%<extra></extra>",
    ))
    fig_trend.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Month", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        yaxis=dict(title="Return Rate (%)", showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    _n_months = len(_monthly)
    _mean_rate = _monthly["return_rate"].mean()
    _max_rate = _monthly["return_rate"].max()
    _max_month = _monthly.loc[_monthly["return_rate"].idxmax(), "month"].strftime("%b %Y")
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("Months Observed", f"{_n_months}")
    col_t2.metric("Mean Monthly Return Rate", f"{_mean_rate:.1f}%")
    col_t3.metric("Peak Month", _max_month, f"{_max_rate:.1f}%")

except Exception as _e:
    st.info("Return rate trend chart unavailable — processed dataset not loaded.", icon="ℹ️")
