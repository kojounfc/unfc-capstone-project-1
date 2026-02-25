"""
EDA Page — TheLook E-Commerce Dataset
Exploratory Data Analysis overview for the full dataset.
This page is a placeholder; charts are generated from processed data.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
PARQUET = PROCESSED / "returns_eda_v1.parquet"
FEATURE_PARQUET = PROCESSED / "feature_engineered_dataset.parquet"

st.set_page_config(
    page_title="EDA — TheLook",
    page_icon="🔍",
    layout="wide",
)

st.title("Exploratory Data Analysis — TheLook E-Commerce")
st.markdown(
    """
**Source**: `bigquery-public-data.thelook_ecommerce` (Google BigQuery)

Four tables were merged at the **order-item grain** into a single analytical dataset:

| Table | Key Columns |
|---|---|
| `order_items` | `order_item_id`, `order_id`, `product_id`, `sale_price`, `status` |
| `orders` | `order_id`, `user_id`, `order_status`, timestamps |
| `products` | `product_id`, `brand`, `category`, `department`, `retail_price`, `cost` |
| `users` | `user_id`, `age`, `gender`, `country`, `traffic_source`, `created_at` |

The pipeline in `src/data_processing.py` loads, merges, and standardizes types.
Feature engineering (`src/feature_engineering.py`) adds return flags, margins, and
profit erosion metrics.
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
    st.warning(
        "Processed dataset not found. Run the master notebook "
        "(`notebooks/profit_erosion_analysis.ipynb`) to generate it."
    )
    st.stop()

# ── Section 1: Dataset Overview ───────────────────────────────────────────────
st.header("1. Dataset Overview")

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
st.header("2. Schema & Data Types")
with st.expander("View column list", expanded=False):
    schema_df = pd.DataFrame(
        {"column": df.columns, "dtype": df.dtypes.astype(str).values}
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

st.divider()

# ── Section 3: Item Status Distribution ───────────────────────────────────────
st.header("3. Item Status Distribution")
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
        st.dataframe(status_counts, use_container_width=True, hide_index=True)
else:
    st.info("Column `item_status` not found in dataset.")

st.divider()

# ── Section 4: Return Rate by Category ────────────────────────────────────────
st.header("4. Return Rate by Product Category")
st.markdown(
    "Categories vary widely in return rates. "
    "The processing cost model applies tiered multipliers (1.0x–1.3x) based on category."
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
st.header("5. Item Margin Distribution")
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
st.header("6. Geographic Distribution")
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
st.header("7. Data Quality Summary")
st.markdown(
    "The cleaning pipeline in `src/data_cleaning.py` checks for duplicates, missing values, "
    "price inconsistencies, status logic errors, and temporal ordering violations. "
    "Flagged rows are written to `data/processed/data_to_review.csv` for review."
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
    st.caption(f"Flagged records for review: {len(review_df):,} rows → `data/processed/data_to_review.csv`")

st.divider()

# ── Section 8: Feature Engineering Overview ───────────────────────────────────
st.header("8. Engineered Features — Quick Reference")
st.markdown("Key derived columns added by `src/feature_engineering.py`:")

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
st.dataframe(features_table, use_container_width=True, hide_index=True)

st.caption(
    "For the full data dictionary see `docs/DATA_DICTIONARY.md`. "
    "For processing cost methodology see `docs/PROCESSING_COST_METHODOLOGY.md`."
)
