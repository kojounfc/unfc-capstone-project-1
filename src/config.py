"""
Configuration constants for the Profit Erosion E-commerce Capstone Project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
RQ1_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq1"
RQ1_FIGURES_DIR = FIGURES_DIR / "rq1"
EDA_PROCESSED_DIR = PROCESSED_DATA_DIR / "eda"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
FEATURES_PROCESSED_DIR = PROCESSED_DATA_DIR / "feature_engineering"
FEATURES_FIGURES_DIR = FIGURES_DIR / "feature_engineering"
US07_PROCESSED_DIR = PROCESSED_DATA_DIR / "us07"
US07_FIGURES_DIR = FIGURES_DIR / "us07"
RQ2_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq2"
RQ2_FIGURES_DIR = FIGURES_DIR / "rq2"
RQ3_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq3"
RQ3_FIGURES_DIR = FIGURES_DIR / "rq3"
RQ4_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq4"
RQ4_FIGURES_DIR = FIGURES_DIR / "rq4"

# Raw data files
RAW_FILES = {
    "order_items": RAW_DATA_DIR / "order_items.csv",
    "orders": RAW_DATA_DIR / "orders.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "users": RAW_DATA_DIR / "users.csv",
}

# Processed data files
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "returns_eda_v1.parquet"
PROCESSED_CSV = PROCESSED_DATA_DIR / "returns_eda_v1.csv"
RQ1_PROCESSED_PARQUET = RQ1_PROCESSED_DIR / "returns_eda_v1.parquet"
RQ1_PROCESSED_CSV = RQ1_PROCESSED_DIR / "returns_eda_v1.csv"

# Date columns for parsing
DATETIME_COLS = [
    "item_created_at",
    "item_shipped_at",
    "item_delivered_at",
    "item_returned_at",
    "order_created_at",
    "order_shipped_at",
    "order_delivered_at",
    "order_returned_at",
    "user_created_at",
]

# Numeric columns
NUMERIC_COLS = ["sale_price", "retail_price", "cost", "age", "num_of_item"]

# String columns (mixed-type handling)
STRING_COLS = ["postal_code", "sku", "user_geom", "street_address", "email"]

# Minimum rows threshold for aggregation analysis
MIN_ROWS_THRESHOLD = 200

# --- RQ3: Predictive Modeling ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
AUC_THRESHOLD = 0.70
CUSTOMER_TARGETS_CSV = PROCESSED_DATA_DIR / "customer_profit_erosion_targets.csv"
SSL_RETURNS_CSV = RAW_DATA_DIR / "SSL_Returns_df_yoy.csv"
RQ3_TARGET = "is_high_erosion_customer"
RQ3_CANDIDATE_FEATURES = [
    "order_frequency",
    "return_frequency",
    "customer_return_rate",
    "avg_basket_size",
    "avg_order_value",
    "customer_tenure_days",
    "purchase_recency_days",
    "total_items",
    "total_sales",
    "total_margin",
    "avg_item_price",
    "avg_item_margin",
]
RQ3_LEAKAGE_COLUMNS = [
    "total_profit_erosion",
    "total_margin_reversal",
    "total_process_cost",
    "is_high_erosion_customer",
    "profit_erosion_quartile",
    "erosion_percentile_rank",
    "user_id",
]

# --- RQ3: Sensitivity Analysis ---
SENSITIVITY_BASE_COSTS = [8.0, 10.0, 12.0, 14.0, 18.0]
SENSITIVITY_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
