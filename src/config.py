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
