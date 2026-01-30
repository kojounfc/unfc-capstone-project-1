# Data Processing Module - Technical Reference

## Executive Summary

The `data_processing.py` module orchestrates the complete data pipeline for the Profit Erosion E-commerce Capstone Project. It handles data loading, cleaning, merging, feature engineering, and type standardization to produce a unified, analysis-ready dataset at the order-item grain level.

---

## 1. Module Overview

### 1.1 Data Pipeline Architecture

```
Raw CSV Files (4 tables)
        ↓
    Load & Clean Column Names
        ↓
    Merge at Order-Item Grain
        ↓
    Standardize Data Types
        ↓
    Engineer Return Features
        ↓
    Calculate Margin Metrics
        ↓
    Save to Parquet & CSV
```

**Output Dataset:**
- **Grain:** One row per order item (~465,000 rows)
- **Columns:** 25+ merged attributes from all 4 source tables
- **Format:** Parquet (optimized) + CSV (portable)

---

## 2. Core Functions

### 2.1 clean_columns()

**Purpose:** Normalize column names by removing whitespace and BOM characters

**Function Signature:**
```python
def clean_columns(df: pd.DataFrame) -> pd.DataFrame
```

**Input:**
- DataFrame with potentially messy column names
- Common issues:
  - Leading/trailing spaces: `" column_name "`
  - Byte Order Marks (BOM): `"\ufeffcolumn_name"`
  - Mixed case inconsistencies

**Processing:**
1. Convert column names to string type
2. Strip leading/trailing whitespace
3. Remove UTF-8 BOM characters
4. Return cleaned DataFrame

**Output:**
- DataFrame with standardized column names
- Example: `" Status "` → `"Status"`

**Usage Example:**
```python
df_raw = pd.read_csv("data.csv")
df_clean = clean_columns(df_raw)
```

---

### 2.2 load_raw_data()

**Purpose:** Load four raw CSV files with optional column selection and datetime parsing

**Function Signature:**
```python
def load_raw_data(
    raw_dir: Optional[Path] = None,
    order_items_cols: Optional[list] = None,
    orders_cols: Optional[list] = None,
    products_cols: Optional[list] = None,
    users_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| raw_dir | Path | config.RAW_DATA_DIR | Directory containing CSV files |
| order_items_cols | list | None (load all) | Columns to load from order_items.csv |
| orders_cols | list | None (load all) | Columns to load from orders.csv |
| products_cols | list | None (load all) | Columns to load from products.csv |
| users_cols | list | None (load all) | Columns to load from users.csv |

**Processing:**
1. Load each CSV file with specified columns
2. Auto-parse datetime columns (shipped_at, delivered_at, returned_at)
3. Clean column names in each table
4. Return tuple of four DataFrames

**Datetime Auto-Parsing:**
- order_items: `shipped_at`, `delivered_at`, `returned_at`
- orders: `shipped_at`, `delivered_at`, `returned_at`
- users: `created_at`

**Return Value:**
- Tuple: `(order_items_df, orders_df, products_df, users_df)`

**Error Handling:**
- Raises `FileNotFoundError` if CSV files missing from raw_dir

**Usage Example:**
```python
# Load all columns (default)
oi, orders, products, users = load_raw_data()

# Load specific columns only (memory efficient)
oi, orders, products, users = load_raw_data(
    order_items_cols=["id", "order_id", "product_id", "status", "sale_price"],
    orders_cols=["order_id", "user_id", "status"],
)
```

---

### 2.3 remove_unnecessary_columns()

**Purpose:** Remove or keep specific columns with detailed reporting

**Function Signature:**
```python
def remove_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[list] = None,
    columns_to_keep: Optional[list] = None,
) -> Tuple[pd.DataFrame, Dict]
```

**Modes of Operation:**

#### Mode 1: Drop Specific Columns
```python
df_reduced, report = remove_unnecessary_columns(
    df,
    columns_to_drop=["user_geom", "street_address"]
)
```

#### Mode 2: Keep Only Specific Columns (Whitelist)
```python
df_subset, report = remove_unnecessary_columns(
    df,
    columns_to_keep=["order_id", "user_id", "sale_price", "cost"]
)
```

**Return Value:** Tuple of (DataFrame, report dictionary)

**Report Contents:**
```python
{
    "mode": "drop_mode|keep_mode|no_action",
    "columns_actually_dropped": [...],      # or columns_actually_kept
    "non_existing_columns": [...],           # Requested but not found
    "initial_column_count": 35,
    "final_column_count": 33,
    "columns_removed": 2,
    "remaining_columns": [...]
}
```

---

### 2.4 merge_datasets()

**Purpose:** Merge four source tables into a single unified dataset at order-item grain

**Function Signature:**
```python
def merge_datasets(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame
```

**Merge Strategy:**

```
order_items
    ↓ (merge on order_id, LEFT JOIN)
orders → Extract: order_status, order_shipped_at, order_delivered_at, order_returned_at, user_id, num_of_item
    ↓ (merge on product_id, LEFT JOIN)
products → Extract: category, brand, department, cost, retail_price
    ↓ (merge on user_id, LEFT JOIN)
users → Extract: first_name, last_name, gender, city, country, age
    ↓
Unified Dataset (order-item grain)
```

**Column Renaming:**

Prevents collisions between tables with overlapping column names:

| Original Column | Renamed To | Source | Reason |
|-----------------|------------|--------|--------|
| status | item_status | order_items | Distinguish from order_status |
| id | order_item_id | order_items | Clarity on grain |
| shipped_at | item_shipped_at | order_items | Distinguish from order level |
| delivered_at | item_delivered_at | order_items | Distinguish from order level |
| returned_at | item_returned_at | order_items | Distinguish from order level |
| status | order_status | orders | Distinguish from item_status |
| shipped_at | order_shipped_at | orders | Order-level timestamp |
| delivered_at | order_delivered_at | orders | Order-level timestamp |
| returned_at | order_returned_at | orders | Order-level timestamp |
| id | user_id | users | Clarity on dimension |
| created_at | user_created_at | users | Distinguish from order created_at |
| gender | user_gender | users | Distinguish from item gender |
| id | product_id | products | Already expected in order_items |

**Join Types:**
- All joins are LEFT JOIN on order_items
- Preserves all order items even if product/user metadata missing
- Validates grain: `validate="many_to_one"` ensures correct key relationships

**Resulting Dataset:**
- **Grain:** One row per order item
- **Record Count:** Same as input order_items (465k+)
- **Column Count:** 28+ columns (depending on input column selection)

---

### 2.5 engineer_return_features()

**Purpose:** Create return-related flags for analysis

**Function Signature:**
```python
def engineer_return_features(df: pd.DataFrame) -> pd.DataFrame
```

**Features Created:**

| Column | Type | Logic | Usage |
|--------|------|-------|-------|
| is_returned_item | int (0/1) | item_status.lower() == "returned" | Item-level return indicator |
| is_returned_order | int (0/1) | order_status.lower() == "returned" | Order-level return indicator |

**Processing:**
1. Case-insensitive status comparison (handles "Returned", "RETURNED", etc.)
2. Convert boolean to integer (0/1) for mathematical operations
3. Return DataFrame with new columns added

**Usage Example:**
```python
df = engineer_return_features(df)

# Filter to returned items
returned_items = df[df["is_returned_item"] == 1]

# Calculate return rate
return_rate = df["is_returned_item"].sum() / len(df)
```

---

### 2.6 calculate_margins()

**Purpose:** Compute profit and discount metrics for profit erosion analysis

**Function Signature:**
```python
def calculate_margins(df: pd.DataFrame) -> pd.DataFrame
```

**Features Created:**

#### Discount Metrics
| Column | Formula | Logic |
|--------|---------|-------|
| discount_amount | retail_price - sale_price | Absolute discount in dollars |
| discount_pct | discount_amount / retail_price | Discount as percentage (handles div by zero) |

#### Margin Metrics
| Column | Formula | Logic |
|--------|---------|-------|
| item_margin | sale_price - cost | Absolute profit per item |
| item_margin_pct | item_margin / sale_price | Profit margin percentage |

**Numeric Coercion:**
- All price columns converted to numeric with `errors="coerce"`
- Invalid values (non-numeric) become NaN
- NaN handling:
  - When retail_price = 0: discount_pct = NaN (prevents division by zero)
  - When sale_price = 0: item_margin_pct = NaN (prevents division by zero)

**Usage Example:**
```python
df = calculate_margins(df)

# Analyze return-related margin loss
returned_margin_loss = df[df["is_returned_item"] == 1]["item_margin"].sum()

# Find most profitable items
profitable = df.nlargest(10, "item_margin")
```

---

### 2.7 standardize_dtypes()

**Purpose:** Ensure consistent data types for processing and serialization

**Function Signature:**
```python
def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame
```

**Type Standardization Rules:**

| Pattern/Column | Target Type | Reason |
|---|---|---|
| Columns containing 'id' | string | Prevents arithmetic operations on categorical IDs |
| postal_code, sku, user_geom, street_address, email | string | Mixed-type columns requiring text handling |
| item_created_at, item_shipped_at, etc. (config.DATETIME_COLS) | datetime64[ns, UTC] | Enables time-based analysis |
| sale_price, retail_price, cost, age, num_of_item | float64 | Enables numerical calculations |

**Processing Steps:**
1. Identify all ID columns (pattern matching on column name)
2. Convert each category to appropriate type
3. Handle datetime columns with UTC timezone
4. Use `errors="coerce"` to convert invalid values to NaN
5. Return standardized DataFrame

**Error Handling:**
- Invalid values don't raise errors, become NaN
- Allows graceful handling of messy real-world data
- NaN values can be investigated separately

---

### 2.8 build_analysis_dataset()

**Purpose:** Execute the complete data pipeline end-to-end

**Function Signature:**
```python
def build_analysis_dataset(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    save_output: bool = True,
) -> pd.DataFrame
```

**Pipeline Steps:**
1. Load raw CSV files
2. Merge datasets at order-item grain
3. Standardize data types
4. Engineer return features
5. Calculate margin metrics
6. Save to Parquet (snappy compression) and CSV

**Parameters:**
- raw_dir: Source for CSV files (default: config.RAW_DATA_DIR)
- output_dir: Destination for processed files (default: config.PROCESSED_DATA_DIR)
- save_output: Whether to write files to disk (default: True)

**Output Files:**
- `returns_eda_v1.parquet` (optimized for analysis)
- `returns_eda_v1.csv` (universal compatibility)

**Usage Example:**
```python
# Build full pipeline with default settings
df = build_analysis_dataset()

# Custom directories
df = build_analysis_dataset(
    raw_dir=Path("data/custom_raw"),
    output_dir=Path("data/custom_output"),
    save_output=True
)

# Build but don't save (for testing/validation)
df = build_analysis_dataset(save_output=False)
```

---

### 2.9 load_processed_data()

**Purpose:** Load pre-processed dataset for immediate analysis

**Function Signature:**
```python
def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame
```

**Parameters:**
- file_path: Path to parquet file (default: config.PROCESSED_PARQUET)

**Return Value:**
- DataFrame with all preprocessing applied

**Error Handling:**
- Raises `FileNotFoundError` if parquet file doesn't exist
- Use `build_analysis_dataset()` if file missing

**Usage Example:**
```python
# Load from default location
df = load_processed_data()

# Load from custom location
df = load_processed_data(Path("data/custom/dataset.parquet"))
```

---

## 3. Data Transformation Summary

### 3.1 Record Count Changes
| Stage | Record Count | Change |
|-------|--------------|--------|
| order_items (raw) | ~465,000 | Baseline |
| After merge | ~465,000 | No change (order-item grain preserved) |
| After filtering | ~465,000 | No removal (flagging strategy) |

### 3.2 Column Count Growth
| Stage | Column Count | New Columns |
|-------|---|---|
| order_items (raw) | 10 | - |
| After merge | 28+ | +18 from orders, products, users |
| After engineering | 30+ | +is_returned_item, is_returned_order |
| After margins | 34+ | +discount_amount, discount_pct, item_margin, item_margin_pct |

### 3.3 Data Type Distribution
- **String:** 15+ (IDs, names, categories, status)
- **Numeric:** 12+ (prices, costs, margins, discounts)
- **Datetime:** 9 (timestamps for events)
- **Integer Flag:** 2 (return indicators)

---

## 4. Integration Points

| Downstream Module | Functions Used | Purpose |
|---|---|---|
| data_cleaning.py | load_processed_data() | Get data for validation |
| modeling.py | load_processed_data() | Input for analytical functions |
| visualization.py | load_processed_data() | Source for plots and charts |
| notebooks | build_analysis_dataset(), load_processed_data() | Exploratory analysis |

---

## 5. Error Handling & Edge Cases

### 5.1 Missing Files
**Scenario:** Raw CSV file doesn't exist
**Handling:** `FileNotFoundError` raised with clear path
**Solution:** Verify files in data/raw directory

### 5.2 Invalid Numeric Values
**Scenario:** Price column contains "N/A", "unknown", etc.
**Handling:** `pd.to_numeric(..., errors="coerce")` converts to NaN
**Impact:** Calculations with NaN propagate (e.g., NaN + 5 = NaN)

### 5.3 Mismatched Keys
**Scenario:** order_items references product_id not in products table
**Handling:** LEFT JOIN preserves order_items, products columns are NaN
**Impact:** Some rows have product metadata as NaN

### 5.4 Datetime Parsing Issues
**Scenario:** Timestamp in unexpected format
**Handling:** `pd.to_datetime(..., errors="coerce")` converts to NaT
**Resolution:** Investigate problematic rows separately

---

## 6. Performance Optimization

### 6.1 Selective Column Loading
```python
# Load only needed columns (faster, less memory)
oi, orders, products, users = load_raw_data(
    order_items_cols=["id", "order_id", "product_id", "sale_price", "cost", "status"]
)
```

### 6.2 Parquet vs CSV
- **Parquet:** 10-20x smaller file size, maintains types, faster loading
- **CSV:** Universal compatibility, human-readable

### 6.3 Compression
- Parquet uses Snappy compression by default
- Reduces storage from 500MB+ to ~50MB

---

## Summary

The `data_processing.py` module provides:
- ✅ End-to-end data pipeline execution
- ✅ Flexible column selection for efficiency
- ✅ Proper handling of categorical IDs and mixed-type data
- ✅ Return feature engineering for analysis
- ✅ Complete margin and discount calculations
- ✅ Type standardization for consistency
- ✅ Dual-format output (Parquet + CSV)
- ✅ Clear error messages for troubleshooting
