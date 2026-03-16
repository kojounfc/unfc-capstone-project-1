# Configuration Module - Technical Reference (`src/config.py`)

_Last updated: 2026-02-23_

## Executive Summary

`config.py` is the single source of truth for **directory paths**, **input/output file locations**, and **shared analysis parameters** used across the Profit Erosion E-commerce Capstone Project.

It ensures:
- Consistent path resolution on any machine (Windows/macOS/Linux)
- One place to change thresholds and modeling parameters
- Stable references for notebooks and Python modules

---

## 1. Project Root and Directory Paths

### 1.1 Project Root

```python
PROJECT_ROOT = Path(__file__).parent.parent
```

All other directories are defined relative to `PROJECT_ROOT`.

### 1.2 Core Directories

```python
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
```

### 1.3 RQ-Specific Directories (currently defined)

```python
RQ1_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq1"
RQ1_FIGURES_DIR = FIGURES_DIR / "rq1"
RQ4_FIGURES_DIR = FIGURES_DIR / "rq4"
```

---

## 2. File Mappings

### 2.1 Raw file map

`RAW_FILES` provides named access to the raw CSVs:

```python
RAW_FILES = {
    "order_items": RAW_DATA_DIR / "order_items.csv",
    "orders": RAW_DATA_DIR / "orders.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "users": RAW_DATA_DIR / "users.csv",
}
```

### 2.2 Processed dataset outputs

```python
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "returns_eda_v1.parquet"
PROCESSED_CSV = PROCESSED_DATA_DIR / "returns_eda_v1.csv"
RQ1_PROCESSED_PARQUET = RQ1_PROCESSED_DIR / "returns_eda_v1.parquet"
RQ1_PROCESSED_CSV = RQ1_PROCESSED_DIR / "returns_eda_v1.csv"
```

---

## 3. Column Type Configuration

These lists are used to standardize parsing and type casting during data loading and cleaning.

### 3.1 Date/time columns

```python
DATETIME_COLS = [
    "item_created_at", "item_shipped_at", "item_delivered_at", "item_returned_at",
    "order_created_at", "order_shipped_at", "order_delivered_at", "order_returned_at",
    "user_created_at",
]
```

### 3.2 Numeric columns

```python
NUMERIC_COLS = ["sale_price", "retail_price", "cost", "age", "num_of_item"]
```

### 3.3 String columns (mixed-type handling)

```python
STRING_COLS = ["postal_code", "sku", "user_geom", "street_address", "email"]
```

---

## 4. Shared Analysis Parameters

### 4.1 Minimum aggregation threshold

```python
MIN_ROWS_THRESHOLD = 200
```

Used in grouped summaries and visualizations to avoid unstable rates or misleading small-sample results.

---

## 5. RQ3: Predictive Modeling Parameters

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
AUC_THRESHOLD = 0.70
RQ3_TARGET = "is_high_erosion_customer"
```

### 5.1 RQ3 data inputs / outputs

```python
CUSTOMER_TARGETS_CSV = PROCESSED_DATA_DIR / "customer_profit_erosion_targets.csv"
SSL_RETURNS_CSV = RAW_DATA_DIR / "SSL_Returns_df_yoy.csv"
```

### 5.2 Candidate features and leakage controls

`RQ3_CANDIDATE_FEATURES` defines the baseline features considered for modeling.

`RQ3_LEAKAGE_COLUMNS` lists columns that must be excluded to prevent training on target leakage (including IDs and any direct erosion totals).

---

## 6. RQ3 Sensitivity Analysis

```python
SENSITIVITY_BASE_COSTS = [8.0, 10.0, 12.0, 14.0, 18.0]
SENSITIVITY_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
```

Used to test how conclusions change under different operational cost assumptions and “high-erosion” threshold definitions.

---

## 7. RQ4: Behavioral Econometrics Parameters

```python
RQ4_TARGET_COL = "total_profit_erosion"
RQ4_BEHAVIORAL_CONTROLS = ["order_frequency", "avg_order_value", "customer_tenure_days", "customer_return_rate"]
RQ4_HYPOTHESIS_PREDICTORS = ["return_frequency", "avg_basket_size", "purchase_recency_days"]
RQ4_ALPHA = 0.05
RQ4_COLLINEARITY_THRESHOLD = 0.85
```

These constants support regression specification and common validation rules (significance level and multicollinearity threshold).

---

## Appendix: Current Constant Assignments (from `config.py`)

Below is a direct list of top-level constant assignments currently present in `config.py`:

```text
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
RQ1_PROCESSED_DIR = PROCESSED_DATA_DIR / "rq1"
RQ1_FIGURES_DIR = FIGURES_DIR / "rq1"
RQ4_FIGURES_DIR = FIGURES_DIR / "rq4"
RAW_FILES = {
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "returns_eda_v1.parquet"
PROCESSED_CSV = PROCESSED_DATA_DIR / "returns_eda_v1.csv"
RQ1_PROCESSED_PARQUET = RQ1_PROCESSED_DIR / "returns_eda_v1.parquet"
RQ1_PROCESSED_CSV = RQ1_PROCESSED_DIR / "returns_eda_v1.csv"
DATETIME_COLS = [
NUMERIC_COLS = ["sale_price", "retail_price", "cost", "age", "num_of_item"]
STRING_COLS = ["postal_code", "sku", "user_geom", "street_address", "email"]
MIN_ROWS_THRESHOLD = 200
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
AUC_THRESHOLD = 0.70
CUSTOMER_TARGETS_CSV = PROCESSED_DATA_DIR / "customer_profit_erosion_targets.csv"
SSL_RETURNS_CSV = RAW_DATA_DIR / "SSL_Returns_df_yoy.csv"
RQ3_TARGET = "is_high_erosion_customer"
RQ3_CANDIDATE_FEATURES = [
RQ3_LEAKAGE_COLUMNS = [
SENSITIVITY_BASE_COSTS = [8.0, 10.0, 12.0, 14.0, 18.0]
SENSITIVITY_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
RQ4_TARGET_COL = "total_profit_erosion"
RQ4_BEHAVIORAL_CONTROLS = ['order_frequency', 'avg_order_value', 'customer_tenure_days', 'customer_return_rate']
RQ4_HYPOTHESIS_PREDICTORS = ['return_frequency', 'avg_basket_size', 'purchase_recency_days']
RQ4_ALPHA = 0.05
RQ4_COLLINEARITY_THRESHOLD = 0.85
```
