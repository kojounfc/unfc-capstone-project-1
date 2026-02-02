# Configuration Module - Technical Reference

## Executive Summary

The `config.py` module centralizes all configuration constants and directory paths for the Profit Erosion E-commerce Capstone Project. This module ensures consistent path resolution across development environments and provides a single source of truth for file locations and data parameters.

---

## 1. Directory Structure

### 1.1 Project Root
```python
PROJECT_ROOT = Path(__file__).parent.parent
```
- **Purpose:** Dynamically identifies the project root directory
- **Usage:** Base reference for all relative paths
- **Benefit:** Environment-agnostic path resolution

### 1.2 Data Directories

#### Raw Data Directory
```python
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
```
- **Contents:** Original CSV files from TheLook e-commerce dataset
- **Expected Files:** `order_items.csv`, `orders.csv`, `products.csv`, `users.csv`

#### Processed Data Directory
```python
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
```
- **Contents:** Cleaned and processed datasets after data cleaning pipeline
- **Output Format:** Both CSV and Parquet for compatibility

#### Figures Directory
```python
FIGURES_DIR = PROJECT_ROOT / "figures"
```
- **Purpose:** Storage location for generated visualizations
- **Output Format:** PNG with 150 DPI resolution

#### Reports Directory
```python
REPORTS_DIR = PROJECT_ROOT / "reports"
```
- **Purpose:** Location for analysis reports and findings
- **Output Format:** Markdown and HTML documents

---

## 2. File Mappings

### 2.1 Raw Data Files Dictionary
```python
RAW_FILES = {
    "order_items": RAW_DATA_DIR / "order_items.csv",
    "orders": RAW_DATA_DIR / "orders.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "users": RAW_DATA_DIR / "users.csv",
}
```

**File Descriptions:**

| File | Records | Purpose | Key Columns |
|------|---------|---------|-------------|
| order_items.csv | ~465k | Individual line items per order | id, order_id, product_id, status, sale_price, cost |
| orders.csv | ~95k | Order-level aggregations | order_id, user_id, status, num_of_item |
| products.csv | ~7k | Product catalog | id, name, category, brand, cost, retail_price |
| users.csv | ~130k | Customer master data | id, first_name, country, city, gender, created_at |

### 2.2 Processed Data Files
```python
PROCESSED_PARQUET = PROCESSED_DATA_DIR / "returns_eda_v1.parquet"
PROCESSED_CSV = PROCESSED_DATA_DIR / "returns_eda_v1.csv"
```

- **Format:** Dual format storage for flexibility
  - **Parquet:** Optimal for analysis (compression, type preservation)
  - **CSV:** Universal compatibility (Excel, Python, R)
- **Content:** Merged, cleaned, and feature-engineered dataset at order-item grain
- **Grain:** One row per order item (465k+ rows)

---

## 3. Column Type Definitions

### 3.1 DateTime Columns
```python
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
```

**Purpose:** Configures pandas datetime parsing during data loading
**Usage:** Ensures timestamps are parsed as datetime objects, not strings
**Count:** 9 datetime columns across the merged dataset

### 3.2 Numeric Columns
```python
NUMERIC_COLS = [
    "sale_price",
    "retail_price", 
    "cost",
    "age",
    "num_of_item"
]
```

**Purpose:** Identifies columns requiring numeric type handling
**Usage:** Reference for feature engineering and statistical analysis
**Note:** Other numeric columns exist (order_id, user_id) but are categorized as IDs, not metrics

### 3.3 String Columns
```python
STRING_COLS = [
    "postal_code",
    "sku",
    "user_geom",
    "street_address",
    "email"
]
```

**Purpose:** Identifies columns with mixed-type characteristics
**Challenge:** These columns may contain numeric or string data
**Handling:** Special parsing logic required to preserve data integrity
**Note:** `postal_code` is treated as string to preserve leading zeros (e.g., "01234")

---

## 4. Analysis Parameters

### 4.1 Minimum Rows Threshold
```python
MIN_ROWS_THRESHOLD = 200
```

**Purpose:** Ensures statistical reliability in aggregated analyses
**Usage:** Applied in return rate analysis and segmentation functions
**Rationale:** Groups with < 200 records are excluded from analysis due to:
- Statistical instability at small sample sizes
- Potential for misleading conclusions
- Recommendation from data quality standards (minimum 20-30x effect size)

**Application Examples:**
- Return rate calculations by category require ≥ 200 items per category
- Customer segmentation analysis filters to customers with ≥ 200 purchases
- Geographic breakdowns only show regions with ≥ 200 transaction records

---

## 5. Usage Examples

### 5.1 Loading Raw Data
```python
from src.config import RAW_FILES
import pandas as pd

# Access specific file path
order_items_path = RAW_FILES["order_items"]
df = pd.read_csv(order_items_path)
```

### 5.2 Saving Processed Results
```python
from src.config import PROCESSED_PARQUET, PROCESSED_CSV

# Save to both formats
df.to_parquet(PROCESSED_PARQUET, compression="snappy")
df.to_csv(PROCESSED_CSV, index=False)
```

### 5.3 Saving Visualizations
```python
from src.config import FIGURES_DIR

# Create figure and save
fig.savefig(
    FIGURES_DIR / "return_rate_by_category.png",
    dpi=150,
    bbox_inches="tight"
)
```

### 5.4 Configuration in Functions
```python
from src.config import MIN_ROWS_THRESHOLD

def analyze_returns(df):
    # Automatically uses configured minimum threshold
    filtered = df.groupby("category").filter(
        lambda x: len(x) >= MIN_ROWS_THRESHOLD
    )
    return filtered
```

---

## 6. Configuration Best Practices

### 6.1 Environment Independence
- Uses `Path` objects for cross-platform compatibility (Windows, macOS, Linux)
- Relative paths ensure code works regardless of installation location
- No hardcoded absolute paths

### 6.2 Centralized Updates
- Changing `MIN_ROWS_THRESHOLD` updates all analyses using it
- Modifying data directory structure only requires config.py edit
- File naming conventions change in one location

### 6.3 Type Safety
- All paths are `Path` objects with full pathlib functionality
- Type hints support IDE autocompletion and type checking
- Constants are uppercase for easy identification

---

## 7. Integration with Other Modules

| Module | Usage | References |
|--------|-------|-----------|
| data_processing.py | Directory and column definitions | RAW_DATA_DIR, DATETIME_COLS, NUMERIC_COLS |
| data_cleaning.py | Data file locations | PROCESSED_DATA_DIR |
| feature_engineering.py | Output directory, cost constants | PROCESSED_DATA_DIR |
| analytics.py | Statistical threshold | MIN_ROWS_THRESHOLD |
| visualization.py | Output directory | FIGURES_DIR |

---

## 8. Troubleshooting

### Issue: "No such file or directory" errors
**Solution:** Verify data files exist in RAW_DATA_DIR structure
```bash
ls -la data/raw/  # Check file existence
```

### Issue: Import errors for config module
**Solution:** Ensure src directory is in Python path
```python
import sys
sys.path.insert(0, '/path/to/project')
from src.config import RAW_DATA_DIR
```

### Issue: Inconsistent paths across environments
**Solution:** Use Path objects consistently throughout codebase
```python
# Correct
from pathlib import Path
file_path = PROCESSED_DATA_DIR / "file.csv"

# Avoid
file_path = "data/processed/file.csv"  # Platform-dependent
```

---

## Summary

The `config.py` module provides:
- ✅ Centralized path management
- ✅ Environment-independent configuration
- ✅ Single source of truth for file locations
- ✅ Consistent data type handling across the project
- ✅ Easy updates for parameter changes (threshold, file locations)
