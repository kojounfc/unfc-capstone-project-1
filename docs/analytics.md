# Analytics Module - Technical Reference

## Executive Summary

The `analytics.py` module provides analytical functions for profit erosion analysis, customer segmentation, return behavior analysis, and feature quality validation. These functions transform transaction data into actionable business insights about the economic impact of product returns.

**Note:** This module was renamed from `modeling.py` to better reflect its purpose. ML models will be added in a separate module later.

---

## 1. Module Overview

### 1.1 Analysis Domains

The analytics module addresses five key business areas:

1. **Return Rate Analysis:** Calculate return rates by product categories, brands, and other dimensions
2. **Margin Erosion:** Quantify profit lost due to returns across different segments
3. **Customer Segmentation:** Classify customers by return behavior patterns
4. **Product-Level Features:** Engineer features at category and brand level for analysis
5. **Temporal Features:** Extract time-based patterns from order data
6. **Feature Quality Validation:** Ensure data quality for downstream modeling

### 1.2 Key Metrics

| Metric | Definition | Business Use |
|--------|------------|--------------|
| Return Rate | % of items returned | Identify high-risk products/regions |
| Margin Loss | Revenue not collected due to returns | Quantify financial impact |
| Customer Segments | Behavioral groupings | Targeted retention strategies |
| Category/Brand Return Rates | Aggregated return rates | Product risk profiling |

---

## 2. Core Functions

### 2.1 calculate_return_rates_by_group()

**Purpose:** Calculate return rate statistics grouped by one or more dimensions

**Function Signature:**
```python
def calculate_return_rates_by_group(
    df: pd.DataFrame,
    group_cols: list,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Input data with is_returned_item column |
| group_cols | list | Required | Column names to group by (e.g., ["category", "country"]) |
| min_rows | int | 200 | Minimum sample size for statistical reliability |

**Output Structure:**

| Column | Description |
|--------|-------------|
| item_rows | Count of all items in group |
| returned_items | Sum of is_returned_item flag |
| return_rate | returned_items / item_rows |

**Usage Examples:**

```python
# Return rate by product category
ret_by_category = calculate_return_rates_by_group(df, ["category"])

# Return rate by country and category (2-dimensional)
ret_by_region_category = calculate_return_rates_by_group(
    df,
    ["country", "category"]
)

# Custom minimum sample size (more conservative)
ret_filtered = calculate_return_rates_by_group(
    df,
    ["brand"],
    min_rows=500  # Only brands with 500+ items
)
```

---

### 2.2 calculate_margin_loss_by_group()

**Purpose:** Quantify total margin loss (profit erosion) grouped by dimension

**Function Signature:**
```python
def calculate_margin_loss_by_group(
    df: pd.DataFrame,
    group_cols: list,
) -> pd.DataFrame
```

**Output Structure:**

| Column | Description |
|--------|-------------|
| returned_items | Count of returned items |
| total_lost_sales | Sum of sale_price for returns |
| total_lost_margin | Sum of item_margin for returns |
| median_margin_per_return | Median item_margin for returns |
| avg_margin_per_return | Mean item_margin for returns |

**Usage Example:**

```python
# Margin loss by product category
margin_by_category = calculate_margin_loss_by_group(df, ["category"])

# Top 5 categories by margin loss
top_5_losses = margin_by_category.nlargest(5, "total_lost_margin")
```

---

### 2.3 calculate_customer_margin_exposure()

**Purpose:** Quantify the profit impact of returns at customer level

**Function Signature:**
```python
def calculate_customer_margin_exposure(df: pd.DataFrame) -> pd.DataFrame
```

**Output Structure:**

| Column | Description |
|--------|-------------|
| return_events | COUNT of returned items per customer |
| total_lost_margin | SUM of item_margin for returned items |
| total_lost_sales | SUM of sale_price for returned items |
| median_margin_per_return | MEDIAN item_margin for returns |
| max_single_return_margin | MAX item_margin for returns |

**Usage Example:**

```python
exposure = calculate_customer_margin_exposure(df)

# Top 10 customers by margin loss
biggest_risks = exposure.nlargest(10, "total_lost_margin")
```

---

### 2.4 segment_customers_by_return_behavior()

**Purpose:** Classify customers into risk segments based on return behavior

**Function Signature:**
```python
def segment_customers_by_return_behavior(
    df: pd.DataFrame,
    return_rate_thresholds: Tuple[float, float] = (0.05, 0.15),
) -> pd.DataFrame
```

**Segmentation Logic:**

| Segment | Condition | Characteristics |
|---------|-----------|-----------------|
| no_returns | return_rate == 0 | Ideal customers |
| low_returner | return_rate <= 0.05 | Good customers |
| moderate_returner | 0.05 < return_rate <= 0.15 | Manageable risk |
| high_returner | return_rate > 0.15 | Problematic customers |

**Usage Example:**

```python
# Default thresholds
segments = segment_customers_by_return_behavior(df)

# Custom thresholds (stricter)
segments = segment_customers_by_return_behavior(
    df,
    return_rate_thresholds=(0.02, 0.10)
)

# Analyze segment value
value_by_segment = segments.groupby("return_segment").agg({
    "total_margin": ["sum", "mean"],
    "total_items": "sum"
})
```

---

### 2.5 calculate_price_margin_returned_by_country()

**Purpose:** Geographic analysis of margin loss from returned items

**Function Signature:**
```python
def calculate_price_margin_returned_by_country(
    df: pd.DataFrame,
) -> pd.DataFrame
```

**Output Structure:**

| Column | Description |
|--------|-------------|
| item_count | COUNT of returned items per country |
| avg_cost | MEAN of cost |
| total_cost | SUM of cost |
| avg_sale_price | MEAN of sale_price |
| total_sale_price | SUM of sale_price |
| avg_margin | MEAN of item_margin |
| total_margin | SUM of item_margin |
| median_margin | MEDIAN of item_margin |
| min_margin | MIN of item_margin |
| max_margin | MAX of item_margin |

---

## 3. Product-Level Features (Task 3)

### 3.1 calculate_category_return_rates()

**Purpose:** Calculate return rates aggregated by product category

**Function Signature:**
```python
def calculate_category_return_rates(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame
```

**Usage Example:**

```python
category_rates = calculate_category_return_rates(df, min_rows=100)
high_risk_categories = category_rates[category_rates["return_rate"] > 0.10]
```

---

### 3.2 calculate_brand_return_rates()

**Purpose:** Calculate return rates aggregated by brand

**Function Signature:**
```python
def calculate_brand_return_rates(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame
```

---

### 3.3 engineer_product_level_features()

**Purpose:** Add product-level aggregated features to item-level data

**Function Signature:**
```python
def engineer_product_level_features(
    df: pd.DataFrame,
    min_rows: int = MIN_ROWS_THRESHOLD,
) -> pd.DataFrame
```

**Output Columns:**

| Column | Description |
|--------|-------------|
| category_return_rate | Return rate for item's category |
| brand_return_rate | Return rate for item's brand |
| price_tier | 'low', 'medium', or 'high' based on sale_price quantiles |

**Usage Example:**

```python
df = engineer_product_level_features(df)
high_risk_items = df[df["category_return_rate"] > 0.15]
```

---

## 3A. Descriptive & Group-Level Transformations (US07)

### Purpose

User Story 07 transforms **item-level engineered features** into **analysis-ready aggregated datasets** used for descriptive analysis and downstream modeling.  
These transformations intentionally separate **feature creation (US06)** from **analytical summarization (US07)** to ensure reproducibility, prevent data leakage, and maintain methodological clarity across research questions.

---

### Input Dependencies

US07 consumes the output of the following pipeline steps:

- `build_analysis_dataset()`
- `engineer_return_features()`
- `calculate_margins()`
- `calculate_profit_erosion()`

All transformations operate on a **fully denormalized, order-item–level dataset**.

---

### Task #57 – Product-Level Profit Erosion Metrics (RQ1)

**Function:** `build_product_profit_erosion_metrics()`

**Purpose:**  
Aggregate item-level profit erosion into interpretable **product-group summaries**.

**Aggregation Levels:**
- Category  
- Brand  
- Department  

**Core Metrics:**

| Metric | Description |
|------|-------------|
| item_rows | Total item count per group |
| returned_items | Number of returned items |
| return_rate | returned_items / item_rows |
| total_profit_erosion | Total margin loss due to returns |
| avg_profit_erosion | Mean erosion per item |
| median_profit_erosion | Median erosion per item |

**Analytical Value:**  
Supports **RQ1 (product-level profit erosion drivers)** by enabling hypothesis testing and descriptive comparisons across product dimensions without introducing customer-level bias.

---

### Task #58 – Product-Level Return Behavior Metrics

**Function:** `build_product_return_behavior_metrics()`

**Purpose:**  
Summarize **return behavior propensity** independently from financial impact.

**Outputs:**
- Category-level return rates
- Brand-level return rates
- Optional department-level return rates

**Design Notes:**
- Validates minimum sample sizes per group
- Ensures consistent denominators to avoid aggregation bias
- Explicitly separated from profit erosion calculations

**Analytical Value:**  
Provides behavioral context used for modeling joins and explanatory analysis.

---

### Task #59 – Customer-Level Profit Erosion Summaries (RQ2–RQ4)

**Function:** `build_customer_profit_erosion_summaries()`

**Purpose:**  
Aggregate profit erosion at the **customer level** to support segmentation and predictive modeling.

**Key Outputs:**

| Column | Description |
|------|-------------|
| return_rows | Number of returned items |
| total_profit_erosion | Total margin loss due to returns |
| avg_profit_erosion_per_return | Mean erosion per return event |

**Analytical Value:**  
Creates a stable customer-grain analytical table used for:
- Customer segmentation (RQ2)
- Binary classification (RQ3)
- Regression modeling (RQ4)

---

### Task #60 – Modeling Dataset Assembly

**Function:** `build_product_modeling_dataset()`

**Purpose:**  
Prepare **model-ready analytical datasets** by joining:
- Product-level profit erosion summaries (Task #57)
- Product-level return behavior metrics (Task #58)

**Design Decisions:**
- Explicit aggregation level control (`by_category`, `by_brand`, `by_department`)
- Standardized `return_rate` column for modeling consistency
- Defensive schema validation prior to joins
- No file outputs or side effects

**Output:**  
A clean, single-table dataset suitable for regression analysis, feature importance evaluation, and predictive modeling workflows.

---

### Reproducibility & Validation

All US07 transformations:
- Are deterministic
- Are validated using `pytest`
- Enforce schema checks before aggregation and joins
- Do not write files during execution

---

### Relationship to Other Tasks

- **US06:** Feature engineering (input layer)
- **US07:** Descriptive aggregation and modeling preparation
- **US08+:** Visualization, modeling, and inference

---

## 4. Temporal Features (Task 4)

### 4.1 engineer_temporal_features()

**Purpose:** Create time-based features for analysis

**Function Signature:**
```python
def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame
```

**Output Columns:**

| Column | Description |
|--------|-------------|
| order_day_of_week | 0-6 (Monday=0) |
| order_month | 1-12 |
| order_quarter | 1-4 |
| order_year | Year |
| is_weekend_order | Boolean |
| season | 'winter', 'spring', 'summer', 'fall' |
| days_to_delivery | Days from order creation to delivery |
| days_to_return | Days from delivery to return (NaN if not returned) |

**Usage Example:**

```python
df = engineer_temporal_features(df)
weekend_returns = df[df["is_weekend_order"] == True]
seasonal_analysis = df.groupby("season")["is_returned_item"].mean()
```

---

## 5. Feature Quality Validation (Task 6)

### 5.1 validate_feature_quality()

**Purpose:** Validate engineered features for quality issues

**Function Signature:**
```python
def validate_feature_quality(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    vif_threshold: float = 5.0,
) -> Dict[str, Any]
```

**Return Dictionary:**

| Key | Description |
|-----|-------------|
| total_rows | Number of rows in dataset |
| total_features | Number of features analyzed |
| missing_values | {col: count} for columns with missing values |
| missing_pct | {col: pct} percentage missing |
| distribution_stats | {col: {mean, std, min, max, skew, kurtosis}} |
| correlation_matrix | Correlation matrix for numeric features |
| high_correlations | List of column pairs with \|corr\| > 0.8 |
| constant_columns | Columns with zero variance |
| low_variance_columns | Columns with variance < 0.01 |

**Usage Example:**

```python
report = validate_feature_quality(df)
print(f"Missing values: {report['missing_values']}")
print(f"High correlations: {report['high_correlations']}")
```

---

### 5.2 generate_feature_quality_report()

**Purpose:** Generate a human-readable feature quality report

**Function Signature:**
```python
def generate_feature_quality_report(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> str
```

**Usage Example:**

```python
# Print report
report = generate_feature_quality_report(df)
print(report)

# Save to file
generate_feature_quality_report(df, output_path="reports/quality_report.txt")
```

---

## 6. Legacy Functions

The following functions are maintained for backward compatibility but may be superseded by functions in `feature_engineering.py`:

### 6.1 estimate_return_process_cost()

**Note:** For profit erosion calculations, prefer using `calculate_profit_erosion()` from `feature_engineering.py` which implements category-tiered processing costs.

**Function Signature:**
```python
def estimate_return_process_cost(
    df: pd.DataFrame,
    cost_per_return: float = 15.0,
    cost_components: Optional[Dict[str, float]] = None,
) -> pd.DataFrame
```

### 6.2 summarize_profit_erosion()

**Note:** A more comprehensive version exists in `feature_engineering.py`.

---

## 7. Workflow Integration

### 7.1 Typical Analysis Flow

```
Load Data → Engineer Features → Calculate Returns → Segment Customers
                ↓                     ↓                  ↓
            Add Product       Margin Loss          Target Groups
            Features             Analysis
                ↓
           Validate
           Features
```

### 7.2 Example: Complete Return Analysis Pipeline

```python
from src.data_processing import build_analysis_dataset
from src.feature_engineering import engineer_return_features, calculate_margins
from src.analytics import (
    calculate_return_rates_by_group,
    calculate_margin_loss_by_group,
    engineer_product_level_features,
    engineer_temporal_features,
    segment_customers_by_return_behavior,
    validate_feature_quality,
)

# Load and prepare data
df = build_analysis_dataset()
df = engineer_return_features(df)
df = calculate_margins(df)

# Add product and temporal features
df = engineer_product_level_features(df)
df = engineer_temporal_features(df)

# Analysis
returns_by_category = calculate_return_rates_by_group(df, ["category"])
losses_by_category = calculate_margin_loss_by_group(df, ["category"])
customer_segments = segment_customers_by_return_behavior(df)

# Validate features
quality_report = validate_feature_quality(df)
print(f"Features validated: {quality_report['total_features']}")
```

---

## 8. Module Dependencies

| Dependency | Usage |
|------------|-------|
| pandas | DataFrame operations |
| numpy | Numerical operations |
| src.config | MIN_ROWS_THRESHOLD constant |
| src.feature_engineering | engineer_customer_behavioral_features |

---

## Summary

The `analytics.py` module provides:
- ✅ Return rate analysis by any dimension
- ✅ Profit erosion quantification
- ✅ Customer behavioral segmentation
- ✅ Product-level feature engineering
- ✅ Temporal feature engineering
- ✅ Geographic analysis
- ✅ Feature quality validation
- ✅ Flexible grouping and thresholds
