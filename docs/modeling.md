# Modeling Module - Technical Reference

## Executive Summary

The `modeling.py` module provides analytical functions for profit erosion modeling, customer segmentation, and return behavior analysis. These functions transform raw transaction data into actionable business insights about the economic impact of product returns.

---

## 1. Module Overview

### 1.1 Analysis Domains

The modeling module addresses three key business questions:

1. **Return Rate Analysis:** Where are customers returning products most frequently?
2. **Margin Erosion:** How much profit is lost due to returns?
3. **Customer Segmentation:** Which customer groups are most problematic?

### 1.2 Key Metrics

| Metric | Definition | Business Use |
|--------|------------|--------------|
| Return Rate | % of items returned | Identify high-risk products/regions |
| Margin Loss | Revenue not collected due to returns | Quantify financial impact |
| Customer Segments | Behavioral groupings | Targeted retention strategies |
| Process Costs | Cost to handle returns | Total profit erosion calculation |

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

**Calculation Logic:**

```
For each group:
  item_rows = count of all items in group
  returned_items = sum of is_returned_item flag (items with status='Returned')
  return_rate = returned_items / item_rows
  
Filter: Keep only groups with item_rows >= min_rows
Sort: Descending by return_rate (highest first)
```

**Output Structure:**

```
┌────────────┬───────────┬─────────────────┬─────────────┐
│ category   │ item_rows │ returned_items  │ return_rate │
├────────────┼───────────┼─────────────────┼─────────────┤
│ Jeans      │   45,000  │     7,200       │    0.160    │
│ Dresses    │   38,000  │     5,320       │    0.140    │
│ Tops       │   52,000  │     5,720       │    0.110    │
└────────────┴───────────┴─────────────────┴─────────────┘
```

**Business Insights:**
- Identifies which product categories/regions have highest return rates
- Enables targeted improvement efforts
- Reveals customer preference patterns

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

# Get top 10 highest-return categories
top_returns = ret_by_category.head(10)
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

**Parameters:**

| Parameter | Type | Purpose |
|-----------|------|---------|
| df | DataFrame | Input data with is_returned_item and item_margin columns |
| group_cols | list | Column names to group by (e.g., ["category"]) |

**Calculation Logic:**

```
For each group:
  Filter to returned items only (is_returned_item == 1)
  returned_items = count of returned items
  total_lost_sales = sum of sale_price for returns
  total_lost_margin = sum of item_margin for returns
  median_margin_per_return = median item_margin for returns
  avg_margin_per_return = mean item_margin for returns

Sort: Descending by total_lost_margin (biggest impact first)
```

**Output Structure:**

```
┌───────────────┬─────────────────┬──────────────────┬──────────────────────┬──────────────────┐
│ category      │ returned_items  │ total_lost_sales │ total_lost_margin    │ avg_margin...    │
├───────────────┼─────────────────┼──────────────────┼──────────────────────┼──────────────────┤
│ Jeans         │     7,200       │   $1,080,000     │   $324,000           │   $45.00         │
│ Dresses       │     5,320       │     $798,000     │   $239,400           │   $45.00         │
│ Tops & Tees   │     5,720       │     $572,000     │   $171,600           │   $30.00         │
└───────────────┴─────────────────┴──────────────────┴──────────────────────┴──────────────────┘
```

**Business Insights:**
- Jeans category loses most profit despite similar return rates to other categories
- May indicate higher unit cost/margin for jeans
- Drives allocation of quality improvement efforts

**Usage Examples:**

```python
# Margin loss by product category
margin_by_category = calculate_margin_loss_by_group(df, ["category"])

# Margin loss by geography
margin_by_country = calculate_margin_loss_by_group(df, ["country"])

# Multi-dimensional analysis
margin_by_region_category = calculate_margin_loss_by_group(
    df,
    ["country", "category"]
)

# Identify biggest loss drivers
top_5_losses = margin_by_category.nlargest(5, "total_lost_margin")
```

---

### 2.3 build_customer_behavior_profile()

**Purpose:** Create customer-level behavioral metrics for each user

**Function Signature:**
```python
def build_customer_behavior_profile(df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- df: DataFrame with order item grain (one row per item)

**Metrics Calculated:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| total_items | COUNT(order_item_id) | Number of items purchased |
| total_orders | NUNIQUE(order_id) | Number of distinct orders |
| return_events | SUM(is_returned_item) | Number of items returned |
| total_sales | SUM(sale_price) | Total revenue from customer |
| total_margin | SUM(item_margin) | Total profit from customer |
| avg_item_price | MEAN(sale_price) | Average item cost |
| avg_item_margin | MEAN(item_margin) | Average profit per item |
| avg_discount_pct | MEAN(discount_pct) | Average discount received |
| delivered_items | COUNT(item_delivered_at IS NOT NULL) | Items successfully delivered |
| **return_rate** | return_events / total_items | Percentage of items returned |
| **items_per_order** | total_items / total_orders | Items per purchase |

**Output Structure:**

```
┌──────────┬──────────────┬──────────────┬───────────────┬──────────────┬──────────────────┐
│ user_id  │ total_items  │ total_orders │ return_events │ total_margin │ return_rate      │
├──────────┼──────────────┼──────────────┼───────────────┼──────────────┼──────────────────┤
│ 1001     │     45       │      12      │      8        │   $1,200     │   0.1778 (17.8%) │
│ 1002     │     18       │       5      │      0        │   $450       │   0.0000 (0%)    │
│ 1003     │     62       │      18      │     12        │   $1,550     │   0.1935 (19.4%) │
└──────────┴──────────────┴──────────────┴───────────────┴──────────────┴──────────────────┘
```

**Business Use:**
- Identify high-value customers with low return rates (ideal customers)
- Identify low-value customers with high return rates (problematic)
- Understand customer purchase patterns
- Target retention or feedback campaigns

**Usage Example:**

```python
profile = build_customer_behavior_profile(df)

# Find high-value, low-return customers (VIP segment)
vip_customers = profile[
    (profile["total_margin"] > profile["total_margin"].quantile(0.75)) &
    (profile["return_rate"] < 0.05)
]

# Find problematic customers (high returns despite spending)
problem_customers = profile[
    (profile["total_sales"] > 1000) &
    (profile["return_rate"] > 0.25)
]
```

---

### 2.4 calculate_customer_margin_exposure()

**Purpose:** Quantify the profit impact of returns at customer level

**Function Signature:**
```python
def calculate_customer_margin_exposure(df: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- df: DataFrame with order item grain

**Metrics Calculated:**

| Metric | Logic | Interpretation |
|--------|-------|---|
| return_events | COUNT of returned items per customer | Number of returns |
| total_lost_margin | SUM of item_margin for returned items | Total profit lost |
| total_lost_sales | SUM of sale_price for returned items | Revenue at risk |
| median_margin_per_return | MEDIAN item_margin for returns | Typical return impact |
| max_single_return_margin | MAX item_margin for returns | Largest single loss |

**Output Structure:**

```
┌──────────┬────────────────┬──────────────────┬──────────────────┬─────────────────────┐
│ user_id  │ return_events  │ total_lost_margin │ total_lost_sales │ max_single_return.. │
├──────────┼────────────────┼──────────────────┼──────────────────┼─────────────────────┤
│ 2001     │      15        │   $450           │   $1,500         │   $50               │
│ 2002     │       8        │   $160           │   $640           │   $40               │
└──────────┴────────────────┴──────────────────┴──────────────────┴─────────────────────┘
```

**Business Use:**
- Identify customers with highest margin exposure (return risk)
- Quantify financial risk per customer relationship
- Prioritize customer service interventions

**Usage Example:**

```python
exposure = calculate_customer_margin_exposure(df)

# Top 10 customers by margin loss
biggest_risks = exposure.nlargest(10, "total_lost_margin")

# Customers losing more than $500 in margins
high_exposure = exposure[exposure["total_lost_margin"] > 500]
```

---

### 2.5 estimate_return_process_cost()

**Purpose:** Model operational costs of handling returns

**Function Signature:**
```python
def estimate_return_process_cost(
    df: pd.DataFrame,
    cost_per_return: float = 15.0,
    cost_components: Optional[Dict[str, float]] = None,
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Input data filtered to returned items |
| cost_per_return | float | $15.00 | Total operational cost per return |
| cost_components | dict | See below | Granular cost breakdown |

**Default Cost Components:**
```python
{
    "customer_care": $5.00,    # Customer service handling
    "inspection": $3.00,        # Product inspection/testing
    "restocking": $4.00,        # Shelving/inventory management
    "logistics": $3.00,         # Return shipping + handling
}
```

**Calculation Logic:**

```
For each returned item:
  process_cost = SUM of all cost_components
  total_profit_erosion = item_margin + process_cost
```

**Output:**
- Returns DataFrame with returned items only
- Adds columns: process_cost, cost_*, total_profit_erosion
- Allows analysis of true economic impact

**Example:**

```
Total Margin Loss:    $324,000
Process Costs:        $108,000  (7,200 returns × $15)
Total Erosion:        $432,000
```

**Usage Example:**

```python
# Default cost model ($15 per return)
erosion = estimate_return_process_cost(df)

# Custom cost model (higher logistics cost)
erosion = estimate_return_process_cost(
    df,
    cost_components={
        "customer_care": 4.0,
        "inspection": 2.5,
        "restocking": 3.5,
        "logistics": 8.0,  # International shipping
    }
)

# Analyze by product
erosion_by_category = erosion.groupby("category").agg({
    "total_profit_erosion": "sum",
    "item_margin": "sum",
    "process_cost": "sum"
})
```

---

### 2.6 summarize_profit_erosion()

**Purpose:** Generate executive summary of profit erosion metrics

**Function Signature:**
```python
def summarize_profit_erosion(
    df: pd.DataFrame,
    cost_per_return: float = 15.0,
) -> Dict[str, float]
```

**Return Dictionary Keys:**

| Key | Calculation | Interpretation |
|-----|---|---|
| total_items | COUNT of all items | Dataset size |
| total_returned | COUNT of returned items | Number of returns |
| return_rate_pct | (total_returned / total_items) × 100 | Return rate percentage |
| total_margin_reversal | SUM of item_margin for returned items | Lost revenue |
| avg_margin_per_return | MEAN of item_margin for returns | Average impact per return |
| median_margin_per_return | MEDIAN of item_margin for returns | Typical impact |
| estimated_process_costs | total_returned × cost_per_return | Operational burden |
| total_profit_erosion | margin_reversal + process_costs | Total financial impact |
| max_single_margin_loss | MAX item_margin for returns | Largest single loss |
| pct_margin_lost_to_returns | (margin_reversal / total_margin) × 100 | % of profit lost |

**Output Example:**

```python
{
    'total_items': 465000,
    'total_returned': 74400,
    'return_rate_pct': 16.0,
    'total_margin_reversal': 2232000.0,
    'avg_margin_per_return': 30.0,
    'median_margin_per_return': 25.0,
    'estimated_process_costs': 1116000.0,
    'total_profit_erosion': 3348000.0,
    'max_single_margin_loss': 189.99,
    'pct_margin_lost_to_returns': 18.5
}
```

**Business Interpretation:**
- 16% of items are returned
- Returns cost $3.3M in lost profit + process costs
- 18.5% of total profit is eroded by returns

**Usage Example:**

```python
summary = summarize_profit_erosion(df, cost_per_return=20.0)

print(f"Return Rate: {summary['return_rate_pct']:.1f}%")
print(f"Total Erosion: ${summary['total_profit_erosion']:,.0f}")
print(f"% of Profit Lost: {summary['pct_margin_lost_to_returns']:.1f}%")
```

---

### 2.7 segment_customers_by_return_behavior()

**Purpose:** Classify customers into risk segments based on return behavior

**Function Signature:**
```python
def segment_customers_by_return_behavior(
    df: pd.DataFrame,
    return_rate_thresholds: Tuple[float, float] = (0.05, 0.15),
) -> pd.DataFrame
```

**Parameters:**

| Parameter | Type | Default | Interpretation |
|-----------|------|---------|---|
| df | DataFrame | Required | Order item data |
| return_rate_thresholds | tuple | (0.05, 0.15) | (low_thresh, high_thresh) |

**Segmentation Logic:**

```
IF return_rate == 0:
    "no_returns" (ideal customers)
ELSE IF return_rate <= 0.05:
    "low_returner" (good customers)
ELSE IF return_rate <= 0.15:
    "moderate_returner" (manageable risk)
ELSE:
    "high_returner" (problematic customers)
```

**Output:**
- Returns customer profile with segment labels
- One row per customer
- Segment column added to dataframe

**Example Distribution:**

```
Return Segment Distribution:
  no_returns:        45,000 customers (34.6%)
  low_returner:      52,000 customers (40.0%)
  moderate_returner: 23,000 customers (17.7%)
  high_returner:     10,000 customers (7.7%)
```

**Business Strategies by Segment:**

| Segment | Characteristics | Strategy |
|---------|---|---|
| no_returns | 0% return rate, loyal | Reward loyalty, VIP benefits |
| low_returner | <5% return rate, good value | Maintain relationships, feedback |
| moderate_returner | 5-15% return rate, manageable | Monitor, offer better fit help |
| high_returner | >15% return rate, problematic | Intervention, assess relationship ROI |

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

# Target high returners for intervention
high_returners = segments[segments["return_segment"] == "high_returner"]
```

---

### 2.8 calculate_price_margin_returned_by_country()

**Purpose:** Geographic analysis of margin loss from returned items

**Function Signature:**
```python
def calculate_price_margin_returned_by_country(
    df: pd.DataFrame,
) -> pd.DataFrame
```

**Parameters:**
- df: Order item data with country and item_status columns

**Filtering:**
- Only includes items with `item_status == "Returned"`
- Returns empty DataFrame if no returned items

**Metrics Calculated:**

| Metric | Aggregation | Interpretation |
|--------|---|---|
| item_count | COUNT of returned items | Number of returns per country |
| avg_cost | MEAN of cost | Average COGS |
| total_cost | SUM of cost | Total COGS for returns |
| avg_sale_price | MEAN of sale_price | Average selling price |
| total_sale_price | SUM of sale_price | Total revenue lost |
| avg_margin | MEAN of item_margin | Average profit per return |
| total_margin | SUM of item_margin | Total margin lost |
| median_margin | MEDIAN of item_margin | Typical return impact |
| min_margin | MIN of item_margin | Best case return |
| max_margin | MAX of item_margin | Worst case return |

**Output Structure:**

```
┌──────────────┬─────────────┬───────────┬──────────────┬──────────────┐
│ country      │ item_count  │ avg_cost  │ total_margin │ max_margin   │
├──────────────┼─────────────┼───────────┼──────────────┼──────────────┤
│ United States│    45,200   │  $28.50   │  $1,356,000  │    $189.99   │
│ Canada       │     8,900   │  $27.25   │    $267,000  │    $185.00   │
│ Mexico       │     6,300   │  $29.10   │    $189,000  │    $180.00   │
└──────────────┴─────────────┴───────────┴──────────────┴──────────────┘
```

**Business Use:**
- Identify high-loss geographic markets
- Compare margin impact across countries
- Inform logistics and support resource allocation

**Usage Example:**

```python
margin_by_country = calculate_price_margin_returned_by_country(df)

# Top 5 countries by margin loss
top_losses = margin_by_country.head(5)

# Geographic comparison
us_margin_loss = margin_by_country.loc["United States", "total_margin"]
can_margin_loss = margin_by_country.loc["Canada", "total_margin"]
```

---

## 3. Workflow Integration

### 3.1 Typical Analysis Flow

```
Load Data → Build Profile → Calculate Returns → Segment Customers
                ↓                   ↓                  ↓
            Margin Loss    Process Costs         Target Groups
```

### 3.2 Example: Complete Return Analysis

```python
# Load data
df = load_processed_data()

# 1. Overall impact
summary = summarize_profit_erosion(df)

# 2. Category analysis
returns_by_category = calculate_return_rates_by_group(df, ["category"])
losses_by_category = calculate_margin_loss_by_group(df, ["category"])

# 3. Customer analysis
profile = build_customer_behavior_profile(df)
segments = segment_customers_by_return_behavior(df)
exposure = calculate_customer_margin_exposure(df)

# 4. Geographic analysis
geo_margin = calculate_price_margin_returned_by_country(df)

# 5. Cost modeling
with_costs = estimate_return_process_cost(df, cost_per_return=20.0)
```

---

## Summary

The `modeling.py` module provides:
- ✅ Return rate analysis by any dimension
- ✅ Profit erosion quantification
- ✅ Customer behavioral profiling
- ✅ Risk segmentation
- ✅ Geographic analysis
- ✅ Process cost estimation
- ✅ Executive summary metrics
- ✅ Flexible grouping and thresholds
