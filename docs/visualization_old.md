# Visualization Module - Technical Reference

## Executive Summary

The `visualization.py` module provides plotting functions for exploratory data analysis (EDA) and analytical reporting. These functions generate publication-quality charts that communicate key findings about return rates, margin loss, and customer behavior.

---

## 1. Module Overview

### 1.1 Plot Categories

| Category | Purpose | Functions |
|----------|---------|-----------|
| **Status Distribution** | Overview of data composition | `plot_status_distribution()` |
| **Return Analysis** | Return rates and patterns | `plot_return_rate_by_category()`, `plot_return_rate_heatmap()` |
| **Margin Analysis** | Loss visualization and exposure | `plot_margin_distribution()`, `plot_margin_loss_by_category()`, `plot_price_margin_returned_by_status_country()` |
| **Customer Analysis** | Risk identification by customer | `plot_customer_margin_exposure()` |

### 1.2 Design Principles

- **Consistent Style:** All plots use `set_plot_style()` for uniform appearance
- **Publication Quality:** 150 DPI, tight layouts, labeled values
- **File Saving:** Optional `save_path` parameter for saving PNG files
- **Flexibility:** Configurable dimensions, thresholds, and grouping

---

## 2. Core Functions

### 2.1 set_plot_style()

**Purpose:** Configure matplotlib/seaborn styling for all visualizations

**Function Signature:**
```python
def set_plot_style() -> None
```

**Configuration Applied:**
```python
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
```

**Usage:**
```python
# Call once at start of visualization script
set_plot_style()

# All subsequent plots will use this style
plot_status_distribution(df)
plot_return_rate_by_category(df)
```

---

### 2.2 plot_status_distribution()

**Purpose:** Visualize the distribution of order item statuses

**Function Signature:**
```python
def plot_status_distribution(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with item_status column |
| figsize | tuple | (10, 6) | Figure dimensions (width, height) |
| save_path | str | None | Path to save PNG (e.g., "figures/status.png") |

**Output:**
- Bar chart with status categories on x-axis
- Count of items for each status on y-axis
- Value labels displayed on top of bars

**Example Output:**
```
        Completed   Cancelled   Returned   Shipped
Count   390,600      10,000      74,400      [...]
```

**Usage Example:**

```python
# Basic plot
fig = plot_status_distribution(df)
plt.show()

# Save to file
fig = plot_status_distribution(
    df,
    save_path="reports/figures/status_dist.png"
)

# Custom dimensions
fig = plot_status_distribution(
    df,
    figsize=(14, 8),
    save_path="status_report.png"
)
```

---

### 2.3 plot_return_rate_by_category()

**Purpose:** Compare return rates across product categories

**Function Signature:**
```python
def plot_return_rate_by_category(
    df: pd.DataFrame,
    top_n: int = 15,
    min_rows: int = MIN_ROWS_THRESHOLD,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with category and is_returned_item |
| top_n | int | 15 | Number of top categories to show |
| min_rows | int | 200 | Minimum items for category to include |
| figsize | tuple | (12, 8) | Figure dimensions |
| save_path | str | None | Path to save PNG |

**Filtering Logic:**
1. Groups by category
2. Calculates: total items, returned items, return rate
3. Filters to categories with ≥ min_rows items
4. Sorts descending by return rate
5. Takes top_n categories

**Output:**
- Horizontal bar chart (easier category label reading)
- Categories on y-axis
- Return rate (%) on x-axis
- Percentage labels on bars

**Example Output:**
```
Jeans           ████████████████ 16.0%
Dresses         ███████████████  14.0%
Shorts          █████████████    13.0%
```

**Business Use:**
- Identify which product categories have highest return rates
- Compare performance across product lines
- Prioritize quality or fit improvement efforts

**Usage Example:**

```python
# Top 15 categories (default)
fig = plot_return_rate_by_category(df)

# Top 10 with strict sample size
fig = plot_return_rate_by_category(
    df,
    top_n=10,
    min_rows=500
)

# All categories with minimum threshold
fig = plot_return_rate_by_category(
    df,
    top_n=50,
    min_rows=100,
    save_path="figures/return_by_category.png"
)
```

---

### 2.4 plot_margin_distribution()

**Purpose:** Visualize the distribution of profit margins per item

**Function Signature:**
```python
def plot_margin_distribution(
    df: pd.DataFrame,
    returned_only: bool = False,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with item_margin column |
| returned_only | bool | False | If True, show only returned items |
| figsize | tuple | (12, 5) | Figure dimensions |
| save_path | str | None | Path to save PNG |

**Output:** Two-subplot figure
1. **Left:** Histogram of margin values with median line
2. **Right:** Box plot showing distribution quartiles

**Histogram Details:**
- 50 bins for margin values
- Red dashed line marks median
- Shows central tendency and spread

**Box Plot Details:**
- Shows Q1, median, Q3, whiskers, outliers
- Single vertical box for easy comparison

**Usage Example:**

```python
# All items
fig = plot_margin_distribution(df)

# Only returned items (to see if returns skew low/high margin)
fig = plot_margin_distribution(
    df,
    returned_only=True,
    save_path="figures/margin_returned.png"
)

# Comparison visualization
fig1 = plot_margin_distribution(df, returned_only=False)
fig2 = plot_margin_distribution(df, returned_only=True)
# Compare side-by-side
```

---

### 2.5 plot_margin_loss_by_category()

**Purpose:** Show which product categories lose most profit from returns

**Function Signature:**
```python
def plot_margin_loss_by_category(
    df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with category, is_returned_item, item_margin |
| top_n | int | 15 | Number of top categories |
| figsize | tuple | (12, 8) | Figure dimensions |
| save_path | str | None | Path to save PNG |

**Filtering & Aggregation:**
1. Filters to returned items only (is_returned_item == 1)
2. Groups by category
3. Sums total margin lost per category
4. Sorts descending by total loss
5. Takes top_n categories

**Output:**
- Horizontal bar chart
- Categories on y-axis
- Total margin loss ($) on x-axis
- Dollar labels on bars

**Example Output:**
```
Jeans       ████████████████ $324,000
Dresses     █████████████    $239,400
Tops & Tees ████████████     $171,600
```

**Business Interpretation:**
- Dollar impact (not just rate) of returns
- Combines return frequency AND margin per item
- Jeans: high return rate + high unit margin = huge impact

**Usage Example:**

```python
# Top 15 categories by loss (default)
fig = plot_margin_loss_by_category(df)

# Top 10 largest losses
fig = plot_margin_loss_by_category(
    df,
    top_n=10,
    save_path="figures/margin_loss.png"
)

# All categories (can be many)
fig = plot_margin_loss_by_category(
    df,
    top_n=50,
    figsize=(12, 20)  # Taller figure
)
```

---

### 2.6 plot_customer_margin_exposure()

**Purpose:** Identify customers who represent highest profit risk from returns

**Function Signature:**
```python
def plot_customer_margin_exposure(
    df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with user_id, is_returned_item, item_margin |
| top_n | int | 20 | Number of top customers to show |
| figsize | tuple | (12, 8) | Figure dimensions |
| save_path | str | None | Path to save PNG |

**Filtering & Aggregation:**
1. Filters to returned items only
2. Groups by user_id
3. Counts return events per customer
4. Sums total lost margin per customer
5. Sorts descending by total loss
6. Takes top_n customers

**Output:**
- Vertical bar chart
- Customer User IDs on x-axis
- Total lost margin ($) on y-axis
- 45° rotated x-axis labels for readability

**Example Output:**
```
User 2001: $450 (15 returns)
User 2005: $380 (12 returns)
User 1998: $340 (10 returns)
```

**Business Use:**
- Identify customers with highest return risk
- Target for quality feedback or customer service
- Assess customer lifetime value vs. return cost

**Usage Example:**

```python
# Top 20 customers at risk (default)
fig = plot_customer_margin_exposure(df)

# Top 50 customers
fig = plot_customer_margin_exposure(
    df,
    top_n=50,
    figsize=(16, 10),
    save_path="figures/customer_exposure.png"
)

# Smaller set for focused intervention
fig = plot_customer_margin_exposure(
    df,
    top_n=10,
    figsize=(10, 6)
)
```

---

### 2.7 plot_return_rate_heatmap()

**Purpose:** Visualize return rates across two dimensions simultaneously

**Function Signature:**
```python
def plot_return_rate_heatmap(
    df: pd.DataFrame,
    row_col: str = "category",
    col_col: str = "traffic_source",
    min_rows: int = 100,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure
```

**Parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| df | DataFrame | Required | Data with is_returned_item column |
| row_col | str | "category" | Column for heatmap rows |
| col_col | str | "traffic_source" | Column for heatmap columns |
| min_rows | int | 100 | Minimum sample size per cell |
| figsize | tuple | (14, 10) | Figure dimensions |
| save_path | str | None | Path to save PNG |

**Filtering & Aggregation:**
1. Groups by row_col and col_col
2. Calculates: total items, returned items, return rate
3. Filters cells with < min_rows to suppress noise
4. Creates pivot table
5. Converts to percentage format

**Output:**
- Color gradient heatmap (yellow → orange → red)
- Darker red = higher return rate
- Annotated with percentage values
- Rows: chosen dimension (e.g., categories)
- Columns: chosen dimension (e.g., traffic sources)

**Example Output:**
```
              Direct   Organic   Paid Ads   Social
Jeans         12.0%    14.5%     18.2%      16.1%
Dresses       10.0%    11.8%     15.3%      14.2%
Shirts         9.5%    10.2%     12.8%      11.5%
```

**Business Insights:**
- Which category × source combination has highest returns
- Are returns driven by traffic source or category
- Regional patterns (if using country × category)

**Usage Example:**

```python
# Default: category vs traffic_source
fig = plot_return_rate_heatmap(df)

# Category vs country
fig = plot_return_rate_heatmap(
    df,
    row_col="category",
    col_col="country",
    min_rows=200
)

# Category vs discount level
fig = plot_return_rate_heatmap(
    df,
    row_col="category",
    col_col="discount_range",
    min_rows=50,
    save_path="figures/return_heatmap.png"
)

# Very detailed: brand vs region
fig = plot_return_rate_heatmap(
    df,
    row_col="brand",
    col_col="state",
    min_rows=30,
    figsize=(20, 16)  # Large for many dimensions
)
```

---

### 2.8 plot_price_margin_returned_by_status_country()

**Purpose:** Create comprehensive grid visualization of cost, price, margin, and volume metrics for returned items by country

**Function Signature:**
```python
def plot_price_margin_returned_by_status_country(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None
```

**Parameters:**

| Parameter | Type | Purpose |
|-----------|------|---------|
| df | DataFrame | Aggregated metrics for returned items by country (output of `calculate_price_margin_returned_by_country()`) |
| save_path | str | Optional path to save PNG grid |

**⚠️ Important Note:**
- Input DataFrame should contain **only returned items** (pre-filtered)
- Expected to be output of `modeling.calculate_price_margin_returned_by_country()`
- No additional filtering is applied in this function

**Grid Layout:** 4 rows × 2 columns = 7 charts

| Position | Metric | Y-Axis Label |
|----------|--------|---|
| (1,1) | avg_cost | Average Cost ($) |
| (1,2) | total_cost | Total Cost ($) |
| (2,1) | avg_sale_price | Average Sale Price ($) |
| (2,2) | total_sale_price | Total Sale Price ($) |
| (3,1) | avg_margin | Average Margin ($) |
| (3,2) | total_margin | Total Margin ($) |
| (4,1) | item_count | Item Count (Volume) |
| (4,2) | (hidden) | N/A |

**Output:**
- 4×2 subplot grid
- Bar charts for each metric
- X-axis: Country names (rotated 45°)
- Title: "Cost, Price, Margin, and Volume Analysis for RETURNED Items by Country"
- 16×14 figure size for readability

**Business Insights:**
- Which countries have highest return volumes
- Average margin impact by country
- Cost structure differences across regions
- Total financial impact by geography

**Usage Example:**

```python
# Get returned items by country
returned_by_country = calculate_price_margin_returned_by_country(df)

# Create grid visualization
plot_price_margin_returned_by_status_country(returned_by_country)

# Save to file
plot_price_margin_returned_by_status_country(
    returned_by_country,
    save_path="reports/figures/returned_by_country"
)
```

---

## 3. Utility Functions

### 3.1 _safe_tight_layout()

**Purpose:** Apply matplotlib tight_layout with warning suppression

**Function Signature:**
```python
def _safe_tight_layout() -> None
```

**Why Needed:**
- Tight layout prevents label cutoff
- Warnings can occur on small figures
- Wrapped in warning filter for clean output

**Usage:**
- Called automatically within all plotting functions
- No need to call directly

---

## 4. Workflow Integration

### 4.1 Complete EDA Visualization Pipeline

```python
from src.visualization import set_plot_style, plot_*

# Initialize style
set_plot_style()

# Overview
fig1 = plot_status_distribution(df, save_path="figures/01_status.png")

# Return analysis
fig2 = plot_return_rate_by_category(df, save_path="figures/02_return_rate.png")
fig3 = plot_return_rate_heatmap(df, row_col="category", col_col="country", 
                                save_path="figures/03_heatmap.png")

# Margin analysis
fig4 = plot_margin_distribution(df, save_path="figures/04_margin_all.png")
fig5 = plot_margin_distribution(df, returned_only=True, 
                                save_path="figures/05_margin_returned.png")
fig6 = plot_margin_loss_by_category(df, save_path="figures/06_margin_loss.png")

# Customer analysis
fig7 = plot_customer_margin_exposure(df, save_path="figures/07_customer_exposure.png")

# Geographic analysis
returned_geo = calculate_price_margin_returned_by_country(df)
plot_price_margin_returned_by_status_country(returned_geo, 
                                             save_path="figures/08_geo_metrics")
```

### 4.2 Report Generation Workflow

```python
# Generate all visualizations for report
from src.config import FIGURES_DIR

set_plot_style()

# Create comprehensive set
plots = [
    ("status_distribution", plot_status_distribution(df)),
    ("return_by_category", plot_return_rate_by_category(df, top_n=20)),
    ("margin_loss", plot_margin_loss_by_category(df, top_n=15)),
    ("customer_exposure", plot_customer_margin_exposure(df, top_n=25)),
]

for name, fig in plots:
    fig.savefig(f"{FIGURES_DIR}/{name}.png", dpi=150, bbox_inches="tight")
```

---

## 5. Configuration & Customization

### 5.1 Figure Size Guidelines

| Use Case | figsize | Notes |
|----------|---------|-------|
| Report slide | (10, 6) | Standard slide ratio |
| Presentation | (12, 8) | More readable |
| Detailed analysis | (14, 10) | Heatmaps, many categories |
| Poster/Print | (16, 12) | High detail required |
| Tall figure | (12, 20) | For 50+ categories |

### 5.2 Color Palette

| Function | Color | Rationale |
|----------|-------|-----------|
| plot_status_distribution | steelblue | Professional, neutral |
| plot_return_rate_by_category | coral | Highlights concern |
| plot_margin_loss_by_category | indianred | Danger/loss indication |
| plot_customer_margin_exposure | darkorange | Warning/risk |
| plot_return_rate_heatmap | YlOrRd | Standard diverging |

### 5.3 DPI & Quality Settings

All functions save at:
- **DPI:** 150 (publication quality)
- **Format:** PNG (web-friendly)
- **Bbox:** "tight" (removes whitespace)

---

## 6. Common Patterns

### 6.1 Saving Multiple Figures

```python
# Batch save
figures_dir = "reports/figures"
os.makedirs(figures_dir, exist_ok=True)

fig1 = plot_return_rate_by_category(df, save_path=f"{figures_dir}/returns.png")
fig2 = plot_margin_loss_by_category(df, save_path=f"{figures_dir}/losses.png")
fig3 = plot_customer_margin_exposure(df, save_path=f"{figures_dir}/exposure.png")
```

### 6.2 Creating Comparison Plots

```python
# Side-by-side comparison
fig1 = plot_margin_distribution(df, returned_only=False)
fig2 = plot_margin_distribution(df, returned_only=True)
# Save both for comparison
fig1.savefig("all_items.png")
fig2.savefig("returned_items.png")
```

### 6.3 Adjusting for Different Datasets

```python
# Adapt parameters based on data size
if len(df) > 1_000_000:
    top_n = 20  # More categories possible
    min_rows = 500  # Stricter thresholds
else:
    top_n = 10  # Show only top performers
    min_rows = 100  # More lenient

fig = plot_return_rate_by_category(df, top_n=top_n, min_rows=min_rows)
```

---

## Summary

The `visualization.py` module provides:
- ✅ 8 core plotting functions for comprehensive analysis
- ✅ Consistent, professional styling
- ✅ Publication-quality output (150 DPI PNG)
- ✅ Flexible customization options
- ✅ Clear business interpretation labels
- ✅ Multi-dimensional heatmap analysis
- ✅ Automatic tight layout handling
- ✅ Optional file saving for reporting
