# Data Dictionary - Profit Erosion Analysis

This document provides comprehensive definitions for all columns in the feature-engineered dataset used for the Profit Erosion E-commerce Capstone Project.

## Table of Contents

1. [Raw Data Columns](#raw-data-columns)
2. [Engineered Features](#engineered-features)
   - [Return Features](#return-features)
   - [Margin Features](#margin-features)
   - [Profit Erosion Features](#profit-erosion-features-task-1)
   - [Customer Behavioral Features](#customer-behavioral-features-task-2)
   - [Product-Level Features](#product-level-features-task-3)
   - [Temporal Features](#temporal-features-task-4)
   - [Target Variables](#target-variables-task-5)
3. [Assumptions and Methodology](#assumptions-and-methodology)
4. [Research Question Mapping](#research-question-mapping)

---

## Raw Data Columns

These columns originate from the source BigQuery tables (`order_items`, `orders`, `products`, `users`).

### Identifiers

| Column | Type | Description | Source Table |
|--------|------|-------------|--------------|
| `order_item_id` | int | Unique identifier for each order line item | order_items.id |
| `order_id` | int | Unique identifier for each order | orders.order_id |
| `user_id` | int | Unique identifier for each customer | orders.user_id |
| `product_id` | int | Unique identifier for each product | order_items.product_id |
| `product_dim_id` | int | Product dimension ID (same as product_id) | products.id |
| `user_dim_id` | int | User dimension ID (same as user_id) | users.id |
| `inventory_item_id` | int | Inventory item identifier | order_items.inventory_item_id |

### Status Columns

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `item_status` | string | Status of the order item: Complete, Returned, Shipped, Cancelled, Processing | order_items.status |
| `order_status` | string | Status of the order | orders.status |

### Price and Cost Columns

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `sale_price` | float | Actual selling price of the item | order_items.sale_price |
| `retail_price` | float | Original retail price before discount | products.retail_price |
| `cost` | float | Product cost (COGS) | products.cost |

### Product Attributes

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `category` | string | Product category (e.g., Jeans, Tops & Tees) | products.category |
| `brand` | string | Product brand name | products.brand |
| `department` | string | Product department (Women, Men) | products.department |
| `sku` | string | Stock keeping unit | products.sku |
| `name` | string | Product name | products.name |
| `distribution_center_id` | int | Distribution center ID | products.distribution_center_id |

### Customer Attributes

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `user_gender` | string | Customer gender (M/F) | users.gender |
| `age` | int | Customer age | users.age |
| `city` | string | Customer city | users.city |
| `state` | string | Customer state/province | users.state |
| `country` | string | Customer country | users.country |
| `postal_code` | string | Customer postal code | users.postal_code |
| `latitude` | float | Customer location latitude | users.latitude |
| `longitude` | float | Customer location longitude | users.longitude |
| `traffic_source` | string | Acquisition channel (Search, Organic, Email, Display, Facebook) | users.traffic_source |

### Date/Time Columns

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `order_created_at` | datetime | When the order was created | orders.created_at |
| `item_created_at` | datetime | When the order item was created | order_items.created_at |
| `item_shipped_at` | datetime | When the item was shipped | order_items.shipped_at |
| `item_delivered_at` | datetime | When the item was delivered | order_items.delivered_at |
| `item_returned_at` | datetime | When the item was returned (null if not returned) | order_items.returned_at |
| `user_created_at` | datetime | When the customer account was created | users.created_at |
| `num_of_item` | int | Number of items in the order | orders.num_of_item |

---

## Engineered Features

### Return Features

Created by `engineer_return_features()` in `src/feature_engineering.py`.

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `is_returned_item` | int | Binary flag: 1 if item was returned | `(item_status.lower() == 'returned').astype(int)` | RQ1-4 |
| `is_returned_order` | int | Binary flag: 1 if order status is returned | `(order_status.lower() == 'returned').astype(int)` | RQ1-4 |

### Margin Features

Created by `calculate_margins()` in `src/feature_engineering.py`.

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `item_margin` | float | Gross margin per item (profit before return) | `sale_price - cost` | RQ1 |
| `item_margin_pct` | float | Margin as percentage of sale price | `item_margin / sale_price` | RQ1 |
| `discount_amount` | float | Discount from retail price | `retail_price - sale_price` | RQ1 |
| `discount_pct` | float | Discount as percentage of retail price | `discount_amount / retail_price` | RQ1 |

### Profit Erosion Features (Task 1)

Created by `calculate_profit_erosion()` in `src/feature_engineering.py`.

**Note:** These columns are calculated on **returned items only** (where `is_returned_item == 1`).

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `margin_reversal` | float | Margin lost on returned item | `item_margin` (for returned items) | RQ1 |
| `process_cost` | float | Estimated processing cost per return | `base_cost ($12) * category_multiplier` | RQ1 |
| `profit_erosion` | float | Total economic loss per returned item | `margin_reversal + process_cost` | RQ1 |

#### Processing Cost Category Multipliers

| Tier | Categories | Multiplier | Avg Margin |
|------|------------|------------|------------|
| Premium (1.3x) | Outerwear & Coats, Suits & Sport Coats, Blazers & Jackets, Jeans, Dresses, Suits, Sweaters, Pants | 1.3 | $52.25 |
| Moderate (1.15x) | Skirts, Active, Swim, Maternity, Sleep & Lounge, Accessories, Pants & Capris, Fashion Hoodies & Sweatshirts, Shorts | 1.15 | $27.49 |
| Standard (1.0x) | Plus, Tops & Tees, Intimates, Underwear, Leggings, Socks & Hosiery, Socks, Jumpsuits & Rompers, Clothing Sets | 1.0 | $15.25 |

### Customer Behavioral Features (Task 2)

Created by `engineer_customer_behavioral_features()` in `src/feature_engineering.py`.

**Note:** These are customer-level features (one row per customer).

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `order_frequency` | int | Total unique orders per customer | `nunique(order_id)` per user_id | RQ2, RQ3 |
| `return_frequency` | int | Total return events per customer | `sum(is_returned_item)` per user_id | RQ2, RQ3 |
| `customer_return_rate` | float | Proportion of items returned | `return_frequency / total_items` | RQ2, RQ3 |
| `avg_basket_size` | float | Average items per order | `total_items / order_frequency` | RQ2, RQ3 |
| `avg_order_value` | float | Average total order value | `mean(order_total)` per customer | RQ2, RQ3 |
| `customer_tenure_days` | int | Days since account creation | `reference_date - user_created_at` | RQ2, RQ3 |
| `purchase_recency_days` | int | Days since last order | `reference_date - max(order_created_at)` | RQ2, RQ3 |
| `total_items` | int | Total items purchased | `count(order_item_id)` per user_id | RQ2 |
| `total_sales` | float | Sum of sale prices | `sum(sale_price)` per user_id | RQ2 |
| `total_margin` | float | Sum of item margins | `sum(item_margin)` per user_id | RQ2 |
| `avg_item_price` | float | Average sale price per item | `mean(sale_price)` per user_id | RQ2 |
| `avg_item_margin` | float | Average margin per item | `mean(item_margin)` per user_id | RQ2 |

### Product-Level Features (Task 3)

Created by `engineer_product_level_features()` in `src/analytics.py`.

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `category_return_rate` | float | Return rate for item's category | `returned_items / total_items` per category | RQ1 |
| `brand_return_rate` | float | Return rate for item's brand | `returned_items / total_items` per brand | RQ1 |
| `price_tier` | string | Price tier classification | Tercile of sale_price: 'low', 'medium', 'high' | RQ1, RQ3 |

### Temporal Features (Task 4)

Created by `engineer_temporal_features()` in `src/analytics.py`.

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `order_day_of_week` | int | Day of week (0=Monday, 6=Sunday) | `order_created_at.dayofweek` | RQ1, RQ3 |
| `order_month` | int | Month of year (1-12) | `order_created_at.month` | RQ1, RQ3 |
| `order_quarter` | int | Quarter of year (1-4) | `order_created_at.quarter` | RQ1 |
| `order_year` | int | Year | `order_created_at.year` | RQ1 |
| `is_weekend_order` | bool | True if ordered on Saturday or Sunday | `order_day_of_week in [5, 6]` | RQ3 |
| `season` | string | Season of order | Mapped from month: winter, spring, summer, fall | RQ1, RQ3 |
| `days_to_delivery` | int | Days from order to delivery | `item_delivered_at - order_created_at` | RQ1 |
| `days_to_return` | int | Days from delivery to return (NaN if not returned) | `item_returned_at - item_delivered_at` | RQ1 |

### Target Variables (Task 5)

Created by `create_profit_erosion_targets()` in `src/feature_engineering.py`.

**Note:** These are customer-level targets for predictive modeling.

| Column | Type | Definition | Formula | RQ |
|--------|------|------------|---------|-----|
| `total_profit_erosion` | float | Total profit erosion per customer | `sum(profit_erosion)` per user_id | RQ3, RQ4 |
| `is_high_erosion_customer` | int | Binary flag: 1 if above 75th percentile | `total_profit_erosion > quantile(0.75)` | RQ3 |
| `profit_erosion_quartile` | int | Quartile assignment (1=lowest, 4=highest) | `pd.qcut(total_profit_erosion, 4)` | RQ3 |
| `erosion_percentile_rank` | float | Percentile rank (0-100) | `rank(pct=True) * 100` | RQ3 |

---

## Mathematical Formulation

### Profit Erosion Model

The profit erosion framework quantifies the total economic loss from product returns through two components:

**Total Profit Erosion (per returned item):**

```
Profit_Erosion_i = Margin_Reversal_i + Processing_Cost_i
```

Where:

**1. Margin Reversal** - The gross margin lost when a customer returns an item:

```
Margin_Reversal_i = Sale_Price_i - Cost_i = Item_Margin_i
```

- `Sale_Price_i`: Actual selling price of item i
- `Cost_i`: Product cost (COGS) of item i
- Only calculated for items where `is_returned_item = 1`

**2. Processing Cost** - The operational cost to handle the return:

```
Processing_Cost_i = Base_Cost × Category_Multiplier_c
```

Where:
- `Base_Cost = $12.00` (sum of component costs)
- `Category_Multiplier_c ∈ {1.0, 1.15, 1.3}` based on category tier

**Aggregation Formulas:**

*Customer-Level Profit Erosion:*
```
Total_Profit_Erosion_u = Σ Profit_Erosion_i  (for all returned items by user u)
```

*Order-Level Profit Erosion:*
```
Total_Profit_Erosion_o = Σ Profit_Erosion_i  (for all returned items in order o)
```

### Margin Calculations

**Item Margin (Gross Profit per Item):**
```
Item_Margin = Sale_Price - Cost
```

**Item Margin Percentage:**
```
Item_Margin_Pct = Item_Margin / Sale_Price = (Sale_Price - Cost) / Sale_Price
```

**Discount Metrics:**
```
Discount_Amount = Retail_Price - Sale_Price
Discount_Pct = Discount_Amount / Retail_Price
```

### Customer Behavioral Metrics

**Return Rate:**
```
Customer_Return_Rate = Return_Frequency / Total_Items
```

Where:
- `Return_Frequency = Σ is_returned_item` (count of returned items)
- `Total_Items` = total items purchased by customer

**RFM-Style Metrics:**
```
Order_Frequency = COUNT(DISTINCT order_id) per user
Avg_Basket_Size = Total_Items / Order_Frequency
Avg_Order_Value = MEAN(Order_Total) per user
Customer_Tenure_Days = Reference_Date - User_Created_At
Purchase_Recency_Days = Reference_Date - MAX(Order_Created_At)
```

---

## Assumptions and Methodology

### Return Processing Cost Model

#### Why Model Processing Costs?

The TheLook e-commerce dataset does not include actual operational cost data for processing returns. To quantify total profit erosion (margin reversal + operational costs), we must model the processing cost component based on:

1. **Academic literature** on reverse logistics costs
2. **Industry benchmarks** from comparable e-commerce operations
3. **Empirical analysis** of the dataset's margin and return rate distributions

#### Base Cost Assumption: $12.00 per Return

The base processing cost of $12 per return represents a **mid-range conservative estimate** derived from the following sources:

| Source | Finding | Relevance |
|--------|---------|-----------|
| Rogers & Tibben-Lembke (2001) | Reverse logistics costs typically 4-10% of original sale revenue | Establishes industry baseline |
| Guide & Van Wassenhove (2009) | Closed-loop supply chain processing costs range $10-25 per unit | Provides academic validation |
| Optoro (2019 Industry Report) | Average e-commerce return processing cost $10-20 | Confirms contemporary benchmarks |
| School Specialty LLC (validation data) | Directional validation from real operations | Supports reasonableness |

**Justification for $12.00:** This value falls within the lower half of the $10-25 academic range, making it a conservative estimate that is less likely to overstate profit erosion. For a synthetic dataset, using a mid-low estimate ensures findings are directionally valid without exaggerating economic impact.

#### Cost Component Breakdown

| Component | Amount | Calculation Basis | Justification |
|-----------|--------|-------------------|---------------|
| Customer Care | $4.00 | 10-15 min handling time × $16-24/hr labor rate | Returns require customer service interaction for authorization, tracking, and resolution |
| Inspection | $2.50 | 5-8 min assessment time | Returned items must be visually and functionally inspected before restocking or disposal |
| Restocking | $3.00 | Physical handling + system updates | Includes shelving labor, inventory system updates, and warehouse management |
| Logistics | $2.50 | Administrative + label costs | Return label generation, carrier coordination, and administrative processing |
| **Total** | **$12.00** | | |

#### Category-Tiered Multiplier Justification

**Decision Basis:** Empirical analysis of 18,208 returned items revealed **59.4% coefficient of variation (CV)** in margins across product categories. This exceeds the 15% threshold warranting differentiated treatment.

| Analysis Finding | Value | Implication |
|------------------|-------|-------------|
| Margin CV across categories | 59.4% | High variance justifies category-specific treatment |
| Return rate CV across countries | 3.58% | Low variance does NOT justify geographic tiers |
| Average margin range | $7.89 - $82.38 | 10x difference in margin-at-risk |

**Tier Assignment Rationale:**

| Tier | Multiplier | Avg Margin | Justification |
|------|------------|------------|---------------|
| **Premium (1.3x)** | $15.60 effective | $52.25 | High-value items (Outerwear, Suits, Jeans) require more careful handling, detailed inspection, and represent greater margin-at-risk. Higher processing cost reflects operational reality of premium item returns. |
| **Moderate (1.15x)** | $13.80 effective | $27.49 | Mid-range items (Swim, Active, Accessories) with standard processing but above-average value justifying moderate uplift. |
| **Standard (1.0x)** | $12.00 effective | $15.25 | Lower-margin items (Tops & Tees, Socks, Underwear) where processing cost already represents a larger share of item value. Base rate applies. |

**Why Not Flat Rate?** A flat $12 rate would understate profit erosion for high-margin categories where operational complexity is higher, and overstate relative impact for low-margin categories. The tiered approach better reflects the economic reality that processing a $150 coat involves more care than processing a $15 t-shirt.

#### Geographic Cost Decision

**Decision:** Uniform processing costs across all countries (no geographic tiers).

**Justification:**
- Return rate CV across 15 countries: **3.58%** (below 10% threshold)
- Return rate range: 9.61% (France) to 10.80% (Germany) - only 1.19 percentage point spread
- Insufficient variance to justify country-specific cost multipliers

#### Key Assumptions Summary

| Assumption | Rationale | Sensitivity |
|------------|-----------|-------------|
| $12 base cost | Mid-range of $10-25 literature range; conservative for synthetic data | Test with $8-18 range |
| Category tiers (1.0x, 1.15x, 1.3x) | 59.4% margin CV justifies differentiation | Test alternative multipliers |
| No geographic variation | 3.58% return rate CV indicates uniform operations | N/A - variance too low |
| Uniform returns processing | All returns follow similar operational flow | Could vary by return reason |
| No seasonality in costs | Processing costs constant year-round | Could vary with volume |
| No salvage value modeling | Returns assumed to have zero residual value | Understates recovery potential |

#### Limitations

1. **No Real Cost Data:** Base cost derived from literature, not actual TheLook operations (which don't exist for synthetic data)
2. **Category Granularity:** Some categories may have sub-segments with different cost profiles
3. **Excludes Shipping Costs:** Model does not include reverse shipping postage paid by retailer
4. **No Refurbishment Costs:** Does not model costs to repair/refurbish damaged returns

For complete methodology documentation including raw analysis output, see `docs/PROCESSING_COST_METHODOLOGY.md`.

### Margin Calculations

- **Item Margin:** `sale_price - cost` represents gross margin before overhead
- **Margin Percentage:** Calculated as a proportion of sale price, not cost

### Customer Segmentation Thresholds

Default thresholds for customer segmentation by return behavior:

| Segment | Return Rate Threshold |
|---------|----------------------|
| no_returns | return_rate = 0 |
| low_returner | return_rate <= 5% |
| moderate_returner | 5% < return_rate <= 15% |
| high_returner | return_rate > 15% |

---

## Research Question Mapping

### Overview

| Research Question | Method | Target Variable | Key Predictors |
|-------------------|--------|-----------------|----------------|
| **RQ1** | Descriptive Analysis | `profit_erosion`, `margin_reversal` | `category`, `brand`, `category_return_rate`, `brand_return_rate` |
| **RQ2** | Unsupervised Learning (Clustering) | Customer segments | `customer_return_rate`, `return_frequency`, `order_frequency`, `avg_basket_size`, `customer_tenure_days` |
| **RQ3** | Predictive Modeling (Classification) | `is_high_erosion_customer` | All behavioral, product, and temporal features |
| **RQ4** | Econometric Regression | `total_profit_erosion` (continuous) | All behavioral and temporal features |
| **RQ5** | Prescriptive Analytics (Optional) | Intervention threshold | `erosion_percentile_rank`, `profit_erosion_quartile` |

### Detailed Feature-to-RQ Mapping

#### RQ1: Profit Erosion Differences Across Product Categories/Brands

**Objective:** Identify high-risk product categories and brands that contribute disproportionately to profit erosion.

| Feature | Role | Analysis |
|---------|------|----------|
| `profit_erosion` | Dependent variable | Aggregated by category/brand |
| `margin_reversal` | Component analysis | Margin loss contribution |
| `process_cost` | Component analysis | Processing cost contribution |
| `category_return_rate` | Risk indicator | Proportion of returns by category |
| `brand_return_rate` | Risk indicator | Proportion of returns by brand |
| `item_margin` | Context | Profitability before return |

#### RQ2: Customer Behavioral Segments

**Objective:** Identify customer segments with differential profit erosion through clustering.

| Feature | Role | Clustering Use |
|---------|------|----------------|
| `customer_return_rate` | Segmentation driver | Primary clustering variable |
| `return_frequency` | Behavior indicator | Return volume metric |
| `order_frequency` | Behavior indicator | Purchase frequency |
| `avg_basket_size` | Behavior indicator | Order size pattern |
| `avg_order_value` | Value indicator | Customer spending level |
| `customer_tenure_days` | Context | Customer maturity |
| `total_margin` | Value indicator | Total profit contribution |

#### RQ3: Predict High Profit Erosion Customers

**Objective:** Build classification model to identify high-risk customers (target AUC > 0.70).

| Feature | Role | Model Input |
|---------|------|-------------|
| `is_high_erosion_customer` | **Target (Y)** | Binary: 1 if above 75th percentile |
| `customer_return_rate` | Predictor (X) | Historical return behavior |
| `return_frequency` | Predictor (X) | Return count |
| `order_frequency` | Predictor (X) | Purchase frequency |
| `avg_basket_size` | Predictor (X) | Order pattern |
| `customer_tenure_days` | Predictor (X) | Account age |
| `purchase_recency_days` | Predictor (X) | Recency (RFM) |
| `category_return_rate` | Predictor (X) | Product risk exposure |
| `price_tier` | Predictor (X) | Price sensitivity |
| `is_weekend_order` | Predictor (X) | Temporal pattern |
| `season` | Predictor (X) | Seasonal pattern |

#### RQ4: Marginal Associations Between Behaviors and Profit Erosion

**Objective:** Quantify marginal effects of customer behaviors on profit erosion through regression.

| Feature | Role | Regression Specification |
|---------|------|-------------------------|
| `total_profit_erosion` | **Target (Y)** | Continuous dependent variable |
| `customer_return_rate` | Predictor (X) | Key behavioral coefficient |
| `order_frequency` | Predictor (X) | Volume effect |
| `avg_order_value` | Predictor (X) | Value effect |
| `customer_tenure_days` | Control (X) | Customer maturity control |
| `traffic_source` | Control (X) | Acquisition channel effect |
| Temporal features | Controls (X) | Seasonal/time controls |

---

## Data Quality Notes

### Known Issues

1. **Temporal Inconsistencies:** ~38,000 records have `created_at` after `shipped_at` or `delivered_at` due to synthetic data generation. Use `item_shipped_at` and `item_delivered_at` for lifecycle analysis.

2. **Missing Values:**
   - `item_returned_at`: Null for non-returned items (expected)
   - `item_delivered_at`: May be null for orders not yet delivered
   - Geographic fields: Some postal codes may have mixed types

3. **Category Coverage:** The category tier mapping covers all categories in the dataset. Unknown categories default to Standard tier (1.0x multiplier).

### Validation

Feature quality can be validated using:

```python
from src.analytics import validate_feature_quality, generate_feature_quality_report

report = validate_feature_quality(df)
print(generate_feature_quality_report(df))
```

This generates a comprehensive quality report including:
- Missing value counts and percentages
- Distribution statistics (mean, std, skew, kurtosis)
- High correlations (|r| > 0.8)
- Constant and low-variance columns

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-31 | Initial data dictionary with all Task 1-6 features |
