# Baseline Exploratory Data Analysis (EDA) Technical Documentation

**Capstone Project — Master of Data Analytics**

---

## 1. Objective of Baseline EDA

The objective of the baseline Exploratory Data Analysis (EDA) phase is
to understand the structural composition of the dataset prior to
advanced modeling and hypothesis testing.

This stage focuses on:

- Dataset source, grain, and table structure
- Order status distribution and return volume
- Return rate behavior across product categories
- Item margin distribution for all items vs. returned items
- Geographic variation in return rates
- Data quality flags and cleaning decisions

This phase is strictly descriptive. No inferential statistical testing
is performed here.

---

## 2. Dataset Source & Structure

**Source:** `bigquery-public-data.thelook_ecommerce` (Google BigQuery)

Four tables were merged at the **order-item grain** into a single analytical dataset:

| Table | Key Columns |
|---|---|
| `order_items` | `order_item_id`, `order_id`, `product_id`, `sale_price`, `status` |
| `orders` | `order_id`, `user_id`, `order_status`, timestamps |
| `products` | `product_id`, `brand`, `category`, `department`, `retail_price`, `cost` |
| `users` | `user_id`, `age`, `gender`, `country`, `traffic_source`, `created_at` |

The analytical unit is **one row per order-item**, not per order. The
pipeline in `src/data_processing.py` loads, merges, and standardizes types.
Feature engineering (`src/feature_engineering.py`) adds return flags,
margins, and profit erosion metrics.

---

## 3. Order Item Status Distribution

Order items progress through five statuses:
`Shipped`, `Complete`, `Returned`, `Cancelled`, `Processing`

**Observed counts:**

| Status | Count |
|---|---|
| Shipped | 53,931 |
| Complete | 45,277 |
| Processing | 36,215 |
| Cancelled | 27,277 |
| Returned | 18,208 |

**Interpretation:**

Only items with `item_status == 'Returned'` are included in profit
erosion calculations. With over 18,000 returned items, returns are not
marginal events — they represent a financially significant and
operationally relevant subset of total order volume. This validates
the relevance of profit erosion modeling in subsequent research questions.

---

## 4. Return Rate by Product Category

Return rate is defined as: `returned items / total items` per category.
Categories with fewer than 200 items are excluded for stability.

**Highest return rate categories (~10–11% range):**

- Blazers & Jackets
- Maternity
- Suits
- Clothing Sets
- Active

**Interpretation:**

Return rates are relatively clustered within a narrow 10–11% band
across high-volume categories. There is no extreme outlier category in
terms of return frequency alone.

This suggests that **frequency alone may not fully explain financial
risk**, reinforcing the need to examine margin severity — the motivation
for the profit erosion framework developed in RQ1.

The processing cost model applies tiered multipliers based on category
return behavior: **1.0× (standard), 1.15× (moderate), 1.3× (high)**.

---

## 5. Item Margin Distribution

`item_margin = sale_price − cost`

For returned items, the margin is reversed and added to the processing
cost to form `profit_erosion`:

`profit_erosion = margin_reversal + process_cost`

**Key observations:**

- The all-items margin distribution is approximately symmetric and
  centred above zero, confirming positive margins in normal operations.
- The returned-items distribution shows a heavy left tail, indicating
  that a small number of returns generate disproportionately large losses.
- This non-normality motivates the use of non-parametric tests
  (Kruskal–Wallis) in RQ1 rather than ANOVA.

---

## 6. Geographic Distribution

Return rates were compared across all countries in the dataset.

**Finding:**

The coefficient of variation (CV) across country-level return rates
was **3.58%**, well below the **10% threshold** for meaningful geographic
tiering.

**Decision:**

No geographic multipliers were applied to the processing cost model.
The low CV indicates that return behavior is structurally consistent
across markets and that a geography-based cost tier would not be
analytically justified.

---

## 7. Data Quality Summary

The cleaning pipeline in `src/data_cleaning.py` applies the following checks:

| Check | Action |
|---|---|
| Duplicate rows | Detected and removed (keep first) |
| Missing values | Reported; numeric imputed with median for ML (RQ3 only) |
| Price inconsistencies | Flagged: sale > retail, cost > sale, negative prices |
| Status consistency | Flagged: returned-not-delivered, item/order status mismatch |
| Temporal consistency | Flagged: delivered-before-shipped, returned-before-delivered |
| Categorical cleaning | Stripped whitespace; case-normalised for grouping |

Flagged rows are written to `data/processed/data_to_review.csv` for audit
and are excluded from downstream analysis.

---

## 8. Baseline EDA Summary

Baseline EDA establishes:

1. Returns represent a non-trivial operational event (~18,200 items, ~10% of total volume).
2. Return rates are moderately stable across categories (10–11% band).
3. Margin distributions for returned items exhibit a heavy left tail, motivating non-parametric analysis.
4. Geographic variation in return rates is low (CV = 3.58%) — no geographic cost tiers applied.
5. Frequency-based analysis alone does not reveal economic severity.
6. All downstream analysis uses the cleaned, feature-engineered dataset.

This phase provides contextual grounding before margin and profit
erosion analysis in RQ1–RQ4.

---
