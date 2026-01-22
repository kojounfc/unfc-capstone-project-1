# Dataset Description – thelook_ecommerce

## Dataset Source

The primary dataset used for this project is the **thelook_ecommerce** dataset, available through **Google BigQuery Public Datasets**.

**Provider:**  
Google BigQuery Public Datasets  
Origin: Looker (Google) – Demonstration e-commerce dataset

**BigQuery Location:**
- Project: `bigquery-public-data`
- Dataset: `thelook_ecommerce`

This dataset is designed as a **simulated e-commerce environment** intended for analytics education, SQL training, and business intelligence demonstrations.

Official documentation is provided through Google Developers resources for BigQuery public datasets and Looker demo datasets.

---

## Dataset Purpose

The dataset represents a fictional online retail business and contains structured transactional data suitable for:

- SQL-based analytics  
- Data modeling and transformation  
- Business intelligence use cases  
- End-to-end analytics pipeline development  
- Methodological experimentation and validation  

---

## Core Tables in Scope

The following tables are relevant to this project’s analytical objectives:

- `orders`  
- `order_items`  
- `products`  
- `users`  
- `events`  
- `inventory_items`  
- `distribution_centers`  

These tables collectively support transaction-level analysis, customer behavior analysis, and operational modeling.

---

## Important Limitation

This dataset is **synthetic (not real commercial data)**. While the structure and relationships are realistic, all values are simulated. Therefore, conclusions derived from this dataset should be interpreted as:

- Demonstrations of analytical methodology  
- Validation of framework design  
- Educational exploration of behavioral and transactional patterns  

rather than direct empirical claims about real-world business performance.

This limitation will be explicitly considered in the methodology and interpretation of findings throughout the project.

---

## Data Acquisition Method

The raw dataset used in this project was extracted from the public BigQuery dataset:

`bigquery-public-data.thelook_ecommerce`

The following tables were downloaded as CSV files and stored under `data/raw/`:
- orders.csv  
- order_items.csv  
- products.csv  
- users.csv  

Extraction was performed using SQL queries executed in the Google BigQuery console, and files were exported using the BigQuery "Export to CSV" functionality.

Schema discovery and structural profiling were subsequently conducted on the extracted raw files and are documented in:
`notebooks/dataset_profiling.ipynb`

---

## Task 3 — Dataset Suitability and Limitations (thelook_ecommerce)

### Suitability for Project Objectives

The `thelook_ecommerce` dataset is suitable for this capstone because it supports **transaction-level analysis** across customers, orders, and products, which is required to quantify and model profit erosion mechanisms related to returns.

Key suitability factors based on the extracted tables used in this project (`orders`, `order_items`, `products`, `users`) include:

- **Relational join structure**
  - `orders.order_id` joins to `order_items.order_id`
  - `orders.user_id` and `order_items.user_id` support customer-level linking
  - `order_items.product_id` joins to `products.id`

- **Return and lifecycle analysis fields**
  - Both `orders` and `order_items` contain lifecycle timestamps (`created_at`, `shipped_at`, `delivered_at`, `returned_at`)
  - These fields allow analysis of return frequency and return timing (e.g., days-to-return)

- **Revenue and margin proxy variables**
  - `order_items.sale_price` provides a revenue proxy at the item level
  - `products.cost` and `products.retail_price` enable an approximate gross margin calculation
  - These support quantification of “value at risk” when items are returned

- **Customer segmentation support**
  - The `users` table includes demographic and geographic attributes (e.g., `age`, `gender`, `city`, `country`, `traffic_source`)
  - These enable segmentation and analysis of return behavior by customer cohorts and acquisition channels

Overall, the dataset provides sufficient structural realism and fields to build and demonstrate a complete analytics pipeline for analyzing return-related profit erosion drivers.

---

### Key Limitations

Despite its structural suitability, the dataset has limitations that constrain real-world economic conclusions:

1. **Synthetic / simulated nature**
   - The dataset is designed as a demonstration environment, meaning values are simulated rather than observed from real commerce operations.
   - Findings must therefore be framed as methodological insights rather than empirical business outcomes.

2. **Incomplete return-cost representation**
   - While return events can be inferred using `returned_at` and status fields, the dataset does not explicitly include full operational return costs such as:
     - reverse logistics/shipping cost
     - warehouse receiving and inspection labor
     - refurbishment, repackaging, disposal
     - customer service and administrative overhead
   - As a result, “profit erosion” must be modeled using proxies and assumptions.

3. **Revenue is represented via sale_price only**
   - The dataset includes `sale_price` at the item level, but does not include:
     - discounts/promotions as separate components
     - taxes, shipping revenue, or payment fees
     - multi-item order pricing logic beyond `num_of_item`
   - This limits precision for total order profitability modeling.

4. **Potential data-type and formatting inconsistencies from CSV export**
   - Local extraction may result in mixed-type fields (e.g., postal codes).
   - These issues require careful preprocessing and explicit typing rules.

---

### Implications for Methodology and Interpretation

To ensure analytical validity given the limitations above, this project will:

- Treat results as a **demonstration of analytical framework utility** rather than an exact economic estimate of real-world return costs.
- Use **proxy variables** for profitability, such as:
  - item-level margin proxy: `sale_price - cost`
  - return indicator: presence of `returned_at` and/or return status
- Include explicit assumptions where operational costs are not observed, and test sensitivity using scenario ranges where appropriate.
- Emphasize **pattern discovery and risk profiling** (e.g., high-return cohorts, low-margin items at risk) rather than claiming precise real-business financial impact.

---

### Data Quality Checks Applied

The following checks are required before downstream analysis:

- Uniqueness of join keys (`order_id`, `id`, `product_id`, `user_id`)
- Missingness assessment for `returned_at`, `sale_price`, `cost`, and join fields
- Date parsing and consistency validation for lifecycle timestamps
- Validation that `order_items.user_id` aligns with `orders.user_id` for the same `order_id` (spot-check)

These checks will be documented in the profiling notebook and preprocessing pipeline.
