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
