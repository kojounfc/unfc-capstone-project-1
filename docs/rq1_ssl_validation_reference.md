# RQ1 SSL Validation Reference (Field Mapping + Validation Notes)
**Capstone Project – Master of Data Analytics**  
**External Validation Dataset: SSL_Returns_df_yoy (real-world transactional returns)**  
**Scope: RQ1 (Category/Brand differences in Profit Erosion)**

---

## 1. Purpose

This document defines the **field mappings** used to validate the RQ1 profit erosion pipeline on the SSL dataset and explains how SSL fields align to the TheLook (synthetic) dataset fields used in the original RQ1 analysis.

The goal is **directional and structural validation**: confirm that (1) the same derived metric (**profit_erosion**) can be produced from SSL data using consistent logic, and (2) RQ1 hypothesis tests produce comparable conclusions (differences exist across product groupings), even if magnitudes differ due to business context, pricing, and return policy differences.

---

## 2. Unit of Analysis and Filtering

- **Unit of analysis (RQ1):** Returned order line / returned item.
- **TheLook:** Filtered to returned items (returned order-items only).
- **SSL:** Dataset is an operational returns extract; **all rows represent returns** (e.g., `Returns = TRUE`).  
  **Implication:** Any “return rate” visual/metric must be interpreted carefully because SSL does **not** include the full non-returned population in the same file.

---

## 3. Canonical RQ1 Fields (What the pipeline expects)

RQ1 analysis (both datasets) uses a canonical set of columns:

| Canonical RQ1 Field | Meaning | Used in |
|---|---|---|
| `order_id` | Unique order identifier | grouping, sanity checks |
| `order_line_id` | Unique line identifier | item-level grain |
| `category` | Product category | hypothesis test + visuals |
| `brand` | Supplier/brand | hypothesis test + visuals |
| `department` | Higher-level department grouping | descriptive visual |
| `sale_amount` | Revenue impact baseline (optional) | descriptive context |
| `profit_erosion` | Loss per returned item (key metric) | visuals + hypothesis tests |
| `returned_flag` | return indicator | filtering |
| `date` | transaction date | optional slicing |

---

## 4. Field Mappings: TheLook ↔ SSL (RQ1 Validation)

> **Note:** These mappings were selected to preserve the **business meaning** of each field, not just similar names.

| Canonical RQ1 Field | TheLook Field (Synthetic) | SSL Field (Real-world) | Notes / Transformations |
|---|---|---|---|
| `order_id` | `order_id` | `Order Number` | Cast to string for stability |
| `order_line_id` | `order_item_id` | `Order Line ID` | Cast to string |
| `category` | `category` / `product_category` | `Class` (or `Department` when Class is too granular) | For RQ1, we used **`Class`** to approximate product category |
| `brand` | `brand` | `Supplier Name` | If missing/“Not Available”, keep as category “Unknown/Not Available” |
| `department` | `department` | `Department` | Direct mapping |
| `sale_amount` | `sale_price` (or item revenue) | `CreditReturn Sales` (absolute value) | SSL returns are often negative; take absolute value for “size” measures |
| `profit_erosion` | `profit_erosion` (engineered) | `total_loss` (absolute value) | RQ1 SSL pipeline uses `profit_erosion = abs(total_loss)` |
| `returned_flag` | return status flag | `Returns` | SSL extract contains returns only; keep flag for auditing |
| `date` | `created_at` / `shipped_at` | `Booked Date` / `Billed Date` / `Reference Booked Date` | Prefer `Booked Date` as the primary event date |

---

## 5. Profit Erosion Definition in SSL (Validation Logic)

TheLook profit erosion is modeled as:

- **Profit Erosion = Margin Reversal + Reverse-logistics Processing Cost**

SSL already includes engineered economic outcomes. For RQ1 validation, we map:

- **`profit_erosion` (SSL) = abs(`total_loss`)**

Why this is acceptable for validation:
- It represents realized financial loss at the returned line level (economic impact).
- It allows the same downstream analysis (group comparisons and distribution diagnostics) while acknowledging that SSL loss accounting may include costs not explicitly modeled in TheLook.

---

## 6. Known Limitations (Important for interpretation)

1. **Return rate is not comparable**  
   SSL file does not include non-return transactions, so “return rate” will often appear as **1.0** if computed naïvely. For SSL, return-rate visuals should be framed as **composition/volume** and **severity**, not frequency.

2. **Category granularity differs**  
   SSL `Class` can be much more granular than TheLook categories. This increases the number of tested groups and can amplify significance due to sample size. For reporting, consider optionally rolling up to `Department` for a higher-level validation view.

3. **Accounting definitions differ**  
   `total_loss` may include policy-related, freight, vendor, or operational adjustments that do not exist in TheLook’s simplified model. This is expected and should be stated in the validation narrative.

---

## 7. Validation Artifacts Produced (RQ1 SSL)

The RQ1 SSL notebook produces the same artifact types as the TheLook RQ1 notebook:

- `rq1_ssl_engineered.parquet` (feature-engineered SSL dataset)
- `rq1_ssl_returned_items.parquet` (canonical return-line dataset)
- `rq1_ssl_base_canonical.parquet` (canonical base dataset incl. non-return-safe fields)
- `rq1_ssl_by_category.csv`
- `rq1_ssl_by_brand.csv`
- `rq1_ssl_by_department.csv`
- `rq1_ssl_posthoc_category.csv` (if Dunn post-hoc enabled)
- `rq1_ssl_posthoc_brand.csv` (if Dunn post-hoc enabled)
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_test_summary_brand.csv`
- Figures (7): top categories, top brands, return-rate vs mean erosion, top departments, severity vs volume, erosion distribution (log), bootstrap CI plot

---

## References (APA)

- Looker. (n.d.). *TheLook eCommerce dataset* [Demo dataset]. Looker.  
- School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].
- Conover, W. J. (1999). *Practical nonparametric statistics* (3rd ed.). Wiley.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.
