# RQ1 SSL Validation Reference (Field Mapping + Validation Notes)
**Capstone Project – Master of Data Analytics**
**External Validation Dataset: SSL_Returns_df_yoy (real-world transactional returns)**
**Scope: RQ1 (Category differences in Profit Erosion — category-only validation)**

---

## 1. Purpose

This document defines the **field mappings** used to validate the RQ1 profit erosion pipeline on
the SSL dataset and explains how SSL fields align to the TheLook (synthetic) dataset fields used
in the original RQ1 analysis.

The goal is **directional and structural validation**: confirm that (1) the same derived metric
(**profit_erosion**) can be produced from SSL data using consistent logic, and (2) the RQ1
category-level hypothesis test produces comparable conclusions (differences exist across product
categories), even if magnitudes differ due to business context, pricing, and return policy
differences.

**Scope of validation:** Category level only. Brand and department are **not** validated
independently.

---

## 2. Unit of Analysis and Filtering

- **Unit of analysis (RQ1):** Returned order line / returned item.
- **TheLook:** Filtered to returned items (returned order-items only).
- **SSL:** Dataset is an operational returns extract; **all rows represent returns** (e.g., `Returns = TRUE`).
  **Implication:** Any "return rate" visual/metric must be interpreted carefully because SSL does
  **not** include the full non-returned population in the same file.

---

## 3. SSL Category Construction

SSL does not have a single category field equivalent to TheLook's `category`. Instead, category
is constructed by **concatenating three SSL fields** with a hyphen separator:

```
category = Pillar + "-" + Major Market Cat + "-" + Department
```

**Example output:** `STEM-Science-Physics`, `Art-Visual Arts-Ceramics/Sculpture`

This composite label captures the full product hierarchy and is used as the single grouping
dimension for all RQ1 statistical tests and visuals. Missing values in any component are filled
with `"Unknown"` before concatenation.

---

## 4. Canonical RQ1 Fields (What the pipeline expects)

RQ1 category-level analysis uses a canonical set of columns:

| Canonical RQ1 Field | Meaning | Used in |
|---|---|---|
| `order_id` | Unique order identifier | grouping, sanity checks |
| `order_line_id` | Unique line identifier | item-level grain |
| `category` | Composite product category (see Section 3) | hypothesis test + visuals |
| `profit_erosion` | Loss per returned item (key metric) | visuals + hypothesis tests |
| `is_returned_item` | Return indicator | filtering |
| `date` | Transaction date | optional slicing |

---

## 5. Field Mappings: TheLook ↔ SSL (RQ1 Validation)

> **Note:** Only the fields required for category-level validation are mapped.
> Brand and department are **not** mapped as independent validation dimensions.

| Canonical RQ1 Field | TheLook Field (Synthetic) | SSL Field(s) (Real-world) | Notes / Transformations |
|---|---|---|---|
| `order_id` | `order_id` | `Order Number` | Cast to string for stability |
| `order_line_id` | `order_item_id` | `Order Line ID` | Cast to string |
| `category` | `category` / `product_category` | `Pillar` + `Major Market Cat` + `Department` | Concatenated with `-` separator |
| `profit_erosion` | `profit_erosion` (engineered) | `total_loss` (absolute value) | `profit_erosion = abs(total_loss)` |
| `is_returned_item` | return status flag | `Returns` | SSL extract contains returns only |
| `date` | `created_at` / `shipped_at` | `Booked Date` / `Billed Date` | Prefer `Booked Date` as primary event date |

---

## 6. Profit Erosion Definition in SSL (Validation Logic)

TheLook profit erosion is modeled as:

- **Profit Erosion = Margin Reversal + Reverse-logistics Processing Cost**

SSL already includes engineered economic outcomes. For RQ1 validation, we map:

- **`profit_erosion` (SSL) = abs(`total_loss`)**

Why this is acceptable for validation:
- It represents realized financial loss at the returned line level (economic impact).
- It allows the same downstream analysis (group comparisons and distribution diagnostics) while
  acknowledging that SSL loss accounting may include costs not explicitly modeled in TheLook.

---

## 7. Known Limitations (Important for interpretation)

1. **Return rate is not comparable**
   SSL file does not include non-return transactions, so "return rate" will often appear as
   **1.0** if computed naïvely. For SSL, the return-rate visual is retained for structural
   parity with TheLook but must be framed as **composition/volume** and **severity**, not
   frequency.

2. **Category granularity**
   The composite `Pillar-Major Market Cat-Department` label is more granular than TheLook
   categories. This increases the number of tested groups and can amplify significance due to
   sample size. For high-level summaries, consider rolling up to `Pillar` alone.

3. **Accounting definitions differ**
   `total_loss` may include policy-related, freight, vendor, or operational adjustments that
   do not exist in TheLook's simplified model. This is expected and should be stated in the
   validation narrative.

4. **Brand not validated**
   The SSL dataset's brand/supplier field is not used for hypothesis testing. The validation
   scope is intentionally limited to the composite category dimension.

---

## 8. Validation Artifacts Produced (RQ1 SSL)

The RQ1 SSL notebook produces the following reproducible artifacts:

**Data (under `data/processed/rq1_ssl/`):**
- `rq1_ssl_engineered.parquet` (feature-engineered SSL dataset)
- `rq1_ssl_returned_items.parquet` (canonical return-line dataset)
- `rq1_ssl_base_canonical.parquet` (canonical base dataset)
- `rq1_ssl_by_category.csv` / `.parquet`
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_posthoc_category.csv` (if Dunn post-hoc enabled)
- `rq1_ssl_bootstrap_ci_category_mean.csv` / `.parquet`

**Figures (under `figures/rq1_ssl/`):**
- `fig1_top_categories_total_erosion.png`
- `fig2_return_rate_vs_mean_erosion_category.png`
- `fig3_severity_vs_volume_category.png`
- `fig4_profit_erosion_distribution_log.png`
- `fig5_bootstrap_ci_category_mean.png`

---

## References (APA)

- Looker. (n.d.). *TheLook eCommerce dataset* [Demo dataset]. Looker.
- School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].
- Conover, W. J. (1999). *Practical nonparametric statistics* (3rd ed.). Wiley.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.
