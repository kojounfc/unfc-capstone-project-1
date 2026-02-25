# RQ1 SSL Analysis Documentation (External Validation)
**Capstone Project – Master of Data Analytics**  
**Research Question 1 (RQ1) – External Validation Using SSL Dataset**

---

## 1. Objective

This document validates the **RQ1 profit erosion analysis** on a **real-world SSL returns dataset**. The intent is to confirm that the same RQ1 business question and statistical conclusions hold when applying the pipeline to operational returns data:

**RQ1:** *Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?*

Validation checks two things:

1. **Metric compatibility:** We can construct a line-level `profit_erosion` measure from SSL data that represents economic impact per return.
2. **Structural consistency:** Profit erosion differs across product groupings (category and brand) in SSL, consistent with TheLook results.

---

## 2. Dataset Scope and Unit of Analysis

**Primary engineered artifact:** `data/processed/rq1_ssl/rq1_ssl_engineered.parquet` (optional CSV also saved for audit).

- **Dataset:** `SSL_Returns_df_yoy.csv` (cleaned + feature engineered prior to validation)
- **Rows (engineered):** 133800
- **Columns:** 82
- **Unit of analysis:** returned order line (returned item)

**Important:** SSL extract is returns-only. Therefore, return-frequency metrics (e.g., return rate) are not directly comparable to TheLook without additional non-return data.

---

## 3. Field Mapping and Canonical Schema

Field alignment between datasets is defined in:

- `rq1_ssl_validation_reference.md` (RQ1 mapping + notes)

At a minimum, SSL provides:

- `Department` → department grouping
- `Class` → category proxy
- `Supplier Name` → brand proxy
- `total_loss` → economic loss per return-line (mapped to `profit_erosion`)

---

## 4. Profit Erosion Definition (SSL)

Unlike TheLook (where profit erosion is modeled), SSL contains realized loss accounting. For validation we define:

- **`profit_erosion` (SSL) = abs(`total_loss`)**

This preserves the core RQ1 intent: compare **economic impact per returned item** across categories and brands.

---

## 5. Visual Diagnostics (SSL)

The SSL validation notebook reproduces the same **7-figure RQ1 visual set** used in TheLook, with SSL-specific titles.

### 5.1 Distribution of Profit Erosion (Log Scale)

The SSL distribution is strongly right-skewed with a heavy tail, consistent with returns-loss behavior in the synthetic dataset. This supports non-parametric hypothesis testing.

### 5.2 Top Categories / Brands / Departments by Total Profit Erosion

- **Top categories** show large cumulative loss concentrated in specific product groupings (e.g., Sensory Processing, Paint, Ceramics/Sculpture).
- **Top brands** show highly concentrated loss among a small set of suppliers (including “Not Available”, which likely represents missing supplier attribution in the extract).
- **Top departments** identify operational “loss hotspots” at a business-function level (e.g., Art Supplies, Paper).

### 5.3 Severity vs Volume Decomposition

The SSL severity-volume bubble plot illustrates the same RQ1 decomposition:

- **Volume (x):** returned line count (how often returns happen for the group)
- **Severity (y):** mean profit erosion per return (how expensive each return is)
- **Total loss:** implicitly proportional to volume × severity

This provides the business interpretation layer for RQ1: high total loss can result from **many moderate losses** or **few extreme losses**.

### 5.4 Return Rate vs Mean Profit Erosion

In SSL, return rate can appear artificially concentrated (often ~1.0) because the dataset is a returns-only extract. This plot is retained for structural parity with TheLook, but should be interpreted as a **validation placeholder** unless non-return transactions are integrated.

---

## 6. Hypothesis Testing Results (SSL)

Because the profit erosion distribution is non-normal and group counts differ substantially, the SSL notebook uses non-parametric testing consistent with RQ1 methodology.

### 6.1 Category-level Differences

- **Test used:** Kruskal-Wallis
- **Groups tested:** (see notebook)
- **p-value:** 0.0000e+00
- **Conclusion:** Reject H₀. Mean/median profit erosion differs across categories.

### 6.2 Brand-level Differences

- **Test used:** Kruskal-Wallis
- **Groups tested:** (see notebook)
- **p-value:** 0.0000e+00
- **Conclusion:** Reject H₀. Mean/median profit erosion differs across brands.

**Interpretation (plain language):** Not all product groups cost the same when returned. Some categories and suppliers consistently show higher loss per return, which supports targeted mitigation strategies (policy, vendor accountability, packaging/quality improvements).

---

## 7. Cross-Dataset Validation Summary (TheLook vs SSL)

### 7.1 What matches (strong validation)

1. **Distribution shape:** Both datasets show heavy-tailed profit erosion distributions.
2. **Statistical conclusion:** Both reject “no differences” across categories and brands.
3. **Business structure:** High total erosion is explained by a combination of volume and severity.

### 7.2 What differs (expected differences)

1. **Magnitude:** SSL losses are larger in absolute terms because they reflect real accounting rather than modeled reverse-logistics assumptions.
2. **Grouping:** SSL category proxy (`Class`) is more granular than TheLook categories; more groups increases sensitivity.
3. **Return rate:** SSL extract is returns-only; return rate is not directly comparable without non-return transactions.

---

## References (APA)

- Looker. (n.d.). *TheLook eCommerce dataset* [Demo dataset]. Looker.  
- School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].  
- Conover, W. J. (1999). *Practical nonparametric statistics* (3rd ed.). Wiley.  
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.


---

## 8. Validation Artifacts Produced (SSL)

The SSL validation workflow produces the following reproducible artifacts under `data/processed/rq1_ssl/`:

- `rq1_ssl_engineered.parquet` (+ optional `.csv`)
- `rq1_ssl_returned_items.parquet`
- `rq1_ssl_base_canonical.parquet`
- `rq1_ssl_by_category.csv`
- `rq1_ssl_by_brand.csv`
- `rq1_ssl_by_department.csv`
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_test_summary_brand.csv`
- `rq1_ssl_posthoc_category.csv` (optional)
- `rq1_ssl_posthoc_brand.csv` (optional)
- `rq1_ssl_bootstrap_ci_category_mean.csv` (+ optional `.parquet`)
