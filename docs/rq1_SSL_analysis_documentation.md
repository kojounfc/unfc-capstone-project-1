# RQ1 SSL Analysis Documentation (External Validation)
**Capstone Project – Master of Data Analytics**
**Research Question 1 (RQ1) – External Validation Using SSL Dataset**
**Validation scope: Category level only**

---

## 1. Objective

This document validates the **RQ1 profit erosion analysis** on a **real-world SSL returns dataset**.
The intent is to confirm that the RQ1 category-level conclusion holds when applying the pipeline
to operational returns data:

**RQ1:** *Do returned items exhibit statistically significant differences in profit erosion across
product categories?*

Validation checks two things:

1. **Metric compatibility:** We can construct a line-level `profit_erosion` measure from SSL data
   that represents economic impact per return.
2. **Structural consistency:** Profit erosion differs across product categories in SSL, consistent
   with TheLook results.

**Note:** Brand and department are **not** validated independently. Only the composite category
dimension is used for hypothesis testing.

---

## 2. Dataset Scope and Unit of Analysis

**Primary engineered artifact:** `data/processed/rq1_ssl/rq1_ssl_engineered.parquet`
(optional CSV also saved for audit).

- **Dataset:** `SSL_Returns_df_yoy.csv` (cleaned + feature engineered prior to validation)
- **Rows (engineered):** 133,800
- **Columns:** 82
- **Unit of analysis:** Returned order line (returned item)

**Important:** SSL extract is returns-only. Therefore, return-frequency metrics (e.g., return rate)
are not directly comparable to TheLook without additional non-return data.

---

## 3. SSL Category Construction

SSL does not have a single field equivalent to TheLook's `category`. The category dimension is
constructed by concatenating three SSL fields:

```
category = Pillar + "-" + Major Market Cat + "-" + Department
```

**Example labels:** `STEM-Science-Physics`, `Art-Visual Arts-Ceramics/Sculpture`

Missing values in any component are filled with `"Unknown"` before concatenation. This composite
label captures the full SSL product hierarchy and is the sole grouping dimension for all
statistical tests and visuals.

---

## 4. Field Mapping and Canonical Schema

Field alignment between datasets is defined in `rq1_ssl_validation_reference.md`.

At a minimum, SSL provides:

- `Pillar` + `Major Market Cat` + `Department` → composite `category` (concatenated with `-`)
- `total_loss` → economic loss per return-line (mapped to `profit_erosion`)
- `Returns` → return indicator

---

## 5. Profit Erosion Definition (SSL)

Unlike TheLook (where profit erosion is modeled), SSL contains realized loss accounting.
For validation we define:

- **`profit_erosion` (SSL) = abs(`total_loss`)**

This preserves the core RQ1 intent: compare **economic impact per returned item** across
categories.

---

## 6. Visual Diagnostics (SSL)

The SSL validation notebook reproduces **5 category-focused RQ1 visuals** with SSL-specific titles.

### 6.1 Top Categories by Total Profit Erosion

The SSL distribution shows large cumulative loss concentrated in specific product groupings.
Category labels reflect the composite `Pillar-Major Market Cat-Department` hierarchy.

### 6.2 Return Rate vs Mean Profit Erosion per Return (Category)

**Interpret cautiously:** SSL is a returns-only extract, so return_rate values may be uniformly
close to 1.0. This visual is retained for structural parity with TheLook but should not be
used to draw return-frequency conclusions.

### 6.3 Severity vs Volume Decomposition (Category)

The SSL severity-volume bubble plot illustrates the same RQ1 decomposition:

- **Volume (x):** Returned line count per category
- **Severity (y):** Mean profit erosion per return
- **Total loss:** Proportional to volume × severity

This provides the business interpretation layer for RQ1: high total loss can result from
**many moderate losses** or **few extreme losses**.

### 6.4 Distribution of Profit Erosion (Log Scale)

The SSL distribution is strongly right-skewed with a heavy tail, consistent with returns-loss
behaviour in the synthetic dataset. This supports non-parametric hypothesis testing.

### 6.5 Bootstrap 95% CI for Mean Profit Erosion (Category)

Bootstrap confidence intervals around category means show estimate stability and uncertainty.
Limited overlap across categories increases confidence that differences are reliable.

---

## 7. Hypothesis Testing Results (SSL)

Because the profit erosion distribution is non-normal and group counts differ substantially,
the SSL notebook uses non-parametric testing consistent with RQ1 methodology.

### 7.1 Category-level Differences

- **Test used:** Kruskal-Wallis
- **Grouping:** `Pillar-Major Market Cat-Department`
- **p-value:** 0.0000e+00
- **Conclusion:** Reject H₀. Mean/median profit erosion differs across categories.

**Interpretation (plain language):** Not all product categories cost the same when returned.
Some composite category groups consistently show higher loss per return, which supports targeted
mitigation strategies (policy, vendor accountability, product quality improvements).

---

## 8. Cross-Dataset Validation Summary (TheLook vs SSL)

### 8.1 What matches (strong validation)

1. **Distribution shape:** Both datasets show heavy-tailed profit erosion distributions.
2. **Statistical conclusion:** Both reject "no category differences" at p < 0.001.
3. **Business structure:** High total erosion is explained by a combination of volume and severity.

### 8.2 What differs (expected differences)

1. **Magnitude:** SSL losses are larger in absolute terms because they reflect real accounting
   rather than modeled reverse-logistics assumptions.
2. **Category granularity:** SSL composite category (`Pillar-Major Market Cat-Department`) is
   more granular than TheLook categories; more groups increases sensitivity.
3. **Return rate:** SSL extract is returns-only; return rate is not directly comparable without
   non-return transactions.
4. **Brand not compared:** TheLook validates both category and brand; SSL validates category only.

---

## 9. Validation Artifacts Produced (SSL)

The SSL validation workflow produces the following reproducible artifacts:

**Data (under `data/processed/rq1_ssl/`):**
- `rq1_ssl_engineered.parquet` (+ optional `.csv`)
- `rq1_ssl_returned_items.parquet`
- `rq1_ssl_base_canonical.parquet`
- `rq1_ssl_by_category.csv` / `.parquet`
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_posthoc_category.csv` (optional)
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
