# RQ1 SSL Analysis Documentation
**Capstone Project - Master of Data Analytics**  
**Research Question 1 (RQ1) - External Validation Using SSL Dataset**  
**Validation scope: Category level only**

---

## 1. Objective

This document validates the RQ1 profit erosion analysis on a real-world SSL returns dataset.

**RQ1:** *Do returned items exhibit statistically significant differences in profit erosion across product categories?*

Validation checks two things:

1. Metric compatibility: a line-level `profit_erosion` measure can be constructed from SSL data.
2. Structural consistency: profit erosion differs across SSL categories in the same direction as the TheLook findings.

Brand and department are not validated independently. SSL validation is category-only.

---

## 2. Dataset Scope and Unit of Analysis

**Primary engineered artifact:** `data/processed/rq1_ssl/rq1_ssl_engineered.parquet`

- Dataset: `data/raw/SSL_Returns_df_yoy.csv`
- Unit of analysis: returned order line
- Canonical outputs: `rq1_ssl_engineered.parquet`, `rq1_ssl_returned_items.parquet`, `rq1_ssl_base_canonical.parquet`

Because SSL is a returns-focused operational extract, return-frequency metrics are not directly comparable to TheLook without a full non-return population.

---

## 3. SSL Category Construction

SSL does not expose a single TheLook-style category field. The category used for validation is:

```text
category = Pillar + "-" + Major Market Cat + "-" + Department
```

Missing values are filled with `"Unknown"` before concatenation.

---

## 4. Field Mapping and Canonical Schema

Field alignment between datasets is documented in `docs/rq1_ssl_validation_reference.md`.

Core mappings:

- `Pillar + Major Market Cat + Department -> category`
- `total_loss -> profit_erosion` using absolute value
- `Returns -> is_returned_item`

---

## 5. Profit Erosion Definition (SSL)

Unlike TheLook, SSL contains realized loss accounting. For validation:

- `profit_erosion = abs(total_loss)`

This preserves the RQ1 objective of comparing economic impact per returned item across categories.

---

## 6. Visual Diagnostics (SSL)

The validation workflow reproduces the core category-level RQ1 visuals:

### 6.1 Top Categories by Total Profit Erosion

Shows the categories contributing the largest cumulative losses.

### 6.2 Return Composition vs Mean Profit Erosion

Retained for structural parity, but interpreted cautiously because SSL is not a full purchase-and-return population.

### 6.3 Severity vs Volume Decomposition

Decomposes total loss into:

- Volume: returned line count per category
- Severity: mean profit erosion per return

### 6.4 Profit Erosion Distribution (Log Scale)

Confirms right-skew and heavy-tailed loss behavior.

### 6.5 Bootstrap 95% CI for Mean Profit Erosion

Shows uncertainty around category-level mean erosion estimates.

---

## 7. Hypothesis Testing Results (SSL)

The SSL workflow applies the same non-parametric logic as the main RQ1 notebook.

### 7.1 Category-Level Differences

- Test: Kruskal-Wallis
- Grouping: `Pillar-Major Market Cat-Department`
- Conclusion: reject the null hypothesis; profit erosion differs across categories

Operationally, this means some SSL category groups are consistently more expensive when returned.

---

## 8. Cross-Dataset Validation Summary (TheLook vs SSL)

### 8.1 What matches

1. Both datasets show heavy-tailed profit erosion distributions.
2. Both workflows reject the null hypothesis of no category differences.
3. In both datasets, high total erosion is explained by a combination of volume and severity.

### 8.2 What differs

1. SSL magnitudes are larger because they reflect realized accounting losses.
2. SSL categories are more granular because they use a composite hierarchy.
3. SSL return-composition metrics are not directly comparable to TheLook return rates.
4. SSL validation does not reproduce a separate brand-level hypothesis test.

---

## 9. Validation Artifacts Produced (SSL)

The SSL validation workflow writes artifacts using the same folder discipline as the main notebook:

**Processed data (`data/processed/rq1_ssl/`):**

- `rq1_ssl_engineered.parquet`
- `rq1_ssl_returned_items.parquet`
- `rq1_ssl_base_canonical.parquet`

**Report CSVs (`reports/rq1_ssl/`):**

- `rq1_ssl_by_category.csv`
- `rq1_ssl_test_summary_category.csv`
- `rq1_ssl_posthoc_category.csv`
- `rq1_ssl_bootstrap_ci_category_mean.csv`

**Figures (`figures/rq1_ssl/`):**

- `fig1_top_categories_total_erosion.png`
- `fig2_return_rate_vs_mean_erosion_category.png`
- `fig3_severity_vs_volume_category.png`
- `fig4_profit_erosion_distribution_log.png`
- `fig5_bootstrap_ci_category_mean.png`

---

## References

- Looker. *TheLook eCommerce dataset*.
- School Specialty, Inc. *SSL_Returns_df_yoy*.
- Conover, W. J. *Practical Nonparametric Statistics*.
- Efron, B., and Tibshirani, R. J. *An Introduction to the Bootstrap*.
