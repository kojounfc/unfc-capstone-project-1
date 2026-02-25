# RQ1 Technical Documentation

**Capstone Project – Master of Data Analytics**
**Research Question 1 (RQ1)**

---

## 1. Research Question

**RQ1:**
*Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?*

This research question evaluates whether the economic impact of product returns — operationalized as **profit erosion per returned item** — varies systematically across product categories and brands within the e-commerce dataset.

While return rate is frequently used in retail analytics, prior research demonstrates that frequency alone does not capture financial exposure (Petersen & Kumar, 2009). Therefore, RQ1 focuses on a profit-based metric that integrates both margin reversal and reverse logistics cost.

Establishing statistically significant structural variation is a prerequisite for predictive modeling in later research questions.

---

## 2. Hypotheses

All hypothesis testing was conducted at significance level α = 0.05.

### 2.1 Category-Level Hypothesis

* **H₀:** Mean profit erosion is equal across product categories.
* **H₁:** Mean profit erosion differs across product categories.

### 2.2 Brand-Level Hypothesis

* **H₀:** Mean profit erosion is equal across brands.
* **H₁:** Mean profit erosion differs across brands.

---

## 3. Data Scope and Unit of Analysis

* **Unit of analysis:** Returned order-item
* **Dataset:** TheLook synthetic e-commerce dataset
* **Returned observations (category analysis):** 436
* **Returned observations (brand analysis subset):** 56
* **Groups tested (categories):** 21
* **Groups tested (brands):** 7

Only returned items were included in profit erosion calculations, ensuring that the metric reflects realized return-related financial loss.

Upstream feature engineering was implemented via **US06 (Return Feature Engineering Pipeline)**, and descriptive aggregation via **US07 (Profit Erosion Descriptive Module)**.

---

## 4. Profit Erosion Definition (US06)

Profit erosion was operationalized using the standardized US06 pipeline as:

[
\text{Profit Erosion} = \text{Margin Reversal} + \text{Processing Cost}
]

Where:

* **Margin Reversal:** Lost item-level contribution margin
* **Processing Cost:** Modeled reverse logistics cost
* Category-tier multipliers applied

This formulation aligns with reverse logistics cost frameworks in operations research (Guide & Van Wassenhove, 2009).

---

## 5. Descriptive Analysis (US07 Outputs)

### 5.1 Distribution of Profit Erosion

**Figure 1: Distribution of Profit Erosion (Log Scale)**

The distribution of profit erosion is strongly right-skewed with a heavy tail.

Observations:

* Extreme high-loss returns
* Non-normal distribution
* Log transformation required

Implication:

Normality assumptions are violated, justifying non-parametric testing (Conover, 1999).

---

### 5.2 Category-Level Total Profit Erosion

**Figure 2: Top Categories by Total Profit Erosion**

Top contributors:

| Category            | Total Profit Erosion |
| ------------------- | -------------------- |
| Outerwear & Coats   | ~$2.0K               |
| Sweaters            | ~$1.6K               |
| Jeans               | ~$1.4K               |
| Suits & Sport Coats | ~$1.3K               |
| Pants               | ~$1.2K               |

Interpretation:

Structured and higher-margin apparel categories generate the largest financial losses. This reflects the amplification effect of high unit margins when returns occur (Petersen & Kumar, 2010).

Loss concentration is economically meaningful rather than evenly distributed.

---

### 5.3 Department-Level Aggregation

**Figure 3: Top Departments by Total Profit Erosion**

| Department | Total Profit Erosion |
| ---------- | -------------------- |
| Men        | ~$10.7K              |
| Women      | ~$8.1K               |

The Men’s department exhibits materially higher total erosion.

This suggests department-level structural asymmetry in return-related financial exposure.

---

### 5.4 Brand-Level Concentration

**Figure 4: Top Brands by Total Profit Erosion**

Top brands include:

* Orvis
* Allegra K
* Tommy Hilfiger
* Volcom

Profit erosion is concentrated among a small subset of brands, indicating heterogeneity in brand-level economic risk.

---

### 5.5 Severity vs Volume Decomposition

**Figure 5: Severity vs Volume (Category Decomposition)**

Total erosion decomposed as:

[
\text{Total Erosion} = \text{Return Volume} \times \text{Mean Erosion per Return}
]

Findings:

* **Outerwear & Coats:** High severity + moderate volume
* **Suits & Sport Coats:** Very high severity
* **Fashion Hoodies:** High volume, moderate severity

Conclusion:

Different categories generate losses through different mechanisms. This distinction is critical for operational targeting (Guide & Van Wassenhove, 2009).

---

### 5.6 Return Rate vs Mean Profit Erosion

**Figure 6: Return Rate vs Mean Profit Erosion per Return**

The relationship between return rate and erosion severity is weak.

Implication:

Return rate alone is insufficient as a financial risk indicator.

This aligns with findings that frequency-based metrics do not fully capture profitability impact (Petersen & Kumar, 2009).

---

## 6. Statistical Analysis

### 6.1 Category-Level Kruskal–Wallis Test

* **p-value:** 4.99 × 10⁻²⁷
* **Effect size (ε²):** 0.377
* **Groups:** 21
* **Observations:** 436
* **Decision:** Reject H₀

The extremely small p-value provides overwhelming evidence against equality of group means.

The epsilon-squared value (0.377) indicates a **large practical effect** (Tomczak & Tomczak, 2014).

Therefore, category-level differences are both statistically and economically significant.

---

### 6.2 Brand-Level Kruskal–Wallis Test

* **p-value:** 4.44 × 10⁻⁵
* **Effect size (ε²):** 0.484
* **Groups:** 7
* **Decision:** Reject H₀

The brand-level effect size exceeds the category-level effect.

This suggests brand-specific structural drivers of financial loss.

---

### 6.3 Post-Hoc Testing

Dunn’s post-hoc tests with Bonferroni correction were conducted.

Significant pairwise differences were observed between:

* Structured apparel categories and casual categories
* High-margin and low-margin brand groups

These results confirm that specific groups drive overall heterogeneity.

---

## 7. Bootstrap Confidence Intervals

**Figure 7: Bootstrap 95% Confidence Intervals (Category Means)**

Bootstrap resampling (Efron & Tibshirani, 1993) was used to estimate robust uncertainty intervals under non-normal conditions.

Example:

| Category        | Mean  | 95% CI         |
| --------------- | ----- | -------------- |
| Fashion Hoodies | 26.36 | [22.35, 30.67] |
| Tops & Tees     | 16.99 | [13.74, 21.39] |

Limited overlap between intervals reinforces structural differences.

---

## 8. Integrated Interpretation

RQ1 provides converging descriptive and inferential evidence that:

1. Profit erosion is not uniformly distributed.
2. Category-level differences are large and economically meaningful.
3. Brand-level heterogeneity is pronounced.
4. Department-level asymmetry exists.
5. Return rate alone is insufficient as a financial KPI.

The null hypotheses are rejected at both levels.

The effect sizes indicate meaningful structural economic variation rather than trivial statistical artifacts.

---

## 9. Business Implications

From a managerial perspective:

1. High-margin structured apparel requires targeted mitigation strategies.
2. Brand-level erosion dashboards should supplement traditional return metrics.
3. Men’s department requires enhanced monitoring.
4. Reverse logistics cost optimization may materially reduce losses.
5. Profit erosion should complement return rate as a primary performance metric.

These findings support financially informed return management policy rather than frequency-only monitoring.

---

## 10. Conclusion (RQ1)

RQ1 establishes that profit erosion differs significantly across product categories and brands therefore rejects the null hypotheses.

The statistical evidence is robust, and effect sizes confirm practical relevance.

These findings provide the inferential foundation for predictive modeling in subsequent research questions.

---

## 11. Traceability to User Stories

* **US06:** Return feature engineering and profit erosion computation
* **US07:** Descriptive aggregation and visualization outputs
* **RQ1:** Statistical validation of structural heterogeneity

---

## 12. References

Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Guide, V. D. R., & Van Wassenhove, L. N. (2009). The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10–18.

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? *Journal of Marketing*, 73(3), 35–51.

Petersen, J. A., & Kumar, V. (2010). Can product returns make you money? *MIT Sloan Management Review*, 51(3), 85–89.

Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19–25.

---

## 13. External Validation Using SSL Dataset (Real-world Returns)

### 13.1 Validation Purpose

This external validation is implemented in `profit_erosion_analysis.ipynb` (Section **6.14–6.27**) and reproduces the full RQ1 artifact + visual set on SSL.

TheLook is a synthetic dataset designed for analytics benchmarking. To strengthen external validity, we replicated the full RQ1 workflow on a real-world returns dataset (SSL). The objective is **structural validation**: confirm that the same RQ1 conclusion (profit erosion differs across product groupings) holds under operational data.

Dataset citation (APA): *SSL_Returns_df_yoy* (School Specialty, Inc., 2025).

### 13.2 Field Mapping and Canonical Alignment

Field mappings were documented in `rq1_ssl_validation_reference.md`. In summary:

- **Category proxy (SSL):** `Class`
- **Brand proxy (SSL):** `Supplier Name`
- **Department (SSL):** `Department`
- **Profit erosion (SSL):** `profit_erosion = abs(total_loss)`

This differs from TheLook where profit erosion is modeled as margin reversal + processing cost (Guide & Van Wassenhove, 2009), but it preserves the RQ1 intent: measure **economic impact per returned item**.

### 13.3 SSL Dataset Scope

- **Rows (returned lines):** 133800
- **Unit of analysis:** Returned order line (returned item)
- **Note:** SSL extract is returns-only; therefore **return rate is not directly comparable** to TheLook without the non-return population.

### 13.4 Visual Replication (SSL)

The SSL notebook recreates and displays the full **7-figure RQ1 visual suite**:

1. Top Categories by Total Profit Erosion  
2. Top Brands by Total Profit Erosion  
3. Return Rate vs Mean Profit Erosion (Category) *(interpret cautiously in SSL)*  
4. Top Departments by Total Profit Erosion  
5. Severity vs Volume Decomposition (Category)  
6. Distribution of Profit Erosion (Log Scale)  
7. Bootstrap 95% CI for Mean Profit Erosion (Category)

Across SSL, the same pattern holds: losses are **right-skewed** and concentrated in specific categories, brands, and departments, with total erosion explained by a mix of **return volume** and **loss severity**.

### 13.5 Hypothesis Testing Results (SSL)

The SSL dataset exhibits non-normal profit erosion distributions; therefore, non-parametric testing is used (Conover, 1999).

- **Category-level test:** Kruskal-Wallis  
  p = 0.0000e+00 → **Reject H₀**

- **Brand-level test:** Kruskal-Wallis  
  p = 0.0000e+00 → **Reject H₀**

**Conclusion:** The RQ1 findings generalize directionally to real-world data: profit erosion differs significantly across product groupings (category and brand). This supports targeted risk-mitigation and supplier/category-specific interventions.


### 13.6 Validation Artifacts Produced (SSL)

All SSL validation outputs are written under `data/processed/rq1_ssl/` to keep traceability consistent with the main pipeline:

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


---

## References (APA) – Addendum for Validation

- School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].  
- Conover, W. J. (1999). *Practical nonparametric statistics* (3rd ed.). Wiley.  
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.  
- Looker. (n.d.). *TheLook eCommerce dataset* [Demo dataset]. Looker.
