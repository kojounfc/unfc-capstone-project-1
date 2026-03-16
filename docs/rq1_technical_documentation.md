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

- **p-value:** 2.63 × 10⁻³³  
- **Effect size (ε²):** 0.454  
- **Groups:** 21  
- **Observations:** 436  
- **Decision:** Reject H₀  

The p-value (2.63 × 10⁻³³) is far below 0.05, which means profit erosion is not equal across product categories. The differences we observe are extremely unlikely to be caused by random variation.

The effect size (ε² = 0.454) is considered large (Tomczak & Tomczak, 2014). This confirms that category differences are not just statistically significant — they are also meaningful in real business terms.

---

### 6.2 Brand-Level Kruskal–Wallis Test

- **p-value:** 1.08 × 10⁻⁴  
- **Effect size (ε²):** 0.442  
- **Groups:** 7  
- **Observations:** 56  
- **Decision:** Reject H₀  

The brand-level results also lead us to reject the null hypothesis. The p-value (1.08 × 10⁻⁴) shows strong statistical evidence that profit erosion differs across brands.

The effect size (ε² = 0.442) indicates a large difference, suggesting that brand identity is an important driver of financial loss from returns.

---

### 6.3 Post-Hoc Testing

Dunn’s post-hoc tests with Bonferroni correction were conducted on the top groups. Significant pairwise differences confirm that the overall statistical result is driven by clear and consistent group-level differences.

## 7. Bootstrap Confidence Intervals

Bootstrap resampling (1,000 iterations) was used to estimate 95% confidence intervals without assuming normality (Efron & Tibshirani, 1993).

Example:

| Category        | Mean  | 95% CI         |
|-----------------|-------|---------------|
| Fashion Hoodies | 26.36 | [22.35, 30.67] |
| Tops & Tees     | 16.99 | [13.74, 21.39] |

Limited overlap between intervals supports the conclusion that category differences are stable and reliable.

---

## 8. Integrated Interpretation

When we look at the descriptive charts and statistical tests together, a clear pattern emerges:

1. Profit erosion is not evenly distributed.  
2. Category differences are statistically significant and meaningful in practice.  
3. Brand-level differences are even stronger.  
4. The Men’s department generates higher total erosion.  
5. Return rate alone does not fully capture financial risk.

Overall, both null hypotheses are rejected. Category and brand membership clearly influence return-related financial loss.

---

## 9. Business Implications

From a business perspective, these findings suggest:

1. High-margin structured apparel should be prioritized for return mitigation.  
2. Brand-level monitoring should complement return-rate dashboards.  
3. The Men’s department requires closer monitoring due to higher total erosion.  
4. Improving reverse logistics efficiency could reduce financial impact.  
5. Profit erosion should be tracked alongside return rate as a key performance indicator.

These recommendations are based on statistically significant and practically meaningful results.

---

## 10. Conclusion (RQ1)

RQ1 confirms that profit erosion differs significantly across product categories and brands at α = 0.05.

The results are statistically robust, and the effect sizes show that the differences matter in practice. These findings provide a strong foundation for later predictive modeling work.

---

## 11. Dashboard Element Interpretations

### KPI Cards

**Total Profit Erosion** – Total financial loss from all returned items.  
**Total Returns Analyzed** – Number of returned items included in the analysis.  
**Mean Erosion per Return** – Average loss per returned item.  
**Highest-Risk Category** – Category contributing the largest total erosion.

### Statistical Summary

The Kruskal–Wallis tests show statistically significant differences across categories (ε² = 0.454) and brands (ε² = 0.442). These are large effect sizes, meaning the differences are meaningful for business decision-making.

### Key Takeaway

Profit erosion is not evenly distributed. Category and brand are important drivers of financial loss from returns.


## 12. Traceability to User Stories

* **US06:** Return feature engineering and profit erosion computation
* **US07:** Descriptive aggregation and visualization outputs
* **RQ1:** Statistical validation of structural heterogeneity

---

## 13. References

Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.

Guide, V. D. R., & Van Wassenhove, L. N. (2009). The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10–18.

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? *Journal of Marketing*, 73(3), 35–51.

Petersen, J. A., & Kumar, V. (2010). Can product returns make you money? *MIT Sloan Management Review*, 51(3), 85–89.

Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19–25.

---


---

## 14. External Validation Using SSL Dataset (Real-world Returns)

### 14.1 Validation Purpose

TheLook is a synthetic dataset designed for analytics benchmarking. To strengthen external
validity, we replicated the RQ1 **category-level** workflow on a real-world returns dataset
(SSL). The objective is **structural validation**: confirm that the same RQ1 conclusion (profit
erosion differs across product categories) holds under operational data.

**Validation scope:** Category level only. Brand and department are **not** validated
independently.

Dataset citation (APA): *SSL_Returns_df_yoy* (School Specialty, Inc., 2025).

### 14.2 SSL Category Construction

SSL does not have a single category field equivalent to TheLook's `category`. The category
dimension is constructed by concatenating three SSL fields:

```
category = Pillar + "-" + Major Market Cat + "-" + Department
```

**Example:** `STEM-Science-Physics`, `Art-Visual Arts-Ceramics/Sculpture`

This composite label captures the full SSL product hierarchy and is the sole grouping dimension
for all statistical tests and visuals. Full field mapping is documented in
`rq1_ssl_validation_reference.md`.

**Profit erosion (SSL):** `profit_erosion = abs(total_loss)`

This differs from TheLook where profit erosion is modeled as margin reversal + processing cost
(Guide & Van Wassenhove, 2009), but it preserves the RQ1 intent: measure **economic impact per
returned item**.

### 14.3 SSL Dataset Scope

- **Rows (returned lines):** 133,800
- **Unit of analysis:** Returned order line (returned item)
- **Note:** SSL extract is returns-only; therefore **return rate is not directly comparable** to
  TheLook without the non-return population.

### 14.4 Visual Replication (SSL)

The SSL notebook reproduces 5 category-focused RQ1 visuals:

1. Top Categories by Total Profit Erosion
2. Return Rate vs Mean Profit Erosion per Return (Category) *(interpret cautiously — SSL is returns-only)*
3. Severity vs Volume Decomposition (Category)
4. Distribution of Profit Erosion (Log Scale)
5. Bootstrap 95% CI for Mean Profit Erosion (Category)

Across SSL, the same pattern holds: losses are **right-skewed** and concentrated in specific
categories, with total erosion explained by a mix of **return volume** and **loss severity**.

### 14.5 Hypothesis Testing Results (SSL)

The SSL dataset exhibits non-normal profit erosion distributions; therefore, non-parametric
testing is used (Conover, 1999).

- **Category-level test:** Kruskal-Wallis (grouping: `Pillar-Major Market Cat-Department`)
  p = 0.0000e+00 → **Reject H₀**

**Conclusion:** The RQ1 category-level finding generalizes to real-world data: profit erosion
differs significantly across product categories. This supports targeted risk-mitigation and
category-specific interventions.

---

## References (APA) — Addendum for Validation

- School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].  
- Conover, W. J. (1999). *Practical nonparametric statistics* (3rd ed.). Wiley.  
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.  
- Looker. (n.d.). *TheLook eCommerce dataset* [Demo dataset]. Looker.
