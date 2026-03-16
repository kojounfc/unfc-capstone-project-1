# RQ2 Technical Documentation

**Capstone Project – Master of Data Analytics**
**Research Question 2 (RQ2)**

---

## 1. Research Question

**RQ2:**
*Can unsupervised learning identify distinct customer behavioral segments, and do these segments differ significantly in profit erosion intensity?*

This research question evaluates whether return-driven financial losses are disproportionately concentrated among a subset of customers, and whether statistically distinct, economically meaningful behavioral segments can be identified through unsupervised learning.

Concentration analysis (Gini coefficient, Pareto and Lorenz curves) establishes the descriptive foundation: if all customers generated equal erosion, segment-specific interventions would have no incremental value. The segmentation analysis then tests whether K-Means clustering can identify groups that differ significantly in profit erosion — providing the customer-level targeting framework that bridges RQ1 (product-level findings) and RQ3 (predictive modeling).

---

## 2. Hypothesis

All hypothesis testing was conducted at significance level α = 0.05.

**H₀₂ (Null):** Customer segments identified through clustering algorithms do not differ significantly in mean profit erosion from returns.

**H₁₂ (Alternative):** Customer segments identified through clustering algorithms exhibit statistically significant differences in mean profit erosion from returns.

*Note: Concentration analysis (Gini, Pareto, Lorenz) is the descriptive analytical method used to motivate and contextualise the segmentation hypothesis. It is not a separate hypothesis.*

---

## 3. Data Scope and Unit of Analysis

- **Unit of analysis:** Customer (aggregated from order-item transactions)
- **Dataset:** TheLook synthetic e-commerce dataset
- **Total customers:** 79,944
- **Analysis population (returners only):** 11,988 customers with at least one returned item (15.0% of total)
- **Total profit erosion (analysis population):** $816,500.94
- **Total returned items:** 18,208

Only customers with at least one return are included. The restriction is intentional: the concentration and segmentation analysis targets the population that generates return-driven losses, not the full customer base.

---

## 4. Profit Erosion Definition (US06)

Profit erosion was operationalized using the standardized US06 feature engineering pipeline, consistent with RQ1 and RQ3:

$$\text{Profit Erosion} = \text{Margin Reversal} + \text{Processing Cost}$$

- **Margin Reversal:** Item-level contribution margin lost due to the return. Total across dataset: $564,294.54.
- **Processing Cost:** Modeled reverse-logistics cost ($12 base × category-tiered multiplier: Premium 1.3× = $15.60, Moderate 1.15× = $13.80, Standard 1.0× = $12.00). Total: $252,206.40.
- **Average profit erosion per return:** $44.84

Processing cost tier distribution across 18,208 returned items:

| Tier | Rate | Items | Share |
|------|------|-------|-------|
| Standard | $12.00 | 5,042 | 27.7% |
| Moderate | $13.80 | 7,604 | 41.8% |
| Premium | $15.60 | 5,562 | 30.5% |

---

## 5. Concentration Analysis

### 5.1 Overview

Concentration analysis measures how unevenly profit erosion is distributed across the 11,988 returning customers. Two primary tools are used: the Gini coefficient and Pareto / Lorenz curves. Both are descriptive and do not require a hypothesis test — they characterise the shape of the problem before segmentation.

### 5.2 Gini Coefficient

**Gini = 0.409 (Moderate concentration)**

The Gini coefficient quantifies distributional inequality on a scale from 0 (perfect equality) to 1 (perfect concentration). A value of 0.409 indicates moderate inequality. Bootstrap significance testing (n = 1,000 resamples) confirmed this result is not attributable to chance (p < 0.0001).

### 5.3 Pareto Concentration

Cumulative erosion shares by customer rank:

| Customer Group | Share of Total Erosion | Absolute Amount |
|----------------|------------------------|-----------------|
| Top 20% | 47.6% | ~$388,814 |
| Top 50 customers | 2.9% | $23,808.31 |

The top 20% of customers account for 47.6% of total profit erosion — an approximate 80/20 relationship consistent with Pareto concentration (Koch, 1997). Concentration is spread across the top quintile rather than localized in a handful of extreme outliers.

### 5.4 Feature Concentration Ranking

All 18 customer-level behavioral features were scored by their own Gini coefficient — measuring how unequally erosion is distributed when customers are ranked by that feature:

| Feature | Gini | Top 20% Share | Level |
|---------|------|---------------|-------|
| `purchase_recency_days` | 0.528 | 54.2% | High |
| `total_margin_reversal` | 0.494 | 53.8% | Moderate |
| `total_margin` | 0.472 | 50.2% | Moderate |
| `total_sales` | 0.464 | 49.4% | Moderate |
| `total_profit_erosion` | 0.409 | 47.6% | Moderate |
| `avg_order_value` | 0.402 | — | Moderate |
| `avg_item_margin` | 0.367 | — | Moderate |
| `total_items` | 0.353 | — | Moderate |

`purchase_recency_days` is the most concentrated behavioral feature (Gini = 0.528) — recently active customers drive a disproportionate share of total losses. This makes recency-based alerting the highest-precision early-warning signal available.

### 5.5 Lorenz Curve

The Lorenz curve plots cumulative share of customers (x-axis) against cumulative share of profit erosion (y-axis). The further the curve bows below the 45° equality diagonal, the more unequal the distribution. The shaded area between the curve and the diagonal is proportional to the Gini coefficient.

---

## 6. Clustering Methodology

### 6.1 Feature Selection

Eight candidate features were selected for the clustering pipeline after leakage exclusion:

| Dimension | Features |
|-----------|----------|
| Purchase behaviour | `avg_order_value`, `avg_basket_size`, `order_frequency`, `total_sales` |
| Return behaviour | `customer_return_rate` |
| Margin structure | `total_margin` |
| Temporal | `customer_tenure_days`, `purchase_recency_days` |

### 6.2 Feature Screening (2 Sequential Gates)

| Gate | Method | Criterion | Outcome |
|------|--------|-----------|---------|
| 1 — Variance | `VarianceThreshold` (scikit-learn) | Variance < 0.01 | 8/8 features passed |
| 2 — Correlation | Pearson correlation matrix | \|r\| > 0.85 → drop lower-variance feature | `total_margin` dropped (r = 0.995 with `total_sales`) |

**Surviving features (7):** `avg_order_value`, `avg_basket_size`, `order_frequency`, `customer_return_rate`, `customer_tenure_days`, `purchase_recency_days`, `total_sales`.

### 6.3 Optimal k Selection

Two complementary indices were evaluated for k = 2 through k = 8:

| k | Silhouette Score |
|---|-----------------|
| **2** | **0.2921** ← peak |
| 3 | 0.2485 |
| 4 | 0.2490 |
| 5 | 0.2484 |
| 6 | 0.2452 |
| 7 | 0.2460 |
| 8 | 0.2402 |

The silhouette score peaks at k = 2 (0.2921) and is not exceeded at any higher k through k = 8. The inertia (elbow) curve shows an inflection at k ≈ 2. **k = 2 is the statistically optimal solution — not a simplification.**

### 6.4 Pipeline Architecture

```
1. Load customer-level feature data (8 candidate features)
2. Gate 1 — Variance threshold filtering
3. Gate 2 — Correlation screening (|r| > 0.85, drop lower-variance feature)
4. Standardize surviving features using StandardScaler
5. Determine optimal k via elbow (inertia) + silhouette analysis
6. Fit K-Means (k=2, random_state=42)
7. Assign cluster labels and compute cluster profiles
8. Post-hoc statistical testing (ANOVA + Kruskal-Wallis)
```

---

## 7. Clustering Results

### 7.1 Cluster Size and Erosion Distribution

| Cluster | Count | % of Population | Mean Erosion | Median Erosion | Total Erosion | % of Total |
|---------|-------|-----------------|--------------|----------------|---------------|------------|
| **Cluster 0** | 4,375 | 36.5% | $96.68 | $68.48 | $422,960.18 | 51.8% |
| **Cluster 1** | 7,613 | 63.5% | $51.69 | $40.89 | $393,540.76 | 48.2% |

Cluster 0 is the smaller, higher-erosion segment. Despite comprising only 36.5% of returners, it accounts for 51.8% of total profit erosion. The mean erosion difference between clusters is $44.99 (ratio: 1.87×).

### 7.2 Cluster Profiles

| Feature | Cluster 0 Mean | Cluster 1 Mean | Key Difference |
|---------|---------------|---------------|----------------|
| `order_frequency` | 2.96 | 1.43 | Cluster 0: frequent buyers |
| `avg_order_value` | $127.56 | $61.97 | Cluster 0: higher spend per order |
| `total_sales` | $326.37 | $85.46 | Cluster 0: much higher lifetime spend |
| `customer_return_rate` | 0.41 | 0.82 | Cluster 1: very high return rate |
| `purchase_recency_days` | 317 | 558 | Cluster 0: more recently active |
| `avg_basket_size` | 1.72 | 1.28 | Cluster 0: larger baskets |
| `customer_tenure_days` | 1,234 | 1,263 | Negligible difference |

**Cluster 0 — High-Activity, Moderate Return Rate:** Frequent buyers ($96.68 avg erosion) who purchase often, spend more, and return a minority of what they buy. High absolute erosion driven by volume and spend, not return propensity.

**Cluster 1 — Low-Activity, High Return Rate:** Predominantly single-order customers ($51.69 avg erosion) with very high return rates (median = 1.00 — they return everything they buy in this dataset). Lower absolute erosion per customer despite near-complete return behaviour.

### 7.3 Feature Importance (ANOVA F-Statistics)

| Feature | F-Statistic | η² (Effect Size) |
|---------|-------------|------------------|
| `order_frequency` | 11,549.10 | 0.491 |
| `total_sales` | 13,988.07 | 0.539 |
| `total_margin` | 13,033.80 | 0.521 |
| `customer_return_rate` | 7,488.41 | 0.385 |
| `avg_order_value` | 2,731.74 | 0.186 |
| `avg_basket_size` | 1,410.94 | 0.105 |
| `purchase_recency_days` | 621.85 | 0.049 |
| `customer_tenure_days` | 4.04 | 0.000 |

Order frequency dominates cluster separation (F = 11,549, η² = 0.491) — it alone explains 49.1% of between-cluster variance. Customer tenure is essentially uninformative (η² ≈ 0.000).

### 7.4 Hypothesis Test Results

| Test | Statistic | p-value | Decision |
|------|-----------|---------|----------|
| One-Way ANOVA | F = 1,794.23 | < 0.000001 | **Reject H₀₂** |
| Kruskal-Wallis | H = 968.13 | < 0.000001 | **Reject H₀₂** |

**Effect size η² = 0.1302 (13.0% of variance explained) — medium practical effect.**

Both parametric and non-parametric tests independently confirm H₀₂ is rejected. H₁₂ is supported: customer segments identified through clustering exhibit statistically significant differences in mean profit erosion from returns.

---

## 8. Statistical Evidence Summary

### 8.1 Concentration

- Gini coefficient = 0.409, bootstrap p < 0.0001
- Top 20% of returning customers → 47.6% of total profit erosion
- Highest-concentration feature: `purchase_recency_days` (Gini = 0.528)

### 8.2 Segmentation (H₀₂ Test)

- ANOVA F = 1,794.23, p < 0.0001
- Kruskal-Wallis H = 968.13, p < 0.0001
- η² = 0.130 (medium effect)
- Silhouette score at k = 2: 0.2921 (highest across k = 2 to 8)
- H₀₂ **Rejected** — H₁₂ **Supported**

### 8.3 Integrated Interpretation

Concentration identifies **WHO** to target: the top 20% of returning customers, best identified via `purchase_recency_days`. Segmentation reveals **HOW** to intervene differently: Cluster 0 (frequent buyers, $96.68 avg) requires personalised retention strategies; Cluster 1 (single-order, high return rate, $51.69 avg) requires policy-level guardrails. The optimal targeting zone is the top 20% of Cluster 0 — smallest group, highest per-customer impact.

---

## 9. External Validation (SSL — Pattern-Based)

### 9.1 Validation Objective

Since RQ2 uses unsupervised clustering, direct model transfer is not possible. Level 1 (Pattern Validation) was applied: do the same behavioral features that discriminate high-loss customers in TheLook also do so in the external SSL dataset (School Specialty LLC; 133,800 transactions, 13,616 accounts)?

**Success criterion:** Agreement rate ≥ 50%.

### 9.2 Pattern Validation Results

| Feature | TheLook | SSL | Status | Agreement |
|---------|---------|-----|--------|-----------|
| `avg_order_value` | Pass | Pass | Both Pass | **Yes** |
| `customer_return_rate` | Pass | Pass | Both Pass | **Yes** |
| `avg_basket_size` | Pass | Pass | Both Pass | **Yes** |
| `total_margin` | Pass | Pass | Both Pass | **Yes** |
| `avg_item_margin` | Pass | Pass | Both Pass | **Yes** |
| `order_frequency` | Fail | Fail | Both Fail | **Yes** |
| `total_sales` | Fail | Fail | Both Fail | **Yes** |
| `return_frequency` | Pass | Fail | Disagree | No |
| `total_items` | Pass | Fail | Disagree | No |
| `customer_tenure_days` | Fail | Pass | Disagree | No |
| `purchase_recency_days` | Fail | Pass | Disagree | No |
| `avg_item_price` | Fail | Pass | Disagree | No |

**Agreement: 7/12 features (58.3%) — validation threshold exceeded (≥ 50%).**

- **Both Pass (5):** `avg_order_value`, `customer_return_rate`, `avg_basket_size`, `total_margin`, `avg_item_margin` — discriminate high-loss accounts in both datasets. Use confidently for cross-domain targeting.
- **Both Fail (2):** `order_frequency`, `total_sales` — uninformative in both datasets. Agreement on uselessness still counts toward the threshold.
- **Disagree (5):** `return_frequency`, `total_items` pass in TheLook but fail in SSL (returns-only scope collapses their independent signal); `customer_tenure_days`, `purchase_recency_days`, `avg_item_price` pass in SSL (defined 2-year window) but fail in TheLook's longer time horizon.

---

## 10. Dashboard Elements

### KPI Cards

| Card | Value | Interpretation |
|------|-------|---------------|
| **Top 20% Share** | 47.6% | Top quintile of returners drives nearly half of all losses |
| **H₀₂ Result** | Rejected | Clusters differ significantly in mean profit erosion (p < 0.0001) |
| **Customer Segments** | 2 | K-Means optimal solution at k = 2 (silhouette peak) |
| **Gini Coefficient** | 0.409 | Moderate concentration — erosion is unequal but not extreme |

### Overview Tab (3-Panel Logic Chain)

- **STEP 1 — IS THE PROBLEM CONCENTRATED?** Lorenz curve shows the inequality bow below the equality diagonal.
- **STEP 2 — WHO ARE THE HIGH-RISK CUSTOMERS?** Cluster comparison shows Cluster 0 costs 1.80× more per customer.
- **STEP 3 — WHAT DRIVES THEM?** Feature concentration ranking shows `purchase_recency_days` as the top signal.

### Statistical Evidence (Overview)

- Left panel: Concentration analysis — Gini = 0.409, bootstrap p < 0.0001.
- Right panel: H₀₂ test — ANOVA F = 1,794.23, KW H = 968.13, η² = 0.130, Decision: **Rejected**.

### Concentration Tab

Pareto curve, Lorenz curve, feature concentration ranking (Gini by feature), and Gini vs Pareto cross-validation scatter confirming both concentration measures agree.

### Segmentation Tab

Cluster erosion comparison (mean vs median), clustering diagnostics (elbow + silhouette), feature importance (ANOVA F-statistic ranked), and interactive cluster explorer.

### Validation Tab

5-metric validation summary (Features Tested, Both Pass, Both Fail, Disagree, Agreement Rate) with horizontal bar chart and generalising vs domain-specific feature breakdown.

### Conclusion Tab

Full H₀₂ / H₁₂ hypothesis table, integrated WHO/HOW/Driver metrics, cluster erosion donut chart, strategic action plan (P1: Cluster 0 / P2: Top 20% by Recency / P3: Cluster 1), and RQ2 summary table.

---

## 11. Limitations

- The TheLook dataset is synthetic and may not fully capture behavioral complexity present in real-world e-commerce data. Findings should be validated against production customer data before operational deployment.
- Return processing costs are modeled using literature-based estimates rather than directly observed operational costs.
- Recovery or resale value of returned items is not incorporated, which may overstate net profit erosion.
- K-Means assumes spherical cluster geometry. The moderate silhouette score (0.2921) reflects a behavioral continuum rather than clearly separated populations — the two-cluster assignment is a useful operational approximation.
- External validation uses a returns-only SSL dataset, limiting interpretability of return-rate features whose denominators reflect return activity rather than total purchasing behavior.

---

## 12. Conclusion (RQ2)

**H₀₂ is rejected.** Customer segments identified through K-Means clustering exhibit statistically significant differences in mean profit erosion from returns (ANOVA F = 1,794.23, p < 0.0001; Kruskal-Wallis H = 968.13, p < 0.0001; η² = 0.130 — medium effect). H₁₂ is supported.

Concentration analysis provides the descriptive context: Gini = 0.409 (bootstrap p < 0.0001), with the top 20% of returning customers accounting for 47.6% of $816,500.94 in total profit erosion across 11,988 returners.

The two-cluster solution identifies:

- **Cluster 0** (n = 4,375, 36.5%): High-frequency buyers, mean erosion $96.68, 51.8% of total erosion. Primary driver: order frequency (F = 11,549, η² = 0.491).
- **Cluster 1** (n = 7,613, 63.5%): Single-order, high return rate customers, mean erosion $51.69, 48.2% of total erosion.

External pattern validation (SSL) shows 58.3% agreement (7/12 features) — exceeding the 50% cross-domain validity threshold. Five features pass in both datasets (`customer_return_rate`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`); two fail in both (`order_frequency`, `total_sales`). The core behavioral dimensions generalise across datasets.

**Strategic implication:** Focus Cluster 0 (frequent buyers) on personalised fit guidance and loyalty retention. Apply policy-level guardrails to Cluster 1 (high return rate). Use `purchase_recency_days` (Gini = 0.530) as the highest-precision alerting signal for both segments.

These findings extend RQ1's product-level descriptive analysis into a customer-level targeting framework, and directly motivate the predictive modeling approach in RQ3.

---

## 13. Traceability to User Stories

- **US06:** Return feature engineering and profit erosion computation (upstream data pipeline)
- **US07:** Descriptive aggregation and customer-level behavioral feature construction
- **RQ1:** Established statistically significant cross-category and cross-brand profit erosion differences (foundational finding)
- **RQ2:** Concentration and segmentation analysis of customer-level profit erosion
- **RQ3:** Extends RQ2 findings into a predictive classification framework

---

## 14. References

Anderson, E. T., Hansen, K., & Simester, D. (2009). The option value of returns: Theory and empirical evidence. *Marketing Science*, 28(3), 405–423.

Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., & Perona, I. (2013). An extensive comparative study of cluster validity indices. *Pattern Recognition*, 46(1), 243–256.

Cowell, F. A. (2011). *Measuring Inequality* (3rd ed.). Oxford University Press.

Dormann, C. F., et al. (2013). Collinearity: A review of methods to deal with it and a simulation study evaluating their performance. *Ecography*, 36(1), 27–46.

Guide, V. D. R., Jr., & Van Wassenhove, L. N. (2009). The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10–18.

Koch, R. (1997). *The 80/20 Principle: The Secret to Achieving More with Less*. Doubleday.

Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? *Journal of Marketing*, 73(3), 35–51.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

School Specialty, Inc. (2025). *SSL_Returns_df_yoy* [Unpublished internal dataset].

Steinley, D. (2006). K-means clustering: A half-century synthesis. *British Journal of Mathematical and Statistical Psychology*, 59(1), 1–34.

Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 1(21), 19–25.
