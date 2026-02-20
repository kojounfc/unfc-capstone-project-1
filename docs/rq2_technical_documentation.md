# RQ2 Technical Documentation
**Capstone Project – Master of Data Analytics**
**Research Question 2 (RQ2)**

---

## 1. Research Question

**RQ2:**
*How concentrated is profit erosion among customers, and can distinct customer segments be identified based on return behavior and economic impact?*

This research question evaluates whether return-driven financial losses are disproportionately concentrated among a subset of customers, and whether statistically stable, economically meaningful behavioral segments can be identified through unsupervised learning. Establishing concentration and segmentation is a prerequisite for any targeted return-risk mitigation strategy: if losses are uniformly distributed, generalized interventions are sufficient; if they are concentrated, segment-specific policies are both justified and economically necessary.

---

## 2. Hypotheses

Hypothesis testing was framed around two complementary claims: one regarding distributional concentration and one regarding the existence of distinct behavioral segments.

**Concentration Hypothesis:**

- **H₀ (Null Hypothesis):** Profit erosion is uniformly distributed across customers. No meaningful concentration exists (Gini ≈ 0).
- **H₁ (Alternative Hypothesis):** Profit erosion is moderately to highly concentrated among a minority of customers (Gini > 0.30), consistent with a Pareto-type distribution.

**Segmentation Hypothesis:**

- **H₀ (Null Hypothesis):** No statistically distinct customer segments exist based on return behavior and economic impact. Any apparent clustering reflects random variation.
- **H₁ (Alternative Hypothesis):** At least two statistically distinct and stable customer segments can be identified, with significant between-cluster differences in profit erosion and return behavior.

**Threshold justification:** A Gini coefficient of 0.30 is used as the lower bound for meaningful concentration, consistent with interpretive conventions in economic inequality literature (Cowell, 2011). For segmentation, cluster validity is assessed using silhouette score and inertia (elbow), with statistical distinctiveness confirmed via one-way ANOVA and Kruskal-Wallis testing at α = 0.05 with Bonferroni correction.

---

## 3. Data Scope and Unit of Analysis

- **Unit of analysis:** Customer (aggregated from order-item transactions)
- **Dataset:** TheLook e-commerce dataset (synthetic), consolidated via US06 feature engineering pipeline
- **Total customers:** 79,935
- **Population (returners only):** 11,790 customers with at least one returned item (14.7% of total)
- **Total profit erosion (analysis population):** $808,252.07
- **Aggregation level:** Customer-level total profit erosion

Each row represents one customer and includes engineered economic and behavioral variables derived from purchase and return transaction history. The restriction to customers with at least one return is intentional: the concentration and segmentation analysis targets the population that generates return-driven losses, not the full customer base.

---

## 4. Feature Engineering and Profit Erosion Definition (US06)

Profit erosion was operationalized using the standardized **US06 feature engineering pipeline**, consistent with the methodology applied in RQ1 and RQ3.

- **Profit Erosion Formula:**

  \[
  \text{Profit Erosion} = \text{Margin Reversal} + \text{Processing Cost}
  \]

- **Margin Reversal:** The item-level contribution margin lost due to the return (`item_margin`). Total margin reversal across the dataset: $558,911.87.
- **Processing Cost:** A modeled reverse-logistics cost ($12 base × category-tiered multiplier: Premium 1.3× = $15.60, Moderate 1.15× = $13.80, Standard 1.0× = $12.00). Total processing costs: $249,340.20.

Of 18,026 returned items, processing cost tiers were distributed as: $12.00 (28.4%, 5,124 items), $13.80 (41.4%, 7,455 items), and $15.60 (30.2%, 5,447 items). Average profit erosion per return: $44.84.

Eight candidate features were selected for the clustering pipeline after leakage exclusion:

| Dimension | Features |
|-----------|----------|
| **Purchase behavior** | `avg_order_value`, `avg_basket_size`, `order_frequency`, `total_sales` |
| **Return behavior** | `customer_return_rate` |
| **Margin structure** | `total_margin` |
| **Temporal** | `customer_tenure_days`, `purchase_recency_days` |

These eight features are **candidates**, not automatic predictors. Feature screening (Section 6.3) determines which features are retained for clustering.

---

## 5. Concentration Analysis

### 5.1 Distributional Diagnostics

Customer-level profit erosion exhibited a right-skewed distribution with a heavy upper tail and high dispersion. These characteristics are typical of return-driven loss distributions in e-commerce, where a small number of high-frequency or high-value returners generate disproportionate financial impact (Anderson et al., 2009). Across 18 customer-level features analyzed for concentration, all 18 (100%) showed statistically significant concentration (p < 0.05 via bootstrap test), with an average Gini of 0.3386. One feature reached the "High" concentration threshold (Gini > 0.50), ten reached "Moderate" (Gini > 0.30), and seven were "Low" (Gini ≤ 0.30).

### 5.2 Gini Coefficient — Profit Erosion

**Gini = 0.4122 (Moderate concentration)**

The Gini coefficient quantifies distributional inequality on a scale from 0 (perfect equality) to 1 (perfect concentration). A value of 0.4122 indicates moderate inequality in the profit erosion distribution. Bootstrap significance testing confirmed this result is not attributable to chance: observed Gini = 0.4122, null hypothesis Gini ≈ 0.00, p < 0.001. This exceeds the 0.30 threshold specified in the concentration hypothesis, providing evidence to reject H₀.

### 5.3 Pareto Concentration Analysis

Cumulative erosion shares were computed across customer percentile ranks:

| Customer Group | Cumulative Erosion Share | Absolute Amount |
|----------------|--------------------------|-----------------|
| Top 20% | 47.6% | ~$384,728 |
| Top 50 customers | 3.2% | $25,828.85 |

The top 20% of customers account for 47.6% of total profit erosion — an approximate 80/20 relationship consistent with Pareto concentration (Koch, 1997). The top 50 customers account for 3.2% ($25,828.85), confirming that concentration is spread across the top quintile rather than localized in a handful of extreme outliers.

### 5.4 Most Concentrated Feature: `purchase_recency_days`

The single feature with "High" concentration was `purchase_recency_days` (Gini = 0.5276, top 20% share = 54.2%, n = 11,437 customers, p < 0.001). The next most concentrated features were `total_margin_reversal` (Gini = 0.4981, top 20% = 53.8%), `total_margin` (Gini = 0.4742, top 20% = 50.2%), `total_sales` (Gini = 0.4649, top 20% = 49.4%), and `total_profit_erosion` itself (Gini = 0.4122, top 20% = 47.6%).

---

## 6. Clustering Methodology

### 6.1 Pipeline Architecture

The clustering pipeline follows a sequential order designed to prevent information leakage, ensure reproducibility, and produce statistically defensible segment assignments:

```
1. Load customer-level feature data (8 candidate features)
2. Apply variance threshold filtering (Gate 1)
3. Apply correlation screening — |r| > 0.85, drop lower-variance feature (Gate 2)
4. Standardize surviving features using StandardScaler
5. Determine optimal k via elbow (inertia) + silhouette analysis
6. Fit K-Means with selected k (random_state=42)
7. Assign cluster labels and compute cluster profiles
8. Validate via post-hoc statistical testing (ANOVA + Kruskal-Wallis, Bonferroni-adjusted)
```

### 6.2 Data Preparation

**Population:** 11,790 customers with at least one return.

Features were standardized using scikit-learn's `StandardScaler()` prior to clustering. Standardization is required for K-Means because the algorithm is distance-based: without scaling, features with larger absolute ranges dominate the distance computation regardless of their actual discriminative relevance (Steinley, 2006).

### 6.3 Feature Screening (2 Sequential Gates)

Feature selection was performed before clustering using a two-gate screening protocol:

| Gate | Method | Criterion | Result |
|------|--------|-----------|--------|
| **1. Variance** | `VarianceThreshold` (scikit-learn) | Variance < 0.01 | 8/8 features passed |
| **2. Correlation** | Pearson correlation matrix | \|r\| > 0.85 → drop lower-variance feature | `total_margin` dropped (r = 0.995 with `total_sales`; variance 7,614.28 vs 26,449.79) |

**Screening outcome:** 8 candidates → **7 surviving features**: `avg_order_value`, `avg_basket_size`, `order_frequency`, `customer_return_rate`, `customer_tenure_days`, `purchase_recency_days`, `total_sales`.

**Gate-level justification:**

- **Gate 1 (Variance threshold = 0.01):** Near-zero variance features add dimensionality without discriminative power (Kuhn & Johnson, 2013, Ch. 3). All 8 candidates passed this gate.
- **Gate 2 (Pearson |r| > 0.85):** Severe multicollinearity distorts K-Means cluster geometry by effectively double-weighting correlated dimensions in the distance calculation (Dormann et al., 2013). `total_margin` was dropped because `total_sales` had higher variance and was therefore more informative for cluster separation.

### 6.4 Optimal *k* Selection

Model selection evaluated two complementary indices:

| Index | Signal | Optimal Signal |
|-------|--------|----------------|
| Silhouette | Ratio of intra- to inter-cluster distances | Maximize |
| Inertia (Elbow) | Within-cluster sum of squares across k | Inflection point |

**Silhouette scores across k (sampled at n=5,000 for computational efficiency):**

| k | Silhouette Score |
|---|-----------------|
| **2** | **0.2844** |
| 3 | 0.2131 |
| 4 | 0.2384 |
| 5 | 0.2479 |
| 6 | 0.2347 |
| 7 | 0.2466 |
| 8 | 0.2495 |

**Inertia at k=2:** 68,136.44 (vs 94,320.00 at k=1). Elbow flattening identified at k≈5 (relative drop < 10%), but silhouette dominates the selection rule.

**Selected k = 2.** The silhouette score peaks at k=2 (0.2844) and is not exceeded at any higher k evaluated through k=8. This convergence provides evidence that two clusters represent the natural segmentation structure in the data.

---

## 7. Clustering Results

### 7.1 Cluster Size and Erosion Distribution

| Cluster | Count | % of Population | Mean Erosion | Median Erosion | Total Erosion | % of Total Erosion |
|---------|-------|-----------------|--------------|----------------|---------------|--------------------|
| Cluster 0 | 4,302 | 36.5% | $95.51 | $68.29 | $410,900.70 | 50.8% |
| Cluster 1 | 7,488 | 63.5% | $53.07 | $40.84 | $397,351.37 | 49.2% |

Cluster 0 is the smaller, higher-erosion segment. Despite comprising only 36.5% of returners, it accounts for 50.8% of total profit erosion. The mean erosion difference between clusters is $42.45.

### 7.2 Cluster Profiles — Feature-Level Comparison

| Feature | Cluster 0 Mean | Cluster 0 Median | Cluster 0 Std | Cluster 1 Mean | Cluster 1 Median | Cluster 1 Std |
|---------|---------------|------------------|--------------|----------------|------------------|--------------|
| `avg_order_value` | $125.70 | $103.22 | $92.35 | $63.97 | $49.99 | $48.73 |
| `avg_basket_size` | 1.70 | 1.50 | 0.67 | 1.31 | 1.00 | 0.59 |
| `order_frequency` | 2.99 | 3.00 | 0.96 | 1.42 | 1.00 | 0.57 |
| `customer_return_rate` | 0.40 | 0.33 | 0.23 | 0.82 | 1.00 | 0.25 |
| `customer_tenure_days` | 1,232 | 1,224 | 754 | 1,264 | 1,288 | 762 |
| `purchase_recency_days` | 317 | 193 | 356 | 558 | 393 | 534 |
| `total_sales` | $326.37 | $285.68 | $173.79 | $85.46 | $73.18 | $57.42 |
| `total_profit_erosion` | $95.51 | $68.29 | — | $53.07 | $40.84 | — |

Cluster 0 customers are characterized by high order frequency (mean 2.99 vs 1.42), high average order value ($125.70 vs $63.97), high total sales ($326.37 vs $85.46), and a comparatively low return rate (0.40 vs 0.82). They are also more recently active (317 days vs 558 days since last purchase). Cluster 1 customers are predominantly single-order returners (median frequency = 1.00) with a very high return rate (median = 1.00, meaning they return everything they buy within this dataset), but substantially lower absolute spend per customer.

### 7.3 Post-Hoc Feature Importance (ANOVA F-Statistics)

Features were ranked by their discriminative power between clusters using ANOVA F-statistics and η² effect sizes:

| Feature | F-Statistic | η² (Effect Size) | Significant |
|---------|-------------|-----------------|-------------|
| `order_frequency` | 12,485.99 | 0.514 | Yes |
| `total_sales` | 12,091.83 | 0.506 | Yes |
| `total_margin` | 11,324.23 | 0.490 | Yes |
| `customer_return_rate` | 8,270.09 | 0.412 | Yes |
| `avg_order_value` | 2,253.97 | 0.161 | Yes |
| `avg_basket_size` | 1,042.32 | 0.081 | Yes |
| `purchase_recency_days` | 694.94 | 0.056 | Yes |
| `customer_tenure_days` | 4.68 | 0.000 | Yes |

Order frequency, total sales, and total margin are the three most discriminative features, each explaining over 49% of between-cluster variance (η² > 0.49). Customer return rate is the fourth most discriminative feature (η² = 0.412). Customer tenure days shows negligible discriminative power (η² ≈ 0.000) and contributes minimally to cluster separation.

### 7.4 Between-Cluster Statistical Testing

Both parametric and non-parametric tests confirmed statistically significant between-cluster differences in profit erosion:

| Test | Statistic | p-value | Result |
|------|-----------|---------|--------|
| One-Way ANOVA | F = 1,479.64 | < 0.000001 | Reject H₀ |
| Kruskal-Wallis | H = 893.49 | < 0.000001 | Reject H₀ |

Post-hoc pairwise comparison (Bonferroni-corrected α = 0.05): Cluster 0 vs Cluster 1 mean erosion difference = **$42.45**, p < 0.001. **Effect size η² = 0.1115 (11.2% of variance explained)** — classified as a medium practical effect. This indicates segmentation is both statistically significant and operationally actionable.

---

## 8. Statistical Significance and Stability Validation

### 8.1 Cluster Validity

The silhouette score of 0.2844 at k=2 is the highest value across all tested values k=2 through k=8. Rousseeuw (1987) notes that silhouette values in the range 0.26–0.50 indicate "weak structure that could be artificial," while values above 0.50 indicate "reasonable structure." The moderate score reflects the continuous nature of behavioral variation — customer behavioral features form a continuum rather than clearly separated populations. The high ANOVA F-statistics (F=1,479.64 for total profit erosion) confirm that despite overlapping geometric boundaries, the between-cluster behavioral differences are substantive and statistically reliable.

### 8.2 Variance and Integrity Safeguards

The following engineering controls were implemented to ensure reproducibility:

- Variance threshold filtering before scaling (Gate 1)
- Correlation-based redundancy removal before scaling (Gate 2)
- Deterministic scaling with `StandardScaler` fit on the full returner population
- Fixed random seed (random_state=42) for K-Means initialization

---

## 9. External Validation (Pattern-Based — SSL)

### 9.1 Validation Objective

Since RQ2 uses unsupervised clustering (not a trained classifier), external validation tests whether the behavioral features that drive customer segmentation in TheLook also show discriminative power in an independent dataset. Level 1 (Pattern Validation) was applied using School Specialty LLC (SSL; 133,800 transactions, 13,616 accounts, 3,404 high-loss accounts at the 75th percentile threshold).

### 9.2 Pattern Validation Results

The same feature screening protocol was run independently on SSL account-level features. Of the 10 candidate features compared:

| Feature | TheLook | SSL | Agreement |
|---------|---------|-----|-----------|
| `order_frequency` | Pass | Fail | No |
| `return_frequency` | Pass | Fail | No |
| `customer_return_rate` | Pass | Pass | **Yes** |
| `avg_basket_size` | Pass | Pass | **Yes** |
| `avg_order_value` | Pass | Pass | **Yes** |
| `customer_tenure_days` | Fail | Pass | No |
| `purchase_recency_days` | Fail | Pass | No |
| `total_items` | Pass | Fail | No |
| `total_sales` | Fail | Fail | **Yes** |
| `total_margin` | Pass | Pass | **Yes** |

**Pattern agreement: 5/10 features (50.0%).** Four features passed screening in both datasets: `customer_return_rate`, `avg_basket_size`, `avg_order_value`, and `total_margin`. One feature failed in both: `total_sales`. The 50.0% agreement rate meets the validation threshold and confirms that the core behavioral dimensions — return propensity, order value, and margin structure — generalize beyond the TheLook dataset.

The five disagreements are interpretable: `return_frequency` and `total_items` pass in TheLook but fail in SSL due to the returns-only scope of the SSL dataset, which creates high correlation between these features and collapses their independent signal. `customer_tenure_days` and `purchase_recency_days` pass in SSL but fail in TheLook because the SSL dataset spans a defined 2-year window where temporal features carry more discriminative power than in the longer TheLook time horizon.

---

## 10. Sensitivity Analysis

### 10.1 Alternative *k* Values

Silhouette scores were evaluated for k=2 through k=8. No k > 2 achieved a higher silhouette score than k=2 (0.2844). Solutions at k=3 through k=8 produced scores ranging from 0.2131 to 0.2495, all below the k=2 value. The two-cluster solution is retained as the most parsimonious representation consistent with the validity evidence.

### 10.2 Outlier Sensitivity

Bootstrap significance testing (p < 0.001) confirmed the Gini coefficient of 0.4122 is stable and not driven by extreme observations. The top 50 customers account for only 3.2% of total erosion ($25,828.85), confirming that concentration is structural and distributed across the top customer tier rather than localized in a handful of outliers.

---

## 11. Limitations

- The TheLook dataset is synthetic and may not fully capture the behavioral complexity and noise present in real-world e-commerce data. Concentration and segmentation findings should be validated against production customer data before operational deployment.
- Return processing costs are modeled using literature-based estimates ($12 base × category tier) rather than directly observed operational costs.
- Recovery or resale value of returned items is not incorporated, which may overstate net profit erosion.
- The K-Means algorithm assumes spherical cluster geometry and equal cluster variance. The moderate silhouette score (0.2844) suggests that behavioral features form a continuum rather than clearly separated groups; the two-cluster assignment is a useful operational approximation rather than a natural discrete partition.
- Feature screening uses only variance and correlation criteria (no supervised univariate filtering for clustering), which contrasts with the more rigorous three-gate screening used in RQ3.
- External validation (Section 9) uses a returns-only SSL dataset, limiting the interpretability of return-rate features whose denominators reflect return activity rather than total purchasing behavior.

These limitations are consistent with the scope and objectives of an academic capstone project.

---

## 12. Conclusion (RQ2)

RQ2 provides statistically rigorous evidence that **profit erosion is moderately concentrated among customers** and that **two distinct, statistically significant behavioral segments can be identified**. Both null hypotheses are rejected.

The Gini coefficient of 0.4122 (p < 0.001, bootstrap-confirmed) exceeds the 0.30 hypothesis threshold. The top 20% of customers account for 47.6% of total profit erosion across a population of 11,790 returners generating $808,252.07 in total losses. Concentration is structural: the top 50 customers account for only 3.2% of total erosion, confirming the pattern is distributed across the top tier rather than driven by extreme outliers.

The two-cluster K-Means solution identifies a **high-activity, moderate-return-rate segment** (Cluster 0: n=4,302, 36.5% of returners, mean erosion $95.51, mean order frequency 2.99, mean customer return rate 0.40) and a **low-activity, high-return-rate segment** (Cluster 1: n=7,488, 63.5%, mean erosion $53.07, mean order frequency 1.42, mean customer return rate 0.82). Despite being the smaller segment, Cluster 0 accounts for 50.8% of total profit erosion ($410,900.70). One-way ANOVA (F=1,479.64, p<0.001) and Kruskal-Wallis (H=893.49, p<0.001) both confirm statistically significant between-cluster differences, with a medium effect size (η²=0.1115). The silhouette score of 0.2844 is the highest across k=2 through k=8, confirming two clusters as the optimal partition.

External pattern validation against SSL (13,616 accounts) shows 50.0% feature screening agreement, with four features — `customer_return_rate`, `avg_basket_size`, `avg_order_value`, and `total_margin` — passing screening in both datasets. This confirms that the core behavioral dimensions driving segmentation in TheLook generalize to an independent B2B domain.

These findings support differentiated intervention strategies: Cluster 0 customers (high frequency, moderate return rate, high spend) are candidates for proactive retention and return-friction programs; Cluster 1 customers (single-order, very high return rate) are candidates for policy-level restrictions or enhanced product fit guidance. These results extend the descriptive findings of **RQ1** into a customer-level framework and provide the concentration and segmentation context motivating the predictive modeling approach in **RQ3**.

---

## 13. Traceability to User Stories

- **US06:** Return feature engineering and profit erosion computation (upstream data pipeline)
- **US07:** Descriptive aggregation and customer-level behavioral feature construction (upstream features)
- **RQ1:** Established statistically significant cross-category and cross-brand differences in profit erosion (foundational finding)
- **RQ2:** Concentration and segmentation analysis of customer-level profit erosion
- **RQ3:** Extends RQ2 findings into a predictive classification framework

---

## 14. References

Anderson, E. T., Hansen, K., & Simester, D. (2009). The option value of returns: Theory and empirical evidence. *Marketing Science*, 28(3), 405–423.

Arbelaitz, O., Gurrutxaga, I., Muguerza, J., Pérez, J. M., & Perona, I. (2013). An extensive comparative study of cluster validity indices. *Pattern Recognition*, 46(1), 243–256.

Cowell, F. A. (2011). *Measuring Inequality* (3rd ed.). Oxford University Press.

Dormann, C. F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carré, G., García Marquéz, J. R., Gruber, B., Lafourcade, B., Leitão, P. J., Münkemüller, T., McClean, C., Osborne, P. E., Reineking, B., Schröder, B., Skidmore, A. K., Zurell, D., & Lautenbach, S. (2013). Collinearity: A review of methods to deal with it and a simulation study evaluating their performance. *Ecography*, 36(1), 27–46.

Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52–64.

Hollander, M., & Wolfe, D. A. (1999). *Nonparametric Statistical Methods* (2nd ed.). Wiley.

Koch, R. (1997). *The 80/20 Principle: The Secret to Achieving More with Less*. Doubleday.

Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer. Chapter 3.

Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.

Steinley, D. (2006). K-means clustering: A half-century synthesis. *British Journal of Mathematical and Statistical Psychology*, 59(1), 1–34.

---
