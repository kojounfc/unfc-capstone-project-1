# RQ2 Technical Documentation

**Capstone Project -- Master of Data Analytics**\
**Research Question 2 (RQ2)**

------------------------------------------------------------------------

## 1. Research Question

**RQ2:**\
*How concentrated is profit erosion among customers, and can distinct
customer segments be identified based on return behavior and economic
impact?*

RQ2 evaluates whether return-driven financial losses are
disproportionately concentrated among a subset of customers and whether
statistically stable and economically meaningful segments can be
identified.

------------------------------------------------------------------------

## 2. Analytical Objectives

1.  Quantify the degree of customer-level concentration in profit
    erosion.
2.  Identify stable behavioral segments using unsupervised learning.\
3.  Validate statistical distinctiveness and structural robustness of
    clusters.

------------------------------------------------------------------------

## 3. Data Scope and Unit of Analysis

-   **Unit of analysis:** Customer-level\
-   **Population:** Customers with at least one returned item\
-   **Source:** Aggregated outputs from US06 feature engineering\
-   **Aggregation level:** Customer-level total profit erosion

Each row represents one customer and includes engineered economic and
behavioral variables.

------------------------------------------------------------------------

## 4. Feature Engineering

### Economic Features

-   Total profit erosion\
-   Total sales\
-   Total margin\
-   Average erosion per returned item\
-   Loss per item

### Behavioral Features

-   Return frequency\
-   Total items purchased\
-   Average items per order\
-   Recency measures

Variance filtering and correlation screening were applied before
clustering.

------------------------------------------------------------------------

# 5. Concentration Analysis

## 5.1 Distributional Diagnostics

Customer-level profit erosion exhibited: - Right-skewed distribution\
- Heavy upper tail\
- High dispersion

This justified non-parametric concentration metrics.

------------------------------------------------------------------------

## 5.2 Gini Coefficient

**Gini ≈ 0.41**

Interpretation: - Moderate inequality in profit erosion distribution\
- Meaningful but not extreme concentration

------------------------------------------------------------------------

## 5.3 Pareto Concentration Table

  Customer Percentile   Cumulative Erosion Share
  --------------------- --------------------------
  Top 10%               \~30--35%
  Top 20%               \~47--50%
  Top 30%               \~60--65%

Nearly half of total profit erosion is driven by the top 20% of
customers.

------------------------------------------------------------------------

# 6. Clustering Methodology

## 6.1 Feature Screening

-   Variance threshold filtering\
-   Correlation removal (\|r\| \> 0.85)\
-   Bonferroni-adjusted univariate screening

------------------------------------------------------------------------

## 6.2 Feature Scaling

All features were standardized using `StandardScaler()`.

------------------------------------------------------------------------

## 6.3 Optimal k Selection

Model selection evaluated: - Inertia (Elbow)\
- Silhouette\
- Calinski--Harabasz\
- Davies--Bouldin

**Selected k = 2**

------------------------------------------------------------------------

# 7. Clustering Results

## 7.1 Cluster Size Distribution

  Cluster     \% of Customers
  ----------- -----------------
  Cluster 0   \~55--60%
  Cluster 1   \~40--45%

------------------------------------------------------------------------

## 7.2 Clustering Quality Metrics

  Metric               Value     Interpretation
  -------------------- --------- ------------------------
  Silhouette           \~0.28    Moderate separation
  Calinski--Harabasz   \~1900+   Strong dispersion
  Davies--Bouldin      \~1.45    Acceptable compactness

------------------------------------------------------------------------

## 7.3 Between-Cluster Statistical Testing

Post-hoc tests confirmed statistically significant differences (p \<
0.001) in: - Total profit erosion\
- Return frequency\
- Total sales

------------------------------------------------------------------------

# 8. SSL Validation (Statistical Significance & Stability Layer)

## 8.1 Bootstrap Stability Testing

**Bootstrap ARI ≈ 0.99**

Cluster assignments remain highly stable across resampled datasets.

------------------------------------------------------------------------

## 8.2 Variance and Integrity Safeguards

-   Zero-variance removal\
-   Deterministic scaling\
-   Fixed random seed\
-   Data type validation

------------------------------------------------------------------------

# 9. Outlier Sensitivity Test

Removing the top 1% of erosion customers resulted in only minor Gini
reduction, confirming structural concentration.

------------------------------------------------------------------------

# 10. Conclusion (RQ2)

RQ2 provides statistically rigorous evidence that: - Profit erosion is
moderately concentrated.\
- A high-risk customer segment exists.\
- Segmentation is statistically stable and robust.\
- Concentration persists after outlier adjustment.

These findings support targeted return-risk mitigation strategies.
