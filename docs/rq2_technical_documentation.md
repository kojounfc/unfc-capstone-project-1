# RQ2 Technical Documentation

**Capstone Project – Master of Data Analytics**  
**Research Question 2: Customer Concentration and Behavioral Segmentation of Profit Erosion**

---

## Executive Summary

This document provides comprehensive technical documentation for Research Question 2 (RQ2), which investigates the concentration and customer-level patterns of profit erosion in e-commerce returns. The analysis combines concentration metrics (Pareto, Lorenz, Gini) with unsupervised customer segmentation (K-Means clustering) to identify actionable customer segments with differential return behavior and economic impact.

**Key Findings:**
- Profit erosion is moderately concentrated (Gini = 0.41)
- Top 20% of customers account for 47.6% of total profit erosion
- Two statistically distinct customer segments identified with high stability (Bootstrap ARI = 0.99)
- High-erosion customers are high-value but risky; low-erosion customers exhibit sustainable purchasing patterns

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Methodology Overview](#2-methodology-overview)
3. [Data Preparation](#3-data-preparation)
4. [Phase 1: Concentration Analysis](#4-phase-1-concentration-analysis)
5. [Phase 2: Customer Segmentation](#5-phase-2-customer-segmentation)
6. [Statistical Validation](#6-statistical-validation)
7. [Results and Interpretation](#7-results-and-interpretation)
8. [Implementation Details](#8-implementation-details)
9. [Reproducibility](#9-reproducibility)
10. [Business Recommendations](#10-business-recommendations)
11. [Limitations](#11-limitations)
12. [References](#12-references)

---

## 1. Research Question

### 1.1 Primary Research Question

**RQ2:** *To what extent is profit erosion concentrated among a small subset of customers, and can customers be meaningfully segmented based on behavioral and erosion characteristics?*

This research question examines:
1. **Concentration:** Whether return-driven profit erosion is unevenly distributed across the customer base
2. **Segmentation:** Whether unsupervised learning can identify behaviorally distinct customer segments with differential economic impact

### 1.2 Conceptual Hypotheses

Although RQ2 is exploratory and descriptive, conceptual hypotheses maintain alignment with the approved capstone proposal:

- **H₀₂ (Null Hypothesis):** Customer segments identified through unsupervised clustering do not differ meaningfully in profit erosion intensity.

- **H₁₂ (Alternative Hypothesis):** Customer segments identified through unsupervised clustering exhibit materially different profit erosion profiles.

**Methodological Note:** The analysis combines exploratory methods (clustering, concentration metrics) with statistical validation (significance testing, effect size estimation, stability analysis) to provide both descriptive insights and empirical evidence of segment differentiation.

---

## 2. Methodology Overview

### 2.1 Two-Phase Analytical Approach

**Phase 1: Concentration Analysis (Descriptive)**
- Pareto analysis to identify top contributors
- Lorenz curve visualization of inequality
- Gini coefficient quantification
- Bootstrap significance testing vs. uniform distribution
- Comparative concentration analysis (erosion vs. baseline sales)

**Phase 2: Customer Segmentation (Unsupervised ML)**
- Feature engineering (behavioral + value metrics)
- Optimal K selection (silhouette analysis)
- K-Means clustering
- Cluster quality validation (silhouette, Calinski-Harabasz, Davies-Bouldin)
- Bootstrap stability testing (Adjusted Rand Index)
- Statistical significance testing (Kruskal-Wallis/ANOVA)
- Effect size quantification (epsilon-squared/eta-squared)
- Cluster profiling and interpretation

### 2.2 Analytical Philosophy

This analysis is **exploratory with statistical validation:**
- **Primary goal:** Generate actionable customer insights through data-driven segmentation
- **Statistical rigor:** Validate that discovered segments represent meaningful, stable, and statistically significant differences
- **Business focus:** Translate statistical findings into operational recommendations

---

## 3. Data Preparation

### 3.1 Data Sources

**Primary Input:**
- File: `data/processed/returns_eda_v1.parquet`
- Grain: Item-level transaction records
- Source: Processed from raw CSV files via `src.data_processing`

**Derived Datasets:**
1. **Customer Behavioral Features** (`customer_behavior`)
   - Created by: `engineer_customer_behavioral_features()`
   - Grain: 1 row per customer
   - Features: RFM-style metrics (recency, frequency, monetary value)

2. **Customer Profit Erosion** (`customer_erosion`)
   - Created by: `build_customer_erosion()`
   - Grain: 1 row per customer (with returns only)
   - Features: Aggregated profit erosion metrics

3. **Customer Segmentation Table** (`customer_segmentation`)
   - Created by: `build_customer_segmentation_table()`
   - Grain: 1 row per customer
   - Features: Behavioral + erosion metrics merged

### 3.2 Data Quality

**Dataset Characteristics:**
- Total customers analyzed: 11,790
- Customers with returns: 11,790 (100% of analysis set)
- Total profit erosion: $1,622,495.20
- Date range: Full dataset history

**Quality Checks:**
- ✅ No duplicate user IDs
- ✅ No infinite values in clustering features
- ✅ Missing values handled via median imputation (when present)
- ✅ All numeric features standardized before clustering

### 3.3 Profit Erosion Construction

Profit erosion is calculated at the item level and aggregated to customers using the standardized pipeline from `src.feature_engineering`:

**Item-Level Formula:**
```
Profit Erosion = Margin Reversal + Return Processing Cost
```

Where:
- **Margin Reversal** = `item_margin` (sale_price - cost)
- **Return Processing Cost** = Fixed operational cost per return

**Processing Tiers (Category-Based):**
The pipeline applies category-specific processing costs:
- **Low complexity** (e.g., accessories): $10 per return
- **Medium complexity** (e.g., apparel): $15 per return  
- **High complexity** (e.g., electronics): $25 per return

**Customer-Level Aggregation:**
```python
customer_erosion = aggregate_profit_erosion_by_customer(returned_items)
```

This produces:
- `total_profit_erosion`: Sum of item-level erosion
- `total_margin_reversal`: Sum of margin reversals
- `total_processing_cost`: Sum of processing costs
- `returned_items`: Count of items returned

**Pipeline Consistency:**
- Same profit erosion logic used in RQ1 (product-level analysis)
- Ensures cross-RQ metric consistency
- Maintains CI-safe execution with no manual calculations

---

## 4. Phase 1: Concentration Analysis

### 4.1 Pareto Analysis

**Purpose:** Identify the top customers contributing disproportionately to profit erosion.

**Implementation:**
```python
from src.rq2_concentration import compute_pareto_table

pareto_table = compute_pareto_table(
    df=customer_erosion,
    value_col='total_profit_erosion',
    id_col='user_id'
)
```

**Output Fields:**
- `user_id`: Customer identifier
- `total_profit_erosion`: Customer's total erosion value
- `rank`: Customer rank (1 = highest erosion)
- `customer_share`: Cumulative % of customers up to this rank
- `cum_value`: Cumulative profit erosion
- `value_share`: Cumulative % of total profit erosion
- `concentration_category`: "Vital Few" (top 20%) or "Useful Many"

**Key Metrics Extracted:**
- Top 20% customers: 47.6% of total erosion
- Top 10% customers: 31.2% of total erosion
- Top 50% customers: 83.4% of total erosion

**Interpretation:**
The Pareto distribution demonstrates moderate-to-high concentration. Nearly half of all profit erosion comes from just one-fifth of customers, indicating clear opportunities for targeted intervention.

**Output File:** `data/processed/rq2/pareto_table.csv`

---

### 4.2 Lorenz Curve

**Purpose:** Visualize the cumulative distribution of profit erosion across customers.

**Implementation:**
```python
from src.rq2_concentration import lorenz_curve_points

lorenz_df = lorenz_curve_points(
    df=customer_erosion,
    value_col='total_profit_erosion'
)
```

**Output:**
- `population_share`: Cumulative proportion of customers (0 to 1)
- `value_share`: Cumulative proportion of profit erosion (0 to 1)

**Visualization Elements:**
- **Line of perfect equality:** 45° diagonal (y = x), representing uniform distribution
- **Actual Lorenz curve:** Lies below equality line, quantifying concentration
- **Area between curves:** Used to calculate Gini coefficient

**Interpretation:**
The Lorenz curve shows substantial deviation from equality:
- Bottom 50% of customers contribute only ~16% of erosion
- Top 20% account for nearly half of all erosion
- Curve steepens sharply in upper tail, indicating extreme concentration among top customers

**Output File:** `figures/rq2/lorenz_curve.png`

---

### 4.3 Gini Coefficient

**Purpose:** Quantify the degree of concentration using a single summary metric.

**Implementation:**
```python
from src.rq2_concentration import gini_coefficient

gini = gini_coefficient(
    df=customer_erosion,
    value_col='total_profit_erosion'
)
```

**Formula:**
The Gini coefficient is calculated using the standard formula:
```
G = (Σ (2i - n - 1) * x_i) / (n * Σ x_i)
```

Where:
- `x_i` = profit erosion for customer i (sorted ascending)
- `n` = number of customers
- `i` = rank index

**Result:** Gini = 0.41

**Interpretation Scale:**
- 0.0 = Perfect equality (all customers contribute equally)
- 0.4-0.5 = Moderate concentration (**observed value**)
- 0.7+ = High concentration
- 1.0 = Perfect inequality (one customer causes all erosion)

**Contextual Interpretation:**
A Gini of 0.41 indicates **moderate concentration**, comparable to:
- Income inequality in many developed nations
- Customer concentration in B2B contexts
- Revenue distribution across customer bases

This level justifies segmentation strategies while recognizing that erosion is not dominated by a tiny minority.

---

### 4.4 Statistical Significance Testing

**Purpose:** Test whether observed concentration exceeds random variation.

**Implementation:**
```python
from src.rq2_concentration import bootstrap_gini_p_value

bootstrap_result = bootstrap_gini_p_value(
    df=customer_erosion,
    value_col='total_profit_erosion',
    n_bootstrap=1000,
    random_state=42
)
```

**Null Hypothesis:** Profit erosion is uniformly distributed (equal share per customer).

**Method:**
1. Create null distribution: Divide total erosion equally among all customers
2. Bootstrap resample 1,000 times with replacement
3. Calculate Gini for each bootstrap sample
4. Compare observed Gini to null distribution

**Results:**
- Observed Gini: 0.41
- Null mean Gini: 0.00 (by definition, uniform distribution)
- p-value: < 0.001

**Interpretation:**
The observed concentration is **statistically significant** (p < 0.001). The probability of observing a Gini coefficient this high under random uniform allocation is effectively zero, confirming that concentration is a structural feature of the data, not random noise.

---

### 4.5 Comparative Concentration Analysis

**Purpose:** Compare profit erosion concentration against baseline business metrics.

**Implementation:**
```python
from src.rq2_concentration import concentration_comparison

comparison = concentration_comparison(
    df=customer_segmentation,
    erosion_col='total_profit_erosion',
    baseline_col='total_sales'
)
```

**Results:**
- Gini (Profit Erosion): 0.41
- Gini (Total Sales): 0.28

**Interpretation:**
Profit erosion is **more concentrated** than general sales activity:
- Sales are relatively evenly distributed across customers (Gini = 0.28)
- Erosion concentration (Gini = 0.41) exceeds sales concentration by 46%
- This suggests returns are not simply proportional to purchase activity
- Certain customers disproportionately generate erosion relative to their sales contribution

**Business Implication:**
High-sales customers do not automatically equal high-erosion customers. This justifies behavioral segmentation beyond simple sales volume stratification.

---

## 5. Phase 2: Customer Segmentation

### 5.1 Feature Engineering

**Objective:** Create behavioral and value-based features for clustering that **exclude outcome leakage**.

**Critical Rule: No Outcome Leakage**
Clustering features must **NOT** include:
- `total_profit_erosion` (this is the outcome we're trying to segment)
- Any derivatives of profit erosion (e.g., `erosion_per_return`, `high_erosion_flag`)
- Direct return processing costs

**Feature Selection Implementation:**
```python
from src.rq2_segmentation import select_numeric_features

X_df, feature_cols = select_numeric_features(
    customer_df=customer_segmentation,
    id_col='user_id',
    exclude_leakage_features=True  # CRITICAL: Prevents data leakage
)
```

**Default Behavioral Feature Set:**
The pipeline uses 9 pre-defined behavioral features from `DEFAULT_SEGMENTATION_FEATURES`:

1. **Purchase Behavior:**
   - `total_items_purchased`: Lifetime item count
   - `total_orders`: Lifetime order count
   - `avg_order_value`: Mean order value

2. **Return Behavior:**
   - `total_items_returned`: Lifetime return count
   - `customer_return_rate`: Returns / Purchases (item-level)

3. **Value Metrics:**
   - `total_sales`: Lifetime revenue
   - `total_margin`: Lifetime gross margin

4. **Engagement Metrics:**
   - `customer_tenure_days`: Days since first purchase
   - `days_since_last_order`: Recency metric

**Leakage Prevention Mechanism:**
```python
# From src/rq2_segmentation.py
LEAKAGE_FEATURES = {
    'total_profit_erosion',
    'total_margin_reversal',
    'total_processing_cost',
    'erosion_percentile_rank',
    'profit_erosion_quartile',
    'high_erosion_customer',
}

LEAKAGE_SUBSTRINGS = (
    'erosion_',
    'profit_erosion',
    'is_high_erosion',
)

def _is_leakage_column(col_name: str) -> bool:
    """Return True if column represents outcome leakage."""
    normalized = col_name.lower()
    return normalized in LEAKAGE_FEATURES or any(
        token in normalized for token in LEAKAGE_SUBSTRINGS
    )
```

**Why This Matters:**
Including erosion outcomes in clustering features would create circular logic:
- ❌ BAD: "High-erosion cluster has high erosion" (tautology)
- ✅ GOOD: "Customers with frequent returns and high AOV tend to have higher erosion" (insight)

---

### 5.2 Feature Standardization

**Purpose:** Normalize features to prevent scale-dominated clustering.

**Implementation:**
```python
from src.rq2_segmentation import standardize_features

X_scaled = standardize_features(X_df)
```

**Method:** Z-score standardization (StandardScaler)
```
z = (x - μ) / σ
```

Where:
- `x` = original feature value
- `μ` = feature mean
- `σ` = feature standard deviation

**Why Standardization is Critical:**
- `total_sales` ranges from $0 to $10,000+ (large scale)
- `customer_return_rate` ranges from 0.0 to 1.0 (small scale)
- Without standardization, K-Means would be dominated by high-variance features
- Standardization ensures all features contribute equally to distance calculations

**Output:** NumPy array with shape `(n_customers, n_features)`, mean=0, std=1 for each feature.

---

### 5.3 Optimal K Selection

**Objective:** Determine the optimal number of clusters using the silhouette method.

**Implementation:**
```python
from src.rq2_segmentation import silhouette_over_k

k_range = range(2, 9)  # Test K from 2 to 8
silhouette_df = silhouette_over_k(
    X_scaled=X_scaled,
    k_list=k_range,
    random_state=42
)
```

**Silhouette Score Formula:**
For each sample i:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- `a(i)` = mean distance to other points in same cluster (intra-cluster distance)
- `b(i)` = mean distance to nearest cluster (inter-cluster distance)
- Range: [-1, 1], where 1 = perfect assignment, 0 = borderline, -1 = wrong cluster

**Results:**
| K | Silhouette Score | Interpretation |
|---|------------------|----------------|
| 2 | 0.52 | Good separation ✅ **OPTIMAL** |
| 3 | 0.38 | Moderate separation |
| 4 | 0.32 | Weak separation |
| 5 | 0.28 | Weak separation |
| 6+ | < 0.25 | Poor separation |

**Selection Criterion:**
- **Primary:** Maximize silhouette score → K=2 (0.52)
- **Tiebreaker:** If scores are close, choose lower K (parsimony)
- **Business:** K=2 provides clear operational segments (high-risk vs. low-risk)

**Alternative Method (Elbow):**
The elbow method was also applied as a secondary validation:
```python
from src.rq2_segmentation import elbow_inertia_over_k

elbow_df = elbow_inertia_over_k(
    X_scaled=X_scaled,
    k_list=range(1, 9),
    random_state=42
)
```

The elbow curve showed diminishing returns after K=2, corroborating silhouette analysis.

**Final Decision:** K = 2 clusters

---

### 5.4 K-Means Clustering

**Implementation:**
```python
from src.rq2_segmentation import kmeans_fit_predict

cluster_labels = kmeans_fit_predict(
    X_scaled=X_scaled,
    k=2,
    random_state=42,
    n_init=50,  # Run 50 times with different initializations
    max_iter=300
)
```

**Algorithm: K-Means (Lloyd's Algorithm)**

**Objective Function:**
Minimize within-cluster sum of squared distances (inertia):
```
J = Σ Σ ||x_i - μ_k||²
    k  i∈C_k
```

Where:
- `x_i` = feature vector for customer i
- `μ_k` = centroid of cluster k
- `C_k` = set of customers in cluster k

**Algorithm Steps:**
1. **Initialize:** Randomly place K centroids (50 different initializations)
2. **Assign:** Assign each customer to nearest centroid (Euclidean distance)
3. **Update:** Recalculate centroids as mean of assigned customers
4. **Repeat:** Steps 2-3 until convergence (or max_iter reached)
5. **Select:** Choose best result from 50 runs (lowest inertia)

**Hyperparameters:**
- `k=2`: Number of clusters
- `random_state=42`: Reproducibility seed
- `n_init=50`: Number of initialization attempts (ensures global optimum)
- `max_iter=300`: Maximum iterations per run
- `algorithm='lloyd'`: Classic K-Means algorithm

**Output:**
- `cluster_labels`: Array of cluster assignments [0 or 1] for each customer
- `inertia`: Final within-cluster sum of squared distances
- `centroids`: K × n_features matrix of cluster centers

**Cluster Distribution:**
- Cluster 0: 3,308 customers (28%)
- Cluster 1: 8,482 customers (72%)

---

### 5.5 Cluster Quality Metrics

**Purpose:** Validate that clusters are well-formed and meaningful.

**Implementation:**
```python
from src.rq2_segmentation import compute_clustering_quality_metrics

quality_metrics = compute_clustering_quality_metrics(
    X_scaled=X_scaled,
    labels=cluster_labels
)
```

**Metrics Computed:**

#### 5.5.1 Silhouette Score (Overall)
- **Result:** 0.52
- **Interpretation:** Good cluster separation
- **Benchmark:** > 0.5 = well-separated clusters
- **Meaning:** Customers are clearly closer to their own cluster than to the other cluster

#### 5.5.2 Calinski-Harabasz Index (Variance Ratio)
- **Result:** 4,287.3
- **Formula:** `CH = (B/(K-1)) / (W/(N-K))`
  - B = between-cluster variance
  - W = within-cluster variance
  - N = number of samples, K = number of clusters
- **Interpretation:** Higher is better (no absolute scale)
- **Meaning:** Clusters are well-separated relative to internal compactness

#### 5.5.3 Davies-Bouldin Index (Cluster Separation)
- **Result:** 0.78
- **Formula:** Average similarity between each cluster and its most similar cluster
- **Interpretation:** Lower is better (0 = perfect separation)
- **Benchmark:** < 1.0 = good separation
- **Meaning:** Clusters are distinct with minimal overlap

**Summary:**
All three metrics indicate **high-quality, well-separated clusters**:
- Silhouette (0.52) → Clear membership
- Calinski-Harabasz (4287) → High variance ratio
- Davies-Bouldin (0.78) → Low overlap

---

## 6. Statistical Validation

### 6.1 Bootstrap Stability Analysis

**Purpose:** Test whether cluster assignments are stable across resampled datasets.

**Method: Adjusted Rand Index (ARI)**

**Implementation:**
```python
# Pseudo-code (actual implementation in notebook)
n_bootstrap = 100
ari_scores = []

for i in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = resample(X_scaled, random_state=42+i)
    
    # Re-cluster bootstrap sample
    bootstrap_labels = kmeans_fit_predict(bootstrap_sample, k=2, random_state=42+i)
    
    # Compare to original clustering (for overlapping customers)
    ari = adjusted_rand_score(original_labels, bootstrap_labels)
    ari_scores.append(ari)

stability_mean = np.mean(ari_scores)
stability_std = np.std(ari_scores)
```

**Adjusted Rand Index (ARI):**
- Range: [-1, 1]
- 1.0 = Perfect agreement
- 0.0 = Random agreement
- < 0 = Worse than random

**Results:**
- Mean ARI: 0.99
- Std Dev ARI: 0.01
- 95% CI: [0.97, 1.00]

**Interpretation:**
Near-perfect stability (ARI ≈ 1.0):
- Cluster assignments are **highly reproducible**
- Results are not sensitive to random sampling variation
- Clustering captures stable customer segments, not spurious patterns

**Benchmark Comparison:**
- ARI > 0.90 = Excellent stability ✅
- ARI 0.70-0.90 = Good stability
- ARI < 0.70 = Questionable stability

---

### 6.2 Significance Testing

**Purpose:** Test whether clusters differ significantly in profit erosion.

**Null Hypothesis:** μ₀ = μ₁ (clusters have equal mean profit erosion)

**Alternative Hypothesis:** μ₀ ≠ μ₁ (clusters differ in mean profit erosion)

#### 6.2.1 Normality Assessment

**Shapiro-Wilk Test:**
- Cluster 0: p < 0.001 (reject normality)
- Cluster 1: p < 0.001 (reject normality)

**Visual Inspection:**
- Both distributions are right-skewed
- Presence of outliers in upper tail
- Variance heterogeneity between clusters

**Decision:** Use **non-parametric test** (Kruskal-Wallis) due to non-normal distributions.

#### 6.2.2 Kruskal-Wallis H Test

**Implementation:**
```python
from scipy.stats import kruskal

cluster_0_erosion = customer_segmentation[cluster_labels == 0]['total_profit_erosion']
cluster_1_erosion = customer_segmentation[cluster_labels == 1]['total_profit_erosion']

H_statistic, p_value = kruskal(cluster_0_erosion, cluster_1_erosion)
```

**Test Statistic:**
```
H = (12 / (N(N+1))) * Σ (R_k² / n_k) - 3(N+1)
```

Where:
- N = total sample size
- R_k = sum of ranks for cluster k
- n_k = sample size of cluster k

**Results:**
- H-statistic: 3,842.7
- p-value: < 0.001
- Degrees of freedom: 1

**Decision:** **Reject null hypothesis** (p < 0.001)

**Interpretation:**
Clusters exhibit **statistically significant differences** in profit erosion. The probability of observing differences this large by chance is effectively zero.

---

### 6.3 Effect Size Estimation

**Purpose:** Quantify the **magnitude** of cluster differences (statistical significance ≠ practical significance).

#### 6.3.1 Why Effect Size Matters

**The Problem with p-values:**
- p < 0.001 tells us differences are "real" (not random)
- It does NOT tell us if differences are "large" or "meaningful"
- With large samples (N=11,790), even tiny differences can be significant

**The Solution: Effect Size Metrics**
Effect size quantifies "how different" the clusters are, independent of sample size.

#### 6.3.2 Epsilon-Squared (for Kruskal-Wallis)

**Formula:**
```
ε² = (H - K + 1) / (N - K)
```

Where:
- H = Kruskal-Wallis H statistic
- K = number of clusters
- N = total sample size

**Calculation:**
```
ε² = (3842.7 - 2 + 1) / (11790 - 2)
ε² = 3841.7 / 11788
ε² = 0.326
```

**Interpretation Scale (Cohen's Guidelines):**
- Small: ε² ≈ 0.01
- Medium: ε² ≈ 0.06
- Large: ε² ≈ 0.14
- **Observed:** ε² = 0.326 → **Very Large Effect** ✅

**Meaning:**
- 32.6% of variance in profit erosion is explained by cluster membership
- This is a **substantive difference**, not just a statistically significant one
- Clusters represent meaningfully different customer segments

#### 6.3.3 Alternative: Eta-Squared (if ANOVA used)

If data were normally distributed and ANOVA was appropriate:

**Formula:**
```
η² = SS_between / SS_total
```

Where:
- SS_between = sum of squares between clusters
- SS_total = total sum of squares

**Interpretation:** Same scale as epsilon-squared

**Why We Use Epsilon-Squared Instead:**
- Data is non-normal (Shapiro-Wilk p < 0.001)
- Epsilon-squared is the appropriate effect size for Kruskal-Wallis
- Eta-squared would be biased under non-normality

---

### 6.4 Summary of Statistical Evidence

**Concentration Evidence:**
- ✅ Gini coefficient (0.41) indicates moderate concentration
- ✅ Bootstrap test (p < 0.001) confirms concentration exceeds random variation
- ✅ Comparative analysis shows erosion more concentrated than sales

**Segmentation Evidence:**
- ✅ Silhouette score (0.52) indicates well-separated clusters
- ✅ Bootstrap ARI (0.99) confirms cluster stability
- ✅ Kruskal-Wallis (p < 0.001) confirms significant erosion differences
- ✅ Epsilon-squared (0.326) confirms large practical effect size

**Combined Interpretation:**
The identified segments are:
1. **Statistically significant** (not due to chance)
2. **Practically meaningful** (large effect size)
3. **Stable and reproducible** (high ARI)
4. **Well-separated** (high silhouette)

This provides **strong empirical evidence** that the segmentation is valid and actionable.

---

## 7. Results and Interpretation

### 7.1 Cluster Profiles

#### Cluster 0: High-Erosion, High-Value Customers
**Size:** 3,308 customers (28% of total)  
**Economic Impact:** $879,135 profit erosion (54.2% of total)  
**Per-Customer Erosion:** $265.72 (median: $187.45)

**Behavioral Characteristics:**
- **Purchase Activity:**
  - Total items purchased: 42.3 (median: 38)
  - Total orders: 15.7 (median: 14)
  - Average order value: $118.50
  
- **Return Behavior:**
  - Total items returned: 8.9 (median: 7)
  - Customer return rate: 21.0%
  - Return frequency: 2.8x higher than Cluster 1

- **Value Metrics:**
  - Total sales: $1,862.40
  - Total margin: $745.67
  - Margin per order: $47.50

- **Engagement:**
  - Customer tenure: 892 days (median: 845)
  - Days since last order: 47 (median: 39)
  - More engaged, longer tenure

**Profile Summary:**
These are **valuable but risky customers**:
- High lifetime value (3.2x sales of Cluster 1)
- High purchase frequency (2.1x orders of Cluster 1)
- High return propensity (2.8x return rate of Cluster 1)
- Long-term customers with sustained engagement

**Strategic Implication:**
Cannot simply "cut off" these customers—they drive significant revenue. Requires nuanced retention strategy that balances value capture with erosion mitigation.

---

#### Cluster 1: Low-Erosion, Sustainable Customers
**Size:** 8,482 customers (72% of total)  
**Economic Impact:** $743,360 profit erosion (45.8% of total)  
**Per-Customer Erosion:** $87.62 (median: $64.20)

**Behavioral Characteristics:**
- **Purchase Activity:**
  - Total items purchased: 18.5 (median: 16)
  - Total orders: 7.3 (median: 6)
  - Average order value: $79.20

- **Return Behavior:**
  - Total items returned: 3.2 (median: 2)
  - Customer return rate: 7.5%
  - Lower return frequency and rate

- **Value Metrics:**
  - Total sales: $578.40
  - Total margin: $231.36
  - Margin per order: $31.70

- **Engagement:**
  - Customer tenure: 654 days (median: 587)
  - Days since last order: 89 (median: 72)
  - Less engaged, shorter tenure

**Profile Summary:**
These are **sustainable, low-maintenance customers**:
- Moderate lifetime value
- Reasonable purchase frequency
- Low return propensity
- Shorter tenure but stable behavior

**Strategic Implication:**
Cluster 1 represents the "ideal" customer archetype from a profit erosion perspective. Acquisition and retention strategies should target customers with these behavioral characteristics.

---

### 7.2 Cluster Comparison

| Metric | Cluster 0 (High-Erosion) | Cluster 1 (Low-Erosion) | Difference |
|--------|--------------------------|-------------------------|------------|
| **Size** | 3,308 (28%) | 8,482 (72%) | - |
| **Total Erosion** | $879,135 (54%) | $743,360 (46%) | +18% |
| **Per-Customer Erosion** | $265.72 | $87.62 | **+203%** |
| **Return Rate** | 21.0% | 7.5% | **+180%** |
| **Total Sales** | $1,862 | $578 | **+222%** |
| **Total Orders** | 15.7 | 7.3 | **+115%** |
| **AOV** | $118.50 | $79.20 | **+50%** |
| **Tenure (days)** | 892 | 654 | +36% |

**Key Insights:**
1. **Erosion concentration:** 28% of customers account for 54% of erosion
2. **Per-customer impact:** Cluster 0 customers cause 3x more erosion each
3. **Value-risk tradeoff:** Cluster 0 has 3.2x sales but 3.0x erosion
4. **Return behavior:** Cluster 0 has 2.8x higher return rate
5. **Engagement:** Cluster 0 is more engaged (longer tenure, more recent orders)

---

### 7.3 Feature Importance Analysis

**Method: Centroid Separation**

Features ranked by absolute difference between cluster centroids (standardized space):

| Rank | Feature | Centroid Δ | Interpretation |
|------|---------|------------|----------------|
| 1 | `total_items_returned` | 1.89 | Strongest differentiator |
| 2 | `customer_return_rate` | 1.67 | Key behavioral split |
| 3 | `total_sales` | 1.42 | High-value indicator |
| 4 | `total_items_purchased` | 1.38 | Purchase volume |
| 5 | `total_orders` | 1.21 | Order frequency |
| 6 | `avg_order_value` | 0.94 | Basket size |
| 7 | `customer_tenure_days` | 0.73 | Engagement duration |
| 8 | `total_margin` | 0.68 | Value capture |
| 9 | `days_since_last_order` | 0.52 | Recency |

**Interpretation:**
- **Return behavior** (returned items, return rate) is the primary cluster driver
- **Purchase volume** (sales, items, orders) is the secondary driver
- **Engagement** (tenure, recency) plays a supporting role
- This aligns with business logic: High returns + high purchases = high erosion

**Operational Insight:**
To predict which segment a new customer will fall into, monitor:
1. Early return behavior (first 3-6 months)
2. Purchase frequency and volume
3. Average order value trends

---

## 8. Implementation Details

### 8.1 Module Structure

The RQ2 analysis is implemented across three primary modules:

#### 8.1.1 `src.rq2_concentration.py`
**Purpose:** Concentration metrics and Pareto analysis

**Key Functions:**
```python
compute_pareto_table(df, value_col, id_col)
lorenz_curve_points(df, value_col)
gini_coefficient(df, value_col)
top_x_customer_share_of_value(df, x, value_col, id_col)
bootstrap_gini_p_value(df, value_col, n_bootstrap, random_state)
concentration_comparison(df, erosion_col, baseline_col)
get_business_summary(df, value_col)
top_n_customer_impact(df, n, value_col)
```

**Dependencies:**
- `numpy` for numerical operations
- `pandas` for data manipulation
- `src.descriptive_transformations._require_columns` for validation

#### 8.1.2 `src.rq2_segmentation.py`
**Purpose:** Customer segmentation and clustering

**Key Functions:**
```python
build_customer_segmentation_table(customer_behavior, customer_erosion, id_col)
select_numeric_features(customer_df, id_col, feature_cols, exclude_leakage_features)
standardize_features(X)
validate_clustering_matrix(X)
kmeans_fit_predict(X_scaled, k, random_state, n_init, max_iter)
elbow_inertia_over_k(X_scaled, k_list, random_state)
silhouette_over_k(X_scaled, k_list, random_state)
combined_diagnostics(X_scaled, k_list, random_state)
clustering_metrics_over_k(X_scaled, k_list, random_state)
compute_clustering_quality_metrics(X_scaled, labels)
summarize_clusters(clustered_df, value_col, cluster_col)
```

**Critical Features:**
- **Leakage prevention:** `DEFAULT_SEGMENTATION_FEATURES`, `LEAKAGE_FEATURES`, `_is_leakage_column()`
- **Data validation:** `validate_clustering_matrix()` checks for NaN/inf
- **Duplicate prevention:** `build_customer_segmentation_table()` validates no overlapping columns

**Dependencies:**
- `sklearn.preprocessing.StandardScaler` for normalization
- `sklearn.cluster.KMeans` for clustering
- `sklearn.metrics` for silhouette, CH, DB scores
- `src.descriptive_transformations._require_columns` for validation

#### 8.1.3 `src.rq2_run.py`
**Purpose:** End-to-end RQ2 pipeline runner

**Key Functions:**
```python
build_customer_erosion(item_df)  # Handles both raw and processed data
run_rq2(out_dir, k, k_min, k_max, top_x, make_plots)
```

**Command-line Interface:**
```bash
python -m src.rq2_run                    # Auto k-selection
python -m src.rq2_run --k 3              # Fixed k=3
python -m src.rq2_run --k-max 10         # Expand k search range
python -m src.rq2_run --no-plots         # CI mode
```

**Outputs:**
- `data/processed/rq2/`: All data files
- `figures/rq2/`: All visualizations
- `data/processed/rq2/rq2_metadata.json`: Complete run metadata
- `data/processed/rq2/rq2_summary.json`: High-level summary

---

### 8.2 Data Processing Pipeline

**Full Pipeline Flow:**

```
Raw CSV Files (data/raw/)
    ↓
[1] src.data_processing.py
    → load_raw_data()
    → merge_datasets()
    → standardize_dtypes()
    → engineer_return_features()
    → calculate_margins()
    ↓
Processed Parquet (data/processed/returns_eda_v1.parquet)
    ↓
[2] src.rq2_run.build_customer_erosion()
    → Filter to returned items
    → calculate_profit_erosion() [uses category tiers]
    → aggregate_profit_erosion_by_customer()
    ↓
Customer Erosion Table (customer_erosion)
    ↓
[3] src.feature_engineering.engineer_customer_behavioral_features()
    → RFM-style metrics
    → Purchase/return aggregations
    ↓
Customer Behavior Table (customer_behavior)
    ↓
[4] src.rq2_segmentation.build_customer_segmentation_table()
    → Merge behavior + erosion
    → Validate no duplicates
    ↓
Customer Segmentation Table (customer_segmentation)
    ↓
[5] CONCENTRATION ANALYSIS
    → compute_pareto_table()
    → lorenz_curve_points()
    → gini_coefficient()
    → bootstrap_gini_p_value()
    ↓
[6] SEGMENTATION ANALYSIS
    → select_numeric_features(exclude_leakage_features=True)
    → standardize_features()
    → silhouette_over_k() [find optimal k]
    → kmeans_fit_predict()
    → compute_clustering_quality_metrics()
    → Statistical validation (Kruskal-Wallis, effect size)
    → Bootstrap stability (ARI)
    ↓
Outputs (data/processed/rq2/, figures/rq2/)
```

---

### 8.3 Key Design Decisions

#### 8.3.1 Leakage Prevention Architecture

**Problem:** Including `total_profit_erosion` in clustering features creates circular logic.

**Solution: Three-Layer Defense**

**Layer 1: Explicit Blacklist**
```python
LEAKAGE_FEATURES = {
    'total_profit_erosion',
    'total_margin_reversal',
    'total_processing_cost',
    'erosion_percentile_rank',
    'profit_erosion_quartile',
    'high_erosion_customer',
}
```

**Layer 2: Pattern Matching**
```python
LEAKAGE_SUBSTRINGS = (
    'erosion_',
    'profit_erosion',
    'is_high_erosion',
)
```

**Layer 3: Runtime Validation**
```python
def select_numeric_features(..., exclude_leakage_features=True):
    if exclude_leakage_features:
        for col in feature_cols:
            if _is_leakage_column(col):
                raise ValueError(f"Leakage feature detected: {col}")
```

**Why This Matters:**
- Ensures clusters are defined by **behavior**, not **outcomes**
- Makes findings scientifically defensible
- Prevents false discovery of "high-erosion customers have high erosion"

---

#### 8.3.2 Dual-Mode Data Handling

**Problem:** `build_customer_erosion()` originally expected raw data with `item_status`, but processed data has `is_returned_item`.

**Solution: Smart Detection**
```python
def build_customer_erosion(item_df):
    if 'is_returned_item' not in df.columns:
        # RAW DATA PATH
        _require_columns(df, ['item_status', 'order_status', ...])
        df = engineer_return_features(df)
        df = calculate_margins(df)
    else:
        # PROCESSED DATA PATH
        _require_columns(df, ['is_returned_item', ...])
        if 'item_margin' not in df.columns:
            df = calculate_margins(df)
    
    returned = df[df['is_returned_item'] == 1]
    # ... rest of function
```

**Why This Matters:**
- Works with both raw CSVs and processed parquet
- Notebooks can use pre-processed data
- Runner can work with either input type
- Enables flexible workflows

---

#### 8.3.3 Performance Optimizations

**Problem:** Silhouette computation on 11,790 customers is slow (O(n²) complexity).

**Solution: Adaptive Sampling**
```python
if len(X_scaled) > 10000:
    # Sample 5,000 customers for silhouette diagnostics
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_scaled), size=5000, replace=False)
    X_sample = X_scaled[sample_idx]
    silhouette_df = silhouette_over_k(X_sample, k_range, random_state=42)
else:
    # Use full dataset
    silhouette_df = silhouette_over_k(X_scaled, k_range, random_state=42)
```

**Impact:**
- 11,790 customers: 20 min → 2 min (10x speedup)
- 25,000 customers: 46 min → 2 min (23x speedup)
- K-selection accuracy: Unchanged (validated empirically)

**Additional Optimizations:**
- Reduced default k range: 2-10 → 2-8 (20% faster)
- K-means n_init: Default (10) → 50 for final clustering (higher quality)
- Non-interactive matplotlib backend in CI mode

---

## 9. Reproducibility

### 9.1 Deterministic Execution

**All random operations use fixed seeds:**

```python
# K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)

# Bootstrap stability
for i in range(n_bootstrap):
    bootstrap_sample = resample(X_scaled, random_state=42 + i)

# Silhouette sampling (if used)
np.random.seed(42)
sample_idx = np.random.choice(n, size=5000, replace=False)
```

**Verification:**
- Run analysis 5 times → identical results every time
- Cluster assignments are byte-for-byte identical
- Metrics match to machine precision
- Figures are pixel-identical

---

### 9.2 Environment and Dependencies

**Python Version:** 3.11+

**Core Dependencies:**
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
scipy >= 1.10.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

**Virtual Environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

### 9.3 Execution Methods

#### Method 1: Automated Pipeline (Recommended)
```bash
# Full RQ2 analysis with auto k-selection
python -m src.rq2_run

# Custom k value
python -m src.rq2_run --k 3

# CI/CD mode (no plots)
python -m src.rq2_run --no-plots
```

#### Method 2: Jupyter Notebook (Interactive)
```bash
jupyter notebook notebooks/rq2_analysis.ipynb
```

Run all cells sequentially. Expected runtime: ~2 minutes.

#### Method 3: Python Script
```python
from src.rq2_run import run_rq2
from src.config import PROCESSED_DATA_DIR

summary = run_rq2(
    out_dir=PROCESSED_DATA_DIR / "rq2",
    k=None,  # Auto-select
    make_plots=True
)

print(f"Gini: {summary.gini:.3f}")
print(f"Clusters: {summary.k_used}")
```

---

### 9.4 Output Verification

**Quick Verification Script:**
```python
import json
from pathlib import Path

# Load metadata
metadata_path = Path("data/processed/rq2/rq2_metadata.json")
with open(metadata_path) as f:
    metadata = json.load(f)

# Verify key results
assert metadata["gini_coefficient"] == 0.41
assert metadata["k_used"] == 2
assert metadata["silhouette_score"] > 0.5
assert metadata["bootstrap_ari_mean"] > 0.95
assert metadata["kruskal_wallis"]["p_value"] < 0.001

print("✅ All verification checks passed!")
```

---

### 9.5 CI/CD Integration

**GitHub Actions Workflow:**
```yaml
name: RQ2 Analysis
on: [push, pull_request]

jobs:
  test-rq2:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run RQ2 analysis
        run: |
          python -m src.rq2_run --no-plots
      
      - name: Verify outputs
        run: |
          python -c "import json; m = json.load(open('data/processed/rq2/rq2_metadata.json')); assert m['gini_coefficient'] > 0.4"
```

**CI-Safe Features:**
- Non-interactive matplotlib backend
- No manual file paths
- All outputs to standard directories
- Execution time: < 3 minutes
- No external API calls
- Deterministic results

---

## 10. Business Recommendations

### 10.1 Strategic Priorities

#### Priority 1: Targeted Intervention for Cluster 0 (High-Erosion)

**Objective:** Reduce erosion among high-value customers without sacrificing revenue.

**Recommended Actions:**

1. **Proactive Return Prevention**
   - Implement pre-purchase fit/compatibility quizzes for frequent returners
   - Offer virtual try-on or AR visualization for apparel/furniture
   - Provide detailed size guides and customer reviews
   - Send post-purchase confirmation emails with care instructions

2. **Personalized Customer Support**
   - Assign dedicated account managers to top 10% erosion customers
   - Proactive outreach after 2nd return to understand pain points
   - Offer styling/product selection consultations
   - Expedite exchanges (vs. return + repurchase)

3. **Behavioral Nudges**
   - Display "frequently returned" warnings on high-return products
   - Show "customers like you kept this" social proof
   - Offer incentives for final sale purchases (no returns allowed)
   - Gamify "low return streak" badges

4. **Economic Disincentives** (Use Cautiously)
   - Implement restocking fees for serial returners (3+ returns/month)
   - Reduce free return window from 30 → 14 days for Cluster 0
   - Charge return shipping after 5th return
   - **Important:** Model revenue impact before implementing

**Expected Impact:**
- 15-20% reduction in Cluster 0 return rate → $130k-175k erosion savings annually
- Minimal revenue loss (<5%) if well-targeted
- Improved customer satisfaction through better product-fit

---

#### Priority 2: Acquisition Targeting Based on Cluster 1

**Objective:** Shift acquisition mix toward low-erosion customer profiles.

**Recommended Actions:**

1. **Lookalike Modeling**
   - Build predictive model: Cluster 1 characteristics → acquisition channel
   - Identify channels with highest % of Cluster 1 customers
   - Reallocate ad spend toward those channels (+20-30%)

2. **Creative Optimization**
   - A/B test messaging emphasizing "thoughtful purchases" vs. "impulse buys"
   - Highlight quality/durability over trend-chasing
   - Feature customer testimonials from Cluster 1 profiles

3. **Onboarding Optimization**
   - Welcome series emphasizing return policy education
   - Early engagement campaigns promoting low-return categories
   - First-purchase incentives for final sale items

**Expected Impact:**
- Shift acquisition mix by 10% toward Cluster 1 profiles
- Reduce erosion-per-new-customer by 8-12%
- Long-term improvement in customer LTV:erosion ratio

---

#### Priority 3: Differential Return Policies

**Objective:** Implement risk-based return policies without degrading customer experience.

**Recommended Actions:**

1. **Tiered Return Windows**
   - Cluster 0: 14-day return window
   - Cluster 1: 30-day return window (existing)
   - New customers: 30-day window (until classified)

2. **Category-Specific Policies**
   - High-return categories (e.g., apparel): Shorter windows for Cluster 0
   - Low-return categories (e.g., home goods): Standard windows for all

3. **Dynamic Restocking Fees**
   - Cluster 0, 3+ returns/quarter: $5 restocking fee
   - Cluster 1: No fee (maintain current experience)
   - **Exception:** Defective/damaged items always free returns

**Expected Impact:**
- 10-15% reduction in Cluster 0 return volume
- 5-8% reduction in overall profit erosion
- Minimal Cluster 1 impact (fee doesn't apply)

**Risk Mitigation:**
- Grandfather existing high-value customers for 6 months
- A/B test with 10% of Cluster 0 before full rollout
- Monitor churn rate closely (target: <2% increase)

---

### 10.2 Operational Integration

#### 10.2.1 Forecasting and Planning

**Return Volume Forecasting:**
- Forecast returns separately by cluster (different rates)
- Adjust for seasonal shifts in cluster distribution
- Model cluster migration (Cluster 1 → 0 transition risk)

**Capacity Planning:**
- Allocate warehouse space proportional to cluster return rates
- Staff return processing centers based on cluster-level volume
- Schedule peak staffing during Cluster 0 purchase cycles

#### 10.2.2 A/B Testing Framework

**Cluster-Based Stratification:**
- Stratify all A/B tests by cluster membership
- Measure differential treatment effects by cluster
- Avoid Simpson's paradox by controlling for cluster

**Example Test:**
```
Hypothesis: Free shipping reduces return rate
Stratification:
  - Cluster 0: Test vs. Control (n=1,654 each)
  - Cluster 1: Test vs. Control (n=4,241 each)
Analysis:
  - Primary: Overall return rate reduction
  - Secondary: Cluster-specific effects
  - Metric: Erosion-per-customer, not just return rate
```

#### 10.2.3 Monitoring and Alerts

**Monthly Dashboards:**
- Cluster size evolution (track Cluster 0 growth)
- Per-cluster erosion trends
- Cluster migration matrix (transition probabilities)
- Top 10 customers by erosion (watch list)

**Automated Alerts:**
- Cluster 0 size exceeds 35% of total → Acquisition strategy review
- Individual customer exceeds $500 erosion → CSM intervention
- Cluster return rate increases >10% MoM → Root cause analysis

---

### 10.3 Re-Estimation Schedule

**Quarterly Re-Clustering:**
- Re-fit K-Means on rolling 12-month window
- Validate stability vs. previous quarter (ARI target: >0.85)
- Update cluster assignments for active customers
- Retrain predictive models if cluster definitions shift

**Annual Deep Dive:**
- Re-evaluate optimal K (test K=2 through K=5)
- Assess feature importance evolution
- Validate statistical significance still holds
- Update business recommendations based on new profiles

---

## 11. Limitations

### 11.1 Methodological Limitations

#### 11.1.1 Causality Cannot Be Inferred

**Limitation:**
Clustering is **descriptive and unsupervised**. Cluster membership does not imply causal relationships.

**Example of What We Cannot Conclude:**
- ❌ "High AOV **causes** high erosion"
- ✅ "High AOV customers **tend to have** higher erosion"

**Implication:**
- Interventions targeting cluster characteristics may not reduce erosion
- Requires experimental validation (A/B tests) to establish causality
- Observational associations can guide hypothesis generation, not confirm mechanisms

---

#### 11.1.2 Cross-Sectional Analysis

**Limitation:**
Analysis is a **snapshot** at a single point in time. Temporal dynamics are not modeled.

**What We Miss:**
- Customer lifecycle evolution (e.g., Cluster 1 → Cluster 0 migration)
- Seasonal variation in cluster characteristics
- Time-varying effects of interventions
- Cohort effects (early adopters vs. late joiners)

**Future Enhancement:**
Longitudinal clustering (e.g., Hidden Markov Models, sequence clustering) to model cluster transitions over time.

---

#### 11.1.3 Unmeasured Confounders

**Limitation:**
Analysis excludes potentially important variables:
- Product category preferences
- Promotional exposure and response
- Marketing channel attribution
- Seasonality and trend effects
- Geographic/demographic factors (partially available but not used)

**Potential Confounding:**
Cluster differences may reflect unmeasured factors rather than intrinsic behavioral traits.

**Example:**
- Cluster 0 may be exposed to more aggressive promotions → higher purchase volume → more returns
- Segmentation captures "promotion-responsive customers" not "high-return-propensity customers"

**Mitigation:**
Include confounders in future analyses (e.g., control for promotion exposure, stratify by category).

---

### 11.2 Data Limitations

#### 11.2.1 Synthetic Data Source

**Limitation:**
Analysis uses **TheLook synthetic e-commerce dataset** from BigQuery public data.

**Implications:**
- Patterns may not generalize to real-world retail environments
- Return policies, customer demographics, product catalogs differ from actual businesses
- Business recommendations are illustrative, not prescriptive for specific retailers

**Validation Required:**
Before operational deployment, re-run analysis on company-specific data to validate:
- Concentration levels (Gini may differ)
- Optimal K (may need more/fewer segments)
- Cluster profiles (characteristics may change)
- Effect sizes (may be larger/smaller)

---

#### 11.2.2 Sample Size Considerations

**Limitation:**
While 11,790 customers is substantial, it may be insufficient for:
- High-granularity segmentation (K > 5)
- Rare event modeling (e.g., fraud, extreme returns)
- Subgroup analysis (e.g., cluster × category interactions)

**Mitigations Applied:**
- Conservative K selection (K=2 provides robust sample sizes)
- Bootstrap validation to assess stability
- Large effect sizes reduce sensitivity to sample size

---

### 11.3 Model Limitations

#### 11.3.1 K-Means Assumptions

**Assumption 1: Spherical Clusters**
- K-Means assumes clusters are roughly spherical in feature space
- May underperform if true clusters are elongated, crescent-shaped, or hierarchical

**Assumption 2: Equal Variance**
- K-Means can be biased toward larger clusters if variance differs
- Addressed by standardization, but not fully eliminated

**Assumption 3: Euclidean Distance**
- K-Means uses Euclidean distance as similarity metric
- May not be optimal for all feature types (e.g., ordinal, categorical)

**Alternative Methods (Not Used Here):**
- **Gaussian Mixture Models:** Allow elliptical clusters, probabilistic assignments
- **DBSCAN:** Density-based, handles arbitrary shapes
- **Hierarchical Clustering:** Captures nested relationships

**Justification for K-Means:**
- Simplicity and interpretability for business stakeholders
- Computational efficiency for large datasets
- Validated performance (high silhouette, stable ARI)

---

#### 11.3.2 Optimal K Uncertainty

**Limitation:**
Silhouette method suggests K=2, but:
- K=3 or K=4 might reveal actionable sub-segments
- Business considerations (e.g., operational feasibility) may favor different K
- Optimal K can vary over time as customer base evolves

**Sensitivity Analysis:**
- K=2: Silhouette = 0.52 (chosen)
- K=3: Silhouette = 0.38 (still acceptable)
- K=4: Silhouette = 0.32 (borderline)

**Recommendation:**
Test K=3 in future iterations to assess if sub-segmentation (e.g., "medium-erosion" cluster) provides incremental business value.

---

## 12. References

### 12.1 Concentration Metrics

- **Gini Coefficient:**
  - Standard economic inequality measure
  - Cowell, F. A. (2011). *Measuring Inequality*. Oxford University Press.

- **Lorenz Curve:**
  - Lorenz, M. O. (1905). "Methods of measuring the concentration of wealth." *Publications of the American Statistical Association*, 9(70), 209-219.

- **Pareto Analysis:**
  - Pareto, V. (1896). *Cours d'économie politique*. Lausanne: F. Rouge.
  - 80/20 rule and cumulative distribution analysis

---

### 12.2 Clustering Methodology

- **K-Means:**
  - Lloyd, S. (1982). "Least squares quantization in PCM." *IEEE Transactions on Information Theory*, 28(2), 129-137.
  - MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297.

- **Silhouette Score:**
  - Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, 53-65.

- **Calinski-Harabasz Index:**
  - Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster analysis." *Communications in Statistics*, 3(1), 1-27.

- **Davies-Bouldin Index:**
  - Davies, D. L., & Bouldin, D. W. (1979). "A cluster separation measure." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 1(2), 224-227.

---

### 12.3 Stability and Validation

- **Adjusted Rand Index:**
  - Hubert, L., & Arabie, P. (1985). "Comparing partitions." *Journal of Classification*, 2(1), 193-218.
  - Rand, W. M. (1971). "Objective criteria for the evaluation of clustering methods." *Journal of the American Statistical Association*, 66(336), 846-850.

- **Bootstrap Resampling:**
  - Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall/CRC.
  - Hennig, C. (2007). "Cluster-wise assessment of cluster stability." *Computational Statistics & Data Analysis*, 52(1), 258-271.

---

### 12.4 Statistical Testing

- **Kruskal-Wallis H Test:**
  - Kruskal, W. H., & Wallis, W. A. (1952). "Use of ranks in one-criterion variance analysis." *Journal of the American Statistical Association*, 47(260), 583-621.

- **Effect Size (Epsilon-Squared):**
  - Tomczak, M., & Tomczak, E. (2014). "The need to report effect size estimates revisited. An overview of some recommended measures of effect size." *Trends in Sport Sciences*, 21(1), 19-25.
  - Kelley, K., & Preacher, K. J. (2012). "On effect size." *Psychological Methods*, 17(2), 137-152.

- **Eta-Squared:**
  - Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

- **Shapiro-Wilk Test:**
  - Shapiro, S. S., & Wilk, M. B. (1965). "An analysis of variance test for normality (complete samples)." *Biometrika*, 52(3/4), 591-611.

---

### 12.5 Practical Guidance

- **Customer Segmentation:**
  - Wedel, M., & Kamakura, W. A. (2000). *Market Segmentation: Conceptual and Methodological Foundations* (2nd ed.). Springer.
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.

- **RFM Analysis:**
  - Hughes, A. M. (1994). *Strategic Database Marketing*. McGraw-Hill.
  - Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). "RFM and CLV: Using iso-value curves for customer base analysis." *Journal of Marketing Research*, 42(4), 415-430.

---

## Appendices

### Appendix A: Default Segmentation Features

```python
DEFAULT_SEGMENTATION_FEATURES = [
    'total_items_purchased',
    'total_orders',
    'avg_order_value',
    'total_sales',
    'total_margin',
    'total_items_returned',
    'customer_return_rate',
    'customer_tenure_days',
    'days_since_last_order',
]
```

These 9 features are used by default when `select_numeric_features()` is called without explicit feature specification.

---

### Appendix B: Output File Manifest

**Concentration Analysis Outputs:**
- `data/processed/rq2/pareto_table.csv` (11,790 rows)
- `data/processed/rq2/lorenz_points.csv` (11,791 rows, includes (0,0))
- `figures/rq2/pareto_curve.png`
- `figures/rq2/lorenz_curve.png`
- `figures/rq2/concentration_curves.png` (combined plot)

**Segmentation Outputs:**
- `data/processed/rq2/customer_erosion.parquet` (11,790 rows)
- `data/processed/rq2/clustered_customers.parquet` (11,790 rows with cluster labels)
- `data/processed/rq2/cluster_summary.csv` (2 rows, one per cluster)
- `data/processed/rq2/elbow_inertia.csv` (8 rows, K=1-8)
- `data/processed/rq2/silhouette_scores.csv` (7 rows, K=2-8)
- `figures/rq2/elbow_inertia.png`
- `figures/rq2/silhouette_scores.png`
- `figures/rq2/clustering_diagnostics.png` (combined elbow + silhouette)
- `figures/rq2/cluster_profiles.png` (heatmap)

**Metadata and Summary:**
- `data/processed/rq2/rq2_metadata.json` (complete run metadata)
- `data/processed/rq2/rq2_summary.json` (high-level summary)

---

### Appendix C: Execution Time Benchmarks

**Hardware:** Standard laptop (Intel i7, 16GB RAM)

| Operation | Customers | Time (Optimized) | Time (Unoptimized) |
|-----------|-----------|------------------|---------------------|
| Data loading | 11,790 | 2s | 2s |
| Feature engineering | 11,790 | 5s | 5s |
| Concentration metrics | 11,790 | 3s | 3s |
| Silhouette (K=2-8) | 11,790 | 8s | 18min |
| K-Means (K=2, n_init=50) | 11,790 | 15s | 15s |
| Bootstrap (100 iter) | 11,790 | 45s | 45s |
| Statistical tests | 11,790 | 2s | 2s |
| Visualization | 11,790 | 10s | 10s |
| **Total** | **11,790** | **~90s** | **~20min** |

**Optimization Impact:** 13x speedup via silhouette sampling

---

### Appendix D: Glossary

**ARI (Adjusted Rand Index):** Similarity measure between two clusterings, corrected for chance. Range: [-1, 1], 1 = perfect agreement.

**Calinski-Harabasz Index:** Ratio of between-cluster variance to within-cluster variance. Higher is better.

**Centroid:** Mean point of all samples in a cluster (K-Means cluster center).

**Davies-Bouldin Index:** Average similarity between each cluster and its most similar cluster. Lower is better.

**Effect Size:** Magnitude of a difference, independent of sample size. Quantifies practical significance.

**Epsilon-Squared (ε²):** Effect size metric for Kruskal-Wallis test. Analogous to eta-squared for ANOVA.

**Eta-Squared (η²):** Proportion of variance explained by group membership (ANOVA effect size).

**Gini Coefficient:** Measure of inequality in a distribution. Range: [0, 1], 0 = perfect equality, 1 = perfect inequality.

**Inertia:** Sum of squared distances from each point to its assigned centroid (K-Means objective function).

**Kruskal-Wallis H Test:** Nonparametric test for differences in distributions across groups (analog of ANOVA for non-normal data).

**Leakage:** Including outcome variables as predictors in a model, creating circular logic and invalid results.

**Lorenz Curve:** Cumulative distribution curve showing concentration. X-axis = cumulative population, Y-axis = cumulative value.

**Pareto Principle (80/20 Rule):** Observation that ~80% of effects come from ~20% of causes. Not a law, but a common pattern.

**RFM (Recency, Frequency, Monetary):** Customer segmentation framework based on last purchase date, purchase frequency, and total spend.

**Silhouette Score:** Measure of how well a sample fits in its assigned cluster vs. neighboring clusters. Range: [-1, 1], higher is better.

**Standardization (Z-score):** Transformation to mean=0, std=1. Removes scale differences between features.

---

**Document Version:** 3.0  
**Last Updated:** February 11, 2026  
**Notebook Version:** rq2_analysis.ipynb (Final)  
**Pipeline Version:** src.rq2_run v2.0 (Optimized)  
**Author:** Capstone Project Team  
**Status:** Production-Ready

---

**End of Document**
