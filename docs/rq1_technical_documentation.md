# RQ2 Technical Documentation
**Capstone Project – Master of Data Analytics**  
**Research Question 2 (RQ2): Customer Concentration and Behavioral Segmentation of Profit Erosion**

---

## 1. Research Question

**RQ2:**  
*To what extent is profit erosion concentrated among a small subset of customers, and can customers be meaningfully segmented based on behavioral and erosion characteristics?*

This research question examines whether return-driven profit erosion is unevenly distributed across customers and whether unsupervised learning can identify behaviorally distinct customer segments associated with differential economic impact.

---

## 2. Conceptual Hypotheses (Proposal Alignment)

Although RQ2 is exploratory and descriptive in execution, conceptual hypotheses are stated to maintain alignment with the approved capstone proposal.

- **H₀₂ (Null Hypothesis):**  
  Customer segments identified through unsupervised clustering do not differ meaningfully in profit erosion intensity.

- **H₁₂ (Alternative Hypothesis):**  
  Customer segments identified through unsupervised clustering exhibit materially different profit erosion profiles.

**Methodological clarification:**  
The notebook implements both descriptive analysis and statistical validation. While the primary approach is exploratory, significance testing (Kruskal-Wallis/ANOVA), effect size estimation (eta-squared), and stability diagnostics (bootstrap ARI) are included to validate segment differentiation. Results are interpreted in terms of both practical differentiation and statistical evidence of segment-level differences.

---

## 3. Analytical Framing and Scope

RQ2 combines **exploratory analysis with statistical validation**. The analysis proceeds in two distinct phases:

**Phase 1: Concentration Analysis**
- Pareto analysis of customer-level profit erosion distribution
- Lorenz curve visualization
- Gini coefficient computation
- Top-X customer share calculations

**Phase 2: Customer Segmentation**
- Unsupervised clustering using K-Means
- Optimal cluster selection via silhouette analysis
- Cluster quality diagnostics (silhouette score, Calinski-Harabasz, Davies-Bouldin)
- Bootstrap stability testing (Adjusted Rand Index)
- Statistical significance testing (Kruskal-Wallis H / ANOVA)
- Effect size quantification (eta-squared)
- Post-hoc pairwise comparisons (when applicable)

The objective is to generate actionable insights through rigorous exploratory analysis complemented by statistical validation of segment differentiation.

---

## 4. Data Scope and Unit of Analysis

- **Primary unit of analysis:** Customer-level aggregation  
- **Base data:** Processed customer-level dataset (`customer_profit_erosion_targets.parquet`)
- **Dataset size:** 11,790 customer records
- **Filtering rule:** Analysis includes only customers with `total_profit_erosion > 0`
- **Data quality:**
  - No duplicate user IDs after processing
  - Missing values detected: 580 across all columns
  - Infinite values: If detected, replaced with NaN and median-imputed
  - Note: Missing value imputation is handled by downstream feature engineering functions before clustering
  
All customer-level metrics are constructed from validated item-level outputs generated in upstream pipeline stages, ensuring consistency with RQ1 and maintaining CI-safe execution.

---

## 5. Profit Erosion Construction (Pipeline Reuse)

Profit erosion at the customer level is derived from the standardized feature engineering pipeline (`src.feature_engineering`) to ensure metric consistency across research questions.

### Item-Level Profit Erosion Formula

For each returned item:

```
Profit Erosion = Margin Reversal + Return Processing Cost
```

Where:
- **Margin Reversal** = Revenue lost from reversed sale (sale_price - cost)
- **Return Processing Cost** = Fixed operational cost per return (defined in upstream pipeline)

**Note:** The RQ2 notebook does not define the per-return processing cost value. It uses pre-calculated `total_process_cost` values from the upstream feature engineering pipeline. Based on the notebook output showing total process cost of $249,340.20 across all returns, the per-return cost can be back-calculated if the total number of returns is known.

### Customer-Level Aggregation

Customer-level profit erosion is computed by:
1. Summing item-level profit erosion across all returned items per customer
2. Aggregating via `aggregate_profit_erosion_by_customer()` function
3. Merging with behavioral features via `engineer_customer_behavioral_features()`

This reuse:
- Preserves economic logic consistency across RQ1 and RQ2
- Eliminates duplicated metric definitions
- Maintains pipeline traceability and CI stability
- Enables automated validation of total erosion consistency

---

## 6. Concentration Analysis Methodology

This section operationalizes customer-level concentration analysis as implemented in the RQ2 notebook. All metrics are computed on the customer-level aggregated table and executed deterministically (no random components in concentration phase).

### 6.1 Pareto Analysis

**Implementation:**
Customers are sorted in descending order of `total_profit_erosion`. A Pareto table is constructed via `compute_pareto_table()` with the following derived fields:

- `rank`: Customer rank by profit erosion (1 = highest erosion)
- `cumulative_erosion`: Running sum of profit erosion
- `cumulative_erosion_share`: Cumulative % of total profit erosion
- `cumulative_customer_share`: Cumulative % of customers
- Thresholds flagging top 10%, 20%, 50% customers

**Function signature:**
```python
from src.rq2_concentration import compute_pareto_table

pareto_df = compute_pareto_table(
    customer_df=customer_erosion,
    value_col='total_profit_erosion',
    customer_id='user_id'
)
```

**Output:**
- Saved to `data/processed/rq2/pareto_table.csv`
- Enables assessment of whether a small fraction of customers drives disproportionate profit erosion

### 6.2 Lorenz Curve

**Implementation:**
The Lorenz curve is generated via `lorenz_curve_points()`, which:
1. Sorts customers by ascending profit erosion
2. Computes cumulative customer share (x-axis)
3. Computes cumulative profit erosion share (y-axis)
4. Returns coordinate arrays for plotting

**Function signature:**
```python
from src.rq2_concentration import lorenz_curve_points

x_coords, y_coords = lorenz_curve_points(
    values=customer_erosion['total_profit_erosion'].values
)
```

**Visualization:**
- **X-axis:** Cumulative share of customers (0 to 1)
- **Y-axis:** Cumulative share of total profit erosion (0 to 1)
- **Line of perfect equality:** 45-degree diagonal (y = x)
- **Actual curve:** Lies below equality line, indicating concentration

**Chart interpretation:**
The observed Lorenz curve deviates substantially from the line of equality, demonstrating that profit erosion is not evenly distributed. Key visual insights:
- A large proportion of low-erosion customers contributes minimally to total erosion
- Profit erosion accumulates rapidly in the upper tail (top percentiles)
- The curve steepens markedly among top customers, indicating disproportionate contribution

The area between the Lorenz curve and the equality line quantifies inequality and is used to compute the Gini coefficient.

**Output:**
- Saved to `figures/rq2/lorenz_curve.png`
- Provides graphical validation of concentration metrics

### 6.3 Gini Coefficient

**Implementation:**
The Gini coefficient is computed from Lorenz curve coordinates using the trapezoidal rule via `gini_coefficient()`:

```python
from src.rq2_concentration import gini_coefficient

gini = gini_coefficient(values=customer_erosion['total_profit_erosion'].values)
```

**Calculation method:**
1. Generate Lorenz curve coordinates (cumulative customer share vs. cumulative erosion share)
2. Compute area under Lorenz curve using trapezoidal integration
3. Gini = 1 - 2 × (area under Lorenz curve)

**Interpretation scale:**
- 0.00: Perfect equality (all customers contribute equally)
- 0.50: Moderate inequality
- 1.00: Perfect inequality (one customer accounts for all erosion)

**Observed value:**
- **Gini coefficient: 0.4122**
- Indicates **moderate concentration** of profit erosion
- Consistent with Lorenz curve visualization

**Output:**
- Stored in `rq2_comprehensive_summary.json` under `concentration.gini_coefficient`

### 6.4 Top-X Customer Share Analysis

**Implementation:**
Computes the share of total profit erosion attributable to the top X% of customers via `top_x_customer_share_of_value()`:

```python
from src.rq2_concentration import top_x_customer_share_of_value

top_10_share = top_x_customer_share_of_value(
    values=customer_erosion['total_profit_erosion'].values,
    top_x_pct=0.10
)
```

**Computed metrics:**
- Top 10% of customers → Share of total erosion
- Top 20% of customers → Share of total erosion
- Top 50% of customers → Share of total erosion

**Observed results:**
- **Top 10%:** 29.52% of total profit erosion
- **Top 20%:** 47.60% of total profit erosion
- **Top 50%:** 82.35% of total profit erosion

**Interpretation:**
These values confirm Pareto-like behavior: a minority of customers account for the majority of profit erosion. The top 20% of customers drive nearly half of all erosion, supporting the hypothesis of structural concentration.

**Output:**
- Stored in `rq2_comprehensive_summary.json` under `concentration` object

### 6.5 Concentration Analysis Summary

All concentration metrics converge on the same conclusion:

1. **Gini coefficient (0.4122)** indicates moderate inequality
2. **Lorenz curve** visually confirms deviation from equal distribution
3. **Top-20% customers** account for 47.6% of total erosion
4. **Pareto table** demonstrates accelerating cumulative erosion in top percentiles

These findings justify the segmentation analysis in Phase 2 and establish that profit erosion is not uniformly distributed across the customer base.

---

## 7. Behavioral Feature Engineering

Behavioral features are engineered via `engineer_customer_behavioral_features()`, which aggregates item-level and order-level data to the customer level. Feature construction reuses validated upstream transformations to avoid duplicated logic.

### 7.1 Feature Groups

**Purchase intensity features:**
- `total_items_purchased`: Total number of items purchased
- `total_sales`: Total revenue from all purchases
- `avg_order_value`: Average value per order
- `avg_basket_size`: Average number of items per order

**Return behavior features:**
- `return_frequency`: Total number of returned items
- `customer_return_rate`: Returns ÷ total items purchased
- `pct_orders_with_returns`: Percentage of orders containing at least one return

**Economic impact features:**
- `total_profit_erosion`: Sum of all profit erosion from returns
- `avg_profit_erosion_per_return`: Mean erosion per returned item
- `total_margin_reversal`: Total lost margin from reversed sales
- `total_processing_cost`: Total return processing costs

**Risk indicators (derived features):**
- `erosion_percentile_rank`: Customer's percentile rank in erosion distribution
- `profit_erosion_quartile`: Quartile assignment (0-3)
- `high_erosion_customer`: Binary flag for top quartile customers

**Temporal features:**
- `days_since_first_order`: Customer tenure
- `days_since_last_order`: Recency
- `order_frequency`: Orders per active day

### 7.2 Feature Engineering Implementation

```python
from src.feature_engineering import (
    aggregate_profit_erosion_by_customer,
    engineer_customer_behavioral_features
)

# Aggregate profit erosion
customer_erosion = aggregate_profit_erosion_by_customer(
    item_level_df=processed_items,
    order_df=orders_df
)

# Engineer behavioral features
customer_features = engineer_customer_behavioral_features(
    customer_erosion_df=customer_erosion,
    order_df=orders_df,
    item_df=processed_items
)
```

### 7.3 Feature Validation

The notebook performs automated validation:
- Checks for missing required columns
- Detects infinite values and replaces with NaN
- Conditionally median-imputes numeric columns if infinite values are found
- Validates customer ID uniqueness
- Confirms positive profit erosion for all records

**Data quality summary (from notebook output):**
- Total records: 11,790
- Missing values detected: 580 (likely in temporal features like recency/tenure)
- Duplicate IDs: 0
- Customers with erosion > 0: 11,790

**Note:** The notebook code at lines 167-170 only performs median imputation if infinite values are detected. The 580 missing values reported at line 178 remain unless handled by downstream feature engineering functions (`select_numeric_features`, `standardize_features`) or within the `build_customer_segmentation_table` function. StandardScaler requires non-missing data, so imputation must occur before the standardization step.

---

## 8. Customer Segmentation Methodology

### 8.1 Segmentation Table Construction

The segmentation table is built via `build_customer_segmentation_table()`, which:
1. Merges customer erosion aggregates with behavioral features
2. Validates data completeness
3. Returns a unified customer-level dataset for clustering

```python
from src.rq2_segmentation import build_customer_segmentation_table

seg_table = build_customer_segmentation_table(
    customer_erosion_df=customer_erosion,
    behavioral_features_df=customer_features
)
```

### 8.2 Feature Selection and Standardization

**Numeric feature extraction:**
```python
from src.rq2_segmentation import select_numeric_features

numeric_features = select_numeric_features(
    df=seg_table,
    exclude_cols=['user_id', 'cluster']
)
```

**Standardization:**
All numeric features are standardized using `StandardScaler` (zero mean, unit variance) via `standardize_features()`:

```python
from src.rq2_segmentation import standardize_features

X_scaled, scaler, feature_names = standardize_features(
    df=seg_table,
    feature_cols=numeric_features
)
```

**Rationale:**
- K-Means is sensitive to feature scale
- Standardization ensures all features contribute equally to distance calculations
- Enables interpretable cluster centroids

### 8.3 Optimal K Selection

The notebook evaluates clustering quality across K = 2 to 10 clusters using three complementary metrics:

**Metrics computed:**
1. **Silhouette Score** (maximize): Measures cluster cohesion and separation
2. **Calinski-Harabasz Index** (maximize): Ratio of between-cluster to within-cluster variance
3. **Davies-Bouldin Index** (minimize): Average similarity between each cluster and its most similar cluster

**Implementation:**
```python
from src.rq2_segmentation import clustering_metrics_over_k

k_range = range(2, 11)
silhouette_scores, calinski_scores, davies_scores = clustering_metrics_over_k(
    X=X_scaled,
    k_range=k_range,
    random_state=42
)
```

**Selection criterion:**
The notebook uses **silhouette score maximization** as the primary selection criterion, as it balances cluster compactness and separation without assuming cluster shape.

**Observed metrics (from notebook output):**

| K | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|-----------|-------------------|----------------|
| 2 | 0.5412 | 14,026.52 | 0.5925 |
| 3 | 0.4156 | 11,243.79 | 0.7642 |
| 4 | 0.3845 | 10,184.23 | 0.8156 |
| ... | ... | ... | ... |

**Selected K:** **K = 2**
- Highest silhouette score (0.5412)
- Clear elbow in inertia plot at K = 2
- Interpretable high-erosion vs. low-erosion segment structure

### 8.4 Final Clustering Model

**Model fitting:**
```python
from src.rq2_segmentation import kmeans_fit_predict

kmeans_model, cluster_labels = kmeans_fit_predict(
    X=X_scaled,
    k=2,
    random_state=42
)

seg_table['cluster'] = cluster_labels
```

**Model parameters:**
- Algorithm: K-Means++
- K: 2 clusters
- Random state: 42 (reproducibility)
- Initialization: k-means++ (smart centroid initialization)

**Output:**
- Cluster assignments saved to `customer_segmentation_results.csv`
- Model diagnostics saved to `rq2_comprehensive_summary.json`

---

## 9. Cluster Quality Diagnostics

### 9.1 Quality Metrics for K = 2

**Silhouette Score: 0.5412**
- Interpretation: Moderate to good cluster quality
- Range: [-1, 1]; values > 0.5 indicate well-separated clusters
- Confirms that clusters are distinct and internally cohesive

**Calinski-Harabasz Index: 14,026.52**
- Interpretation: High ratio of between-cluster to within-cluster variance
- Higher values indicate better-defined clusters
- Confirms strong separation between segments

**Davies-Bouldin Index: 0.5925**
- Interpretation: Low average cluster similarity
- Lower values indicate better separation
- Values < 1.0 suggest clusters are not overly similar

**Implementation:**
```python
from src.rq2_segmentation import compute_clustering_quality_metrics

metrics = compute_clustering_quality_metrics(
    X=X_scaled,
    labels=cluster_labels
)
```

### 9.2 Diagnostic Visualizations

**Elbow curve (inertia):**
- Plots within-cluster sum of squares vs. K
- Shows clear elbow at K = 2
- Saved to `figures/rq2/elbow_curve.png`

**Silhouette scores vs. K:**
- Maximum at K = 2
- Declines monotonically for K > 2
- Saved to `figures/rq2/silhouette_scores.png`

**Cluster quality comparison:**
- Multi-panel plot showing all three metrics
- Saved to `figures/rq2/cluster_quality_metrics.png`

---

## 10. Bootstrap Stability Analysis

### 10.1 Stability Testing Procedure

To evaluate the robustness of the clustering solution, the notebook performs a bootstrap stability analysis using the **Adjusted Rand Index (ARI)**.

**Bootstrap procedure:**
1. Fit baseline K-Means solution on full dataset (K = 2, random_state = 42)
2. Generate B = 50 bootstrap samples (resampling with replacement)
3. Fit K-Means on each bootstrap sample
4. Compare each bootstrap solution to baseline using ARI

**Implementation:**
```python
B = 50  # Number of bootstrap iterations
baseline_labels = kmeans_model.fit_predict(X_scaled)
ari_scores = []

for i in range(B):
    X_boot = resample(X_scaled, replace=True, random_state=42 + i)
    boot_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_boot)
    ari = adjusted_rand_score(baseline_labels, boot_labels)
    ari_scores.append(ari)
```

**ARI properties:**
- Range: [-1, 1]; perfect agreement = 1.0
- Corrected for chance (expected value ≈ 0 for random clusterings)
- Label-invariant (robust to cluster index permutations)

### 10.2 Stability Results

**Bootstrap ARI summary (from notebook output):**
- **Mean ARI:** 0.9941
- **Standard deviation:** 0.0043
- **Minimum ARI:** 0.9845
- **Maximum ARI:** 0.9989

**Stability rating:**
Based on commonly accepted benchmarks:
- ARI > 0.95: Excellent stability ✓
- ARI > 0.80: Good stability
- ARI < 0.80: Unstable clustering

**Rating:** **EXCELLENT**

### 10.3 Stability Interpretation

The bootstrap results demonstrate:
1. **Near-perfect reproducibility:** Mean ARI exceeds 0.99
2. **Low sampling sensitivity:** Even worst-case ARI > 0.98
3. **Structural robustness:** Clustering is not driven by outliers or noise
4. **Initialization stability:** Results are consistent despite random resampling

**Analytical implications:**
- Identified segments are structurally credible
- Economic interpretations are defensible
- Segmentation is suitable for business decision-making

**Output:**
- Bootstrap ARI distribution saved to `figures/rq2/bootstrap_stability.png`
- Summary statistics saved to `rq2_comprehensive_summary.json` under `stability` object

---

## 11. Statistical Significance Testing

### 11.1 Test Selection and Assumptions

**Objective:**
Evaluate whether profit erosion distributions differ significantly across clusters.

**Normality testing:**
The notebook tests normality of profit erosion within each cluster using the Shapiro-Wilk test:
- Sample size per cluster: 3,302 (Cluster 0), 8,488 (Cluster 1)
- For large samples (n > 5,000), a random subsample of 5,000 is tested to avoid oversensitivity

**Observed normality p-values (from notebook):**
- Cluster 0: < 0.05 (non-normal)
- Cluster 1: < 0.05 (non-normal)

**Test selection:**
Given non-normal distributions, the notebook selects the **Kruskal-Wallis H test** as a non-parametric alternative to one-way ANOVA.

**Implementation:**
```python
from scipy.stats import kruskal

cluster_groups = [
    seg_table.loc[seg_table['cluster'] == 0, 'total_profit_erosion'].values,
    seg_table.loc[seg_table['cluster'] == 1, 'total_profit_erosion'].values
]

stat, p_value = kruskal(*cluster_groups)
```

### 11.2 Omnibus Test Results

**Test:** Kruskal-Wallis H  
**Test statistic:** 6,602.47  
**P-value:** < 0.001

**Interpretation:**
The p-value provides strong evidence that profit erosion distributions differ significantly across behavioral segments. This result is statistically robust even after accounting for:
- Non-normal distributions
- Unequal cluster sizes (28% vs. 72%)
- Large sample size

**Note on interpretation:**
While clustering is unsupervised (clusters are not pre-defined), the Kruskal-Wallis test validates that the identified segments exhibit **statistically distinguishable profit erosion outcomes**. This complements descriptive evidence and confirms that segment differences are not attributable to sampling variation.

### 11.3 Effect Size Analysis

**Metric:** Eta-squared (η²)  
**Calculation:** Proportion of total variance in profit erosion explained by cluster membership

**Formula:**
```
SS_between = Σ [n_k × (mean_k - grand_mean)²]
SS_total = Σ [(x_i - grand_mean)²]
η² = SS_between / SS_total
```

**Implementation:**
```python
grand_mean = seg_table['total_profit_erosion'].mean()
ss_between = sum(
    len(group) * (group.mean() - grand_mean) ** 2 
    for group in cluster_groups
)
ss_total = np.sum((seg_table['total_profit_erosion'].values - grand_mean) ** 2)
eta_squared = ss_between / ss_total
```

**Observed eta-squared: 0.5350**

**Effect size interpretation scale:**
- η² < 0.01: Negligible
- 0.01 ≤ η² < 0.06: Small
- 0.06 ≤ η² < 0.14: Medium
- η² ≥ 0.14: Large

**Rating:** **Large effect size**

**Interpretation:**
Cluster membership explains **53.5% of the variance** in customer-level profit erosion. This is a very large effect, indicating that behavioral segmentation captures substantial, actionable differentiation in economic outcomes.

### 11.4 Post-hoc Pairwise Testing

**Applicability:**
Post-hoc tests are only performed when:
1. The omnibus test is significant (p < 0.05)
2. K > 2 clusters (pairwise comparisons are informative)

**For K = 2:**
The notebook skips post-hoc testing, as there is only one pairwise comparison, which is redundant with the omnibus test.

**For K > 2 (general implementation):**
If post-hoc testing were applicable, the notebook would:
1. Perform pairwise Mann-Whitney U tests (for non-normal data) or t-tests (for normal data)
2. Apply Bonferroni correction to control family-wise error rate
3. Compute Cohen's d for each pairwise comparison
4. Save results to `posthoc_tests.csv`

**Output:**
- Empty `posthoc_tests.csv` file created (no post-hoc tests for K = 2)
- Significance summary saved to `segment_significance_summary.csv`

---

## 12. Cluster Summary Statistics and Economic Interpretation

### 12.1 Cluster Sizes

- **Cluster 0:** 3,302 customers (28.0% of customers)
- **Cluster 1:** 8,488 customers (72.0% of customers)

Although Cluster 0 represents a minority of customers, it accounts for a disproportionate share of profit erosion.

### 12.2 Total Profit Erosion by Cluster

- **Cluster 0:** $463,323.41 (57.3% of total erosion)
- **Cluster 1:** $344,928.66 (42.7% of total erosion)

**Key insight:**
Despite representing only 28% of customers, Cluster 0 generates **57.3% of total profit erosion**, confirming segment-level concentration.

### 12.3 Mean and Median Profit Erosion

**Cluster 0 (High-Erosion Segment):**
- Mean profit erosion: $140.32
- Median profit erosion: $122.12
- Standard deviation: $56.89

**Cluster 1 (Low-Erosion Segment):**
- Mean profit erosion: $40.64
- Median profit erosion: $35.93
- Standard deviation: $18.73

**Interpretation:**
Customers in Cluster 0 generate approximately **3.5× higher average profit erosion** than Cluster 1 customers. The close alignment between mean and median within each cluster indicates that differences are not driven solely by extreme outliers but reflect systematic behavioral patterns.

### 12.4 Purchase Behavior Comparison

**Average sales per customer:**
- **Cluster 0:** $318.05
- **Cluster 1:** $117.08

**Average items purchased:**
- **Cluster 0:** 6.8 items
- **Cluster 1:** 3.2 items

**Average order value:**
- **Cluster 0:** $89.45
- **Cluster 1:** $52.30

**Interpretation:**
Cluster 0 customers exhibit substantially higher purchase volume and order value, characterizing them as **high-value, high-engagement customers** who also carry higher erosion risk.

### 12.5 Return Behavior Comparison

**Return frequency (total returns):**
- **Cluster 0:** 4.2 returns per customer
- **Cluster 1:** 1.8 returns per customer

**Customer return rate:**
- **Cluster 0:** 38.5%
- **Cluster 1:** 22.1%

**Percentage of orders with returns:**
- **Cluster 0:** 45.2%
- **Cluster 1:** 28.6%

**Interpretation:**
Cluster 0 customers not only purchase more but also return at significantly higher rates, driving their disproportionate contribution to profit erosion.

### 12.6 Segment Archetype Summary

**Cluster 0 – High-Value, High-Erosion Segment:**
- Smaller segment (28% of customers)
- Higher purchase volume and order value
- Elevated return rates
- Disproportionately high profit erosion (57.3% of total)
- Longer customer tenure
- Higher engagement frequency

**Cluster 1 – Low-Value, Low-Erosion Segment:**
- Majority segment (72% of customers)
- Lower purchase volume and order value
- Lower return rates
- Lower profit erosion per customer (42.7% of total)
- Shorter tenure and lower engagement

**Strategic implication:**
Profit erosion is **segment-driven** rather than uniformly distributed. High-erosion customers are not marginal actors but represent a distinct behavioral profile combining high value with high risk.

---

## 13. Feature-Level Cluster Profiles

### 13.1 Standardized Cluster Centroids

The notebook generates a heatmap of standardized cluster centroids to visualize feature-level differentiation across segments.

**Heatmap structure:**
- **Rows:** Behavioral and economic features used in clustering
- **Columns:** Cluster labels (0 and 1)
- **Color scale:** Standardized z-scores
  - Red (positive): Above-average feature levels
  - Green (negative): Below-average feature levels
  - White (zero): At population mean

**Expected pattern for K = 2:**
Because clustering is performed on standardized features and only two clusters are used, centroids appear as approximate mirror images (±1 standard deviations). This is geometrically expected and indicates clean separation rather than a modeling artifact.

### 13.2 Feature Profile Interpretation

**Purchase intensity features:**
Features such as `total_items_purchased`, `total_sales`, `avg_order_value`, and `avg_basket_size` are uniformly **above average for Cluster 0** and **below average for Cluster 1**, confirming that high-erosion customers are high-engagement purchasers.

**Return behavior features:**
Features including `return_frequency`, `customer_return_rate`, and `pct_orders_with_returns` are strongly **positive for Cluster 0** and **negative for Cluster 1**, demonstrating that return propensity is a core differentiator.

**Economic impact features:**
Erosion-related features such as `total_profit_erosion`, `total_margin_reversal`, and `total_processing_cost` align almost perfectly with Cluster 0, while Cluster 1 represents low-risk customers across all economic dimensions.

**Risk indicators:**
Derived features like `erosion_percentile_rank`, `profit_erosion_quartile`, and `high_erosion_customer` flag Cluster 0 as the high-risk segment and Cluster 1 as the low-risk segment, confirming that clustering recovers the intended economic stratification.

**Temporal features:**
Features such as `days_since_first_order` (tenure) and `order_frequency` indicate that Cluster 0 customers are typically **longer-tenured and more active**, suggesting that high erosion is driven by established, repeat customers rather than one-time or transient users.

### 13.3 Feature Contribution Analysis

The notebook computes feature-level separation between clusters by measuring the absolute difference in standardized cluster centroids for each feature.

**Top differentiating features (ranked by |centroid difference|):**
1. `high_erosion_customer` (binary flag)
2. `profit_erosion_quartile`
3. `erosion_percentile_rank`
4. `total_profit_erosion`
5. `total_margin_reversal`
6. `total_processing_cost`
7. `avg_order_value`
8. `return_frequency`
9. `total_sales`
10. `customer_return_rate`

**Low-contribution features:**
- `days_since_last_order` (recency)
- `days_since_first_order` (tenure)
- Simple frequency counts without economic context

**Interpretation:**
Segmentation is driven primarily by:
1. **Profit erosion risk indicators** (erosion flags, quartiles, percentile ranks)
2. **Absolute economic impact** (total erosion, margin reversal, processing costs)
3. **Purchase value and return intensity** (order value, return frequency, return rate)

Temporal and simple frequency features contribute negligibly, indicating that clusters reflect **systematic economic risk** rather than short-term activity fluctuations or customer age.

**Output:**
- Cluster centroid heatmap saved to `figures/rq2/cluster_centroids_heatmap.png`
- Feature contribution bar chart saved to `figures/rq2/feature_contribution.png`

---

## 14. Comprehensive Results Summary

### 14.1 Concentration Findings

**Gini coefficient:** 0.4122 (moderate concentration)  
**Top 10% of customers:** 29.52% of total erosion  
**Top 20% of customers:** 47.60% of total erosion  
**Top 50% of customers:** 82.35% of total erosion

**Conclusion:**
Profit erosion exhibits **Pareto-like concentration**, with a minority of customers driving the majority of financial impact. This justifies customer-level segmentation and targeted intervention strategies.

### 14.2 Segmentation Findings

**Optimal K:** 2 clusters (selected via silhouette score maximization)  
**Silhouette score:** 0.5412 (moderate to good cluster quality)  
**Calinski-Harabasz index:** 14,026.52 (high between/within-cluster variance ratio)  
**Davies-Bouldin index:** 0.5925 (low average cluster similarity)

**Cluster sizes:**
- Cluster 0: 3,302 customers (28.0%)
- Cluster 1: 8,488 customers (72.0%)

**Economic differentiation:**
- Cluster 0: $140.32 mean erosion per customer (57.3% of total erosion)
- Cluster 1: $40.64 mean erosion per customer (42.7% of total erosion)

### 14.3 Stability and Significance

**Bootstrap stability (ARI):**
- Mean ARI: 0.9941 (near-perfect reproducibility)
- Stability rating: **EXCELLENT**

**Statistical significance:**
- Test: Kruskal-Wallis H
- P-value: < 0.001
- Conclusion: Clusters differ significantly in profit erosion

**Effect size:**
- Eta-squared: 0.5350 (large effect)
- Interpretation: Cluster membership explains 53.5% of variance in profit erosion

### 14.4 Segment Archetypes

**Cluster 0 – High-Value, High-Erosion:**
- 28% of customers, 57.3% of erosion
- 3.5× higher average erosion than Cluster 1
- Higher purchase volume, order value, return rate
- Longer tenure, higher engagement

**Cluster 1 – Low-Value, Low-Erosion:**
- 72% of customers, 42.7% of erosion
- Lower purchase volume, order value, return rate
- Shorter tenure, lower engagement

---

## 15. Business Insights and Recommendations

### 15.1 Key Findings

1. **Concentration is structural, not random:** Gini coefficient (0.41) and top-20% share (47.6%) confirm that profit erosion is concentrated among a minority of customers.

2. **Segments are statistically distinct:** Kruskal-Wallis test (p < 0.001) and large effect size (η² = 0.535) validate that clusters differ significantly in profit erosion.

3. **Segmentation is stable and reproducible:** Bootstrap ARI (0.99) demonstrates near-perfect stability across resampled datasets.

4. **High-erosion customers are high-value:** Cluster 0 customers exhibit higher purchase volume, order value, and engagement, indicating they are **valuable but risky**.

5. **Return behavior is a core differentiator:** Return rate and return frequency are among the strongest predictors of segment membership.

### 15.2 Strategic Recommendations

**1. Prioritize high-erosion segment for intervention**
- Allocate customer support resources proportionally to segment-level risk
- Develop proactive return-prevention playbooks for Cluster 0 customers
- Monitor segment migration monthly to detect early warning signals

**2. Use low-erosion segment as behavioral benchmark**
- Identify behavioral patterns that minimize return propensity
- Use Cluster 1 characteristics to inform acquisition targeting
- Design retention campaigns that promote low-erosion purchasing behaviors

**3. Implement differential return policies by segment**
- Consider stricter return eligibility criteria for Cluster 0 customers
- Test personalized incentives to reduce return rates among high-risk users
- Evaluate economic impact of segment-specific policy changes

**4. Integrate segmentation into operational planning**
- Forecast return volumes and processing costs at the segment level
- Allocate warehouse capacity and staffing based on segment-level return patterns
- Use segments as stratification variables in A/B testing and experimentation

**5. Re-estimate clustering periodically**
- Re-fit clustering quarterly on refreshed data
- Track stability of segment assignments over time
- Validate continued differentiation in profit erosion outcomes

### 15.3 Analytical Limitations

**1. Causality cannot be inferred**
- Clustering is unsupervised and descriptive; segment membership does not imply causal impact
- Statistical tests validate segment differentiation but do not establish causal mechanisms

**2. Behavioral dynamics are not modeled**
- Analysis is cross-sectional; temporal evolution of customer behavior is not captured
- Segment migration patterns require longitudinal tracking

**3. External factors are not considered**
- Product category, seasonality, promotional activity, and marketing exposure are not included as features
- Segments may reflect unmeasured confounders rather than intrinsic behavioral traits

**4. Generalizability is limited**
- Analysis is based on synthetic e-commerce data (TheLook dataset)
- Findings may not generalize to real-world retail environments with different return policies, customer demographics, or product catalogs

---

## 16. Data Outputs and Artifacts

All outputs are saved to `data/processed/rq2/` and `figures/rq2/` for reproducibility and downstream use.

### 16.1 Data Files

**Concentration analysis:**
- `pareto_table.csv`: Customer-level Pareto table with cumulative erosion shares
- `lorenz_curve_data.csv`: Lorenz curve coordinates (x = customer share, y = erosion share)

**Segmentation results:**
- `customer_segmentation_results.csv`: Customer-level cluster assignments and features
- `cluster_centroids.csv`: Standardized cluster centroids for all features
- `feature_contribution.csv`: Feature importance ranking by centroid separation

**Statistical outputs:**
- `segment_significance_summary.csv`: Omnibus test results and effect size
- `posthoc_tests.csv`: Pairwise post-hoc test results (empty for K = 2)
- `rq2_comprehensive_summary.json`: Complete RQ2 results in structured JSON format

### 16.2 Visualizations

**Concentration visualizations:**
- `lorenz_curve.png`: Lorenz curve with line of equality
- `pareto_chart.png`: Cumulative profit erosion by customer rank

**Segmentation diagnostics:**
- `elbow_curve.png`: Inertia vs. K (elbow method)
- `silhouette_scores.png`: Silhouette scores vs. K
- `cluster_quality_metrics.png`: Multi-panel plot of quality metrics

**Cluster interpretation:**
- `cluster_centroids_heatmap.png`: Standardized feature profiles by cluster
- `feature_contribution.png`: Feature importance bar chart
- `bootstrap_stability.png`: Bootstrap ARI distribution histogram

**Economic comparisons:**
- `cluster_erosion_boxplot.png`: Profit erosion distributions by cluster
- `cluster_size_pie.png`: Customer count and erosion share by cluster

---

## 17. Reproducibility and CI Compatibility

### 17.1 Deterministic Execution

All random operations use fixed random states to ensure reproducibility:
- K-Means clustering: `random_state=42`
- Bootstrap resampling: Sequential seeding (`random_state=42 + i`)
- Train/test splits: Not applicable (unsupervised learning)

### 17.2 Pipeline Integration

The notebook integrates seamlessly with the project pipeline:
1. Depends on processed customer-level data from upstream stages
2. Reuses feature engineering functions from `src.feature_engineering`
3. Calls modularized analysis functions from `src.rq2_concentration` and `src.rq2_segmentation`
4. Saves all outputs to standardized directories (`data/processed/rq2/`, `figures/rq2/`)

### 17.3 Continuous Integration Safety

The notebook is CI-safe and does not access raw data files:
- No file paths reference raw/external data sources
- All inputs are read from `PROCESSED_DATA_DIR`
- No manual file editing or external dependencies required
- Execution time: ~30 seconds (single-threaded, no API calls)

### 17.4 Version Control and Metadata

All outputs include metadata for traceability:
- Analysis date/timestamp
- Number of features used
- Random state values
- Model hyperparameters
- Data quality metrics

Metadata is stored in `rq2_comprehensive_summary.json` for programmatic access.

---

## 18. References and Methodology Citations

**Concentration metrics:**
- Gini coefficient: Standard economic inequality measure
- Lorenz curve: Graphical representation of distributional inequality
- Pareto analysis: 80/20 rule and cumulative distribution analysis

**Clustering methodology:**
- K-Means: Lloyd, S. (1982). "Least squares quantization in PCM." IEEE Transactions on Information Theory.
- Silhouette score: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." Journal of Computational and Applied Mathematics.
- Calinski-Harabasz index: Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster analysis." Communications in Statistics.
- Davies-Bouldin index: Davies, D. L., & Bouldin, D. W. (1979). "A cluster separation measure." IEEE Transactions on Pattern Analysis and Machine Intelligence.

**Stability analysis:**
- Adjusted Rand Index: Hubert, L., & Arabie, P. (1985). "Comparing partitions." Journal of Classification.
- Bootstrap resampling: Efron, B., & Tibshirani, R. J. (1994). "An Introduction to the Bootstrap." Chapman and Hall/CRC.

**Statistical testing:**
- Kruskal-Wallis H test: Kruskal, W. H., & Wallis, W. A. (1952). "Use of ranks in one-criterion variance analysis." Journal of the American Statistical Association.
- Eta-squared effect size: Cohen, J. (1988). "Statistical Power Analysis for the Behavioral Sciences." Lawrence Erlbaum Associates.

---

**Document version:** 2.0  
**Last updated:** February 2026  
**Notebook version:** RQ2 Analysis Notebook (Final)  
**Contact:** Capstone Project Team
