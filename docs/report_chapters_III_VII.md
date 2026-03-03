# Capstone Report — Chapters III through VII

**Course:** DAMO-699-4 Capstone Project
**Institution:** University of Niagara Falls, Canada
**Professor:** Omid Isfahanialamdari
**Date:** February 2026

---

# Chapter III: Statistical Analysis — RQ1

## 3.1 Overview

Research Question 1 asks whether statistically significant differences in profit erosion exist across product categories and brands. This chapter presents the non-parametric hypothesis tests, post-hoc analysis, and bootstrap confidence intervals used to answer that question, followed by the descriptive findings characterizing the distribution of profit erosion across the product catalog.

## 3.2 Method Selection

Profit erosion per returned item is strongly right-skewed with a heavy tail (range: $0.70–$95.70; distribution confirmed non-normal by visual inspection of log-scale histogram). Parametric ANOVA assumes normally distributed group residuals, making it inappropriate here. The **Kruskal-Wallis H test** was selected as the non-parametric equivalent of one-way ANOVA. It tests whether group samples were drawn from the same distribution without requiring normality, making it robust to the skewed structure of the profit erosion outcome. Effect size is reported as **epsilon-squared (ε²)**, the rank-based analog of eta-squared.

Where the omnibus test is significant, pairwise post-hoc analysis was conducted using the **Dunn test with Bonferroni correction** to control family-wise error rate across all category and brand pairs.

## 3.3 Hypothesis Tests

### 3.3.1 Category-Level Test

**H₀:** Mean profit erosion per returned item is equal across all product categories.
**H₁:** At least one pair of product categories differs significantly in mean profit erosion.

| Statistic | Value |
|-----------|-------|
| Test | Kruskal-Wallis H |
| H statistic | — |
| p-value | 2.63 × 10⁻³³ |
| Effect size (ε²) | 0.454 |
| Interpretation | Large effect |

**Decision: Reject H₀.** The probability of observing this result under the null is effectively zero. The effect size of ε² = 0.454 indicates that category membership accounts for approximately 45% of the variance in profit erosion ranks — a large and practically meaningful difference.

### 3.3.2 Brand-Level Test

**H₀:** Mean profit erosion per returned item is equal across all brands.
**H₁:** At least one pair of brands differs significantly in mean profit erosion.

| Statistic | Value |
|-----------|-------|
| Test | Kruskal-Wallis H |
| p-value | 1.08 × 10⁻⁴ |
| Effect size (ε²) | 0.442 |
| Interpretation | Large effect |

**Decision: Reject H₀.** Brand-level differences in profit erosion are statistically significant with a large effect, confirming that the brand dimension carries independent explanatory power beyond product category alone.

## 3.4 Descriptive Findings

### 3.4.1 Category-Level Distribution

The five categories generating the highest total profit erosion in the TheLook dataset are:

| Rank | Category | Total Erosion | Driver |
|------|----------|--------------|--------|
| 1 | Outerwear & Coats | ~$2,000 | High severity + moderate volume |
| 2 | Sweaters | ~$1,600 | High per-item margin reversal |
| 3 | Jeans | ~$1,400 | Premium tier processing cost |
| 4 | Suits & Sport Coats | ~$1,300 | Highest per-item severity |
| 5 | Pants | ~$1,200 | High volume |

A critical finding emerges from decomposing total erosion into its volume and severity components: return rate (frequency per customer) is a weak predictor of financial risk at the category level. Outerwear generates high total erosion primarily because each individual return is costly (high severity), not because customers return items at unusually high rates. This confirms the finding of Petersen and Kumar (2009) that return volume metrics alone are insufficient to characterize economic impact.

### 3.4.2 Department-Level Asymmetry

At the department level, Men's apparel generates $10,700 in total profit erosion versus $8,100 for Women's — a 32% gap — driven primarily by the concentration of premium categories (Suits, Outerwear, Jeans) in the Men's department.

### 3.4.3 Bootstrap Confidence Intervals

Bootstrap 95% confidence intervals for category-level mean profit erosion show limited overlap across the highest-risk categories, confirming that the Kruskal-Wallis result is not driven by a small number of extreme outliers and that the rank ordering of categories is statistically stable.

## 3.5 External Validation

RQ1 findings were directionally validated using SSL return data (133,800 returned order lines, B2B educational supplies). The Kruskal-Wallis test on SSL data yields p ≈ 0.000 at both the category level (mapped from SSL product class) and the brand level (mapped from SSL supplier), with H₀ rejected in both cases. This confirms that the finding — profit erosion differs significantly across product categories and brands — generalizes to real-world return data from a structurally different sector.

## 3.6 Summary

All four null hypotheses tested under RQ1 are rejected. Profit erosion is not uniformly distributed across the product catalog; it is concentrated in premium categories with high margin reversal risk. Category membership explains approximately 45% of rank-variance in profit erosion, making category the single most important structural variable for return-risk management.

---

# Chapter IV: Advanced Analytics — RQ2, RQ3, and RQ4

## 4.1 Overview

This chapter presents three advanced analytical methods applied to the customer-level profit erosion dataset: unsupervised customer segmentation (RQ2), machine learning classification (RQ3), and econometric regression (RQ4). Each method answers a distinct dimension of the central research problem: *who* drives concentrated losses (RQ2), *which* customers can be identified in advance (RQ3), and *what* behavioral mechanisms drive erosion magnitude (RQ4).

---

## 4.2 RQ2: Customer Behavioral Segmentation

### 4.2.1 Concentration Analysis

Before segmentation, the concentration of profit erosion across the customer base was quantified using the **Gini coefficient** and **Lorenz curve**. The Gini coefficient measures the degree to which the distribution departs from perfect equality (Gini = 0) toward perfect concentration (Gini = 1).

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Gini coefficient | 0.4122 | Moderate concentration (threshold: > 0.30) |
| Pareto ratio | Top 20% → 47.6% of erosion | Near-Pareto concentration |
| Bootstrap p-value | < 0.001 | Concentration is statistically significant |

The Pareto ratio of 47.6% approaches but does not fully match the classical 80/20 rule, indicating that erosion is meaningfully concentrated but not extreme. A small number of high-activity customers drive disproportionate losses; targeted intervention on the top quintile would address nearly half of total erosion without requiring broad policy changes.

### 4.2.2 K-Means Clustering

**Feature preparation:** Eight candidate behavioral features were constructed at the customer level. Variance and correlation screening reduced this to seven survivors (dropping `total_margin`, which showed Pearson r > 0.85 with `total_sales`). All features were standardized prior to clustering.

**Cluster selection:** K-Means was evaluated for k = 2 through k = 8. The optimal solution was k = 2, which achieved the highest silhouette score (0.2844). The relatively moderate silhouette score reflects a behavioral continuum rather than sharply discrete customer typologies — a finding consistent with the literature on customer segmentation in e-commerce (Cui et al., 2020).

**Cluster profiles (n = 11,790 returners; $808,300 total profit erosion):**

| Characteristic | Cluster 0 — High-Activity | Cluster 1 — High-Return-Rate |
|---|---|---|
| Size | 4,302 customers (36.5%) | 7,488 customers (63.5%) |
| Mean profit erosion | $95.51 | $53.07 |
| Share of total erosion | 50.8% | 49.2% |
| Mean order frequency | 2.99 | 1.42 |
| Mean return rate | 0.40 | 0.82 |
| Mean order value | $125.70 | $63.97 |
| Mean purchase recency | 317 days | 558 days |

**Cluster 0 (High-Activity)** comprises the minority of customers (36.5%) but accounts for over half of total erosion. These customers order frequently, spend more per order, and return at moderate rates. Their erosion is driven by volume and value, not by return rate alone.

**Cluster 1 (High-Return-Rate)** comprises the majority of customers (63.5%) with a mean return rate of 82% — nearly every item they purchase is returned. Despite this extreme return behavior, their total erosion contribution is roughly equal to Cluster 0, because their individual orders are smaller and less frequent.

**Statistical significance of cluster differences:** One-Way ANOVA confirms that the two clusters differ significantly in mean profit erosion (F = 1,479.64, p < 0.001, η² = 0.1115 — medium effect).

### 4.2.3 External Validation

SSL account-level data (13,616 accounts) was used to assess whether the same behavioral features that drive segmentation in TheLook also show discriminating power in a real-world B2B context. Five of ten core behavioral features agreed between datasets (50% pattern agreement), with `customer_return_rate`, `avg_basket_size`, `avg_order_value`, and `total_margin` all passing in both TheLook and SSL. `order_frequency` failed in SSL, likely because institutional B2B accounts place fewer, larger orders, reducing its discriminating power relative to the B2C context.

---

## 4.3 RQ3: Predictive Modeling

### 4.3.1 Pipeline and Leakage Prevention

The RQ3 pipeline follows a strict sequence designed to prevent data leakage — the inadvertent inclusion of information derived from the target variable in the training features (Rosenblatt et al., 2024; Kaufman et al., 2012). Six features were excluded from all modeling as direct components or derivatives of the target (`total_profit_erosion`, `total_margin_reversal`, `total_process_cost`, `profit_erosion_quartile`, `erosion_percentile_rank`, `user_id`).

The 12 remaining candidate features were subjected to a three-gate sequential screening process applied exclusively to the training set:

| Gate | Method | Dropped Features |
|------|--------|-----------------|
| 1. Variance | VarianceThreshold < 0.01 | None |
| 2. Correlation | Pearson \|r\| > 0.85 | `order_frequency`, `total_sales`, `avg_item_price` |
| 3. Univariate | Point-biserial p > 0.05 (Bonferroni) | `customer_tenure_days`, `purchase_recency_days` |

Seven of 12 features survived to model training: `return_frequency`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, and `customer_return_rate`.

The dataset (11,988 customers with ≥1 return; 25% positive class) was split 80/20 using stratified sampling. Three classifiers were trained with GridSearchCV using stratified 5-fold cross-validation.

### 4.3.2 Model Results

| Model | CV AUC | Test AUC | Precision | Recall | F1 |
|-------|--------|----------|-----------|--------|----|
| **Random Forest ★** | 0.9792 | **0.9798** | 0.7822 | 0.9115 | 0.8419 |
| Gradient Boosting | 0.9797 | 0.9795 | 0.7801 | 0.9299 | 0.8484 |
| Logistic Regression | 0.9646 | 0.9687 | 0.7591 | 0.9048 | 0.8256 |

**H₀** (best model AUC ≤ 0.70): **Rejected.** The Random Forest champion achieves a Test AUC of 0.9798, exceeding the minimum threshold of 0.70 by 0.28. All three models independently exceed the threshold, ruling out model-specific artifacts.

**Champion selection:** Random Forest was designated champion on the basis of highest Test AUC (0.9798) with minimal overfitting (CV–test gap = 0.0006). The near-equivalence of all three models (AUC spread = 0.011) is itself a finding: profit erosion is sufficiently well-structured in behavioral features that even the simplest classifier (Logistic Regression, AUC = 0.9687) greatly exceeds the operational threshold.

**Business context for model selection:** In practice, the preferred model depends on intervention economics. Where interventions are cheap and scalable (e.g., automated email), Gradient Boosting's higher Recall (0.9299) minimizes missed high-erosion customers. Where interventions are costly per customer (e.g., account manager calls), Random Forest's higher Precision (0.7822) minimizes wasted contacts. Logistic Regression is preferred where probability calibration or regulatory interpretability is required (Elkan, 2001; Verbeke et al., 2012).

### 4.3.3 Feature Importance

Across all three models, the following features consistently rank as the strongest predictors of high profit erosion:

| Feature | Consensus Rank | Interpretation |
|---------|---------------|---------------|
| `return_frequency` | Top tier | Volume of return events drives erosion directly |
| `avg_order_value` | Top tier | Higher-value orders generate larger margin reversals |
| `total_margin` | Top tier | Captures cumulative financial exposure |
| `avg_item_margin` | Mid tier | Per-item margin determines reversal magnitude |
| `customer_return_rate` | Mid tier | Behavioral propensity signal |

`return_frequency`, `avg_order_value`, and `total_margin` appear in the top three across at least two of three models, indicating robust predictive signal independent of algorithm.

### 4.3.4 Error Analysis

The Random Forest champion's confusion matrix on the held-out test set (n = 2,398):

| | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | 1,647 (True Negatives) | 152 (False Positives) |
| **Actual High** | 53 (False Negatives) | 546 (True Positives) |

The critical business error is the False Negative (FN) — high-erosion customers who receive no intervention. With only 53 FNs (8.8% miss rate), the model captures 91.2% of the highest-risk customers. The False Positive rate of 8.4% represents wasted intervention expenditure but carries a lower unit cost than a missed high-erosion customer (Verbeke et al., 2012).

### 4.3.5 Sensitivity Analysis

The core conclusion is robust to parameter variation across 11 tested scenarios:

- **Processing cost sensitivity ($8–$18):** Test AUC ranges from 0.9759 to 0.9810. The 0.5% of customers whose classification label changes across cost assumptions are near-threshold cases with negligible strategic impact.
- **Threshold sensitivity (50th–90th percentile):** Test AUC ranges from 0.9664 to 0.9879. The hypothesis conclusion (AUC > 0.70) holds at every threshold tested.

### 4.3.6 External Validation — School Specialty LLC

The Random Forest champion was applied to 13,616 SSL accounts using the same seven surviving features mapped to SSL-equivalent fields:

| Validation Level | Metric | Value |
|---|---|---|
| Feature pattern agreement | 7/12 features consistent | 58.3% |
| Directional accuracy | Predicted vs. actual high-loss | 76.4% |
| Rank correlation | Spearman ρ | 0.7526 (p ≈ 0.00) |
| Recall (SSL) | High-loss accounts captured | 64.1% |
| Specificity (SSL) | Low-loss accounts cleared | 80.5% |

A Spearman ρ of 0.7526 indicates strong positive monotonic alignment between the model's predicted risk scores and observed loss outcomes in an independent real-world dataset. This constitutes meaningful evidence of external transportability (Steyerberg & Harrell, 2016), particularly given the structural difference between B2C fashion retail (TheLook) and B2B educational supplies (SSL).

---

## 4.4 RQ4: Behavioral Econometrics

### 4.4.1 Method and Specification

RQ4 quantifies the marginal association between key behavioral variables and profit erosion magnitude using **Ordinary Least Squares (OLS) regression**. Unlike the classification approach in RQ3, the regression models the continuous outcome `total_profit_erosion` (or its log-transform), allowing coefficient-level interpretation of how much profit erosion changes per unit change in each predictor after controlling for confounds.

**Model specification — Log-Linear OLS (primary):**

> log(total_profit_erosion) ~ return_frequency + avg_basket_size + purchase_recency_days + avg_order_value + customer_return_rate + customer_tenure_days + age + gender + 25 category dummies

The log-linear form was selected because profit erosion is strongly right-skewed at the customer level. Log-transforming the outcome reduces the Jarque-Bera normality test statistic from 619,317 (linear OLS) to 2,198 (log-linear OLS) — a 281.8× improvement — substantially better satisfying OLS residual assumptions. Coefficients from the log-linear model are interpreted as the **percentage change in profit erosion** per unit change in the predictor.

Heteroscedasticity was confirmed by the Breusch-Pagan test (BP = 1,556, p < 0.001); HC3 heteroscedasticity-robust standard errors were applied throughout. The sample is 11,694 customers with ≥1 return (after listwise deletion on regression covariates). The model includes 35 parameters (7 numeric predictors + 26 category dummies + intercept).

### 4.4.2 Results

**Overall model fit:** R² = 0.7765. The model explains 77.7% of variance in log-profit-erosion, indicating strong explanatory power for a cross-sectional behavioral dataset.

**Hypothesis predictor coefficients (log-linear model):**

| Predictor | log β | % Effect | SE | t | p |
|-----------|-------|----------|----|---|---|
| `return_frequency` | +0.4454 | **+56.1%** | 0.0067 | +66.6 | < 0.0001 |
| `avg_basket_size` | −0.1559 | **−14.4%** | 0.0057 | −27.5 | < 0.0001 |
| `purchase_recency_days` | −0.0009 | ~0% | 0.0026 | −0.22 | 0.824 |

Each additional unit increase in return frequency is associated with a 56.1% increase in total profit erosion, holding all other predictors constant. This is the largest effect in the model. Each unit increase in average basket size is associated with a 14.4% decrease in erosion — customers who purchase more items per order tend to select lower-unit-margin items, reducing the average margin reversal per return. Purchase recency days shows no statistically significant marginal association with profit erosion after controlling for frequency and basket behavior (t = −0.22, p = 0.824).

**Control variable findings:** `avg_order_value` is equally significant (t ≈ +66; p < 0.0001), confirming that higher-value orders generate larger margin reversals when returned. Demographic controls (`age`, `gender`, `customer_tenure_days`) are not significant after behavioral and category controls are included, consistent with the hypothesis that return behavior — not demographic characteristics — is the primary driver of profit erosion.

**Category effects:** 20 of 25 category dummy variables are statistically significant. Premium categories (Suits, Outerwear, Jeans, Sweaters, Dresses) carry positive coefficients; commodity categories (Socks, Underwear, Leggings, Tops & Tees) carry negative coefficients, consistent with the tier multiplier structure of the cost model.

**Hypothesis test outcome:** H₀ (behavioral variables exhibit no statistically significant marginal associations) is **rejected**. The joint F-test on the hypothesis predictor block (p < 0.0001) and individual t-tests on `return_frequency` and `avg_basket_size` both support rejection.

### 4.4.3 Practical Magnitude

For the median returner ($47.20 total erosion), a one-standard-deviation increase in return frequency is associated with an increase to approximately $86.36 — an 83% rise — illustrating why frequent returners are the primary target for profit erosion intervention. Conversely, a one-standard-deviation increase in basket size is associated with a 30% decrease in erosion for the mean-erosion customer, suggesting that encouraging multi-item purchases may reduce per-return risk.

### 4.4.4 External Validation — School Specialty LLC

The log-linear OLS model was re-estimated on SSL account-level data (13,600 accounts) using analogous feature mappings. The SSL model achieves R² = 0.6185 (R² ratio SSL/TheLook = 0.80), indicating moderately good transportability of explanatory power across domains.

Coefficient alignment for the three hypothesis predictors:

| Predictor | TheLook β | SSL β | Direction | Both Significant |
|-----------|-----------|-------|-----------|-----------------|
| `return_frequency` | +0.445 | +0.104 | ✓ Aligned | ✗ (SSL p = 0.578) |
| `avg_basket_size` | −0.156 | +0.320 | ✗ Diverged | ✓ Both significant |
| `purchase_recency_days` | −0.001 | +0.027 | ✗ Diverged | ✗ (SSL only) |

The divergence on `avg_basket_size` reflects a genuine structural difference between B2C and B2B purchasing: in B2B institutional buying, larger orders contain higher-value items (opposite of fashion retail), so larger baskets produce costlier — not cheaper — returns. This represents a meaningful domain difference rather than model failure. `return_frequency` is directionally consistent across sectors; its SSL coefficient is attenuated because institutional return processes are structurally higher-volume, reducing the discriminating power of raw count. Overall generalization score: 0.33 (moderate), reflecting directional consistency on the primary predictor but structural B2C–B2B differences in basket composition.

---

# Chapter V: Data Visualizations

## 5.1 Overview

This chapter summarizes the key visualizations produced across all research questions. All figures were generated programmatically from processed data artifacts and are available in the `figures/` directory of the project repository. Interactive versions of select visualizations are available through the Streamlit dashboard application.

## 5.2 RQ1 Visualizations

Seven figures characterize the category and brand distribution of profit erosion:

| Figure | Content | Key Insight |
|--------|---------|------------|
| Fig. 1 | Top categories by total profit erosion (bar) | Outerwear and Sweaters dominate total erosion |
| Fig. 2 | Top brands by total profit erosion (bar) | Brand concentration mirrors category concentration |
| Fig. 3 | Return rate vs. mean erosion scatter (by category) | Weak correlation: high return rate ≠ high erosion |
| Fig. 4 | Top departments by total erosion (bar) | Men's department 32% higher than Women's |
| Fig. 5 | Severity vs. volume decomposition (scatter) | Distinct mechanisms: volume-driven vs. severity-driven |
| Fig. 6 | Profit erosion distribution — log scale (histogram) | Right-skewed with heavy tail; justifies non-parametric tests |
| Fig. 7 | Bootstrap 95% confidence intervals by category | Category rank order is statistically stable |

The severity-vs.-volume decomposition (Fig. 5) is the most analytically important RQ1 visualization: it reveals that Outerwear generates high erosion primarily through severity (high cost per return) while categories like Active wear generate moderate erosion primarily through volume (high return frequency). These two mechanisms require different management responses.

## 5.3 RQ2 Visualizations

Six figures characterize customer concentration and segmentation:

| Figure | Content | Key Insight |
|--------|---------|------------|
| Lorenz curve | Cumulative erosion share vs. customer share | Visual confirmation of Gini = 0.4122 |
| Pareto chart | Top-N customers' cumulative erosion share | Top 20% → 47.6% of erosion |
| Cluster scatter | Customer scatter colored by segment | Two clusters separated primarily on order frequency and basket size |
| Cluster erosion comparison | Box plots by cluster | Cluster 0 mean erosion ($95.51) nearly double Cluster 1 ($53.07) |
| Clustering diagnostics | Silhouette scores by k | k = 2 is the clear optimum |
| Feature concentration ranking | Feature importance bar chart | Order frequency and total sales as top discriminating features |

## 5.4 RQ3 Visualizations

Four figures support the predictive modeling results:

| Figure | Content | Key Insight |
|--------|---------|------------|
| ROC curves | All three models vs. random baseline | All models far above the diagonal; RF AUC = 0.9798 |
| Confusion matrices | 2×2 matrices for all three models | Low FN rate (8.8%) confirms strong Recall |
| Feature importance bars | Horizontal bars per model | Return frequency and order value consistently top-ranked |
| Sensitivity line charts | AUC and F1 across cost/threshold ranges | Hypothesis conclusion stable across all 11 scenarios |

## 5.5 RQ4 Visualizations

Four figures support the econometric findings:

| Figure | Content | Key Insight |
|--------|---------|------------|
| Coefficient forest plot | Log β with 95% CI (significant features) | Return frequency dominates; basket size negative |
| Target distribution | log(profit_erosion) histogram | Approximate normality of log-transformed outcome |
| Residual diagnostics | Fitted vs. residuals, scale-location | Heteroscedasticity confirmed; HC3 correction appropriate |
| SSL forest comparison | TheLook vs. SSL coefficient comparison | Direction alignment on return_frequency; basket sign diverges |

---

# Chapter VI: Discussion

## 6.1 Synthesis of Findings

Across four research questions, this study consistently finds that profit erosion from product returns is concentrated, predictable, and behaviorally driven. Table 6.1 summarizes the hypothesis outcomes.

**Table 6.1 — Hypothesis Outcomes**

| RQ | Null Hypothesis | Test | Outcome |
|----|----------------|------|---------|
| RQ1 | No differences in erosion across categories/brands | Kruskal-Wallis | **Rejected** (ε² = 0.45) |
| RQ2a | Erosion is uniformly distributed (Gini ≈ 0) | Gini + bootstrap | **Rejected** (Gini = 0.41) |
| RQ2b | No distinct customer segments exist | K-Means + ANOVA | **Rejected** (k=2, F = 1,479.64) |
| RQ3 | Best model AUC ≤ 0.70 | Test AUC comparison | **Rejected** (RF AUC = 0.9798) |
| RQ4 | Behavioral variables not significant | OLS joint F-test | **Rejected** (p < 0.0001; 2/3 predictors) |

**The overarching finding** is that return-related profit erosion is not a random operational cost but a structured economic phenomenon with identifiable drivers. Category and brand determine the unit cost of each return (RQ1); customer behavioral typology determines who accumulates losses across many return events (RQ2); and specific behavioral metrics — especially return frequency and order value — are both predictive (RQ3) and causally interpretable (RQ4) proxies for erosion risk.

## 6.2 Behavioral Mechanisms

Three behavioral mechanisms emerge as consistent explanations across methods:

**1. Return frequency is the primary driver.** Across RQ3 (feature importance), RQ4 (largest coefficient, +56.1% per unit), and RQ2 (top ANOVA F-statistic for cluster separation), return frequency is consistently the most powerful predictor of profit erosion. This aligns with Petersen and Kumar (2009), who identified habitual returners as the primary driver of customer-level margin erosion in longitudinal CLV analysis.

**2. Order value amplifies erosion.** Average order value appears in the top three features for two of three RQ3 models and carries a coefficient comparable to return frequency in the RQ4 regression. Higher-value orders generate larger margin reversals when returned, creating a compounding interaction: frequent returners who also buy expensive items are disproportionately costly.

**3. Basket size is a protective factor.** The negative RQ4 coefficient on `avg_basket_size` (−14.4% per unit) suggests that customers who purchase more items per order tend to select lower-margin, lower-risk items on average. This is a non-obvious finding with practical implications: promotional strategies that encourage multi-item orders may incidentally reduce per-return erosion risk.

## 6.3 External Validity

The multi-level external validation against SSL data provides evidence that core findings extend beyond the synthetic TheLook environment:

- **RQ1** generalizes fully: Kruskal-Wallis p ≈ 0.000 in SSL confirms that category-level profit erosion differences are a general phenomenon in returns data.
- **RQ3** demonstrates meaningful transportability: 76.4% directional accuracy and Spearman ρ = 0.75 in an independent B2B dataset are strong results given the structural differences between sectors (Steyerberg & Harrell, 2016; Debray et al., 2015).
- **RQ4** shows partial transportability: the R² ratio of 0.80 (SSL/TheLook) indicates reasonable explanatory power retention, but the sign reversal on `avg_basket_size` is a genuine structural difference reflecting B2C vs. B2B purchasing composition. This divergence is not a model failure; it is a domain-specific finding that enriches interpretation.

## 6.4 Limitations

**Dataset limitations.** The primary dataset is synthetic. While relative comparisons and model-inferred behavioral patterns are expected to reflect realistic structure, the absolute dollar values of profit erosion should not be extrapolated to real-world business decisions without recalibration against observed operational costs.

**Model assumptions.** The OLS regression (RQ4) assumes linear marginal effects and does not capture interaction terms (the RESET test statistic F = 1,525 indicates some non-linearity). Return frequency and order value likely interact: the marginal erosion impact of an additional return may be higher for customers who also have high order values. Future work should explore interaction specifications.

**Selection bias.** All analyses are conditioned on customers who returned at least one item. The approximately 70,000+ TheLook customers with no returns are excluded from the primary analysis. If non-returners systematically differ on unobserved dimensions, estimates of behavioral associations may not generalize to the full customer population.

**Causal inference.** All reported associations are observational. The RQ4 coefficients on return frequency and basket size represent partial correlations controlling for observed confounds, not causal effects in the experimental sense. Endogeneity is possible: customers who have already experienced high erosion may alter their return behavior in response.

**Processing cost model.** The base cost of $12.00 per return is derived from the reverse logistics literature and applied uniformly within tier. Actual operational costs vary by retailer, return channel, and item condition. Sensitivity analysis confirms the hypothesis conclusion holds across $8–$18 base costs, but the absolute erosion estimates are contingent on cost model assumptions.

---

# Chapter VII: Conclusions and Recommendations

## 7.1 Conclusions

This study successfully applied a multi-method analytics framework to quantify, segment, predict, and model product return-related profit erosion in e-commerce. Four research questions were completed; all corresponding null hypotheses were rejected.

The central finding is operationally actionable: **profit erosion is concentrated among a identifiable minority of customers, predictable from behavioral data with high accuracy, and driven primarily by return frequency and order value.** The top 20% of customers account for 47.6% of total erosion (RQ2). A Random Forest model identifies high-erosion customers with a Test AUC of 0.9798 and a False Negative rate of 8.8% (RQ3). Return frequency carries a marginal effect of +56.1% erosion per unit (RQ4). These findings, validated directionally against real-world return data from School Specialty LLC, constitute an evidence-based foundation for return-risk management.

## 7.2 Recommendations

**Recommendation 1 — Tier-based return policy differentiation (RQ1).**
Premium categories (Outerwear, Suits, Jeans, Sweaters) generate disproportionate erosion per return event. Retailers should apply stricter return windows, restocking fees, or inspection requirements to these categories while maintaining liberal policies for low-margin commodity categories (Socks, Intimates, Tops & Tees) where per-return erosion is substantially lower.

**Recommendation 2 — Targeted intervention for Cluster 0 (RQ2).**
The High-Activity segment (Cluster 0, 36.5% of customers, 50.8% of erosion) is the primary intervention priority. These customers are high-value in gross revenue terms, making blanket punitive policies counterproductive. Targeted interventions — fit advisory services, personalized size guidance, return reason surveys — can reduce return probability without alienating high-spending customers.

**Recommendation 3 — Deploy the predictive model for early-warning scoring (RQ3).**
The Random Forest model should be integrated into the customer data platform to produce a monthly erosion-risk score per customer. New customers with return frequency ≥ 3 and order value above the dataset median should be flagged for Cluster 0-type interventions before erosion compounds. Where intervention budgets are limited, Gradient Boosting's higher Recall (0.9299) minimizes missed high-risk cases; where intervention cost per customer is high, Random Forest's higher Precision (0.7822) minimizes wasted expenditure.

**Recommendation 4 — Target return frequency as the primary lever (RQ4).**
Given the +56.1% marginal association between return frequency and total erosion, even small reductions in per-customer return frequency generate meaningful erosion reduction at scale. Behavioral interventions that reduce repeat returns — virtual try-on tools, enhanced product descriptions, size verification prompts — should be prioritized and evaluated through controlled experiments with erosion as the primary outcome metric.

**Recommendation 5 — Multi-item order incentives as a passive erosion hedge (RQ4).**
The negative coefficient on basket size (−14.4%) suggests that customers who purchase more items per order generate lower per-return erosion. Bundle promotions and multi-item discounts may incidentally reduce return risk by shifting customers toward lower-margin product combinations. This hypothesis should be tested in an A/B experiment before being adopted as a stated erosion-reduction strategy.

## 7.3 Future Work

**RQ5 — Optimal intervention thresholds (proposed).** The prescriptive extension of RQ3 would determine the probability threshold at which the expected cost of intervention is less than the expected erosion prevented. This requires uplift modeling (estimating the return reduction attributable to intervention versus counterfactual) and empirical data on intervention costs and effectiveness by channel. A randomized controlled trial assigning high-risk customers to intervention vs. control arms would provide the necessary uplift estimates.

**Longitudinal analysis.** The current study is cross-sectional. A longitudinal cohort design tracking customers across multiple return events would enable causal identification of whether high-frequency returners are habitually predisposed (selection) or whether early return experiences reinforce return behavior (treatment effect) — a distinction with direct implications for intervention targeting.

**Interaction modeling.** The RESET test on the RQ4 regression signals unexplained non-linearity. A second-stage analysis incorporating return frequency × order value interactions, threshold effects (does erosion accelerate beyond a frequency threshold?), and customer lifetime stage moderators would provide a richer behavioral model.

**Real-world operational cost calibration.** Partnering with a live e-commerce retailer to measure actual processing costs by category and return channel would replace the literature-based cost model with observed values, substantially improving the precision of total erosion estimates and intervention cost-benefit calculations.

---

# References

Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., … & Lindauer, M. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *WIREs Data Mining and Knowledge Discovery*, 13(2), e1484. https://arxiv.org/abs/2107.05173

Bousquet, O., & Elisseeff, A. (2002). Stability and generalization. *Journal of Machine Learning Research*, 2, 499–526. https://www.jmlr.org/papers/v2/bousquet02a.html

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Cui, R., Rajagopalan, S., & Ward, A. R. (2020). Predicting product return volume using machine learning methods. *European Journal of Operational Research*, 281(3), 612–627.

Debray, T. P. A., Vergouwe, Y., Koffijberg, H., Nieboer, D., Steyerberg, E. W., & Moons, K. G. M. (2015). A new framework to enhance the interpretation of external validation studies of clinical prediction models. *Journal of Clinical Epidemiology*, 68(3), 279–289. https://doi.org/10.1016/j.jclinepi.2014.06.018

Dormann, C. F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carré, G., … & Lautenbach, S. (2013). Collinearity: A review of methods to deal with it and a simulation study evaluating their performance. *Ecography*, 36(1), 27–46. https://doi.org/10.1111/j.1600-0587.2012.07348.x

Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52–64.

Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of the 17th International Joint Conference on Artificial Intelligence*, 973–978. https://cseweb.ucsd.edu/~elkan/rescale.pdf

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

Guide, V. D. R., Jr., & Van Wassenhove, L. N. (2009). The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10–18. https://doi.org/10.1287/opre.1080.0628

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182. https://www.jmlr.org/papers/v3/guyon03a.html

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. https://hastie.su.domains/ElemStatLearn/

Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley. https://doi.org/10.1002/0471722146

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21. https://doi.org/10.1145/2382577.2382579

Kornbrot, D. (2014). Point biserial correlation. *Wiley StatsRef: Statistics Reference Online*. https://doi.org/10.1002/9781118445112.stat06227

National Retail Federation. (2023). *2023 Consumer Returns in the Retail Industry*. NRF/Appriss Retail.

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research*, 18(174), 1–54. https://www.jmlr.org/papers/v18/17-514.html

Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461–465.

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? Antecedents and consequences. *Journal of Marketing*, 73(3), 35–51. https://doi.org/10.1509/jmkg.73.3.035

Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery*, 9(3), e1301. https://www.jmlr.org/papers/v20/18-444.html

Rogers, D. S., & Tibben-Lembke, R. (2001). An examination of reverse logistics practices. *Journal of Business Logistics*, 22(2), 129–148.

Rosenblatt, J. D., Vink, M., Bhatt, P., Drton, M., & Hansen, N. R. (2024). Leakage and the reproducibility crisis in machine learning-based science. *Nature Communications*, 15, 2091. https://doi.org/10.1038/s41467-024-46150-w

Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics*, 23(19), 2507–2517. https://academic.oup.com/bioinformatics/article/23/19/2507/185254

Saltelli, A., Tarantola, S., Campolongo, F., & Ratto, M. (2004). *Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models*. Wiley. https://doi.org/10.1002/0470870958

Schober, P., Boer, C., & Schwarte, L. A. (2018). Correlation coefficients: Appropriate use and interpretation. *Anesthesia & Analgesia*, 126(5), 1763–1768. https://doi.org/10.1213/ANE.0000000000002864

Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Steyerberg, E. W., & Harrell, F. E. (2016). Prediction models need appropriate internal, internal–external, and external validation. *Journal of Clinical Epidemiology*, 69, 245–247. https://doi.org/10.1016/j.jclinepi.2015.04.005

Stevenson, M., & Rieck, M. (2024). Returns management and sustainability in e-commerce: A systematic review. *International Journal of Production Economics*, 268, 109117.

Toktay, L. B. (2003). Forecasting product returns. In Guide, V. D. R., Jr., & Van Wassenhove, L. N. (Eds.), *Business Aspects of Closed-Loop Supply Chains* (pp. 203–219). Carnegie Mellon University Press.

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031

---

*All data processing, modeling, and visualization code is available in the project repository. Appendix A contains the complete data dictionary. Appendix B contains the full processing cost derivation, sensitivity analysis tables, and supplementary statistical outputs.*
