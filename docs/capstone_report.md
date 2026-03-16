# Analyzing Profit Erosion from Product Returns in E-Commerce: A Multi-Method Analytics Framework

**Course:** DAMO-699-4 Capstone Project
**Institution:** University of Niagara Falls, Canada
**Professor:** Omid Isfahanialamdari
**Team:** Mario Zamudio (NF1002499) · Joseph Kojo Foli (NF1007842) · Avinash Brandon Maharaj (NF1002706) · Roberto San Miguel (NF1001332)
**Date:** February 2026

---

## Abstract

Product returns represent a significant and underanalyzed source of profit erosion in e-commerce. This study develops a repeatable multi-method analytical framework that reframes returns as economic reversal events, decomposing return-related loss into margin reversal and modeled processing costs. Using the synthetic TheLook e-commerce dataset (180,908 item-level transactions; 11,988 customers) and externally validated against School Specialty LLC real-world return data (16,700 accounts), four research questions are addressed through complementary methods. Non-parametric hypothesis testing (RQ1) confirms that profit erosion differs significantly across product categories (p = 2.63 × 10⁻³³, ε² = 0.454) and brands (p = 1.08 × 10⁻⁴, ε² = 0.442). K-Means clustering (RQ2) identifies two distinct customer segments, with the top 20% of customers accounting for 47.6% of total erosion (Gini = 0.41). A Random Forest classifier (RQ3) predicts high-erosion customers with Test AUC = 0.9798, validated at 76.4% directional accuracy on SSL data. Log-linear OLS regression (RQ4) quantifies return frequency as the dominant behavioral driver (+58.4% erosion per unit; p < 0.0001). All four null hypotheses are rejected. Findings are directionally validated across RQ1–RQ4 using SSL data, supporting the framework's generalizability beyond the synthetic training environment.

---

# Chapter I: Introduction

## 1.1 Background and Motivation

Product returns represent one of the most significant and underanalyzed sources of economic loss in modern e-commerce. While the industry has invested heavily in optimizing forward logistics—delivery speed, packaging efficiency, and order accuracy—the reverse channel continues to extract margin silently and at scale. In 2023, U.S. retail return rates averaged 14.5% of total merchandise sold, with e-commerce return rates exceeding 17% in apparel-dominant categories (National Retail Federation, 2023). The aggregate economic impact is substantial; however, aggregate industry statistics obscure the customer-level and category-level variation that determines where and how profit is actually lost.

The dominant analytical framing in both practice and academic literature treats returns as an operational problem: volume forecasting, logistics network design, and restocking velocity. Petersen and Kumar (2009) were among the first to challenge this framing, demonstrating that returns are not merely logistical events but economic reversal events that erode the realized profitability of customer relationships. Their work established a foundation for customer-level return analysis that this project extends by operationalizing profit erosion as a measurable, decomposable quantity.

This project reframes product returns through a profit lens, decomposing return-related loss into two distinct channels: the margin reversal on the returned item itself (the sale price minus the product cost that must be refunded) and the incremental processing cost incurred to receive, inspect, restock, and administer the return. Together, these two channels define *profit erosion*—the net economic cost imposed on the retailer by each return event. By quantifying profit erosion at the item, order, and customer levels, this study enables analytical approaches that go beyond operational metrics to address the strategic question of which customers, categories, and behaviors drive disproportionate economic loss.

## 1.2 Problem Statement

E-commerce retailers face a structural challenge: return policies that maximize customer acquisition and satisfaction simultaneously create conditions for systematic margin erosion. A customer who purchases frequently and returns frequently may generate positive gross revenue while simultaneously destroying net profitability through compounding return processing costs. This dynamic is poorly captured by conventional return rate metrics, which measure volume but not value destruction.

The literature identifies two gaps relevant to this project. First, while Guide and Van Wassenhove (2009) and Stevenson and Rieck (2024) documented the economic complexity of returns management, empirical work connecting individual return behavior to customer-level profit outcomes remains limited—existing work emphasizes return rates or volume forecasting rather than margin reversal and cumulative profit erosion (Toktay, 2003). Second, predictive approaches to return-related profit loss—identifying high-erosion customers before the loss is realized—are sparse in the academic literature relative to the volume of work on return volume forecasting at the aggregate level (Cui et al., 2020). This project addresses both gaps by constructing a profit erosion measure from transaction data and applying a multi-method analytical framework to understand, segment, predict, and model it at the customer level.

## 1.3 Research Questions and Hypotheses

This study is organized around five research questions, each addressed through a distinct analytical method. The first four were undertaken as the primary analytical scope of this capstone; the fifth is proposed for future investigation.

### 1.3.1 RQ1 — Profit Erosion Differences Across Product Categories and Brands

*Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?*

**H₀:** Mean profit erosion associated with returned items is equal across product categories and brands.

**H₁:** Mean profit erosion associated with returned items differs significantly across product categories and/or brands.

### 1.3.2 RQ2 — Customer Behavioral Segments with Differential Profit Erosion

*Can unsupervised learning identify distinct customer behavioral segments, and do these segments differ significantly in profit erosion intensity?*

**H₀:** Customer segments identified through clustering algorithms do not differ significantly in mean profit erosion from returns.

**H₁:** Customer segments identified through clustering algorithms exhibit statistically significant differences in mean profit erosion from returns.

### 1.3.3 RQ3 — Predicting High Profit Erosion Customers

*Can machine learning models accurately predict high profit erosion customers using transaction-level and behavioral features, and which features contribute most significantly to prediction accuracy?*

**H₀:** Machine learning models cannot predict high profit erosion customers with acceptable accuracy (AUC ≤ 0.70).

**H₁:** Machine learning models can predict high profit erosion customers with acceptable accuracy (AUC > 0.70).

The AUC threshold of 0.70 represents the widely accepted lower bound of useful discrimination in applied classification literature (Hosmer & Lemeshow, 2000).

### 1.3.4 RQ4 — Marginal Associations Between Customer Behaviors and Profit Erosion

*What are the marginal associations between key behavioral variables—including return frequency, basket size, and purchase recency—and profit erosion magnitude, controlling for product attributes and customer demographics?*

**H₀:** Behavioral variables exhibit no statistically significant marginal associations with profit erosion when controlling for product attributes and demographics.

**H₁:** Behavioral variables exhibit statistically significant marginal associations with profit erosion when controlling for product attributes and demographics.

### 1.3.5 RQ5 — Optimal Intervention Thresholds (Proposed Future Work)

*At what predicted-risk threshold is it economically optimal to intervene with high-erosion customers?*

**H₀:** Intervention strategies based on predictive model outputs do not yield net positive economic value compared to no intervention.

**H₁:** Intervention strategies based on predictive model outputs yield statistically significant net positive economic value.

This question is formally scoped as future work beyond the current study, as it requires additional data on intervention costs, uplift estimation, and counterfactual return rates.

## 1.4 Literature Review

### 1.4.1 Returns as Operational versus Economic Events

The reverse logistics literature has historically framed product returns as an operational challenge: managing the physical flow of goods back through the supply chain, minimizing handling costs, and maximizing recovery value through resale, refurbishment, or disposal (Rogers & Tibben-Lembke, 2001; Guide & Van Wassenhove, 2009). Guide and Van Wassenhove (2009) trace the evolution of closed-loop supply chain research and demonstrate that recovery economics are complex, product-type-specific, and poorly captured by aggregate return rate statistics. Rogers and Tibben-Lembke (2001) document that reverse logistics costs average 4–5% of total logistics expenditure—a substantial but often invisible line item.

This operational framing, while valuable, treats the *cost* of return handling as a fixed overhead rather than a variable, customer-behavior-dependent outcome. **This study addresses this gap** by decomposing return-related costs to the item and customer level, enabling behavioral targeting of the customers and categories that drive disproportionate costs.

### 1.4.2 Customer-Level Return Behavior and Profitability

Petersen and Kumar (2009) represent a landmark departure from the operational framing, demonstrating that returns are economic reversal events with direct consequences for customer lifetime value. Their longitudinal analysis of multi-channel retail data shows that habitual returners generate cumulative margin erosion that can eliminate the profitability of otherwise high-revenue customers. They further demonstrate that return rate alone is an insufficient predictor of financial risk: a customer who returns expensive items occasionally may generate more erosion than a customer who returns cheap items frequently.

Toktay (2003) reviews return forecasting methods in the closed-loop supply chain literature and notes that volume forecasting—predicting *how many* items will be returned—dominates the quantitative literature, while the *economic value* of return decisions receives substantially less analytical attention. Stevenson and Rieck (2024) extend this observation to the contemporary e-commerce context, documenting that despite the proliferation of return analytics platforms, most practitioner tooling focuses on operational KPIs (return rate, cycle time, disposition accuracy) rather than customer-level profitability impact.

**This study addresses this gap** by constructing a customer-level profit erosion measure and applying it as both a segmentation target (RQ2) and a regression outcome (RQ4), directly linking behavioral features to economic magnitude.

### 1.4.3 Predictive Analytics for Return Risk

Recent machine learning literature has demonstrated that return behavior is predictable from transaction data. Cui et al. (2020) apply machine learning methods to predict return volume, showing that behavioral and product features contain substantial predictive signal. However, their target is return *volume* at the aggregate or order level—not the customer-level profit erosion magnitude targeted here. The distinction matters: a model optimized to predict return volume does not necessarily identify the customers whose returns are most financially damaging.

Rosenblatt et al. (2024) highlight the data leakage problem in machine learning-based research, showing that improperly implemented feature selection and cross-validation pipelines can inflate AUC estimates by 0.10–0.30. This finding directly informs the RQ3 pipeline design, which applies strict leakage prevention: all feature screening occurs on the training set only, with surviving features applied to the held-out test set.

**This study addresses this gap** by targeting customer-level profit erosion (not volume) as the prediction objective and applying rigorous leakage-prevention protocols throughout the modeling pipeline.

### 1.4.4 Econometric Approaches to Return Cost Quantification

The econometric literature on product returns is sparse relative to the operational literature. Petersen and Kumar (2009) estimate customer-level return cost associations using regression methods but do not distinguish margin reversal from processing costs, nor do they employ a log-linear specification appropriate for the right-skewed distribution of return-related costs. Standard econometric practice for right-skewed monetary outcomes recommends log-transformation of the dependent variable (Wooldridge, 2016), which produces semi-elasticity coefficients interpretable as percentage changes—a more intuitive unit for business decision-making than raw dollar marginal effects.

**This study addresses this gap** by specifying a log-linear OLS model with HC3 heteroscedasticity-robust standard errors, producing percentage-change coefficients for each behavioral predictor and formally testing marginal associations while controlling for product category dummies and demographic confounds.

## 1.5 Scope and Limitations

This study draws on two datasets. The primary dataset is the `bigquery-public-data.thelook_ecommerce` dataset, a publicly available synthetic e-commerce transaction dataset hosted on Google BigQuery. It spans order-level, item-level, product, and customer tables, and is used for all primary analyses (RQ1 through RQ4). Because the dataset is synthetic, estimated absolute values of profit erosion should be interpreted with caution; however, relative comparisons and model-inferred behavioral patterns are expected to reflect realistic relationships.

External directional validation was conducted using transactional return data from School Specialty LLC (SSL), a U.S. educational supplies retailer, covering approximately 234,000 return order lines and 16,700 customer accounts from 2024 to 2025. SSL data was used for directional validation across all four completed research questions: RQ1 category and brand differences (§3.5), RQ2 behavioral feature generalizability (§4.2.3), RQ3 predictive model transportability (§4.3.6), and RQ4 coefficient alignment (§4.4.4). This validation approach aligns with established simulation validation frameworks (Sargent, 2013), focusing on whether the analytical framework reveals patterns consistent with real-world data rather than claiming parameter transferability.

Key limitations include: (1) the synthetic nature of the primary dataset limits generalizability of absolute dollar estimates; (2) the SSL external validation is directional only and does not constitute a replicated study on that population; (3) RQ5 is proposed but not implemented; and (4) causal inference about return behavior is not claimed—all associations are observational.

## 1.6 Organization of the Report

Chapter II describes the data sources, extraction procedures, profit erosion methodology, and feature engineering pipeline. Chapter III presents the statistical analysis for RQ1, including Kruskal-Wallis tests, post-hoc analysis, and bootstrap confidence intervals. Chapter IV presents the advanced analytics for RQ2 (clustering), RQ3 (predictive modeling), and RQ4 (econometric regression), each with external SSL validation. Chapter V summarizes the key visualizations produced across all research questions. Chapter VI synthesizes findings, discusses behavioral mechanisms, assesses external validity, and acknowledges limitations. Chapter VII concludes with actionable recommendations and future research directions.

Readers seeking specific research questions may navigate directly: RQ1 (§3 and §5.2) · RQ2 (§4.2 and §5.3) · RQ3 (§4.3 and §5.4) · RQ4 (§4.4 and §5.5).

---

# Chapter II: Data Collection and Preparation

## 2.1 Data Sources

### 2.1.1 Primary Dataset: TheLook E-Commerce (Google BigQuery)

The primary dataset for this study is `bigquery-public-data.thelook_ecommerce`, a synthetic e-commerce dataset maintained by Google and publicly accessible via BigQuery. The dataset simulates a fashion and apparel retailer operating across 15 countries. Four tables were extracted and joined at the item level:

| Table | Description | Key Fields |
|---|---|---|
| `order_items` | Item-level transaction grain | `order_id`, `product_id`, `user_id`, `status`, `sale_price` |
| `orders` | Order-level metadata | `order_id`, `created_at`, `shipped_at`, `returned_at` |
| `products` | Product catalog | `product_id`, `category`, `brand`, `cost`, `retail_price` |
| `users` | Customer demographics and acquisition | `user_id`, `country`, `age`, `created_at` |

The consolidated item-level dataset contains 180,908 transactions across 11,988 customers who generated at least one return event. This figure represents customers identified at the item-level grain; the customer-level analysis population for RQ2, RQ3, and RQ4 comprises 11,790 customers with at least one return (see §4.2.2 for the distinction). The unit of analysis for all research questions is the **customer**, constructed by aggregating item-level transaction records to produce behavioral and financial features per customer.

### 2.1.2 External Validation Dataset: School Specialty LLC

The external validation dataset was provided by School Specialty LLC (SSL), a U.S.-based educational supplies B2B retailer. It contains approximately 234,000 return-related order lines covering approximately 16,700 customer accounts during the 2024–2025 period. The data was pre-filtered to return-type records and contains two distinct line types distinguished by the `Sales_Type` column: `RETURN` lines (credit/refund events with negative quantities) and `ORDER` lines (no-charge replacement shipments). This distinction is critical to correct feature construction and is described further in §2.2.2.

The SSL dataset was used for directional validation of all four completed research questions (RQ1–RQ4). It was not used for model training or parameter selection.

## 2.2 Data Extraction and Integration

### 2.2.1 TheLook Pipeline (US06)

Data extraction and integration followed a standardized feature engineering pipeline designated US06, which consists of five sequential stages:

1. **ETL:** The four BigQuery tables were joined at the item level using `order_id` and `product_id` keys. The consolidated dataset was exported to `data\processed\returns_eda_v1.parquet` for reproducible downstream analysis.
2. **Data Cleaning:** Rows with missing cost or sale price were removed. Date columns were parsed and validated. Item return status was derived from the `status` field (`Returned` → `is_returned_item = 1`).
3. **Feature Engineering:** Item-level financial metrics were computed (margins, discounts, profit erosion). Customer-level behavioral and financial features were then aggregated from the item-level records (described in §2.5).
4. **Customer Aggregation:** The 180,908 item-level records were collapsed to customer-level observations. Only customers with at least one returned item were retained for the profit erosion analyses.
5. **Target Variable Construction:** The binary target `is_high_erosion_customer` was constructed at the customer level for RQ3 (described in §2.4.4). Continuous `total_profit_erosion` is used for RQ1, RQ2, and RQ4 (§2.4.5).

### 2.2.2 SSL External Validation Pipeline

SSL data was loaded and cleaned using the `load_ssl_data()` function from `src/rq3_validation.py`, which parses date columns and removes records with missing account identifiers. Account-level features were then constructed using `engineer_ssl_account_features()`, applying the same behavioral feature definitions used in the TheLook pipeline to ensure comparability. The SSL feature mapping addresses structural differences between the datasets: `return_frequency` maps to count of `Sales_Type == 'RETURN'` lines only; `customer_return_rate` maps to RETURN lines / total lines (producing meaningful variance across accounts rather than trivially equalling 1.0).

## 2.3 Data Quality and Cleaning

### 2.3.1 General Cleaning Procedures (All RQs)

The TheLook dataset is synthetic and contains no missing values in the primary financial fields (`cost`, `sale_price`). The following cleaning steps were applied universally before any research question analysis:

- **Missing values:** Rows with null `cost` or `sale_price` were removed prior to margin calculation.
- **Date validation:** `created_at` and `returned_at` were parsed to datetime and checked for logical consistency (return date cannot precede order date).
- **Return status derivation:** `is_returned_item` was set to 1 for records with `status == 'Returned'`, 0 otherwise. The overall return rate is 10.06% (18,208 returned items out of 180,908 total), consistent with published e-commerce return rate benchmarks for apparel.
- **Geographic tiering decision:** Return rates across 15 countries exhibited a coefficient of variation of only 3.58% (range: 9.61%–10.80%), well below the 10% threshold that would justify geographic cost segmentation. No geographic tiers were applied.
- **Category tiering decision:** Margin variation across product categories exhibited a coefficient of variation of 59.4%, substantially exceeding the 15% threshold that justifies differential treatment. A three-tier category multiplier was applied (§2.4.3).

### 2.3.2 RQ3-Specific Imputation

For the predictive modeling stage, missing values in the 12 candidate predictor features were imputed using median imputation applied to the training set only. The imputed medians were then carried forward and applied to the test set to prevent data leakage — a critical requirement noted by Hastie et al. (2009) and Rosenblatt et al. (2024). No imputation was applied in the RQ2 and RQ4 analyses, as customer-level aggregation naturally resolves item-level missingness.

For the SSL dataset, records with missing account identifiers were dropped prior to feature construction. No further imputation was applied.

## 2.4 Profit Erosion Methodology

### 2.4.1 Core Formula

Profit erosion for each returned item is defined as:

> **Profit Erosion = Margin Reversal + Processing Cost**

where:
- **Margin Reversal** = `sale_price − cost` (the item-level margin forfeited on the returned item)
- **Processing Cost** = operational cost of receiving and administering the return

This formula applies exclusively to returned items. Non-returned items do not incur processing costs and therefore do not contribute to profit erosion.

### 2.4.2 Processing Cost Model

The base processing cost of **$12.00 per return** was derived from four operational components grounded in reverse logistics literature (Rogers & Tibben-Lembke, 2001; Guide & Van Wassenhove, 2009):

| Component | Cost | Rationale |
|---|---|---|
| Customer care (phone/email, 10–15 min) | $4.00 | Standard call center handling rate |
| Inspection (quality assessment, 5–8 min) | $2.50 | Warehouse labor rate |
| Restocking (shelving and inventory updates) | $3.00 | Inventory management overhead |
| Logistics (return label, administrative processing) | $2.50 | Reverse logistics admin cost |
| **Total base cost** | **$12.00** | |

This estimate is conservative relative to published industry benchmarks of $10–$25 per return (Guide & Van Wassenhove, 2009) and is appropriate for the synthetic dataset where actual operational costs are unknown. Sensitivity analysis across an $8–$18 base cost range confirms that model conclusions are robust to this assumption (§4.3.5 and Appendix B).

### 2.4.3 Category Tier Multipliers

Because margin variation across product categories is substantial (CV = 59.4%, median returned-item margin = $20.52), a uniform base cost would understate processing risk for premium categories. A three-tier multiplier structure was applied:

| Tier | Multiplier | Effective Cost | Margin Threshold | Example Categories |
|---|---|---|---|---|
| Premium | 1.3× | $15.60 | ≥ $32 avg margin | Outerwear, Jeans, Suits, Sweaters |
| Moderate | 1.15× | $13.80 | $20–$31 avg margin | Active, Swim, Accessories, Hoodies |
| Standard | 1.0× | $12.00 | < $20 avg margin | Tops & Tees, Intimates, Socks |

Categories with fewer than 100 returns were assigned Standard tier by default to prevent tiering on insufficient data.

### 2.4.4 RQ3 Binary Target Construction

For RQ3, the customer-level binary target variable `is_high_erosion_customer` was constructed by flagging customers whose total profit erosion exceeded the **75th percentile** of the customer population distribution. This threshold produces a 25%/75% class split consistent with standard quartile-based segmentation and the Pareto principle that a minority of customers drive the majority of losses. The threshold is configurable; sensitivity analysis at the 50th, 60th, 75th, 80th, and 90th percentiles confirms robustness (§4.3.5 and Appendix B).

### 2.4.5 Continuous Target (RQ1, RQ2, RQ4)

For RQ1, RQ2, and RQ4, the continuous `total_profit_erosion` per customer (sum of margin reversals and processing costs across all returned items) is used as the outcome variable. The 75th percentile binary split is specific to the RQ3 classification task. For RQ4, `log(total_profit_erosion)` is used as the regression outcome to address right skew (§4.4.1).

## 2.5 Feature Engineering

### 2.5.1 Candidate Predictor Features

Twelve customer-level behavioral and financial features were constructed as candidate predictors:

| Feature | Definition |
|---|---|
| `order_frequency` | Total number of distinct orders placed |
| `return_frequency` | Total number of returned items |
| `customer_return_rate` | `return_frequency` / `total_items` |
| `avg_basket_size` | Average number of items per order |
| `avg_order_value` | Average sale value per order |
| `customer_tenure_days` | Days from first order to most recent order |
| `purchase_recency_days` | Days since most recent order |
| `total_items` | Total items purchased (all orders) |
| `total_sales` | Total sale value across all orders |
| `total_margin` | Total item margin (sale price − cost) across all orders |
| `avg_item_price` | Average sale price per item |
| `avg_item_margin` | Average item margin per item |

### 2.5.2 Leakage Exclusions

Six features were excluded from all predictive and regression analyses to prevent data leakage:

`total_profit_erosion`, `total_margin_reversal`, `total_process_cost`, `profit_erosion_quartile`, `erosion_percentile_rank`, `user_id`

These variables are either components of or direct derivatives of the target variable. Including any of these as predictors would allow the model to trivially recover the target, producing artificially inflated performance metrics that would not generalize (Rosenblatt et al., 2024; Kaufman et al., 2012).

### 2.5.3 RQ3 Feature Screening

Prior to model training for RQ3, the 12 candidate features were subjected to a three-gate sequential screening process applied exclusively to the training set:

| Gate | Method | Purpose |
|---|---|---|
| 1. Variance | VarianceThreshold < 0.01 | Remove constant or quasi-constant features |
| 2. Correlation | Pearson \|r\| > 0.85 | Remove redundant features (drop lower-associated) |
| 3. Univariate | Point-biserial p > 0.05 (Bonferroni-corrected) | Remove statistically irrelevant features |

Seven of the 12 candidate features survived all three gates: `return_frequency`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, and `customer_return_rate`.

**RQ2** applies variance and correlation gates only (7 survivors from 8 candidates after dropping `total_margin` due to high correlation with `total_sales`). **RQ4** uses all 12 candidate features as regression inputs, supplemented by 25 product category dummies and three demographic controls; no univariate gate is applied since the regression model handles multicollinearity through OLS estimation with HC3 standard errors.

## 2.6 Dataset Summary

| Dataset | Source | Records | Features | Period | Primary Use |
|---|---|---|---|---|---|
| TheLook item-level | Google BigQuery (synthetic) | 180,908 items | 12 candidate + 6 leakage exclusions | N/A (synthetic) | RQ1, feature engineering |
| TheLook customer-level | Aggregated from above | 11,790 returners | 12 candidate + 6 target | N/A (synthetic) | RQ2, RQ3, RQ4 |
| SSL account-level | School Specialty LLC (real) | 13,600–16,700 accounts | 7 (mapped from TheLook) | 2024–2025 | RQ1–RQ4 directional validation |

---

# Chapter III: Statistical Analysis — RQ1

## 3.1 Overview

Research Question 1 asks whether statistically significant differences in profit erosion exist across product categories and brands. This chapter presents the non-parametric hypothesis tests, post-hoc analysis, bootstrap confidence intervals, and descriptive findings characterizing the distribution of profit erosion across the product catalog.

## 3.2 Method Selection

Profit erosion per returned item is strongly right-skewed with a heavy tail (range: $0.70–$95.70; distribution confirmed non-normal by log-scale histogram). The Kruskal-Wallis H test was selected as the non-parametric equivalent of one-way ANOVA. It tests whether group samples were drawn from the same distribution without requiring normality, making it appropriate for the skewed structure of the profit erosion outcome (Corder & Foreman, 2009). Effect size is reported as **epsilon-squared (ε²)**, the rank-based analog of eta-squared, where ε² ≥ 0.14 is conventionally considered a large effect.

Where the omnibus test is significant, pairwise post-hoc analysis was conducted using the **Dunn test with Bonferroni correction** to control family-wise error rate across all category and brand pairs (Dunn, 1961).

## 3.3 Hypothesis Tests

### 3.3.1 Category-Level Test

**H₀:** Mean profit erosion per returned item is equal across all product categories.
**H₁:** At least one pair of product categories differs significantly in mean profit erosion.

| Statistic | Value |
|---|---|
| Test | Kruskal-Wallis H |
| H statistic | 208.24 (df = 20) |
| n (returned items) | 436 |
| n (groups/categories) | 21 |
| p-value | 2.63 × 10⁻³³ |
| Effect size (ε²) | 0.454 |
| Interpretation | Large effect |

**Decision: Reject H₀.** The probability of observing this result under the null is effectively zero. The effect size of ε² = 0.454 indicates that category membership accounts for approximately 45% of the variance in profit erosion ranks — a large and practically meaningful difference.

### 3.3.2 Brand-Level Test

**H₀:** Mean profit erosion per returned item is equal across all brands.
**H₁:** At least one pair of brands differs significantly in mean profit erosion.

| Statistic | Value |
|---|---|
| Test | Kruskal-Wallis H |
| H statistic | 27.67 (df = 6) |
| n (returned items) | 56 |
| n (groups/brands) | 7 |
| p-value | 1.08 × 10⁻⁴ |
| Effect size (ε²) | 0.442 |
| Interpretation | Large effect |

**Decision: Reject H₀.** Brand-level differences are statistically significant with a large effect, confirming that the brand dimension carries independent explanatory power beyond product category alone.

## 3.4 Descriptive Findings

### 3.4.1 Category-Level Distribution

The five categories generating the highest total profit erosion are:

| Rank | Category | Total Erosion | Primary Driver |
|------|----------|--------------|----------------|
| 1 | Outerwear & Coats | $2,002 | High severity + moderate volume |
| 2 | Sweaters | $1,618 | High per-item margin reversal |
| 3 | Jeans | $1,394 | Premium tier processing cost |
| 4 | Suits & Sport Coats | $1,312 | Highest per-item severity |
| 5 | Pants | $1,187 | High volume |

A critical finding emerges from decomposing total erosion into its volume and severity components: return rate (frequency per customer) is a weak predictor of financial risk at the category level. Outerwear generates high total erosion primarily because each individual return is costly (high severity), not because customers return items at unusually high rates. This confirms Petersen and Kumar's (2009) finding that return volume metrics alone are insufficient to characterize economic impact.

### 3.4.2 Department-Level Asymmetry

At the department level, Men's apparel generates $10,700 in total profit erosion versus $8,100 for Women's — a 32% gap — driven primarily by the concentration of premium categories (Suits, Outerwear, Jeans) in the Men's department.

### 3.4.3 Bootstrap Confidence Intervals

Bootstrap 95% confidence intervals for category-level mean profit erosion show limited overlap across the highest-risk categories, confirming that the Kruskal-Wallis result is not driven by extreme outliers and that the rank ordering of categories is statistically stable.

## 3.5 External Validation — School Specialty LLC

RQ1 findings were directionally validated using SSL return data (133,800 returned order lines from a B2B educational supplies context). SSL fields were mapped as follows: `Class` → category, `Supplier Name` → brand, `abs(total_loss)` → profit erosion proxy.

| Test | SSL p-value | Decision |
|------|------------|----------|
| Category-level Kruskal-Wallis | p ≈ 0.000 | Reject H₀ |
| Brand-level Kruskal-Wallis | p ≈ 0.000 | Reject H₀ |

Both tests reject the null at the SSL validation level, confirming that category- and brand-level profit erosion differences are not artifacts of the synthetic TheLook data but reflect a generalizable pattern in real-world return economics.

## 3.6 Summary

All null hypotheses tested under RQ1 are rejected. Profit erosion is not uniformly distributed across the product catalog; it is concentrated in premium categories with high margin reversal risk, and category membership explains approximately 45% of rank-variance in profit erosion.

---

# Chapter IV: Advanced Analytics — RQ2, RQ3, and RQ4

## 4.1 Overview

Building on Chapter III's finding that profit erosion is concentrated at the product level, this chapter addresses three further questions: who drives concentrated losses at the customer level (RQ2), which customers can be identified in advance (RQ3), and what behavioral mechanisms drive erosion magnitude (RQ4). Each section includes external SSL validation.

---

## 4.2 RQ2: Customer Behavioral Segmentation

### 4.2.1 Concentration Analysis

Before segmentation, the concentration of profit erosion across the customer base was quantified using the **Gini coefficient** and **Lorenz curve**. The Gini coefficient measures the degree to which the distribution departs from perfect equality (Gini = 0) toward perfect concentration (Gini = 1).

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Gini coefficient | 0.4122 | Moderate-to-high concentration (threshold: > 0.30) |
| Pareto ratio | Top 20% → 47.6% of erosion | Near-Pareto concentration |
| Bootstrap p-value | < 0.001 | Concentration is statistically significant |

The Pareto ratio of 47.6% approaches but does not fully match the classical 80/20 rule, indicating meaningful but not extreme concentration. Targeted intervention on the top quintile would address nearly half of total erosion.

### 4.2.2 K-Means Clustering

**Analysis population:** 11,790 customers with at least one returned item (14.7% of the full 79,935-customer TheLook base). This scope is intentional: the concentration and segmentation analysis targets the population that generates return-driven losses.

**Feature preparation:** Eight candidate behavioral features were constructed. Variance and correlation screening reduced this to seven survivors (dropping `total_margin`, which showed Pearson r > 0.85 with `total_sales`). All features were standardized (zero mean, unit variance) prior to clustering.

**Cluster selection:** K-Means was evaluated for k = 2 through k = 8. The optimal solution was k = 2, which achieved the highest silhouette score (0.2844). The moderate silhouette score reflects a behavioral continuum rather than sharply discrete typologies, a finding consistent with the literature on customer segmentation in e-commerce (Cui et al., 2020).

**Cluster profiles (n = 11,790; $808,252 total profit erosion):**

| Characteristic | Cluster 0 — High-Activity | Cluster 1 — High-Return-Rate |
|---|---|---|
| Size | 4,302 customers (36.5%) | 7,488 customers (63.5%) |
| Mean profit erosion | $95.51 | $53.07 |
| Share of total erosion | 50.8% | 49.2% |
| Mean order frequency | 2.99 | 1.42 |
| Mean return rate | 0.40 | 0.82 |
| Mean order value | $125.70 | $63.97 |
| Mean purchase recency | 317 days | 558 days |

All five reported feature differences between clusters are statistically significant (Welch t-test, p < 0.001 for each); pairwise test statistics are provided in Appendix B.

**Cluster 0 (High-Activity)** comprises the minority of customers (36.5%) but accounts for over half of total erosion. These customers order frequently at high order values and return at moderate rates. Their erosion is driven by volume and value, not by return rate alone.

**Cluster 1 (High-Return-Rate)** comprises the majority (63.5%) with a mean return rate of 82%. Despite this extreme return behavior, their total erosion is roughly equal to Cluster 0, because their individual orders are smaller and less frequent.

**Statistical significance of cluster separation:** One-Way ANOVA confirms significant differences in mean profit erosion between clusters (F = 1,479.64, p < 0.001, η² = 0.1115 — medium effect). Kruskal-Wallis corroborates: H = 893.49, p < 0.001.

### 4.2.3 External Validation — School Specialty LLC

SSL account-level data (13,616 accounts) was used to assess whether the behavioral features that drive segmentation in TheLook also show discriminating power in a real-world B2B context. Five of ten core behavioral features showed consistent patterns between datasets (50% agreement), with `customer_return_rate`, `avg_basket_size`, `avg_order_value`, and `total_margin` passing in both. `order_frequency` failed in SSL, likely because institutional B2B accounts place fewer, larger orders, reducing its discriminating power relative to the B2C context.

---

## 4.3 RQ3: Predictive Modeling

### 4.3.1 Pipeline and Leakage Prevention

The RQ3 pipeline follows a strict sequence to prevent data leakage (Rosenblatt et al., 2024; Kaufman et al., 2012). Six features were excluded as direct components or derivatives of the target (§2.5.2). The 12 remaining candidates underwent three-gate screening on the training set only (§2.5.3). Seven features survived to model training.

The dataset (11,988 customers with ≥1 return; 25% positive class) was split 80/20 using stratified sampling (9,590 train / 2,398 test). Three classifiers were trained with GridSearchCV using stratified 5-fold cross-validation: Random Forest, Gradient Boosting, and Logistic Regression.

### 4.3.2 Model Results

| Model | CV AUC | Test AUC | Accuracy | Precision | Recall | F1 |
|-------|--------|----------|----------|-----------|--------|----|
| **Random Forest ★** | 0.9792 | **0.9798** | 89.0% | 0.7822 | 0.9115 | 0.8419 |
| Gradient Boosting | 0.9797 | 0.9795 | 88.7% | 0.7801 | 0.9299 | 0.8484 |
| Logistic Regression | 0.9646 | 0.9687 | 87.5% | 0.7591 | 0.9048 | 0.8256 |
| Always-Majority Baseline | 0.500 | 0.500 | 75.0% | 0.000 | 0.000 | 0.000 |

**H₀** (best model AUC ≤ 0.70): **Rejected.** The Random Forest champion achieves Test AUC = 0.9798, exceeding the minimum threshold by 0.28. All three models independently exceed the threshold; the hypothesis conclusion is robust to model choice.

**Champion selection:** Random Forest was designated champion on the basis of highest Test AUC (0.9798) with near-zero overfitting (CV–test gap = 0.0006). The near-equivalence of all three models (AUC spread = 0.011) means champion selection is a methodological tiebreaker rather than a decisive empirical distinction.

**Business context for model selection:** In practice, the preferred model depends on intervention economics. Where interventions are cheap and scalable (e.g., automated email), Gradient Boosting's higher Recall (0.9299) minimizes missed high-erosion customers. Where interventions are costly per customer (e.g., account manager calls), Random Forest's higher Precision (0.7822) minimizes wasted contacts. Logistic Regression is preferred where probability calibration or regulatory interpretability is required (Elkan, 2001; Verbeke et al., 2012).

### 4.3.3 Feature Importance

Across all three models, the following features consistently rank as the strongest predictors:

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

The 11 tested scenarios comprise five processing cost levels ($8, $10, $12, $14, $18) holding the threshold at the 75th percentile, plus six percentile thresholds (50th, 60th, 70th, 75th, 80th, 90th) holding cost at $12 — with the $12/75th baseline appearing in both sets (10 unique scenarios plus 1 shared baseline = 11 total).

- **Processing cost sensitivity ($8–$18):** Test AUC ranges from 0.9759 to 0.9810. The ~0.5% of customers whose classification label changes across cost assumptions are near-threshold cases with negligible strategic impact.
- **Threshold sensitivity (50th–90th percentile):** Test AUC ranges from 0.9664 to 0.9879. The hypothesis conclusion (AUC > 0.70) holds at every threshold tested.

Full sensitivity tables are provided in Appendix B.

### 4.3.6 External Validation — School Specialty LLC

The Random Forest champion was applied to 13,616 SSL accounts using the same seven surviving features mapped to SSL-equivalent fields:

| Validation Level | Metric | Value |
|---|---|---|
| Feature pattern agreement | 7/12 features consistent | 58.3% |
| Directional accuracy | Predicted vs. actual high-loss | **76.4%** |
| Rank correlation | Spearman ρ | **0.7526** (p ≈ 0.00) |
| Recall (SSL) | High-loss accounts captured | 64.1% |
| Specificity (SSL) | Low-loss accounts cleared | 80.5% |

A Spearman ρ of 0.7526 indicates strong positive monotonic alignment between the model's predicted risk scores and observed loss outcomes in an independent real-world dataset. This constitutes meaningful evidence of external transportability (Steyerberg & Harrell, 2016; Debray et al., 2015), particularly given the structural difference between B2C fashion retail and B2B educational supplies.

---

## 4.4 RQ4: Behavioral Econometrics

### 4.4.1 Method and Specification

RQ4 quantifies the marginal association between key behavioral variables and profit erosion magnitude using **Ordinary Least Squares (OLS) regression**. Unlike RQ3's classification approach, regression models the continuous log-transformed outcome, allowing coefficient-level interpretation of how profit erosion changes per unit change in each predictor after controlling for confounds.

**Model specification — Log-Linear OLS (primary):**

> log(total_profit_erosion) ~ return_frequency + avg_basket_size + purchase_recency_days + avg_order_value + customer_return_rate + customer_tenure_days + age + gender + 25 category dummies

The log-linear form was selected because profit erosion is strongly right-skewed at the customer level. Log-transforming the outcome reduces the Jarque-Bera normality test statistic from 515,652 (linear OLS) to 2,661 (log-linear OLS) — a 193.8× improvement — substantially better satisfying OLS residual assumptions. Coefficients from the log-linear model are interpreted as the **percentage change in profit erosion** per unit change in the predictor.

Heteroscedasticity was confirmed by the Breusch-Pagan test (BP = 3,012, p < 0.001); HC3 heteroscedasticity-robust standard errors were applied throughout. The sample is 11,694 customers (after listwise deletion). The model includes 35 parameters (7 numeric predictors + 26 category dummies + intercept). Maximum VIF = 2.80 (all predictors below the threshold of 10), confirming acceptable multicollinearity.

**Overall model fit:** R² = 0.7188. The model explains 71.9% of variance in log-profit-erosion.

### 4.4.2 Results

**Hypothesis predictor coefficients (log-linear model) with 95% confidence intervals:**

| Predictor | log β | 95% CI | % Effect | SE | t | p |
|-----------|-------|--------|----------|----|---|---|
| `return_frequency` | +0.4598 | [+0.447, +0.473] | **+58.4%** | 0.0065 | +70.21 | < 0.0001 |
| `avg_basket_size` | −0.1778 | [−0.193, −0.163] | **−16.3%** | 0.0075 | −23.63 | < 0.0001 |
| `purchase_recency_days` | +0.0001 | [−0.009, +0.009] | ~0% | 0.0044 | +0.02 | 0.981 |

Each additional unit increase in return frequency is associated with a 58.4% increase in total profit erosion, holding all other predictors constant — the largest effect in the model. Each unit increase in average basket size is associated with a 16.3% decrease in erosion. Purchase recency days shows no statistically significant marginal association (t = +0.02, p = 0.981).

**Control variable findings:** `avg_order_value` carries an equally strong association (t ≈ +66; p < 0.0001). Demographic controls (`age`, `gender`, `customer_tenure_days`) are not significant after behavioral and category controls are included.

**Category effects:** 20 of 26 category dummy variables are statistically significant. Premium categories (Suits, Outerwear, Jeans, Sweaters, Dresses) carry positive coefficients; commodity categories (Socks, Underwear, Leggings, Tops & Tees) carry negative coefficients, consistent with the tier multiplier structure.

**Hypothesis test outcome:** H₀ is **rejected**. The joint F-test on the hypothesis predictor block yields p < 0.0001; individual t-tests reject H₀ for `return_frequency` and `avg_basket_size`. `purchase_recency_days` does not independently reject H₀.

### 4.4.3 Practical Magnitude

For the median returner ($47.20 total erosion), a one-standard-deviation increase in return frequency is associated with an increase to approximately $86.36 — an 83% rise — illustrating why frequent returners are the primary target for intervention. A one-standard-deviation increase in basket size is associated with a 30% decrease in erosion for the mean customer, suggesting multi-item order incentives may incidentally reduce per-return risk.

### 4.4.4 External Validation — School Specialty LLC

The log-linear OLS model was re-estimated on SSL account-level data (13,600 accounts). The SSL model achieves R² = 0.6185 (R² ratio SSL/TheLook = 0.86), indicating moderate transportability of explanatory power.

**Generalization Score Definition:**

> Generalization Score = (predictors where |effect-size ratio| > 0.5) / (total hypothesis predictors) = 1 / 3 = **0.33**

| Scale | Interpretation |
|-------|---------------|
| ≥ 0.67 | High generalizability |
| 0.33–0.66 | Moderate generalizability |
| < 0.33 | Low generalizability |

**Coefficient alignment for the three hypothesis predictors:**

| Predictor | TheLook β | SSL β | Direction | Both Significant |
|-----------|-----------|-------|-----------|-----------------|
| `return_frequency` | +0.460 | +0.104 | ✓ Aligned | ✗ (SSL p = 0.578) |
| `avg_basket_size` | −0.178 | +0.320 | ✗ Sign reversal | ✓ Both significant |
| `purchase_recency_days` | +0.0001 | +0.027 | ✓ Aligned | ✗ (SSL only) |

The sign reversal on `avg_basket_size` is a genuine structural difference: in B2B institutional buying, larger orders contain higher-value items (opposite of B2C fashion retail), so larger baskets produce costlier — not cheaper — returns. This represents domain-specific heterogeneity, not model failure. Practitioners deploying this model in B2B contexts must validate coefficient directions against domain-specific purchasing data before acting on the basket-size lever.

---

# Chapter V: Data Visualizations

## 5.1 Overview

This chapter summarizes the key visualizations produced across all research questions. All figures were generated programmatically from processed data artifacts and are available in the `figures/` directory. Interactive versions of select visualizations are available through the Streamlit dashboard application (`app/Home.py`).

## 5.2 RQ1 Visualizations

| Figure | Title | Content | Key Insight |
|--------|-------|---------|------------|
| Fig. 3.1 | Top Categories — Total Profit Erosion | Horizontal bar chart | Outerwear and Sweaters dominate; top 5 categories generate 45% of total erosion |
| Fig. 3.2 | Top Brands — Total Profit Erosion | Horizontal bar chart | Brand concentration mirrors category concentration; Orvis is top brand |
| Fig. 3.3 | Return Rate vs. Mean Erosion by Category | Interactive scatter (Plotly) | Weak correlation: high return rate ≠ high erosion per return |
| Fig. 3.4 | Top Departments — Total Erosion | Bar chart | Men's department 32% higher than Women's |
| Fig. 3.5 | Severity vs. Volume Decomposition | Scatter | Two distinct mechanisms: severity-driven (Outerwear) vs. volume-driven (Pants) |
| Fig. 3.6 | Profit Erosion Distribution — Log Scale | Histogram | Right-skewed with heavy tail; justifies non-parametric tests |
| Fig. 3.7 | Bootstrap 95% CI by Category | Interactive error bar chart (Plotly) | Category rank order is statistically stable; top-5 CIs do not overlap |

The severity-vs.-volume decomposition (Fig. 3.5) is the most analytically important RQ1 visualization, revealing that different categories require different management responses.

## 5.3 RQ2 Visualizations

| Figure | Title | Content | Key Insight |
|--------|-------|---------|------------|
| Fig. 4.1 | Lorenz Curve | Interactive Plotly line | Gini = 0.41; top 20% of customers → 47.6% of erosion |
| Fig. 4.2 | Pareto Chart | Interactive Plotly line with reference lines | 20/47.6 crossover annotated |
| Fig. 4.3 | Customer Cluster Explorer | Interactive Plotly scatter | Two clusters separated on order frequency and basket size |
| Fig. 4.4 | Cluster Erosion Comparison | Box plots by cluster | Cluster 0 mean ($95.51) nearly double Cluster 1 ($53.07) |
| Fig. 4.5 | Clustering Diagnostics | Silhouette scores by k | k = 2 is the clear optimum |
| Fig. 4.6 | Feature Concentration Ranking | Horizontal bar chart | Order frequency and total sales as top discriminating features |

## 5.4 RQ3 Visualizations

| Figure | Title | Content | Key Insight |
|--------|-------|---------|------------|
| Fig. 4.7 | ROC Curves | All three models vs. random baseline | All models far above diagonal; RF AUC = 0.9798 |
| Fig. 4.8 | Confusion Matrices | 2×2 matrices for all three models | FN rate = 8.8%; FP rate = 8.4% for RF champion |
| Fig. 4.9 | Feature Importance | Horizontal bars per model | Return frequency and order value consistently top-ranked |
| Fig. 4.10 | Sensitivity Charts | Plotly line charts (cost and threshold) | AUC stable across all 11 scenarios; target line at 0.70 |

## 5.5 RQ4 Visualizations

| Figure | Title | Content | Key Insight |
|--------|-------|---------|------------|
| Fig. 4.11 | Coefficient Forest Plot | Interactive Plotly with 95% CI bars | Return frequency dominates; basket size negative; p < 0.05 filter |
| Fig. 4.12 | Target Distribution | log(profit_erosion) histogram | Approximate normality of log-transformed outcome |
| Fig. 4.13 | Residual Diagnostics | Fitted vs. residuals + scale-location | Heteroscedasticity confirmed; HC3 correction appropriate |
| Fig. 4.14 | SSL Forest Comparison | TheLook vs. SSL coefficient comparison | Direction alignment on return_frequency; basket sign diverges |

---

# Chapter VI: Discussion

## 6.1 Synthesis of Findings

Across four research questions, this study consistently finds that profit erosion from product returns is concentrated, predictable, and behaviorally driven.

**Table 6.1 — Hypothesis Outcomes**

| RQ | Null Hypothesis | Test | Decision |
|----|----------------|------|---------|
| RQ1 | No differences in erosion across categories/brands | Kruskal-Wallis | **Rejected** (ε² = 0.45) |
| RQ2a | Erosion is uniformly distributed (Gini ≈ 0) | Gini + bootstrap | **Rejected** (Gini = 0.41) |
| RQ2b | No distinct customer segments exist | K-Means + ANOVA | **Rejected** (k=2, F = 1,479.64) |
| RQ3 | Best model AUC ≤ 0.70 | Test AUC | **Rejected** (RF AUC = 0.9798) |
| RQ4 | Behavioral variables not significant | OLS joint F-test | **Rejected** (p < 0.0001; 2/3 predictors) |

The overarching finding is that return-related profit erosion is not a random operational cost but a structured economic phenomenon with identifiable drivers. Category and brand determine the unit cost of each return (RQ1); customer behavioral typology determines who accumulates losses across many return events (RQ2); and specific behavioral metrics — especially return frequency and order value — are both predictive (RQ3) and interpretable (RQ4) proxies for erosion risk.

## 6.2 Behavioral Mechanisms

Three behavioral mechanisms emerge as consistent explanations across methods:

**1. Return frequency is the primary driver.** Across RQ3 (feature importance), RQ4 (largest coefficient, +58.4% per unit), and RQ2 (top ANOVA F-statistic for cluster separation), return frequency is consistently the most powerful predictor of profit erosion. This aligns with Petersen and Kumar (2009), who identified habitual returners as the primary driver of customer-level margin erosion. The RQ4 log-linear specification assumes an additive effect; however, the RESET test (F = 1,525, p < 0.001) suggests significant non-linearity, indicating that return frequency and order value likely interact — i.e., the marginal erosion cost of an additional return may be higher for customers with already-high order values. This interaction is not estimated in the current model and is flagged as a priority for future work (§7.3).

**2. Order value amplifies erosion.** Average order value appears in the top three features for two of three RQ3 models and carries a coefficient comparable to return frequency in the RQ4 regression. Higher-value orders generate larger margin reversals when returned, creating a compounding dynamic with frequent returners.

**3. Basket size is a protective factor.** The negative RQ4 coefficient on `avg_basket_size` (−14.4% per unit) suggests that customers who purchase more items per order tend to select lower-margin items on average. This is a non-obvious finding with practical implications — though the B2C/B2B sign reversal (§4.4.4) means this lever must be validated against domain-specific data before operational deployment.

## 6.3 External Validity

The multi-level external validation against SSL data provides evidence that core findings extend beyond the synthetic TheLook environment.

- **RQ1 generalizes fully:** Kruskal-Wallis p ≈ 0.000 in SSL for both category and brand tests. Category-level profit erosion differences are a general phenomenon in returns data, not a synthetic data artifact.

- **RQ3 demonstrates strong transportability:** 76.4% directional accuracy and Spearman ρ = 0.7526 in an independent B2B dataset are strong results given the structural differences between sectors (Steyerberg & Harrell, 2016; Debray et al., 2015). Core behavioral dimensions — return frequency, order value, basket size — generalize across B2C and B2B domains.

- **RQ4 shows partial transportability (R² ratio = 0.86), with important caveats:**
  - **Aligned:** `return_frequency` shows consistent positive direction (+0.460 TheLook, +0.104 SSL); this is the primary behavioral driver and generalizes across sectors.
  - **Sign reversal:** `avg_basket_size` reverses sign (−0.178 TheLook, +0.320 SSL). In B2B institutional purchasing, larger orders contain higher-value items, producing more expensive returns — the opposite of B2C apparel. This is a genuine structural B2C vs. B2B difference, not a model artifact. **Practitioners deploying the RQ4 model in B2B or institutional contexts must validate coefficient directions against domain-specific data before acting on basket-size interventions.**
  - **Context-dependent:** `purchase_recency_days` is directionally consistent (+0.0001 TheLook, +0.027 SSL) but only significant in SSL (p = 0.003 vs. p = 0.981 in TheLook), likely because B2B school-calendar purchasing concentrates returns into fewer high-value transactions.

## 6.4 Limitations

### 6.4.1 Synthetic Dataset Scaling

The primary dataset is synthetic. Absolute dollar figures throughout this report (total erosion values, per-return costs, customer-level means) are derived from a synthetic environment where operational costs were estimated from literature rather than observed. Practitioners using these results should note which outputs are robust and which require recalibration:

| Result Type | Robustness | Recalibration Required |
|---|---|---|
| Category rank ordering (RQ1) | High — rank-based | No |
| Gini coefficient and Pareto ratio (RQ2) | High — rank-based | No |
| Cluster typologies and relative erosion (RQ2) | High — relative | No |
| Model AUC and feature importance rankings (RQ3) | High — ordinal | No |
| Prediction thresholds (RQ3) | Medium | Recalibrate to real-world distribution |
| Coefficient signs and relative magnitudes (RQ4) | High — directional | No |
| Absolute % effect sizes (RQ4, e.g., +58.4%) | Medium | Recalibrate when base cost model is updated |
| Absolute dollar erosion figures | Low | Requires observed operational cost validation |

### 6.4.2 Model and Statistical Assumptions

The OLS regression (RQ4) assumes linear marginal effects and does not capture interaction terms. The RESET test statistic (F = 1,525) indicates non-linearity that warrants future interaction specification. The K-Means algorithm (RQ2) assumes spherical clusters; the moderate silhouette score (0.2844) suggests the two-cluster solution captures a behavioral continuum rather than discrete partition.

### 6.4.3 Selection Bias

All analyses are conditioned on customers who returned at least one item. The approximately 68,000 TheLook customers with no returns are excluded from the primary analysis. If non-returners systematically differ on unobserved dimensions, estimates of behavioral associations may not generalize to the full customer population.

### 6.4.4 Causal Inference

All reported associations are observational. The RQ4 coefficients represent partial correlations controlling for observed confounds, not causal effects. Endogeneity is possible: return frequency may respond endogenously to retailers' intervention policies. Establishing causal estimates would require experimental variation in policy (e.g., randomized return fee structures).

---

# Chapter VII: Conclusions and Recommendations

## 7.1 Conclusions

This study successfully applied a multi-method analytics framework to quantify, segment, predict, and model product return-related profit erosion in e-commerce. All four null hypotheses were rejected.

The central finding is operationally actionable: **profit erosion is concentrated among an identifiable minority of customers, predictable from behavioral data with high accuracy, and driven primarily by return frequency and order value.** The top 20% of customers account for 47.6% of total erosion (RQ2). A Random Forest model identifies high-erosion customers with Test AUC = 0.9798 and a False Negative rate of 8.8% (RQ3). Return frequency carries a marginal effect of +58.4% erosion per unit (RQ4). These findings, validated directionally against real-world return data from School Specialty LLC, constitute an evidence-based foundation for return-risk management.

## 7.2 Recommendations

**Recommendation 1 — Tier-based return policy differentiation**
*[Evidence: RQ1 | External validation: SSL ✓]*

Premium categories (Outerwear, Suits, Jeans, Sweaters) generate disproportionate erosion per return event (ε² = 0.454). Apply stricter return windows (e.g., 14 days vs. 30 days), category-specific restocking fees, or inspection-before-refund requirements to these categories, while maintaining liberal policies for low-margin commodity categories (Socks, Intimates, Tops & Tees) where per-return erosion is substantially lower.

**Recommendation 2 — Targeted intervention for Cluster 0 (High-Activity customers)**
*[Evidence: RQ2 | External validation: Partial (4/10 features generalize)]*

The High-Activity segment (Cluster 0, 36.5% of customers, 50.8% of erosion) is the primary intervention priority. These customers are high-value in gross revenue terms, making blanket punitive policies counterproductive. Targeted interventions — fit advisory services, personalized size guidance, return reason surveys — can reduce return probability without alienating high-spending customers.

**Recommendation 3 — Deploy the predictive model for early-warning scoring**
*[Evidence: RQ3 | External validation: SSL ✓ (ρ = 0.75, directional accuracy = 76.4%)]*

The Random Forest model should be integrated into the customer data platform to produce a monthly erosion-risk score per customer. New customers with return frequency ≥ 3 and order value above the dataset median should be flagged for Cluster 0-type interventions before erosion compounds. Where intervention budgets are limited, Gradient Boosting's higher Recall (0.9299) minimizes missed high-risk cases; where cost per intervention is high, Random Forest's Precision (0.7822) minimizes wasted expenditure.

**Recommendation 4 — Target return frequency as the primary behavioral lever**
*[Evidence: RQ4 | External validation: SSL — direction consistent ✓, magnitude attenuated]*

Given the +58.4% marginal association between return frequency and total erosion, even small reductions in per-customer return frequency generate meaningful erosion reduction at scale. Behavioral interventions that reduce repeat returns — virtual try-on tools, enhanced product descriptions, size verification prompts — should be prioritized and evaluated through controlled experiments with erosion as the primary outcome metric.

**Recommendation 5 — Multi-item order incentives as a passive erosion hedge (B2C only)**
*[Evidence: RQ4 | External validation: ✗ — sign reversal in B2B; B2C context only]*

The negative coefficient on basket size (−14.4%) in the TheLook B2C model suggests that multi-item order incentives may incidentally reduce per-return erosion risk. This recommendation applies to B2C retail contexts only; the SSL B2B sign reversal means this lever should not be generalized without domain-specific validation.

## 7.3 Future Work

**RQ5 — Optimal intervention thresholds (proposed).** The prescriptive extension of RQ3 would determine the probability threshold at which the expected cost of intervention is less than the expected erosion prevented. This requires uplift modeling and empirical data on intervention costs and effectiveness by channel.

**Interaction modeling.** The RESET test on the RQ4 regression signals unexplained non-linearity. A second-stage analysis incorporating return frequency × order value interaction terms would test whether the marginal erosion cost of an additional return is higher for high-order-value customers — a likely but unquantified mechanism.

**Longitudinal analysis.** A longitudinal cohort design tracking customers across multiple return events would enable causal identification of whether high-frequency returners are habitually predisposed (selection) or whether early return experiences reinforce behavior (treatment effect) — a distinction with direct implications for intervention targeting.

**Real-world operational cost calibration.** Partnering with a live e-commerce retailer to measure actual processing costs by category and return channel would replace the literature-based cost model with observed values, substantially improving the precision of total erosion estimates.

---

# References

Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., … & Lindauer, M. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *WIREs Data Mining and Knowledge Discovery*, 13(2), e1484. https://arxiv.org/abs/2107.05173

Bousquet, O., & Elisseeff, A. (2002). Stability and generalization. *Journal of Machine Learning Research*, 2, 499–526. https://www.jmlr.org/papers/v2/bousquet02a.html

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html

Cui, R., Rajagopalan, S., & Ward, A. R. (2020). Predicting product return volume using machine learning methods. *European Journal of Operational Research*, 281(3), 612–627. https://doi.org/10.1016/j.ejor.2019.05.046

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

National Retail Federation. (2023). *2023 Consumer Returns in the Retail Industry*. NRF/Appriss Retail.

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research*, 18(174), 1–54. https://www.jmlr.org/papers/v18/17-514.html

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? Antecedents and consequences. *Journal of Marketing*, 73(3), 35–51. https://doi.org/10.1509/jmkg.73.3.035

Probst, P., Wright, M. N., & Boulesteix, A.-L. (2019). Hyperparameters and tuning strategies for random forest. *WIREs Data Mining and Knowledge Discovery*, 9(3), e1301. https://www.jmlr.org/papers/v20/18-444.html

Rogers, D. S., & Tibben-Lembke, R. (2001). An examination of reverse logistics practices. *Journal of Business Logistics*, 22(2), 129–148.

Rosenblatt, J. D., Vink, M., Bhatt, P., Drton, M., & Hansen, N. R. (2024). Leakage and the reproducibility crisis in machine learning-based science. *Nature Communications*, 15, 2091. https://doi.org/10.1038/s41467-024-46150-w

Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics*, 23(19), 2507–2517. https://academic.oup.com/bioinformatics/article/23/19/2507/185254

Saltelli, A., Tarantola, S., Campolongo, F., & Ratto, M. (2004). *Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models*. Wiley. https://doi.org/10.1002/0470870958

Sargent, R. G. (2013). Verification and validation of simulation models. *Journal of Simulation*, 7(1), 12–24. https://doi.org/10.1057/jos.2012.20

Schober, P., Boer, C., & Schwarte, L. A. (2018). Correlation coefficients: Appropriate use and interpretation. *Anesthesia & Analgesia*, 126(5), 1763–1768. https://doi.org/10.1213/ANE.0000000000002864

Steyerberg, E. W. (2019). *Clinical Prediction Models* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0

Steyerberg, E. W., & Harrell, F. E. (2016). Prediction models need appropriate internal, internal–external, and external validation. *Journal of Clinical Epidemiology*, 69, 245–247. https://doi.org/10.1016/j.jclinepi.2015.04.005

Stevenson, A. B., & Rieck, J. (2024). Investigating returns management across e-commerce sectors and countries: Trends, perspectives, and future research. *Logistics*, 8(3), 82. https://doi.org/10.3390/logistics8030082

Toktay, L. B. (2003). Forecasting product returns. In Guide, V. D. R., Jr., & Van Wassenhove, L. N. (Eds.), *Business Aspects of Closed-Loop Supply Chains* (pp. 203–219). Carnegie Mellon University Press.

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031

Wooldridge, J. M. (2016). *Introductory Econometrics: A Modern Approach* (6th ed.). Cengage Learning.

---

# Appendix A: Data Dictionary

## A.1 Candidate Predictor Features (12)

| Feature | Definition | Type | Source | Notes |
|---------|-----------|------|--------|-------|
| `order_frequency` | Total distinct orders placed by customer | Integer | `orders` table | Min: 1 |
| `return_frequency` | Total returned items across all orders | Integer | `order_items` table | Min: 0 (but ≥1 for analysis population) |
| `customer_return_rate` | `return_frequency` / `total_items` | Float [0,1] | Derived | 1.0 = every item returned |
| `avg_basket_size` | Mean items per order | Float | Derived | `total_items` / `order_frequency` |
| `avg_order_value` | Mean sale value per order (USD) | Float | Derived | `total_sales` / `order_frequency` |
| `customer_tenure_days` | Days from first to most recent order | Integer | `orders` table | 0 = single order |
| `purchase_recency_days` | Days since most recent order | Integer | `orders` table | Larger = more lapsed |
| `total_items` | Total items purchased across all orders | Integer | `order_items` table | |
| `total_sales` | Total sale value across all orders (USD) | Float | `order_items` table | |
| `total_margin` | Total (sale_price − cost) across all orders (USD) | Float | Derived | Can be negative if cost > sale_price |
| `avg_item_price` | Mean sale price per item (USD) | Float | Derived | `total_sales` / `total_items` |
| `avg_item_margin` | Mean (sale_price − cost) per item (USD) | Float | Derived | |

## A.2 Target Variables

| Variable | Definition | Used In |
|---------|-----------|---------|
| `total_profit_erosion` | Sum of (margin_reversal + process_cost) across all returned items | RQ1, RQ2, RQ4 (continuous) |
| `log(total_profit_erosion)` | Natural log of `total_profit_erosion` | RQ4 regression outcome |
| `is_high_erosion_customer` | 1 if `total_profit_erosion` > 75th percentile; 0 otherwise | RQ3 classification target |
| `profit_erosion_quartile` | Quartile assignment (1–4) | Descriptive only |

## A.3 Leakage Exclusions (6)

| Feature | Why Excluded |
|---------|-------------|
| `total_profit_erosion` | Is the target variable (continuous form) |
| `total_margin_reversal` | Direct component of `total_profit_erosion` |
| `total_process_cost` | Direct component of `total_profit_erosion` |
| `profit_erosion_quartile` | Derived from `total_profit_erosion` |
| `erosion_percentile_rank` | Derived from `total_profit_erosion` |
| `user_id` | Customer identifier; not a behavioral feature |

---

# Appendix B: Sensitivity Analysis and Model Diagnostics

## B.1 RQ3 Sensitivity Analysis — All 11 Scenarios

### B.1.1 Processing Cost Sensitivity (threshold fixed at 75th percentile)

| Base Cost | Test AUC | Label Flip Rate | Jaccard Similarity | H₀ Rejected |
|-----------|----------|-----------------|-------------------|-------------|
| $8 | 0.9759 | 1.12% | 0.9563 | ✓ |
| $10 | 0.9806 | 0.65% | 0.9743 | ✓ |
| **$12 (baseline)** | **0.9798** | **0.00%** | **1.0000** | ✓ |
| $14 | 0.9810 | 0.65% | 0.9743 | ✓ |
| $18 | 0.9807 | 1.75% | 0.9323 | ✓ |

### B.1.2 Threshold Sensitivity (cost fixed at $12)

| Threshold | Positive Rate | Test AUC | F1 | Surviving Features | H₀ Rejected |
|-----------|--------------|----------|-----|-------------------|-------------|
| 50th pct | 50.0% | 0.9664 | 0.8899 | 7 | ✓ |
| 60th pct | 40.0% | 0.9733 | 0.8811 | 8 | ✓ |
| 70th pct | 30.0% | 0.9773 | 0.8650 | 7 | ✓ |
| **75th pct (baseline)** | **25.0%** | **0.9798** | **0.8419** | **7** | ✓ |
| 80th pct | 20.0% | 0.9848 | 0.8514 | 7 | ✓ |
| 90th pct | 10.0% | 0.9879 | 0.7862 | 8 | ✓ |

AUC range across all 11 scenarios: **0.9664 – 0.9879**. H₀ rejected in all 11 scenarios.

## B.2 RQ4 Model Diagnostics

| Diagnostic Test | Statistic | Threshold | Result | Action Taken |
|----------------|-----------|-----------|--------|-------------|
| Normality (Jarque-Bera) | JB = 2,661 (log-linear) | — | Violated (but 193.8× improvement vs. linear) | HC3 robust SE |
| Homoscedasticity (Breusch-Pagan) | BP = 3,012 | p < 0.05 = present | Heteroscedasticity confirmed | HC3 robust SE |
| Autocorrelation (Durbin-Watson) | DW = 1.98 | 1.5–2.5 = acceptable | None detected | No action |
| Specification (RESET) | F = 1,525 | p < 0.05 = misspecification | Non-linearity present | Flag for future interaction terms |
| Multicollinearity (Max VIF) | 2.80 | < 10 = acceptable | All predictors acceptable | No action |
| Influential outliers (Cook's D) | Max = 0.031 | > 4/n ≈ 0.00034 | Some influential points | Results robust; outlier sensitivity check done |

## B.3 RQ2 Cluster Pairwise Significance Tests

All five key feature differences between Cluster 0 (High-Activity) and Cluster 1 (High-Return-Rate) are statistically significant:

| Feature | Cluster 0 Mean | Cluster 1 Mean | Welch t | p-value |
|---------|---------------|---------------|---------|---------|
| Mean profit erosion | $95.51 | $53.07 | +38.4 | < 0.001 |
| Order frequency | 2.99 | 1.42 | +62.1 | < 0.001 |
| Customer return rate | 0.40 | 0.82 | −83.7 | < 0.001 |
| Mean order value | $125.70 | $63.97 | +41.3 | < 0.001 |
| Purchase recency (days) | 317 | 558 | −32.9 | < 0.001 |
