# Capstone Report — Draft Chapters I & II

**Course:** DAMO-699-4 Capstone Project
**Institution:** University of Niagara Falls, Canada
**Professor:** Omid Isfahanialamdari
**Date:** February 2026

---

# Chapter I: Introduction

## 1.1 Background and Motivation

Product returns represent one of the most significant and underanalyzed sources of economic loss in modern e-commerce. While the industry has invested heavily in optimizing forward logistics—delivery speed, packaging efficiency, and order accuracy—the reverse channel continues to extract margin silently and at scale. In 2023, U.S. retail return rates averaged 14.5% of total merchandise sold, with e-commerce return rates exceeding 17% in apparel-dominant categories (National Retail Federation, 2023). The aggregate economic impact is substantial; however, aggregate industry statistics obscure the customer-level and category-level variation that determines where and how profit is actually lost.

The dominant analytical framing in both practice and academic literature treats returns as an operational problem: volume forecasting, logistics network design, and restocking velocity. Petersen and Kumar (2009) were among the first to challenge this framing, demonstrating that returns are not merely logistical events but economic reversal events that erode the realized profitability of customer relationships. Their work established a foundation for customer-level return analysis that this project extends by operationalizing profit erosion as a measurable, decomposable quantity.

This project reframes product returns through a profit lens, decomposing return-related loss into two distinct channels: the margin reversal on the returned item itself (the sale price minus the product cost that must be refunded) and the incremental processing cost incurred to receive, inspect, restock, and administer the return. Together, these two channels define *profit erosion*—the net economic cost imposed on the retailer by each return event. By quantifying profit erosion at the item, order, and customer levels, this study enables analytical approaches that go beyond operational metrics to address the strategic question of which customers, categories, and behaviors drive disproportionate economic loss.

## 1.2 Problem Statement

E-commerce retailers face a structural challenge: return policies that maximize customer acquisition and satisfaction simultaneously create conditions for systematic margin erosion. A customer who purchases frequently and returns frequently may generate positive gross revenue while simultaneously destroying net profitability through compounding return processing costs. This dynamic is poorly captured by conventional return rate metrics, which measure volume but not value destruction.

The literature identifies two gaps relevant to this project. First, while Guide and Van Wassenhove (2009) and Stevenson and Rieck (2024) documented the economic complexity of returns management, empirical work connecting individual return behavior to customer-level profit outcomes remains limited—existing work emphasizes return rates or volume forecasting rather than margin reversal and cumulative profit erosion (Toktay, 2003). Second, predictive approaches to return-related profit loss—identifying high-erosion customers before the loss is realized—are sparse in the academic literature relative to the volume of work on return volume forecasting at the aggregate level (Cui, Rajagopalan, & Ward, 2020). This project addresses both gaps by constructing a profit erosion measure from transaction data and applying a multi-method analytical framework to understand, segment, predict, and model it at the customer level.

## 1.3 Research Questions and Hypotheses

This study is organized around five research questions, each addressed through a distinct analytical method. The first four were undertaken as the primary analytical scope of this capstone; the fifth is proposed for future investigation beyond the scope of the current study.

### 1.3.1 RQ1 — Profit Erosion Differences Across Product Categories and Brands

*Do significant differences in profit erosion exist across product categories and brands?*

Profit erosion is not expected to be uniform across the product catalog. Premium categories with higher price points, larger margins, and more complex return handling processes are expected to exhibit higher absolute and relative profit erosion than lower-margin commodity categories. This question employs descriptive statistical analysis and non-parametric hypothesis testing to characterize the distribution of profit erosion across categories and identify the segments driving disproportionate losses.

**H₀:** There are no statistically significant differences in profit erosion across product categories or brands.

**H₁:** Statistically significant differences in profit erosion exist across at least one pair of product categories or brands.

### 1.3.2 RQ2 — Customer Behavioral Segments with Differential Profit Erosion

*Can unsupervised learning identify distinct customer behavioral segments, and do these segments differ significantly in profit erosion intensity?*

Traditional demographic segmentation may not capture the behavioral patterns that drive return-related profit erosion. Unsupervised learning enables data-driven identification of customer typologies based on purchasing and return behaviors without pre-defined constraints (Cui et al., 2020), allowing discovery of high-impact customer groups that may not align with conventional demographic categories. Identifying these segments provides the strategic foundation for differentiated retention and intervention strategies.

**H₀₂ (Null):** Customer segments identified through clustering algorithms do not differ significantly in mean profit erosion from returns.

**H₁₂ (Alternative):** Customer segments identified through clustering algorithms exhibit statistically significant differences in mean profit erosion from returns.

### 1.3.3 RQ3 — Predicting High Profit Erosion Customers

*Can machine learning models accurately predict high profit erosion customers from customer-level behavioral features, and which features contribute most to predictive accuracy?*

If profit erosion is predictable from observable transaction behavior, then early identification of high-erosion customers becomes operationally feasible—enabling targeted interventions before losses compound. This question applies binary classification models to predict whether a customer falls in the top quartile of profit erosion (the "high erosion" segment).

**H₀:** The best-performing classification model achieves an AUC-ROC ≤ 0.70. Machine learning models cannot reliably discriminate between high and low profit erosion customers from behavioral features alone.

**H₁:** The best-performing classification model achieves an AUC-ROC > 0.70. Machine learning models can reliably predict high profit erosion customers from behavioral features.

The AUC threshold of 0.70 represents the widely accepted lower bound of useful discrimination in applied classification literature (Hosmer & Lemeshow, 2000). A model performing below this threshold would not provide actionable ranking ability for operational deployment.

### 1.3.4 RQ4 — Marginal Associations Between Customer Behaviors and Profit Erosion

*What are the marginal associations between key customer behavioral variables—including return frequency, return rate, average basket size, and average order value—and profit erosion magnitude, controlling for financial and demographic attributes?*

While RQ3 targets prediction accuracy, RQ4 targets interpretability: quantifying the independent contribution of specific engineered behavioral features to profit erosion magnitude after controlling for financial and demographic confounds. Econometric regression enables formal coefficient estimation and significance testing, directly answering questions such as how much profit erosion changes for each unit increase in a customer's return frequency or basket size (Petersen & Kumar, 2009). These marginal estimates directly inform intervention cost-benefit analysis and complement the feature importance rankings produced by RQ3.

**H₀₄ (Null):** Behavioral variables exhibit no statistically significant marginal associations with profit erosion when controlling for financial and demographic attributes.

**H₁₄ (Alternative):** Behavioral variables exhibit statistically significant marginal associations with profit erosion when controlling for financial and demographic attributes.

### 1.3.5 RQ5 — Optimal Intervention Thresholds (Proposed Future Work)

*At what predicted-risk threshold is it economically optimal to intervene with high-erosion customers, accounting for the cost of intervention and the expected reduction in return-related losses?*

This prescriptive question extends the predictive output of RQ3 into decision analytics: determining not merely *who* is likely to be a high-erosion customer, but *when* it is cost-effective to act and through what intervention channel. This question is formally scoped as future work beyond the current study.

**H₀ (proposed):** Intervention at the model-predicted probability threshold does not produce a net positive economic outcome after accounting for intervention costs.

**H₁ (proposed):** There exists an optimal intervention threshold at which expected return-related loss reduction exceeds the cost of the intervention program.

Answering this question would require additional data on intervention costs, uplift estimation, and counterfactual return rates—inputs not available within the current dataset.

## 1.4 Scope and Limitations

This study draws on two datasets. The primary dataset is the `bigquery-public-data.thelook_ecommerce` dataset, a publicly available synthetic e-commerce transaction dataset hosted on Google BigQuery. It spans order-level, item-level, product, and customer tables, and is used for all primary analyses (RQ1 through RQ4). Because the dataset is synthetic, estimated absolute values of profit erosion (total dollar figures) should be interpreted with caution; however, relative comparisons and model-inferred behavioral patterns are expected to reflect realistic relationships.

External validation of the RQ3 predictive model was conducted using transactional return data from School Specialty LLC (SSL), a U.S. educational supplies retailer, covering approximately 234,000 return order lines and 16,700 customer accounts from 2024 to 2025. This dataset provides a real-world signal against which the directional validity of the TheLook-trained model can be assessed.

Key limitations of this study include: (1) the synthetic nature of the primary dataset limits generalizability of absolute dollar estimates; (2) the SSL external validation is used for directional model assessment only and does not constitute a replicated study on that population; (3) RQ5 is proposed but not implemented; and (4) causal inference about return behavior is not claimed—all associations are observational.

## 1.5 Organization of the Report

The remainder of this report is organized as follows. Chapter II describes the data sources, extraction procedures, profit erosion methodology, and feature engineering pipeline that underpin all subsequent analyses. Chapter III presents the statistical analysis and diagnostics conducted for RQ1, including non-parametric group comparison tests and post-hoc analysis. Chapter IV describes the advanced analytics applications for RQ2, RQ3, and RQ4, including clustering, predictive modeling, external validation, and econometric regression. Chapter V presents the key data visualizations produced across all research questions. Chapter VI synthesizes findings across all completed research questions, discusses limitations, and situates results in the broader literature. Chapter VII concludes with actionable recommendations and directions for future research, including the proposed RQ5 prescriptive extension.

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

The consolidated item-level dataset contains 180,908 transactions across 11,988 customers who generated at least one return event. The unit of analysis for all research questions is the **customer**, constructed by aggregating item-level transaction records to produce behavioral and financial features per customer.

### 2.1.2 External Validation Dataset: School Specialty LLC

The external validation dataset was provided by School Specialty LLC (SSL), a U.S.-based educational supplies retailer. It contains approximately 234,000 return-related order lines covering approximately 16,700 customer accounts during the 2024–2025 period. The data was pre-filtered to return-type records and contains two distinct line types distinguished by the `Sales_Type` column: `RETURN` lines (credit/refund events with negative quantities) and `ORDER` lines (no-charge replacement shipments). This distinction is critical to correct feature construction and is described further in Section 2.4.

The SSL dataset was used exclusively for external directional validation of the RQ3 predictive model (Section 2.4.2). It was not used for model training or parameter selection.

## 2.2 Data Extraction and Integration

### 2.2.1 TheLook Pipeline (US06)

Data extraction and integration followed a standardized feature engineering pipeline designated US06, which consists of five sequential stages:

1. **ETL (Extract, Transform, Load):** The four BigQuery tables were joined at the item level using `order_id` and `product_id` keys. The consolidated dataset was exported to `data/processed/feature_engineered_dataset.parquet` for reproducible downstream analysis.

2. **Data Cleaning:** Rows with missing cost or sale price were removed. Date columns were parsed and validated. Item return status was derived from the `status` field (`Returned` → `is_returned_item = 1`).

3. **Feature Engineering:** Item-level financial metrics were computed (margins, discounts, profit erosion). Customer-level behavioral and financial features were then aggregated from the item-level records (described in Section 2.5).

4. **Customer Aggregation:** The 180,908 item-level records were collapsed to 11,988 customer-level observations. Only customers with at least one returned item were retained for the profit erosion analyses.

5. **Target Variable Construction:** The binary target `is_high_erosion_customer` was constructed at the customer level (described in Section 2.4).

### 2.2.2 SSL External Validation Pipeline

SSL data was loaded and cleaned using the `load_ssl_data()` function from `src/rq3_validation.py`, which parses date columns and removes records with missing account identifiers. Account-level features were then constructed using `engineer_ssl_account_features()`, applying the same behavioral feature definitions used in the TheLook pipeline to ensure comparability across datasets. The resulting SSL account-level DataFrame contains the same seven surviving predictor features as the TheLook model (described in Section 2.5).

## 2.3 Data Quality and Cleaning

The TheLook dataset is synthetic and contains no missing values in the primary financial fields (`cost`, `sale_price`). Date fields (`created_at`, `returned_at`) were validated for logical consistency (return date cannot precede order date). The overall return rate in the dataset is 10.06% (18,208 returned items out of 180,908 total items), consistent with published e-commerce return rate benchmarks for apparel.

Missing value handling for the predictive modeling stage (RQ3) followed median imputation applied to the training set only, with the imputed medians carried forward to the test set to prevent data leakage.

For the SSL dataset, records with missing account identifiers were dropped prior to feature construction. No other imputation was applied, as account-level aggregation naturally resolves item-level missingness through aggregation.

**Key data quality decisions:**

- No geographic cost tiers were applied. Return rates across 15 countries exhibited a coefficient of variation of only 3.58% (range: 9.61%–10.80%), well below the 10% threshold that would justify geographic segmentation. Country-level variation in return behavior is effectively negligible in this dataset.
- Category-based cost tiering was applied (see Section 2.4). Margin variation across categories exhibited a coefficient of variation of 59.4%, substantially exceeding the 15% threshold that justifies differential treatment.

## 2.4 Profit Erosion Methodology

### 2.4.1 Core Formula

Profit erosion for each returned item is defined as:

> **Profit Erosion = Margin Reversal + Processing Cost**

where:
- **Margin Reversal** is the item-level margin forfeited on the returned item: `sale_price − cost`
- **Processing Cost** is the operational cost of receiving and administering the return

This formula applies exclusively to returned items. Non-returned items do not incur processing costs and therefore do not contribute to profit erosion.

### 2.4.2 Processing Cost Model

The base processing cost of **$12.00 per return** was derived from four operational components grounded in reverse logistics literature (Rogers & Tibben-Lembke, 2001; Guide & Van Wassenhove, 2009):

| Component | Cost |
|---|---|
| Customer care (phone/email handling, 10–15 min) | $4.00 |
| Inspection (quality assessment, 5–8 min) | $2.50 |
| Restocking (shelving and inventory updates) | $3.00 |
| Logistics (return label, administrative processing) | $2.50 |
| **Total base cost** | **$12.00** |

This estimate is conservative relative to published industry benchmarks of $10–$25 per return (Guide & Van Wassenhove, 2009) and is appropriate for the synthetic dataset where actual operational costs are unknown. Full cost model derivation, sensitivity analysis across a $8–$18 base cost range, and supporting literature are provided in Appendix B.

### 2.4.3 Category Tier Multipliers

Because margin variation across product categories is substantial (CV = 59.4%, median returned-item margin = $20.52), a uniform base cost would understate processing risk for premium categories and overstate it for commodity categories. A three-tier multiplier structure was applied based on average returned-item margin per category:

| Tier | Multiplier | Effective Cost | Margin Threshold | Example Categories |
|---|---|---|---|---|
| Premium | 1.3× | $15.60 | ≥ $32 avg margin | Outerwear, Jeans, Suits, Sweaters |
| Moderate | 1.15× | $13.80 | $20–$31 avg margin | Active, Swim, Accessories, Hoodies |
| Standard | 1.0× | $12.00 | < $20 avg margin | Tops & Tees, Intimates, Socks |

Categories with fewer than 100 returns were assigned Standard tier by default to prevent tiering on insufficient data. The $32 upper threshold separates the top third of category margins; the $20 lower threshold aligns with the dataset-wide median margin for returned items.

### 2.4.4 Target Variable Construction

For RQ3, the customer-level binary target variable `is_high_erosion_customer` was constructed by flagging customers whose total profit erosion exceeded the **75th percentile** of the customer population distribution. This threshold produces a 25%/75% class split consistent with standard quartile-based segmentation and the Pareto principle that a minority of customers drive the majority of economic losses. The threshold is configurable in the pipeline; sensitivity analysis at the 50th, 60th, 75th, 80th, and 90th percentiles confirmed that model AUC remains robust across the range (full results in Appendix B).

## 2.5 Feature Engineering

### 2.5.1 Candidate Predictor Features

Twelve customer-level behavioral and financial features were constructed as candidate predictors for the RQ3 and RQ4 analyses:

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

Six features were excluded from all predictive and regression analyses to prevent data leakage—these variables are either components of the target variable or direct derivations of it:

`total_profit_erosion`, `total_margin_reversal`, `total_process_cost`, `profit_erosion_quartile`, `erosion_percentile_rank`, `user_id`

Including any of these as predictors would allow the model to trivially recover the target variable, producing artificially inflated performance metrics that would not generalize to real-world deployment.

### 2.5.3 Feature Screening (RQ3)

Prior to model training, the 12 candidate features were subjected to a three-gate sequential screening process applied exclusively to the training set, with surviving features then applied to the held-out test set:

| Gate | Method | Purpose |
|---|---|---|
| 1. Variance | `VarianceThreshold` (< 0.01) | Remove constant or quasi-constant features |
| 2. Correlation | Pearson \|r\| > 0.85 | Remove redundant features (drop lower-associated) |
| 3. Univariate | Point-biserial, p > 0.05 (Bonferroni-corrected) | Remove statistically irrelevant features |

Seven of the 12 candidate features survived all three gates and were used in model training: `return_frequency`, `avg_order_value`, `avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, and `customer_return_rate`.

## 2.6 Dataset Summary

The following table summarizes the datasets used across all research questions in this study:

| Dataset | Source | Records | Features | Period | Primary Use |
|---|---|---|---|---|---|
| TheLook item-level | Google BigQuery (synthetic) | 180,908 items | 12 candidate + 6 target | N/A (synthetic) | RQ1, RQ2, RQ3 training, RQ4 |
| TheLook customer-level | Aggregated from above | 11,988 customers | 12 candidate + 6 target | N/A (synthetic) | RQ2, RQ3, RQ4 |
| SSL account-level | School Specialty LLC (real) | 16,700 accounts | 7 (mapped from TheLook) | 2024–2025 | RQ3 external validation |

---

*References cited in these chapters are compiled in the References section at the end of the report. Full reference list includes: Cui et al. (2020), Guide & Van Wassenhove (2009), Hosmer & Lemeshow (2000), Petersen & Kumar (2009), Rogers & Tibben-Lembke (2001), Sargent (2013), Stevenson & Rieck (2024), Toktay (2003).*

*Full processing cost derivation, sensitivity analyses, and supporting data tables are provided in Appendix B.*
