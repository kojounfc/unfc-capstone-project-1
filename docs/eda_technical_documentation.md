# Baseline Exploratory Data Analysis (EDA) Technical Documentation

**Capstone Project -- Master of Data Analytics**

------------------------------------------------------------------------

## 1. Objective of Baseline EDA

The objective of the baseline Exploratory Data Analysis (EDA) phase is
to understand the structural composition of the dataset prior to
advanced modeling and hypothesis testing.

This stage focuses on:

-   Order status distribution
-   Return rate behavior across product categories
-   Geographic variation in return rates
-   Marketing channel variation in return rates

This phase is strictly descriptive. No inferential statistical testing
is performed here.

------------------------------------------------------------------------

## 2. Order Item Status Distribution

### Figure: Order Item Status Distribution

**Observation:**

-   Shipped: 53,931
-   Complete: 45,277
-   Processing: 36,215
-   Cancelled: 27,277
-   Returned: 18,208

Returned items represent a meaningful subset of total order volume.

**Interpretation:**

Returns are not marginal events. The presence of over 18,000 returned
items indicates operational and financial exposure significant enough to
warrant deeper economic analysis.

This validates the relevance of profit erosion modeling in subsequent
research questions.

------------------------------------------------------------------------

## 3. Return Rate by Category (Top 15, Min 200 Items)

### Figure: Top Categories by Return Rate

Highest return rate categories (\~10--11% range):

-   Blazers & Jackets
-   Maternity
-   Suits
-   Clothing Sets
-   Active

**Interpretation:**

Return rates are relatively clustered within a narrow 10--11% band
across high-volume categories. There is no extreme outlier category in
terms of return frequency alone.

This suggests that **frequency alone may not fully explain financial
risk**, reinforcing the need to examine margin severity.

------------------------------------------------------------------------

## 4. Return Rate by Category and Country

### Figure: Heatmap -- Category × Country

**Observations:**

-   Moderate geographic variability (typically 8--13% range)
-   Some country-category pairs exceed 14--15%
-   Germany, Spain, and UK show higher values in certain apparel
    categories

**Interpretation:**

Return behavior varies across markets, but differences are moderate
rather than extreme. Geographic patterns may reflect:

-   Sizing differences
-   Customer expectations
-   Local return policies

However, variability does not appear structurally extreme at this stage.

------------------------------------------------------------------------

## 5. Return Rate by Category and Traffic Source

### Figure: Heatmap -- Category × Traffic Source

**Observations:**

-   Email and Facebook traffic sometimes show elevated return rates
    (\~12--14%)
-   Organic and Search appear more stable (\~9--11%)

**Interpretation:**

Marketing channel may influence return propensity. Promotional campaigns
or ad-driven purchases may lead to higher mismatch between expectation
and product experience.

This suggests marketing-source interaction could be a candidate feature
for predictive modeling.

------------------------------------------------------------------------

## 6. Baseline EDA Summary

Baseline EDA establishes:

1.  Returns represent a non-trivial operational event.
2.  Return rates are moderately stable across categories.
3.  Geographic and traffic source variations exist but are not extreme.
4.  Frequency-based analysis alone does not reveal economic severity.

This phase provides contextual grounding before margin and profit
erosion analysis.

------------------------------------------------------------------------
