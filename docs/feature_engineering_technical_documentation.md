# Feature Engineering Validation Technical Documentation

**Capstone Project -- Master of Data Analytics**

------------------------------------------------------------------------

## 1. Objective

This section validates the outputs of the US06 Feature Engineering
Pipeline, particularly:

-   Item-level margin calculation
-   Margin distribution structure
-   Margin exposure concentration
-   Category-level total margin loss

This phase confirms that engineered economic features behave
consistently and logically before use in RQ1--RQ4.

------------------------------------------------------------------------

## 2. Margin Distribution -- All Items

### Figure: Margin Distribution (All Items)

**Median Margin:** \~\$20.23

**Observations:**

-   Strong right skew
-   Heavy upper tail
-   Numerous high-value outliers

**Interpretation:**

The distribution is non-normal. Most products generate moderate margins,
while a small number of items generate extremely high margins.

This heavy-tail structure implies that returns of high-margin items will
disproportionately impact profit erosion.

------------------------------------------------------------------------

## 3. Margin Distribution -- Returned Items Only

### Figure: Margin Distribution (Returned Only)

**Median Margin:** \~\$20.52

**Observation:**

The distribution of margins among returned items closely mirrors the
full-item distribution.

**Interpretation:**

Returns are not concentrated only among low-margin items. High-margin
products are also being returned.

This supports the hypothesis that **profit erosion is severity-driven**,
not purely frequency-driven.

------------------------------------------------------------------------

## 4. Customer Margin Exposure

### Figure: Top 20 Customers by Margin Exposure

**Observation:**

Top customers individually generate \$450--\$660 in total lost margin
from returns.

**Interpretation:**

Loss exposure is partially concentrated at the customer level. A small
subset of customers contribute disproportionately to total margin loss.

This finding later supports RQ2 concentration analysis.

------------------------------------------------------------------------

## 5. Category-Level Margin Loss from Returns

### Figure: Top 15 Categories by Margin Loss

Highest categories:

-   Outerwear & Coats (\~\$76K)
-   Jeans (\~\$57K)
-   Sweaters (\~\$44K)
-   Suits & Sport Coats (\~\$36K)

**Interpretation:**

High-margin structured apparel categories dominate total economic loss.
Even if return rates are similar, margin severity amplifies financial
impact.

This validates the operational relevance of category-level profit
erosion modeling in RQ1.

------------------------------------------------------------------------

## 6. Feature Engineering Validation Summary

The engineered economic variables demonstrate:

1.  Logical distributional structure (right-skewed margins)
2.  Presence of economically meaningful outliers
3.  Customer-level exposure concentration
4.  Category-level severity asymmetry

The feature engineering pipeline produces economically coherent signals
suitable for inferential and predictive modeling.

------------------------------------------------------------------------
