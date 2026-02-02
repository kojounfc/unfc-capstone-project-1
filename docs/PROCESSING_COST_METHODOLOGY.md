# Processing Cost Methodology for Profit Erosion Analysis

## Document Purpose

This document defines the methodology, assumptions, and rationale for calculating return processing costs in the Profit Erosion E-commerce Capstone Project. These decisions are based on empirical analysis of the TheLook e-commerce dataset and industry benchmarks.

---

## 1. Executive Summary

**Decision:** Implement a **category-tiered processing cost model** with three tiers based on product category margin characteristics.

| Tier | Multiplier | Base Cost | Effective Cost | Categories |
|------|------------|-----------|----------------|------------|
| Standard | 1.0x | $12.00 | $12.00 | 10 categories |
| Moderate | 1.15x | $12.00 | $13.80 | 8 categories |
| Premium | 1.3x | $12.00 | $15.60 | 8 categories |

**Key Justification:**
- Margin variation across categories: **59.4% CV** (exceeds 15% threshold)
- Return rate variation across countries: **3.58% CV** (below 10% threshold)
- Category tiering captures margin-at-risk differences; country-specific rates not warranted

---

## 2. Research Questions Addressed

This methodology supports:
- **RQ1 (Descriptive):** Quantify profit erosion across product categories
- **RQ2 (Segmentation):** Identify customer segments with differential profit erosion
- **RQ3 (Predictive):** Predict high profit erosion customers
- **RQ4 (Regression):** Model marginal associations between behaviors and profit erosion

---

## 3. Data Analysis Summary

### 3.1 Dataset Overview

| Metric | Value |
|--------|-------|
| Total items | 180,908 |
| Total returned items | 18,208 |
| Overall return rate | 10.06% |
| Countries | 15 |
| Product categories | 26 |

### 3.2 Geographic Analysis (Country-Level)

**Finding:** Return rates are remarkably consistent across all markets.

| Country | Items | Returns | Return Rate | Avg Margin |
|---------|-------|---------|-------------|------------|
| China | 61,761 | 6,045 | 9.79% | $30.83 |
| United States | 40,802 | 4,193 | 10.28% | $30.91 |
| Brasil | 25,893 | 2,651 | 10.24% | $31.40 |
| South Korea | 9,678 | 974 | 10.06% | $30.92 |
| France | 8,458 | 813 | 9.61% | $30.88 |
| United Kingdom | 8,216 | 829 | 10.09% | $31.31 |
| Germany | 7,722 | 834 | 10.80% | $31.84 |
| Spain | 7,311 | 715 | 9.78% | $31.35 |
| Japan | 4,554 | 482 | 10.58% | $29.88 |
| Australia | 3,846 | 395 | 10.27% | $31.80 |

**Variance Analysis:**
- Return rate range: 9.61% (France) to 10.80% (Germany)
- Difference: 1.19 percentage points
- Mean return rate: 10.04%
- Standard deviation: 0.36%
- **Coefficient of variation: 3.58%**

**Decision:** Country-specific processing costs are **NOT warranted**. The low variance (CV < 10%) indicates operational costs are effectively uniform across geographies.

### 3.3 Category-Level Analysis (Returned Items Only)

**Finding:** Significant margin variation across product categories for returned items.

**Top Categories by Margin Lost (Returned Items):**

| Category | Returns | Avg Margin | Total Margin Lost | % of Total |
|----------|---------|------------|-------------------|------------|
| Outerwear & Coats | 931 | $82.38 | $76,694 | 13.59% |
| Jeans | 1,286 | $44.60 | $57,360 | 10.16% |
| Sweaters | 1,109 | $39.72 | $44,053 | 7.81% |
| Suits & Sport Coats | 492 | $74.24 | $36,526 | 6.47% |
| Fashion Hoodies & Sweatshirts | 1,192 | $26.18 | $31,204 | 5.53% |
| Sleep & Lounge | 1,094 | $27.08 | $29,630 | 5.25% |

**Variance Analysis (Returned Items):**
- Average margin for returned items: $30.99
- Median margin for returned items: $20.52
- Margin CV across categories: **59.4%**
- Price CV across categories: **55.0%**

**Decision:** Category-based tiering is **JUSTIFIED**. The high margin variance (CV > 15%) indicates different processing cost multipliers are appropriate based on the margin at risk.

---

## 4. Processing Cost Components

### 4.1 Base Cost Structure ($12.00 per return)

| Component | Cost | Description | Rationale |
|-----------|------|-------------|-----------|
| Customer Care | $4.00 | Phone/email support time for return requests | Based on avg handling time of 10-15 min at $16-24/hr |
| Inspection | $2.50 | Quality assessment upon receipt | Visual and functional check, avg 5-8 min |
| Restocking | $3.00 | Shelving, inventory system updates | Physical handling plus system updates |
| Logistics | $2.50 | Reverse shipping handling, label generation | Administrative processing, label costs |
| **Total** | **$12.00** | | |

### 4.2 Literature Support

The base cost of $12 per return is conservative compared to industry benchmarks:

1. **Rogers & Tibben-Lembke (2001):** Reverse logistics costs typically 4-10% of revenue
2. **Guide & Van Wassenhove (2009):** Closed-loop supply chain processing costs $10-25 per unit
3. **Optoro (2019):** Average return processing cost $10-20 for e-commerce

Our $12 base represents a mid-range estimate appropriate for a synthetic dataset where exact operational costs are unknown.

### 4.3 Category Tier Multipliers

Based on returned item analysis, categories are assigned multipliers reflecting:
- **Price complexity:** Higher-priced items require more careful handling
- **Margin at stake:** Higher margins represent greater economic risk
- **Inspection intensity:** Premium items often require detailed quality checks

---

## 5. Category-to-Tier Mapping

### 5.1 Premium Tier (1.3x = $15.60 per return)

**Characteristics:** High-margin items ($40-85 avg margin), require careful handling

| Category | Returns | Avg Margin | Avg Price | Total Lost |
|----------|---------|------------|-----------|------------|
| Outerwear & Coats | 931 | $82.38 | $148.17 | $76,694 |
| Suits & Sport Coats | 492 | $74.24 | $124.32 | $36,526 |
| Blazers & Jackets | 354 | $55.53 | $89.78 | $19,657 |
| Jeans | 1,286 | $44.60 | $96.35 | $57,360 |
| Dresses | 541 | $44.32 | $80.39 | $23,978 |
| Suits | 110 | $44.43 | $111.56 | $4,887 |
| Sweaters | 1,109 | $39.72 | $76.82 | $44,053 |
| Pants | 739 | $32.81 | $60.78 | $24,245 |

**Tier Summary:**
- Total returns: 5,562
- Average margin: $52.25
- Total margin lost: $287,399

### 5.2 Moderate Tier (1.15x = $13.80 per return)

**Characteristics:** Mid-margin items ($22-35 avg margin), standard handling

| Category | Returns | Avg Margin | Avg Price | Total Lost |
|----------|---------|------------|-----------|------------|
| Skirts | 222 | $34.79 | $57.81 | $7,723 |
| Active | 947 | $28.24 | $48.91 | $26,747 |
| Swim | 1,069 | $27.62 | $56.40 | $29,523 |
| Maternity | 570 | $27.62 | $49.66 | $15,743 |
| Sleep & Lounge | 1,094 | $27.08 | $52.26 | $29,630 |
| Accessories | 996 | $26.40 | $43.89 | $26,297 |
| Pants & Capris | 356 | $26.15 | $55.56 | $9,310 |
| Fashion Hoodies & Sweatshirts | 1,192 | $26.18 | $54.52 | $31,204 |
| Shorts | 1,158 | $21.95 | $44.04 | $25,418 |

**Tier Summary:**
- Total returns: 7,248
- Average margin: $27.49
- Total margin lost: $192,284

### 5.3 Standard Tier (1.0x = $12.00 per return)

**Characteristics:** Lower-margin items ($8-20 avg margin), basic handling

| Category | Returns | Avg Margin | Avg Price | Total Lost |
|----------|---------|------------|-----------|------------|
| Plus | 430 | $19.21 | $38.78 | $8,259 |
| Tops & Tees | 1,185 | $17.95 | $40.78 | $21,265 |
| Intimates | 1,302 | $15.82 | $33.76 | $20,596 |
| Underwear | 727 | $15.08 | $28.44 | $10,962 |
| Leggings | 333 | $10.77 | $27.03 | $3,585 |
| Socks & Hosiery | 356 | $9.13 | $15.31 | $3,250 |
| Socks | 602 | $7.89 | $19.87 | $4,750 |
| Jumpsuits & Rompers | 84 | $21.98 | - | $1,847 |
| Clothing Sets | 23 | $34.24 | - | $788 |

**Note:** Low-volume categories (< 100 returns) assigned to Standard tier by default.

**Tier Summary:**
- Total returns: 5,291 (including low-volume)
- Average margin: $15.25
- Total margin lost: $81,977

---

## 6. Implementation Specification

### 6.1 Constants Definition

```python
# Base processing cost components ($12 total per return)
DEFAULT_COST_COMPONENTS: Dict[str, float] = {
    "customer_care": 4.0,   # Phone/email support time
    "inspection": 2.5,      # Quality assessment
    "restocking": 3.0,      # Shelving, inventory updates
    "logistics": 2.5,       # Reverse shipping handling
}

# Category tier multipliers based on margin-at-risk analysis
CATEGORY_TIER_MULTIPLIERS: Dict[str, float] = {
    # Premium Tier (1.3x) - High-margin, careful handling
    "Outerwear & Coats": 1.3,
    "Suits & Sport Coats": 1.3,
    "Blazers & Jackets": 1.3,
    "Jeans": 1.3,
    "Dresses": 1.3,
    "Suits": 1.3,
    "Sweaters": 1.3,
    "Pants": 1.3,
    # Moderate Tier (1.15x) - Mid-margin, standard handling
    "Skirts": 1.15,
    "Active": 1.15,
    "Swim": 1.15,
    "Maternity": 1.15,
    "Sleep & Lounge": 1.15,
    "Accessories": 1.15,
    "Pants & Capris": 1.15,
    "Fashion Hoodies & Sweatshirts": 1.15,
    "Shorts": 1.15,
    # Standard Tier (1.0x) - Lower-margin, basic handling
    "Plus": 1.0,
    "Tops & Tees": 1.0,
    "Intimates": 1.0,
    "Underwear": 1.0,
    "Leggings": 1.0,
    "Socks & Hosiery": 1.0,
    "Socks": 1.0,
    "Jumpsuits & Rompers": 1.0,
    "Clothing Sets": 1.0,
}

# Default multiplier for unknown categories
DEFAULT_CATEGORY_MULTIPLIER: float = 1.0
```

### 6.2 Profit Erosion Formula

For each returned item:

```
profit_erosion = margin_reversal + process_cost

Where:
  margin_reversal = item_margin
  process_cost = base_cost × category_multiplier
  base_cost = sum(DEFAULT_COST_COMPONENTS) = $12.00
  category_multiplier = CATEGORY_TIER_MULTIPLIERS.get(category, 1.0)
```

### 6.3 Function Signature

**Design Decision:** Functions receive **pre-filtered returned items only** (where `is_returned_item == 1`).

**Rationale:**
- All research questions (RQ1-RQ4) focus specifically on returns and profit erosion
- EDA uses full dataset; profit erosion calculations only need returned items
- Improves efficiency: 18,208 returned items vs 180,908 total items
- Cleaner logic without conditional checks for non-returned items

```python
def calculate_profit_erosion(
    df: pd.DataFrame,
    cost_components: Optional[Dict[str, float]] = None,
    category_multipliers: Optional[Dict[str, float]] = None,
    use_category_tiers: bool = True,
) -> pd.DataFrame:
    """
    Calculate profit erosion metrics for returned items.

    Args:
        df: DataFrame containing ONLY returned items (pre-filtered where
            is_returned_item == 1). Must have item_margin and category columns.
        cost_components: Base cost breakdown (default $12 total).
        category_multipliers: Category-to-multiplier mapping.
        use_category_tiers: If True, apply category multipliers; if False, flat rate.

    Returns:
        DataFrame with columns added:
        - margin_reversal: item_margin (the margin lost on this return)
        - process_cost: base_cost × category_multiplier
        - profit_erosion: margin_reversal + process_cost
    """
```

### 6.4 Pipeline Usage

```python
# EDA uses full dataset
df_full = build_analysis_dataset()
df_full = engineer_return_features(df_full)
df_full = calculate_margins(df_full)

# Filter to returned items for profit erosion analysis
returned_df = df_full[df_full["is_returned_item"] == 1].copy()

# Calculate profit erosion (only on returned items)
returned_df = calculate_profit_erosion(returned_df)

# Aggregate to order/customer level
order_erosion = aggregate_profit_erosion_by_order(returned_df)
customer_erosion = aggregate_profit_erosion_by_customer(returned_df)
```

---

## 7. Sensitivity Analysis Recommendations

To validate the robustness of findings, the following sensitivity analyses are recommended:

### 7.1 Base Cost Sensitivity
- Test range: $8.00 to $18.00 per return
- Expected impact: Linear scaling of total processing costs

### 7.2 Tier Multiplier Sensitivity
- Test alternative multipliers: (0.9x, 1.0x, 1.2x) vs (1.0x, 1.15x, 1.3x)
- Expected impact: Moderate change in category-level erosion rankings

### 7.3 Tier Boundary Sensitivity
- Test moving categories between tiers
- Focus on borderline categories (Shorts, Pants & Capris)

---

## 8. Limitations and Assumptions

### 8.1 Assumptions

1. **Synthetic Data:** TheLook is a synthetic dataset; actual operational costs may differ
2. **Uniform Returns Processing:** Assumes all returns follow similar operational flow
3. **No Seasonality in Costs:** Processing costs assumed constant year-round
4. **No Salvage Value Modeling:** Does not account for resale/liquidation value of returns

### 8.2 Limitations

1. **No Real Cost Data:** Base cost derived from literature, not actual operations
2. **Category Granularity:** Some categories may have sub-segments with different profiles
3. **Country Effects:** While return rates are uniform, actual logistics costs may vary

---

## 9. References

1. Rogers, D. S., & Tibben-Lembke, R. S. (2001). An examination of reverse logistics practices. *Journal of Business Logistics*, 22(2), 129-148.

2. Guide, V. D. R., & Van Wassenhove, L. N. (2009). The evolution of closed-loop supply chain research. *Operations Research*, 57(1), 10-18.

3. Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? Antecedents and consequences. *Journal of Marketing*, 73(3), 35-51.

4. Cui, H., Rajagopalan, S., & Ward, A. R. (2020). Predicting product return volume using machine learning methods. *European Journal of Operational Research*, 281(3), 612-627.

---

## 10. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-30 | Analysis Team | Initial methodology document |

---

## Appendix A: Raw Analysis Output

### A.1 Country-Level Return Rate Variance

```
Highest return rate (Germany): 10.80%
Lowest return rate (France):    9.61%
Difference:                     1.19 percentage points
Mean return rate:              10.04%
Standard deviation:             0.36%
Coefficient of variation:       3.58%
```

### A.2 Category-Level Margin Variance (Returned Items)

```
Overall stats for RETURNED items:
  Total returns: 18,208
  Average margin: $30.99
  Median margin: $20.52
  Average sale price: $59.71

Coefficient of Variation (CV) across categories:
  Margin CV: 0.594 (59.4%)
  Price CV: 0.550 (55.0%)
```

### A.3 Decision Thresholds Applied

| Metric | Threshold | Observed | Decision |
|--------|-----------|----------|----------|
| Country return rate CV | < 10% | 3.58% | No country tiers |
| Category margin CV | > 15% | 59.4% | Category tiers needed |

---

*Document prepared for: DAMO-699-4 Capstone Project, University of Niagara Falls, Canada*
