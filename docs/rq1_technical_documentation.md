# RQ1 Technical Documentation  
**Capstone Project – Master of Data Analytics**  
**Research Question 1 (RQ1)**

---

## 1. Research Question

**RQ1:**  
*Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?*

This research question evaluates whether the **economic impact of product returns**, measured through profit erosion, varies systematically across product categories and brands in an e-commerce environment.

---

## 2. Hypotheses

All hypothesis testing was conducted at a significance level of **α = 0.05**, in accordance with the standardized RQ methodology template.

- **H₀ (Null Hypothesis):**  
  Mean profit erosion is equal across product categories and brands.

- **H₁ (Alternative Hypothesis):**  
  Mean profit erosion differs significantly across product categories and brands.

---

## 3. Data Scope and Unit of Analysis

- **Unit of analysis:** Order-item level  
- **Dataset:** Consolidated e-commerce dataset derived from TheLook (synthetic)  
- **Filtering rule:** Only returned items (`is_returned_item = 1`) were included in profit erosion calculations  

Each observation represents a single returned product item, ensuring that profit erosion is measured at the most granular transactional level.

---

## 4. Feature Engineering and Profit Erosion Definition (US06)

Profit erosion was operationalized using the standardized **US06 feature engineering pipeline**.

For each returned item, profit erosion was defined as follows:

- **Margin Reversal:**  
  The item-level contribution margin lost due to the return (`item_margin`).

- **Return Processing Cost:**  
  A modeled reverse-logistics cost composed of:
  - customer service handling  
  - inspection  
  - restocking  
  - logistics  

  A **category-tiered multiplier** was applied to the base processing cost to reflect differences in handling complexity across product categories.

- **Profit Erosion Formula:**  

  \[
  \text{Profit Erosion} = \text{Margin Reversal} + \text{Processing Cost}
  \]

Processing costs are modeled based on documented industry benchmarks and assumptions defined in the project proposal. These assumptions are explicitly acknowledged as a limitation.

---

## 5. Descriptive Aggregation (US07)

Using **US07 descriptive transformation functions**, profit erosion metrics were aggregated across the following dimensions:

- Product category  
- Brand  
- Department  

For each aggregation level, the following metrics were computed:

- Total profit erosion  
- Mean profit erosion per return  
- Number of returned items  
- Return rate (computed using all items as the denominator for contextual comparison)

These descriptive outputs were used for ranking, visualization, and subsequent statistical analysis.

---

## 6. Statistical Methodology

### 6.1 Normality Assessment
Distributional normality of profit erosion within each category and brand group was assessed using the **Shapiro–Wilk test**. The distributions exhibited significant skewness and heavy tails, consistent with cost and margin data.

Given the violation of normality assumptions, parametric tests were deemed inappropriate.

---

### 6.2 Primary Statistical Test
The **Kruskal–Wallis H test** was applied to evaluate differences in mean profit erosion across:

- Product categories  
- Brands  

This non-parametric test is suitable for comparing group medians when distributional assumptions are not met.

---

### 6.3 Effect Size
Effect size was quantified using **epsilon-squared (ε²)** to assess the practical significance of observed differences beyond statistical significance.

A threshold of **ε² ≥ 0.06** was used to determine meaningful effect magnitude, as specified in the RQ methodology template.

---

### 6.4 Post-Hoc Analysis
Following statistically significant Kruskal–Wallis results, **Dunn’s post-hoc tests with Bonferroni correction** were conducted to identify pairwise differences between categories and brands while controlling for multiple comparisons.

---

## 7. Results

### 7.1 Statistical Test Results

The statistical analysis produced the following outcomes:

- **Product Categories**
  - Null hypothesis rejected (p < 0.05)
  - Effect size exceeded the predefined threshold (ε² ≥ 0.06)

- **Brands**
  - Null hypothesis rejected (p < 0.05)
  - Effect size exceeded the predefined threshold (ε² ≥ 0.06)

Both category-level and brand-level analyses meet the **success criteria** defined in the RQ methodology template.

---

### 7.2 Descriptive Results

#### Categories
Several product categories, including **Outerwear & Coats**, **Jeans**, and **Sweaters**, were identified as leading contributors to total profit erosion. These categories exhibit either high return volumes, high profit erosion per return, or a combination of both.

#### Brands
At the brand level, profit erosion was unevenly distributed. Some brands contributed heavily to total profit erosion due to return volume, while others exhibited high erosion per return driven by premium pricing and margin structures.

#### Return Rate vs. Profit Erosion
Analysis of return rate versus mean profit erosion per return revealed a weak linear relationship, indicating that return frequency alone does not adequately explain the economic impact of returns.

---

## 8. Interpretation

### 8.1 Category-Level Interpretation
The results demonstrate that product categories differ substantially in the economic cost of returns. Categories with similar return rates can generate markedly different profit erosion outcomes due to differences in margin structure and return processing complexity.

High-margin and structurally complex categories incur significantly higher losses per return compared to basic or commodity categories.

---

### 8.2 Brand-Level Interpretation
Brand-level heterogeneity in profit erosion suggests that return-related financial risk is not evenly distributed across brands. This variation cannot be explained by return rate alone and reflects differences in pricing strategies, margin composition, and product mix.

---

### 8.3 Implications of Return Rate as a Metric
The weak association between return rate and profit erosion confirms that return rate is an incomplete proxy for financial risk. Profit erosion provides a more economically meaningful measure for prioritizing return management strategies.

---

## 9. Limitations

- The dataset used is synthetic and may not fully capture real-world consumer behavior.
- Return processing costs are modeled rather than directly observed.
- Recovery or resale value of returned items is not explicitly incorporated.

These limitations are consistent with the scope and objectives of an academic capstone project.

---

## 10. Conclusion (RQ1)

RQ1 provides strong empirical evidence that **profit erosion from product returns differs significantly across both product categories and brands**. The null hypothesis was rejected in all tested dimensions, and effect size thresholds were satisfied.

These findings establish a rigorous descriptive and inferential foundation for subsequent research questions, particularly **RQ2**, where customer segmentation will incorporate economically meaningful return behavior rather than return frequency alone.

---

## 11. Traceability to User Stories

- **US06:** Return feature engineering and profit erosion computation  
- **US07:** Descriptive aggregation and reporting of profit erosion metrics  
- **RQ1:** Statistical validation of cross-category and cross-brand differences  

---