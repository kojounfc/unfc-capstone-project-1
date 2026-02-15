# RQ1 Technical Documentation  
**Capstone Project – Master of Data Analytics**  
**Research Question 1 (RQ1)**

---

## 1. Research Question

**RQ1:**  
*Do returned items exhibit statistically significant differences in profit erosion across product categories and brands?*

This research question evaluates whether the **economic impact of product returns**, measured through profit erosion, varies systematically across product categories and brands in an e-commerce enviro[...]  

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

Note: US07 processed outputs used by RQ1 are return-focused and are written to `data/processed/rq1/`. If return rate denominators that include non-returned exposure are needed, downstream RQ pipelines recompute exposure from raw transactions.

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
Distributional normality of profit erosion within each category and brand group was assessed using the **Shapiro–Wilk test**. The distributions exhibited significant skewness and heavy tails, consis[...]  

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
Following statistically significant Kruskal–Wallis results, **Dunn’s post-hoc tests with Bonferroni correction** were conducted to identify pairwise differences between categories and brands while[...]  

---

## 6.5 Bootstrap Confidence Intervals (Supplementary)

To complement non-parametric hypothesis testing and provide robust uncertainty estimates for group means, bootstrap 95% confidence intervals were computed for mean profit erosion by group. Bootstrap tables and CI visualizations are produced and saved as reproducible artifacts (see Reproducibility & Visualization Standards).

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
Several product categories, including **Outerwear & Coats**, **Jeans**, and **Sweaters**, were identified as leading contributors to total profit erosion. These categories exhibit either high return v[...]  

#### Brands
At the brand level, profit erosion was unevenly distributed. Some brands contributed heavily to total profit erosion due to return volume, while others exhibited high erosion per return driven by prem[...]  

#### Return Rate vs. Profit Erosion
Analysis of return rate versus mean profit erosion per return revealed a weak linear relationship, indicating that return frequency alone does not adequately explain the economic impact of returns.

---

## 8. Interpretation

### 8.1 Category-Level Interpretation
The results demonstrate that product categories differ substantially in the economic cost of returns. Categories with similar return rates can generate markedly different profit erosion outcomes due t[...]  

High-margin and structurally complex categories incur significantly higher losses per return compared to basic or commodity categories.

---

### 8.2 Brand-Level Interpretation
Brand-level heterogeneity in profit erosion suggests that return-related financial risk is not evenly distributed across brands. This variation cannot be explained by return rate alone and reflects di[...]  

---

### 8.3 Implications of Return Rate as a Metric
The weak association between return rate and profit erosion confirms that return rate is an incomplete proxy for financial risk. Profit erosion provides a more economically meaningful measure for prio[...]  

---

## 9. Limitations

- The dataset used is synthetic and may not fully capture real-world consumer behavior.
- Return processing costs are modeled rather than directly observed.
- Recovery or resale value of returned items is not explicitly incorporated.

These limitations are consistent with the scope and objectives of an academic capstone project.

---

## 10. Conclusion (RQ1)

RQ1 provides strong empirical evidence that **profit erosion from product returns differs significantly across both product categories and brands**. The null hypothesis was rejected in all tested dime[...]  

These findings establish a rigorous descriptive and inferential foundation for subsequent research questions, particularly **RQ2**, where customer segmentation will incorporate economically meaningful[...]  

---

## 11. Traceability to User Stories

- **US06:** Return feature engineering and profit erosion computation  
- **US07:** Descriptive aggregation and reporting of profit erosion metrics  
- **RQ1:** Statistical validation of cross-category and cross-brand differences  

---

## 12. Reproducibility & Visualization Standards (post-refactor)

Following the RQ1 refactor, visualization and pipeline orchestration were standardized to improve reproducibility and CI-safety:

- Centralized visualization API: All RQ1 visuals are implemented in `src/rq1_visuals.py` (functions include `plot_top_groups_total_erosion`, `plot_return_rate_vs_mean_erosion`, `plot_profit_erosion_distribution_log`, and `plot_bootstrap_ci_mean_by_group`). This centralization enforces consistent styling and deterministic outputs.
- Deterministic artifact locations: All RQ1 figures and CI tables are written deterministically to Figures and processed data directories:
  - Figures: `figures/rq1/`
  - Processed inputs and CI tables: `data/processed/rq1/`
  Storing artifacts at fixed paths enables CI to assert outputs and makes notebook rendering headless-friendly.
- Log-scale distributions & bootstrap confidence intervals: To support non-parametric inference and robust effect-size interpretation, a log-transformed erosion distribution plot is included and bootstrap 95% confidence interval visualizations and CSV tables are produced and saved. These supplement the Kruskal–Wallis and Dunn post-hoc tests.
- Notebook orchestration: The primary analysis notebook (`profit_erosion_analysis.ipynb`) was refactored to remove inline rendering. Notebooks now operate as orchestration layers that reference saved PNGs (one-visual-per-cell) so figures are generated deterministically by the pipeline and displayed by embedding the saved image files.
- Code structure: RQ1 code was migrated from the legacy `src/rq1/` package to top-level modules: `src/rq1_run.py`, `src/rq1_stats.py`, and `src/rq1_visuals.py`. Tests were added to validate statistical outputs and figure generation to improve CI reliability.

These refactorings do not change the analytical decisions documented above (units of analysis, tests, effect size), but they do affect how figures and intermediate tables are produced, stored, and consumed by downstream reporting.