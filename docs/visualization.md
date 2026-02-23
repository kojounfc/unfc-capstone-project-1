# Visualization Module - Technical Reference (`src/visualization.py`)

_Last updated: 2026-02-23_

## Executive Summary

`visualization.py` centralizes plotting functions for:
- **Baseline / descriptive EDA** (dataset composition and return patterns)
- **Feature Engineering validation** (post-FE sanity checks)
- **RQ1 visuals** (profit erosion narrative visuals)
- **RQ2 visuals** (concentration + clustering diagnostics)

The notebook should:
1) build/prepare the dataset
2) call plotting functions
3) save and display figures from disk

The visualization module should:
- validate required columns
- generate a figure deterministically
- save to a provided `save_path` / `out_path`

---

## 1. Plot Style and Utility

### 1.1 `set_plot_style()`

**Purpose:** Applies consistent plot style across the project.

**Signature:**
```python
def set_plot_style():
    ...
```

### 1.2 `_safe_tight_layout()`

**Purpose:** Applies `plt.tight_layout()` while suppressing layout warnings for small figures.

---

## 2. Baseline Descriptive EDA (Legacy EDA)

These are **visual EDA** functions. They do **not** run hypothesis tests; they provide descriptive evidence used to guide later analysis.

### 2.1 Status distribution

- **Function:** `plot_status_distribution(df, ...)`
- **Required columns:** `item_status`

### 2.2 Return rate by category

- **Function:** `plot_return_rate_by_category(df, ...)`
- **Required columns:** `category`, `order_id`, `is_returned_item`
- **Note:** `is_returned_item` is expected to exist **after** return-flag feature engineering (or must be created minimally from status).

### 2.3 Return-rate heatmap

- **Function:** `plot_return_rate_heatmap(df, row_col="category", col_col="traffic_source", ...)`
- **Required columns:** `row_col`, `col_col`, `order_id`, `is_returned_item`

### 2.4 Margin distribution

- **Function:** `plot_margin_distribution(df, returned_only=False, ...)`
- **Required columns:** `item_margin`
- **Placement:** **Post Feature Engineering** (after margins are computed)

### 2.5 Margin loss by category (returned items)

- **Function:** `plot_margin_loss_by_category(df, ...)`
- **Required columns:** `category`, `is_returned_item`, `item_margin`

### 2.6 Customer margin exposure (returned items)

- **Function:** `plot_customer_margin_exposure(df, ...)`
- **Required columns:** `user_id`, `is_returned_item`, `item_margin`, `order_id`

### 2.7 Returned item cost/price/margin grid by country

- **Function:** `plot_price_margin_returned_by_status_country(agg_df, ...)`
- **Expected input:** output of an aggregator such as `calculate_price_margin_returned_by_country(...)`
- **Required columns in input:** `country`, `avg_cost`, `total_cost`, `avg_sale_price`, `total_sale_price`, `avg_margin`, `total_margin`, `item_count`

---

## 3. RQ1 Visuals (now centralized in `visualization.py`)

These plots are designed to support RQ1’s narrative:

Descriptive Impact → Mechanism → Statistical Justification → Inference Stability

### 3.1 Top groups by total erosion

- `plot_top_groups_total_erosion(df, group_col=..., value_col="total_profit_erosion", ...)`

### 3.2 Return rate vs mean erosion (mechanism view)

- `plot_return_rate_vs_mean_erosion(df, x_col="return_rate", y_col="avg_profit_erosion", ...)`

### 3.3 Severity vs volume decomposition

- `plot_severity_vs_volume_decomposition(df, ...)`

### 3.4 Profit erosion distribution (log scale)

- `plot_profit_erosion_distribution_log(returned_df, value_col="profit_erosion", ...)`

### 3.5 Bootstrap confidence intervals

- `plot_bootstrap_ci_mean_by_group(returned_df, group_col=..., value_col="profit_erosion", ...)`
- **Output:** returns a CI table (DataFrame) and saves a CI figure to `out_path`

---

## 4. RQ2 Visuals (Concentration & Segmentation)

### 4.1 Concentration ranking

- `plot_feature_concentration_ranking(concentration_df, ...)`

### 4.2 Gini vs Pareto share scatter

- `plot_gini_vs_pareto_scatter(concentration_df, ...)`

### 4.3 Pareto curve

- `plot_pareto_curve(pareto_df, gini, ...)`

### 4.4 Lorenz curve

- `plot_lorenz_curve(lorenz_df, gini, ...)`

### 4.5 Clustering diagnostics

- `plot_clustering_diagnostics(elbow_df, silhouette_df, optimal_k, ...)`

### 4.6 Cluster erosion comparison

- `plot_cluster_erosion_comparison(cluster_summary_df, optimal_k, ...)`

### 4.7 Clustering feature importance

- `plot_clustering_feature_importance(feature_importance_df, ...)`

---

## 5. Notebook Integration Pattern (No Double-Rendering)

Recommended notebook pattern:
1) call plot function with save path
2) display the saved image
3) close the matplotlib figure if returned

Example:

```python
out_path = FIGURES_DIR / "eda" / "status_distribution.png"
fig = plot_status_distribution(df, save_path=str(out_path))
plt.close(fig)
display(Image(filename=str(out_path)))
```

This prevents the “figure appears twice” behavior in Jupyter.

---

## 6. Where Each Visual Belongs in the Pipeline

**Baseline Descriptive EDA (pre-FE):**
- status distribution
- return rate by category (if `is_returned_item` exists or is minimally created)

**Post Feature Engineering validation (post-FE):**
- margin distribution
- margin loss by category
- customer margin exposure
- country cost/price/margin grid

**RQ1 / RQ2:**
- use their dedicated visuals after the corresponding transformation steps.

---

## Summary

`visualization.py` provides a unified, reproducible plotting layer that supports:
- early descriptive profiling
- post-FE validation
- RQ1 research narrative visuals
- RQ2 concentration and segmentation diagnostics
