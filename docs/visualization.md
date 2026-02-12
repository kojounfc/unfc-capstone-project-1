# RQ1 Visual Standards & Module Reference

## Executive Summary

The `src/rq1_visuals.py` module provides publication-quality, research-aligned visualizations for **Research Question 1 (RQ1)**:

> Do product-level dimensions (category, brand, department) significantly differ in profit erosion?

This module replaces the legacy `visualization.py` EDA module.

The RQ1 visuals are designed to:

- Identify high-impact product groups
- Explain erosion mechanisms (frequency vs severity)
- Justify statistical testing assumptions
- Provide bootstrap-based inference robustness
- Produce thesis-ready, reproducible figures

All figures are saved under:

    figures/rq1/

Bootstrap CI tables are saved under:

    data/processed/rq1/

The notebook orchestrates execution only.  
All matplotlib logic is centralized inside `src/rq1_visuals.py`.

---

# 1. RQ1 Visual Pipeline

RQ1 visuals follow a structured academic narrative:

1. Top Categories by Total Profit Erosion
2. Top Brands by Total Profit Erosion
3. Return Rate vs Mean Erosion (Mechanism View)
4. Severity vs Volume Decomposition
5. Distribution of Profit Erosion (Log Scale)
6. Bootstrap 95% Confidence Intervals

Narrative structure:

Descriptive Impact → Mechanism → Statistical Justification → Inference Stability

---

# 2. Core Visualization Functions

## 2.1 plot_top_groups_total_erosion()

Purpose:
Rank product groups by total financial exposure.

Used For:
- Category
- Brand
- Department

Output:
- Horizontal bar chart
- Sorted descending
- Value annotations
- Saved under figures/rq1/

RQ1 Role:
Provides descriptive baseline for hypothesis testing.

---

## 2.2 plot_return_rate_vs_mean_erosion()

Purpose:
Distinguish whether high total erosion groups are driven by:
- High return frequency
- High per-return severity
- Or both

Axes:
- X: Return Rate
- Y: Mean Profit Erosion per Return
- Bubble Size: Returned Item Volume

Design:
- Top contributors annotated (default: top 10)
- Clean scatter layout
- Publication-ready typography

RQ1 Role:
Explains mechanism behind total impact.

---

## 2.3 plot_severity_vs_volume_decomposition()

Identity Visualized:

Total Profit Erosion = Returned Items × Avg Profit Erosion

Axes:
- X: Returned Items
- Y: Average Erosion
- Bubble Size: Total Erosion

Design:
- Limited annotation for clarity
- Top contributors labeled
- Clean grid background

RQ1 Role:
Confirms mathematical structure of erosion.

---

## 2.4 plot_profit_erosion_distribution_log()

Purpose:
Visualize skewness of item-level profit erosion.

Design:
- Log-scale histogram
- High bin count (default 60)
- Clean axis labeling

RQ1 Role:
Justifies non-parametric testing (e.g., Kruskal–Wallis).

---

## 2.5 plot_bootstrap_ci_mean_by_group()

Purpose:
Estimate 95% bootstrap confidence intervals for group mean erosion.

Output:
- Horizontal CI chart
- Mean marker
- Error bars representing 95% interval

Additional Output:
Returns CI DataFrame for persistence:

    data/processed/rq1/rq1_bootstrap_ci_*.parquet

RQ1 Role:
Strengthens inference robustness beyond p-values.

---

# 3. Visual Standards (Updated)

## 3.1 Styling Principles

All visuals follow strict consistency standards:

- Clean white grid
- Neutral academic color palette
- No excessive color saturation
- Limited annotation (top contributors only)
- Tight layout to prevent clipping
- High readability font sizes
- No notebook-dependent rendering logic

---

## 3.2 Figure Quality Standards

Format: PNG  
DPI: 150–200  
Layout: tight  
Transparency: Disabled  
Background: White  

All figures are saved deterministically before notebook display.

---

## 3.3 Annotation Rules

To prevent clutter:

- Maximum annotated groups: 8–12
- Annotate only high-impact contributors
- Avoid full-label scatter clutter
- No overlapping label stacks

---

## 3.4 Output Standards

Every figure must:

1. Save to figures/rq1/
2. Be displayed inline via saved file
3. Return output path (CI-safe)
4. Not rely on implicit matplotlib state

Notebook pattern:

rq1_visuals.plot_*(...)
display(Image(filename=str(out_path)))

---

# 4. Separation of Concerns

Notebook Responsibilities:
- Load data
- Call visualization functions
- Display saved PNG
- Save CI tables

Module Responsibilities:
- Validate schema
- Generate plot
- Save figure
- Return path
- Close figure

---

# 5. Differences from Legacy visualization.py

The old module:
- Focused on exploratory EDA
- Mixed multiple business views
- Included margin and customer analysis

The new rq1_visuals.py:
- Is RQ1-specific
- Supports academic hypothesis testing
- Enforces standardized output structure
- Aligns with statistical workflow
- Produces thesis-ready figures

---

# 6. Summary

The RQ1 visualization module provides:

- Structured academic storytelling
- Mechanism-based interpretation
- Distribution justification for statistical tests
- Bootstrap inference validation
- Publication-ready output standards
- Clean module separation
- CI-safe deterministic rendering

This module directly supports RQ1 hypothesis validation and final report production.
