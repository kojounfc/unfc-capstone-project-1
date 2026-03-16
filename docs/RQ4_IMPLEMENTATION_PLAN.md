# RQ4 Econometric Regression Implementation Plan

## Context

RQ4 is no longer maintained as a separate standalone notebook workflow. The final implementation is integrated into the master notebook:

- `notebooks/profit_erosion_analysis.ipynb`
- Main section: `9. RQ4: Econometric Analysis`

This file remains as a project record, but the canonical execution path, outputs, and section references are now tied to the master notebook.

---

## Final Implementation State

RQ4 is implemented through:

- `src/rq4_econometrics.py`
- `src/rq4_validation.py`
- `src/rq4_visuals.py`
- `tests/test_rq4_econometrics.py`
- `docs/rq4_technical_documentation.md`

The notebook workflow writes:

- processed model inputs to `data/processed/rq4`
- CSV report artifacts to `reports/rq4`
- figures to `figures/rq4`

---

## Master Notebook Placement

RQ4 now lives inside the main notebook structure:

1. Setup
2. Data Loading
3. Data Cleaning and Baseline / Descriptive EDA
4. Feature Engineering
5. Descriptive and Group-Level Transformations
6. RQ1: Descriptive Analysis
7. RQ2: Customer Segmentation
8. RQ3: Predictive Modeling
9. RQ4: Econometric Analysis
10. Summary and Conclusions

Any previous references to a standalone RQ4 notebook should be read as historical only.

---

## RQ4 Deliverables

### Source Modules

- `src/rq4_econometrics.py`: feature screening, regression preparation, OLS fitting, diagnostics, and summary extraction
- `src/rq4_validation.py`: SSL external validation workflow
- `src/rq4_visuals.py`: coefficient plots and diagnostics

### Outputs

**Processed data (`data/processed/rq4/`):**

- model-ready parquet artifacts produced during the RQ4 workflow

**Report CSVs (`reports/rq4/`):**

- `rq4_thelook_coefficients.csv`
- `rq4_ssl_coefficients.csv`
- `rq4_ssl_coefficient_alignment.csv`
- `rq4_validation_summary.csv`

**Figures (`figures/rq4/`):**

- `rq4_coefficient_plot.png`
- `rq4_residual_diagnostics.png`
- `rq4_qq_plot_comparison.png`

---

## Notes for Maintenance

When updating RQ4 documentation or code references:

1. Treat `profit_erosion_analysis.ipynb` as the canonical notebook.
2. Keep section numbering aligned to main section `9`.
3. Keep parquet outputs in `data/processed/rq4`.
4. Keep CSV artifacts in `reports/rq4`.
5. Keep figures in `figures/rq4`.
