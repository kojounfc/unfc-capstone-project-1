# RQ4 Econometric Regression — Implementation Plan

## Context

A teammate submitted PR #14 (`feat/us11-behavioral-profit-associations`) implementing RQ4 with 1,472 lines across 2 modules. The implementation uses all 82K customers (85% have zero returns), necessitating a two-stage model (logistic + conditional OLS), a 647-line preprocessing module, and a 4-level logistic fallback chain.

This plan provides a **focused alternative** scoped to the returns population (`customer_profit_erosion_targets.csv`, 11,988 returners). Since every customer in this dataset has `total_profit_erosion > 0`, a single OLS regression suffices — no two-stage model, no separate preprocessing module.

**This is a new, parallel implementation on branch `feat/rq4-econometric-regression` from `dev`.** The teammate's code on `feat/us11-behavioral-profit-associations` is NOT modified.

**RQ4 (from proposal, p.6):** "What are the marginal associations between key behavioral variables (return frequency, basket size, purchase recency) and profit erosion magnitude, controlling for product attributes and customer demographics?"

---

## Comparison: Teammate vs. Clean Approach

| Aspect | Teammate (PR #14) | Clean Approach |
|--------|-------------------|----------------|
| Population | 82K customers (85% zeros) | 11,988 returners only |
| Model | Two-stage (logistic + OLS) | Single OLS with HC3 |
| Source modules | 2 files, 1,472 lines | 1 file, ~310 lines |
| Test files | 2 files, 1,071 lines | 1 file, 37 tests |
| Preprocessing | 647-line audit pipeline | ~10 lines inside `load_rq4_data()` |
| Logistic fallback | 4-level convergence chain | Not needed |
| Category controls | Not included | `dominant_return_category` (26 levels) |
| Ramsey RESET test | Not included | Included |
| Feature screening | None — hardcoded features | 3-gate data-driven screening |
| Explicit H0/H1 test | Not explicit | Explicit evaluation |
| Robustness check | None | Log-transformed model comparison |
| Technical documentation | None | Full `docs/rq4_technical_documentation.md` |

---

## Implementation Status: COMPLETE

All steps below have been implemented, tested, and verified.

---

## Step 0: Create new branch from `dev` — DONE

```bash
git checkout dev
git checkout -b feat/rq4-econometric-regression
```

Branch: `feat/rq4-econometric-regression`

---

## Step 1: Update `src/config.py` — DONE

Added after `MIN_ROWS_THRESHOLD` (line 48):

```python
# RQ4 Econometric Regression constants
RQ4_TARGET_COL = "total_profit_erosion"
RQ4_HYPOTHESIS_PREDICTORS = ["return_frequency", "avg_basket_size", "purchase_recency_days"]
RQ4_BEHAVIORAL_CONTROLS = [
    "order_frequency", "avg_order_value", "customer_tenure_days", "customer_return_rate",
]
RQ4_ALPHA = 0.05
RQ4_VIF_THRESHOLD = 10.0
```

---

## Step 2: Create `src/rq4_econometrics.py` (~310 lines) — DONE

Single module with 8 functions implementing the full RQ4 pipeline:

### Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `LEAKAGE_COLUMNS` | 5 columns | Features that leak target info |
| `NUMERIC_CANDIDATES` | 8 features | Behavioral + age |
| `CATEGORICAL_CANDIDATES` | 3 features | Gender, traffic source, category |
| `COLLINEARITY_THRESHOLD` | 0.85 | Pairwise collinearity cutoff |
| `LOG_TARGET_COL` | `"log_total_profit_erosion"` | Log-transformed DV name |

### Functions

| Function | Purpose |
|----------|---------|
| `load_rq4_data()` | Load & merge customer targets with demographics and dominant return category |
| `screen_features()` | 3-gate data-driven feature screening (correlation, collinearity, ANOVA) |
| `prepare_regression_data()` | Z-score standardize, one-hot encode, add constant; supports `log_transform` parameter |
| `fit_ols_robust()` | Fit OLS with HC3 robust standard errors |
| `calculate_vif()` | Variance Inflation Factor assessment |
| `run_diagnostics()` | Jarque-Bera, Breusch-Pagan, Ramsey RESET, Durbin-Watson |
| `extract_coefficient_table()` | Full coefficient table sorted by |coef| |
| `generate_summary()` | Comprehensive summary dict with hypothesis test evaluation |

### Key Design Decisions

1. **`screen_features()` — 3-gate data-driven selection:**
   - Gate 1: Pearson correlation with target (informational, all features retained)
   - Gate 2: Pairwise multicollinearity |r| > 0.85 (drop weaker of pair)
   - Gate 3: One-way ANOVA for categoricals (drop if p > alpha)

2. **`prepare_regression_data()` — `log_transform` parameter:**
   - When `True`, applies `np.log()` to the target variable
   - Replaces `total_profit_erosion` with `log_total_profit_erosion`
   - Safe because all returners have erosion > 0 (min $13.18)

3. **`load_rq4_data()` — data merge strategy:**
   - `user_id` cast from string (parquet) to int (CSV) for merge
   - `item_status` compared case-insensitively (`str.lower() == "returned"`)
   - Demographics extracted via `groupby('user_id').first()`
   - Dominant return category via `groupby('user_id')['category'].agg(mode)`

---

## Step 3: Create `tests/test_rq4_econometrics.py` (37 tests) — DONE

All tests use synthetic data fixtures (no data files needed for CI):

| Test Class | Tests | What's Tested |
|------------|-------|---------------|
| `TestLoadRQ4Data` | 4 | Shape, no leakage cols, required cols, all erosion > 0 (integration, skip w/o data) |
| `TestScreenFeatures` | 7 | Returns dict, correlation table structure, values in range, collinearity detection, ANOVA table structure, surviving features as lists, respects alpha threshold |
| `TestPrepareRegressionData` | 8 | Returns DF, has constant, numerics standardized, target preserved, errors on missing columns, errors on empty data, log transform creates column, log transform values correct |
| `TestFitOLSRobust` | 4 | Returns results, HC3 covariance, has coefficients, errors on missing target |
| `TestCalculateVIF` | 4 | Returns DF, correct columns, excludes constant, VIF >= 1 |
| `TestRunDiagnostics` | 4 | Returns dict, all 4 tests present, p-values in [0,1], DW in [0,4] |
| `TestExtractCoefficientTable` | 3 | Returns DF, required columns, CIs ordered correctly |
| `TestGenerateSummary` | 3 | Returns dict, all sections present, R-squared in [0,1] |

Integration tests (`TestLoadRQ4Data`) use `_integration_data_available()` helper that validates both file existence AND schema (parquet must have `user_id` column with > 100 rows).

---

## Step 4: Create `notebooks/rq4_behavioral_associations.ipynb` — DONE

### Linear Model (Sections 1-12)

| Section | Content |
|---------|---------|
| 1 | Load data — 11,988 returners, 12 features |
| 2 | EDA — target distribution, correlation heatmap, category bar chart |
| 3 | Feature screening — 3-gate results, surviving features |
| 4 | Prepare regression data — 11,694 obs x 35 columns |
| 5 | VIF check — all < 3.0 |
| 6 | Fit OLS model — R² = 0.808, F = 908.18 |
| 7 | Coefficient table — sorted by |coef| |
| 8 | Diagnostics — JB, BP, RESET, DW |
| 9 | Hypothesis test — H0 rejected (2/3 predictors significant) |
| 10 | Coefficient plot — saved to `figures/rq4_coefficient_plot.png` |
| 11 | Residual diagnostics plots — saved to `figures/rq4_residual_diagnostics.png` |
| 12 | Summary table for report |

### Log-Transformed Model (Sections 13-14)

| Section | Content |
|---------|---------|
| 13 | Log model — fit, diagnostics, coefficient table with % change interpretation |
| 14 | Model comparison — fit metrics, diagnostics, hypothesis test, key coefficients side-by-side |

---

## Step 5: Create `docs/rq4_technical_documentation.md` — DONE

Full technical documentation following the style of `docs/rq1_technical_documentation.md`. Contains:

1. Research Question & Hypotheses
2. Data Scope (11,988 returners, population rationale)
3. Dependent Variable descriptive statistics
4. Feature Screening (3-gate process with full results)
5. Regression Methodology (HC3, z-score, one-hot encoding)
6. Diagnostic Tests (VIF, residual diagnostics)
7. Results (model fit, behavioral coefficients, demographic controls, category effects, log model)
8. Hypothesis Test (H0 rejected, 2/3 predictors significant)
9. Interpretation (per-variable analysis)
10. Limitations (5 items; cross-sectional design removed per research scope)
11. Conclusion
12. Traceability (US06, US07, US11)
13. Artifacts

---

## Key Results

### Feature Screening Outcomes

- **Dropped (Gate 2):** `order_frequency` — collinear with `customer_return_rate` (r = 0.8509)
- **Dropped (Gate 3):** `traffic_source` — ANOVA not significant (p = 0.193)
- **Surviving numeric (7):** `return_frequency`, `avg_basket_size`, `purchase_recency_days`, `avg_order_value`, `customer_tenure_days`, `customer_return_rate`, `age`
- **Surviving categorical (2):** `user_gender`, `dominant_return_category`

### Linear Model

| Metric | Value |
|--------|-------|
| R-squared | 0.8082 |
| Adj. R-squared | 0.8076 |
| F-statistic | 908.18 (p < 0.0001) |
| AIC | 109,730.8 |
| Observations | 11,694 |

### Hypothesis Test

| Predictor | Coefficient | p-value | Significant |
|-----------|-------------|---------|-------------|
| `return_frequency` | +39.16 | < 0.0001 | **Yes** |
| `avg_basket_size` | -20.52 | < 0.0001 | **Yes** |
| `purchase_recency_days` | +0.11 | 0.6770 | No |

**Decision:** H0 rejected — 2 of 3 hypothesis predictors show significant marginal associations.

Additional finding: `avg_order_value` (beta = +39.28, p < 0.0001) is equally strong as `return_frequency`.

### Log Model (Robustness Check)

| Metric | Linear | Log |
|--------|--------|-----|
| R-squared | 0.8082 | 0.7765 |
| Jarque-Bera | 619,317 | 2,198 (282x improvement) |
| Breusch-Pagan | 1,556 | 2,756 |
| Ramsey RESET | 262 | 1,525 |
| H0 Rejected | Yes | Yes |
| Significant H-predictors | return_frequency, avg_basket_size | return_frequency, avg_basket_size |

The log model dramatically improves normality but does not uniformly improve all diagnostics. Both models agree on which predictors are significant, confirming robustness.

---

## Files Summary

| File | Action | Status |
|------|--------|--------|
| `src/config.py` | **Modified** — added RQ4 constants (8 lines) | DONE |
| `src/rq4_econometrics.py` | **Created** — ~310 lines, single OLS module with feature screening and log transform | DONE |
| `tests/test_rq4_econometrics.py` | **Created** — 37 tests with synthetic fixtures | DONE |
| `notebooks/rq4_behavioral_associations.ipynb` | **Created** — 33 cells (linear + log model comparison) | DONE |
| `docs/rq4_technical_documentation.md` | **Created** — full technical documentation | DONE |
| `figures/rq4_coefficient_plot.png` | **Generated** — coefficient plot with CIs | DONE |
| `figures/rq4_residual_diagnostics.png` | **Generated** — linear model residual plots | DONE |
| `figures/rq4_log_residual_diagnostics.png` | **Generated** — log model residual plots | DONE |

**No files from the teammate's branch are modified or deleted.**

---

## Verification

| Check | Status |
|-------|--------|
| `pytest tests/test_rq4_econometrics.py -v` — 37 tests pass | DONE |
| Notebook runs end-to-end with results saved | DONE |
| Regression includes explicit H0/H1 evaluation | DONE |
| No leakage columns used as predictors | DONE |
| Log-transformed robustness check confirms findings | DONE |
| Technical documentation complete | DONE |
