# RQ3 Sensitivity Analysis Plan

## Context

Two modeling choices in the RQ3 predictive pipeline were documented as limitations needing sensitivity analysis:

1. **Processing cost base ($12)** — Selected from $10–$25 literature range. `docs/PROCESSING_COST_METHODOLOGY.md` Section 7 recommends testing base costs $8–$18, alternative tier multipliers, and tier boundary changes.
2. **High-erosion threshold (75th percentile)** — `docs/rq3_technical_documentation.md` Section 10 states: *"The 75th percentile threshold for high-erosion classification is a modeling choice. Sensitivity analysis across alternative thresholds was not conducted."*

**Goal**: Determine whether the RQ3 findings (AUC > 0.70, hypothesis rejection) are robust to alternative parameter values.

---

## Key Insight: Existing Functions Are Already Parameterized

No changes needed to existing `src/` modules. All entry points accept the parameters we need to vary:

| Function | File | Parameter |
|----------|------|-----------|
| `calculate_profit_erosion()` | `src/feature_engineering.py` | `cost_components` dict (sums to base cost) |
| `create_profit_erosion_targets()` | `src/feature_engineering.py` | `high_erosion_percentile` float |
| `prepare_modeling_data()` | `src/rq3_modeling.py` | `target` column name |
| `screen_features()` | `src/rq3_modeling.py` | reusable as-is |
| `train_and_evaluate()` | `src/rq3_modeling.py` | reusable as-is |

**What changes between scenarios**: Only the **target labels** shift. The 12 predictor features are computed from transactions and do not depend on processing cost or threshold.

---

## Sensitivity Parameter Grids

### Analysis A: Processing Cost ($8–$18)

| Scenario | Base Cost | Rationale |
|----------|-----------|-----------|
| Low | $8 | Floor of plausible range |
| Conservative-low | $10 | Lower bound of literature ($10–$25) |
| **Baseline** | **$12** | **Current default** |
| Moderate-high | $14 | Mid-range |
| High | $18 | Upper-mid of literature range |

Category tier multipliers (1.0x / 1.15x / 1.3x) stay constant. Only the base cost varies.

### Analysis B: Threshold Percentile (50th–90th)

| Threshold | Approx. Positive Rate | Rationale |
|-----------|----------------------|-----------|
| 50th | ~50% | Median split |
| 60th | ~40% | |
| 70th | ~30% | |
| **75th** | **~25%** | **Current default** |
| 80th | ~20% | |
| 90th | ~10% | Extreme tail |

Processing cost stays at $12 baseline.

---

## Implementation Steps

### Step 1: Add config constants — `src/config.py`

```python
# --- RQ3: Sensitivity Analysis ---
SENSITIVITY_BASE_COSTS = [8.0, 10.0, 12.0, 14.0, 18.0]
SENSITIVITY_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
```

### Step 2: Create `src/rq3_sensitivity.py` (3 functions)

**`run_cost_sensitivity(item_df, customer_behavioral_df, base_costs, ...)`**

For each base cost:
1. Scale `DEFAULT_COST_COMPONENTS` proportionally to sum to new base
2. `calculate_profit_erosion()` → `aggregate_profit_erosion_by_customer()`
3. Merge with `customer_behavioral_df` (computed once — doesn't change)
4. `create_profit_erosion_targets(percentile=0.75)`
5. `prepare_modeling_data()` → `screen_features()` → `train_and_evaluate()`
6. Collect: best AUC, F1, precision, recall, threshold $, n_positive, labels

Returns: summary DataFrame + labels dict

**`run_threshold_sensitivity(customer_targets_df, thresholds, ...)`**

For each threshold:
1. `create_profit_erosion_targets(percentile=threshold)` on baseline ($12) data
2. `prepare_modeling_data()` → `screen_features()` → `train_and_evaluate()`
3. Collect: best AUC, F1, precision, recall, positive rate, surviving features

Returns: summary DataFrame

**`compute_label_stability(labels_dict, baseline_key)`**

- Jaccard similarity of flagged customer sets vs baseline
- Flip rate (% customers whose label changed)

Returns: stability DataFrame

### Step 3: Create `tests/test_rq3_sensitivity.py`

Synthetic fixtures (CI-safe), same pattern as `test_rq3_modeling.py`:

- `run_cost_sensitivity`: expected columns, correct row count, AUC in [0,1]
- `run_threshold_sensitivity`: expected columns, positive rate decreases with threshold
- `compute_label_stability`: Jaccard=1.0 for self-comparison, correct flip rates

### Step 4: Create `notebooks/rq3_sensitivity_analysis.ipynb`

| Section | Content |
|---------|---------|
| 1. Setup | Imports, reload modules |
| 2. Data Prep | Load `feature_engineered_dataset.parquet`, filter to returned items, compute `engineer_customer_behavioral_features()` once |
| 3. Cost Sensitivity | Run 5 scenarios, display summary table + label stability |
| 4. Cost Visualizations | AUC/F1 vs base cost; Jaccard similarity vs baseline |
| 5. Threshold Sensitivity | Run 6 scenarios, display summary table |
| 6. Threshold Visualizations | AUC vs threshold (with positive rate on secondary axis) |
| 7. Summary | Consolidated findings for tech doc |
| 8. Export | Save CSVs/figures to `reports/rq3/` |

### Step 5: Update `docs/rq3_technical_documentation.md` (after notebook run)

- Add new section between current Section 8 (Interpretation) and Section 9 (External Validation): **"Sensitivity Analysis"** with cost and threshold results tables
- Update Section 10 (Limitations): replace "Sensitivity analysis... was not conducted" with reference to the new section

---

## Files Summary

| File | Action |
|------|--------|
| `src/config.py` | **Modify** — add `SENSITIVITY_BASE_COSTS`, `SENSITIVITY_THRESHOLDS` |
| `src/rq3_sensitivity.py` | **Create** — 3 orchestration functions |
| `tests/test_rq3_sensitivity.py` | **Create** — unit tests |
| `notebooks/rq3_sensitivity_analysis.ipynb` | **Create** — analysis notebook |
| `docs/rq3_technical_documentation.md` | **Modify** — add results section after notebook run |

Existing modules reused **without changes**:

- `src/feature_engineering.py` — `calculate_profit_erosion()`, `aggregate_profit_erosion_by_customer()`, `create_profit_erosion_targets()`, `engineer_customer_behavioral_features()`
- `src/rq3_modeling.py` — `prepare_modeling_data()`, `screen_features()`, `train_and_evaluate()`, `test_hypothesis()`, `build_comparison_table()`

---

## Runtime Consideration

11 scenarios × ~2–3 min each = ~25–35 min total. Mitigation: run only Random Forest (the established best model) for sensitivity sweeps instead of all 3 model families. The full 3-model comparison was already completed in the primary notebook.

---

## Verification

1. `pytest tests/test_rq3_sensitivity.py -v` — all tests pass
2. `pytest tests/ -v` — full suite passes, no regressions
3. Run `notebooks/rq3_sensitivity_analysis.ipynb` — all cells execute, artifacts saved to `reports/rq3/`
4. Key expectation: AUC remains above 0.70 across all scenarios if the predictive signal is genuine
