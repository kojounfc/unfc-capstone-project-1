# Plan: Improve `avg_item_price` SSL Feature Using Reference Sale Amount

## Context

The SSL external validation pipeline (`src/rq3_validation.py`) computes `avg_item_price` from `|CreditReturn Sales / Ordered Qty|` using RETURN lines only. This produces non-null values for only **37.7% of SSL accounts** (5,135 of 13,616) because the remaining 62.3% of accounts have only ORDER (replacement) lines with no RETURN activity.

## Data Exploration Results (Pre-Implementation)

A quick empirical test was run against the actual SSL data (`data/raw/SSL_Returns_df_yoy.csv`) to evaluate six strategies. **The results invalidated the original assumption** that simply switching columns on RETURN lines would improve coverage.

### Account Composition
- Accounts with RETURN lines: **5,135 (37.7%)**
- Accounts with ORDER lines only: **8,481 (62.3%)**
- Accounts with both: 3,469

### Column Availability on RETURN Lines
| Column | Non-null | Positive/Nonzero |
|--------|----------|-------------------|
| `Reference Sale Amount` | 90.0% | 85.6% |
| `CreditReturn Sales` | 100.0% | 79.9% |

### Strategy Comparison (Account-Level Coverage)

| # | Strategy | Scope | Column | Coverage | vs Current |
|---|----------|-------|--------|----------|------------|
| S1 | **Current implementation** | RETURN only | CreditReturn Sales | **37.7%** (5,135) | — |
| S2 | RefSale only | RETURN only | Reference Sale Amount | 33.3% (4,537) | **-4.4pp (worse)** |
| S3 | Hybrid RefSale→Credit | RETURN only | RefSale preferred, CreditReturn fallback | 37.7% (5,135) | +0.0pp (same) |
| **S4** | **RefSale all lines** | **ALL lines** | **Reference Sale Amount** | **89.0%** (12,117) | **+51.3pp** |
| **S5** | **Hybrid all lines** | **ALL lines** | **RefSale→CreditReturn fallback** | **90.0%** (12,254) | **+52.3pp** |
| S6 | Hybrid RETURN only | RETURN only | RefSale→CreditReturn fallback | 33.8% (4,607) | -3.9pp (worse) |

### Key Finding

**The coverage bottleneck is the RETURN-only line restriction, NOT the column choice.**

- Restricting to RETURN lines caps coverage at 37.7% regardless of which column is used, because 62.3% of accounts have zero RETURN lines
- Switching from `CreditReturn Sales` to `Reference Sale Amount` on RETURN lines *reduces* coverage (from 37.7% to 33.3%) because 14.4% of RETURN lines have zero/null Reference Sale Amount
- The dramatic improvement (**37.7% → 89–90%**) comes from expanding scope to ALL lines using `Reference Sale Amount`

### Semantic Justification for Using ALL Lines

| Aspect | TheLook `avg_item_price` | SSL Proposed |
|--------|--------------------------|--------------|
| **Definition** | Mean `sale_price` per item | Mean `Reference Sale Amount / |Ordered Qty|` per line |
| **Source lines** | All purchased items | All return-related lines (RETURN + ORDER) |
| **What it represents** | Original sale price per unit | Original sale price of items in the return ecosystem |

- `Reference Sale Amount` on ORDER lines = the original sale price of the item being replaced (sent for free as a no-charge replacement)
- This IS the unit price of items involved in the return event — the closest semantic analog to TheLook's per-unit sale price
- ORDER lines with `CreditReturn Sales ≈ $0` were correctly excluded before (refund amount = 0 for replacements), but their `Reference Sale Amount` is meaningful

### Descriptive Statistics Comparison

| Stat | S1 (Current, n=5,135) | S4 (RefSale all, n=12,117) | S5 (Hybrid all, n=12,254) |
|------|------------------------|----------------------------|---------------------------|
| mean | $85.14 | $193.91 | $188.19 |
| median | $19.87 | $42.44 | $40.85 |
| std | $240.93 | $723.79 | $712.50 |
| 25th | $4.65 | $16.99 | $16.24 |
| 75th | $63.28 | $121.62 | $117.96 |

The higher values for S4/S5 reflect inclusion of the original sale price (which is the full retail amount) vs. the refund credit (which may be partial). This is expected and desirable — TheLook's `avg_item_price` also uses the full sale price.

### Downstream Impact: Feature Screening, Model Performance, Feature Importance

A full end-to-end test (`avg_item_price_impact_test.py`) was run comparing the current implementation against S5 through the complete SSL pipeline: imputation → train/test split → 3-gate screening → GridSearchCV (RF) → test-set evaluation → feature importance.

**Feature Screening — No Change:**

| Feature | Current | S5 |
|---------|---------|-----|
| `order_frequency` | FAIL | FAIL |
| `return_frequency` | FAIL | FAIL |
| `customer_return_rate` | PASS | PASS |
| `avg_basket_size` | PASS | PASS |
| `avg_order_value` | PASS | PASS |
| `customer_tenure_days` | PASS | PASS |
| `purchase_recency_days` | PASS | PASS |
| `total_items` | FAIL | FAIL |
| `total_sales` | FAIL | FAIL |
| `total_margin` | PASS | PASS |
| **`avg_item_price`** | **PASS** | **PASS** |
| `avg_item_margin` | PASS | PASS |

All 12 features produce identical pass/fail outcomes. The surviving set remains **8 features** in both scenarios. `avg_item_price` passes screening in both cases.

**Model Performance — Marginal Improvement:**

| Metric | Current | S5 | Delta |
|--------|---------|-----|-------|
| avg_item_price coverage | 37.7% | 90.0% | **+52.3pp** |
| Imputation rate | 62.3% | 10.0% | **-52.3pp** |
| Surviving features | 8 | 8 | 0 |
| CV AUC | 0.9986 | 0.9986 | +0.0001 |
| Test AUC | 0.9992 | 0.9992 | +0.0000 |
| F1 | 0.9697 | 0.9761 | **+0.0064** |
| Precision | 0.9545 | 0.9642 | **+0.0097** |
| Recall | 0.9853 | 0.9883 | +0.0029 |

AUC is virtually identical (both 0.9992). F1 and precision show small improvements (+0.6pp and +1.0pp respectively), likely from reduced imputation noise.

**Feature Importance — `avg_item_price` Nearly Doubles:**

| Rank | Current | S5 |
|------|---------|-----|
| 1 | `total_margin` (0.5213) | `total_margin` (0.5202) |
| 2 | `avg_basket_size` (0.1450) | `avg_basket_size` (0.1403) |
| 3 | `customer_tenure_days` (0.1056) | `customer_tenure_days` (0.1048) |
| 4 | `avg_item_margin` (0.1032) | `avg_item_margin` (0.1042) |
| 5 | `customer_return_rate` (0.0417) | `customer_return_rate` (0.0425) |
| 6 | `avg_order_value` (0.0391) | `avg_order_value` (0.0370) |
| 7 | `purchase_recency_days` (0.0346) | `purchase_recency_days` (0.0335) |
| **8** | **`avg_item_price` (0.0095)** | **`avg_item_price` (0.0174)** |

`avg_item_price` remains rank 8/8 in both cases, but its importance nearly doubles (0.0095 → 0.0174). This makes sense: with 62.3% imputation, the current version is mostly median fill — the model can barely extract signal from it. At 10% imputation, the feature carries substantially more real information.

**Impact Assessment:** The change is **safe and beneficial**. No screening outcomes change, no model degradation occurs, and `avg_item_price` becomes a more informative (though still lowest-ranked) predictor. The primary benefit is methodological: reducing a 62.3% imputation rate to 10% strengthens the validity of the feature and the overall analysis.

---

## Recommended Strategy: S5 (Hybrid All Lines)

**Use `Reference Sale Amount / |Ordered Qty|` from ALL lines (RETURN + ORDER), with `|CreditReturn Sales / Ordered Qty|` fallback on RETURN lines where Reference Sale Amount is unavailable.**

- Coverage: **90.0%** (12,254 / 13,616) — up from 37.7%
- Falls back gracefully when Reference Sale Amount is null
- Includes ORDER-only accounts (8,481) that were previously all-NaN
- Best semantic alignment with TheLook's definition

---

## Implementation Steps

### Step 1: Update `engineer_ssl_account_features()` in `src/rq3_validation.py`

**File:** `src/rq3_validation.py`, lines 108–115

**Current code:**
```python
# Compute per-line item price from RETURN lines only
# ORDER lines have CreditReturn Sales ≈ 0, which would distort avg_item_price
is_return = df["Sales_Type"] == "RETURN" if "Sales_Type" in df.columns else pd.Series(True, index=df.index)
df["_item_price"] = np.where(
    is_return & (df["Ordered Qty"].abs() > 0),
    df["CreditReturn Sales"].abs() / df["Ordered Qty"].abs(),
    np.nan,
)
```

**Proposed change:**
```python
# Compute per-line item price using Reference Sale Amount (original sale price)
# from ALL line types — semantically closest to TheLook's avg_item_price.
# ORDER lines carry Reference Sale Amount (price of the item being replaced)
# but have CreditReturn Sales ≈ $0, so CreditReturn Sales is only used as
# fallback on RETURN lines where Reference Sale Amount is unavailable.
# This increases account-level coverage from 37.7% to ~90%.
is_return = (
    df["Sales_Type"] == "RETURN"
    if "Sales_Type" in df.columns
    else pd.Series(True, index=df.index)
)
has_qty = df["Ordered Qty"].abs() > 0

if "Reference Sale Amount" in df.columns:
    has_ref = df["Reference Sale Amount"].notna() & (df["Reference Sale Amount"] > 0)
    df["_item_price"] = np.where(
        has_qty & has_ref,
        df["Reference Sale Amount"] / df["Ordered Qty"].abs(),
        np.where(
            is_return & has_qty,
            df["CreditReturn Sales"].abs() / df["Ordered Qty"].abs(),
            np.nan,
        ),
    )
else:
    # Fallback for DataFrames without Reference Sale Amount
    df["_item_price"] = np.where(
        is_return & has_qty,
        df["CreditReturn Sales"].abs() / df["Ordered Qty"].abs(),
        np.nan,
    )
```

**Key change from original plan:** The first `np.where` condition is `has_qty & has_ref` (ALL lines), NOT `is_return & has_qty & has_ref`. This allows ORDER lines with valid Reference Sale Amount to contribute.

### Step 2: Update synthetic test fixture in `tests/test_rq3_validation.py`

**File:** `tests/test_rq3_validation.py`, lines 22–80 (synthetic fixture)

Add `Reference Sale Amount` column to the synthetic fixture:
- For RETURN lines: generate positive values (e.g., `rng.uniform(10, 500, n_return)`), set ~10% to NaN
- For ORDER lines: generate positive values (e.g., `rng.uniform(10, 300, n_order)`), set ~20% to NaN
- Ensures test coverage of both the primary path (RefSale) and fallback (CreditReturn)

### Step 3: Add unit-level tests for `avg_item_price` computation

**File:** `tests/test_rq3_validation.py`

Add new tests:
1. **`test_avg_item_price_uses_reference_sale_amount`** — verify that when `Reference Sale Amount` is available on ALL line types, it is used
2. **`test_avg_item_price_fallback_to_credit_sales`** — verify fallback to `|CreditReturn Sales / Ordered Qty|` on RETURN lines when RefSale is null
3. **`test_avg_item_price_includes_order_lines`** — verify ORDER lines with valid RefSale contribute to avg_item_price (key behavioral change)
4. **`test_avg_item_price_coverage_improvement`** — verify that account coverage improves when RefSale is available on ORDER lines
5. **`test_avg_item_price_without_reference_column`** — verify graceful fallback when `Reference Sale Amount` column is absent entirely

### Step 4: Update documentation in `notebooks/rq3_ssl_validation.ipynb`

**File:** `notebooks/rq3_ssl_validation.ipynb`, Cell 9 (feature mapping table)

Update the `avg_item_price` row:
```
| avg_item_price | Mean Reference Sale Amount / |Ordered Qty| (all lines; CreditReturn fallback on RETURN lines) | All lines |
```

Add a markdown note explaining:
- Why scope expanded from RETURN-only to ALL lines
- Coverage improvement (37.7% → ~90%)
- Semantic justification

### Step 5: Re-execute `notebooks/rq3_ssl_validation.ipynb`

User must re-run the notebook to produce updated results:
- New account-level `avg_item_price` coverage statistics (~90% vs 37.7%)
- Updated feature screening results (Level 1) — `avg_item_price` screening outcome may change
- Updated directional validation results (Level 2) — note: `avg_item_price` was NOT among the 7 TheLook surviving features used in directional prediction, so Level 2 results should be unchanged
- Updated descriptive statistics

### Step 6: Update `docs/rq3_technical_documentation.md`

Update three locations:

1. **Section 10.3 (Feature Mapping table):** Change the `avg_item_price` row from:
   ```
   | `avg_item_price` | Mean |CreditReturn Sales / Ordered Qty| | RETURN only |
   ```
   to:
   ```
   | `avg_item_price` | Mean Reference Sale Amount / |Ordered Qty| (CreditReturn fallback on RETURN lines) | All lines |
   ```

2. **Section 10.3 (Key mapping decisions):** Update the `avg_item_price` bullet to reflect: (a) expanded scope to all lines, (b) Reference Sale Amount as primary source, (c) CreditReturn Sales fallback, (d) coverage improvement.

3. **Section 11 (Limitations):** Update the `avg_item_price` limitation bullet to reflect the improved coverage (~90% vs 37.7%) and the remaining ~10% still imputed.

### Step 7: Run full test suite

```bash
pytest tests/ -v
```

Verify all existing tests pass plus the new tests from Step 3.

---

## File Summary

| File | Action |
|------|--------|
| `src/rq3_validation.py` | **Modify** — update `_item_price` to use RefSale from ALL lines with CreditReturn fallback |
| `tests/test_rq3_validation.py` | **Modify** — add `Reference Sale Amount` to fixture + 5 new unit tests |
| `notebooks/rq3_ssl_validation.ipynb` | **Modify** — update documentation cell, user re-executes |
| `docs/rq3_technical_documentation.md` | **Modify** — update feature mapping table, mapping decisions, limitations |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| `Reference Sale Amount` not present in all SSL data loads | Fallback to original RETURN-only CreditReturn logic when column is absent |
| ORDER-line prices skew the average | Semantically valid: ORDER lines represent replacement items at original sale price |
| Feature screening results change (pass/fail status) | Expected and acceptable — document any changes |
| Directional validation (Level 2) results change | Unlikely — `avg_item_price` is not among the 7 TheLook surviving features |
| Higher mean/std than current ($188 vs $85) | Expected — reflects original sale price vs. partial refund amount; aligns with TheLook semantics |
| Existing tests break | Fixture update in Step 2 + column-presence guard ensures backward compatibility |

---

## Verification Checklist

1. `src/rq3_validation.py` uses `Reference Sale Amount` from ALL lines as primary source with `CreditReturn Sales` fallback on RETURN lines
2. Account-level `avg_item_price` coverage ≈ 90% (up from 37.7%)
3. All existing tests pass (no regressions)
4. 5 new unit tests pass for `avg_item_price` computation logic
5. Notebook re-executed with updated coverage statistics documented
6. Technical documentation updated in Sections 10.3 and 11
7. Full test suite: `pytest tests/ -v` — all green
