# Validation Module Reference
**Capstone Project – Master of Data Analytics**
**`src/rq3_validation.py`**

---

## 1. Overview

`src/rq3_validation.py` provides a reusable external validation pipeline. It was built for RQ3 (predictive modeling) but is designed to be callable by any RQ that needs to test whether TheLook findings generalize to an external dataset.

The module implements two validation levels:

| Level | Question | Key Function |
|-------|----------|--------------|
| **Level 1 — Pattern** | Do the same features matter in both datasets? | `validate_feature_patterns()` |
| **Level 2 — Directional** | Does a TheLook-trained model generalize to external data? | `validate_directional_predictions()` |

Only **two** functions are SSL-specific (`load_ssl_data`, `engineer_ssl_account_features`). The remaining four functions are dataset-agnostic and can be called by any RQ with appropriately prepared data.

---

## 2. Function Reference

### 2.1 `load_ssl_data(filepath=None)`

*SSL-specific. Not reusable for other RQs.*

**Prerequisite:** The raw data file must be present at:  
```
data/raw/SSL_Returns_df_yoy.csv
```
This file is not tracked in version control. It is located in the project repository in [onedrive](https://guscanada-my.sharepoint.com/personal/mario_zamudio2499_myunfc_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fmario%5Fzamudio2499%5Fmyunfc%5Fca%2FDocuments%2FDocuments%2F1%2E%20MDA%2F05%2ETERM%2F1%2E%20Capstone%2Fdataset&viewid=f0d3ff3a%2Ddbe3%2D401e%2Db204%2Dd119e89d4152&sharingv2=true&fromShare=true&at=9&CT=1768319801109&OR=OWA%2DNT%2DMail&CID=6505da53%2De49c%2D5481%2Dc88e%2D9f65f485854e&FolderCTID=0x0120000C62785EC1C5A045B9855897FD62CFEF&view=0) Contact the project team to obtain it, then place it at the path above before calling any SSL function.

Loads and cleans `data/raw/SSL_Returns_df_yoy.csv`. Parses date columns, drops rows with missing account IDs. Returns a cleaned DataFrame.

```python
from src.rq3_validation import load_ssl_data
ssl_df = load_ssl_data()                          # uses config.SSL_RETURNS_CSV
ssl_df = load_ssl_data("path/to/custom.csv")      # override path
```

---

### 2.2 `engineer_ssl_account_features(df)`

*SSL-specific. Not reusable for other RQs — but serves as the template for writing your own account-level aggregation.*

Aggregates SSL order lines to account level, producing 12 features analogous to TheLook's candidate predictors. See `docs/rq3_technical_documentation.md` Section 10.3 for the full feature mapping.

```python
from src.rq3_validation import engineer_ssl_account_features
account_df = engineer_ssl_account_features(ssl_df)
# Returns one row per account with columns:
# account_id, order_frequency, return_frequency, customer_return_rate,
# avg_basket_size, avg_order_value, total_items, total_sales, total_margin,
# avg_item_price, avg_item_margin, customer_tenure_days, purchase_recency_days,
# total_loss
```

---

### 2.3 `create_ssl_targets(account_df, loss_column="total_loss", percentile=75.0)`

Creates a binary `is_high_loss_account` column using the nth percentile of a loss column, mirroring the TheLook 75th-percentile methodology.

```python
from src.rq3_validation import create_ssl_targets
account_df = create_ssl_targets(account_df)
# Adds: is_high_loss_account (0 or 1)

# With custom column and threshold:
account_df = create_ssl_targets(
    account_df,
    loss_column="total_profit_erosion",
    percentile=75.0
)
```

**For other RQs:** Call this on your own account-level DataFrame with whatever column represents "high impact." The `loss_column` and `percentile` parameters are fully configurable.

---

### 2.4 `validate_feature_patterns(ssl_account_df, thelook_screening, feature_columns=None, target_column="is_high_loss_account")`

**Reusable by RQ1, RQ2, RQ4.**

Runs the same 3-gate feature screening (variance → correlation → univariate significance) independently on external data and compares results against TheLook screening output. Returns a comparison DataFrame showing which features passed or failed in each dataset.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ssl_account_df` | `pd.DataFrame` | External data at the unit-of-analysis level (one row per entity) |
| `thelook_screening` | `pd.DataFrame` | Screening report from `screen_features()` — must have `feature` and `final_status` columns |
| `feature_columns` | `list[str]` or `None` | Features to compare. Defaults to `RQ3_CANDIDATE_FEATURES` present in external data |
| `target_column` | `str` | Binary target column in the external DataFrame |

**Returns:** DataFrame with columns `feature`, `thelook_status`, `ssl_status`, `both_pass`, `both_fail`, `agreement`.

```python
from src.rq3_validation import validate_feature_patterns

pattern_comparison = validate_feature_patterns(
    ssl_account_df=external_df,
    thelook_screening=screening_report,       # from screen_features()
    feature_columns=["return_frequency", "avg_order_value", "total_margin"],
    target_column="is_high_impact"
)
print(pattern_comparison)
# feature              thelook_status  ssl_status  agreement
# return_frequency     pass            pass        True
# avg_order_value      pass            fail        False
# ...
```

---

### 2.5 `validate_directional_predictions(ssl_account_df, thelook_model, thelook_features, scaler=None, target_column="is_high_loss_account", loss_column="total_loss")`

**Reusable by RQ3 and RQ4.**

Applies a TheLook-trained model to external data and measures directional alignment between predictions and actual values. Works with any sklearn-compatible classifier (`predict` + `predict_proba`).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ssl_account_df` | `pd.DataFrame` | External data at the unit-of-analysis level |
| `thelook_model` | sklearn estimator | Trained model — must implement `predict()` and `predict_proba()` |
| `thelook_features` | `list[str]` | Feature names the model was trained on (in training order) |
| `scaler` | sklearn scaler or `None` | Optional `StandardScaler` if model requires scaled input |
| `target_column` | `str` | Binary target column in the external DataFrame |
| `loss_column` | `str` | Continuous loss column for Spearman rank correlation |

**Returns:** Dict with keys:

| Key | Type | Description |
|-----|------|-------------|
| `directional_accuracy` | float | % of accounts where predicted label matches actual label |
| `rank_correlation` | float | Spearman ρ between predicted probability and actual loss |
| `rank_pvalue` | float | p-value for rank correlation |
| `predicted_high_pct` | float | % of accounts predicted as high-risk |
| `actual_high_pct` | float | % of accounts actually high-loss |
| `confusion_at_directional` | ndarray | 2×2 confusion matrix |
| `n_accounts` | int | Total external accounts evaluated |
| `n_features_available` | int | Features successfully matched |
| `n_features_missing` | int | Features not found (imputed with 0) |
| `missing_features` | list | Names of missing features |
| `predictions_df` | `pd.DataFrame` | Per-account predictions and actuals |

```python
from src.rq3_validation import validate_directional_predictions

result = validate_directional_predictions(
    ssl_account_df=external_df,
    thelook_model=best_model,                  # trained RandomForest / GradientBoosting etc.
    thelook_features=surviving_features,        # list from screen_features()
    scaler=None,                                # or your StandardScaler
    target_column="is_high_impact",
    loss_column="total_profit_erosion"
)
print(f"Directional accuracy: {result['directional_accuracy']:.1%}")
print(f"Rank correlation: {result['rank_correlation']:.3f} (p={result['rank_pvalue']:.2e})")
```

---

### 2.6 `build_validation_summary(pattern_comparison, directional_result)`

**Reusable by all RQs.**

Combines Level 1 and Level 2 results into a single summary DataFrame with 13 standard metrics. Suitable for inclusion in notebook output or reports.

```python
from src.rq3_validation import build_validation_summary

summary = build_validation_summary(pattern_comparison, result)
print(summary.to_string(index=False))
# metric                         value
# pattern_features_compared     12
# pattern_agreement_count        7
# pattern_agreement_pct         58.3
# directional_accuracy          0.7640
# rank_correlation              0.7526
# ...
```

---

## 3. Adapting for RQ1 — Profit Erosion by Category/Brand

**Goal:** Confirm that the same product categories/brands driving profit erosion in TheLook also show elevated erosion in an external dataset.

RQ1 is a **descriptive/statistical RQ**, so only Level 1 (pattern) validation applies. There is no trained ML model to apply directionally.

### Step-by-step

**Step 1 — Prepare external data at the group level.**

Build a DataFrame where each row is a category (or brand), with aggregate features analogous to the TheLook group-level metrics. At minimum you need:
- A numeric target column indicating "high erosion" (e.g., above-median erosion rate)
- The same feature columns present in your TheLook screening report

```python
import pandas as pd

# Example: external data already aggregated to category level
external_category_df = pd.DataFrame({
    "category":          ["Electronics", "Apparel", "Books"],
    "avg_erosion_rate":  [0.32, 0.21, 0.08],
    "return_rate":       [0.18, 0.25, 0.05],
    "avg_margin_pct":    [0.40, 0.35, 0.55],
    "is_high_erosion":   [1, 1, 0],          # binary target (e.g., top 50%)
})
```

**Step 2 — Get the TheLook screening report.**

This is the `screening_report` DataFrame produced by `screen_features()` in your RQ1 analysis:

```python
from src.rq3_modeling import screen_features

surviving_features, screening_report = screen_features(X_train, y_train)
# screening_report has columns: feature, final_status, ...
```

If you ran RQ1 in `profit_erosion_analysis.ipynb`, the screening report is available as `screening_report` in scope after Section 7.4.

**Step 3 — Run pattern validation.**

```python
from src.rq3_validation import validate_feature_patterns

pattern_comparison = validate_feature_patterns(
    ssl_account_df=external_category_df,
    thelook_screening=screening_report,
    feature_columns=["return_rate", "avg_margin_pct"],   # features available in both
    target_column="is_high_erosion"
)
print(pattern_comparison)
```

**Step 4 — Interpret.**

- `agreement` column: features that behave the same way (pass or fail screening) in both datasets
- Focus on `both_pass` rows: these are features that carry predictive signal in both domains
- Agreement rate ≥ 50% supports generalizability of the RQ1 finding

---

## 4. Adapting for RQ2 — Customer Behavioral Segments

**Goal:** Confirm that the same behavioral clusters found in TheLook also appear in an external dataset (e.g., SSL accounts cluster similarly).

RQ2 uses unsupervised clustering, so **there is no single trained model to apply directionally**. Only Level 1 (pattern) validation is applicable. The goal is to check whether the same features that drove cluster separation in TheLook also show discriminative power in external data.

### Step-by-step

**Step 1 — Prepare external data at the customer/account level.**

Your external DataFrame needs one row per customer/account, with the same features used in RQ2 clustering:

```python
# External data: one row per account, same behavioral features as RQ2
external_account_df = pd.DataFrame({
    "account_id":           [...],
    "return_frequency":     [...],
    "avg_order_value":      [...],
    "customer_return_rate": [...],
    # ... other RQ2 features
    "is_high_erosion":      [...],   # binary target: e.g. top-quartile total loss
})
```

**Step 2 — Define a binary target.**

Cluster-based validation needs a binary anchor. Use the loss column from your external dataset with `create_ssl_targets()`:

```python
from src.rq3_validation import create_ssl_targets

external_account_df = create_ssl_targets(
    external_account_df,
    loss_column="total_financial_loss",
    percentile=75.0
)
# Adds: is_high_loss_account
```

**Step 3 — Get the TheLook screening report from your RQ2 feature prep.**

If your RQ2 notebook ran `screen_features()` during feature selection, reuse that report. Alternatively, run it on the RQ2 customer-level features:

```python
from src.rq3_modeling import screen_features

surviving_features, screening_report = screen_features(
    X_rq2_train,
    y_rq2_train   # binary target from create_ssl_targets() on TheLook data
)
```

**Step 4 — Run pattern validation.**

```python
from src.rq3_validation import validate_feature_patterns

pattern_comparison = validate_feature_patterns(
    ssl_account_df=external_account_df,
    thelook_screening=screening_report,
    feature_columns=surviving_features,      # RQ2 cluster features
    target_column="is_high_loss_account"
)
```

**Step 5 — Interpret.**

- Agreement on "pass" features confirms that the behavioral dimensions driving TheLook clusters also discriminate high-loss accounts in external data
- Agreement on "fail" features confirms that non-discriminative features generalize as non-discriminative

---

## 5. Adapting for RQ4 — Econometric Regression

**Goal:** Confirm that the marginal behavioral associations identified in RQ4 (e.g., "a one-unit increase in return frequency predicts +X% profit erosion") also hold directionally in external data.

RQ4 uses regression, but a trained classifier (or a calibrated regression-based classifier) can be applied using `validate_directional_predictions()`. The closest approach is to train a binary classifier on the RQ4 features and apply it to external data.

### Step-by-step

**Step 1 — Prepare external data at the customer/account level.**

Same as RQ2 Step 1 — one row per account, features matching RQ4 predictors, a binary target, and a continuous loss column.

```python
external_account_df["is_high_erosion"] = (
    external_account_df["total_profit_erosion"]
    >= external_account_df["total_profit_erosion"].quantile(0.75)
).astype(int)
```

**Step 2 — Train a classifier on RQ4 features (TheLook training set).**

RQ4's econometric model is a regression model, not a classifier. To use `validate_directional_predictions()`, train a lightweight Random Forest on the same features using the TheLook training set:

```python
from sklearn.ensemble import RandomForestClassifier
from src.rq3_modeling import screen_features

# Feature screening on TheLook training data
surviving_features, screening_report = screen_features(X_rq4_train, y_rq4_train)
X_train_screened = X_rq4_train[surviving_features]

clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train_screened, y_rq4_train)
```

**Step 3 — Run Level 1 pattern validation.**

```python
from src.rq3_validation import validate_feature_patterns

pattern_comparison = validate_feature_patterns(
    ssl_account_df=external_account_df,
    thelook_screening=screening_report,
    feature_columns=surviving_features,
    target_column="is_high_erosion"
)
```

**Step 4 — Run Level 2 directional validation.**

```python
from src.rq3_validation import validate_directional_predictions

result = validate_directional_predictions(
    ssl_account_df=external_account_df,
    thelook_model=clf,
    thelook_features=surviving_features,
    scaler=None,
    target_column="is_high_erosion",
    loss_column="total_profit_erosion"
)

print(f"Directional accuracy: {result['directional_accuracy']:.1%}")
print(f"Rank correlation (Spearman ρ): {result['rank_correlation']:.3f}")
```

**Step 5 — Build summary.**

```python
from src.rq3_validation import build_validation_summary

summary = build_validation_summary(pattern_comparison, result)
print(summary.to_string(index=False))
```

**Step 6 — Interpret.**

- **Rank correlation > 0.50** (Cohen's "large" threshold): The model's predicted risk probabilities are strongly ordered relative to actual loss, supporting the directional validity of RQ4 behavioral associations across domains
- **Directional accuracy > 0.70**: Classification generalizes beyond chance
- Feature agreement in Level 1 confirms which RQ4 predictors are domain-transferable

---

## 6. Required External Dataset Schema

Any external DataFrame passed to `validate_feature_patterns()` or `validate_directional_predictions()` must meet these requirements:

| Requirement | Details |
|-------------|---------|
| **Unit of analysis** | One row per entity (customer, account, category, etc.) |
| **Feature columns** | Numeric. Must include at least the features you pass in `feature_columns`. Missing features are imputed with 0 by `validate_directional_predictions()`. |
| **Binary target column** | Integer 0/1. Required by `validate_feature_patterns()` (for univariate significance gate) and `validate_directional_predictions()` (for directional accuracy). |
| **Continuous loss column** | Numeric. Required by `validate_directional_predictions()` for Spearman rank correlation. Pass the same column name in `loss_column`. |
| **No leakage columns** | Exclude any column derived from the target (e.g., percentile rank, quartile assignment, total erosion if it directly defines the target). |

### Minimum viable DataFrame

```python
external_df = pd.DataFrame({
    "entity_id":              [...],          # identifier — not used in validation
    "return_frequency":       [...],          # numeric feature
    "avg_order_value":        [...],          # numeric feature
    # ... additional features
    "is_high_impact":         [...],          # binary 0/1 target
    "total_financial_loss":   [...],          # continuous loss for rank correlation
})
```

---

## 7. Complete Example — RQ4 End-to-End Walkthrough

This example shows all steps for RQ4 external validation in a single code block, suitable for a notebook cell or standalone script.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.rq3_modeling import screen_features
from src.rq3_validation import (
    validate_feature_patterns,
    validate_directional_predictions,
    build_validation_summary,
)

# ── 1. Prepare external data ────────────────────────────────────────────────
# Load and aggregate your external dataset to entity level
# (replace this block with your own data preparation logic)
external_df = pd.read_csv("data/raw/your_external_dataset.csv")

# Aggregate to entity level and engineer analogous features
# (write your own aggregation — see engineer_ssl_account_features for the SSL pattern)
external_account_df = your_aggregation_function(external_df)

# Create binary target: top quartile of financial loss
external_account_df["is_high_impact"] = (
    external_account_df["total_financial_loss"]
    >= external_account_df["total_financial_loss"].quantile(0.75)
).astype(int)

# ── 2. Load TheLook training data ────────────────────────────────────────────
# customer_targets is already in scope if running from profit_erosion_analysis.ipynb
# Otherwise load from the processed CSV:
from src.config import CUSTOMER_TARGETS_CSV
from src.rq3_modeling import prepare_modeling_data

customer_targets = pd.read_csv(CUSTOMER_TARGETS_CSV)
X_train, X_test, y_train, y_test = prepare_modeling_data(customer_targets)

# ── 3. Feature screening on TheLook training set ─────────────────────────────
surviving_features, screening_report = screen_features(X_train, y_train)
X_train_screened = X_train[surviving_features]
X_test_screened  = X_test[surviving_features]

# ── 4. Train classifier on TheLook data ──────────────────────────────────────
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train_screened, y_train)

# ── 5. Level 1: Pattern validation ────────────────────────────────────────────
# Check which features screen similarly in external data
pattern_comparison = validate_feature_patterns(
    ssl_account_df=external_account_df,
    thelook_screening=screening_report,
    feature_columns=surviving_features,
    target_column="is_high_impact",
)
print("=== Level 1: Feature Pattern Comparison ===")
print(pattern_comparison.to_string(index=False))
n_agree = pattern_comparison["agreement"].sum()
print(f"\nAgreement: {n_agree}/{len(pattern_comparison)} features ({n_agree/len(pattern_comparison):.0%})")

# ── 6. Level 2: Directional prediction ────────────────────────────────────────
result = validate_directional_predictions(
    ssl_account_df=external_account_df,
    thelook_model=clf,
    thelook_features=surviving_features,
    scaler=None,
    target_column="is_high_impact",
    loss_column="total_financial_loss",
)
print("\n=== Level 2: Directional Prediction ===")
print(f"Directional accuracy : {result['directional_accuracy']:.1%}")
print(f"Rank correlation (ρ) : {result['rank_correlation']:.3f}  (p={result['rank_pvalue']:.2e})")
print(f"Predicted high-risk  : {result['predicted_high_pct']:.1f}%")
print(f"Actual high-loss     : {result['actual_high_pct']:.1f}%")

# ── 7. Summary table ──────────────────────────────────────────────────────────
summary = build_validation_summary(pattern_comparison, result)
print("\n=== Validation Summary ===")
print(summary.to_string(index=False))
```

---

## 8. Interpreting Results

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Pattern agreement | ≥ 50% | The same behavioral features discriminate high-impact entities in both domains |
| `both_pass` count | ≥ 3 features | Core feature set generalizes across domains |
| Directional accuracy | ≥ 0.70 | Model correctly classifies > 70% of external entities |
| Rank correlation (ρ) | ≥ 0.50 (Cohen's "large") | Strong monotonic alignment between predicted risk and actual loss |
| Rank p-value | < 0.05 | Rank correlation is statistically significant |

A validation is considered **successful** when:
- Pattern agreement ≥ 50%, AND
- Rank correlation ≥ 0.50 with p < 0.05

These are the thresholds used in RQ3's SSL validation (7/12 agreement = 58.3%, ρ = 0.75, p ≈ 0.00).

---

## 9. Notes and Caveats

- **Do not modify `src/rq3_validation.py`** for your RQ. Write your own data preparation function that produces the required schema (Section 6), then call the shared validation functions.
- **Feature name matching is exact.** If your external features have different column names than the TheLook features, rename them before calling `validate_directional_predictions()`. The function logs a warning for missing features and imputes them with 0.
- **Level 2 requires a trained classifier**, not a regression model. If your RQ uses only regression (e.g., OLS), train a supplementary Random Forest on the same features using TheLook data to enable directional validation.
- **`create_ssl_targets()` is configurable** — pass any `loss_column` and `percentile` to match your RQ's target definition.
- All functions use `logging` at the `INFO` level. To see log output in a notebook: `import logging; logging.basicConfig(level=logging.INFO)`.
