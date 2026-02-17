# RQ3 Predictive Modeling - Implementation Plan

## Context

RQ3 asks: *"Can machine learning models accurately predict high profit erosion customers using transaction-level and behavioral features, and which features contribute most significantly to prediction accuracy?"*

- **Success criterion**: AUC > 0.70
- **H0**: AUC <= 0.70 (cannot predict) | **H1**: AUC > 0.70 (can predict)
- **Upstream work complete**: Target variable (`is_high_erosion_customer`) and 12 customer behavioral features are available in `data/processed/customer_profit_erosion_targets.csv` (~11,988 customers with returns)
- **External validation data**: School Specialty LLC (SSL) return transactions in `data/raw/SSL_Returns_df_yoy.csv` (~234K order lines, ~16.7K accounts, 2024-2025)

This plan follows the established flat-module structure under `src/` (e.g., `config.py`, `feature_engineering.py`, `analytics.py`, `visualization.py`, `descriptive_transformations.py`).

---

## File Structure

```
src/
├── rq3_modeling.py          # Core ML pipeline: data prep, feature screening, training, evaluation
├── rq3_validation.py        # SSL external validation: feature mapping, pattern & directional validation
├── rq3_visuals.py           # ROC curves, feature importance plots, confusion matrices
├── config.py                # Add RQ3 constants (modify existing)
tests/
├── test_rq3_modeling.py     # Unit tests for modeling pipeline
└── test_rq3_validation.py   # Unit tests for SSL validation
notebooks/
├── rq3_predictive_modeling.ipynb  # Primary modeling notebook (TheLook)
└── rq3_ssl_validation.ipynb       # External validation notebook (SSL)
```

No new sub-packages or `__init__.py` files. New modules sit alongside existing `src/` files.

---

## Data Leakage Prevention

Features derived from the target MUST be excluded as predictors:

| Exclude | Reason |
|---------|--------|
| `total_profit_erosion` | Target derived from this |
| `total_margin_reversal` | Component of target |
| `total_process_cost` | Component of target |
| `profit_erosion_quartile` | Derived from target |
| `erosion_percentile_rank` | Derived from target |
| `user_id` | Identifier |

**Candidate Predictor Features (12)**:
`order_frequency`, `return_frequency`, `customer_return_rate`, `avg_basket_size`, `avg_order_value`, `customer_tenure_days`, `purchase_recency_days`, `total_items`, `total_sales`, `total_margin`, `avg_item_price`, `avg_item_margin`

> These are **candidates**, not automatic predictors. The feature screening step determines which features survive into the final model.

---

## Pipeline Order of Operations

The sequence below is critical for consistency and leakage prevention:

```
1. Load data (all 12 candidates + target + leakage cols)
       |
2. Drop leakage columns (remove 6 cols; keep 12 candidates + target)
       |
3. Impute missing values (median strategy on 12 candidates)
       |
4. Stratified train/test split 80/20 (all 12 candidates present in both sets)
       |
5. Feature screening on TRAINING SET ONLY
   +-- Gate 1: Variance check         -> drop near-zero variance features
   +-- Gate 2: Correlation analysis    -> drop redundant features
   +-- Gate 3: Univariate stat test    -> drop statistically irrelevant features
   Result: surviving feature list (<=12 features)
       |
6. Apply surviving feature list to BOTH train and test sets
       |
7. Train models (GridSearchCV on training set, using surviving features only)
       |
8. Evaluate on test set (using same surviving features)
       |
9. Extract feature importance (post-hoc, from trained models)
```

**Key point**: All 12 candidate features enter the 80/20 split. Feature screening (step 5) runs on the training set only, producing a reduced feature list. That same reduced list is then applied to the test set before evaluation. This ensures no test-set information leaks into feature selection.

---

## External Validation Strategy (School Specialty LLC)

### Rationale

A second holdout from TheLook provides limited additional evidence -- it's the same distribution. External validation against a different domain (educational supplies vs. general e-commerce) provides a far stronger generalizability claim for the capstone paper.

### Data Source

`data/raw/SSL_Returns_df_yoy.csv` -- 234K order lines, 16.7K accounts, 2024-2025

| Aspect | TheLook | SSL (School Specialty) |
|--------|---------|------------------------|
| Grain | Order line | Order line |
| Customers | ~12K with returns | ~16.7K accounts |
| Financial | `sale_price`, `cost`, `item_margin` | `CreditReturn Sales`, `Product Cost`, `Gross Profit` |
| Return cost | Estimated ($12 base x tier) | Actual (`estimated_labor_cost`, `total_return_cogs`) |
| Target | `is_high_erosion_customer` (75th pct) | Constructed from `total_loss` (75th pct) |
| Categories | Product categories | `Major Market Cat`,`Department`, `Class`, `Pillar` |

### Validation Levels

**Level 1: Pattern Validation** -- Do the same features matter?

- Engineer analogous RFM-style features at the SSL account level
- Run the same 3-gate feature screening on SSL data independently
- Compare which features survive and their relative importance rankings
- If overlapping features dominate in both datasets, that's strong external evidence

**Level 2: Directional Prediction** -- Does the TheLook model generalize?

- Map SSL account-level features to the TheLook feature space
- Apply the TheLook-trained model to SSL data
- Check if accounts with high actual `total_loss` tend to be flagged as high-risk
- Report directional alignment (not raw AUC, since domains differ)

### SSL Feature Mapping

**Note**: SSL data is pre-filtered to returns only. All rows represent return transactions. Return type is indicated by the `Return_Type` column (values: Credit Only, No-Charge Replacement, FC Return, Vendor Return, Unauthorized Return).

| TheLook Feature | SSL Equivalent | SSL Source Columns |
|-----------------|----------------|--------------------|
| `order_frequency` | Unique order count per account | `Order Number` grouped by `Bill To Act #` |
| `return_frequency` | Count of return lines per account | `Order Line ID` count (all rows are returns) |
| `customer_return_rate` | return_frequency / total_lines | Derived |
| `avg_basket_size` | Avg lines per order | `Lines Per Order` |
| `avg_order_value` | Avg `Reference Sale Amount` per order | `Reference Sale Amount` |
| `total_items` | Total return lines | `Order Line ID` count |
| `total_sales` | Sum of `Reference Sale Amount` | `Reference Sale Amount` |
| `total_margin` | Sum of `gross_financial_loss` | `gross_financial_loss` |
| `avg_item_price` | Avg unit price per line | `CreditReturn Sales` / `Ordered Qty` |
| `avg_item_margin` | Avg `gross_financial_loss` per line | `gross_financial_loss` |
| `customer_tenure_days` | Not directly available | May approximate from date range |
| `purchase_recency_days` | Days since last order | `Booked Date` |

**Target variable**: `is_high_loss_account` = 1 if account's `total_loss` > 75th percentile (mirrors TheLook methodology)

### Validation Outputs

- `reports/rq3/rq3_ssl_feature_mapping.csv` -- SSL features mapped to TheLook equivalents
- `reports/rq3/rq3_ssl_feature_screening.csv` -- SSL screening results (for pattern comparison)
- `reports/rq3/rq3_ssl_directional_validation.csv` -- Predicted vs actual high-loss accounts
- `reports/rq3/rq3_validation_summary.csv` -- Pattern alignment and directional metrics

---

## Feature Selection: Multi-Method Screening

Feature selection is performed **before model training** but **after train/test split** to ensure only statistically justified predictors enter the pipeline.

### Pre-Modeling Screening (3 gates -- training set only)

| Gate | Method | Criteria | Removes |
|------|--------|----------|---------|
| **1. Variance check** | `VarianceThreshold` (scikit-learn) | Drop features with near-zero variance (< 0.01) | Constant or quasi-constant features |
| **2. Correlation analysis** | Pearson correlation matrix | If two features have \|r\| > 0.85, drop the one with lower univariate association to target | Redundant/collinear features |
| **3. Univariate statistical test** | Point-biserial correlation (continuous vs binary target) | Drop features with p-value > 0.05 after Bonferroni correction | Statistically irrelevant features |

### Post-Modeling Validation

After model training, feature importance is extracted and cross-referenced against the pre-modeling screening results:

| Model | Importance Method |
|-------|-------------------|
| Logistic Regression | Standardized coefficients (absolute values) |
| Random Forest | `feature_importances_` (Gini importance) |
| Gradient Boosting | `feature_importances_` (split-based) |

### Screening Outputs

- `reports/rq3/rq3_feature_screening.csv` -- screening results per feature (variance, correlation, univariate p-value, pass/fail per gate, final status)
- `reports/rq3/rq3_feature_importance.csv` -- model-based importance rankings (post-training)
- `reports/rq3/rq3_feature_importance.png` -- horizontal bar chart of importance per model

---

## Implementation Tasks

### Task 1: Add RQ3 Constants to `src/config.py` (DONE)

```python
# --- RQ3: Predictive Modeling ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
AUC_THRESHOLD = 0.70
CUSTOMER_TARGETS_CSV = PROCESSED_DATA_DIR / "customer_profit_erosion_targets.csv"
SSL_RETURNS_CSV = RAW_DATA_DIR / "SSL_Returns_df_yoy.csv"
RQ3_TARGET = "is_high_erosion_customer"
RQ3_CANDIDATE_FEATURES = [
    "order_frequency", "return_frequency", "customer_return_rate",
    "avg_basket_size", "avg_order_value", "customer_tenure_days",
    "purchase_recency_days", "total_items", "total_sales",
    "total_margin", "avg_item_price", "avg_item_margin",
]
RQ3_LEAKAGE_COLUMNS = [
    "total_profit_erosion", "total_margin_reversal", "total_process_cost",
    "is_high_erosion_customer", "profit_erosion_quartile",
    "erosion_percentile_rank", "user_id",
]
```

### Task 2: Create `src/rq3_modeling.py` (DONE)

| Function | Purpose |
|----------|---------|
| `prepare_modeling_data()` | Drop leakage cols, impute missing (median), stratified 80/20 split with all 12 candidates |
| `screen_features()` | Multi-method screening on training set: variance -> correlation -> univariate stats; returns surviving feature list + screening report DataFrame |
| `build_model_configs()` | Return dict of 3 models with param grids |
| `train_and_evaluate()` | GridSearchCV (stratified k-fold, scoring=roc_auc) per model using surviving features, test-set metrics |
| `get_feature_importance()` | Extract coefficients (LR) / feature_importances_ (trees) for surviving features |
| `build_comparison_table()` | DataFrame: model, cv_auc, test_auc, precision, recall, F1, meets_threshold |
| `test_hypothesis()` | Compare best AUC to 0.70 threshold, return reject/fail-to-reject |
| `main()` | End-to-end pipeline: load -> prep -> screen -> train -> evaluate -> export to `reports/rq3/` |

### Task 3: Create `src/rq3_visuals.py` (DONE)

| Function | Purpose |
|----------|---------|
| `plot_roc_curves()` | All models on same axes with AUC in legend |
| `plot_feature_importance()` | Horizontal bar chart per model |
| `plot_confusion_matrices()` | Side-by-side for each model |
| `plot_precision_recall_curves()` | PR curves for all models |

### Task 4: Create `tests/test_rq3_modeling.py` (DONE)

Test classes with small synthetic fixtures (CI-safe, no real data):
- `TestPrepareModelingData` -- split sizes, leakage exclusion, all 12 candidates present in split, stratification preserved
- `TestScreenFeatures` -- variance gate drops constant cols, correlation gate drops redundant cols, univariate gate drops irrelevant cols, screening report contains all candidates with pass/fail status
- `TestBuildModelConfigs` -- returns 3 models with estimator + params
- `TestTrainAndEvaluate` -- returns results for all models, AUC in [0,1], uses only surviving features
- `TestGetFeatureImportance` -- correct columns, only surviving features present
- `TestBuildComparisonTable` -- expected columns, threshold flag
- `TestHypothesisTest` -- correct reject/fail logic

### Task 5: Create `src/rq3_validation.py` (DONE)

| Function | Purpose |
|----------|---------|
| `load_ssl_data()` | Load SSL CSV, parse dates, basic cleaning |
| `engineer_ssl_account_features()` | Aggregate SSL order lines to account level, producing analogous features to TheLook's 12 candidates |
| `create_ssl_targets()` | Create `is_high_loss_account` target using 75th percentile of `total_loss` (mirrors TheLook methodology) |
| `validate_feature_patterns()` | Run the same 3-gate screening on SSL data independently; compare surviving features and importance rankings against TheLook results |
| `validate_directional_predictions()` | Map SSL features to TheLook feature space, apply TheLook-trained model, measure directional alignment between predicted risk and actual `total_loss` |
| `build_validation_summary()` | Summarize pattern alignment (feature overlap %) and directional metrics (rank correlation, confusion at directional level) |

### Task 6: Create `tests/test_rq3_validation.py` (DONE)

Test classes with small synthetic SSL-like fixtures (CI-safe, no real data):
- `TestLoadSslData` -- expected columns present, date parsing, drops missing accounts (4 tests)
- `TestEngineerSslAccountFeatures` -- one row per account, expected feature columns, return rate = 1.0 (7 tests)
- `TestCreateSslTargets` -- binary target created, 75th percentile logic, custom percentile (5 tests)
- `TestValidateFeaturePatterns` -- returns comparison DataFrame with TheLook and SSL screening results (5 tests)
- `TestValidateDirectionalPredictions` -- returns alignment metrics, handles missing features gracefully (6 tests)
- `TestBuildValidationSummary` -- returns summary DataFrame with expected metrics (3 tests)

---

## Models & Hyperparameter Grids

| Model | Key Params | Class Balance |
|-------|-----------|---------------|
| **Logistic Regression** | C=[0.01,0.1,1,10], penalty=[l1,l2], solver=saga | `class_weight='balanced'` |
| **Random Forest** | n_estimators=[100,200], max_depth=[5,10,None], min_samples_leaf=[5,10] | `class_weight='balanced'` |
| **Gradient Boosting** | n_estimators=[100,200], max_depth=[3,5], learning_rate=[0.01,0.1], subsample=[0.8,1.0] | `sample_weight` from class distribution |

---

## Files to Create/Modify

| File | Action | Status |
|------|--------|--------|
| `src/config.py` | Modify -- add RQ3 constants | DONE |
| `src/rq3_modeling.py` | Create -- ML pipeline + `main()` entry point | DONE |
| `src/rq3_visuals.py` | Create -- visualization functions | DONE |
| `tests/test_rq3_modeling.py` | Create -- unit tests (32 tests, all pass) | DONE |
| `notebooks/rq3_predictive_modeling.ipynb` | Create -- primary modeling notebook | DONE |
| `src/rq3_validation.py` | Create -- SSL external validation pipeline (6 functions) | DONE |
| `tests/test_rq3_validation.py` | Create -- validation unit tests (30 tests, all pass) | DONE |
| `notebooks/rq3_ssl_validation.ipynb` | Create -- external validation notebook | DONE |

---

## Verification

### Primary Pipeline (TheLook)

1. `pytest tests/test_rq3_modeling.py -v` -- all 32 unit tests pass (DONE)
2. `pytest tests/ -v --tb=short` -- no regressions, 293/293 pass (DONE)
3. `python -m src.rq3_modeling` -- generates artifacts in `reports/rq3/`:
   - `rq3_feature_screening.csv` -- which features passed/failed each gate
   - `rq3_model_comparison.csv` -- model performance summary
   - `rq3_feature_importance.csv` -- model-based importance rankings
   - `rq3_roc_curves.png`
   - `rq3_confusion_matrices.png`
   - `rq3_feature_importance.png`
4. Confirm best model AUC > 0.70 (or document if not met)
5. Verify feature screening report shows clear justification for included/excluded features

### External Validation (SSL)

6. `pytest tests/test_rq3_validation.py -v` -- all validation unit tests pass
7. `notebooks/rq3_ssl_validation.ipynb` -- generates validation artifacts in `reports/rq3/`:
   - `rq3_ssl_feature_mapping.csv` -- SSL features mapped to TheLook equivalents
   - `rq3_ssl_feature_screening.csv` -- SSL screening results for pattern comparison
   - `rq3_ssl_directional_validation.csv` -- predicted vs actual high-loss accounts
   - `rq3_validation_summary.csv` -- pattern alignment and directional metrics
8. Compare feature screening results: which features survive in both datasets?
9. Report directional alignment: do high-`total_loss` SSL accounts tend to be flagged as high-risk by the TheLook model?
