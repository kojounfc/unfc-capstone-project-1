Design a detailed implementation plan for adding sensitivity analysis to the RQ3 predictive modeling pipeline for a capstone project. Here's the full context:

## Background
The project models profit erosion from e-commerce returns. Two parameters were chosen by judgment and documented as limitations needing sensitivity analysis:

### Parameter 1: Processing Cost Base ($12)
- Located in `src/feature_engineering.py` as `DEFAULT_COST_COMPONENTS` (sum = $12)
- Category tier multipliers: Premium=1.3x ($15.60), Moderate=1.15x ($13.80), Standard=1.0x ($12)
- `PROCESSING_COST_METHODOLOGY.md` Section 7 recommends testing: base $8-$18, alternative multipliers (0.9x/1.0x/1.2x vs 1.0x/1.15x/1.3x), and tier boundary changes
- The function `calculate_profit_erosion()` already accepts `cost_components` and `category_multipliers` as parameters

### Parameter 2: High Erosion Threshold (75th percentile)
- Located in `src/feature_engineering.py` function `create_profit_erosion_targets()` with parameter `high_erosion_percentile: float = 0.75`
- RQ3 tech doc Section 10 explicitly states: "The 75th percentile threshold for high-erosion classification is a modeling choice. Sensitivity analysis across alternative thresholds was not conducted."
- The target variable `is_high_erosion_customer` is binary: 1 if customer's total_profit_erosion >= 75th percentile

### RQ3 Pipeline (already implemented)
1. Load customer-level data (from `data/processed/customer_profit_erosion_targets.csv`)
2. Drop leakage columns, impute, stratified split 80/20
3. 3-gate feature screening on training set only
4. Train 3 models (LR, RF, GBM) with GridSearchCV
5. Evaluate on test set (AUC, F1, etc.)
6. Best model: Random Forest, AUC=0.9798

### Existing code structure
- `src/config.py` — path constants, `RQ3_CANDIDATE_FEATURES`, `RQ3_TARGET`, `AUC_THRESHOLD=0.70`
- `src/rq3_modeling.py` — `prepare_modeling_data()`, `screen_features()`, `train_and_evaluate()`, `test_hypothesis()`, `build_comparison_table()`
- `src/rq3_visuals.py` — plotting functions for ROC, feature importance, confusion matrices
- `src/feature_engineering.py` — `calculate_profit_erosion()`, `create_profit_erosion_targets()`
- `notebooks/rq3_predictive_modeling.ipynb` — main RQ3 notebook
- `tests/test_rq3_modeling.py` — 32 tests
- Flat module structure under `src/` (no sub-packages)

### What the sensitivity analysis should answer:
1. **Processing cost sensitivity**: If the base cost were $8, $10, $14, or $18 instead of $12, would the same customers be flagged as high-erosion? Would the model's AUC change materially?
2. **Threshold sensitivity**: If we used the 50th, 60th, 70th, 80th, or 90th percentile instead of 75th, how does model performance (AUC, F1, precision, recall) change? Is the 75th percentile robust or was it a lucky choice?

### Design constraints:
- All source modules go under `src/` (flat, no sub-packages)
- Follow existing patterns: functions in src modules, analysis in notebooks, tests in tests/
- The sensitivity analysis should be a notebook for review (user prefers notebook-based analysis)
- Reuse existing functions wherever possible (the pipeline functions are already parameterized)
- Keep it focused — this is a capstone project, not a production system

Please provide:
1. A clear scope of what sensitivity analyses to run
2. Whether new src module functions are needed or if existing ones suffice
3. Notebook structure (sections/cells)
4. What results/artifacts to produce
5. How to update the tech doc once results are available