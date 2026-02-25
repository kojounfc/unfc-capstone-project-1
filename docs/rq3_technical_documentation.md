# RQ3 Technical Documentation
**Capstone Project – Master of Data Analytics**
**Research Question 3 (RQ3)**

---

## 1. Research Question

**RQ3:**
*Can machine learning models accurately predict high profit erosion customers using transaction-level and behavioral features, and which features contribute most significantly to prediction accuracy?*

This research question evaluates whether customer-level behavioral features, derived from purchase and return activity, contain sufficient predictive signal to identify customers who generate disproportionately high profit erosion. Establishing this predictive capability is a prerequisite for any operationally deployable early-warning or intervention system.

---

## 2. Hypotheses

Hypothesis testing was framed around the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), a standard discrimination metric for binary classification.

- **H₀ (Null Hypothesis):**
  The best-performing model achieves AUC ≤ 0.70. Machine learning models cannot reliably discriminate between high and low profit erosion customers.

- **H₁ (Alternative Hypothesis):**
  The best-performing model achieves AUC > 0.70. Machine learning models can reliably predict high profit erosion customers from behavioral features.

**Threshold justification:** An AUC of 0.70 represents the lower bound of acceptable discrimination in applied classification literature (Hosmer & Lemeshow, 2000). Below this threshold, a model's ranking ability is considered insufficient for operational use.

---

## 3. Data Scope and Unit of Analysis

- **Unit of analysis:** Customer (aggregated from order-item transactions)
- **Dataset:** TheLook e-commerce dataset (synthetic), consolidated via US06 feature engineering pipeline
- **Population:** 11,988 customers with at least one returned item
- **Target variable:** `is_high_erosion_customer` — binary indicator (1 = customer's total profit erosion ≥ 75th percentile)
- **Class distribution:** 2,997 positive (25.0%) / 8,991 negative (75.0%)

The 75th percentile threshold segments the top quartile of profit-eroding customers, creating a class split appropriate for binary classification while reflecting the Pareto principle that a minority of customers drive the majority of return-related losses.

---

## 4. Feature Engineering and Profit Erosion Definition (US06)

Profit erosion was operationalized using the standardized **US06 feature engineering pipeline**, consistent with the methodology applied in RQ1.

- **Profit Erosion Formula:**

  \[
  \text{Profit Erosion} = \text{Margin Reversal} + \text{Processing Cost}
  \]

- **Margin Reversal:** The item-level contribution margin lost due to the return (`item_margin`).
- **Processing Cost:** A modeled reverse-logistics cost ($12 base × category-tiered multiplier).

Twelve candidate predictor features were engineered at the customer level from order-item transaction data, capturing four behavioral dimensions:

| Dimension | Features |
|-----------|----------|
| **Purchase behavior** | `order_frequency`, `avg_basket_size`, `avg_order_value`, `total_items`, `total_sales` |
| **Return behavior** | `return_frequency`, `customer_return_rate` |
| **Margin structure** | `total_margin`, `avg_item_price`, `avg_item_margin` |
| **Temporal** | `customer_tenure_days`, `purchase_recency_days` |

These twelve features are **candidates**, not automatic predictors. Feature screening (Section 6) determines which features are statistically justified for inclusion in the final model.

---

## 5. Data Leakage Prevention

Six columns were identified as data leakage risks and excluded from the predictor set prior to any modeling:

| Excluded Column | Reason |
|-----------------|--------|
| `total_profit_erosion` | Target is derived directly from this value |
| `total_margin_reversal` | Arithmetic component of the target |
| `total_process_cost` | Arithmetic component of the target |
| `profit_erosion_quartile` | Derived from the target distribution |
| `erosion_percentile_rank` | Derived from the target distribution |
| `user_id` | Row identifier with no predictive meaning |

This exclusion is enforced programmatically before any train/test split to guarantee that no target-derived information enters the modeling pipeline.

---

## 6. Modeling Methodology

### 6.1 Pipeline Architecture

The pipeline follows a strict sequential order designed to prevent information leakage and ensure reproducibility. This architecture implements what Hastie, Tibshirani, and Friedman (2009, Sec 7.10.2) term the "right way" to combine feature selection with cross-validation — performing all data-driven selection steps *after* the train/test split, inside the training fold only. The alternative ("wrong way") applies feature selection to the entire dataset before splitting, allowing test-set distributional information to leak into model selection and producing optimistically biased performance estimates.

```
1. Load data (12 candidates + target)
2. Drop leakage columns (6 columns removed)
3. Impute missing values (median strategy)
4. Stratified train/test split 80/20 (all 12 candidates in both sets)
5. Feature screening on TRAINING SET ONLY (3 sequential gates)
6. Apply surviving feature list to BOTH train and test sets
7. Train models (GridSearchCV, stratified 5-fold CV)
8. Evaluate on held-out test set
9. Extract feature importance (post-hoc, from trained models)
```

**Critical design decision:** Feature screening (Step 5) executes on the training set exclusively. The resulting feature list is then applied to the test set. This enforces the "learn-predict separation" principle formalized by Kaufman et al. (2012), which requires that any data-driven decision — including feature selection — must be learned only from training data and never from the prediction (test) set. Rosenblatt et al. (2024) demonstrated empirically that violating this separation through feature leakage "drastically inflates prediction performance," producing AUC estimates that may be 0.10–0.30 higher than the model's true discrimination ability. By restricting screening to the training fold, this pipeline prevents such optimistic bias.

### 6.2 Data Preparation

- **Split:** 9,590 train / 2,398 test (80/20 stratified)
- **Train positive rate:** 25.01% | **Test positive rate:** 24.98%
- **Imputation:** 294 missing values in `customer_tenure_days` and `purchase_recency_days` imputed with column median
- **Missing after imputation:** 0 (train and test)

### 6.3 Feature Screening (3 Sequential Gates)

Feature selection was performed before model training using a three-gate screening protocol applied to the training set only. This hybrid multi-stage filter approach follows the taxonomy of Guyon and Elisseeff (2003) and the recommendations of Saeys, Inza, and Larrañaga (2007): apply cheap unsupervised filters first to remove uninformative features, then progressively apply more rigorous supervised statistical tests. Each gate addresses a distinct source of noise:

| Gate | Method | Criterion | Removes |
|------|--------|-----------|---------|
| **1. Variance** | `VarianceThreshold` (scikit-learn) | Variance < 0.01 | Constant or quasi-constant features |
| **2. Correlation** | Pearson correlation matrix | \|r\| > 0.85 between two features → drop the one with lower univariate association to the target | Redundant collinear features |
| **3. Univariate** | Point-biserial correlation, Bonferroni correction | Adjusted p-value > 0.05 | Statistically irrelevant features |

**Gate-level justification:**

- **Gate 1 (Variance threshold = 0.01):** Near-zero variance features add dimensionality without discriminative power (Kuhn & Johnson, 2013, Ch. 3). A threshold of 0.01 is consistent with standard usage of scikit-learn's `VarianceThreshold`, removing only features that are effectively constant across the training set.
- **Gate 2 (Pearson |r| > 0.85):** Severe multicollinearity inflates coefficient variance in linear models and can cause regression coefficients to change sign (Dormann et al., 2013). The commonly cited threshold is |r| > 0.7; the 0.85 threshold used here is a conservative choice that removes only the most severely collinear pairs. When two features exceed the threshold, the one with lower point-biserial target association is dropped to retain the more predictive feature.
- **Gate 3 (Point-biserial + Bonferroni):** Point-biserial correlation is mathematically equivalent to Pearson's r when one variable is binary — the correct measure of association between continuous predictors and a binary target (Kornbrot, 2014). Bonferroni correction (α/m = 0.05/9 ≈ 0.0056) controls the family-wise error rate across simultaneous tests and is appropriate for small numbers of comparisons (Dunn, 1961). This gate removes features that show no statistically significant association with the target after multiple-testing correction.

### 6.4 Baseline Reference

To contextualize model performance, the following baselines apply:

| Baseline | AUC | Description |
|----------|-----|-------------|
| Random classifier | 0.500 | No discrimination ability |
| Majority-class classifier | 0.500 | Predicts all customers as low-erosion (75% accuracy but zero recall) |
| **Acceptable threshold** | **0.700** | Minimum for operational use (Hosmer & Lemeshow) |

Any trained model must substantially exceed these baselines to demonstrate that learned behavioral patterns provide genuine discrimination beyond chance.

### 6.5 Model Selection

Three model families were selected to provide complementary perspectives:

| Model | Rationale | Class Imbalance Handling |
|-------|-----------|--------------------------|
| **Logistic Regression** | Linear baseline; interpretable coefficients; tests linear separability | `class_weight='balanced'` |
| **Random Forest** | Non-linear ensemble; captures feature interactions without explicit specification | `class_weight='balanced'` |
| **Gradient Boosting** | Sequential boosting; typically achieves highest discrimination on tabular data | `sample_weight` from class distribution |

### 6.6 Hyperparameter Optimization

Optimization was conducted via `GridSearchCV` with stratified 5-fold cross-validation, scoring on AUC-ROC:

| Model | Hyperparameter Grid | Combinations |
|-------|---------------------|--------------|
| Logistic Regression | C: [0.01, 0.1, 1, 10], penalty: [L1, L2] | 8 |
| Random Forest | n_estimators: [100, 200], max_depth: [5, 10, None], min_samples_leaf: [5, 10] | 12 |
| Gradient Boosting | n_estimators: [100, 200], max_depth: [3, 5], learning_rate: [0.01, 0.1], subsample: [0.8, 1.0] | 16 |

**Why hyperparameter tuning is justified despite AUC > 0.96:** Although initial results exceeded the 0.70 success criterion, systematic hyperparameter optimization serves three purposes beyond raw performance improvement: (1) **Reproducibility** — documented search spaces with cross-validated selection are a scientific standard for transparent, reproducible modeling (Bischl et al., 2023); (2) **Fair model comparison** — without tuning all candidate models under equivalent conditions, performance comparisons are not valid, as one model may benefit from favorable defaults while another suffers from poor ones (Probst et al., 2019); (3) **Generalization confidence** — a high AUC on a single default configuration does not guarantee generalization to new data. CV-based tuning provides distributional evidence that performance is stable across data partitions (Cawley & Talbot, 2010).

---

## 7. Results

### 7.1 Feature Screening Results

All 12 candidate features passed Gate 1 (variance check). The subsequent gates removed five features:

| Feature | Gate Failed | Reason |
|---------|-------------|--------|
| `order_frequency` | Gate 2 (Correlation) | \|r\| > 0.85 with `return_frequency`; lower target association |
| `total_sales` | Gate 2 (Correlation) | \|r\| > 0.85 with `total_margin`; lower target association |
| `avg_item_price` | Gate 2 (Correlation) | \|r\| > 0.85 with `avg_item_margin`; lower target association |
| `customer_tenure_days` | Gate 3 (Univariate) | p = 0.4053; not significant after Bonferroni correction |
| `purchase_recency_days` | Gate 3 (Univariate) | p = 0.2730; not significant after Bonferroni correction |

**Surviving features (7 of 12):**

| Feature | Point-Biserial r | p-value | Status |
|---------|-----------------|---------|--------|
| `return_frequency` | 0.6141 | < 1e-16 | Pass |
| `avg_order_value` | 0.5846 | < 1e-16 | Pass |
| `avg_basket_size` | 0.4707 | < 1e-16 | Pass |
| `total_margin` | 0.4520 | < 1e-16 | Pass |
| `avg_item_margin` | 0.4062 | < 1e-16 | Pass |
| `total_items` | 0.2909 | < 1e-16 | Pass |
| `customer_return_rate` | 0.1022 | < 1e-16 | Pass |

### 7.2 Model Performance

| Model | CV AUC (5-fold) | Test AUC | Precision | Recall | F1 | Accuracy | Best Parameters |
|-------|-----------------|----------|-----------|--------|----|----------|-----------------|
| **Random Forest** | 0.9792 | **0.9798** | 0.7822 | 0.9115 | 0.8419 | 0.9145 | n_estimators=200, max_depth=10, min_samples_leaf=10 |
| **Gradient Boosting** | 0.9797 | 0.9795 | 0.7801 | 0.9299 | 0.8484 | 0.9170 | learning_rate=0.1, max_depth=3, n_estimators=100 |
| Logistic Regression | 0.9646 | 0.9687 | 0.7591 | 0.9048 | 0.8256 | 0.9045 | C=10, penalty=L1 |
| *Majority-class baseline* | *0.500* | *0.500* | *—* | *0.000* | *0.000* | *0.750* | *—* |

All three models exceed the AUC > 0.70 threshold by a substantial margin (test AUC range: 0.9687–0.9798), representing a +0.48 improvement over the random baseline.

### 7.3 Champion Model Selection Rationale

#### 7.3.1 Primary Criterion: Test AUC

All three models exceed the AUC > 0.70 hypothesis threshold, making the hypothesis conclusion robust to model choice. Among them, **Random Forest achieves the highest Test AUC (0.9798)** with near-zero overfitting (CV–test gap = 0.0006), and is designated champion for external validation and deployment guidance.

AUC is selected as the primary criterion because it is threshold-independent — it measures overall ranking ability across all possible decision thresholds rather than at a single operating point — and is the accepted standard in binary classification benchmarking (Hastie, Tibshirani, & Friedman, 2009; Hosmer & Lemeshow, 2000). This is particularly appropriate here, where the optimal intervention threshold depends on cost-per-contact assumptions that are organization-specific and not fixed in advance.

#### 7.3.2 Cost-Asymmetry and the Recall vs. Precision Tradeoff

In cost-sensitive classification, not all prediction errors carry equal cost. For profit erosion intervention in e-commerce, the two error types have asymmetric economic consequences:

- **False Negative cost** (missing a high-erosion customer): unbounded — the customer continues eroding margin without any intervention, compounding over future transactions. Petersen & Kumar (2009) demonstrate that habitual returners generate disproportionate lifetime margin erosion, making missed detection the dominant cost driver in return management programs.
- **False Positive cost** (flagging a low-erosion customer): bounded — one wasted intervention contact, email, or loyalty offer. The per-unit cost is small relative to the revenue recovery potential from true positives.

This economic asymmetry — unbounded false negative cost vs. bounded false positive cost — provides the principled justification for favoring Recall over Precision in the design of return intervention systems. Elkan (2001) formalizes this as the cost-sensitive learning framework: when misclassification costs are asymmetric, classifier design should minimize expected total cost rather than error rate. Verbeke et al. (2012) apply this directly to customer churn modeling, demonstrating that profit-oriented model selection (which maximizes the expected revenue recovered from true positives minus intervention cost on false positives) outperforms accuracy-optimized selection when false negative costs substantially exceed false positive costs.

In the Random Forest champion, the resulting error profile is:

| Error Type | Count | Rate | Business Impact |
|------------|-------|------|----------------|
| False Negatives (missed high-erosion) | 53 | 8.8% | High-erosion customers receive no intervention |
| False Positives (unnecessary intervention) | 137 | 1.5% | One wasted intervention per flagged customer |

With 91.2% of high-erosion customers correctly identified and only 1.5% of the customer base unnecessarily flagged, the model is calibrated consistent with the asymmetric cost structure.

#### 7.3.3 Model Selection Under Near-Equivalent Performance

All three models occupy a narrow performance band (AUC range: 0.9687–0.9798; Recall range: 0.9048–0.9299; Precision range: 0.7591–0.7822). When models are this close, champion selection by Test AUC is a methodological convention rather than a decisive empirical distinction. Practitioners deploying these models should select by intervention cost context:

| Business Context | Recommended Model | Justification |
|-----------------|------------------|---------------|
| Automated, low-cost intervention (email, push notification) | **Gradient Boosting** | Recall = 0.9299 — maximizes caught cases when false alarm cost ≈ $0 |
| High-cost per-customer intervention (account manager call, loyalty offer) | **Random Forest** | Precision = 0.7822 — minimizes wasted high-cost contacts |
| Regulatory or interpretability requirement | **Logistic Regression** | Calibrated probabilities; coefficients interpretable as log-odds |

For the purposes of this research (no specific intervention cost defined for the TheLook dataset), Random Forest is the champion by Test AUC, consistent with standard ML benchmarking practice. The hypothesis conclusion — **reject H₀** — is robust regardless of which model is designated champion, as all three independently exceed the 0.70 threshold.

### 7.4 Hypothesis Test Outcome

| Component | Result |
|-----------|--------|
| Best model | Random Forest |
| Best test AUC | 0.9798 |
| Threshold | 0.70 |
| **Decision** | **Reject H₀** |

The null hypothesis is rejected. All three model families independently exceed the success criterion, indicating robust and model-agnostic predictive signal.

### 7.5 Cross-Validation Stability

The close agreement between CV AUC and test AUC across all models (maximum gap: 0.0041 for Logistic Regression) indicates minimal overfitting and stable generalization. This stability is attributable to strict separation of feature screening to the training set, stratified splitting, and regularization.

### 7.6 Feature Importance (Post-Hoc)

Feature importance was extracted from each trained model using the method appropriate to the model family:

| Rank | Logistic Regression | Random Forest | Gradient Boosting |
|------|---------------------|---------------|-------------------|
| 1 | `return_frequency` (3.083) | `total_margin` (0.246) | `avg_order_value` (0.469) |
| 2 | `avg_item_margin` (2.238) | `avg_order_value` (0.235) | `return_frequency` (0.185) |
| 3 | `avg_order_value` (1.759) | `return_frequency` (0.211) | `total_margin` (0.176) |
| 4 | `total_margin` (1.080) | `avg_item_margin` (0.133) | `avg_item_margin` (0.083) |
| 5 | `customer_return_rate` (0.942) | `customer_return_rate` (0.077) | `customer_return_rate` (0.069) |
| 6 | `avg_basket_size` (0.499) | `avg_basket_size` (0.057) | `avg_basket_size` (0.017) |
| 7 | `total_items` (0.273) | `total_items` (0.041) | `total_items` (0.001) |

Despite fundamentally different learning mechanisms, the three models converge on the same feature tiers:

- **Top-tier** (ranked 1–3 across all models): `return_frequency`, `avg_order_value`, `total_margin`
- **Mid-tier** (ranked 4–5): `avg_item_margin`, `customer_return_rate`
- **Lower-tier** (ranked 6–7): `avg_basket_size`, `total_items`

This cross-model consistency strengthens confidence that the identified features represent genuine predictive signals rather than model-specific artifacts.

### 7.7 Error Analysis

With a 25% positive class rate, the confusion matrix breakdown for the best model (Random Forest) is:

| | Predicted: Low Erosion | Predicted: High Erosion |
|---|---|---|
| **Actual: Low Erosion** | 1,662 (TN) | 137 (FP) |
| **Actual: High Erosion** | 53 (FN) | 546 (TP) |

- **False Positives (137):** Low-erosion customers flagged as high-risk. These are customers who share behavioral patterns with high-erosion customers (e.g., high return frequency or high order value) but whose returns have not yet crossed the 75th percentile threshold. In cost-sensitive classification, the cost of a false positive (reviewing a low-risk customer) is a bounded, known operational expense, whereas the cost of a false negative (missing a high-erosion customer) is an unbounded potential loss in unrealized intervention savings (Elkan, 2001; Verbeke et al., 2012). With 137 FPs vs. 53 FNs, the model appropriately favors recall, accepting a manageable review cost to minimize missed high-erosion cases.
- **False Negatives (53):** High-erosion customers missed by the model. At a miss rate of 8.8% (53/599), the model captures over 91% of high-erosion customers. The missed cases likely represent customers whose erosion is driven by a small number of very high-value returns rather than recurring behavioral patterns.

---

## 8. Interpretation

### 8.1 Feature-Level Interpretation

The most important predictors consistently across all models are:

1. **Return frequency** — the count of return events, not the rate. Each additional return compounds both margin reversal and processing cost.
2. **Average order value** — higher-value orders amplify margin reversal upon return. This captures the economic exposure per transaction.
3. **Total margin** — the cumulative economic stake at risk. Greater cumulative margin provides more margin to reverse.

### 8.2 Temporal Features Are Not Predictive

The exclusion of `customer_tenure_days` and `purchase_recency_days` indicates that how long a customer has been active or how recently they purchased has no significant bearing on high profit erosion. Profit erosion is driven by transactional behavior (what and how much customers buy and return), not by lifecycle position.

### 8.3 Return Rate vs. Return Frequency

`customer_return_rate` survived screening but ranked as the weakest predictor, while `return_frequency` ranked strongest. This reinforces the RQ1 finding that return rate alone is an incomplete proxy for economic risk — a customer with 1 return out of 2 items (50% rate) generates far less erosion than a customer with 5 returns out of 10 items (50% rate).

### 8.4 Why Performance Is High

The AUC values (> 0.96) are notably high. Three factors explain this:

1. **Strong feature-target signal:** `return_frequency` alone has a point-biserial correlation of 0.61 with the target, providing a strong univariate baseline.
2. **Well-defined target variable:** The 75th percentile threshold creates a clean separation between customer groups that differ meaningfully in transactional behavior.
3. **Synthetic data structure:** The TheLook dataset, while realistic in structure, lacks the noise and edge cases of real-world transaction data. This is explicitly acknowledged as a limitation (Section 11).

---

## 9. Sensitivity Analysis

Two modeling choices documented as limitations were subjected to systematic sensitivity analysis to determine whether the RQ3 findings are robust to alternative parameter values.

### 9.1 Motivation

1. **Processing cost base ($12):** Selected as the conservative mid-range of the $10–$25 literature range (see `docs/PROCESSING_COST_METHODOLOGY.md`, Section 7). The cost model directly affects the target variable (`is_high_erosion_customer`) through the `total_profit_erosion` computation.
2. **High-erosion threshold (75th percentile):** Determines the binary classification boundary. Alternative thresholds change the class balance and may affect model discrimination.

**Key insight:** Only the target labels shift across scenarios. The 12 predictor features are computed from transaction-level data and do not depend on either the processing cost or the percentile threshold.

### 9.2 Methodology

- **Model:** Random Forest only (the established best model from the primary analysis). The full 3-model comparison was completed in the primary notebook.
- **Pipeline:** The same leakage-prevention pipeline (Section 6) was applied for each scenario: drop leakage → impute → stratified split → 3-gate screening → GridSearchCV → test-set evaluation.
- **Label stability** was assessed using two complementary metrics, following the sensitivity analysis methodology of Saltelli et al. (2004) — systematically varying model inputs to apportion output uncertainty to specific parameter choices:
  - **Jaccard similarity:** J = |A ∩ B| / |A ∪ B|, where A is the set of customers flagged under the baseline scenario and B is the set flagged under an alternative scenario. Jaccard is a standard formal stability measure for feature selection and classification outputs (Nogueira, Sechidis, & Brown, 2018). A value of 1.0 indicates identical flagged sets; lower values indicate divergence.
  - **Flip rate:** The proportion of customers whose binary label changes relative to baseline, measuring prediction instability. Grounded in algorithmic stability theory (Bousquet & Elisseeff, 2002), a stable model should produce small output changes in response to small input perturbations. A flip rate near 0% indicates that the classification is insensitive to the varied parameter.

### 9.3 Analysis A: Processing Cost Sensitivity ($8–$18)

Five base cost values were tested while holding the threshold at the 75th percentile and keeping category tier multipliers (1.0×/1.15×/1.3×) constant:

| Base Cost | Test AUC | F1 | Precision | Recall | Threshold ($) | Surviving Features |
|-----------|----------|----|-----------|--------|---------------|--------------------|
| $8 | 0.9759 | 0.8393 | 0.7877 | 0.8982 | $77.00 | 7 |
| $10 | 0.9806 | 0.8565 | 0.8117 | 0.9065 | $81.45 | 8 |
| **$12 (baseline)** | **0.9798** | **0.8419** | **0.7822** | **0.9115** | **$85.92** | **7** |
| $14 | 0.9810 | 0.8508 | 0.7818 | 0.9332 | $90.09 | 7 |
| $18 | 0.9807 | 0.8549 | 0.7986 | 0.9199 | $98.35 | 7 |

**AUC range: [0.9759, 0.9810].** All five scenarios exceed the 0.70 threshold by a substantial margin. Performance is remarkably stable: the AUC varies by only 0.005 across a 2.25× cost range ($8 to $18).

**Label stability (vs $12 baseline):**

| Base Cost | Jaccard Similarity | Flip Rate | Flagged Customers |
|-----------|--------------------|-----------|-------------------|
| $8 | 0.9563 | 1.12% | 2,997 |
| $10 | 0.9743 | 0.65% | 2,997 |
| $12 | 1.0000 | 0.00% | 2,997 |
| $14 | 0.9743 | 0.65% | 2,997 |
| $18 | 0.9323 | 1.75% | 2,997 |

The number of flagged customers remains constant at 2,997 (25th percentile by definition), but the composition shifts slightly. Even at the extreme ($18), only 1.75% of customers change their label — Jaccard similarity remains above 0.93, indicating high label stability.

### 9.4 Analysis B: Threshold Sensitivity (50th–90th Percentile)

Six threshold percentiles were tested while holding the processing cost at $12:

| Threshold | Positive Rate | Test AUC | F1 | Precision | Recall | Surviving Features |
|-----------|---------------|----------|----|-----------|--------|--------------------|
| 50th | 50.0% | 0.9664 | 0.8899 | 0.9037 | 0.8766 | 7 |
| 60th | 40.0% | 0.9733 | 0.8811 | 0.8593 | 0.9041 | 8 |
| 70th | 30.0% | 0.9773 | 0.8650 | 0.8223 | 0.9125 | 7 |
| **75th (baseline)** | **25.0%** | **0.9798** | **0.8419** | **0.7822** | **0.9115** | **7** |
| 80th | 20.0% | 0.9848 | 0.8514 | 0.7886 | 0.9250 | 7 |
| 90th | 10.0% | 0.9879 | 0.7862 | 0.6706 | 0.9500 | 8 |

**AUC range: [0.9664, 0.9879].** All six thresholds exceed 0.70 by a substantial margin. AUC increases monotonically as the threshold becomes more selective (50th → 90th), which is expected: extreme quantiles are easier for the model to discriminate.

**Key observations:**
- **F1 peaks at the 50th percentile (0.89)** where class balance is 50/50, then decreases at extreme thresholds (90th: 0.79) as the positive class shrinks and precision drops.
- **Recall is consistently high (0.88–0.95)** across all thresholds, indicating the model reliably identifies customers above the erosion boundary regardless of where that boundary is set.
- **Precision decreases from 0.90 (50th) to 0.67 (90th)** as the positive class shrinks — the model increasingly over-predicts at extreme thresholds.
- The 75th percentile baseline represents a practical balance between AUC (0.9798), F1 (0.84), and operational feasibility (~25% positive rate).

### 9.5 Sensitivity Analysis Conclusion

**The RQ3 findings are robust across all 11 sensitivity scenarios.** AUC exceeds 0.70 in every case, with a combined range of [0.9664, 0.9879]. The hypothesis test conclusion (reject H₀) holds regardless of whether the processing cost is set at $8 or $18, and regardless of whether the high-erosion threshold is set at the 50th or 90th percentile.

The predictive signal arises from behavioral patterns in the 12 candidate features — particularly return frequency, average order value, and total margin — which are independent of the cost model and threshold choice. This confirms that the finding is a property of the underlying data structure, not an artifact of specific parameter values.

---

## 10. External Validation (School Specialty LLC)

### 10.1 Rationale

A holdout from the same TheLook dataset would test within-distribution generalization but cannot assess whether predictive patterns are domain-specific or transferable. External validation tests *transportability* — the ability of a model to give valid predictions in populations related to but different from the development population (Steyerberg & Harrell, 2016; Debray et al., 2015). Transportability provides stronger evidence than *reproducibility* (internal validation within the same domain), because it demonstrates that the learned relationships reflect genuine behavioral patterns rather than dataset-specific artifacts (Justice, Covinsky, & Berlin, 1999).

### 10.2 Validation Data

| Attribute | TheLook (Primary) | SSL (Validation) |
|-----------|-------------------|-------------------|
| Domain | General e-commerce (fashion, B2C) | Educational supplies (B2B) |
| Customers | 11,988 with returns | 13,616 accounts |
| Return order lines | — | 133,800 (37,978 actual returns + 95,822 no-charge replacements) |
| Date range | Synthetic | Jan 2024 – Nov 2025 |
| Data scope | Full transaction history | Return-related transactions only |
| Financial fields | `sale_price`, `cost`, `item_margin` | `CreditReturn Sales`, `Product Cost`, `gross_financial_loss` |
| Return cost | Estimated ($12 × category tier) | Observed (`estimated_labor_cost`, `total_return_cogs`) |
| Target | `is_high_erosion_customer` (75th pct) | `is_high_loss_account` (75th pct of `total_loss` = $570.50) |
| Class distribution | 25.0% positive | 25.0% positive (3,404 / 13,616) |

**SSL data structure:** The `Sales_Type` column distinguishes two line types within return-related orders:
- **RETURN** (37,978 lines, 28.4%): Actual return of goods — credit/refund issued, negative ordered quantity, negative product cost reversal.
- **ORDER** (95,822 lines, 71.6%): No-charge replacement shipments — CreditReturn Sales ≈ $0, positive ordered quantity, company bears replacement cost.

Both line types represent economic costs of the return event, but only RETURN lines correspond to the physical act of returning goods. Feature engineering distinguishes these to produce accurate mappings (see Section 9.3).

SSL return type distribution: No-Charge Replacement (82,261), FC Return (27,819), Vendor Return (13,044), Unauthorized Return (10,676).

### 10.3 Feature Mapping

Analogous features were constructed at the SSL account level. The `Sales_Type` column was used to distinguish actual returns from no-charge replacements, ensuring accurate feature computation:

| TheLook Feature | SSL Mapping | Scope |
|-----------------|-------------|-------|
| `order_frequency` | Unique Order Number count | All lines |
| `return_frequency` | Count of RETURN lines only | RETURN only |
| `customer_return_rate` | RETURN lines / total lines | Both (ratio) |
| `avg_basket_size` | Mean Lines Per Order | All lines |
| `avg_order_value` | Mean Reference Sale Amount per order | All lines |
| `total_items` | Total lines per account | All lines |
| `total_sales` | Sum of Reference Sale Amount | All lines |
| `total_margin` | Sum of `gross_financial_loss` | All lines |
| `avg_item_price` | Mean Reference Sale Amount / \|Ordered Qty\| (CreditReturn Sales fallback on RETURN lines) | All lines |
| `avg_item_margin` | Mean `gross_financial_loss` per line | All lines |
| `customer_tenure_days` | Date range of Booked Date | All lines |
| `purchase_recency_days` | Days since last Booked Date | All lines |

Key mapping decisions:
- **`return_frequency`** counts only `Sales_Type == 'RETURN'` lines. ORDER lines (no-charge replacements) are replacement shipments, not actual returns of goods.
- **`customer_return_rate`** = RETURN lines / total lines per account. This produces meaningful variance (mean = 0.22, std = 0.35) rather than a constant 1.0, reflecting that 62.3% of accounts have only replacement activity (rate = 0), 12.2% have only actual returns (rate = 1), and 25.5% have mixed behavior.
- **`avg_item_price`** uses `Reference Sale Amount / |Ordered Qty|` from ALL line types as the primary source — semantically closest to TheLook's `avg_item_price` (original sale price per unit). ORDER lines carry `Reference Sale Amount` (the price of the item being replaced) but have `CreditReturn Sales ≈ $0`, so `|CreditReturn Sales / Ordered Qty|` is used as fallback on RETURN lines where Reference Sale Amount is unavailable. This produces non-null values for approximately 90% of accounts (up from 37.7% under the previous RETURN-only approach).
- **`total_margin`** maps to `gross_financial_loss` (not `Gross Profit`), consistent with the profit erosion framing.

### 10.4 Level 1 Results — Pattern Validation

The same three-gate feature screening was run independently on SSL account-level features. Of the 12 candidate features compared:

| Feature | TheLook | SSL | Agreement |
|---------|---------|-----|-----------|
| `order_frequency` | Fail | Fail | Yes |
| `return_frequency` | Pass | Fail | No |
| `customer_return_rate` | Pass | Pass | **Yes** |
| `avg_basket_size` | Pass | Pass | **Yes** |
| `avg_order_value` | Pass | Pass | **Yes** |
| `customer_tenure_days` | Fail | Pass | No |
| `purchase_recency_days` | Fail | Pass | No |
| `total_items` | Pass | Fail | No |
| `total_sales` | Fail | Fail | Yes |
| `total_margin` | Pass | Pass | **Yes** |
| `avg_item_price` | Fail | Pass | No |
| `avg_item_margin` | Pass | Pass | **Yes** |

**Pattern agreement: 7/12 features (58.3%).** Five features passed screening in both datasets: `customer_return_rate`, `avg_basket_size`, `avg_order_value`, `total_margin`, `avg_item_margin`. Two features failed in both: `order_frequency`, `total_sales`.

The five disagreements are interpretable:
- `return_frequency` and `total_items` pass in TheLook but fail in SSL. In TheLook, return and non-return transactions coexist, giving these features independent predictive variance. In SSL, the returns-only scope causes high correlation between return frequency and total line count, triggering the correlation gate.
- `customer_tenure_days`, `purchase_recency_days`, and `avg_item_price` pass in SSL but fail in TheLook. The SSL dataset spans a defined 2-year window where temporal features carry more discriminative power. `avg_item_price` is computed from RETURN lines only in SSL, giving it a cleaner signal than in TheLook where it was dropped due to high correlation with `avg_item_margin`.

The **core behavioral features** — those capturing margin structure, return behavior, and order value (`customer_return_rate`, `avg_order_value`, `total_margin`, `avg_item_margin`, `avg_basket_size`) — are consistent across both domains.

### 10.5 Level 2 Results — Directional Prediction

The TheLook-trained Random Forest model (AUC = 0.9798) was applied directly to 13,616 SSL accounts using all 7 surviving features: 

| Metric | Value |
|--------|-------|
| Directional accuracy | **0.7640 (76.4%)** |
| Spearman rank correlation | **0.7526** (p ≈ 0.00) |
| Predicted high-risk | 30.7% of accounts |
| Actual high-loss | 25.0% of accounts |
| Features available | 7 / 7 |

**Directional confusion matrix (SSL):**

| | Predicted: Low Risk | Predicted: High Risk |
|---|---|---|
| **Actual: Low Loss** | 8,220 (TN) | 1,992 (FP) |
| **Actual: High Loss** | 1,221 (FN) | 2,183 (TP) |

**Methodological definitions:**

- **Directional accuracy** evaluates whether a model correctly predicts the *direction* (high vs. low risk) rather than the exact magnitude of loss (Pesaran & Timmermann, 1992). It is calculated as the proportion of accounts where the predicted binary label matches the actual binary label — equivalent to classification accuracy. A value of 76.4% means the model correctly classifies over three-quarters of SSL accounts into their actual risk category.
- **Spearman rank correlation** measures monotonic association between predicted probability and actual loss, appropriate for non-normally distributed financial data (Schober, Boer, & Schwarte, 2018). A value of 0.75 exceeds Cohen's (1988) "large" effect size threshold of 0.50, indicating a strong monotonic relationship.
- **Confusion matrix metrics** (Fawcett, 2006) provide the operational breakdown of classification performance into true/false positives and negatives.

**Key observations:**
- **Recall = 64.1%** (2,183 / 3,404): The model captures nearly two-thirds of high-loss SSL accounts despite being trained on a different domain.
- **Precision = 52.3%** (2,183 / 4,175): Over half of flagged accounts are truly high-loss. The model predicts 30.7% of accounts as high-risk versus 25.0% actual — a modest over-prediction that is operationally reasonable.
- **Specificity = 80.5%** (8,220 / 10,212): The model correctly clears the majority of low-loss accounts, minimizing false alarms.
- **Spearman rank correlation = 0.75**: The model's predicted probability strongly ranks accounts in the correct order relative to their actual total loss. This is the strongest evidence of generalizability — the model correctly identifies which accounts are *more* at risk, even across domains.

### 10.6 Interpretation

The external validation provides two complementary lines of evidence, following the transportability framework of Debray et al. (2015):

1. **Feature-level transferability:** Five of twelve candidate features pass independent screening in both datasets: `customer_return_rate`, `avg_basket_size`, `avg_order_value`, `total_margin`, and `avg_item_margin`. These represent the core behavioral dimensions (return propensity, order value, margin structure) and are predictive in both B2C fashion (TheLook) and B2B educational supplies (SSL). The five disagreements are attributable to structural differences in the returns-only SSL dataset, not to substantive divergence in predictive patterns.

2. **Directional generalizability (strong):** A Spearman rank correlation of 0.75 (p ≈ 0.00) demonstrates that the TheLook model's risk ranking transfers meaningfully to an independent domain. Cross-domain transfer theory predicts some performance degradation when the source and target distributions differ (Ben-David et al., 2010); the retained ordinal discrimination (rho = 0.75, exceeding Cohen's (1988) "large" effect threshold of 0.50) indicates genuine predictive signal rather than domain-specific overfitting (Steyerberg, 2019). Unlike the prior iteration where all SSL lines were treated as returns (producing `customer_return_rate = 1.0` for all accounts and inflated over-prediction), the corrected feature engineering — distinguishing actual returns from no-charge replacements via `Sales_Type` — produces a well-calibrated model that flags 30.7% of accounts as high-risk versus 25.0% actual. The 76.4% directional accuracy with 80.5% specificity indicates the model generalizes both in ranking (Spearman) and in absolute classification with a reasonable false-alarm rate.

---

## 11. Limitations

- The TheLook dataset is synthetic and may not fully capture the complexity and noise of real-world e-commerce data. The high AUC values (> 0.96) should be interpreted with this caveat.
- Return processing costs are modeled using literature-based estimates ($12 base × category tier) rather than directly observed operational costs.
- Recovery or resale value of returned items is not incorporated, which may overstate net profit erosion.
- Feature screening uses univariate methods (point-biserial correlation), which do not capture multivariate interactions. A feature with low univariate association may still contribute in combination with others.
- The 75th percentile threshold for high-erosion classification and the $12 processing cost base are modeling choices. Sensitivity analysis (Section 9) confirmed robustness across alternative values: AUC exceeds 0.70 for all tested cost values ($8–$18) and threshold percentiles (50th–90th).
- SSL validation uses a returns-only dataset. Although the `Sales_Type` distinction (RETURN vs ORDER) was used to compute a meaningful `customer_return_rate`, the dataset still lacks non-return purchase history, meaning the denominator reflects return-related activity only — not total purchasing behavior as in TheLook.
- `avg_item_price` uses `Reference Sale Amount / |Ordered Qty|` from all line types with `|CreditReturn Sales / Ordered Qty|` fallback on RETURN lines, providing approximately 90% account-level coverage. The remaining ~10% of accounts (those with no `Reference Sale Amount` on any line and no RETURN lines) are imputed with the median during screening and modeling.

These limitations are consistent with the scope and objectives of an academic capstone project.

---

## 12. Conclusion (RQ3)

RQ3 provides strong empirical evidence that **machine learning models can accurately predict high profit erosion customers** using behavioral and transactional features. The null hypothesis was rejected, with all three model families exceeding the AUC > 0.70 success criterion by a substantial margin (best: Random Forest, AUC = 0.9798).

The multi-method feature screening reduced the predictor set from 12 candidates to 7 statistically justified features. The most important predictors — return frequency, average order value, and total margin — are consistent across all three model families, confirming that the signal is robust and model-agnostic.

External validation against School Specialty LLC (13,616 accounts, B2B educational supplies) strengthens these findings on two levels. At the feature level, 7 of 12 candidate features (58.3%) showed agreement between independent screening on TheLook and SSL data, with 5 features — `customer_return_rate`, `avg_basket_size`, `avg_order_value`, `total_margin`, and `avg_item_margin` — passing in both datasets. At the directional level, the TheLook-trained Random Forest produced a Spearman rank correlation of 0.75 (p ≈ 0.00) against actual SSL losses, with 76.4% directional accuracy and 80.5% specificity. The model flagged 30.7% of SSL accounts as high-risk versus 25.0% actual, demonstrating well-calibrated cross-domain generalization.

These results were enabled by careful feature mapping that distinguished actual returns (`Sales_Type = RETURN`) from no-charge replacement shipments (`Sales_Type = ORDER`) in the SSL dataset, producing meaningful variance in features like `customer_return_rate` (mean = 0.22, std = 0.35) rather than the constant 1.0 that would result from treating all lines as returns.

Sensitivity analysis (Section 9) confirmed that these findings are robust to alternative parameter values. Across 11 scenarios varying the processing cost base ($8–$18) and the high-erosion threshold percentile (50th–90th), AUC ranged from 0.9664 to 0.9879 — all well above the 0.70 threshold. Label stability analysis showed Jaccard similarity above 0.93 across cost scenarios, with a maximum flip rate of only 1.75%. The predictive signal arises from behavioral patterns in the candidate features, not from specific choices about cost modeling or threshold placement.

These findings extend the descriptive results of **RQ1** into a predictive framework and provide a foundation for **RQ4**, where econometric regression will quantify the marginal associations between specific behaviors and profit erosion while controlling for confounders.

---

## 13. Traceability to User Stories

- **US06:** Return feature engineering and profit erosion computation (upstream data pipeline)
- **US07:** Descriptive aggregation and customer-level behavioral feature construction (upstream features)
- **RQ1:** Established statistically significant cross-category and cross-brand differences in profit erosion (foundational finding)
- **RQ3:** Predictive validation that behavioral features enable accurate identification of high-erosion customers

---

## 14. References

Ben-David, S., Blitzer, J., Crammer, K., Kuber, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175. https://doi.org/10.1007/s10994-009-5152-4

Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., Thomas, J., Ullmann, T., Becker, M., Boulesteix, A.-L., Deng, D., & Lindauer, M. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *WIREs Data Mining and Knowledge Discovery*, 13(2), e1484. https://doi.org/10.1002/widm.1484 — [Free preprint: arXiv:2107.05173](https://arxiv.org/abs/2107.05173)

Bousquet, O., & Elisseeff, A. (2002). Stability and generalization. *Journal of Machine Learning Research*, 2, 499–526. https://www.jmlr.org/papers/v2/bousquet02a.html *(open access)*

Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107. https://www.jmlr.org/papers/v11/cawley10a.html *(open access)*

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates. https://www.routledge.com/Statistical-Power-Analysis-for-the-Behavioral-Sciences/Cohen/p/book/9780805802832

Debray, T. P. A., Vergouwe, Y., Koffijberg, H., Nieboer, D., Steyerberg, E. W., & Moons, K. G. M. (2015). A new framework to enhance the interpretation of external validation studies of clinical prediction models. *Journal of Clinical Epidemiology*, 68(3), 279–289. https://doi.org/10.1016/j.jclinepi.2014.06.018 — [Free full text: PMC4384703](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4384703/) *(open access)*

Dormann, C. F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carré, G., García Marquéz, J. R., Gruber, B., Lafourcade, B., Leitão, P. J., Münkemüller, T., McClean, C., Osborne, P. E., Reineking, B., Schröder, B., Skidmore, A. K., Zurell, D., & Lautenbach, S. (2013). Collinearity: A review of methods to deal with it and a simulation study evaluating their performance. *Ecography*, 36(1), 27–46. https://doi.org/10.1111/j.1600-0587.2012.07348.x

Dunn, O. J. (1961). Multiple comparisons among means. *Journal of the American Statistical Association*, 56(293), 52–64. https://doi.org/10.1080/01621459.1961.10482090 — [JSTOR](https://www.jstor.org/stable/2282330)

Elkan, C. (2001). The foundations of cost-sensitive learning. In *Proceedings of the 17th International Joint Conference on Artificial Intelligence* (IJCAI-01), 973–978. https://cseweb.ucsd.edu/~elkan/rescale.pdf *(free PDF — author's UCSD page)*

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of Machine Learning Research*, 3, 1157–1182. https://www.jmlr.org/papers/v3/guyon03a.html *(open access)*

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Section 7.10.2. https://doi.org/10.1007/978-0-387-84858-7 — [Free official PDF](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf) *(open access — authors' Stanford page)*

Hosmer, D. W., & Lemeshow, S. (2000). *Applied Logistic Regression* (2nd ed.). Wiley. https://doi.org/10.1002/0471722146

Justice, A. C., Covinsky, K. E., & Berlin, J. A. (1999). Assessing the generalizability of prognostic information. *Annals of Internal Medicine*, 130(6), 515–524. https://doi.org/10.7326/0003-4819-130-6-199903160-00016

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15. https://doi.org/10.1145/2382577.2382579

Kornbrot, D. (2014). Point biserial correlation. In *Wiley StatsRef: Statistics Reference Online*. Wiley. https://doi.org/10.1002/9781118445112.stat06227

Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer. Chapter 3. https://doi.org/10.1007/978-1-4614-6849-3

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research*, 18(174), 1–54. https://www.jmlr.org/papers/v18/17-514.html *(open access)*

Pesaran, M. H., & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461–465. https://doi.org/10.1080/07350015.1992.10509922 — [JSTOR](https://www.jstor.org/stable/1391822)

Petersen, J. A., & Kumar, V. (2009). Are product returns a necessary evil? Antecedents and consequences. *Journal of Marketing*, 73(3), 35–51. https://doi.org/10.1509/jmkg.73.3.035 — [ResearchGate](https://www.researchgate.net/publication/247837106_Are_Product_Returns_a_Necessary_Evil_Antecedents_and_Consequences)

Probst, P., Boulesteix, A.-L., & Bischl, B. (2019). Tunability: Importance of hyperparameters of machine learning algorithms. *Journal of Machine Learning Research*, 20(53), 1–32. https://www.jmlr.org/papers/v20/18-444.html *(open access)* — [arXiv preprint](https://arxiv.org/abs/1802.09596)

Rosenblatt, M., Tejavibulya, L., Jiang, R., Noble, S., & Scheinost, D. (2024). Data leakage inflates prediction performance in connectome-based machine learning models. *Nature Communications*, 15, 1829. https://doi.org/10.1038/s41467-024-46150-w — [Free full text: PMC10912291](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10912291/) *(open access)*

Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. *Bioinformatics*, 23(19), 2507–2517. https://doi.org/10.1093/bioinformatics/btm344 — [Free full text at Oxford Academic](https://academic.oup.com/bioinformatics/article/23/19/2507/185254) *(open access)*

Saltelli, A., Tarantola, S., Campolongo, F., & Ratto, M. (2004). *Sensitivity Analysis in Practice: A Guide to Assessing Scientific Models*. Wiley. https://doi.org/10.1002/0470870958

Schober, P., Boer, C., & Schwarte, L. A. (2018). Correlation coefficients: Appropriate use and interpretation. *Anesthesia & Analgesia*, 126(5), 1763–1768. https://doi.org/10.1213/ANE.0000000000002864 *(open access)*

Steyerberg, E. W. (2019). *Clinical Prediction Models: A Practical Approach to Development, Validation, and Updating* (2nd ed.). Springer. https://doi.org/10.1007/978-3-030-16399-0 *(open access — Springer Open)*

Steyerberg, E. W., & Harrell, F. E. (2016). Prediction models need appropriate internal, internal-external, and external validation. *Journal of Clinical Epidemiology*, 69, 245–247. https://doi.org/10.1016/j.jclinepi.2015.04.005 — [Free full text: PMC4688400](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4688400/) *(open access)*

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research*, 218(1), 211–229. https://doi.org/10.1016/j.ejor.2011.09.031 — [ResearchGate](https://www.researchgate.net/publication/220288606_New_insights_into_churn_prediction_in_the_telecommunication_sector_A_profit_driven_data_mining_approach)

---
