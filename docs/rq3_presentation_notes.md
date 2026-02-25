# RQ3 Predictive Modeling — 2-Minute Presentation Notes

---

## 1. Problem Framing (15 sec)

- Predicting which customers cause **high profit erosion** (top 25%) to the organization from return behavior
- **11,988 returners**, binary target at the 75th percentile threshold ($85.92)
- Success criterion: **AUC > 0.70**

---

## 2. Leakage Prevention Pipeline (20 sec)

- **6 leakage columns excluded** before any modeling:
  - `total_profit_erosion`, `total_margin_reversal`, `total_process_cost`
  - `profit_erosion_quartile`, `erosion_percentile_rank`, `user_id`
- Feature screening runs on **training set only** — surviving feature list then applied to both sets
- This strict ordering prevents optimistic bias — a common mistake in applied ML

---

## 3. Data-Driven Feature Screening (20 sec)

- Started with **12 candidates**, 3-gate screening reduced to **7 survivors**:
  - **Gate 1:** Variance threshold (< 0.01) — all passed
  - **Gate 2:** Pairwise correlation (|r| > 0.85) — dropped `order_frequency`, `total_sales`, `avg_item_price`
  - **Gate 3:** Univariate significance (Bonferroni) — dropped `customer_tenure_days` (p=0.41), `purchase_recency_days` (p=0.27)
- **Key insight:** Temporal features are not predictive — how long a customer has been active doesn't matter; *what* they buy and return does

---

## 4. Results — All 3 Models Exceed Threshold (20 sec)

| Model | CV AUC | Test AUC | F1 | Recall |
|-------|--------|----------|----|--------|
| **Random Forest** | 0.9792 | **0.9798** | 0.8419 | 91.2% |
| Gradient Boosting | 0.9797 | 0.9795 | 0.8484 | 93.0% |
| Logistic Regression | 0.9646 | 0.9687 | 0.8256 | 90.5% |

- **H0 rejected** — all three models independently exceed AUC > 0.70
- CV AUC and test AUC nearly identical (max gap = 0.004) — **no overfitting**

---

## 5. Feature Importance Convergence (15 sec)

All 3 models agree on the same top drivers:

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| 1 | **`return_frequency`** | Each return compounds margin reversal + processing cost |
| 2 | **`avg_order_value`** | Higher-value orders amplify margin reversal |
| 3 | **`total_margin`** | Cumulative economic stake at risk |

- Cross-model consistency = **signal is real**, not a model artifact
- Return *frequency* matters more than return *rate* — absolute count drives erosion, not proportion

---

## 6. Robustness — Sensitivity + External Validation (30 sec)

### Sensitivity Analysis (11 scenarios)

| Parameter | Range Tested | AUC Range | Conclusion |
|-----------|-------------|-----------|------------|
| Processing cost | $8 – $18 | 0.9759 – 0.9810 | Stable |
| Threshold percentile | 50th – 90th | 0.9664 – 0.9879 | Stable |

- **AUC > 0.96 in all 11 scenarios** — finding is not an artifact of the $12 cost assumption
- Label stability: Jaccard similarity > 0.93 across cost scenarios (max flip rate = 1.75%)

### External Validation — School Specialty LLC (real B2B data)

| Metric | Value |
|--------|-------|
| Accounts | 13,616 (educational supplies, B2B) |
| **Spearman rank correlation** | **0.75** (p ≈ 0.00) |
| **Directional accuracy** | **76.4%** |
| Specificity | 80.5% |
| Feature agreement | 7/12 (58.3%) |

- TheLook-trained model generalizes to a **completely different domain**
- 5 core features pass screening in both datasets: `customer_return_rate`, `avg_basket_size`, `avg_order_value`, `total_margin`, `avg_item_margin`

---

## Anticipated Question

> **"Why is AUC so high?"**

1. Strong univariate signal — `return_frequency` alone has r = 0.61 with the target
2. Clean 75th-percentile threshold creates meaningful separation between groups
3. Synthetic data lacks real-world noise and edge cases
4. **BUT** — sensitivity analysis (11 scenarios) and SSL external validation (Spearman r = 0.75) confirm the signal is genuine, not an artifact
