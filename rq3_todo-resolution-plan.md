# Plan: Address TODOs in RQ3 Technical Documentation

## Context

The user embedded 10 inline `to do` markers throughout `docs/rq3_technical_documentation.md` requesting literature references, methodological justifications, and one code improvement. These need to be resolved to strengthen the document for capstone defense. Web research has been completed for all items.

---

## TODO Inventory

| # | Location | Topic | Action |
|---|----------|-------|--------|
| 1 | Sec 6.1, line 89 | Data leakage prevention pipeline literature | Add references |
| 2 | Sec 6.1, line 103 | Training-only feature screening literature | Add references |
| 3 | Sec 6.3, line 114 | Justification for variance threshold, Pearson correlation, point-biserial, Bonferroni | Add narrative + references |
| 4 | Sec 6.6, line 154 | Is hyperparameter tuning important given AUC > 0.96? | Add justification + references |
| 5 | Sec 7.6, line 240 | False positives as "manageable review cost" literature | Add references |
| 6 | Sec 9.2, line 288 | Jaccard similarity & flip rate justification | Add narrative + references |
| 7 | Sec 10.1, line 349 | External/cross-domain validation literature | Add references |
| 8 | Sec 10.5, line 444 | Directional accuracy methodology explanation | Add narrative + references |
| 9 | Sec 10.6, line 454 | Inline references for interpretation claims | Add inline citations |
| 10 | Sec 11, line 470 | Improve `avg_item_price` using Reference Sale Amount | Deferred — rewrite as documented limitation |

---

## Implementation Steps

### Step 1: Add references section to the document

Add a **References** section at the end of `docs/rq3_technical_documentation.md` (before or after Section 13) with the full bibliography. All inline TODOs will cite from this list using author-year format.

Key references to include:

**Data Leakage (TODOs 1-2):**
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*, Sec 7.10.2 — "Wrong way" vs "right way" for feature selection with CV
- Kaufman, Rosset, Perlich & Stitelman (2012). Leakage in data mining. *ACM TKDD* — formal "learn-predict separation" framework
- Rosenblatt et al. (2024). Data leakage inflates prediction performance. *Nature Communications* — empirical demonstration that feature leakage drastically inflates metrics

**Feature Screening Methods (TODO 3):**
- Kuhn & Johnson (2013). *Applied Predictive Modeling*, Ch. 3 — near-zero variance removal rationale
- Dormann et al. (2013). Collinearity review. *Ecography* — |r| > 0.7 as standard threshold; 0.85 is recognized conservative choice
- Kornbrot (2014). Point Biserial Correlation. *Wiley StatsRef* — correct measure for continuous vs. binary association
- Dunn (1961). Multiple comparisons among means. *JASA* — Bonferroni correction; appropriate for small number of comparisons (≤12)
- Guyon & Elisseeff (2003). Variable and feature selection. *JMLR* — multi-stage filter methods taxonomy
- Saeys, Inza & Larranaga (2007). Feature selection in bioinformatics. *Bioinformatics* — hybrid multi-step selection

**Hyperparameter Tuning (TODO 4):**
- Probst, Boulesteix & Bischl (2019). Tunability. *JMLR* — tuned models outperform defaults regardless of initial performance
- Bischl et al. (2023). Hyperparameter optimization. *WIREs* — systematic HPO is essential for reproducibility and fair comparison
- Cawley & Talbot (2010). Over-fitting in model selection. *JMLR* — CV-based tuning prevents false confidence in high AUC

**False Positive Cost (TODO 5):**
- Elkan (2001). Foundations of cost-sensitive learning. *IJCAI* — asymmetric misclassification costs
- Verbeke et al. (2012). Profit-driven churn prediction. *EJOR* — FPs have bounded known cost; FNs have unbounded potential cost

**Sensitivity Analysis Metrics (TODO 6):**
- Nogueira, Sechidis & Brown (2018). Stability of feature selection. *JMLR* — Jaccard as formal stability measure
- Saltelli et al. (2004). *Sensitivity Analysis in Practice*. Wiley — standard SA methodology
- Bousquet & Elisseeff (2002). Stability and generalization. *JMLR* — theoretical basis for "small input change → small output change"

**External Validation (TODO 7):**
- Steyerberg & Harrell (2016). Prediction models need appropriate validation. *J Clinical Epidemiology* — reproducibility vs. transportability distinction
- Debray et al. (2015). Framework for external validation. *J Clinical Epidemiology* — transportability framework
- Justice, Covinsky & Berlin (1999). Assessing generalizability. *Annals of Internal Medicine* — seminal paper on cross-population validation

**Directional Accuracy (TODO 8):**
- Pesaran & Timmermann (1992). Predictive performance test. *J Business & Economic Statistics* — directional accuracy as formal test
- Fawcett (2006). Introduction to ROC analysis. *Pattern Recognition Letters* — confusion matrix metrics reference
- Schober et al. (2018). Correlation coefficients. *Anesthesia & Analgesia* — Spearman for non-normal data

**Interpretation (TODO 9):**
- Steyerberg (2019). *Clinical Prediction Models*, 2nd ed. — retained ordinal discrimination = real signal
- Ben-David et al. (2010). Learning from different domains. *Machine Learning* — formal cross-domain error bounds
- Cohen (1988). *Statistical Power Analysis* — effect size benchmarks (rho > 0.50 = large)

### Step 2: Resolve TODOs 1-2 (Section 6.1 — Data Leakage)

Replace the two `to do` markers in Section 6.1 with inline narrative and citations:

- After the pipeline description (line 89): Add a sentence citing Hastie et al. (2009) Sec 7.10.2 on the "wrong way" (feature selection before split) vs. "right way" (after split).
- After the critical design decision (line 103): Expand with citations to Kaufman et al. (2012) on "learn-predict separation" and Rosenblatt et al. (2024) showing feature leakage "drastically inflates prediction performance."

### Step 3: Resolve TODO 3 (Section 6.3 — Feature Screening Justification)

Replace the `to do` marker with an expanded narrative explaining each gate's rationale:

- **Gate 1 (Variance):** Near-zero variance features add complexity without discrimination (Kuhn & Johnson, 2013). Threshold of 0.01 is consistent with scikit-learn's `VarianceThreshold` default usage.
- **Gate 2 (Pearson Correlation):** |r| > 0.85 removes severe multicollinearity where regression coefficients can change sign (Dormann et al., 2013). When two features exceed the threshold, the one with lower point-biserial target association is dropped to retain predictive value.
- **Gate 3 (Point-Biserial + Bonferroni):** Point-biserial correlation is mathematically equivalent to Pearson's r when one variable is binary — the correct association measure for continuous predictors vs. a binary target (Kornbrot, 2014). Bonferroni correction (alpha/m = 0.05/9 ≈ 0.0056) controls family-wise error rate across simultaneous tests; appropriate for small comparison counts (Dunn, 1961).
- **Sequential multi-gate rationale:** Hybrid multi-stage filter methods leverage complementary strengths — cheap unsupervised filters first, then statistically rigorous supervised tests (Guyon & Elisseeff, 2003; Saeys et al., 2007).

### Step 4: Resolve TODO 4 (Section 6.6 — Hyperparameter Tuning)

Replace the `to do` marker with a brief paragraph explaining why HPO is justified even at AUC > 0.96:

1. **Reproducibility** — systematic tuning with documented search spaces is a scientific standard (Bischl et al., 2023)
2. **Fair model comparison** — without tuning all candidates, performance comparison is not valid (Probst et al., 2019)
3. **Generalization confidence** — high AUC on a single configuration does not guarantee generalization; CV-based tuning provides this assurance (Cawley & Talbot, 2010)

### Step 5: Resolve TODO 5 (Section 7.6 — False Positive Cost)

Replace the `to do` marker with a citation-backed statement:

- In cost-sensitive classification, the cost of a false positive (reviewing a low-risk customer) is a bounded, known operational expense, whereas the cost of a false negative (missing a high-erosion customer) is an unbounded loss (Elkan, 2001; Verbeke et al., 2012). With 137 FPs vs. 53 FNs, the model appropriately favors recall.

### Step 6: Resolve TODO 6 (Section 9.2 — Jaccard & Flip Rate)

Replace the `to do` marker with a narrative defining both metrics and their justification:

- **Jaccard similarity** (Jaccard, 1912): J = |A ∩ B| / |A ∪ B|, measures overlap between the set of customers flagged under the baseline scenario and each alternative scenario. Widely used as a formal stability measure for feature selection and classification outputs (Nogueira et al., 2018).
- **Flip rate**: The proportion of customers whose binary label changes relative to baseline, measuring prediction instability. Grounded in algorithmic stability theory (Bousquet & Elisseeff, 2002) — a stable model should produce small output changes in response to small input perturbations.
- **Sensitivity analysis methodology** follows Saltelli et al. (2004): systematically varying model inputs to apportion output uncertainty to specific parameter choices.

### Step 7: Resolve TODO 7 (Section 10.1 — External Validation Rationale)

Replace the `to do` marker with literature-backed rationale:

- External validation tests *transportability* — the ability of a model to give valid predictions in populations related to but different from the development population (Steyerberg & Harrell, 2016; Debray et al., 2015). This is stronger evidence than *reproducibility* testing within the same domain (Justice et al., 1999).

### Step 8: Resolve TODO 8 (Section 10.5 — Directional Accuracy Explanation)

Replace the `to do` marker with a methodological explanation:

- **Directional accuracy** evaluates whether a model correctly predicts the *direction* (high vs. low) rather than the exact magnitude (Pesaran & Timmermann, 1992). Calculated as the proportion of accounts where the predicted binary label matches the actual binary label.
- **Spearman rank correlation** measures monotonic association between predicted probability and actual loss, appropriate for non-normally distributed data (Schober et al., 2018). A value of 0.75 exceeds Cohen's (1988) "large" effect size threshold of 0.50.
- **Confusion matrix metrics** (Fawcett, 2006) provide the operational breakdown of classification accuracy.

### Step 9: Resolve TODO 9 (Section 10.6 — Inline References)

Add inline author-year citations to the interpretation paragraph:

- Feature-level transferability: cite Debray et al. (2015) on transportability framework
- Directional generalizability: cite Ben-David et al. (2010) on expected cross-domain degradation, Steyerberg (2019) on retained ordinal discrimination as evidence of genuine signal

### Step 10: Resolve TODO 10 (Section 11 — `avg_item_price` Improvement)

**DEFERRED to a separate task.** Replace the inline `to do` marker with a documented note that this is a known improvement opportunity, keeping it as a limitation with clearer framing:

- Current: `avg_item_price` computed from `|CreditReturn Sales / Ordered Qty|` (RETURN lines only) → 37.7% coverage
- Improvement: `Reference Sale Amount / |Ordered Qty|` would increase coverage to ~50-53% (available for 90% of RETURN lines) and is semantically closer to TheLook's `avg_item_price` (original sale price vs. refund amount)
- Rewrite the limitation bullet to acknowledge this as a documented future improvement rather than leaving it as a raw TODO

---

## File Summary

| File | Action |
|------|--------|
| `docs/rq3_technical_documentation.md` | **Modify** — resolve all 10 TODOs, add References section |

---

## Verification

1. All 10 `to do` markers removed from `docs/rq3_technical_documentation.md`
2. TODOs 1-9 replaced with narrative + citations in author-year format
3. TODO 10 rewritten as a documented limitation/future improvement (no code change)
4. References section added at end of document
5. Grep for `to do` in the file returns zero matches
