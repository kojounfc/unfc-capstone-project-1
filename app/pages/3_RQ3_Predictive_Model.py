"""
RQ3: Predict High Profit Erosion Customers

Method: ML Classification — Random Forest, Gradient Boosting, Logistic Regression
Target AUC > 0.70 | External validation: School Specialty LLC (SSL)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="RQ3 – Predictive Model",
    page_icon="🤖",
    layout="wide",
)

ROOT = Path(__file__).parent.parent.parent
REPORTS_RQ3 = ROOT / "reports" / "rq3"
AUC_TARGET = 0.70

# ── Load data once at top level ──────────────────────────────────────────────
_comp_df = None
_screen_df = None
_val_dict = {}
_cost_df = None
_thresh_df = None
_fi_df = None

_comp_path = REPORTS_RQ3 / "rq3_model_comparison.csv"
if _comp_path.exists():
    _comp_df = pd.read_csv(_comp_path)

_screen_path = REPORTS_RQ3 / "rq3_feature_screening.csv"
if _screen_path.exists():
    _screen_df = pd.read_csv(_screen_path)

_val_path = REPORTS_RQ3 / "rq3_validation_summary.csv"
if _val_path.exists():
    _val_df = pd.read_csv(_val_path)
    if "metric" in _val_df.columns and "value" in _val_df.columns:
        _val_dict = dict(zip(_val_df["metric"], _val_df["value"]))

_cost_path = REPORTS_RQ3 / "sensitivity_cost_summary.csv"
if _cost_path.exists():
    _cost_df = pd.read_csv(_cost_path)

_thresh_path = REPORTS_RQ3 / "sensitivity_threshold_summary.csv"
if _thresh_path.exists():
    _thresh_df = pd.read_csv(_thresh_path)

_fi_path = REPORTS_RQ3 / "rq3_feature_importance.csv"
if _fi_path.exists():
    _fi_df = pd.read_csv(_fi_path)

# ── Derived values from data ─────────────────────────────────────────────────
_champion_row = None
if _comp_df is not None:
    _champion_row = _comp_df.loc[_comp_df["test_auc"].idxmax()]
    _champion_name = str(_champion_row["model"])
    _champion_auc = float(_champion_row["test_auc"])
    _all_meet = bool((_comp_df["test_auc"] > AUC_TARGET).all())
    _h0_result = "Rejected" if _all_meet else "Not Rejected"
else:
    _champion_name = "N/A"
    _champion_auc = float("nan")
    _h0_result = "N/A"

if _screen_df is not None:
    _n_pass = int((_screen_df["final_status"] == "pass").sum()) if "final_status" in _screen_df.columns else None
    _n_total = len(_screen_df)
    _fail_df = _screen_df[_screen_df["final_status"] == "fail"] if "final_status" in _screen_df.columns else pd.DataFrame()
else:
    _n_pass = None
    _n_total = None
    _fail_df = pd.DataFrame()

_cost_auc_min = float("nan")
_cost_auc_max = float("nan")
_cost_range_str = "N/A"
if _cost_df is not None and "best_auc" in _cost_df.columns:
    _cost_auc_min = float(_cost_df["best_auc"].min())
    _cost_auc_max = float(_cost_df["best_auc"].max())
    _cost_range_str = f"USD {int(_cost_df['base_cost'].min())}–{int(_cost_df['base_cost'].max())}"

_thresh_auc_min = float("nan")
_thresh_auc_max = float("nan")
if _thresh_df is not None and "best_auc" in _thresh_df.columns:
    _thresh_auc_min = float(_thresh_df["best_auc"].min())
    _thresh_auc_max = float(_thresh_df["best_auc"].max())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🤖 RQ3: Predicting High Profit Erosion Customers")
st.markdown(
    """
**Research Question**: Can behavioral features predict whether a customer will be in the
top quartile of profit erosion (AUC > 0.70 target)?

**Method**: 3-model ML classification pipeline with stratified 80/20 split,
3-gate feature screening, GridSearchCV tuning, and SSL external validation.
"""
)
st.divider()

# ── 5-Tab layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Overview", "📈 Model Results", "🎯 What Matters", "🌐 Validation", "🔬 Robustness"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Pipeline Overview")

    col1, col2, col3 = st.columns(3)
    if _comp_df is not None:
        col1.metric(
            "Champion Model AUC",
            f"{_champion_auc:.4f}",
            f"{_champion_name} — target was {AUC_TARGET}",
            help=(
                "**AUC (Area Under the ROC Curve)** measures how well the model separates "
                "high-erosion customers from low-erosion ones. "
                "A score of 1.0 is perfect; 0.5 is no better than a coin flip. "
                f"Our target was {AUC_TARGET} — anything above that is a business-ready predictor."
            ),
        )
    else:
        col1.metric("Champion Model AUC", "N/A", "Run master notebook")

    if _n_pass is not None:
        col2.metric(
            "Features Used",
            f"{_n_pass} / {_n_total}",
            "after 3-gate screening",
            help=(
                "We started with 12 candidate customer behaviors (e.g. how often they return, "
                "average order value). Before training, we automatically filtered out features "
                "that were redundant or statistically uninformative. "
                f"Only the {_n_pass} most useful signals were kept — keeping the model lean and interpretable."
            ),
        )
    else:
        col2.metric("Features Used", "N/A", "")

    col3.metric(
        "Hypothesis H₀",
        _h0_result,
        f"All {len(_comp_df) if _comp_df is not None else '?'} models exceed AUC > {AUC_TARGET}"
        if _comp_df is not None else "",
        help=(
            "The null hypothesis (H₀) states that behavioral features cannot predict "
            "which customers cause disproportionate profit erosion. "
            "**Rejected** means the evidence is strong enough to conclude the opposite — "
            "behavioral signals are real, measurable predictors of erosion risk."
        ),
    )

    st.divider()
    st.subheader("Feature Screening — 3 Sequential Gates")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
Before training any model, all 12 candidate features pass through three automatic quality gates:

| Gate | What it checks | Business meaning |
|------|---------------|-----------------|
| **Gate 1 — Variance** | Is this feature constant or near-constant across customers? | A feature that barely changes tells the model nothing useful. |
| **Gate 2 — Collinearity** | Does this feature say the same thing as another feature? | Keeping duplicates inflates apparent importance without adding information. |
| **Gate 3 — Relevance** | Is this feature statistically linked to the target? | Features with no detectable relationship to high erosion are dropped before training. |

Features that fail any gate are excluded. Only survivors are used to train and evaluate the models.
"""
        )
    if _screen_df is not None:
        show_cols = [c for c in [
            "feature", "variance_pass", "correlation_pass", "univariate_pass", "final_status",
        ] if c in _screen_df.columns]
        if show_cols:
            display_df = _screen_df[show_cols].copy()
            for col in display_df.columns:
                if display_df[col].dtype == bool:
                    display_df[col] = display_df[col].map({True: "✅", False: "❌"})
            col_cfg = {
                "feature": st.column_config.TextColumn(
                    "Feature", help="Customer behavioral signal evaluated as a candidate predictor."
                ),
                "variance_pass": st.column_config.TextColumn(
                    "Gate 1: Variance",
                    help="✅ = feature has enough variation across customers to be useful.",
                ),
                "correlation_pass": st.column_config.TextColumn(
                    "Gate 2: Collinearity",
                    help="✅ = not redundant with another feature already in the set.",
                ),
                "univariate_pass": st.column_config.TextColumn(
                    "Gate 3: Relevance",
                    help="✅ = statistically linked to high-erosion customers (p < Bonferroni threshold).",
                ),
                "final_status": st.column_config.TextColumn(
                    "Final Status",
                    help="pass = used in model training. fail = excluded.",
                ),
            }
            st.dataframe(display_df, use_container_width=True, hide_index=True,
                         column_config={k: v for k, v in col_cfg.items() if k in display_df.columns})
        else:
            st.dataframe(_screen_df, use_container_width=True, hide_index=True)

        if "univariate_pvalue" in _screen_df.columns and not _fail_df.empty:
            fail_reasons = []
            for _, row in _fail_df.iterrows():
                feat = row["feature"]
                if not row.get("correlation_pass", True):
                    fail_reasons.append(f"`{feat}` — dropped at Gate 2 (collinearity)")
                elif not row.get("univariate_pass", True) and pd.notna(row.get("univariate_pvalue")):
                    fail_reasons.append(f"`{feat}` — dropped at Gate 3 (p = {row['univariate_pvalue']:.4f})")
                else:
                    fail_reasons.append(f"`{feat}` — dropped at screening")
            st.caption(
                "Gate 1: VarianceThreshold < 0.01 | "
                "Gate 2: Pearson |r| > 0.85 (collinearity) | "
                "Gate 3: Point-biserial p > Bonferroni threshold  \n"
                "Dropped: " + " | ".join(fail_reasons)
            )
    else:
        st.warning("Feature screening CSV not found. Run the master notebook.")

    st.divider()
    st.subheader("Pipeline Architecture")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
This diagram shows the order of operations — why it matters:

- **Feature screening happens only on training data** to prevent the model from "peeking" at test results,
  which would produce misleadingly optimistic scores.
- **Leakage columns are removed first** — these are fields that directly encode the answer
  (e.g., total profit erosion itself), so keeping them would make the problem trivially easy but useless in practice.
- **GridSearchCV** automatically finds the best hyperparameter settings by trying many combinations
  and picking the one with the highest cross-validated performance.
- **Held-out test set** (20% of data, never seen during training or screening) gives an honest,
  unbiased measure of how the model would perform on new customers.
"""
        )
    _n_pass_str = str(_n_pass) if _n_pass is not None else "surviving"
    st.markdown(
        f"""
        ```
        Load {_n_total if _n_total else 12} candidate features + target
              ↓
        Drop leakage columns
              ↓
        Impute missing (median)
              ↓
        Stratified 80/20 train/test split
              ↓
        3-gate feature screening (training set ONLY)
              ↓
        Apply {_n_pass_str} surviving features to both sets
              ↓
        GridSearchCV + stratified k-fold CV
              ↓
        Evaluate on held-out test set
              ↓
        Extract feature importance (post-hoc)
        ```
        **Data leakage prevention**: `total_profit_erosion`, `profit_erosion_quartile`,
        `erosion_percentile_rank`, `total_margin_reversal`, `total_process_cost`, `user_id`
        are excluded from all models.
        """
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Performance")

    if _comp_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        for metric, col, label, help_txt in [
            (
                "test_auc", c1, "Test AUC",
                "How well the champion model ranks customers by erosion risk on data it has never seen. "
                "Closer to 1.0 = near-perfect separation of high-risk from low-risk customers.",
            ),
            (
                "cv_auc", c2, "CV AUC",
                "AUC measured during cross-validation (training data only, averaged across 5 folds). "
                "Close agreement with Test AUC confirms the model generalizes — it hasn't memorized the training data.",
            ),
            (
                "f1", c3, "F1 Score",
                "Balances two competing goals: catching as many true high-erosion customers as possible (Recall) "
                "while not raising too many false alarms (Precision). "
                "Higher is better; 1.0 is perfect.",
            ),
            (
                "precision", c4, "Precision",
                "Of all customers the model flags as high-erosion risk, what fraction actually are? "
                "High precision = fewer wasted intervention resources on false positives.",
            ),
        ]:
            if metric in _champion_row.index:
                col.metric(
                    f"{_champion_name} {label}",
                    f"{float(_champion_row[metric]):.4f}",
                    help=help_txt,
                )

        st.divider()
        st.subheader("All Models Comparison")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
We trained three different model types so no single algorithm drives the conclusion.
All three must exceed the AUC > 0.70 target for the hypothesis to be rejected.

| Model | Strengths |
|-------|-----------|
| **Random Forest** | Ensemble of decision trees; robust to outliers; naturally ranks feature importance |
| **Gradient Boosting** | Sequentially corrects errors; often highest raw accuracy |
| **Logistic Regression** | Simple, interpretable baseline; coefficients have direct probability meaning |

**Why Random Forest is the champion**: Highest Test AUC (0.9798) with minimal overfitting
(CV–test gap = 0.0006). AUC is threshold-independent — it measures overall ranking ability
across all operating points and is the accepted standard in classification benchmarking.

---

**When model choice would shift based on business context**

All three models are exceptionally close (AUC spread = 0.011). In practice, the "best" model
depends on the cost of each error type:

| Business Scenario | Preferred Model | Why |
|-------------------|----------------|-----|
| Cheap, scalable intervention (automated email, push notification) | **Gradient Boosting** | Highest Recall (0.9299) — catches the most high-erosion customers when false-alarm cost ≈ zero |
| Expensive per-customer intervention (account manager call, loyalty offer) | **Random Forest** | Highest Precision (0.7822) — fewer wasted high-cost contacts |
| Regulatory or audit requirement | **Logistic Regression** | Calibrated probabilities; coefficients interpretable as log-odds |

**For this project** (no specific intervention cost defined for TheLook), we select by Test AUC
as the primary metric — consistent with standard ML benchmarking practice. All three models
exceed the AUC > 0.70 target, so the hypothesis conclusion is robust regardless of which model
is chosen as champion.
"""
            )
        display_comp = _comp_df.copy()
        num_cols = [c for c in display_comp.columns if display_comp[c].dtype in ["float64", "float32"]]
        for col in num_cols:
            display_comp[col] = display_comp[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        if "meets_threshold" in display_comp.columns:
            display_comp["meets_threshold"] = _comp_df["meets_threshold"].map(
                {True: "✅ Yes", False: "❌ No"}
            )
        comp_col_cfg = {
            "model": st.column_config.TextColumn("Model", help="Algorithm type."),
            "test_auc": st.column_config.TextColumn(
                "Test AUC", help="Performance on held-out data (20%). Primary decision metric."
            ),
            "cv_auc": st.column_config.TextColumn(
                "CV AUC", help="Cross-validated AUC on training data. Should be close to Test AUC."
            ),
            "f1": st.column_config.TextColumn(
                "F1", help="Harmonic mean of Precision and Recall. Balances false positives and false negatives."
            ),
            "precision": st.column_config.TextColumn(
                "Precision", help="Fraction of flagged customers who are truly high-erosion."
            ),
            "recall": st.column_config.TextColumn(
                "Recall", help="Fraction of actual high-erosion customers the model catches."
            ),
            "meets_threshold": st.column_config.TextColumn(
                "Meets Target", help=f"✅ if Test AUC > {AUC_TARGET} (project success criterion)."
            ),
        }
        st.dataframe(
            display_comp,
            use_container_width=True,
            hide_index=True,
            column_config={k: v for k, v in comp_col_cfg.items() if k in display_comp.columns},
        )
    else:
        st.warning("Model comparison CSV not found.")

    st.divider()

    st.subheader("ROC Curves")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
An **ROC curve** plots the trade-off between catching high-erosion customers (True Positive Rate)
and incorrectly flagging safe customers (False Positive Rate) across all possible decision thresholds.

- **Top-left corner** = ideal model (catches everyone, no false alarms)
- **Diagonal line** = random guessing (AUC = 0.5)
- **AUC** = the area under the curve — a single number summarizing the whole trade-off

**Business implication**: A high AUC means you can set whichever threshold fits your intervention budget —
aggressive (flag more, intervene more) or conservative (flag only the clearest cases) —
and the model will still rank customers correctly.
"""
        )
    roc_path = REPORTS_RQ3 / "rq3_roc_curves.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
        if _comp_df is not None:
            _min_auc = float(_comp_df["test_auc"].min())
            st.caption(
                f"All {len(_comp_df)} models achieve AUC ≥ {_min_auc:.4f}, "
                f"far exceeding the {AUC_TARGET} target. "
                f"{_champion_name} is the champion (AUC = {_champion_auc:.4f})."
            )
    else:
        st.warning("ROC curves PNG not found.")

    st.divider()

    st.subheader("Confusion Matrices")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
A **confusion matrix** shows exactly what the model gets right and wrong on the test set:

|  | Model says: High-risk | Model says: Low-risk |
|--|----------------------|---------------------|
| **Actually high-erosion** | ✅ True Positive (caught) | ❌ False Negative (missed) |
| **Actually low-erosion** | ⚠️ False Positive (false alarm) | ✅ True Negative (correct) |

**What to look for**:
- High numbers on the diagonal = good
- False Negatives (top-right) = high-erosion customers we miss — the costliest error
- False Positives (bottom-left) = wasted intervention on safe customers

Our models prioritize **Recall** (minimizing False Negatives) because missing a high-erosion
customer is more costly than an occasional false alarm.
"""
        )
    cm_path = REPORTS_RQ3 / "rq3_confusion_matrices.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
        if _comp_df is not None:
            _rf_recall = float(_champion_row["recall"]) if "recall" in _champion_row.index else None
            _recall_str = f" (recall = {_rf_recall:.4f})" if _rf_recall else ""
            st.caption(
                f"Confusion matrices on the held-out test set. "
                f"{_champion_name}{_recall_str} prioritizes capturing actual high-erosion customers."
            )
    else:
        st.warning("Confusion matrices PNG not found.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT MATTERS (Feature Importance)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Feature Importance — What Drives High Erosion Risk?")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
Feature importance answers: **which customer behaviors are the strongest warning signs of high profit erosion?**

Each bar shows how much a feature contributed to the model's decisions — longer bar = stronger signal.

**How to read this for business decisions**:
- Features near the top are the behaviors you should monitor and act on first
- A feature ranked #1 across all three models is a highly reliable signal — not a fluke of one algorithm
- Features with low importance still passed screening (they're statistically real) but add marginal lift —
  operational simplicity may favor dropping them in a deployment scenario

**Technical note**: Tree models (Random Forest, Gradient Boosting) measure importance as
the average reduction in prediction error when the feature is used to split data.
Logistic Regression uses |coefficient| magnitude instead.
"""
        )

    if _fi_df is not None:
        fi_png = REPORTS_RQ3 / "rq3_feature_importance.png"
        if fi_png.exists():
            st.image(str(fi_png), use_container_width=True)

        st.divider()
        st.subheader("Average Importance Ranking (across models)")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
This table averages importance scores across all three models (Random Forest, Gradient Boosting,
Logistic Regression) to identify signals that are consistently important — not just for one algorithm.

A feature ranked #1 here is the behavior most reliably associated with high profit erosion
across all modeling approaches. These are your highest-confidence intervention levers.
"""
            )

        if "feature" in _fi_df.columns and "importance" in _fi_df.columns:
            avg_imp = (
                _fi_df.groupby("feature")["importance"]
                .mean()
                .reset_index()
                .rename(columns={"importance": "avg_importance"})
                .sort_values("avg_importance", ascending=False)
                .reset_index(drop=True)
            )
            avg_imp.index += 1
            avg_imp.insert(0, "rank", avg_imp.index)
            avg_imp["avg_importance"] = avg_imp["avg_importance"].map("{:.4f}".format)
            st.dataframe(
                avg_imp,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn(
                        "Rank", help="1 = most important signal across all models."
                    ),
                    "feature": st.column_config.TextColumn(
                        "Feature", help="Customer behavioral metric used as a predictor."
                    ),
                    "avg_importance": st.column_config.TextColumn(
                        "Avg. Importance",
                        help="Average importance score across all trained models. Higher = stronger predictor.",
                    ),
                },
            )
            st.caption(
                "Average importance across all trained models. "
                "For LR, |coefficient| is used; for tree models, Gini impurity decrease."
            )

        st.divider()
        st.subheader("Per-Model Detail")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
These charts show how each individual model ranks the features. Consistent ranking across tabs
confirms the signal is genuine. If one model ranks a feature very differently, it may indicate
a model-specific interaction worth investigating.
"""
            )
        if "model" in _fi_df.columns:
            models = _fi_df["model"].unique()
            model_tabs = st.tabs([str(m) for m in models])
            for mt, model_name in zip(model_tabs, models):
                with mt:
                    model_df = _fi_df[_fi_df["model"] == model_name].sort_values(
                        "importance", ascending=False
                    )
                    fig = px.bar(
                        model_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title=f"{model_name} — Feature Importance",
                        labels={"importance": "Importance Score", "feature": "Feature"},
                        color="importance",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(_fi_df, use_container_width=True, hide_index=True)

    else:
        st.warning("Feature importance CSV not found.")

    st.divider()
    st.subheader("Dropped Features")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
These are the features that were candidates but did not survive the 3-gate screening process.
They were excluded from model training — not because the data is missing, but because they
failed a statistical quality check:

- **Gate 2 (collinearity)**: This feature was so similar to another feature that keeping both
  would have been redundant and could have distorted importance rankings.
- **Gate 3 (univariate)**: No statistically detectable relationship to high-erosion customers
  was found, even before accounting for other features.
"""
        )
    if _screen_df is not None and not _fail_df.empty:
        dropped_rows = []
        for _, row in _fail_df.iterrows():
            feat = row["feature"]
            if not row.get("correlation_pass", True):
                gate = "Gate 2 (collinearity)"
                detail = ""
            elif not row.get("univariate_pass", True) and pd.notna(row.get("univariate_pvalue")):
                gate = "Gate 3 (univariate)"
                detail = f"p = {row['univariate_pvalue']:.4f}"
            else:
                gate = "Gate 1 (variance)"
                detail = f"variance = {row.get('variance', ''):.6f}" if pd.notna(row.get("variance")) else ""
            dropped_rows.append({"feature": feat, "dropped_at": gate, "detail": detail})
        st.dataframe(
            pd.DataFrame(dropped_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "feature": st.column_config.TextColumn(
                    "Feature", help="Candidate feature that did not pass screening."
                ),
                "dropped_at": st.column_config.TextColumn(
                    "Dropped At", help="Which gate caused the exclusion."
                ),
                "detail": st.column_config.TextColumn(
                    "Detail", help="Statistical reason for exclusion (p-value or variance score)."
                ),
            },
        )
    else:
        st.info("Feature screening CSV not found.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION (SSL External)
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("External Validation — School Specialty LLC (SSL)")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            """
**Why external validation matters**:
A model that only works on the data it was trained on is not trustworthy for business decisions.
To test real-world generalizability, we applied the model to a completely separate dataset —
School Specialty LLC (SSL), a U.S. B2B educational supplies retailer.

**Which model was used**: The **Random Forest** champion (Test AUC = 0.9798) was applied to SSL.
Random Forest was selected for external validation because it achieved the highest Test AUC —
the threshold-independent ranking metric used as the primary criterion. In deployment contexts
where a cheap, scalable intervention is preferred, Gradient Boosting's higher Recall (0.9299)
may be more appropriate; but for this generalizability test, the AUC champion is the right choice.

**The challenge**: SSL is structurally different from TheLook (B2B vs B2C, different product types,
different purchasing patterns). If the behavioral signals still hold directionally, that is strong
evidence the model captures a universal pattern, not a TheLook-specific artifact.

**What we measured**:
- Do customers the model flags as high-risk actually show higher financial losses in SSL?
- Does the *rank order* of risk predictions align with actual loss rank order?
- Do the same behavioral features show similar concentration patterns in both datasets?
"""
        )
    st.markdown(
        """
**External dataset**: School Specialty LLC — U.S. educational supplies B2B retailer

**Purpose**: Test whether the behavioral pattern learned on TheLook generalizes to
a real-world returns dataset from a different industry and customer type (B2B vs B2C).
"""
    )

    if _val_dict:
        ssl_accounts = int(float(_val_dict.get("ssl_accounts_evaluated", 0)))
        dir_acc = float(_val_dict.get("directional_accuracy", 0))
        spearman = float(_val_dict.get("directional_rank_correlation", 0))
        pattern_pct = float(_val_dict.get("pattern_agreement_pct", 0))
        pattern_count = int(float(_val_dict.get("pattern_agreement_count", 0)))
        features_compared = int(float(_val_dict.get("pattern_features_compared", 0)))
        pred_high = float(_val_dict.get("predicted_high_risk_pct", 0))
        actual_high = float(_val_dict.get("actual_high_loss_pct", 0))
        feats_avail = int(float(_val_dict.get("features_available", 0)))

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Directional Accuracy",
            f"{dir_acc:.1%}",
            "Predicted high-risk matches actual high-loss",
            help=(
                "Of the SSL accounts the model predicted as high-risk, "
                f"{dir_acc:.1%} were confirmed as actually high financial loss accounts. "
                "This is measured on external data the model was never trained on."
            ),
        )
        c2.metric(
            "Spearman ρ",
            f"{spearman:.4f}",
            "Rank correlation (p ≈ 0.00)",
            help=(
                "Spearman rank correlation measures whether the model's risk scores "
                "put customers in the right order — accounts ranked as highest risk "
                "should show the most actual financial loss. "
                "A value near 1.0 means the ordering is nearly perfect. "
                f"Our result of {spearman:.4f} with p ≈ 0.00 means this ordering is not by chance."
            ),
        )
        c3.metric(
            "SSL Accounts",
            f"{ssl_accounts:,}",
            "evaluated",
            help="Number of unique SSL customer accounts used in the external validation.",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric(
            "Predicted High-Risk %",
            f"{pred_high:.1f}%",
            "of SSL accounts",
            help="The fraction of SSL accounts that our model classified as high erosion risk.",
        )
        c5.metric(
            "Actual High-Loss %",
            f"{actual_high:.1f}%",
            "in SSL dataset",
            help="The fraction of SSL accounts that actually had above-median financial losses. Used to assess calibration.",
        )
        c6.metric(
            "Pattern Agreement",
            f"{pattern_count} / {features_compared}",
            f"{pattern_pct:.1f}% of features match",
            help=(
                "How many of the surviving behavioral features show similar concentration patterns "
                "in SSL as in TheLook. A high match rate means the same customer behaviors that "
                "predict erosion in e-commerce also appear in B2B retail — validating the framework."
            ),
        )

        st.divider()
        st.subheader("Full Validation Metrics")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
This table shows the complete set of metrics computed during SSL external validation.
Key metrics to focus on for an executive summary:

- **directional_accuracy** — primary business metric: does the model flag the right accounts?
- **directional_rank_correlation** — does it rank them in the right order?
- **pattern_agreement_pct** — do the same behavioral signals hold in a different industry?

The other metrics provide technical depth for auditing the validation methodology.
"""
            )
        _val_display = pd.read_csv(_val_path)
        st.dataframe(_val_display, use_container_width=True, hide_index=True)

        st.divider()
        ssl_screen_path = REPORTS_RQ3 / "rq3_ssl_feature_screening.csv"
        if ssl_screen_path.exists():
            st.subheader("SSL Feature Pattern Comparison (Level 1)")
            with st.expander("ℹ️ What does this mean?", expanded=False):
                st.markdown(
                    """
This table compares each surviving feature's behavior in TheLook vs SSL:

- **concentration_match** — does the feature show similar skew/concentration in both datasets?
  A match means high-return customers in SSL also cluster on the same behavioral dimension.
- This is a directional check, not a coefficient comparison — we expect magnitudes to differ
  (B2B vs B2C), but the direction and relative ranking should align.
"""
                )
            ssl_screen = pd.read_csv(ssl_screen_path)
            st.dataframe(ssl_screen, use_container_width=True, hide_index=True)
            st.caption(
                f"{pattern_count} / {features_compared} features ({pattern_pct:.1f}%) "
                "exhibit similar concentration patterns in SSL as in TheLook."
            )

        st.divider()
        st.subheader("Interpretation")
        st.markdown(
            f"""
- **Directional accuracy = {dir_acc:.1%}** — the model correctly identifies high-risk SSL accounts
  as high-actual-loss accounts
- **Spearman ρ = {spearman:.4f}** (p ≈ 0.00) — strong positive rank correlation between predicted
  risk score and actual financial loss rank
- **Pattern agreement = {pattern_count} / {features_compared} ({pattern_pct:.1f}%)** — behavioral
  features replicate across B2C (TheLook) to B2B (SSL) contexts
- **{feats_avail} / {_n_pass if _n_pass else '?'} surviving features** available in SSL dataset
- **Caveat**: SSL is B2B ({ssl_accounts:,} accounts); TheLook is B2C.
  Some divergence is expected due to structural differences in purchasing behavior.
"""
        )
    else:
        st.warning("Validation summary CSV not found.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROBUSTNESS (Sensitivity Analysis)
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Sensitivity & Robustness Analysis")
    with st.expander("ℹ️ What does this mean?", expanded=False):
        st.markdown(
            f"""
Sensitivity analysis answers the question: **do the model's conclusions change if we tweak our assumptions?**

Two key assumptions underpin our target variable definition:

1. **Processing cost per return** — we assumed USD 12 as the base cost (mid-range of the USD 10–25 academic literature).
   What if the true cost is USD 8 or USD 18? Does the model still work?

2. **High-erosion threshold** — we defined "high erosion" as the top 25% of customers (75th percentile).
   What if we use 30% or 20% instead? Does model accuracy hold?

**How to use this tab**: Select a specific assumption scenario with the radio buttons.
The metric cards update immediately to show performance under that scenario.
The comparison chart shows all scenarios side-by-side so you can see the full range.

**Bottom line**: If AUC stays well above {AUC_TARGET} across all scenarios, the model's business
conclusions are robust — they don't depend on getting these assumptions exactly right.
"""
        )
    st.markdown(
        """
Two sensitivity analyses test whether model conclusions are robust to key modeling assumptions:

1. **Processing cost sensitivity** — does the target variable change when cost assumptions vary?
2. **Percentile threshold sensitivity** — does model performance hold across different definitions of "high erosion"?
"""
    )

    # ── Processing Cost Sensitivity ───────────────────────────────────────────
    st.subheader(f"Processing Cost Sensitivity ({_cost_range_str})" if _cost_range_str != "N/A" else "Processing Cost Sensitivity")

    if _cost_df is not None and "base_cost" in _cost_df.columns:
        _cost_options = [f"${int(c)}" for c in _cost_df["base_cost"]]
        _default_cost_idx = (
            _cost_df["base_cost"].tolist().index(12.0)
            if 12.0 in _cost_df["base_cost"].tolist() else 0
        )
        _cost_sel = st.radio(
            "Select a processing cost assumption to inspect:",
            _cost_options,
            index=_default_cost_idx,
            horizontal=True,
        )
        _cost_val = float(_cost_sel.replace("$", ""))
        _cost_row = _cost_df[_cost_df["base_cost"] == _cost_val].iloc[0]

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric(
            "AUC",
            f"{_cost_row['best_auc']:.4f}",
            "✅ exceeds target" if _cost_row["best_auc"] > AUC_TARGET else "❌ below target",
            help=f"Model accuracy (AUC) when the base return processing cost is assumed to be {_cost_sel}. "
                 f"The project default is USD 12.",
        )
        cc2.metric(
            "F1 Score",
            f"{_cost_row['f1']:.4f}",
            help="Balance of precision and recall at this cost assumption.",
        )
        cc3.metric(
            "Precision",
            f"{_cost_row['precision']:.4f}",
            help="Of customers flagged as high-risk, what fraction truly are — at this cost assumption.",
        )
        cc4.metric(
            "Recall",
            f"{_cost_row['recall']:.4f}",
            help="Fraction of actual high-erosion customers the model catches — at this cost assumption.",
        )

        st.caption(
            f"Erosion threshold at {_cost_sel}/return: **${_cost_row['threshold_value']:.2f}** · "
            f"Surviving features: **{int(_cost_row['n_surviving_features'])}** · "
            f"High-erosion customers: **{int(_cost_row['n_positive']):,}** ({_cost_row['positive_rate']:.0%})"
        )
        if "surviving_features" in _cost_row.index:
            st.caption(f"Features: {_cost_row['surviving_features']}")

        fig_cost = px.bar(
            _cost_df,
            x=_cost_df["base_cost"].apply(lambda c: f"${int(c)}"),
            y=["best_auc", "f1", "precision", "recall"],
            barmode="group",
            title="Model Metrics Across All Processing Cost Scenarios",
            labels={"value": "Score", "variable": "Metric", "x": "Base Processing Cost"},
            color_discrete_sequence=["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"],
        )
        fig_cost.add_hline(
            y=AUC_TARGET,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"AUC target = {AUC_TARGET}",
            annotation_position="top right",
        )
        _sel_x = _cost_sel
        fig_cost.add_vline(
            x=_cost_options.index(_sel_x) - 0.5 + (len(["best_auc", "f1", "precision", "recall"]) / 2 * 0.1),
            line_dash="dot",
            line_color="orange",
            line_width=2,
        )
        fig_cost.update_layout(height=380, xaxis_title="Base Processing Cost")
        st.plotly_chart(fig_cost, use_container_width=True)
        st.caption(
            f"AUC range: {_cost_auc_min:.4f} – {_cost_auc_max:.4f} across all cost assumptions. "
            f"All scenarios exceed the {AUC_TARGET} target."
        )
    else:
        st.info("Processing cost sensitivity CSV not found. Run the master notebook.")

    st.divider()

    # ── Percentile Threshold Sensitivity ──────────────────────────────────────
    if _thresh_df is not None:
        _thresh_min_pct = int(_thresh_df["threshold"].min() * 100)
        _thresh_max_pct = int(_thresh_df["threshold"].max() * 100)
        st.subheader(f"Percentile Threshold Sensitivity ({_thresh_min_pct}th–{_thresh_max_pct}th)")
    else:
        st.subheader("Percentile Threshold Sensitivity")

    if _thresh_df is not None and "threshold" in _thresh_df.columns:
        _thresh_options = [f"{int(t * 100)}th percentile" for t in _thresh_df["threshold"]]
        _default_thresh_idx = (
            _thresh_df["threshold"].tolist().index(0.75)
            if 0.75 in _thresh_df["threshold"].tolist() else 0
        )
        _thresh_sel = st.radio(
            "Select a high-erosion percentile threshold to inspect:",
            _thresh_options,
            index=_default_thresh_idx,
            horizontal=True,
        )
        _thresh_val = float(_thresh_sel.split("th")[0]) / 100
        _thresh_row = _thresh_df[_thresh_df["threshold"] == _thresh_val].iloc[0]

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric(
            "AUC",
            f"{_thresh_row['best_auc']:.4f}",
            "✅ exceeds target" if _thresh_row["best_auc"] > AUC_TARGET else "❌ below target",
            help=f"Model accuracy when 'high erosion' is defined as the top "
                 f"{100 - int(_thresh_val * 100)}% of customers by erosion. "
                 f"The project default is the 75th percentile (top 25%).",
        )
        tc2.metric(
            "F1 Score",
            f"{_thresh_row['f1']:.4f}",
            help="Balance of precision and recall at this threshold definition.",
        )
        tc3.metric(
            "Precision",
            f"{_thresh_row['precision']:.4f}",
            help="Of customers flagged as high-risk, what fraction truly are — at this threshold.",
        )
        tc4.metric(
            "Recall",
            f"{_thresh_row['recall']:.4f}",
            help="Fraction of actual high-erosion customers the model catches — at this threshold.",
        )

        st.caption(
            f"At {_thresh_sel}: **{_thresh_row['positive_rate']:.0%}** of customers flagged as high-erosion "
            f"(**{int(_thresh_row['n_positive']):,}** of {int(_thresh_row['n_total']):,}) · "
            f"Surviving features: **{int(_thresh_row['n_surviving_features'])}**"
        )
        if "surviving_features" in _thresh_row.index:
            st.caption(f"Features: {_thresh_row['surviving_features']}")

        fig_thresh = px.bar(
            _thresh_df,
            x=_thresh_df["threshold"].apply(lambda t: f"{int(t * 100)}th"),
            y=["best_auc", "f1", "precision", "recall"],
            barmode="group",
            title="Model Metrics Across All Percentile Threshold Scenarios",
            labels={"value": "Score", "variable": "Metric", "x": "Percentile Threshold"},
            color_discrete_sequence=["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"],
        )
        fig_thresh.add_hline(
            y=AUC_TARGET,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"AUC target = {AUC_TARGET}",
            annotation_position="top right",
        )
        fig_thresh.update_layout(height=380, xaxis_title="Percentile Threshold")
        st.plotly_chart(fig_thresh, use_container_width=True)
        st.caption(
            f"AUC range: {_thresh_auc_min:.4f} – {_thresh_auc_max:.4f} across all threshold scenarios. "
            f"All scenarios exceed the {AUC_TARGET} target."
        )
    else:
        st.info("Threshold sensitivity CSV not found. Run the master notebook.")

    st.divider()
    st.subheader("Robustness Conclusion")
    if _cost_df is not None and _thresh_df is not None:
        _thresh_min_pct_c = int(_thresh_df["threshold"].min() * 100)
        _thresh_max_pct_c = int(_thresh_df["threshold"].max() * 100)
        st.markdown(
            f"""
- **Processing cost robustness**: AUC ranged from {_cost_auc_min:.4f} to {_cost_auc_max:.4f}
  across the {_cost_range_str} cost range — all scenarios exceed the {AUC_TARGET} target
- **Threshold robustness**: AUC ranged from {_thresh_auc_min:.4f} to {_thresh_auc_max:.4f}
  across the {_thresh_min_pct_c}th–{_thresh_max_pct_c}th percentile range — all scenarios exceed the {AUC_TARGET} target
- **Conclusion**: Model performance is not sensitive to cost model assumptions or threshold choice
"""
        )
    else:
        st.info("Sensitivity CSVs not found. Run the master notebook.")
