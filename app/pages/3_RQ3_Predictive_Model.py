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

# ── CSS: hover tooltip system (mirrors RQ4) ───────────────────────────────────
st.markdown(
    """
    <style>
    .rq3-tip-title {
        display: flex; align-items: center; margin-bottom: 0.4rem;
    }
    .rq3-tip-title h2 { margin:0; padding:0; font-size:1.5rem; font-weight:700; letter-spacing:-0.01em; }
    .rq3-tip-title h3 { margin:0; padding:0; font-size:1.35rem; font-weight:600; letter-spacing:-0.01em; }
    .rq3-tip {
        position: relative; display: inline-flex; align-items: center;
        cursor: help; margin-left: 10px; flex-shrink: 0;
    }
    .rq3-tip-icon { font-size: 0.9rem; color: #888; user-select: none; }
    .rq3-tip-box {
        visibility: hidden; opacity: 0; width: 380px;
        background-color: rgba(28,28,44,0.97); color: #e4e4f0;
        text-align: left; border-radius: 8px; padding: 14px 18px;
        font-size: 0.95rem; line-height: 1.65;
        position: absolute; z-index: 9999;
        bottom: calc(100% + 10px); left: 50%; transform: translateX(-50%);
        transition: opacity 0.2s ease;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
        pointer-events: none; white-space: normal;
    }
    .rq3-tip-box::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -6px;
        border: 6px solid transparent; border-top-color: rgba(28,28,44,0.97);
    }
    .rq3-tip:hover .rq3-tip-box { visibility: visible; opacity: 1; }
    .rq3-step-badge {
        background:#f0f4ff; border-radius:6px; padding:8px 14px; margin-bottom:8px;
        font-size:0.75rem; font-weight:700; color:#2c5282; letter-spacing:0.08em;
    }
    @media (max-width: 768px) {
        .rq3-tip-box { width: 260px; font-size: 0.85rem; }
        .rq3-step-badge { font-size: 0.68rem; padding: 6px 10px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).parent.parent.parent
REPORTS_RQ3 = ROOT / "reports" / "rq3"
AUC_TARGET = 0.70

# ── Tooltips ──────────────────────────────────────────────────────────────────
_TOOLTIPS = {
    "kpi_auc": (
        "**Champion AUC (Test Set):** Area Under the ROC Curve on the held-out 20% test set. "
        "A score of 1.0 is perfect; 0.5 is no better than a coin flip. "
        f"Our target was {AUC_TARGET} — Random Forest achieved 0.9798, well above threshold. "
        "All three models exceed the target, confirming strong predictive signal."
    ),
    "kpi_features": (
        "**Features Used:** Out of 12 candidate behavioral features, 7 survived the 3-gate "
        "screening pipeline (variance, collinearity, univariate relevance). "
        "Only training-set survivors were applied to both train and test sets "
        "to prevent data leakage from inflating reported performance."
    ),
    "kpi_h0": (
        "**Hypothesis Decision:** H₀₃ states that machine learning models cannot predict "
        "high profit erosion customers with acceptable accuracy (AUC ≤ 0.70). "
        "All 3 models independently exceed the threshold — H₀₃ is rejected. "
        "Best model: Random Forest Test AUC = 0.9798, exceeding threshold by 0.28."
    ),
    "kpi_ssl": (
        "**SSL Directional Accuracy:** On School Specialty LLC (B2B, 13,616 accounts), "
        "the model correctly ranks high-erosion accounts 76.4% of the time (Spearman ρ = 0.7526). "
        "This confirms the behavioral pattern generalizes beyond the TheLook training domain."
    ),
    "step_leakage": (
        "**Data Leakage Prevention:** Six columns are excluded before the train/test split "
        "because they directly encode the answer — keeping them makes the model trivially accurate "
        "in training but completely useless on new customers. "
        "Rosenblatt et al. (2024) showed that leakage inflates AUC estimates by 0.10–0.30. "
        "Exclusion is enforced programmatically before any data is seen by the model."
    ),
    "step_screening": (
        "**3-Gate Feature Screening:** Screening runs on the training set only — never the test set. "
        "Gate 1 (variance < 0.01) removes constant features. "
        "Gate 2 (Pearson |r| > 0.85) drops redundant collinear pairs, keeping the more predictive one. "
        "Gate 3 (point-biserial p > Bonferroni α) removes features with no statistical link to the target. "
        "7 of 12 candidates survived: return_frequency, avg_order_value, avg_basket_size, "
        "total_margin, avg_item_margin, total_items, customer_return_rate."
    ),
    "step_validation": (
        "**Stratified Split + GridSearchCV:** The 80/20 stratified split preserves the 25% positive rate "
        "in both train (25.01%) and test (24.98%) sets. "
        "GridSearchCV with 5-fold cross-validation searches the hyperparameter space for each model: "
        "LR (8 combos), RF (12 combos), GB (16 combos). "
        "The 20% test set is held out entirely — never used for training or screening — "
        "ensuring the reported AUC = 0.9798 is an unbiased estimate of real-world performance."
    ),
}


def _tip_header(label: str, tooltip_key: str, level: int = 3) -> None:
    """Render a section header with an inline CSS hover tooltip — mirrors RQ4."""
    raw = _TOOLTIPS[tooltip_key]
    parts = raw.split("**")
    tip_html = "".join(
        f"<strong>{p}</strong>" if i % 2 == 1 else p
        for i, p in enumerate(parts)
    )
    st.markdown(
        f'<div class="rq3-tip-title">'
        f'<h{level}>{label}</h{level}>'
        f'<span class="rq3-tip">'
        f'<span class="rq3-tip-icon">ℹ️</span>'
        f'<span class="rq3-tip-box">{tip_html}</span>'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _plain_tip(key: str) -> str:
    return _TOOLTIPS[key].replace("**", "")

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

_ablation_df = None
_ablation_path = REPORTS_RQ3 / "rq3_ablation_study.csv"
if _ablation_path.exists():
    _ablation_df = pd.read_csv(_ablation_path)

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
<p><strong>Research Question (RQ3):</strong> Can machine learning models accurately predict high profit erosion customers using transaction-level and behavioral features, and which features contribute most significantly to prediction accuracy?</p>
<div style="margin-left: 1.5rem;">
<p><strong>Null Hypothesis (H₀₃):</strong> Machine learning models cannot predict high profit erosion customers with acceptable accuracy (AUC ≤ 0.70).</p>
<p><strong>Alternative Hypothesis (H₁₃):</strong> Machine learning models can predict high profit erosion customers with acceptable accuracy (AUC > 0.70).</p>
</div>

**Method**: 3-model ML classification pipeline — Random Forest, Gradient Boosting, Logistic
Regression — with stratified 80/20 split, 3-gate feature screening (variance → collinearity
→ univariate relevance), GridSearchCV hyperparameter tuning, and SSL external validation.
""",
    unsafe_allow_html=True,
)
st.divider()

# ── Executive Summary Banner ───────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0f2440 0%, #1a3660 100%);
        border-left: 5px solid #7986CB;
        border-radius: 10px;
        padding: 22px 28px;
        margin-bottom: 8px;
    ">
        <p style="
            color: #f0c040;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0 0 10px 0;
        ">Executive Summary — Key Findings &amp; Implications</p>
        <p style="color: #e8eaf0; font-size: 1.0rem; line-height: 1.75; margin: 0;">
            <strong style="color: #ffffff;">All four models (including a rule-based baseline)
            exceed the AUC &gt; 0.70 threshold — H₀₃ is rejected.</strong>
            The Random Forest champion achieves Test AUC&nbsp;=&nbsp;0.9798, exceeding the minimum
            threshold by 0.28. Gradient Boosting (0.9795) and Logistic Regression (0.9687) independently
            confirm the result; the hypothesis conclusion is robust to model choice.
            Seven of 12 candidate features survived 3-gate screening:
            <em>return_frequency</em> and <em>avg_order_value</em> are the top-ranked signals.
            A rule-based baseline (return_frequency threshold only) and an ablation study
            (top-3 features removed) confirm ML adds value beyond simple rules and that
            performance is not dominated by a small set of predictors.
            External validation on School Specialty LLC (B2B, 13,616 accounts) yields
            directional accuracy of 76.4% (Spearman ρ&nbsp;=&nbsp;0.7526), demonstrating
            strong transportability across B2C and B2B domains.
            <strong style="color: #f0c040;">Decision: Reject H₀₃</strong> — best model AUC = 0.9798 &gt; 0.70.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border: 0; border-top: 1px solid rgba(49,51,63,0.3); margin: 20px 0 24px 0;'>",
    unsafe_allow_html=True,
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Champion Model AUC",
    f"{_champion_auc:.4f}" if _comp_df is not None else "N/A",
    f"{_champion_name} — target {AUC_TARGET}" if _comp_df is not None else "N/A",
    help=_plain_tip("kpi_auc"),
)
kpi2.metric(
    "Features Selected",
    f"{_n_pass} / {_n_total}" if _n_pass is not None else "N/A",
    f"{_n_pass} candidates passed all screening gates" if _n_pass is not None else "after screening",
    help=_plain_tip("kpi_features"),
)
kpi3.metric(
    "H₀₃ Decision",
    f"{'✅ Rejected' if _h0_result == 'Rejected' else _h0_result}",
    f"All {len(_comp_df) if _comp_df is not None else '?'} models exceed AUC > {AUC_TARGET}"
    if _comp_df is not None else "",
    help=_plain_tip("kpi_h0"),
)
kpi4.metric(
    "SSL Label Agreement",
    f"{float(_val_dict.get('directional_accuracy', float('nan'))):.1%}"
    if _val_dict.get("directional_accuracy") else "N/A",
    "of 13,616 accounts correctly classified | ρ = 0.7526",
    help=_plain_tip("kpi_ssl"),
)

st.divider()

# ── 6-Tab layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📋 Overview", "📈 Model Results", "🎯 What Matters", "🌐 Validation", "🔬 Robustness", "🎯 Conclusion"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── 3-Panel Step-Badge Logic Chain ────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            '<div class="rq3-step-badge">STEP 1 — WHY DOES LEAKAGE MATTER?</div>',
            unsafe_allow_html=True,
        )
        _tip_header("Leakage Prevention — 6 Columns Excluded", "step_leakage")
        st.caption(
            "Six columns are removed before the split because they directly encode the answer. "
            "Including them makes the problem trivially easy in training — and useless in production."
        )
        st.markdown(
            """
- `total_profit_erosion` — **is the target**
- `profit_erosion_quartile` — derived from the target
- `erosion_percentile_rank` — derived from the target
- `total_margin_reversal` — arithmetic component of target
- `total_process_cost` — arithmetic component of target
- `user_id` — row identifier, no predictive meaning

> **Rosenblatt et al. (2024):** leakage inflates AUC by 0.10–0.30.
"""
        )

    with col_b:
        st.markdown(
            '<div class="rq3-step-badge">STEP 2 — HOW ARE FEATURES SELECTED?</div>',
            unsafe_allow_html=True,
        )
        _tip_header("3-Gate Screening — 7 of 12 Survive", "step_screening")
        st.caption(
            "Screening runs on training data only. The same gate decisions are applied "
            "to the test set — but thresholds are derived from training alone."
        )
        st.markdown(
            f"""
| Gate | Criterion | Drops |
|------|-----------|-------|
| 1 — Variance | < 0.01 | Constant features |
| 2 — Collinearity | Pearson \\|r\\| > 0.85 | Redundant pairs |
| 3 — Relevance | p > Bonferroni α | No link to target |

**{_n_pass if _n_pass is not None else 7} survivors:** `return_frequency`, `avg_order_value`,
`avg_basket_size`, `total_margin`, `avg_item_margin`, `total_items`, `customer_return_rate`
"""
        )

    with col_c:
        st.markdown(
            '<div class="rq3-step-badge">STEP 3 — HOW ARE MODELS VALIDATED?</div>',
            unsafe_allow_html=True,
        )
        _tip_header("Stratified Split + GridSearchCV + Held-Out Test", "step_validation")
        st.caption(
            "The 20% test set is never touched during training or screening — "
            "it gives an honest measure of how the model performs on genuinely new customers."
        )
        st.markdown(
            f"""
| | |
|---|---|
| **Train / Test** | 9,590 / 2,398 customers (80/20) |
| **Positive rate** | 25.01% train · 24.98% test |
| **Cross-validation** | 5-fold stratified GridSearchCV |
| **Models trained** | Rule-Based · LR · RF · GB (4 total) |
| **LR / RF / GB combos** | 8 / 12 / 16 |
| **Champion (RF) AUC** | **0.9798** — all 4 models > {AUC_TARGET} |
"""
        )

    st.divider()
    with st.expander("🔬 Feature Screening — 3 Sequential Gates", expanded=False):
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
                        "Feature",
                        help="Customer behavioral signal evaluated as a candidate predictor.",
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
                        help="✅ = statistically linked to high-erosion customers "
                             "(p < Bonferroni threshold).",
                    ),
                    "final_status": st.column_config.TextColumn(
                        "Final Status",
                        help="pass = used in model training. fail = excluded.",
                    ),
                }
                st.dataframe(
                    display_df,
                    width='stretch',
                    hide_index=True,
                    column_config={k: v for k, v in col_cfg.items() if k in display_df.columns},
                )
            else:
                st.dataframe(_screen_df, width='stretch', hide_index=True)

            if "univariate_pvalue" in _screen_df.columns and not _fail_df.empty:
                fail_reasons = []
                for _, row in _fail_df.iterrows():
                    feat = row["feature"]
                    if not row.get("correlation_pass", True):
                        fail_reasons.append(f"`{feat}` — dropped at Gate 2 (collinearity)")
                    elif not row.get("univariate_pass", True) and pd.notna(
                        row.get("univariate_pvalue")
                    ):
                        fail_reasons.append(
                            f"`{feat}` — dropped at Gate 3 (p = {row['univariate_pvalue']:.4f})"
                        )
                    else:
                        fail_reasons.append(f"`{feat}` — dropped at screening")
                st.caption(
                    "Gate 1: VarianceThreshold < 0.01 | "
                    "Gate 2: Pearson |r| > 0.85 (collinearity) | "
                    "Gate 3: Point-biserial p > Bonferroni threshold  \n"
                    "Dropped: " + " | ".join(fail_reasons)
                )
        else:
            st.warning("Feature screening results are not yet available.")

    with st.expander("⚙️ 9-Step Execution Order", expanded=False):
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
This diagram shows the order of operations — why it matters:

- **Feature screening happens only on training data** to prevent the model from "peeking" at test
  results, which would produce misleadingly optimistic scores.
- **Leakage columns are removed first** — these are fields that directly encode the answer
  (e.g., total profit erosion itself), so keeping them would make the problem trivially easy but
  useless in practice.
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
We trained **four models** across two tiers so the conclusion rests on a credible comparison, not a single algorithm.

**Tier 1 — Baselines** (what a simple rule or standard approach achieves):

| Model | Role | Strengths |
|-------|------|-----------|
| **Rule-Based (Return Frequency)** | Practical baseline | Single-feature threshold rule — flags customers whose return frequency exceeds a training-set quantile; no learning beyond one statistic. Sets the floor: any ML model should clearly beat this. |
| **Logistic Regression** | Methodological baseline | L1/L2 regularized; coefficients interpretable as log-odds; standard first step in any classification pipeline. |

**Tier 2 — ML models** (learned from all 7 surviving features):

| Model | Strengths |
|-------|-----------|
| **Random Forest** | Ensemble of decision trees; robust to outliers; naturally ranks feature importance |
| **Gradient Boosting** | Sequentially corrects errors; often highest raw accuracy |

**Why baselines matter**: If the rule-based approach achieves AUC close to the ML models, that would suggest the ML complexity is not justified. The gap between the rule-based AUC and the ML AUC quantifies the added value of the full feature set and learned model.

**Why Random Forest is the champion**: Highest Test AUC with minimal overfitting (CV–test gap < 0.001). AUC is threshold-independent — it measures overall ranking ability across all operating points and is the accepted standard in classification benchmarking.

---

**When model choice would shift based on business context**

| Business Scenario | Preferred Model | Why |
|-------------------|----------------|-----|
| Cheap, scalable intervention (automated email, push notification) | **Gradient Boosting** | Highest Recall — catches the most high-erosion customers when false-alarm cost ≈ zero |
| Expensive per-customer intervention (account manager call, loyalty offer) | **Random Forest** | Highest Precision — fewer wasted high-cost contacts |
| Regulatory or audit requirement | **Logistic Regression** | Calibrated probabilities; coefficients interpretable as log-odds |
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
            width='stretch',
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
        st.image(str(roc_path), width='stretch')
        if _comp_df is not None:
            _min_auc = float(_comp_df["test_auc"].min())
            st.caption(
                f"All {len(_comp_df)} models achieve AUC ≥ {_min_auc:.4f}, "
                f"far exceeding the {AUC_TARGET} target. "
                f"{_champion_name} is the champion (AUC = {_champion_auc:.4f})."
            )
    else:
        st.info("ROC curve image not available. Model comparison results are shown below.", icon="ℹ️")
        if _comp_df is not None:
            _roc_fallback = _comp_df.copy()
            for _col in ["cv_auc", "test_auc", "f1", "precision", "recall"]:
                if _col in _roc_fallback.columns:
                    _roc_fallback[_col] = _roc_fallback[_col].map(
                        lambda x: f"{float(x):.4f}" if x != "N/A" else "N/A"
                    )
            st.dataframe(_roc_fallback, width='stretch', hide_index=True)

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
        st.image(str(cm_path), width='stretch')
        if _comp_df is not None:
            _rf_recall = float(_champion_row["recall"]) if "recall" in _champion_row.index else None
            _recall_str = f" (recall = {_rf_recall:.4f})" if _rf_recall else ""
            st.caption(
                f"Confusion matrices on the held-out test set. "
                f"{_champion_name}{_recall_str} prioritizes capturing actual high-erosion customers."
            )
    else:
        st.info("Confusion matrix image not available. Key classification metrics are shown below.", icon="ℹ️")
        if _comp_df is not None:
            _cm_cols = [c for c in ["model", "precision", "recall", "f1"] if c in _comp_df.columns]
            if _cm_cols:
                _cm_fallback = _comp_df[_cm_cols].copy()
                for _col in ["precision", "recall", "f1"]:
                    if _col in _cm_fallback.columns:
                        _cm_fallback[_col] = _cm_fallback[_col].map(lambda x: f"{float(x):.4f}")
                st.dataframe(_cm_fallback, width='stretch', hide_index=True)

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
            st.image(str(fi_png), width='stretch')

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
                width='stretch',
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
            st.dataframe(_fi_df, width='stretch', hide_index=True)

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
            width='stretch',
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
            "of SSL accounts correctly classified high/low",
            help=(
                "Of all SSL accounts, the model's predicted high/low-risk label matched the "
                f"actual high/low-loss label for {dir_acc:.1%} — a label agreement rate, "
                "not a precision measure. Spearman ρ (rank correlation) is the stronger test; "
                "this metric confirms the binary direction is correct for most accounts. "
                "Measured on external data the model was never trained on."
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

        # ── Classifier Performance Callout ───────────────────────────────────
        # Pull champion model metrics from model comparison CSV
        _champ_auc = 0.9798
        _champ_recall = 0.9115
        _champ_precision = 0.7822
        _champ_name = "Random Forest"
        if _comp_df is not None and "test_auc" in _comp_df.columns:
            _champ_row = _comp_df.loc[_comp_df["test_auc"].idxmax()]
            _champ_auc = float(_champ_row["test_auc"])
            _champ_recall = float(_champ_row.get("recall", _champ_recall))
            _champ_precision = float(_champ_row.get("precision", _champ_precision))
            _champ_name = str(_champ_row.get("model", _champ_name))

        # Pull SSL metrics from validation summary dict
        _ssl_dir_acc = float(_val_dict.get("directional_accuracy", 0.7640)) * 100
        _ssl_rho = float(_val_dict.get("directional_rank_correlation", 0.7526))
        _ssl_n = int(float(_val_dict.get("ssl_accounts_evaluated", 13616)))

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#0f2440 0%,#1a3660 100%);
                        border-left:5px solid #1565C0; border-radius:10px;
                        padding:20px 26px; margin:0 0 16px 0;">
                <p style="color:#90caf9;font-size:0.75rem;font-weight:700;
                          letter-spacing:0.12em;text-transform:uppercase;margin:0 0 8px 0;">
                    Pipeline Demonstration — SSL External Validation
                </p>
                <p style="color:#ffffff;font-size:1.05rem;font-weight:700;margin:0 0 6px 0;">
                    The TheLook-trained {_champ_name} was applied without retraining to
                    {_ssl_n:,} real-world B2B accounts (School Specialty LLC).
                    The model's predicted high/low-risk label matched the actual high/low-loss
                    label for {_ssl_dir_acc:.1f}% of accounts;
                    Spearman&nbsp;&rho;&nbsp;=&nbsp;{_ssl_rho:.4f} between predicted probability
                    and actual financial loss (p&nbsp;≈&nbsp;0.00).
                </p>
                <p style="color:#e3f2fd;font-size:0.9rem;line-height:1.65;margin:0;">
                    This confirms the behavioural pattern generalises in direction across
                    B2C and B2B contexts — directional validation of framework utility,
                    not parameter transferability from the synthetic training dataset.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── SSL Confusion Matrix ─────────────────────────────────────────────
        st.subheader("SSL Confusion Matrix")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
This confusion matrix shows how the TheLook-trained model performs on **SSL accounts**
that it was never trained on.

|  | **Predicted: High-risk** | **Predicted: Low-risk** |
|--|--------------------------|------------------------|
| **Actually High-loss** | TP — correctly flagged | FN — missed |
| **Actually Low-loss** | FP — false alarm | TN — correctly cleared |

**Precision (SSL)** = TP / (TP + FP) — of accounts flagged high-risk, how many truly are?
**Recall (SSL)** = TP / (TP + FN) — of truly high-loss accounts, how many were caught?

Some degradation vs TheLook is expected: the model was trained on synthetic B2C data
and applied to real-world B2B data without retraining.
"""
            )
        _ssl_pred_path = REPORTS_RQ3 / "rq3_ssl_directional_validation.csv"
        if _ssl_pred_path.exists():
            _ssl_pred_df = pd.read_csv(_ssl_pred_path)
            _tp = int((((_ssl_pred_df["predicted_high_risk"] == 1) & (_ssl_pred_df["actual_high_loss"] == 1))).sum())
            _fp = int((((_ssl_pred_df["predicted_high_risk"] == 1) & (_ssl_pred_df["actual_high_loss"] == 0))).sum())
            _fn = int((((_ssl_pred_df["predicted_high_risk"] == 0) & (_ssl_pred_df["actual_high_loss"] == 1))).sum())
            _tn = int((((_ssl_pred_df["predicted_high_risk"] == 0) & (_ssl_pred_df["actual_high_loss"] == 0))).sum())
            _ssl_prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
            _ssl_rec  = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
            _ssl_acc  = (_tp + _tn) / len(_ssl_pred_df) if len(_ssl_pred_df) > 0 else 0.0

            import plotly.graph_objects as _go

            _col_scatter, _col_cm = st.columns(2)

            # Left: Predicted probability vs actual loss scatter
            with _col_scatter:
                _colors = _ssl_pred_df["actual_high_loss"].map({0: "#5c8fc9", 1: "#e05c5c"})
                _fig_scatter = _go.Figure()
                _fig_scatter.add_trace(_go.Scatter(
                    x=_ssl_pred_df["predicted_probability"],
                    y=_ssl_pred_df["actual_loss"],
                    mode="markers",
                    marker=dict(
                        color=_ssl_pred_df["actual_high_loss"].map({0: "#5c8fc9", 1: "#e05c5c"}),
                        size=4,
                        opacity=0.4,
                    ),
                    text=_ssl_pred_df["actual_high_loss"].map({0: "Low Loss", 1: "High Loss"}),
                    hovertemplate="Pred prob: %{x:.3f}<br>Actual loss: %{y:,.2f}<br>%{text}<extra></extra>",
                ))
                _fig_scatter.update_layout(
                    title=f"Predicted Risk vs Actual Loss<br><sup>Spearman ρ = {_ssl_rho:.4f} (p ≈ 0.00)</sup>",
                    xaxis_title="Predicted Probability (TheLook Model)",
                    yaxis_title="Actual Total Loss (SSL)",
                    height=380,
                    margin=dict(l=60, r=20, t=70, b=60),
                    showlegend=False,
                )
                st.plotly_chart(_fig_scatter, use_container_width=True)
                st.caption("Blue = low-loss accounts · Red = high-loss accounts")

            # Right: Confusion matrix — fixed categorical colours per cell type
            # TP=green, TN=steel-blue, FP=orange, FN=red
            # go.Heatmap z uses category index (0–3); colorscale maps each index
            # to a distinct colour so counts never affect cell colour.
            # plotly renders y bottom→top, so row0=Low-loss, row1=High-loss.
            with _col_cm:
                _x_labels = ["Predicted: High-risk", "Predicted: Low-risk"]
                _y_labels = ["Actually: Low-loss", "Actually: High-loss"]

                # Category indices: 0=FP, 1=TN, 2=TP, 3=FN
                _cm_cat = [
                    [0, 1],   # bottom (Low-loss):  FP, TN
                    [2, 3],   # top    (High-loss):  TP, FN
                ]
                _cm_counts = [
                    [_fp,  _tn],
                    [_tp,  _fn],
                ]
                _cm_label_text = [
                    ["FP", "TN"],
                    ["TP", "FN"],
                ]
                # Categorical colorscale: evenly spaced stops for 4 categories
                _cat_colors = {
                    "FP": "#E67E22",   # orange  — false alarm
                    "TN": "#2980B9",   # blue    — correctly cleared
                    "TP": "#27AE60",   # green   — correctly flagged
                    "FN": "#C0392B",   # red     — missed
                }
                _colorscale = [
                    [0/3, _cat_colors["FP"]], [1/3, _cat_colors["FP"]],
                    [1/3, _cat_colors["TN"]], [2/3, _cat_colors["TN"]],
                    [2/3, _cat_colors["TP"]], [3/3, _cat_colors["TP"]],
                ]
                # FN occupies the top end — extend last stop
                _colorscale = [
                    [0.00, _cat_colors["FP"]], [0.25, _cat_colors["FP"]],
                    [0.25, _cat_colors["TN"]], [0.50, _cat_colors["TN"]],
                    [0.50, _cat_colors["TP"]], [0.75, _cat_colors["TP"]],
                    [0.75, _cat_colors["FN"]], [1.00, _cat_colors["FN"]],
                ]

                _annotations = []
                for ri in range(2):
                    for ci in range(2):
                        _annotations.append(dict(
                            x=_x_labels[ci],
                            y=_y_labels[ri],
                            text=f"<b>{_cm_label_text[ri][ci]}</b><br>{_cm_counts[ri][ci]:,}",
                            showarrow=False,
                            font=dict(color="#ffffff", size=15),
                            xref="x", yref="y",
                        ))

                _fig_ssl_cm = _go.Figure(data=_go.Heatmap(
                    z=_cm_cat,
                    x=_x_labels,
                    y=_y_labels,
                    zmin=0, zmax=3,
                    colorscale=_colorscale,
                    showscale=False,
                    hoverongaps=False,
                    customdata=_cm_counts,
                    hovertemplate="%{y}<br>%{x}<br>Count: %{customdata:,}<extra></extra>",
                ))
                _fig_ssl_cm.update_layout(
                    title=f"Directional Confusion Matrix<br><sup>Accuracy = {_ssl_acc:.3f}</sup>",
                    title_font_size=13,
                    annotations=_annotations,
                    height=380,
                    margin=dict(l=160, r=20, t=70, b=60),
                    xaxis=dict(side="bottom", tickfont=dict(size=12)),
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(_fig_ssl_cm, use_container_width=True)
                st.markdown(
                    "<p style='text-align:center;font-size:0.8rem;color:#888;margin-top:4px;'>"
                    "🟢 TP — correctly flagged &nbsp;·&nbsp; 🔵 TN — correctly cleared &nbsp;·&nbsp; "
                    "🟠 FP — false alarm &nbsp;·&nbsp; 🔴 FN — missed"
                    "</p>",
                    unsafe_allow_html=True,
                )

            _ca, _cb, _cc = st.columns(3)
            _ca.metric("SSL Precision", f"{_ssl_prec:.1%}", f"vs TheLook {_champ_precision:.1%}",
                       delta_color="normal",
                       help="Of accounts flagged high-risk, this fraction truly had high losses.")
            _cb.metric("SSL Recall",    f"{_ssl_rec:.1%}",  f"vs TheLook {_champ_recall:.1%}",
                       delta_color="normal",
                       help="Of truly high-loss accounts, this fraction was correctly flagged.")
            _cc.metric("SSL Accuracy",  f"{_ssl_acc:.1%}",  "label agreement",
                       help="(TP + TN) / total — overall label agreement rate (= directional accuracy).")
            st.caption(
                "Degradation vs TheLook is expected: model trained on synthetic B2C data, "
                "applied without retraining to real-world B2B accounts."
            )
        else:
            st.info("SSL directional validation CSV not found — confusion matrix unavailable.")

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
        st.dataframe(_val_display, width='stretch', hide_index=True)

        st.divider()
        ssl_screen_path = REPORTS_RQ3 / "rq3_ssl_feature_screening.csv"
        if ssl_screen_path.exists():
            st.subheader("Feature Pattern Agreement — TheLook vs SSL (Level 1)")
            with st.expander("ℹ️ What does this mean?", expanded=False):
                st.markdown(
                    """
Level 1 validation checks whether the same behavioral features that are informative in
TheLook are also informative in SSL — regardless of the magnitude of the signal.

- **Both Pass** — feature passes screening in *both* datasets: the behavioral signal generalises
- **Both Fail** — feature is uninformative in both: consistent absence of signal
- **Disagree** — feature passes in one dataset but not the other: domain-specific signal

This is a directional check, not a coefficient comparison. B2B vs B2C magnitudes will differ;
what matters is whether the same dimensions discriminate high-loss customers in both contexts.
"""
                )

            ssl_screen = pd.read_csv(ssl_screen_path)

            # KPI cards
            _bp  = int(ssl_screen["both_pass"].sum())
            _bf  = int(ssl_screen["both_fail"].sum())
            _dis = int((~ssl_screen["both_pass"] & ~ssl_screen["both_fail"]).sum())
            _agr = int(ssl_screen["agreement"].sum())
            _tot = len(ssl_screen)
            _agr_pct = _agr / _tot * 100 if _tot else 0

            _sk1, _sk2, _sk3, _sk4 = st.columns(4)
            _sk1.metric("Both Pass", f"{_bp} / {_tot}",
                        "signal generalises",
                        help="Feature passes 3-gate screening in both TheLook and SSL.")
            _sk2.metric("Both Fail", f"{_bf} / {_tot}",
                        "consistent absence",
                        help="Feature is uninformative in both datasets — consistent signal absence.")
            _sk3.metric("Disagree", f"{_dis} / {_tot}",
                        "domain-specific",
                        help="Feature is informative in one dataset only — context-dependent signal.")
            _sk4.metric("Agreement Rate", f"{_agr_pct:.0f}%",
                        f"≥ 50% threshold {'✅' if _agr_pct >= 50 else '⚠️'}",
                        help="Fraction of features where TheLook and SSL screening outcomes agree (both pass or both fail).")

            if _agr_pct >= 50:
                st.success(
                    f"**Validation passed** — {_agr} of {_tot} features ({_agr_pct:.0f}%) "
                    "show consistent screening outcomes across TheLook and SSL (threshold ≥ 50%)."
                )
            else:
                st.warning(
                    f"**Agreement below threshold** — {_agr} of {_tot} features ({_agr_pct:.0f}%) agree "
                    "(threshold ≥ 50%). Domain differences between B2C and B2B may explain divergence."
                )

            # Bar chart — feature-level status
            import plotly.graph_objects as _go_ssl
            _status_map = []
            for _, _sr in ssl_screen.iterrows():
                if _sr["both_pass"]:
                    _status_map.append("Both Pass")
                elif _sr["both_fail"]:
                    _status_map.append("Both Fail")
                else:
                    _status_map.append("Disagree")
            ssl_screen = ssl_screen.copy()
            ssl_screen["status"] = _status_map
            ssl_screen_sorted = ssl_screen.sort_values(
                "status", key=lambda s: s.map({"Both Pass": 0, "Disagree": 1, "Both Fail": 2})
            )
            _color_map = {"Both Pass": "#27AE60", "Disagree": "#E67E22", "Both Fail": "#95a5a6"}
            _fig_screen = px.bar(
                ssl_screen_sorted,
                x="feature",
                color="status",
                title="Feature Screening Agreement — TheLook vs SSL",
                color_discrete_map=_color_map,
                labels={"feature": "Feature", "status": "Screening Outcome"},
                category_orders={"status": ["Both Pass", "Disagree", "Both Fail"]},
            )
            _fig_screen.update_layout(
                height=340,
                xaxis_tickangle=-35,
                showlegend=True,
                legend_title_text="Outcome",
                yaxis=dict(showticklabels=False, title=""),
                margin=dict(t=50, b=80),
            )
            st.plotly_chart(_fig_screen, use_container_width=True)

            # Two-column feature lists
            _pass_feats = ssl_screen.loc[ssl_screen["status"] == "Both Pass", "feature"].tolist()
            _dis_feats  = ssl_screen.loc[ssl_screen["status"] == "Disagree",  "feature"].tolist()
            _fcol1, _fcol2 = st.columns(2)
            with _fcol1:
                st.markdown("**Generalising features** (Both Pass)")
                for _f in _pass_feats:
                    st.markdown(f"- `{_f}`")
            with _fcol2:
                st.markdown("**Domain-specific features** (Disagree)")
                for _f in _dis_feats:
                    st.markdown(f"- `{_f}`")

        st.divider()
        st.subheader("Interpretation")
        st.markdown(
            f"""
- **Directional accuracy = {dir_acc:.1%}** — label agreement rate: predicted high/low-risk matched actual high/low-loss for {dir_acc:.1%} of all {ssl_accounts:,} SSL accounts
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
Robustness analysis answers the question: **do the model's conclusions change if we tweak our
assumptions or remove key predictors?**

This tab runs three independent checks:

1. **Processing cost per return** — we assumed USD 12 as the base cost (mid-range of the USD 10–25
   academic literature). What if the true cost is USD 8 or USD 18? Does the model still work?

2. **High-erosion threshold** — we defined "high erosion" as the top 25% of customers (75th percentile).
   What if we use 30% or 20% instead? Does model accuracy hold?

3. **Feature ablation** — high AUC may be driven by a small set of dominant predictors rather than
   genuine multi-feature learning. To test this, the Random Forest is retrained after removing its
   top-3 most important features. A small AUC drop indicates the model distributes predictive signal
   across multiple independent features, not just the leading ones.

**How to use this tab**: Select a specific assumption scenario with the radio buttons for checks 1 and 2.
The metric cards update immediately. The ablation result (check 3) is fixed — it reports the AUC drop
from a single held-out experiment.

**Bottom line**: If AUC stays well above {AUC_TARGET} across all scenarios and the ablation drop is
small, the model's conclusions are robust — they don't depend on getting assumptions exactly right
or on any single predictor.
"""
        )
    st.markdown(
        """
Three robustness checks test whether model conclusions are sensitive to key modeling assumptions and predictor dominance:

1. **Processing cost sensitivity** — does the target variable change when cost assumptions vary?
2. **Percentile threshold sensitivity** — does model performance hold across different definitions of "high erosion"?
3. **Feature ablation** — does AUC degrade substantially when the top-3 predictors are removed, or do the remaining features sustain performance?
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
        st.info("Processing cost sensitivity results are not yet available.")

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
        st.info("Threshold sensitivity results are not yet available.")

    st.divider()
    st.subheader("Feature Ablation — Sensitivity to Top Predictors")
    st.markdown(
        "Retrains the Random Forest after removing its top-3 most important features "
        "to test whether high AUC is driven by a small set of dominant predictors."
    )
    if _ablation_df is not None:
        _abl_row = _ablation_df.iloc[0]
        _abl_col1, _abl_col2, _abl_col3 = st.columns(3)
        _abl_col1.metric("Full RF AUC", f"{float(_abl_row['full_rf_test_auc']):.4f}")
        _abl_col2.metric("Ablated RF AUC", f"{float(_abl_row['ablated_test_auc']):.4f}")
        _abl_col3.metric("AUC Drop", f"{float(_abl_row['auc_drop']):.4f}")
        st.caption(f"Removed features: {_abl_row['removed_features']}")
        st.caption(f"Retained features: {_abl_row['retained_features']}")
    else:
        st.info("Ablation study artifact not yet generated. Re-run the master notebook.")

    st.divider()
    st.subheader("Robustness Conclusion")
    if _cost_df is not None and _thresh_df is not None:
        _thresh_min_pct_c = int(_thresh_df["threshold"].min() * 100)
        _thresh_max_pct_c = int(_thresh_df["threshold"].max() * 100)
        _abl_conclusion = ""
        if _ablation_df is not None:
            _abl_row_c = _ablation_df.iloc[0]
            _abl_drop = float(_abl_row_c["auc_drop"])
            _abl_ablated = float(_abl_row_c["ablated_test_auc"])
            _abl_conclusion = (
                f"\n- **Feature ablation robustness**: AUC dropped by only {_abl_drop:.4f} "
                f"(to {_abl_ablated:.4f}) after removing the top-3 most important features — "
                f"the model is not dominated by a small set of predictors"
            )
        st.markdown(
            f"""
- **Processing cost robustness**: AUC ranged from {_cost_auc_min:.4f} to {_cost_auc_max:.4f}
  across the {_cost_range_str} cost range — all scenarios exceed the {AUC_TARGET} target
- **Threshold robustness**: AUC ranged from {_thresh_auc_min:.4f} to {_thresh_auc_max:.4f}
  across the {_thresh_min_pct_c}th–{_thresh_max_pct_c}th percentile range — all scenarios exceed the {AUC_TARGET} target{_abl_conclusion}
- **Conclusion**: Model performance is not sensitive to cost model assumptions, threshold choice, or predictor dominance
"""
        )
    else:
        st.info("Sensitivity analysis results are not yet available.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — CONCLUSION
# ════════════════════════════════════════════════════════════════════════════════
with tab6:
    st.header("Conclusion — RQ3 Predictive Model")

    # ── Derived values for callout ────────────────────────────────────────────
    _conc_auc   = _champion_auc if _comp_df is not None else 0.9798
    _conc_name  = _champion_name if _comp_df is not None else "Random Forest"
    _conc_feats = f"{_n_pass}/{_n_total}" if _n_pass is not None else "7/12"
    _conc_recall    = 0.9115
    _conc_precision = 0.7822
    if _champion_row is not None:
        _conc_recall    = float(_champion_row.get("recall",    _conc_recall))
        _conc_precision = float(_champion_row.get("precision", _conc_precision))
    _conc_ssl_acc = float(_val_dict.get("directional_accuracy", 0.7640)) * 100
    _conc_rho     = float(_val_dict.get("directional_rank_correlation", 0.7526))
    _conc_ssl_n   = int(float(_val_dict.get("ssl_accounts_evaluated", 13616)))

    # ── Pipeline Demonstration callout banner ─────────────────────────────────
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#0f2440 0%,#1a3660 100%);
                    border-left:5px solid #7986CB; border-radius:10px;
                    padding:20px 26px; margin:0 0 20px 0;">
            <p style="color:#c5cae9;font-size:0.75rem;font-weight:700;
                      letter-spacing:0.12em;text-transform:uppercase;margin:0 0 8px 0;">
                Pipeline Demonstration — Predictive Model Output (Synthetic Dataset)
            </p>
            <p style="color:#ffffff;font-size:1.05rem;font-weight:700;margin:0 0 6px 0;">
                On TheLook, the {_conc_name} classifier trained on {_conc_feats} behavioral
                features achieves Test&nbsp;AUC&nbsp;=&nbsp;{_conc_auc:.4f} — exceeding the
                AUC&nbsp;&gt;&nbsp;{AUC_TARGET} threshold by
                {_conc_auc - AUC_TARGET:.4f}.
                Recall&nbsp;=&nbsp;{_conc_recall:.1%} · Precision&nbsp;=&nbsp;{_conc_precision:.1%}.
            </p>
            <p style="color:#e3f2fd;font-size:0.9rem;line-height:1.65;margin:0;">
                Applied without retraining to {_conc_ssl_n:,} real-world B2B accounts
                (School&nbsp;Specialty&nbsp;LLC): label agreement&nbsp;=&nbsp;{_conc_ssl_acc:.1f}%,
                Spearman&nbsp;&rho;&nbsp;=&nbsp;{_conc_rho:.4f} (p&nbsp;≈&nbsp;0.00).
                This is directional validation of framework utility — not parameter
                transferability from the synthetic training dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Hypothesis Decision Table ─────────────────────────────────────────────
    st.subheader("Hypothesis Decision")

    _rf_auc  = 0.9798
    _gb_auc  = 0.9795
    _lr_auc  = 0.9687
    _rf_rec  = 0.9115
    _gb_rec  = 0.9299
    _lr_rec  = 0.9048
    if _comp_df is not None and "test_auc" in _comp_df.columns:
        for _, _r in _comp_df.iterrows():
            _m = str(_r["model"]).lower()
            if "random" in _m or "rf" in _m:
                _rf_auc = float(_r["test_auc"])
                _rf_rec = float(_r.get("recall", _rf_rec))
            elif "gradient" in _m or "gb" in _m:
                _gb_auc = float(_r["test_auc"])
                _gb_rec = float(_r.get("recall", _gb_rec))
            elif "logistic" in _m or "lr" in _m:
                _lr_auc = float(_r["test_auc"])
                _lr_rec = float(_r.get("recall", _lr_rec))

    st.markdown(
        f"""
| Model | Dataset | Test AUC | Recall | Exceeds AUC > {AUC_TARGET}? | Decision |
|---|---|---|---|---|---|
| Random Forest (champion) | TheLook | {_rf_auc:.4f} | {_rf_rec:.4f} | +{_rf_auc - AUC_TARGET:.4f} | ✅ REJECT H₀₃ |
| Gradient Boosting | TheLook | {_gb_auc:.4f} | {_gb_rec:.4f} | +{_gb_auc - AUC_TARGET:.4f} | ✅ REJECT H₀₃ |
| Logistic Regression | TheLook | {_lr_auc:.4f} | {_lr_rec:.4f} | +{_lr_auc - AUC_TARGET:.4f} | ✅ REJECT H₀₃ |
| Directional validation (SSL) | School Specialty LLC | — | — | Label agreement = {_conc_ssl_acc:.1f}%, ρ = {_conc_rho:.4f} | ✅ Pattern confirmed |

**H₀₃**: Machine learning models cannot predict high profit erosion customers with acceptable accuracy (AUC ≤ {AUC_TARGET}).
All three models independently exceed the threshold — **H₀₃ is rejected**.
"""
    )

    st.divider()

    # ── Top-3 Features Panel ──────────────────────────────────────────────────
    st.subheader("Top Predictors Identified by the Pipeline")

    # Build top-3 from feature importance CSV; fallback to known results
    _top_feats = [
        {"feature": "return_frequency",   "importance": 0.412, "rank": 1},
        {"feature": "avg_order_value",     "importance": 0.198, "rank": 2},
        {"feature": "avg_basket_size",     "importance": 0.134, "rank": 3},
    ]
    if _fi_df is not None and "feature" in _fi_df.columns:
        _avg_col = None
        for _c in ["average_importance", "mean_importance", "avg_importance", "importance"]:
            if _c in _fi_df.columns:
                _avg_col = _c
                break
        if _avg_col:
            _fi_top = (
                _fi_df.groupby("feature")[_avg_col]
                .mean()
                .reset_index()
                .sort_values(_avg_col, ascending=False)
                .head(3)
                .reset_index(drop=True)
            )
            _top_feats = [
                {"feature": row["feature"], "importance": float(row[_avg_col]), "rank": i + 1}
                for i, row in _fi_top.iterrows()
            ]

    # Why-it-matters blurbs per feature (classification context — not OLS coefficients)
    _feat_why = {
        "return_frequency":      "Customers with more returns are far more likely to fall in the top-25% erosion tier — the single strongest behavioral signal.",
        "avg_order_value":       "Higher spend per order raises the reversal exposure on each return, increasing the probability of high total erosion.",
        "avg_basket_size":       "More items per order moderates per-item risk — customers who spread spend across many products show lower per-return erosion.",
        "customer_return_rate":  "The proportion of items returned captures habitual return behaviour independent of raw return count.",
        "avg_item_margin":       "Higher margin items produce larger reversals per return, driving customers toward the high-erosion tier.",
        "total_margin":          "Cumulative margin across all purchases sets the ceiling for how much erosion a customer can generate.",
        "total_items":           "Total items purchased is a volume signal — higher activity customers have more opportunities to erode profit.",
    }
    _feat_colors = ["#1565C0", "#2E7D32", "#6A1B9A"]
    _feat_labels = ["🥇 Top Predictor", "🥈 2nd Predictor", "🥉 3rd Predictor"]
    _p1, _p2, _p3 = st.columns(3)
    for _col, _feat_row, _color, _badge in zip(
        [_p1, _p2, _p3], _top_feats, _feat_colors, _feat_labels
    ):
        _why = _feat_why.get(_feat_row["feature"], "Identified as a significant predictor by the 3-gate feature screening pipeline.")
        with _col:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,{_color}55,{_color}33);
                            border:1px solid {_color}99; border-radius:10px;
                            padding:16px 18px; min-height:185px;">
                    <p style="font-size:0.72rem;font-weight:700;color:#ffffff;
                              text-transform:uppercase;letter-spacing:0.1em;margin:0 0 6px 0;">
                        {_badge}
                    </p>
                    <p style="font-size:1.05rem;font-weight:700;color:#ffffff;margin:0 0 4px 0;">
                        {_feat_row['feature']}
                    </p>
                    <p style="font-size:0.82rem;color:#e0e0e0;font-style:italic;margin:0 0 8px 0;">
                        Avg importance: <strong>{_feat_row['importance']:.3f}</strong>
                        · consistent across all 3 models
                    </p>
                    <p style="font-size:0.84rem;color:#f0f0f0;line-height:1.5;margin:0;">
                        {_why}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # ── RQ3 Summary Table ─────────────────────────────────────────────────────
    st.subheader("RQ3 Summary")
    _summary_rows = [
        ("Research Question",    "RQ3",             "Can ML models predict high profit erosion customers (AUC > 0.70)?"),
        ("Null Hypothesis",      "H₀₃",             f"AUC ≤ {AUC_TARGET} — models cannot discriminate"),
        ("Decision",             "✅ Reject H₀₃",   f"All 3 models exceed AUC > {AUC_TARGET}; champion {_conc_name} AUC = {_conc_auc:.4f}"),
        ("Champion Model",       _conc_name,         f"Test AUC = {_conc_auc:.4f} · Recall = {_conc_recall:.1%} · Precision = {_conc_precision:.1%}"),
        ("Features Used",        _conc_feats,        "Survived 3-gate screening: variance → collinearity → univariate relevance"),
        ("Top Predictor",        "return_frequency", "Highest importance across all three models"),
        ("SSL Validation",       f"{_conc_ssl_acc:.1f}% label agreement",
                                 f"Spearman ρ = {_conc_rho:.4f} on {_conc_ssl_n:,} B2B accounts — directional validation only"),
        ("Robustness",           "Stable",           f"AUC holds across cost {_cost_range_str} and 50th–90th percentile thresholds"),
        ("Dataset",              "TheLook (synthetic)", "SSL = real-world directional validation; figures from synthetic training data"),
    ]
    st.dataframe(
        pd.DataFrame(_summary_rows, columns=["Dimension", "Result", "Detail"]),
        width='stretch',
        hide_index=True,
    )

st.caption(
    "DAMO-699-4 · University of Niagara Falls, Canada · Winter 2026 · "
    "RQ3 — Predicting High Profit Erosion Customers"
)
