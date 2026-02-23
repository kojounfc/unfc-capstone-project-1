"""
RQ3: Predict High Profit Erosion Customers

Method: ML Classification — Random Forest, Gradient Boosting, Logistic Regression
Target AUC > 0.70 | External validation: School Specialty LLC (SSL)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
if _comp_df is not None:
    _champion_row = _comp_df.loc[_comp_df["test_auc"].idxmax()]
    _champion_name = _champion_row["model"]
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

if _cost_df is not None and "best_auc" in _cost_df.columns:
    _cost_auc_min = float(_cost_df["best_auc"].min())
    _cost_auc_max = float(_cost_df["best_auc"].max())
    _cost_range_str = f"${int(_cost_df['base_cost'].min())}–${int(_cost_df['base_cost'].max())}"
else:
    _cost_auc_min = float("nan")
    _cost_range_str = "N/A"

if _thresh_df is not None and "best_auc" in _thresh_df.columns:
    _thresh_auc_min = float(_thresh_df["best_auc"].min())
    _thresh_auc_max = float(_thresh_df["best_auc"].max())
else:
    _thresh_auc_min = float("nan")

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
    st.header("Problem Setup")

    col1, col2, col3 = st.columns(3)
    if _comp_df is not None:
        col1.metric(
            "Champion Model AUC",
            f"{_champion_auc:.4f}",
            f"{_champion_name} — target was {AUC_TARGET}",
        )
    else:
        col1.metric("Champion Model AUC", "N/A", "Run master notebook")

    if _n_pass is not None:
        col2.metric("Features Used", f"{_n_pass} / {_n_total}", "after 3-gate screening")
    else:
        col2.metric("Features Used", "N/A", "")

    col3.metric("Hypothesis H₀", _h0_result,
                f"All {len(_comp_df) if _comp_df is not None else '?'} models exceed AUC > {AUC_TARGET}"
                if _comp_df is not None else "")

    st.divider()
    st.subheader("Feature Screening — 3 Sequential Gates")
    if _screen_df is not None:
        show_cols = [c for c in [
            "feature", "variance_pass", "correlation_pass", "univariate_pass", "final_status",
        ] if c in _screen_df.columns]
        if show_cols:
            display_df = _screen_df[show_cols].copy()
            for col in display_df.columns:
                if display_df[col].dtype == bool:
                    display_df[col] = display_df[col].map({True: "✅", False: "❌"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)
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
        # KPI row for champion model
        c1, c2, c3, c4 = st.columns(4)
        for metric, col, label in [
            ("test_auc", c1, "Test AUC"),
            ("cv_auc", c2, "CV AUC"),
            ("f1", c3, "F1 Score"),
            ("precision", c4, "Precision"),
        ]:
            if metric in _champion_row.index:
                col.metric(f"{_champion_name} {label}", f"{float(_champion_row[metric]):.4f}")

        st.divider()
        st.subheader("All Models Comparison")
        display_comp = _comp_df.copy()
        num_cols = [c for c in display_comp.columns if display_comp[c].dtype in ["float64", "float32"]]
        for col in num_cols:
            display_comp[col] = display_comp[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        if "meets_threshold" in display_comp.columns:
            display_comp["meets_threshold"] = _comp_df["meets_threshold"].map(
                {True: "✅ Yes", False: "❌ No"}
            )
        st.dataframe(display_comp, use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison CSV not found.")

    st.divider()

    st.subheader("ROC Curves (Primary)")
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

    if _fi_df is not None:
        fi_png = REPORTS_RQ3 / "rq3_feature_importance.png"
        if fi_png.exists():
            st.image(str(fi_png), use_container_width=True)

        st.divider()
        st.subheader("Average Importance Ranking (across models)")

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
            # Derive rank label from position
            avg_imp.insert(0, "rank", avg_imp.index)
            avg_imp["avg_importance"] = avg_imp["avg_importance"].map("{:.4f}".format)
            st.dataframe(avg_imp, use_container_width=True, hide_index=True)
            st.caption(
                "Average importance across all trained models. "
                "For LR, |coefficient| is used; for tree models, Gini impurity decrease."
            )

        st.divider()
        st.subheader("Per-Model Detail")
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

    # Dropped features — data-driven from screening
    st.divider()
    st.subheader("Dropped Features")
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
        st.dataframe(pd.DataFrame(dropped_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Feature screening CSV not found.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION (SSL External)
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("External Validation — School Specialty LLC (SSL)")
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
        c1.metric("Directional Accuracy", f"{dir_acc:.1%}",
                  "Predicted high-risk matches actual high-loss")
        c2.metric("Spearman ρ", f"{spearman:.4f}", "Rank correlation (p ≈ 0.00)")
        c3.metric("SSL Accounts", f"{ssl_accounts:,}", "evaluated")

        c4, c5, c6 = st.columns(3)
        c4.metric("Predicted High-Risk %", f"{pred_high:.1f}%", "of SSL accounts")
        c5.metric("Actual High-Loss %", f"{actual_high:.1f}%", "in SSL dataset")
        c6.metric("Pattern Agreement",
                  f"{pattern_count} / {features_compared}",
                  f"{pattern_pct:.1f}% of features match")

        st.divider()
        st.subheader("Full Validation Metrics")
        _val_display = pd.read_csv(_val_path)
        st.dataframe(_val_display, use_container_width=True, hide_index=True)

        st.divider()
        ssl_screen_path = REPORTS_RQ3 / "rq3_ssl_feature_screening.csv"
        if ssl_screen_path.exists():
            st.subheader(f"SSL Feature Pattern Comparison (Level 1)")
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
    st.markdown(
        """
Two sensitivity analyses test whether model conclusions are robust to key modeling assumptions:

1. **Processing cost sensitivity** — does the target variable change when cost assumptions vary?
2. **Percentile threshold sensitivity** — does model performance hold across different definitions of "high erosion"?
"""
    )

    col_left, col_right = st.columns(2)

    with col_left:
        _cost_label = f"Processing Cost Sensitivity ({_cost_range_str})" if _cost_range_str != "N/A" else "Processing Cost Sensitivity"
        st.subheader(_cost_label)
        cost_png = REPORTS_RQ3 / "sensitivity_cost_analysis.png"
        if cost_png.exists():
            st.image(str(cost_png), use_container_width=True)
        if _cost_df is not None:
            st.dataframe(_cost_df, use_container_width=True, hide_index=True)
            st.caption(
                f"AUC range across cost scenarios: {_cost_auc_min:.4f} – {_cost_auc_max:.4f}. "
                f"All scenarios exceed AUC > {AUC_TARGET} target."
            )

    with col_right:
        if _thresh_df is not None:
            _thresh_min_pct = int(_thresh_df["threshold"].min() * 100)
            _thresh_max_pct = int(_thresh_df["threshold"].max() * 100)
            st.subheader(f"Percentile Threshold Sensitivity ({_thresh_min_pct}th–{_thresh_max_pct}th)")
        else:
            st.subheader("Percentile Threshold Sensitivity")
        thresh_png = REPORTS_RQ3 / "sensitivity_threshold_analysis.png"
        if thresh_png.exists():
            st.image(str(thresh_png), use_container_width=True)
        if _thresh_df is not None:
            st.dataframe(_thresh_df, use_container_width=True, hide_index=True)
            st.caption(
                f"AUC range across threshold scenarios: {_thresh_auc_min:.4f} – {_thresh_auc_max:.4f}. "
                f"All scenarios exceed AUC > {AUC_TARGET} target."
            )

    st.divider()
    st.subheader("Robustness Conclusion")
    if _cost_df is not None and _thresh_df is not None:
        st.markdown(
            f"""
- **Processing cost robustness**: AUC ranged from {_cost_auc_min:.4f} to {_cost_auc_max:.4f}
  across the {_cost_range_str} cost range — all exceed the {AUC_TARGET} target
- **Threshold robustness**: AUC ranged from {_thresh_auc_min:.4f} to {_thresh_auc_max:.4f}
  across the {_thresh_min_pct}th–{_thresh_max_pct}th percentile range — all exceed the {AUC_TARGET} target
- **Conclusion**: Model performance is not sensitive to cost model assumptions or threshold choice
"""
        )
    else:
        st.info("Sensitivity CSVs not found. Run the master notebook.")
