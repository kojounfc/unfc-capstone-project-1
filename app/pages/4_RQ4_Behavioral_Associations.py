"""
RQ4: Marginal Associations Between Behaviors and Profit Erosion

Method: Log-Linear OLS Regression
Model: log(total_profit_erosion) ~ behavioral + category + demographics
External validation: School Specialty LLC (SSL)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="RQ4 – Behavioral Associations",
    page_icon="📐",
    layout="wide",
)

ROOT = Path(__file__).parent.parent.parent
REPORTS_RQ4 = ROOT / "reports" / "rq4"
FIGURES_RQ4 = ROOT / "figures" / "rq4"

# ── Load data once ────────────────────────────────────────────────────────────
_coef_df = None
_align_df = None
_val_dict = {}

_coef_path = REPORTS_RQ4 / "rq4v2_thelook_coefficients.csv"
if _coef_path.exists():
    _coef_df = pd.read_csv(_coef_path)

_align_path = REPORTS_RQ4 / "rq4v2_ssl_coefficient_alignment.csv"
if _align_path.exists():
    _align_df = pd.read_csv(_align_path)

_val_path = REPORTS_RQ4 / "rq4v2_validation_summary.csv"
if _val_path.exists():
    _val_raw = pd.read_csv(_val_path)
    if _val_raw.shape[1] == 2:
        _val_dict = dict(zip(_val_raw.iloc[:, 0], _val_raw.iloc[:, 1]))

# ── Derived values ────────────────────────────────────────────────────────────
_thelook_r2 = float(_val_dict.get("thelook_r_squared", float("nan")))
_ssl_r2 = float(_val_dict.get("ssl_r_squared", float("nan")))
_r2_ratio = float(_val_dict.get("r_squared_ratio_ssl_to_thelook", float("nan")))
_thelook_n = int(float(_val_dict.get("thelook_nobs", 0))) if _val_dict.get("thelook_nobs") else None
_ssl_n = int(float(_val_dict.get("ssl_accounts_validated", 0))) if _val_dict.get("ssl_accounts_validated") else None
_gen_score = float(_val_dict.get("generalization_score", float("nan")))
_n_hyp = int(float(_val_dict.get("n_hypothesis_predictors", 0))) if _val_dict.get("n_hypothesis_predictors") else None
_dir_aligned = int(float(_val_dict.get("direction_aligned_count", 0))) if _val_dict.get("direction_aligned_count") else None
_dir_pct = float(_val_dict.get("direction_aligned_pct", float("nan")))
_sig_agree = int(float(_val_dict.get("significance_agreement_count", 0))) if _val_dict.get("significance_agreement_count") else None

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📐 RQ4: Behavioral Associations with Profit Erosion")
st.markdown(
    """
**Research Question**: What is the marginal association between customer behavioral features
and profit erosion, holding other factors constant?

**Method**: Log-Linear OLS Regression — `log(total_profit_erosion) ~ behaviors + categories + demographics`

Log-linear specification chosen because profit erosion is right-skewed; log transform achieves
normality. Coefficients are interpreted as **% change in profit erosion** per unit change in predictor.
"""
)
st.divider()

# ── KPI cards (data-driven) ───────────────────────────────────────────────────
st.header("Model Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "R² (TheLook)",
    f"{_thelook_r2:.4f}" if not pd.isna(_thelook_r2) else "N/A",
    f"{_thelook_r2*100:.1f}% of log-erosion variance explained" if not pd.isna(_thelook_r2) else "",
)
col2.metric(
    "R² (SSL)",
    f"{_ssl_r2:.4f}" if not pd.isna(_ssl_r2) else "N/A",
    f"R² ratio = {_r2_ratio:.2f}" if not pd.isna(_r2_ratio) else "",
)
col3.metric(
    "Observations (TheLook)",
    f"{_thelook_n:,}" if _thelook_n else "N/A",
    "customers with ≥1 return",
)
col4.metric(
    "SSL Accounts Validated",
    f"{_ssl_n:,}" if _ssl_n else "N/A",
    "",
)

st.divider()

# ── Visualizations ────────────────────────────────────────────────────────────
st.header("Visualizations")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Target Distribution (Log-Transformed)")
    target_path = FIGURES_RQ4 / "rq4_target_distribution.png"
    if target_path.exists():
        st.image(str(target_path), use_container_width=True)
        st.caption(
            "Log transformation of total_profit_erosion. "
            "Near-normal distribution validates OLS assumptions."
        )
    else:
        st.warning("Target distribution figure not found.")

with col_b:
    st.subheader("Coefficient Forest Plot")
    if _coef_df is not None and "coefficient" in _coef_df.columns:
        # Build interactive forest plot (significant features, p < 0.05, by default)
        _sig_only = st.checkbox("Show significant predictors only (p < 0.05)", value=True)
        _forest_df = _coef_df.copy()
        if _sig_only and "p_value" in _forest_df.columns:
            _forest_df = _forest_df[
                pd.to_numeric(_forest_df["p_value"], errors="coerce") < 0.05
            ]
        _forest_df = (
            _forest_df
            .assign(_abs_coef=_forest_df["coefficient"].abs())
            .sort_values("_abs_coef", ascending=True)
        )
        _colors = [
            "#EF5350" if c > 0 else "#42A5F5"
            for c in _forest_df["coefficient"]
        ]
        fig_forest = go.Figure()
        fig_forest.add_trace(go.Scatter(
            x=_forest_df["coefficient"],
            y=_forest_df["feature"],
            mode="markers",
            marker=dict(size=9, color=_colors),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(
                    (_forest_df["ci_upper"] - _forest_df["coefficient"]).tolist()
                    if "ci_upper" in _forest_df.columns else None
                ),
                arrayminus=(
                    (_forest_df["coefficient"] - _forest_df["ci_lower"]).tolist()
                    if "ci_lower" in _forest_df.columns else None
                ),
                visible=True,
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coefficient: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        ))
        fig_forest.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        fig_forest.update_layout(
            title="OLS Coefficient Forest Plot (Log-Linear)",
            xaxis_title="Coefficient (log-scale — % effect on profit erosion)",
            yaxis_title="",
            height=max(400, len(_forest_df) * 22),
        )
        st.plotly_chart(fig_forest, use_container_width=True)
        st.caption(
            "Red = positive association (more erosion); blue = negative (less erosion). "
            "Error bars show 95% CIs. Points right of zero → higher profit erosion."
        )
    else:
        coef_path = FIGURES_RQ4 / "rq4_coefficient_plot.png"
        if coef_path.exists():
            st.image(str(coef_path), use_container_width=True)
            st.caption(
                "OLS coefficients with 95% CIs. Points to the right = positive association "
                "(more erosion); left = negative (less erosion)."
            )
        else:
            st.warning("Coefficient plot not found.")

st.divider()

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Residual Diagnostics")
    resid_path = FIGURES_RQ4 / "rq4_residual_diagnostics.png"
    if resid_path.exists():
        st.image(str(resid_path), use_container_width=True)
        st.caption("Residual plots validate OLS assumptions: linearity, homoscedasticity, normality.")
    else:
        st.warning("Residual diagnostics figure not found.")

with col_d:
    st.subheader("QQ Plot — Normality Check")
    qq_path = FIGURES_RQ4 / "rq4_qq_plot_comparison.png"
    if qq_path.exists():
        st.image(str(qq_path), use_container_width=True)
        st.caption(
            "QQ plot compares residual quantiles to the normal distribution. "
            "Points on diagonal = normality assumption satisfied."
        )
    else:
        st.warning("QQ plot figure not found.")

st.divider()

st.subheader("TheLook vs SSL Coefficient Comparison")
forest_path = FIGURES_RQ4 / "rq4_ssl_forest_comparison.png"
if forest_path.exists():
    st.image(str(forest_path), use_container_width=True)
    st.caption(
        "Side-by-side forest plot comparing OLS coefficients for hypothesis predictors "
        "across TheLook (B2C) and SSL (B2B) datasets."
    )
else:
    st.warning("SSL forest comparison figure not found.")

st.divider()

# ── Coefficient table (interactive) ──────────────────────────────────────────
st.header("OLS Coefficient Table (TheLook)")

if _coef_df is not None and "coefficient" in _coef_df.columns:
    st.subheader("Interactive Coefficient Explorer")
    n_features = st.slider(
        "Show top N features by |coefficient|", 5, len(_coef_df), min(15, len(_coef_df))
    )
    top_coef = (
        _coef_df.assign(abs_coef=_coef_df["coefficient"].abs())
        .nlargest(n_features, "abs_coef")
        .sort_values("coefficient")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=top_coef["coefficient"],
            y=top_coef["feature"],
            mode="markers",
            marker=dict(
                size=10,
                color=top_coef["coefficient"].apply(
                    lambda x: "steelblue" if x > 0 else "tomato"
                ),
            ),
            error_x=dict(
                type="data",
                array=(top_coef["ci_upper"] - top_coef["coefficient"]).tolist()
                if "ci_upper" in top_coef.columns else None,
                arrayminus=(top_coef["coefficient"] - top_coef["ci_lower"]).tolist()
                if "ci_lower" in top_coef.columns else None,
                visible=True,
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="OLS Coefficients with 95% CI",
        xaxis_title="Coefficient (log-scale — interpreted as % effect)",
        yaxis_title="Feature",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Full Coefficient Table")
    display_coef = _coef_df.copy()
    for col in ["coefficient", "std_error", "t_stat", "ci_lower", "ci_upper"]:
        if col in display_coef.columns:
            display_coef[col] = display_coef[col].map("{:.4f}".format)
    if "p_value" in display_coef.columns:
        display_coef["p_value"] = pd.to_numeric(_coef_df["p_value"], errors="coerce").map(
            lambda x: f"{x:.2e}" if pd.notna(x) else ""
        )
    st.dataframe(display_coef, use_container_width=True, hide_index=True)
    st.caption(
        "Coefficients = % change in profit erosion per unit increase in predictor "
        "(log-linear specification). Positive = more erosion; negative = less erosion."
    )
else:
    st.warning("TheLook coefficient CSV not found.")

st.divider()

# ── SSL alignment (data-driven) ───────────────────────────────────────────────
st.header("SSL External Validation — Coefficient Alignment")

if _align_df is not None:
    # KPI cards from validation summary
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Direction Aligned",
        f"{_dir_aligned} / {_n_hyp}" if _dir_aligned is not None and _n_hyp else "N/A",
        f"{_dir_pct:.1f}%" if not pd.isna(_dir_pct) else "",
    )
    c2.metric(
        "Significance Agreement",
        f"{_sig_agree} / {_n_hyp}" if _sig_agree is not None and _n_hyp else "N/A",
        "",
    )
    c3.metric(
        "Generalization Score",
        f"{_gen_score:.2f}" if not pd.isna(_gen_score) else "N/A",
        "0 = none, 1 = perfect",
    )

    st.divider()
    st.subheader("Hypothesis Predictor Alignment Table")

    display_align = _align_df.copy()
    bool_cols = ["thelook_significant", "ssl_significant", "direction_aligned", "significance_agreement"]
    for col in bool_cols:
        if col in display_align.columns:
            display_align[col] = _align_df[col].map({True: "✅", False: "❌"})
    for col in ["thelook_coefficient", "ssl_coefficient", "thelook_pct_effect", "ssl_pct_effect",
                "thelook_p_value", "ssl_p_value", "coefficient_ratio"]:
        if col in display_align.columns:
            display_align[col] = pd.to_numeric(_align_df[col], errors="coerce").map(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
    st.dataframe(display_align, use_container_width=True, hide_index=True)

    # Interpretation — derived from alignment data
    st.divider()
    st.subheader("Interpretation")
    rows = []
    for _, row in _align_df.iterrows():
        feat = row["feature"]
        tl_coef = float(row["thelook_coefficient"])
        ssl_coef = float(row["ssl_coefficient"])
        tl_pct = float(row["thelook_pct_effect"])
        ssl_pct = float(row["ssl_pct_effect"])
        tl_sig = bool(row["thelook_significant"])
        ssl_sig = bool(row["ssl_significant"])
        aligned = bool(row["direction_aligned"])

        tl_sig_str = "sig." if tl_sig else "n.s."
        ssl_sig_str = "sig." if ssl_sig else "n.s."
        aligned_str = "✅ Yes" if aligned else "❌ No"
        rows.append({
            "Feature": f"`{feat}`",
            "TheLook coef. (pct effect)": f"{tl_coef:+.3f} ({tl_pct:+.1f}%, {tl_sig_str})",
            "SSL coef. (pct effect)": f"{ssl_coef:+.3f} ({ssl_pct:+.1f}%, {ssl_sig_str})",
            "Direction aligned": aligned_str,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.warning("SSL alignment CSV not found.")

st.divider()

# ── Key findings (data-driven) ────────────────────────────────────────────────
st.header("Key Findings")

if _coef_df is not None and _align_df is not None and _val_dict:
    # Top behavioral predictor by |coefficient| (excluding category dummies and const)
    behav_feats = _coef_df[
        ~_coef_df["feature"].str.startswith("dominant_return_category") &
        (_coef_df["feature"] != "const")
    ].copy()
    if not behav_feats.empty:
        top_behav = behav_feats.loc[behav_feats["coefficient"].abs().idxmax()]
        top_feat_name = top_behav["feature"]
        top_feat_coef = float(top_behav["coefficient"])
        top_feat_pct = top_feat_coef * 100  # approximate % effect for log-linear

    # Category dummies — positive vs negative
    cat_dummies = _coef_df[_coef_df["feature"].str.startswith("dominant_return_category")].copy()
    if "p_value" in cat_dummies.columns:
        sig_cat = cat_dummies[pd.to_numeric(cat_dummies["p_value"], errors="coerce") < 0.05]
        pos_cats = sig_cat[sig_cat["coefficient"] > 0]["feature"].str.replace(
            "dominant_return_category_", ""
        ).tolist()
        neg_cats = sig_cat[sig_cat["coefficient"] < 0]["feature"].str.replace(
            "dominant_return_category_", ""
        ).tolist()
    else:
        pos_cats, neg_cats = [], []

    # Demographics
    demo_feats = ["age", "user_gender_M", "customer_tenure_days", "purchase_recency_days"]
    demo_present = _coef_df[_coef_df["feature"].isin(demo_feats)]
    if "p_value" in demo_present.columns:
        demo_sig = demo_present[pd.to_numeric(demo_present["p_value"], errors="coerce") < 0.05]
        demos_significant = len(demo_sig) > 0
    else:
        demos_significant = None

    # Alignment for top predictor
    top_align_row = _align_df[_align_df["feature"] == top_feat_name] if top_feat_name in _align_df["feature"].values else None

    findings = [
        f"**R² = {_thelook_r2:.4f}** — behavioral + category features explain "
        f"{_thelook_r2*100:.1f}% of log-profit-erosion variance",
        f"**`{top_feat_name}`** is the strongest behavioral predictor "
        f"(coefficient = {top_feat_coef:+.4f})",
    ]

    if top_align_row is not None and not top_align_row.empty:
        r = top_align_row.iloc[0]
        tl_pct = float(r["thelook_pct_effect"])
        ssl_pct = float(r["ssl_pct_effect"])
        direction = "consistent direction" if bool(r["direction_aligned"]) else "opposite directions"
        findings.append(
            f"  TheLook pct effect: {tl_pct:+.1f}% | SSL pct effect: {ssl_pct:+.1f}% — {direction}"
        )

    if pos_cats:
        findings.append(
            f"**Positive category effects** (significant): {', '.join(pos_cats[:5])}"
        )
    if neg_cats:
        findings.append(
            f"**Negative category effects** (significant): {', '.join(neg_cats[:5])}"
        )

    if demos_significant is not None:
        if not demos_significant:
            findings.append(
                "**Demographics not significant**: age, gender, tenure, recency show no "
                "significant marginal effect after controlling for behavioral and category features"
            )
        else:
            sig_names = demo_sig["feature"].tolist()
            findings.append(f"**Significant demographics**: {', '.join(sig_names)}")

    findings.append(
        f"**SSL generalization**: R² = {_ssl_r2:.4f} (ratio = {_r2_ratio:.2f}), "
        f"generalization score = {_gen_score:.2f} ({_dir_aligned}/{_n_hyp} hypothesis predictors direction-aligned)"
    )
    findings.append(
        "**Hypothesis**: Reject H₀ — behavioral and category features have statistically "
        "significant associations with profit erosion"
    )

    st.markdown("\n".join(f"- {f}" for f in findings))
else:
    st.info("Key findings require coefficient and validation CSVs. Run the master notebook.")
