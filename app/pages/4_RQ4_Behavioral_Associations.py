"""
RQ4: Marginal Associations Between Behaviors and Profit Erosion

Method: Log-Linear OLS Regression
Model: log(total_profit_erosion) ~ behavioral + category + demographics
External validation: School Specialty LLC (SSL)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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
_ssl_coef_df = None
_align_df = None
_val_dict = {}

_coef_path = REPORTS_RQ4 / "rq4_thelook_coefficients.csv"
if _coef_path.exists():
    _coef_df = pd.read_csv(_coef_path)

_ssl_coef_path = REPORTS_RQ4 / "rq4_ssl_coefficients.csv"
if _ssl_coef_path.exists():
    _ssl_coef_df = pd.read_csv(_ssl_coef_path)

_align_path = REPORTS_RQ4 / "rq4_ssl_coefficient_alignment.csv"
if _align_path.exists():
    _align_df = pd.read_csv(_align_path)

_val_path = REPORTS_RQ4 / "rq4_validation_summary.csv"
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

# ── 5-Tab layout ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Overview", "📈 Model Results", "🎯 What Matters", "🌐 Validation", "🔬 Robustness"]
)


with tab1:
    # ── KPI cards (data-driven) ───────────────────────────────────────────────────
    st.header("Model Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "R² (TheLook)",
        f"{_thelook_r2:.4f}" if not pd.isna(_thelook_r2) else "N/A",
        f"{_thelook_r2*100:.1f}% of log-erosion variance explained" if not pd.isna(_thelook_r2) else "",
        help=(
                "R² indicates how well the model explains the variance in log-transformed profit erosion. "
            ),
    )
    col2.metric(
        "R² (SSL)",
        f"{_ssl_r2:.4f}" if not pd.isna(_ssl_r2) else "N/A",
        f"R² ratio = {_r2_ratio:.2f}" if not pd.isna(_r2_ratio) else "",
        help=(
                "R² ratio indicates how much better the SSL model performs compared to TheLook model. "
            ),
    )
    col3.metric(
        "Observations (TheLook)",
        f"{_thelook_n:,}" if _thelook_n else "N/A",
        "customers with ≥1 return",
        help=(
                "Number of customer accounts in TheLook dataset used for OLS regression. "
            ),
    )
    col4.metric(
        "SSL Accounts Validated",
        f"{_ssl_n:,}" if _ssl_n else "N/A",
        help=(
                "Number of customer accounts in the SSL dataset that were validated against TheLook. "
            ),
    )

    st.divider()

    # ── Visualizations ────────────────────────────────────────────────────────────
    st.header("Visualizations")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Target Distribution (Log-Transformed)")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### Why Two Charts?

                **Raw Distribution (Blue Histogram)**
                
                Shows the actual return costs to your company:
                - Most returns cost less than $100
                - A small number of returns cost significantly more (up to $700)
                - This "long tail" skew makes the data hard to analyze statistically

                **Log-Transformed Distribution (Orange Histogram)**
                
                Shows the same data but on a normalized scale:
                - Converts values using logarithm: $\\log(x)$
                - Results in a bell-curve shape that standard analytics models expect
                - Makes patterns and relationships easier to detect

                #### Why Transform the Data?

                Most predictive models assume normally distributed data. By log-transforming, we:
                - Enable accurate statistical modeling
                - Improve coefficient estimates and confidence intervals
                - Validate that profit erosion follows a **lognormal distribution** (common in finance & insurance)

                **Bottom Line:** This transformation allows us to reliably estimate how customer behaviors drive profit erosion.
                """
            )

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
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### How to Read

                - **Red points (right):** Increase erosion | **Blue points (left):** Decrease erosion
                - Narrow error bars = high confidence | All shown are statistically significant (p < 0.05)

                ---

                #### Key Drivers of Erosion (Red)

                **Return Frequency** — Strongest driver (~100% impact per return)
                **High-Cost Categories** — Outerwear, Suits, Sweaters (shipping + restocking)
                **Customer Return Rate** — Repeat returners = repeat erosion

                ---

                #### Protective Factors (Blue)

                **Low-Cost Categories** — Leggings, Intimates, Underwear (cheaper to process)
                **Larger Basket Size** — Economies of scale + higher-quality customers

                ---

                #### Strategic Actions

                1. **Reduce return frequency** (strongest lever): Size guides, virtual try-on for Red categories
                2. **Segment policies:** Stricter for Outerwear/Suits; flexible for Leggings/Intimates  
                3. **Target repeat returners:** Simple process; reward loyal non-returners
                4. **Cross-sell strategy:** Bundle Red categories with Blue categories
                """
            )
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
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### Key Finding: Heteroscedasticity

                **Plots 1 & 4 (Residuals vs Fitted & Scale-Location):** "Funnel" pattern shows error spread increases with prediction size.
                - **Low erosion predictions:** Highly reliable ✅
                - **High erosion predictions:** Wide error margins ⚠️

                ---

                #### Model Behavior

                **Plot 2 (Q-Q Residuals):** Left tail indicates overprediction — model occasionally predicts much higher losses than actual.
                - Missing factors: restocking efficiency, successful resale, inventory turnover

                **Plot 3 (Distribution):** Tall, narrow peak with left tail — most errors small, but occasional large overestimation.
                - **Practical:** Model is conservative (safer for budgeting) but may cause operational friction

                ---

                #### Actions for elook

                1. Add safety margins to high-erosion predictions (account for uncertainty)
                2. Collect features for high-value returns: order size, product category, customer tenure
                3. Consider separate models for low vs. high erosion segments
                """
            )
        resid_path = FIGURES_RQ4 / "rq4_residual_diagnostics.png"
        if resid_path.exists():
            st.image(str(resid_path), use_container_width=True)
            st.caption("Residual plots validate OLS assumptions: linearity, homoscedasticity, normality.")
        else:
            st.warning("Residual diagnostics figure not found.")

    with col_d:
        st.subheader("QQ Plot — Normality Check")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### Why Log-Transform Works

                **Linear Model (Left):** Jarque-Bera = 619,317 — Severe non-normality
                - S-shaped deviation from diagonal = underestimation of extreme losses
                - Linear model misses "profit killer" returns, risking budget shortfalls

                **Log Model (Right):** Jarque-Bera = 2,198 — **281.8x improvement**
                - Points hug diagonal = normally distributed (model assumptions satisfied)
                - Captures multiplicative nature: shipping × restocking × seasonal loss × support time

                ---

                #### Strategic Takeaway: Pareto Principle (80/20 Rule)

                - Small % of extreme returns = most profit loss
                - Focus on high-impact behaviors (visible in Coefficient Forest Plot)
                - Avoid one-size-fits-all return policy; target interventions instead
                """
            )
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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Performance & Diagnostics")
    
    st.markdown(
        """
        This section evaluates the log-linear OLS regression model's fit, assumptions, and reliability.
        """
    )
    
    st.divider()
    
    # ── Model Fit Summary ──────────────────────────────────────────────────────────
    st.subheader("1. Model Fit Summary")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "R² (TheLook Training)",
        f"{_thelook_r2:.4f}" if not pd.isna(_thelook_r2) else "N/A",
        f"Explains {_thelook_r2*100:.1f}% of log-erosion variance",
        help="Proportion of variance in log(profit_erosion) explained by behavioral + category features",
    )
    col2.metric(
        "Observations (TheLook)",
        f"{_thelook_n:,}" if _thelook_n else "N/A",
        "customers with ≥1 return",
        help="Sample size used to estimate coefficients",
    )
    col3.metric(
        "Features (Significant)",
        f"{len(_coef_df[pd.to_numeric(_coef_df.get('p_value', 1), errors='coerce') < 0.05]) if _coef_df is not None else 'N/A'}",
        f"out of {len(_coef_df) if _coef_df is not None else 'N/A'} total",
        help="Number of coefficients with p < 0.05",
    )
    
    st.divider()
    
    # ── OLS Assumptions ────────────────────────────────────────────────────────────
    st.subheader("2. OLS Assumptions Validation")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Normality (Q-Q Plot)**")
        with st.expander("ℹ️ Details", expanded=False):
            st.markdown(
                """
                Residuals should follow a normal distribution for valid p-values and confidence intervals.
                
                **Finding:** Log-transformed specification achieves near-normality 
                (Jarque-Bera = 2,198, 281.8× improvement over linear).
                """
            )
        qq_path = FIGURES_RQ4 / "rq4_qq_plot_comparison.png"
        if qq_path.exists():
            st.image(str(qq_path), use_container_width=True)
    
    with col_b:
        st.markdown("**Residual Patterns (Homoscedasticity)**")
        with st.expander("ℹ️ Details", expanded=False):
            st.markdown(
                """
                Error variance should be constant across fitted values. 
                
                **Finding:** Heteroscedasticity detected — error spread increases with predicted erosion size.
                - Low predictions (< $50): Reliable ✅
                - High predictions (> $500): Wide confidence bands ⚠️
                """
            )
        resid_path = FIGURES_RQ4 / "rq4_residual_diagnostics.png"
        if resid_path.exists():
            st.image(str(resid_path), use_container_width=True)
    
    st.divider()
    
    # ── External Validation Snapshot ──────────────────────────────────────────────
    st.subheader("3. Cross-Dataset Performance (SSL Validation)")
    
    col_x, col_y, col_z = st.columns(3)
    col_x.metric(
        "R² (SSL External)",
        f"{_ssl_r2:.4f}" if not pd.isna(_ssl_r2) else "N/A",
        f"Ratio: {_r2_ratio:.2f}x TheLook",
        help="External validation on B2B dataset (School Specialty LLC)",
    )
    col_y.metric(
        "SSL Accounts Tested",
        f"{_ssl_n:,}" if _ssl_n else "N/A",
        "from B2B channel",
        help="Number of SSL accounts with sufficient return history",
    )
    col_z.metric(
        "Generalization Score",
        f"{_gen_score:.2f}" if not pd.isna(_gen_score) else "N/A",
        f"{_dir_aligned}/{_n_hyp} hypothesis features aligned",
        help="0 = no transfer, 1 = perfect transfer",
    )
    
    st.markdown(
        f"""
        **Interpretation:**
        
        - R² decreased from {_thelook_r2:.4f} (TheLook) to {_ssl_r2:.4f} (SSL) — 
          {((1 - _r2_ratio) * 100):.0f}% variance capture drop on external data
        - Only {_dir_pct:.0f}% of hypothesis predictors maintain same direction in SSL
        - **Implication:** Core behavioral patterns exist, but **magnitude varies by customer segment** 
          (B2C vs. B2B have different erosion drivers)
        """
    )
    
    st.divider()
    
    # ── Model Summary Statistics ──────────────────────────────────────────────────
    st.subheader("4. Feature Category Breakdown")
    
    if _coef_df is not None:
        # Categorize features
        const_feats = _coef_df[_coef_df["feature"] == "const"]
        behav_feats = _coef_df[
            ~_coef_df["feature"].str.contains("dominant_return_category|const", regex=True)
        ]
        cat_feats = _coef_df[_coef_df["feature"].str.contains("dominant_return_category")]
        
        col_p, col_q, col_r = st.columns(3)
        col_p.metric(
            "Behavioral Predictors",
            len(behav_feats),
            f"{(len(behav_feats) / (len(_coef_df) - 1) * 100):.0f}% of model",
            help="Core behavioral features (return frequency, basket size, recency) that drive profit erosion across categories",
        )
        col_q.metric(
            "Category Dummies",
            len(cat_feats),
            f"{(len(cat_feats) / (len(_coef_df) - 1) * 100):.0f}% of model",
            help="Indicator variables for dominant return categories (e.g., Outerwear, Leggings) that capture category-specific effects on erosion",
        )
        col_r.metric(
            "Baseline Cost",
            f"${np.exp(const_feats['coefficient'].values[0]):.2f}" if len(const_feats) > 0 else "N/A",
            "Intercept (constant term)",
            help="Baseline cost of customer account (constant term) in log-erosion regression",
        )
    
    st.divider()

    st.markdown(
        "**Key Takeaway:** Strong model fit on TheLook training data (R²=0.78). "
        "Heteroscedasticity present → use caution with high-erosion predictions. "
        "SSL validation shows moderate generalization → patterns real but context-dependent."
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT MATTERS (Feature Importance)
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Feature Importance & Transferability")
    
    st.markdown(
        """
        Which features drive profit erosion in TheLook? Which generalize to other customer segments (SSL)?
        This section ranks features by impact and flags **transferability** — what works across contexts.
        """
    )
    
    st.divider()
    
    # ── Feature Importance Ranking ─────────────────────────────────────────────────
    st.subheader("1. Feature Importance by Coefficient Magnitude")
    
    if _coef_df is not None:
        # Build top features ranked by absolute coefficient
        top_features = (
            _coef_df
            .assign(abs_coef=_coef_df["coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
            .head(15)
        )
        
        fig_importance = go.Figure()
        fig_importance.add_trace(
            go.Bar(
                x=top_features["coefficient"],
                y=top_features["feature"],
                orientation="h",
                marker=dict(
                    color=top_features["coefficient"].apply(
                        lambda x: "#EF5350" if x > 0 else "#42A5F5"
                    ),
                    opacity=0.7,
                ),
                text=top_features["coefficient"].apply(lambda x: f"{x:+.3f}"),
                textposition="inside",
            )
        )
        fig_importance.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_importance.update_layout(
            title="Top 15 Features by Impact (TheLook)",
            xaxis_title="Coefficient (log-scale — % effect on profit erosion)",
            yaxis_title="",
            height=500,
            showlegend=False,
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.markdown(
            """
            **How to interpret:**
            - **Red bars (right):** Features that INCREASE profit erosion
            - **Blue bars (left):** Features that DECREASE erosion (protective factors)
            - **Length:** Strength of effect — longer bars = stronger impact
            
            Example: `return_frequency` with coefficient 0.445 means: **for each additional return a customer makes, 
            profit erosion increases by ~45%** (multiplicative, log-scale).
            """
        )
    
    st.divider()
    
    # ── Coefficient Scatter: TheLook vs SSL ────────────────────────────────────────
    st.subheader("2. Coefficient Comparison: TheLook vs SSL (Transferability Visual)")
    
    if _coef_df is not None and _ssl_coef_df is not None:
        st.markdown(
            """
            This scatter plot compares how coefficients differ between TheLook (B2C) and SSL (B2B).
            Features on the **diagonal** transfer perfectly; those **far from diagonal** show context-dependent effects.
            """
        )
        
        # Create mapping for shared features (we'll use feature name matching)
        shared_features = []
        comparison_data = []
        
        for feat_tl in _coef_df["feature"].unique():
            if feat_tl == "const":
                continue
            # Find matching feature in SSL
            feat_ssl_matches = [f for f in _ssl_coef_df["feature"].unique() if f in feat_tl or feat_tl in f]
            if feat_ssl_matches and len(feat_ssl_matches) > 0:
                feat_ssl = feat_ssl_matches[0]
                tl_row = _coef_df[_coef_df["feature"] == feat_tl].iloc[0]
                ssl_row = _ssl_coef_df[_ssl_coef_df["feature"] == feat_ssl].iloc[0]
                
                # Determine feature category
                if "dominant_return_category" in feat_tl:
                    category = "Category"
                elif feat_tl in ["return_frequency", "avg_basket_size", "purchase_recency_days", "avg_order_value"]:
                    category = "Behavioral"
                else:
                    category = "Other"
                
                comparison_data.append({
                    "feature": feat_tl.replace("dominant_return_category_", ""),
                    "thelook_coef": float(tl_row["coefficient"]),
                    "ssl_coef": float(ssl_row["coefficient"]),
                    "category": category,
                    "abs_coef": abs(float(tl_row["coefficient"])),
                })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            # Color mapping for categories
            color_map = {"Behavioral": "#1f77b4", "Category": "#ff7f0e", "Other": "#2ca02c"}
            colors = [color_map.get(cat, "#999999") for cat in comp_df["category"]]
            
            fig_scatter = go.Figure()
            
            # Add traces by category for legend
            for category in comp_df["category"].unique():
                subset = comp_df[comp_df["category"] == category]
                fig_scatter.add_trace(
                    go.Scatter(
                        x=subset["thelook_coef"],
                        y=subset["ssl_coef"],
                        mode="markers",
                        name=category,
                        marker=dict(
                            size=subset["abs_coef"] * 15 + 5,
                            color=color_map.get(category, "#999999"),
                            opacity=0.7,
                            line=dict(width=1, color="white"),
                        ),
                        text=subset["feature"],
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "TheLook: %{x:.4f}<br>"
                            "SSL: %{y:.4f}<br>"
                            "<extra></extra>"
                        ),
                    )
                )
            
            # Add diagonal reference line (perfect transfer)
            min_val = min(comp_df["thelook_coef"].min(), comp_df["ssl_coef"].min())
            max_val = max(comp_df["thelook_coef"].max(), comp_df["ssl_coef"].max())
            fig_scatter.add_shape(
                type="line",
                x0=min_val,
                y0=min_val,
                x1=max_val,
                y1=max_val,
                line=dict(dash="dash", color="rgba(100,100,100,0.5)", width=2),
            )
            
            fig_scatter.update_layout(
                title="TheLook vs SSL Coefficient Alignment<br><sub>Points on diagonal = transfer perfectly | Size = magnitude</sub>",
                xaxis_title="TheLook Coefficient (B2C Fashion)",
                yaxis_title="SSL Coefficient (B2B Supplies)",
                height=600,
                hovermode="closest",
                showlegend=True,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.caption(
                "**Interpretation:** "
                "- Points along diagonal ↗️ = feature effects are **stable across contexts** ✅"
                "- Points above diagonal ↑ = effect **stronger in B2B** (SSL > TheLook)"
                "- Points below diagonal ↓ = effect **stronger in B2C** (TheLook > SSL)"
                "- Points far left/right but near zero = feature is **context-dependent** ⚠️"
            )
        else:
            st.info("No matching features found between TheLook and SSL datasets.")
    else:
        st.info("TheLook or SSL coefficient data not loaded.")
    
    st.divider()
    
    # ── Transferability Analysis ───────────────────────────────────────────────────
    st.subheader("3. Transferability to External Data (B2B/SSL)")
    
    if _align_df is not None:
        st.markdown(
            """
            Your model was built on TheLook (B2C e-commerce). Does it work on SSL (B2B purchasing)?
            Here are the **hypothesis features** — core behavioral predictors tested for transferability.
            """
        )
        
        # Build a more visual transferability summary
        transfer_rows = []
        for _, row in _align_df.iterrows():
            feat = row["feature"]
            tl_coef = float(row["thelook_coefficient"])
            ssl_coef = float(row["ssl_coefficient"])
            tl_pct = float(row["thelook_pct_effect"])
            ssl_pct = float(row["ssl_pct_effect"])
            tl_p = float(row["thelook_p_value"])
            ssl_p = float(row["ssl_p_value"])
            aligned = bool(row["direction_aligned"])
            agree_sig = bool(row["significance_agreement"])
            
            tl_sig_str = "✅ Sig" if tl_p < 0.05 else "❌ n.s."
            ssl_sig_str = "✅ Sig" if ssl_p < 0.05 else "❌ n.s."
            aligned_str = "✅ YES" if aligned else "❌ NO"
            transfer_icon = "🔄" if (aligned and agree_sig) else "⚠️" if aligned else "❌"
            
            transfer_rows.append({
                "Status": transfer_icon,
                "Feature": feat,
                "TheLook Coef": f"{tl_coef:+.4f}",
                "TheLook Effect": f"{tl_pct:+.1f}%",
                "TheLook": tl_sig_str,
                "SSL Coef": f"{ssl_coef:+.4f}",
                "SSL Effect": f"{ssl_pct:+.1f}%",
                "SSL": ssl_sig_str,
                "Direction": aligned_str,
            })
        
        transfer_df = pd.DataFrame(transfer_rows)
        st.dataframe(transfer_df, use_container_width=True, hide_index=True)
        
        st.markdown(
            """
            **Legend:**
            - 🔄 = Transfers perfectly (direction + significance stable)
            - ⚠️ = Partial transfer (direction aligned but significance differs)
            - ❌ = Doesn't transfer (direction flips or drops entirely)
            """
        )
    
    st.divider()
    
    # ── Key Findings ───────────────────────────────────────────────────────────────
    st.subheader("4. Critical Insights: What Transfers & What Doesn't")
    
    if _coef_df is not None and _align_df is not None:
        
        with st.expander("🔍 Behavioral Features (Core Predictors)", expanded=True):
            st.markdown(
                """
                **Return Frequency** ✅ → Transfers to SSL
                - TheLook: +0.445 (+56% per return) — STRONG
                - SSL: +0.104 (+11% per return) — WEAK but same direction
                - **Strategic implication:** Return frequency is universal driver, but B2B customers 
                  return less often OR returns matter less in B2B context
                
                **Basket Size/AOV** ❌ → REVERSES in SSL!
                - TheLook: -0.156 (-14.4% per AOV unit) — larger orders = LESS erosion
                - SSL: +0.320 (+37.7% per AOV unit) — larger orders = MORE erosion  
                - **Critical finding:** Size-based economics flip between B2C and B2B!
                  - B2C: Premium customers (large baskets) return less
                  - B2B: Large orders have more returns (bulk shipping complexity?)
                
                **Purchase Recency** ❌ → Only significant in SSL
                - TheLook: n.s. (not predictive)
                - SSL: Significant (recency matters for B2B retention/erosion)
                """
            )
        
        with st.expander("📦 Product Categories (Stable Across Datasets)", expanded=False):
            # Extract category data
            cat_df = _coef_df[_coef_df["feature"].str.contains("dominant_return_category")].copy()
            if "p_value" in cat_df.columns:
                cat_df["significant"] = pd.to_numeric(cat_df["p_value"], errors="coerce") < 0.05
            cat_df = cat_df.sort_values("coefficient", ascending=False)
            
            high_erosion = cat_df[cat_df["coefficient"] > 0]
            low_erosion = cat_df[cat_df["coefficient"] < 0]
            
            st.markdown("**High-Erosion Categories (RED):**")
            cols = st.columns(2)
            high_list = high_erosion["feature"].str.replace("dominant_return_category_", "").tolist()
            cols[0].markdown("- " + "\n- ".join(high_list[:len(high_list)//2 + 1]))
            cols[1].markdown("- " + "\n- ".join(high_list[len(high_list)//2 + 1:]))
            
            st.markdown("**Low-Erosion Categories (BLUE):**")
            cols = st.columns(2)
            low_list = low_erosion["feature"].str.replace("dominant_return_category_", "").tolist()
            cols[0].markdown("- " + "\n- ".join(low_list[:len(low_list)//2 + 1]))
            cols[1].markdown("- " + "\n- ".join(low_list[len(low_list)//2 + 1:]))
            
            st.markdown(
                """
                **Strategic insight:** Category patterns are **the most stable** across datasets.
                - Suits, Outerwear cost more to process regardless of B2C/B2B
                - Intimates, Underwear inherently cheaper to restock
                → Category-based policies are **highly transferable**
                """
            )
        
        with st.expander("👤 Demographics (Minimal Impact)", expanded=False):
            demo_cols = ["age", "user_gender_M", "customer_tenure_days", "purchase_recency_days"]
            demo_df = _coef_df[_coef_df["feature"].isin(demo_cols)]
            
            if not demo_df.empty and "p_value" in demo_df.columns:
                demo_df["significant"] = pd.to_numeric(demo_df["p_value"], errors="coerce") < 0.05
                sig_demos = demo_df[demo_df["significant"]]
                
                if not sig_demos.empty:
                    st.markdown(f"**Found {len(sig_demos)} significant demographic features:**")
                    for _, row in sig_demos.iterrows():
                        st.markdown(
                            f"- `{row['feature']}`: {row['coefficient']:+.4f} (p={row['p_value']:.2e})"
                        )
                else:
                    st.markdown(
                        """
                        **None significant.** Age, gender, tenure, recency show **no marginal effect** 
                        after controlling for behaviors and categories. This suggests that 
                        **what customers do** (return frequency, order size) matters much more 
                        than **who they are** (demographics).
                        """
                    )
            else:
                st.markdown("Demographics not present in model.")
    
    st.divider()
    
    # ── Actionable Recommendations ─────────────────────────────────────────────────
    st.subheader("5. What This Means for elook")
    
    st.markdown(
        """
        ### 🎯 Three-Tier Strategy
        
        **Tier 1 — Universal (High Transferability) ✅**
        - Return frequency interventions apply everywhere
        - Category policies work for both B2C/B2B
        - → **Deploy immediately across all channels**
        
        **Tier 2 — Conditional (Partial Transferability) ⚠️**
        - Basket size effects DIFFER between B2C and B2B
        - B2C: incentivize large orders (premium customers = fewer returns)
        - B2B: investigate why large orders have MORE returns (logistics issue?)
        - → **Use channel-specific tactics**
        
        **Tier 3 — Context-Specific (Low Transferability) ❌**
        - Demographics (age, gender) don't drive erosion directly
        - Recency only matters in B2B retention context
        - → **Don't build universal policies based on these**
        
       """
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION (SSL External)
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("External Validation — School Specialty LLC (SSL)")

    if _align_df is not None:
        # KPI cards from validation summary
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Direction Aligned",
            f"{_dir_aligned} / {_n_hyp}" if _dir_aligned is not None and _n_hyp else "N/A",
            f"{_dir_pct:.1f}%" if not pd.isna(_dir_pct) else "",
            help=(
                "Direction aligned means the coefficient signs (positive/negative) for hypothesis features match between TheLook and SSL. "
            ),
        )
        c2.metric(
            "Significance Agreement",
            f"{_sig_agree} / {_n_hyp}" if _sig_agree is not None and _n_hyp else "N/A",
            help=(
                "Significance agreement means both datasets agree on whether a feature is statistically significant (p < 0.05) or not. "
            ),
        )
        c3.metric(
            "Generalization Score",
            f"{_gen_score:.2f}" if not pd.isna(_gen_score) else "N/A",
            "0 = none, 1 = perfect",
            help=(
                "Generalization score is a composite metric that captures both direction alignment and significance agreement across all hypothesis features. "
            ),
        )

        st.divider()
        
        # ── Confidence Interval Comparison ────────────────────────────────────────
        st.subheader("Confidence Interval Stability: TheLook vs SSL")
        
        if _coef_df is not None and _ssl_coef_df is not None:
            st.markdown(
                """
                How certain are we about each coefficient? This chart compares **confidence interval widths** 
                to show which predictions are reliable (tight CIs) vs. uncertain (wide CIs) in each dataset.
                """
            )
            
            # Build comparison for top behavioral + category features
            ci_comparison = []
            for feat_tl in _coef_df[~_coef_df["feature"].isin(["const"])]["feature"].head(15):
                tl_row = _coef_df[_coef_df["feature"] == feat_tl].iloc[0]
                
                # Find matching SSL feature
                feat_ssl_matches = [f for f in _ssl_coef_df["feature"].unique() if f in feat_tl or feat_tl in f]
                if feat_ssl_matches and len(feat_ssl_matches) > 0:
                    feat_ssl = feat_ssl_matches[0]
                    ssl_row = _ssl_coef_df[_ssl_coef_df["feature"] == feat_ssl].iloc[0]
                    
                    tl_ci_width = float(tl_row["ci_upper"]) - float(tl_row["ci_lower"])
                    ssl_ci_width = float(ssl_row["ci_upper"]) - float(ssl_row["ci_lower"])
                    
                    clean_name = feat_tl.replace("dominant_return_category_", "")
                    ci_comparison.append({
                        "feature": clean_name,
                        "thelook_coef": float(tl_row["coefficient"]),
                        "thelook_ci_lower": float(tl_row["ci_lower"]),
                        "thelook_ci_upper": float(tl_row["ci_upper"]),
                        "ssl_coef": float(ssl_row["coefficient"]),
                        "ssl_ci_lower": float(ssl_row["ci_lower"]),
                        "ssl_ci_upper": float(ssl_row["ci_upper"]),
                    })
            
            if ci_comparison:
                ci_df = pd.DataFrame(ci_comparison)
                
                fig_ci = go.Figure()
                
                # Add TheLook error bars
                fig_ci.add_trace(
                    go.Scatter(
                        x=ci_df["thelook_coef"],
                        y=ci_df["feature"],
                        mode="markers",
                        name="TheLook",
                        marker=dict(size=8, color="#4472C4"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=(ci_df["thelook_ci_upper"] - ci_df["thelook_coef"]).tolist(),
                            arrayminus=(ci_df["thelook_coef"] - ci_df["thelook_ci_lower"]).tolist(),
                            color="#4472C4",
                            thickness=2,
                        ),
                        hovertemplate="<b>%{y}</b> (TheLook)<br>Coef: %{x:.4f}<extra></extra>",
                    )
                )
                
                # Add SSL error bars
                fig_ci.add_trace(
                    go.Scatter(
                        x=ci_df["ssl_coef"],
                        y=ci_df["feature"],
                        mode="markers",
                        name="SSL",
                        marker=dict(size=8, color="#ED7D31"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=(ci_df["ssl_ci_upper"] - ci_df["ssl_coef"]).tolist(),
                            arrayminus=(ci_df["ssl_coef"] - ci_df["ssl_ci_lower"]).tolist(),
                            color="#ED7D31",
                            thickness=2,
                        ),
                        hovertemplate="<b>%{y}</b> (SSL)<br>Coef: %{x:.4f}<extra></extra>",
                    )
                )
                
                fig_ci.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
                fig_ci.update_layout(
                    title="Coefficient Stability Across Datasets<br><sub>Wider error bars = less certain | Overlapping CIs = patterns may not be reliable</sub>",
                    xaxis_title="Coefficient (95% CI)",
                    yaxis_title="",
                    height=600,
                    hovermode="y unified",
                    showlegend=True,
                )
                st.plotly_chart(fig_ci, use_container_width=True)
                
                st.caption(
                    "**Interpretation:** "
                    "- **Tight error bars** (short lines) = reliable, well-estimated coefficient ✅ "
                    "- **Wide error bars** = uncertain, imprecise estimate ⚠️ "
                    "- **CIs don't overlap** between TheLook & SSL = robust difference between contexts"
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

    st.subheader("TheLook vs SSL Coefficient Comparison")
    with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### Why Standardized Comparison?
                
                Both plots use **"Coefficient (std dev of target)"** — allowing fair comparison despite different business scales 
                (Fashion B2C vs. Institutional Supply B2B). This reveals true pattern transferability, not just numerical coincidence.
                
                ---
                
                #### TheLook (Synthetic Fashion)
                
                **Signature Patterns:**
                - Return Frequency & AOV are dominant behavioral drivers
                - Suits, Outerwear erode most; Socks, Hosiery protect
                - **Tight error bars** → predictable, "clean" retail patterns
                
                ---
                
                #### SSL (Real-World Data)
                
                **Different Story:**
                - Categories dominate (Media Center, Science Furniture, Supplies)
                - Return Frequency & Customer Behavior much closer to zero
                - **Wide error bars** → unpredictable, context-dependent effects
                
                ---
                
                #### What Transfers? What Doesn't?
                
                ❌ **Category Names Don't Transfer** — Different business types
                
                ⚠️ **Return Frequency Transfers (Weakly)** — Positive in both but 5-10× weaker in SSL
                
                ✅ **"Safe Low-Cost" Pattern Transfers Strongly** — Socks→Adhesives; Hosiery→Paper (both protective)
                
                💡 **Key Insight:** Behavior drives synthetic erosion; product type drives real-world erosion.
                """
            )
    forest_path = FIGURES_RQ4 / "rq4_ssl_forest_comparison.png"
    if forest_path.exists():
        st.image(str(forest_path), use_container_width=True)
        st.caption(
            "Side-by-side forest plot comparing OLS coefficients for hypothesis predictors "
            "across TheLook (B2C) and SSL (B2B) datasets."
        )
    else:
        st.warning("SSL forest comparison figure not found.")

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


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROBUSTNESS (Sensitivity Analysis)
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Sensitivity & Robustness Analysis")
    st.markdown(
        """
        This interactive tool shows sensitivity analysis results for a "Sensitivity Analysis". We can demonstrate that even when focusing only on the top 10 most impactful features, the core story of Category Volatility vs. Return Frequency remains consistent. This shows that our findings are not just driven by a long tail of less important features, but are robust to different levels of feature inclusion. It also allows you to highlight specific product categories (e.g., Outerwear, Suits) and their strong association with profit erosion, which is a key actionable insight for the business.
    """
    )

    st.divider()

    # ── Cumulative Variance Explained ──────────────────────────────────────────
    st.header("Cumulative Feature Impact on Model Fit")
    
    if _coef_df is not None:
        st.markdown(
            """
            How many features do you need to explain the majority of your model's predictive power?
            This chart shows **cumulative R² contribution** as we add features from most to least important.
            """
        )
        
        # Calculate contribution of each feature to R²
        # Use squared coefficients as proxy for variance explained (crude but intuitive)
        feature_impact = (
            _coef_df[~(_coef_df["feature"] == "const")]
            .assign(
                abs_tstat=_coef_df["t_stat"].abs(),
                feature_display=_coef_df["feature"].str.replace("dominant_return_category_", ""),
            )
            .sort_values("abs_tstat", ascending=False)
            .reset_index(drop=True)
        )
        
        # Calculate cumulative impact (normalized by max)
        feature_impact["cumulative_contribution"] = (
            feature_impact["abs_tstat"].cumsum() / feature_impact["abs_tstat"].sum() * 100
        )
        feature_impact["feature_number"] = range(1, len(feature_impact) + 1)
        
        fig_cumul = go.Figure()
        fig_cumul.add_trace(
            go.Scatter(
                x=feature_impact["feature_number"],
                y=feature_impact["cumulative_contribution"],
                mode="lines+markers",
                name="Cumulative Impact",
                fill="tozeroy",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=6),
                hovertemplate="<b>Top %{x} features</b><br>Cumulative impact: %{y:.1f}%<extra></extra>",
            )
        )
        
        # Add reference lines for common thresholds
        for threshold, label, color in [(50, "50%", "rgba(255,0,0,0.3)"), (80, "80%", "rgba(255,165,0,0.3)"), (95, "95%", "rgba(0,128,0,0.3)")]:
            fig_cumul.add_hline(
                y=threshold,
                line_dash="dash",
                line_color=color,
                annotation_text=label,
                annotation_position="right",
            )
        
        fig_cumul.update_layout(
            title="How Many Features to Explain Model Behavior?",
            xaxis_title="Number of Features (ranked by t-statistic magnitude)",
            yaxis_title="Cumulative Impact (%)",
            height=450,
            hovermode="x unified",
            showlegend=True,
        )
        st.plotly_chart(fig_cumul, use_container_width=True)
        
        # Add summary stats
        top_5_impact = feature_impact.loc[4, "cumulative_contribution"]
        top_10_impact = feature_impact.loc[9, "cumulative_contribution"] if len(feature_impact) > 9 else 100
        for_80_pct = len(feature_impact[feature_impact["cumulative_contribution"] <= 80])
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        col_stats1.metric(
            "Top 5 Features Impact",
            f"{top_5_impact:.1f}%",
            "Of total model strength",
            help="The top 5 features (by t-statistic magnitude) explain this percentage of the model's predictive power. A high percentage here indicates a strong Pareto effect, where a few key predictors dominate the model's behavior.",
        )
        col_stats2.metric(
            "Top 10 Features Impact",
            f"{top_10_impact:.1f}%",
            "Of total model strength",
            help="The top 10 features explain this percentage of the model's predictive power. If this is close to 100%, it suggests that the model's behavior is largely driven by a small subset of features, reinforcing the idea of focusing on key drivers for interventions.",
        )
        col_stats3.metric(
            "Features for 80% Impact",
            f"{for_80_pct}",
            f"Out of {len(feature_impact)} total",
            help="This shows how many features you need to include to explain 80% of the model's predictive power. A small number here indicates that the model is highly concentrated around a few key predictors, which can be a strategic advantage for targeting interventions effectively.",
        )
        
        st.caption(
            "**Pareto Principle in Action:** A small number of high-impact features explain most of the model's behavior. "
            "Focus interventions on these top predictors for maximum ROI."
        )
    
    st.divider()

    # ── Coefficient table (interactive) ──────────────────────────────────────────
    st.header("OLS Coefficient Table (TheLook)")

    if _coef_df is not None and "coefficient" in _coef_df.columns:
        st.subheader("Interactive Coefficient Explorer")
        with st.expander("ℹ️ What does this mean?", expanded=False):
            st.markdown(
                """
                #### Why This Slider?

                Your model has dozens of statistically significant predictors. Displaying all at once creates visual clutter.
                This slider lets you **zoom in on importance** — focusing analysis on what matters most.

                ---

                #### How It Works

                **Drag left (fewer features):** View only the strongest predictors by absolute coefficient magnitude.
                Reveals the "core story" — the factors with biggest impact on profit erosion.

                **Drag right (more features):** Progressively add more granular predictors (smaller effects).
                Explore secondary patterns and category-specific insights.

                **Dynamic range:** Slider automatically adjusts to your dataset size.

                ---

                #### What You'll Discover

                **The "Big Movers"** (left side of slider):
                - Return Frequency (most stable predictor)
                - Baseline erosion cost (Constant term)
                - High-impact product categories (Suits, Outerwear, Sport Coats)

                **Secondary Drivers** (right side of slider):
                - Low-cost categories (Socks, Hosiery, Underwear)
                - Basket size effects (economies of scale)
                - Niche category coefficients

                ---

                #### Practical Use

                Start **wide** (right) to see full picture. Then **focus narrow** (left) to identify 
                which 3-5 factors drive majority of erosion. These become your intervention targets.
                """
            )
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
