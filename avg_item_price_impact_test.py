"""
Quick test: Compare avg_item_price computation strategies and their
downstream impact on SSL feature screening, model performance, and
feature importance.

NOT production code — exploratory analysis only.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.config import RQ3_CANDIDATE_FEATURES
from src.feature_engineering import create_profit_erosion_targets
from src.rq3_modeling import screen_features
from src.rq3_validation import engineer_ssl_account_features


def full_pipeline(acct_df, label):
    """Run screening + RF training + evaluation on an account DataFrame."""
    print("=" * 70)
    print(label)
    print("=" * 70)

    aip = acct_df["avg_item_price"]
    coverage = aip.notna().mean()
    print(
        f"avg_item_price coverage: {aip.notna().sum():,} / {len(acct_df):,}"
        f" ({coverage*100:.1f}%)"
    )
    print(
        f"avg_item_price stats: mean=${aip.mean():.2f},"
        f" median=${aip.median():.2f}, std=${aip.std():.2f}"
    )
    print()

    # Create targets
    targets = create_profit_erosion_targets(
        acct_df.rename(columns={"total_loss": "total_profit_erosion"}),
        high_erosion_percentile=0.75,
        erosion_column="total_profit_erosion",
    )

    feature_cols = [c for c in RQ3_CANDIDATE_FEATURES if c in targets.columns]
    X = targets[feature_cols].copy()
    y = targets["is_high_erosion_customer"].copy()

    # Imputation
    for col in X.columns:
        n_miss = X[col].isna().sum()
        if n_miss > 0:
            pct = n_miss / len(X) * 100
            print(f"  Imputed {col}: {n_miss:,} ({pct:.1f}%)")
            X[col] = X[col].fillna(X[col].median())

    print()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Screen features
    surviving, report = screen_features(X_train, y_train)

    print("  Screening results:")
    for feat in feature_cols:
        s = "PASS" if feat in surviving else "FAIL"
        marker = " <<<" if feat == "avg_item_price" else ""
        print(f"    {feat:<25} {s}{marker}")
    print(f"  Surviving: {len(surviving)} features")
    print()

    # Train RF model
    X_train_s = X_train[surviving]
    X_test_s = X_test[surviving]

    rf = RandomForestClassifier(class_weight="balanced", random_state=42)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }
    gs = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    gs.fit(X_train_s, y_train)

    y_prob = gs.predict_proba(X_test_s)[:, 1]
    y_pred = gs.predict(X_test_s)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print("  Model Performance (RF, GridSearchCV 5-fold):")
    print(f"    CV AUC:     {gs.best_score_:.4f}")
    print(f"    Test AUC:   {auc:.4f}")
    print(f"    F1:         {f1:.4f}")
    print(f"    Precision:  {prec:.4f}")
    print(f"    Recall:     {rec:.4f}")
    print(f"    Best params: {gs.best_params_}")
    print()

    # Feature importance
    importances = pd.Series(
        gs.best_estimator_.feature_importances_,
        index=surviving,
    ).sort_values(ascending=False)
    print("  Feature Importance:")
    for feat, imp in importances.items():
        marker = " <<<" if feat == "avg_item_price" else ""
        print(f"    {feat:<25} {imp:.4f}{marker}")
    print()

    return {
        "surviving": surviving,
        "auc": auc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "cv_auc": gs.best_score_,
        "importances": importances,
        "coverage": coverage,
    }


def main():
    df_raw = pd.read_csv(
        "data/raw/SSL_Returns_df_yoy.csv", parse_dates=["Booked Date"]
    )
    print(f"SSL data: {len(df_raw):,} lines, {df_raw['Bill To Act #'].nunique():,} accounts\n")

    # ==================================================================
    # SCENARIO 1: Current implementation
    # ==================================================================
    acct_current = engineer_ssl_account_features(df_raw)
    r1 = full_pipeline(
        acct_current, "SCENARIO 1: CURRENT (CreditReturn Sales, RETURN only)"
    )

    # ==================================================================
    # SCENARIO 2: S5 — RefSale ALL lines + CreditReturn fallback
    # ==================================================================
    # Recompute _item_price at line level with S5 logic
    df_work = df_raw.copy()
    is_return = df_work["Sales_Type"] == "RETURN"
    has_qty = df_work["Ordered Qty"].abs() > 0
    has_ref = df_work["Reference Sale Amount"].notna() & (
        df_work["Reference Sale Amount"] > 0
    )

    df_work["_item_price_s5"] = np.where(
        has_qty & has_ref,
        df_work["Reference Sale Amount"] / df_work["Ordered Qty"].abs(),
        np.where(
            is_return & has_qty,
            df_work["CreditReturn Sales"].abs() / df_work["Ordered Qty"].abs(),
            np.nan,
        ),
    )

    # Account-level mean
    new_avg = df_work.groupby("Bill To Act #")["_item_price_s5"].mean()

    # Get base features from current engineer function, overwrite avg_item_price
    acct_s5 = engineer_ssl_account_features(df_raw)

    # Find the account ID column
    if "Bill To Act #" in acct_s5.columns:
        acct_col = "Bill To Act #"
    else:
        # engineer function may rename it — check all columns
        acct_col = None
        for col in acct_s5.columns:
            if acct_s5[col].nunique() == len(acct_s5):
                # likely the account ID
                acct_col = col
                break

    if acct_col:
        acct_s5["avg_item_price"] = acct_s5[acct_col].map(new_avg)
    else:
        # Try index
        acct_s5["avg_item_price"] = acct_s5.index.map(new_avg)

    r2 = full_pipeline(
        acct_s5, "SCENARIO 2: S5 (RefSale ALL lines + CreditReturn fallback)"
    )

    # ==================================================================
    # SIDE-BY-SIDE COMPARISON
    # ==================================================================
    print("=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print()
    hdr = f"{'Metric':<30} {'Current':>15} {'S5 Proposed':>15} {'Delta':>10}"
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'avg_item_price coverage':<30}"
        f" {r1['coverage']*100:>14.1f}%"
        f" {r2['coverage']*100:>14.1f}%"
        f" {(r2['coverage']-r1['coverage'])*100:>+9.1f}pp"
    )
    print(
        f"{'Imputation rate':<30}"
        f" {(1-r1['coverage'])*100:>14.1f}%"
        f" {(1-r2['coverage'])*100:>14.1f}%"
        f" {(r1['coverage']-r2['coverage'])*100:>+9.1f}pp"
    )
    print(
        f"{'Surviving features':<30}"
        f" {len(r1['surviving']):>15}"
        f" {len(r2['surviving']):>15}"
        f" {len(r2['surviving'])-len(r1['surviving']):>+10}"
    )
    print(
        f"{'CV AUC':<30}"
        f" {r1['cv_auc']:>15.4f}"
        f" {r2['cv_auc']:>15.4f}"
        f" {r2['cv_auc']-r1['cv_auc']:>+10.4f}"
    )
    print(
        f"{'Test AUC':<30}"
        f" {r1['auc']:>15.4f}"
        f" {r2['auc']:>15.4f}"
        f" {r2['auc']-r1['auc']:>+10.4f}"
    )
    print(
        f"{'F1':<30}"
        f" {r1['f1']:>15.4f}"
        f" {r2['f1']:>15.4f}"
        f" {r2['f1']-r1['f1']:>+10.4f}"
    )
    print(
        f"{'Precision':<30}"
        f" {r1['prec']:>15.4f}"
        f" {r2['prec']:>15.4f}"
        f" {r2['prec']-r1['prec']:>+10.4f}"
    )
    print(
        f"{'Recall':<30}"
        f" {r1['rec']:>15.4f}"
        f" {r2['rec']:>15.4f}"
        f" {r2['rec']-r1['rec']:>+10.4f}"
    )
    print()

    # Feature screening comparison
    print("Feature Screening Comparison:")
    print(f"  {'Feature':<25} {'Current':>10} {'S5':>10} {'Change':>10}")
    print("  " + "-" * 55)
    for feat in RQ3_CANDIDATE_FEATURES:
        c_s = "PASS" if feat in r1["surviving"] else "FAIL"
        s_s = "PASS" if feat in r2["surviving"] else "FAIL"
        changed = " *CHANGED*" if c_s != s_s else ""
        print(f"  {feat:<25} {c_s:>10} {s_s:>10}{changed}")
    print()

    # avg_item_price importance comparison
    for label, r in [("Current", r1), ("S5", r2)]:
        if "avg_item_price" in r["importances"].index:
            rank = list(r["importances"].index).index("avg_item_price") + 1
            imp = r["importances"]["avg_item_price"]
            print(f"  avg_item_price ({label}): rank={rank}/{len(r['importances'])}, importance={imp:.4f}")
        else:
            print(f"  avg_item_price ({label}): NOT in surviving features")


if __name__ == "__main__":
    main()
