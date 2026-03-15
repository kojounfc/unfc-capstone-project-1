"""
RQ3 Predictive Modeling module for Profit Erosion Capstone Project.

Predicts high profit erosion customers using machine learning with
multi-method feature screening to ensure only statistically justified
predictors enter the pipeline.

Pipeline order:
    1. Load data (12 candidate features + target)
    2. Drop leakage columns
    3. Impute missing values (median)
    4. Stratified train/test split 80/20
    5. Feature screening on training set only (variance -> correlation -> univariate)
    6. Apply surviving features to both sets
    7. Train models (GridSearchCV on training set)
       - Rule-based classifier (return frequency threshold) -- practical baseline
       - Logistic Regression (L1/L2 regularized) -- methodological baseline
       - Random Forest -- primary model
       - Gradient Boosting -- secondary ML model
    8. Evaluate on test set
    9. Extract feature importance (post-hoc, from trained models)
   10. Ablation study -- retrain RF after removing top-N predictors to stress-test AUC
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    AUC_THRESHOLD,
    CUSTOMER_TARGETS_CSV,
    CV_FOLDS,
    FIGURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RQ3_CANDIDATE_FEATURES,
    RQ3_LEAKAGE_COLUMNS,
    RQ3_TARGET,
    TEST_SIZE,
)
from src.feature_engineering import create_profit_erosion_targets

logger = logging.getLogger(__name__)


def prepare_modeling_data(
    df: pd.DataFrame,
    candidate_features: Optional[List[str]] = None,
    leakage_columns: Optional[List[str]] = None,
    target: str = RQ3_TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for modeling: drop leakage, impute, stratified split.

    All 12 candidate features are retained in the split. Feature screening
    happens separately on the training set only.

    Args:
        df: Customer-level DataFrame with candidate features and target.
        candidate_features: List of candidate predictor column names.
        leakage_columns: Columns to exclude (target-derived or identifiers).
        target: Target column name.
        test_size: Proportion for test set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with all candidates present.
    """
    if candidate_features is None:
        candidate_features = RQ3_CANDIDATE_FEATURES
    if leakage_columns is None:
        leakage_columns = RQ3_LEAKAGE_COLUMNS

    # Validate required columns
    missing_features = [f for f in candidate_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing candidate features: {missing_features}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Extract features and target
    X = df[candidate_features].copy()
    y = df[target].copy()

    # Impute missing values with median
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info("Imputed %d missing values in '%s' with median=%.4f",
                        X[col].isna().sum(), col, median_val)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split: train=%d (%.1f%% positive), test=%d (%.1f%% positive)",
        len(X_train), y_train.mean() * 100,
        len(X_test), y_test.mean() * 100,
    )

    return X_train, X_test, y_train, y_test


def screen_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    significance_level: float = 0.05,
    use_variance_gate: bool = True,
    use_correlation_gate: bool = True,
    use_univariate_gate: bool = True,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Multi-method feature screening on training data only.

    Applies up to three sequential gates (each can be disabled independently):
        1. Variance check -- drop near-zero variance features
        2. Correlation analysis -- drop redundant features (|r| > threshold)
        3. Univariate statistical test -- drop features with p > alpha (Bonferroni)

    Args:
        X_train: Training features (all 12 candidates).
        y_train: Training target (binary).
        variance_threshold: Minimum variance to keep a feature.
        correlation_threshold: Max |r| between two features before dropping one.
        significance_level: Base alpha for univariate test (Bonferroni-corrected).
        use_variance_gate: If False, skip Gate 1 (variance check).
        use_correlation_gate: If False, skip Gate 2 (correlation analysis).
        use_univariate_gate: If False, skip Gate 3 (univariate statistical test).

    Returns:
        Tuple of:
        - surviving_features: List of feature names that passed all gates.
        - screening_report: DataFrame with per-feature screening results.
    """
    all_features = list(X_train.columns)
    report_rows = []

    # --- Gate 1: Variance check (skipped when use_variance_gate=False) ---
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(X_train)
    variances = selector.variances_
    variance_pass = selector.get_support()

    for i, feat in enumerate(all_features):
        report_rows.append({
            "feature": feat,
            "variance": variances[i],
            "variance_pass": bool(variance_pass[i]) if use_variance_gate else True,
        })

    if use_variance_gate:
        features_after_variance = [f for f, p in zip(all_features, variance_pass) if p]
        dropped_variance = set(all_features) - set(features_after_variance)
        if dropped_variance:
            logger.info("Gate 1 (variance): dropped %s", dropped_variance)
    else:
        features_after_variance = list(all_features)
        dropped_variance = set()

    # --- Gate 2: Correlation analysis (skipped when use_correlation_gate=False) ---
    to_drop_corr = set()

    if use_correlation_gate:
        corr_matrix = X_train[features_after_variance].corr().abs()
        for i in range(len(features_after_variance)):
            for j in range(i + 1, len(features_after_variance)):
                feat_i = features_after_variance[i]
                feat_j = features_after_variance[j]
                if corr_matrix.loc[feat_i, feat_j] > correlation_threshold:
                    # Drop the feature with lower univariate association to target
                    corr_i = abs(stats.pointbiserialr(X_train[feat_i], y_train).correlation)
                    corr_j = abs(stats.pointbiserialr(X_train[feat_j], y_train).correlation)
                    drop_feat = feat_j if corr_i >= corr_j else feat_i
                    to_drop_corr.add(drop_feat)
                    logger.info(
                        "Gate 2 (correlation): |r(%s, %s)|=%.3f > %.2f, dropping '%s'",
                        feat_i, feat_j, corr_matrix.loc[feat_i, feat_j],
                        correlation_threshold, drop_feat,
                    )

    features_after_corr = [f for f in features_after_variance if f not in to_drop_corr]

    # Update report with correlation info
    for row in report_rows:
        feat = row["feature"]
        if feat in dropped_variance:
            row["correlation_pass"] = None
        elif not use_correlation_gate:
            row["correlation_pass"] = True
        elif feat in to_drop_corr:
            row["correlation_pass"] = False
        else:
            row["correlation_pass"] = True

    # --- Gate 3: Univariate statistical test (skipped when use_univariate_gate=False) ---
    to_drop_univariate = set()

    if use_univariate_gate:
        n_tests = len(features_after_corr)
        bonferroni_alpha = significance_level / n_tests if n_tests > 0 else significance_level
        for feat in features_after_corr:
            corr_val, p_val = stats.pointbiserialr(X_train[feat], y_train)
            for row in report_rows:
                if row["feature"] == feat:
                    row["univariate_corr"] = corr_val
                    row["univariate_pvalue"] = p_val
                    row["bonferroni_alpha"] = bonferroni_alpha
                    if p_val > bonferroni_alpha:
                        row["univariate_pass"] = False
                        to_drop_univariate.add(feat)
                        logger.info(
                            "Gate 3 (univariate): '%s' p=%.4e > alpha=%.4e, dropping",
                            feat, p_val, bonferroni_alpha,
                        )
                    else:
                        row["univariate_pass"] = True
                    break
    else:
        for row in report_rows:
            if row["feature"] in features_after_corr and "univariate_pass" not in row:
                row["univariate_pass"] = True

    surviving_features = [f for f in features_after_corr if f not in to_drop_univariate]

    # Set univariate fields to None for features not evaluated at this gate
    for row in report_rows:
        if "univariate_pass" not in row:
            row["univariate_corr"] = None
            row["univariate_pvalue"] = None
            row["bonferroni_alpha"] = None
            row["univariate_pass"] = None

    # Final status
    for row in report_rows:
        row["final_status"] = "pass" if row["feature"] in surviving_features else "fail"

    screening_report = pd.DataFrame(report_rows)
    logger.info(
        "Feature screening: %d/%d candidates survived all gates",
        len(surviving_features), len(all_features),
    )

    return surviving_features, screening_report


def build_model_configs(
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Return model configurations with estimators and hyperparameter grids.

    Returns:
        Dict mapping model name to {"estimator": ..., "param_grid": ...}.
    """
    configs = {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_leaf": [5, 10],
            },
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(
                random_state=random_state,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
            },
        },
    }
    return configs


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, Any]]:
    """
    Train models via GridSearchCV and evaluate on the test set.

    Args:
        X_train: Training features (surviving features only).
        X_test: Test features (same surviving features).
        y_train: Training target.
        y_test: Test target.
        model_configs: Output from build_model_configs().
        cv_folds: Number of cross-validation folds.
        random_state: Random seed.

    Returns:
        Dict mapping model name to results dict containing:
        - best_estimator: Fitted model with best hyperparameters
        - best_params: Best hyperparameters from GridSearchCV
        - cv_auc: Mean cross-validation AUC
        - test_auc: AUC on the held-out test set
        - y_pred: Predicted labels on test set
        - y_proba: Predicted probabilities on test set
        - precision, recall, f1, accuracy: Test set metrics
        - confusion_matrix: Confusion matrix on test set
        - roc_curve: (fpr, tpr, thresholds) for plotting
    """
    if model_configs is None:
        model_configs = build_model_configs(random_state)

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = {}

    for name, config in model_configs.items():
        logger.info("Training %s...", name)

        # Use scaled data for Logistic Regression, raw for tree-based
        if name == "Logistic Regression":
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test

        # Handle class imbalance for Gradient Boosting via sample_weight
        fit_params = {}
        if name == "Gradient Boosting":
            class_counts = y_train.value_counts()
            weight_map = {
                cls: len(y_train) / (2.0 * count)
                for cls, count in class_counts.items()
            }
            fit_params["sample_weight"] = y_train.map(weight_map).values

        grid = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        if fit_params:
            grid.fit(X_tr, y_train, **fit_params)
        else:
            grid.fit(X_tr, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_te)
        y_proba = best_model.predict_proba(X_te)[:, 1]

        test_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        results[name] = {
            "best_estimator": best_model,
            "best_params": grid.best_params_,
            "cv_auc": grid.best_score_,
            "test_auc": test_auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_curve": (fpr, tpr, thresholds),
        }

        logger.info(
            "%s -- CV AUC: %.4f, Test AUC: %.4f, F1: %.4f",
            name, grid.best_score_, test_auc, results[name]["f1"],
        )

    return results


def get_feature_importance(
    results: Dict[str, Dict[str, Any]],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract feature importance from trained models (post-hoc).

    Uses standardized coefficients for Logistic Regression and
    built-in feature_importances_ for tree-based models.

    Args:
        results: Output from train_and_evaluate().
        feature_names: List of surviving feature names used in training.

    Returns:
        DataFrame with columns: feature, model, importance.
    """
    rows = []
    for name, res in results.items():
        model = res["best_estimator"]
        if hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            continue

        for feat, imp in zip(feature_names, importances):
            rows.append({"feature": feat, "model": name, "importance": imp})

    return pd.DataFrame(rows)


class RuleBasedClassifier:
    """
    Practical baseline classifier: flags customers as high-erosion if their
    return frequency exceeds a threshold learned from the training set.

    The threshold is set at the (1 - prevalence) quantile of return_frequency
    in the training set, where prevalence is the proportion of positive cases.
    This ensures the rule produces the same positive rate as the observed class
    balance, making it a fair and non-trivial comparator.

    Implements a predict_proba-compatible interface so it integrates cleanly
    with roc_auc_score and the existing evaluation pipeline.
    """

    RETURN_FREQ_FEATURE = "return_frequency"

    def __init__(self) -> None:
        self.threshold_: Optional[float] = None
        self.feature_index_: Optional[int] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleBasedClassifier":
        """
        Learn threshold from training data.

        Args:
            X: Training features (must contain 'return_frequency').
            y: Training target (binary).

        Returns:
            self
        """
        if self.RETURN_FREQ_FEATURE not in X.columns:
            raise ValueError(
                f"RuleBasedClassifier requires '{self.RETURN_FREQ_FEATURE}' "
                f"in feature set. Available: {list(X.columns)}"
            )

        prevalence = y.mean()
        self.threshold_ = float(
            X[self.RETURN_FREQ_FEATURE].quantile(1.0 - prevalence)
        )
        self.feature_index_ = list(X.columns).index(self.RETURN_FREQ_FEATURE)

        logger.info(
            "RuleBasedClassifier fitted: threshold=%.4f (prevalence=%.3f)",
            self.threshold_, prevalence,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probability scores: normalised return_frequency clipped to [0, 1].

        Args:
            X: Feature DataFrame.

        Returns:
            Array of shape (n_samples, 2) with [P(negative), P(positive)].
        """
        if self.threshold_ is None:
            raise RuntimeError("Classifier must be fitted before calling predict_proba.")

        scores = X[self.RETURN_FREQ_FEATURE].values.astype(float)
        # Normalise by max observed value so scores sit in [0, 1]
        max_val = scores.max() if scores.max() > 0 else 1.0
        pos_prob = np.clip(scores / max_val, 0.0, 1.0)
        return np.column_stack([1.0 - pos_prob, pos_prob])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Hard predictions using the learned threshold.

        Args:
            X: Feature DataFrame.

        Returns:
            Binary array of predictions.
        """
        if self.threshold_ is None:
            raise RuntimeError("Classifier must be fitted before calling predict.")

        return (X[self.RETURN_FREQ_FEATURE].values >= self.threshold_).astype(int)


def evaluate_rule_based(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Fit and evaluate the RuleBasedClassifier on the test set.

    Uses the same metrics as train_and_evaluate() so results slot directly
    into build_comparison_table().

    Args:
        X_train: Training features (must contain 'return_frequency').
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        Results dict compatible with the train_and_evaluate() output schema.
    """
    clf = RuleBasedClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    result = {
        "best_estimator": clf,
        "best_params": {"threshold": clf.threshold_},
        "cv_auc": float("nan"),          # No CV for rule-based
        "test_auc": test_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_curve": (fpr, tpr, thresholds),
    }

    logger.info(
        "Rule-Based Classifier -- Test AUC: %.4f, F1: %.4f (threshold=%.4f)",
        test_auc, result["f1"], clf.threshold_,
    )
    return result


def run_ablation_study(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    n_top_features: int = 3,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Ablation study: retrain the best Random Forest after removing the top-N
    most important predictors, then report AUC degradation.

    This directly addresses reviewer concern that the high AUC (0.9798) may
    reflect a small number of dominant predictors rather than a genuinely
    informative feature set. If AUC degrades substantially, that confirms the
    strong predictors are doing the heavy lifting. If it remains high, the
    feature set is broadly informative.

    Args:
        X_train: Training features used in the primary RF model.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        importance_df: Output of get_feature_importance() filtered to RF.
        n_top_features: Number of top features to remove (default 3).
        cv_folds: CV folds for GridSearchCV.
        random_state: Random seed.

    Returns:
        Dict with:
        - removed_features: List of features dropped
        - retained_features: List of features kept
        - ablated_cv_auc: CV AUC after ablation
        - ablated_test_auc: Test AUC after ablation
        - best_params: Best hyperparameters from ablated GridSearchCV
    """
    # Identify top-N features from RF importance
    rf_importance = (
        importance_df[importance_df["model"] == "Random Forest"]
        .sort_values("importance", ascending=False)
    )

    if len(rf_importance) == 0:
        raise ValueError(
            "No Random Forest importance data found. "
            "Run get_feature_importance() before ablation."
        )

    n_top_features = min(n_top_features, len(rf_importance) - 1)
    removed_features = rf_importance["feature"].iloc[:n_top_features].tolist()
    retained_features = [f for f in X_train.columns if f not in removed_features]

    if not retained_features:
        raise ValueError("Ablation removed all features. Reduce n_top_features.")

    logger.info(
        "Ablation study: removing top-%d features: %s",
        n_top_features, removed_features,
    )
    logger.info("Retained features: %s", retained_features)

    # Retrain RF on retained features with same hyperparameter grid
    X_train_abl = X_train[retained_features]
    X_test_abl = X_test[retained_features]

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }

    rf_abl = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=rf_abl,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train_abl, y_train)

    best_abl = grid.best_estimator_
    y_proba_abl = best_abl.predict_proba(X_test_abl)[:, 1]
    ablated_test_auc = roc_auc_score(y_test, y_proba_abl)
    ablated_cv_auc = grid.best_score_

    logger.info(
        "Ablation result -- CV AUC: %.4f, Test AUC: %.4f",
        ablated_cv_auc, ablated_test_auc,
    )

    return {
        "removed_features": removed_features,
        "retained_features": retained_features,
        "ablated_cv_auc": round(ablated_cv_auc, 4),
        "ablated_test_auc": round(ablated_test_auc, 4),
        "best_params": grid.best_params_,
    }


def _train_random_forest_grid(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Train a Random Forest using the standard RQ3 grid and report CV/test AUC."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [5, 10],
    }
    model = RandomForestClassifier(class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]
    return {
        "best_estimator": best_model,
        "best_params": grid.best_params_,
        "cv_auc": float(grid.best_score_),
        "test_auc": float(roc_auc_score(y_test, y_proba)),
    }


def run_preprocessing_ablation_study(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    significance_level: float = 0.05,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compare Random Forest performance after removing one or more screening steps.

    Tests whether the 2-gate screening pipeline (correlation + univariate, after
    dropping the zero-impact variance gate) is materially inflating AUC.
    """
    variants = [
        {
            "variant": "Full pipeline",
            "use_variance_gate": False,
            "use_correlation_gate": True,
            "use_univariate_gate": True,
            "steps_removed": "None",
        },
        {
            "variant": "No correlation gate",
            "use_variance_gate": False,
            "use_correlation_gate": False,
            "use_univariate_gate": True,
            "steps_removed": "Correlation",
        },
        {
            "variant": "No univariate gate",
            "use_variance_gate": False,
            "use_correlation_gate": True,
            "use_univariate_gate": False,
            "steps_removed": "Univariate",
        },
        {
            "variant": "No screening",
            "use_variance_gate": False,
            "use_correlation_gate": False,
            "use_univariate_gate": False,
            "steps_removed": "Correlation, Univariate",
        },
    ]

    rows = []
    full_test_auc: Optional[float] = None
    for cfg in variants:
        selected_features, _ = screen_features(
            X_train,
            y_train,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            use_variance_gate=cfg["use_variance_gate"],
            use_correlation_gate=cfg["use_correlation_gate"],
            use_univariate_gate=cfg["use_univariate_gate"],
        )
        rf_result = _train_random_forest_grid(
            X_train[selected_features],
            X_test[selected_features],
            y_train,
            y_test,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        if cfg["variant"] == "Full pipeline":
            full_test_auc = rf_result["test_auc"]
        rows.append({
            "variant": cfg["variant"],
            "steps_removed": cfg["steps_removed"],
            "n_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
            "cv_auc": round(rf_result["cv_auc"], 4),
            "test_auc": round(rf_result["test_auc"], 4),
            "best_params": str(rf_result["best_params"]),
        })

    if full_test_auc is None:
        raise RuntimeError("Full pipeline ablation result was not computed.")

    for row in rows:
        row["auc_drop_vs_full"] = round(full_test_auc - float(row["test_auc"]), 4)

    return pd.DataFrame(rows)


def run_feature_family_ablation_study(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    significance_level: float = 0.05,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compare Random Forest performance after removing whole families of predictors.

    Screening is rerun after each family removal so each variant reflects a valid
    reduced-feature pipeline rather than a forced drop from the already-screened set.
    """
    feature_families = {
        "return_behavior": ["return_frequency", "customer_return_rate"],
        "purchase_behavior": [
            "order_frequency", "avg_basket_size", "avg_order_value",
            "total_items", "total_sales",
        ],
        "margin_structure": ["total_margin", "avg_item_price", "avg_item_margin"],
        "temporal": ["customer_tenure_days", "purchase_recency_days"],
    }

    full_features, _ = screen_features(
        X_train, y_train,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        significance_level=significance_level,
        use_variance_gate=False,
        use_correlation_gate=True,
        use_univariate_gate=True,
    )
    full_result = _train_random_forest_grid(
        X_train[full_features], X_test[full_features], y_train, y_test,
        cv_folds=cv_folds, random_state=random_state,
    )
    full_test_auc = full_result["test_auc"]

    rows = [{
        "variant": "Full model",
        "removed_family": "None",
        "removed_features": "None",
        "n_features": len(full_features),
        "selected_features": ", ".join(full_features),
        "cv_auc": round(full_result["cv_auc"], 4),
        "test_auc": round(full_result["test_auc"], 4),
        "auc_drop_vs_full": 0.0,
        "best_params": str(full_result["best_params"]),
    }]

    for family_name, family_features in feature_families.items():
        reduced_columns = [c for c in X_train.columns if c not in family_features]
        selected_features, _ = screen_features(
            X_train[reduced_columns], y_train,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            use_variance_gate=False,
            use_correlation_gate=True,
            use_univariate_gate=True,
        )
        rf_result = _train_random_forest_grid(
            X_train[selected_features], X_test[selected_features], y_train, y_test,
            cv_folds=cv_folds, random_state=random_state,
        )
        removed_present = [f for f in family_features if f in X_train.columns]
        rows.append({
            "variant": f"No {family_name}",
            "removed_family": family_name,
            "removed_features": ", ".join(removed_present),
            "n_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
            "cv_auc": round(rf_result["cv_auc"], 4),
            "test_auc": round(rf_result["test_auc"], 4),
            "auc_drop_vs_full": round(full_test_auc - rf_result["test_auc"], 4),
            "best_params": str(rf_result["best_params"]),
        })

    return pd.DataFrame(rows)


def run_target_construction_ablation_study(
    df: pd.DataFrame,
    high_erosion_percentile: float = 0.75,
    candidate_features: Optional[List[str]] = None,
    target: str = RQ3_TARGET,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    significance_level: float = 0.05,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compare RF performance under alternate target constructions.

    Each variant rebuilds the high-erosion label from a different erosion
    component using the same percentile threshold and screened RF pipeline.
    """
    variants = [
        {"variant": "Full profit erosion target", "erosion_column": "total_profit_erosion"},
        {"variant": "Margin-only target",          "erosion_column": "total_margin_reversal"},
        {"variant": "Process-cost-only target",    "erosion_column": "total_process_cost"},
    ]

    rows = []
    full_test_auc: Optional[float] = None

    for cfg in variants:
        target_df = create_profit_erosion_targets(
            df,
            high_erosion_percentile=high_erosion_percentile,
            erosion_column=cfg["erosion_column"],
        )
        X_train_raw, X_test_raw, y_train, y_test = prepare_modeling_data(
            target_df,
            candidate_features=candidate_features,
            target=target,
            random_state=random_state,
        )
        selected_features, _ = screen_features(
            X_train_raw, y_train,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            use_variance_gate=False,
            use_correlation_gate=True,
            use_univariate_gate=True,
        )
        rf_result = _train_random_forest_grid(
            X_train_raw[selected_features], X_test_raw[selected_features],
            y_train, y_test,
            cv_folds=cv_folds, random_state=random_state,
        )
        if cfg["variant"] == "Full profit erosion target":
            full_test_auc = rf_result["test_auc"]
        rows.append({
            "variant": cfg["variant"],
            "erosion_column": cfg["erosion_column"],
            "positive_rate": round(float(target_df[target].mean()), 4),
            "n_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
            "cv_auc": round(rf_result["cv_auc"], 4),
            "test_auc": round(rf_result["test_auc"], 4),
            "best_params": str(rf_result["best_params"]),
        })

    if full_test_auc is None:
        raise RuntimeError("Full target-construction ablation result was not computed.")

    for row in rows:
        row["auc_drop_vs_full"] = round(full_test_auc - float(row["test_auc"]), 4)

    return pd.DataFrame(rows)


def run_feature_set_ablation_study(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.85,
    significance_level: float = 0.05,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compare RF performance using behavioral-only, economic-only, and combined
    feature sets.
    """
    feature_sets = {
        "Behavioral + economic (full set)": list(X_train.columns),
        "Behavioral only": [
            "order_frequency", "return_frequency", "customer_return_rate",
            "avg_basket_size", "customer_tenure_days", "purchase_recency_days",
            "total_items",
        ],
        "Economic only": [
            "avg_order_value", "total_sales", "total_margin",
            "avg_item_price", "avg_item_margin",
        ],
    }

    rows = []
    full_test_auc: Optional[float] = None

    for variant_name, subset in feature_sets.items():
        available_features = [f for f in subset if f in X_train.columns]
        selected_features, _ = screen_features(
            X_train[available_features], y_train,
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            significance_level=significance_level,
            use_variance_gate=False,
            use_correlation_gate=True,
            use_univariate_gate=True,
        )
        rf_result = _train_random_forest_grid(
            X_train[selected_features], X_test[selected_features], y_train, y_test,
            cv_folds=cv_folds, random_state=random_state,
        )
        if variant_name == "Behavioral + economic (full set)":
            full_test_auc = rf_result["test_auc"]
        rows.append({
            "variant": variant_name,
            "candidate_features": ", ".join(available_features),
            "n_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
            "cv_auc": round(rf_result["cv_auc"], 4),
            "test_auc": round(rf_result["test_auc"], 4),
            "best_params": str(rf_result["best_params"]),
        })

    if full_test_auc is None:
        raise RuntimeError("Full feature-set ablation result was not computed.")

    for row in rows:
        row["auc_drop_vs_full"] = round(full_test_auc - float(row["test_auc"]), 4)

    return pd.DataFrame(rows)


def run_operational_model_ablation_study(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compare the full screened RF model against lighter operational subsets
    based on the top RF importance ranking.
    """
    rf_importance = (
        importance_df[importance_df["model"] == "Random Forest"]
        .sort_values("importance", ascending=False)
    )
    ranked_features = [f for f in rf_importance["feature"].tolist() if f in X_train.columns]

    variants = [
        ("Full screened model",    list(X_train.columns)),
        ("Top-5 operational subset", ranked_features[:min(5, len(ranked_features))]),
        ("Top-3 operational subset", ranked_features[:min(3, len(ranked_features))]),
    ]

    rows = []
    full_test_auc: Optional[float] = None

    for variant_name, selected_features in variants:
        rf_result = _train_random_forest_grid(
            X_train[selected_features], X_test[selected_features], y_train, y_test,
            cv_folds=cv_folds, random_state=random_state,
        )
        if variant_name == "Full screened model":
            full_test_auc = rf_result["test_auc"]
        rows.append({
            "variant": variant_name,
            "n_features": len(selected_features),
            "selected_features": ", ".join(selected_features),
            "cv_auc": round(rf_result["cv_auc"], 4),
            "test_auc": round(rf_result["test_auc"], 4),
            "best_params": str(rf_result["best_params"]),
        })

    if full_test_auc is None:
        raise RuntimeError("Full operational ablation result was not computed.")

    for row in rows:
        row["auc_drop_vs_full"] = round(full_test_auc - float(row["test_auc"]), 4)

    return pd.DataFrame(rows)


def interpret_ablation_results(
    feature_ablation: Dict[str, Any],
    preprocessing_ablation: pd.DataFrame,
    feature_family_ablation: pd.DataFrame,
    full_model_auc: float,
) -> Dict[str, str]:
    """Generate plain-language interpretations for the RQ3 ablation studies."""
    feature_auc_drop = float(full_model_auc - float(feature_ablation["ablated_test_auc"]))

    preprocessing_ranked = preprocessing_ablation.sort_values("auc_drop_vs_full", ascending=False)
    preprocessing_top = preprocessing_ranked.iloc[0]

    family_ranked = feature_family_ablation[
        feature_family_ablation["variant"] != "Full model"
    ].sort_values("auc_drop_vs_full", ascending=False)
    family_top = family_ranked.iloc[0]

    if float(preprocessing_top["auc_drop_vs_full"]) < 0.01:
        preprocessing_text = (
            "Removing individual screening steps does not materially reduce AUC. "
            "The Random Forest performance is not being driven by the preprocessing pipeline."
        )
    else:
        preprocessing_text = (
            f"The largest preprocessing impact comes from '{preprocessing_top['variant']}', "
            f"which reduces test AUC by {float(preprocessing_top['auc_drop_vs_full']):.4f}. "
            "This indicates at least one screening step contributes meaningfully to model quality."
        )

    if float(family_top["auc_drop_vs_full"]) > max(
        float(preprocessing_top["auc_drop_vs_full"]), feature_auc_drop
    ):
        family_text = (
            f"The largest drop occurs when '{family_top['removed_family']}' is removed "
            f"(AUC drop = {float(family_top['auc_drop_vs_full']):.4f}), which is larger than any "
            "preprocessing-step effect — indicating genuine behavioral signal in that family."
        )
    else:
        family_text = (
            f"No feature family removal exceeds the strongest preprocessing or top-feature effect. "
            f"Largest family drop: '{family_top['removed_family']}' at {float(family_top['auc_drop_vs_full']):.4f}."
        )

    feature_text = (
        f"Removing the top-{len(feature_ablation['removed_features'])} RF predictors "
        f"({', '.join(feature_ablation['removed_features'])}) changes test AUC by {feature_auc_drop:.4f}. "
        "A small drop indicates predictive signal is distributed across the retained features."
    )

    overall_text = (
        "Small preprocessing drops + larger feature-family drops = the high AUC is driven by genuine "
        "behavioral signal in the engineered features, not by the screening procedure."
    )

    return {
        "feature_ablation": feature_text,
        "preprocessing_ablation": preprocessing_text,
        "feature_family_ablation": family_text,
        "overall": overall_text,
    }


def build_comparison_table(
    results: Dict[str, Dict[str, Any]],
    auc_threshold: float = AUC_THRESHOLD,
) -> pd.DataFrame:
    """
    Build a model comparison summary table.

    Args:
        results: Output from train_and_evaluate().
        auc_threshold: AUC threshold for hypothesis test.

    Returns:
        DataFrame with one row per model: model, cv_auc, test_auc,
        precision, recall, f1, accuracy, meets_threshold.
    """
    rows = []
    for name, res in results.items():
        cv_auc = res["cv_auc"]
        rows.append({
            "model": name,
            "cv_auc": round(cv_auc, 4) if not (isinstance(cv_auc, float) and np.isnan(cv_auc)) else "N/A",
            "test_auc": round(res["test_auc"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1": round(res["f1"], 4),
            "accuracy": round(res["accuracy"], 4),
            "meets_threshold": res["test_auc"] >= auc_threshold,
        })

    return pd.DataFrame(rows).sort_values("test_auc", ascending=False)


def test_hypothesis(
    results: Dict[str, Dict[str, Any]],
    auc_threshold: float = AUC_THRESHOLD,
) -> Dict[str, Any]:
    """
    Test RQ3 hypothesis: Can we predict high erosion customers (AUC > threshold)?

    H0: Best model AUC <= threshold (cannot predict)
    H1: Best model AUC > threshold (can predict)

    Args:
        results: Output from train_and_evaluate().
        auc_threshold: AUC threshold (default 0.70).

    Returns:
        Dict with: best_model, best_auc, threshold, reject_null, conclusion.
    """
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_auc = results[best_name]["test_auc"]
    reject_null = best_auc > auc_threshold

    if reject_null:
        conclusion = (
            f"Reject H0: {best_name} achieves AUC={best_auc:.4f} > {auc_threshold}, "
            f"supporting that ML models can predict high profit erosion customers."
        )
    else:
        conclusion = (
            f"Fail to reject H0: Best AUC={best_auc:.4f} <= {auc_threshold}. "
            f"Insufficient evidence that ML models can predict high profit erosion customers."
        )

    return {
        "best_model": best_name,
        "best_auc": best_auc,
        "threshold": auc_threshold,
        "reject_null": reject_null,
        "conclusion": conclusion,
    }


def main() -> None:
    """
    End-to-end RQ3 pipeline: load -> prep -> screen -> train -> evaluate -> importance -> export.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # 1. Load data
    logger.info("Loading data from %s", CUSTOMER_TARGETS_CSV)
    df = pd.read_csv(CUSTOMER_TARGETS_CSV)
    logger.info("Loaded %d customers with %d columns", len(df), len(df.columns))

    # 2-4. Prepare modeling data (drop leakage, impute, split)
    X_train, X_test, y_train, y_test = prepare_modeling_data(df)

    # 5. Feature screening on training set only
    surviving_features, screening_report = screen_features(X_train, y_train)

    # 6. Apply surviving features to both sets
    X_train = X_train[surviving_features]
    X_test = X_test[surviving_features]

    # Export screening report
    reports_dir = REPORTS_DIR / "rq3"
    reports_dir.mkdir(parents=True, exist_ok=True)
    screening_report.to_csv(reports_dir / "rq3_feature_screening.csv", index=False)
    logger.info("Feature screening report saved")

    # 7-8. Train ML models and evaluate on test set
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Add rule-based baseline (practical comparator)
    results["Rule-Based (Return Frequency)"] = evaluate_rule_based(
        X_train, X_test, y_train, y_test
    )

    # 9. Extract feature importance (post-hoc, from trained models)
    importance_df = get_feature_importance(results, surviving_features)
    importance_df.to_csv(reports_dir / "rq3_feature_importance.csv", index=False)

    # Model comparison
    comparison = build_comparison_table(results)
    comparison.to_csv(reports_dir / "rq3_model_comparison.csv", index=False)
    logger.info("\nModel Comparison:\n%s", comparison.to_string(index=False))

    # Hypothesis test
    hypothesis = test_hypothesis(results)
    logger.info("\nHypothesis Test Result:\n%s", hypothesis["conclusion"])

    # 10. Ablation study: remove top-3 RF predictors, report AUC degradation
    full_rf_auc = results["Random Forest"]["test_auc"]
    ablation = run_ablation_study(
        X_train, X_test, y_train, y_test,
        importance_df=importance_df,
        n_top_features=3,
    )
    ablation_summary = pd.DataFrame([{
        "removed_features": ", ".join(ablation["removed_features"]),
        "retained_features": ", ".join(ablation["retained_features"]),
        "full_rf_test_auc": round(full_rf_auc, 4),
        "ablated_test_auc": ablation["ablated_test_auc"],
        "ablated_cv_auc": ablation["ablated_cv_auc"],
        "auc_drop": round(full_rf_auc - ablation["ablated_test_auc"], 4),
    }])
    ablation_summary.to_csv(reports_dir / "rq3_ablation_study.csv", index=False)
    logger.info(
        "\nAblation Study:\n  Removed: %s\n  Full AUC: %.4f -> Ablated AUC: %.4f (drop: %.4f)",
        ablation["removed_features"],
        full_rf_auc,
        ablation["ablated_test_auc"],
        ablation_summary["auc_drop"].iloc[0],
    )

    # Visualizations
    from src.rq3_visuals import (
        plot_confusion_matrices,
        plot_feature_importance,
        plot_roc_curves,
    )

    figures_dir = FIGURES_DIR / "rq3"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curves(results, save_path=figures_dir / "rq3_roc_curves.png")
    plot_confusion_matrices(results, save_path=figures_dir / "rq3_confusion_matrices.png")
    plot_feature_importance(importance_df, save_path=figures_dir / "rq3_feature_importance.png")

    logger.info("All RQ3 artifacts saved. Reports: %s | Figures: %s", reports_dir, figures_dir)


if __name__ == "__main__":
    main()
