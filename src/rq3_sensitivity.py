"""
RQ3 Sensitivity Analysis module for Profit Erosion Capstone Project.

Tests robustness of RQ3 predictive findings against two modeling choices:
    1. Processing cost base ($12 default) — swept $8–$18
    2. High-erosion threshold (75th percentile default) — swept 50th–90th

Only the target labels shift across scenarios. The 12 predictor features
are computed from transactions and do not depend on processing cost or threshold.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    RANDOM_STATE,
    RQ3_CANDIDATE_FEATURES,
    RQ3_LEAKAGE_COLUMNS,
    SENSITIVITY_BASE_COSTS,
    SENSITIVITY_THRESHOLDS,
)
from src.feature_engineering import (
    DEFAULT_COST_COMPONENTS,
    aggregate_profit_erosion_by_customer,
    calculate_profit_erosion,
    create_profit_erosion_targets,
)
from src.rq3_modeling import (
    prepare_modeling_data,
    screen_features,
    train_and_evaluate,
)

logger = logging.getLogger(__name__)

# Minimal RF-only config for sensitivity sweeps (established best model)
SENSITIVITY_MODEL_CONFIGS = {
    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_leaf": [5, 10],
        },
    },
}


def _scale_cost_components(
    base_components: Dict[str, float],
    target_total: float,
) -> Dict[str, float]:
    """Scale cost components proportionally to sum to a new base cost.

    Args:
        base_components: Original cost breakdown (e.g., DEFAULT_COST_COMPONENTS).
        target_total: Desired total cost (e.g., 8.0, 10.0, 18.0).

    Returns:
        New cost components dict summing to target_total.
    """
    current_total = sum(base_components.values())
    if current_total == 0:
        return {k: 0.0 for k in base_components}
    scale_factor = target_total / current_total
    return {k: round(v * scale_factor, 4) for k, v in base_components.items()}


def run_cost_sensitivity(
    returned_items_df: pd.DataFrame,
    customer_behavioral_df: pd.DataFrame,
    base_costs: Optional[List[float]] = None,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[float, pd.Series]]:
    """Run sensitivity analysis over processing cost base values.

    For each base cost, recalculates profit erosion, creates targets at the
    75th percentile, merges with behavioral features, and trains/evaluates
    a Random Forest classifier.

    Args:
        returned_items_df: Item-level DataFrame of returned items with
            columns needed by calculate_profit_erosion() (item_margin,
            category, order_id, user_id, etc.).
        customer_behavioral_df: Customer-level behavioral features from
            engineer_customer_behavioral_features(). Computed once and
            reused across scenarios (features don't change with cost).
        base_costs: List of base cost values to test.
            Defaults to config SENSITIVITY_BASE_COSTS.
        model_configs: Model configs for train_and_evaluate().
            Defaults to RF-only SENSITIVITY_MODEL_CONFIGS.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of:
        - summary_df: DataFrame with one row per cost scenario containing
          base_cost, best_auc, f1, precision, recall, threshold_value,
          n_positive, n_total, positive_rate, surviving_features.
        - labels_dict: Dict mapping base_cost -> Series of target labels
          (indexed by user_id) for label stability analysis.
    """
    if base_costs is None:
        base_costs = SENSITIVITY_BASE_COSTS
    if model_configs is None:
        model_configs = SENSITIVITY_MODEL_CONFIGS

    summary_rows = []
    labels_dict = {}

    for cost in base_costs:
        logger.info("=== Cost sensitivity: base_cost=$%.2f ===", cost)

        # 1. Scale cost components proportionally
        scaled_components = _scale_cost_components(DEFAULT_COST_COMPONENTS, cost)

        # 2. Recalculate profit erosion with new cost
        erosion_df = calculate_profit_erosion(
            returned_items_df, cost_components=scaled_components
        )

        # 3. Aggregate to customer level
        customer_erosion = aggregate_profit_erosion_by_customer(erosion_df)

        # 4. Merge with behavioral features
        merged = customer_behavioral_df.merge(
            customer_erosion[["user_id", "total_profit_erosion"]],
            on="user_id",
            how="inner",
        )

        # 5. Create targets at 75th percentile
        targets_df = create_profit_erosion_targets(
            merged, high_erosion_percentile=0.75
        )

        # Store labels for stability analysis
        labels_dict[cost] = targets_df.set_index("user_id")[
            "is_high_erosion_customer"
        ]

        n_positive = targets_df["is_high_erosion_customer"].sum()
        n_total = len(targets_df)
        threshold_value = targets_df["total_profit_erosion"].quantile(0.75)

        # 6. Prepare modeling data, screen, train, evaluate
        try:
            X_train, X_test, y_train, y_test = prepare_modeling_data(
                targets_df, random_state=random_state
            )
            surviving, _ = screen_features(X_train, y_train)

            if len(surviving) == 0:
                logger.warning(
                    "No features survived screening for cost=$%.2f", cost
                )
                summary_rows.append({
                    "base_cost": cost,
                    "best_auc": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "threshold_value": threshold_value,
                    "n_positive": n_positive,
                    "n_total": n_total,
                    "positive_rate": n_positive / n_total,
                    "n_surviving_features": 0,
                    "surviving_features": "",
                })
                continue

            X_train_s = X_train[surviving]
            X_test_s = X_test[surviving]

            results = train_and_evaluate(
                X_train_s, X_test_s, y_train, y_test,
                model_configs=model_configs,
            )

            # Extract best model metrics
            best_name = max(results, key=lambda k: results[k]["test_auc"])
            best = results[best_name]

            summary_rows.append({
                "base_cost": cost,
                "best_auc": best["test_auc"],
                "f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "threshold_value": threshold_value,
                "n_positive": n_positive,
                "n_total": n_total,
                "positive_rate": n_positive / n_total,
                "n_surviving_features": len(surviving),
                "surviving_features": ", ".join(surviving),
            })

        except Exception as e:
            logger.error("Cost sensitivity failed for $%.2f: %s", cost, e)
            summary_rows.append({
                "base_cost": cost,
                "best_auc": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "threshold_value": threshold_value,
                "n_positive": n_positive,
                "n_total": n_total,
                "positive_rate": n_positive / n_total,
                "n_surviving_features": 0,
                "surviving_features": "",
            })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(
        "Cost sensitivity complete: %d scenarios, AUC range [%.4f, %.4f]",
        len(summary_df),
        summary_df["best_auc"].min(),
        summary_df["best_auc"].max(),
    )

    return summary_df, labels_dict


def run_threshold_sensitivity(
    customer_targets_base_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Run sensitivity analysis over high-erosion threshold percentiles.

    Uses the baseline ($12) profit erosion data and varies only the
    percentile threshold for the binary target variable.

    Args:
        customer_targets_base_df: Customer-level DataFrame with behavioral
            features and total_profit_erosion column (baseline $12 data).
            Must contain all RQ3_CANDIDATE_FEATURES plus total_profit_erosion.
        thresholds: List of percentile thresholds to test (0-1 scale).
            Defaults to config SENSITIVITY_THRESHOLDS.
        model_configs: Model configs for train_and_evaluate().
            Defaults to RF-only SENSITIVITY_MODEL_CONFIGS.
        random_state: Random seed for reproducibility.

    Returns:
        Summary DataFrame with one row per threshold containing
        threshold, best_auc, f1, precision, recall, positive_rate,
        n_positive, n_total, n_surviving_features, surviving_features.
    """
    if thresholds is None:
        thresholds = SENSITIVITY_THRESHOLDS
    if model_configs is None:
        model_configs = SENSITIVITY_MODEL_CONFIGS

    summary_rows = []

    for pct in thresholds:
        logger.info("=== Threshold sensitivity: percentile=%.2f ===", pct)

        # 1. Create targets at this percentile
        targets_df = create_profit_erosion_targets(
            customer_targets_base_df, high_erosion_percentile=pct
        )

        n_positive = targets_df["is_high_erosion_customer"].sum()
        n_total = len(targets_df)

        # 2. Prepare modeling data, screen, train, evaluate
        try:
            X_train, X_test, y_train, y_test = prepare_modeling_data(
                targets_df, random_state=random_state
            )
            surviving, _ = screen_features(X_train, y_train)

            if len(surviving) == 0:
                logger.warning(
                    "No features survived screening for threshold=%.2f", pct
                )
                summary_rows.append({
                    "threshold": pct,
                    "best_auc": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "positive_rate": n_positive / n_total,
                    "n_positive": n_positive,
                    "n_total": n_total,
                    "n_surviving_features": 0,
                    "surviving_features": "",
                })
                continue

            X_train_s = X_train[surviving]
            X_test_s = X_test[surviving]

            results = train_and_evaluate(
                X_train_s, X_test_s, y_train, y_test,
                model_configs=model_configs,
            )

            best_name = max(results, key=lambda k: results[k]["test_auc"])
            best = results[best_name]

            summary_rows.append({
                "threshold": pct,
                "best_auc": best["test_auc"],
                "f1": best["f1"],
                "precision": best["precision"],
                "recall": best["recall"],
                "positive_rate": n_positive / n_total,
                "n_positive": n_positive,
                "n_total": n_total,
                "n_surviving_features": len(surviving),
                "surviving_features": ", ".join(surviving),
            })

        except Exception as e:
            logger.error(
                "Threshold sensitivity failed for %.2f: %s", pct, e
            )
            summary_rows.append({
                "threshold": pct,
                "best_auc": np.nan,
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "positive_rate": n_positive / n_total,
                "n_positive": n_positive,
                "n_total": n_total,
                "n_surviving_features": 0,
                "surviving_features": "",
            })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(
        "Threshold sensitivity complete: %d scenarios, AUC range [%.4f, %.4f]",
        len(summary_df),
        summary_df["best_auc"].min(),
        summary_df["best_auc"].max(),
    )

    return summary_df


def compute_label_stability(
    labels_dict: Dict[Any, pd.Series],
    baseline_key: Any,
) -> pd.DataFrame:
    """Compute label stability metrics across sensitivity scenarios.

    Measures how much the set of flagged customers changes when parameters
    are varied, using Jaccard similarity and flip rate relative to baseline.

    Args:
        labels_dict: Dict mapping scenario key (e.g., base_cost float) to
            a Series of binary labels indexed by user_id.
        baseline_key: Key identifying the baseline scenario for comparison.

    Returns:
        DataFrame with one row per scenario containing:
        - scenario: The scenario key
        - jaccard_similarity: Jaccard index of flagged sets vs baseline
        - flip_rate: Proportion of customers whose label changed vs baseline
        - n_flagged: Number of customers flagged in this scenario
        - n_flagged_baseline: Number flagged in baseline

    Raises:
        KeyError: If baseline_key is not in labels_dict.
    """
    if baseline_key not in labels_dict:
        raise KeyError(
            f"Baseline key '{baseline_key}' not found in labels_dict. "
            f"Available keys: {list(labels_dict.keys())}"
        )

    baseline_labels = labels_dict[baseline_key]
    baseline_flagged = set(baseline_labels[baseline_labels == 1].index)

    rows = []
    for scenario_key, labels in labels_dict.items():
        # Align to common index
        common_idx = baseline_labels.index.intersection(labels.index)
        bl = baseline_labels.loc[common_idx]
        sc = labels.loc[common_idx]

        scenario_flagged = set(sc[sc == 1].index)

        # Jaccard similarity
        intersection = len(baseline_flagged & scenario_flagged)
        union = len(baseline_flagged | scenario_flagged)
        jaccard = intersection / union if union > 0 else 1.0

        # Flip rate
        n_flips = (bl != sc).sum()
        flip_rate = n_flips / len(common_idx) if len(common_idx) > 0 else 0.0

        rows.append({
            "scenario": scenario_key,
            "jaccard_similarity": jaccard,
            "flip_rate": flip_rate,
            "n_flagged": len(scenario_flagged),
            "n_flagged_baseline": len(baseline_flagged),
        })

    stability_df = pd.DataFrame(rows)
    logger.info(
        "Label stability: Jaccard range [%.4f, %.4f], flip rate range [%.4f, %.4f]",
        stability_df["jaccard_similarity"].min(),
        stability_df["jaccard_similarity"].max(),
        stability_df["flip_rate"].min(),
        stability_df["flip_rate"].max(),
    )

    return stability_df
