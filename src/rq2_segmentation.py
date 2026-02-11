"""
RQ2 / US09 - Customer behavioral segmentation
(feature matrix + clustering + diagnostics).

This module is designed to consume:
- customer_behavior: output of engineer_customer_behavioral_features
  (item-level -> customer level)
- customer_erosion: output of aggregate_profit_erosion_by_customer
  (returned-item level -> customer level)

Then:
- join into a segmentation table
- select numeric features
- standardize features
- apply KMeans clustering (deterministic via random_state)
- produce elbow + silhouette diagnostics

All functions are unit-testable with tiny in-memory DataFrames (CI-safe).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.descriptive_transformations import _require_columns

# Default RQ2 feature set intentionally excludes direct erosion outcomes to
# prevent data leakage when testing downstream erosion differences by segment.
DEFAULT_SEGMENTATION_FEATURES = [
    "total_items_purchased",
    "avg_order_value",
    "avg_basket_size",
    "order_frequency",
    "customer_return_rate",
    "customer_tenure_days",
    "purchase_recency_days",
    "total_sales",
    "total_margin",
]

LEAKAGE_FEATURES = {
    "total_profit_erosion",
    "total_margin_reversal",
    "total_processing_cost",
    "erosion_percentile_rank",
    "profit_erosion_quartile",
    "high_erosion_customer",
}


def build_customer_segmentation_table(
    customer_behavior: pd.DataFrame,
    customer_erosion: pd.DataFrame,
    id_col: str = "user_id",
) -> pd.DataFrame:
    """
    Join behavioral features and erosion metrics into a single customer-level table.

    - Left-join keeps all customers present in customer_behavior.
    - Numeric erosion columns that are missing become 0 (customers with no returns).

    Args:
        customer_behavior: Customer-level behavioral feature table.
        customer_erosion: Customer-level profit erosion table.
        id_col: Customer identifier column shared by both tables.

    Returns:
        Merged customer-level table with behavioral and erosion features.
    """
    _require_columns(
        customer_behavior,
        [id_col],
        "build_customer_segmentation_table:customer_behavior",
    )
    _require_columns(
        customer_erosion, [id_col], "build_customer_segmentation_table:customer_erosion"
    )

    df = customer_behavior.merge(customer_erosion, on=id_col, how="left")

    # Fill numeric NaNs with 0.0 (especially erosion columns)
    for col in df.columns:
        if col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0.0)

    return df


def select_numeric_features(
    customer_df: pd.DataFrame,
    id_col: str = "user_id",
    feature_cols: Optional[List[str]] = None,
    exclude_leakage_features: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select numeric feature columns for clustering.

    If feature_cols is None:
        prefer the default non-leakage behavioral feature set, falling back
        to numeric columns (excluding id_col and leakage columns).

    Args:
        customer_df: Customer-level table containing candidate feature columns.
        id_col: Customer identifier column to exclude from features.
        feature_cols: Optional list of explicit feature columns to use.
        exclude_leakage_features: If True, disallow direct erosion-derived
            leakage features.

    Returns:
        X (DataFrame of selected features), used_cols (list)
    """
    df = customer_df.copy()

    leakage_present = sorted(c for c in LEAKAGE_FEATURES if c in df.columns)

    if feature_cols is None:
        default_available = [
            c
            for c in DEFAULT_SEGMENTATION_FEATURES
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if default_available:
            used = default_available
        else:
            used = [
                c
                for c in df.columns
                if c != id_col and pd.api.types.is_numeric_dtype(df[c])
            ]

        if exclude_leakage_features:
            used = [c for c in used if c not in LEAKAGE_FEATURES]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        if exclude_leakage_features:
            leakage_requested = [c for c in feature_cols if c in LEAKAGE_FEATURES]
            if leakage_requested:
                raise ValueError(
                    "Feature list contains leakage columns: "
                    f"{sorted(leakage_requested)}"
                )
        used = feature_cols

    if exclude_leakage_features and not used and leakage_present:
        raise ValueError(
            "No valid segmentation features available after leakage exclusion."
        )

    X = df[used].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X, used


def standardize_features(X: pd.DataFrame) -> np.ndarray:
    """
    Standardize features to zero mean / unit variance.
    Returns a numpy array suitable for clustering.

    Args:
        X: DataFrame of numeric feature columns.

    Returns:
        Standardized feature matrix as a NumPy array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X.to_numpy())


def kmeans_fit_predict(
    X_scaled: np.ndarray,
    k: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit KMeans and return labels (deterministic with random_state).

    Args:
        X_scaled: Standardized feature matrix.
        k: Number of clusters to fit.
        random_state: Random seed for reproducible clustering.

    Returns:
        NumPy array of cluster labels.
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return model.fit_predict(X_scaled)


def summarize_clusters(
    clustered_df: pd.DataFrame,
    value_col: str = "total_profit_erosion",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Generate segment-level summary statistics for interpreting customer clusters.

    This function fulfills US09 requirement #89:
    "Segment-level profit erosion compared".
    It produces tabular summaries of economic characteristics by cluster, supporting
    the comparison of mean profit erosion across identified customer segments.

    The output enables analysts to:
    - Identify high-risk customer segments (clusters with elevated mean erosion)
    - Compare economic impact distribution across behavioral groups
    - Prioritize intervention strategies based on segment-level metrics
    - Document findings for Research Question 2 (RQ2) in the technical report

    Args:
        clustered_df: Customer-level DataFrame with cluster assignments.
                      Must contain at minimum: user_id, cluster assignments,
                      and profit erosion values.
        value_col: Column name containing profit erosion values.
                   Default: 'total_profit_erosion'
        cluster_col: Column name containing cluster labels/assignments.
                     Default: 'cluster_id'

    Returns:
        DataFrame with one row per cluster containing:
            - cluster_id (or specified cluster_col): Cluster identifier (0, 1, 2, ...)
            - Count: Number of customers in the cluster
            - Total_Erosion: Sum of profit erosion for all customers in cluster
            - Mean_Erosion: Average profit erosion per customer in cluster
            - Median_Erosion: Median profit erosion per customer in cluster

        Sorted by cluster_id in ascending order.

    Raises:
        KeyError: If value_col or cluster_col not found in clustered_df

    Example:
        >>> # After clustering with k=4
        >>> clustered = seg_table.copy()
        >>> clustered['cluster_id'] = kmeans_fit_predict(X_scaled, k=4, random_state=42)
        >>>
        >>> # Generate segment summary
        >>> summary = summarize_clusters(clustered)
        >>> print(summary)
           cluster_id  Count  Total_Erosion  Mean_Erosion  Median_Erosion
        0           0   2756      230490.32         83.63           78.11
        1           1   2740      124109.07         45.30           40.28
        2           2   1853      318380.82        171.82          151.24
        3           3   4441      135271.85         30.46           29.17

        >>> # Interpretation: Cluster 2 shows highest mean erosion ($171.82)
        >>> # despite having fewer customers (1,853), indicating a high-risk segment

    Notes:
        - All monetary values are rounded to 2 decimal places
        - Clusters are presented in ascending order by cluster_id
        - This function is designed for exploratory analysis (US09 scope)
        - No causal claims should be made from segment differences

    See Also:
        - kmeans_fit_predict: Function that generates cluster assignments
        - build_customer_segmentation_table: Creates the input DataFrame
        - US09 Acceptance Criteria #89 in project documentation
    """
    if cluster_col not in clustered_df.columns:
        raise KeyError(f"Column '{cluster_col}' not found in clustered_df")
    if value_col not in clustered_df.columns:
        raise KeyError(f"Column '{value_col}' not found in clustered_df")

    summary = (
        clustered_df.groupby(cluster_col)[value_col]
        .agg(
            Count="size",
            Total_Erosion="sum",
            Mean_Erosion="mean",
            Median_Erosion="median",
        )
        .round(2)
        .reset_index()
    )

    return summary


def elbow_inertia_over_k(
    X_scaled: np.ndarray,
    k_list: List[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute inertia values for a list of k (elbow diagnostic).

    Args:
        X_scaled: Standardized feature matrix.
        k_list: List of k values to evaluate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns [k, inertia].
    """
    rows = []
    for k in k_list:
        if k < 1:
            raise ValueError("All k values must be >= 1")
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        model.fit(X_scaled)
        rows.append({"k": int(k), "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def silhouette_over_k(
    X_scaled: np.ndarray,
    k_list: List[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute silhouette scores for a list of k (k must be >= 2).

    Args:
        X_scaled: Standardized feature matrix.
        k_list: List of k values to evaluate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns [k, silhouette].
    """
    rows = []
    for k in k_list:
        if k < 2:
            raise ValueError("Silhouette requires k >= 2")
        labels = kmeans_fit_predict(X_scaled, k=k, random_state=random_state)
        score = silhouette_score(X_scaled, labels)
        rows.append({"k": int(k), "silhouette": float(score)})
    return pd.DataFrame(rows)


def clustering_metrics_over_k(
    X_scaled: np.ndarray,
    k_list: List[int],
    random_state: int = 42,
    n_init: int | str = "auto",
) -> pd.DataFrame:
    """
    Compute k-means quality diagnostics for each k in k_list.

    Args:
        X_scaled: Standardized feature matrix.
        k_list: List of k values to evaluate.
        random_state: Random seed for reproducibility.
        n_init: Number of KMeans initializations or "auto".

    Returns:
        DataFrame with columns: k, inertia, silhouette, calinski_harabasz,
        and davies_bouldin.
    """
    rows = []

    for k in k_list:
        if k < 1:
            raise ValueError("All k values must be >= 1")

        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)

        row = {"k": int(k), "inertia": float(model.inertia_)}

        if k >= 2:
            row["silhouette"] = float(silhouette_score(X_scaled, labels))
            row["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
            row["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
        else:
            row["silhouette"] = np.nan
            row["calinski_harabasz"] = np.nan
            row["davies_bouldin"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_clustering_quality_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Compute clustering quality metrics for an existing label assignment.

    Returns a dict with silhouette, calinski_harabasz, davies_bouldin.
    For <2 unique clusters, all metrics are NaN.

    Args:
        X_scaled: Standardized feature matrix.
        labels: Cluster labels for each sample.

    Returns:
        Dictionary containing silhouette, calinski_harabasz, and davies_bouldin.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }

    return {
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
    }


def combined_diagnostics(
    X_scaled: np.ndarray,
    k_list: List[int],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute both elbow (inertia) and silhouette metrics efficiently.
    Fits k-means once per k value and calculates both metrics.

    This is ~2x faster than calling elbow_inertia_over_k
    and silhouette_over_k separately
    because it reuses the same k-means fit for both metrics.

    Args:
        X_scaled: Standardized feature matrix (n_samples, n_features)
        k_list: List of k values to evaluate (e.g., [1, 2, 3, ..., 10])
        random_state: Random seed for reproducibility

    Returns:
        (elbow_df, silhouette_df) - Two DataFrames:
            - elbow_df: columns ['k', 'inertia']
            - silhouette_df: columns ['k', 'silhouette'] (only k >= 2)
    """
    elbow_rows = []
    sil_rows = []

    for k in k_list:
        if k < 1:
            raise ValueError("All k values must be >= 1")

        # Fit k-means once
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        model.fit(X_scaled)

        # Get inertia (always available)
        elbow_rows.append({"k": int(k), "inertia": float(model.inertia_)})

        # Get silhouette (only for k >= 2)
        if k >= 2:
            labels = model.labels_
            score = silhouette_score(X_scaled, labels)
            sil_rows.append({"k": int(k), "silhouette": float(score)})

    elbow_df = pd.DataFrame(elbow_rows)
    silhouette_df = pd.DataFrame(sil_rows)

    return elbow_df, silhouette_df
