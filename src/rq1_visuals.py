"""
RQ1 visualization utilities (publication-ready).

All RQ1 figures MUST be saved under: figures/rq1/

Design goals
------------
- Publication-ready defaults (consistent typography, grid, sizing).
- Deterministic outputs suitable for CI execution.
- Notebook/CLI orchestration only calls functions here (no ad-hoc plotting in notebooks).

Notes
-----
- This module intentionally does not write any data outputs. It only saves figures.
- Statistical testing lives in `rq1_stats.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlotStyle:
    """Centralized, consistent style choices for RQ1 figures."""
    figsize_wide: Tuple[float, float] = (12.0, 7.0)
    figsize_standard: Tuple[float, float] = (10.5, 6.5)
    dpi: int = 300
    title_size: int = 15
    label_size: int = 12
    tick_size: int = 10
    grid_alpha: float = 0.25
    bar_color: str = "#2E86AB"          # consistent primary
    accent_color: str = "#A23B72"       # consistent accent
    neutral_color: str = "#4D4D4D"
    ci_color: str = "#1F77B4"
    edge_color: str = "#222222"


DEFAULT_STYLE = PlotStyle()


def _apply_style(style: PlotStyle = DEFAULT_STYLE) -> None:
    """
    Apply a consistent matplotlib style for publication-ready plots.

    This is intentionally lightweight (no seaborn dependency) and safe to call per plot.
    """
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.dpi,
            "font.size": style.tick_size,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "axes.grid": True,
            "grid.alpha": style.grid_alpha,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _validate_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{context}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)[:40]}..."
        )


def _ensure_parent_dir(out_path: Path) -> None:
    """Create parent directory for output path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)


def _format_currency(x: float) -> str:
    """Compact currency formatter for annotations."""
    if np.isnan(x):
        return "n/a"
    absx = abs(x)
    if absx >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if absx >= 1_000:
        return f"${x/1_000:.1f}K"
    return f"${x:.0f}"


def _safe_float_series(s: pd.Series) -> pd.Series:
    """Convert series to float safely."""
    return pd.to_numeric(s, errors="coerce").astype(float)


def _default_figures_dir() -> Path:
    """
    Resolve the default RQ1 figures output directory.

    Uses `project_root/figures/rq1` when a `src/` folder is detected; otherwise falls back
    to `figures/rq1` relative to the current working directory.
    """
    cwd = Path.cwd().resolve()
    project_root = cwd
    while not (project_root / "src").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    if (project_root / "src").exists():
        return project_root / "figures" / "rq1"
    return Path("figures") / "rq1"


def plot_top_groups_total_erosion(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str = "total_profit_erosion",
    top_n: int = 15,
    out_path: Path,
    title: Optional[str] = None,
    style: PlotStyle = DEFAULT_STYLE,
    annotate_top_k: int = 10,
) -> Path:
    """
    Plot a ranked horizontal bar chart of the top-N groups by total profit erosion.

    Parameters
    ----------
    df:
        Aggregated dataframe containing at least [group_col, value_col].
    group_col:
        Categorical column name (e.g., 'category', 'brand', 'department').
    value_col:
        Numeric metric column (default: 'total_profit_erosion').
    top_n:
        Number of top groups to display.
    out_path:
        File path where the figure will be saved (must be under figures/rq1/).
    title:
        Optional plot title. If None, a sensible default is used.
    style:
        PlotStyle controlling typography and colors.
    annotate_top_k:
        Number of top bars to annotate with values (avoid clutter).

    Returns
    -------
    Path
        The saved figure path.
    """
    _validate_columns(df, [group_col, value_col], context="plot_top_groups_total_erosion")
    _ensure_parent_dir(out_path)
    _apply_style(style)

    df2 = df[[group_col, value_col]].copy()
    df2[value_col] = _safe_float_series(df2[value_col])
    df2 = df2.dropna(subset=[value_col])
    df_top = df2.sort_values(value_col, ascending=False).head(top_n).copy()

    # Reverse for horizontal bars (largest at top)
    df_top = df_top.iloc[::-1]

    fig, ax = plt.subplots(figsize=style.figsize_wide)
    bars = ax.barh(
        df_top[group_col].astype(str),
        df_top[value_col],
        color=style.bar_color,
        edgecolor=style.edge_color,
        linewidth=0.6,
    )

    ax.set_xlabel("Total Profit Erosion")
    ax.set_ylabel(group_col.replace("_", " ").title())
    ax.set_title(title or f"Top {top_n} {group_col.replace('_', ' ').title()} by Total Profit Erosion")

    maxv = float(df_top[value_col].max()) if not df_top.empty else 0.0
    ax.set_xlim(0, maxv * 1.12 if maxv > 0 else 1.0)

    k = min(annotate_top_k, len(df_top))
    if k > 0:
        for bar in bars[-k:]:
            v = float(bar.get_width())
            ax.text(
                v + (maxv * 0.01 if maxv > 0 else 0.01),
                bar.get_y() + bar.get_height() / 2,
                _format_currency(v),
                va="center",
                ha="left",
                fontsize=style.tick_size,
                color=style.neutral_color,
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_return_rate_vs_mean_erosion(
    df: pd.DataFrame,
    *,
    x_col: str = "return_rate",
    y_col: str = "avg_profit_erosion",
    label_col: str = "category",
    size_col: Optional[str] = "returned_items",
    out_path: Path,
    title: str = "Return Rate vs Mean Profit Erosion per Return",
    style: PlotStyle = DEFAULT_STYLE,
    annotate_top_k: int = 10,
) -> Path:
    """
    Scatter plot: Return Rate (x) vs Mean Profit Erosion per Return (y).

    Purpose (RQ1):
    - Distinguishes whether high erosion groups are driven by return frequency (return_rate)
      or per-return severity (avg_profit_erosion).
    - Bubble size optionally reflects returned volume (returned_items).

    Parameters
    ----------
    df:
        Aggregated dataframe containing x_col, y_col, label_col, and optionally size_col.
    x_col:
        X-axis variable (default: 'return_rate').
    y_col:
        Y-axis variable (default: 'avg_profit_erosion').
    label_col:
        Label column used for annotations (default: 'category').
    size_col:
        Bubble sizing variable (default: 'returned_items'). If None or missing, uses constant size.
    out_path:
        File path where the figure will be saved.
    title:
        Plot title.
    style:
        PlotStyle controlling typography and colors.
    annotate_top_k:
        Annotate top-k points to keep the figure readable (recommended 8–12).

    Returns
    -------
    Path
        The saved figure path.
    """
    req = [x_col, y_col, label_col]
    if size_col is not None:
        req.append(size_col)

    _validate_columns(df, req, context="plot_return_rate_vs_mean_erosion")
    _ensure_parent_dir(out_path)
    _apply_style(style)

    d = df[req].copy()
    d[x_col] = _safe_float_series(d[x_col])
    d[y_col] = _safe_float_series(d[y_col])
    if size_col is not None:
        d[size_col] = _safe_float_series(d[size_col]).clip(lower=0)

    d = d.dropna(subset=[x_col, y_col])

    # Bubble size scaling (safe default if size_col missing/None)
    if size_col is None or size_col not in d.columns or d[size_col].max() <= 0:
        sizes = np.full(len(d), 140.0)
    else:
        s_raw = d[size_col].to_numpy(dtype=float)
        s_norm = s_raw / (np.max(s_raw) if np.max(s_raw) > 0 else 1.0)
        sizes = s_norm * 1200 + 60  # readable range

    fig, ax = plt.subplots(figsize=style.figsize_wide)

    ax.scatter(
        d[x_col],
        d[y_col],
        s=sizes,
        alpha=0.65,
        color=style.bar_color,
        edgecolor=style.edge_color,
        linewidth=0.6,
    )

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").title())
    ax.set_title(title)

    # Annotate top contributors:
    # Prefer using total_profit_erosion if present, otherwise use y_col (severity).
    rank_col = "total_profit_erosion" if "total_profit_erosion" in df.columns else y_col
    if rank_col in df.columns:
        d_rank = df[[label_col, rank_col]].copy()
        d_rank[rank_col] = _safe_float_series(d_rank[rank_col])
        top_labels = (
            d_rank.dropna()
            .sort_values(rank_col, ascending=False)
            .head(min(annotate_top_k, len(d_rank)))[label_col]
            .astype(str)
            .tolist()
        )
    else:
        top_labels = (
            d.sort_values(y_col, ascending=False)
            .head(min(annotate_top_k, len(d)))[label_col]
            .astype(str)
            .tolist()
        )

    # Create a quick lookup for coordinates to annotate
    d[label_col] = d[label_col].astype(str)
    for _, r in d.iterrows():
        if str(r[label_col]) in top_labels:
            ax.annotate(
                str(r[label_col]),
                (float(r[x_col]), float(r[y_col])),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=style.neutral_color,
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_severity_vs_volume_decomposition(
    df: pd.DataFrame,
    *,
    group_col: str,
    returned_items_col: str = "returned_items",
    avg_erosion_col: str = "avg_profit_erosion",
    total_erosion_col: str = "total_profit_erosion",
    out_path: Path,
    title: str = "Severity vs Volume Decomposition (Total Erosion = Volume × Severity)",
    style: PlotStyle = DEFAULT_STYLE,
    annotate_top_k: int = 10,
) -> Path:
    """
    Bubble chart decomposing total erosion into Volume × Severity.

    Identity:
        total_profit_erosion = returned_items × avg_profit_erosion

    Annotation recommendation (best practice):
    - Annotate ONLY the top-k groups by total erosion to keep the figure readable.

    Parameters
    ----------
    df:
        Aggregated dataframe containing [group_col, returned_items_col, avg_erosion_col, total_erosion_col].
    group_col:
        Categorical label column (e.g., 'category').
    returned_items_col:
        Volume column (returned item count).
    avg_erosion_col:
        Severity column (average erosion per return).
    total_erosion_col:
        Bubble-size driver (total erosion).
    out_path:
        File path where the figure will be saved.
    title:
        Plot title.
    style:
        PlotStyle controlling typography and colors.
    annotate_top_k:
        Annotate top-k bubbles by total erosion (recommended to avoid clutter).

    Returns
    -------
    Path
        The saved figure path.
    """
    _validate_columns(
        df,
        [group_col, returned_items_col, avg_erosion_col, total_erosion_col],
        context="plot_severity_vs_volume_decomposition",
    )
    _ensure_parent_dir(out_path)
    _apply_style(style)

    df2 = df[[group_col, returned_items_col, avg_erosion_col, total_erosion_col]].copy()
    df2[returned_items_col] = _safe_float_series(df2[returned_items_col]).clip(lower=0)
    df2[avg_erosion_col] = _safe_float_series(df2[avg_erosion_col]).clip(lower=0)
    df2[total_erosion_col] = _safe_float_series(df2[total_erosion_col]).clip(lower=0)
    df2 = df2.dropna()

    s_raw = df2[total_erosion_col]
    s = (s_raw / (s_raw.max() if s_raw.max() > 0 else 1.0)) * 1400 + 40

    fig, ax = plt.subplots(figsize=style.figsize_wide)
    ax.scatter(
        df2[returned_items_col],
        df2[avg_erosion_col],
        s=s,
        alpha=0.60,
        color=style.bar_color,
        edgecolor=style.edge_color,
        linewidth=0.6,
    )

    ax.set_xlabel("Returned Items (Volume)")
    ax.set_ylabel("Average Profit Erosion per Return (Severity)")
    ax.set_title(title)

    k = min(annotate_top_k, len(df2))
    top = df2.sort_values(total_erosion_col, ascending=False).head(k)
    for _, r in top.iterrows():
        ax.annotate(
            str(r[group_col]),
            (float(r[returned_items_col]), float(r[avg_erosion_col])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color=style.neutral_color,
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_profit_erosion_distribution_log(
    returned_df: pd.DataFrame,
    *,
    value_col: str = "profit_erosion",
    out_path: Path,
    title: str = "Distribution of Profit Erosion for Returned Items (Log Scale)",
    style: PlotStyle = DEFAULT_STYLE,
    bins: int = 60,
) -> Path:
    """
    Histogram of profit erosion for returned items with a log-scale x-axis.

    Why it matters for RQ1:
    - Profit erosion is typically right-skewed (few extreme items).
    - This supports the use of Kruskal–Wallis when normality fails.

    Parameters
    ----------
    returned_df:
        Item-level dataframe filtered to returned items only.
    value_col:
        Numeric column to plot (default: 'profit_erosion').
    out_path:
        File path where the figure will be saved.
    title:
        Plot title.
    style:
        PlotStyle controlling typography and colors.
    bins:
        Number of histogram bins.

    Returns
    -------
    Path
        The saved figure path.
    """
    _validate_columns(returned_df, [value_col], context="plot_profit_erosion_distribution_log")
    _ensure_parent_dir(out_path)
    _apply_style(style)

    x = _safe_float_series(returned_df[value_col]).dropna()
    x = x[x > 0]
    if x.empty:
        raise ValueError("[plot_profit_erosion_distribution_log] No positive values available for log-scale plot.")

    fig, ax = plt.subplots(figsize=style.figsize_standard)
    ax.hist(x, bins=bins, edgecolor=style.edge_color, alpha=0.85, color=style.accent_color)
    ax.set_xscale("log")
    ax.set_xlabel(f"{value_col} (log scale)")
    ax.set_ylabel("Count of returned items")
    ax.set_title(title)

    p50 = float(np.percentile(x, 50))
    p90 = float(np.percentile(x, 90))
    p99 = float(np.percentile(x, 99))
    txt = f"Median: {_format_currency(p50)}\nP90: {_format_currency(p90)}\nP99: {_format_currency(p99)}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10, color=style.neutral_color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_bootstrap_ci_mean_by_group(
    returned_df: pd.DataFrame,
    *,
    group_col: str = "category",
    value_col: str = "profit_erosion",
    out_path: Path,
    title: str = "Bootstrap 95% CI for Mean Profit Erosion (Top Groups)",
    style: PlotStyle = DEFAULT_STYLE,
    n_boot: int = 800,
    min_group_size: int = 30,
    top_n_plot: int = 15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Path]:
    """
    Bootstrap confidence intervals for mean profit erosion by group.

    This is recommended to include in RQ1 because it:
    - adds robustness beyond a single p-value
    - shows uncertainty around group mean estimates
    - improves interpretability in a thesis/capstone context

    Parameters
    ----------
    returned_df:
        Returned-items-only dataframe.
    group_col:
        Grouping variable (default: 'category').
    value_col:
        Numeric outcome (default: 'profit_erosion').
    out_path:
        File path where the figure will be saved.
    title:
        Plot title.
    style:
        PlotStyle controlling typography and colors.
    n_boot:
        Bootstrap iterations.
    min_group_size:
        Minimum returned items per group to compute CI.
    top_n_plot:
        Number of groups to show in plot (sorted by mean desc).
    random_state:
        RNG seed.

    Returns
    -------
    (ci_df, out_path)
        ci_df contains mean and 95% CI bounds per group; out_path is the saved figure path.
    """
    _validate_columns(returned_df, [group_col, value_col], context="plot_bootstrap_ci_mean_by_group")
    _ensure_parent_dir(out_path)
    _apply_style(style)

    df = returned_df[[group_col, value_col]].dropna().copy()
    df[value_col] = _safe_float_series(df[value_col])

    counts = df[group_col].value_counts(dropna=False)
    keep = counts[counts >= min_group_size].index
    df = df[df[group_col].isin(keep)].copy()
    if df.empty:
        raise ValueError(f"[plot_bootstrap_ci_mean_by_group] No groups meet min_group_size={min_group_size}.")

    rng = np.random.default_rng(random_state)

    rows = []
    for g, sub in df.groupby(group_col):
        vals = sub[value_col].to_numpy()
        n = len(vals)

        boot_means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(vals, size=n, replace=True)
            boot_means[i] = float(np.mean(sample))

        mean_hat = float(np.mean(vals))
        ci_low = float(np.quantile(boot_means, 0.025))
        ci_high = float(np.quantile(boot_means, 0.975))

        rows.append(
            {
                group_col: g,
                "n_returned_items": int(n),
                "mean_profit_erosion": mean_hat,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "ci_width": ci_high - ci_low,
            }
        )

    ci_df = pd.DataFrame(rows).sort_values("mean_profit_erosion", ascending=False).reset_index(drop=True)

    plot_df = ci_df.head(top_n_plot).sort_values("mean_profit_erosion", ascending=True).copy()
    y = np.arange(len(plot_df))
    means = plot_df["mean_profit_erosion"].to_numpy()
    err_low = means - plot_df["ci_low_95"].to_numpy()
    err_high = plot_df["ci_high_95"].to_numpy() - means

    fig, ax = plt.subplots(figsize=style.figsize_wide)
    ax.errorbar(
        means,
        y,
        xerr=[err_low, err_high],
        fmt="o",
        capsize=4,
        color=style.ci_color,
        ecolor=style.edge_color,
        linewidth=1.2,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df[group_col].astype(str))
    ax.set_xlabel("Mean Profit Erosion per Returned Item")
    ax.set_title(title)

    # Add n context at the end of CI bars
    for yi, mean, n in zip(y, means, plot_df["n_returned_items"].to_numpy()):
        ax.text(mean, yi, f"  n={int(n)}", va="center", ha="left", fontsize=9, color=style.neutral_color)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return ci_df, out_path
