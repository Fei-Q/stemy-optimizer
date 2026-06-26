"""Plotting utilities for the SteMy BO dashboard.

These functions are intentionally independent from Streamlit. They read the
existing SteMy result files and return matplotlib Figure objects, so the same
plots can be used from a local GUI, notebooks, or standalone scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def study_paths(study_id: str, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> dict[str, Path]:
    root = Path(results_dir) / study_id
    return {
        "root": root,
        "config": root / f"{study_id}_config.tsv",
        "log": root / f"{study_id}_log.tsv",
        "summary": root / f"{study_id}_summary.json",
        "metadata": root / f"{study_id}_metadata.json",
    }


def load_study_tables(
    study_id: str,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load config TSV, log TSV, and summary JSON for one study."""
    paths = study_paths(study_id, results_dir)

    config_df = pd.read_csv(paths["config"], sep="\t") if paths["config"].exists() else pd.DataFrame()
    log_df = pd.read_csv(paths["log"], sep="\t") if paths["log"].exists() else pd.DataFrame()

    if paths["summary"].exists():
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    else:
        summary = {}

    return config_df, log_df, summary


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Convert a DataFrame column to numeric values with invalid entries as NaN."""
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def completed_log(log_df: pd.DataFrame) -> pd.DataFrame:
    """Return completed rows with numeric cTnT values, preserving original order."""
    if log_df.empty or "ctnt_pct" not in log_df.columns:
        return pd.DataFrame()
    df = log_df.copy()
    df["ctnt_pct_num"] = pd.to_numeric(df["ctnt_pct"], errors="coerce")
    df = df[(df.get("run_status", "") == "completed") & df["ctnt_pct_num"].notna()].copy()
    df["completed_order"] = np.arange(1, len(df) + 1)
    return df


def delta_columns(log_df: pd.DataFrame, group: str = "all") -> list[str]:
    """Return delta columns, optionally restricted to dynamic x or control z groups."""
    cols = [c for c in log_df.columns if c.endswith("_delta")]
    if group == "dynamic":
        return [c for c in cols if f"{c[:-6]}_planned" in log_df.columns]
    if group == "control":
        return [c for c in cols if f"{c[:-6]}_target" in log_df.columns]
    return cols


def available_parameters_for_planned_actual(log_df: pd.DataFrame) -> list[str]:
    """Return parameter names that have planned/target and actual columns."""
    params: list[str] = []
    for col in log_df.columns:
        if col.endswith("_actual"):
            name = col[: -len("_actual")]
            if f"{name}_planned" in log_df.columns or f"{name}_target" in log_df.columns:
                params.append(name)
    return sorted(set(params))


def plot_ctnt_over_time(log_df: pd.DataFrame):
    """Plot observed cTnT% and best-so-far across completed runs."""
    df = completed_log(log_df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No completed runs with cTnT% yet", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = df["completed_order"].to_numpy()
    y = df["ctnt_pct_num"].to_numpy()
    best = np.maximum.accumulate(y)

    ax.scatter(x, y, label="Observed cTnT%")
    ax.plot(x, best, marker="o", label="Best so far")
    ax.set_xlabel("Completed run order")
    ax.set_ylabel("cTnT%")
    ax.set_title("Differentiation outcome over time")
    ax.set_ylim(0, max(100, np.nanmax(y) * 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_batch_performance(log_df: pd.DataFrame):
    """Plot per-batch cTnT% distributions and batch means."""
    df = completed_log(log_df)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty or "batch_id" not in df.columns:
        ax.text(0.5, 0.5, "No completed batch data yet", ha="center", va="center")
        ax.set_axis_off()
        return fig

    batches = [b for b in sorted(df["batch_id"].dropna().unique())]
    data = [df.loc[df["batch_id"] == b, "ctnt_pct_num"].to_numpy() for b in batches]
    positions = np.arange(1, len(batches) + 1)

    ax.boxplot(data, positions=positions, widths=0.55)
    for pos, values in zip(positions, data):
        if len(values):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), pos) + jitter, values, alpha=0.8)
            ax.scatter([pos], [np.nanmean(values)], marker="D", s=55, label="Batch mean" if pos == 1 else None)

    ax.set_xticks(positions)
    ax.set_xticklabels(batches, rotation=45, ha="right")
    ax.set_xlabel("Batch")
    ax.set_ylabel("cTnT%")
    ax.set_title("Batch performance")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    if len(batches):
        ax.legend()
    fig.tight_layout()
    return fig


def plot_planned_vs_actual(log_df: pd.DataFrame, param_name: str):
    """Plot planned/target vs actual values for one parameter."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if log_df.empty:
        ax.text(0.5, 0.5, "No log data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    planned_col = f"{param_name}_planned"
    target_col = f"{param_name}_target"
    actual_col = f"{param_name}_actual"

    x_col = planned_col if planned_col in log_df.columns else target_col
    if x_col not in log_df.columns or actual_col not in log_df.columns:
        ax.text(0.5, 0.5, f"Missing planned/target or actual columns for {param_name}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = pd.to_numeric(log_df[x_col], errors="coerce")
    y = pd.to_numeric(log_df[actual_col], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        ax.text(0.5, 0.5, f"No actual values for {param_name}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(x[mask], y[mask], alpha=0.8)
    lo = float(min(x[mask].min(), y[mask].min()))
    hi = float(max(x[mask].max(), y[mask].max()))
    if np.isfinite(lo) and np.isfinite(hi):
        pad = 0.05 * max(hi - lo, 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", label="planned = actual")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    label = "planned" if x_col == planned_col else "target"
    ax.set_xlabel(f"{param_name} {label}")
    ax.set_ylabel(f"{param_name} actual")
    ax.set_title(f"Planned/target vs actual: {param_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_delta_by_parameter(log_df: pd.DataFrame, group: str = "all"):
    """Plot mean absolute deviation by parameter."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cols = delta_columns(log_df, group=group)
    if log_df.empty or not cols:
        ax.text(0.5, 0.5, "No deviation columns available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    values = []
    labels = []
    for col in cols:
        s = pd.to_numeric(log_df[col], errors="coerce").abs()
        if s.notna().any():
            values.append(float(s.mean()))
            labels.append(col[: -len("_delta")])

    if not values:
        ax.text(0.5, 0.5, "No numeric deviation values yet", ha="center", va="center")
        ax.set_axis_off()
        return fig

    order = np.argsort(values)[::-1]
    values = [values[i] for i in order]
    labels = [labels[i] for i in order]

    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Mean absolute deviation")
    ax.set_title("Deviation magnitude by parameter")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_delta_heatmap(log_df: pd.DataFrame, group: str = "all", max_rows: int = 50):
    """Plot a run-by-parameter heatmap of delta values."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = delta_columns(log_df, group=group)
    if log_df.empty or not cols:
        ax.text(0.5, 0.5, "No deviation columns available", ha="center", va="center")
        ax.set_axis_off()
        return fig

    df = log_df.copy().tail(max_rows)
    mat = df[cols].apply(pd.to_numeric, errors="coerce")
    keep_cols = [c for c in mat.columns if mat[c].notna().any()]
    mat = mat[keep_cols]
    if mat.empty:
        ax.text(0.5, 0.5, "No numeric deviation values yet", ha="center", va="center")
        ax.set_axis_off()
        return fig

    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(keep_cols)))
    ax.set_xticklabels([c[: -len("_delta")] for c in keep_cols], rotation=60, ha="right")
    y_labels = df.get("run_id", pd.Series(range(len(df)))).astype(str).to_list()
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Run")
    ax.set_title("Planned/target to actual deviation heatmap")
    fig.colorbar(im, ax=ax, label="delta")
    fig.tight_layout()
    return fig


def plot_delta_vs_ctnt(log_df: pd.DataFrame, param_name: str):
    """Plot absolute deviation for one parameter against cTnT%."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    delta_col = f"{param_name}_delta"
    if log_df.empty or delta_col not in log_df.columns or "ctnt_pct" not in log_df.columns:
        ax.text(0.5, 0.5, f"Missing delta or cTnT% for {param_name}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    x = pd.to_numeric(log_df[delta_col], errors="coerce").abs()
    y = pd.to_numeric(log_df["ctnt_pct"], errors="coerce")
    mask = x.notna() & y.notna()
    if not mask.any():
        ax.text(0.5, 0.5, "No numeric values yet", ha="center", va="center")
        ax.set_axis_off()
        return fig

    ax.scatter(x[mask], y[mask], alpha=0.8)
    ax.set_xlabel(f"|{param_name} delta|")
    ax.set_ylabel("cTnT%")
    ax.set_title(f"Deviation vs outcome: {param_name}")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
