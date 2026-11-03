#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-oriented visualization for hyperparameter sensitivity results.

Expected input interface
------------------------
Pass either --summary-xlsx with a combined summary file or --results-dir with
the output directory produced by run_parameter_sensitivity_window_observability.py.
The latter automatically resolves combined_summary.xlsx (or its CSV fallback).

The combined summary should contain:
    parameter_group, parameter_name, parameter_value, default_value,
    is_default_configuration, run_status.
OT/repair rows use the window-only analysis interface:
    mean_pre_onset_coverage, mean_ignition_window_completeness,
    observable_transition_rate.
GMM/RTV rows retain:
    ignition_precision, ignition_recall, ignition_f1,
    false_ignition_rate, onset_mae.
Tracking deltas and rankings are derived from the successful default setting,
never from ignition F1. Old tracking summaries without window metrics must be
regenerated using run_parameter_sensitivity_window_observability.py.

Confidence-interval columns may be present in the input but are intentionally
not drawn, so all primary metric curves use the same point-estimate convention.

Only successful configurations are plotted as experimental results.
Failed/skipped configurations are summarized separately and never interpreted
as performance measurements.

Outputs
-------
output_dir/
    figures/
        fig00_configuration_status.png/.pdf
        fig01_core_hyperparameter_sensitivity.png/.pdf
        fig02_all_delta_f1_sensitivity.png/.pdf
        fig02_all_window_observability_deltas.png/.pdf (OT/repair)
        fig03_parameter_robustness_ranking.png/.pdf
        fig04_rtv_voting_rule.png/.pdf  (if available)
        fig05_all_hyperparameter_performance.png/.pdf
        per_parameter/*.png/.pdf
    data/
        successful_configurations.csv
        unsuccessful_configurations.csv
        parameter_robustness_summary.csv
        parameter_coverage_summary.csv
    hyperparameter_visualization_summary.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


PARAMETER_LABELS = {
    "ot_reg": "Sinkhorn regularization",
    "max_center_dist": "Maximum center distance",
    "dummy_cost": "Dummy assignment cost",
    "rescue_center_factor": "Rescue center factor",
    "max_gap_fill": "Maximum gap length",
    "prebirth_max_backward_frames": "Prebirth backward length",
    "prebirth_search_radius_base": "Prebirth search radius",
    "gmm_max_states": "Maximum GMM states",
    "gmm_bic_improvement_ratio": "BIC improvement ratio",
    "gmm_min_state_fraction": "Minimum state fraction",
    "gmm_min_state_count": "Minimum state count",
    "min_state_run": "Minimum state run",
    "cold_hot_persistence": "Cold/hot persistence",
    "rtv_alpha": "RTV threshold scale",
    "rtv_voting_rule": "RTV voting rule",
}

EXPECTED_PARAMETER_GROUPS = {
    "ot_reg": "ot",
    "max_center_dist": "ot",
    "dummy_cost": "ot",
    "rescue_center_factor": "repair",
    "max_gap_fill": "repair",
    "prebirth_max_backward_frames": "repair",
    "prebirth_search_radius_base": "repair",
    "gmm_max_states": "gmm",
    "gmm_bic_improvement_ratio": "gmm",
    "gmm_min_state_fraction": "gmm",
    "gmm_min_state_count": "gmm",
    "min_state_run": "gmm",
    "cold_hot_persistence": "gmm",
    "rtv_alpha": "rtv",
    "rtv_voting_rule": "rtv",
}

SUCCESS_STATUSES = frozenset(
    {"success", "success_with_track_errors", "ok", "completed"}
)

VALUE_LABELS = {
    "any_1_of_5": "Any 1 of 5",
    "at_least_2_of_5": "At least 2 of 5",
    "all_5": "All 5",
}

DEFAULT_CORE_PARAMETERS = [
    "gmm_max_states",
    "min_state_run",
    "cold_hot_persistence",
    "rtv_alpha",
]

TRACKING_GROUPS = frozenset({"ot", "repair"})

WINDOW_METRICS = {
    "mean_pre_onset_coverage": "Pre-onset coverage",
    "mean_ignition_window_completeness": "Window completeness",
    "observable_transition_rate": "Observable-transition rate",
}
WINDOW_STYLES = (("#1f77b4", "o"), ("#ff7f0e", "s"), ("#2ca02c", "^"))

TRACKING_PARAMETERS = [
    "ot_reg",
    "max_center_dist",
    "dummy_cost",
    "rescue_center_factor",
    "max_gap_fill",
    "prebirth_max_backward_frames",
    "prebirth_search_radius_base",
]


@dataclass
class Settings:
    dpi: int
    formats: tuple[str, ...]
    core_parameters: tuple[str, ...]
    font_family: str
    font_size: float


def configure_matplotlib(settings: Settings) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = float(settings.font_size)
    plt.rcParams.update(
        {
            "font.family": settings.font_family,
            "font.size": size,
            "axes.labelsize": size,
            "axes.titlesize": size,
            "legend.fontsize": size,
            "xtick.labelsize": size,
            "ytick.labelsize": size,
            "figure.titlesize": size,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_summary(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(
        f"Unsupported summary format {path.suffix!r}; expected .xlsx, .xls, or .csv"
    )


def resolve_summary_path(
    summary_path: Path | None,
    results_dir: Path | None,
) -> Path:
    """Resolve the combined summary emitted by the sensitivity runner."""
    if summary_path is not None:
        path = Path(summary_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if results_dir is None:
        raise ValueError("Either --summary-xlsx or --results-dir is required")
    root = Path(results_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidates = (
        root / "combined_summary.xlsx",
        root / "combined_summary.csv",
        root / "all_parameter_sensitivity.xlsx",
        root / "all_parameter_sensitivity.csv",
        root / "per_run_summary.xlsx",
        root / "per_run_summary.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No combined sensitivity summary was found under "
        f"{root}. Expected combined_summary.xlsx or combined_summary.csv."
    )


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Hyperparameter summary is missing required columns: {missing}")


def save_figure(fig: Any, base_path: Path, settings: Settings) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in settings.formats:
        path = base_path.with_suffix(f".{extension}")
        kwargs = {"dpi": settings.dpi} if extension.lower() in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
        fig.savefig(path, **kwargs)
        print(f"[OK] Saved: {path}")


def panel_label(ax: Any, text: str) -> None:
    ax.text(
        -0.12,
        1.07,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def numeric_or_original(series: pd.Series) -> tuple[pd.Series, bool]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float), True
    return series.astype(str), False


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "parameter_group",
            "parameter_name",
            "parameter_value",
            "default_value",
            "is_default_configuration",
            "run_status",
        ],
    )
    df = df.copy()
    df["parameter_group"] = df["parameter_group"].astype(str).str.strip().str.lower()
    df["parameter_name"] = df["parameter_name"].astype(str).str.strip()
    status = df["run_status"].astype(str).str.strip().str.lower()
    successful = status.isin(SUCCESS_STATUSES)
    tracking = df["parameter_group"].isin(TRACKING_GROUPS)
    if df.loc[successful].duplicated(["parameter_group", "parameter_name", "parameter_value"]).any():
        raise ValueError("Duplicate successful parameter settings; select one sensitivity run instead of merging repeated sweeps")
    if (successful & tracking).any():
        missing = [metric for metric in WINDOW_METRICS if metric not in df]
        if missing:
            raise ValueError(
                f"Tracking summary is missing window-observability metrics: {missing}. "
                "Re-run run_parameter_sensitivity_window_observability.py; "
                "ignition F1/reference coverage cannot substitute for these metrics."
            )
    if (successful & ~tracking).any():
        require_columns(df, ["ignition_precision", "ignition_recall", "ignition_f1",
                             "false_ignition_rate", "onset_mae"])
    for column in [
        *WINDOW_METRICS,
        *[f"delta_{metric}" for metric in WINDOW_METRICS],
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "false_ignition_rate",
        "onset_mae",
        "within_1_frame_acc",
        "within_3_frame_acc",
        "delta_f1",
        "delta_false_ignition_rate",
        "delta_onset_mae",
        "f1_ci_low",
        "f1_ci_high",
        "far_ci_low",
        "far_ci_high",
        "delta_f1_ci_low",
        "delta_f1_ci_high",
        "reference_track_coverage",
        "delta_reference_coverage",
        "suspicious_merge_proxy_count",
        "delta_suspicious_merge_proxy_count",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for metric in WINDOW_METRICS:
        if metric in df:
            values = df.loc[successful & tracking, metric]
            # Pre-onset coverage can be undefined only when the GT onset is
            # exactly at acquisition start (zero target frames). Keep it NaN.
            if metric != "mean_pre_onset_coverage" and values.isna().any():
                raise ValueError(f"Successful tracking rows contain missing {metric}")
            finite = values.dropna().to_numpy(dtype=float)
            if not np.isfinite(finite).all() or ((finite < 0) | (finite > 1)).any():
                raise ValueError(f"{metric} must be a fraction in [0, 1], not a percentage")
    if df["is_default_configuration"].dtype != bool:
        df["is_default_configuration"] = (
            df["is_default_configuration"]
            .astype(str)
            .str.strip()
            .str.lower()
                .isin(["true", "1", "1.0", "yes"])
        )
    return df


def select_tracking_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only OT-association and trajectory-repair configurations."""
    groups = df["parameter_group"].astype(str).str.strip().str.lower()
    selected = df[groups.isin(TRACKING_GROUPS)].copy()
    if selected.empty:
        available = sorted(set(groups))
        raise RuntimeError(
            "No OT or repair configurations were found in the input summary; "
            f"available groups={available}"
        )
    return selected


def split_status(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    status = df["run_status"].astype(str).str.strip().str.lower()
    successful = df[status.isin(SUCCESS_STATUSES)].copy()
    unsuccessful = df[~status.isin(SUCCESS_STATUSES)].copy()
    return successful, unsuccessful


def ordered_parameter_data(df: pd.DataFrame, parameter_name: str) -> tuple[pd.DataFrame, np.ndarray, list[str], bool]:
    subset = df[df["parameter_name"].astype(str).eq(parameter_name)].copy()
    if subset.empty:
        return subset, np.array([]), [], False

    values, numeric = numeric_or_original(subset["parameter_value"])
    if numeric:
        subset["_numeric_value"] = values
        subset = subset.sort_values("_numeric_value").reset_index(drop=True)
        x = subset["_numeric_value"].to_numpy(dtype=float)
        labels = [f"{value:g}" for value in x]
    else:
        order = list(dict.fromkeys(subset["parameter_value"].astype(str).tolist()))
        subset["_categorical_order"] = pd.Categorical(
            subset["parameter_value"].astype(str),
            categories=order,
            ordered=True,
        )
        subset = subset.sort_values("_categorical_order").reset_index(drop=True)
        x = np.arange(len(subset), dtype=float)
        labels = [VALUE_LABELS.get(value, value) for value in subset["parameter_value"].astype(str)]
    return subset, x, labels, numeric


def draw_default_line(ax: Any, subset: pd.DataFrame, x: np.ndarray) -> None:
    positions = np.where(subset["is_default_configuration"].to_numpy(dtype=bool))[0]
    for position in positions:
        ax.axvline(x[position], linestyle="--", linewidth=0.9, alpha=0.7)


def draw_default_marker(ax: Any, subset: pd.DataFrame, x: np.ndarray) -> None:
    draw_default_line(ax, subset, x)
    positions = np.where(subset["is_default_configuration"].to_numpy(dtype=bool))[0]
    for position in positions:
        y = float(subset.iloc[position]["ignition_f1"])
        if np.isfinite(y):
            ax.scatter([x[position]], [y], marker="*", s=100, edgecolors="black", linewidths=0.7, zorder=6)


def _is_tracking_parameter(subset: pd.DataFrame) -> bool:
    return len(subset) > 0 and set(subset["parameter_group"]).issubset(TRACKING_GROUPS)


def _window_baseline(subset: pd.DataFrame) -> pd.Series | None:
    defaults = subset[subset["is_default_configuration"]]
    if len(defaults) > 1:
        raise ValueError("Tracking parameter has multiple successful default configurations")
    return defaults.iloc[0] if len(defaults) else None


def plot_window_observability(
    ax: Any, successful: pd.DataFrame, parameter_name: str,
    show_legend: bool = True, metric: str | None = None,
) -> bool:
    """Same three window metrics and colors in every tracking figure."""
    subset, x, labels, numeric = ordered_parameter_data(successful, parameter_name)
    if subset.empty:
        ax.set_axis_off()
        return False
    for (column, label), (color, marker) in zip(WINDOW_METRICS.items(), WINDOW_STYLES):
        if metric is not None and column != metric:
            continue
        ax.plot(x, subset[column], color=color, marker=marker, linewidth=1.3, label=label)
    draw_default_line(ax, subset, x)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(WINDOW_METRICS[metric] if metric else "GT-centered observability")
    ax.set_xlabel("Parameter value" if numeric else "Configuration")
    if not numeric:
        ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.grid(alpha=0.2)
    ax.set_title(PARAMETER_LABELS.get(parameter_name, parameter_name))
    if show_legend:
        ax.legend(frameon=False, loc="best")
    return True


def plot_parameter_performance(
    ax: Any,
    successful: pd.DataFrame,
    parameter_name: str,
    show_legend: bool,
) -> bool:
    subset, x, labels, numeric = ordered_parameter_data(successful, parameter_name)
    if subset.empty:
        ax.text(0.5, 0.5, f"No successful data\n{parameter_name}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    if _is_tracking_parameter(subset):
        return plot_window_observability(ax, successful, parameter_name, show_legend)

    ax.plot(
        x,
        subset["ignition_f1"],
        marker="o",
        linewidth=1.4,
        label="F1",
    )
    ax.plot(x, subset["ignition_precision"], marker="s", linewidth=1.0, label="Precision")
    ax.plot(x, subset["ignition_recall"], marker="^", linewidth=1.0, label="Recall")
    ax.set_ylabel("Precision / recall / F1")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.set_title(PARAMETER_LABELS.get(parameter_name, parameter_name))
    if numeric:
        ax.set_xlabel("Parameter value")
    else:
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_xlabel("Configuration")

    far_ax = ax.twinx()
    far_values = pd.to_numeric(
        subset["false_ignition_rate"], errors="coerce"
    ).to_numpy(dtype=float)
    finite_far = far_values[np.isfinite(far_values)]
    if finite_far.size:
        far_ax.plot(
            x,
            far_values,
            marker="D",
            linestyle="--",
            linewidth=1.0,
            label="False ignition rate",
        )
        far_max = max(0.1, float(finite_far.max()) * 1.35)
        far_ax.set_ylim(0, min(1.0, far_max))
        far_ax.set_ylabel("False ignition rate")
    else:
        far_ax.set_axis_off()
    draw_default_marker(ax, subset, x)

    if show_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = far_ax.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    return True


def plot_parameter_timing(
    ax: Any,
    successful: pd.DataFrame,
    parameter_name: str,
    show_legend: bool,
) -> bool:
    subset, x, labels, numeric = ordered_parameter_data(successful, parameter_name)
    if subset.empty:
        ax.set_axis_off()
        return False

    if _is_tracking_parameter(subset):
        return plot_window_observability(
            ax, successful, parameter_name, show_legend,
            metric="mean_ignition_window_completeness",
        )

    available = False
    mae = pd.to_numeric(subset["onset_mae"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(mae).any():
        ax.plot(x, mae, marker="o", linewidth=1.3, label="Onset MAE")
        ax.set_ylabel("Onset MAE (frames)")
        available = True

    accuracy_ax = ax.twinx()
    for column, label, marker in (
        ("within_1_frame_acc", r"Acc@$\pm$1 frame", "s"),
        ("within_3_frame_acc", r"Acc@$\pm$3 frames", "^"),
    ):
        if column not in subset.columns:
            continue
        values = pd.to_numeric(subset[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            accuracy_ax.plot(x, values, marker=marker, linewidth=1.0, label=label)
            available = True
    if not available:
        ax.text(0.5, 0.5, "No finite timing metrics", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        accuracy_ax.set_axis_off()
        return False

    accuracy_ax.set_ylim(0, 1.05)
    accuracy_ax.set_ylabel("Temporal accuracy")
    draw_default_line(ax, subset, x)
    if numeric:
        ax.set_xlabel("Parameter value")
    else:
        ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_xlabel("Configuration")
    ax.grid(alpha=0.2)
    ax.set_title(f"{PARAMETER_LABELS.get(parameter_name, parameter_name)}: timing")
    if show_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = accuracy_ax.get_legend_handles_labels()
        if handles1 or handles2:
            ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    return True


def plot_tracking_diagnostics(
    ax: Any, successful: pd.DataFrame, parameter_name: str,
) -> bool:
    """Compatibility entry point, now showing only GT-centered window metrics."""
    return plot_window_observability(ax, successful, parameter_name, True)


def plot_status_overview(
    full: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    status_table = (
        full.assign(
            normalized_status=full["run_status"].astype(str).str.strip().str.lower()
        )
        .groupby(["parameter_group", "normalized_status"])
        .size()
        .unstack(fill_value=0)
    )
    groups = status_table.index.astype(str).tolist()
    statuses = status_table.columns.astype(str).tolist()
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for status in statuses:
        values = status_table[status].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=status)
        bottom += values
    ax.set_xticks(x, groups)
    ax.set_ylabel("Number of configurations")
    ax.set_title("Hyperparameter experiment execution status")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for index, total in enumerate(bottom):
        ax.text(index, total, f"{int(total)}", ha="center", va="bottom")
    save_figure(fig, output_dir / "fig00_configuration_status", settings)
    plt.close(fig)


def plot_core_parameters(
    successful: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    available = [
        parameter
        for parameter in settings.core_parameters
        if parameter in set(successful["parameter_name"].astype(str))
    ]
    if not available:
        print("[WARN] None of the requested core parameters have successful rows.")
        return

    n = len(available)
    columns = 2
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.0, 4.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for index, ax in enumerate(axes.flat):
        if index >= n:
            ax.set_axis_off()
            continue
        parameter = available[index]
        plot_parameter_performance(ax, successful, parameter, show_legend=True)
        panel_label(ax, f"({chr(ord('a') + index)})")
    save_figure(fig, output_dir / "fig01_core_hyperparameter_sensitivity", settings)
    plt.close(fig)


def plot_all_parameter_performance(
    successful: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    """Plot the primary performance curves for every available parameter."""
    import matplotlib.pyplot as plt

    parameters = list(dict.fromkeys(successful["parameter_name"].astype(str).tolist()))
    if not parameters:
        return
    columns = 3
    rows = math.ceil(len(parameters) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 3.8 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for index, ax in enumerate(axes.flat):
        if index >= len(parameters):
            ax.set_axis_off()
            continue
        plot_parameter_performance(
            ax,
            successful,
            parameters[index],
            show_legend=True,
        )
    save_figure(
        fig,
        output_dir / "fig05_all_hyperparameter_performance",
        settings,
    )
    plt.close(fig)


def plot_window_deltas(
    successful: pd.DataFrame, output_dir: Path, settings: Settings,
) -> None:
    """Default-relative differences in fractions (0.01 = one percentage point)."""
    import matplotlib.pyplot as plt
    parameters = list(dict.fromkeys(successful["parameter_name"].astype(str)))
    if not parameters:
        return
    columns = 2
    rows = math.ceil(len(parameters) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12., 4.5 * rows),
                             squeeze=False, constrained_layout=True)
    for index, ax in enumerate(axes.flat):
        if index >= len(parameters):
            ax.set_axis_off()
            continue
        parameter = parameters[index]
        subset, x, labels, numeric = ordered_parameter_data(successful, parameter)
        baseline = _window_baseline(subset)
        if baseline is None:
            ax.text(.5, .5, "No successful default;\nwindow deltas unavailable",
                    transform=ax.transAxes, ha="center", va="center")
        else:
            for (metric, label), (color, marker) in zip(WINDOW_METRICS.items(), WINDOW_STYLES):
                ax.plot(x, subset[metric] - float(baseline[metric]),
                        color=color, marker=marker, linewidth=1.3, label=label)
            ax.legend(frameon=False, loc="best")
        ax.axhline(0, color="gray", linewidth=.8)
        draw_default_line(ax, subset, x)
        ax.set_xlabel("Parameter value" if numeric else "Configuration")
        if not numeric:
            ax.set_xticks(x, labels, rotation=15, ha="right")
        ax.set_ylabel("Window-metric change from default")
        ax.set_title(PARAMETER_LABELS.get(parameter, parameter))
        ax.grid(alpha=.2)
    save_figure(fig, output_dir / "fig02_all_window_observability_deltas", settings)
    plt.close(fig)


def plot_window_robustness(
    robustness: pd.DataFrame, output_dir: Path, settings: Settings,
    filename: str = "fig03_parameter_robustness_ranking",
) -> None:
    """Rank tracking parameters using window metrics only; all are fractions."""
    import matplotlib.pyplot as plt
    ranked = robustness.dropna(subset=["max_abs_window_deviation"]).sort_values(
        "max_abs_window_deviation", ascending=True
    ).reset_index(drop=True)
    if ranked.empty:
        print("[WARN] Tracking ranking unavailable: no successful default window metrics.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14., max(5.5, .6 * len(ranked) + 2)),
                             constrained_layout=True)
    y = np.arange(len(ranked))
    ax = axes[0]
    bars = ax.barh(y, ranked["max_abs_window_deviation"], color="#1f77b4")
    ax.set_yticks(y, ranked["parameter_label"])
    ax.set_xlabel("Maximum absolute window-metric change")
    ax.set_title("Tracking sensitivity across three window metrics")
    ax.grid(axis="x", alpha=.2)
    for bar, value in zip(bars, ranked["max_abs_window_deviation"]):
        ax.text(value, bar.get_y() + bar.get_height()/2, f" {value:.3f}", va="center")
    ax = axes[1]
    width = .23
    for index, ((metric, label), (color, _marker)) in enumerate(zip(WINDOW_METRICS.items(), WINDOW_STYLES)):
        ax.barh(y + (index - 1) * width, ranked[f"max_abs_delta_{metric}"],
                height=width, label=label, color=color)
    ax.set_yticks(y, ranked["parameter_label"])
    ax.set_xlabel("Maximum absolute change from default")
    ax.set_title("Metric-specific tracking sensitivity")
    ax.legend(frameon=False, loc="best")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, output_dir / filename, settings)
    plt.close(fig)


def plot_all_delta_f1(
    successful: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    tracking = successful[successful["parameter_group"].isin(TRACKING_GROUPS)]
    if len(tracking):
        plot_window_deltas(tracking, output_dir, settings)
    successful = successful[~successful["parameter_group"].isin(TRACKING_GROUPS)]

    parameters = list(dict.fromkeys(successful["parameter_name"].astype(str).tolist()))
    if not parameters:
        return

    columns = 3
    rows = math.ceil(len(parameters) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(13.5, 3.5 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    for index, ax in enumerate(axes.flat):
        if index >= len(parameters):
            ax.set_axis_off()
            continue
        parameter = parameters[index]
        subset, x, labels, numeric = ordered_parameter_data(successful, parameter)
        if "delta_f1" in subset.columns and subset["delta_f1"].notna().any():
            y = subset["delta_f1"].to_numpy(dtype=float)
        else:
            defaults = subset[subset["is_default_configuration"]]
            baseline = (
                float(defaults.iloc[0]["ignition_f1"])
                if len(defaults)
                else float(subset.iloc[0]["ignition_f1"])
            )
            y = subset["ignition_f1"].to_numpy(dtype=float) - baseline

        ax.plot(x, y, marker="o", linewidth=1.2)
        ax.axhline(0, linewidth=0.8)
        defaults = np.where(subset["is_default_configuration"].to_numpy(dtype=bool))[0]
        for default_index in defaults:
            ax.scatter(
                [x[default_index]],
                [y[default_index]],
                marker="*",
                s=90,
                edgecolors="black",
                linewidths=0.7,
                zorder=6,
            )
        if not numeric:
            ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("$\\Delta$F1 from default")
        ax.set_title(PARAMETER_LABELS.get(parameter, parameter))
        ax.grid(alpha=0.2)
        panel_label(ax, f"({chr(ord('a') + index)})")

    save_figure(fig, output_dir / "fig02_all_delta_f1_sensitivity", settings)
    plt.close(fig)


def build_robustness_summary(successful: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parameter, subset in successful.groupby("parameter_name", sort=False):
        subset = subset.copy()
        if _is_tracking_parameter(subset):
            baseline = _window_baseline(subset)
            entry = {
                "parameter_group": str(subset.iloc[0]["parameter_group"]),
                "parameter_name": str(parameter),
                "parameter_label": PARAMETER_LABELS.get(str(parameter), str(parameter)),
                "n_successful_values": len(subset),
                "metric_family": "window_observability",
                "default_available": baseline is not None,
            }
            deviations = []
            for metric in WINDOW_METRICS:
                values = subset[metric]
                default = float(baseline[metric]) if baseline is not None else float("nan")
                deviation = float((values - default).abs().max())
                entry.update({f"default_{metric}": default, f"min_{metric}": float(values.min()),
                              f"max_{metric}": float(values.max()),
                              f"range_{metric}": float(values.max() - values.min()),
                              f"max_abs_delta_{metric}": deviation})
                deviations.append(deviation)
            finite = [value for value in deviations if np.isfinite(value)]
            entry["max_abs_window_deviation"] = max(finite) if finite else float("nan")
            rows.append(entry)
            continue
        default_rows = subset[subset["is_default_configuration"]]
        if len(default_rows):
            default_f1 = float(default_rows.iloc[0]["ignition_f1"])
            default_far = float(default_rows.iloc[0]["false_ignition_rate"])
            default_mae = float(default_rows.iloc[0]["onset_mae"])
        else:
            default_f1 = float(subset.iloc[0]["ignition_f1"])
            default_far = float(subset.iloc[0]["false_ignition_rate"])
            default_mae = float(subset.iloc[0]["onset_mae"])

        delta_f1 = subset["ignition_f1"].astype(float) - default_f1
        delta_far = subset["false_ignition_rate"].astype(float) - default_far
        delta_mae = subset["onset_mae"].astype(float) - default_mae
        rows.append(
            {
                "parameter_group": str(subset.iloc[0]["parameter_group"]),
                "parameter_name": str(parameter),
                "parameter_label": PARAMETER_LABELS.get(str(parameter), str(parameter)),
                "n_successful_values": int(len(subset)),
                "metric_family": "ignition_accuracy",
                "default_f1": default_f1,
                "min_f1": float(subset["ignition_f1"].min()),
                "max_f1": float(subset["ignition_f1"].max()),
                "max_abs_delta_f1": float(np.abs(delta_f1).max()),
                "f1_range": float(subset["ignition_f1"].max() - subset["ignition_f1"].min()),
                "max_abs_delta_far": float(np.abs(delta_far).max()),
                "max_abs_delta_onset_mae": float(np.abs(delta_mae).max()),
            }
        )
    return pd.DataFrame(rows)


def plot_robustness_ranking(
    robustness: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    if robustness.empty:
        return
    import matplotlib.pyplot as plt

    tracking = robustness[robustness["metric_family"].eq("window_observability")]
    if len(tracking):
        destination = "fig03_parameter_robustness_ranking" if len(tracking) == len(robustness) else "fig03_tracking_window_robustness_ranking"
        plot_window_robustness(tracking, output_dir, settings, destination)
    robustness = robustness[robustness["metric_family"].eq("ignition_accuracy")]
    if robustness.empty:
        return
    robustness = robustness.sort_values("max_abs_delta_f1", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), constrained_layout=True)
    y = np.arange(len(robustness))

    ax = axes[0]
    bars = ax.barh(y, robustness["max_abs_delta_f1"])
    ax.set_yticks(y, robustness["parameter_label"])
    ax.set_xlabel("Maximum absolute $\\Delta$F1")
    ax.set_title("Sensitivity ranking by F1 deviation")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, robustness["max_abs_delta_f1"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center")
    panel_label(ax, "(a)")

    ax = axes[1]
    width = 0.38
    bars1 = ax.barh(
        y - width / 2,
        robustness["max_abs_delta_far"],
        height=width,
        label="Max |ΔFAR|",
    )
    bars2 = ax.barh(
        y + width / 2,
        robustness["max_abs_delta_onset_mae"],
        height=width,
        label="Max |Δ onset MAE|",
    )
    ax.set_yticks(y, robustness["parameter_label"])
    ax.set_xlabel("Absolute deviation")
    ax.set_title("Error-rate and onset-deviation sensitivity")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    panel_label(ax, "(b)")

    save_figure(fig, output_dir / "fig03_parameter_robustness_ranking", settings)
    plt.close(fig)


def plot_voting_rule(
    successful: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    subset, x, labels, _ = ordered_parameter_data(successful, "rtv_voting_rule")
    if subset.empty:
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)

    ax = axes[0]
    width = 0.24
    for offset, (column, label) in zip(
        (-width, 0.0, width),
        (
            ("ignition_precision", "Precision"),
            ("ignition_recall", "Recall"),
            ("ignition_f1", "F1"),
        ),
    ):
        ax.bar(x + offset, subset[column], width, label=label)
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Classification trade-off of RTV voting rules")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(a)")

    ax = axes[1]
    bars = ax.bar(x, subset["false_ignition_rate"])
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("False ignition rate")
    ax.set_title("False-positive suppression of voting rules")
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, subset["false_ignition_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    panel_label(ax, "(b)")

    save_figure(fig, output_dir / "fig04_rtv_voting_rule", settings)
    plt.close(fig)


def plot_per_parameter(
    successful: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    parameters = list(dict.fromkeys(successful["parameter_name"].astype(str).tolist()))
    for parameter in parameters:
        parameter_rows = successful[
            successful["parameter_name"].astype(str).eq(parameter)
        ]
        group = (
            str(parameter_rows.iloc[0]["parameter_group"])
            if len(parameter_rows)
            else ""
        )
        is_tracking = group in {"ot", "repair"}
        columns = 3 if is_tracking else 2
        fig, axes = plt.subplots(
            1,
            columns,
            figsize=(6.3 * columns, 4.6),
            squeeze=False,
            constrained_layout=True,
        )
        if is_tracking:
            for ax, metric in zip(axes[0], WINDOW_METRICS):
                plot_window_observability(ax, successful, parameter, True, metric=metric)
        else:
            plot_parameter_performance(axes[0, 0], successful, parameter, show_legend=True)
            plot_parameter_timing(axes[0, 1], successful, parameter, show_legend=True)
        safe_name = "".join(character if character.isalnum() or character in "_-" else "_" for character in parameter)
        save_figure(fig, output_dir / safe_name, settings)
        plt.close(fig)


def build_parameter_coverage_summary(
    full: pd.DataFrame,
    successful: pd.DataFrame,
    expected_parameter_groups: dict[str, str] | None = None,
) -> pd.DataFrame:
    expected = (
        EXPECTED_PARAMETER_GROUPS
        if expected_parameter_groups is None
        else expected_parameter_groups
    )
    rows: list[dict[str, Any]] = []
    observed = list(dict.fromkeys(full["parameter_name"].astype(str).tolist()))
    ordered_names = list(expected)
    ordered_names.extend(name for name in observed if name not in expected)
    for name in ordered_names:
        attempted = full[full["parameter_name"].astype(str).eq(name)]
        plotted = successful[successful["parameter_name"].astype(str).eq(name)]
        statuses = sorted(
            set(attempted["run_status"].dropna().astype(str).str.strip().tolist())
        )
        attempted_values = list(
            dict.fromkeys(attempted["parameter_value"].dropna().astype(str).tolist())
        )
        plotted_values = list(
            dict.fromkeys(plotted["parameter_value"].dropna().astype(str).tolist())
        )
        rows.append(
            {
                "parameter_group": expected.get(
                    name,
                    str(attempted.iloc[0]["parameter_group"]) if len(attempted) else "",
                ),
                "parameter_name": name,
                "parameter_label": PARAMETER_LABELS.get(name, name),
                "expected_parameter": name in expected,
                "attempted_configurations": int(len(attempted)),
                "plotted_configurations": int(len(plotted)),
                "attempted_values": " | ".join(attempted_values),
                "plotted_values": " | ".join(plotted_values),
                "run_statuses": " | ".join(statuses),
                "has_input_rows": bool(len(attempted)),
                "has_plotted_rows": bool(len(plotted)),
                "all_attempted_rows_plotted": bool(
                    len(attempted) > 0 and len(attempted) == len(plotted)
                ),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    full: pd.DataFrame,
    successful: pd.DataFrame,
    unsuccessful: pd.DataFrame,
    robustness: pd.DataFrame,
    coverage: pd.DataFrame,
    expected_parameter_count: int,
    selected_groups: Sequence[str],
) -> None:
    lines = [
        "=== Hyperparameter visualization summary ===",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"Total configurations: {len(full)}",
        f"Successful configurations: {len(successful)}",
        f"Unsuccessful configurations: {len(unsuccessful)}",
        f"Successful parameter names: {successful['parameter_name'].nunique()}",
        f"Selected parameter groups: {', '.join(selected_groups)}",
        f"Expected parameter names: {expected_parameter_count}",
        "",
        "Status by parameter group:",
    ]
    status = (
        full.groupby(["parameter_group", "run_status"]).size().reset_index(name="count")
    )
    for _, row in status.iterrows():
        lines.append(
            f"  {row['parameter_group']} / {row['run_status']}: {int(row['count'])}"
        )
    tracking_robustness = robustness[robustness["metric_family"].eq("window_observability")]
    if len(tracking_robustness):
        lines.extend(["", "Tracking window-observability metrics:",
                      "  Pre-onset coverage: frames -10..-1 relative to GT onset.",
                      "  Window completeness: frames -5..+5 relative to GT onset.",
                      "  Observable transition: >=3 observations before and after GT onset on one matched trajectory.",
                      "  Unmatched GT-positive events retain zero support; negative GT events are excluded from window means.",
                      "  All changes are fractions; 0.01 corresponds to one percentage point.",
                      "  Tracking sensitivity ranking (largest maximum window change first):"])
        ranked_tracking = tracking_robustness.sort_values("max_abs_window_deviation", ascending=False)
        for _, row in ranked_tracking.iterrows():
            value = row["max_abs_window_deviation"]
            if pd.notna(value):
                lines.append(f"  {row['parameter_name']}: max absolute window change={value:.6f}")
            else:
                lines.append(f"  {row['parameter_name']}: no successful default; ranking unavailable")
    ignition_robustness = robustness[robustness["metric_family"].eq("ignition_accuracy")]
    if len(ignition_robustness):
        lines.extend(["", "Ignition F1 sensitivity ranking (largest first):"])
        ranked = ignition_robustness.sort_values("max_abs_delta_f1", ascending=False)
        for _, row in ranked.iterrows():
            lines.append(
                f"  {row['parameter_name']}: max |ΔF1|={row['max_abs_delta_f1']:.6f}"
            )
    if len(unsuccessful):
        lines.extend(
            [
                "",
                "Important:",
                "  Failed or skipped configurations are not included in performance plots.",
                "  Inspect data/unsuccessful_configurations.csv before interpreting missing groups.",
            ]
        )
    missing_parameters = coverage[
        coverage["expected_parameter"] & ~coverage["has_plotted_rows"]
    ]
    partially_plotted = coverage[
        coverage["has_input_rows"] & ~coverage["all_attempted_rows_plotted"]
    ]
    if len(missing_parameters):
        lines.extend(
            [
                "",
                "Expected parameters without plotted rows:",
                *[
                    f"  {name}"
                    for name in missing_parameters["parameter_name"].astype(str)
                ],
            ]
        )
    if len(partially_plotted):
        lines.extend(
            [
                "",
                "Parameters with unsuccessful configurations omitted from performance curves:",
                *[
                    f"  {row['parameter_name']}: plotted "
                    f"{int(row['plotted_configurations'])}/"
                    f"{int(row['attempted_configurations'])}"
                    for _, row in partially_plotted.iterrows()
                ],
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize successful hyperparameter sensitivity configurations."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--summary-xlsx",
        type=Path,
        help="Combined summary in XLSX or CSV format.",
    )
    input_group.add_argument(
        "--results-dir",
        type=Path,
        help=(
            "Output directory produced by run_parameter_sensitivity_window_observability.py; "
            "combined_summary.xlsx/.csv is resolved automatically"
        ),
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Plot only OT/repair using the three GT-centered window-observability metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hyperparameter_visualization_results"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-family",
        default="Times New Roman",
        help="Font family shared by titles, labels, ticks, legends, and annotations.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="Unified font size in points for all figure text.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    parser.add_argument(
        "--core-parameters",
        nargs="+",
        default=DEFAULT_CORE_PARAMETERS,
        help="Parameters included in the compact main sensitivity figure.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.dpi) <= 0:
        raise ValueError("--dpi must be positive")
    if not math.isfinite(float(args.font_size)) or float(args.font_size) <= 0:
        raise ValueError("--font-size must be a finite positive number")
    core_parameters = tuple(args.core_parameters)
    if args.tracking_only and core_parameters == tuple(DEFAULT_CORE_PARAMETERS):
        core_parameters = tuple(TRACKING_PARAMETERS)
    settings = Settings(
        dpi=int(args.dpi),
        formats=tuple(args.formats),
        core_parameters=core_parameters,
        font_family=str(args.font_family).strip(),
        font_size=float(args.font_size),
    )
    if not settings.font_family:
        raise ValueError("--font-family must not be empty")
    configure_matplotlib(settings)

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    per_parameter_dir = figure_dir / "per_parameter"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    per_parameter_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = resolve_summary_path(args.summary_xlsx, args.results_dir)
    full = read_summary(summary_path)
    if args.tracking_only:
        require_columns(full, ["parameter_group"])
        full = select_tracking_rows(full)
    full = clean_input(full)
    selected_groups = sorted(
        set(full["parameter_group"].astype(str).str.strip().str.lower())
    )
    if set(selected_groups).issubset(TRACKING_GROUPS) and settings.core_parameters == tuple(DEFAULT_CORE_PARAMETERS):
        settings.core_parameters = tuple(TRACKING_PARAMETERS)
    expected_parameter_groups = {
        name: group
        for name, group in EXPECTED_PARAMETER_GROUPS.items()
        if group in selected_groups
    }
    successful, unsuccessful = split_status(full)
    if successful.empty:
        status_counts = (
            full["run_status"].astype(str).str.strip().str.lower().value_counts()
        )
        raise RuntimeError(
            "No successful configurations are available for plotting. "
            "Run-status counts: "
            + ", ".join(
                f"{status}={int(count)}"
                for status, count in status_counts.items()
            )
        )

    successful.to_csv(
        data_dir / "successful_configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unsuccessful.to_csv(
        data_dir / "unsuccessful_configurations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_status_overview(full, figure_dir, settings)
    plot_core_parameters(successful, figure_dir, settings)
    plot_all_parameter_performance(successful, figure_dir, settings)
    plot_all_delta_f1(successful, figure_dir, settings)
    robustness = build_robustness_summary(successful)
    robustness.to_csv(
        data_dir / "parameter_robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage = build_parameter_coverage_summary(
        full,
        successful,
        expected_parameter_groups=expected_parameter_groups,
    )
    coverage.to_csv(
        data_dir / "parameter_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_parameters = coverage[
        coverage["expected_parameter"] & ~coverage["has_plotted_rows"]
    ]
    if len(missing_parameters):
        print(
            "[WARN] Expected parameters without successful rows: "
            + ", ".join(missing_parameters["parameter_name"].astype(str))
        )
    plot_robustness_ranking(robustness, figure_dir, settings)
    if not args.tracking_only:
        plot_voting_rule(successful, figure_dir, settings)
    plot_per_parameter(successful, per_parameter_dir, settings)

    write_summary(
        output_dir / "hyperparameter_visualization_summary.txt",
        full,
        successful,
        unsuccessful,
        robustness,
        coverage,
        expected_parameter_count=len(expected_parameter_groups),
        selected_groups=selected_groups,
    )
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": str(summary_path.resolve()),
        "tracking_only": bool(args.tracking_only),
        "tracking_metrics": list(WINDOW_METRICS),
        "tracking_delta_units": "fractions (0.01 = one percentage point)",
        "tracking_rank_basis": "max absolute default-relative change across the three window metrics",
        "selected_groups": selected_groups,
        "settings": {
            "dpi": settings.dpi,
            "formats": list(settings.formats),
            "core_parameters": list(settings.core_parameters),
            "font_family": settings.font_family,
            "font_size": settings.font_size,
        },
        "counts": {
            "total": int(len(full)),
            "successful": int(len(successful)),
            "unsuccessful": int(len(unsuccessful)),
            "expected_parameters": int(len(expected_parameter_groups)),
            "plotted_parameters": int(successful["parameter_name"].nunique()),
            "missing_expected_parameters": int(len(missing_parameters)),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[OK] Completed. Output: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
