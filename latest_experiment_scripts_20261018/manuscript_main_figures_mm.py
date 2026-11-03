#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate only the four figures recommended for the main manuscript.

Figures
-------
fig01_representative_cases
    Representative clear ignition, gap filling, prebirth recovery, and
    rejected weak/ambiguous transition cases. Each row combines raw-image
    snapshots, the 2-D trajectory, coordinate evolution, and the GMM state
    sequence.

fig02_combined_ablation
    A compact three-panel ablation figure aligned with the final ablation logic:
      (a) tracking restoration and GT-onset-centered observability;
      (b) baseline decision performance plus signed decision-module effects;
      (c) incremental feature-group effects.

fig03_key_parameter_sensitivity
    Four principal sensitivity analyses:
      maximum center distance, maximum GMM states,
      cold/hot persistence, and RTV threshold scale.

fig04_spatiotemporal_measurement
    Application-value figure showing ignition locations, ignition-front
    migration, temporal event activity with cumulative count, and
    stage-wise spatial density. All spatial axes in this application figure
    use mm (default calibration: 3.96 um/pixel); input coordinates remain pixels.

The script intentionally does NOT generate the supplementary figures produced
by the earlier visualization scripts.

All manuscript figures use caption-only styling: saved outputs contain no
subplot letters or axes titles. Complete figures and independently rendered
panels are both retained. Font sizes are controlled with
``--fig2-font-size`` through ``--fig5-font-size`` using the manuscript figure
numbering (internal prefixes ``fig01`` through ``fig04``).

Use ``--only-fig5`` (alias ``--only-application``) to generate only the final
application figure and its standalone panels. This mode requires only the
ignition summary and tracks table; it skips GT, raw images, and other figures.
Example: python manuscript_main_figures_mm.py --only-fig5 --fig5-font-size 18 --pixel-size-um 3.96

Example
-------
python manuscript_main_figures.py \
    --tracking-summary tracking_ablation_summary.xlsx \
    --method-summary method_summary.xlsx \
    --feature-summary feature_group_summary.xlsx \
    --parameter-summary combined_summary.xlsx \
    --ignition-summary ignition_summary.xlsx \
    --tracks-csv tracks_long.csv \
    --frames-dir /path/to/original/frames \
    --output-dir manuscript_figures \
    --representative-track-ids 3153 922 2599 840 \
    --fig2-font-size 16 --fig3-font-size 16 \
    --fig4-font-size 16 --fig5-font-size 16

If representative IDs are omitted, one candidate is selected automatically for
four predefined categories. Automatic selection is a screening aid; inspect the
chosen tracks before final submission.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TRACKING_ORDER = [
    "T0_center_nn",
    "T1_basic_ot_kalman",
    "T2_ot_kalman_rescue",
    "T2b_ot_kalman_rescue_gap",
    "T3_full_tracker",
]
TRACKING_LABELS = {
    "T0_center_nn": "T0\nCenter-NN",
    "T1_basic_ot_kalman": "T1\nOT-Kalman",
    "T2_ot_kalman_rescue": "T2\n+ Rescue",
    "T2b_ot_kalman_rescue_gap": "T2b\n+ Gap filling",
    "T3_full_tracker": "T3\n+ Prebirth",
}

# Panel (c) of Fig. 4 uses two explicitly contrasting, color-vision-friendly
# encodings: blue for the per-bin event count and vermilion for accumulation.
TEMPORAL_BAR_COLOR = "#4C78A8"
TEMPORAL_CUMULATIVE_COLOR = "#D55E00"

METHOD_LABELS = {
    "gray_p99_threshold": "Gray threshold",
    "multi_feature_change_score": "Change score",
    "multi_feature_change_score_rtv": "Change score + RTV",
    "gmm_segmentation_only": "GMM only",  # legacy alias
    "gmm_without_rtv": "GMM w/o RTV",
    "gmm_no_shape_precheck": "w/o shape precheck",
    "gmm_no_initial_hot_precheck": "w/o initial-hot precheck",
    "gmm_no_run_length_smoothing": "w/o run smoothing",
    "gmm_no_prior_cold_guard": "w/o prior-cold guard",
    "gmm_rtv_full": "Proposed GMM-RTV",
    "gmm_with_onset_refinement": "+ onset refinement",
}

FEATURE_LABELS = {
    "gray_only": "Gray only",
    "gray_bg": "Gray + background",
    "radiometric_core": "Radiometric core",
    "radiometric_color": "Radiometric + color",
    "radiometric_color_morph": "Radiometric + color + morphology",
    "full_state_no_diff": "Full state w/o difference",
    "full_state_plus_diff": "Full state + difference",
}

PARAMETER_LABELS = {
    "max_center_dist": "Maximum center distance",
    "gmm_max_states": "Maximum GMM states",
    "cold_hot_persistence": "Cold/hot persistence",
    "rtv_alpha": "RTV threshold scale",
}

REPRESENTATIVE_DEFAULT_LABELS = [
    "Clear ignition",
    "Ignition with gap filling",
    "Ignition with prebirth recovery",
    "Rejected weak/ambiguous transition",
]
AUXILIARY_TRACK_IDS = [5489, 6682, 957, 713, 290, 14, 6,1147]

POINT_TYPE_LABELS = {
    "detected": "Measured detection",
    "candidate": "Gap-filled point",
    "prebirth_candidate": "Prebirth recovery",
}
POINT_TYPE_MARKERS = {
    "detected": "o",
    "candidate": "s",
    "prebirth_candidate": "^",
}


@dataclass
class Settings:
    dpi: int
    formats: tuple[str, ...]
    fps: float
    temporal_bin_width: int
    heatmap_bins_x: int
    heatmap_bins_y: int
    heatmap_sigma: float
    crop_margin: int
    sequence_offsets: tuple[int, ...]
    # These follow the manuscript numbering: Figure 2--Figure 5 correspond to
    # the internal output prefixes fig01--fig04, respectively.
    fig2_font_size: float
    fig3_font_size: float
    fig4_font_size: float
    fig5_font_size: float
    # Calibration applies only to the final application figure (manuscript Fig. 5).
    pixel_size_um: float = 3.96


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            # Journal typography: use Times New Roman for normal text and math variables.
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            # Larger global typography for all manuscript figures.
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "lines.linewidth": 1.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: Any, base_path: Path, settings: Settings) -> None:
    prepare_figure_for_output(fig, base_path, settings)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in settings.formats:
        path = base_path.with_suffix(f".{extension}")
        kwargs = {"dpi": settings.dpi} if extension.lower() in {
            "png", "jpg", "jpeg", "tif", "tiff"
        } else {}
        fig.savefig(path, **kwargs)
        print(f"[OK] Saved: {path}")


def figure_font_size_for_path(base_path: Path, settings: Settings) -> float:
    """Map internal fig01--fig04 paths to manuscript Figure 2--Figure 5."""
    normalized = base_path.as_posix().lower()
    mapping = (
        ("fig01", settings.fig2_font_size),
        ("fig02", settings.fig3_font_size),
        ("fig03", settings.fig4_font_size),
        ("fig04", settings.fig5_font_size),
    )
    for token, size in mapping:
        if token in normalized:
            return float(size)
    return 14.0


def prepare_figure_for_output(
    fig: Any,
    base_path: Path,
    settings: Settings,
) -> None:
    """Apply adjustable typography and remove panel labels/subplot titles.

    Existing relative typography is retained: ``--figN-font-size`` scales all
    text in that manuscript figure relative to the script's original 14-pt
    base.  This keeps axis labels, ticks, legends, annotations, and colorbars
    visually consistent while exposing one simple interface per figure.
    """
    from matplotlib.text import Text

    if getattr(fig, "_manuscript_output_prepared", False):
        return

    target_size = figure_font_size_for_path(base_path, settings)
    if not np.isfinite(target_size) or target_size <= 0:
        raise ValueError(f"Figure font size must be positive, got {target_size!r}")

    # Subplot explanations belong in the manuscript caption, not inside axes.
    for ax in fig.axes:
        ax.set_title("")

    scale = target_size / 14.0
    for text_artist in fig.findobj(match=Text):
        text_value = str(text_artist.get_text()).strip()
        if re.fullmatch(r"\([A-Za-z]\)", text_value):
            text_artist.set_visible(False)
            continue
        current_size = float(text_artist.get_fontsize())
        if np.isfinite(current_size) and current_size > 0:
            text_artist.set_fontsize(current_size * scale)

    fig._manuscript_output_prepared = True


def panel_label(ax: Any, text: str, x: float = -0.12, y: float = 1.06) -> None:
    """Legacy no-op retained so older drawing calls produce no (a)--(d)."""
    return None


def require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def clean_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def ordered_subset(df: pd.DataFrame, key: str, order: Sequence[str]) -> pd.DataFrame:
    available = set(df[key].dropna().astype(str))
    sequence = [value for value in order if value in available]
    sequence.extend(
        value for value in df[key].dropna().astype(str).unique() if value not in sequence
    )
    categorical = pd.Categorical(df[key].astype(str), categories=sequence, ordered=True)
    return (
        df.assign(_order=categorical)
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_state_sequence(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(float(item)) for item in value]
    text = str(value).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, (list, tuple)):
        return []
    output: list[int] = []
    for item in parsed:
        try:
            output.append(int(float(item)))
        except (TypeError, ValueError):
            continue
    return output


def read_tracks_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    for kwargs in (
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": "\t", "encoding": "utf-8-sig"},
        {"sep": None, "engine": "python", "encoding": "utf-8-sig"},
    ):
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception:
            continue
        if "track_id" in df.columns and len(df.columns) >= 8:
            return df
    raise ValueError(f"Could not parse tracks table: {path}")


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return values.astype(float, copy=False)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(values.astype(float, copy=False), sigma=sigma)
    except ImportError:
        return values.astype(float, copy=False)


def frame_to_display(values: np.ndarray | pd.Series | float, fps: float) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    converted = array / fps * 1000.0 if fps > 0 else array
    return float(converted) if np.isscalar(values) else converted


def time_axis_label(fps: float) -> str:
    return "Time (ms)" if fps > 0 else "Global frame"


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------

def load_ablation_tables(
    tracking_path: Path,
    method_path: Path,
    feature_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the final tracking / decision / feature ablation summaries used by Fig. 2.

    The tracking panel intentionally uses GT-onset-centered observability metrics rather
    than downstream F1, because the restoration modules are evaluated by how much of the
    cold-to-hot trajectory history they make observable around the annotated ignition.
    """
    tracking = pd.read_excel(tracking_path)
    method = pd.read_excel(method_path)
    feature = pd.read_excel(feature_path)

    require_columns(
        tracking,
        [
            "variant",
            "fresh_t3_reference_coverage",
            "mean_pre_onset_coverage",
            "mean_ignition_window_completeness",
            "observable_transition_rate",
        ],
        "tracking summary",
    )
    require_columns(
        method,
        [
            "method",
            "ignition_precision",
            "ignition_recall",
            "ignition_f1",
            "false_ignition_rate",
        ],
        "method summary",
    )
    require_columns(
        feature,
        [
            "protocol",
            "feature_group",
            "ignition_precision",
            "ignition_recall",
            "ignition_f1",
            "false_ignition_rate",
            "onset_mae_common",
        ],
        "feature summary",
    )

    numeric = [
        "fresh_t3_reference_coverage",
        "mean_pre_onset_coverage",
        "mean_ignition_window_completeness",
        "observable_transition_rate",
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "false_ignition_rate",
        "missed_ignition_rate",
        "onset_mae",
        "onset_mae_common",
        "within_1_frame_acc_common",
        "within_3_frame_acc_common",
        "n_onset_eval_common",
    ]
    tracking = clean_numeric(tracking[tracking["variant"].notna()], numeric)
    method = clean_numeric(method[method["method"].notna()], numeric)
    feature = clean_numeric(
        feature[feature["protocol"].notna() & feature["feature_group"].notna()],
        numeric,
    )

    tracking = ordered_subset(tracking, "variant", TRACKING_ORDER)
    return tracking, method, feature


def load_parameter_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    require_columns(
        df,
        [
            "parameter_name",
            "parameter_value",
            "is_default_configuration",
            "run_status",
            "ignition_precision",
            "ignition_recall",
            "ignition_f1",
            "false_ignition_rate",
        ],
        "parameter summary",
    )
    numeric = [
        "parameter_value",
        "default_value",
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "false_ignition_rate",
        "reference_track_coverage",
        "suspicious_merge_proxy_count",
        "f1_ci_low",
        "f1_ci_high",
    ]
    df = clean_numeric(df, numeric)
    if df["is_default_configuration"].dtype != bool:
        df["is_default_configuration"] = (
            df["is_default_configuration"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    status = df["run_status"].astype(str).str.strip().str.lower()
    return df[status.isin(["success", "ok", "completed"])].copy()


def load_trajectory_inputs(
    ignition_path: Path,
    tracks_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_excel(ignition_path)
    tracks = read_tracks_table(tracks_path)
    require_columns(
        summary,
        [
            "track_id",
            "has_ignition",
            "ignition_status",
            "ignition_onset_frame",
            "gmm_state_sequence",
        ],
        "ignition summary",
    )
    require_columns(
        tracks,
        [
            "track_id",
            "frame_id",
            "image_name",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "point_type",
        ],
        "tracks table",
    )
    summary = summary.copy()
    tracks = tracks.copy()
    summary["track_id"] = pd.to_numeric(summary["track_id"], errors="raise").astype(int)
    tracks["track_id"] = pd.to_numeric(tracks["track_id"], errors="raise").astype(int)
    tracks["frame_id"] = pd.to_numeric(tracks["frame_id"], errors="raise").astype(int)
    summary["has_ignition"] = summary["has_ignition"].map(parse_bool)
    for column in (
        "ignition_idx_candidate",
        "ignition_idx_onset",
        "ignition_idx_confirmed",
        "ignition_onset_frame",
        "ignition_confidence",
        "gmm_n_states",
    ):
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    for column in ("x1", "y1", "x2", "y2", "cx", "cy"):
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    tracks["point_type"] = tracks["point_type"].fillna("detected").astype(str)
    tracks = tracks.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    summary = summary.sort_values("track_id").reset_index(drop=True)
    return summary, tracks


def _parse_gt_track_id(value: Any) -> Optional[int]:
    """Parse GT track IDs such as ``track_6`` or numeric values into integers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(-?\d+)$", text)
    return int(match.group(1)) if match else None


def load_gt_ignition_track_ids(path: Path) -> set[int]:
    """Load GT-positive track IDs from ``effective_trace.xlsx``.

    The GT file is expected to contain ``track_id`` and ``frame_id`` columns.
    A track is considered a valid ignition sequence only when its ``frame_id``
    entry is present and numeric. Rows with a missing/blank/non-numeric
    ``frame_id`` are explicitly excluded. Track IDs may be written as
    ``track_6`` or as bare numeric values.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"GT ignition table was not found: {path}. "
            "Pass the correct file with --gt-track-list."
        )

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix in {".tsv", ".txt"} else ","
        table = pd.read_csv(path, sep=separator)
    else:
        raise ValueError(
            f"Unsupported GT table format: {path.suffix}. "
            "Use xlsx, xls, csv, txt, or tsv."
        )

    table = table.copy()
    table.columns = [str(column).strip().lower() for column in table.columns]
    required = {"track_id", "frame_id"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"GT table {path} is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(table.columns)}"
        )

    parsed_frame = pd.to_numeric(table["frame_id"], errors="coerce")
    valid_rows = table[parsed_frame.notna()].copy()
    valid_rows["_parsed_track_id"] = valid_rows["track_id"].map(_parse_gt_track_id)
    valid_rows = valid_rows[valid_rows["_parsed_track_id"].notna()].copy()

    track_ids = {
        int(value)
        for value in valid_rows["_parsed_track_id"].tolist()
        if int(value) >= 0
    }
    if not track_ids:
        raise ValueError(
            f"No GT-positive tracks with a valid frame_id were found in: {path}"
        )

    excluded = int(len(table) - len(valid_rows))
    print(
        f"[OK] Loaded {len(track_ids)} GT-positive tracks from {path.name}; "
        f"excluded {excluded} rows without a valid frame_id or track_id."
    )
    return track_ids


# -----------------------------------------------------------------------------
# Figure 1: representative cases and tracking-mechanism candidate export
# -----------------------------------------------------------------------------

def _numeric_column(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def _mask_for_mechanism(trajectory: pd.DataFrame, mechanism: str) -> pd.Series:
    point_type = trajectory["point_type"].fillna("detected").astype(str)
    if mechanism == "gap_fill":
        return point_type.eq("candidate")
    if mechanism == "prebirth":
        return point_type.eq("prebirth_candidate")
    if mechanism == "rescue":
        rescue = np.isclose(_numeric_column(trajectory, "ot_prob", default=np.nan), -1.0, equal_nan=False)
        if "gap_fill_method" in trajectory.columns:
            method = trajectory["gap_fill_method"].fillna("").astype(str).str.lower()
            rescue = rescue | method.str.contains("rescue", regex=False)
        return pd.Series(rescue, index=trajectory.index)
    if mechanism == "measured":
        return point_type.eq("detected") & ~_mask_for_mechanism(trajectory, "rescue")
    raise ValueError(f"Unknown mechanism: {mechanism}")


def _contiguous_mask_segments(
    trajectory: pd.DataFrame,
    mask: pd.Series,
) -> list[pd.DataFrame]:
    ordered = trajectory.sort_values("frame_id").copy()
    active = mask.reindex(ordered.index, fill_value=False).to_numpy(dtype=bool)
    frames = ordered["frame_id"].to_numpy(dtype=int)
    segments: list[pd.DataFrame] = []
    start: Optional[int] = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        ends_here = start is not None and (
            not flag
            or index == len(active) - 1
            or (flag and index + 1 < len(active) and frames[index + 1] != frames[index] + 1)
        )
        if ends_here:
            stop = index + 1 if flag else index
            if stop > start:
                segments.append(ordered.iloc[start:stop].copy())
            start = None
    return segments


def _internal_gap_descriptors(trajectory: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = trajectory.sort_values("frame_id").reset_index(drop=True)
    gap_mask = _mask_for_mechanism(ordered, "gap_fill")
    measured_mask = ordered["point_type"].astype(str).eq("detected")
    track_start = int(ordered["frame_id"].min())
    track_end = int(ordered["frame_id"].max())
    duration = max(track_end - track_start, 1)
    descriptors: list[dict[str, Any]] = []

    for segment in _contiguous_mask_segments(ordered, gap_mask):
        start_frame = int(segment["frame_id"].min())
        end_frame = int(segment["frame_id"].max())
        before = ordered[measured_mask & (ordered["frame_id"] < start_frame)]
        after = ordered[measured_mask & (ordered["frame_id"] > end_frame)]
        if before.empty or after.empty:
            continue
        left = before.iloc[-1]
        right = after.iloc[0]
        midpoint = 0.5 * (start_frame + end_frame)
        relative_position = float((midpoint - track_start) / duration)
        centrality = float(max(0.0, 1.0 - abs(relative_position - 0.5) / 0.5))
        post_span = float(max(0.0, (track_end - float(right["frame_id"])) / duration))

        sample = pd.concat([left.to_frame().T, segment, right.to_frame().T], ignore_index=True)
        sample_frames = sample["frame_id"].to_numpy(dtype=float)
        frame_denominator = max(float(right["frame_id"] - left["frame_id"]), 1.0)
        alpha = (sample_frames - float(left["frame_id"])) / frame_denominator
        expected_x = float(left["cx"]) + alpha * (float(right["cx"]) - float(left["cx"]))
        expected_y = float(left["cy"]) + alpha * (float(right["cy"]) - float(left["cy"]))
        residual = np.hypot(
            sample["cx"].to_numpy(dtype=float) - expected_x,
            sample["cy"].to_numpy(dtype=float) - expected_y,
        )
        median_residual = float(np.nanmedian(residual)) if len(residual) else 0.0
        smoothness = float(1.0 / (1.0 + median_residual / 5.0))

        descriptors.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length": int(len(segment)),
                "mid_frame": float(midpoint),
                "relative_position": relative_position,
                "centrality": centrality,
                "post_span": post_span,
                "median_linear_residual_px": median_residual,
                "smoothness": smoothness,
                "left_measured_frame": int(left["frame_id"]),
                "right_measured_frame": int(right["frame_id"]),
            }
        )
    return descriptors


def summarize_track_mechanisms(trajectory: pd.DataFrame) -> dict[str, Any]:
    ordered = trajectory.sort_values("frame_id").copy()
    gap_segments = _contiguous_mask_segments(ordered, _mask_for_mechanism(ordered, "gap_fill"))
    prebirth_segments = _contiguous_mask_segments(ordered, _mask_for_mechanism(ordered, "prebirth"))
    internal_gaps = _internal_gap_descriptors(ordered)
    rescue_mask = _mask_for_mechanism(ordered, "rescue")
    rescue_frames = ordered.loc[rescue_mask, "frame_id"].astype(int).tolist()
    prebirth_frames = ordered.loc[_mask_for_mechanism(ordered, "prebirth"), "frame_id"].astype(int).tolist()

    mechanisms = []
    if internal_gaps:
        mechanisms.append("gap_fill")
    if rescue_frames:
        mechanisms.append("rescue")
    if prebirth_frames:
        mechanisms.append("prebirth")

    if mechanisms:
        # Keep the exact mechanism combination only as metadata. It is not a
        # fourth mechanism and no separate "combined" folder is created.
        category = "+".join(mechanisms)
    else:
        category = "none"

    best_gap = max(
        internal_gaps,
        key=lambda item: (
            item["length"],
            item["centrality"],
            item["smoothness"],
            item["post_span"],
        ),
        default=None,
    )
    max_prebirth_span = 0
    if prebirth_segments:
        max_prebirth_span = max(
            int(segment["frame_id"].max() - segment["frame_id"].min() + 1)
            for segment in prebirth_segments
        )

    return {
        "track_length": int(len(ordered)),
        "start_frame": int(ordered["frame_id"].min()),
        "end_frame": int(ordered["frame_id"].max()),
        "n_gap_points": int(_mask_for_mechanism(ordered, "gap_fill").sum()),
        "n_gap_segments": int(len(gap_segments)),
        "n_internal_gap_segments": int(len(internal_gaps)),
        "max_internal_gap_length": int(max((item["length"] for item in internal_gaps), default=0)),
        "best_gap_start_frame": int(best_gap["start_frame"]) if best_gap else -1,
        "best_gap_end_frame": int(best_gap["end_frame"]) if best_gap else -1,
        "best_gap_mid_frame": float(best_gap["mid_frame"]) if best_gap else np.nan,
        "best_gap_relative_position": float(best_gap["relative_position"]) if best_gap else np.nan,
        "best_gap_centrality": float(best_gap["centrality"]) if best_gap else 0.0,
        "best_gap_smoothness": float(best_gap["smoothness"]) if best_gap else 0.0,
        "best_gap_post_span": float(best_gap["post_span"]) if best_gap else 0.0,
        "n_rescue_points": int(len(rescue_frames)),
        "rescue_frames": rescue_frames,
        "n_prebirth_points": int(len(prebirth_frames)),
        "prebirth_start_frame": int(min(prebirth_frames)) if prebirth_frames else -1,
        "prebirth_end_frame": int(max(prebirth_frames)) if prebirth_frames else -1,
        "max_prebirth_span": int(max_prebirth_span),
        "mechanism_count": int(len(mechanisms)),
        "mechanisms": mechanisms,
        "mechanism_category": category,
        "internal_gap_details": internal_gaps,
    }


def build_track_statistics(
    summary: pd.DataFrame,
    tracks: pd.DataFrame,
    gt_ignition_track_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    gt_ids = gt_ignition_track_ids or set()
    for track_id, trajectory in tracks.groupby("track_id", sort=True):
        track_id_int = int(track_id)
        record = {
            "track_id": track_id_int,
            "gt_has_ignition": track_id_int in gt_ids,
        }
        record.update(summarize_track_mechanisms(trajectory))
        records.append(record)
    stats = pd.DataFrame(records)
    keep = [
        column for column in (
            "track_id",
            "has_ignition",
            "ignition_status",
            "ignition_onset_frame",
            "ignition_confidence",
            "gmm_n_states",
        ) if column in summary.columns
    ]
    stats = stats.merge(summary[keep], on="track_id", how="left")
    if "has_ignition" in stats.columns:
        stats["pred_has_ignition"] = stats["has_ignition"].map(parse_bool)
    else:
        stats["pred_has_ignition"] = False
    stats["is_true_positive"] = (
        stats["gt_has_ignition"].astype(bool)
        & stats["pred_has_ignition"].astype(bool)
        & stats.get("ignition_status", pd.Series(index=stats.index, dtype=str))
            .astype(str).eq("ignition_detected")
    )
    return stats


def _normalized_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    if numeric.empty:
        return numeric
    low = float(numeric.min())
    high = float(numeric.max())
    if np.isclose(low, high):
        return pd.Series(np.ones(len(numeric), dtype=float), index=numeric.index)
    return (numeric - low) / (high - low)


def select_representative_tracks(
    summary: pd.DataFrame,
    tracks: pd.DataFrame,
    requested: Optional[Sequence[int]],
    gt_ignition_track_ids: Optional[set[int]] = None,
) -> pd.DataFrame:
    stats = build_track_statistics(summary, tracks, gt_ignition_track_ids)
    if requested:
        rows = []
        for track_id, label in zip(requested, REPRESENTATIVE_DEFAULT_LABELS):
            match = stats[stats["track_id"] == int(track_id)]
            if match.empty:
                raise ValueError(f"Requested representative track {track_id} was not found.")
            record = match.iloc[0].to_dict()
            record["case_label"] = label
            rows.append(record)
        return pd.DataFrame(rows)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def choose(candidates: pd.DataFrame, label: str, score: pd.Series) -> None:
        candidates = candidates[~candidates["track_id"].isin(used)].copy()
        if candidates.empty:
            return
        score = score.reindex(candidates.index).fillna(0.0)
        candidates["_score"] = score
        row = candidates.sort_values(
            ["_score", "track_length"], ascending=False
        ).iloc[0].to_dict()
        row["case_label"] = label
        selected.append(row)
        used.add(int(row["track_id"]))

    # The first three rows must be true-positive ignition tracks: the track is
    # present in the GT list and the final detector also reports ignition.
    ignition = stats[stats["is_true_positive"].astype(bool)].copy()

    clear = ignition[
        (ignition["n_internal_gap_segments"] == 0)
        & (ignition["n_rescue_points"] == 0)
        & (ignition["n_prebirth_points"] == 0)
    ]
    clear_score = (
        0.75 * _normalized_score(clear.get("ignition_confidence", pd.Series(index=clear.index, dtype=float)))
        + 0.25 * _normalized_score(np.log1p(clear["track_length"]))
    )
    choose(clear, REPRESENTATIVE_DEFAULT_LABELS[0], clear_score)

    gap = ignition[
        (ignition["n_internal_gap_segments"] > 0)
        & (ignition["n_prebirth_points"] == 0)
    ]
    gap_score = (
        0.30 * _normalized_score(gap["max_internal_gap_length"])
        + 0.25 * _normalized_score(gap["best_gap_centrality"])
        + 0.20 * _normalized_score(gap["best_gap_smoothness"])
        + 0.15 * _normalized_score(gap["best_gap_post_span"])
        + 0.10 * _normalized_score(gap.get("ignition_confidence", pd.Series(index=gap.index, dtype=float)))
    )
    choose(gap, REPRESENTATIVE_DEFAULT_LABELS[1], gap_score)

    prebirth = ignition[ignition["n_prebirth_points"] > 0]
    prebirth_score = (
        0.45 * _normalized_score(prebirth["max_prebirth_span"])
        + 0.25 * _normalized_score(prebirth["n_prebirth_points"])
        + 0.20 * _normalized_score(prebirth.get("ignition_confidence", pd.Series(index=prebirth.index, dtype=float)))
        + 0.10 * _normalized_score(np.log1p(prebirth["track_length"]))
    )
    choose(prebirth, REPRESENTATIVE_DEFAULT_LABELS[2], prebirth_score)

    # The rejected row is selected only from GT-negative tracks, so it is a
    # genuine non-ignition example rather than a missed positive event.
    negative = stats[
        ~stats["gt_has_ignition"].astype(bool)
        & ~stats["pred_has_ignition"].astype(bool)
        & stats["ignition_status"].astype(str).isin(
            ["no_ignition_weak_transition", "ambiguous", "ambiguous_gmm_unstable"]
        )
    ]
    negative_score = (
        0.70 * _normalized_score(np.log1p(negative["track_length"]))
        + 0.30 * _normalized_score(negative["mechanism_count"])
    )
    choose(negative, REPRESENTATIVE_DEFAULT_LABELS[3], negative_score)

    if len(selected) < 4:
        fallback = stats[~stats["track_id"].isin(used)].sort_values(
            "track_length", ascending=False
        )
        for _, row in fallback.iterrows():
            if len(selected) >= 4:
                break
            record = row.to_dict()
            record["case_label"] = f"Additional representative case {len(selected) + 1}"
            selected.append(record)
            used.add(int(record["track_id"]))
    return pd.DataFrame(selected[:4])


def build_frame_index(frames_dir: Optional[Path]) -> dict[str, Path]:
    if frames_dir is None or not frames_dir.is_dir():
        return {}
    index: dict[str, Path] = {}
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"):
        for path in frames_dir.rglob(extension):
            index.setdefault(path.stem, path)
    return index


def representative_anchor(
    summary_row: pd.Series,
    trajectory: pd.DataFrame,
    case_label: str = "",
) -> int:
    label = str(case_label).lower()
    mechanisms = summarize_track_mechanisms(trajectory)
    if "gap" in label and mechanisms["best_gap_mid_frame"] == mechanisms["best_gap_mid_frame"]:
        return int(round(float(mechanisms["best_gap_mid_frame"])))
    if "prebirth" in label and mechanisms["n_prebirth_points"] > 0:
        return int(mechanisms["prebirth_end_frame"])
    if "rescue" in label and mechanisms["n_rescue_points"] > 0:
        return int(mechanisms["rescue_frames"][0])

    onset = pd.to_numeric(
        pd.Series([summary_row.get("ignition_onset_frame")]), errors="coerce"
    ).iloc[0]
    if np.isfinite(onset) and onset >= 0:
        return int(onset)
    candidate = pd.to_numeric(
        pd.Series([summary_row.get("ignition_idx_candidate")]), errors="coerce"
    ).iloc[0]
    if np.isfinite(candidate) and candidate >= 0 and int(candidate) < len(trajectory):
        return int(trajectory.iloc[int(candidate)]["frame_id"])
    states = parse_state_sequence(summary_row.get("gmm_state_sequence", []))
    if len(states) >= 2:
        changes = np.where(np.diff(np.asarray(states)) != 0)[0]
        if len(changes):
            index = int(changes[0] + 1)
            if index < len(trajectory):
                return int(trajectory.iloc[index]["frame_id"])
    return int(trajectory.iloc[len(trajectory) // 2]["frame_id"])


def closest_rows(trajectory: pd.DataFrame, requested_frames: Sequence[int]) -> list[pd.Series]:
    frames = trajectory["frame_id"].to_numpy(dtype=int)
    rows: list[pd.Series] = []
    for frame in requested_frames:
        index = int(np.argmin(np.abs(frames - int(frame))))
        rows.append(trajectory.iloc[index])
    return rows


def fixed_crop_bounds(rows: Sequence[pd.Series], margin: int) -> tuple[int, int, int, int]:
    x1 = min(float(row["x1"]) for row in rows)
    y1 = min(float(row["y1"]) for row in rows)
    x2 = max(float(row["x2"]) for row in rows)
    y2 = max(float(row["y2"]) for row in rows)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) + 2.0 * margin
    side = max(side, 48.0)
    return (
        int(math.floor(cx - side / 2)),
        int(math.floor(cy - side / 2)),
        int(math.ceil(cx + side / 2)),
        int(math.ceil(cy + side / 2)),
    )


def load_crop(path: Path, bounds: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = bounds
    width, height = image.size
    clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
    crop = image.crop(clipped)
    return np.asarray(crop), (clipped[0], clipped[1])


def _smoothed_path(trajectory: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = trajectory.sort_values("frame_id")
    x = ordered["cx"].to_numpy(dtype=float)
    y = ordered["cy"].to_numpy(dtype=float)
    if len(ordered) < 5:
        return x, y
    window = min(15, len(ordered) if len(ordered) % 2 == 1 else len(ordered) - 1)
    window = max(window, 5)
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(x, window_length=window, polyorder=2, mode="interp"),
            savgol_filter(y, window_length=window, polyorder=2, mode="interp"),
        )
    except Exception:
        return (
            pd.Series(x).rolling(window=5, center=True, min_periods=1).mean().to_numpy(),
            pd.Series(y).rolling(window=5, center=True, min_periods=1).mean().to_numpy(),
        )


def _segment_with_context(
    trajectory: pd.DataFrame,
    segment: pd.DataFrame,
) -> pd.DataFrame:
    ordered = trajectory.sort_values("frame_id").reset_index(drop=True)
    start_frame = int(segment["frame_id"].min())
    end_frame = int(segment["frame_id"].max())
    before = ordered[ordered["frame_id"] < start_frame].tail(1)
    after = ordered[ordered["frame_id"] > end_frame].head(1)
    return pd.concat([before, segment, after], ignore_index=True)


def _draw_frame_spans(
    ax: Any,
    segments: Sequence[pd.DataFrame],
    y: float,
    color: str,
    linewidth: float,
    label: Optional[str] = None,
    linestyle: str = "-",
) -> None:
    first = True
    for segment in segments:
        start = float(segment["frame_id"].min())
        end = float(segment["frame_id"].max())
        if np.isclose(start, end):
            ax.scatter(
                [start], [y], s=32, color=color, zorder=4,
                label=label if first else None,
            )
        else:
            ax.hlines(
                y, start, end, color=color, linewidth=linewidth,
                linestyle=linestyle, zorder=3,
                label=label if first else None,
            )
        first = False


def _mechanism_summary_text(trajectory: pd.DataFrame) -> str:
    info = summarize_track_mechanisms(trajectory)
    lines: list[str] = []
    if info["n_internal_gap_segments"]:
        suffix = "segment" if info["n_internal_gap_segments"] == 1 else "segments"
        lines.append(
            f"{info['n_internal_gap_segments']} internal gap {suffix}; "
            f"max {info['max_internal_gap_length']} frames"
        )
    if info["n_rescue_points"]:
        suffix = "match" if info["n_rescue_points"] == 1 else "matches"
        lines.append(f"{info['n_rescue_points']} rescue {suffix}")
    if info["n_prebirth_points"]:
        lines.append(f"{info['n_prebirth_points']}-frame prebirth extension")
    return "\n".join(lines) if lines else "No trajectory restoration"


def _safe_slug(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z]+", "_", str(text).strip().lower())
    text = text.strip("_")
    return text or "case"


def _build_case_context(
    summary_by_id: pd.DataFrame,
    tracks: pd.DataFrame,
    case: pd.Series,
    settings: Settings,
) -> dict[str, Any]:
    track_id = int(case["track_id"])
    trajectory = tracks[tracks["track_id"] == track_id].sort_values("frame_id").copy()
    summary_row = summary_by_id.loc[track_id]
    label = str(case.get("case_label", f"Track {track_id}"))
    anchor = representative_anchor(summary_row, trajectory, label)
    requested_frames = [anchor + offset for offset in settings.sequence_offsets]
    image_rows = closest_rows(trajectory, requested_frames)
    bounds = fixed_crop_bounds(image_rows, settings.crop_margin)
    onset = (
        int(summary_row.get("ignition_onset_frame", -1))
        if pd.notna(summary_row.get("ignition_onset_frame"))
        else -1
    )
    return {
        "track_id": track_id,
        "trajectory": trajectory,
        "summary_row": summary_row,
        "label": label,
        "image_rows": image_rows,
        "bounds": bounds,
        "onset": onset,
    }


def _draw_montage_panel(
    fig: Any,
    subplot_spec: Any,
    case_ctx: dict[str, Any],
    frame_index: dict[str, Path],
    sample_colors: Sequence[str],
    *,
    add_panel_label_text: Optional[str] = None,
) -> Any:
    montage_grid = subplot_spec.subgridspec(2, len(case_ctx["image_rows"]), height_ratios=[1.0, 0.13], hspace=0.02, wspace=0.04)
    first_ax = None
    for image_index, track_row in enumerate(case_ctx["image_rows"]):
        ax = fig.add_subplot(montage_grid[0, image_index])
        if first_ax is None:
            first_ax = ax
        stem = Path(str(track_row["image_name"])).stem
        image_path = frame_index.get(stem)
        frame_color = sample_colors[image_index % len(sample_colors)]
        if image_path is None:
            ax.text(0.5, 0.5, "image unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=9.5)
            ax.set_facecolor("0.94")
        else:
            image, origin = load_crop(image_path, case_ctx["bounds"])
            ax.imshow(image)
            point_type = str(track_row.get("point_type", "detected"))
            ax.text(
                0.03, 0.95,
                POINT_TYPE_LABELS.get(point_type, point_type),
                transform=ax.transAxes,
                ha="left", va="top", fontsize=8.8,
                bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none", "pad": 1.4},
            )
        ax.set_title("", pad=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
            spine.set_linewidth(1.5)
    label_ax = fig.add_subplot(montage_grid[1, :])
    label_ax.axis("off")
    label_ax.text(0.5, 0.55, case_ctx["label"], transform=label_ax.transAxes, ha="center", va="center", fontsize=12.5, fontweight="bold")
    if add_panel_label_text and first_ax is not None:
        panel_label(first_ax, add_panel_label_text, x=-0.42, y=1.18)
    return first_ax


def _draw_path_panel(
    fig: Any,
    subplot_spec: Any,
    case_ctx: dict[str, Any],
    sample_colors: Sequence[str],
    mechanism_colors: dict[str, str],
    *,
    add_panel_label_text: Optional[str] = None,
    show_legend: bool = False,
) -> Any:
    from matplotlib.lines import Line2D

    trajectory = case_ctx["trajectory"]
    image_rows = case_ctx["image_rows"]
    onset = case_ctx["onset"]
    ax_path = fig.add_subplot(subplot_spec)
    smooth_x, smooth_y = _smoothed_path(trajectory)
    ax_path.plot(smooth_x, smooth_y, color="0.55", linewidth=1.7, zorder=1)

    gap_segments = _contiguous_mask_segments(trajectory, _mask_for_mechanism(trajectory, "gap_fill"))
    prebirth_segments = _contiguous_mask_segments(trajectory, _mask_for_mechanism(trajectory, "prebirth"))
    rescue_rows = trajectory[_mask_for_mechanism(trajectory, "rescue")]

    for segment in gap_segments:
        bridge = _segment_with_context(trajectory, segment)
        ax_path.plot(bridge["cx"], bridge["cy"], color=mechanism_colors["gap_fill"], linewidth=2.8, linestyle="--", zorder=4)
    for segment in prebirth_segments:
        bridge = _segment_with_context(trajectory, segment)
        ax_path.plot(bridge["cx"], bridge["cy"], color=mechanism_colors["prebirth"], linewidth=2.8, linestyle="--", zorder=4)
    if not rescue_rows.empty:
        ax_path.scatter(rescue_rows["cx"], rescue_rows["cy"], marker="D", s=52, color=mechanism_colors["rescue"], edgecolors="white", linewidths=0.8, zorder=5)

    for image_index, track_row in enumerate(image_rows):
        ax_path.scatter(float(track_row["cx"]), float(track_row["cy"]), s=44, facecolors="none", edgecolors=sample_colors[image_index % len(sample_colors)], linewidths=1.5, zorder=6)

    if onset >= 0:
        onset_row = trajectory[trajectory["frame_id"] == onset]
        if not onset_row.empty:
            ax_path.scatter(onset_row.iloc[0]["cx"], onset_row.iloc[0]["cy"], marker="*", s=145, color=mechanism_colors["onset"], edgecolors="black", linewidths=0.7, zorder=7)
    ax_path.invert_yaxis()
    ax_path.set_xlabel("$c_x$ (px)", labelpad=3)
    ax_path.set_ylabel("$c_y$ (px)", labelpad=3)
    ax_path.tick_params(labelsize=11.5)
    ax_path.grid(alpha=0.20)
    if add_panel_label_text:
        panel_label(ax_path, add_panel_label_text)
    if show_legend:
        handles = [
            Line2D([0], [0], color="0.55", linewidth=1.7, label="Smoothed trajectory"),
            Line2D([0], [0], color=mechanism_colors["gap_fill"], linewidth=2.8, linestyle="--", label="Gap fill"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=mechanism_colors["rescue"], markeredgecolor="white", markersize=7, label="Rescue"),
            Line2D([0], [0], color=mechanism_colors["prebirth"], linewidth=2.8, linestyle="--", label="Prebirth"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=mechanism_colors["onset"], markeredgecolor="black", markersize=10, label="Ignition onset"),
        ]
        ax_path.legend(handles=handles, frameon=False, fontsize=8.8, loc="best")
    return ax_path


def _draw_lifecycle_panel(
    fig: Any,
    subplot_spec: Any,
    case_ctx: dict[str, Any],
    mechanism_colors: dict[str, str],
    *,
    add_panel_label_text: Optional[str] = None,
) -> Any:
    trajectory = case_ctx["trajectory"]
    onset = case_ctx["onset"]
    ax_life = fig.add_subplot(subplot_spec)
    lane_order = ["prebirth", "measured", "gap_fill", "rescue"]
    lane_y = {name: index for index, name in enumerate(lane_order)}
    gap_segments = _contiguous_mask_segments(trajectory, _mask_for_mechanism(trajectory, "gap_fill"))
    prebirth_segments = _contiguous_mask_segments(trajectory, _mask_for_mechanism(trajectory, "prebirth"))
    measured_segments = _contiguous_mask_segments(trajectory, _mask_for_mechanism(trajectory, "measured"))
    rescue_rows = trajectory[_mask_for_mechanism(trajectory, "rescue")]
    _draw_frame_spans(ax_life, prebirth_segments, lane_y["prebirth"], mechanism_colors["prebirth"], 5.0, label="Prebirth")
    _draw_frame_spans(ax_life, measured_segments, lane_y["measured"], mechanism_colors["measured"], 4.0, label="Measured")
    _draw_frame_spans(ax_life, gap_segments, lane_y["gap_fill"], mechanism_colors["gap_fill"], 5.0, label="Gap fill", linestyle="--")
    if not rescue_rows.empty:
        ax_life.scatter(rescue_rows["frame_id"], np.full(len(rescue_rows), lane_y["rescue"], dtype=float), marker="D", s=46, color=mechanism_colors["rescue"], edgecolors="white", linewidths=0.7, zorder=5)
    if onset >= 0:
        ax_life.axvline(onset, linestyle="--", linewidth=1.15, color=mechanism_colors["onset"], zorder=2)
    frame_min = float(trajectory["frame_id"].min())
    frame_max = float(trajectory["frame_id"].max())
    frame_padding = 0.02 * max(frame_max - frame_min, 1.0)
    ax_life.set_xlim(frame_min - frame_padding, frame_max + frame_padding)
    ax_life.set_ylim(-0.55, len(lane_order) - 0.45)
    ax_life.set_yticks([lane_y[name] for name in lane_order])
    ax_life.set_yticklabels(["Prebirth", "Measured", "Gap fill", "Rescue"])
    ax_life.set_xlabel("Global frame", labelpad=3)
    ax_life.set_ylabel("Trajectory support", labelpad=3)
    ax_life.tick_params(labelsize=11.0)
    ax_life.grid(alpha=0.18, axis="x")
    if add_panel_label_text:
        panel_label(ax_life, add_panel_label_text)
    return ax_life


def _draw_state_panel(
    fig: Any,
    subplot_spec: Any,
    case_ctx: dict[str, Any],
    *,
    add_panel_label_text: Optional[str] = None,
) -> Any:
    trajectory = case_ctx["trajectory"]
    summary_row = case_ctx["summary_row"]
    onset = case_ctx["onset"]
    ax_state = fig.add_subplot(subplot_spec)
    states = parse_state_sequence(summary_row.get("gmm_state_sequence", []))
    n = min(len(states), len(trajectory))
    if n:
        state_frames = trajectory.iloc[:n]["frame_id"].to_numpy(dtype=float)
        ax_state.step(state_frames, np.asarray(states[:n]), where="post", linewidth=1.6)
        ax_state.set_yticks(sorted(set(states[:n])))
    else:
        ax_state.text(0.5, 0.5, "State sequence unavailable", ha="center", va="center", transform=ax_state.transAxes)
    candidate_idx = summary_row.get("ignition_idx_candidate", np.nan)
    if pd.notna(candidate_idx) and float(candidate_idx) >= 0 and int(candidate_idx) < len(trajectory):
        ax_state.axvline(float(trajectory.iloc[int(candidate_idx)]["frame_id"]), linestyle=":", linewidth=1.1, label="Candidate")
    if onset >= 0:
        ax_state.axvline(onset, linestyle="--", linewidth=1.1, label="Onset")
    ax_state.set_xlabel("Global frame", labelpad=3)
    ax_state.set_ylabel("GMM state", labelpad=3)
    ax_state.tick_params(labelsize=11.0)
    ax_state.grid(alpha=0.18)
    if add_panel_label_text:
        panel_label(ax_state, add_panel_label_text)
    return ax_state


def _save_single_case_outputs(
    case_ctx: dict[str, Any],
    frame_index: dict[str, Path],
    output_dir: Path,
    settings: Settings,
    sample_colors: Sequence[str],
    mechanism_colors: dict[str, str],
    *,
    stem_prefix: str,
    save_row_figure: bool = True,
    save_individual_panels: bool = True,
) -> None:
    case_slug = f"track_{case_ctx['track_id']}_{_safe_slug(case_ctx['label'])}"
    case_dir = output_dir / case_slug
    case_dir.mkdir(parents=True, exist_ok=True)

    if save_row_figure:
        import matplotlib.pyplot as plt
        row_fig = plt.figure(figsize=(18.5, 4.6))
        gs = row_fig.add_gridspec(1, 4, width_ratios=[3.9, 1.65, 1.75, 1.55], wspace=0.38)
        _draw_montage_panel(row_fig, gs[0, 0], case_ctx, frame_index, sample_colors, add_panel_label_text="(a)")
        _draw_path_panel(row_fig, gs[0, 1], case_ctx, sample_colors, mechanism_colors, add_panel_label_text="(b)", show_legend=True)
        _draw_lifecycle_panel(row_fig, gs[0, 2], case_ctx, mechanism_colors, add_panel_label_text="(c)")
        _draw_state_panel(row_fig, gs[0, 3], case_ctx, add_panel_label_text="(d)")
        save_figure(row_fig, case_dir / f"{stem_prefix}_{case_slug}_1x4", settings)
        plt.close(row_fig)

    if save_individual_panels:
        import matplotlib.pyplot as plt
        panel_specs = [
            ("panel_a_montage", (11.2, 2.8), lambda f: _draw_montage_panel(f, f.add_gridspec(1,1)[0,0], case_ctx, frame_index, sample_colors, add_panel_label_text="(a)")),
            ("panel_b_path", (4.9, 4.2), lambda f: _draw_path_panel(f, f.add_gridspec(1,1)[0,0], case_ctx, sample_colors, mechanism_colors, add_panel_label_text="(b)", show_legend=True)),
            ("panel_c_support", (5.2, 4.2), lambda f: _draw_lifecycle_panel(f, f.add_gridspec(1,1)[0,0], case_ctx, mechanism_colors, add_panel_label_text="(c)")),
            ("panel_d_gmm_state", (4.4, 4.2), lambda f: _draw_state_panel(f, f.add_gridspec(1,1)[0,0], case_ctx, add_panel_label_text="(d)")),
        ]
        for panel_name, figsize, builder in panel_specs:
            fig = plt.figure(figsize=figsize)
            builder(fig)
            save_figure(fig, case_dir / f"{stem_prefix}_{case_slug}_{panel_name}", settings)
            plt.close(fig)


def _select_tracks_by_id(summary: pd.DataFrame, track_ids: Sequence[int], *, label_prefix: str = "Auxiliary track") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for tid in track_ids:
        if tid in seen:
            continue
        seen.add(int(tid))
        hit = summary[summary["track_id"] == int(tid)]
        if hit.empty:
            print(f"[WARN] Auxiliary track {tid} not found in ignition summary; skipped.")
            continue
        row = hit.iloc[0].to_dict()
        row["case_label"] = f"{label_prefix} {int(tid)}"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_auxiliary_track_cases(
    summary: pd.DataFrame,
    tracks: pd.DataFrame,
    track_ids: Sequence[int],
    frames_dir: Optional[Path],
    output_dir: Path,
    settings: Settings,
) -> None:
    selected = _select_tracks_by_id(summary, track_ids)
    if selected.empty:
        return
    frame_index = build_frame_index(frames_dir)
    summary_by_id = summary.set_index("track_id", drop=False)
    sample_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    mechanism_colors = {"measured": "#4c78a8", "gap_fill": "#f58518", "rescue": "#8e5cc2", "prebirth": "#54a24b", "onset": "#d62728"}
    aux_dir = output_dir / "fig01_auxiliary_track_cases"
    aux_dir.mkdir(parents=True, exist_ok=True)
    for _, case in selected.reset_index(drop=True).iterrows():
        case_ctx = _build_case_context(summary_by_id, tracks, case, settings)
        _save_single_case_outputs(case_ctx, frame_index, aux_dir, settings, sample_colors, mechanism_colors, stem_prefix="aux")


def plot_representative_cases(
    summary: pd.DataFrame,
    tracks: pd.DataFrame,
    selected: pd.DataFrame,
    frames_dir: Optional[Path],
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    frame_index = build_frame_index(frames_dir)
    summary_by_id = summary.set_index("track_id", drop=False)
    sample_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    mechanism_colors = {
        "measured": "#4c78a8",
        "gap_fill": "#f58518",
        "rescue": "#8e5cc2",
        "prebirth": "#54a24b",
        "onset": "#d62728",
    }

    # Save each representative case separately to avoid label/axis overlap during later layout composition.
    per_case_dir = output_dir / "fig01_representative_cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    case_contexts: list[dict[str, Any]] = []
    for _, case in selected.reset_index(drop=True).iterrows():
        case_ctx = _build_case_context(summary_by_id, tracks, case, settings)
        case_contexts.append(case_ctx)
        _save_single_case_outputs(
            case_ctx,
            frame_index,
            per_case_dir,
            settings,
            sample_colors,
            mechanism_colors,
            stem_prefix="fig01",
            save_row_figure=True,
            save_individual_panels=True,
        )

    # Also keep a single overview figure for quick inspection, but use larger spacing.
    fig = plt.figure(figsize=(18.8, 4.0 * len(case_contexts)))
    outer = fig.add_gridspec(
        len(case_contexts),
        4,
        width_ratios=[3.9, 1.8, 1.95, 1.65],
        hspace=0.60,
        wspace=0.40,
    )
    for row_index, case_ctx in enumerate(case_contexts):
        _draw_montage_panel(fig, outer[row_index, 0], case_ctx, frame_index, sample_colors, add_panel_label_text="(a)" if row_index == 0 else None)
        _draw_path_panel(fig, outer[row_index, 1], case_ctx, sample_colors, mechanism_colors, add_panel_label_text="(b)" if row_index == 0 else None, show_legend=(row_index == 0))
        _draw_lifecycle_panel(fig, outer[row_index, 2], case_ctx, mechanism_colors, add_panel_label_text="(c)" if row_index == 0 else None)
        _draw_state_panel(fig, outer[row_index, 3], case_ctx, add_panel_label_text="(d)" if row_index == 0 else None)
    save_figure(fig, output_dir / "fig01_representative_cases_overview", settings)
    plt.close(fig)



def _contact_sheet_frame_rows(
    trajectory: pd.DataFrame,
    max_frames: int,
) -> list[pd.Series]:
    ordered = trajectory.sort_values("frame_id").reset_index(drop=True)
    mechanism_mask = (
        _mask_for_mechanism(ordered, "gap_fill")
        | _mask_for_mechanism(ordered, "prebirth")
        | _mask_for_mechanism(ordered, "rescue")
    )
    mechanism_indices = np.flatnonzero(mechanism_mask.to_numpy(dtype=bool))
    context: set[int] = set()
    for index in mechanism_indices:
        context.update(range(max(0, index - 2), min(len(ordered), index + 3)))
    if not context:
        context.update(
            np.linspace(0, max(len(ordered) - 1, 0), min(max_frames, len(ordered)), dtype=int).tolist()
        )
    indices = sorted(context)
    if len(indices) > max_frames:
        mandatory = sorted(set(int(index) for index in mechanism_indices))
        if len(mandatory) >= max_frames:
            chosen = np.linspace(0, len(mandatory) - 1, max_frames, dtype=int)
            indices = [mandatory[int(position)] for position in chosen]
        else:
            remaining_slots = max_frames - len(mandatory)
            optional = [index for index in indices if index not in mandatory]
            sampled_optional = (
                np.asarray(optional)[
                    np.linspace(0, len(optional) - 1, remaining_slots, dtype=int)
                ].tolist()
                if optional and remaining_slots > 0
                else []
            )
            indices = sorted(set(mandatory + sampled_optional))
    return [ordered.iloc[index] for index in indices]


def _save_candidate_contact_sheet(
    trajectory: pd.DataFrame,
    frame_index: dict[str, Path],
    output_path: Path,
    crop_margin: int,
    max_frames: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rows = _contact_sheet_frame_rows(trajectory, max_frames=max_frames)
    if not rows:
        return
    cell_size = 170
    caption_height = 30
    columns = min(6, len(rows))
    n_rows = int(math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell_size, n_rows * (cell_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = {
        "measured": "#4c78a8",
        "gap_fill": "#f58518",
        "rescue": "#8e5cc2",
        "prebirth": "#54a24b",
    }

    for index, row in enumerate(rows):
        stem = Path(str(row["image_name"])).stem
        image_path = frame_index.get(stem)
        column = index % columns
        row_index = index // columns
        x0 = column * cell_size
        y0 = row_index * (cell_size + caption_height)
        if image_path is not None:
            local_bounds = fixed_crop_bounds([row], crop_margin)
            crop, _ = load_crop(image_path, local_bounds)
            tile = Image.fromarray(crop).resize((cell_size, cell_size))
            canvas.paste(tile, (x0, y0))
        else:
            draw.rectangle((x0, y0, x0 + cell_size - 1, y0 + cell_size - 1), fill="#eeeeee")

        one = pd.DataFrame([row])
        if bool(_mask_for_mechanism(one, "gap_fill").iloc[0]):
            mechanism = "gap_fill"
            label = "gap fill"
        elif bool(_mask_for_mechanism(one, "prebirth").iloc[0]):
            mechanism = "prebirth"
            label = "prebirth"
        elif bool(_mask_for_mechanism(one, "rescue").iloc[0]):
            mechanism = "rescue"
            label = "rescue"
        else:
            mechanism = "measured"
            label = "measured"
        color = colors[mechanism]
        draw.rectangle((x0 + 1, y0 + 1, x0 + cell_size - 2, y0 + cell_size - 2), outline=color, width=4)
        draw.text(
            (x0 + 5, y0 + cell_size + 5),
            f"Frame {int(row['frame_id'])} | {label}",
            fill="black", font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def export_tracking_mechanism_candidates(
    summary: pd.DataFrame,
    tracks: pd.DataFrame,
    gt_ignition_track_ids: set[int],
    frames_dir: Optional[Path],
    output_dir: Path,
    settings: Settings,
    export_mode: str = "full",
    contact_max_frames: int = 24,
) -> pd.DataFrame:
    """Export GT-positive trajectories using gap fill, rescue, or prebirth.

    Only tracks listed in the GT ignition-track file are exported. A canonical
    folder is written once per track under ``tracks/track_<id>``. The three
    mechanism folders may reference the same canonical track when more than one
    mechanism occurs; no misleading fourth "combined mechanism" is created.
    """
    if export_mode == "none":
        return pd.DataFrame()

    import shutil

    root = output_dir / "tracking_mechanism_candidates"
    canonical_root = root / "tracks"
    canonical_root.mkdir(parents=True, exist_ok=True)
    frame_index = build_frame_index(frames_dir)
    stats = build_track_statistics(summary, tracks, gt_ignition_track_ids)
    candidates = stats[
        (stats["mechanism_count"] > 0)
        & stats["gt_has_ignition"].astype(bool)
    ].copy()
    if candidates.empty:
        print(
            "[WARN] No GT-positive trajectories containing gap fill, rescue, "
            "or prebirth recovery were found."
        )
        return candidates

    csv_safe = candidates.copy()
    for column in ("mechanisms", "rescue_frames", "internal_gap_details"):
        if column in csv_safe.columns:
            csv_safe[column] = csv_safe[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
            )
    root.mkdir(parents=True, exist_ok=True)
    csv_safe.to_csv(root / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    try:
        csv_safe.to_excel(root / "candidate_summary.xlsx", index=False)
    except Exception as exc:
        print(f"[WARN] Could not write candidate_summary.xlsx: {exc}")

    category_filters = {
        "gap_fill_internal": csv_safe["n_internal_gap_segments"] > 0,
        "rescue": csv_safe["n_rescue_points"] > 0,
        "prebirth": csv_safe["n_prebirth_points"] > 0,
    }
    for category, category_mask in category_filters.items():
        category_dir = root / category
        category_dir.mkdir(parents=True, exist_ok=True)
        subset = csv_safe[category_mask].copy()
        subset.to_csv(category_dir / "index.csv", index=False, encoding="utf-8-sig")

    summary_by_id = summary.set_index("track_id", drop=False)
    for _, candidate in candidates.iterrows():
        track_id = int(candidate["track_id"])
        trajectory = tracks[tracks["track_id"] == track_id].sort_values("frame_id").copy()
        track_dir = canonical_root / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        trajectory.to_csv(track_dir / "track_points.csv", index=False, encoding="utf-8-sig")

        metadata = candidate.to_dict()
        if track_id in summary_by_id.index:
            metadata["ignition_summary"] = summary_by_id.loc[track_id].to_dict()
        (track_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        contact_path = track_dir / "contact_sheet.png"
        if frame_index:
            _save_candidate_contact_sheet(
                trajectory,
                frame_index,
                contact_path,
                crop_margin=settings.crop_margin,
                max_frames=max(6, int(contact_max_frames)),
            )

        if export_mode == "full" and frame_index:
            sequence_dir = track_dir / "full_sequence"
            sequence_dir.mkdir(parents=True, exist_ok=True)
            for _, point in trajectory.iterrows():
                stem = Path(str(point["image_name"])).stem
                source = frame_index.get(stem)
                if source is None:
                    continue
                destination = sequence_dir / f"frame_{int(point['frame_id']):07d}_{source.name}"
                if not destination.exists():
                    shutil.copy2(source, destination)

        # Put a lightweight contact-sheet copy in each applicable mechanism
        # folder, while the complete sequence remains in the canonical track dir.
        applicable = []
        if int(candidate["n_internal_gap_segments"]) > 0:
            applicable.append("gap_fill_internal")
        if int(candidate["n_rescue_points"]) > 0:
            applicable.append("rescue")
        if int(candidate["n_prebirth_points"]) > 0:
            applicable.append("prebirth")
        for category in applicable:
            category_dir = root / category
            link_file = category_dir / f"track_{track_id}.txt"
            link_file.write_text(
                f"Canonical folder: ../tracks/track_{track_id}\n",
                encoding="utf-8",
            )
            if contact_path.is_file():
                shutil.copy2(contact_path, category_dir / f"track_{track_id}_contact_sheet.png")

    print(
        f"[OK] Exported {len(candidates)} GT-positive tracking-mechanism "
        f"candidates: {root}"
    )
    return candidates


# -----------------------------------------------------------------------------
# Figure 2: combined ablation
# -----------------------------------------------------------------------------

def row_by_key(df: pd.DataFrame, key: str, value: str) -> Optional[pd.Series]:
    rows = df[df[key].astype(str).eq(value)]
    return None if rows.empty else rows.iloc[0]


def improvement(before: pd.Series, after: pd.Series, column: str, lower_better: bool = False) -> float:
    b = pd.to_numeric(pd.Series([before.get(column)]), errors="coerce").iloc[0]
    a = pd.to_numeric(pd.Series([after.get(column)]), errors="coerce").iloc[0]
    if not np.isfinite(b) or not np.isfinite(a):
        return np.nan
    return float(b - a if lower_better else a - b)


def build_method_baseline(method: pd.DataFrame) -> pd.DataFrame:
    """Absolute performance of the five representative decision baselines.

    These points answer a different question from the module-effect heatmap: they show
    why simple instantaneous/change-score criteria are insufficient, whereas the heatmap
    explains what each decision module changes inside the final pipeline.
    """
    keys = [
        "gray_p99_threshold",
        "multi_feature_change_score",
        "multi_feature_change_score_rtv",
        "gmm_without_rtv",
        "gmm_rtv_full",
    ]
    rows: list[dict[str, Any]] = []
    for key in keys:
        row = row_by_key(method, "method", key)
        if row is None:
            continue
        rows.append(
            {
                "method": key,
                "label": METHOD_LABELS.get(key, key),
                "Precision": float(row["ignition_precision"]),
                "Recall": float(row["ignition_recall"]),
                "F1": float(row["ignition_f1"]),
                "FAR": float(row["false_ignition_rate"]),
            }
        )
    return pd.DataFrame(rows)


def build_method_gain(method: pd.DataFrame) -> pd.DataFrame:
    """Signed effect obtained when each decision module is restored.

    Positive values always mean an improvement for the named metric. In particular,
    FAR reduction is computed as FAR(ablated/before) - FAR(after/full).
    """
    pairs = [
        ("RTV on change score", "multi_feature_change_score", "multi_feature_change_score_rtv"),
        ("RTV on GMM states", "gmm_without_rtv", "gmm_rtv_full"),
        ("Shape-sequence precheck", "gmm_no_shape_precheck", "gmm_rtv_full"),
        ("Initial-hot precheck", "gmm_no_initial_hot_precheck", "gmm_rtv_full"),
        ("Run-length smoothing", "gmm_no_run_length_smoothing", "gmm_rtv_full"),
        ("Prior-cold guard", "gmm_no_prior_cold_guard", "gmm_rtv_full"),
    ]
    rows: list[dict[str, Any]] = []
    for label, before_key, after_key in pairs:
        before = row_by_key(method, "method", before_key)
        after = row_by_key(method, "method", after_key)
        if before is None or after is None:
            continue
        rows.append(
            {
                "module": label,
                "$\\Delta$P": improvement(before, after, "ignition_precision"),
                "$\\Delta$R": improvement(before, after, "ignition_recall"),
                "$\\Delta$F1": improvement(before, after, "ignition_f1"),
                "FAR reduction": improvement(
                    before,
                    after,
                    "false_ignition_rate",
                    lower_better=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def final_feature_row(feature: pd.DataFrame, key: str) -> Optional[pd.Series]:
    rows = feature[
        feature["protocol"].astype(str).eq("final_detection")
        & feature["feature_group"].astype(str).eq(key)
    ]
    return None if rows.empty else rows.iloc[0]


def build_feature_gain(feature: pd.DataFrame) -> pd.DataFrame:
    """Sequential feature effects following the final 6-group feature hierarchy."""
    pairs = [
        ("+ Background", "gray_only", "gray_bg"),
        ("+ Radiometric core", "gray_bg", "radiometric_core"),
        ("+ Color", "radiometric_core", "radiometric_color"),
        ("+ Morphology", "radiometric_color", "radiometric_color_morph"),
        ("+ Temporal differences", "radiometric_color_morph", "full_state_plus_diff"),
    ]
    rows: list[dict[str, Any]] = []
    for label, before_key, after_key in pairs:
        before = final_feature_row(feature, before_key)
        after = final_feature_row(feature, after_key)
        if before is None or after is None:
            continue
        rows.append(
            {
                "module": label,
                "$\\Delta$P": improvement(before, after, "ignition_precision"),
                "$\\Delta$R": improvement(before, after, "ignition_recall"),
                "$\\Delta$F1": improvement(before, after, "ignition_f1"),
                "FAR reduction": improvement(
                    before,
                    after,
                    "false_ignition_rate",
                    lower_better=True,
                ),
                "Common MAE reduction": improvement(
                    before,
                    after,
                    "onset_mae_common",
                    lower_better=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def annotated_heatmap(
    ax: Any,
    table: pd.DataFrame,
    title: str,
    value_format: str = ".3f",
    annotation_fontsize: float = 10.0,
    tick_fontsize: float = 10.5,
    *,
    normalize_columns_for_color: bool = False,
    colorbar_label: str = "Signed effect (positive is better)",
) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        ax.text(0.5, 0.5, "No paired results", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    labels = table.iloc[:, 0].astype(str).tolist()
    columns = table.columns[1:].tolist()
    raw_matrix = table.iloc[:, 1:].to_numpy(dtype=float)

    if normalize_columns_for_color:
        display_matrix = raw_matrix.copy()
        for column in range(display_matrix.shape[1]):
            finite_col = display_matrix[:, column][np.isfinite(display_matrix[:, column])]
            bound_col = max(abs(float(finite_col.min())), abs(float(finite_col.max()))) if finite_col.size else 1.0
            if bound_col < 1e-12:
                bound_col = 1.0
            display_matrix[:, column] = display_matrix[:, column] / bound_col
        vmin, vmax = -1.0, 1.0
    else:
        display_matrix = raw_matrix
        finite = display_matrix[np.isfinite(display_matrix)]
        bound = max(abs(float(finite.min())), abs(float(finite.max()))) if finite.size else 1.0
        bound = max(bound, 1e-6)
        vmin, vmax = -bound, bound

    image = ax.imshow(display_matrix, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=22, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_title(title, fontsize=13.5, pad=10)
    for row in range(raw_matrix.shape[0]):
        for column in range(raw_matrix.shape[1]):
            value = raw_matrix[row, column]
            label_text = "NA" if not np.isfinite(value) else format(value, value_format)
            ax.text(
                column,
                row,
                label_text,
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
            )
    colorbar = plt.colorbar(image, ax=ax, fraction=0.038, pad=0.025)
    colorbar.set_label(colorbar_label, fontsize=10.5)
    colorbar.ax.tick_params(labelsize=9.5)


def _plot_tracking_observability(ax: Any, tracking: pd.DataFrame, *, show_panel_label: bool = True) -> None:
    x = np.arange(len(tracking))
    labels = [TRACKING_LABELS.get(value, value) for value in tracking["variant"].astype(str)]
    series = [
        ("Reference coverage", "fresh_t3_reference_coverage", "s", "--"),
        ("Pre-onset coverage", "mean_pre_onset_coverage", "o", "-"),
        ("Ignition-window completeness", "mean_ignition_window_completeness", "D", "-"),
        ("Observable-transition rate", "observable_transition_rate", "^", "-"),
    ]
    for label, column, marker, linestyle in series:
        ax.plot(
            x,
            tracking[column],
            marker=marker,
            markersize=6.5,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.tick_params(axis="y", labelsize=10.5)
    finite = tracking[[item[1] for item in series]].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    ymin = min(0.78, float(finite.min()) - 0.02) if finite.size else 0.78
    ax.set_ylim(ymin, 1.015)
    ax.set_ylabel("Coverage / observability", fontsize=11.5)
    ax.set_title("Tracking restoration and ignition-oriented observability", fontsize=13.5, pad=10)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, loc="lower right", fontsize=9.2)
    if show_panel_label:
        panel_label(ax, "(a)", x=-0.15, y=1.10)


def _plot_baseline_performance(ax: Any, baseline: pd.DataFrame) -> None:
    """Compact F1-vs-FAR performance map for representative decision baselines."""
    if baseline.empty:
        ax.text(0.5, 0.5, "No baseline results", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    markers = ["o", "s", "D", "^", "*"]
    for index, row in baseline.reset_index(drop=True).iterrows():
        marker = markers[min(index, len(markers) - 1)]
        size = 115 if str(row["method"]) == "gmm_rtv_full" else 62
        ax.scatter(float(row["FAR"]), float(row["F1"]), marker=marker, s=size, zorder=3)
        dx = 5
        dy = 5 if index % 2 == 0 else -12
        ax.annotate(
            str(row["label"]),
            (float(row["FAR"]), float(row["F1"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.3,
            ha="left",
            va="center",
        )
    xmax = max(0.56, float(baseline["FAR"].max()) * 1.08)
    ymin = min(0.30, float(baseline["F1"].min()) - 0.03)
    ymax = max(0.72, float(baseline["F1"].max()) + 0.03)
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("False ignition rate (lower is better)", fontsize=10.5)
    ax.set_ylabel("F1 score (higher is better)", fontsize=10.5)
    ax.set_title("Representative decision baselines", fontsize=12.5, pad=8)
    ax.grid(alpha=0.20)
    ax.tick_params(labelsize=9.5)


def plot_combined_ablation(
    tracking: pd.DataFrame,
    method: pd.DataFrame,
    feature: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import matplotlib.pyplot as plt

    baseline = build_method_baseline(method)
    method_gain = build_method_gain(method)
    feature_gain = build_feature_gain(feature)

    # Three manuscript panels in a spacious two-row layout. Panel (b) contains
    # two coordinated views across the bottom row. This remains readable when
    # --fig3-font-size is increased for manuscript typesetting.
    fig = plt.figure(figsize=(18.8, 11.2), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.30],
        height_ratios=[1.0, 1.05],
    )

    ax_a = fig.add_subplot(grid[0, 0])
    _plot_tracking_observability(ax_a, tracking, show_panel_label=True)

    bgrid = grid[1, :].subgridspec(
        1,
        2,
        width_ratios=[0.78, 1.42],
        wspace=0.18,
    )
    ax_b_left = fig.add_subplot(bgrid[0, 0])
    ax_b_right = fig.add_subplot(bgrid[0, 1])
    _plot_baseline_performance(ax_b_left, baseline)
    annotated_heatmap(
        ax_b_right,
        method_gain,
        "Ignition-decision module effects",
        annotation_fontsize=9.5,
        tick_fontsize=9.6,
        colorbar_label="Signed effect (positive is better)",
    )
    panel_label(ax_b_left, "(b)", x=-0.16, y=1.11)

    ax_c = fig.add_subplot(grid[0, 1])
    annotated_heatmap(
        ax_c,
        feature_gain,
        "Incremental feature-group effects",
        annotation_fontsize=9.6,
        tick_fontsize=10.0,
        normalize_columns_for_color=True,
        colorbar_label="Column-normalized signed effect",
    )
    panel_label(ax_c, "(c)", x=-0.13, y=1.10)

    save_figure(fig, output_dir / "fig02_combined_ablation", settings)
    plt.close(fig)

    # Save every Fig. 2 component separately for later manuscript assembly.
    panel_dir = output_dir / "fig02_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig_a, ax_a = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    _plot_tracking_observability(ax_a, tracking, show_panel_label=False)
    ax_a.set_title("Tracking restoration and ignition-oriented observability", fontsize=15.0, pad=10)
    ax_a.set_ylabel("Coverage / observability", fontsize=13)
    ax_a.tick_params(labelsize=11.3)
    panel_label(ax_a, "(a)", x=-0.14, y=1.07)
    save_figure(fig_a, panel_dir / "fig02a_tracking_ignition_observability", settings)
    plt.close(fig_a)

    # Full conceptual panel (b): baseline performance + module effects.
    fig_b = plt.figure(figsize=(13.0, 5.8), constrained_layout=True)
    bgrid = fig_b.add_gridspec(1, 2, width_ratios=[0.95, 1.45])
    ax_b1 = fig_b.add_subplot(bgrid[0, 0])
    ax_b2 = fig_b.add_subplot(bgrid[0, 1])
    _plot_baseline_performance(ax_b1, baseline)
    annotated_heatmap(
        ax_b2,
        method_gain,
        "Ignition-decision module effects",
        annotation_fontsize=11.0,
        tick_fontsize=11.0,
        colorbar_label="Signed effect (positive is better)",
    )
    panel_label(ax_b1, "(b)", x=-0.15, y=1.07)
    save_figure(fig_b, panel_dir / "fig02b_ignition_decision_baselines_and_module_effects", settings)
    plt.close(fig_b)

    # Also save the two internal parts of panel (b) independently.
    fig_b1, ax_b1 = plt.subplots(figsize=(5.7, 5.3), constrained_layout=True)
    _plot_baseline_performance(ax_b1, baseline)
    save_figure(fig_b1, panel_dir / "fig02b1_representative_decision_baselines", settings)
    plt.close(fig_b1)

    fig_b2, ax_b2 = plt.subplots(figsize=(8.2, 5.7), constrained_layout=True)
    annotated_heatmap(
        ax_b2,
        method_gain,
        "Ignition-decision module effects",
        annotation_fontsize=11.3,
        tick_fontsize=11.2,
        colorbar_label="Signed effect (positive is better)",
    )
    save_figure(fig_b2, panel_dir / "fig02b2_ignition_decision_module_effects", settings)
    plt.close(fig_b2)

    fig_c, ax_c = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    annotated_heatmap(
        ax_c,
        feature_gain,
        "Incremental feature-group effects",
        annotation_fontsize=11.0,
        tick_fontsize=11.0,
        normalize_columns_for_color=True,
        colorbar_label="Column-normalized signed effect",
    )
    panel_label(ax_c, "(c)", x=-0.12, y=1.07)
    save_figure(fig_c, panel_dir / "fig02c_incremental_feature_group_effects", settings)
    plt.close(fig_c)

    return baseline, method_gain, feature_gain


# -----------------------------------------------------------------------------
# Figure 3: key parameter sensitivity
# -----------------------------------------------------------------------------

def parameter_subset(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    subset = df[df["parameter_name"].astype(str).eq(name)].copy()
    if subset.empty:
        return subset, np.array([]), []
    numeric = pd.to_numeric(subset["parameter_value"], errors="coerce")
    if numeric.notna().all():
        subset["_x"] = numeric.astype(float)
        subset = subset.sort_values("_x").reset_index(drop=True)
        x = subset["_x"].to_numpy(dtype=float)
        labels = [f"{value:g}" for value in x]
    else:
        values = subset["parameter_value"].astype(str).tolist()
        subset = subset.reset_index(drop=True)
        x = np.arange(len(subset), dtype=float)
        labels = values
    return subset, x, labels


def f1_errorbars(subset: pd.DataFrame) -> Optional[np.ndarray]:
    if not {"f1_ci_low", "f1_ci_high"}.issubset(subset.columns):
        return None
    low = pd.to_numeric(subset["f1_ci_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(subset["f1_ci_high"], errors="coerce").to_numpy(dtype=float)
    center = pd.to_numeric(subset["ignition_f1"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        return None
    return np.vstack([center - low, high - center])


def mark_default(ax: Any, subset: pd.DataFrame, x: np.ndarray) -> None:
    positions = np.where(subset["is_default_configuration"].to_numpy(dtype=bool))[0]
    for position in positions:
        ax.axvline(x[position], linestyle="--", linewidth=0.9, alpha=0.7)
        y = float(subset.iloc[position]["ignition_f1"])
        ax.scatter([x[position]], [y], marker="*", s=90, edgecolors="black", linewidths=0.6, zorder=7)


def plot_tracking_parameter(ax: Any, df: pd.DataFrame, name: str) -> None:
    subset, x, labels = parameter_subset(df, name)
    if subset.empty:
        ax.set_axis_off()
        return
    ax.errorbar(x, subset["ignition_f1"], yerr=f1_errorbars(subset), marker="o", capsize=3, label="Ignition F1")
    ax.plot(x, subset["reference_track_coverage"], marker="s", label="Reference coverage")
    ax.set_ylim(0.40, 1.03)
    ax.set_ylabel("F1 / coverage")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Parameter value")
    ax.set_title(PARAMETER_LABELS[name])
    ax.grid(alpha=0.2)
    merge_ax = ax.twinx()
    merge_ax.plot(x, subset["suspicious_merge_proxy_count"], marker="D", linestyle="--", label="Merge-risk proxy")
    finite = subset["suspicious_merge_proxy_count"].dropna()
    if len(finite):
        pad = max(5.0, (float(finite.max()) - float(finite.min())) * 0.2)
        merge_ax.set_ylim(float(finite.min()) - pad, float(finite.max()) + pad)
    merge_ax.set_ylabel("Suspicious-merge proxy")
    mark_default(ax, subset, x)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = merge_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best")


def plot_decision_parameter(ax: Any, df: pd.DataFrame, name: str, show_legend: bool) -> None:
    subset, x, labels = parameter_subset(df, name)
    if subset.empty:
        ax.set_axis_off()
        return
    ax.errorbar(x, subset["ignition_f1"], yerr=f1_errorbars(subset), marker="o", capsize=3, label="F1")
    ax.plot(x, subset["ignition_precision"], marker="s", label="Precision")
    ax.plot(x, subset["ignition_recall"], marker="^", label="Recall")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Precision / recall / F1")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Parameter value")
    ax.set_title(PARAMETER_LABELS[name])
    ax.grid(alpha=0.2)
    far_ax = ax.twinx()
    far_ax.plot(x, subset["false_ignition_rate"], marker="D", linestyle="--", label="FAR")
    finite = subset["false_ignition_rate"].dropna()
    far_ax.set_ylim(0, max(0.11, float(finite.max()) * 1.25) if len(finite) else 0.11)
    far_ax.set_ylabel("False ignition rate")
    mark_default(ax, subset, x)
    if show_legend:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = far_ax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best")


def plot_key_parameter_sensitivity(
    parameters: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    plot_tracking_parameter(axes[0, 0], parameters, "max_center_dist")
    panel_label(axes[0, 0], "(a)")
    plot_decision_parameter(axes[0, 1], parameters, "gmm_max_states", show_legend=True)
    panel_label(axes[0, 1], "(b)")
    plot_decision_parameter(axes[1, 0], parameters, "cold_hot_persistence", show_legend=False)
    panel_label(axes[1, 0], "(c)")
    plot_decision_parameter(axes[1, 1], parameters, "rtv_alpha", show_legend=False)
    panel_label(axes[1, 1], "(d)")
    save_figure(fig, output_dir / "fig03_key_parameter_sensitivity", settings)
    plt.close(fig)

    # Save every Fig. 3 sensitivity panel as an independent figure.
    panel_dir = output_dir / "fig03_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        ("a", "max_center_dist", True, "fig03a_max_center_distance"),
        ("b", "gmm_max_states", False, "fig03b_maximum_gmm_states"),
        ("c", "cold_hot_persistence", False, "fig03c_cold_hot_persistence"),
        ("d", "rtv_alpha", False, "fig03d_rtv_threshold_scale"),
    ]
    for letter, name, is_tracking, stem in panel_specs:
        single_fig, single_ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
        if is_tracking:
            plot_tracking_parameter(single_ax, parameters, name)
        else:
            # Each independent panel gets its own legend.
            plot_decision_parameter(single_ax, parameters, name, show_legend=True)
        panel_label(single_ax, f"({letter})", x=-0.14, y=1.07)
        save_figure(single_fig, panel_dir / stem, settings)
        plt.close(single_fig)


# -----------------------------------------------------------------------------
# Figure 4: spatiotemporal application value
# -----------------------------------------------------------------------------

def build_ignition_events(summary: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    ignition = summary[
        summary["has_ignition"].map(parse_bool)
        & summary["ignition_status"].astype(str).eq("ignition_detected")
        & (pd.to_numeric(summary["ignition_onset_frame"], errors="coerce") >= 0)
    ].copy()
    merged = ignition.merge(
        tracks[["track_id", "frame_id", "cx", "cy", "point_type"]],
        left_on=["track_id", "ignition_onset_frame"],
        right_on=["track_id", "frame_id"],
        how="inner",
    )
    merged = merged.rename(columns={"ignition_onset_frame": "global_ignition_frame"})
    return merged.sort_values(["global_ignition_frame", "track_id"]).reset_index(drop=True)


def spatial_bounds(tracks: pd.DataFrame, margin_ratio: float = 0.03) -> tuple[float, float, float, float]:
    x = pd.concat([tracks["x1"], tracks["x2"], tracks["cx"]]).dropna().to_numpy(dtype=float)
    y = pd.concat([tracks["y1"], tracks["y2"], tracks["cy"]]).dropna().to_numpy(dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    xmargin = max(10.0, (xmax - xmin) * margin_ratio)
    ymargin = max(10.0, (ymax - ymin) * margin_ratio)
    return xmin - xmargin, xmax + xmargin, ymin - ymargin, ymax + ymargin


def make_heatmap(
    events: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    settings: Settings,
) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    hist, _, _ = np.histogram2d(
        events["cx"].to_numpy(dtype=float),
        events["cy"].to_numpy(dtype=float),
        bins=(settings.heatmap_bins_x, settings.heatmap_bins_y),
        range=((xmin, xmax), (ymin, ymax)),
    )
    return gaussian_smooth(hist.T, settings.heatmap_sigma)


def add_stage_annotation(
    ax: Any,
    stage_name: str,
    sample_count: int,
) -> None:
    """Keep stage information as an in-panel annotation, not a subplot title."""
    ax.text(
        0.04,
        0.95,
        f"{stage_name} stage  ($n$={sample_count})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=12.5,
        fontweight="bold",
        bbox={
            "facecolor": "black",
            "alpha": 0.55,
            "edgecolor": "none",
            "pad": 2.2,
        },
        zorder=5,
    )


def plot_spatiotemporal_measurement(
    events: pd.DataFrame,
    tracks: pd.DataFrame,
    output_dir: Path,
    settings: Settings,
) -> None:
    """Plot the spatiotemporal distribution of detected ignition events.

    Layout
    ------
    (a) Event locations colored by ignition time.
    (b) Ignition-front migration along the dominant spatial axis.
    (c) Temporal event rate and cumulative event count in a shared panel.
    (d) Early/middle/late spatial-density maps spanning the full figure width.
    """
    if events.empty:
        print("[WARN] No matched ignition events; skip fig04.")
        return
    import matplotlib.pyplot as plt

    mm_per_pixel = float(settings.pixel_size_um) / 1000.0
    if not np.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("pixel_size_um must be a finite positive value")
    # Keep PCA, histogram binning, smoothing, and exported source data in pixels.
    # Convert only coordinates passed to the plotting artists, including panels.
    event_positions_mm = events[["cx", "cy"]].to_numpy(dtype=float) * mm_per_pixel

    # Three-row layout: the stage-wise density maps occupy the entire bottom
    # row, preventing each image from being compressed into one quarter of the
    # original canvas.
    fig = plt.figure(figsize=(14.8, 11.2))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 0.82, 1.18],
        hspace=0.38,
        wspace=0.28,
    )

    bounds_px = spatial_bounds(tracks)
    bounds = tuple(value * mm_per_pixel for value in bounds_px)
    xmin, xmax, ymin, ymax = bounds
    frames = events["global_ignition_frame"].to_numpy(dtype=float)
    display_time = np.asarray(frame_to_display(frames, settings.fps), dtype=float)
    time_label = time_axis_label(settings.fps)

    width = max(1, int(settings.temporal_bin_width))
    frame_min = int(np.floor(frames.min()))
    frame_max = int(np.ceil(frames.max()))
    edges = np.arange(frame_min, frame_max + width + 1, width, dtype=float)
    if len(edges) < 2:
        edges = np.array([frame_min, frame_min + width], dtype=float)
    counts, edges = np.histogram(frames, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    display_edges = np.asarray(frame_to_display(edges, settings.fps), dtype=float)
    display_centers = np.asarray(frame_to_display(centers, settings.fps), dtype=float)
    display_width = (
        width / settings.fps * 1000.0
        if settings.fps > 0
        else float(width)
    )

    # ------------------------------------------------------------------
    # (a) Spatiotemporal ignition locations
    # ------------------------------------------------------------------
    ax_loc = fig.add_subplot(grid[0, 0])
    scatter = ax_loc.scatter(
        event_positions_mm[:, 0],
        event_positions_mm[:, 1],
        c=display_time,
        cmap="viridis",
        s=24,
        alpha=0.82,
        linewidths=0,
    )
    ax_loc.set_xlim(xmin, xmax)
    ax_loc.set_ylim(ymax, ymin)
    ax_loc.set_aspect("equal", adjustable="box")
    ax_loc.set_xlabel("Horizontal position, $C_x$ (mm)", fontsize=15)
    ax_loc.set_ylabel("Vertical position, $C_y$ (mm)", fontsize=15)
    ax_loc.set_title("Spatiotemporal ignition locations", fontsize=17, pad=10)
    ax_loc.tick_params(labelsize=13)
    colorbar = fig.colorbar(scatter, ax=ax_loc, fraction=0.046, pad=0.035)
    colorbar.set_label(time_label, fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    panel_label(ax_loc, "(a)")

    # ------------------------------------------------------------------
    # (b) Ignition-front migration along the dominant spatial direction
    # ------------------------------------------------------------------
    ax_front = fig.add_subplot(grid[0, 1])
    coordinates = events[["cx", "cy"]].to_numpy(dtype=float)
    spatial_center = np.nanmean(coordinates, axis=0)
    centered = coordinates - spatial_center

    if len(coordinates) >= 2 and np.all(np.isfinite(centered)):
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    else:
        principal_axis = np.array([1.0, 0.0], dtype=float)

    # Fix the arbitrary PCA sign so that the dominant component is positive;
    # this makes repeated runs and different datasets easier to compare.
    dominant_component = int(np.argmax(np.abs(principal_axis)))
    if principal_axis[dominant_component] < 0:
        principal_axis = -principal_axis

    principal_coordinate = centered @ principal_axis
    s_min = float(np.nanmin(principal_coordinate))
    s_max = float(np.nanmax(principal_coordinate))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or np.isclose(s_min, s_max):
        s_min, s_max = -0.5, 0.5
    s_pad = max(1.0, 0.04 * (s_max - s_min))
    spatial_edges = np.linspace(
        s_min - s_pad,
        s_max + s_pad,
        max(24, int(settings.heatmap_bins_y // 2)) + 1,
    )
    front_hist, _, _ = np.histogram2d(
        frames,
        principal_coordinate,
        bins=(edges, spatial_edges),
    )
    front_density = gaussian_smooth(front_hist.T, sigma=1.0)
    spatial_edges = spatial_edges * mm_per_pixel
    front_image = ax_front.pcolormesh(
        display_edges,
        spatial_edges,
        front_density,
        shading="auto",
        cmap="magma",
    )

    median_s = np.full(len(centers), np.nan, dtype=float)
    q25_s = np.full(len(centers), np.nan, dtype=float)
    q75_s = np.full(len(centers), np.nan, dtype=float)
    for bin_index in range(len(centers)):
        if bin_index < len(centers) - 1:
            mask = (frames >= edges[bin_index]) & (frames < edges[bin_index + 1])
        else:
            mask = (frames >= edges[bin_index]) & (frames <= edges[bin_index + 1])
        if np.any(mask):
            values = principal_coordinate[mask]
            median_s[bin_index] = float(np.nanmedian(values))
            q25_s[bin_index] = float(np.nanpercentile(values, 25))
            q75_s[bin_index] = float(np.nanpercentile(values, 75))

    median_s *= mm_per_pixel
    q25_s *= mm_per_pixel
    q75_s *= mm_per_pixel
    valid_front = np.isfinite(median_s)
    if np.any(valid_front):
        ax_front.plot(
            display_centers[valid_front],
            median_s[valid_front],
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            label="Median front position",
        )
        ax_front.fill_between(
            display_centers[valid_front],
            q25_s[valid_front],
            q75_s[valid_front],
            alpha=0.22,
            label="Interquartile range",
        )
        ax_front.legend(frameon=False, fontsize=11.5, loc="best")

    axis_angle = float(np.degrees(np.arctan2(principal_axis[1], principal_axis[0])))
    ax_front.text(
        0.02,
        0.97,
        f"Principal-axis angle: {axis_angle:.1f}°",
        transform=ax_front.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
    )
    ax_front.set_xlabel(time_label, fontsize=15)
    ax_front.set_ylabel("Principal spatial coordinate, $S$ (mm)", fontsize=15)
    ax_front.set_title("Ignition-front migration", fontsize=17, pad=10)
    ax_front.tick_params(labelsize=13)
    front_cbar = fig.colorbar(front_image, ax=ax_front, fraction=0.046, pad=0.035)
    front_cbar.set_label("Event density", fontsize=14)
    front_cbar.ax.tick_params(labelsize=12)
    panel_label(ax_front, "(b)")

    # ------------------------------------------------------------------
    # (c) Event rate + cumulative count in one coordinated panel
    # ------------------------------------------------------------------
    ax_time = fig.add_subplot(grid[1, :])
    bars = ax_time.bar(
        display_centers,
        counts,
        width=display_width * 0.88,
        color=TEMPORAL_BAR_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.7,
        label="Ignition events per bin",
        zorder=2,
    )
    ax_time.set_xlabel(time_label, fontsize=15)
    ax_time.set_ylabel("Ignition events per bin", fontsize=15, color=TEMPORAL_BAR_COLOR)
    ax_time.set_title("Temporal ignition activity and accumulation", fontsize=17, pad=10)
    ax_time.tick_params(axis="y", labelsize=13, colors=TEMPORAL_BAR_COLOR)
    ax_time.grid(axis="y", alpha=0.22, zorder=0)

    cumulative_ax = ax_time.twinx()
    cumulative_counts = np.cumsum(counts)
    cumulative_line, = cumulative_ax.plot(
        display_centers,
        cumulative_counts,
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        color=TEMPORAL_CUMULATIVE_COLOR,
        label="Cumulative ignition count",
        zorder=4,
    )
    cumulative_ax.fill_between(
        display_centers,
        0,
        cumulative_counts,
        color=TEMPORAL_CUMULATIVE_COLOR,
        alpha=0.10,
        zorder=1,
    )
    cumulative_ax.set_ylabel("Cumulative ignition count", fontsize=15, color=TEMPORAL_CUMULATIVE_COLOR)
    cumulative_ax.set_ylim(0, max(1.0, float(cumulative_counts.max()) * 1.08))
    cumulative_ax.tick_params(axis="y", labelsize=13, colors=TEMPORAL_CUMULATIVE_COLOR)
    ax_time.legend(
        [bars, cumulative_line],
        ["Ignition events per bin", "Cumulative ignition count"],
        frameon=False,
        ncol=2,
        loc="upper left",
        fontsize=12,
    )
    panel_label(ax_time, "(c)", x=-0.055)

    # ------------------------------------------------------------------
    # (d) Stage-wise spatial-density maps across the full figure width
    # ------------------------------------------------------------------
    stage_grid = grid[2, :].subgridspec(1, 3, wspace=0.12)
    stage_edges = np.linspace(float(frames.min()), float(frames.max()) + 1e-9, 4)
    stage_names = ["Early", "Middle", "Late"]
    stage_data: list[tuple[pd.DataFrame, np.ndarray]] = []
    vmax = 0.0
    for index in range(3):
        if index < 2:
            subset = events[
                (events["global_ignition_frame"] >= stage_edges[index])
                & (events["global_ignition_frame"] < stage_edges[index + 1])
            ]
        else:
            subset = events[
                (events["global_ignition_frame"] >= stage_edges[index])
                & (events["global_ignition_frame"] <= stage_edges[index + 1])
            ]
        heatmap = make_heatmap(subset, bounds_px, settings)
        vmax = max(vmax, float(heatmap.max()) if heatmap.size else 0.0)
        stage_data.append((subset, heatmap))

    image = None
    stage_axes = []
    for index, (subset, heatmap) in enumerate(stage_data):
        stage_ax = fig.add_subplot(stage_grid[0, index])
        stage_axes.append(stage_ax)
        image = stage_ax.imshow(
            heatmap,
            origin="upper",
            extent=(xmin, xmax, ymax, ymin),
            cmap="inferno",
            vmin=0,
            vmax=max(vmax, 1e-8),
            interpolation="bilinear",
            aspect="equal",
        )
        stage_ax.set_xlim(xmin, xmax)
        stage_ax.set_ylim(ymax, ymin)
        stage_ax.set_aspect("equal", adjustable="box")
        add_stage_annotation(stage_ax, stage_names[index], len(subset))
        stage_ax.set_xlabel("$C_x$ (mm)", fontsize=15)
        stage_ax.tick_params(labelsize=12.5)
        if index == 0:
            stage_ax.set_ylabel("$C_y$ (mm)", fontsize=15)
            panel_label(stage_ax, "(d)", x=-0.22)
        else:
            stage_ax.tick_params(labelleft=False)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=stage_axes,
            fraction=0.024,
            pad=0.025,
            aspect=35,
        )
        cbar.set_label("Ignition-event density", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    save_figure(fig, output_dir / "fig04_spatiotemporal_measurement", settings)
    plt.close(fig)

    # Save every Fig. 4 subpanel separately. Panel (d) is saved both as the
    # three-stage group and as three additional single-stage maps.
    panel_dir = output_dir / "fig04_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # (a) Spatiotemporal ignition locations
    fig_a, ax_a = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    scatter_a = ax_a.scatter(event_positions_mm[:, 0], event_positions_mm[:, 1], c=display_time, cmap="viridis", s=28, alpha=0.82, linewidths=0)
    ax_a.set_xlim(xmin, xmax)
    ax_a.set_ylim(ymax, ymin)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Horizontal position, $C_x$ (mm)", fontsize=15)
    ax_a.set_ylabel("Vertical position, $C_y$ (mm)", fontsize=15)
    ax_a.set_title("Spatiotemporal ignition locations", fontsize=17, pad=10)
    ax_a.tick_params(labelsize=13)
    cb_a = fig_a.colorbar(scatter_a, ax=ax_a, fraction=0.048, pad=0.04)
    cb_a.set_label(time_label, fontsize=14)
    cb_a.ax.tick_params(labelsize=12)
    panel_label(ax_a, "(a)", x=-0.16, y=1.07)
    save_figure(fig_a, panel_dir / "fig04a_spatiotemporal_ignition_locations", settings)
    plt.close(fig_a)

    # (b) Ignition-front migration
    fig_b, ax_b = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    front_b = ax_b.pcolormesh(display_edges, spatial_edges, front_density, shading="auto", cmap="magma")
    if np.any(valid_front):
        ax_b.plot(display_centers[valid_front], median_s[valid_front], marker="o", markersize=5.0, linewidth=1.9, label="Median front position")
        ax_b.fill_between(display_centers[valid_front], q25_s[valid_front], q75_s[valid_front], alpha=0.22, label="Interquartile range")
        ax_b.legend(frameon=False, fontsize=11.5, loc="best")
    ax_b.text(0.02, 0.97, f"Principal-axis angle: {axis_angle:.1f}°", transform=ax_b.transAxes, ha="left", va="top", fontsize=11.5, bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5})
    ax_b.set_xlabel(time_label, fontsize=15)
    ax_b.set_ylabel("Principal spatial coordinate, $S$ (mm)", fontsize=15)
    ax_b.set_title("Ignition-front migration", fontsize=17, pad=10)
    ax_b.tick_params(labelsize=13)
    cb_b = fig_b.colorbar(front_b, ax=ax_b, fraction=0.046, pad=0.04)
    cb_b.set_label("Event density", fontsize=14)
    cb_b.ax.tick_params(labelsize=12)
    panel_label(ax_b, "(b)", x=-0.13, y=1.07)
    save_figure(fig_b, panel_dir / "fig04b_ignition_front_migration", settings)
    plt.close(fig_b)

    # (c) Temporal ignition activity and cumulative count
    fig_c, ax_c = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    bars_c = ax_c.bar(display_centers, counts, width=display_width * 0.88, color=TEMPORAL_BAR_COLOR, alpha=0.72, edgecolor="white", linewidth=0.7, label="Ignition events per bin", zorder=2)
    ax_c.set_xlabel(time_label, fontsize=15)
    ax_c.set_ylabel("Ignition events per bin", fontsize=15, color=TEMPORAL_BAR_COLOR)
    ax_c.set_title("Temporal ignition activity and accumulation", fontsize=17, pad=10)
    ax_c.tick_params(axis="y", labelsize=13, colors=TEMPORAL_BAR_COLOR)
    ax_c.grid(axis="y", alpha=0.22, zorder=0)
    cumulative_c = ax_c.twinx()
    cumulative_line_c, = cumulative_c.plot(display_centers, cumulative_counts, marker="o", markersize=4.8, linewidth=2.0, color=TEMPORAL_CUMULATIVE_COLOR, label="Cumulative ignition count", zorder=4)
    cumulative_c.fill_between(display_centers, 0, cumulative_counts, color=TEMPORAL_CUMULATIVE_COLOR, alpha=0.10, zorder=1)
    cumulative_c.set_ylabel("Cumulative ignition count", fontsize=15, color=TEMPORAL_CUMULATIVE_COLOR)
    cumulative_c.set_ylim(0, max(1.0, float(cumulative_counts.max()) * 1.08))
    cumulative_c.tick_params(axis="y", labelsize=13, colors=TEMPORAL_CUMULATIVE_COLOR)
    ax_c.legend([bars_c, cumulative_line_c], ["Ignition events per bin", "Cumulative ignition count"], frameon=False, ncol=2, loc="upper left", fontsize=12)
    panel_label(ax_c, "(c)", x=-0.075, y=1.07)
    save_figure(fig_c, panel_dir / "fig04c_temporal_ignition_activity", settings)
    plt.close(fig_c)

    # (d) Three stage-wise density maps as one independent subfigure
    fig_d = plt.figure(figsize=(12.8, 4.2), constrained_layout=True)
    dgrid = fig_d.add_gridspec(1, 3, wspace=0.08)
    d_axes = []
    d_image = None
    for index, (subset, heatmap) in enumerate(stage_data):
        dax = fig_d.add_subplot(dgrid[0, index])
        d_axes.append(dax)
        d_image = dax.imshow(heatmap, origin="upper", extent=(xmin, xmax, ymax, ymin), cmap="inferno", vmin=0, vmax=max(vmax, 1e-8), interpolation="bilinear", aspect="equal")
        dax.set_xlim(xmin, xmax)
        dax.set_ylim(ymax, ymin)
        dax.set_aspect("equal", adjustable="box")
        add_stage_annotation(dax, stage_names[index], len(subset))
        dax.set_xlabel("$C_x$ (mm)", fontsize=15)
        dax.tick_params(labelsize=12.5)
        if index == 0:
            dax.set_ylabel("$C_y$ (mm)", fontsize=15)
            panel_label(dax, "(d)", x=-0.22, y=1.07)
        else:
            dax.tick_params(labelleft=False)
    if d_image is not None:
        cb_d = fig_d.colorbar(d_image, ax=d_axes, fraction=0.025, pad=0.025, aspect=35)
        cb_d.set_label("Ignition-event density", fontsize=14)
        cb_d.ax.tick_params(labelsize=12)
    save_figure(fig_d, panel_dir / "fig04d_stagewise_spatial_density", settings)
    plt.close(fig_d)

    # Additional single-stage maps, useful if the three maps are laid out manually.
    for index, (subset, heatmap) in enumerate(stage_data):
        single_fig, single_ax = plt.subplots(figsize=(5.0, 4.5), constrained_layout=True)
        single_img = single_ax.imshow(heatmap, origin="upper", extent=(xmin, xmax, ymax, ymin), cmap="inferno", vmin=0, vmax=max(vmax, 1e-8), interpolation="bilinear", aspect="equal")
        single_ax.set_xlim(xmin, xmax)
        single_ax.set_ylim(ymax, ymin)
        single_ax.set_aspect("equal", adjustable="box")
        add_stage_annotation(single_ax, stage_names[index], len(subset))
        single_ax.set_xlabel("$C_x$ (mm)", fontsize=15)
        single_ax.set_ylabel("$C_y$ (mm)", fontsize=15)
        single_ax.tick_params(labelsize=12.5)
        cb_single = single_fig.colorbar(single_img, ax=single_ax, fraction=0.05, pad=0.04)
        cb_single.set_label("Ignition-event density", fontsize=14)
        cb_single.ax.tick_params(labelsize=12)
        save_figure(single_fig, panel_dir / f"fig04d_{stage_names[index].lower()}_stage_density", settings)
        plt.close(single_fig)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------

def positive_font_size(value: str) -> float:
    """Argparse type for a positive font size expressed in points."""
    try:
        size = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("font size must be a number") from exc
    if not np.isfinite(size) or size <= 0:
        raise argparse.ArgumentTypeError("font size must be greater than zero")
    return size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate only the optimized main-manuscript figures."
    )
    parser.add_argument("--tracking-summary", default=r'draw_res_final\tracking_ablation_summary_with_ignition.xlsx', type=Path)
    parser.add_argument("--method-summary", default=r'draw_res_final\method_summary.xlsx', type=Path)
    parser.add_argument("--feature-summary", default=r'draw_res_final\feature_group_summary.xlsx', type=Path)
    parser.add_argument("--parameter-summary", default=r'draw_res_final\combined_summary.xlsx', type=Path)
    parser.add_argument("--ignition-summary", default=r'tracking_ablation_results\run_20260719_121436\T3_full_tracker\ignition_summary.xlsx', type=Path)
    parser.add_argument("--tracks-csv", default=r'tracking_ablation_results\run_20260719_121436\T3_full_tracker\tracks_long.csv', type=Path)
    parser.add_argument(
        "--gt-track-list",
        type=Path,
        default=Path("effective_trace.xlsx"),
        help=(
            "GT table containing track_id and frame_id columns. Only rows with a "
            "valid frame_id are treated as true ignition tracks and exported."
        ),
    )
    parser.add_argument("--frames-dir", type=Path, default='inputs/MCFF0004-N001')
    parser.add_argument("--output-dir", type=Path, default=Path("manuscript_main_figures"))
    parser.add_argument("--representative-track-ids", nargs=4, type=int, default=None)
    parser.add_argument("--auxiliary-track-ids", nargs="*", type=int, default=AUXILIARY_TRACK_IDS, help="Additional candidate track IDs to render in the same 4-panel style, saved one figure at a time for later manual selection.")
    figure_mode = parser.add_mutually_exclusive_group()
    figure_mode.add_argument("--only-fig1", action="store_true", help="Generate Fig. 1 and its tracking-candidate folders only; skip Figs. 2-4.")
    figure_mode.add_argument("--only-fig2", action="store_true", help="Generate only the final Fig. 2 ablation figure from tracking/method/feature summary spreadsheets; Fig. 1/3/4 inputs are not required.")
    figure_mode.add_argument("--only-fig5", "--only-application", dest="only_fig5", action="store_true", help="Generate only manuscript Figure 5 (internal fig04 application statistics), including standalone panels. Requires only --ignition-summary and --tracks-csv; skips GT, frames, candidate exports, and other summary spreadsheets.")
    parser.add_argument("--candidate-export-mode", choices=("none", "contact", "full"), default="full", help="Export no candidates, contact sheets only, or complete frame sequences.")
    parser.add_argument("--candidate-contact-max-frames", type=int, default=24, help="Maximum number of mechanism-centered frames in each candidate contact sheet.")
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument(
        "--pixel-size-um", type=float, default=3.96,
        help="Spatial calibration in um/pixel for manuscript Fig. 5 only (default: 3.96). Input tracks and exported CSV coordinates remain in pixels.",
    )
    parser.add_argument("--temporal-bin-width", type=int, default=100)
    parser.add_argument("--sequence-frame-offsets", nargs="+", type=int, default=(-6, -3, 0, 3, 6))
    parser.add_argument("--crop-margin", type=int, default=24)
    parser.add_argument("--heatmap-bins-x", type=int, default=96)
    parser.add_argument("--heatmap-bins-y", type=int, default=72)
    parser.add_argument("--heatmap-sigma", type=float, default=1.8)
    parser.add_argument(
        "--fig2-font-size",
        type=positive_font_size,
        default=16.0,
        help="Base font size for manuscript Figure 2 (internal fig01 representative cases).",
    )
    parser.add_argument(
        "--fig3-font-size",
        type=positive_font_size,
        default=16.0,
        help="Base font size for manuscript Figure 3 (internal fig02 ablation results).",
    )
    parser.add_argument(
        "--fig4-font-size",
        type=positive_font_size,
        default=16.0,
        help="Base font size for manuscript Figure 4 (internal fig03 parameter sensitivity).",
    )
    parser.add_argument(
        "--fig5-font-size",
        type=positive_font_size,
        default=16.0,
        help="Base font size for manuscript Figure 5 (internal fig04 spatiotemporal results).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=("png", "pdf"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not np.isfinite(args.pixel_size_um) or args.pixel_size_um <= 0:
        parser.error("--pixel-size-um must be a finite positive number")
    configure_matplotlib()
    settings = Settings(
        dpi=int(args.dpi),
        formats=tuple(args.formats),
        fps=float(args.fps),
        temporal_bin_width=int(args.temporal_bin_width),
        heatmap_bins_x=int(args.heatmap_bins_x),
        heatmap_bins_y=int(args.heatmap_bins_y),
        heatmap_sigma=float(args.heatmap_sigma),
        crop_margin=int(args.crop_margin),
        sequence_offsets=tuple(int(value) for value in args.sequence_frame_offsets),
        fig2_font_size=float(args.fig2_font_size),
        fig3_font_size=float(args.fig3_font_size),
        fig4_font_size=float(args.fig4_font_size),
        fig5_font_size=float(args.fig5_font_size),
        pixel_size_um=float(args.pixel_size_um),
    )

    output_dir = args.output_dir
    figure_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.only_fig5:
        summary, tracks = load_trajectory_inputs(args.ignition_summary, args.tracks_csv)
        events = build_ignition_events(summary, tracks)
        if events.empty:
            raise ValueError("No ignition events matched their onset-frame trajectory samples; cannot generate Figure 5")
        events.to_csv(data_dir / "matched_ignition_events.csv", index=False, encoding="utf-8-sig")
        plot_spatiotemporal_measurement(events, tracks, figure_dir, settings)
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "only_fig5": True,
            "inputs": {
                "ignition_summary": str(args.ignition_summary.resolve()),
                "tracks_csv": str(args.tracks_csv.resolve()),
            },
            "n_matched_ignition_events": int(len(events)),
            "figures": [
                "fig04_spatiotemporal_measurement",
                "fig04_panels/fig04a_spatiotemporal_ignition_locations",
                "fig04_panels/fig04b_ignition_front_migration",
                "fig04_panels/fig04c_temporal_ignition_activity",
                "fig04_panels/fig04d_stagewise_spatial_density",
                "fig04_panels/fig04d_early_stage_density",
                "fig04_panels/fig04d_middle_stage_density",
                "fig04_panels/fig04d_late_stage_density",
            ],
            "settings": {
                "dpi": settings.dpi,
                "formats": list(settings.formats),
                "fps": settings.fps,
                "temporal_bin_width": settings.temporal_bin_width,
                "heatmap_bins_x": settings.heatmap_bins_x,
                "heatmap_bins_y": settings.heatmap_bins_y,
                "heatmap_sigma": settings.heatmap_sigma,
                "fig5_font_size": settings.fig5_font_size,
                "pixel_size_um": settings.pixel_size_um,
                "application_spatial_unit": "mm",
                "input_coordinate_unit": "pixel",
            },
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[OK] Manuscript Figure 5 completed ({len(events)} events). Output: {output_dir.resolve()}")
        return

    if args.only_fig2:
        tracking, method, feature = load_ablation_tables(
            args.tracking_summary,
            args.method_summary,
            args.feature_summary,
        )
        baseline, method_gain, feature_gain = plot_combined_ablation(
            tracking, method, feature, figure_dir, settings
        )
        baseline.to_csv(data_dir / "method_baseline_performance.csv", index=False, encoding="utf-8-sig")
        method_gain.to_csv(data_dir / "method_module_effect.csv", index=False, encoding="utf-8-sig")
        feature_gain.to_csv(data_dir / "feature_incremental_effect.csv", index=False, encoding="utf-8-sig")
        manifest = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "only_fig2": True,
            "inputs": {
                "tracking_summary": str(args.tracking_summary.resolve()),
                "method_summary": str(args.method_summary.resolve()),
                "feature_summary": str(args.feature_summary.resolve()),
            },
            "figures": [
                "fig02_combined_ablation",
                "fig02_panels/fig02a_tracking_ignition_observability",
                "fig02_panels/fig02b_ignition_decision_baselines_and_module_effects",
                "fig02_panels/fig02b1_representative_decision_baselines",
                "fig02_panels/fig02b2_ignition_decision_module_effects",
                "fig02_panels/fig02c_incremental_feature_group_effects",
            ],
            "settings": {
                "dpi": settings.dpi,
                "formats": list(settings.formats),
                "fig3_font_size": settings.fig3_font_size,
            },
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[OK] Fig. 2 completed. Output: {output_dir.resolve()}")
        return

    # Fig. 1 is independent of the ablation and parameter spreadsheets. Load
    # only the trajectory-level files first so --only-fig1 does not require
    # any Fig. 2-4 inputs to exist.
    summary, tracks = load_trajectory_inputs(args.ignition_summary, args.tracks_csv)
    gt_ignition_track_ids = load_gt_ignition_track_ids(args.gt_track_list)
    pd.DataFrame(
        {"track_id": sorted(gt_ignition_track_ids), "gt_has_ignition": True}
    ).to_csv(
        data_dir / "gt_ignition_track_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidate_summary = export_tracking_mechanism_candidates(
        summary,
        tracks,
        gt_ignition_track_ids,
        args.frames_dir,
        output_dir,
        settings,
        export_mode=str(args.candidate_export_mode),
        contact_max_frames=int(args.candidate_contact_max_frames),
    )
    if not candidate_summary.empty:
        safe_candidates = candidate_summary.copy()
        for column in ("mechanisms", "rescue_frames", "internal_gap_details"):
            if column in safe_candidates.columns:
                safe_candidates[column] = safe_candidates[column].apply(
                    lambda value: json.dumps(value, ensure_ascii=False)
                )
        safe_candidates.to_csv(
            data_dir / "tracking_mechanism_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    selected = select_representative_tracks(
        summary,
        tracks,
        args.representative_track_ids,
        gt_ignition_track_ids,
    )
    selected_safe = selected.copy()
    for column in ("mechanisms", "rescue_frames", "internal_gap_details"):
        if column in selected_safe.columns:
            selected_safe[column] = selected_safe[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False)
            )
    selected_safe.to_csv(
        data_dir / "selected_representative_tracks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_representative_cases(
        summary,
        tracks,
        selected,
        args.frames_dir,
        figure_dir,
        settings,
    )
    if args.auxiliary_track_ids:
        plot_auxiliary_track_cases(
            summary,
            tracks,
            args.auxiliary_track_ids,
            args.frames_dir,
            figure_dir,
            settings,
        )

    generated_figures = ["fig01_representative_cases_overview", "fig01_representative_cases/*", "fig01_auxiliary_track_cases/*"]
    if not args.only_fig1:
        tracking, method, feature = load_ablation_tables(
            args.tracking_summary,
            args.method_summary,
            args.feature_summary,
        )
        parameters = load_parameter_table(args.parameter_summary)

        baseline, method_gain, feature_gain = plot_combined_ablation(
            tracking,
            method,
            feature,
            figure_dir,
            settings,
        )
        baseline.to_csv(data_dir / "method_baseline_performance.csv", index=False, encoding="utf-8-sig")
        method_gain.to_csv(data_dir / "method_module_effect.csv", index=False, encoding="utf-8-sig")
        feature_gain.to_csv(data_dir / "feature_incremental_effect.csv", index=False, encoding="utf-8-sig")

        plot_key_parameter_sensitivity(parameters, figure_dir, settings)

        events = build_ignition_events(summary, tracks)
        events.to_csv(data_dir / "matched_ignition_events.csv", index=False, encoding="utf-8-sig")
        plot_spatiotemporal_measurement(events, tracks, figure_dir, settings)
        generated_figures.extend(
            [
                "fig02_combined_ablation",
                "fig02_panels/*",
                "fig03_key_parameter_sensitivity",
                "fig03_panels/*",
                "fig04_spatiotemporal_measurement",
                "fig04_panels/*",
            ]
        )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "tracking_summary": None if args.only_fig1 else str(args.tracking_summary.resolve()),
            "method_summary": None if args.only_fig1 else str(args.method_summary.resolve()),
            "feature_summary": None if args.only_fig1 else str(args.feature_summary.resolve()),
            "parameter_summary": None if args.only_fig1 else str(args.parameter_summary.resolve()),
            "ignition_summary": str(args.ignition_summary.resolve()),
            "tracks_csv": str(args.tracks_csv.resolve()),
            "gt_track_list": str(args.gt_track_list.resolve()),
            "frames_dir": str(args.frames_dir.resolve()) if args.frames_dir else None,
        },
        "only_fig1": bool(args.only_fig1),
        "only_fig2": bool(args.only_fig2),
        "candidate_export_mode": str(args.candidate_export_mode),
        "n_gt_ignition_tracks": int(len(gt_ignition_track_ids)),
        "representative_track_ids": [int(value) for value in selected["track_id"].tolist()],
        "figures": generated_figures,
        "settings": {
            "dpi": settings.dpi,
            "formats": list(settings.formats),
            "fps": settings.fps,
            "temporal_bin_width": settings.temporal_bin_width,
            "sequence_offsets": list(settings.sequence_offsets),
            "fig2_font_size": settings.fig2_font_size,
            "fig3_font_size": settings.fig3_font_size,
            "fig4_font_size": settings.fig4_font_size,
            "fig5_font_size": settings.fig5_font_size,
            "pixel_size_um": settings.pixel_size_um,
            "application_spatial_unit": "mm",
            "input_coordinate_unit": "pixel",
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
