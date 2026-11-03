#!/usr/bin/env python3
"""OT/repair: fixed-GT window observability; GMM/RTV: ignition accuracy.

Tracking uses pre-onset coverage (-10..-1 frames), window completeness
(-5..+5), and three observations per side of GT onset on one matched track.
Detected and restored observations count alike; unmatched positive GT events
remain in the denominator. Tracking does not run ignition judgment.
Existing input paths and OT/repair/GMM/RTV output directories are preserved.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import pickle
import re
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

SCRIPT_TITLE = "Parameter Sensitivity Analysis"


# ---------------------------------------------------------------------------
# Immutable default snapshots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackerSensitivityConfig:
    enable_rescue: bool = True
    enable_gap_fill: bool = True
    enable_prebirth: bool = True
    max_center_dist: float = 45.0
    ot_reg: float = 0.08
    dummy_cost: float = 0.65
    rescue_center_factor: float = 1.3
    rescue_maha_factor: float = 1.3
    max_gap_fill: int = 10
    prebirth_max_backward_frames: int = 8
    prebirth_search_radius_base: float = 12.0


@dataclass(frozen=True)
class ParameterSpec:
    group: str
    name: str
    fields: tuple[str, ...]
    default_value: Any
    values: tuple[Any, ...]
    skip_reason: str = ""
    coupled_note: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    group: str
    parameter_name: str
    parameter_value: Any
    default_value: Any
    ratio_to_default: Optional[float]
    is_default_configuration: bool
    overrides: Mapping[str, Any]
    config_id: str
    skip_reason: str = ""


@dataclass
class FeatureCacheEntry:
    """One fixed-ROI track's formally extracted feature table."""

    track_name: str
    track_id: Optional[int]
    roi_path: str
    image_paths: tuple[str, ...]
    feature_df: pd.DataFrame
    source_signature: str
    status: str = "ok"
    error: str = ""


@dataclass
class FixedTrackingContext:
    """Immutable-by-contract reference inputs shared by every tracking run."""

    reference_tracks: pd.DataFrame
    reference_ignition: pd.DataFrame
    listed_no_ignition_ids: tuple[int, ...]
    onset_tolerance: int
    annotations: Mapping[str, Mapping[str, Any]]
    content_hash: str


@dataclass(frozen=True)
class ManualGtSource:
    """Source-space manual GT before transfer into T3 reference identities."""

    detection_annotations: Mapping[str, Mapping[str, Any]]
    source_universe: tuple[str, ...]
    explicit_names: tuple[str, ...]
    crop_sequences: Mapping[str, tuple[int, ...]]
    source_tracks: pd.DataFrame
    metadata: Mapping[str, Any]
    content_hash: str


@dataclass(frozen=True)
class ManualGtTransfer:
    """Formal source→T3 identity transfer for tracking annotations."""

    source_to_reference: Mapping[str, str]
    reference_to_source: Mapping[str, str]
    tracking_annotations: Mapping[str, Mapping[str, Any]]
    diagnostics: pd.DataFrame
    metadata: Mapping[str, Any]
    content_hash: str


@dataclass(frozen=True)
class GtExecutionPlan:
    """Unified legacy/manual GT plan consumed by the sensitivity sweep."""

    format: str
    detection_annotations: Mapping[str, Mapping[str, Any]]
    tracking_annotations: Mapping[str, Mapping[str, Any]]
    source_to_reference: Mapping[str, str]
    reference_to_source: Mapping[str, str]
    source_universe: tuple[str, ...]
    metadata: Mapping[str, Any]
    content_hash: str


DEFAULT_ANNOTATION_PATH = "effective_trace.xlsx"
DEFAULT_TRACKING_MODULE_PATH = "ot_kalman_tracker.py"
DEFAULT_IGNITION_MODULE_PATH = "ignition_detection_gmm_rtv_new.py"
DEFAULT_TRACKING_HELPER_PATH = (
    "run_tracking_ablation_experiments_gmm_rtv_"
    "effective_trace_positive_gt_v2.py"
)
DEFAULT_MANUAL_GT_SOURCE_TRACKS_CSV = (
    "locate_results/run_20260703_104159/postprocessed/"
    "tracking_output/run_20260706_105300/tracks.csv"
)
_MANUAL_GT_REPRODUCIBILITY_NOT_COMPARABLE: frozenset[str] = frozenset(
    {
        "tp",
        "fp",
        "fn",
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "onset_mae",
    }
)
_RESUME_EXECUTION_CONTEXT_KEYS: tuple[str, ...] = (
    "execution_context_hash",
    "gt_format",
    "gt_content_hash",
    "manual_gt_transfer_hash",
    "source_test_track_set_hash",
    "tracking_target_track_set_hash",
    "dataset_split_hash",
)


FORMAL_MANUAL_GT_MATCHER_GATES: Mapping[str, Any] = MappingProxyType(
    {
        "max_center_dist": 30.0,
        "min_shared_frames": 3,
        "min_ref_coverage": 0.20,
        "min_query_coverage": 0.20,
        "min_temporal_iou": 0.15,
        "max_assignment_cost": 1.50,
        "detected_only": True,
    }
)
MANUAL_GT_TRANSFER_CACHE_PROTOCOL = "manual_gt_transfer_v2"
_MANUAL_GT_TRANSFER_DIAG_COLUMNS: tuple[str, ...] = (
    "source_track_name",
    "reference_track_name",
    "source_track_id",
    "reference_track_id",
    "annotation_origin",
    "gt_category",
    "local_onset_idx",
    "global_onset_frame",
    "match_status",
    "exclusion_reason",
    "onset_outside_reference_track",
    "shared_frames",
    "reference_coverage",
    "query_coverage",
    "temporal_iou",
    "assignment_cost",
    "mean_center_distance",
)
_MANUAL_GT_TRANSFER_STRING_COLUMNS: tuple[str, ...] = (
    "source_track_name",
    "reference_track_name",
    "annotation_origin",
    "gt_category",
    "match_status",
    "exclusion_reason",
)
_MANUAL_GT_TRANSFER_INTEGER_COLUMNS: tuple[str, ...] = (
    "source_track_id",
    "reference_track_id",
    "local_onset_idx",
    "global_onset_frame",
    "shared_frames",
)
_MANUAL_GT_TRANSFER_NULLABLE_INTEGER_COLUMNS: frozenset[str] = frozenset(
    {"reference_track_id", "shared_frames"}
)
_MANUAL_GT_TRANSFER_BOOLEAN_COLUMNS: tuple[str, ...] = (
    "onset_outside_reference_track",
)
_MANUAL_GT_TRANSFER_FLOAT_COLUMNS: tuple[str, ...] = (
    "reference_coverage",
    "query_coverage",
    "temporal_iou",
    "assignment_cost",
    "mean_center_distance",
)
_MANUAL_GT_TRANSFER_METADATA_COMPARISON_WHITELIST: frozenset[str] = frozenset(
    {
        "display_path",
        "output_path",
        "temporary_path",
        "created_at",
    }
)
_MANUAL_GT_TRANSFER_HASH_EXCLUDED_METADATA_FIELDS: frozenset[str] = frozenset(
    {"content_hash"} | set(_MANUAL_GT_TRANSFER_METADATA_COMPARISON_WHITELIST)
)


@dataclass
class RunResult:
    """Per-track and aggregate output for one sensitivity configuration."""

    experiment: ExperimentSpec
    rows: pd.DataFrame
    summary: dict[str, Any]
    evaluated_track_names: tuple[str, ...]
    test_track_set_hash: str
    config: Any
    reference_ignition: Optional[pd.DataFrame] = None
    alignment_rows: Optional[pd.DataFrame] = None


@dataclass
class TrackingOutput:
    """Formal tracker objects plus statistics emitted by its public APIs."""

    tracks: list[Any]
    stats: dict[str, Any]


RTV_THRESHOLD_FIELDS: tuple[str, ...] = (
    "min_raw_delta_gray_p99",
    "min_raw_delta_corr_p99",
    "min_raw_delta_bright_energy",
    "min_raw_delta_bright_area",
    "min_raw_delta_hsv_v_p99",
)

GROUP_ALIASES: dict[str, str] = {
    "ot": "ot",
    "ot_association": "ot",
    "repair": "repair",
    "trajectory_repair": "repair",
    "gmm": "gmm",
    "gmm_temporal": "gmm",
    "rtv": "rtv",
    "all": "all",
}

VALID_GROUPS: frozenset[str] = frozenset({"ot", "repair", "gmm", "rtv"})
TRACKING_GROUPS: frozenset[str] = frozenset({"ot", "repair"})
TRACKING_ABLATION_REQUIRED_APIS: tuple[str, ...] = (
    "build_track_centroids_by_frame",
    "normalize_track_name",
    "match_tracks_to_reference",
)

# GT-centered availability, not downstream ignition classification accuracy.
WINDOW_PRE_FRAMES = 10
WINDOW_RADIUS = 5
WINDOW_MIN_SIDE_OBSERVATIONS = 3
WINDOW_METRIC_PROTOCOL = "gt_window_v1_pre10_pm5_fulltrack_side3"
WINDOW_METRICS = {
    "mean_pre_onset_coverage": "pre_onset_coverage",
    "mean_ignition_window_completeness": "ignition_window_completeness",
    "observable_transition_rate": "observable_transition",
}


def evaluate_tracking_windows(
    tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    annotations: Mapping[str, Mapping[str, Any]],
    query_to_ref: Mapping[Any, Any],
    frame_map: Mapping[int, Path],
    formal_module: Any,
    identity_aliases: Optional[Mapping[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use fixed positive GT events, including unmatched events as zero support.

    All valid detected/restored samples count equally. Windows are clipped to
    sequence boundaries only; missing internal frames count as missing support.
    Each event uses ONE matched trajectory, never a union of separate identities.
    This measures geometric observation availability, not thermal correctness.
    """
    domain = sorted(int(f) for f in frame_map)
    if not domain:
        raise ValueError("Window evaluation requires indexed source-image frames")
    first, last = domain[0], domain[-1]
    normalize = lambda value: _track_name_from_raw_id(
        value, tracking_ablation=formal_module
    )
    sample_sets = {
        normalize(track_id): set(group["frame_id"].astype(int))
        for track_id, group in tracks.groupby("track_id", sort=False)
    }
    reference_sets = {
        normalize(track_id): set(group["frame_id"].astype(int))
        for track_id, group in reference_tracks.groupby("track_id", sort=False)
    }
    candidates: dict[str, list[str]] = {}
    for query, reference in query_to_ref.items():
        candidates.setdefault(normalize(reference), []).append(normalize(query))
    rows = []
    for name, annotation in annotations.items():
        ref_name = normalize(name)
        positive = _annotation_is_positive(annotation)
        row = {
            "track_name": (identity_aliases or {}).get(name, name),
            "reference_track_name": ref_name,
            "gt_has_ignition": positive,
            "window_evaluated": positive,
            "gt_onset_frame": int(annotation.get("gt_onset_frame", -1)),
            **{column: float("nan") for column in WINDOW_METRICS.values()},
        }
        if not positive:
            rows.append(row)
            continue
        onset = row["gt_onset_frame"]
        if not first <= onset <= last:
            raise ValueError(f"GT onset outside source sequence: {name}: {onset}")
        choices = candidates.get(ref_name, [])
        chosen = min(
            choices,
            key=lambda query: (
                -len(sample_sets.get(query, set()) & reference_sets.get(ref_name, set())),
                query,
            ),
        ) if choices else None
        observed = sample_sets.get(chosen, set())
        pre = set(range(max(first, onset - WINDOW_PRE_FRAMES), onset))
        window = set(range(max(first, onset - WINDOW_RADIUS), min(last, onset + WINDOW_RADIUS) + 1))
        # The manuscript's observable-transition criterion uses at least three
        # samples on each side along the same trajectory (not only +/-5 frames).
        n_before = sum(frame < onset for frame in observed)
        n_after = sum(frame > onset for frame in observed)
        row.update({
            "matched_query_track_name": chosen or "",
            "matched_reference": chosen is not None,
            "pre_window_target_frames": len(pre),
            "pre_window_observed_frames": len(observed & pre),
            "ignition_window_target_frames": len(window),
            "ignition_window_observed_frames": len(observed & window),
            "observations_before_onset": n_before,
            "observations_after_onset": n_after,
            "pre_onset_coverage": len(observed & pre) / len(pre) if pre else float("nan"),
            "ignition_window_completeness": len(observed & window) / len(window),
            "observable_transition": float(
                n_before >= WINDOW_MIN_SIDE_OBSERVATIONS
                and n_after >= WINDOW_MIN_SIDE_OBSERVATIONS
            ),
        })
        rows.append(row)
    result = pd.DataFrame(rows)
    positives = result[result["window_evaluated"] == True] if len(result) else result
    if positives.empty:
        raise ValueError("Window evaluation requires at least one positive GT ignition event")
    summary = {
        "tracking_metric_protocol": WINDOW_METRIC_PROTOCOL,
        "n_gt_positive_events": len(positives),
        "n_matched_gt_positive_events": int(positives["matched_reference"].sum()),
        "n_unmatched_gt_positive_events": int((~positives["matched_reference"].astype(bool)).sum()),
        "n_pre_window_evaluable_events": int(positives["pre_onset_coverage"].notna().sum()),
        "n_ignition_window_evaluable_events": len(positives),
        "n_observable_transitions": int(positives["observable_transition"].sum()),
        **{metric: float(positives[column].mean()) for metric, column in WINDOW_METRICS.items()},
    }
    return result, summary


def _run_tracking_experiment_impl(**kwargs: Any) -> RunResult:
    """Tracking-only window protocol: deliberately does not run GMM or RTV."""
    context = kwargs["fixed_context"]
    _validate_fixed_tracking_context(context)
    experiment = kwargs["experiment"]
    tracker = kwargs["tracker_mod"]
    helper = kwargs["tracking_ablation"]
    run_dir = Path(kwargs["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    state = kwargs["diagnostic_state"]
    if kwargs.get("t3_reproducibility_dir") and experiment.is_default_configuration:
        warnings.warn(
            "--tracking-ablation-t3-dir is retained for compatibility, but legacy "
            "T3 ignition-metric comparisons are not applicable to window-only evaluation."
        )
    state.update(stage="formal_tracker", run_dir=run_dir, experiment=experiment,
                 reference_tracks=context.reference_tracks, annotations=context.annotations)
    config = apply_tracker_experiment(kwargs["tracker_default"], experiment)
    output = run_formal_tracker(tracker, kwargs["detections"], config, kwargs["frame_map"])
    tracker.save_tracks_csv(output.tracks, run_dir / "tracks_long.csv", min_track_len=2)
    tracker.save_track_summary_csv(output.tracks, run_dir / "track_summary.csv", min_track_len=2)
    tracks = pd.read_csv(run_dir / "tracks_long.csv", encoding="utf-8-sig")
    valid = pd.Series(True, index=tracks.index)
    numeric_frames = pd.to_numeric(tracks["frame_id"], errors="coerce")
    valid &= numeric_frames.notna() & numeric_frames.mod(1).eq(0)
    valid &= numeric_frames.isin(kwargs["frame_map"])
    coordinate_columns = _coordinate_columns(tracks, "query_tracks")
    for column in coordinate_columns:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
        valid &= np.isfinite(tracks[column])
    if all(column in tracks for column in _BOX_COLUMNS):
        valid &= tracks["x2"].gt(tracks["x1"]) & tracks["y2"].gt(tracks["y1"])
    if tracks["track_id"].isna().any():
        raise ValueError("Tracking output contains missing track_id")
    valid_tracks = tracks.loc[valid].copy()
    valid_tracks["frame_id"] = numeric_frames.loc[valid].astype(int)
    valid_tracks = valid_tracks.drop_duplicates(["track_id", "frame_id"])
    state.update(stage="formal_matcher", query_tracks=valid_tracks)
    if valid_tracks.empty:
        mapping, alignment = {}, pd.DataFrame()
    else:
        mapping, alignment = helper.match_tracks_to_reference(
            valid_tracks, context.reference_tracks,
            **dict(FORMAL_MANUAL_GT_MATCHER_GATES), return_diagnostics=True,
        )
    state.update(query_to_ref=mapping, stage="window_evaluation")
    aliases = kwargs.get("evaluation_identity_aliases")
    if aliases is not None:
        aliases = build_evaluation_identity_aliases(
            tracking_annotations=context.annotations, reference_to_source=aliases
        )
    rows, summary = evaluate_tracking_windows(
        valid_tracks, context.reference_tracks, context.annotations, mapping,
        kwargs["frame_map"], helper, aliases,
    )
    if summary["n_matched_gt_positive_events"] == 0:
        warnings.warn(
            f"{experiment.config_id}: no positive GT event matched a query track. "
            "Window metrics retain zero support; check alignment_rows.csv and "
            "the reference/source frame and identity conventions."
        )
    summary.update(tracks_long_row_count=len(tracks), valid_observation_row_count=len(valid_tracks),
                   invalid_observation_row_count=int((~valid).sum()))
    pd.DataFrame(alignment).to_csv(run_dir / "alignment_rows.csv", index=False, encoding="utf-8-sig")
    valid_tracks.to_csv(run_dir / "valid_window_observations.csv", index=False, encoding="utf-8-sig")
    _atomic_write_json(run_dir / "window_evaluation_protocol.json", {
        "protocol": WINDOW_METRIC_PROTOCOL, "pre_onset_frames": WINDOW_PRE_FRAMES,
        "ignition_window_radius": WINDOW_RADIUS, "minimum_observations_per_side": WINDOW_MIN_SIDE_OBSERVATIONS,
        "onset_excluded_from_side_counts": True, "sequence_boundary_clipping": True,
        "transition_support_scope": "entire_matched_trajectory_before_and_after_gt_onset",
        "unmatched_gt_positive_policy": "zero_observations_fixed_denominator",
        "observation_policy": "valid_geometry_at_indexed_image_frame_including_restored_samples",
        "negative_gt_policy": "retained_in_identity_set_excluded_from_window_means",
        "stable_max_absolute_drop": STABLE_MAX_COVERAGE_DROP,
        "note": "Geometric availability only; no ignition-classification or thermal-correctness inference.",
    })
    evaluated_names = tuple((aliases or {}).get(name, name) for name in context.annotations)
    _validate_fixed_tracking_context(context)
    return RunResult(experiment=experiment, rows=rows, summary=summary,
                     evaluated_track_names=evaluated_names,
                     test_track_set_hash=_test_track_set_hash(evaluated_names), config=config,
                     reference_ignition=context.reference_ignition, alignment_rows=pd.DataFrame(alignment))

STABLE_MAX_F1_DROP_PCT = 5.0
STABLE_MAX_ONSET_MAE_INCREASE = 0.5
STABLE_MAX_FALSE_IGNITION_INCREASE = 0.03
STABLE_MAX_COVERAGE_DROP = 0.03

BOOTSTRAP_METRIC_KEYS: tuple[str, ...] = (
    "ignition_precision",
    "ignition_recall",
    "ignition_f1",
    "false_ignition_rate",
    "onset_mae",
)

ERROR_PREDICTION_STATUSES: frozenset[str] = frozenset(
    {"track_error", "method_error"}
)
UNMATCHED_PREDICTION_POOL_KEY = "__unmatched_prediction_pool__"
RTV_CANDIDATE_SEMANTICS = (
    "formal_first_cold_hot_candidate_is_terminal_after_rtv_failure"
)

_TRACK_SUMMARY_ERROR = (
    "Reference tracks must be a long-format per-frame table; received "
    "a track_summary table with one row per track."
)
_CENTER_COLUMNS = ("cx", "cy")
_BOX_COLUMNS = ("x1", "y1", "x2", "y2")


class TrackingAlignmentError(RuntimeError):
    """The canonical default tracking run failed an alignment invariant."""


def _require_dataframe(frame: pd.DataFrame, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")


def _integer_series(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"{name} must include {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(values.to_numpy(dtype=float))
    integral = finite & (values.to_numpy(dtype=float) == np.floor(values))
    if not bool(np.all(integral)):
        raise ValueError(f"{name} must contain integer-convertible {column} values")
    return values.astype(np.int64)


def _coordinate_columns(frame: pd.DataFrame, name: str) -> tuple[str, ...]:
    has_center = all(column in frame.columns for column in _CENTER_COLUMNS)
    has_box = all(column in frame.columns for column in _BOX_COLUMNS)
    if has_center or has_box:
        return (
            (*_CENTER_COLUMNS, *_BOX_COLUMNS)
            if has_center and has_box
            else (_CENTER_COLUMNS if has_center else _BOX_COLUMNS)
        )
    raise ValueError(
        f"{name} must include track_id, frame_id, and either cx/cy "
        "or x1/y1/x2/y2"
    )


def _finite_coordinate_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> pd.DataFrame:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if values.size == 0 or not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain non-empty finite coordinates")
    return numeric


def validate_reference_tracks_long_format(
    df: pd.DataFrame,
    path: str | Path,
) -> dict[str, Any]:
    """Validate per-frame reference tracks and return a JSON-ready summary."""
    name = "reference tracks"
    _require_dataframe(df, name)
    if df.empty:
        raise ValueError("reference tracks must not be empty")
    missing = [column for column in ("track_id", "frame_id") if column not in df]
    if missing:
        raise ValueError(
            "reference tracks must include track_id, frame_id, and either cx/cy "
            "or x1/y1/x2/y2"
        )
    if df["track_id"].isna().any() or df["track_id"].map(
        lambda value: not str(value).strip()
    ).any():
        raise ValueError("reference tracks must contain non-empty track_id values")

    frames = _integer_series(df, "frame_id", name)
    coordinate_columns = _coordinate_columns(df, name)
    coordinates = _finite_coordinate_values(df, coordinate_columns, name)

    canonical_ids = df["track_id"].map(_normalize_identity)
    sequences = (
        df["sequence_id"].map(lambda value: str(value).strip())
        if "sequence_id" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    canonical_variants: dict[tuple[str, str], set[str]] = {}
    for sequence, canonical_id, raw_id in zip(
        sequences, canonical_ids, df["track_id"]
    ):
        canonical_variants.setdefault((sequence, canonical_id), set()).add(
            f"{type(raw_id).__name__}:{str(raw_id).strip()}"
        )
    collisions = [
        {
            "sequence_id": sequence or None,
            "canonical_id": canonical_id,
            "raw_ids": sorted(raw_ids),
        }
        for (sequence, canonical_id), raw_ids in sorted(canonical_variants.items())
        if len(raw_ids) > 1
    ]
    if collisions:
        raise ValueError(
            "reference tracks canonical track identity collision: "
            + json.dumps(collisions, sort_keys=True)
        )

    canonical_frame_keys = pd.DataFrame(
        {
            "sequence_id": sequences,
            "track_id": canonical_ids,
            "frame_id": frames,
        }
    )
    key_columns = (
        ["sequence_id", "track_id", "frame_id"]
        if "sequence_id" in df.columns
        else ["track_id", "frame_id"]
    )
    if canonical_frame_keys.duplicated(key_columns, keep=False).any():
        label = (
            "(sequence_id, track_id, frame_id)"
            if "sequence_id" in df.columns
            else "(track_id, frame_id)"
        )
        raise ValueError(f"{label} must be unique in reference tracks")

    track_keys = pd.DataFrame(
        {"sequence_id": sequences, "track_id": canonical_ids}
    )
    track_columns = (
        ["sequence_id", "track_id"]
        if "sequence_id" in df.columns
        else ["track_id"]
    )
    rows_per_track = track_keys.groupby(track_columns, dropna=False).size()
    median_rows = float(rows_per_track.median())
    single_row_count = int((rows_per_track == 1).sum())
    single_row_ratio = float(single_row_count / len(rows_per_track))
    if (
        rows_per_track.empty
        or median_rows <= 1.0
        or single_row_ratio > 0.5
    ):
        raise ValueError(_TRACK_SUMMARY_ERROR)

    coordinate_ranges = {
        column: [
            float(coordinates[column].min()),
            float(coordinates[column].max()),
        ]
        for column in coordinate_columns
    }
    return {
        "path": str(path),
        "rows": int(len(df)),
        "tracks": int(len(rows_per_track)),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "coordinate_columns": list(coordinate_columns),
        "coordinate_ranges": coordinate_ranges,
        "median_rows_per_track": median_rows,
        "single_row_track_count": single_row_count,
        "single_row_track_ratio": single_row_ratio,
        "canonical_identity_collision_count": 0,
        "canonical_identity_collisions": [],
    }


def _resolve_normalize_manual_track_identity(ignition_ablation: Any) -> Any:
    normalize = getattr(ignition_ablation, "normalize_manual_track_identity", None)
    if callable(normalize):
        return normalize
    framework = getattr(ignition_ablation, "framework", None)
    normalize = getattr(framework, "normalize_manual_track_identity", None)
    if callable(normalize):
        return normalize
    raise AttributeError(
        "ignition_ablation must expose normalize_manual_track_identity"
    )


def validate_source_tracks(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    ignition_ablation: Any,
) -> dict[str, Any]:
    """Validate source long-format tracks and return a strict identity universe."""
    geometry = validate_reference_tracks_long_format(frame, path)
    normalize = _resolve_normalize_manual_track_identity(ignition_ablation)

    canonical_ids: list[str] = []
    for row_number, raw_id in enumerate(frame["track_id"].tolist(), start=2):
        canonical_ids.append(
            normalize(
                raw_id,
                source_name=f"source tracks {path} row {row_number}",
            )
        )
    frames = _integer_series(frame, "frame_id", "source tracks")
    key_frame = pd.DataFrame(
        {"track_id": canonical_ids, "frame_id": frames.to_numpy(dtype=np.int64)}
    )
    if key_frame.duplicated(["track_id", "frame_id"], keep=False).any():
        raise ValueError(
            f"(track_id, frame_id) must be unique in source tracks: {path}"
        )

    ordered_universe: list[str] = []
    seen: set[str] = set()
    for name in canonical_ids:
        if name not in seen:
            seen.add(name)
            ordered_universe.append(name)
    summary = dict(geometry)
    summary["path"] = str(path)
    summary["source_universe"] = tuple(ordered_universe)
    summary["tracks"] = int(len(ordered_universe))
    return summary


def validate_source_crop_index(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    source_universe: Sequence[str],
    ignition_ablation: Any,
) -> dict[str, Any]:
    """Validate crop-index rows against a source identity universe."""
    _require_dataframe(frame, "source crop index")
    required = ("track_id", "frame_id", "status", "crop_path")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "source crop index must include track_id/frame_id/status/crop_path; "
            f"missing {missing}: {path}"
        )
    normalize = _resolve_normalize_manual_track_identity(ignition_ablation)

    frames = _integer_series(frame, "frame_id", "source crop index")
    canonical_ids: list[str] = []
    for row_number, raw_id in enumerate(frame["track_id"].tolist(), start=2):
        canonical_ids.append(
            normalize(
                raw_id,
                source_name=f"source crop index {path} row {row_number}",
            )
        )
    key_frame = pd.DataFrame(
        {
            "track_id": canonical_ids,
            "frame_id": frames.to_numpy(dtype=np.int64),
            "status": frame["status"].map(lambda value: str(value).strip().lower()),
        }
    )
    if key_frame.duplicated(["track_id", "frame_id"], keep=False).any():
        raise ValueError(
            f"(track_id, frame_id) must be unique in source crop index: {path}"
        )
    successful = key_frame.loc[key_frame["status"] == "ok"]
    successful_paths = frame.loc[
        key_frame["status"] == "ok",
        "crop_path",
    ]
    missing_successful_path = successful_paths.isna() | successful_paths.map(
        lambda value: not str(value).strip()
    )
    if missing_successful_path.any():
        bad_rows = [
            int(position) + 2
            for position in np.flatnonzero(missing_successful_path.to_numpy())
        ]
        raise ValueError(
            "source crop index status=ok rows require non-empty crop_path; "
            f"rows={bad_rows}: {path}"
        )

    universe = set(source_universe)
    orphans = sorted(set(canonical_ids) - universe)
    if orphans:
        raise ValueError(
            "crop-index track absent from source tracks: "
            + ", ".join(orphans)
        )
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "tracks": int(len(set(canonical_ids))),
        "successful_crop_rows": int(len(successful)),
    }


def build_source_crop_sequences(
    frame: pd.DataFrame,
    framework: Any,
) -> dict[str, tuple[int, ...]]:
    """Build per-track successful crop frame sequences in actual row order."""
    _require_dataframe(frame, "source crop index")
    for column in ("track_id", "frame_id", "status"):
        if column not in frame.columns:
            raise ValueError(f"source crop index must include {column}")
    normalize = _resolve_normalize_manual_track_identity(framework)

    sequences: dict[str, list[int]] = {}
    frames = _integer_series(frame, "frame_id", "source crop index")
    for row_number, (raw_id, frame_id, status) in enumerate(
        zip(frame["track_id"], frames, frame["status"]),
        start=2,
    ):
        track_name = normalize(
            raw_id,
            source_name=f"source crop sequence row {row_number}",
        )
        if str(status).strip().lower() != "ok":
            continue
        sequences.setdefault(track_name, []).append(int(frame_id))
    return {name: tuple(values) for name, values in sequences.items()}


def map_local_onset_to_global_frame(
    track_name: str,
    local_onset: int,
    crop_sequences: Mapping[str, tuple[int, ...]],
    *,
    source_name: str = "manual GT local onset",
) -> int:
    """Map a local successful-crop onset index to a global source frame."""
    onset = int(local_onset)
    if onset < 0:
        return -1
    sequence = crop_sequences.get(str(track_name))
    if sequence is None:
        raise ValueError(
            f"{source_name}: no successful crop sequence for {track_name!r}"
        )
    if onset >= len(sequence):
        raise ValueError(
            f"{source_name}: local onset {onset} out of range for "
            f"{track_name} (successful crop length {len(sequence)})"
        )
    return int(sequence[onset])


def _manual_gt_source_hash(
    detection_annotations: Mapping[str, Mapping[str, Any]],
    source_universe: Sequence[str],
    explicit_names: Sequence[str],
    crop_sequences: Mapping[str, tuple[int, ...]],
    source_tracks: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        pd.util.hash_pandas_object(source_tracks, index=True, categorize=True)
        .to_numpy(dtype=np.uint64)
        .tobytes()
    )
    digest.update(
        json.dumps(
            {
                "detection_annotations": detection_annotations,
                "source_universe": list(source_universe),
                "explicit_names": list(explicit_names),
                "crop_sequences": {
                    str(name): list(frames)
                    for name, frames in crop_sequences.items()
                },
                "metadata": metadata,
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def load_manual_gt_source(
    annotation_path: str | Path,
    source_tracks_path: str | Path,
    crop_index_path: Optional[str | Path],
    *,
    ignition_ablation: Any,
    ignition_mod: Any,
    onset_tolerance: int = 2,
    require_crop_index: bool = True,
) -> ManualGtSource:
    """Parse source-space manual GT, crop sequences, and implicit negatives."""
    del ignition_mod  # reserved for later formal identity adapters
    annotation_path = Path(annotation_path)
    source_tracks_path = Path(source_tracks_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Manual GT annotation not found: {annotation_path}")
    if not source_tracks_path.is_file():
        raise FileNotFoundError(f"Source tracks not found: {source_tracks_path}")

    source_tracks = pd.read_csv(source_tracks_path)
    track_summary = validate_source_tracks(
        source_tracks,
        source_tracks_path,
        ignition_ablation=ignition_ablation,
    )
    source_universe = tuple(track_summary["source_universe"])

    read_annotation_table = getattr(ignition_ablation, "read_annotation_table", None)
    if callable(read_annotation_table):
        annotation_frame = read_annotation_table(annotation_path)
        for format_name, detector_name in (
            ("effective_trace_new", "is_effective_trace_new_format"),
            ("effective_trace", "is_effective_trace_format"),
        ):
            detector = getattr(ignition_ablation, detector_name, None)
            if callable(detector) and detector(annotation_frame):
                raise ValueError(
                    "load_manual_gt_source requires manual_review_gt annotations, "
                    f"got {format_name!r} from {annotation_path}"
                )

    source_metadata: dict[str, Any] = {}
    explicit, fmt = ignition_ablation.load_annotations(
        annotation_path,
        reference_summary=None,
        onset_tolerance=onset_tolerance,
        metadata_out=source_metadata,
    )
    if fmt != "manual_review_gt":
        raise ValueError(
            "load_manual_gt_source requires manual_review_gt annotations, "
            f"got {fmt!r} from {annotation_path}"
        )

    strict_mapping = getattr(ignition_ablation, "_strict_manual_identity_mapping", None)
    if not callable(strict_mapping):
        framework = getattr(ignition_ablation, "framework", None)
        strict_mapping = getattr(framework, "_strict_manual_identity_mapping", None)
    if not callable(strict_mapping):
        raise AttributeError(
            "ignition_ablation must expose _strict_manual_identity_mapping"
        )
    no_ignition_gt = getattr(ignition_ablation, "_no_ignition_gt", None)
    if not callable(no_ignition_gt):
        framework = getattr(ignition_ablation, "framework", None)
        no_ignition_gt = getattr(framework, "_no_ignition_gt", None)
    if not callable(no_ignition_gt):
        raise AttributeError("ignition_ablation must expose _no_ignition_gt")
    explicit_canonical = strict_mapping(
        list(explicit.items()),
        source_name="manual GT annotations",
    )
    outside = sorted(set(explicit_canonical) - set(source_universe))
    if outside:
        raise ValueError(
            "Manual GT identities absent from source tracks universe: "
            f"{outside}"
        )

    detection_annotations: dict[str, dict[str, Any]] = {}
    for name in source_universe:
        if name in explicit_canonical:
            detection_annotations[name] = copy.deepcopy(explicit_canonical[name])
        else:
            detection_annotations[name] = copy.deepcopy(no_ignition_gt())

    crop_sequences: dict[str, tuple[int, ...]] = {}
    crop_summary: dict[str, Any] = {}
    if crop_index_path is None:
        if require_crop_index:
            raise ValueError("crop_index_path is required when require_crop_index=True")
    else:
        crop_index_path = Path(crop_index_path)
        if not crop_index_path.is_file():
            raise FileNotFoundError(f"Source crop index not found: {crop_index_path}")
        crop_frame = pd.read_csv(crop_index_path)
        crop_summary = validate_source_crop_index(
            crop_frame,
            crop_index_path,
            source_universe=source_universe,
            ignition_ablation=ignition_ablation,
        )
        crop_sequences = build_source_crop_sequences(crop_frame, ignition_ablation)
    if crop_index_path is not None:
        for name, annotation in detection_annotations.items():
            if annotation.get("gt_category") != "ignition":
                continue
            map_local_onset_to_global_frame(
                name,
                int(annotation["gt_onset_idx"]),
                crop_sequences,
                source_name=(
                    f"manual GT local onset for {name} in {annotation_path}"
                ),
            )

    explicit_names = tuple(
        name for name in source_universe if name in explicit_canonical
    )
    positive_count = sum(
        detection_annotations[name].get("gt_category") == "ignition"
        for name in explicit_names
    )
    explicit_negative_count = sum(
        detection_annotations[name].get("gt_category") == "no_ignition"
        for name in explicit_names
    )
    implicit_negative_count = len(source_universe) - len(explicit_names)
    metadata = dict(source_metadata)
    annotation_sha256 = _file_content_sha256(annotation_path)
    source_tracks_sha256 = _file_content_sha256(source_tracks_path)
    crop_index_sha256 = (
        None
        if crop_index_path is None
        else _file_content_sha256(Path(crop_index_path))
    )
    if require_crop_index and crop_index_sha256 is None:
        raise ValueError(
            "crop_index_path hash is required when require_crop_index=True"
        )
    metadata.update(
        {
            "format": "manual_review_gt",
            "onset_tolerance": int(onset_tolerance),
            "annotation_sha256": annotation_sha256,
            "source_tracks_sha256": source_tracks_sha256,
            "crop_index_sha256": crop_index_sha256,
            "source_tracks_path": str(source_tracks_path),
            "crop_index_path": (
                None if crop_index_path is None else str(Path(crop_index_path))
            ),
            "require_crop_index": bool(require_crop_index),
            "source_universe_count": int(len(source_universe)),
            "explicit_track_count": int(len(explicit_names)),
            "explicit_positive_count": int(positive_count),
            "explicit_negative_count": int(explicit_negative_count),
            "implicit_negative_count": int(implicit_negative_count),
            "total_negative_count": int(
                explicit_negative_count + implicit_negative_count
            ),
            "uses_reference_predictions": False,
            "source_track_summary": track_summary,
            "crop_index_summary": crop_summary,
        }
    )
    metadata = _to_jsonable(metadata)
    content_hash = _manual_gt_source_hash(
        detection_annotations,
        source_universe,
        explicit_names,
        crop_sequences,
        source_tracks,
        metadata,
    )
    return ManualGtSource(
        detection_annotations=detection_annotations,
        source_universe=source_universe,
        explicit_names=explicit_names,
        crop_sequences=crop_sequences,
        source_tracks=source_tracks,
        metadata=metadata,
        content_hash=content_hash,
    )


def _file_content_sha256(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_semantic_hash(frame: pd.DataFrame) -> str:
    """Hash ordered columns/rows while normalizing ordinary pandas dtypes."""
    _require_dataframe(frame, "semantic hash frame")
    if frame.columns.duplicated().any():
        raise ValueError("semantic hash frame must have unique columns")

    column_specs: list[dict[str, str]] = []
    normalized_columns: list[list[Any]] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[raw_column]
        nonnull = series.loc[~series.isna()]
        values = nonnull.tolist()
        if values and all(
            isinstance(value, (bool, np.bool_)) for value in values
        ):
            kind = "boolean"
        elif pd.api.types.is_numeric_dtype(series.dtype) or (
            values
            and all(
                isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                and not isinstance(value, (bool, np.bool_))
                for value in values
            )
        ):
            kind = "number"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            kind = "datetime"
        else:
            kind = "string"
        column_specs.append({"name": column, "kind": kind})

        normalized: list[Any] = []
        for value in series.tolist():
            if _is_blank_transfer_value(value) and not isinstance(value, str):
                normalized.append(None)
                continue
            if kind == "boolean":
                normalized.append(bool(value))
            elif kind == "number":
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"semantic hash column {column!r} contains non-finite values"
                    )
                normalized.append(
                    int(numeric) if numeric == math.floor(numeric) else numeric
                )
            elif kind == "datetime":
                normalized.append(pd.Timestamp(value).isoformat())
            else:
                normalized.append(str(value))
        normalized_columns.append(normalized)

    rows = [
        [normalized_columns[column_index][row_index]
         for column_index in range(len(normalized_columns))]
        for row_index in range(len(frame))
    ]
    payload = {
        "columns": column_specs,
        "rows": rows,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def _track_name_from_raw_id(raw_id: Any, *, tracking_ablation: Any) -> str:
    normalize = getattr(tracking_ablation, "normalize_track_name", None)
    if callable(normalize):
        name = str(normalize(raw_id)).strip()
        if name:
            return name
    text = str(raw_id).strip()
    if text.lower().startswith("track_"):
        return text
    numeric = pd.to_numeric(raw_id, errors="coerce")
    if pd.notna(numeric) and float(numeric) == float(np.floor(float(numeric))):
        return f"track_{int(numeric)}"
    raise ValueError(f"Cannot canonicalize track id: {raw_id!r}")


def _numeric_track_id_from_name(track_name: str) -> int:
    text = str(track_name).strip()
    lowered = text.lower()
    if lowered.startswith("track_"):
        text = text[6:]
    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric) or float(numeric) != float(np.floor(float(numeric))):
        raise ValueError(f"Cannot parse numeric track id from {track_name!r}")
    return int(numeric)


def _reference_track_frame_sets(
    reference_tracks: pd.DataFrame,
    *,
    tracking_ablation: Any,
) -> dict[str, set[int]]:
    frames = _integer_series(reference_tracks, "frame_id", "reference tracks")
    frame_sets: dict[str, set[int]] = {}
    for raw_id, frame_id in zip(reference_tracks["track_id"], frames):
        name = _track_name_from_raw_id(raw_id, tracking_ablation=tracking_ablation)
        frame_sets.setdefault(name, set()).add(int(frame_id))
    return frame_sets


def _alignment_diagnostics_lookup(
    alignment: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    if not isinstance(alignment, pd.DataFrame) or alignment.empty:
        return {}
    if "query_track_id" not in alignment.columns:
        raise ValueError(
            "formal matcher diagnostics require query_track_id column"
        )
    if "ref_track_id" in alignment.columns:
        ref_column = "ref_track_id"
    elif "reference_track_id" in alignment.columns:
        ref_column = "reference_track_id"
    else:
        raise ValueError(
            "formal matcher diagnostics require ref_track_id or "
            "reference_track_id"
        )
    shared_column = (
        "n_shared_frames"
        if "n_shared_frames" in alignment.columns
        else ("shared_frames" if "shared_frames" in alignment.columns else None)
    )
    lookup: dict[int, dict[str, Any]] = {}
    for _, row in alignment.iterrows():
        query_id = int(row["query_track_id"])
        record = {
            "reference_track_id": int(row[ref_column]),
            "shared_frames": (
                int(row[shared_column]) if shared_column is not None else None
            ),
            "reference_coverage": (
                float(row["reference_coverage"])
                if "reference_coverage" in alignment.columns
                and pd.notna(row["reference_coverage"])
                else None
            ),
            "query_coverage": (
                float(row["query_coverage"])
                if "query_coverage" in alignment.columns
                and pd.notna(row["query_coverage"])
                else None
            ),
            "temporal_iou": (
                float(row["temporal_iou"])
                if "temporal_iou" in alignment.columns
                and pd.notna(row["temporal_iou"])
                else None
            ),
            "assignment_cost": (
                float(row["assignment_cost"])
                if "assignment_cost" in alignment.columns
                and pd.notna(row["assignment_cost"])
                else None
            ),
            "mean_center_distance": (
                float(row["mean_center_distance"])
                if "mean_center_distance" in alignment.columns
                and pd.notna(row["mean_center_distance"])
                else None
            ),
        }
        lookup[query_id] = record
    return lookup


def _tracking_no_ignition_annotation() -> dict[str, Any]:
    return {
        "gt_category": "no_ignition",
        "gt_has_ignition": False,
        "gt_onset_frame": -1,
        "gt_onset_start": -1,
        "gt_onset_end": -1,
    }


def _tracking_ignition_annotation(
    global_onset: int,
    *,
    onset_tolerance: int,
) -> dict[str, Any]:
    onset = int(global_onset)
    if onset < 0:
        return {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_frame": -1,
            "gt_onset_start": -1,
            "gt_onset_end": -1,
        }
    return {
        "gt_category": "ignition",
        "gt_has_ignition": True,
        "gt_onset_frame": onset,
        "gt_onset_start": max(0, onset - int(onset_tolerance)),
        "gt_onset_end": onset + int(onset_tolerance),
    }


def build_reference_ignition_from_tracking_annotations(
    tracking_annotations: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Materialize one reference-level ignition row for every annotated track."""
    columns = (
        "track_id",
        "gt_category",
        "gt_has_ignition",
        "gt_onset_frame",
        "gt_onset_start",
        "gt_onset_end",
    )
    rows: list[dict[str, Any]] = []
    for track_name, annotation in tracking_annotations.items():
        rows.append(
            {
                "track_id": _numeric_track_id_from_name(str(track_name)),
                "gt_category": str(
                    annotation.get("gt_category", "no_ignition")
                ),
                "gt_has_ignition": bool(
                    annotation.get("gt_has_ignition", False)
                ),
                "gt_onset_frame": int(
                    annotation.get("gt_onset_frame", -1)
                ),
                "gt_onset_start": int(
                    annotation.get("gt_onset_start", -1)
                ),
                "gt_onset_end": int(
                    annotation.get("gt_onset_end", -1)
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        raise ValueError(
            "Tracking GT did not produce any reference-level annotations"
        )
    normalized = result["track_id"].map(_normalize_identity)
    if normalized.duplicated(keep=False).any():
        duplicates = sorted(set(normalized[normalized.duplicated(keep=False)]))
        raise ValueError(
            "Tracking GT produced duplicate reference identities: "
            + ", ".join(duplicates[:20])
        )
    return result


def _validate_tracking_annotation_schema(
    annotations: Mapping[str, Mapping[str, Any]],
) -> None:
    required = (
        "gt_category",
        "gt_has_ignition",
        "gt_onset_frame",
        "gt_onset_start",
        "gt_onset_end",
    )
    for name, annotation in annotations.items():
        missing = [field for field in required if field not in annotation]
        if missing:
            raise ValueError(
                f"tracking annotation {name!r} missing fields: {missing}"
            )
        category = str(annotation["gt_category"])
        if category not in {"ignition", "no_ignition"}:
            raise ValueError(
                f"tracking annotation {name!r} has unsupported category "
                f"{category!r}"
            )


def _validate_transfer_mapping_uniqueness(
    source_to_reference: Mapping[str, str],
    reference_to_source: Mapping[str, str],
) -> None:
    if len(source_to_reference) != len(set(source_to_reference.values())):
        raise ValueError(
            "manual GT transfer target collision: duplicate reference identities"
        )
    if len(reference_to_source) != len(set(reference_to_source.values())):
        raise ValueError(
            "manual GT transfer reverse alias collision: duplicate source identities"
        )
    for source_name, reference_name in source_to_reference.items():
        reverse = reference_to_source.get(reference_name)
        if reverse != source_name:
            raise ValueError(
                "manual GT transfer aliases are not one-to-one: "
                f"{source_name!r} -> {reference_name!r} vs reverse {reverse!r}"
            )


def _canonical_transfer_track_name(
    value: Any,
    *,
    allow_empty: bool = False,
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if allow_empty:
            return ""
        raise ValueError("manual GT transfer track identity must not be empty")
    text = str(value).strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError("manual GT transfer track identity must not be empty")
    return f"track_{_numeric_track_id_from_name(text)}"


def _is_blank_transfer_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _strict_transfer_bool(value: Any, column: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(
        f"{column} must contain only true/false/0/1, got {value!r}"
    )


def _coerce_manual_gt_transfer_diagnostics(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    _require_dataframe(diagnostics, "manual GT transfer diagnostics")
    if list(diagnostics.columns) != list(_MANUAL_GT_TRANSFER_DIAG_COLUMNS):
        raise ValueError("manual GT transfer diagnostics schema mismatch")
    frame = diagnostics.copy(deep=True)

    for column in _MANUAL_GT_TRANSFER_STRING_COLUMNS:
        values = frame[column].map(
            lambda value: ""
            if _is_blank_transfer_value(value)
            else str(value).strip()
        )
        frame[column] = values.astype("string")
    frame["source_track_name"] = frame["source_track_name"].map(
        _canonical_transfer_track_name
    ).astype("string")
    frame["reference_track_name"] = frame["reference_track_name"].map(
        lambda value: _canonical_transfer_track_name(value, allow_empty=True)
    ).astype("string")

    for column in _MANUAL_GT_TRANSFER_INTEGER_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        array = values.to_numpy(dtype=float, na_value=np.nan)
        nonempty_invalid = frame[column].map(
            lambda value: not _is_blank_transfer_value(value)
        ) & values.isna()
        if bool(nonempty_invalid.any()):
            raise ValueError(f"{column} contains non-integer values")
        invalid = np.isfinite(array) & (array != np.floor(array))
        if bool(np.any(invalid)) or bool(np.any(np.isinf(array))):
            raise ValueError(f"{column} must contain finite integer values")
        if (
            column not in _MANUAL_GT_TRANSFER_NULLABLE_INTEGER_COLUMNS
            and bool(np.any(np.isnan(array)))
        ):
            raise ValueError(f"{column} must not contain null values")
        frame[column] = values.astype("Int64")

    for column in _MANUAL_GT_TRANSFER_BOOLEAN_COLUMNS:
        frame[column] = frame[column].map(
            lambda value: _strict_transfer_bool(value, column)
        ).astype("boolean")

    for column in _MANUAL_GT_TRANSFER_FLOAT_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce")
        array = values.to_numpy(dtype=float, na_value=np.nan)
        if bool(np.any(np.isinf(array))):
            raise ValueError(f"{column} must contain finite values or null")
        nonempty_invalid = frame[column].map(
            lambda value: not _is_blank_transfer_value(value)
        ) & values.isna()
        if bool(nonempty_invalid.any()):
            raise ValueError(f"{column} contains non-numeric values")
        frame[column] = values.astype("Float64")

    if frame["source_track_name"].duplicated().any():
        raise ValueError("diagnostics source_track_name must be unique")
    if not frame["annotation_origin"].isin(["explicit", "implicit"]).all():
        raise ValueError("diagnostics annotation_origin is invalid")
    if not frame["gt_category"].isin(["ignition", "no_ignition"]).all():
        raise ValueError("diagnostics gt_category is invalid")
    if not frame["match_status"].isin(["matched", "excluded_unmatched"]).all():
        raise ValueError("diagnostics match_status is invalid")

    matched = frame["match_status"] == "matched"
    if (frame.loc[matched, "reference_track_name"] == "").any():
        raise ValueError("matched diagnostics require reference_track_name")
    if frame.loc[matched, "reference_track_id"].isna().any():
        raise ValueError("matched diagnostics require reference_track_id")
    if (frame.loc[~matched, "reference_track_name"] != "").any():
        raise ValueError("unmatched diagnostics must not name a reference track")
    if frame.loc[~matched, "reference_track_id"].notna().any():
        raise ValueError("unmatched diagnostics must not have reference_track_id")
    if frame.loc[matched, "reference_track_name"].duplicated().any():
        raise ValueError("matched reference_track_name must be unique")

    for _, row in frame.iterrows():
        if _numeric_track_id_from_name(str(row["source_track_name"])) != int(
            row["source_track_id"]
        ):
            raise ValueError("source track name/id mismatch in diagnostics")
        if row["match_status"] == "matched":
            if _numeric_track_id_from_name(str(row["reference_track_name"])) != int(
                row["reference_track_id"]
            ):
                raise ValueError("reference track name/id mismatch in diagnostics")
    return frame.sort_values("source_track_name", kind="stable").reset_index(drop=True)


def _normalize_transfer_mapping(
    mapping: Mapping[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_source, raw_reference in mapping.items():
        source = _canonical_transfer_track_name(raw_source)
        reference = _canonical_transfer_track_name(raw_reference)
        if source in normalized:
            raise ValueError(f"duplicate canonical transfer source: {source}")
        normalized[source] = reference
    return dict(sorted(normalized.items()))


def _normalize_tracking_annotations(
    annotations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_annotation in annotations.items():
        name = _canonical_transfer_track_name(raw_name)
        if name in normalized:
            raise ValueError(f"duplicate canonical tracking annotation: {name}")
        if not isinstance(raw_annotation, Mapping):
            raise ValueError(f"tracking annotation {name!r} must be an object")
        annotation = dict(raw_annotation)
        category = str(annotation.get("gt_category", "")).strip()
        if category not in {"ignition", "no_ignition"}:
            raise ValueError(
                f"tracking annotation {name!r} has unsupported category {category!r}"
            )
        annotation["gt_category"] = category
        annotation["gt_has_ignition"] = _strict_transfer_bool(
            annotation.get("gt_has_ignition"),
            f"{name}.gt_has_ignition",
        )
        for field in (
            "gt_onset_frame",
            "gt_onset_start",
            "gt_onset_end",
            "local_onset_idx",
        ):
            value = pd.to_numeric(annotation.get(field), errors="coerce")
            if pd.isna(value) or not math.isfinite(float(value)) or float(
                value
            ) != math.floor(float(value)):
                raise ValueError(f"{name}.{field} must be a finite integer")
            annotation[field] = int(value)
        annotation["source_track_name"] = _canonical_transfer_track_name(
            annotation.get("source_track_name")
        )
        annotation["onset_outside_reference_track"] = _strict_transfer_bool(
            annotation.get("onset_outside_reference_track"),
            f"{name}.onset_outside_reference_track",
        )
        if annotation["gt_has_ignition"] != (category == "ignition"):
            raise ValueError(f"{name} category/gt_has_ignition mismatch")
        normalized[name] = _to_jsonable(annotation)
    return dict(sorted(normalized.items()))


def _metadata_without_nonsemantic_fields(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): _to_jsonable(value)
        for key, value in metadata.items()
        if str(key) not in _MANUAL_GT_TRANSFER_HASH_EXCLUDED_METADATA_FIELDS
    }


def _semantic_diagnostics_records(
    diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    normalized = _coerce_manual_gt_transfer_diagnostics(diagnostics)
    records: list[dict[str, Any]] = []
    for row in normalized.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for column in _MANUAL_GT_TRANSFER_DIAG_COLUMNS:
            value = row[column]
            if value is pd.NA or pd.isna(value):
                record[column] = None
            elif isinstance(value, (np.bool_, bool)):
                record[column] = bool(value)
            elif isinstance(value, (np.integer, int)):
                record[column] = int(value)
            elif isinstance(value, (np.floating, float)):
                record[column] = float(value)
            else:
                record[column] = str(value)
        records.append(record)
    return records


def _manual_gt_transfer_content_hash(
    source_to_reference: Mapping[str, str],
    reference_to_source: Mapping[str, str],
    tracking_annotations: Mapping[str, Mapping[str, Any]],
    diagnostics: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "source_to_reference": _normalize_transfer_mapping(source_to_reference),
        "reference_to_source": _normalize_transfer_mapping(reference_to_source),
        "tracking_annotations": _normalize_tracking_annotations(
            tracking_annotations
        ),
        "diagnostics": _semantic_diagnostics_records(diagnostics),
        "metadata": _metadata_without_nonsemantic_fields(metadata),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_equal_ignoring_whitelist(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    def comparable(metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): _to_jsonable(value)
            for key, value in metadata.items()
            if str(key)
            not in _MANUAL_GT_TRANSFER_METADATA_COMPARISON_WHITELIST
        }

    return comparable(left) == comparable(right)


def _validate_transfer_artifact_consistency(
    source_to_reference: Mapping[str, str],
    reference_to_source: Mapping[str, str],
    tracking_annotations: Mapping[str, Mapping[str, Any]],
    diagnostics: pd.DataFrame,
) -> None:
    _validate_transfer_mapping_uniqueness(
        source_to_reference,
        reference_to_source,
    )
    matched = diagnostics.loc[diagnostics["match_status"] == "matched"]
    csv_mapping = {
        str(row["source_track_name"]): str(row["reference_track_name"])
        for _, row in matched.iterrows()
    }
    if dict(source_to_reference) != csv_mapping:
        raise ValueError("JSON source_to_reference disagrees with matched CSV rows")
    csv_reverse = {
        reference: source for source, reference in csv_mapping.items()
    }
    if dict(reference_to_source) != csv_reverse:
        raise ValueError("JSON reference_to_source disagrees with matched CSV rows")
    if set(tracking_annotations) != set(csv_reverse):
        raise ValueError("JSON tracking annotations disagree with matched CSV rows")

    for _, row in matched.iterrows():
        source_name = str(row["source_track_name"])
        reference_name = str(row["reference_track_name"])
        annotation = tracking_annotations[reference_name]
        expected = {
            "source_track_name": source_name,
            "gt_category": str(row["gt_category"]),
            "local_onset_idx": int(row["local_onset_idx"]),
            "gt_onset_frame": int(row["global_onset_frame"]),
            "onset_outside_reference_track": bool(
                row["onset_outside_reference_track"]
            ),
        }
        actual = {
            "source_track_name": str(annotation["source_track_name"]),
            "gt_category": str(annotation["gt_category"]),
            "local_onset_idx": int(annotation["local_onset_idx"]),
            "gt_onset_frame": int(annotation["gt_onset_frame"]),
            "onset_outside_reference_track": bool(
                annotation["onset_outside_reference_track"]
            ),
        }
        if actual != expected:
            raise ValueError(
                "JSON tracking annotation disagrees with matched CSV row "
                f"for {source_name}: expected={expected}, actual={actual}"
            )


_MANUAL_GT_TRANSFER_REQUIRED_INPUTS: tuple[str, ...] = (
    "annotation_path",
    "source_tracks_path",
    "crop_index_path",
    "reference_tracks_path",
)


def _resolve_manual_gt_transfer_input_paths(
    input_paths: Mapping[str, Any],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key in _MANUAL_GT_TRANSFER_REQUIRED_INPUTS:
        raw_path = input_paths.get(key)
        if raw_path is None:
            raise ValueError(f"{key} must not be None for manual GT transfer")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"manual GT transfer input {key} is not a file: {path}"
            )
        resolved[key] = path
    return resolved


def _manual_gt_transfer_input_hashes(
    input_paths: Mapping[str, Path],
) -> dict[str, str]:
    return {
        key: _file_content_sha256(input_paths[key])
        for key in _MANUAL_GT_TRANSFER_REQUIRED_INPUTS
    }


def _validate_manual_gt_source_input_hashes(
    source: ManualGtSource,
    input_hashes: Mapping[str, str],
) -> None:
    source_fields = {
        "annotation_path": "annotation_sha256",
        "source_tracks_path": "source_tracks_sha256",
        "crop_index_path": "crop_index_sha256",
    }
    mismatches: list[str] = []
    for input_key, metadata_key in source_fields.items():
        stored = source.metadata.get(metadata_key)
        current = input_hashes[input_key]
        if not isinstance(stored, str) or stored != current:
            mismatches.append(
                f"{input_key}: source={stored!r}, current={current!r}"
            )
    if mismatches:
        raise ValueError(
            "ManualGtSource input hashes are stale; reload ManualGtSource "
            "from the current annotation/source/crop files: "
            + "; ".join(mismatches)
        )


def _manual_gt_transfer_cache_key(
    input_hashes: Mapping[str, str],
    *,
    onset_tolerance: int,
    source_content_hash: str,
    reference_semantic_hash: str,
) -> dict[str, Any]:
    return {
        "cache_protocol": MANUAL_GT_TRANSFER_CACHE_PROTOCOL,
        "input_hashes": {
            key: str(input_hashes[key])
            for key in _MANUAL_GT_TRANSFER_REQUIRED_INPUTS
        },
        "matcher_gates": dict(FORMAL_MANUAL_GT_MATCHER_GATES),
        "onset_tolerance": int(onset_tolerance),
        "source_content_hash": str(source_content_hash),
        "reference_semantic_hash": str(reference_semantic_hash),
    }


def build_manual_gt_transfer(
    source: ManualGtSource,
    reference_tracks: pd.DataFrame,
    *,
    tracking_ablation: Any,
    onset_tolerance: int,
    output_dir: str | Path,
) -> ManualGtTransfer:
    """Match source tracks to T3 and materialize tracking annotations."""
    if not isinstance(source, ManualGtSource):
        raise TypeError("source must be a ManualGtSource")
    if source.metadata.get("crop_index_path") is None:
        raise ValueError(
            "crop_index_path must not be None for manual GT tracking transfer"
        )
    _require_dataframe(reference_tracks, "reference tracks")
    if reference_tracks.empty:
        raise ValueError("reference tracks must not be empty")
    matcher = getattr(tracking_ablation, "match_tracks_to_reference", None)
    if not callable(matcher):
        raise AttributeError(
            "tracking_ablation must expose match_tracks_to_reference"
        )

    query_to_ref, alignment = matcher(
        source.source_tracks,
        reference_tracks,
        max_center_dist=FORMAL_MANUAL_GT_MATCHER_GATES["max_center_dist"],
        min_shared_frames=FORMAL_MANUAL_GT_MATCHER_GATES["min_shared_frames"],
        min_ref_coverage=FORMAL_MANUAL_GT_MATCHER_GATES["min_ref_coverage"],
        min_query_coverage=FORMAL_MANUAL_GT_MATCHER_GATES["min_query_coverage"],
        min_temporal_iou=FORMAL_MANUAL_GT_MATCHER_GATES["min_temporal_iou"],
        max_assignment_cost=FORMAL_MANUAL_GT_MATCHER_GATES["max_assignment_cost"],
        detected_only=FORMAL_MANUAL_GT_MATCHER_GATES["detected_only"],
        return_diagnostics=True,
    )
    if not isinstance(query_to_ref, Mapping):
        raise TypeError("match_tracks_to_reference must return a mapping")
    alignment_lookup = _alignment_diagnostics_lookup(alignment)

    source_names = {str(name) for name in source.source_universe}
    reference_frames = _reference_track_frame_sets(
        reference_tracks,
        tracking_ablation=tracking_ablation,
    )
    reference_names = set(reference_frames)
    canonical_pairs: list[tuple[str, str]] = []
    unknown_query_ids: list[str] = []
    unknown_reference_ids: list[str] = []
    for raw_query, raw_ref in query_to_ref.items():
        try:
            source_name = _track_name_from_raw_id(
                raw_query,
                tracking_ablation=tracking_ablation,
            )
        except (TypeError, ValueError):
            source_name = ""
        try:
            reference_name = _track_name_from_raw_id(
                raw_ref,
                tracking_ablation=tracking_ablation,
            )
        except (TypeError, ValueError):
            reference_name = ""
        if source_name not in source_names:
            unknown_query_ids.append(str(raw_query))
        if reference_name not in reference_names:
            unknown_reference_ids.append(str(raw_ref))
        if source_name in source_names and reference_name in reference_names:
            canonical_pairs.append((source_name, reference_name))
    if unknown_query_ids or unknown_reference_ids:
        details = []
        if unknown_query_ids:
            details.append(
                "unknown query IDs: "
                + json.dumps(sorted(set(unknown_query_ids)))
            )
        if unknown_reference_ids:
            details.append(
                "unknown reference IDs: "
                + json.dumps(sorted(set(unknown_reference_ids)))
            )
        raise ValueError(
            "manual GT transfer matcher returned " + "; ".join(details)
        )

    source_to_reference: dict[str, str] = {}
    reference_to_source: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    for source_name, reference_name in canonical_pairs:
        existing_source = reference_to_source.get(reference_name)
        if existing_source is not None and existing_source != source_name:
            collisions.append(
                {
                    "reference_track_name": reference_name,
                    "source_track_name_a": existing_source,
                    "source_track_name_b": source_name,
                }
            )
            continue
        source_to_reference[source_name] = reference_name
        reference_to_source[reference_name] = source_name
    if collisions:
        raise ValueError(
            "manual GT transfer target collision: "
            + json.dumps(collisions, sort_keys=True)
        )
    _validate_transfer_mapping_uniqueness(source_to_reference, reference_to_source)
    explicit_set = set(source.explicit_names)
    tracking_annotations: dict[str, dict[str, Any]] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    onset_conversion_failures = 0
    outside_reference_count = 0
    mapped_explicit_positive = 0
    mapped_explicit_negative = 0
    mapped_implicit_negative = 0
    mapped_positive = 0
    mapped_negative = 0
    excluded_unmatched = 0
    excluded_unmatched_positive = 0
    excluded_unmatched_negative = 0

    for source_name in source.source_universe:
        annotation = source.detection_annotations[source_name]
        category = str(annotation.get("gt_category", ""))
        local_onset = int(annotation.get("gt_onset_idx", -1))
        origin = "explicit" if source_name in explicit_set else "implicit"
        source_id = _numeric_track_id_from_name(source_name)
        reference_name = source_to_reference.get(source_name)
        align = alignment_lookup.get(source_id, {})
        row: dict[str, Any] = {
            "source_track_name": source_name,
            "reference_track_name": reference_name or "",
            "source_track_id": source_id,
            "reference_track_id": (
                _numeric_track_id_from_name(reference_name)
                if reference_name
                else None
            ),
            "annotation_origin": origin,
            "gt_category": category,
            "local_onset_idx": local_onset,
            "global_onset_frame": -1,
            "match_status": "excluded_unmatched",
            "exclusion_reason": "",
            "onset_outside_reference_track": False,
            "shared_frames": align.get("shared_frames"),
            "reference_coverage": align.get("reference_coverage"),
            "query_coverage": align.get("query_coverage"),
            "temporal_iou": align.get("temporal_iou"),
            "assignment_cost": align.get("assignment_cost"),
            "mean_center_distance": align.get("mean_center_distance"),
        }
        if reference_name is None:
            row["exclusion_reason"] = "excluded_unmatched"
            excluded_unmatched += 1
            if category == "ignition":
                excluded_unmatched_positive += 1
            else:
                excluded_unmatched_negative += 1
            diagnostic_rows.append(row)
            continue

        row["match_status"] = "matched"
        row["exclusion_reason"] = ""
        if category == "ignition":
            try:
                global_onset = map_local_onset_to_global_frame(
                    source_name,
                    local_onset,
                    source.crop_sequences,
                    source_name=(
                        f"manual GT transfer local onset for {source_name}"
                    ),
                )
            except ValueError:
                onset_conversion_failures += 1
                raise
            row["global_onset_frame"] = int(global_onset)
            outside = global_onset >= 0 and global_onset not in reference_frames.get(
                reference_name,
                set(),
            )
            row["onset_outside_reference_track"] = bool(outside)
            if outside:
                outside_reference_count += 1
            tracking_annotation = _tracking_ignition_annotation(
                global_onset,
                onset_tolerance=int(onset_tolerance),
            )
            tracking_annotation.update(
                {
                    "source_track_name": source_name,
                    "local_onset_idx": int(local_onset),
                    "onset_outside_reference_track": bool(outside),
                }
            )
            tracking_annotations[reference_name] = tracking_annotation
            mapped_positive += 1
            if origin == "explicit":
                mapped_explicit_positive += 1
        else:
            tracking_annotation = _tracking_no_ignition_annotation()
            tracking_annotation.update(
                {
                    "source_track_name": source_name,
                    "local_onset_idx": int(local_onset),
                    "onset_outside_reference_track": False,
                }
            )
            tracking_annotations[reference_name] = tracking_annotation
            mapped_negative += 1
            if origin == "explicit":
                mapped_explicit_negative += 1
            else:
                mapped_implicit_negative += 1
        diagnostic_rows.append(row)

    if mapped_positive <= 0:
        raise ValueError(
            "manual GT transfer produced zero mapped positives; "
            "refusing to continue a tracking sweep"
        )

    diagnostics = _coerce_manual_gt_transfer_diagnostics(
        pd.DataFrame(
            diagnostic_rows,
            columns=list(_MANUAL_GT_TRANSFER_DIAG_COLUMNS),
        )
    )
    source_to_reference = _normalize_transfer_mapping(source_to_reference)
    reference_to_source = _normalize_transfer_mapping(reference_to_source)
    tracking_annotations = _normalize_tracking_annotations(tracking_annotations)
    _validate_transfer_artifact_consistency(
        source_to_reference,
        reference_to_source,
        tracking_annotations,
        diagnostics,
    )
    metadata = {
        "format": "manual_review_gt",
        "uses_reference_predictions": False,
        "onset_tolerance": int(onset_tolerance),
        "matcher_gates": dict(FORMAL_MANUAL_GT_MATCHER_GATES),
        "source_universe_count": int(len(source.source_universe)),
        "source_explicit_track_count": int(len(source.explicit_names)),
        "source_explicit_positive_count": int(
            source.metadata.get("explicit_positive_count", 0)
        ),
        "source_explicit_negative_count": int(
            source.metadata.get("explicit_negative_count", 0)
        ),
        "source_implicit_negative_count": int(
            source.metadata.get("implicit_negative_count", 0)
        ),
        "matched_count": int(len(source_to_reference)),
        "excluded_unmatched_count": int(excluded_unmatched),
        "excluded_unmatched_positive_count": int(excluded_unmatched_positive),
        "excluded_unmatched_negative_count": int(excluded_unmatched_negative),
        "mapped_positive_count": int(mapped_positive),
        "mapped_negative_count": int(mapped_negative),
        "mapped_explicit_positive_count": int(mapped_explicit_positive),
        "mapped_explicit_negative_count": int(mapped_explicit_negative),
        "mapped_implicit_negative_count": int(mapped_implicit_negative),
        "mapped_explicit_count": int(
            mapped_explicit_positive + mapped_explicit_negative
        ),
        "onset_conversion_failure_count": int(onset_conversion_failures),
        "onset_outside_reference_track_count": int(outside_reference_count),
        "target_collision_count": 0,
        "tracking_annotation_count": int(len(tracking_annotations)),
        "source_content_hash": str(source.content_hash),
    }
    metadata = _to_jsonable(metadata)
    content_hash = _manual_gt_transfer_content_hash(
        source_to_reference,
        reference_to_source,
        tracking_annotations,
        diagnostics,
        metadata,
    )
    metadata["content_hash"] = content_hash

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(output_dir / "manual_gt_transfer.csv", diagnostics)
    _atomic_write_json(
        output_dir / "manual_gt_transfer_summary.json",
        metadata,
    )
    return ManualGtTransfer(
        source_to_reference=dict(source_to_reference),
        reference_to_source=dict(reference_to_source),
        tracking_annotations=copy.deepcopy(tracking_annotations),
        diagnostics=diagnostics.copy(deep=True),
        metadata=metadata,
        content_hash=content_hash,
    )


def _load_manual_gt_transfer_from_cache(
    output_dir: Path,
    *,
    expected_cache_key: Mapping[str, Any],
) -> Optional[ManualGtTransfer]:
    cache_path = output_dir / "manual_gt_transfer_cache.json"
    csv_path = output_dir / "manual_gt_transfer.csv"
    summary_path = output_dir / "manual_gt_transfer_summary.json"
    if not (cache_path.is_file() and csv_path.is_file() and summary_path.is_file()):
        return None
    try:
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        diagnostics = pd.read_csv(csv_path, keep_default_na=False)
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return None
    if not isinstance(cache_payload, Mapping) or not isinstance(
        summary_payload,
        Mapping,
    ):
        return None
    for key in (
        "cache_protocol",
        "input_hashes",
        "matcher_gates",
        "onset_tolerance",
        "reference_semantic_hash",
    ):
        if cache_payload.get(key) != expected_cache_key.get(key):
            return None
    if cache_payload.get("source_content_hash") != expected_cache_key.get(
        "source_content_hash"
    ):
        return None

    source_to_reference = cache_payload.get("source_to_reference")
    reference_to_source = cache_payload.get("reference_to_source")
    tracking_annotations = cache_payload.get("tracking_annotations")
    cache_metadata = cache_payload.get("metadata")
    if not (
        isinstance(source_to_reference, Mapping)
        and isinstance(reference_to_source, Mapping)
        and isinstance(tracking_annotations, Mapping)
        and isinstance(cache_metadata, Mapping)
    ):
        return None
    try:
        source_to_reference = _normalize_transfer_mapping(source_to_reference)
        reference_to_source = _normalize_transfer_mapping(reference_to_source)
        tracking_annotations = _normalize_tracking_annotations(
            tracking_annotations
        )
        diagnostics = _coerce_manual_gt_transfer_diagnostics(diagnostics)
        _validate_transfer_artifact_consistency(
            source_to_reference,
            reference_to_source,
            tracking_annotations,
            diagnostics,
        )
        metadata = _to_jsonable(dict(summary_payload))
        normalized_cache_metadata = _to_jsonable(dict(cache_metadata))
        if not _metadata_equal_ignoring_whitelist(
            metadata,
            normalized_cache_metadata,
        ):
            raise ValueError("cache metadata disagrees with summary JSON")
        content_hash = _manual_gt_transfer_content_hash(
            source_to_reference,
            reference_to_source,
            tracking_annotations,
            diagnostics,
            metadata,
        )
        stored_hashes = {
            str(cache_payload.get("transfer_content_hash", "")),
            str(metadata.get("content_hash", "")),
            str(normalized_cache_metadata.get("content_hash", "")),
        }
        if stored_hashes != {content_hash}:
            raise ValueError("manual GT transfer cache content hash mismatch")
    except (TypeError, ValueError, KeyError, OverflowError):
        return None

    return ManualGtTransfer(
        source_to_reference=dict(source_to_reference),
        reference_to_source=dict(reference_to_source),
        tracking_annotations=copy.deepcopy(tracking_annotations),
        diagnostics=diagnostics.copy(deep=True),
        metadata=metadata,
        content_hash=content_hash,
    )


def load_or_build_manual_gt_transfer(
    source: ManualGtSource,
    reference_tracks: pd.DataFrame,
    *,
    tracking_ablation: Any,
    onset_tolerance: int,
    output_dir: Path,
    input_paths: Mapping[str, Path],
) -> ManualGtTransfer:
    """Reuse a validated JSON/CSV transfer cache or rebuild formally."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_paths = _resolve_manual_gt_transfer_input_paths(input_paths)
    input_hashes = _manual_gt_transfer_input_hashes(resolved_paths)
    _validate_manual_gt_source_input_hashes(source, input_hashes)
    recomputed_source_hash = _manual_gt_source_hash(
        source.detection_annotations,
        source.source_universe,
        source.explicit_names,
        source.crop_sequences,
        source.source_tracks,
        source.metadata,
    )
    if recomputed_source_hash != source.content_hash:
        raise ValueError(
            "ManualGtSource content changed after loading; reload ManualGtSource"
        )

    reference_from_file = pd.read_csv(resolved_paths["reference_tracks_path"])
    reference_semantic_hash = _dataframe_semantic_hash(reference_tracks)
    reference_file_semantic_hash = _dataframe_semantic_hash(reference_from_file)
    if reference_semantic_hash != reference_file_semantic_hash:
        raise ValueError(
            "reference_tracks DataFrame does not match reference_tracks_path file"
        )
    cache_key = _manual_gt_transfer_cache_key(
        input_hashes,
        onset_tolerance=int(onset_tolerance),
        source_content_hash=str(source.content_hash),
        reference_semantic_hash=reference_semantic_hash,
    )
    cached = _load_manual_gt_transfer_from_cache(
        output_dir,
        expected_cache_key=cache_key,
    )
    if cached is not None:
        return cached

    transfer = build_manual_gt_transfer(
        source,
        reference_tracks,
        tracking_ablation=tracking_ablation,
        onset_tolerance=int(onset_tolerance),
        output_dir=output_dir,
    )
    cache_payload = {
        **cache_key,
        "transfer_content_hash": transfer.content_hash,
        "source_to_reference": dict(transfer.source_to_reference),
        "reference_to_source": dict(transfer.reference_to_source),
        "tracking_annotations": copy.deepcopy(transfer.tracking_annotations),
        "metadata": dict(transfer.metadata),
    }
    _atomic_write_json(output_dir / "manual_gt_transfer_cache.json", cache_payload)
    return transfer


def _source_identity_sort_key(name: str) -> tuple[Any, ...]:
    text = str(name)
    try:
        track_id = _numeric_track_id_from_name(text)
    except (TypeError, ValueError):
        return (True, text)
    return (False, int(track_id))


def read_effective_trace_positive_onsets(
    annotation_path: str | Path,
) -> dict[str, int]:
    """Read ``track_id``/``frame_id`` as positive local-onset annotations.

    A track present in the workbook with a numeric ``frame_id`` is a positive
    ignition example. Tracks in the evaluated population that are absent from
    the workbook are treated as negative examples.
    """
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"--annotation-path file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix in {".txt", ".tsv"} else ","
        table = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported annotation format: {path.suffix}")
    table = table.copy()
    table.columns = [str(column).strip().lower() for column in table.columns]
    required = {"track_id", "frame_id"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "effective_trace GT requires track_id and frame_id columns; "
            f"missing={missing}, columns={list(table.columns)}"
        )

    positives: dict[str, int] = {}
    for row_number, row in table.iterrows():
        raw_id = row.get("track_id")
        try:
            track_id = _numeric_track_id_from_name(str(raw_id))
        except ValueError as exc:
            raise ValueError(
                f"Invalid track_id at annotation row {int(row_number) + 2}: "
                f"{raw_id!r}"
            ) from exc
        numeric_onset = pd.to_numeric(
            pd.Series([row.get("frame_id")]), errors="coerce"
        ).iloc[0]
        if pd.isna(numeric_onset):
            raise ValueError(
                f"Non-numeric frame_id at annotation row {int(row_number) + 2}"
            )
        onset = float(numeric_onset)
        if not math.isfinite(onset) or onset != math.floor(onset) or onset < 0:
            raise ValueError(
                f"frame_id must be a non-negative integer at annotation row "
                f"{int(row_number) + 2}: {numeric_onset!r}"
            )
        name = f"track_{track_id}"
        onset_index = int(onset)
        if name in positives and positives[name] != onset_index:
            raise ValueError(
                f"Conflicting onset labels for {name}: "
                f"{positives[name]} and {onset_index}"
            )
        positives[name] = onset_index
    if not positives:
        raise ValueError(f"No positive ignition annotations found in {path}")
    return positives


def _detection_gt_annotation(
    onset_index: Optional[int], onset_tolerance: int
) -> dict[str, Any]:
    if onset_index is None:
        return {
            "gt_category": "no_ignition",
            "gt_has_ignition": False,
            "gt_onset_idx": -1,
            "gt_onset_start": -1,
            "gt_onset_end": -1,
        }
    onset = int(onset_index)
    return {
        "gt_category": "ignition",
        "gt_has_ignition": True,
        "gt_onset_idx": onset,
        "gt_onset_start": max(0, onset - int(onset_tolerance)),
        "gt_onset_end": onset + int(onset_tolerance),
    }


def build_effective_trace_positive_gt_plan(
    annotation_path: str | Path,
    *,
    detection_track_names: Sequence[str],
    reference_tracks: Optional[pd.DataFrame],
    tracking_ablation: Any,
    onset_tolerance: int,
    detection_selected: bool,
    tracking_selected: bool,
) -> GtExecutionPlan:
    """Build detector-local and tracking-global GT from positive annotations."""
    if onset_tolerance < 0:
        raise ValueError("--onset-tolerance must be non-negative")
    positives = read_effective_trace_positive_onsets(annotation_path)

    detection_names = tuple(
        sorted(
            {canonical_track_name(name) for name in detection_track_names},
            key=_source_identity_sort_key,
        )
    )
    detection_annotations = {
        name: _detection_gt_annotation(positives.get(name), onset_tolerance)
        for name in detection_names
    } if detection_selected else {}

    tracking_annotations: dict[str, Mapping[str, Any]] = {}
    tracking_names: tuple[str, ...] = ()
    bad_onsets: list[str] = []
    if tracking_selected:
        if reference_tracks is None:
            raise ValueError(
                "Tracking sensitivity requires --reference-tracks-csv so local "
                "GT onset indices can be mapped to global frames"
            )
        _require_dataframe(reference_tracks, "reference tracks")
        if "track_id" not in reference_tracks or "frame_id" not in reference_tracks:
            raise ValueError("reference tracks require track_id and frame_id columns")
        mapped = reference_tracks[["track_id", "frame_id"]].copy()
        mapped["track_name"] = mapped["track_id"].map(
            lambda value: _track_name_from_raw_id(
                value, tracking_ablation=tracking_ablation
            )
        )
        mapped["frame_id"] = pd.to_numeric(mapped["frame_id"], errors="raise")
        if (mapped["frame_id"] != np.floor(mapped["frame_id"])).any():
            raise ValueError("reference track frame_id values must be integers")
        mapped["frame_id"] = mapped["frame_id"].astype(np.int64)
        if mapped.duplicated(["track_name", "frame_id"], keep=False).any():
            raise ValueError(
                "reference tracks contain duplicate (track_id, frame_id) rows"
            )
        tracking_names = tuple(
            sorted(mapped["track_name"].unique(), key=_source_identity_sort_key)
        )
        for name in tracking_names:
            onset_index = positives.get(name)
            if onset_index is None:
                tracking_annotations[name] = _tracking_no_ignition_annotation()
                continue
            frames = (
                mapped.loc[mapped["track_name"] == name, "frame_id"]
                .sort_values(kind="stable")
                .tolist()
            )
            if onset_index >= len(frames):
                bad_onsets.append(
                    f"{name}: onset={onset_index}, track_length={len(frames)}"
                )
                continue
            tracking_annotations[name] = _tracking_ignition_annotation(
                int(frames[onset_index]), onset_tolerance=int(onset_tolerance)
            )
        if bad_onsets:
            raise ValueError(
                "GT local onset index is outside its reference trajectory: "
                + "; ".join(bad_onsets[:20])
            )
        _validate_tracking_annotation_schema(tracking_annotations)

    source_to_reference, reference_to_source = _identity_alias_maps(
        tracking_names
    )
    universe = tuple(
        sorted(
            set(detection_annotations) | set(tracking_annotations),
            key=_source_identity_sort_key,
        )
    )
    missing_from_detection = sorted(
        set(positives) - set(detection_annotations), key=_source_identity_sort_key
    )
    missing_from_tracking = sorted(
        set(positives) - set(tracking_annotations), key=_source_identity_sort_key
    )
    metadata = {
        "format": "effective_trace_positive",
        "semantics": (
            "positive iff effective_trace contains track_id with numeric "
            "frame_id; all other tracks in the evaluated population are negative"
        ),
        "uses_reference_predictions": False,
        "positive_annotation_count": int(len(positives)),
        "detection_population_count": int(len(detection_annotations)),
        "tracking_population_count": int(len(tracking_annotations)),
        "detection_positive_count": int(
            sum(bool(row["gt_has_ignition"]) for row in detection_annotations.values())
        ),
        "tracking_positive_count": int(
            sum(bool(row["gt_has_ignition"]) for row in tracking_annotations.values())
        ),
        "positive_labels_absent_from_detection_population": missing_from_detection,
        "positive_labels_absent_from_tracking_population": missing_from_tracking,
        "listed_no_ignition_ids": [],
        "annotation_sha256": _file_content_sha256(annotation_path),
    }
    content_hash = _gt_execution_plan_content_hash(
        format_name="effective_trace_positive",
        detection_annotations=detection_annotations,
        tracking_annotations=tracking_annotations,
        source_to_reference=source_to_reference,
        reference_to_source=reference_to_source,
        source_universe=universe,
        metadata=metadata,
    )
    return GtExecutionPlan(
        format="effective_trace_positive",
        detection_annotations=detection_annotations,
        tracking_annotations=tracking_annotations,
        source_to_reference=source_to_reference,
        reference_to_source=reference_to_source,
        source_universe=universe,
        metadata=_to_jsonable(metadata),
        content_hash=content_hash,
    )


def detect_sensitivity_gt_format(
    annotation_path: str | Path,
    ignition_ablation: Any = None,
) -> str:
    """Validate and identify the positive-onset ``effective_trace`` format."""
    del ignition_ablation
    read_effective_trace_positive_onsets(annotation_path)
    return "effective_trace_positive"


def resolve_candidate_source_identities(
    *,
    source_universe: Sequence[str],
    detection_annotation_names: Mapping[str, Any] | Sequence[str],
    detection_folder_names: set[str] | Sequence[str],
    source_to_reference: Mapping[str, str],
    detection_selected: bool,
    tracking_selected: bool,
) -> tuple[str, ...]:
    """Build the shared source-ID candidate population for the selected groups."""
    # source_universe is retained in GtExecutionPlan for audit only. Candidate
    # eligibility must come from the annotation view actually used by a branch.
    _ = source_universe
    detection_names = {str(name) for name in detection_annotation_names}
    folders = {str(name) for name in detection_folder_names}
    tracking_names = {str(name) for name in source_to_reference}
    if detection_selected and tracking_selected:
        selected = detection_names & tracking_names & folders
    elif detection_selected:
        selected = detection_names & folders
    elif tracking_selected:
        selected = tracking_names
    else:
        selected = set()
    return tuple(sorted(selected, key=_source_identity_sort_key))


def resolve_split_annotation_view(
    gt_plan: GtExecutionPlan,
    candidate_names: Sequence[str],
    *,
    detection_selected: bool,
    tracking_selected: bool,
) -> dict[str, Mapping[str, Any]]:
    """Return source-keyed GT used for splitting the effective candidates."""
    candidates = tuple(str(name) for name in candidate_names)
    if detection_selected:
        split_view = _filter_annotations(
            gt_plan.detection_annotations, candidates
        )
        if tracking_selected:
            for source_name in candidates:
                reference_name = gt_plan.source_to_reference.get(source_name)
                if reference_name is None:
                    raise ValueError(
                        "Mixed candidate missing source→T3 mapping: "
                        f"{source_name!r}"
                    )
                tracking_gt = gt_plan.tracking_annotations.get(
                    str(reference_name)
                )
                if tracking_gt is None:
                    raise ValueError(
                        "Mixed candidate missing tracking annotation: "
                        f"{source_name!r}→{reference_name!r}"
                    )
                detection_gt = split_view[source_name]
                detection_semantics = (
                    str(detection_gt.get("gt_category", "")).strip().lower(),
                    bool(detection_gt.get("gt_has_ignition", False)),
                )
                tracking_semantics = (
                    str(tracking_gt.get("gt_category", "")).strip().lower(),
                    bool(tracking_gt.get("gt_has_ignition", False)),
                )
                if detection_semantics != tracking_semantics:
                    raise ValueError(
                        "Mixed detection/tracking GT category/presence "
                        f"mismatch for {source_name!r}: "
                        f"{detection_semantics!r} != {tracking_semantics!r}"
                    )
        return split_view
    if tracking_selected:
        split_view: dict[str, Mapping[str, Any]] = {}
        for source_name in candidates:
            reference_name = gt_plan.source_to_reference.get(source_name)
            if reference_name is None:
                raise ValueError(
                    "Tracking split identity missing source→T3 mapping: "
                    f"{source_name!r}"
                )
            if str(reference_name) not in gt_plan.tracking_annotations:
                raise ValueError(
                    "Tracking split identity missing tracking annotation: "
                    f"{source_name!r}→{reference_name!r}"
                )
            split_view[source_name] = copy.deepcopy(
                dict(gt_plan.tracking_annotations[str(reference_name)])
            )
        return split_view
    return {}


def resolve_tracking_evaluation_keys(
    evaluation_names: Sequence[str],
    source_to_reference: Mapping[str, str],
) -> tuple[str, ...]:
    """Map source evaluation identities to T3 annotation keys (order-preserving)."""
    tracking_keys: list[str] = []
    for source_name in evaluation_names:
        reference_name = source_to_reference.get(str(source_name))
        if reference_name is None:
            raise ValueError(
                "Tracking evaluation identity missing source→T3 mapping: "
                f"{source_name!r}"
            )
        tracking_keys.append(str(reference_name))
    return tuple(tracking_keys)


def build_evaluation_identity_aliases(
    *,
    tracking_annotations: Mapping[str, Mapping[str, Any]],
    reference_to_source: Mapping[str, str],
) -> dict[str, str]:
    """Map T3 annotation keys to unique source evaluation identities."""
    aliases: dict[str, str] = {}
    seen_sources: set[str] = set()
    for reference_name in tracking_annotations:
        key = str(reference_name)
        if key not in reference_to_source:
            raise ValueError(
                f"Missing evaluation identity alias for reference track {key!r}"
            )
        source_name = str(reference_to_source[key])
        if source_name in seen_sources:
            raise ValueError(
                f"duplicate evaluation identity alias target {source_name!r}"
            )
        seen_sources.add(source_name)
        aliases[key] = source_name
    return aliases


def build_execution_context_payload(
    *,
    gt_format: str,
    gt_content_hash: str,
    manual_gt_transfer_hash: Optional[str],
    source_test_track_set_hash: str,
    tracking_target_track_set_hash: Optional[str],
    dataset_split_hash: str,
) -> dict[str, Any]:
    """Build resume/config execution-context fields and a combined hash."""
    payload = {
        "gt_format": str(gt_format),
        "gt_content_hash": str(gt_content_hash),
        "manual_gt_transfer_hash": (
            None
            if manual_gt_transfer_hash is None
            else str(manual_gt_transfer_hash)
        ),
        "source_test_track_set_hash": str(source_test_track_set_hash),
        "tracking_target_track_set_hash": (
            None
            if tracking_target_track_set_hash is None
            else str(tracking_target_track_set_hash)
        ),
        "dataset_split_hash": str(dataset_split_hash),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    payload["execution_context_hash"] = digest
    return payload


def _validate_resume_execution_context(
    existing_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
) -> None:
    if "tracking_metric_protocol" in expected_config and existing_config.get(
        "tracking_metric_protocol"
    ) != expected_config["tracking_metric_protocol"]:
        raise ValueError("Resume rejected: old tracking metrics/window protocol. Use a new output directory or --overwrite.")
    for key in _RESUME_EXECUTION_CONTEXT_KEYS:
        if key not in expected_config and key not in existing_config:
            continue
        if key not in existing_config or existing_config.get(key) != expected_config.get(
            key
        ):
            raise ValueError(
                "Resume rejected because execution context "
                f"{key!r} is missing or mismatched. Use --overwrite or a new "
                "output directory."
            )


def _gt_execution_plan_content_hash(
    *,
    format_name: str,
    detection_annotations: Mapping[str, Mapping[str, Any]],
    tracking_annotations: Mapping[str, Mapping[str, Any]],
    source_to_reference: Mapping[str, str],
    reference_to_source: Mapping[str, str],
    source_universe: Sequence[str],
    metadata: Mapping[str, Any],
) -> str:
    payload = {
        "format": str(format_name),
        "detection_annotations": _to_jsonable(dict(detection_annotations)),
        "tracking_annotations": _to_jsonable(dict(tracking_annotations)),
        "source_to_reference": dict(source_to_reference),
        "reference_to_source": dict(reference_to_source),
        "source_universe": list(source_universe),
        "metadata": _to_jsonable(dict(metadata)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _identity_alias_maps(
    names: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    source_to_reference = {str(name): str(name) for name in names}
    reference_to_source = dict(source_to_reference)
    return source_to_reference, reference_to_source


def load_sensitivity_gt(
    annotation_path: Path,
    reference_ignition: Optional[pd.DataFrame],
    *,
    source_tracks_path: Optional[Path],
    crop_index_path: Optional[Path],
    reference_tracks: Optional[pd.DataFrame],
    ignition_ablation: Any,
    ignition_mod: Any,
    tracking_ablation: Any,
    onset_tolerance: int,
    selected_groups: set[str],
    output_dir: Path,
    reference_tracks_path: Optional[Path] = None,
    detection_track_names: Sequence[str] = (),
) -> GtExecutionPlan:
    """Load GT and produce a dual-mode execution plan."""
    annotation_path = Path(annotation_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(selected_groups)
    tracking_selected = bool(selected & TRACKING_GROUPS)
    detection_selected = bool(selected & {"gmm", "rtv"})
    format_name = detect_sensitivity_gt_format(annotation_path, ignition_ablation)

    if format_name == "effective_trace_positive":
        return build_effective_trace_positive_gt_plan(
            annotation_path,
            detection_track_names=detection_track_names,
            reference_tracks=reference_tracks,
            tracking_ablation=tracking_ablation,
            onset_tolerance=int(onset_tolerance),
            detection_selected=bool(detection_selected),
            tracking_selected=bool(tracking_selected),
        )

    if format_name == "manual_review_gt":
        if source_tracks_path is None:
            raise ValueError(
                "manual_review_gt requires --manual-gt-source-tracks-csv "
                "(source tracks)"
            )
        source = load_manual_gt_source(
            annotation_path,
            source_tracks_path,
            crop_index_path,
            ignition_ablation=ignition_ablation,
            ignition_mod=ignition_mod,
            onset_tolerance=int(onset_tolerance),
            require_crop_index=bool(tracking_selected),
        )
        detection_annotations = canonicalize_annotation_mapping(
            source.detection_annotations, formal_module=ignition_mod
        )
        tracking_annotations: dict[str, Mapping[str, Any]] = {}
        source_to_reference: dict[str, str] = {}
        reference_to_source: dict[str, str] = {}
        transfer_hash: Optional[str] = None
        if tracking_selected:
            if reference_tracks is None:
                raise ValueError(
                    "manual_review_gt tracking groups require preloaded "
                    "reference_tracks before GT transfer"
                )
            if tracking_ablation is None:
                raise ValueError(
                    "manual_review_gt tracking groups require tracking_ablation"
                )
            if crop_index_path is None:
                raise ValueError(
                    "manual_review_gt tracking groups require "
                    "--manual-gt-source-crop-index-csv"
                )
            resolved_reference_path = (
                Path(reference_tracks_path)
                if reference_tracks_path is not None
                else output_dir / "manual_gt_reference_tracks.csv"
            )
            if reference_tracks_path is None:
                reference_tracks.to_csv(resolved_reference_path, index=False)
            transfer = load_or_build_manual_gt_transfer(
                source,
                reference_tracks,
                tracking_ablation=tracking_ablation,
                onset_tolerance=int(onset_tolerance),
                output_dir=output_dir,
                input_paths={
                    "annotation_path": annotation_path,
                    "source_tracks_path": Path(source_tracks_path),
                    "crop_index_path": Path(crop_index_path),
                    "reference_tracks_path": resolved_reference_path,
                },
            )
            tracking_annotations = canonicalize_annotation_mapping(
                transfer.tracking_annotations, formal_module=tracking_ablation
            )
            source_to_reference = dict(transfer.source_to_reference)
            reference_to_source = dict(transfer.reference_to_source)
            transfer_hash = str(transfer.content_hash)
        metadata = {
            **dict(source.metadata),
            "format": "manual_review_gt",
            "uses_reference_predictions": False,
            "manual_gt_transfer_hash": transfer_hash,
            "detection_selected": bool(detection_selected),
            "tracking_selected": bool(tracking_selected),
            "source_content_hash": source.content_hash,
        }
        content_hash = _gt_execution_plan_content_hash(
            format_name="manual_review_gt",
            detection_annotations=detection_annotations,
            tracking_annotations=tracking_annotations,
            source_to_reference=source_to_reference,
            reference_to_source=reference_to_source,
            source_universe=source.source_universe,
            metadata=metadata,
        )
        return GtExecutionPlan(
            format="manual_review_gt",
            detection_annotations=detection_annotations,
            tracking_annotations=tracking_annotations,
            source_to_reference=source_to_reference,
            reference_to_source=reference_to_source,
            source_universe=tuple(source.source_universe),
            metadata=_to_jsonable(metadata),
            content_hash=content_hash,
        )

    # Legacy effective_trace_new path — preserve existing builders.
    listed_ids = ignition_ablation.framework.read_invalid_track_ids(
        annotation_path
    )
    detection_annotations = canonicalize_annotation_mapping(
        _build_detection_annotations_from_reference(
            listed_ids,
            reference_ignition,
            onset_tolerance=int(onset_tolerance),
            framework=ignition_ablation.framework,
        ),
        formal_module=ignition_mod,
    )
    tracking_annotations = {}
    if tracking_selected:
        if tracking_ablation is None:
            raise ValueError(
                "effective_trace_new tracking groups require tracking_ablation"
            )
        tracking_annotations = canonicalize_annotation_mapping(
            tracking_ablation.build_gt_from_effective_trace_new_and_ref_ignition(
                list(listed_ids),
                reference_ignition,
                onset_tolerance=int(onset_tolerance),
            ),
            formal_module=tracking_ablation,
        )
    universe_names = tuple(
        sorted(
            set(detection_annotations) | set(tracking_annotations),
            key=_source_identity_sort_key,
        )
    )
    source_to_reference, reference_to_source = _identity_alias_maps(
        tuple(tracking_annotations)
    )
    metadata = {
        "format": "effective_trace_new",
        "uses_reference_predictions": True,
        "listed_no_ignition_ids": [int(value) for value in listed_ids],
        "manual_gt_transfer_hash": None,
        "detection_selected": bool(detection_selected),
        "tracking_selected": bool(tracking_selected),
        "source_universe_count": int(len(universe_names)),
    }
    content_hash = _gt_execution_plan_content_hash(
        format_name="effective_trace_new",
        detection_annotations=detection_annotations,
        tracking_annotations=tracking_annotations,
        source_to_reference=source_to_reference,
        reference_to_source=reference_to_source,
        source_universe=universe_names,
        metadata=metadata,
    )
    return GtExecutionPlan(
        format="effective_trace_new",
        detection_annotations=detection_annotations,
        tracking_annotations=tracking_annotations,
        source_to_reference=source_to_reference,
        reference_to_source=reference_to_source,
        source_universe=universe_names,
        metadata=_to_jsonable(metadata),
        content_hash=content_hash,
    )


def _range_or_none(values: pd.Series) -> Optional[list[int]]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=float))]
    if numeric.empty:
        return None
    return [int(numeric.min()), int(numeric.max())]


def _ignition_onset_diagnostics(
    ignition: Optional[pd.DataFrame],
    name: str,
) -> dict[str, Any]:
    if ignition is None:
        return {"range": None, "invalid_count": 0, "negative_count": 0}
    _require_dataframe(ignition, name)
    for column in (
        "ignition_onset_frame",
        "onset_frame",
        "gt_onset_frame",
        "pred_onset_frame",
    ):
        if column in ignition.columns:
            numeric = pd.to_numeric(ignition[column], errors="coerce")
            values = numeric.to_numpy(dtype=float)
            finite = np.isfinite(values)
            integral = finite & (values == np.floor(values))
            nonnegative = integral & (values >= 0)
            valid = numeric[nonnegative]
            return {
                "range": (
                    [int(valid.min()), int(valid.max())]
                    if not valid.empty
                    else None
                ),
                "invalid_count": int((~finite | (finite & ~integral)).sum()),
                "negative_count": int((integral & (values < 0)).sum()),
            }
    raise ValueError(f"{name} must include an ignition onset frame column")


def _sequence_comparison_mode(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> bool:
    left_has = "sequence_id" in left.columns
    right_has = "sequence_id" in right.columns
    left_count = int(left["sequence_id"].nunique(dropna=False)) if left_has else 0
    right_count = int(right["sequence_id"].nunique(dropna=False)) if right_has else 0
    if left_count > 1 and not right_has:
        raise ValueError(
            f"{left_name} contains multiple sequences; {right_name} must include sequence_id"
        )
    if right_count > 1 and not left_has:
        raise ValueError(
            f"{right_name} contains multiple sequences; {left_name} must include sequence_id"
        )
    return left_has and right_has


def _frame_keys(
    frame: pd.DataFrame,
    frames: pd.Series,
    *,
    use_sequence: bool,
    offset: int = 0,
) -> pd.Series:
    shifted = frames + int(offset)
    if not use_sequence:
        return shifted
    return pd.Series(
        list(zip(frame["sequence_id"].map(str), shifted.astype(int))),
        index=frame.index,
    )


def diagnose_frame_alignment(
    query_tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    *,
    query_ignition: Optional[pd.DataFrame] = None,
    reference_ignition: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Report frame overlap for offsets -2..2 without changing either table."""
    _require_dataframe(query_tracks, "query_tracks")
    _require_dataframe(reference_tracks, "reference_tracks")
    query_frames = _integer_series(query_tracks, "frame_id", "query_tracks")
    reference_frames = _integer_series(
        reference_tracks, "frame_id", "reference_tracks"
    )
    sequence_aware = _sequence_comparison_mode(
        query_tracks, reference_tracks, "query_tracks", "reference_tracks"
    )
    reference_keys = _frame_keys(
        reference_tracks, reference_frames, use_sequence=sequence_aware
    )
    reference_frame_set = set(reference_keys)
    offset_statistics: list[dict[str, int]] = []
    for offset in range(-2, 3):
        shifted = _frame_keys(
            query_tracks,
            query_frames,
            use_sequence=sequence_aware,
            offset=offset,
        )
        shared = set(shifted) & reference_frame_set
        offset_statistics.append(
            {
                "offset": offset,
                "shared_frames": int(len(shared)),
                "query_rows_on_shared_frames": int(shifted.isin(shared).sum()),
                "reference_rows_on_shared_frames": int(
                    reference_keys.isin(shared).sum()
                ),
            }
        )
    best = max(
        offset_statistics,
        key=lambda row: (
            row["shared_frames"],
            row["query_rows_on_shared_frames"]
            + row["reference_rows_on_shared_frames"],
            -abs(row["offset"]),
            -row["offset"],
        ),
    )
    query_onset = _ignition_onset_diagnostics(query_ignition, "query_ignition")
    reference_onset = _ignition_onset_diagnostics(
        reference_ignition, "reference_ignition"
    )
    return {
        "query_frame_range": _range_or_none(query_frames),
        "reference_frame_range": _range_or_none(reference_frames),
        "query_ignition_onset_range": query_onset["range"],
        "query_ignition_invalid_onset_count": query_onset["invalid_count"],
        "query_ignition_negative_onset_count": query_onset["negative_count"],
        "reference_ignition_onset_range": reference_onset["range"],
        "reference_ignition_invalid_onset_count": reference_onset["invalid_count"],
        "reference_ignition_negative_onset_count": reference_onset["negative_count"],
        "offset_statistics": offset_statistics,
        "best_offset": int(best["offset"]),
        "sequence_aware": sequence_aware,
        "frames_modified": False,
    }


def _center_geometry(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    columns = _coordinate_columns(frame, name)
    coordinates = _finite_coordinate_values(frame, columns, name)
    has_center = all(column in coordinates for column in _CENTER_COLUMNS)
    has_box = all(column in coordinates for column in _BOX_COLUMNS)
    if has_center:
        cx = coordinates["cx"]
        cy = coordinates["cy"]
        center_source = "cx/cy"
    else:
        cx = (coordinates["x1"] + coordinates["x2"]) / 2.0
        cy = (coordinates["y1"] + coordinates["y2"]) / 2.0
        center_source = "box"
    width = (
        coordinates["x2"] - coordinates["x1"]
        if has_box
        else pd.Series(np.nan, index=frame.index)
    )
    height = (
        coordinates["y2"] - coordinates["y1"]
        if has_box
        else pd.Series(np.nan, index=frame.index)
    )
    return pd.DataFrame(
        {
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "x1": coordinates["x1"] if has_box else np.nan,
            "y1": coordinates["y1"] if has_box else np.nan,
            "x2": coordinates["x2"] if has_box else np.nan,
            "y2": coordinates["y2"] if has_box else np.nan,
            "matching_center_source": center_source,
        },
        index=frame.index,
    )


def _geometry_summary(geometry: pd.DataFrame) -> dict[str, Any]:
    width = geometry["width"].dropna()
    height = geometry["height"].dropna()
    has_box = not width.empty
    return {
        "matching_center_source": str(geometry["matching_center_source"].iloc[0]),
        "center_x_range": [
            float(geometry["cx"].min()),
            float(geometry["cx"].max()),
        ],
        "center_y_range": [
            float(geometry["cy"].min()),
            float(geometry["cy"].max()),
        ],
        "median_width": float(width.median()) if not width.empty else None,
        "median_height": float(height.median()) if not height.empty else None,
        "box_x_range": (
            [
                float(geometry["x1"].min()),
                float(geometry["x2"].max()),
            ]
            if has_box
            else None
        ),
        "box_y_range": (
            [
                float(geometry["y1"].min()),
                float(geometry["y2"].max()),
            ]
            if has_box
            else None
        ),
    }


def diagnose_coordinate_alignment(
    query_tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
) -> dict[str, Any]:
    """Describe coordinate scales and same-frame nearest-center distances."""
    _require_dataframe(query_tracks, "query_tracks")
    _require_dataframe(reference_tracks, "reference_tracks")
    query_frames = _integer_series(query_tracks, "frame_id", "query_tracks")
    reference_frames = _integer_series(
        reference_tracks, "frame_id", "reference_tracks"
    )
    query_geometry = _center_geometry(query_tracks, "query_tracks")
    reference_geometry = _center_geometry(reference_tracks, "reference_tracks")
    sequence_aware = _sequence_comparison_mode(
        query_tracks, reference_tracks, "query_tracks", "reference_tracks"
    )
    query_keys = _frame_keys(
        query_tracks, query_frames, use_sequence=sequence_aware
    ).reset_index(drop=True)
    reference_keys = _frame_keys(
        reference_tracks, reference_frames, use_sequence=sequence_aware
    ).reset_index(drop=True)
    query_centers = query_geometry[["cx", "cy"]].to_numpy(dtype=float)
    reference_centers = reference_geometry[["cx", "cy"]].to_numpy(dtype=float)
    query_groups = {
        key: positions.to_numpy(dtype=int)
        for key, positions in query_keys.groupby(query_keys).groups.items()
    }
    reference_groups = {
        key: positions.to_numpy(dtype=int)
        for key, positions in reference_keys.groupby(reference_keys).groups.items()
    }

    distance_arrays: list[np.ndarray] = []
    shared_keys = set(query_groups) & set(reference_groups)
    for key in shared_keys:
        query_points = query_centers[query_groups[key]]
        reference_points = reference_centers[reference_groups[key]]
        pairwise = np.linalg.norm(
            query_points[:, np.newaxis, :] - reference_points[np.newaxis, :, :],
            axis=2,
        )
        distance_arrays.append(pairwise.min(axis=1))
    distances = (
        np.concatenate(distance_arrays)
        if distance_arrays
        else np.asarray([], dtype=float)
    )

    percentile_summary: dict[str, Any] = {"count": int(distances.size)}
    for percentile in (10, 25, 50, 75, 90, 95):
        percentile_summary[f"p{percentile}"] = (
            float(np.percentile(distances, percentile))
            if distances.size
            else None
        )
    return {
        "query": _geometry_summary(query_geometry),
        "reference": _geometry_summary(reference_geometry),
        "shared_frames": int(len(shared_keys)),
        "sequence_aware": sequence_aware,
        "nearest_reference_center_distance": percentile_summary,
    }


def _normalize_identity(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("track_id values must not be empty")
    text = str(value).strip()
    if not text:
        raise ValueError("track_id values must not be empty")
    lowered = text.lower()
    for prefix in ("track_", "track-", "track "):
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    try:
        numeric = float(text)
    except ValueError:
        return text.lower()
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text.lower()


def _identity_records(
    frame: pd.DataFrame,
    name: str,
) -> tuple[set[tuple[str, ...]], list[str], int]:
    _require_dataframe(frame, name)
    if "track_id" not in frame.columns:
        raise ValueError(f"{name} must include track_id")
    normalized = frame["track_id"].map(_normalize_identity)
    has_sequence = "sequence_id" in frame.columns
    sequences = (
        frame["sequence_id"].map(lambda value: str(value).strip())
        if has_sequence
        else pd.Series([""] * len(frame), index=frame.index)
    )
    sequence_count = int(sequences.nunique(dropna=False)) if len(frame) else 0

    variants: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for sequence, raw, identity in zip(sequences, frame["track_id"], normalized):
        key = (sequence, identity)
        raw_token = (type(raw).__name__, str(raw).strip())
        variants.setdefault(key, set()).add(raw_token)
    mixed = next(
        ((identity, values) for (_, identity), values in variants.items() if len(values) > 1),
        None,
    )
    if mixed is not None:
        raise ValueError(
            f"{name} mixes raw track IDs that normalize to '{mixed[0]}'"
        )

    columns = ["sequence_id", "track_id"] if has_sequence else ["track_id"]
    identities = (
        set(zip(sequences.astype(str), normalized.astype(str)))
        if has_sequence
        else set((value,) for value in normalized.astype(str))
    )
    return identities, columns, sequence_count


def _normalized_identity_set(
    frame: pd.DataFrame,
    *,
    use_sequence: bool,
) -> set[tuple[str, ...]]:
    normalized = frame["track_id"].map(_normalize_identity).astype(str)
    if use_sequence:
        return set(zip(frame["sequence_id"].map(str), normalized))
    return set((value,) for value in normalized)


def _ignition_identity_coverage(
    ignition: pd.DataFrame,
    ignition_name: str,
    track_frame: pd.DataFrame,
    track_name: str,
) -> dict[str, Any]:
    track_has_sequence = "sequence_id" in track_frame.columns
    ignition_has_sequence = "sequence_id" in ignition.columns
    track_sequence_count = (
        int(track_frame["sequence_id"].nunique(dropna=False))
        if track_has_sequence
        else 0
    )
    ignition_sequence_count = (
        int(ignition["sequence_id"].nunique(dropna=False))
        if ignition_has_sequence
        else 0
    )
    if track_sequence_count > 1 and "sequence_id" not in ignition.columns:
        raise ValueError(
            f"{ignition_name} must include sequence_id because "
            f"{track_name} contains multiple sequences"
        )
    if ignition_sequence_count > 1 and not track_has_sequence:
        raise ValueError(
            f"{ignition_name} contains multiple sequences; "
            f"{track_name} must include sequence_id"
        )
    _identity_records(ignition, ignition_name)
    use_sequence = track_has_sequence and ignition_has_sequence
    track_identities = _normalized_identity_set(
        track_frame, use_sequence=use_sequence
    )
    identities = _normalized_identity_set(ignition, use_sequence=use_sequence)
    columns = ["sequence_id", "track_id"] if use_sequence else ["track_id"]
    canonical_keys = pd.Series(
        list(_normalized_identity_set(ignition, use_sequence=use_sequence))
    )
    if len(canonical_keys) != len(ignition):
        raise ValueError(f"{ignition_name} identities must be unique")
    missing = track_identities - identities
    extra = identities - track_identities
    if missing:
        raise ValueError(
            f"{ignition_name} does not cover all {track_name} identities"
        )
    return {
        "identity_columns": columns,
        "rows": int(len(ignition)),
        "covered_tracks": int(len(track_identities) - len(missing)),
        "coverage": float(1.0 if not track_identities else 1.0 - len(missing) / len(track_identities)),
        "extra_identities": int(len(extra)),
    }


def assert_track_identity_uniqueness(
    query_tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    *,
    query_ignition: Optional[pd.DataFrame] = None,
    reference_ignition: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Validate normalized track identities and optional ignition coverage."""
    query_ids, query_columns, query_sequences = _identity_records(
        query_tracks, "query_tracks"
    )
    reference_ids, reference_columns, reference_sequences = _identity_records(
        reference_tracks, "reference_tracks"
    )
    result: dict[str, Any] = {
        "query": {
            "identity_columns": query_columns,
            "tracks": int(len(query_ids)),
            "sequences": query_sequences,
        },
        "reference": {
            "identity_columns": reference_columns,
            "tracks": int(len(reference_ids)),
            "sequences": reference_sequences,
        },
    }
    if query_ignition is not None:
        result["query_ignition"] = _ignition_identity_coverage(
            query_ignition,
            "query_ignition",
            query_tracks,
            "query_tracks",
        )
    if reference_ignition is not None:
        result["reference_ignition"] = _ignition_identity_coverage(
            reference_ignition,
            "reference_ignition",
            reference_tracks,
            "reference_tracks",
        )
    return result


def format_tracking_alignment_startup_summary(
    reference_validation: Mapping[str, Any],
    frame_alignment: Mapping[str, Any],
    coordinate_alignment: Mapping[str, Any],
    identity_diagnostics: Mapping[str, Any],
) -> str:
    """Format diagnostics for startup logging without performing any I/O."""
    nearest = coordinate_alignment.get(
        "nearest_reference_center_distance", {}
    )
    query_identity = identity_diagnostics.get("query", {})
    reference_identity = identity_diagnostics.get("reference", {})
    return "\n".join(
        (
            (
                "reference "
                f"rows={reference_validation.get('rows')} "
                f"tracks={reference_validation.get('tracks')} "
                f"frames={reference_validation.get('frame_min')}"
                f"..{reference_validation.get('frame_max')}"
            ),
            (
                f"best frame offset={frame_alignment.get('best_offset')} "
                "(diagnostic only)"
            ),
            (
                "nearest reference center distance "
                f"count={nearest.get('count')} p50={nearest.get('p50')}"
            ),
            (
                f"identity tracks query={query_identity.get('tracks')} "
                f"reference={reference_identity.get('tracks')}"
            ),
        )
    )


def _flatten_diagnostic_values(
    section: str,
    value: Any,
    prefix: str = "",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            metric = f"{prefix}.{key}" if prefix else str(key)
            records.extend(
                _flatten_diagnostic_values(section, child, metric)
            )
        return records
    if isinstance(value, (list, tuple)):
        scalar: Any = json.dumps(value, sort_keys=True)
    elif isinstance(value, np.generic):
        scalar = value.item()
    elif value is None or isinstance(value, (str, int, float, bool)):
        scalar = value
    else:
        scalar = str(value)
    records.append({"section": section, "metric": prefix, "value": scalar})
    return records


def build_tracking_alignment_diagnostic_records(
    reference_validation: Mapping[str, Any],
    frame_alignment: Mapping[str, Any],
    coordinate_alignment: Mapping[str, Any],
    identity_diagnostics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return scalar records suitable for JSON or a three-column CSV."""
    sections = (
        ("reference_validation", reference_validation),
        ("frame_alignment", frame_alignment),
        ("coordinate_alignment", coordinate_alignment),
        ("identity_diagnostics", identity_diagnostics),
    )
    return [
        record
        for section, diagnostics in sections
        for record in _flatten_diagnostic_values(section, diagnostics)
    ]


_ALIGNMENT_CANDIDATE_COLUMNS: tuple[str, ...] = (
    "query_track_id",
    "reference_track_id",
    "shared_frames",
    "mean_center_distance",
    "median_center_distance",
    "p90_center_distance",
    "query_coverage",
    "reference_coverage",
    "temporal_iou",
    "assignment_cost",
    "total_match_cost",
    "accepted",
    "reject_reason",
    "diagnostic_warning",
)
_ALIGNMENT_REJECTION_REASONS: tuple[str, ...] = (
    "no_shared_frames",
    "insufficient_shared_frames",
    "center_distance_too_large",
    "reference_coverage_too_low",
    "query_coverage_too_low",
    "temporal_overlap_too_low",
    "total_match_cost_too_large",
    "duplicate_assignment",
    "invalid_coordinates",
)


def _diagnostic_alignment_row_center(row: pd.Series) -> tuple[float, float]:
    """Mirror revised `_row_center`: missing cx/cy falls back to the bbox."""
    if (
        "cx" in row
        and pd.notna(row["cx"])
        and "cy" in row
        and pd.notna(row["cy"])
    ):
        center = (float(row["cx"]), float(row["cy"]))
    else:
        center = (
            float(row["x1"] + row["x2"]) * 0.5,
            float(row["y1"] + row["y2"]) * 0.5,
        )
    if not all(math.isfinite(value) for value in center):
        raise ValueError("non-finite center")
    return center


def _count_invalid_alignment_coordinates(
    frame: pd.DataFrame,
    tracking_ablation_mod: Any,
) -> int:
    """Count invalid detected points using the revised filtering semantics."""
    filter_points = getattr(
        tracking_ablation_mod, "filter_detected_track_points", None
    )
    if callable(filter_points):
        detected = filter_points(frame)
    elif "point_type" in frame:
        point_type = frame["point_type"].fillna("detected").astype(str)
        detected = frame[point_type == "detected"].copy()
    else:
        detected = frame.copy()
    invalid = 0
    for _, row in detected.iterrows():
        try:
            _diagnostic_alignment_row_center(row)
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid += 1
    return int(invalid)


def _centroid_candidate_pairs(
    query_centroids: Mapping[int, Mapping[int, tuple[float, float]]],
    reference_centroids: Mapping[int, Mapping[int, tuple[float, float]]],
) -> set[tuple[int, int]]:
    query_by_frame: dict[int, list[int]] = {}
    reference_by_frame: dict[int, list[int]] = {}
    for track_id, points in query_centroids.items():
        for frame_id in points:
            query_by_frame.setdefault(int(frame_id), []).append(int(track_id))
    for track_id, points in reference_centroids.items():
        for frame_id in points:
            reference_by_frame.setdefault(int(frame_id), []).append(int(track_id))
    pairs: set[tuple[int, int]] = set()
    for frame_id in set(query_by_frame) & set(reference_by_frame):
        pairs.update(
            (query_id, reference_id)
            for query_id in query_by_frame[frame_id]
            for reference_id in reference_by_frame[frame_id]
        )
    return pairs


def build_alignment_candidate_diagnostics(
    query_tracks: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    formal_mapping: Mapping[int, int],
    *,
    tracking_ablation_mod: Any,
    max_center_dist: float = 30.0,
    min_shared_frames: int = 3,
    min_ref_coverage: float = 0.20,
    min_query_coverage: float = 0.20,
    min_temporal_iou: float = 0.15,
    max_assignment_cost: float = 1.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explain formal matching gates without producing the accepted mapping."""
    query_centroids = tracking_ablation_mod.build_track_centroids_by_frame(
        query_tracks, detected_only=True
    )
    reference_centroids = tracking_ablation_mod.build_track_centroids_by_frame(
        reference_tracks, detected_only=True
    )
    query_ids = sorted(int(value) for value in query_centroids)
    reference_ids = sorted(int(value) for value in reference_centroids)
    formal_pairs = {
        (int(query_id), int(reference_id))
        for query_id, reference_id in formal_mapping.items()
    }
    shared_pairs = _centroid_candidate_pairs(
        query_centroids, reference_centroids
    )
    no_shared_count = len(query_ids) * len(reference_ids) - len(shared_pairs)
    invalid_input_coordinates = (
        _count_invalid_alignment_coordinates(
            query_tracks, tracking_ablation_mod
        )
        + _count_invalid_alignment_coordinates(
            reference_tracks, tracking_ablation_mod
        )
    )
    rows: list[dict[str, Any]] = []

    for query_id, reference_id in sorted(shared_pairs):
        query_points = query_centroids[query_id]
        reference_points = reference_centroids[reference_id]
        query_frames = set(query_points)
        reference_frames = set(reference_points)
        shared = query_frames & reference_frames
        distances: list[float] = []
        invalid_coordinates = False
        for frame_id in shared:
            try:
                query_x, query_y = query_points[frame_id]
                reference_x, reference_y = reference_points[frame_id]
                values = np.asarray(
                    [query_x, query_y, reference_x, reference_y], dtype=float
                )
            except (TypeError, ValueError):
                invalid_coordinates = True
                continue
            if not bool(np.isfinite(values).all()):
                invalid_coordinates = True
                continue
            distances.append(
                float(math.hypot(query_x - reference_x, query_y - reference_y))
            )

        shared_count = int(len(shared))
        query_coverage = float(shared_count / max(len(query_frames), 1))
        reference_coverage = float(
            shared_count / max(len(reference_frames), 1)
        )
        temporal_iou = float(
            shared_count / max(len(query_frames | reference_frames), 1)
        )
        mean_distance = (
            float(np.mean(distances)) if distances else float("nan")
        )
        p90_distance = (
            float(np.percentile(distances, 90))
            if distances
            else float("nan")
        )
        median_distance = (
            float(np.median(distances)) if distances else float("nan")
        )
        total_cost = (
            mean_distance / max(float(max_center_dist), 1e-6)
            + 0.50 * (1.0 - reference_coverage)
            + 0.25 * (1.0 - query_coverage)
            + 0.25 * (1.0 - temporal_iou)
            if np.isfinite(mean_distance)
            else float("nan")
        )

        if invalid_coordinates or len(distances) != shared_count:
            reason = "invalid_coordinates"
        elif shared_count < int(min_shared_frames):
            reason = "insufficient_shared_frames"
        elif (
            mean_distance > float(max_center_dist)
            or p90_distance > 1.5 * float(max_center_dist)
        ):
            reason = "center_distance_too_large"
        elif reference_coverage < float(min_ref_coverage):
            reason = "reference_coverage_too_low"
        elif query_coverage < float(min_query_coverage):
            reason = "query_coverage_too_low"
        elif temporal_iou < float(min_temporal_iou):
            reason = "temporal_overlap_too_low"
        elif total_cost > float(max_assignment_cost):
            reason = "total_match_cost_too_large"
        elif (query_id, reference_id) not in formal_pairs:
            reason = "duplicate_assignment"
        else:
            reason = ""
        accepted = (query_id, reference_id) in formal_pairs
        diagnostic_warning = (
            f"formal_mapping_failed_local_diagnostic_gate:{reason}"
            if accepted and reason
            else ""
        )
        rows.append(
            {
                "query_track_id": query_id,
                "reference_track_id": reference_id,
                "shared_frames": shared_count,
                "mean_center_distance": mean_distance,
                "median_center_distance": median_distance,
                "p90_center_distance": p90_distance,
                "query_coverage": query_coverage,
                "reference_coverage": reference_coverage,
                "temporal_iou": temporal_iou,
                "assignment_cost": float(total_cost),
                "total_match_cost": float(total_cost),
                "accepted": bool(accepted),
                "reject_reason": reason,
                "diagnostic_warning": diagnostic_warning,
            }
        )

    represented_formal_pairs = {
        (int(row["query_track_id"]), int(row["reference_track_id"]))
        for row in rows
        if bool(row["accepted"])
    }
    for query_id, reference_id in sorted(
        formal_pairs - represented_formal_pairs
    ):
        rows.append(
            {
                "query_track_id": query_id,
                "reference_track_id": reference_id,
                "shared_frames": np.nan,
                "mean_center_distance": np.nan,
                "median_center_distance": np.nan,
                "p90_center_distance": np.nan,
                "query_coverage": np.nan,
                "reference_coverage": np.nan,
                "temporal_iou": np.nan,
                "assignment_cost": np.nan,
                "total_match_cost": np.nan,
                "accepted": True,
                "reject_reason": "",
                "diagnostic_warning": (
                    "formal_mapping_missing_local_candidate"
                ),
            }
        )
    candidates = pd.DataFrame(rows, columns=_ALIGNMENT_CANDIDATE_COLUMNS)
    rejection_counts = (
        candidates.loc[~candidates["accepted"], "reject_reason"]
        .value_counts()
        .to_dict()
        if len(candidates)
        else {}
    )
    rejection_counts["no_shared_frames"] = int(max(no_shared_count, 0))
    rejection_counts["invalid_coordinates"] = int(
        rejection_counts.get("invalid_coordinates", 0)
        + invalid_input_coordinates
    )
    for reason in _ALIGNMENT_REJECTION_REASONS:
        rejection_counts.setdefault(reason, 0)
    rejection_summary = pd.DataFrame(
        [
            {"reject_reason": reason, "count": int(count)}
            for reason, count in sorted(rejection_counts.items())
        ],
        columns=["reject_reason", "count"],
    )
    return candidates, rejection_summary


def build_mapping_sample_first20(
    formal_mapping: Mapping[int, int],
    formal_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return the first formal query->reference mappings with formal fields."""
    aliases = {
        "ref_track_id": "reference_track_id",
        "n_shared_frames": "shared_frames",
        "assignment_cost": "total_match_cost",
    }
    diagnostics = formal_diagnostics.rename(columns=aliases).copy()
    rows: list[dict[str, Any]] = []
    for query_id, reference_id in sorted(
        ((int(q), int(r)) for q, r in formal_mapping.items())
    )[:20]:
        if {
            "query_track_id",
            "reference_track_id",
        }.issubset(diagnostics.columns):
            matched = diagnostics[
                (
                    pd.to_numeric(
                        diagnostics["query_track_id"], errors="coerce"
                    )
                    == query_id
                )
                & (
                    pd.to_numeric(
                        diagnostics["reference_track_id"], errors="coerce"
                    )
                    == reference_id
                )
            ]
        else:
            matched = pd.DataFrame()
        if matched.empty:
            row = {
                "diagnostic_warning": (
                    "formal_mapping_missing_formal_diagnostic"
                )
            }
        else:
            row = matched.iloc[0].to_dict()
            row["diagnostic_warning"] = ""
        row["query_track_id"] = query_id
        row["reference_track_id"] = reference_id
        rows.append(row)
    preferred = [
        "query_track_id",
        "reference_track_id",
        "shared_frames",
        "mean_center_distance",
        "query_coverage",
        "reference_coverage",
        "temporal_iou",
        "total_match_cost",
        "diagnostic_warning",
    ]
    result = pd.DataFrame(rows)
    return result.reindex(
        columns=preferred
        + [column for column in result.columns if column not in preferred]
    )


def _annotation_is_positive(annotation: Mapping[str, Any]) -> bool:
    category = str(annotation.get("gt_category", "")).strip().lower()
    return category == "ignition" or bool(
        annotation.get("gt_has_ignition", False)
    )


def build_alignment_funnel(
    *,
    detections: Mapping[int, Sequence[Mapping[str, Any]]],
    query_tracks: pd.DataFrame,
    crop_data: Any,
    query_ignition: pd.DataFrame,
    reference_tracks: pd.DataFrame,
    annotations: Mapping[str, Mapping[str, Any]],
    candidate_diagnostics: pd.DataFrame,
    query_to_ref: Mapping[int, int],
    projected_ignition: pd.DataFrame,
    evaluation_metrics: Mapping[str, Any],
    row_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-ready end-to-end alignment funnel."""
    def has_successful_crop(track_result: Any) -> bool:
        if not isinstance(track_result, Mapping):
            return False
        success_count = pd.to_numeric(
            pd.Series([track_result.get("n_success_crops")]),
            errors="coerce",
        ).iloc[0]
        if pd.notna(success_count):
            return float(success_count) > 0
        crop_paths = track_result.get("crop_paths")
        if crop_paths is not None:
            try:
                return len(crop_paths) > 0
            except TypeError:
                return False
        crop_meta = track_result.get("crop_meta")
        return isinstance(crop_meta, pd.DataFrame) and not crop_meta.empty

    query_ids = (
        set(pd.to_numeric(query_tracks["track_id"], errors="coerce").dropna().astype(int))
        if "track_id" in query_tracks
        else set()
    )
    reference_ids = (
        set(
            pd.to_numeric(reference_tracks["track_id"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if "track_id" in reference_tracks
        else set()
    )
    positive_reference_ids = {
        int(str(name).split("_", 1)[1])
        for name, annotation in annotations.items()
        if _annotation_is_positive(annotation)
        and str(name).startswith("track_")
        and str(name).split("_", 1)[1].isdigit()
    }
    mapped_reference_ids = {int(value) for value in query_to_ref.values()}
    query_frames = (
        set(pd.to_numeric(query_tracks["frame_id"], errors="coerce").dropna().astype(int))
        if "frame_id" in query_tracks
        else set()
    )
    reference_frames = (
        set(
            pd.to_numeric(reference_tracks["frame_id"], errors="coerce")
            .dropna()
            .astype(int)
        )
        if "frame_id" in reference_tracks
        else set()
    )
    crop_count = (
        sum(has_successful_crop(result) for result in crop_data.values())
        if isinstance(crop_data, Mapping)
        else 0
    )
    ignition_tracks = (
        int(query_ignition["track_id"].nunique())
        if "track_id" in query_ignition
        else int(len(query_ignition))
    )
    accepted = (
        candidate_diagnostics["accepted"].fillna(False).astype(bool)
        if "accepted" in candidate_diagnostics
        else pd.Series(False, index=candidate_diagnostics.index)
    )
    reasons = (
        candidate_diagnostics["reject_reason"].fillna("")
        if "reject_reason" in candidate_diagnostics
        else pd.Series("", index=candidate_diagnostics.index)
    )
    all_candidate_rows = pd.Series(True, index=candidate_diagnostics.index)

    def has_numeric_diagnostics(*columns: str) -> pd.Series:
        available = all_candidate_rows.copy()
        for column in columns:
            if column in candidate_diagnostics:
                available &= pd.to_numeric(
                    candidate_diagnostics[column], errors="coerce"
                ).notna()
        return available

    diagnostic_warnings = (
        candidate_diagnostics["diagnostic_warning"].fillna("").astype(str)
        if "diagnostic_warning" in candidate_diagnostics
        else pd.Series("", index=candidate_diagnostics.index)
    )
    missing_local_diagnostics = diagnostic_warnings.str.startswith(
        "formal_mapping_missing_local_"
    )
    shared_frame_diagnostics = (
        has_numeric_diagnostics("shared_frames")
        & ~missing_local_diagnostics
    )
    if "shared_frames" in candidate_diagnostics:
        shared_frame_diagnostics &= pd.to_numeric(
            candidate_diagnostics["shared_frames"], errors="coerce"
        ).gt(0)
    minimum_shared_frame_gate = shared_frame_diagnostics
    center_distance_gate = (
        minimum_shared_frame_gate
        & has_numeric_diagnostics(
            "mean_center_distance", "p90_center_distance"
        )
    )
    reference_coverage_gate = (
        center_distance_gate
        & has_numeric_diagnostics("reference_coverage")
    )
    query_coverage_gate = (
        reference_coverage_gate
        & has_numeric_diagnostics("query_coverage")
    )
    temporal_overlap_gate = (
        query_coverage_gate
        & has_numeric_diagnostics("temporal_iou")
    )
    total_match_cost_gate = (
        temporal_overlap_gate
        & has_numeric_diagnostics("total_match_cost")
    )
    return {
        "n_detections": int(sum(len(rows) for rows in detections.values())),
        "n_query_tracks": int(len(query_ids)),
        "n_query_track_rows": int(len(query_tracks)),
        "n_query_tracks_with_crops": int(crop_count),
        "n_query_tracks_with_ignition_results": ignition_tracks,
        "n_reference_tracks": int(len(reference_ids)),
        "n_reference_track_rows": int(len(reference_tracks)),
        "n_reference_positive_ignition": int(len(positive_reference_ids)),
        "n_candidate_track_pairs": int(len(query_ids) * len(reference_ids)),
        "n_pairs_with_shared_frames": int(shared_frame_diagnostics.sum()),
        "n_pairs_passing_minimum_shared_frame_gate": int(
            (
                minimum_shared_frame_gate
                & ~reasons.isin(
                    ["insufficient_shared_frames", "invalid_coordinates"]
                )
            ).sum()
        ),
        "n_pairs_passing_center_distance_gate": int(
            (
                center_distance_gate
                & ~reasons.isin([
                    "insufficient_shared_frames",
                    "invalid_coordinates",
                    "center_distance_too_large",
                ])
            ).sum()
        ),
        "n_pairs_passing_reference_coverage_gate": int(
            (
                reference_coverage_gate
                & ~reasons.isin([
                    "insufficient_shared_frames",
                    "invalid_coordinates",
                    "center_distance_too_large",
                    "reference_coverage_too_low",
                ])
            ).sum()
        ),
        "n_pairs_passing_query_coverage_gate": int(
            (
                query_coverage_gate
                & ~reasons.isin([
                    "insufficient_shared_frames",
                    "invalid_coordinates",
                    "center_distance_too_large",
                    "reference_coverage_too_low",
                    "query_coverage_too_low",
                ])
            ).sum()
        ),
        "n_pairs_passing_temporal_overlap_gate": int(
            (
                temporal_overlap_gate
                & ~reasons.isin([
                    "insufficient_shared_frames",
                    "invalid_coordinates",
                    "center_distance_too_large",
                    "reference_coverage_too_low",
                    "query_coverage_too_low",
                    "temporal_overlap_too_low",
                ])
            ).sum()
        ),
        "n_pairs_passing_total_match_cost_gate": int(
            (
                total_match_cost_gate
                & ~reasons.isin([
                    "insufficient_shared_frames",
                    "invalid_coordinates",
                    "center_distance_too_large",
                    "reference_coverage_too_low",
                    "query_coverage_too_low",
                    "temporal_overlap_too_low",
                    "total_match_cost_too_large",
                ])
            ).sum()
        ),
        "n_matched_track_pairs": int(len(query_to_ref)),
        "n_matched_reference_positive": int(
            len(mapped_reference_ids & positive_reference_ids)
        ),
        "n_projected_ignition_predictions": int(
            _truthy_series(projected_ignition["has_ignition"]).sum()
            if "has_ignition" in projected_ignition
            else len(projected_ignition)
        ),
        "n_unmatched_reference_tracks": int(
            len(reference_ids - mapped_reference_ids)
        ),
        "n_unmatched_predicted_ignition_tracks": int(
            evaluation_metrics.get(
                "unmatched_predicted_ignition_count", 0
            )
            or 0
        ),
        "shared_frame_count": int(len(query_frames & reference_frames)),
        "reference_track_coverage": float(
            len(mapped_reference_ids) / max(len(reference_ids), 1)
        ),
        "tp": int(
            evaluation_metrics.get("end_to_end_tp", row_metrics.get("tp", 0))
            or 0
        ),
        "fp": int(
            evaluation_metrics.get("end_to_end_fp", row_metrics.get("fp", 0))
            or 0
        ),
        "fn": int(
            evaluation_metrics.get("end_to_end_fn", row_metrics.get("fn", 0))
            or 0
        ),
        "tn": int(row_metrics.get("tn", 0) or 0),
        "n_accepted_candidate_rows": int(accepted.sum()),
    }


def validate_default_tracking_alignment(
    funnel: Mapping[str, Any],
) -> None:
    """Reject a canonical default run that violates any required invariant."""
    conditions = (
        ("n_query_tracks", float(funnel.get("n_query_tracks", 0)) > 0),
        ("n_reference_tracks", float(funnel.get("n_reference_tracks", 0)) > 0),
        ("shared_frame_count", float(funnel.get("shared_frame_count", 0)) > 0),
        (
            "n_matched_track_pairs",
            float(funnel.get("n_matched_track_pairs", 0)) > 0,
        ),
        (
            "reference_track_coverage",
            float(funnel.get("reference_track_coverage", 0)) > 0.1,
        ),
        (
            "n_reference_positive_ignition",
            float(funnel.get("n_reference_positive_ignition", 0)) > 0,
        ),
        ("tp", float(funnel.get("tp", 0)) > 0),
        (
            "n_projected_ignition_predictions",
            float(funnel.get("n_projected_ignition_predictions", 0)) > 0,
        ),
    )
    failed = [name for name, passed in conditions if not passed]
    if failed:
        if (
            float(funnel.get("n_reference_positive_ignition", 0)) > 0
            and float(funnel.get("tp", 0)) <= 0
        ):
            message = (
                "Default T3 produced zero matched true-positive ignition events. "
                "Track/reference alignment is invalid."
            )
        else:
            message = (
                "Default T3 tracking alignment failed required conditions: "
                + ", ".join(failed)
            )
        error = TrackingAlignmentError(message)
        error.alignment_funnel = dict(funnel)  # type: ignore[attr-defined]
        raise error


_REPRODUCIBILITY_METRICS: tuple[
    tuple[str, str], ...
] = (
    ("n_tracks", "n_tracks"),
    ("mean_track_length", "mean_track_len"),
    ("short_track_count", "n_short_tracks"),
    ("n_matched", "n_matched_to_ref"),
    ("reference_coverage", "reference_track_coverage"),
    ("tp", "end_to_end_tp"),
    ("fp", "end_to_end_fp"),
    ("fn", "end_to_end_fn"),
    ("ignition_precision", "ignition_precision"),
    ("ignition_recall", "ignition_recall"),
    ("ignition_f1", "ignition_f1"),
    ("onset_mae", "onset_mae"),
)


def _tracks_long_reproducibility_stats(tracks: pd.DataFrame) -> dict[str, float]:
    lengths = (
        tracks.groupby("track_id").size().astype(float)
        if "track_id" in tracks
        else pd.Series(dtype=float)
    )
    return {
        "n_tracks": float(len(lengths)),
        "tracks_long_row_count": float(len(tracks)),
        "mean_track_len": float(lengths.mean()) if len(lengths) else 0.0,
        "n_short_tracks": float((lengths < 5).sum()),
    }


def build_default_t3_reproducibility_check(
    sensitivity_metrics: Mapping[str, Any],
    sensitivity_tracks_long: pd.DataFrame,
    *,
    t3_variant_metrics_path: str | Path,
    t3_tracks_long_path: str | Path,
    warning_path: Optional[str | Path] = None,
    gt_format: str = "effective_trace_new",
) -> pd.DataFrame:
    """Compare a successful sensitivity default with established T3 artifacts."""
    metrics_path = Path(t3_variant_metrics_path)
    tracks_path = Path(t3_tracks_long_path)
    missing = [str(path) for path in (metrics_path, tracks_path) if not path.is_file()]
    if missing:
        message = "Default T3 reproducibility artifacts are missing: " + ", ".join(missing)
        warnings.warn(message)
        if warning_path is not None:
            Path(warning_path).write_text(message + "\n", encoding="utf-8")
        return pd.DataFrame(
            [
                {
                    "metric": "warning",
                    "tracking_ablation_T3": np.nan,
                    "sensitivity_default_T3": np.nan,
                    "absolute_difference": np.nan,
                    "relative_difference": np.nan,
                    "match_status": "warning_missing_artifact",
                }
            ]
        )
    reference_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if str(reference_metrics.get("status", "")).lower() not in {"ok", "success"}:
        raise ValueError("Established T3 variant_metrics.json is not successful")
    if str(sensitivity_metrics.get("run_status", "success")).lower() not in {
        "success",
        "success_with_track_errors",
    }:
        raise ValueError("Sensitivity default T3 metrics are not successful")
    reference_tracks = pd.read_csv(tracks_path, encoding="utf-8-sig")
    reference = {
        **reference_metrics,
        **_tracks_long_reproducibility_stats(reference_tracks),
    }
    sensitivity = {
        **sensitivity_metrics,
        **_tracks_long_reproducibility_stats(sensitivity_tracks_long),
    }
    requested = (
        ("tracks_long_row_count", "tracks_long_row_count"),
        *_REPRODUCIBILITY_METRICS,
    )
    manual_mode = str(gt_format) == "manual_review_gt"
    rows: list[dict[str, Any]] = []
    for label, source in requested:
        reference_value = float(reference[source])
        sensitivity_value = float(sensitivity[source])
        absolute = abs(sensitivity_value - reference_value)
        relative = (
            absolute / abs(reference_value)
            if reference_value != 0
            else (0.0 if absolute == 0 else float("inf"))
        )
        if manual_mode and label in _MANUAL_GT_REPRODUCIBILITY_NOT_COMPARABLE:
            match_status = "not_comparable_gt_changed"
        else:
            match_status = (
                "match"
                if math.isclose(
                    reference_value,
                    sensitivity_value,
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
                else "different"
            )
        rows.append(
            {
                "metric": label,
                "tracking_ablation_T3": reference_value,
                "sensitivity_default_T3": sensitivity_value,
                "absolute_difference": float(absolute),
                "relative_difference": float(relative),
                "match_status": match_status,
            }
        )
    return pd.DataFrame(rows)


default_t3_reproducibility_check = build_default_t3_reproducibility_check


# ---------------------------------------------------------------------------
# Scan normalization
# ---------------------------------------------------------------------------

def scaled_values(
    default: float,
    ratios: Sequence[float],
    *,
    integer: bool = False,
    minimum: Optional[float] = None,
) -> list[float | int]:
    """Scale *default* by *ratios*, optionally round, clamp, and deduplicate."""
    seen: set[float | int] = set()
    ordered: list[float | int] = []
    for ratio in ratios:
        raw = round(float(default) * float(ratio), 9)
        if integer:
            value = int(math.floor(raw + 0.5))
        else:
            value = raw
        if minimum is not None and value < minimum:
            value = type(value)(minimum) if integer else float(minimum)
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    default_out: float | int
    if integer:
        default_out = int(math.floor(round(float(default), 9) + 0.5))
    else:
        default_out = float(default)
    if minimum is not None and default_out < minimum:
        default_out = int(minimum) if integer else float(minimum)
    if default_out not in seen:
        ordered.append(default_out)
        ordered.sort()
        seen = set()
        deduped: list[float | int] = []
        for item in ordered:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped
    return ordered


def _ratio_to_default(value: Any, default_value: Any) -> Optional[float]:
    if default_value in (None, 0, False, ""):
        return None
    try:
        return float(value) / float(default_value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def make_config_id(
    group: str,
    parameter_name: str,
    overrides: Mapping[str, Any],
) -> str:
    payload = {
        "group": group,
        "parameter_name": parameter_name,
        "overrides": {key: overrides[key] for key in sorted(overrides)},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    safe_name = parameter_name.replace("/", "_")
    return f"{group}__{safe_name}__{digest}"


def _rtv_alpha_overrides(gmm_default: Any, alpha: float) -> dict[str, float]:
    return {
        field: round(float(getattr(gmm_default, field)) * float(alpha), 12)
        for field in RTV_THRESHOLD_FIELDS
    }


def _importable_module_name(resolved: Path, role: str) -> str:
    """Build a valid, re-importable module name for ProcessPool pickling."""
    stem = "".join(
        ch if (ch.isalnum() or ch == "_") else "_" for ch in resolved.stem
    )
    if not stem:
        stem = "module"
    if stem[0].isdigit():
        stem = f"m_{stem}"
    # Prefer the file stem so Windows spawn workers can import it from sys.path.
    existing = sys.modules.get(stem)
    if existing is None:
        return stem
    existing_file = getattr(existing, "__file__", None)
    if existing_file and Path(existing_file).resolve() == resolved:
        return stem
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    role_safe = "".join(
        ch if (ch.isalnum() or ch == "_") else "_" for ch in str(role)
    ).strip("_") or "mod"
    return f"_parameter_sensitivity_{role_safe}_{digest}"


def _load_module(module_path: str, role: str) -> ModuleType:
    path = Path(module_path)
    if path.suffix == ".py" or path.exists():
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{role} module not found: {module_path}")
        parent = str(resolved.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        module_name = _importable_module_name(resolved, role)
        existing = sys.modules.get(module_name)
        if existing is not None:
            existing_file = getattr(existing, "__file__", None)
            if existing_file and Path(existing_file).resolve() == resolved:
                return existing
        spec = importlib.util.spec_from_file_location(module_name, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {role} module: {resolved}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_path)


def load_tracking_ablation_module(path: str | Path) -> ModuleType:
    """Load the revised formal tracking-ablation module and verify its API."""
    module = _load_module(str(path), "tracking_ablation")
    missing = [
        name
        for name in TRACKING_ABLATION_REQUIRED_APIS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise AttributeError(
            "Tracking ablation module missing required APIs: "
            + ", ".join(missing)
        )
    return module


def _signature_default(callable_obj: Any, name: str) -> Any:
    parameter = inspect.signature(callable_obj).parameters.get(name)
    if parameter is None or parameter.default is inspect.Parameter.empty:
        raise ValueError(f"Formal callable has no default for {name!r}")
    return parameter.default


def load_formal_defaults(
    tracking_module_path: str = DEFAULT_TRACKING_MODULE_PATH,
    ignition_module_path: str = DEFAULT_IGNITION_MODULE_PATH,
) -> tuple[TrackerSensitivityConfig, Any]:
    """Load formal modules and derive immutable T3/default snapshots."""
    tracker_mod = _load_module(tracking_module_path, "tracker")
    ignition_mod = _load_module(ignition_module_path, "ignition")
    tracker_init = tracker_mod.OTKalmanTracker.__init__
    postprocess = tracker_mod.postprocess_tracks
    prebirth = tracker_mod.recover_prebirth_dark_particles
    tracker_defaults = TrackerSensitivityConfig(
        enable_rescue=True,
        enable_gap_fill=True,
        enable_prebirth=True,
        # The confirmed T3 baseline overrides any standalone CLI/module value.
        max_center_dist=45.0,
        ot_reg=float(_signature_default(tracker_init, "ot_reg")),
        dummy_cost=float(_signature_default(tracker_init, "dummy_cost")),
        rescue_center_factor=float(
            _signature_default(tracker_init, "rescue_center_factor")
        ),
        rescue_maha_factor=float(
            _signature_default(tracker_init, "rescue_maha_factor")
        ),
        max_gap_fill=int(_signature_default(postprocess, "max_gap_fill")),
        prebirth_max_backward_frames=int(
            _signature_default(prebirth, "max_backward_frames")
        ),
        prebirth_search_radius_base=float(
            _signature_default(prebirth, "search_radius_base")
        ),
    )
    gmm_defaults = ignition_mod.IgnitionGMMConfig()
    if not is_dataclass(gmm_defaults):
        raise TypeError("Formal IgnitionGMMConfig must be a dataclass")
    missing_thresholds = [
        name for name in RTV_THRESHOLD_FIELDS if not hasattr(gmm_defaults, name)
    ]
    if missing_thresholds:
        raise ValueError(
            f"Formal IgnitionGMMConfig lacks RTV fields: {missing_thresholds}"
        )
    return tracker_defaults, gmm_defaults


# ---------------------------------------------------------------------------
# Parameter registry
# ---------------------------------------------------------------------------

def _values_with_default(values: Sequence[Any], default: Any) -> tuple[Any, ...]:
    ordered = list(values)
    if default not in ordered:
        ordered.append(default)
        try:
            ordered.sort()
        except TypeError:
            pass
    return tuple(ordered)


def build_parameter_registry(
    tracker_default: Optional[TrackerSensitivityConfig] = None,
    gmm_default: Any = None,
    *,
    tracking_module_path: str = DEFAULT_TRACKING_MODULE_PATH,
    ignition_module_path: str = DEFAULT_IGNITION_MODULE_PATH,
) -> list[ParameterSpec]:
    if tracker_default is None or gmm_default is None:
        loaded_tracker, loaded_gmm = load_formal_defaults(
            tracking_module_path, ignition_module_path
        )
        tracker_default = tracker_default or loaded_tracker
        gmm_default = gmm_default or loaded_gmm
    tracker_defaults = tracker_default
    gmm_defaults = gmm_default
    specs: list[ParameterSpec] = []

    # OT association
    specs.append(
        ParameterSpec(
            group="ot",
            name="ot_reg",
            fields=("ot_reg",),
            default_value=tracker_defaults.ot_reg,
            values=(
                (0.04, 0.06, 0.08, 0.10, 0.12)
                if tracker_defaults.ot_reg == 0.08
                else tuple(
                    scaled_values(
                        tracker_defaults.ot_reg,
                        [0.5, 0.75, 1.0, 1.25, 1.5],
                    )
                )
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="ot",
            name="max_center_dist",
            fields=("max_center_dist",),
            default_value=tracker_defaults.max_center_dist,
            values=tuple(
                scaled_values(
                    tracker_defaults.max_center_dist,
                    [0.70, 0.85, 1.00, 1.15, 1.30],
                    integer=True,
                    minimum=1,
                )
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="ot",
            name="dummy_cost",
            fields=("dummy_cost",),
            default_value=tracker_defaults.dummy_cost,
            values=tuple(
                scaled_values(
                    tracker_defaults.dummy_cost,
                    [0.80, 0.90, 1.00, 1.10, 1.20],
                )
            ),
            coupled_note=(
                "Birth and death share one formal dummy_cost field; "
                "scanning unmatched track/detection cost only."
            ),
        )
    )

    # Trajectory repair
    specs.append(
        ParameterSpec(
            group="repair",
            name="rescue_center_factor",
            fields=("rescue_center_factor",),
            default_value=tracker_defaults.rescue_center_factor,
            values=tuple(
                scaled_values(
                    tracker_defaults.rescue_center_factor,
                    [0.70, 0.85, 1.00, 1.15, 1.30],
                )
            ),
            coupled_note=(
                "Only the principal rescue center gate is scanned; "
                "rescue_maha_factor remains at the formal default."
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="repair",
            name="max_gap_fill",
            fields=("max_gap_fill",),
            default_value=tracker_defaults.max_gap_fill,
            values=(
                (5, 8, 10, 12, 15)
                if tracker_defaults.max_gap_fill == 10
                else tuple(
                    scaled_values(
                        tracker_defaults.max_gap_fill,
                        [0.5, 0.8, 1.0, 1.2, 1.5],
                        integer=True,
                        minimum=1,
                    )
                )
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="repair",
            name="prebirth_max_backward_frames",
            fields=("prebirth_max_backward_frames",),
            default_value=tracker_defaults.prebirth_max_backward_frames,
            values=tuple(
                scaled_values(
                    tracker_defaults.prebirth_max_backward_frames,
                    [0.60, 0.80, 1.00, 1.20, 1.40],
                    integer=True,
                    minimum=1,
                )
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="repair",
            name="prebirth_search_radius_base",
            fields=("prebirth_search_radius_base",),
            default_value=tracker_defaults.prebirth_search_radius_base,
            values=tuple(
                scaled_values(
                    tracker_defaults.prebirth_search_radius_base,
                    [0.70, 0.85, 1.00, 1.15, 1.30],
                )
            ),
        )
    )

    # GMM temporal-state
    specs.append(
        ParameterSpec(
            group="gmm",
            name="gmm_max_states",
            fields=("gmm_max_states",),
            default_value=gmm_defaults.gmm_max_states,
            values=_values_with_default(
                (2, 3, 4), gmm_defaults.gmm_max_states
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="gmm",
            name="gmm_bic_improvement_ratio",
            fields=("gmm_bic_improvement_ratio",),
            default_value=gmm_defaults.gmm_bic_improvement_ratio,
            values=_values_with_default(
                (0.00, 0.015, 0.03, 0.05, 0.08),
                gmm_defaults.gmm_bic_improvement_ratio,
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="gmm",
            name="gmm_min_state_fraction",
            fields=("gmm_min_state_fraction",),
            default_value=gmm_defaults.gmm_min_state_fraction,
            values=_values_with_default(
                (0.04, 0.06, 0.08, 0.10, 0.12),
                gmm_defaults.gmm_min_state_fraction,
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="gmm",
            name="gmm_min_state_count",
            fields=("gmm_min_state_count",),
            default_value=gmm_defaults.gmm_min_state_count,
            values=_values_with_default(
                (1, 2, 3, 4), gmm_defaults.gmm_min_state_count
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="gmm",
            name="min_state_run",
            fields=("min_state_run",),
            default_value=gmm_defaults.min_state_run,
            values=_values_with_default(
                (1, 2, 3, 4), gmm_defaults.min_state_run
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="gmm",
            name="cold_hot_persistence",
            fields=("min_cold_run", "min_hot_run"),
            default_value=gmm_defaults.min_cold_run,
            values=_values_with_default(
                (2, 3, 4, 5), gmm_defaults.min_cold_run
            ),
            coupled_note=(
                "Symmetric persistence: min_cold_run = min_hot_run = L. "
                "This also changes RTV before/after averaging windows."
            ),
        )
    )

    # RTV threshold
    specs.append(
        ParameterSpec(
            group="rtv",
            name="rtv_alpha",
            fields=RTV_THRESHOLD_FIELDS,
            default_value=1.0,
            values=(0.60, 0.80, 1.00, 1.20, 1.40),
            coupled_note=(
                "Unified alpha scales all five formal min_raw_delta_* thresholds."
            ),
        )
    )
    specs.append(
        ParameterSpec(
            group="rtv",
            name="rtv_voting_rule",
            fields=("rtv_minimum_votes",),
            default_value="any_1_of_5",
            values=("any_1_of_5", "at_least_2_of_5", "all_5"),
            coupled_note=(
                "Voting comparisons run at alpha_RTV=1.0 using formal raw deltas; "
                "wrapper changes only the minimum passing feature count."
            ),
        )
    )

    return specs


def _build_overrides(
    spec: ParameterSpec, value: Any, gmm_default: Any
) -> dict[str, Any]:
    if spec.name == "rtv_alpha":
        return _rtv_alpha_overrides(gmm_default, float(value))
    if spec.name == "cold_hot_persistence":
        length = int(value)
        return {"min_cold_run": length, "min_hot_run": length}
    if spec.name == "rtv_voting_rule":
        vote_map = {
            "any_1_of_5": 1,
            "at_least_2_of_5": 2,
            "all_5": 5,
        }
        return {"rtv_minimum_votes": vote_map[str(value)]}
    if len(spec.fields) == 1:
        return {spec.fields[0]: value}
    raise ValueError(f"Unsupported parameter spec: {spec.name}")


def dataclass_diff(before: Any, after: Any) -> dict[str, tuple[Any, Any]]:
    """Return changed dataclass fields as ``name: (before, after)``."""
    if not is_dataclass(before) or not is_dataclass(after):
        raise TypeError("dataclass_diff requires dataclass instances")
    if type(before) is not type(after):
        raise TypeError("dataclass_diff inputs must have the same type")
    return {
        item.name: (getattr(before, item.name), getattr(after, item.name))
        for item in fields(before)
        if getattr(before, item.name) != getattr(after, item.name)
    }


def _apply_dataclass_experiment(
    default_cfg: Any,
    experiment: ExperimentSpec,
    *,
    allowed_adapter_fields: frozenset[str] = frozenset(),
) -> Any:
    dataclass_fields = {item.name for item in fields(default_cfg)}
    overrides = dict(experiment.overrides)
    unknown = set(overrides) - dataclass_fields - allowed_adapter_fields
    if unknown:
        raise ValueError(
            f"{experiment.parameter_name} has invalid overrides: {sorted(unknown)}"
        )
    config_overrides = {
        name: value for name, value in overrides.items() if name in dataclass_fields
    }
    applied = replace(default_cfg, **config_overrides)
    expected_changed = {
        name
        for name, value in config_overrides.items()
        if getattr(default_cfg, name) != value
    }
    actual_changed = set(dataclass_diff(default_cfg, applied))
    if actual_changed != expected_changed:
        raise AssertionError(
            f"OFAT diff mismatch for {experiment.config_id}: "
            f"expected {sorted(expected_changed)}, got {sorted(actual_changed)}"
        )
    return applied


def apply_tracker_experiment(
    default_cfg: TrackerSensitivityConfig,
    experiment: ExperimentSpec,
) -> TrackerSensitivityConfig:
    """Apply one OT/repair experiment to a fresh tracker default snapshot."""
    if experiment.group not in {"ot", "repair"}:
        raise ValueError(f"Not a tracker experiment: {experiment.group}")
    return _apply_dataclass_experiment(default_cfg, experiment)


def apply_gmm_experiment(default_cfg: Any, experiment: ExperimentSpec) -> Any:
    """Apply one GMM/RTV experiment, keeping vote adapter fields separate."""
    if experiment.group not in {"gmm", "rtv"}:
        raise ValueError(f"Not a GMM/RTV experiment: {experiment.group}")
    return _apply_dataclass_experiment(
        default_cfg,
        experiment,
        allowed_adapter_fields=frozenset({"rtv_minimum_votes"}),
    )


def _tracking_context_hash(
    reference_tracks: pd.DataFrame,
    reference_ignition: pd.DataFrame,
    listed_no_ignition_ids: Sequence[int],
    onset_tolerance: int,
    annotations: Mapping[str, Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for label, frame in (
        ("reference_tracks", reference_tracks),
        ("reference_ignition", reference_ignition),
    ):
        digest.update(label.encode("utf-8"))
        digest.update(
            json.dumps(
                {
                    "columns": [str(column) for column in frame.columns],
                    "dtypes": [str(dtype) for dtype in frame.dtypes],
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(
            pd.util.hash_pandas_object(frame, index=True, categorize=True)
            .to_numpy(dtype=np.uint64)
            .tobytes()
        )
    digest.update(
        json.dumps(
            {
                "listed_no_ignition_ids": [int(value) for value in listed_no_ignition_ids],
                "onset_tolerance": int(onset_tolerance),
                "annotations": annotations,
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def build_fixed_tracking_context(
    reference_tracks: pd.DataFrame,
    reference_ignition: pd.DataFrame,
    listed_no_ignition_ids: Sequence[int],
    onset_tolerance: int = 2,
    *,
    tracking_ablation_mod: Any,
    prebuilt_annotations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> FixedTrackingContext:
    """Build formal effective_trace_new GT once for reuse across the OFAT sweep."""
    if not isinstance(reference_tracks, pd.DataFrame):
        raise TypeError("reference_tracks must be a preloaded DataFrame")
    if not isinstance(reference_ignition, pd.DataFrame):
        raise TypeError("reference_ignition must be a preloaded DataFrame")
    listed_ids = tuple(int(value) for value in listed_no_ignition_ids)
    if prebuilt_annotations is None:
        annotations = (
            tracking_ablation_mod.build_gt_from_effective_trace_new_and_ref_ignition(
                list(listed_ids),
                reference_ignition,
                onset_tolerance=int(onset_tolerance),
            )
        )
    else:
        annotations = canonicalize_annotation_mapping(
            copy.deepcopy(dict(prebuilt_annotations))
        )
    content_hash = _tracking_context_hash(
        reference_tracks,
        reference_ignition,
        listed_ids,
        int(onset_tolerance),
        annotations,
    )
    return FixedTrackingContext(
        reference_tracks=reference_tracks,
        reference_ignition=reference_ignition,
        listed_no_ignition_ids=listed_ids,
        onset_tolerance=int(onset_tolerance),
        annotations=annotations,
        content_hash=content_hash,
    )


def _validate_fixed_tracking_context(context: FixedTrackingContext) -> None:
    if not isinstance(context, FixedTrackingContext):
        raise TypeError("fixed_context must be a preloaded FixedTrackingContext")
    actual_hash = _tracking_context_hash(
        context.reference_tracks,
        context.reference_ignition,
        context.listed_no_ignition_ids,
        context.onset_tolerance,
        context.annotations,
    )
    if actual_hash != context.content_hash:
        raise ValueError("FixedTrackingContext inputs changed after GT construction")


def _require_module_apis(module: Any, names: Sequence[str]) -> None:
    missing = [name for name in names if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(
            "Formal tracker module missing required APIs: " + ", ".join(missing)
        )


def run_formal_tracker(
    tracker_mod: Any,
    detections: Mapping[int, Sequence[Mapping[str, Any]]],
    cfg: TrackerSensitivityConfig,
    frame_map: Mapping[int, Path],
) -> TrackingOutput:
    """Run the formal T3 tracker, gap fill, and prebirth recovery APIs."""
    if not cfg.enable_rescue:
        raise ValueError(
            "Formal OTKalmanTracker always uses rescue matching; "
            "algorithm substitution is not allowed"
        )
    _require_module_apis(
        tracker_mod,
        (
            "OTKalmanTracker",
            "kf_params_from_tracker",
            "postprocess_tracks",
            "recover_prebirth_dark_particles",
        ),
    )

    tracker = tracker_mod.OTKalmanTracker(
        max_age=4,
        max_age_tentative=1,
        max_age_confirmed=5,
        min_hits=3,
        min_contiguous_observed=3,
        max_center_dist=float(cfg.max_center_dist),
        max_mahalanobis=12.0,
        ot_reg=float(cfg.ot_reg),
        ot_iters=100,
        dummy_cost=float(cfg.dummy_cost),
        min_ot_prob=0.15,
        max_match_cost=0.80,
        w_maha=0.45,
        w_center=0.45,
        w_iou=0.15,
        w_size=0.15,
        w_direction=0.05,
        min_direction_speed=1.0,
        kf_process_pos_std=2.0,
        kf_process_vel_std=12.0,
        kf_process_size_std=2.0,
        kf_meas_pos_std=5.0,
        kf_meas_size_std=12.0,
        min_det_score=0.0,
        image_width=None,
        image_height=None,
        border_margin=20.0,
        rescue_center_factor=float(cfg.rescue_center_factor),
        rescue_maha_factor=float(cfg.rescue_maha_factor),
    )
    stats: dict[str, Any] = {
        "n_rescue_matches": 0,
        "n_rescue_candidates": 0,
        "gap_filled_points": 0,
        "prebirth_points_added": 0,
    }
    frame_ids = {int(frame_id) for frame_id in detections}
    frame_ids.update(int(frame_id) for frame_id in frame_map)
    if frame_ids:
        for frame_id in range(min(frame_ids), max(frame_ids) + 1):
            frame_detections = detections.get(frame_id, [])
            step_stats = tracker.step(frame_id, frame_detections)
            stats["n_rescue_matches"] += int(
                (step_stats or {}).get("rescue_matches", 0)
            )
            stats["n_rescue_candidates"] += int(
                (step_stats or {}).get("rescue_candidates", 0)
            )

    tracks = tracker.finish()
    tracks, postprocess_stats = tracker_mod.postprocess_tracks(
        tracks,
        kf_params=tracker_mod.kf_params_from_tracker(tracker),
        enable_gap_fill=bool(cfg.enable_gap_fill),
        max_gap_fill=int(cfg.max_gap_fill),
        min_contiguous_observed=3,
        min_observed_hits=3,
    )
    stats.update(
        {
            f"postprocess_{name}": value
            for name, value in (postprocess_stats or {}).items()
        }
    )
    gap_points = int((postprocess_stats or {}).get("gap_filled_points", 0))
    stats["gap_filled_points"] = gap_points

    if cfg.enable_prebirth and frame_map:
        tracks, prebirth_stats = tracker_mod.recover_prebirth_dark_particles(
            tracks,
            frame_image_map=frame_map,
            max_backward_frames=int(cfg.prebirth_max_backward_frames),
            search_radius_base=float(cfg.prebirth_search_radius_base),
        )
        stats.update(
            {
                f"prebirth_{name}": value
                for name, value in (prebirth_stats or {}).items()
            }
        )
        stats["prebirth_points_added"] = int(
            (prebirth_stats or {}).get("prebirth_points_added", 0)
        )
    return TrackingOutput(tracks=list(tracks), stats=stats)


def _tracking_track_id(raw: Any, tracking_ablation_mod: Any) -> int:
    normalized = str(tracking_ablation_mod.normalize_track_name(raw)).strip()
    if normalized.startswith("track_"):
        normalized = normalized.split("_", 1)[1]
    return int(float(normalized))


def _build_tracking_evaluation_rows(
    annotations: Mapping[str, Mapping[str, Any]],
    projected_ignition: pd.DataFrame,
    query_ignition: pd.DataFrame,
    query_to_ref: Mapping[int, int],
    *,
    tracking_ablation_mod: Any,
    evaluation_identity_aliases: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    projected_by_ref = {
        _tracking_track_id(row["track_id"], tracking_ablation_mod): row
        for _, row in projected_ignition.iterrows()
    } if not projected_ignition.empty else {}
    query_by_ref = {
        _tracking_track_id(ref_id, tracking_ablation_mod):
        _tracking_track_id(query_id, tracking_ablation_mod)
        for query_id, ref_id in query_to_ref.items()
    }
    rows: list[dict[str, Any]] = []
    aliases = (
        {str(key): str(value) for key, value in evaluation_identity_aliases.items()}
        if evaluation_identity_aliases is not None
        else None
    )
    if aliases is not None:
        missing = sorted(set(str(name) for name in annotations) - set(aliases))
        if missing:
            raise ValueError(
                "Missing evaluation identity aliases for annotation keys: "
                f"{missing}"
            )
        source_values = [aliases[str(name)] for name in annotations]
        if len(source_values) != len(set(source_values)):
            raise ValueError("duplicate evaluation identity alias targets")

    for track_name, gt in annotations.items():
        ref_id = _tracking_track_id(track_name, tracking_ablation_mod)
        prediction = projected_by_ref.get(ref_id)
        pred_has = (
            tracking_ablation_mod.prediction_has_ignition(prediction)
            if prediction is not None
            else False
        )
        pred_onset = (
            int(prediction.get("ignition_onset_frame", -1))
            if prediction is not None
            else -1
        )
        gt_has = bool(gt.get("gt_has_ignition", False))
        gt_onset = int(gt.get("gt_onset_frame", -1))
        onset_error: Any = np.nan
        within_1: Any = np.nan
        within_3: Any = np.nan
        inside_interval: Any = np.nan
        if gt_has and pred_has and gt_onset >= 0 and pred_onset >= 0:
            onset_error = float(abs(pred_onset - gt_onset))
            within_1 = bool(onset_error <= 1.0)
            within_3 = bool(onset_error <= 3.0)
            gt_start = int(gt.get("gt_onset_start", -1))
            gt_end = int(gt.get("gt_onset_end", -1))
            if gt_start >= 0 and gt_end >= gt_start:
                inside_interval = bool(gt_start <= pred_onset <= gt_end)
        is_matched = ref_id in query_by_ref
        source_track_name = (
            aliases[str(track_name)] if aliases is not None else str(track_name)
        )
        rows.append(
            {
                "track_name": str(track_name),
                "track_id": ref_id,
                "reference_track_id": ref_id,
                "source_track_name": source_track_name,
                "query_track_id": query_by_ref.get(ref_id, np.nan),
                "row_type": "reference_track",
                "sample_weight": 1.0,
                "match_status": "matched" if is_matched else "unmatched_reference",
                "is_matched": is_matched,
                "gt_category": str(gt.get("gt_category", "")),
                "gt_has_ignition": gt_has,
                "gt_onset_frame": gt_onset,
                "gt_onset_start": int(gt.get("gt_onset_start", -1)),
                "gt_onset_end": int(gt.get("gt_onset_end", -1)),
                "pred_has_ignition": pred_has,
                "pred_onset_frame": pred_onset,
                "pred_ignition_status": (
                    str(prediction.get("ignition_status", ""))
                    if prediction is not None
                    else "unmatched"
                ),
                "onset_abs_error": onset_error,
                "onset_within_1": within_1,
                "onset_within_3": within_3,
                "within_1_frame": within_1,
                "within_3_frame": within_3,
                "onset_inside_interval": inside_interval,
            }
        )

    unmatched_predictions: list[pd.Series] = []
    for _, prediction in query_ignition.iterrows():
        query_id = _tracking_track_id(
            prediction["track_id"], tracking_ablation_mod
        )
        if (
            query_id in query_by_ref.values()
            or not tracking_ablation_mod.prediction_has_ignition(prediction)
        ):
            continue
        unmatched_predictions.append(prediction)
    unmatched_count = len(unmatched_predictions)
    rows.append(
        {
            "track_name": UNMATCHED_PREDICTION_POOL_KEY,
            "track_id": np.nan,
            "reference_track_id": np.nan,
            "source_track_name": UNMATCHED_PREDICTION_POOL_KEY,
            "query_track_id": np.nan,
            "row_type": "unmatched_prediction_pool",
            "sample_weight": float(unmatched_count),
            "match_status": "unmatched_prediction_pool",
            "is_matched": False,
            "gt_category": "unmatched_prediction_pool",
            "gt_has_ignition": False,
            "gt_onset_frame": -1,
            "gt_onset_start": -1,
            "gt_onset_end": -1,
            "pred_has_ignition": bool(unmatched_count),
            "pred_onset_frame": -1,
            "pred_ignition_status": (
                "unmatched_prediction_pool"
                if unmatched_count
                else "no_unmatched_predictions"
            ),
            "onset_abs_error": np.nan,
            "onset_within_1": np.nan,
            "onset_within_3": np.nan,
            "within_1_frame": np.nan,
            "within_3_frame": np.nan,
            "onset_inside_interval": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _unmatched_prediction_diagnostics(
    query_ignition: pd.DataFrame,
    query_to_ref: Mapping[int, int],
    *,
    tracking_ablation_mod: Any,
) -> pd.DataFrame:
    matched_query_ids = {
        _tracking_track_id(query_id, tracking_ablation_mod)
        for query_id in query_to_ref
    }
    rows: list[dict[str, Any]] = []
    for _, prediction in query_ignition.iterrows():
        query_id = _tracking_track_id(
            prediction["track_id"], tracking_ablation_mod
        )
        if (
            query_id in matched_query_ids
            or not tracking_ablation_mod.prediction_has_ignition(prediction)
        ):
            continue
        rows.append(
            {
                "diagnostic_type": "unmatched_predicted_ignition",
                "query_track_id": query_id,
                "ref_track_id": np.nan,
                "ignition_onset_frame": int(
                    prediction.get("ignition_onset_frame", -1)
                ),
                "ignition_status": str(
                    prediction.get("ignition_status", "")
                ),
            }
        )
    return pd.DataFrame(rows)


def _unavailable_t3_reproducibility_rows() -> pd.DataFrame:
    requested = (
        ("tracks_long_row_count", "tracks_long_row_count"),
        *_REPRODUCIBILITY_METRICS,
    )
    return pd.DataFrame(
        [
            {
                "metric": label,
                "tracking_ablation_T3": np.nan,
                "sensitivity_default_T3": np.nan,
                "absolute_difference": np.nan,
                "relative_difference": np.nan,
                "match_status": "unavailable",
            }
            for label, _source in requested
        ]
    )


def _persist_tracking_stage_error(
    diagnostic_state: Mapping[str, Any],
    error: Exception,
) -> None:
    run_dir = Path(diagnostic_state.get("run_dir", "."))
    run_dir.mkdir(parents=True, exist_ok=True)
    stage = str(diagnostic_state.get("stage", "unknown"))
    _atomic_write_json(
        run_dir / "alignment_stage_error.json",
        {
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    )

    reference_validation = dict(
        diagnostic_state.get("reference_validation") or {}
    )
    if reference_validation:
        _atomic_write_json(
            run_dir / "reference_validation.json", reference_validation
        )
        pd.DataFrame(
            _flatten_diagnostic_values(
                "reference_validation", reference_validation
            )
        ).to_csv(
            run_dir / "reference_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )
    for name, state_key in (
        ("frame_alignment_diagnostics", "frame_diagnostics"),
        ("coordinate_alignment_diagnostics", "coordinate_diagnostics"),
        ("identity_diagnostics", "identity_diagnostics"),
    ):
        diagnostics = diagnostic_state.get(state_key)
        if diagnostics is None:
            continue
        _atomic_write_json(run_dir / f"{name}.json", diagnostics)
        pd.DataFrame(
            _flatten_diagnostic_values(name, diagnostics)
        ).to_csv(
            run_dir / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    candidates = diagnostic_state.get("candidate_diagnostics")
    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.DataFrame(columns=_ALIGNMENT_CANDIDATE_COLUMNS)
    candidates.to_csv(
        run_dir / "alignment_candidate_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rejection = diagnostic_state.get("rejection_summary")
    if not isinstance(rejection, pd.DataFrame):
        rejection = pd.DataFrame(
            [
                {"reject_reason": reason, "count": 0}
                for reason in _ALIGNMENT_REJECTION_REASONS
            ]
        )
    rejection.to_csv(
        run_dir / "alignment_rejection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mapping_sample = diagnostic_state.get("mapping_sample")
    if not isinstance(mapping_sample, pd.DataFrame):
        mapping_sample = pd.DataFrame(
            columns=[
                "query_track_id",
                "reference_track_id",
                "shared_frames",
                "mean_center_distance",
                "query_coverage",
                "reference_coverage",
                "temporal_iou",
                "total_match_cost",
                "diagnostic_warning",
            ]
        )
    mapping_sample.to_csv(
        run_dir / "mapping_sample_first20.csv",
        index=False,
        encoding="utf-8-sig",
    )

    query_tracks = diagnostic_state.get("query_tracks")
    reference_tracks = diagnostic_state.get("reference_tracks")
    query_ignition = diagnostic_state.get("query_ignition")
    projected = diagnostic_state.get("projected_ignition")
    try:
        partial_funnel = build_alignment_funnel(
            detections=diagnostic_state.get("detections") or {},
            query_tracks=(
                query_tracks
                if isinstance(query_tracks, pd.DataFrame)
                else pd.DataFrame()
            ),
            crop_data=diagnostic_state.get("crop_data") or {},
            query_ignition=(
                query_ignition
                if isinstance(query_ignition, pd.DataFrame)
                else pd.DataFrame()
            ),
            reference_tracks=(
                reference_tracks
                if isinstance(reference_tracks, pd.DataFrame)
                else pd.DataFrame()
            ),
            annotations=diagnostic_state.get("annotations") or {},
            candidate_diagnostics=candidates,
            query_to_ref=diagnostic_state.get("query_to_ref") or {},
            projected_ignition=(
                projected
                if isinstance(projected, pd.DataFrame)
                else pd.DataFrame()
            ),
            evaluation_metrics=diagnostic_state.get("evaluation_metrics") or {},
            row_metrics=diagnostic_state.get("row_metrics") or {},
        )
    except Exception as funnel_error:
        partial_funnel = {
            "partial": True,
            "available_through_stage": stage,
            "funnel_error_type": type(funnel_error).__name__,
            "funnel_error_message": str(funnel_error),
        }
    partial_funnel["partial"] = True
    partial_funnel["available_through_stage"] = stage
    _atomic_write_json(run_dir / "alignment_funnel.json", partial_funnel)
    pd.DataFrame([partial_funnel]).to_csv(
        run_dir / "alignment_funnel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    experiment = diagnostic_state.get("experiment")
    is_default = bool(
        getattr(experiment, "is_default_configuration", False)
    )
    reproducibility_path = (
        run_dir / "default_t3_reproducibility_check.csv"
    )
    if is_default and not reproducibility_path.is_file():
        _unavailable_t3_reproducibility_rows().to_csv(
            reproducibility_path,
            index=False,
            encoding="utf-8-sig",
        )


def run_tracking_experiment(
    *,
    tracker_mod: Any,
    tracking_ablation: Any,
    detections: Mapping[int, Sequence[Mapping[str, Any]]],
    frame_map: Mapping[int, Path],
    image_name_map: Mapping[str, Path],
    experiment: ExperimentSpec,
    tracker_default: TrackerSensitivityConfig,
    fixed_context: FixedTrackingContext,
    output_dir: str | Path,
    ignition_mod: Any,
    gmm_rtv_cfg: Any,
    ignition_module_path: str | Path,
    workers: int = 1,
    crop_padding: float = 4.0,
    crop_format: str = "png",
    jpeg_quality: int = 90,
    save_plot: bool = False,
    skip_existing_crops: bool = False,
    reference_validation: Optional[Mapping[str, Any]] = None,
    t3_reproducibility_dir: Optional[str | Path] = None,
    evaluation_identity_aliases: Optional[Mapping[str, str]] = None,
    gt_format: str = "effective_trace_new",
) -> RunResult:
    """Run tracking while preserving partial alignment diagnostics on failure."""
    diagnostic_state: dict[str, Any] = {
        "stage": "initialization",
        "run_dir": Path(output_dir),
        "experiment": experiment,
        "detections": detections,
        "reference_validation": dict(reference_validation or {}),
        "t3_reproducibility_dir": t3_reproducibility_dir,
        "gt_format": str(gt_format),
    }
    try:
        return _run_tracking_experiment_impl(
            tracker_mod=tracker_mod,
            tracking_ablation=tracking_ablation,
            detections=detections,
            frame_map=frame_map,
            image_name_map=image_name_map,
            experiment=experiment,
            tracker_default=tracker_default,
            fixed_context=fixed_context,
            output_dir=output_dir,
            ignition_mod=ignition_mod,
            gmm_rtv_cfg=gmm_rtv_cfg,
            ignition_module_path=ignition_module_path,
            workers=workers,
            crop_padding=crop_padding,
            crop_format=crop_format,
            jpeg_quality=jpeg_quality,
            save_plot=save_plot,
            skip_existing_crops=skip_existing_crops,
            reference_validation=reference_validation,
            t3_reproducibility_dir=t3_reproducibility_dir,
            diagnostic_state=diagnostic_state,
            evaluation_identity_aliases=evaluation_identity_aliases,
            gt_format=gt_format,
        )
    except Exception as exc:
        _persist_tracking_stage_error(diagnostic_state, exc)
        if experiment.is_default_configuration:
            if isinstance(exc, TrackingAlignmentError):
                raise
            wrapped = TrackingAlignmentError(
                f"Tracking alignment stage {diagnostic_state.get('stage')} "
                f"failed: {exc}"
            )
            consistency = getattr(exc, "consistency_diagnostics", None)
            if consistency is not None:
                wrapped.consistency_diagnostics = consistency  # type: ignore[attr-defined]
            raise wrapped from exc
        raise


def build_experiments(
    groups: set[str],
    tracker_default: Optional[TrackerSensitivityConfig] = None,
    gmm_default: Any = None,
    *,
    tracking_module_path: str = DEFAULT_TRACKING_MODULE_PATH,
    ignition_module_path: str = DEFAULT_IGNITION_MODULE_PATH,
) -> list[ExperimentSpec]:
    selected = set(groups)
    unknown = selected - VALID_GROUPS - {"all"}
    if unknown:
        raise ValueError(f"Unknown parameter groups: {sorted(unknown)}")
    if "all" in selected:
        selected = set(VALID_GROUPS)
    if tracker_default is None or gmm_default is None:
        loaded_tracker, loaded_gmm = load_formal_defaults(
            tracking_module_path, ignition_module_path
        )
        tracker_default = tracker_default or loaded_tracker
        gmm_default = gmm_default or loaded_gmm

    experiments: list[ExperimentSpec] = []
    for spec in build_parameter_registry(
        tracker_default=tracker_default,
        gmm_default=gmm_default,
    ):
        if spec.group not in selected:
            continue
        if spec.skip_reason:
            overrides = _build_overrides(spec, spec.default_value, gmm_default)
            immutable_overrides = MappingProxyType(dict(overrides))
            experiments.append(
                ExperimentSpec(
                    group=spec.group,
                    parameter_name=spec.name,
                    parameter_value=spec.default_value,
                    default_value=spec.default_value,
                    ratio_to_default=None,
                    is_default_configuration=True,
                    overrides=immutable_overrides,
                    config_id=make_config_id(spec.group, spec.name, overrides),
                    skip_reason=spec.skip_reason,
                )
            )
            continue

        for value in spec.values:
            overrides = _build_overrides(spec, value, gmm_default)
            immutable_overrides = MappingProxyType(dict(overrides))
            is_default = value == spec.default_value
            experiments.append(
                ExperimentSpec(
                    group=spec.group,
                    parameter_name=spec.name,
                    parameter_value=value,
                    default_value=spec.default_value,
                    ratio_to_default=_ratio_to_default(value, spec.default_value),
                    is_default_configuration=is_default,
                    overrides=immutable_overrides,
                    config_id=make_config_id(spec.group, spec.name, overrides),
                )
            )
    for experiment in experiments:
        if experiment.skip_reason:
            continue
        if experiment.group in {"ot", "repair"}:
            apply_tracker_experiment(tracker_default, experiment)
        else:
            apply_gmm_experiment(gmm_default, experiment)
    return experiments


def select_canonical_default_tracking_experiment(
    experiments: Sequence[ExperimentSpec],
    selected_groups: set[str],
) -> ExperimentSpec:
    """Choose one deterministic full-default T3 tracking configuration."""
    groups = set(TRACKING_GROUPS) if "all" in selected_groups else (
        set(selected_groups) & set(TRACKING_GROUPS)
    )
    candidates = [
        experiment
        for experiment in experiments
        if experiment.group in groups
        and experiment.is_default_configuration
        and not experiment.skip_reason
    ]
    if not candidates:
        raise ValueError(
            "--only-default-tracking requires a selected tracking group "
            "with an active default configuration"
        )
    group_order = {"ot": 0, "repair": 1}
    return min(
        candidates,
        key=lambda item: (
            group_order.get(item.group, 99),
            item.parameter_name,
            item.config_id,
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation metrics, bootstrap, deltas, and stable intervals
# ---------------------------------------------------------------------------

def _truthy_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: False if pd.isna(value) else bool(value))


def _error_track_mask(rows: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=rows.index, dtype=bool)
    if "is_error_track" in rows.columns:
        mask |= _truthy_series(rows["is_error_track"])
    if "pred_ignition_status" in rows.columns:
        statuses = rows["pred_ignition_status"].fillna("").astype(str).str.strip()
        mask |= statuses.isin(ERROR_PREDICTION_STATUSES)
    if "error_message" in rows.columns:
        messages = rows["error_message"].fillna("").astype(str).str.strip()
        mask |= messages.ne("")
    if "run_status" in rows.columns:
        statuses = rows["run_status"].fillna("").astype(str).str.strip().str.lower()
        mask |= statuses.eq("error")
    return mask


def _valid_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[~_error_track_mask(rows)].copy()


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _metric_delta(current: Any, reference: Any) -> float:
    if pd.isna(current) or pd.isna(reference):
        return float("nan")
    return float(current) - float(reference)


def _onset_accuracy(
    hits: pd.DataFrame,
    explicit_column: str,
    *,
    threshold: Optional[float] = None,
) -> float:
    has_explicit = explicit_column in hits.columns
    values = (
        hits[explicit_column].copy()
        if has_explicit
        else pd.Series(np.nan, index=hits.index, dtype=object)
    )
    if threshold is not None and "onset_abs_error" in hits.columns:
        errors = pd.to_numeric(hits["onset_abs_error"], errors="coerce")
        values = values.where(values.notna(), errors <= threshold)
        has_measurement = errors.notna()
        if has_explicit:
            has_measurement |= hits[explicit_column].notna()
        values = values.where(has_measurement, np.nan)
    valid = values.dropna()
    if not len(valid):
        return float("nan")
    weights = (
        pd.to_numeric(hits.loc[valid.index, "sample_weight"], errors="coerce")
        if "sample_weight" in hits.columns
        else pd.Series(1.0, index=valid.index)
    )
    denominator = float(weights.sum())
    if denominator <= 0:
        return float("nan")
    return float(
        (_truthy_series(valid).astype(float) * weights).sum() / denominator
    )


def compute_binary_metrics(rows: pd.DataFrame) -> dict[str, float]:
    """Aggregate binary ignition, onset accuracy, and confusion diagnostics."""
    rows = _valid_metric_rows(rows)
    if rows.empty:
        return {
            "ignition_precision": 0.0,
            "ignition_recall": 0.0,
            "ignition_f1": 0.0,
            "false_ignition_rate": float("nan"),
            "missed_ignition_rate": float("nan"),
            "onset_mae": float("nan"),
            "within_1_frame_acc": float("nan"),
            "within_3_frame_acc": float("nan"),
            "inside_interval_acc": float("nan"),
            "true_positive_count": 0.0,
            "false_positive_count": 0.0,
            "true_negative_count": 0.0,
            "false_negative_count": 0.0,
            "false_event_count": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "tn": 0.0,
            "fn": 0.0,
        }

    gt = _truthy_series(rows["gt_has_ignition"])
    pred = _truthy_series(rows["pred_has_ignition"])
    weights = (
        pd.to_numeric(rows["sample_weight"], errors="coerce")
        if "sample_weight" in rows.columns
        else pd.Series(1.0, index=rows.index)
    )
    if weights.isna().any() or (~np.isfinite(weights)).any() or (weights < 0).any():
        raise ValueError("sample_weight must contain finite non-negative values")

    tp = float(weights[pred & gt].sum())
    fp = float(weights[pred & ~gt].sum())
    fn = float(weights[~pred & gt].sum())
    tn = float(weights[~pred & ~gt].sum())

    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0.0

    hits = rows[gt & pred]
    onset_mae = float("nan")
    if not hits.empty and "onset_abs_error" in hits.columns:
        onset_values = pd.to_numeric(hits["onset_abs_error"], errors="coerce")
        valid_onset = onset_values.notna()
        if valid_onset.any():
            onset_weights = weights.loc[hits.index[valid_onset]]
            denominator = float(onset_weights.sum())
            if denominator > 0:
                onset_mae = float(
                    (
                        onset_values.loc[valid_onset]
                        * onset_weights
                    ).sum()
                    / denominator
                )

    return {
        "ignition_precision": precision,
        "ignition_recall": recall,
        "ignition_f1": f1,
        "false_ignition_rate": _safe_rate(fp, float(weights[~gt].sum())),
        "missed_ignition_rate": _safe_rate(fn, float(weights[gt].sum())),
        "onset_mae": onset_mae,
        "within_1_frame_acc": _onset_accuracy(
            hits, "onset_within_1", threshold=1.0
        ),
        "within_3_frame_acc": _onset_accuracy(
            hits, "onset_within_3", threshold=3.0
        ),
        "inside_interval_acc": _onset_accuracy(
            hits, "onset_inside_interval"
        ),
        "true_positive_count": tp,
        "false_positive_count": fp,
        "true_negative_count": tn,
        "false_negative_count": fn,
        "false_event_count": fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def validate_tracking_evaluator_consistency(
    formal_metrics: Mapping[str, Any],
    local_metrics: Mapping[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Require formal aligned metrics and local row recomputation to agree."""
    metric_pairs = (
        ("end_to_end_tp", "tp"),
        ("end_to_end_fp", "fp"),
        ("end_to_end_fn", "fn"),
        ("ignition_precision", "ignition_precision"),
        ("ignition_recall", "ignition_recall"),
        ("ignition_f1", "ignition_f1"),
        ("onset_mae", "onset_mae"),
        ("within_1_frame_acc", "within_1_frame_acc"),
        ("within_3_frame_acc", "within_3_frame_acc"),
    )
    diagnostics: dict[str, Any] = {}
    checked: list[str] = []
    mismatches: list[str] = []
    max_abs_diff = 0.0
    for formal_name, local_name in metric_pairs:
        if formal_name not in formal_metrics or local_name not in local_metrics:
            continue
        formal_value = formal_metrics.get(formal_name)
        local_value = local_metrics.get(local_name)
        formal_number = (
            float("nan")
            if formal_value is None or pd.isna(formal_value)
            else float(formal_value)
        )
        local_number = (
            float("nan")
            if local_value is None or pd.isna(local_value)
            else float(local_value)
        )
        both_nan = math.isnan(formal_number) and math.isnan(local_number)
        matched = both_nan or (
            np.isfinite(formal_number)
            and np.isfinite(local_number)
            and math.isclose(
                formal_number,
                local_number,
                rel_tol=float(rtol),
                abs_tol=float(atol),
            )
        )
        difference = (
            float("nan")
            if both_nan
            else abs(formal_number - local_number)
        )
        if np.isfinite(difference):
            max_abs_diff = max(max_abs_diff, float(difference))
        checked.append(formal_name)
        prefix = f"evaluation_consistency_{formal_name}"
        diagnostics[f"{prefix}_formal"] = formal_number
        diagnostics[f"{prefix}_local"] = local_number
        diagnostics[f"{prefix}_abs_diff"] = difference
        diagnostics[f"{prefix}_match"] = bool(matched)
        if not matched:
            mismatches.append(
                f"{formal_name}: formal={formal_number!r}, "
                f"local={local_number!r}"
            )
    diagnostics.update(
        {
            "evaluation_consistency_status": (
                "consistent" if not mismatches else "mismatch"
            ),
            "evaluation_consistency_checked_metrics": tuple(checked),
            "evaluation_consistency_max_abs_diff": float(max_abs_diff),
            "evaluation_consistency_rtol": float(rtol),
            "evaluation_consistency_atol": float(atol),
        }
    )
    if mismatches:
        error = TrackingAlignmentError(
            "Formal/local tracking evaluation mismatch: "
            + " | ".join(mismatches)
        )
        error.consistency_diagnostics = diagnostics  # type: ignore[attr-defined]
        raise error
    return diagnostics


def _bootstrap_ci_keys(include_paired: bool) -> list[str]:
    keys = [
        f"{metric}_ci_{bound}"
        for metric in BOOTSTRAP_METRIC_KEYS
        for bound in ("low", "high")
    ]
    if include_paired:
        keys.extend(
            [
                "delta_f1_ci_low",
                "delta_f1_ci_high",
                "delta_onset_mae_ci_low",
                "delta_onset_mae_ci_high",
            ]
        )
    return keys


def _nanpercentile_or_nan(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    if not np.isfinite(finite).any():
        return float("nan")
    return float(np.nanpercentile(finite, percentile))


def bootstrap_metrics(
    rows: pd.DataFrame,
    samples: int,
    seed: int,
    paired_default: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """Bootstrap one test-track set; paired inputs are aligned by track identity."""
    if "window_evaluated" in rows and rows["window_evaluated"].notna().any():
        keys = [f"{prefix}{metric}_ci_{side}" for metric in WINDOW_METRICS
                for side in ("low", "high")
                for prefix in (("", "delta_") if paired_default is not None else ("",))]
        intervals = dict.fromkeys(keys, float("nan"))
        current = rows.loc[rows["window_evaluated"].eq(True)].copy()
        if current["track_name"].duplicated().any():
            raise ValueError("Duplicate GT-positive identities in window bootstrap")
        current = current.set_index("track_name").sort_index()
        baseline = None
        if paired_default is not None:
            baseline = paired_default.loc[paired_default["window_evaluated"].eq(True)].copy()
            if baseline["track_name"].duplicated().any():
                raise ValueError("Duplicate default GT-positive identities in window bootstrap")
            baseline = baseline.set_index("track_name").sort_index()
            if not current.index.equals(baseline.index):
                raise ValueError("Window bootstrap requires identical fixed GT-positive sets")
        if samples <= 0 or current.empty:
            return intervals
        rng = np.random.default_rng(seed)
        distributions = {metric: [] for metric in WINDOW_METRICS}
        deltas = {metric: [] for metric in WINDOW_METRICS}
        for _ in range(samples):
            indices = rng.integers(0, len(current), size=len(current))
            for metric, column in WINDOW_METRICS.items():
                values = pd.to_numeric(current[column], errors="coerce").to_numpy()[indices]
                value = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
                distributions[metric].append(value)
                if baseline is not None:
                    default_values = pd.to_numeric(baseline[column], errors="coerce").to_numpy()[indices]
                    default_mean = float(np.nanmean(default_values)) if np.isfinite(default_values).any() else float("nan")
                    deltas[metric].append(value - default_mean)
        for prefix, distribution in (("", distributions), ("delta_", deltas)):
            if prefix and baseline is None:
                continue
            for metric, values in distribution.items():
                finite = np.asarray(values)[np.isfinite(values)]
                if len(finite):
                    low, high = np.quantile(finite, [.025, .975])
                    intervals[f"{prefix}{metric}_ci_low"] = float(low)
                    intervals[f"{prefix}{metric}_ci_high"] = float(high)
        return intervals
    include_paired = paired_default is not None
    nan_result = {key: float("nan") for key in _bootstrap_ci_keys(include_paired)}
    if samples <= 0:
        return nan_result
    if paired_default is not None:
        rows, paired_default = _align_paired_track_rows(rows, paired_default)
        valid_mask = ~(_error_track_mask(rows) | _error_track_mask(paired_default))
        rows = rows.loc[valid_mask].reset_index(drop=True)
        paired_default = paired_default.loc[valid_mask].reset_index(drop=True)
    else:
        rows = _valid_metric_rows(rows).reset_index(drop=True)
    if rows.empty:
        return nan_result

    def split_pool(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pool_mask = (
            frame["track_name"].astype(str).eq(
                UNMATCHED_PREDICTION_POOL_KEY
            )
            if "track_name" in frame.columns
            else pd.Series(False, index=frame.index)
        )
        pool = frame.loc[pool_mask].reset_index(drop=True)
        if len(pool) > 1:
            raise ValueError(
                "Bootstrap input must contain at most one unmatched prediction pool"
            )
        references = frame.loc[~pool_mask].reset_index(drop=True)
        return references, pool

    reference_rows, current_pool = split_pool(rows)
    default_reference_rows: Optional[pd.DataFrame] = None
    default_pool: Optional[pd.DataFrame] = None
    if paired_default is not None:
        default_reference_rows, default_pool = split_pool(paired_default)
        if len(current_pool) != len(default_pool):
            raise ValueError(
                "Paired bootstrap inputs must both contain the fixed "
                "unmatched prediction pool"
            )
    n_tracks = len(reference_rows)
    if n_tracks == 0:
        return nan_result
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n_tracks, size=(samples, n_tracks))

    boot_values = {metric: [] for metric in BOOTSTRAP_METRIC_KEYS}
    boot_delta_f1: list[float] = []
    boot_delta_mae: list[float] = []

    for sample_idx in sample_indices:
        current_sample = reference_rows.iloc[sample_idx].reset_index(drop=True)
        if not current_pool.empty:
            current_sample = pd.concat(
                [current_sample, current_pool],
                ignore_index=True,
                sort=False,
            )
        current_metrics = compute_binary_metrics(current_sample)
        for metric in BOOTSTRAP_METRIC_KEYS:
            boot_values[metric].append(current_metrics[metric])

        if paired_default is not None:
            assert default_reference_rows is not None
            assert default_pool is not None
            default_sample = default_reference_rows.iloc[
                sample_idx
            ].reset_index(drop=True)
            if not default_pool.empty:
                default_sample = pd.concat(
                    [default_sample, default_pool],
                    ignore_index=True,
                    sort=False,
                )
            default_metrics = compute_binary_metrics(default_sample)
            boot_delta_f1.append(
                _metric_delta(
                    current_metrics["ignition_f1"],
                    default_metrics["ignition_f1"],
                )
            )
            boot_delta_mae.append(
                _metric_delta(
                    current_metrics["onset_mae"],
                    default_metrics["onset_mae"],
                )
            )

    result: dict[str, float] = {}
    for metric in BOOTSTRAP_METRIC_KEYS:
        values = np.asarray(boot_values[metric], dtype=float)
        result[f"{metric}_ci_low"] = _nanpercentile_or_nan(values, 2.5)
        result[f"{metric}_ci_high"] = _nanpercentile_or_nan(values, 97.5)

    if include_paired:
        delta_f1 = np.asarray(boot_delta_f1, dtype=float)
        delta_mae = np.asarray(boot_delta_mae, dtype=float)
        result["delta_f1_ci_low"] = _nanpercentile_or_nan(delta_f1, 2.5)
        result["delta_f1_ci_high"] = _nanpercentile_or_nan(delta_f1, 97.5)
        result["delta_onset_mae_ci_low"] = _nanpercentile_or_nan(
            delta_mae, 2.5
        )
        result["delta_onset_mae_ci_high"] = _nanpercentile_or_nan(
            delta_mae, 97.5
        )

    return result


def _align_paired_track_rows(
    rows: pd.DataFrame,
    paired_default: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = next(
        (
            candidate
            for candidate in ("track_name", "track_id")
            if candidate in rows.columns and candidate in paired_default.columns
        ),
        None,
    )
    if key is None:
        raise ValueError(
            "Paired bootstrap inputs require a common track_name or track_id column"
        )
    for label, frame in (("rows", rows), ("paired_default", paired_default)):
        if frame[key].isna().any():
            raise ValueError(f"{label}.{key} must not contain missing identities")
        if frame[key].duplicated().any():
            raise ValueError(f"{label}.{key} values must be unique")

    current_ids = set(rows[key].tolist())
    default_ids = set(paired_default[key].tolist())
    if current_ids != default_ids:
        missing = sorted(current_ids - default_ids, key=str)
        extra = sorted(default_ids - current_ids, key=str)
        raise ValueError(
            "Paired bootstrap inputs must contain the same test track set; "
            f"missing_from_default={missing}, extra_in_default={extra}"
        )

    aligned_default = (
        paired_default.set_index(key, drop=False)
        .loc[rows[key].tolist()]
        .reset_index(drop=True)
    )
    return rows.reset_index(drop=True), aligned_default


def _row_is_excluded_from_deltas(row: pd.Series) -> bool:
    status = str(row.get("run_status", "success")).strip().lower()
    if status in {"error", "skipped"}:
        return True
    skip_reason = row.get("skip_reason", "")
    return pd.notna(skip_reason) and bool(str(skip_reason).strip())


def add_default_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Attach default-relative deltas using each parameter group's own default row."""
    result = summary.copy()
    for metric in WINDOW_METRICS:
        result[f"delta_{metric}"] = float("nan")
    delta_columns = (
        "delta_ignition_precision",
        "delta_ignition_recall",
        "delta_ignition_f1",
        "relative_f1_change_percent",
        "delta_false_ignition_rate",
        "delta_onset_mae",
        "relative_onset_mae_change_percent",
        "delta_reference_coverage",
        "delta_suspicious_merge_proxy_count",
    )
    for column in delta_columns:
        result[column] = float("nan")

    group_columns = ("parameter_group", "parameter_name")
    for group_key, group_df in result.groupby(list(group_columns), sort=False):
        default_rows = group_df[group_df["is_default_configuration"] == True]  # noqa: E712
        if len(default_rows) != 1:
            raise ValueError(
                f"{group_key} must have exactly one default row; "
                f"found {len(default_rows)}"
            )
        default_row = default_rows.iloc[0]
        default_status = str(default_row.get("run_status", "success")).strip().lower()
        if default_status not in {"success", "success_with_track_errors"}:
            raise ValueError(
                f"{group_key} default row must be successful; "
                f"run_status={default_status!r}"
            )
        default_skip_reason = default_row.get("skip_reason", "")
        if pd.notna(default_skip_reason) and str(default_skip_reason).strip():
            raise ValueError(
                f"{group_key} default row must be not skipped; "
                f"skip_reason={default_skip_reason!r}"
            )

        for idx in group_df.index:
            row = result.loc[idx]
            if _row_is_excluded_from_deltas(row):
                continue

            if str(group_key[0]) in TRACKING_GROUPS:
                for metric in WINDOW_METRICS:
                    result.at[idx, f"delta_{metric}"] = _metric_delta(
                        row.get(metric), default_row.get(metric)
                    )
                continue

            result.at[idx, "delta_ignition_precision"] = _metric_delta(
                row.get("ignition_precision"), default_row.get("ignition_precision")
            )
            result.at[idx, "delta_ignition_recall"] = _metric_delta(
                row.get("ignition_recall"), default_row.get("ignition_recall")
            )
            result.at[idx, "delta_ignition_f1"] = _metric_delta(
                row.get("ignition_f1"), default_row.get("ignition_f1")
            )

            current_f1 = row.get("ignition_f1")
            default_f1 = default_row.get("ignition_f1")
            if pd.notna(current_f1) and pd.notna(default_f1) and float(default_f1) != 0.0:
                result.at[idx, "relative_f1_change_percent"] = (
                    (float(current_f1) - float(default_f1))
                    / float(default_f1)
                    * 100.0
                )

            result.at[idx, "delta_false_ignition_rate"] = _metric_delta(
                row.get("false_ignition_rate"),
                default_row.get("false_ignition_rate"),
            )
            result.at[idx, "delta_onset_mae"] = _metric_delta(
                row.get("onset_mae"), default_row.get("onset_mae")
            )
            current_mae = row.get("onset_mae")
            default_mae = default_row.get("onset_mae")
            if (
                pd.notna(current_mae)
                and pd.notna(default_mae)
                and np.isfinite(float(current_mae))
                and np.isfinite(float(default_mae))
                and float(default_mae) > 0.0
            ):
                result.at[
                    idx, "relative_onset_mae_change_percent"
                ] = (
                    (float(current_mae) - float(default_mae))
                    / float(default_mae)
                    * 100.0
                )
            if "reference_track_coverage" in row.index:
                result.at[idx, "delta_reference_coverage"] = _metric_delta(
                    row.get("reference_track_coverage"),
                    default_row.get("reference_track_coverage"),
                )
            if "suspicious_merge_proxy_count" in row.index:
                result.at[
                    idx, "delta_suspicious_merge_proxy_count"
                ] = _metric_delta(
                    row.get("suspicious_merge_proxy_count"),
                    default_row.get("suspicious_merge_proxy_count"),
                )

    return result


def _is_stable_parameter_value(row: pd.Series, parameter_group: str) -> bool:
    if _row_is_excluded_from_deltas(row):
        return False

    if parameter_group in TRACKING_GROUPS:
        return all(pd.notna(row.get(f"delta_{metric}")) and
                   float(row[f"delta_{metric}"]) >= -STABLE_MAX_COVERAGE_DROP - 1e-12
                   for metric in WINDOW_METRICS)

    relative_f1 = row.get("relative_f1_change_percent")
    if pd.isna(relative_f1) or float(relative_f1) < -STABLE_MAX_F1_DROP_PCT:
        return False

    delta_onset = row.get("delta_onset_mae")
    if pd.isna(delta_onset) or float(delta_onset) > STABLE_MAX_ONSET_MAE_INCREASE:
        return False

    delta_false = row.get("delta_false_ignition_rate")
    if (
        pd.isna(delta_false)
        or float(delta_false) > STABLE_MAX_FALSE_IGNITION_INCREASE
    ):
        return False

    return True


def compute_stable_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    """Tracking: <=3 percentage-point drops in all three window metrics.

    GMM/RTV retain original F1, onset-MAE, and false-ignition constraints.
    These intervals are descriptive tested regions, not confidence guarantees.
    """
    rows: list[dict[str, Any]] = []
    group_columns = ("parameter_group", "parameter_name")

    for (parameter_group, parameter_name), group_df in summary.groupby(
        list(group_columns), sort=False
    ):
        default_value = (
            group_df["default_value"].iloc[0]
            if "default_value" in group_df.columns
            else float("nan")
        )
        tested_mask = ~group_df.apply(_row_is_excluded_from_deltas, axis=1)
        n_tested_values = int(tested_mask.sum())
        stable_df = group_df[
            group_df.apply(
                lambda row: _is_stable_parameter_value(row, str(parameter_group)),
                axis=1,
            )
        ]
        stable_values = tuple(stable_df["parameter_value"].tolist())
        if str(parameter_group) in TRACKING_GROUPS:
            numeric = pd.to_numeric(stable_df["parameter_value"], errors="coerce")
            entry = {
                "parameter_group": parameter_group, "parameter_name": parameter_name,
                "default_value": default_value,
                "stable_min": stable_df.loc[numeric.idxmin(), "parameter_value"] if len(numeric) and numeric.notna().all() else (stable_values[0] if stable_values else float("nan")),
                "stable_max": stable_df.loc[numeric.idxmax(), "parameter_value"] if len(numeric) and numeric.notna().all() else (stable_values[-1] if stable_values else float("nan")),
                "n_tested_values": n_tested_values, "n_stable_values": len(stable_df),
                "stable_values": stable_values, "tracking_metric_protocol": WINDOW_METRIC_PROTOCOL,
                "stable_max_absolute_window_drop": STABLE_MAX_COVERAGE_DROP,
            }
            for metric in WINDOW_METRICS:
                worst = float(stable_df[f"delta_{metric}"].min()) if len(stable_df) else float("nan")
                entry[f"worst_delta_{metric}"] = worst
                entry[f"max_drop_{metric}"] = max(0.0, -worst) if np.isfinite(worst) else float("nan")
            rows.append(entry)
            continue
        if stable_df.empty:
            rows.append(
                {
                    "parameter_group": parameter_group,
                    "parameter_name": parameter_name,
                    "default_value": default_value,
                    "stable_min": float("nan"),
                    "stable_max": float("nan"),
                    "n_tested_values": n_tested_values,
                    "n_stable_values": 0,
                    "stable_values": (),
                    "worst_relative_f1_change_percent": float("nan"),
                    "worst_delta_onset_mae": float("nan"),
                    "worst_delta_false_ignition_rate": float("nan"),
                    "worst_delta_reference_coverage": float("nan"),
                    "max_f1_drop_in_stable_region": float("nan"),
                    "max_mae_increase_in_stable_region": float("nan"),
                    "max_far_increase_in_stable_region": float("nan"),
                    "max_reference_coverage_drop": float("nan"),
                }
            )
            continue

        numeric_values = pd.to_numeric(
            stable_df["parameter_value"], errors="coerce"
        )
        if numeric_values.notna().all():
            stable_min: Any = stable_df.loc[numeric_values.idxmin(), "parameter_value"]
            stable_max: Any = stable_df.loc[numeric_values.idxmax(), "parameter_value"]
        else:
            stable_min = stable_values[0]
            stable_max = stable_values[-1]
        rows.append(
            {
                "parameter_group": parameter_group,
                "parameter_name": parameter_name,
                "default_value": default_value,
                "stable_min": stable_min,
                "stable_max": stable_max,
                "n_tested_values": n_tested_values,
                "n_stable_values": int(len(stable_df)),
                "stable_values": stable_values,
                "worst_relative_f1_change_percent": float(
                    stable_df["relative_f1_change_percent"].min()
                ),
                "worst_delta_onset_mae": float(stable_df["delta_onset_mae"].max()),
                "worst_delta_false_ignition_rate": float(
                    stable_df["delta_false_ignition_rate"].max()
                ),
                "worst_delta_reference_coverage": float(
                    stable_df["delta_reference_coverage"].min()
                )
                if "delta_reference_coverage" in stable_df.columns
                else float("nan"),
                "max_f1_drop_in_stable_region": float(
                    np.maximum(
                        0.0,
                        -pd.to_numeric(
                            stable_df["relative_f1_change_percent"],
                            errors="coerce",
                        ),
                    ).max()
                ),
                "max_mae_increase_in_stable_region": float(
                    np.maximum(
                        0.0,
                        pd.to_numeric(
                            stable_df["delta_onset_mae"], errors="coerce"
                        ),
                    ).max()
                ),
                "max_far_increase_in_stable_region": float(
                    np.maximum(
                        0.0,
                        pd.to_numeric(
                            stable_df["delta_false_ignition_rate"],
                            errors="coerce",
                        ),
                    ).max()
                ),
                "max_reference_coverage_drop": (
                    float(
                        np.maximum(
                            0.0,
                            -pd.to_numeric(
                                stable_df["delta_reference_coverage"],
                                errors="coerce",
                            ),
                        ).max()
                    )
                    if parameter_group in TRACKING_GROUPS
                    and "delta_reference_coverage" in stable_df.columns
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed feature cache and GMM/RTV execution
# ---------------------------------------------------------------------------

_RTV_FEATURE_TO_CONFIG_FIELD: Mapping[str, str] = MappingProxyType(
    {
        "gray_p99": "min_raw_delta_gray_p99",
        "corr_p99": "min_raw_delta_corr_p99",
        "bright_energy_mean": "min_raw_delta_bright_energy",
        "bright_area_ratio": "min_raw_delta_bright_area",
        "hsv_v_p99": "min_raw_delta_hsv_v_p99",
    }
)

_RTV_FEATURE_TO_DIAGNOSTIC_FIELD: Mapping[str, str] = MappingProxyType(
    {
        "gray_p99": "delta_gray_p99",
        "corr_p99": "delta_corr_p99",
        "bright_energy_mean": "delta_bright_energy_mean",
        "bright_area_ratio": "delta_bright_area_ratio",
        "hsv_v_p99": "delta_hsv_v_p99",
    }
)

_FORMAL_DETECTOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "gray_p95",
    "gray_p99",
    "corr_p95",
    "corr_p99",
    "corr_max",
    "bright_energy_mean",
    "bright_area_ratio",
    "bright_mean",
    "red_ratio",
    "rg_ratio",
    "hsv_v_p95",
    "hsv_v_p99",
    "largest_area_ratio",
    "roi_width",
    "roi_height",
    "roi_area",
    "roi_aspect_ratio",
)


def _jsonable_config(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        raw = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        raw = vars(cfg)
    else:
        raise TypeError("Feature extraction config must be a dataclass or object")
    return json.loads(json.dumps(raw, sort_keys=True, default=str))


def canonical_track_name(raw: Any, formal_module: Any = None) -> str:
    """Return the formal unpadded ``track_N`` identity."""
    parser = getattr(formal_module, "track_id_from_name", None)
    if callable(parser):
        try:
            track_id = parser(str(raw).strip())
        except (TypeError, ValueError):
            track_id = None
        if track_id is not None:
            return f"track_{int(track_id)}"
    normalizer = getattr(formal_module, "normalize_track_name", None)
    if callable(normalizer):
        return str(normalizer(raw))
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    normalized = str(raw).strip()
    if not normalized:
        return ""
    suffix = (
        normalized.split("_", 1)[1]
        if normalized.lower().startswith("track_")
        else normalized
    )
    try:
        return f"track_{int(float(suffix))}"
    except (TypeError, ValueError):
        return normalized


def canonicalize_annotation_mapping(
    annotations: Mapping[str, Mapping[str, Any]],
    *,
    formal_module: Any = None,
) -> dict[str, Mapping[str, Any]]:
    canonical: dict[str, Mapping[str, Any]] = {}
    for raw_name, annotation in annotations.items():
        name = canonical_track_name(raw_name, formal_module)
        if not name:
            raise ValueError(f"Invalid annotation track name: {raw_name!r}")
        if name in canonical:
            raise ValueError(
                f"Duplicate canonical annotation track name: {name}"
            )
        canonical[name] = annotation
    return canonical


def _normalize_track_folders(
    track_folders: Sequence[Any],
    *,
    formal_module: Any = None,
) -> list[tuple[str, Path]]:
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in track_folders:
        if isinstance(item, Mapping):
            path = Path(str(item.get("folder", item.get("roi_path", ""))))
            raw_name = str(item.get("track_name", path.name))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            raw_name, raw_path = item
            raw_name, path = str(raw_name), Path(str(raw_path))
        else:
            path = Path(item)
            raw_name = path.name
        name = canonical_track_name(raw_name, formal_module)
        if not name or not str(path):
            raise ValueError(f"Invalid track folder entry: {item!r}")
        if name in seen:
            raise ValueError(f"Duplicate canonical track name: {name}")
        seen.add(name)
        normalized.append((name, path.expanduser().resolve()))
    return normalized


def _track_source_record(
    track_name: str,
    roi_path: Path,
    image_paths: Sequence[str],
    default_cfg: Any,
    formal_module_source: Mapping[str, Any],
) -> dict[str, Any]:
    images = []
    for raw_path in sorted(image_paths, key=lambda value: Path(value).name):
        path = Path(raw_path)
        stat = path.stat()
        images.append(
            {
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "track_name": track_name,
        "roi_path": str(roi_path),
        "images": images,
        "extraction_config": _jsonable_config(default_cfg),
        "formal_module_source": dict(formal_module_source),
    }


def _formal_module_source_record(
    ignition_module_path: str,
    ignition_mod: Any,
) -> dict[str, Any]:
    requested = Path(ignition_module_path).expanduser()
    module_file = requested if requested.is_file() else None
    if module_file is None:
        loaded_file = getattr(ignition_mod, "__file__", None)
        if loaded_file:
            candidate = Path(loaded_file).expanduser()
            if candidate.is_file():
                module_file = candidate
    if module_file is None:
        raise FileNotFoundError(
            "Cannot fingerprint formal ignition module file: "
            f"{ignition_module_path}"
        )
    resolved = module_file.resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _validate_cached_feature_df(
    feature_df: Any,
    image_paths: Sequence[str],
    *,
    expected_columns: Optional[Sequence[str]] = None,
    expected_dtypes: Optional[Sequence[str]] = None,
    expected_identifiers: Optional[Sequence[str]] = None,
) -> bool:
    if not isinstance(feature_df, pd.DataFrame):
        return False
    if feature_df.columns.duplicated().any():
        return False
    if len(feature_df) != len(image_paths):
        return False
    if not set(_FORMAL_DETECTOR_REQUIRED_COLUMNS).issubset(feature_df.columns):
        return False
    current_identifiers = [Path(path).name for path in image_paths]
    if len(current_identifiers) != len(set(current_identifiers)):
        return False
    identifier_columns = [
        column
        for column in ("filename", "image_path")
        if column in feature_df.columns
    ]
    if not identifier_columns:
        return False
    for column in identifier_columns:
        cached_identifiers = [
            Path(str(value)).name for value in feature_df[column].tolist()
        ]
        if cached_identifiers != current_identifiers:
            return False
    if expected_columns is not None:
        if list(expected_columns) != feature_df.columns.tolist():
            return False
    if expected_dtypes is not None:
        actual_dtypes = [str(dtype) for dtype in feature_df.dtypes.tolist()]
        if list(expected_dtypes) != actual_dtypes:
            return False
    if expected_identifiers is not None:
        if list(expected_identifiers) != current_identifiers:
            return False
    for column in _FORMAL_DETECTOR_REQUIRED_COLUMNS:
        values = pd.to_numeric(feature_df[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).any():
            return False
    return True


def _record_signature(record: Mapping[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_cache_filename(track_name: str) -> str:
    digest = hashlib.sha256(track_name.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in track_name)
    return f"{safe}__{digest}.pkl"


def _track_id(ignition_mod: Any, track_name: str) -> Optional[int]:
    parser = getattr(ignition_mod, "track_id_from_name", None)
    if parser is None:
        return None
    value = parser(track_name)
    return None if value is None else int(value)


def load_or_build_feature_cache(
    track_folders: Sequence[Any],
    cache_dir: str | Path,
    default_cfg: Any,
    *,
    ignition_module_path: str = DEFAULT_IGNITION_MODULE_PATH,
    ignition_mod: Optional[ModuleType] = None,
    max_tracks: Optional[int] = None,
    workers: int = 1,
    cache_stats: Optional[dict[str, int]] = None,
) -> dict[str, FeatureCacheEntry]:
    """Load valid per-track feature pickles and rebuild only stale ROI sources."""
    if max_tracks is not None and max_tracks < 0:
        raise ValueError("max_tracks must be non-negative or None")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    formal_mod = ignition_mod or _load_module(ignition_module_path, "ignition")
    selected = _normalize_track_folders(
        track_folders, formal_module=formal_mod
    )
    if max_tracks is not None:
        selected = selected[:max_tracks]
    formal_module_source = _formal_module_source_record(
        ignition_module_path, formal_mod
    )
    output_dir = Path(cache_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "feature_cache_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {"version": 1, "tracks": {}}
    if not isinstance(manifest.get("tracks"), dict):
        manifest = {"version": 1, "tracks": {}}

    result: dict[str, FeatureCacheEntry] = {}
    pending: list[dict[str, Any]] = []
    hits = 0
    rebuilt = 0
    extracted_count = 0
    for track_name, roi_path in selected:
        if not roi_path.is_dir():
            raise FileNotFoundError(f"Track ROI folder not found: {roi_path}")
        image_paths = tuple(
            str(path)
            for path in formal_mod.list_images_in_folder(str(roi_path))
        )
        if not image_paths:
            raise RuntimeError(f"No images found in track folder: {roi_path}")
        source_record = _track_source_record(
            track_name,
            roi_path,
            image_paths,
            default_cfg,
            formal_module_source,
        )
        signature = _record_signature(source_record)
        pickle_name = _safe_cache_filename(track_name)
        pickle_path = output_dir / pickle_name
        old_record = manifest["tracks"].get(track_name, {})
        entry: Optional[FeatureCacheEntry] = None
        if (
            old_record.get("source_signature") == signature
            and old_record.get("pickle_file") == pickle_name
            and pickle_path.is_file()
        ):
            try:
                with pickle_path.open("rb") as handle:
                    payload = pickle.load(handle)
                if (
                    isinstance(payload, Mapping)
                    and payload.get("source_signature") == signature
                    and isinstance(old_record.get("feature_columns"), list)
                    and isinstance(old_record.get("feature_dtypes"), list)
                    and isinstance(old_record.get("image_identifiers"), list)
                    and _validate_cached_feature_df(
                        payload.get("feature_df"),
                        image_paths,
                        expected_columns=old_record["feature_columns"],
                        expected_dtypes=old_record["feature_dtypes"],
                        expected_identifiers=old_record["image_identifiers"],
                    )
                ):
                    entry = FeatureCacheEntry(
                        track_name=track_name,
                        track_id=payload.get("track_id"),
                        roi_path=str(roi_path),
                        image_paths=image_paths,
                        feature_df=payload["feature_df"],
                        source_signature=signature,
                    )
            except Exception:
                entry = None

        if entry is None:
            pending.append(
                {
                    "track_name": track_name,
                    "roi_path": roi_path,
                    "image_paths": image_paths,
                    "source_record": source_record,
                    "signature": signature,
                    "pickle_name": pickle_name,
                    "pickle_path": pickle_path,
                    "is_rebuild": bool(old_record),
                }
            )
            continue

        manifest["tracks"][track_name] = {
            **source_record,
            "source_signature": signature,
            "pickle_file": pickle_name,
            "feature_columns": entry.feature_df.columns.tolist(),
            "feature_dtypes": [
                str(dtype) for dtype in entry.feature_df.dtypes.tolist()
            ],
            "image_identifiers": [Path(path).name for path in image_paths],
        }
        result[track_name] = entry
        hits += 1

    def extract_one(item: Mapping[str, Any]) -> tuple[Mapping[str, Any], pd.DataFrame]:
        feature_df = formal_mod.extract_sequence_features(
            list(item["image_paths"]), copy.deepcopy(default_cfg)
        )
        if not _validate_cached_feature_df(feature_df, item["image_paths"]):
            raise ValueError(
                "Formal extract_sequence_features returned an invalid "
                "DataFrame schema or row count"
            )
        return item, feature_df

    if workers == 1:
        extracted = [extract_one(item) for item in pending]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            extracted = list(executor.map(extract_one, pending))
    for item, feature_df in extracted:
        track_name = str(item["track_name"])
        entry = FeatureCacheEntry(
            track_name=track_name,
            track_id=_track_id(formal_mod, track_name),
            roi_path=str(item["roi_path"]),
            image_paths=tuple(item["image_paths"]),
            feature_df=feature_df,
            source_signature=str(item["signature"]),
        )
        temporary_path = Path(item["pickle_path"]).with_suffix(".pkl.tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(
                {
                    "track_id": entry.track_id,
                    "source_signature": item["signature"],
                    "feature_df": feature_df,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_path.replace(Path(item["pickle_path"]))
        manifest["tracks"][track_name] = {
            **item["source_record"],
            "source_signature": item["signature"],
            "pickle_file": item["pickle_name"],
            "feature_columns": feature_df.columns.tolist(),
            "feature_dtypes": [str(dtype) for dtype in feature_df.dtypes.tolist()],
            "image_identifiers": [
                Path(path).name for path in item["image_paths"]
            ],
        }
        result[track_name] = entry
        if bool(item["is_rebuild"]):
            rebuilt += 1
        else:
            extracted_count += 1

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    if cache_stats is not None:
        cache_stats.clear()
        cache_stats.update(
            {
                "hits": int(hits),
                "rebuilt": int(rebuilt),
                "extracted": int(extracted_count),
                "total": int(len(selected)),
            }
        )
    return {track_name: result[track_name] for track_name, _ in selected}


def evaluate_rtv_votes(
    raw_deltas: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    minimum_votes: int,
) -> tuple[bool, int]:
    """Apply only the requested vote count to five formal raw delta values."""
    if minimum_votes < 1 or minimum_votes > len(_RTV_FEATURE_TO_CONFIG_FIELD):
        raise ValueError("minimum_votes must be between 1 and 5")
    missing = set(_RTV_FEATURE_TO_CONFIG_FIELD) - set(raw_deltas)
    missing |= set(_RTV_FEATURE_TO_CONFIG_FIELD) - set(thresholds)
    if missing:
        raise ValueError(f"Missing RTV raw features: {sorted(missing)}")
    votes = sum(
        float(raw_deltas[name]) >= float(thresholds[name])
        for name in _RTV_FEATURE_TO_CONFIG_FIELD
    )
    return votes >= minimum_votes, int(votes)


def _validate_experiment_track_set(
    feature_cache: Mapping[str, FeatureCacheEntry],
    annotations: Mapping[str, Mapping[str, Any]],
    expected_track_names: set[str] | frozenset[str] | Sequence[str],
) -> tuple[str, ...]:
    if expected_track_names is None:
        raise ValueError("expected_track_names is required for every experiment")
    actual = tuple(feature_cache)
    actual_set = set(actual)
    if len(actual) != len(actual_set):
        raise ValueError("Feature cache contains duplicate test track names")
    expected_values = tuple(expected_track_names)
    expected = set(expected_values)
    if len(expected_values) != len(expected):
        raise ValueError("expected_track_names must contain unique track names")
    if actual_set != expected:
        raise ValueError(
            "Experiment test track set differs from the fixed set; "
            f"missing={sorted(expected - actual_set)}, extra={sorted(actual_set - expected)}"
        )
    missing_annotations = actual_set - set(annotations)
    if missing_annotations:
        raise ValueError(
            f"Missing annotations for test track set: {sorted(missing_annotations)}"
        )
    return actual


def _test_track_set_hash(track_names: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(canonical_track_name(name) for name in track_names),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _error_evaluation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gt_category": str(annotation.get("gt_category", "")).lower(),
        "gt_has_ignition": bool(annotation.get("gt_has_ignition", False)),
        "gt_onset_idx": int(annotation.get("gt_onset_idx", -1)),
        "gt_onset_start": int(annotation.get("gt_onset_start", -1)),
        "gt_onset_end": int(annotation.get("gt_onset_end", -1)),
        "onset_abs_error": np.nan,
        "onset_inside_interval": np.nan,
        "is_correct_presence": np.nan,
        "is_false_ignition": np.nan,
        "is_missed_ignition": np.nan,
        "is_already_hot_correct": np.nan,
    }


def evaluate_detection_prediction(
    prediction: Mapping[str, Any], annotation: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate one formal detector result against local-onset GT."""
    gt_has = bool(annotation.get("gt_has_ignition", False))
    pred_has = bool(prediction.get("has_ignition", False))
    gt_onset = int(annotation.get("gt_onset_idx", -1))
    pred_onset = int(prediction.get("ignition_idx_onset", -1))
    onset_error = (
        abs(pred_onset - gt_onset)
        if gt_has and pred_has and gt_onset >= 0 and pred_onset >= 0
        else float("nan")
    )
    onset_start = int(annotation.get("gt_onset_start", -1))
    onset_end = int(annotation.get("gt_onset_end", -1))
    inside_interval: Any = float("nan")
    if gt_has and pred_has and pred_onset >= 0:
        inside_interval = bool(onset_start <= pred_onset <= onset_end)
    return {
        "gt_category": str(annotation.get("gt_category", "")).lower(),
        "gt_has_ignition": gt_has,
        "gt_onset_idx": gt_onset,
        "gt_onset_start": onset_start,
        "gt_onset_end": onset_end,
        "onset_abs_error": onset_error,
        "onset_inside_interval": inside_interval,
        "is_correct_presence": bool(pred_has == gt_has),
        "is_false_ignition": bool(pred_has and not gt_has),
        "is_missed_ignition": bool(gt_has and not pred_has),
        "is_already_hot_correct": bool(
            str(annotation.get("gt_category", "")).lower() == "already_hot"
            and not pred_has
        ),
    }


def _prediction_row(
    entry: FeatureCacheEntry,
    prediction: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    transition_vote_count: Optional[int] = None,
    raw_deltas: Optional[Mapping[str, float]] = None,
    error_message: str = "",
) -> dict[str, Any]:
    row = {
        "track_name": entry.track_name,
        "track_id": entry.track_id,
        "roi_path": entry.roi_path,
        "n_frames": int(len(entry.feature_df)),
        "pred_has_ignition": bool(prediction.get("has_ignition", False)),
        "pred_ignition_status": str(prediction.get("ignition_status", "")),
        "pred_onset_idx": int(prediction.get("ignition_idx_onset", -1)),
        "pred_confirmed_idx": int(prediction.get("ignition_idx_confirmed", -1)),
        "pred_stable_idx": int(prediction.get("stable_combustion_idx", -1)),
        "selected_k": int(prediction.get("gmm_n_states", 0)),
        "gmm_n_states": int(prediction.get("gmm_n_states", 0)),
        "onset_within_1": (
            float(evaluation.get("onset_abs_error", np.nan)) <= 1.0
            if pd.notna(evaluation.get("onset_abs_error", np.nan))
            else np.nan
        ),
        "onset_within_3": (
            float(evaluation.get("onset_abs_error", np.nan)) <= 3.0
            if pd.notna(evaluation.get("onset_abs_error", np.nan))
            else np.nan
        ),
        "transition_vote_count": (
            np.nan if transition_vote_count is None else int(transition_vote_count)
        ),
        "is_error_track": (
            str(prediction.get("ignition_status", "")).strip()
            in ERROR_PREDICTION_STATUSES
            or bool(str(error_message).strip())
        ),
        "error_message": error_message,
        **dict(evaluation),
    }
    for name in _RTV_FEATURE_TO_CONFIG_FIELD:
        row[f"raw_delta_{name}"] = (
            np.nan if raw_deltas is None else raw_deltas.get(name, np.nan)
        )
    return row


def _summarize_run(
    rows: pd.DataFrame,
    experiment: ExperimentSpec,
    local_cfg: Any,
    test_track_set_hash: str,
) -> dict[str, Any]:
    error_mask = _error_track_mask(rows)
    valid_rows = rows.loc[~error_mask].copy()
    n_error_tracks = int(error_mask.sum())
    if n_error_tracks == len(rows) and len(rows):
        run_status = "error"
    elif n_error_tracks:
        run_status = "success_with_track_errors"
    else:
        run_status = "success"
    error_messages = tuple(
        dict.fromkeys(
            message
            for message in rows.loc[error_mask, "error_message"]
            .fillna("")
            .astype(str)
            .str.strip()
            .tolist()
            if message
        )
    )
    if run_status == "error" and not error_messages:
        error_statuses = sorted(
            set(
                rows.loc[error_mask, "pred_ignition_status"]
                .fillna("unknown_error")
                .astype(str)
                .tolist()
            )
        )
        error_messages = (
            "All test tracks failed with status: " + ", ".join(error_statuses),
        )
    summary: dict[str, Any] = {
        "parameter_group": experiment.group,
        "parameter_name": experiment.parameter_name,
        "parameter_value": experiment.parameter_value,
        "default_value": experiment.default_value,
        "is_default_configuration": experiment.is_default_configuration,
        "config_id": experiment.config_id,
        "run_status": run_status,
        "error_message": " | ".join(error_messages),
        "n_tracks": int(len(rows)),
        "n_valid_tracks": int(len(valid_rows)),
        "n_error_tracks": n_error_tracks,
        "test_track_set_hash": test_track_set_hash,
        **compute_binary_metrics(valid_rows),
    }
    selected_k = pd.to_numeric(
        valid_rows["selected_k"], errors="coerce"
    ).dropna()
    selected_k = selected_k[
        selected_k.isin([1, 2, 3, 4])
    ].astype(int)
    summary["mean_selected_gmm_states"] = (
        float(selected_k.mean()) if len(selected_k) else float("nan")
    )
    selected_counts = selected_k.value_counts()
    for value in range(1, 5):
        count = int(selected_counts.get(value, 0))
        summary[f"n_selected_k_{value}"] = count
        summary[f"n_gmm_k{value}"] = count
    for value, count in (
        valid_rows["pred_ignition_status"].astype(str).value_counts().items()
    ):
        summary[f"n_status_{value}"] = int(count)
    for status in (
        "single_state",
        "ambiguous_gmm_unstable",
        "no_ignition_weak_transition",
    ):
        summary.setdefault(f"n_status_{status}", 0)
    votes = pd.to_numeric(
        valid_rows["transition_vote_count"], errors="coerce"
    ).dropna()
    summary["n_transitions_passing_1"] = int((votes >= 1).sum())
    summary["n_transitions_passing_2"] = int((votes >= 2).sum())
    summary["n_transitions_passing_at_least_3"] = int((votes >= 3).sum())
    for field_name in RTV_THRESHOLD_FIELDS:
        if hasattr(local_cfg, field_name):
            summary[field_name] = float(getattr(local_cfg, field_name))
    return summary


def _run_detection_experiment(
    feature_cache: Mapping[str, FeatureCacheEntry],
    annotations: Mapping[str, Mapping[str, Any]],
    experiment: ExperimentSpec,
    local_cfg: Any,
    *,
    ignition_mod: Any,
    ignition_ablation: Any,
    expected_track_names: set[str] | frozenset[str] | Sequence[str],
    minimum_votes: Optional[int] = None,
    workers: int = 1,
) -> RunResult:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    track_names = _validate_experiment_track_set(
        feature_cache, annotations, expected_track_names
    )
    track_set_hash = _test_track_set_hash(track_names)
    def process_track(track_name: str) -> dict[str, Any]:
        entry = feature_cache[track_name]
        track_cfg = copy.deepcopy(local_cfg)
        prediction: dict[str, Any]
        vote_count: Optional[int] = None
        raw_deltas: Optional[dict[str, float]] = None
        error_message = ""
        try:
            detector_cfg = (
                replace(track_cfg, enable_raw_transition_validation=False)
                if minimum_votes is not None
                else track_cfg
            )
            prediction = dict(
                ignition_mod.detect_ignition_gmm_rtv(
                    entry.feature_df.copy(), detector_cfg
                )
            )
            if minimum_votes is not None and prediction.get("has_ignition", False):
                candidates = prediction.get("all_ignition_candidates", []) or []
                transition_idx = (
                    int(candidates[0]["transition_idx"])
                    if candidates
                    else int(prediction.get("ignition_idx_confirmed", -1))
                )
                _formal_passed, diagnostics = (
                    ignition_mod.raw_transition_has_thermal_increase(
                        entry.feature_df.copy(), transition_idx, track_cfg
                    )
                )
                raw_deltas = {
                    name: float(
                        diagnostics.get(
                            diagnostic_name,
                            float("nan"),
                        )
                    )
                    for name, diagnostic_name in _RTV_FEATURE_TO_DIAGNOSTIC_FIELD.items()
                }
                thresholds = {
                    name: float(getattr(track_cfg, field_name))
                    for name, field_name in _RTV_FEATURE_TO_CONFIG_FIELD.items()
                }
                passed, vote_count = evaluate_rtv_votes(
                    raw_deltas, thresholds, minimum_votes
                )
                prediction["raw_transition_metrics"] = dict(diagnostics)
                if not passed:
                    no_idx = int(getattr(track_cfg, "no_ignition_index", -1))
                    prediction.update(
                        {
                            "has_ignition": False,
                            "ignition_status": "no_ignition_weak_transition",
                            "ignition_idx_onset": no_idx,
                            "ignition_idx_confirmed": no_idx,
                            "ignition_idx_candidate": no_idx,
                            "stable_combustion_idx": no_idx,
                            "n_ignition_candidates": 0,
                            "all_ignition_candidates": [],
                        }
                    )
            elif minimum_votes is None:
                diagnostics = prediction.get("raw_transition_metrics", {}) or {}
                if diagnostics:
                    raw_deltas = {
                        name: float(diagnostics.get(diagnostic_name, float("nan")))
                        for name, diagnostic_name in _RTV_FEATURE_TO_DIAGNOSTIC_FIELD.items()
                    }
                    thresholds = {
                        name: float(getattr(track_cfg, field_name))
                        for name, field_name in _RTV_FEATURE_TO_CONFIG_FIELD.items()
                    }
                    _passed, vote_count = evaluate_rtv_votes(
                        raw_deltas, thresholds, 1
                    )
            evaluation = evaluate_detection_prediction(
                prediction, annotations[track_name]
            )
        except Exception as exc:
            error_message = str(exc)
            prediction = {
                "has_ignition": False,
                "ignition_status": "track_error",
                "ignition_idx_onset": -1,
                "gmm_n_states": 0,
            }
            evaluation = _error_evaluation(annotations[track_name])
        return _prediction_row(
            entry,
            prediction,
            evaluation,
            transition_vote_count=vote_count,
            raw_deltas=raw_deltas,
            error_message=error_message,
        )
    if workers == 1:
        rows = [process_track(track_name) for track_name in track_names]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(executor.map(process_track, track_names))
    rows_df = pd.DataFrame(rows)
    return RunResult(
        experiment=experiment,
        rows=rows_df,
        summary=_summarize_run(
            rows_df, experiment, local_cfg, track_set_hash
        ),
        evaluated_track_names=track_names,
        test_track_set_hash=track_set_hash,
        config=local_cfg,
    )


def run_gmm_experiment(
    feature_cache: Mapping[str, FeatureCacheEntry],
    annotations: Mapping[str, Mapping[str, Any]],
    experiment: ExperimentSpec,
    default_cfg: Any,
    *,
    ignition_mod: Any,
    ignition_ablation: Any,
    expected_track_names: set[str] | frozenset[str] | Sequence[str],
    workers: int = 1,
) -> RunResult:
    """Run one GMM OFAT setting through the formal detector and evaluator."""
    if experiment.group != "gmm":
        raise ValueError(f"Not a GMM experiment: {experiment.group}")
    local_cfg = replace(default_cfg, **dict(experiment.overrides))
    return _run_detection_experiment(
        feature_cache,
        annotations,
        experiment,
        local_cfg,
        ignition_mod=ignition_mod,
        ignition_ablation=ignition_ablation,
        expected_track_names=expected_track_names,
        workers=workers,
    )


def run_rtv_experiment(
    feature_cache: Mapping[str, FeatureCacheEntry],
    annotations: Mapping[str, Mapping[str, Any]],
    experiment: ExperimentSpec,
    default_cfg: Any,
    *,
    ignition_mod: Any,
    ignition_ablation: Any,
    expected_track_names: set[str] | frozenset[str] | Sequence[str],
    workers: int = 1,
) -> RunResult:
    """Run RTV alpha normally, or replace only the formal raw-transition vote gate."""
    if experiment.group != "rtv":
        raise ValueError(f"Not an RTV experiment: {experiment.group}")
    overrides = dict(experiment.overrides)
    minimum_votes = overrides.pop("rtv_minimum_votes", None)
    local_cfg = replace(default_cfg, **overrides)
    if minimum_votes is not None and overrides:
        raise ValueError("RTV voting rules must run at alpha=1 with no threshold overrides")
    return _run_detection_experiment(
        feature_cache,
        annotations,
        experiment,
        local_cfg,
        ignition_mod=ignition_mod,
        ignition_ablation=ignition_ablation,
        expected_track_names=expected_track_names,
        minimum_votes=None if minimum_votes is None else int(minimum_votes),
        workers=workers,
    )


# ---------------------------------------------------------------------------
# Dataset split, manifests, orchestration, and reporting
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dict__"):
        return _to_jsonable(vars(value))
    return value


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _to_jsonable(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _coerce_bool_series(values: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    normalized = values.astype(str).str.strip().str.lower()
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"{column} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def build_local_stratified_dataset_split(
    track_records: Sequence[Mapping[str, Any]],
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible split stratified by ignition presence."""
    if not 0 <= float(test_fraction) < 1:
        raise ValueError("test_fraction must be in [0, 1)")
    rows: list[dict[str, Any]] = []
    for item in track_records:
        name = canonical_track_name(item.get("track_name"))
        gt = dict(item.get("gt", {}))
        rows.append(
            {
                "track_name": name,
                "track_id": item.get("track_id"),
                "gt_category": str(gt.get("gt_category", "")),
                "gt_has_ignition": bool(gt.get("gt_has_ignition", False)),
            }
        )
    split = pd.DataFrame(rows)
    if split.empty:
        raise ValueError("Cannot split an empty track population")
    if split["track_name"].duplicated().any():
        duplicates = sorted(
            split.loc[
                split["track_name"].duplicated(keep=False), "track_name"
            ].unique()
        )
        raise ValueError(f"Duplicate track identities in split input: {duplicates}")

    rng = np.random.default_rng(int(seed))
    test_names: set[str] = set()
    for _, group in split.groupby("gt_has_ignition", sort=True):
        names = group["track_name"].astype(str).tolist()
        order = rng.permutation(len(names))
        if float(test_fraction) <= 0:
            n_test = 0
        else:
            n_test = max(1, int(round(len(names) * float(test_fraction))))
            if len(names) > 1:
                n_test = min(n_test, len(names) - 1)
            else:
                n_test = 1
        test_names.update(names[int(index)] for index in order[:n_test])
    split["in_test"] = split["track_name"].isin(test_names)
    split["in_tune"] = ~split["in_test"]
    split["split"] = np.where(split["in_test"], "test", "tune")
    return split


def resolve_dataset_split(
    track_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset_split_csv: Optional[str | Path] = None,
    evaluate_all_tracks: bool = False,
    test_fraction: float = 0.2,
    seed: int = 42,
    split_module: Any = None,
    max_tracks: Optional[int] = None,
) -> tuple[pd.DataFrame, tuple[str, ...], str]:
    """Resolve one strict split and persist the exact table used by every run."""
    if max_tracks is not None and max_tracks <= 0:
        raise ValueError("max_tracks must be positive or None")
    records = [dict(item) for item in track_records]
    for item in records:
        item["track_name"] = canonical_track_name(item["track_name"])
    current_names = [str(item["track_name"]) for item in records]
    if len(current_names) != len(set(current_names)):
        raise ValueError("Current track records contain duplicate track_name values")
    current_set = set(current_names)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dataset_split_csv is not None:
        source = Path(dataset_split_csv)
        if not source.is_file():
            raise FileNotFoundError(f"dataset split CSV not found: {source}")
        split_df = pd.read_csv(source, encoding="utf-8-sig")
        if "track_name" not in split_df.columns:
            raise ValueError("dataset split CSV requires track_name")
        split_df["track_name"] = split_df["track_name"].map(
            canonical_track_name
        )
        names = split_df["track_name"].astype(str)
        if names.duplicated().any():
            duplicates = sorted(names[names.duplicated(keep=False)].unique())
            raise ValueError(f"dataset split has duplicate track assignments: {duplicates}")
        supplied_set = set(names)
        unknown = sorted(supplied_set - current_set)
        missing = sorted(current_set - supplied_set)
        if unknown:
            raise ValueError(f"dataset split contains unknown tracks: {unknown}")
        if missing:
            raise ValueError(f"dataset split is missing current tracks: {missing}")
        if "in_test" in split_df.columns:
            in_test = _coerce_bool_series(split_df["in_test"], "in_test")
        elif "split" in split_df.columns:
            split_values = split_df["split"].astype(str).str.strip().str.lower()
            allowed = {"tune", "test", "tune_and_test_oracle", "excluded"}
            invalid = sorted(set(split_values) - allowed)
            if invalid:
                raise ValueError(f"dataset split contains invalid split values: {invalid}")
            in_test = split_values.isin({"test", "tune_and_test_oracle"})
        else:
            raise ValueError("dataset split CSV requires in_test or split")
    else:
        if split_module is None:
            split_df = build_local_stratified_dataset_split(
                records, float(test_fraction), int(seed)
            )
        else:
            tune_names, test_names = split_module.select_baseline_tune_test_tracks(
                records, float(test_fraction), int(seed)
            )
            split_df = split_module.build_dataset_split(
                records, set(tune_names), set(test_names)
            )
        in_test = _coerce_bool_series(split_df["in_test"], "in_test")

    if evaluate_all_tracks:
        candidate_set = current_set
        protocol = "full_dataset_sensitivity_analysis"
    else:
        candidate_set = set(
            split_df.loc[in_test, "track_name"].astype(str).tolist()
        )
        protocol = "heldout_test"
        if not candidate_set:
            raise ValueError("heldout_test split contains no test tracks")
    ordered_candidates = [
        name for name in current_names if name in candidate_set
    ]
    if max_tracks is not None:
        ordered_candidates = ordered_candidates[: int(max_tracks)]
    evaluation_names = tuple(ordered_candidates)
    if not evaluation_names:
        raise ValueError(f"{protocol} selection contains no evaluation tracks")

    if max_tracks is None:
        selected_split = split_df.copy()
    else:
        order = {name: index for index, name in enumerate(evaluation_names)}
        selected_split = split_df[
            split_df["track_name"].astype(str).isin(order)
        ].copy()
        selected_split["_selection_order"] = (
            selected_split["track_name"].astype(str).map(order)
        )
        selected_split = (
            selected_split.sort_values("_selection_order", kind="stable")
            .drop(columns="_selection_order")
            .reset_index(drop=True)
        )
    selected_split.to_csv(
        output_path / "dataset_split.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return selected_split, evaluation_names, protocol


class RunManifest:
    """Atomic per-configuration state store for resumable sweeps."""

    TERMINAL_SUCCESS = frozenset({"success", "success_with_track_errors"})

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {"version": 1, "entries": {}}
        entries = payload.get("entries", {})
        self.entries: dict[str, dict[str, Any]] = (
            dict(entries) if isinstance(entries, Mapping) else {}
        )

    def update(self, config_id: str, run_status: str, **details: Any) -> None:
        self.entries[str(config_id)] = {
            "config_id": str(config_id),
            "run_status": str(run_status),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **details,
        }
        _atomic_write_json(
            self.path, {"version": 1, "entries": self.entries}
        )

    def register_pending(self, config_ids: Sequence[str]) -> None:
        registered_at = datetime.now().isoformat(timespec="seconds")
        for config_id in config_ids:
            key = str(config_id)
            previous = self.entries.get(key, {})
            previous_status = str(
                previous.get(
                    "resume_source_status", previous.get("run_status", "")
                )
            )
            self.entries[key] = {
                "config_id": key,
                "run_status": "pending",
                "updated_at": registered_at,
                **(
                    {"resume_source_status": previous_status}
                    if previous_status in self.TERMINAL_SUCCESS
                    else {}
                ),
            }
        _atomic_write_json(
            self.path, {"version": 1, "entries": self.entries}
        )

    def should_skip(
        self, config_id: str, *, resume: bool, overwrite: bool
    ) -> bool:
        if overwrite or not resume:
            return False
        entry = self.entries.get(str(config_id), {})
        status = str(
            entry.get("resume_source_status", entry.get("run_status", ""))
        )
        return status in self.TERMINAL_SUCCESS


def _experiment_row(experiment: ExperimentSpec) -> dict[str, Any]:
    return {
        "parameter_group": experiment.group,
        "parameter_name": experiment.parameter_name,
        "parameter_value": experiment.parameter_value,
        "default_value": experiment.default_value,
        "ratio_to_default": experiment.ratio_to_default,
        "parameter_ratio_to_default": experiment.ratio_to_default,
        "is_default_configuration": experiment.is_default_configuration,
        "config_id": experiment.config_id,
        "skip_reason": experiment.skip_reason,
    }


def _prune_tracking_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove downstream ignition/global-continuity metrics from tracking exports."""
    legacy = {
        "ignition_precision", "ignition_recall", "ignition_f1", "false_ignition_rate",
        "onset_mae", "precision", "recall", "f1", "far", "mae",
        "reference_track_coverage", "reference_coverage", "suspicious_merge_proxy",
        "suspicious_merge_proxy_count", "mean_track_length", "short_track_ratio",
        "tp", "fp", "fn", "tn", "end_to_end_tp", "end_to_end_fp", "end_to_end_fn",
        "relative_f1_change_percent", "relative_onset_mae_change_percent",
        "mean_selected_gmm_states", "worst_relative_f1_change_percent",
        "worst_delta_onset_mae", "worst_delta_false_ignition_rate",
        "worst_delta_reference_coverage", "max_f1_drop_in_stable_region",
        "max_mae_increase_in_stable_region", "max_far_increase_in_stable_region",
        "max_reference_coverage_drop",
    }
    def unwanted(column: str) -> bool:
        base = column
        for prefix in ("worst_delta_", "delta_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
                break
        for suffix in ("_ci_low", "_ci_high"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        return base in legacy or column in legacy or column.startswith(("n_gmm_k", "n_status_"))
    return frame.drop(columns=[column for column in frame.columns if unwanted(str(column))])


def ensure_output_contract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve ignition aliases, but give tracking exports window-only metrics."""
    result = frame.copy()
    if len(result) and "parameter_group" in result and set(result["parameter_group"]).issubset(TRACKING_GROUPS):
        result = _prune_tracking_metric_columns(result)
        if "ratio_to_default" in result:
            result["parameter_ratio_to_default"] = result["ratio_to_default"]
        for metric in WINDOW_METRICS:
            for column in (metric, f"delta_{metric}"):
                if column not in result:
                    result[column] = float("nan")
        return result
    aliases = {
        "parameter_ratio_to_default": "ratio_to_default",
        "delta_precision": "delta_ignition_precision",
        "delta_recall": "delta_ignition_recall",
        "delta_f1": "delta_ignition_f1",
        "delta_suspicious_merge_proxy": (
            "delta_suspicious_merge_proxy_count"
        ),
        "precision_ci_low": "ignition_precision_ci_low",
        "precision_ci_high": "ignition_precision_ci_high",
        "recall_ci_low": "ignition_recall_ci_low",
        "recall_ci_high": "ignition_recall_ci_high",
        "f1_ci_low": "ignition_f1_ci_low",
        "f1_ci_high": "ignition_f1_ci_high",
        "far_ci_low": "false_ignition_rate_ci_low",
        "far_ci_high": "false_ignition_rate_ci_high",
        "delta_mae_ci_low": "delta_onset_mae_ci_low",
        "delta_mae_ci_high": "delta_onset_mae_ci_high",
    }
    required_columns = (
        "parameter_ratio_to_default",
        "delta_precision",
        "delta_recall",
        "delta_f1",
        "relative_f1_change_percent",
        "delta_false_ignition_rate",
        "delta_onset_mae",
        "relative_onset_mae_change_percent",
        "delta_reference_coverage",
        "delta_suspicious_merge_proxy",
        "precision_ci_low",
        "precision_ci_high",
        "recall_ci_low",
        "recall_ci_high",
        "f1_ci_low",
        "f1_ci_high",
        "far_ci_low",
        "far_ci_high",
        "onset_mae_ci_low",
        "onset_mae_ci_high",
        "delta_f1_ci_low",
        "delta_f1_ci_high",
        "delta_mae_ci_low",
        "delta_mae_ci_high",
        "mean_selected_gmm_states",
        "n_gmm_k1",
        "n_gmm_k2",
        "n_gmm_k3",
        "n_gmm_k4",
        "n_status_single_state",
        "n_status_ambiguous_gmm_unstable",
        "n_status_no_ignition_weak_transition",
    )
    def missing_mask(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype(str).str.strip().eq("")

    for target, source in aliases.items():
        if target not in result.columns:
            result[target] = float("nan")
        if source in result.columns:
            missing = missing_mask(result[target])
            result.loc[missing, target] = result.loc[missing, source]
    for column in required_columns:
        if column not in result.columns:
            result[column] = float("nan")

    gmm_mask = (
        result["parameter_group"].astype(str).isin({"gmm", "rtv"})
        if "parameter_group" in result.columns
        else pd.Series(False, index=result.index)
    )
    for value in range(1, 5):
        target = f"n_gmm_k{value}"
        source = f"n_selected_k_{value}"
        if source in result.columns:
            missing = gmm_mask & missing_mask(result[target])
            result.loc[missing, target] = result.loc[missing, source]
        missing = gmm_mask & missing_mask(result[target])
        result.loc[missing, target] = 0

    mean_missing = gmm_mask & missing_mask(
        result["mean_selected_gmm_states"]
    )
    if mean_missing.any():
        counts = pd.DataFrame(
            {
                value: pd.to_numeric(
                    result[f"n_gmm_k{value}"], errors="coerce"
                ).fillna(0.0)
                for value in range(1, 5)
            },
            index=result.index,
        )
        total = counts.sum(axis=1)
        weighted = sum(
            value * counts[value] for value in range(1, 5)
        )
        reconstructed = weighted / total.where(total > 0)
        result.loc[mean_missing, "mean_selected_gmm_states"] = (
            reconstructed.loc[mean_missing]
        )

    status_fallbacks = {
        "n_status_single_state": "n_pred_single_state",
        "n_status_ambiguous_gmm_unstable": (
            "n_pred_ambiguous_gmm_unstable"
        ),
        "n_status_no_ignition_weak_transition": (
            "n_pred_weak_transition_rejected"
        ),
    }
    for target, source in status_fallbacks.items():
        if source in result.columns:
            missing = gmm_mask & missing_mask(result[target])
            result.loc[missing, target] = result.loc[missing, source]
        missing = gmm_mask & missing_mask(result[target])
        result.loc[missing, target] = 0
    return result


def execute_configurations(
    experiments: Sequence[ExperimentSpec],
    output_dir: str | Path,
    *,
    runner: Any,
    expected_track_names: Sequence[str],
    resume: bool = False,
    overwrite: bool = False,
    manifest: Optional[RunManifest] = None,
    execution_mode: str = "sensitivity_sweep",
    execution_context_hash: str = "",
    execution_context: Optional[Mapping[str, Any]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute configurations independently and always emit one summary row."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    state = manifest or RunManifest(root / "manifest.json")
    expected_hash = _test_track_set_hash(expected_track_names)
    context_fields = dict(execution_context or {})
    if execution_context_hash:
        context_fields["execution_context_hash"] = str(
            execution_context_hash
        )

    def build_config_payload(
        experiment: ExperimentSpec,
    ) -> dict[str, Any]:
        payload = {
            **_experiment_row(experiment),
            "overrides": dict(experiment.overrides),
            "expected_track_names": list(expected_track_names),
            "test_track_set_hash": expected_hash,
            "execution_mode": str(execution_mode),
            "execution_context_hash": str(
                context_fields.get(
                    "execution_context_hash", execution_context_hash
                )
            ),
        }
        for key in _RESUME_EXECUTION_CONTEXT_KEYS:
            if key not in payload:
                payload[key] = context_fields.get(key)
        if experiment.group in TRACKING_GROUPS:
            payload["tracking_metric_protocol"] = WINDOW_METRIC_PROTOCOL
        return payload

    # A successful manifest entry is reusable only when its original config
    # proves full execution-context compatibility. Validate every would-skip
    # target before register_pending or any per-run artifact is written.
    if resume and not overwrite:
        for experiment in experiments:
            if not state.should_skip(
                experiment.config_id,
                resume=True,
                overwrite=False,
            ):
                continue
            config_path = (
                root
                / experiment.group
                / experiment.parameter_name
                / experiment.config_id
                / "config.json"
            )
            if not config_path.is_file():
                raise ValueError(
                    "Resume rejected because successful manifest entry "
                    f"{experiment.config_id!r} has no config.json with the "
                    "required execution context. Use --overwrite or a new "
                    "output directory."
                )
            try:
                existing_config = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    "Resume rejected because successful manifest entry "
                    f"{experiment.config_id!r} has an unreadable config.json. "
                    "Use --overwrite or a new output directory."
                ) from exc
            _validate_resume_execution_context(
                existing_config,
                build_config_payload(experiment),
            )

    state.register_pending([item.config_id for item in experiments])
    summary_rows: list[dict[str, Any]] = []
    per_track_frames: list[pd.DataFrame] = []
    tracking_alignment_blocker = ""
    ordered_experiments = list(experiments)
    canonical_default = next(
        (
            item
            for item in ordered_experiments
            if item.group in TRACKING_GROUPS
            and item.is_default_configuration
            and not item.skip_reason
        ),
        None,
    )
    if canonical_default is not None:
        ordered_experiments.remove(canonical_default)
        ordered_experiments.insert(0, canonical_default)
    for experiment in ordered_experiments:
        run_dir = (
            root
            / experiment.group
            / experiment.parameter_name
            / experiment.config_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        base_row = _experiment_row(experiment)
        config_payload = build_config_payload(experiment)
        config_path = run_dir / "config.json"
        if resume and config_path.is_file() and not overwrite:
            existing_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            _validate_resume_execution_context(existing_config, config_payload)
        _atomic_write_json(config_path, config_payload)

        if experiment.skip_reason:
            row = {**base_row, "run_status": "skipped"}
            ensure_output_contract_columns(pd.DataFrame([row])).to_csv(
                run_dir / "summary.csv", index=False
            )
            pd.DataFrame().to_csv(run_dir / "per_track_results.csv", index=False)
            _atomic_write_json(run_dir / "run_status.json", row)
            state.update(
                experiment.config_id,
                "skipped",
                output_dir=str(run_dir),
                skip_reason=experiment.skip_reason,
            )
            summary_rows.append(row)
            continue

        if (
            tracking_alignment_blocker
            and experiment.group in TRACKING_GROUPS
        ):
            row = {
                **base_row,
                "run_status": "skipped",
                "skip_reason": "canonical_default_tracking_alignment_failed",
                "error_message": tracking_alignment_blocker,
                "test_track_set_hash": expected_hash,
            }
            ensure_output_contract_columns(pd.DataFrame([row])).to_csv(
                run_dir / "summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame().to_csv(run_dir / "per_track_results.csv", index=False)
            _atomic_write_json(run_dir / "run_status.json", row)
            state.update(
                experiment.config_id,
                "skipped",
                output_dir=str(run_dir),
                skip_reason=row["skip_reason"],
                error_message=tracking_alignment_blocker,
            )
            summary_rows.append(row)
            continue

        if state.should_skip(
            experiment.config_id, resume=resume, overwrite=overwrite
        ):
            prior = state.entries[experiment.config_id]
            prior_status = str(
                prior.get("resume_source_status", prior.get("run_status", ""))
            )
            prior_summary_path = run_dir / "summary.csv"
            prior_row: dict[str, Any] = {}
            if prior_summary_path.is_file():
                prior_frame = pd.read_csv(
                    prior_summary_path, encoding="utf-8-sig"
                )
                if len(prior_frame):
                    prior_row = prior_frame.iloc[0].to_dict()
            row = {
                **prior_row,
                **base_row,
                "run_status": prior_status,
                "resumed": True,
                "execution_action": "resumed",
            }
            ensure_output_contract_columns(pd.DataFrame([row])).to_csv(
                run_dir / "summary.csv",
                index=False,
                encoding="utf-8-sig",
            )
            _atomic_write_json(run_dir / "run_status.json", row)
            state.update(
                experiment.config_id,
                prior_status,
                output_dir=str(run_dir),
                execution_action="resumed",
            )
            prior_tracks_path = run_dir / "per_track_results.csv"
            if prior_tracks_path.is_file():
                try:
                    prior_tracks = pd.read_csv(
                        prior_tracks_path, encoding="utf-8-sig"
                    )
                    if len(prior_tracks):
                        per_track_frames.append(prior_tracks)
                except pd.errors.EmptyDataError:
                    pass
            summary_rows.append(row)
            continue

        running_status = {
            **base_row,
            "run_status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        _atomic_write_json(run_dir / "run_status.json", running_status)
        state.update(
            experiment.config_id, "running", output_dir=str(run_dir)
        )
        try:
            result: RunResult = runner(experiment, run_dir)
            if set(result.evaluated_track_names) != set(expected_track_names):
                raise ValueError(
                    f"{experiment.config_id} evaluated a different test track set"
                )
            if result.test_track_set_hash != expected_hash:
                raise ValueError(
                    f"{experiment.config_id} emitted inconsistent test track hash"
                )
            status = str(result.summary.get("run_status", "success"))
            row = {
                **base_row,
                **result.summary,
                "run_status": status,
                "test_track_set_hash": result.test_track_set_hash,
            }
            rows = result.rows.copy()
            rows.insert(0, "config_id", experiment.config_id)
            rows.insert(0, "parameter_name", experiment.parameter_name)
            rows.insert(0, "parameter_group", experiment.group)
            per_track_frames.append(rows)
            ensure_output_contract_columns(pd.DataFrame([row])).to_csv(
                run_dir / "summary.csv", index=False, encoding="utf-8-sig"
            )
            rows.to_csv(
                run_dir / "per_track_results.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if result.alignment_rows is not None:
                result.alignment_rows.to_csv(
                    run_dir / "alignment_diagnostics.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            completed = {
                **row,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            _atomic_write_json(run_dir / "run_status.json", completed)
            state.update(
                experiment.config_id, status, output_dir=str(run_dir)
            )
        except Exception as exc:
            consistency_diagnostics = getattr(
                exc, "consistency_diagnostics", {}
            )
            is_tracking_alignment_error = (
                isinstance(exc, TrackingAlignmentError)
                and experiment.group in TRACKING_GROUPS
            )
            is_default_alignment_error = (
                is_tracking_alignment_error
                and experiment.is_default_configuration
            )
            status = (
                "alignment_error" if is_tracking_alignment_error else "error"
            )
            if is_default_alignment_error:
                tracking_alignment_blocker = str(exc)
            row = {
                **base_row,
                "run_status": status,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "test_track_set_hash": expected_hash,
                **consistency_diagnostics,
            }
            ensure_output_contract_columns(pd.DataFrame([row])).to_csv(
                run_dir / "summary.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame().to_csv(
                run_dir / "per_track_results.csv", index=False
            )
            _atomic_write_json(run_dir / "run_status.json", row)
            state.update(
                experiment.config_id,
                status,
                output_dir=str(run_dir),
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=row["traceback"],
            )
        summary_rows.append(row)

    summary = ensure_output_contract_columns(pd.DataFrame(summary_rows))
    per_track = (
        pd.concat(per_track_frames, ignore_index=True, sort=False)
        if per_track_frames
        else pd.DataFrame()
    )
    return summary, per_track


def _save_table(df: pd.DataFrame, path_without_ext: Path) -> Optional[str]:
    path_without_ext.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path_without_ext.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )
    try:
        df.to_excel(path_without_ext.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        message = f"Excel export failed for {path_without_ext}: {exc}"
        warnings.warn(message)
        return message
    return None


def save_group_outputs(
    summary: pd.DataFrame,
    per_track: pd.DataFrame,
    stable_intervals: pd.DataFrame,
    output_dir: str | Path,
) -> list[str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = ensure_output_contract_columns(summary)
    output_warnings = [
        message
        for message in (
            _save_table(summary, root / "combined_summary"),
            _save_table(summary, root / "all_parameter_sensitivity"),
            _save_table(per_track, root / "combined_per_track_results"),
            _save_table(stable_intervals, root / "stable_intervals"),
            _save_table(
                stable_intervals, root / "stable_operating_intervals"
            ),
        )
        if message
    ]
    for group in sorted(VALID_GROUPS):
        group_dir = root / group
        group_dir.mkdir(parents=True, exist_ok=True)
        group_summary = (
            summary[summary["parameter_group"] == group].copy()
            if "parameter_group" in summary
            else pd.DataFrame()
        )
        group_tracks = (
            per_track[per_track["parameter_group"] == group].copy()
            if "parameter_group" in per_track
            else pd.DataFrame()
        )
        group_stable = (
            stable_intervals[
                stable_intervals["parameter_group"] == group
            ].copy()
            if "parameter_group" in stable_intervals
            else pd.DataFrame()
        )
        if group in TRACKING_GROUPS:
            group_summary = ensure_output_contract_columns(group_summary)
            group_tracks = _prune_tracking_metric_columns(group_tracks)
            group_stable = _prune_tracking_metric_columns(group_stable)
        for message in (
            _save_table(group_summary, group_dir / "per_run_summary"),
            _save_table(group_tracks, group_dir / "per_track_results"),
            _save_table(group_stable, group_dir / "stable_intervals"),
        ):
            if message:
                output_warnings.append(message)
    return output_warnings


def save_default_metrics(
    summary: pd.DataFrame, output_dir: str | Path
) -> tuple[pd.DataFrame, list[str]]:
    summary = ensure_output_contract_columns(summary)
    default_rows = summary[
        summary["is_default_configuration"] == True  # noqa: E712
    ].copy()
    root = Path(output_dir)
    warning = _save_table(default_rows, root / "default_metrics")
    _atomic_write_json(
        root / "default_metrics.json",
        {
            str(row["parameter_name"]): row.to_dict()
            for _, row in default_rows.iterrows()
        },
    )
    return default_rows, [warning] if warning else []


def record_output_warnings(
    summary: pd.DataFrame,
    output_dir: str | Path,
    output_warnings: Sequence[str],
) -> None:
    if not output_warnings:
        return
    message = " | ".join(dict.fromkeys(str(item) for item in output_warnings))
    summary["output_warning"] = message
    summary = ensure_output_contract_columns(summary)
    root = Path(output_dir)
    summary.to_csv(
        root / "combined_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        root / "all_parameter_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    for group in sorted(VALID_GROUPS):
        group_summary = summary[summary["parameter_group"] == group]
        if group in TRACKING_GROUPS:
            group_summary = ensure_output_contract_columns(group_summary)
        group_summary.to_csv(
            root / group / "per_run_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )


def _plot_metric(
    rows: pd.DataFrame,
    metric: str,
    destination: Path,
    *,
    ylabel: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = rows.copy()
    plot_rows.to_csv(destination.with_suffix(".csv"), index=False)
    numeric_x = pd.to_numeric(plot_rows["parameter_value"], errors="coerce")
    if numeric_x.notna().all():
        x = numeric_x.to_numpy(dtype=float)
        labels = None
    else:
        x = np.arange(len(plot_rows), dtype=float)
        labels = plot_rows["parameter_value"].astype(str).tolist()
    y = pd.to_numeric(plot_rows[metric], errors="coerce").to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.plot(x, y, marker="o")
    default_mask = plot_rows["is_default_configuration"].fillna(False).astype(bool)
    if default_mask.any():
        default_index = int(np.flatnonzero(default_mask.to_numpy())[0])
        axis.scatter(
            [x[default_index]],
            [y[default_index]],
            marker="*",
            s=180,
            color="tab:red",
            zorder=5,
            label="Default",
        )
        axis.axvline(x[default_index], color="tab:red", linestyle="--", alpha=0.4)
        axis.legend()
    if labels is not None:
        axis.set_xticks(x, labels, rotation=20, ha="right")
    axis.set_xlabel(str(plot_rows["parameter_name"].iloc[0]))
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination.with_suffix(".png"), dpi=180)
    figure.savefig(destination.with_suffix(".pdf"))
    plt.close(figure)


def plot_parameter_results(
    summary: pd.DataFrame, output_dir: str | Path
) -> None:
    if summary.empty:
        return
    metric_specs = {
        "ignition_f1": "F1",
        "false_ignition_rate": "False ignition rate",
        "onset_mae": "Onset MAE (frames)",
    }
    tracking_specs = {
        "mean_pre_onset_coverage": "GT-centered pre-ignition coverage",
        "mean_ignition_window_completeness": "Ignition-window completeness",
        "observable_transition_rate": "Observable-transition rate",
    }
    usable = summary[
        summary["run_status"].isin(["success", "success_with_track_errors"])
    ]
    for (group, parameter), rows in usable.groupby(
        ["parameter_group", "parameter_name"], sort=False
    ):
        destination = Path(output_dir) / str(group) / "plots" / str(parameter)
        destination.mkdir(parents=True, exist_ok=True)
        specs = dict(metric_specs)
        if str(group) in TRACKING_GROUPS:
            specs = dict(tracking_specs)
        for metric, label in specs.items():
            if metric in rows.columns and pd.to_numeric(
                rows[metric], errors="coerce"
            ).notna().any():
                _plot_metric(
                    rows,
                    metric,
                    destination / metric,
                    ylabel=label,
                )

    rtv_rows = usable[usable["parameter_group"] == "rtv"]
    if not rtv_rows.empty:
        for parameter, rows in rtv_rows.groupby("parameter_name", sort=False):
            destination = Path(output_dir) / "rtv" / "plots" / str(parameter)
            if {"ignition_recall", "ignition_precision"}.issubset(rows.columns):
                pr_data = rows[
                    [
                        "parameter_value",
                        "is_default_configuration",
                        "ignition_recall",
                        "ignition_precision",
                    ]
                ].copy()
                pr_data.to_csv(destination / "precision_recall.csv", index=False)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5.5, 5.0))
                ax.plot(
                    pr_data["ignition_recall"],
                    pr_data["ignition_precision"],
                    marker="o",
                )
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
                fig.savefig(destination / "precision_recall.png", dpi=180)
                fig.savefig(destination / "precision_recall.pdf")
                plt.close(fig)
            if (
                str(parameter) == "rtv_alpha"
                and "n_status_no_ignition_weak_transition" in rows.columns
            ):
                _plot_metric(
                    rows,
                    "n_status_no_ignition_weak_transition",
                    destination / "weak_transition_rejections",
                    ylabel="Weak transition rejections",
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_groups(raw: str) -> set[str]:
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if not tokens:
        raise ValueError("At least one group must be provided.")
    normalized = {GROUP_ALIASES.get(token, token) for token in tokens}
    unknown = normalized - VALID_GROUPS - {"all"}
    if unknown:
        raise ValueError(
            f"Unknown groups {sorted(unknown)}; expected ot, repair, gmm, rtv, or all."
        )
    if "all" in normalized:
        return {"all"}
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=SCRIPT_TITLE,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group_selector = parser.add_mutually_exclusive_group()
    group_selector.add_argument(
        "--groups",
        default="all",
        help="Comma-separated groups: ot, repair, gmm, rtv, or all",
    )
    group_selector.add_argument(
        "--tracking-only",
        action="store_true",
        help=(
            "Run only the OT-association and trajectory-repair window-observability sensitivity "
            "groups (equivalent to --groups ot,repair)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default parameter_sensitivity_results/run_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--dataset-split-csv",
        default=None,
        help="Optional precomputed dataset_split.csv with tune/test assignments",
    )
    parser.add_argument(
        "--reference-tracks-csv",
        default=DEFAULT_MANUAL_GT_SOURCE_TRACKS_CSV,
        help=(
            "Long-format reference tracks whose track IDs match effective_trace; "
            "used to map local GT onset indices to global frames and align "
            "tracking-sweep outputs"
        ),
    )
    parser.add_argument(
        "--annotation-path",
        default=DEFAULT_ANNOTATION_PATH,
        help=(
            "Positive-onset GT with track_id and local frame_id columns; "
            "unlisted tracks in the evaluated population are negatives"
        ),
    )
    parser.add_argument(
        "--fixed-roi-dir",
        default='inputs/track_crop',
        help="Fixed T3 track-crop root used by GMM and RTV groups",
    )
    parser.add_argument(
        "--detections-path",
        default='inputs/cleaned_detections_txt',
        help="Detection txt path for tracking",
    )
    parser.add_argument(
        "--max-txt",
        type=int,
        default=8000,
        help="Maximum sorted detection TXT files to load; -1 loads all",
    )
    parser.add_argument(
        "--image-dir",
        default='inputs/MCFF0004-N001',
        help="Global frame image directory",
    )
    parser.add_argument(
        "--image-exts",
        default=".jpg,.jpeg,.png,.bmp,.tif,.tiff",
        help="Image extensions used to build the global frame index",
    )
    parser.add_argument(
        "--onset-tolerance",
        type=int,
        default=2,
        help="Ground-truth onset tolerance in frames",
    )
    parser.add_argument(
        "--tracking-module-path",
        default=DEFAULT_TRACKING_MODULE_PATH,
        help="Formal OT-Kalman tracker module path",
    )
    parser.add_argument(
        "--tracking-ablation-module-path",
        default=DEFAULT_TRACKING_HELPER_PATH,
        help="Tracking identity-alignment/GT helper module path",
    )
    parser.add_argument(
        "--tracking-ablation-t3-dir",
        default=None,
        help="Legacy T3 directory (compatibility only; ignition-metric comparisons disabled for window-only tracking)",
    )
    parser.add_argument(
        "--ignition-module-path",
        default=DEFAULT_IGNITION_MODULE_PATH,
        help="Formal GMM-RTV ignition module path",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Optional bootstrap resamples for numeric output only; plots omit CIs",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Random seed for paired bootstrap resampling",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Held-out test fraction when dataset split is generated",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for stratified dataset split generation",
    )
    parser.add_argument(
        "--evaluate-all-tracks",
        action="store_true",
        help="Evaluate on the full dataset instead of held-out test tracks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip configurations that already completed successfully",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run configurations even if successful outputs already exist",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Workers for safe per-track feature extraction and detection",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Optional cap on the number of tracks for smoke testing",
    )
    parser.add_argument(
        "--only-default-tracking",
        action="store_true",
        help=(
            "Run exactly one canonical full-default T3 tracking configuration; "
            "effective selected groups contain only requested tracking groups"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print registry and experiment plan without loading external data",
    )
    return parser


def load_tracking_detections(
    tracking_ablation: Any,
    tracker_mod: Any,
    detections_path: str | Path,
    *,
    max_txt: int,
) -> Mapping[int, Sequence[Mapping[str, Any]]]:
    """Load detection TXT files through the latest formal tracker API."""
    del tracking_ablation
    limit = int(max_txt)
    if limit == 0 or limit < -1:
        raise ValueError("--max-txt must be -1 or a positive integer")
    loader = getattr(tracker_mod, "load_detections_txt_dir", None)
    if not callable(loader):
        raise AttributeError(
            "Formal tracker module must expose load_detections_txt_dir"
        )
    detections = loader(Path(detections_path), max_txt=limit)
    if not detections:
        raise RuntimeError(f"No detections loaded from {detections_path}")
    return detections


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("parameter_sensitivity_results") / f"run_{stamp}"


def run_dry_run(
    groups: set[str],
    *,
    tracking_module_path: str = DEFAULT_TRACKING_MODULE_PATH,
    ignition_module_path: str = DEFAULT_IGNITION_MODULE_PATH,
    only_default_tracking: bool = False,
    annotation_path: str = DEFAULT_ANNOTATION_PATH,
) -> int:
    selected_groups = set(VALID_GROUPS) if groups == {"all"} else set(groups)
    if only_default_tracking and not (selected_groups & TRACKING_GROUPS):
        raise ValueError(
            "--only-default-tracking requires ot or repair to be selected"
        )
    ignition_path = _required_path(
        ignition_module_path, "--ignition-module-path"
    )
    ignition_mod = _load_module(str(ignition_path), "ignition")
    if selected_groups & TRACKING_GROUPS:
        tracker_path = _required_path(
            tracking_module_path, "--tracking-module-path"
        )
        tracker_defaults, gmm_defaults = load_formal_defaults(
            str(tracker_path), str(ignition_path)
        )
    else:
        tracker_defaults = TrackerSensitivityConfig()
        gmm_defaults = ignition_mod.IgnitionGMMConfig()
    annotation_resolved = _required_path(annotation_path, "--annotation-path")
    positive_count = len(
        read_effective_trace_positive_onsets(annotation_resolved)
    )
    registry = build_parameter_registry(
        tracker_default=tracker_defaults,
        gmm_default=gmm_defaults,
    )
    experiments = build_experiments(
        groups,
        tracker_default=tracker_defaults,
        gmm_default=gmm_defaults,
    )
    if only_default_tracking:
        experiments = [
            select_canonical_default_tracking_experiment(
                experiments, selected_groups
            )
        ]

    print(f"{SCRIPT_TITLE} dry-run")
    print("=" * 72)
    print(f"annotation_path: {annotation_resolved}")
    print(f"positive GT tracks: {positive_count}")
    print("Formal tracker defaults:")
    for key, value in asdict(tracker_defaults).items():
        print(f"  {key}: {value}")
    print()
    print("Formal GMM-RTV defaults (selected fields):")
    gmm_fields = (
        "gmm_max_states",
        "gmm_bic_improvement_ratio",
        "gmm_min_state_fraction",
        "gmm_min_state_count",
        "min_state_run",
        "min_cold_run",
        "min_hot_run",
        *RTV_THRESHOLD_FIELDS,
    )
    for field in gmm_fields:
        print(f"  {field}: {getattr(gmm_defaults, field)}")
    print()
    print("Parameter registry:")
    selected_registry = [
        spec for spec in registry if spec.group in selected_groups
    ]
    for spec in selected_registry:
        status = "SKIPPED" if spec.skip_reason else "ACTIVE"
        print(f"[{spec.group}] {spec.name} ({status})")
        print(f"  fields: {', '.join(spec.fields)}")
        print(f"  default: {spec.default_value}")
        if spec.skip_reason:
            print(f"  skip_reason: {spec.skip_reason}")
        else:
            print(f"  values: {list(spec.values)}")
        if spec.coupled_note:
            print(f"  note: {spec.coupled_note}")
        print()
    print(f"Parameter specifications: {len(selected_registry)}")
    print()

    print("Planned experiments by group:")
    for group in sorted(selected_groups):
        group_experiments = [item for item in experiments if item.group == group]
        print(f"  {group}: {len(group_experiments)} configurations")
    print()
    print(f"Total configurations: {len(experiments)}")
    return 0


def _required_path(
    value: Optional[str], option: str, *, directory: bool = False
) -> Path:
    if not value:
        raise ValueError(f"{option} is required for the selected groups")
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [raw, Path(__file__).resolve().parent / raw]
    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        exists = resolved.is_dir() if directory else resolved.is_file()
        if exists:
            return resolved
    kind = "directory" if directory else "file"
    raise FileNotFoundError(
        f"{option} {kind} not found; checked: "
        + ", ".join(str(path) for path in checked)
    )


def _filter_annotations(
    annotations: Mapping[str, Mapping[str, Any]], names: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    missing = sorted(set(names) - set(annotations))
    if missing:
        raise ValueError(f"Missing annotations for selected tracks: {missing}")
    return {name: annotations[name] for name in names}


def _build_detection_annotations_from_reference(
    listed_no_ignition_ids: Sequence[int],
    reference_ignition: pd.DataFrame,
    *,
    onset_tolerance: int,
    framework: Any,
) -> dict[str, dict[str, Any]]:
    """Adapt one preloaded formal reference table to detector-local GT."""
    if onset_tolerance < 0:
        raise ValueError("--onset-tolerance must be non-negative")
    has_status = "status" in reference_ignition.columns
    valid_rows = [
        row
        for _, row in reference_ignition.iterrows()
        if not has_status or str(row.get("status", "")).strip() == "ok"
    ]
    parsed = [
        (row, *framework._parse_track_id_from_row(row, reference_ignition.columns))
        for row in valid_rows
    ]
    valid_ids = {track_id for _, _, track_id in parsed if track_id is not None}
    listed = {
        int(track_id)
        for track_id in listed_no_ignition_ids
        if int(track_id) in valid_ids
    }
    annotations: dict[str, dict[str, Any]] = {}
    for row, track_name, track_id in parsed:
        if not track_name:
            continue
        if (
            track_id is not None
            and track_id in listed
            or framework._is_no_ignition_candidate(
                row["ignition_idx_candidate"]
            )
        ):
            annotations[str(track_name)] = framework._no_ignition_gt()
            continue
        onset = int(float(row["ignition_idx_onset"]))
        annotations[str(track_name)] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_idx": onset,
            "gt_onset_start": max(0, onset - int(onset_tolerance)),
            "gt_onset_end": onset + int(onset_tolerance),
        }
    return annotations


def _apply_bootstrap_and_deltas(
    summary: pd.DataFrame,
    per_track: pd.DataFrame,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    result = summary.copy()
    for metric in WINDOW_METRICS:
        for column in (f"delta_{metric}", f"{metric}_ci_low", f"{metric}_ci_high",
                       f"delta_{metric}_ci_low", f"delta_{metric}_ci_high"):
            if column not in result:
                result[column] = float("nan")
    result["bootstrap_status"] = (
        "disabled" if int(samples) <= 0 else "pending"
    )
    result["bootstrap_error"] = ""
    result["default_comparison_status"] = "pending"
    result["default_comparison_error"] = ""
    for column in (
        "delta_ignition_precision",
        "delta_ignition_recall",
        "delta_ignition_f1",
        "relative_f1_change_percent",
        "delta_false_ignition_rate",
        "delta_onset_mae",
        "relative_onset_mae_change_percent",
        "delta_reference_coverage",
        "delta_suspicious_merge_proxy_count",
    ):
        if column not in result:
            result[column] = float("nan")
    for column in _bootstrap_ci_keys(include_paired=True):
        if column not in result:
            result[column] = float("nan")
    successful = {"success", "success_with_track_errors"}
    for (_group, _parameter), indices in result.groupby(
        ["parameter_group", "parameter_name"], sort=False
    ).groups.items():
        group_indices = list(indices)
        group = result.loc[group_indices].copy()
        try:
            compared_group = add_default_deltas(group)
        except ValueError as exc:
            message = str(exc)
            result.loc[
                group_indices, "default_comparison_status"
            ] = "error"
            result.loc[
                group_indices, "default_comparison_error"
            ] = message
            if int(samples) > 0:
                result.loc[group_indices, "bootstrap_status"] = "error"
                result.loc[group_indices, "bootstrap_error"] = (
                    "Default comparison unavailable: " + message
                )
            continue
        delta_columns = [
            column
            for column in compared_group.columns
            if column.startswith("delta_")
            or column
            in {
                "relative_f1_change_percent",
                "relative_onset_mae_change_percent",
            }
        ]
        for column in delta_columns:
            result.loc[group_indices, column] = compared_group[column]
        result.loc[
            group_indices, "default_comparison_status"
        ] = "success"
        defaults = compared_group[
            compared_group["is_default_configuration"] == True  # noqa: E712
        ]
        default_id = str(defaults.iloc[0]["config_id"])
        default_rows = per_track[per_track["config_id"] == default_id]
        for index in group_indices:
            if str(result.at[index, "run_status"]) not in successful:
                continue
            config_id = str(result.at[index, "config_id"])
            current_rows = per_track[per_track["config_id"] == config_id]
            try:
                intervals = bootstrap_metrics(
                    current_rows,
                    samples=int(samples),
                    seed=int(seed),
                    paired_default=default_rows,
                )
            except ValueError as exc:
                result.at[index, "bootstrap_status"] = "error"
                result.at[index, "bootstrap_error"] = str(exc)
                continue
            for column, value in intervals.items():
                result.at[index, column] = value
            result.at[index, "bootstrap_status"] = (
                "disabled" if int(samples) <= 0 else "success"
            )
    return ensure_output_contract_columns(result)


def resolve_tracking_execution_protocol(
    args: argparse.Namespace,
) -> tuple[bool, int]:
    """Return the effective full-dataset and bootstrap settings."""
    only_default = bool(getattr(args, "only_default_tracking", False))
    evaluate_all = bool(getattr(args, "evaluate_all_tracks", False)) or only_default
    bootstrap_samples = (
        0 if only_default else int(getattr(args, "bootstrap_samples", 0))
    )
    return evaluate_all, bootstrap_samples


def analysis_exit_code(
    summary: pd.DataFrame,
    *,
    only_default_tracking: bool,
) -> int:
    """Return nonzero after a persisted canonical alignment failure."""
    if "run_status" not in summary:
        return 0
    alignment_errors = summary["run_status"].astype(str).eq("alignment_error")
    if bool(only_default_tracking):
        has_alignment_error = alignment_errors.any()
    elif "is_default_configuration" in summary:
        defaults = _truthy_series(summary["is_default_configuration"])
        has_alignment_error = (alignment_errors & defaults).any()
    else:
        has_alignment_error = False
    return 2 if bool(has_alignment_error) else 0


def run_full_analysis(args: argparse.Namespace, groups: set[str]) -> int:
    """Run the complete formal sensitivity workflow for selected groups."""
    requested_selected_groups = (
        set(VALID_GROUPS) if groups == {"all"} else set(groups)
    )
    requested_tracking = bool(requested_selected_groups & TRACKING_GROUPS)
    if args.only_default_tracking and not requested_tracking:
        raise ValueError(
            "--only-default-tracking requires ot or repair to be selected"
        )
    effective_selected_groups = (
        requested_selected_groups & TRACKING_GROUPS
        if bool(args.only_default_tracking)
        else set(requested_selected_groups)
    )
    tracking_selected = bool(effective_selected_groups & TRACKING_GROUPS)
    detection_selected = bool(
        effective_selected_groups & {"gmm", "rtv"}
    )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if tracking_selected and args.max_tracks is not None:
        raise ValueError(
            "--max-tracks is not supported for tracking groups because query "
            "and reference identities cannot be capped without alignment bias"
        )
    if args.max_tracks is not None and args.max_tracks <= 0:
        raise ValueError("--max-tracks must be positive")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 0 <= args.test_fraction < 1:
        raise ValueError("--test-fraction must be in [0, 1)")

    annotation_path = _required_path(args.annotation_path, "--annotation-path")
    ignition_module_path = _required_path(
        args.ignition_module_path, "--ignition-module-path"
    )
    fixed_roi_dir: Optional[Path] = None
    if detection_selected:
        fixed_roi_dir = _required_path(
            args.fixed_roi_dir, "--fixed-roi-dir", directory=True
        )

    tracking_module_path: Optional[Path] = None
    tracking_ablation_module_path: Optional[Path] = None
    reference_tracks_path: Optional[Path] = None
    detections_path: Optional[Path] = None
    image_dir: Optional[Path] = None
    if tracking_selected:
        tracking_module_path = _required_path(
            args.tracking_module_path, "--tracking-module-path"
        )
        tracking_ablation_module_path = _required_path(
            args.tracking_ablation_module_path,
            "--tracking-ablation-module-path",
        )
        reference_tracks_path = _required_path(
            args.reference_tracks_csv, "--reference-tracks-csv"
        )
        detections_path = _required_path(
            args.detections_path, "--detections-path", directory=True
        )
        image_dir = _required_path(args.image_dir, "--image-dir", directory=True)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else _default_output_dir().resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ignition_ablation = None
    tracking_ablation: Any = (
        load_tracking_ablation_module(tracking_ablation_module_path)
        if tracking_selected and tracking_ablation_module_path is not None
        else None
    )
    ignition_mod = _load_module(str(ignition_module_path), "ignition")
    reference_ignition = pd.DataFrame(
        columns=["track_id", "gt_onset_frame"]
    )

    track_folders: list[tuple[str, Any]] = []
    folder_names: set[str] = set()
    if detection_selected:
        track_folders = _normalize_track_folders(
            ignition_mod.list_track_folders(str(fixed_roi_dir)),
            formal_module=ignition_mod,
        )
        if not track_folders:
            raise RuntimeError(f"No track folders found under {fixed_roi_dir}")
        folder_names = {name for name, _ in track_folders}

    # Tracking reference must be read/validated before manual GT transfer.
    reference_tracks: Optional[pd.DataFrame] = None
    reference_validation: Optional[dict[str, Any]] = None
    if tracking_selected:
        reference_tracks = pd.read_csv(
            reference_tracks_path, encoding="utf-8-sig"
        )
        reference_validation = validate_reference_tracks_long_format(
            reference_tracks, reference_tracks_path
        )
        print(
            "reference "
            f"rows={reference_validation['rows']} "
            f"tracks={reference_validation['tracks']} "
            f"frames={reference_validation['frame_min']}"
            f"..{reference_validation['frame_max']}"
        )
        _atomic_write_json(
            output_dir / "reference_validation.json",
            reference_validation,
        )
        pd.DataFrame(
            _flatten_diagnostic_values(
                "reference_validation", reference_validation
            )
        ).to_csv(
            output_dir / "reference_validation.csv",
            index=False,
            encoding="utf-8-sig",
        )

    gt_format = detect_sensitivity_gt_format(annotation_path)
    gt_plan = load_sensitivity_gt(
        annotation_path,
        reference_ignition,
        source_tracks_path=None,
        crop_index_path=None,
        reference_tracks=reference_tracks,
        ignition_ablation=ignition_ablation,
        ignition_mod=ignition_mod,
        tracking_ablation=tracking_ablation,
        onset_tolerance=int(args.onset_tolerance),
        selected_groups=effective_selected_groups,
        output_dir=output_dir,
        reference_tracks_path=reference_tracks_path,
        detection_track_names=tuple(folder_names),
    )
    annotations = gt_plan.detection_annotations
    tracking_annotations = gt_plan.tracking_annotations
    if tracking_selected:
        if reference_tracks is None:
            raise RuntimeError(
                "reference_tracks must be loaded before reference ignition "
                "annotations are materialized"
            )
        reference_ignition = (
            build_reference_ignition_from_tracking_annotations(
                tracking_annotations
            )
        )
        reference_ignition_coverage = _ignition_identity_coverage(
            reference_ignition,
            "reference_ignition",
            reference_tracks,
            "reference_tracks",
        )
        positive_count = int(
            _truthy_series(reference_ignition["gt_has_ignition"]).sum()
        )
        print(
            "reference ignition "
            f"rows={len(reference_ignition)} "
            f"positive={positive_count} "
            f"negative={len(reference_ignition) - positive_count} "
            f"coverage={reference_ignition_coverage['coverage']:.3f}"
        )
    listed_ids: Sequence[int] = tuple(
        int(value)
        for value in gt_plan.metadata.get("listed_no_ignition_ids", ())
    )

    if tracking_selected:
        tracker_defaults, gmm_defaults = load_formal_defaults(
            str(tracking_module_path),
            str(ignition_module_path),
        )
    else:
        tracker_defaults = TrackerSensitivityConfig()
        gmm_defaults = ignition_mod.IgnitionGMMConfig()
        if not is_dataclass(gmm_defaults):
            raise TypeError("Formal IgnitionGMMConfig must be a dataclass")
    experiments = build_experiments(
        effective_selected_groups,
        tracker_default=tracker_defaults,
        gmm_default=gmm_defaults,
    )
    execution_mode = (
        "canonical_default_tracking"
        if bool(args.only_default_tracking)
        else "sensitivity_sweep"
    )
    if args.only_default_tracking:
        experiments = [
            select_canonical_default_tracking_experiment(
                experiments, effective_selected_groups
            )
        ]
    elif tracking_selected:
        canonical_default = select_canonical_default_tracking_experiment(
            experiments, effective_selected_groups
        )
        experiments = [
            canonical_default,
            *[
                experiment
                for experiment in experiments
                if experiment.config_id != canonical_default.config_id
            ],
        ]
    effective_evaluate_all_tracks, effective_bootstrap_samples = (
        resolve_tracking_execution_protocol(args)
    )

    candidate_names = resolve_candidate_source_identities(
        source_universe=gt_plan.source_universe,
        detection_annotation_names=gt_plan.detection_annotations,
        detection_folder_names=folder_names,
        source_to_reference=gt_plan.source_to_reference,
        detection_selected=detection_selected,
        tracking_selected=tracking_selected,
    )
    if not candidate_names:
        raise RuntimeError(
            "No GT-eligible source identities are available for the effective "
            f"selected groups: detection={detection_selected}, "
            f"tracking={tracking_selected}. Check annotation coverage, fixed "
            "ROI folders, and source→T3 mappings."
        )
    selected_annotations = resolve_split_annotation_view(
        gt_plan,
        candidate_names,
        detection_selected=detection_selected,
        tracking_selected=tracking_selected,
    )
    records = [
        {
            "track_name": name,
            "track_id": (
                ignition_mod.track_id_from_name(name)
                if callable(getattr(ignition_mod, "track_id_from_name", None))
                else None
            ),
            "gt": selected_annotations[name],
        }
        for name in candidate_names
    ]
    split_df, evaluation_names, protocol = resolve_dataset_split(
        records,
        output_dir,
        dataset_split_csv=args.dataset_split_csv,
        evaluate_all_tracks=effective_evaluate_all_tracks,
        test_fraction=float(args.test_fraction),
        seed=int(args.split_seed),
        split_module=None,
        max_tracks=args.max_tracks,
    )
    evaluation_annotations = _filter_annotations(
        selected_annotations, evaluation_names
    )
    evaluation_tracking_annotations: dict[str, Mapping[str, Any]] = {}
    evaluation_identity_aliases: dict[str, str] = {}
    if tracking_selected:
        tracking_keys = resolve_tracking_evaluation_keys(
            evaluation_names, gt_plan.source_to_reference
        )
        evaluation_tracking_annotations = _filter_annotations(
            tracking_annotations, tracking_keys
        )
        evaluation_identity_aliases = build_evaluation_identity_aliases(
            tracking_annotations=evaluation_tracking_annotations,
            reference_to_source=gt_plan.reference_to_source,
        )

    feature_cache: dict[str, FeatureCacheEntry] = {}
    feature_cache_stats = {
        "hits": 0,
        "rebuilt": 0,
        "extracted": 0,
        "total": 0,
    }
    if detection_selected:
        selected_folders = [
            (name, folder)
            for name, folder in track_folders
            if name in set(evaluation_names)
        ]
        feature_cache = load_or_build_feature_cache(
            selected_folders,
            output_dir / "feature_cache",
            gmm_defaults,
            ignition_module_path=str(ignition_module_path),
            ignition_mod=ignition_mod,
            workers=int(args.workers),
            cache_stats=feature_cache_stats,
        )
        feature_cache = {
            name: feature_cache[name] for name in evaluation_names
        }

    tracker_mod: Any = None
    detections: Mapping[int, Sequence[Mapping[str, Any]]] = {}
    frame_map: Mapping[int, Path] = {}
    image_name_map: Mapping[str, Path] = {}
    fixed_context: Optional[FixedTrackingContext] = None
    reference_hash = ""
    if tracking_selected:
        if reference_tracks is None:
            raise RuntimeError("reference_tracks were not preloaded")
        tracker_mod = _load_module(str(tracking_module_path), "tracker")
        detections = load_tracking_detections(
            tracking_ablation,
            tracker_mod,
            detections_path,
            max_txt=int(args.max_txt),
        )
        frame_builder = getattr(tracker_mod, "build_frame_image_map", None)
        if not callable(frame_builder):
            raise AttributeError(
                "Formal tracker module must expose build_frame_image_map"
            )
        frame_map = frame_builder(Path(image_dir), args.image_exts)
        if not frame_map:
            raise RuntimeError(f"No images found under {image_dir}")
        image_name_map = {
            Path(path).name: Path(path) for path in frame_map.values()
        }
        fixed_context = build_fixed_tracking_context(
            reference_tracks,
            reference_ignition,
            listed_ids,
            int(args.onset_tolerance),
            tracking_ablation_mod=tracking_ablation,
            prebuilt_annotations=evaluation_tracking_annotations,
        )
        reference_hash = fixed_context.content_hash

    source_test_hash = _test_track_set_hash(evaluation_names)
    tracking_target_hash = (
        _test_track_set_hash(tuple(evaluation_tracking_annotations))
        if tracking_selected
        else None
    )
    dataset_split_hash = _dataframe_semantic_hash(split_df)
    transfer_hash = gt_plan.metadata.get("manual_gt_transfer_hash")
    execution_context = build_execution_context_payload(
        gt_format=gt_plan.format,
        gt_content_hash=gt_plan.content_hash,
        manual_gt_transfer_hash=transfer_hash,
        source_test_track_set_hash=source_test_hash,
        tracking_target_track_set_hash=tracking_target_hash,
        dataset_split_hash=dataset_split_hash,
    )

    def runner(experiment: ExperimentSpec, run_dir: Path) -> RunResult:
        if experiment.group in TRACKING_GROUPS:
            if fixed_context is None:
                raise RuntimeError("Fixed tracking context was not initialized")
            return run_tracking_experiment(
                tracker_mod=tracker_mod,
                tracking_ablation=tracking_ablation,
                detections=detections,
                frame_map=frame_map,
                image_name_map=image_name_map,
                experiment=experiment,
                tracker_default=tracker_defaults,
                fixed_context=fixed_context,
                output_dir=run_dir,
                ignition_mod=ignition_mod,
                gmm_rtv_cfg=gmm_defaults,
                ignition_module_path=ignition_module_path,
                workers=int(args.workers),
                skip_existing_crops=True,
                reference_validation=reference_validation,
                t3_reproducibility_dir=args.tracking_ablation_t3_dir,
                evaluation_identity_aliases=evaluation_identity_aliases,
                gt_format=gt_plan.format,
            )
        if experiment.group == "gmm":
            return run_gmm_experiment(
                feature_cache,
                evaluation_annotations,
                experiment,
                gmm_defaults,
                ignition_mod=ignition_mod,
                ignition_ablation=ignition_ablation,
                expected_track_names=evaluation_names,
                workers=int(args.workers),
            )
        return run_rtv_experiment(
            feature_cache,
            evaluation_annotations,
            experiment,
            gmm_defaults,
            ignition_mod=ignition_mod,
            ignition_ablation=ignition_ablation,
            expected_track_names=evaluation_names,
            workers=int(args.workers),
        )

    summary, per_track = execute_configurations(
        experiments,
        output_dir,
        runner=runner,
        expected_track_names=evaluation_names,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        execution_mode=execution_mode,
        execution_context_hash=str(
            execution_context["execution_context_hash"]
        ),
        execution_context=execution_context,
    )
    summary = _apply_bootstrap_and_deltas(
        summary,
        per_track,
        samples=effective_bootstrap_samples,
        seed=int(args.bootstrap_seed),
    )
    summary = ensure_output_contract_columns(summary)
    stable = compute_stable_intervals(summary)
    output_warnings = save_group_outputs(
        summary, per_track, stable, output_dir
    )
    _default_rows, default_warnings = save_default_metrics(
        summary, output_dir
    )
    output_warnings.extend(default_warnings)
    record_output_warnings(summary, output_dir, output_warnings)
    plot_parameter_results(summary, output_dir)

    registry = build_parameter_registry(
        tracker_default=tracker_defaults, gmm_default=gmm_defaults
    )
    _atomic_write_json(
        output_dir / "experiment_config.json",
        {
            "script": SCRIPT_TITLE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "cli_arguments": vars(args),
            "selected_groups": sorted(requested_selected_groups),
            "requested_selected_groups": sorted(
                requested_selected_groups
            ),
            "effective_selected_groups": sorted(effective_selected_groups),
            "execution_mode": execution_mode,
            "only_default_tracking": bool(args.only_default_tracking),
            "effective_evaluate_all_tracks": effective_evaluate_all_tracks,
            "effective_bootstrap_samples": effective_bootstrap_samples,
            "tracker_defaults": asdict(tracker_defaults),
            "gmm_rtv_defaults": asdict(gmm_defaults),
            "module_paths": {
                "tracking": (
                    str(tracking_module_path) if tracking_module_path else None
                ),
                "tracking_ablation": (
                    str(tracking_ablation_module_path)
                    if tracking_ablation_module_path
                    else None
                ),
                "ignition": str(ignition_module_path),
            },
            "dataset_protocol": protocol,
            "dataset_split_source": (
                str(Path(args.dataset_split_csv).expanduser().resolve())
                if args.dataset_split_csv
                else None
            ),
            "dataset_split_output": str(
                (output_dir / "dataset_split.csv").resolve()
            ),
            "dataset_split_rows": int(len(split_df)),
            "evaluation_track_count": len(evaluation_names),
            "test_track_set_hash": source_test_hash,
            "gt_format": gt_plan.format,
            "gt_protocol": gt_plan.metadata,
            "gt_content_hash": gt_plan.content_hash,
            "gt_uses_reference_predictions": bool(
                gt_plan.metadata.get("uses_reference_predictions", False)
            ),
            "manual_gt_transfer_hash": transfer_hash,
            "source_test_track_set_hash": source_test_hash,
            "tracking_target_track_set_hash": tracking_target_hash,
            "dataset_split_hash": dataset_split_hash,
            "execution_context_hash": execution_context[
                "execution_context_hash"
            ],
            "split_seed": int(args.split_seed),
            "bootstrap_seed": int(args.bootstrap_seed),
            "tracking_metric_protocol": WINDOW_METRIC_PROTOCOL,
            "bootstrap_tracking_window_policy": (
                "Paired resampling of the fixed GT-positive event identities; "
                "unmatched events retain zero observation support. No ignition "
                "false-positive pool or downstream ignition metrics are used."
            ),
            "workers": {
                "requested": int(args.workers),
                "tracking_formal_batch": (
                    int(args.workers) if tracking_selected else None
                ),
                "fixed_roi_feature_extraction": (
                    int(args.workers) if detection_selected else None
                ),
                "fixed_roi_detection": (
                    int(args.workers) if detection_selected else None
                ),
                "note": (
                    "Tracking uses the formal process-worker batch API. "
                    "Fixed-ROI extraction and OFAT detection use isolated "
                    "per-track DataFrame/config copies with ordered thread maps; "
                    "no shared monkey-patching is used."
                ),
            },
            "feature_cache_stats": feature_cache_stats,
            "bootstrap_errors": summary.loc[
                summary["bootstrap_status"] == "error",
                ["config_id", "bootstrap_error"],
            ].to_dict(orient="records"),
            "output_warning": output_warnings,
            "fixed_tracking_reference_hash": reference_hash or None,
            "dummy_cost": {
                "available": any(spec.name == "dummy_cost" and not spec.skip_reason for spec in registry),
                "skipped_reason": next(
                    (
                        spec.skip_reason
                        for spec in registry
                        if spec.name == "dummy_cost" and spec.skip_reason
                    ),
                    "",
                ),
            },
            "coupling": {
                spec.name: spec.coupled_note
                for spec in registry
                if spec.coupled_note
            },
            "rtv_candidate_semantics": RTV_CANDIDATE_SEMANTICS,
            "skipped_parameters": {
                spec.name: spec.skip_reason
                for spec in registry
                if spec.skip_reason
            },
        },
    )
    return analysis_exit_code(
        summary,
        only_default_tracking=bool(args.only_default_tracking),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        groups = (
            {"ot", "repair"}
            if bool(args.tracking_only)
            else parse_groups(args.groups)
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if args.dry_run:
        try:
            return run_dry_run(
                groups,
                tracking_module_path=args.tracking_module_path,
                ignition_module_path=args.ignition_module_path,
                only_default_tracking=bool(args.only_default_tracking),
                annotation_path=str(args.annotation_path),
            )
        except (ValueError, FileNotFoundError, ImportError, AttributeError) as exc:
            parser.error(str(exc))
            return 2

    try:
        return run_full_analysis(args, groups)
    except (
        ValueError,
        FileNotFoundError,
        ImportError,
        AttributeError,
        RuntimeError,
    ) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
