#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Progressive tracking ablation with a *true* T3 prebirth rerun and aligned ignition evaluation.

Purpose
-------
T0/T1/T2/T2b/T3 are rebuilt from the same detection TXT input. T3_full_tracker
uses the original full-frame image sequence to run the production image-validated
prebirth recovery. After tracking is finished, every variant is evaluated with the
same Full 19-feature GMM-RTV ignition detector and the original effective_trace.xlsx
GT semantics.

GT semantics
------------
The original/reference T3 tracks.csv defines the GT track-ID space:
  - track listed in effective_trace.xlsx with non-empty frame_id -> ignition;
  - every other reference track -> no_ignition.
The effective_trace frame_id is treated as the crop/trajectory onset index and is
mapped to a global frame through the original reference tracks.csv. Fresh tracking
variants are matched to the original reference ID space using detected points only.
Unmatched positive GT tracks count as FN; unmatched predicted ignitions count as FP.

This keeps the tracking rerun independent while preserving the original ignition GT.

Ignition-oriented tracking observability
----------------------------------------
In addition to global tracking metrics and downstream P/R/F1, the script quantifies
how much trajectory information is available around each positive GT onset. It reports
pre-onset history coverage, local ignition-window completeness, whether a cold-to-hot
transition is observable with sufficient support on both sides of the GT onset, and
mechanism-specific near-onset contributions from rescue, gap filling, and prebirth.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd


VARIANTS = [
    "T0_center_nn",
    "T1_basic_ot_kalman",
    "T2_ot_kalman_rescue",
    "T2b_ot_kalman_rescue_gap",
    "T3_full_tracker",
]

DEFAULT_FULL_FRAME_DIR = 'inputs/MCFF0004-N001'
DEFAULT_DETECTIONS_DIR = (
    'inputs/run_20260703_104159'
    r"\postprocessed\cleaned_detections_txt"
)
DEFAULT_OLD_REFERENCE_TRACKS = (
    'inputs/run_20260703_104159'
    r"\postprocessed\tracking_output\run_20260706_105300\tracks.csv"
)


def load_module(path: Path, name: str):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        df.to_excel(stem.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")


def parse_track_id(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and pd.notna(value):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(-?\d+)$", text)
    return int(match.group(1)) if match else None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix in {".txt", ".tsv"} else ","
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    raise ValueError(f"Unsupported table format: {path}")


def build_effective_trace_gt(
    annotation_path: Path,
    old_ref_tracks: pd.DataFrame,
    *,
    onset_tolerance: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """Build binary GT in the original/reference T3 ID space.

    Only effective_trace rows with a non-empty numeric frame_id are positive.
    All other reference tracks are negative. Positive onset indices are mapped
    through the original reference trajectory to global frame_id.
    """
    ann = read_table(annotation_path).copy()
    ann.columns = [str(c).strip().lower() for c in ann.columns]
    if "track_id" not in ann.columns or "frame_id" not in ann.columns:
        raise ValueError(
            f"effective_trace must contain track_id and frame_id; got {list(ann.columns)}"
        )

    ref_ids = sorted(int(v) for v in old_ref_tracks["track_id"].dropna().unique())
    ref_id_set = set(ref_ids)
    gt: Dict[str, Dict[str, Any]] = {}
    for tid in ref_ids:
        gt[f"track_{tid}"] = {
            "gt_category": "no_ignition",
            "gt_has_ignition": False,
            "gt_onset_frame": -1,
            "gt_onset_start": -1,
            "gt_onset_end": -1,
        }

    n_positive = 0
    skipped_positive: List[int] = []
    bad_onset: List[int] = []
    for _, row in ann.iterrows():
        tid = parse_track_id(row.get("track_id"))
        if tid is None:
            continue
        onset_raw = pd.to_numeric(pd.Series([row.get("frame_id")]), errors="coerce").iloc[0]
        if pd.isna(onset_raw):
            # Explicit blank row remains no_ignition; unlisted reference tracks are also negative.
            continue
        if tid not in ref_id_set:
            skipped_positive.append(int(tid))
            continue
        onset_idx = int(float(onset_raw))
        rg = (
            old_ref_tracks[old_ref_tracks["track_id"] == tid]
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        if onset_idx < 0 or onset_idx >= len(rg):
            bad_onset.append(int(tid))
            continue
        onset_frame = int(rg.iloc[onset_idx]["frame_id"])
        gt[f"track_{tid}"] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_frame": onset_frame,
            "gt_onset_start": onset_frame - int(onset_tolerance),
            "gt_onset_end": onset_frame + int(onset_tolerance),
        }
        n_positive += 1

    n_negative = len(gt) - n_positive
    print(
        f"[GT] original_ref_tracks={len(gt)} | ignition={n_positive} | "
        f"no_ignition={n_negative} | onset_tolerance=±{onset_tolerance} frames"
    )
    if skipped_positive:
        print(
            f"[WARN] {len(set(skipped_positive))} positive GT IDs are absent from original reference; "
            f"first={sorted(set(skipped_positive))[:20]}"
        )
    if bad_onset:
        print(
            f"[WARN] {len(set(bad_onset))} positive GT rows have out-of-range onset indices; "
            f"first={sorted(set(bad_onset))[:20]}"
        )
    return gt


def build_full_gmm_rtv_config(ignition_mod: Any) -> Any:
    """Exact full decision configuration used by the aligned 4.2.2/4.2.3 experiments."""
    required = ["IgnitionGMMConfig", "extract_sequence_features", "detect_ignition_gmm_rtv"]
    missing = [name for name in required if not hasattr(ignition_mod, name)]
    if missing:
        raise RuntimeError(f"Ignition module missing API: {missing}")
    return ignition_mod.IgnitionGMMConfig(
        gmm_max_states=3,
        gmm_min_state_count=2,
        gmm_min_state_fraction=0.08,
        gmm_bic_improvement_ratio=0.03,
        min_state_run=2,
        min_cold_run=3,
        min_hot_run=3,
        min_stable_run=3,
        min_sequence_len=8,
        refine_radius=2,
        random_state=42,
        use_diff_features=True,
        require_all_prior_cold=True,
        refine_onset_with_change=False,
        enable_initial_hot_reject=True,
        enable_shape_sequence_reject=True,
        enable_raw_transition_validation=True,
    )


def classify_tracking_support(trajectory: pd.DataFrame) -> pd.Series:
    """Classify each trajectory sample by the tracking mechanism that supplied it.

    The classification is intentionally tracking-oriented rather than image-oriented:
      native   : ordinary main-association support;
      rescue   : a rescue association (commonly marked by ot_prob == -1 or a
                 gap_fill_method containing ``rescue``);
      gap      : an internally gap-filled candidate point;
      prebirth : an image-validated prebirth_candidate point.

    Precedence is prebirth > gap > rescue > native so that every frame belongs to at
    most one mechanism in the observability accounting.
    """
    if trajectory.empty:
        return pd.Series(dtype=str, index=trajectory.index)

    labels = pd.Series("native", index=trajectory.index, dtype="object")
    if "point_type" in trajectory.columns:
        point_type = trajectory["point_type"].fillna("detected").astype(str).str.strip().str.lower()
    else:
        point_type = pd.Series("detected", index=trajectory.index, dtype="object")

    rescue = pd.Series(False, index=trajectory.index)
    if "ot_prob" in trajectory.columns:
        ot_prob = pd.to_numeric(trajectory["ot_prob"], errors="coerce")
        rescue = rescue | (ot_prob.add(1.0).abs() <= 1e-9)
    if "gap_fill_method" in trajectory.columns:
        method = trajectory["gap_fill_method"].fillna("").astype(str).str.lower()
        rescue = rescue | method.str.contains("rescue", regex=False)

    labels.loc[rescue] = "rescue"
    labels.loc[point_type.eq("candidate")] = "gap"
    labels.loc[point_type.eq("prebirth_candidate")] = "prebirth"
    return labels.astype(str)


def _frame_set(values: pd.Series) -> Set[int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return {int(v) for v in numeric.tolist()}


def _window_set(start: int, end: int) -> Set[int]:
    if end < start:
        return set()
    return set(range(int(start), int(end) + 1))


def _count_in_window(frames: Set[int], window: Set[int]) -> int:
    return int(len(frames.intersection(window)))


def build_ignition_oriented_observability(
    tracks_by_variant: Dict[str, pd.DataFrame],
    old_ref_mapping_by_variant: Dict[str, Dict[int, int]],
    gt_ann: Dict[str, Dict[str, Any]],
    *,
    pre_onset_window: int,
    ignition_window_before: int,
    ignition_window_after: int,
    observable_min_pre: int,
    observable_min_post: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify trajectory information available around every positive GT onset.

    All  positive GT events remain in the table, including events not matched by a
    given tracking variant. Unmatched positives therefore receive zero coverage and
    are non-observable. This preserves an end-to-end interpretation instead of
    conditioning the observability statistics on successful tracking.

    Two types of support are reported:
      full support   : all samples supplied by the current tracking variant;
      native support : ordinary main-association samples only, excluding rescue,
                       gap-fill, and prebirth points.

    The difference between the two directly measures how much local temporal support
    was supplied by restoration mechanisms.
    """
    pre_onset_window = max(1, int(pre_onset_window))
    ignition_window_before = max(1, int(ignition_window_before))
    ignition_window_after = max(1, int(ignition_window_after))
    observable_min_pre = max(1, int(observable_min_pre))
    observable_min_post = max(1, int(observable_min_post))

    positive_gt: list[tuple[int, int]] = []
    for key, payload in gt_ann.items():
        if not bool(payload.get("gt_has_ignition", False)):
            continue
        ref_id = parse_track_id(key)
        onset = pd.to_numeric(pd.Series([payload.get("gt_onset_frame")]), errors="coerce").iloc[0]
        if ref_id is None or pd.isna(onset) or int(onset) < 0:
            continue
        positive_gt.append((int(ref_id), int(onset)))
    positive_gt.sort()

    per_gt_rows: list[dict[str, Any]] = []
    grouped_by_variant: Dict[str, Dict[int, pd.DataFrame]] = {}
    for variant in VARIANTS:
        df = tracks_by_variant.get(variant, pd.DataFrame())
        grouped: Dict[int, pd.DataFrame] = {}
        if not df.empty and "track_id" in df.columns:
            for qid, group in df.groupby("track_id", sort=False):
                grouped[int(qid)] = group.copy()
        grouped_by_variant[variant] = grouped

    for variant in VARIANTS:
        query_to_ref = {
            int(query_id): int(ref_id)
            for query_id, ref_id in old_ref_mapping_by_variant.get(variant, {}).items()
        }
        ref_to_query: Dict[int, int] = {}
        duplicate_ref_ids: Set[int] = set()
        for query_id, ref_id in query_to_ref.items():
            if ref_id in ref_to_query:
                duplicate_ref_ids.add(ref_id)
                continue
            ref_to_query[ref_id] = query_id
        if duplicate_ref_ids:
            print(
                f"[WARN] [{variant}] {len(duplicate_ref_ids)} original-reference IDs have more than "
                "one mapped query track; the first one is used for onset observability.",
                flush=True,
            )

        groups = grouped_by_variant[variant]
        for ref_id, onset in positive_gt:
            pre_window = _window_set(onset - pre_onset_window, onset - 1)
            ignition_window = _window_set(
                onset - ignition_window_before,
                onset + ignition_window_after,
            )
            obs_pre_window = _window_set(onset - ignition_window_before, onset - 1)
            # The annotated onset frame is allowed to be the first hot-side sample.
            obs_post_window = _window_set(onset, onset + ignition_window_after)
            local_window = _window_set(onset - pre_onset_window, onset + ignition_window_after)

            query_id = ref_to_query.get(ref_id)
            row: Dict[str, Any] = {
                "variant": variant,
                "gt_ref_track_id": int(ref_id),
                "gt_onset_frame": int(onset),
                "query_track_id": int(query_id) if query_id is not None else -1,
                "mapped_to_variant": bool(query_id is not None and query_id in groups),
                "pre_window_start": int(onset - pre_onset_window),
                "pre_window_end": int(onset - 1),
                "pre_window_total_frames": int(len(pre_window)),
                "ignition_window_start": int(onset - ignition_window_before),
                "ignition_window_end": int(onset + ignition_window_after),
                "ignition_window_total_frames": int(len(ignition_window)),
                "observable_min_pre": int(observable_min_pre),
                "observable_min_post": int(observable_min_post),
            }

            # Defaults keep unmatched positive GT events in the denominator.
            for prefix in ("pre", "ignition", "local"):
                row[f"{prefix}_support_frames"] = 0
                row[f"{prefix}_native_frames"] = 0
                row[f"{prefix}_restored_frames"] = 0
                row[f"{prefix}_rescue_frames"] = 0
                row[f"{prefix}_gap_frames"] = 0
                row[f"{prefix}_prebirth_frames"] = 0
            row.update(
                {
                    "pre_support_coverage": 0.0,
                    "pre_native_coverage": 0.0,
                    "pre_restoration_gain": 0.0,
                    "ignition_window_completeness": 0.0,
                    "ignition_window_native_completeness": 0.0,
                    "ignition_window_restoration_gain": 0.0,
                    "observable_pre_support_frames": 0,
                    "observable_post_support_frames": 0,
                    "observable_pre_native_frames": 0,
                    "observable_post_native_frames": 0,
                    "observable_transition_full": False,
                    "observable_transition_native": False,
                    "restoration_enabled_observable": False,
                    "track_start_frame": -1,
                    "track_end_frame": -1,
                }
            )

            if query_id is not None and query_id in groups:
                trajectory = groups[query_id].copy()
                trajectory["frame_id"] = pd.to_numeric(trajectory["frame_id"], errors="coerce")
                trajectory = trajectory[trajectory["frame_id"].notna()].copy()
                if not trajectory.empty:
                    trajectory["frame_id"] = trajectory["frame_id"].astype(int)
                    mechanism = classify_tracking_support(trajectory)
                    all_frames = _frame_set(trajectory["frame_id"])
                    native_frames = _frame_set(trajectory.loc[mechanism.eq("native"), "frame_id"])
                    rescue_frames = _frame_set(trajectory.loc[mechanism.eq("rescue"), "frame_id"])
                    gap_frames = _frame_set(trajectory.loc[mechanism.eq("gap"), "frame_id"])
                    prebirth_frames = _frame_set(trajectory.loc[mechanism.eq("prebirth"), "frame_id"])
                    restored_frames = rescue_frames | gap_frames | prebirth_frames

                    row["track_start_frame"] = int(min(all_frames)) if all_frames else -1
                    row["track_end_frame"] = int(max(all_frames)) if all_frames else -1

                    for prefix, window in (
                        ("pre", pre_window),
                        ("ignition", ignition_window),
                        ("local", local_window),
                    ):
                        row[f"{prefix}_support_frames"] = _count_in_window(all_frames, window)
                        row[f"{prefix}_native_frames"] = _count_in_window(native_frames, window)
                        row[f"{prefix}_restored_frames"] = _count_in_window(restored_frames, window)
                        row[f"{prefix}_rescue_frames"] = _count_in_window(rescue_frames, window)
                        row[f"{prefix}_gap_frames"] = _count_in_window(gap_frames, window)
                        row[f"{prefix}_prebirth_frames"] = _count_in_window(prebirth_frames, window)

                    row["pre_support_coverage"] = float(
                        row["pre_support_frames"] / max(len(pre_window), 1)
                    )
                    row["pre_native_coverage"] = float(
                        row["pre_native_frames"] / max(len(pre_window), 1)
                    )
                    row["pre_restoration_gain"] = float(
                        row["pre_support_coverage"] - row["pre_native_coverage"]
                    )
                    row["ignition_window_completeness"] = float(
                        row["ignition_support_frames"] / max(len(ignition_window), 1)
                    )
                    row["ignition_window_native_completeness"] = float(
                        row["ignition_native_frames"] / max(len(ignition_window), 1)
                    )
                    row["ignition_window_restoration_gain"] = float(
                        row["ignition_window_completeness"]
                        - row["ignition_window_native_completeness"]
                    )

                    obs_pre_support = _count_in_window(all_frames, obs_pre_window)
                    obs_post_support = _count_in_window(all_frames, obs_post_window)
                    obs_pre_native = _count_in_window(native_frames, obs_pre_window)
                    obs_post_native = _count_in_window(native_frames, obs_post_window)
                    row["observable_pre_support_frames"] = obs_pre_support
                    row["observable_post_support_frames"] = obs_post_support
                    row["observable_pre_native_frames"] = obs_pre_native
                    row["observable_post_native_frames"] = obs_post_native
                    row["observable_transition_full"] = bool(
                        obs_pre_support >= observable_min_pre
                        and obs_post_support >= observable_min_post
                    )
                    row["observable_transition_native"] = bool(
                        obs_pre_native >= observable_min_pre
                        and obs_post_native >= observable_min_post
                    )
                    row["restoration_enabled_observable"] = bool(
                        row["observable_transition_full"]
                        and not row["observable_transition_native"]
                    )

            per_gt_rows.append(row)

    per_gt = pd.DataFrame(per_gt_rows)
    summary_rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = per_gt[per_gt["variant"] == variant].copy()
        n_gt = int(len(group))
        row = {
            "variant": variant,
            "n_gt_positive": n_gt,
            "n_gt_mapped": int(group["mapped_to_variant"].astype(bool).sum()),
            "gt_mapping_rate": float(group["mapped_to_variant"].astype(bool).mean()) if n_gt else 0.0,
            "mean_pre_onset_coverage": float(group["pre_support_coverage"].mean()) if n_gt else 0.0,
            "mean_pre_onset_native_coverage": float(group["pre_native_coverage"].mean()) if n_gt else 0.0,
            "mean_pre_onset_restoration_gain": float(group["pre_restoration_gain"].mean()) if n_gt else 0.0,
            "mean_ignition_window_completeness": float(group["ignition_window_completeness"].mean()) if n_gt else 0.0,
            "mean_ignition_window_native_completeness": float(group["ignition_window_native_completeness"].mean()) if n_gt else 0.0,
            "mean_ignition_window_restoration_gain": float(group["ignition_window_restoration_gain"].mean()) if n_gt else 0.0,
            "observable_transition_count": int(group["observable_transition_full"].astype(bool).sum()),
            "observable_transition_rate": float(group["observable_transition_full"].astype(bool).mean()) if n_gt else 0.0,
            "native_observable_transition_count": int(group["observable_transition_native"].astype(bool).sum()),
            "native_observable_transition_rate": float(group["observable_transition_native"].astype(bool).mean()) if n_gt else 0.0,
            "restoration_enabled_observable_count": int(group["restoration_enabled_observable"].astype(bool).sum()),
            "restoration_enabled_observable_rate": float(group["restoration_enabled_observable"].astype(bool).mean()) if n_gt else 0.0,
            "n_gt_with_any_restoration_local": int((group["local_restored_frames"] > 0).sum()),
            "n_restored_frames_local": int(group["local_restored_frames"].sum()),
        }
        mapped_group = group[group["mapped_to_variant"].astype(bool)]
        if len(mapped_group):
            row.update(
                {
                    "mean_pre_onset_coverage_mapped": float(mapped_group["pre_support_coverage"].mean()),
                    "mean_ignition_window_completeness_mapped": float(
                        mapped_group["ignition_window_completeness"].mean()
                    ),
                    "observable_transition_rate_mapped": float(
                        mapped_group["observable_transition_full"].astype(bool).mean()
                    ),
                }
            )
        else:
            row.update(
                {
                    "mean_pre_onset_coverage_mapped": 0.0,
                    "mean_ignition_window_completeness_mapped": 0.0,
                    "observable_transition_rate_mapped": 0.0,
                }
            )
        for mechanism in ("rescue", "gap", "prebirth"):
            row[f"n_gt_with_{mechanism}_pre"] = int((group[f"pre_{mechanism}_frames"] > 0).sum())
            row[f"n_{mechanism}_frames_pre"] = int(group[f"pre_{mechanism}_frames"].sum())
            row[f"n_gt_with_{mechanism}_ignition_window"] = int(
                (group[f"ignition_{mechanism}_frames"] > 0).sum()
            )
            row[f"n_{mechanism}_frames_ignition_window"] = int(
                group[f"ignition_{mechanism}_frames"].sum()
            )
            row[f"n_gt_with_{mechanism}_local"] = int((group[f"local_{mechanism}_frames"] > 0).sum())
            row[f"n_{mechanism}_frames_local"] = int(group[f"local_{mechanism}_frames"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Mechanism-specific progressive contrasts.  Rescue and gap filling are judged
    # on events where their samples occur in the ± ignition window, while prebirth
    # is judged on events where prebirth samples occur in the pre-onset history.
    contrasts = [
        ("Rescue", "T1_basic_ot_kalman", "T2_ot_kalman_rescue", "rescue", "ignition"),
        ("Gap filling", "T2_ot_kalman_rescue", "T2b_ot_kalman_rescue_gap", "gap", "ignition"),
        ("Prebirth", "T2b_ot_kalman_rescue_gap", "T3_full_tracker", "prebirth", "pre"),
    ]
    contrast_rows: list[dict[str, Any]] = []
    for mechanism_name, before_variant, after_variant, mechanism_key, affected_window in contrasts:
        before = per_gt[per_gt["variant"] == before_variant].set_index("gt_ref_track_id")
        after = per_gt[per_gt["variant"] == after_variant].set_index("gt_ref_track_id")
        common_ids = before.index.intersection(after.index)
        before = before.loc[common_ids]
        after = after.loc[common_ids]
        affected_col = f"{affected_window}_{mechanism_key}_frames"
        affected = after[affected_col] > 0

        before_obs = before["observable_transition_full"].astype(bool)
        after_obs = after["observable_transition_full"].astype(bool)
        newly_observable = (~before_obs) & after_obs
        lost_observable = before_obs & (~after_obs)

        row = {
            "mechanism": mechanism_name,
            "before_variant": before_variant,
            "after_variant": after_variant,
            "affected_window": "GT onset ± local window" if affected_window == "ignition" else "pre-onset history",
            "n_gt_positive": int(len(common_ids)),
            "n_affected_gt": int(affected.sum()),
            "mechanism_frames_in_affected_window": int(after.loc[affected, affected_col].sum()) if affected.any() else 0,
            "mean_pre_coverage_before": float(before["pre_support_coverage"].mean()),
            "mean_pre_coverage_after": float(after["pre_support_coverage"].mean()),
            "delta_mean_pre_coverage": float(
                after["pre_support_coverage"].mean() - before["pre_support_coverage"].mean()
            ),
            "mean_ignition_completeness_before": float(before["ignition_window_completeness"].mean()),
            "mean_ignition_completeness_after": float(after["ignition_window_completeness"].mean()),
            "delta_mean_ignition_completeness": float(
                after["ignition_window_completeness"].mean()
                - before["ignition_window_completeness"].mean()
            ),
            "observable_rate_before": float(before_obs.mean()),
            "observable_rate_after": float(after_obs.mean()),
            "delta_observable_rate": float(after_obs.mean() - before_obs.mean()),
            "newly_observable_gt": int(newly_observable.sum()),
            "lost_observable_gt": int(lost_observable.sum()),
            "net_observable_gain": int(newly_observable.sum() - lost_observable.sum()),
        }
        if affected.any():
            before_aff = before.loc[affected]
            after_aff = after.loc[affected]
            before_aff_obs = before_aff["observable_transition_full"].astype(bool)
            after_aff_obs = after_aff["observable_transition_full"].astype(bool)
            row.update(
                {
                    "affected_mean_pre_coverage_before": float(before_aff["pre_support_coverage"].mean()),
                    "affected_mean_pre_coverage_after": float(after_aff["pre_support_coverage"].mean()),
                    "affected_delta_pre_coverage": float(
                        after_aff["pre_support_coverage"].mean()
                        - before_aff["pre_support_coverage"].mean()
                    ),
                    "affected_mean_ignition_completeness_before": float(
                        before_aff["ignition_window_completeness"].mean()
                    ),
                    "affected_mean_ignition_completeness_after": float(
                        after_aff["ignition_window_completeness"].mean()
                    ),
                    "affected_delta_ignition_completeness": float(
                        after_aff["ignition_window_completeness"].mean()
                        - before_aff["ignition_window_completeness"].mean()
                    ),
                    "affected_observable_rate_before": float(before_aff_obs.mean()),
                    "affected_observable_rate_after": float(after_aff_obs.mean()),
                    "affected_delta_observable_rate": float(
                        after_aff_obs.mean() - before_aff_obs.mean()
                    ),
                    "affected_newly_observable_gt": int(((~before_aff_obs) & after_aff_obs).sum()),
                    "affected_lost_observable_gt": int((before_aff_obs & (~after_aff_obs)).sum()),
                    "affected_mean_added_local_support_frames": float(
                        (after_aff["local_support_frames"] - before_aff["local_support_frames"]).mean()
                    ),
                }
            )
        else:
            row.update(
                {
                    "affected_mean_pre_coverage_before": 0.0,
                    "affected_mean_pre_coverage_after": 0.0,
                    "affected_delta_pre_coverage": 0.0,
                    "affected_mean_ignition_completeness_before": 0.0,
                    "affected_mean_ignition_completeness_after": 0.0,
                    "affected_delta_ignition_completeness": 0.0,
                    "affected_observable_rate_before": 0.0,
                    "affected_observable_rate_after": 0.0,
                    "affected_delta_observable_rate": 0.0,
                    "affected_newly_observable_gt": 0,
                    "affected_lost_observable_gt": 0,
                    "affected_mean_added_local_support_frames": 0.0,
                }
            )
        contrast_rows.append(row)

    return per_gt, summary, pd.DataFrame(contrast_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild T0→T3 with true image-based prebirth, then run the same Full GMM-RTV "
            "on every variant and evaluate against the original effective_trace GT."
        )
    )
    p.add_argument(
        "--aligned-script-path",
        default="run_tracking_ablation_experiments_gmm_rtv_effective_trace_positive_gt_v2.py",
        help="Aligned tracking-ablation helper script reused for tracking/cropping/evaluation.",
    )
    p.add_argument("--tracking-module-path", default="ot_kalman_tracker.py")
    p.add_argument("--ignition-module-path", default="ignition_detection_gmm_rtv_new.py")
    p.add_argument("--detections-txt-dir", default=DEFAULT_DETECTIONS_DIR)
    p.add_argument("--max-txt", type=int, default=7996)
    p.add_argument(
        "--prebirth-image-dir",
        default=DEFAULT_FULL_FRAME_DIR,
        help=(
            "Original full-frame image folder. Used for true T3 prebirth and for downstream "
            "ROI cropping of every tracking variant."
        ),
    )
    p.add_argument("--image-exts", default=".png,.jpg,.jpeg,.tif,.tiff")
    p.add_argument("--event-annotation-csv", default="effective_trace.xlsx")
    p.add_argument(
        "--original-reference-tracks-csv",
        default=DEFAULT_OLD_REFERENCE_TRACKS,
        help="Old/reference T3 tracks.csv whose track IDs are used by effective_trace.xlsx.",
    )
    p.add_argument("--gt-onset-tolerance", type=int, default=2)

    # Ignition-oriented tracking observability around the manual GT onset.
    p.add_argument(
        "--pre-onset-window",
        type=int,
        default=10,
        help="Number of frames immediately before GT onset used to quantify cold-history coverage.",
    )
    p.add_argument(
        "--ignition-window-before",
        type=int,
        default=5,
        help="Frames before GT onset included in the local ignition-window completeness metric.",
    )
    p.add_argument(
        "--ignition-window-after",
        type=int,
        default=5,
        help="Frames from/after GT onset included in the local ignition-window completeness metric.",
    )
    p.add_argument(
        "--observable-min-pre",
        type=int,
        default=3,
        help="Minimum supported frames before GT onset required for an observable transition.",
    )
    p.add_argument(
        "--observable-min-post",
        type=int,
        default=3,
        help="Minimum supported frames from/after GT onset required for an observable transition.",
    )
    p.add_argument("--output-dir", default=None)

    # Tracking settings: same as aligned tracking ablation.
    p.add_argument("--max-center-dist", type=float, default=45.0)
    p.add_argument("--max-age", type=int, default=5)
    p.add_argument("--max-age-confirmed", type=int, default=5)
    p.add_argument("--max-age-tentative", type=int, default=1)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--sinkhorn-reg", type=float, default=0.08)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)

    # Track-ID alignment thresholds.
    p.add_argument("--track-match-max-dist", type=float, default=30.0)
    p.add_argument("--track-match-min-shared-frames", type=int, default=3)
    p.add_argument("--track-match-min-ref-coverage", type=float, default=0.20)
    p.add_argument("--track-match-min-query-coverage", type=float, default=0.20)
    p.add_argument("--track-match-min-temporal-iou", type=float, default=0.15)
    p.add_argument("--track-match-max-cost", type=float, default=1.50)

    # Downstream ignition evaluation.
    p.add_argument("--crop-padding", type=float, default=4.0)
    p.add_argument("--crop-format", choices=["png", "jpg", "jpeg"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--skip-existing-crops", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="GMM-RTV worker processes. On Windows 2-4 is recommended; if a parallel pool crashes, the current variant automatically retries with workers=1.")
    p.add_argument("--save-review-plots", action="store_true")

    p.add_argument(
        "--allow-zero-prebirth",
        action="store_true",
        help="Diagnostic only: allow T3 to finish even when no prebirth points were recovered.",
    )
    return p.parse_args()


def resolve_local(path_text: str, script_dir: Path) -> Path:
    p = Path(path_text)
    if p.is_file() or p.is_dir():
        return p.resolve()
    return (script_dir / p).resolve()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    aligned_path = resolve_local(args.aligned_script_path, script_dir)
    tracker_path = resolve_local(args.tracking_module_path, script_dir)
    ignition_path = resolve_local(args.ignition_module_path, script_dir)
    detections_dir = Path(args.detections_txt_dir).resolve()
    full_frame_dir = Path(args.prebirth_image_dir).resolve()
    old_ref_path = Path(args.original_reference_tracks_csv).resolve()
    annotation_path = resolve_local(args.event_annotation_csv, script_dir)

    if int(args.pre_onset_window) < 1:
        raise ValueError("--pre-onset-window must be >= 1")
    if int(args.ignition_window_before) < 1 or int(args.ignition_window_after) < 1:
        raise ValueError("--ignition-window-before/after must both be >= 1")
    if int(args.observable_min_pre) < 1 or int(args.observable_min_post) < 1:
        raise ValueError("--observable-min-pre/post must both be >= 1")
    if int(args.observable_min_pre) > int(args.ignition_window_before):
        raise ValueError(
            "--observable-min-pre cannot exceed --ignition-window-before because only pre-onset "
            "frames in that local window are eligible."
        )
    if int(args.observable_min_post) > int(args.ignition_window_after) + 1:
        raise ValueError(
            "--observable-min-post cannot exceed --ignition-window-after + 1 because the GT onset "
            "frame itself is counted on the post/hot side."
        )

    if not detections_dir.is_dir():
        raise FileNotFoundError(f"Detection TXT folder not found: {detections_dir}")
    if not full_frame_dir.is_dir():
        raise FileNotFoundError(
            f"Full-frame image folder not found: {full_frame_dir}\n"
            "True prebirth and aligned downstream ROI cropping require the original full frames."
        )
    if not old_ref_path.is_file():
        raise FileNotFoundError(f"Original reference tracks CSV not found: {old_ref_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"effective_trace annotation not found: {annotation_path}")

    base = load_module(aligned_path, "tracking_ablation_aligned_runtime")
    tracker_mod = load_module(tracker_path, "tracking_core_runtime")
    ignition_mod = load_module(ignition_path, "ignition_runtime")

    required_base = [
        "load_detections_txt_dir_compat",
        "run_center_nn_tracker",
        "run_ot_kalman_variant",
        "variant_tracker_config",
        "aggregate_tracking_metrics",
        "match_tracks_to_reference",
        "filter_detected_track_points",
        "build_frame_image_map",
        "crop_all_tracks",
        "run_gmm_rtv_batch",
        "project_ignition_to_ref_ids",
        "evaluate_effective_trace_new_aligned",
        "compute_onset_metrics_on_ids",
    ]
    for name in required_base:
        if not hasattr(base, name):
            raise RuntimeError(f"Aligned script missing helper: {name}")
    if not hasattr(tracker_mod, "recover_prebirth_dark_particles"):
        raise RuntimeError("Tracking module has no recover_prebirth_dark_particles()")

    gmm_cfg = build_full_gmm_rtv_config(ignition_mod)
    old_ref = pd.read_csv(old_ref_path, encoding="utf-8-sig")
    gt_ann = build_effective_trace_gt(
        annotation_path,
        old_ref,
        onset_tolerance=int(args.gt_onset_tolerance),
    )
    old_ref_detected = base.filter_detected_track_points(old_ref)
    n_old_ref = int(old_ref_detected["track_id"].nunique())

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path("tracking_true_prebirth_with_ignition_results")
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] aligned_script={aligned_path}")
    print(f"[INFO] tracker_module={tracker_path}")
    print(f"[INFO] ignition_module={ignition_path}")
    print(f"[INFO] detections={detections_dir}")
    print(f"[INFO] full_frames={full_frame_dir}")
    print(f"[INFO] original_GT_reference={old_ref_path} ({n_old_ref} detected tracks)")
    print(f"[INFO] effective_trace={annotation_path}")
    print(f"[INFO] GMM workers={args.workers}")
    if int(args.workers) > 4:
        print("[WARN] Windows multiprocessing with >4 GMM workers can be memory/native-thread intensive; automatic serial fallback is enabled.")

    dets_by_frame = base.load_detections_txt_dir_compat(
        tracker_mod, detections_dir, max_txt=int(args.max_txt)
    )
    if not dets_by_frame:
        raise RuntimeError("No detections loaded")
    print(
        f"[INFO] frames={len(dets_by_frame)} "
        f"detections={sum(len(v) for v in dets_by_frame.values())}"
    )

    # Build a global full-frame index once; resolve_source_image also uses image_name.
    frame_image_map, image_name_map = base.build_frame_image_map(
        full_frame_dir, args.image_exts
    )
    if not frame_image_map:
        raise RuntimeError(f"No full-frame images found in: {full_frame_dir}")

    metrics_by_variant: Dict[str, Dict[str, Any]] = {}
    tracks_by_variant: Dict[str, pd.DataFrame] = {}
    ignition_by_variant: Dict[str, pd.DataFrame] = {}
    old_ref_mapping_by_variant: Dict[str, Dict[int, int]] = {}
    tp_onset_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}
    fresh_alignment_tables: List[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # 1) Rebuild T0 -> T3. Only T3 uses the full frames during tracking
    #    itself (for true prebirth recovery).
    # ------------------------------------------------------------------
    for variant in VARIANTS:
        vdir = output_root / variant
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 78}\n[{variant}] TRACKING\n{'=' * 78}")

        if variant == "T0_center_nn":
            tracks, run_stats = base.run_center_nn_tracker(
                dets_by_frame,
                max_center_dist=args.max_center_dist,
                max_age=args.max_age,
                max_age_tentative=args.max_age_tentative,
                max_age_confirmed=args.max_age_confirmed,
                min_hits=args.min_hits,
            )
            tracks = [track for track in tracks if bool(track.is_confirmed)]
            tracks, filter_stats = tracker_mod.postprocess_tracks(
                tracks,
                kf_params={},
                enable_gap_fill=False,
                min_contiguous_observed=args.min_hits,
                min_observed_hits=args.min_hits,
            )
            run_stats["removed_short_tracks"] = int(filter_stats.get("removed_short", 0))
        else:
            cfg = base.variant_tracker_config(variant, args)
            image_dir = full_frame_dir if variant == "T3_full_tracker" else None
            tracks, run_stats = base.run_ot_kalman_variant(
                tracker_mod,
                dets_by_frame,
                cfg,
                image_dir=image_dir,
                image_exts=args.image_exts,
                progress_label=variant,
            )

        tracker_mod.save_tracks_csv(tracks, vdir / "tracks_long.csv", min_track_len=2)
        tracker_mod.save_track_summary_csv(tracks, vdir / "track_summary.csv", min_track_len=2)
        tracks_df = pd.read_csv(vdir / "tracks_long.csv", encoding="utf-8-sig")
        track_summary_df = pd.read_csv(vdir / "track_summary.csv", encoding="utf-8-sig")
        tracks_by_variant[variant] = tracks_df

        metrics = base.aggregate_tracking_metrics(tracks_df, track_summary_df, run_stats)
        metrics["variant"] = variant
        metrics["prebirth_points_added_run"] = int(run_stats.get("prebirth_points_added", 0))
        metrics["prebirth_tracks_recovered_run"] = int(run_stats.get("tracks_recovered", 0))
        metrics_by_variant[variant] = metrics

        n_prebirth = 0
        if "point_type" in tracks_df.columns:
            n_prebirth = int(
                (tracks_df["point_type"].fillna("").astype(str) == "prebirth_candidate").sum()
            )
        print(
            f"[{variant}] tracks={tracks_df['track_id'].nunique()} "
            f"mean_len={metrics.get('mean_track_len')} "
            f"short_ratio={metrics.get('short_track_ratio')} "
            f"prebirth_points={n_prebirth}"
        )

    # Verify true T3 before downstream ignition evaluation.
    t3_df = tracks_by_variant["T3_full_tracker"]
    if "point_type" not in t3_df.columns:
        raise RuntimeError("T3 output has no point_type column; cannot verify prebirth recovery")
    n_t3_prebirth = int(
        (t3_df["point_type"].fillna("").astype(str) == "prebirth_candidate").sum()
    )
    if n_t3_prebirth == 0 and not args.allow_zero_prebirth:
        raise RuntimeError(
            "T3_full_tracker produced zero prebirth_candidate points. Check that the full-frame "
            "sequence matches the detection TXT frame numbering."
        )

    # ------------------------------------------------------------------
    # 2) Structural alignment to fresh T3 for paper Ref. cov.
    # ------------------------------------------------------------------
    ref_detected = base.filter_detected_track_points(t3_df)
    n_fresh_ref = int(ref_detected["track_id"].nunique())
    for variant in VARIANTS:
        qdf = tracks_by_variant[variant]
        if variant == "T3_full_tracker":
            qdet = base.filter_detected_track_points(qdf)
            qids = sorted(int(v) for v in qdet["track_id"].unique())
            mapping = {tid: tid for tid in qids}
            diag = pd.DataFrame(
                {
                    "query_track_id": qids,
                    "ref_track_id": qids,
                    "n_shared_frames": [
                        int(qdet[qdet["track_id"] == tid]["frame_id"].nunique()) for tid in qids
                    ],
                    "mean_center_distance": 0.0,
                    "median_center_distance": 0.0,
                    "p90_center_distance": 0.0,
                    "query_coverage": 1.0,
                    "reference_coverage": 1.0,
                    "temporal_iou": 1.0,
                    "assignment_cost": 0.0,
                }
            )
        else:
            mapping, diag = base.match_tracks_to_reference(
                qdf,
                t3_df,
                max_center_dist=float(args.track_match_max_dist),
                min_shared_frames=int(args.track_match_min_shared_frames),
                min_ref_coverage=float(args.track_match_min_ref_coverage),
                min_query_coverage=float(args.track_match_min_query_coverage),
                min_temporal_iou=float(args.track_match_min_temporal_iou),
                max_assignment_cost=float(args.track_match_max_cost),
                detected_only=True,
                return_diagnostics=True,
            )
        qdet = base.filter_detected_track_points(qdf)
        n_query = int(qdet["track_id"].nunique())
        matched_fresh_ids = set(int(v) for v in mapping.values())
        metrics_by_variant[variant].update(
            {
                "n_matched_to_fresh_t3": int(len(mapping)),
                "tracking_match_rate_to_fresh_t3": float(len(mapping) / max(n_query, 1)),
                "fresh_t3_reference_coverage": float(
                    len(matched_fresh_ids) / max(n_fresh_ref, 1)
                ),
                "n_fresh_t3_reference_tracks": n_fresh_ref,
            }
        )
        if not diag.empty:
            diag = diag.copy()
            diag.insert(0, "variant", variant)
            fresh_alignment_tables.append(diag)
            save_table(diag, output_root / variant / "track_id_alignment_to_fresh_t3")

    # ------------------------------------------------------------------
    # 3) Full GMM-RTV on each fresh tracking variant using the same source
    #    images. Then align prediction IDs directly to the original GT space.
    # ------------------------------------------------------------------
    for variant in VARIANTS:
        vdir = output_root / variant
        tracks_df = tracks_by_variant[variant]
        print(f"\n{'=' * 78}\n[{variant}] FULL GMM-RTV + ORIGINAL GT\n{'=' * 78}")

        crop_root = vdir / "track_crop"
        crop_data = base.crop_all_tracks(
            tracks_df,
            crop_root,
            frame_image_map,
            image_name_map,
            float(args.crop_padding),
            crop_format=str(args.crop_format),
            jpeg_quality=int(args.jpeg_quality),
            skip_existing=bool(args.skip_existing_crops),
            progress_label=variant,
        )
        requested_workers = max(1, int(args.workers))
        try:
            ignition_rows = base.run_gmm_rtv_batch(
                crop_data,
                vdir / "gmm_rtv",
                ignition_mod,
                gmm_cfg,
                workers=requested_workers,
                save_plot=bool(args.save_review_plots),
                ignition_module_path=ignition_path,
                variant=variant,
            )
        except Exception as exc:
            # On Windows a ProcessPool worker can be terminated abruptly by the OS
            # (most commonly because several OpenCV / NumPy / sklearn workers compete
            # for memory or native-thread resources).  In that case ProcessPoolExecutor
            # raises BrokenProcessPool while submitting the next track.  The scientific
            # calculation itself is unchanged, so retry the same variant serially rather
            # than aborting the whole T0->T3 experiment.
            exc_name = type(exc).__name__
            if requested_workers <= 1:
                raise
            print(
                f"[WARN] [{variant}] parallel GMM-RTV failed with {exc_name}: {exc}\n"
                "       Retrying this SAME variant with workers=1. "
                "The detector, features, GT, and thresholds are unchanged; only the execution mode changes.",
                flush=True,
            )
            import gc
            gc.collect()
            ignition_rows = base.run_gmm_rtv_batch(
                crop_data,
                vdir / "gmm_rtv",
                ignition_mod,
                gmm_cfg,
                workers=1,
                save_plot=bool(args.save_review_plots),
                ignition_module_path=ignition_path,
                variant=variant,
            )
        ignition_df = pd.DataFrame(ignition_rows)
        ignition_by_variant[variant] = ignition_df
        save_table(ignition_df, vdir / "ignition_summary")

        # Direct detected-only mapping to the original reference ID space used by effective_trace.
        query_to_old, diag_old = base.match_tracks_to_reference(
            tracks_df,
            old_ref,
            max_center_dist=float(args.track_match_max_dist),
            min_shared_frames=int(args.track_match_min_shared_frames),
            min_ref_coverage=float(args.track_match_min_ref_coverage),
            min_query_coverage=float(args.track_match_min_query_coverage),
            min_temporal_iou=float(args.track_match_min_temporal_iou),
            max_assignment_cost=float(args.track_match_max_cost),
            detected_only=True,
            return_diagnostics=True,
        )
        old_ref_mapping_by_variant[variant] = query_to_old
        save_table(diag_old, vdir / "track_id_alignment_to_original_gt_ref")

        projected = base.project_ignition_to_ref_ids(ignition_df, query_to_old)
        save_table(projected, vdir / "ignition_projected_to_original_gt_ref")

        qdet = base.filter_detected_track_points(tracks_df)
        eval_metrics = base.evaluate_effective_trace_new_aligned(
            gt_ann,
            projected,
            query_ignition_df=ignition_df,
            query_to_ref=query_to_old,
            n_query_tracks=int(qdet["track_id"].nunique()),
            n_matched=len(query_to_old),
            n_reference_tracks=n_old_ref,
            count_unmatched_pred_as_fp=True,
        )
        tp_onset_maps[variant] = dict(eval_metrics.pop("tp_onset_by_ref_id", {}) or {})
        metrics_by_variant[variant].update(eval_metrics)

        print(
            f"[{variant}] P={metrics_by_variant[variant].get('ignition_precision', 0):.3f} "
            f"R={metrics_by_variant[variant].get('ignition_recall', 0):.3f} "
            f"F1={metrics_by_variant[variant].get('ignition_f1', 0):.3f} "
            f"TP={metrics_by_variant[variant].get('end_to_end_tp')} "
            f"FP={metrics_by_variant[variant].get('end_to_end_fp')} "
            f"FN={metrics_by_variant[variant].get('end_to_end_fn')} "
            f"onset_MAE={metrics_by_variant[variant].get('onset_mae')}"
        )

    # ------------------------------------------------------------------
    # 4) Strict Common-TP onset metrics across all five tracking variants.
    # ------------------------------------------------------------------
    common_tp_ids: Optional[Set[int]] = None
    for variant in VARIANTS:
        ids = set(int(v) for v in tp_onset_maps.get(variant, {}))
        common_tp_ids = ids if common_tp_ids is None else (common_tp_ids & ids)
    common_tp_ids = common_tp_ids or set()
    print(f"[INFO] strict Common-TP across T0/T1/T2/T2b/T3 = {len(common_tp_ids)}")
    for variant in VARIANTS:
        common_metrics = base.compute_onset_metrics_on_ids(
            tp_onset_maps.get(variant, {}),
            common_tp_ids,
            prefix="_common",
        )
        metrics_by_variant[variant].update(common_metrics)

    # ------------------------------------------------------------------
    # 5) Ignition-oriented tracking observability around the manual GT onset.
    #    This directly answers how much useful temporal support Rescue, Gap
    #    filling, and Prebirth recover where ignition determination needs it.
    # ------------------------------------------------------------------
    observability_per_gt, observability_summary, mechanism_gain = (
        build_ignition_oriented_observability(
            tracks_by_variant,
            old_ref_mapping_by_variant,
            gt_ann,
            pre_onset_window=int(args.pre_onset_window),
            ignition_window_before=int(args.ignition_window_before),
            ignition_window_after=int(args.ignition_window_after),
            observable_min_pre=int(args.observable_min_pre),
            observable_min_post=int(args.observable_min_post),
        )
    )
    save_table(observability_per_gt, output_root / "ignition_observability_per_gt")
    save_table(observability_summary, output_root / "ignition_observability_summary")
    save_table(mechanism_gain, output_root / "restoration_mechanism_near_onset_gain")

    observability_by_variant = observability_summary.set_index("variant", drop=False)
    for variant in VARIANTS:
        if variant not in observability_by_variant.index:
            continue
        obs_row = observability_by_variant.loc[variant]
        for column in observability_summary.columns:
            if column == "variant":
                continue
            metrics_by_variant[variant][column] = obs_row[column]

    print("\n=== Ignition-oriented trajectory observability ===")
    show_obs = [
        "variant",
        "mean_pre_onset_coverage",
        "mean_ignition_window_completeness",
        "observable_transition_rate",
        "restoration_enabled_observable_count",
        "n_gt_with_rescue_ignition_window",
        "n_gt_with_gap_ignition_window",
        "n_gt_with_prebirth_pre",
    ]
    print(observability_summary[[c for c in show_obs if c in observability_summary.columns]].to_string(index=False))
    print("\n=== Near-onset contribution of each restoration mechanism ===")
    print(mechanism_gain.to_string(index=False))

    # Fresh T3 -> original reference identity check (same old GT space).
    fresh_to_old = old_ref_mapping_by_variant.get("T3_full_tracker", {})
    fresh_detected = base.filter_detected_track_points(t3_df)
    matched_old = set(int(v) for v in fresh_to_old.values())
    original_ref_alignment_metrics = {
        "n_fresh_t3_detected_tracks": int(fresh_detected["track_id"].nunique()),
        "n_original_ref_detected_tracks": n_old_ref,
        "n_fresh_t3_mapped_to_original_ref": int(len(fresh_to_old)),
        "fresh_t3_mapping_rate_to_original_ref": float(
            len(fresh_to_old) / max(int(fresh_detected["track_id"].nunique()), 1)
        ),
        "original_ref_coverage_by_fresh_t3": float(len(matched_old) / max(n_old_ref, 1)),
    }
    save_table(
        pd.DataFrame([original_ref_alignment_metrics]),
        output_root / "fresh_t3_to_original_reference_metrics",
    )
    if original_ref_alignment_metrics["original_ref_coverage_by_fresh_t3"] < 0.95:
        print(
            "[WARN] Fresh T3 covers <95% of the original GT-reference ID space. "
            "Review track_id_alignment_to_original_gt_ref before interpreting ignition metrics."
        )

    # ------------------------------------------------------------------
    # 6) Summaries: full diagnostics + paper-facing tables.
    # ------------------------------------------------------------------
    summary = pd.DataFrame([metrics_by_variant[v] for v in VARIANTS])
    preferred = [
        "variant",
        "n_tracks",
        "mean_track_len",
        "median_track_len",
        "short_track_ratio",
        "fresh_t3_reference_coverage",
        "mean_pre_onset_coverage",
        "mean_pre_onset_native_coverage",
        "mean_pre_onset_restoration_gain",
        "mean_pre_onset_coverage_mapped",
        "mean_ignition_window_completeness",
        "mean_ignition_window_completeness_mapped",
        "mean_ignition_window_native_completeness",
        "mean_ignition_window_restoration_gain",
        "observable_transition_rate",
        "observable_transition_rate_mapped",
        "observable_transition_count",
        "restoration_enabled_observable_count",
        "restoration_enabled_observable_rate",
        "n_gt_with_rescue_ignition_window",
        "n_gt_with_gap_ignition_window",
        "n_gt_with_prebirth_pre",
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "onset_mae_common",
        "within_1_frame_acc_common",
        "within_3_frame_acc_common",
        "n_onset_eval_common",
        "end_to_end_tp",
        "end_to_end_fp",
        "end_to_end_fn",
        "gt_track_coverage",
        "unmatched_reference_ignition_count",
        "unmatched_predicted_ignition_count",
        "candidate_ratio",
        "prebirth_ratio",
        "n_rescue_matches",
        "n_gap_candidates",
        "prebirth_points_added_run",
        "suspicious_merge_proxy_count",
        "large_jump_track_count",
    ]
    ordered = [c for c in preferred if c in summary.columns] + [
        c for c in summary.columns if c not in preferred
    ]
    summary = summary[ordered]
    save_table(summary, output_root / "tracking_ablation_summary_with_ignition")

    paper_columns = [
        "variant",
        "mean_track_len",
        "short_track_ratio",
        "fresh_t3_reference_coverage",
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "onset_mae_common",
    ]
    paper = summary[[c for c in paper_columns if c in summary.columns]].copy()
    paper = paper.rename(
        columns={
            "mean_track_len": "Mean len.",
            "short_track_ratio": "Short ratio",
            "fresh_t3_reference_coverage": "Ref. cov.",
            "ignition_precision": "P",
            "ignition_recall": "R",
            "ignition_f1": "F1",
            "onset_mae_common": "Common MAE",
        }
    )
    save_table(paper, output_root / "table_4_2_1_paper")

    observability_paper_columns = [
        "variant",
        "mean_track_len",
        "short_track_ratio",
        "fresh_t3_reference_coverage",
        "mean_pre_onset_coverage",
        "mean_ignition_window_completeness",
        "observable_transition_rate",
        "restoration_enabled_observable_count",
    ]
    observability_paper = summary[
        [c for c in observability_paper_columns if c in summary.columns]
    ].copy()
    observability_paper = observability_paper.rename(
        columns={
            "mean_track_len": "Mean len.",
            "short_track_ratio": "Short ratio",
            "fresh_t3_reference_coverage": "Ref. cov.",
            "mean_pre_onset_coverage": f"Pre-onset cov. (-{int(args.pre_onset_window)}:-1)",
            "mean_ignition_window_completeness": (
                f"Ignition-window completeness (-{int(args.ignition_window_before)}:+{int(args.ignition_window_after)})"
            ),
            "observable_transition_rate": "Observable-transition rate",
            "restoration_enabled_observable_count": "Restoration-enabled observable GT",
        }
    )
    save_table(
        observability_paper,
        output_root / "table_4_2_1_tracking_observability",
    )

    if fresh_alignment_tables:
        save_table(
            pd.concat(fresh_alignment_tables, ignore_index=True),
            output_root / "all_alignment_to_fresh_t3",
        )

    config = {
        "aligned_script": str(aligned_path),
        "tracking_module": str(tracker_path),
        "ignition_module": str(ignition_path),
        "detections_txt_dir": str(detections_dir),
        "full_frame_dir": str(full_frame_dir),
        "original_reference_tracks_csv": str(old_ref_path),
        "event_annotation_csv": str(annotation_path),
        "max_txt": int(args.max_txt),
        "variants": VARIANTS,
        "n_t3_prebirth_points": n_t3_prebirth,
        "n_common_tp": len(common_tp_ids),
        "ignition_observability": {
            "pre_onset_window": int(args.pre_onset_window),
            "ignition_window_before": int(args.ignition_window_before),
            "ignition_window_after": int(args.ignition_window_after),
            "observable_min_pre": int(args.observable_min_pre),
            "observable_min_post": int(args.observable_min_post),
            "denominator": "all positive GT events; unmatched positives receive zero coverage",
            "native_support_definition": (
                "ordinary main-association samples only; rescue, gap-fill, and prebirth are excluded"
            ),
        },
        "original_reference_alignment": original_ref_alignment_metrics,
        "full_gmm_rtv_config": {
            "gmm_max_states": 3,
            "gmm_min_state_count": 2,
            "gmm_min_state_fraction": 0.08,
            "gmm_bic_improvement_ratio": 0.03,
            "min_state_run": 2,
            "min_cold_run": 3,
            "min_hot_run": 3,
            "min_stable_run": 3,
            "min_sequence_len": 8,
            "use_diff_features": True,
            "require_all_prior_cold": True,
            "refine_onset_with_change": False,
            "enable_initial_hot_reject": True,
            "enable_shape_sequence_reject": True,
            "enable_raw_transition_validation": True,
        },
        "gt_semantics": (
            "positive iff effective_trace contains track_id with non-empty frame_id; "
            "all other original-reference tracks are no_ignition"
        ),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Paper-facing Table 4.2.1 ===")
    print(paper.to_string(index=False))
    print(f"\n[OK] output={output_root.resolve()}")
    print("[OK] Main paper table: table_4_2_1_paper.xlsx")
    print("[OK] Tracking + GT-onset observability table: table_4_2_1_tracking_observability.xlsx")
    print("[OK] Per-GT onset observability: ignition_observability_per_gt.xlsx")
    print("[OK] Observability summary: ignition_observability_summary.xlsx")
    print("[OK] Rescue/Gap/Prebirth near-onset gains: restoration_mechanism_near_onset_gain.xlsx")
    print("[OK] Full diagnostics: tracking_ablation_summary_with_ignition.xlsx")


if __name__ == "__main__":
    main()
