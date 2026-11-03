#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute only the four parameter-sensitivity sweeps used by manuscript Fig. 3.

This script is intentionally narrower than the original ~9k-line sensitivity
framework. It keeps only the four parameters actually plotted in Fig. 3:

  (a) max_center_dist        : 32, 38, 45, 52, 59 px
  (b) gmm_max_states         : 2, 3, 4
  (c) cold_hot_persistence   : 2, 3, 4, 5 frames
  (d) rtv_alpha              : 0.6, 0.8, 1.0, 1.2, 1.4

Scientific protocol
-------------------
1) max_center_dist is an end-to-end tracking sensitivity sweep. For every
   distance, the same T3 tracker is rebuilt from the same detections, with
   Rescue + Gap filling + image-validated Prebirth enabled. Fresh ROIs are
   cropped from the original full frames and evaluated by the SAME Full
   19-feature GMM-RTV detector. The default 45-px run is used as the *fresh T3*
   structural reference for reference_track_coverage. Ignition predictions are
   independently aligned to the original GT-reference ID space and evaluated
   against effective_trace.xlsx.

2) gmm_max_states, cold_hot_persistence, and rtv_alpha are evaluated on ONE
   fixed ROI population (the same population intended for the final 4.2.2/4.2.3
   experiments). Features are extracted once and cached in memory. Only the
   selected decision parameter is changed (OFAT).

3) GT semantics are the FINAL manuscript semantics:
      positive  <=> effective_trace.xlsx contains the track AND frame_id is numeric
      negative  <=> every other track in the current evaluable population
   For fixed ROIs, frame_id is the local trajectory/onset index. For the
   tracking sweep, that local index is mapped through the original reference
   tracks.csv to a global frame before aligned evaluation.

4) cold_hot_persistence changes min_cold_run and min_hot_run together. In the
   production detector these same values also set the pre/post averaging windows
   used by raw transition validation (RTV). Therefore panel (c) is an operational
   persistence/window-length sensitivity, not an isolated GMM-only parameter.

Outputs
-------
  <output>/combined_summary.xlsx
  <output>/combined_summary.csv
  <output>/decision_per_track.csv
  <output>/logic_audit.json
  <output>/tracking_max_center_dist/<value>/...

The combined_summary schema is directly compatible with the current Fig. 3
plotter (parameter_name, parameter_value, is_default_configuration, run_status,
ignition_precision/recall/f1, false_ignition_rate, reference_track_coverage,
suspicious_merge_proxy_count, f1_ci_low/high).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Final manuscript parameter grid
# -----------------------------------------------------------------------------

TRACKING_DEFAULT_CENTER_DIST = 45.0
TRACKING_CENTER_RATIOS = (0.70, 0.85, 1.00, 1.15, 1.30)
GMM_MAX_STATES_VALUES = (2, 3, 4)
COLD_HOT_PERSISTENCE_VALUES = (2, 3, 4, 5)
RTV_ALPHA_VALUES = (0.60, 0.80, 1.00, 1.20, 1.40)

RTV_THRESHOLD_FIELDS = (
    "min_raw_delta_gray_p99",
    "min_raw_delta_corr_p99",
    "min_raw_delta_bright_energy",
    "min_raw_delta_bright_area",
    "min_raw_delta_hsv_v_p99",
)

EXPECTED_BASE_GMM_FEATURES = (
    "gray_p95", "gray_p99", "corr_p95", "corr_p99", "corr_max",
    "bright_energy_mean", "bright_area_ratio", "bright_mean",
    "red_ratio", "rg_ratio", "hsv_v_p95", "hsv_v_p99", "largest_area_ratio",
)
EXPECTED_DIFF_GMM_FEATURES = (
    "d1_gray_p99", "d1_corr_p99", "d1_bright_energy_mean",
    "d1_bright_area_ratio", "d1_red_ratio", "d1_hsv_v_p99",
)
EXPECTED_FULL_GMM_FEATURES = EXPECTED_BASE_GMM_FEATURES + EXPECTED_DIFF_GMM_FEATURES

DEFAULT_FULL_FRAME_DIR = 'inputs/MCFF0004-N001'
DEFAULT_DETECTIONS_DIR = (
    'inputs/run_20260703_104159'
    r"\postprocessed\cleaned_detections_txt"
)
DEFAULT_OLD_REFERENCE_TRACKS = (
    'inputs/run_20260703_104159'
    r"\postprocessed\tracking_output\run_20260706_105300\tracks.csv"
)
DEFAULT_FIXED_ROI_DIR = (
    'inputs/run_20260703_104159'
    r"\postprocessed\tracking_output\run_20260706_105300\track_crop"
)


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def load_module(path: Path, name: str) -> Any:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_spawn_safe_module(path: Path) -> Any:
    """Import a Python file under its real importable stem for Windows spawn.

    ProcessPoolExecutor on Windows serializes worker functions by
    ``(module_name, function_name)``.  Loading the tracking helper under a
    synthetic alias such as ``fig3_tracking_helper_runtime`` makes child
    processes try to import a module that does not exist on disk.  Importing
    under ``path.stem`` keeps the worker's ``__module__`` resolvable.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stem = path.stem
    if not stem.isidentifier():
        raise ImportError(
            f"Spawn-safe module filename must be a valid Python identifier: {path.name}"
        )
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    existing = sys.modules.get(stem)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is not None and Path(existing_file).resolve() == path:
            return existing

    spec = importlib.util.spec_from_file_location(stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import spawn-safe module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_local(path_text: str, script_dir: Path) -> Path:
    p = Path(path_text)
    if p.is_file() or p.is_dir():
        return p.resolve()
    return (script_dir / p).resolve()


def save_table(df: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    try:
        df.to_excel(stem.with_suffix(".xlsx"), index=False)
    except Exception as exc:
        print(f"[WARN] Excel export skipped for {stem}: {exc}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_track_id(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
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


def scaled_integer_values(default: float, ratios: Sequence[float]) -> list[int]:
    """Match the original half-up integer scaling (not Python banker's round)."""
    values: list[int] = []
    seen: set[int] = set()
    for ratio in ratios:
        raw = round(float(default) * float(ratio), 9)
        value = int(math.floor(raw + 0.5))
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def stable_seed(base_seed: int, parameter_name: str, value: Any) -> int:
    digest = hashlib.sha256(f"{parameter_name}|{value}".encode("utf-8")).digest()
    return int((int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


# -----------------------------------------------------------------------------
# FINAL effective_trace GT semantics
# -----------------------------------------------------------------------------

def read_effective_trace_positive_onsets(path: Path) -> Dict[int, int]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        table = pd.read_excel(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix in {".txt", ".tsv"} else ","
        table = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    else:
        raise ValueError(f"Unsupported GT format: {path}")

    table = table.copy()
    table.columns = [str(c).strip().lower() for c in table.columns]
    if "track_id" not in table.columns or "frame_id" not in table.columns:
        raise ValueError(
            f"effective_trace requires track_id and frame_id; got {list(table.columns)}"
        )

    positives: Dict[int, int] = {}
    for _, row in table.iterrows():
        tid = parse_track_id(row.get("track_id"))
        if tid is None:
            continue
        onset = pd.to_numeric(pd.Series([row.get("frame_id")]), errors="coerce").iloc[0]
        if pd.isna(onset):
            continue
        onset_idx = int(float(onset))
        if tid in positives and positives[tid] != onset_idx:
            raise ValueError(
                f"Conflicting positive onset labels for track {tid}: "
                f"{positives[tid]} vs {onset_idx}"
            )
        positives[tid] = onset_idx
    if not positives:
        raise ValueError(f"No positive GT rows with numeric frame_id found in {path}")
    return positives


def build_fixed_roi_gt(
    positive_onsets: Mapping[int, int],
    roi_track_names: Sequence[str],
    onset_tolerance: int,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """GT over the CURRENT fixed-ROI population; absent positives are excluded."""
    gt: Dict[str, Dict[str, Any]] = {}
    roi_ids: set[int] = set()
    for name in roi_track_names:
        tid = parse_track_id(name)
        if tid is None:
            raise ValueError(f"Cannot parse ROI track id from {name!r}")
        roi_ids.add(tid)
        if tid in positive_onsets:
            onset = int(positive_onsets[tid])
            gt[str(name)] = {
                "gt_category": "ignition",
                "gt_has_ignition": True,
                "gt_onset_idx": onset,
                "gt_onset_start": max(0, onset - int(onset_tolerance)),
                "gt_onset_end": onset + int(onset_tolerance),
            }
        else:
            gt[str(name)] = {
                "gt_category": "no_ignition",
                "gt_has_ignition": False,
                "gt_onset_idx": -1,
                "gt_onset_start": -1,
                "gt_onset_end": -1,
            }

    missing_positive_ids = sorted(set(int(k) for k in positive_onsets) - roi_ids)
    n_pos = sum(bool(v["gt_has_ignition"]) for v in gt.values())
    audit = {
        "semantics": (
            "positive iff effective_trace contains track_id with numeric frame_id; "
            "all other tracks in current fixed-ROI population are negative"
        ),
        "n_roi_tracks": len(gt),
        "n_evaluable_positive": int(n_pos),
        "n_implicit_negative": int(len(gt) - n_pos),
        "n_positive_labels_absent_from_roi": len(missing_positive_ids),
        "positive_labels_absent_from_roi": missing_positive_ids,
    }
    return gt, audit


def build_tracking_gt_global(
    positive_onsets: Mapping[int, int],
    old_ref_tracks: pd.DataFrame,
    onset_tolerance: int,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Map effective_trace local onset indices through original reference tracks."""
    old_ref_tracks = old_ref_tracks.copy()
    old_ref_tracks["track_id"] = pd.to_numeric(old_ref_tracks["track_id"], errors="raise").astype(int)
    old_ref_tracks["frame_id"] = pd.to_numeric(old_ref_tracks["frame_id"], errors="raise").astype(int)
    ref_ids = sorted(int(v) for v in old_ref_tracks["track_id"].unique())
    ref_set = set(ref_ids)
    gt: Dict[str, Dict[str, Any]] = {
        f"track_{tid}": {
            "gt_category": "no_ignition",
            "gt_has_ignition": False,
            "gt_onset_frame": -1,
            "gt_onset_start": -1,
            "gt_onset_end": -1,
        }
        for tid in ref_ids
    }

    missing_positive: list[int] = []
    bad_onset: list[int] = []
    n_pos = 0
    for tid, onset_idx in positive_onsets.items():
        if int(tid) not in ref_set:
            missing_positive.append(int(tid))
            continue
        rg = (
            old_ref_tracks[old_ref_tracks["track_id"] == int(tid)]
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        if int(onset_idx) < 0 or int(onset_idx) >= len(rg):
            bad_onset.append(int(tid))
            continue
        onset_frame = int(rg.iloc[int(onset_idx)]["frame_id"])
        gt[f"track_{int(tid)}"] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_frame": onset_frame,
            "gt_onset_start": onset_frame - int(onset_tolerance),
            "gt_onset_end": onset_frame + int(onset_tolerance),
        }
        n_pos += 1

    audit = {
        "semantics": (
            "effective_trace positive local onset index mapped through original "
            "reference tracks to global frame"
        ),
        "n_reference_tracks": len(ref_ids),
        "n_positive": n_pos,
        "n_negative": len(ref_ids) - n_pos,
        "missing_positive_ids": sorted(missing_positive),
        "bad_onset_ids": sorted(bad_onset),
    }
    return gt, audit


# -----------------------------------------------------------------------------
# Full production GMM-RTV config and parameter variants
# -----------------------------------------------------------------------------

def build_full_gmm_rtv_config(ignition_mod: Any) -> Any:
    required = ["IgnitionGMMConfig", "extract_sequence_features", "detect_ignition_gmm_rtv"]
    missing = [name for name in required if not hasattr(ignition_mod, name)]
    if missing:
        raise RuntimeError(f"Ignition module missing API: {missing}")
    cfg = ignition_mod.IgnitionGMMConfig(
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
    if not is_dataclass(cfg):
        raise TypeError("IgnitionGMMConfig must be a dataclass")
    for field in RTV_THRESHOLD_FIELDS:
        if not hasattr(cfg, field):
            raise ValueError(f"IgnitionGMMConfig missing RTV threshold field {field}")
    return cfg


def config_fingerprint(cfg: Any) -> str:
    payload = asdict(cfg) if is_dataclass(cfg) else vars(cfg)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def rtv_alpha_config(base_cfg: Any, alpha: float) -> Any:
    overrides = {
        field: round(float(getattr(base_cfg, field)) * float(alpha), 12)
        for field in RTV_THRESHOLD_FIELDS
    }
    return replace(base_cfg, **overrides)


def decision_parameter_specs(base_cfg: Any) -> list[tuple[str, str, Any, Any, str]]:
    """Return (group, name, value, cfg, note)."""
    specs: list[tuple[str, str, Any, Any, str]] = []
    for value in GMM_MAX_STATES_VALUES:
        specs.append((
            "gmm", "gmm_max_states", int(value),
            replace(base_cfg, gmm_max_states=int(value)),
            "Only maximum number of GMM states is changed.",
        ))
    for value in COLD_HOT_PERSISTENCE_VALUES:
        specs.append((
            "gmm", "cold_hot_persistence", int(value),
            replace(base_cfg, min_cold_run=int(value), min_hot_run=int(value)),
            (
                "Operational coupled parameter: min_cold_run=min_hot_run=L. "
                "The production RTV validator also uses these values as pre/post "
                "averaging-window lengths."
            ),
        ))
    for value in RTV_ALPHA_VALUES:
        specs.append((
            "rtv", "rtv_alpha", float(value),
            rtv_alpha_config(base_cfg, float(value)),
            "Alpha scales all five production raw-transition thresholds from the base config.",
        ))
    return specs


# -----------------------------------------------------------------------------
# Fixed-ROI feature cache and decision sensitivity
# -----------------------------------------------------------------------------

def normalize_track_folders(raw: Any) -> list[tuple[str, Path]]:
    folders: list[tuple[str, Path]] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            name, folder = str(item[0]), Path(item[1])
        else:
            folder = Path(item)
            name = folder.name
        folders.append((name, folder))
    return folders


def extract_fixed_roi_features(
    ignition_mod: Any,
    fixed_roi_dir: Path,
    cfg: Any,
    workers: int,
) -> tuple[list[tuple[str, Path]], Dict[str, pd.DataFrame]]:
    raw = ignition_mod.list_track_folders(str(fixed_roi_dir))
    tracks = normalize_track_folders(raw)
    if not tracks:
        raise RuntimeError(f"No track folders found under {fixed_roi_dir}")

    def one(item: tuple[str, Path]) -> tuple[str, pd.DataFrame]:
        name, folder = item
        paths = ignition_mod.list_images_in_folder(str(folder))
        if not paths:
            raise RuntimeError(f"No images in {folder}")
        df = ignition_mod.extract_sequence_features(paths, cfg)
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Feature extraction returned empty table for {name}")
        return name, df

    cache: Dict[str, pd.DataFrame] = {}
    if int(workers) <= 1:
        for i, item in enumerate(tracks, 1):
            name, df = one(item)
            cache[name] = df
            if i % 200 == 0 or i == len(tracks):
                print(f"[FEATURE] {i}/{len(tracks)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            for i, (name, df) in enumerate(executor.map(one, tracks), 1):
                cache[name] = df
                if i % 200 == 0 or i == len(tracks):
                    print(f"[FEATURE] {i}/{len(tracks)}", flush=True)

    # Strong alignment guard: same Full 19-feature GMM representation as 4.2.3.
    first_name = tracks[0][0]
    if hasattr(ignition_mod, "get_gmm_feature_columns"):
        cols = list(ignition_mod.get_gmm_feature_columns(cache[first_name], cfg))
        expected = list(EXPECTED_FULL_GMM_FEATURES)
        missing = [x for x in expected if x not in cols]
        extra = [x for x in cols if x not in expected]
        if missing or extra or len(cols) != len(expected):
            raise RuntimeError(
                "Full GMM feature alignment failed: "
                f"resolved={cols}, missing={missing}, extra={extra}"
            )
        print("[CHECK] Full 19-feature alignment OK (13 absolute + 6 temporal-difference).")
    return tracks, cache


def evaluate_fixed_roi_config(
    *,
    ignition_mod: Any,
    cfg: Any,
    track_names: Sequence[str],
    feature_cache: Mapping[str, pd.DataFrame],
    gt: Mapping[str, Mapping[str, Any]],
    workers: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    def one(name: str) -> Dict[str, Any]:
        # Defensive copy: detector/config sweep must not mutate shared cached features.
        pred = ignition_mod.detect_ignition_gmm_rtv(feature_cache[name].copy(deep=True), cfg)
        gt_row = gt[name]
        pred_has = bool(pred.get("has_ignition", False))
        gt_has = bool(gt_row.get("gt_has_ignition", False))
        pred_onset = int(pred.get("ignition_idx_onset", -1))
        gt_onset = int(gt_row.get("gt_onset_idx", -1))
        onset_error = (
            abs(pred_onset - gt_onset)
            if pred_has and gt_has and pred_onset >= 0 and gt_onset >= 0
            else np.nan
        )
        return {
            "track_name": name,
            "track_id": parse_track_id(name),
            "gt_has_ignition": gt_has,
            "pred_has_ignition": pred_has,
            "pred_ignition_status": str(pred.get("ignition_status", "")),
            "gt_onset_idx": gt_onset,
            "pred_onset_idx": pred_onset,
            "onset_abs_error": onset_error,
            "gmm_n_states": int(pred.get("gmm_n_states", 0) or 0),
        }

    if int(workers) <= 1:
        rows = [one(name) for name in track_names]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(executor.map(one, track_names))
    frame = pd.DataFrame(rows)

    gt_arr = frame["gt_has_ignition"].to_numpy(dtype=bool)
    pred_arr = frame["pred_has_ignition"].to_numpy(dtype=bool)
    tp = int(np.sum(gt_arr & pred_arr))
    fp = int(np.sum(~gt_arr & pred_arr))
    fn = int(np.sum(gt_arr & ~pred_arr))
    tn = int(np.sum(~gt_arr & ~pred_arr))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = fp / max(int(np.sum(~gt_arr)), 1)
    missed = fn / max(int(np.sum(gt_arr)), 1)
    onset_values = pd.to_numeric(
        frame.loc[gt_arr & pred_arr, "onset_abs_error"], errors="coerce"
    ).dropna()
    onset_mae = float(onset_values.mean()) if len(onset_values) else float("nan")
    within1 = float((onset_values <= 1).mean()) if len(onset_values) else float("nan")
    within3 = float((onset_values <= 3).mean()) if len(onset_values) else float("nan")

    ci_low = ci_high = float("nan")
    if int(bootstrap_samples) > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        n = len(frame)
        values = np.empty(int(bootstrap_samples), dtype=float)
        for i in range(int(bootstrap_samples)):
            idx = rng.integers(0, n, size=n)
            g = gt_arr[idx]
            p = pred_arr[idx]
            btp = int(np.sum(g & p))
            bfp = int(np.sum(~g & p))
            bfn = int(np.sum(g & ~p))
            bp = btp / (btp + bfp) if btp + bfp else 0.0
            br = btp / (btp + bfn) if btp + bfn else 0.0
            values[i] = 2 * bp * br / (bp + br) if bp + br else 0.0
        ci_low, ci_high = [float(x) for x in np.percentile(values, [2.5, 97.5])]

    summary = {
        "n_tracks": int(len(frame)),
        "n_valid_gt": int(len(frame)),
        "n_gt_positive": int(np.sum(gt_arr)),
        "n_gt_negative": int(np.sum(~gt_arr)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "ignition_precision": float(precision),
        "ignition_recall": float(recall),
        "ignition_f1": float(f1),
        "false_ignition_rate": float(far),
        "missed_ignition_rate": float(missed),
        "onset_mae": onset_mae,
        "within_1_frame_acc": within1,
        "within_3_frame_acc": within3,
        "f1_ci_low": ci_low,
        "f1_ci_high": ci_high,
    }
    return summary, frame


def run_decision_sensitivity(
    *,
    ignition_mod: Any,
    fixed_roi_dir: Path,
    annotation_path: Path,
    output_root: Path,
    workers: int,
    onset_tolerance: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    base_cfg = build_full_gmm_rtv_config(ignition_mod)
    tracks, feature_cache = extract_fixed_roi_features(
        ignition_mod, fixed_roi_dir, base_cfg, workers
    )
    track_names = [name for name, _ in tracks]
    positive_onsets = read_effective_trace_positive_onsets(annotation_path)
    gt, gt_audit = build_fixed_roi_gt(
        positive_onsets, track_names, onset_tolerance
    )
    print(
        f"[GT fixed ROI] tracks={gt_audit['n_roi_tracks']} | "
        f"positive={gt_audit['n_evaluable_positive']} | "
        f"negative={gt_audit['n_implicit_negative']} | "
        f"missing_positive={gt_audit['n_positive_labels_absent_from_roi']}",
        flush=True,
    )

    run_cache: Dict[str, tuple[Dict[str, Any], pd.DataFrame]] = {}
    summary_rows: list[Dict[str, Any]] = []
    all_per_track: list[pd.DataFrame] = []
    default_values = {
        "gmm_max_states": 3,
        "cold_hot_persistence": 3,
        "rtv_alpha": 1.0,
    }

    for group, name, value, cfg, note in decision_parameter_specs(base_cfg):
        fingerprint = config_fingerprint(cfg)
        print(f"\n[DECISION] {name}={value}", flush=True)
        if fingerprint in run_cache:
            metrics, per_track = run_cache[fingerprint]
            print("  [CACHE] identical Full configuration reused", flush=True)
        else:
            metrics, per_track = evaluate_fixed_roi_config(
                ignition_mod=ignition_mod,
                cfg=cfg,
                track_names=track_names,
                feature_cache=feature_cache,
                gt=gt,
                workers=workers,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=stable_seed(bootstrap_seed, name, value),
            )
            run_cache[fingerprint] = (metrics, per_track)
        tagged = per_track.copy()
        tagged.insert(0, "parameter_value", value)
        tagged.insert(0, "parameter_name", name)
        tagged.insert(0, "parameter_group", group)
        all_per_track.append(tagged)
        summary_rows.append({
            "parameter_group": group,
            "parameter_name": name,
            "parameter_value": value,
            "default_value": default_values[name],
            "is_default_configuration": bool(float(value) == float(default_values[name])),
            "run_status": "success",
            "parameter_note": note,
            **metrics,
            "reference_track_coverage": np.nan,
            "suspicious_merge_proxy_count": np.nan,
        })
        print(
            f"  P={metrics['ignition_precision']:.4f} "
            f"R={metrics['ignition_recall']:.4f} "
            f"F1={metrics['ignition_f1']:.4f} "
            f"FAR={metrics['false_ignition_rate']:.4f}",
            flush=True,
        )

    return (
        pd.DataFrame(summary_rows),
        pd.concat(all_per_track, ignore_index=True),
        {
            "fixed_roi_gt": gt_audit,
            "full_gmm_config": asdict(base_cfg),
            "n_unique_decision_configs_executed": len(run_cache),
            "feature_count_expected": len(EXPECTED_FULL_GMM_FEATURES),
        },
    )


# -----------------------------------------------------------------------------
# Tracking max_center_dist sensitivity: aligned with latest T3 experiment
# -----------------------------------------------------------------------------

def build_tracker_cfg(base: Any, distance: float, ot_reg: float) -> Any:
    if not hasattr(base, "TrackerRunConfig"):
        raise RuntimeError("Tracking helper has no TrackerRunConfig")
    return base.TrackerRunConfig(
        enable_rescue=True,
        enable_gap_fill=True,
        enable_prebirth=True,
        max_center_dist=float(distance),
        max_age=4,
        max_age_tentative=1,
        max_age_confirmed=5,
        min_hits=3,
        min_contiguous_observed=3,
        min_observed_hits=3,
        min_track_len=2,
        max_gap_fill=10,
        ot_reg=float(ot_reg),
        image_width=None,
        image_height=None,
    )


def run_gmm_batch_with_fallback(
    *,
    base: Any,
    crop_data: Any,
    output_dir: Path,
    ignition_mod: Any,
    gmm_cfg: Any,
    workers: int,
    ignition_module_path: Path,
    variant: str,
) -> list[Dict[str, Any]]:
    requested = max(1, int(workers))
    try:
        return base.run_gmm_rtv_batch(
            crop_data,
            output_dir,
            ignition_mod,
            gmm_cfg,
            workers=requested,
            save_plot=False,
            ignition_module_path=ignition_module_path,
            variant=variant,
        )
    except Exception as exc:
        if requested <= 1:
            raise
        print(
            f"[WARN] {variant}: parallel GMM batch failed with "
            f"{type(exc).__name__}: {exc}; retrying serially.",
            flush=True,
        )
        gc.collect()
        return base.run_gmm_rtv_batch(
            crop_data,
            output_dir,
            ignition_mod,
            gmm_cfg,
            workers=1,
            save_plot=False,
            ignition_module_path=ignition_module_path,
            variant=variant,
        )


def run_tracking_sensitivity(
    *,
    base: Any,
    tracker_mod: Any,
    ignition_mod: Any,
    detections_dir: Path,
    full_frame_dir: Path,
    old_reference_tracks_path: Path,
    annotation_path: Path,
    ignition_module_path: Path,
    output_root: Path,
    workers: int,
    max_txt: int,
    image_exts: str,
    onset_tolerance: int,
    crop_padding: float,
    crop_format: str,
    jpeg_quality: int,
    track_match_max_dist: float,
    track_match_min_shared_frames: int,
    track_match_min_ref_coverage: float,
    track_match_min_query_coverage: float,
    track_match_min_temporal_iou: float,
    track_match_max_cost: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    required = [
        "load_detections_txt_dir_compat", "run_ot_kalman_variant",
        "aggregate_tracking_metrics", "match_tracks_to_reference",
        "filter_detected_track_points", "build_frame_image_map", "crop_all_tracks",
        "run_gmm_rtv_batch", "project_ignition_to_ref_ids",
        "evaluate_effective_trace_new_aligned",
    ]
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise RuntimeError(f"Tracking helper missing API: {missing}")

    dets = base.load_detections_txt_dir_compat(
        tracker_mod, detections_dir, max_txt=int(max_txt)
    )
    if not dets:
        raise RuntimeError("No detections loaded")
    frame_map, image_name_map = base.build_frame_image_map(full_frame_dir, image_exts)
    if not frame_map:
        raise RuntimeError(f"No full-frame images found in {full_frame_dir}")

    old_ref = pd.read_csv(old_reference_tracks_path, encoding="utf-8-sig")
    positive_onsets = read_effective_trace_positive_onsets(annotation_path)
    gt_global, gt_audit = build_tracking_gt_global(
        positive_onsets, old_ref, onset_tolerance
    )
    old_ref_detected = base.filter_detected_track_points(old_ref)
    n_old_ref = int(old_ref_detected["track_id"].nunique())

    distances = scaled_integer_values(
        TRACKING_DEFAULT_CENTER_DIST, TRACKING_CENTER_RATIOS
    )
    # Default first, because it defines the fresh structural reference.
    run_order = [int(TRACKING_DEFAULT_CENTER_DIST)] + [
        d for d in distances if d != int(TRACKING_DEFAULT_CENTER_DIST)
    ]
    tracks_by_distance: Dict[int, pd.DataFrame] = {}
    tracking_metrics_by_distance: Dict[int, Dict[str, Any]] = {}
    gmm_cfg = build_full_gmm_rtv_config(ignition_mod)

    # Keep OT regularization at the final experiment default.
    ot_reg = 0.08
    for distance in run_order:
        vdir = output_root / "tracking_max_center_dist" / str(distance)
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"\n[TRACKING] max_center_dist={distance}", flush=True)
        cfg = build_tracker_cfg(base, distance, ot_reg)
        tracks, run_stats = base.run_ot_kalman_variant(
            tracker_mod,
            dets,
            cfg,
            image_dir=full_frame_dir,
            image_exts=image_exts,
            progress_label=f"center_{distance}",
        )
        tracker_mod.save_tracks_csv(tracks, vdir / "tracks_long.csv", min_track_len=2)
        tracker_mod.save_track_summary_csv(tracks, vdir / "track_summary.csv", min_track_len=2)
        tracks_df = pd.read_csv(vdir / "tracks_long.csv", encoding="utf-8-sig")
        track_summary_df = pd.read_csv(vdir / "track_summary.csv", encoding="utf-8-sig")
        tracks_by_distance[distance] = tracks_df
        metrics = base.aggregate_tracking_metrics(tracks_df, track_summary_df, run_stats)
        tracking_metrics_by_distance[distance] = dict(metrics)

    fresh_t3 = tracks_by_distance[int(TRACKING_DEFAULT_CENTER_DIST)]
    fresh_detected = base.filter_detected_track_points(fresh_t3)
    n_fresh_ref = int(fresh_detected["track_id"].nunique())

    rows: list[Dict[str, Any]] = []
    for distance in distances:
        vdir = output_root / "tracking_max_center_dist" / str(distance)
        tracks_df = tracks_by_distance[distance]

        # Structural reference coverage: current distance -> fresh default T3.
        if distance == int(TRACKING_DEFAULT_CENTER_DIST):
            qdet = base.filter_detected_track_points(tracks_df)
            ids = sorted(int(v) for v in qdet["track_id"].unique())
            fresh_map = {tid: tid for tid in ids}
        else:
            fresh_map, fresh_diag = base.match_tracks_to_reference(
                tracks_df,
                fresh_t3,
                max_center_dist=float(track_match_max_dist),
                min_shared_frames=int(track_match_min_shared_frames),
                min_ref_coverage=float(track_match_min_ref_coverage),
                min_query_coverage=float(track_match_min_query_coverage),
                min_temporal_iou=float(track_match_min_temporal_iou),
                max_assignment_cost=float(track_match_max_cost),
                detected_only=True,
                return_diagnostics=True,
            )
            save_table(fresh_diag, vdir / "alignment_to_fresh_default_t3")
        fresh_coverage = len(set(int(v) for v in fresh_map.values())) / max(n_fresh_ref, 1)

        # End-to-end ignition evaluation on fresh variant ROIs.
        crop_data = base.crop_all_tracks(
            tracks_df,
            vdir / "track_crop",
            frame_map,
            image_name_map,
            float(crop_padding),
            crop_format=str(crop_format),
            jpeg_quality=int(jpeg_quality),
            skip_existing=True,
            progress_label=f"center_{distance}",
        )
        ignition_rows = run_gmm_batch_with_fallback(
            base=base,
            crop_data=crop_data,
            output_dir=vdir / "gmm_rtv",
            ignition_mod=ignition_mod,
            gmm_cfg=gmm_cfg,
            workers=workers,
            ignition_module_path=ignition_module_path,
            variant=f"max_center_dist_{distance}",
        )
        ignition_df = pd.DataFrame(ignition_rows)
        save_table(ignition_df, vdir / "ignition_summary")

        query_to_old, diag_old = base.match_tracks_to_reference(
            tracks_df,
            old_ref,
            max_center_dist=float(track_match_max_dist),
            min_shared_frames=int(track_match_min_shared_frames),
            min_ref_coverage=float(track_match_min_ref_coverage),
            min_query_coverage=float(track_match_min_query_coverage),
            min_temporal_iou=float(track_match_min_temporal_iou),
            max_assignment_cost=float(track_match_max_cost),
            detected_only=True,
            return_diagnostics=True,
        )
        save_table(diag_old, vdir / "alignment_to_original_gt_reference")
        projected = base.project_ignition_to_ref_ids(ignition_df, query_to_old)
        save_table(projected, vdir / "ignition_projected_to_original_gt_reference")
        qdet = base.filter_detected_track_points(tracks_df)
        eval_metrics = base.evaluate_effective_trace_new_aligned(
            gt_global,
            projected,
            query_ignition_df=ignition_df,
            query_to_ref=query_to_old,
            n_query_tracks=int(qdet["track_id"].nunique()),
            n_matched=len(query_to_old),
            n_reference_tracks=n_old_ref,
            count_unmatched_pred_as_fp=True,
        )

        metrics = tracking_metrics_by_distance[distance]
        matched_old = set(int(v) for v in query_to_old.values())
        original_gt_ref_coverage = len(matched_old) / max(n_old_ref, 1)
        row = {
            "parameter_group": "ot",
            "parameter_name": "max_center_dist",
            "parameter_value": int(distance),
            "default_value": int(TRACKING_DEFAULT_CENTER_DIST),
            "is_default_configuration": bool(distance == int(TRACKING_DEFAULT_CENTER_DIST)),
            "run_status": "success",
            "parameter_note": (
                "Full T3 tracker rerun; only max_center_dist changes. "
                "Reference coverage is measured against the fresh 45-px T3 run."
            ),
            "ignition_precision": float(eval_metrics.get("ignition_precision", np.nan)),
            "ignition_recall": float(eval_metrics.get("ignition_recall", np.nan)),
            "ignition_f1": float(eval_metrics.get("ignition_f1", np.nan)),
            "false_ignition_rate": float(eval_metrics.get("false_ignition_rate", np.nan)),
            "missed_ignition_rate": float(eval_metrics.get("missed_ignition_rate", np.nan)),
            "onset_mae": float(eval_metrics.get("onset_mae", np.nan)),
            "reference_track_coverage": float(fresh_coverage),
            "fresh_t3_reference_coverage": float(fresh_coverage),
            "original_gt_reference_coverage": float(original_gt_ref_coverage),
            "suspicious_merge_proxy_count": metrics.get("suspicious_merge_proxy_count", np.nan),
            "mean_track_len": metrics.get("mean_track_len", np.nan),
            "short_track_ratio": metrics.get("short_track_ratio", np.nan),
            "n_tracks": int(tracks_df["track_id"].nunique()),
            "f1_ci_low": np.nan,
            "f1_ci_high": np.nan,
        }
        rows.append(row)
        print(
            f"  RefCov={fresh_coverage:.4f} "
            f"P={row['ignition_precision']:.4f} R={row['ignition_recall']:.4f} "
            f"F1={row['ignition_f1']:.4f} FAR={row['false_ignition_rate']:.4f} "
            f"merge_proxy={row['suspicious_merge_proxy_count']}",
            flush=True,
        )

    return pd.DataFrame(rows), {
        "tracking_gt": gt_audit,
        "max_center_dist_values": distances,
        "fresh_reference_distance": int(TRACKING_DEFAULT_CENTER_DIST),
        "n_fresh_reference_tracks": n_fresh_ref,
        "n_original_gt_reference_tracks": n_old_ref,
    }


# -----------------------------------------------------------------------------
# Optional baseline checks against current Fig. 2 tables
# -----------------------------------------------------------------------------

def read_optional_excel(path: Optional[str], script_dir: Path) -> Optional[pd.DataFrame]:
    if not path:
        return None
    p = resolve_local(path, script_dir)
    if not p.is_file():
        print(f"[WARN] Baseline table not found; check skipped: {p}")
        return None
    return pd.read_excel(p)


def baseline_checks(
    combined: pd.DataFrame,
    *,
    method_summary: Optional[pd.DataFrame],
    tracking_summary: Optional[pd.DataFrame],
    tolerance: float,
) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}

    # Decision default should reproduce gmm_rtv_full from 4.2.2.
    if method_summary is not None and {"method", "ignition_f1"}.issubset(method_summary.columns):
        target = method_summary[method_summary["method"].astype(str) == "gmm_rtv_full"]
        current = combined[
            combined["parameter_name"].astype(str).eq("gmm_max_states")
            & combined["is_default_configuration"].astype(bool)
        ]
        if len(target) == 1 and len(current) == 1:
            fields = ["ignition_precision", "ignition_recall", "ignition_f1", "false_ignition_rate"]
            diffs = {
                field: abs(float(current.iloc[0][field]) - float(target.iloc[0][field]))
                for field in fields
                if field in target.columns and field in current.columns
            }
            audit["decision_default_vs_method_summary"] = {
                "absolute_differences": diffs,
                "within_tolerance": bool(all(v <= float(tolerance) for v in diffs.values())),
            }

    # Tracking default should be comparable to current T3 full tracker summary.
    if tracking_summary is not None and {"variant", "ignition_f1"}.issubset(tracking_summary.columns):
        target = tracking_summary[
            tracking_summary["variant"].astype(str).eq("T3_full_tracker")
        ]
        current = combined[
            combined["parameter_name"].astype(str).eq("max_center_dist")
            & combined["is_default_configuration"].astype(bool)
        ]
        if len(target) == 1 and len(current) == 1:
            fields = ["ignition_precision", "ignition_recall", "ignition_f1", "false_ignition_rate"]
            diffs = {
                field: abs(float(current.iloc[0][field]) - float(target.iloc[0][field]))
                for field in fields
                if field in target.columns and pd.notna(target.iloc[0][field])
            }
            audit["tracking_default_vs_tracking_summary"] = {
                "absolute_differences": diffs,
                "within_tolerance": bool(all(v <= float(tolerance) for v in diffs.values())),
            }
    return audit


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute only the four parameter sweeps required by manuscript Fig. 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--parts",
        choices=["all", "tracking", "decision"],
        default="all",
        help="Run all four panels, only panel (a), or only panels (b-d).",
    )
    p.add_argument("--tracking-helper-path", default="run_tracking_ablation_experiments_gmm_rtv_effective_trace_positive_gt_v2.py")
    p.add_argument("--tracking-module-path", default="ot_kalman_tracker.py")
    p.add_argument("--ignition-module-path", default="ignition_detection_gmm_rtv_new.py")
    p.add_argument("--annotation-path", default="effective_trace.xlsx")
    p.add_argument("--old-reference-tracks-csv", default=DEFAULT_OLD_REFERENCE_TRACKS)
    p.add_argument("--fixed-roi-dir", default=DEFAULT_FIXED_ROI_DIR)
    p.add_argument("--detections-txt-dir", default=DEFAULT_DETECTIONS_DIR)
    p.add_argument("--full-frame-dir", default=DEFAULT_FULL_FRAME_DIR)
    p.add_argument("--max-txt", type=int, default=7996)
    p.add_argument("--image-exts", default=".png,.jpg,.jpeg,.tif,.tiff")
    p.add_argument("--onset-tolerance", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--bootstrap-samples", type=int, default=1000, help="Decision-panel F1 bootstrap samples; 0 disables.")
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument("--crop-padding", type=float, default=4.0)
    p.add_argument("--crop-format", choices=["png", "jpg", "jpeg"], default="png")
    p.add_argument("--jpeg-quality", type=int, default=90)

    # Same detected-only identity gates as the final aligned tracking experiment.
    p.add_argument("--track-match-max-dist", type=float, default=30.0)
    p.add_argument("--track-match-min-shared-frames", type=int, default=3)
    p.add_argument("--track-match-min-ref-coverage", type=float, default=0.20)
    p.add_argument("--track-match-min-query-coverage", type=float, default=0.20)
    p.add_argument("--track-match-min-temporal-iou", type=float, default=0.15)
    p.add_argument("--track-match-max-cost", type=float, default=1.50)

    p.add_argument("--baseline-method-summary", default=r"draw_res_final\method_summary.xlsx")
    p.add_argument("--baseline-tracking-summary", default=r"draw_res_final\tracking_ablation_summary_with_ignition.xlsx")
    p.add_argument("--baseline-check-tolerance", type=float, default=1e-6)
    p.add_argument("--strict-baseline-check", action="store_true")
    p.add_argument("--output-dir", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.onset_tolerance < 0:
        raise ValueError("--onset-tolerance must be non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")

    script_dir = Path(__file__).resolve().parent
    ignition_path = resolve_local(args.ignition_module_path, script_dir)
    annotation_path = resolve_local(args.annotation_path, script_dir)
    ignition_mod = load_module(ignition_path, "fig3_ignition_runtime")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("parameter_sensitivity_fig3") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summary_parts: list[pd.DataFrame] = []
    audit: Dict[str, Any] = {
        "gt_semantics": (
            "positive iff effective_trace contains track_id with numeric frame_id; "
            "all other tracks in current evaluable population are negative"
        ),
        "parameter_definitions": {
            "max_center_dist": scaled_integer_values(TRACKING_DEFAULT_CENTER_DIST, TRACKING_CENTER_RATIOS),
            "gmm_max_states": list(GMM_MAX_STATES_VALUES),
            "cold_hot_persistence": list(COLD_HOT_PERSISTENCE_VALUES),
            "rtv_alpha": list(RTV_ALPHA_VALUES),
        },
        "notes": {
            "cold_hot_persistence": (
                "Changes min_cold_run and min_hot_run together; these values also "
                "control RTV pre/post averaging windows in the production detector."
            ),
            "rtv_alpha": "Scales all five raw-transition thresholds from the Full base config.",
            "decision_population": "Full fixed-ROI population; no tune/test split is used.",
            "tracking_reference_coverage": "Coverage against fresh default 45-px T3 run.",
            "windows_spawn_import": "Tracking helper is imported under its real module stem for ProcessPoolExecutor compatibility.",
        },
    }

    decision_per_track = pd.DataFrame()
    if args.parts in {"all", "decision"}:
        fixed_roi_dir = Path(args.fixed_roi_dir).expanduser().resolve()
        if not fixed_roi_dir.is_dir():
            raise FileNotFoundError(f"Fixed ROI dir not found: {fixed_roi_dir}")
        decision_summary, decision_per_track, decision_audit = run_decision_sensitivity(
            ignition_mod=ignition_mod,
            fixed_roi_dir=fixed_roi_dir,
            annotation_path=annotation_path,
            output_root=output_root,
            workers=int(args.workers),
            onset_tolerance=int(args.onset_tolerance),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed),
        )
        summary_parts.append(decision_summary)
        audit.update(decision_audit)
        decision_per_track.to_csv(
            output_root / "decision_per_track.csv", index=False, encoding="utf-8-sig"
        )

    if args.parts in {"all", "tracking"}:
        helper_path = resolve_local(args.tracking_helper_path, script_dir)
        tracker_path = resolve_local(args.tracking_module_path, script_dir)
        # IMPORTANT on Windows: the tracking helper owns ProcessPool worker
        # functions.  It must be imported under its real filename stem so that
        # spawned children can import the same module during unpickling.
        base = load_spawn_safe_module(helper_path)
        tracker_mod = load_spawn_safe_module(tracker_path)
        detections_dir = Path(args.detections_txt_dir).expanduser().resolve()
        full_frame_dir = Path(args.full_frame_dir).expanduser().resolve()
        old_ref_path = Path(args.old_reference_tracks_csv).expanduser().resolve()
        for pth, label, is_dir in [
            (detections_dir, "detections", True),
            (full_frame_dir, "full frames", True),
            (old_ref_path, "old reference tracks", False),
        ]:
            ok = pth.is_dir() if is_dir else pth.is_file()
            if not ok:
                raise FileNotFoundError(f"{label} not found: {pth}")
        tracking_summary, tracking_audit = run_tracking_sensitivity(
            base=base,
            tracker_mod=tracker_mod,
            ignition_mod=ignition_mod,
            detections_dir=detections_dir,
            full_frame_dir=full_frame_dir,
            old_reference_tracks_path=old_ref_path,
            annotation_path=annotation_path,
            ignition_module_path=ignition_path,
            output_root=output_root,
            workers=int(args.workers),
            max_txt=int(args.max_txt),
            image_exts=str(args.image_exts),
            onset_tolerance=int(args.onset_tolerance),
            crop_padding=float(args.crop_padding),
            crop_format=str(args.crop_format),
            jpeg_quality=int(args.jpeg_quality),
            track_match_max_dist=float(args.track_match_max_dist),
            track_match_min_shared_frames=int(args.track_match_min_shared_frames),
            track_match_min_ref_coverage=float(args.track_match_min_ref_coverage),
            track_match_min_query_coverage=float(args.track_match_min_query_coverage),
            track_match_min_temporal_iou=float(args.track_match_min_temporal_iou),
            track_match_max_cost=float(args.track_match_max_cost),
        )
        summary_parts.append(tracking_summary)
        audit.update(tracking_audit)

    if not summary_parts:
        raise RuntimeError("No sensitivity part was executed")
    combined = pd.concat(summary_parts, ignore_index=True, sort=False)
    group_order = {"ot": 0, "gmm": 1, "rtv": 2}
    param_order = {
        "max_center_dist": 0,
        "gmm_max_states": 1,
        "cold_hot_persistence": 2,
        "rtv_alpha": 3,
    }
    combined["_group_order"] = combined["parameter_group"].map(group_order).fillna(99)
    combined["_param_order"] = combined["parameter_name"].map(param_order).fillna(99)
    combined["_value_numeric"] = pd.to_numeric(combined["parameter_value"], errors="coerce")
    combined = (
        combined.sort_values(["_param_order", "_value_numeric"], kind="stable")
        .drop(columns=["_group_order", "_param_order", "_value_numeric"])
        .reset_index(drop=True)
    )

    method_table = read_optional_excel(args.baseline_method_summary, script_dir)
    tracking_table = read_optional_excel(args.baseline_tracking_summary, script_dir)
    checks = baseline_checks(
        combined,
        method_summary=method_table,
        tracking_summary=tracking_table,
        tolerance=float(args.baseline_check_tolerance),
    )
    audit["baseline_checks"] = checks
    failed = [name for name, info in checks.items() if not bool(info.get("within_tolerance", True))]
    if failed:
        message = "Baseline reproducibility check failed: " + ", ".join(failed)
        if args.strict_baseline_check:
            raise RuntimeError(message)
        print(f"[WARN] {message}", flush=True)

    save_table(combined, output_root / "combined_summary")
    write_json(output_root / "logic_audit.json", audit)
    write_json(output_root / "run_config.json", vars(args))

    print("\n=== Fig. 3 combined summary ===")
    show = [
        "parameter_name", "parameter_value", "is_default_configuration",
        "ignition_precision", "ignition_recall", "ignition_f1",
        "false_ignition_rate", "reference_track_coverage",
        "suspicious_merge_proxy_count",
    ]
    print(combined[[c for c in show if c in combined.columns]].to_string(index=False))
    print(f"\n[OK] Output: {output_root.resolve()}")
    print("[OK] Fig. 3 input: combined_summary.xlsx")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
