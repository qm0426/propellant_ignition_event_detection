#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking 模块辅助消融实验：比较不同 tracking 构建策略对下游 GMM-RTV 点火检测的影响。

默认消融链：
    T0_center_nn
    → T1_basic_ot_kalman
    → T2_ot_kalman_rescue
    → T2b_ot_kalman_rescue_gap
    → T3_full_tracker

B0_byte_track 代码作为可选诊断基线保留，但默认不运行、不进入论文汇总。

评估：
  - 默认使用 effective_trace.xlsx。只有“出现在表中且 frame_id 非空”的 T3 reference track
    才定义为有效 ignition GT；其余可评价轨迹（表内 frame_id 为空或未出现在表中）均作为
    no_ignition。
  - T0/T1/T2/T2b 先通过 detected-only 时空匹配对齐到 T3 的 track_id，再在同一 T3
    reference-ID GT 空间上计算端到端 P/R/F1；未能对齐到 T3 的额外预测点火计为 FP。
  - effective_trace 中的 frame_id 是 T3 ROI 序列 onset 索引；跨 tracking 变体评价前统一映射
    为全局 frame_id。
  - 仍兼容 legacy effective_trace_new.xlsx。

图像输入：
  - full_frame：从原始全帧图像按各 tracking 变体 bbox 重新裁剪；
  - roi_cache：当原图盘不可用时，使用既有 T3 track_crop + T3 tracks_long.csv 建立
    (frame_id,bbox)->ROI 缓存。仅对能与缓存中的同一 bbox 匹配的轨迹点复用 ROI；
    T3 轨迹直接读取预计算结果，从而避免在没有全帧图像时重新执行 prebirth。

不修改 ot_kalman_tracker.py / ignition_detection_gmm_rtv_new.py 本体，通过 import 复用。
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import hashlib
import json
import re
import shutil
from io import StringIO
import math
import sys
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment


# ============================================================
# Constants
# ============================================================

VARIANT_ORDER = [
    "T0_center_nn",
    "B0_byte_track",
    "T1_basic_ot_kalman",
    "T2_ot_kalman_rescue",
    "T2b_ot_kalman_rescue_gap",
    "T3_full_tracker",
]

SUMMARY_COLUMNS = [
    "variant",
    "status",
    "error",
    "n_tracks",
    "mean_track_len",
    "median_track_len",
    "mean_detected_len",
    "median_detected_len",
    "short_track_ratio",
    "long_track_ratio",
    "n_short_detected_tracks",
    "n_long_detected_tracks",
    "short_detected_track_ratio",
    "long_detected_track_ratio",
    "candidate_ratio",
    "candidate_ratio_macro",
    "prebirth_ratio",
    "prebirth_ratio_macro",
    "n_short_tracks",
    "n_long_tracks",
    "mean_match_cost",
    "mean_ot_prob",
    "mean_center_match_distance",
    "n_rescue_matches",
    "n_gap_candidates",
    "n_byte_first_stage_matches",
    "n_byte_second_stage_matches",
    "n_high_score_detections",
    "n_low_score_detections",
    "n_byte_removed_short_tracks",
    "suspicious_merge_proxy_count",
    "large_jump_track_count",
    "n_ignition_ok",
    "n_ignition_partial",
    "n_ignition_skip",
    "n_ignition_error",
    "mean_crop_success_rate",
    "n_pred_ignition",
    "n_pred_no_ignition",
    "n_pred_ambiguous",
    "n_pred_shape_rejected",
    "n_pred_initially_hot",
    "n_pred_weak_transition_rejected",
    "n_pred_single_state",
    "n_pred_ambiguous_gmm_unstable",
    "n_valid_ignition_tracks",
    "n_false_like_events",
    "n_gt_events",
    "n_matched_events",
    "event_capture_rate",
    "event_precision",
    "event_recall",
    "event_f1",
    "false_event_count",
    "n_matched_to_ref",
    "match_rate",
    "reference_track_coverage",
    "gt_track_coverage",
    "unmatched_reference_ignition_count",
    "unmatched_predicted_ignition_count",
    "n_valid_gt",
    "n_matched_valid_gt",
    "end_to_end_tp",
    "end_to_end_fp",
    "end_to_end_fn",
    "ignition_precision",
    "ignition_recall",
    "ignition_f1",
    "matched_only_precision",
    "matched_only_recall",
    "matched_only_f1",
    "track_precision",
    "track_recall",
    "track_f1",
    "onset_mae",
    "onset_median_error",
    "within_1_frame_acc",
    "within_3_frame_acc",
    "inside_interval_acc",
    "n_onset_eval",
    "n_onset_eval_common",
    "onset_mae_common",
    "onset_median_error_common",
    "within_1_frame_acc_common",
    "within_3_frame_acc_common",
    "inside_interval_acc_common",
]


# ============================================================
# Utilities
# ============================================================

def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if obj is None:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def validate_variants(raw: str) -> List[str]:
    variants = [value.strip() for value in str(raw).split(",") if value.strip()]
    duplicates = sorted({
        value for value in variants if variants.count(value) > 1
    })
    if duplicates:
        raise ValueError(
            f"Duplicate tracking variants: {', '.join(duplicates)}"
        )
    unknown = sorted(set(variants) - set(VARIANT_ORDER))
    if unknown:
        raise ValueError(
            "Unknown tracking variants: "
            f"{', '.join(unknown)}; allowed: {', '.join(VARIANT_ORDER)}"
        )
    if not variants:
        raise ValueError("At least one tracking variant is required")
    return variants


def resave_csv_utf8_sig(csv_path: Path) -> None:
    """将 tracker 输出的 utf-8 CSV 转为 utf-8-sig。"""
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def save_table(df: pd.DataFrame, path_without_ext: Path, quiet: bool = False) -> None:
    csv_path = Path(f"{path_without_ext}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if not quiet:
        print(f"[OK] Saved: {csv_path}")
    try:
        xlsx_path = Path(f"{path_without_ext}.xlsx")
        df.to_excel(xlsx_path, index=False)
        if not quiet:
            print(f"[OK] Saved: {xlsx_path}")
    except Exception as exc:
        if not quiet:
            warnings.warn(f"Skip Excel export ({exc}); CSV at {csv_path}")


def augment_tracks_csv_from_history(
    tracks: List[Any],
    csv_path: Path,
    fields: Tuple[str, ...],
) -> None:
    """将外部 save_tracks_csv 未导出的诊断字段补回长表。"""
    csv_path = Path(csv_path)
    if not csv_path.is_file() or not fields:
        return
    extra_rows: List[Dict[str, Any]] = []
    for track in tracks:
        tid = int(getattr(track, "track_id"))
        for item in getattr(track, "history", []):
            if "frame_id" not in item:
                continue
            row: Dict[str, Any] = {
                "track_id": tid,
                "frame_id": int(item["frame_id"]),
            }
            found = False
            for field_name in fields:
                if field_name in item:
                    row[field_name] = item.get(field_name)
                    found = True
            if found:
                extra_rows.append(row)
    if not extra_rows:
        return
    base = pd.read_csv(csv_path)
    extra = pd.DataFrame(extra_rows).drop_duplicates(
        ["track_id", "frame_id"], keep="last"
    )
    for field_name in fields:
        if field_name not in extra.columns:
            continue
        patch = extra[["track_id", "frame_id", field_name]].rename(
            columns={field_name: f"__history_{field_name}"}
        )
        base = base.merge(patch, on=["track_id", "frame_id"], how="left")
        patch_col = f"__history_{field_name}"
        if field_name in base.columns:
            base[field_name] = base[patch_col].combine_first(base[field_name])
        else:
            base[field_name] = base[patch_col]
        base = base.drop(columns=[patch_col])
    base.to_csv(csv_path, index=False, encoding="utf-8-sig")


def load_module_from_path(module_path: Path, module_name: str) -> Any:
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_workers(workers: int) -> int:
    """workers>0 用指定进程数；workers<=0 自动取 min(cpu-1, 16)。"""
    if workers > 0:
        return workers
    import os
    return max(1, min((os.cpu_count() or 4) - 1, 16))


def load_detections_txt_dir_compat(
    tracker_mod: Any,
    txt_dir: Path,
    max_txt: int = -1,
) -> Dict[int, List[Dict[str, Any]]]:
    """加载 per-frame txt 检测；兼容 tracker 是否支持 max_txt。"""
    if not hasattr(tracker_mod, "load_detections_txt_dir"):
        raise AttributeError(
            f"{getattr(tracker_mod, '__file__', tracker_mod)} 缺少 load_detections_txt_dir"
        )
    load_fn = tracker_mod.load_detections_txt_dir
    try:
        import inspect
        if "max_txt" in inspect.signature(load_fn).parameters:
            return load_fn(txt_dir, max_txt=int(max_txt))
    except (TypeError, ValueError):
        pass

    dets_by_frame = load_fn(txt_dir)
    if int(max_txt) < 0:
        return dets_by_frame
    keep_n = int(max_txt)
    limited = {fid: dets for fid, dets in dets_by_frame.items() if int(fid) < keep_n}
    print(
        f"[INFO] detections txt limited to first {len(limited)} frames "
        f"(max_txt={keep_n}, loaded_all={len(dets_by_frame)})"
    )
    return limited


def make_default_output_dir(base: str = "tracking_ablation_results") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"run_{stamp}"


def natural_sort_key(path: str) -> list:
    import re

    name = Path(path).name
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def _csv_looks_like_tracks_long(path: Path) -> Tuple[bool, str]:
    """Cheaply validate whether a CSV can serve as per-frame T3 track metadata."""
    try:
        head = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            head = pd.read_csv(path, nrows=5)
        except Exception as exc:
            return False, f"read_error:{exc}"
    except Exception as exc:
        return False, f"read_error:{exc}"
    cols = {str(c).strip().lower() for c in head.columns}
    required = {"track_id", "frame_id"}
    has_center = {"cx", "cy"}.issubset(cols)
    has_box = {"x1", "y1", "x2", "y2"}.issubset(cols)
    if not required.issubset(cols):
        return False, "missing track_id/frame_id"
    if not (has_center or has_box):
        return False, "missing cx/cy or x1/y1/x2/y2"
    return True, "ok"


def resolve_roi_tracks_csv(roi_root: Path, explicit: Optional[str]) -> Path:
    """Resolve the T3 per-frame track metadata CSV used to index cached ROI images.

    The preferred file is tracks_long.csv next to ``track_crop``.  Older runs may
    have used a different CSV name, so if that exact file is absent we inspect
    nearby CSV files and select a unique schema-compatible candidate.  We never
    silently choose between multiple plausible files from different runs.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--roi-tracks-csv not found: {path}")
        ok, reason = _csv_looks_like_tracks_long(path)
        if not ok:
            raise ValueError(
                f"--roi-tracks-csv is not valid per-frame track metadata: {path} ({reason})"
            )
        return path

    preferred = roi_root.parent / "tracks_long.csv"
    if preferred.is_file():
        return preferred

    # Search the current run directory first.  We inspect schemas rather than
    # depending on a historical filename.
    search_roots = [roi_root.parent]
    # Also inspect the parent tracking_output directory for older layouts, but
    # only one level of run subdirectories is considered to avoid scanning the
    # whole project tree.
    if roi_root.parent.parent.is_dir():
        search_roots.append(roi_root.parent.parent)

    candidates: List[Path] = []
    seen: Set[str] = set()
    for root in search_roots:
        patterns = ["*.csv"] if root == roi_root.parent else ["*/tracks_long.csv", "*/tracks*.csv"]
        for pattern in patterns:
            for cand in root.glob(pattern):
                key = str(cand.resolve()).lower()
                if key in seen or not cand.is_file():
                    continue
                seen.add(key)
                ok, _ = _csv_looks_like_tracks_long(cand)
                if ok:
                    candidates.append(cand)

    # Strong preference for a schema-compatible CSV in the same run directory.
    same_run = [c for c in candidates if c.parent.resolve() == roi_root.parent.resolve()]
    if len(same_run) == 1:
        print(f"[INFO] auto-detected T3 per-frame track metadata: {same_run[0]}")
        return same_run[0]
    if len(same_run) > 1:
        names = "\n  - ".join(str(c) for c in same_run)
        raise RuntimeError(
            "Multiple schema-compatible track CSV files were found beside track_crop. "
            "Specify the correct one with --roi-tracks-csv:\n  - " + names
        )
    if len(candidates) == 1:
        print(f"[INFO] auto-detected nearby T3 per-frame track metadata: {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        names = "\n  - ".join(str(c) for c in candidates[:20])
        raise RuntimeError(
            "Multiple possible per-frame track CSV files were found. To avoid mixing "
            "different tracking runs, specify the matching file with --roi-tracks-csv:\n  - "
            + names
        )

    raise FileNotFoundError(
        "ROI-cache mode needs the per-frame T3 track metadata that generated the ROI folders. "
        "No schema-compatible CSV was found near:\n"
        f"  {roi_root}\n\n"
        "Required columns: track_id, frame_id, and either cx/cy or x1/y1/x2/y2.\n"
        "The ROI images alone are insufficient for a strict tracking ablation because the "
        "script must map each T0/T1/T2/T2b detection box to the corresponding cached T3 ROI.\n\n"
        "On Windows you can locate a candidate with:\n"
        r'  dir /s /b "E:\Adia\code\vjepa2-main\*tracks*.csv"' "\n"
        "Then rerun with:\n"
        r'  --roi-tracks-csv "FULL_PATH_TO_THE_MATCHING_CSV"'
    )


def read_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def build_gmm_rtv_config(
    ignition_mod: Any,
    args: Optional[argparse.Namespace] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """采用核心默认；仅应用非 None 的 CLI/字典覆盖，并关闭特征保存。"""
    required_api = (
        "IgnitionGMMConfig",
        "extract_sequence_features",
        "detect_ignition_gmm_rtv",
        "plot_detection_result_gmm_rtv",
    )
    missing = [name for name in required_api if not hasattr(ignition_mod, name)]
    if missing:
        raise AttributeError(
            f"GMM-RTV ignition module missing required API: {', '.join(missing)}"
        )
    config_fields = getattr(
        ignition_mod.IgnitionGMMConfig, "__dataclass_fields__", {}
    )
    missing_config = [
        name
        for name in ("require_all_prior_cold", "refine_onset_with_change")
        if name not in config_fields
    ]
    if missing_config:
        raise RuntimeError(
            "IgnitionGMMConfig missing required configuration: "
            + ", ".join(missing_config)
        )

    if isinstance(args, dict) and overrides is None:
        overrides = args
        args = None

    configurable = (
        "gmm_max_states",
        "gmm_bic_improvement_ratio",
        "gmm_min_state_count",
        "gmm_min_state_fraction",
        "min_state_run",
        "min_cold_run",
        "min_hot_run",
        "min_stable_run",
        "min_sequence_len",
        "refine_radius",
        "use_diff_features",
        "require_all_prior_cold",
        "refine_onset_with_change",
        "enable_shape_sequence_reject",
        "enable_initial_hot_reject",
        "enable_raw_transition_validation",
    )
    selected: Dict[str, Any] = {}
    if args is not None:
        selected.update({
            name: getattr(args, name)
            for name in configurable
            if hasattr(args, name) and getattr(args, name) is not None
        })
    if overrides:
        unknown = sorted(set(overrides) - set(configurable))
        if unknown:
            raise ValueError(f"Unknown GMM-RTV config overrides: {', '.join(unknown)}")
        selected.update({
            name: value for name, value in overrides.items() if value is not None
        })

    cfg = ignition_mod.IgnitionGMMConfig()
    return replace(cfg, **selected, save_features=False)


# ============================================================
# T0: center nearest neighbor tracker
# ============================================================

@dataclass
class CenterNNTrack:
    track_id: int
    start_frame: int
    last_frame: int
    misses: int = 0
    hits: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_confirmed: bool = False
    track_state: str = "tentative"
    birth_reason: str = "unmatched_detection"
    end_reason: str = ""
    prebirth_recovered: bool = False
    kf: Any = None


def _history_item_from_det(det: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "image_name": det.get("image_name", ""),
        "x1": det["x1"],
        "y1": det["y1"],
        "x2": det["x2"],
        "y2": det["y2"],
        "cx": det["cx"],
        "cy": det["cy"],
        "w": det["w"],
        "h": det["h"],
        "det_score": det.get("score", 1.0),
        # Center-NN 与 OT 方法的概率/代价语义不同，不能伪装成 OT=1、cost=0。
        "ot_prob": np.nan,
        "match_cost": np.nan,
        "center_match_distance": np.nan,
        "center_match_distance_norm": np.nan,
        "pred_x1": det["x1"],
        "pred_y1": det["y1"],
        "pred_x2": det["x2"],
        "pred_y2": det["y2"],
        "point_type": "detected",
        "gap_fill_method": "",
    }

def run_center_nn_tracker(
    dets_by_frame: Dict[int, List[Dict[str, Any]]],
    max_center_dist: float = 45.0,
    max_age: int = 5,
    max_age_tentative: Optional[int] = None,
    max_age_confirmed: Optional[int] = None,
    min_hits: int = 3,
) -> Tuple[List[CenterNNTrack], Dict[str, Any]]:
    if not dets_by_frame:
        return [], {"n_rescue_matches": 0, "n_gap_candidates": 0}

    tentative_timeout = int(max_age if max_age_tentative is None else max_age_tentative)
    confirmed_timeout = int(max_age if max_age_confirmed is None else max_age_confirmed)
    if tentative_timeout < 0 or confirmed_timeout < 0:
        raise ValueError("T0 max ages must be non-negative")

    frame_ids = sorted(dets_by_frame.keys())
    active: List[CenterNNTrack] = []
    finished: List[CenterNNTrack] = []
    next_id = 1

    for frame_id in range(frame_ids[0], frame_ids[-1] + 1):
        detections = dets_by_frame.get(frame_id, [])
        n = len(active)
        m = len(detections)

        candidates: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(active):
            tcx = float(tr.history[-1]["cx"])
            tcy = float(tr.history[-1]["cy"])
            for di, det in enumerate(detections):
                dist = math.hypot(det["cx"] - tcx, det["cy"] - tcy)
                if dist <= max_center_dist:
                    candidates.append((dist, ti, di))

        candidates.sort(key=lambda x: x[0])
        matched_tracks: set = set()
        matched_dets: set = set()
        for dist, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            tr = active[ti]
            det = detections[di]
            item = _history_item_from_det(det, frame_id)
            item["center_match_distance"] = float(dist)
            item["center_match_distance_norm"] = float(dist / max(max_center_dist, 1e-6))
            tr.history.append(item)
            tr.last_frame = frame_id
            tr.hits += 1
            tr.misses = 0
            if tr.hits >= min_hits:
                tr.is_confirmed = True
                tr.track_state = "confirmed"
            matched_tracks.add(ti)
            matched_dets.add(di)

        for j, det in enumerate(detections):
            if j not in matched_dets:
                tr = CenterNNTrack(
                    track_id=next_id,
                    start_frame=frame_id,
                    last_frame=frame_id,
                    hits=1,
                    misses=0,
                    history=[_history_item_from_det(det, frame_id)],
                )
                next_id += 1
                active.append(tr)

        n_existing = n
        still_active: List[CenterNNTrack] = []
        for i, tr in enumerate(active):
            if i in matched_tracks:
                still_active.append(tr)
                continue
            if i >= n_existing:
                still_active.append(tr)
                continue
            tr.misses += 1
            allowed_misses = confirmed_timeout if tr.is_confirmed else tentative_timeout
            if tr.misses > allowed_misses:
                tr.track_state = "finished"
                tr.end_reason = "unmatched_timeout"
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active

    for tr in active:
        tr.track_state = "finished"
        tr.end_reason = tr.end_reason or "end_of_sequence"
        finished.append(tr)

    return finished, {"n_rescue_matches": 0, "n_gap_candidates": 0}


# ============================================================
# B0: lightweight ByteTrack (Kalman + two-stage IoU matching)
# ============================================================

def validate_byte_track_thresholds(
    low_thresh: float,
    high_thresh: float,
    new_track_thresh: float,
) -> None:
    if not 0.0 <= low_thresh <= high_thresh <= new_track_thresh:
        raise ValueError(
            "ByteTrack thresholds must satisfy "
            "0 <= low <= high <= new"
        )


def validate_byte_track_parameters(
    low_thresh: float,
    high_thresh: float,
    new_track_thresh: float,
    match_iou_thresh: float,
    second_match_iou_thresh: float,
    track_buffer: int,
    min_hits: int,
) -> None:
    validate_byte_track_thresholds(
        low_thresh, high_thresh, new_track_thresh
    )
    for name, value in (
        ("match_iou_thresh", match_iou_thresh),
        ("second_match_iou_thresh", second_match_iou_thresh),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if track_buffer < 0 or min_hits <= 0:
        raise ValueError("track_buffer must be >= 0 and min_hits must be > 0")


def _bbox_iou(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    intersection_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_w * intersection_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _match_byte_tracks_by_iou(
    tracks: List[Any],
    detections: List[Dict[str, Any]],
    min_iou: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    ious = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    for track_index, track in enumerate(tracks):
        predicted_box = tuple(float(value) for value in track.kf.bbox_xyxy())
        for detection_index, detection in enumerate(detections):
            detection_box = (
                float(detection["x1"]),
                float(detection["y1"]),
                float(detection["x2"]),
                float(detection["y2"]),
            )
            ious[track_index, detection_index] = _bbox_iou(
                predicted_box, detection_box
            )
    costs = 1.0 - ious
    costs[ious < min_iou] = 1e6
    row_indices, column_indices = linear_sum_assignment(costs)
    matches = []
    matched_tracks = set()
    matched_detections = set()
    for track_index, detection_index in zip(row_indices, column_indices):
        iou = float(ious[track_index, detection_index])
        if iou < min_iou:
            continue
        matches.append((int(track_index), int(detection_index), iou))
        matched_tracks.add(int(track_index))
        matched_detections.add(int(detection_index))
    unmatched_tracks = [
        index for index in range(len(tracks)) if index not in matched_tracks
    ]
    unmatched_detections = [
        index for index in range(len(detections))
        if index not in matched_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


def _new_byte_track(
    tracker_mod: Any,
    detection: Dict[str, Any],
    frame_id: int,
    track_id: int,
    min_hits: int,
) -> Any:
    kf = tracker_mod.KalmanBoxFilter(
        detection["cx"], detection["cy"], detection["w"], detection["h"]
    )
    item = _history_item_from_det(detection, frame_id)
    item["match_cost"] = np.nan
    confirmed = min_hits <= 1
    return tracker_mod.Track(
        track_id=track_id,
        kf=kf,
        start_frame=frame_id,
        last_frame=frame_id,
        age=1,
        hits=1,
        misses=0,
        history=[item],
        is_confirmed=confirmed,
        track_state="confirmed" if confirmed else "tentative",
        birth_reason="byte_high_score_detection",
        birth_frame=frame_id,
        birth_x=float(detection["cx"]),
        birth_y=float(detection["cy"]),
        end_reason="",
    )


def run_byte_track_variant(
    tracker_mod: Any,
    dets_by_frame: Dict[int, List[Dict[str, Any]]],
    *,
    high_thresh: float = 0.6,
    low_thresh: float = 0.1,
    new_track_thresh: float = 0.7,
    match_iou_thresh: float = 0.3,
    second_match_iou_thresh: float = 0.2,
    track_buffer: int = 5,
    min_hits: int = 3,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Run dependency-free ByteTrack-style two-stage score/IoU association."""
    validate_byte_track_parameters(
        low_thresh,
        high_thresh,
        new_track_thresh,
        match_iou_thresh,
        second_match_iou_thresh,
        track_buffer,
        min_hits,
    )

    stats = {
        "n_rescue_matches": 0,
        "n_gap_candidates": 0,
        "n_byte_first_stage_matches": 0,
        "n_byte_second_stage_matches": 0,
        "n_high_score_detections": 0,
        "n_low_score_detections": 0,
    }
    if not dets_by_frame:
        return [], stats

    active_tracks: List[Any] = []
    finished_tracks: List[Any] = []
    next_track_id = 1
    frame_ids = sorted(dets_by_frame)

    for frame_id in range(frame_ids[0], frame_ids[-1] + 1):
        tracked_before_prediction = [
            bool(track.is_confirmed and track.track_state == "confirmed")
            for track in active_tracks
        ]
        for track in active_tracks:
            # Track.predict() bases dt on last matched frame. Calling it on every
            # missing frame would therefore apply dt=1,2,3... cumulatively.
            track.kf.predict(dt=1.0)
            track.age += 1
            track.misses += 1

        detections = dets_by_frame.get(frame_id, [])
        high_detections = [
            detection for detection in detections
            if float(detection.get("score", 1.0)) >= high_thresh
        ]
        low_detections = [
            detection for detection in detections
            if low_thresh
            <= float(detection.get("score", 1.0))
            < high_thresh
        ]
        stats["n_high_score_detections"] += len(high_detections)
        stats["n_low_score_detections"] += len(low_detections)

        confirmed_indices = [
            index for index, track in enumerate(active_tracks)
            if track.is_confirmed
        ]
        tentative_indices = [
            index for index, track in enumerate(active_tracks)
            if not track.is_confirmed
        ]
        confirmed_matches, _, unmatched_high_after_confirmed = (
            _match_byte_tracks_by_iou(
                [active_tracks[index] for index in confirmed_indices],
                high_detections,
                match_iou_thresh,
            )
        )
        first_matches = [
            (confirmed_indices[track_index], detection_index, iou)
            for track_index, detection_index, iou in confirmed_matches
        ]

        remaining_high_detections = [
            high_detections[index]
            for index in unmatched_high_after_confirmed
        ]
        tentative_matches, _, unmatched_remaining_high = (
            _match_byte_tracks_by_iou(
                [active_tracks[index] for index in tentative_indices],
                remaining_high_detections,
                match_iou_thresh,
            )
        )
        first_matches.extend(
            (
                tentative_indices[track_index],
                unmatched_high_after_confirmed[detection_index],
                iou,
            )
            for track_index, detection_index, iou in tentative_matches
        )
        unmatched_high_indices = [
            unmatched_high_after_confirmed[index]
            for index in unmatched_remaining_high
        ]
        matched_track_indices = set()
        for track_index, detection_index, iou in first_matches:
            track = active_tracks[track_index]
            track.update(
                high_detections[detection_index],
                frame_id,
                ot_prob=np.nan,
                match_cost=1.0 - iou,
            )
            if track.hits >= min_hits:
                track.is_confirmed = True
                track.track_state = "confirmed"
            matched_track_indices.add(track_index)
        stats["n_byte_first_stage_matches"] += len(first_matches)

        unmatched_track_indices = [
            index for index in range(len(active_tracks))
            if index not in matched_track_indices
        ]
        # ByteTrack's low-score association only rescues tracks that were
        # actively tracked at the beginning of this frame. Tentative and
        # already-lost tracks must be associated with high-score detections.
        second_track_indices = [
            index for index in unmatched_track_indices
            if tracked_before_prediction[index]
        ]
        second_tracks = [
            active_tracks[index] for index in second_track_indices
        ]
        second_matches, _, _ = _match_byte_tracks_by_iou(
            second_tracks, low_detections, second_match_iou_thresh
        )
        for local_track_index, detection_index, iou in second_matches:
            track_index = second_track_indices[local_track_index]
            track = active_tracks[track_index]
            track.update(
                low_detections[detection_index],
                frame_id,
                ot_prob=np.nan,
                match_cost=1.0 - iou,
            )
            if track.hits >= min_hits:
                track.is_confirmed = True
                track.track_state = "confirmed"
            matched_track_indices.add(track_index)
        stats["n_byte_second_stage_matches"] += len(second_matches)

        still_active = []
        for track_index, track in enumerate(active_tracks):
            if track_index in matched_track_indices:
                still_active.append(track)
                continue
            max_misses = track_buffer if track.is_confirmed else 1
            if track.misses > max_misses:
                track.track_state = "finished"
                track.end_reason = "byte_unmatched_timeout"
                finished_tracks.append(track)
            else:
                track.track_state = "lost"
                still_active.append(track)
        active_tracks = still_active

        for detection_index in unmatched_high_indices:
            detection = high_detections[detection_index]
            if float(detection.get("score", 1.0)) < new_track_thresh:
                continue
            active_tracks.append(_new_byte_track(
                tracker_mod,
                detection,
                frame_id,
                next_track_id,
                min_hits,
            ))
            next_track_id += 1

    for track in active_tracks:
        track.track_state = "finished"
        track.end_reason = track.end_reason or "end_of_sequence"
        finished_tracks.append(track)
    confirmed_tracks = [track for track in finished_tracks if track.is_confirmed]
    confirmed_tracks, filter_stats = tracker_mod.postprocess_tracks(
        confirmed_tracks,
        kf_params={},
        enable_gap_fill=False,
        min_contiguous_observed=min_hits,
        min_observed_hits=min_hits,
    )
    stats["byte_removed_short_tracks"] = int(
        filter_stats.get("removed_short", 0)
    )
    return sorted(
        confirmed_tracks, key=lambda track: int(track.track_id)
    ), stats


# ============================================================
# T1/T2/T2b/T3: OT-Kalman variants
# ============================================================

class AblationOTKalmanTracker:
    """轻量 wrapper：通过 enable_rescue 控制 rescue matching，不修改原 tracker 文件。"""

    def __init__(self, tracker_mod: Any, enable_rescue: bool = True, **kwargs: Any):
        self._mod = tracker_mod
        self._tracker = tracker_mod.OTKalmanTracker(**kwargs)
        self.enable_rescue = enable_rescue
        self._orig_rescue = tracker_mod.rescue_unmatched_matches

    def step(self, frame_id: int, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        if not self.enable_rescue:
            self._mod.rescue_unmatched_matches = lambda *a, **k: ([], 0)
        else:
            self._mod.rescue_unmatched_matches = self._orig_rescue
        try:
            return self._tracker.step(frame_id, detections)
        finally:
            self._mod.rescue_unmatched_matches = self._orig_rescue

    def finish(self) -> List[Any]:
        return self._tracker.finish()

    @property
    def inner(self) -> Any:
        return self._tracker


@dataclass
class TrackerRunConfig:
    enable_rescue: bool = False
    enable_gap_fill: bool = False
    enable_prebirth: bool = False
    max_center_dist: float = 45.0
    max_age: int = 4
    max_age_tentative: int = 1
    max_age_confirmed: int = 5
    min_hits: int = 3
    min_contiguous_observed: int = 3
    min_observed_hits: int = 3
    min_track_len: int = 2
    max_gap_fill: int = 10
    ot_reg: float = 0.08
    image_width: Optional[int] = None
    image_height: Optional[int] = None


def run_ot_kalman_variant(
    tracker_mod: Any,
    dets_by_frame: Dict[int, List[Dict[str, Any]]],
    cfg: TrackerRunConfig,
    image_dir: Optional[Path] = None,
    image_exts: str = ".png,.jpg,.jpeg,.tif,.tiff",
    progress_label: str = "",
) -> Tuple[List[Any], Dict[str, Any]]:
    tracker = AblationOTKalmanTracker(
        tracker_mod,
        enable_rescue=cfg.enable_rescue,
        max_age=cfg.max_age,
        max_age_tentative=cfg.max_age_tentative,
        max_age_confirmed=cfg.max_age_confirmed,
        min_hits=cfg.min_hits,
        min_contiguous_observed=cfg.min_contiguous_observed,
        max_center_dist=cfg.max_center_dist,
        ot_reg=cfg.ot_reg,
        image_width=cfg.image_width,
        image_height=cfg.image_height,
    )

    frame_ids = sorted(dets_by_frame.keys())
    run_stats = {
        "n_rescue_matches": 0,
        "n_gap_candidates": 0,
        "gap_filled_points": 0,
        "prebirth_points_added": 0,
    }

    label = f"[{progress_label}] " if progress_label else ""
    frame_lo, frame_hi = frame_ids[0], frame_ids[-1]
    total_frames = frame_hi - frame_lo + 1
    heartbeat = max(1000, total_frames // 10)
    t_start = time.time()
    print(f"  {label}OT-Kalman 步进: {total_frames} 帧 ({frame_lo}..{frame_hi})", flush=True)
    for processed, frame_id in enumerate(range(frame_lo, frame_hi + 1), start=1):
        detections = dets_by_frame.get(frame_id, [])
        stats = tracker.step(frame_id, detections)
        run_stats["n_rescue_matches"] += int(stats.get("rescue_matches", 0))
        if processed % heartbeat == 0 or processed == total_frames:
            elapsed = time.time() - t_start
            print(
                f"  {label}OT-Kalman 步进 {processed}/{total_frames} 帧 | {elapsed:.1f}s",
                flush=True,
            )

    tracks = tracker.finish()
    kf_params = tracker_mod.kf_params_from_tracker(tracker.inner)
    tracks, pp_stats = tracker_mod.postprocess_tracks(
        tracks,
        kf_params=kf_params,
        enable_gap_fill=cfg.enable_gap_fill,
        max_gap_fill=cfg.max_gap_fill,
        min_contiguous_observed=cfg.min_contiguous_observed,
        min_observed_hits=cfg.min_observed_hits,
    )
    run_stats["gap_filled_points"] = int(pp_stats.get("gap_filled_points", 0))
    run_stats["n_gap_candidates"] = int(pp_stats.get("gap_filled_points", 0))

    if cfg.enable_prebirth and image_dir is not None and image_dir.is_dir():
        frame_image_map = tracker_mod.build_frame_image_map(image_dir, image_exts)
        if frame_image_map:
            tracks, prebirth_stats = tracker_mod.recover_prebirth_dark_particles(
                tracks,
                frame_image_map=frame_image_map,
            )
            run_stats["prebirth_points_added"] = int(prebirth_stats.get("prebirth_points_added", 0))

    return tracks, run_stats


def variant_tracker_config(name: str, cli_args: argparse.Namespace) -> TrackerRunConfig:
    if name not in VARIANT_ORDER:
        raise ValueError(
            f"Unknown tracking variant: {name}; allowed: {', '.join(VARIANT_ORDER)}"
        )
    base = TrackerRunConfig(
        max_center_dist=cli_args.max_center_dist,
        max_age=cli_args.max_age,
        max_age_tentative=cli_args.max_age_tentative,
        max_age_confirmed=cli_args.max_age_confirmed,
        min_hits=cli_args.min_hits,
        ot_reg=cli_args.sinkhorn_reg,
        image_width=cli_args.image_width,
        image_height=cli_args.image_height,
    )
    # OT 变体沿用 ot_kalman_tracker 推荐默认 max_age=4；T0 单独使用 --max-age。
    base.max_age = 4
    if name == "T1_basic_ot_kalman":
        base.enable_rescue = False
        base.enable_gap_fill = False
        base.enable_prebirth = False
    elif name == "T2_ot_kalman_rescue":
        base.enable_rescue = True
        base.enable_gap_fill = False
        base.enable_prebirth = False
    elif name == "T2b_ot_kalman_rescue_gap":
        base.enable_rescue = True
        base.enable_gap_fill = True
        base.enable_prebirth = False
    elif name == "T3_full_tracker":
        base.enable_rescue = True
        base.enable_gap_fill = True
        base.enable_prebirth = True
    return base

# ============================================================
# Image map & cropping
# ============================================================

def build_frame_image_map(
    image_dir: Path,
    image_exts: str,
    tracks_df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[int, Path], Dict[str, Path]]:
    exts = {
        e if e.startswith(".") else "." + e
        for e in image_exts.lower().split(",")
        if e.strip()
    }
    paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: natural_sort_key(p.name),
    )
    frame_map: Dict[int, Path] = {i: p for i, p in enumerate(paths)}
    name_map: Dict[str, Path] = {p.name: p for p in paths}
    for p in paths:
        name_map.setdefault(p.stem, p)

    if tracks_df is not None and "image_name" in tracks_df.columns:
        for _, row in tracks_df.drop_duplicates("frame_id").sort_values("frame_id").iterrows():
            fname = str(row.get("image_name", "")).strip()
            fid = int(row["frame_id"])
            if fname and fname in name_map:
                frame_map[fid] = name_map[fname]
            elif fname:
                stem = Path(fname).stem
                if stem in name_map:
                    frame_map[fid] = name_map[stem]
    return frame_map, name_map


def resolve_source_image(
    row: pd.Series,
    frame_image_map: Dict[int, Path],
    image_name_map: Dict[str, Path],
) -> Optional[Path]:
    fname = str(row.get("image_name", "")).strip()
    if fname:
        if fname in image_name_map:
            return image_name_map[fname]
        stem = Path(fname).stem
        if stem in image_name_map:
            return image_name_map[stem]
    fid = int(row["frame_id"])
    return frame_image_map.get(fid)


def crop_track_sequence(
    track_df: pd.DataFrame,
    frame_image_map: Dict[int, Path],
    image_name_map: Dict[str, Path],
    output_crop_dir: Path,
    crop_padding: float = 4.0,
    crop_format: str = "png",
    jpeg_quality: int = 90,
    skip_existing: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    """单轨裁剪（兼容入口）；批量请用 crop_all_tracks。"""
    track_df = track_df.sort_values("frame_id").reset_index(drop=True)
    track_id = int(track_df["track_id"].iloc[0])
    bundled = pd.DataFrame({**{c: track_df[c] for c in track_df.columns}})
    result = crop_all_tracks(
        bundled,
        output_crop_dir.parent,
        frame_image_map,
        image_name_map,
        crop_padding,
        crop_format=crop_format,
        jpeg_quality=jpeg_quality,
        skip_existing=skip_existing,
        progress_label=f"track_{track_id:04d}",
    )[track_id]
    return result["crop_paths"], result["crop_meta"]


def _crop_ext(crop_format: str) -> str:
    fmt = (crop_format or "png").lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        return ".jpg"
    return ".png"


def _crop_plan_fingerprint(
    tracks_df: pd.DataFrame,
    crop_padding: float,
    crop_format: str,
    jpeg_quality: int,
) -> str:
    cols = [
        c for c in (
            "track_id", "frame_id", "image_name", "x1", "y1", "x2", "y2",
            "point_type",
        )
        if c in tracks_df.columns
    ]
    plan = tracks_df[cols].copy()
    sort_cols = [c for c in ("track_id", "frame_id", "point_type") if c in cols]
    if sort_cols:
        plan = plan.sort_values(sort_cols)
    for c in ("x1", "y1", "x2", "y2"):
        if c in plan.columns:
            plan[c] = pd.to_numeric(plan[c], errors="coerce").round(4)
    payload = {
        "crop_padding": float(crop_padding),
        "crop_format": str(crop_format).lower(),
        "jpeg_quality": int(jpeg_quality),
        "tracks": plan.to_dict(orient="records"),
    }
    raw = json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def crop_all_tracks(
    tracks_df: pd.DataFrame,
    crop_root: Path,
    frame_image_map: Dict[int, Path],
    image_name_map: Dict[str, Path],
    crop_padding: float,
    *,
    crop_format: str = "png",
    jpeg_quality: int = 90,
    skip_existing: bool = False,
    progress_label: str = "",
) -> Dict[int, Dict[str, Any]]:
    """
    按源全帧分组裁剪：每张全帧只解码一次，避免「按 track 循环反复读同图」。
    """
    from collections import defaultdict

    crop_root = Path(crop_root)
    crop_root.mkdir(parents=True, exist_ok=True)
    ext = _crop_ext(crop_format)
    pad = float(crop_padding)

    plan_path = crop_root / "crop_plan.json"
    plan_hash = _crop_plan_fingerprint(
        tracks_df, pad, crop_format, int(jpeg_quality)
    )
    effective_skip_existing = False
    if skip_existing and plan_path.is_file():
        try:
            old_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            effective_skip_existing = old_plan.get("fingerprint") == plan_hash
        except Exception:
            effective_skip_existing = False
    if skip_existing and not effective_skip_existing:
        warnings.warn(
            f"[{progress_label or 'crop'}] 已存在裁剪缺少匹配的计划指纹；"
            "为避免复用旧 bbox 生成的 crop，本次将重新裁剪。",
            stacklevel=2,
        )
    skip_existing = effective_skip_existing

    # job: 待裁剪任务；按源图分组
    jobs_by_src: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    issues_by_track: Dict[int, List[str]] = defaultdict(list)
    input_counts = {
        int(track_id): int(len(group))
        for track_id, group in tracks_df.groupby("track_id")
    }
    n_planned = 0
    for track_id, group in tracks_df.groupby("track_id"):
        tid = int(track_id)
        track_dir = crop_root / f"track_{tid:04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for track_row_idx, (_, row) in enumerate(
            group.sort_values("frame_id").iterrows()
        ):
            try:
                src = resolve_source_image(row, frame_image_map, image_name_map)
            except Exception as exc:
                issues_by_track[tid].append(f"invalid_source_reference:{exc}")
                continue
            if src is None or not src.is_file():
                issues_by_track[tid].append("missing_source_image")
                continue
            try:
                x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            except (TypeError, ValueError):
                issues_by_track[tid].append("invalid_box")
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                issues_by_track[tid].append("invalid_box")
                continue
            try:
                frame_id = int(row["frame_id"])
            except (TypeError, ValueError, OverflowError):
                issues_by_track[tid].append("invalid_frame_id")
                continue
            try:
                cx = float(row.get("cx", (x1 + x2) / 2))
                cy = float(row.get("cy", (y1 + y2) / 2))
                if not math.isfinite(cx) or not math.isfinite(cy):
                    raise ValueError("non-finite center")
            except (TypeError, ValueError, OverflowError):
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
            point_type = str(row.get("point_type", "detected") or "detected")
            out_path = track_dir / f"frame_{frame_id:06d}_{point_type}{ext}"
            job = {
                "track_id": tid,
                "track_row_idx": int(track_row_idx),
                "frame_id": frame_id,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": cx,
                "cy": cy,
                "point_type": point_type,
                "det_score": row.get("det_score", np.nan),
                "ot_prob": row.get("ot_prob", np.nan),
                "match_cost": row.get("match_cost", np.nan),
                "gap_fill_method": row.get("gap_fill_method", ""),
                "source_image": str(src),
                "crop_path": out_path,
            }
            jobs_by_src[src].append(job)
            n_planned += 1

    n_src = len(jobs_by_src)
    label = f"[{progress_label}] " if progress_label else ""
    print(f"  {label}裁剪: {n_planned} crops / {n_src} 全帧 (format={ext}, skip_existing={skip_existing})")

    ok_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    n_written = 0
    n_skipped = 0
    for si, (src, jobs) in enumerate(jobs_by_src.items(), start=1):
        need_open = True
        if skip_existing and all(j["crop_path"].is_file() for j in jobs):
            for j in jobs:
                ok_by_track[j["track_id"]].append(j)
                n_skipped += 1
            need_open = False
        if need_open:
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w_img, h_img = im.size
                    for j in jobs:
                        out_path: Path = j["crop_path"]
                        if skip_existing and out_path.is_file():
                            ok_by_track[j["track_id"]].append(j)
                            n_skipped += 1
                            continue
                        x1p = max(0, int(math.floor(min(j["x1"], j["x2"]) - pad)))
                        y1p = max(0, int(math.floor(min(j["y1"], j["y2"]) - pad)))
                        x2p = min(w_img, int(math.ceil(max(j["x1"], j["x2"]) + pad)))
                        y2p = min(h_img, int(math.ceil(max(j["y1"], j["y2"]) + pad)))
                        if x2p <= x1p or y2p <= y1p:
                            issues_by_track[j["track_id"]].append("invalid_crop_box")
                            continue
                        crop = im.crop((x1p, y1p, x2p, y2p))
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if ext == ".jpg":
                            crop.save(out_path, format="JPEG", quality=int(jpeg_quality), optimize=True)
                        else:
                            crop.save(out_path, format="PNG", optimize=False)
                        ok_by_track[j["track_id"]].append(j)
                        n_written += 1
            except Exception as exc:
                reason = f"image_read_error:{src.name}:{exc}"
                for job in jobs:
                    issues_by_track[job["track_id"]].append(reason)
                continue
        if si % 200 == 0 or si == n_src:
            print(
                f"  {label}裁剪进度 {si}/{n_src} 全帧 | "
                f"written={n_written} skipped={n_skipped}"
            )

    result: Dict[int, Dict[str, Any]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for tid in sorted(input_counts):
        items = ok_by_track.get(tid, [])
        items = sorted(items, key=lambda d: (d["frame_id"], d["point_type"]))
        paths: List[str] = []
        meta_rows: List[Dict[str, Any]] = []
        for idx, j in enumerate(items):
            paths.append(str(j["crop_path"]))
            meta_rows.append({
                "track_id": tid,
                "track_row_idx": j.get("track_row_idx", idx),
                "frame_id": j["frame_id"],
                "source_image": j["source_image"],
                "crop_path": str(j["crop_path"]),
                "x1": j["x1"], "y1": j["y1"], "x2": j["x2"], "y2": j["y2"],
                "cx": j["cx"], "cy": j["cy"],
                "point_type": j["point_type"],
                "det_score": j["det_score"],
                "ot_prob": j["ot_prob"],
                "match_cost": j["match_cost"],
                "gap_fill_method": j["gap_fill_method"],
                "crop_idx": idx,
            })
        reasons = list(dict.fromkeys(issues_by_track.get(tid, [])))
        n_success_crops = len(paths)
        n_failed_crops = max(input_counts[tid] - n_success_crops, 0)
        if n_success_crops == 0:
            status = "error"
        elif n_failed_crops == 0:
            status = "ok"
        else:
            status = "partial"
        if n_failed_crops and not reasons:
            reasons.append("crop_count_mismatch")
        error_reason = "|".join(reasons)
        crop_success_rate = float(n_success_crops / max(input_counts[tid], 1))
        result[tid] = {
            "track_id": tid,
            "status": status,
            "error_reason": error_reason,
            "crop_paths": paths,
            "crop_meta": pd.DataFrame(meta_rows),
            "n_success_crops": n_success_crops,
            "n_failed_crops": n_failed_crops,
            "crop_success_rate": crop_success_rate,
        }
        manifest_rows.append({
            "track_id": tid,
            "status": status,
            "error_reason": error_reason,
            "n_input_rows": input_counts[tid],
            "n_crops": len(paths),
            "n_success_crops": n_success_crops,
            "n_failed_crops": n_failed_crops,
            "crop_success_rate": crop_success_rate,
        })
    pd.DataFrame(manifest_rows).to_csv(
        crop_root / "crop_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plan_path.write_text(
        json.dumps(
            {
                "fingerprint": plan_hash,
                "crop_padding": pad,
                "crop_format": ext,
                "jpeg_quality": int(jpeg_quality),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result



# ============================================================
# Precomputed ROI cache (fallback when full-frame images are unavailable)
# ============================================================

ROI_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_roi_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ROI_EXTENSIONS],
        key=lambda p: natural_sort_key(p.name),
    )


def _candidate_track_dirs(roi_root: Path, track_id: int) -> List[Path]:
    return [
        roi_root / f"track_{track_id}",
        roi_root / f"track_{track_id:04d}",
    ]


def _parse_frame_id_from_roi_name(path: Path) -> Optional[int]:
    # 本脚本 crop_all_tracks 的标准命名：frame_000123_detected.png
    m = re.search(r"(?:^|_)frame[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 宽松兼容：若文件名仅含一个明显的长整数，则将其作为 frame 候选。
    nums = re.findall(r"\d+", path.stem)
    if len(nums) == 1 and len(nums[0]) >= 4:
        return int(nums[0])
    return None


def _bbox_tuple_from_row(row: pd.Series) -> Tuple[float, float, float, float]:
    return tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]


def build_precomputed_roi_cache(
    roi_root: Path,
    reference_tracks_df: pd.DataFrame,
) -> Tuple[Dict[int, List[Dict[str, Any]]], pd.DataFrame]:
    """
    由既有 T3 track_crop + T3 tracks_long.csv 构造 frame_id -> ROI entries。

    优先按 ROI 文件名中的 frame_id / image_name 对齐；若无法解析，则仅在
    该 track 的 ROI 数量与轨迹行数完全一致时按自然排序一一配对，避免静默错位。
    """
    roi_root = Path(roi_root)
    if not roi_root.is_dir():
        raise FileNotFoundError(f"ROI root not found: {roi_root}")
    required = {"track_id", "frame_id", "x1", "y1", "x2", "y2"}
    missing = sorted(required - set(reference_tracks_df.columns))
    if missing:
        raise ValueError(f"Reference tracks CSV missing required columns: {missing}")

    cache: Dict[int, List[Dict[str, Any]]] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for tid, group in reference_tracks_df.groupby("track_id"):
        tid = int(tid)
        track_dir = next((p for p in _candidate_track_dirs(roi_root, tid) if p.is_dir()), None)
        if track_dir is None:
            manifest_rows.append({"track_id": tid, "status": "missing_track_dir", "n_rows": len(group), "n_roi": 0})
            continue
        images = list_roi_images(track_dir)
        g = group.sort_values("frame_id").reset_index(drop=True)
        if not images:
            manifest_rows.append({"track_id": tid, "status": "empty_track_dir", "n_rows": len(g), "n_roi": 0})
            continue

        row_by_frame = {int(r["frame_id"]): r for _, r in g.iterrows()}
        row_by_stem: Dict[str, pd.Series] = {}
        if "image_name" in g.columns:
            for _, r in g.iterrows():
                name = str(r.get("image_name", "")).strip()
                if name:
                    row_by_stem.setdefault(Path(name).stem, r)

        paired: List[Tuple[Path, pd.Series]] = []
        used_row_ids: Set[int] = set()
        unresolved: List[Path] = []
        # 1) frame / image_name 明确匹配
        for img in images:
            row = None
            fid = _parse_frame_id_from_roi_name(img)
            if fid is not None and fid in row_by_frame:
                row = row_by_frame[fid]
            elif img.stem in row_by_stem:
                row = row_by_stem[img.stem]
            if row is None:
                unresolved.append(img)
                continue
            ridx = int(row.name)
            if ridx in used_row_ids:
                unresolved.append(img)
                continue
            used_row_ids.add(ridx)
            paired.append((img, row))

        # 2) 无法从文件名识别时，只有在总数完全一致且前面没有混合匹配时才允许顺序配对
        if unresolved:
            if not paired and len(images) == len(g):
                paired = list(zip(images, [r for _, r in g.iterrows()]))
                unresolved = []
            else:
                manifest_rows.append({
                    "track_id": tid,
                    "status": "ambiguous_roi_alignment",
                    "n_rows": len(g),
                    "n_roi": len(images),
                    "n_paired": len(paired),
                    "n_unresolved": len(unresolved),
                })

        for img, row in paired:
            fid = int(row["frame_id"])
            entry = {
                "track_id": tid,
                "frame_id": fid,
                "point_type": str(row.get("point_type", "detected") or "detected"),
                "bbox": _bbox_tuple_from_row(row),
                "crop_path": str(img),
                "image_name": str(row.get("image_name", "")),
            }
            cache.setdefault(fid, []).append(entry)
        if not any(r.get("track_id") == tid for r in manifest_rows):
            manifest_rows.append({
                "track_id": tid,
                "status": "ok" if len(paired) == len(g) else "partial",
                "n_rows": len(g),
                "n_roi": len(images),
                "n_paired": len(paired),
                "n_unresolved": len(unresolved),
            })

    return cache, pd.DataFrame(manifest_rows)


def resolve_roi_cache_entry(
    row: pd.Series,
    roi_cache: Dict[int, List[Dict[str, Any]]],
    bbox_tolerance: float,
) -> Optional[Dict[str, Any]]:
    fid = int(row["frame_id"])
    candidates = roi_cache.get(fid, [])
    if not candidates:
        return None
    target_bbox = np.asarray(_bbox_tuple_from_row(row), dtype=float)
    point_type = str(row.get("point_type", "detected") or "detected")
    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for entry in candidates:
        bbox = np.asarray(entry["bbox"], dtype=float)
        err = float(np.max(np.abs(bbox - target_bbox)))
        # 同类型优先；但 detected/candidate 的 bbox 若完全相同仍允许复用。
        type_penalty = 0 if str(entry.get("point_type", "detected")) == point_type else 1
        scored.append((err, type_penalty, entry))
    scored.sort(key=lambda x: (x[0], x[1]))
    if not scored or scored[0][0] > float(bbox_tolerance):
        return None
    return scored[0][2]


def build_crop_data_from_roi_cache(
    tracks_df: pd.DataFrame,
    roi_cache: Dict[int, List[Dict[str, Any]]],
    *,
    bbox_tolerance: float = 0.75,
    min_success_rate: float = 0.95,
    progress_label: str = "",
) -> Dict[int, Dict[str, Any]]:
    """将当前 tracking 变体的轨迹点映射到既有 ROI 缓存。"""
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("roi_cache min_success_rate must be in [0, 1]")
    result: Dict[int, Dict[str, Any]] = {}
    total_rows = total_hit = 0
    for tid, group in tracks_df.groupby("track_id"):
        tid = int(tid)
        g = group.sort_values("frame_id").reset_index(drop=True)
        paths: List[str] = []
        meta_rows: List[Dict[str, Any]] = []
        miss_reasons: List[str] = []
        for track_row_idx, (_, row) in enumerate(g.iterrows()):
            total_rows += 1
            entry = resolve_roi_cache_entry(row, roi_cache, bbox_tolerance)
            if entry is None:
                miss_reasons.append(f"frame={int(row['frame_id'])}:bbox_not_in_roi_cache")
                continue
            total_hit += 1
            paths.append(str(entry["crop_path"]))
            meta_rows.append({
                "track_id": tid,
                "track_row_idx": int(track_row_idx),
                "crop_idx": len(paths) - 1,
                "frame_id": int(row["frame_id"]),
                "source_image": str(entry.get("crop_path", "")),
                "crop_path": str(entry["crop_path"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
                "cx": float(row.get("cx", (float(row["x1"]) + float(row["x2"])) * 0.5)),
                "cy": float(row.get("cy", (float(row["y1"]) + float(row["y2"])) * 0.5)),
                "point_type": str(row.get("point_type", "detected") or "detected"),
            })
        n_input = len(g)
        n_success = len(paths)
        n_failed = n_input - n_success
        rate = float(n_success / max(n_input, 1))
        if n_success == 0 or rate < min_success_rate:
            status = "error"
        elif n_failed == 0:
            status = "ok"
        else:
            status = "partial"
        reason = "|".join(miss_reasons[:20])
        if len(miss_reasons) > 20:
            reason += f"|...+{len(miss_reasons)-20}_more"
        result[tid] = {
            "track_id": tid,
            "status": status,
            "error_reason": reason,
            "crop_paths": paths,
            "crop_meta": pd.DataFrame(meta_rows),
            "n_success_crops": n_success,
            "n_failed_crops": n_failed,
            "crop_success_rate": rate,
        }
    overall = float(total_hit / max(total_rows, 1))
    print(
        f"  [{progress_label or 'roi-cache'}] ROI cache reuse: "
        f"{total_hit}/{total_rows}={overall:.4f}"
    )
    return result


def resolve_image_source_mode(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "image_source", "auto")).lower()
    image_dir = Path(args.image_dir) if getattr(args, "image_dir", None) else None
    roi_root = Path(args.roi_root) if getattr(args, "roi_root", None) else None
    full_ok = bool(image_dir and image_dir.is_dir())
    roi_ok = bool(roi_root and roi_root.is_dir())
    if requested == "auto":
        if full_ok:
            return "full_frame"
        if roi_ok:
            return "roi_cache"
        raise FileNotFoundError(
            "Neither --image-dir nor --roi-root is available. Provide original full-frame images "
            "or the precomputed T3 track_crop directory."
        )
    if requested == "full_frame":
        if not full_ok:
            raise FileNotFoundError(f"Full-frame image directory not found: {image_dir}")
        return requested
    if requested == "roi_cache":
        if not roi_ok:
            raise FileNotFoundError(f"ROI root not found: {roi_root}")
        return requested
    raise ValueError(f"Unknown --image-source={requested}")

# ============================================================
# Tracking quality metrics
# ============================================================

def compute_track_quality_proxy(track_df: pd.DataFrame) -> Dict[str, Any]:
    df = track_df.sort_values("frame_id").reset_index(drop=True)
    n_points = len(df)
    pt = (
        df["point_type"].fillna("detected").astype(str)
        if "point_type" in df.columns
        else pd.Series(["detected"] * n_points)
    )
    n_detected = int((pt == "detected").sum())
    n_candidate = int((pt == "candidate").sum())
    n_prebirth = int((pt == "prebirth_candidate").sum())
    candidate_ratio = n_candidate / max(n_points, 1)
    prebirth_ratio = n_prebirth / max(n_points, 1)

    ot_prob = pd.to_numeric(df.get("ot_prob", pd.Series(dtype=float)), errors="coerce")
    match_cost = pd.to_numeric(df.get("match_cost", pd.Series(dtype=float)), errors="coerce")
    center_match_distance = pd.to_numeric(
        df.get("center_match_distance", pd.Series(dtype=float)), errors="coerce"
    )
    n_rescue = int((ot_prob == -1.0).sum()) if len(ot_prob) else 0

    detected_df = df[pt == "detected"].reset_index(drop=True)
    frame_id = pd.to_numeric(
        detected_df.get("frame_id", pd.Series(dtype=float)), errors="coerce"
    )
    cx = pd.to_numeric(
        detected_df.get("cx", pd.Series(dtype=float)), errors="coerce"
    )
    cy = pd.to_numeric(
        detected_df.get("cy", pd.Series(dtype=float)), errors="coerce"
    )
    center_speeds: List[float] = []
    for i in range(1, len(detected_df)):
        if not (
            pd.notna(frame_id.iloc[i])
            and pd.notna(frame_id.iloc[i - 1])
            and pd.notna(cx.iloc[i])
            and pd.notna(cx.iloc[i - 1])
            and pd.notna(cy.iloc[i])
            and pd.notna(cy.iloc[i - 1])
        ):
            continue
        dt = int(frame_id.iloc[i] - frame_id.iloc[i - 1])
        if dt <= 0:
            continue
        dist = float(math.hypot(cx.iloc[i] - cx.iloc[i - 1], cy.iloc[i] - cy.iloc[i - 1]))
        center_speeds.append(dist / dt)

    median_speed = float(np.median(center_speeds)) if center_speeds else 0.0
    max_speed = float(np.max(center_speeds)) if center_speeds else 0.0
    speed_ratio = max_speed / max(median_speed, 1e-6) if center_speeds else 0.0
    n_large = int(
        sum(
            1
            for speed in center_speeds
            if speed > max(3.0 * median_speed, median_speed + 5.0)
        )
    )

    valid_ot = ot_prob.replace(-1.0, np.nan)
    mean_ot = float(valid_ot.mean(skipna=True)) if valid_ot.notna().any() else np.nan
    mean_cost = float(match_cost.mean(skipna=True)) if match_cost.notna().any() else np.nan
    mean_center_dist = (
        float(center_match_distance.mean(skipna=True))
        if center_match_distance.notna().any()
        else np.nan
    )

    suspicious = speed_ratio > 4.0 or n_large >= 2

    return {
        "track_len": n_points,
        "n_detected": n_detected,
        "n_candidate": n_candidate,
        "n_prebirth": n_prebirth,
        "candidate_ratio": float(candidate_ratio),
        "prebirth_ratio": float(prebirth_ratio),
        "mean_match_cost": mean_cost,
        "mean_ot_prob": mean_ot,
        "mean_center_match_distance": mean_center_dist,
        "n_rescue_matches": n_rescue,
        "n_gap_candidates": n_candidate,
        "max_center_speed": max_speed,
        "speed_ratio": float(speed_ratio),
        "n_large_center_jumps": n_large,
        "suspicious_merge_proxy": suspicious,
    }

def aggregate_tracking_metrics(
    tracks_df: pd.DataFrame,
    track_summary_df: Optional[pd.DataFrame],
    run_stats: Dict[str, Any],
    short_len: int = 5,
    long_len: int = 30,
) -> Dict[str, Any]:
    per_track = []
    for track_id, group in tracks_df.groupby("track_id"):
        m = compute_track_quality_proxy(group)
        m["track_id"] = int(track_id)
        per_track.append(m)

    if not per_track:
        return {
            "n_tracks": 0,
            "mean_track_len": 0.0,
            "median_track_len": 0.0,
            "mean_detected_len": 0.0,
            "median_detected_len": 0.0,
            "short_track_ratio": 0.0,
            "long_track_ratio": 0.0,
            "n_short_detected_tracks": 0,
            "n_long_detected_tracks": 0,
            "short_detected_track_ratio": 0.0,
            "long_detected_track_ratio": 0.0,
            "candidate_ratio": 0.0,
            "candidate_ratio_macro": 0.0,
            "prebirth_ratio": 0.0,
            "prebirth_ratio_macro": 0.0,
            "n_short_tracks": 0,
            "n_long_tracks": 0,
            "mean_match_cost": np.nan,
            "mean_ot_prob": np.nan,
            "mean_center_match_distance": np.nan,
            "n_rescue_matches": int(run_stats.get("n_rescue_matches", 0)),
            "n_gap_candidates": int(run_stats.get("n_gap_candidates", 0)),
            "n_byte_first_stage_matches": int(
                run_stats.get("n_byte_first_stage_matches", 0)
            ),
            "n_byte_second_stage_matches": int(
                run_stats.get("n_byte_second_stage_matches", 0)
            ),
            "n_high_score_detections": int(
                run_stats.get("n_high_score_detections", 0)
            ),
            "n_low_score_detections": int(
                run_stats.get("n_low_score_detections", 0)
            ),
            "n_byte_removed_short_tracks": int(
                run_stats.get("byte_removed_short_tracks", 0)
            ),
            "suspicious_merge_proxy_count": 0,
            "large_jump_track_count": 0,
        }

    lens = [p["track_len"] for p in per_track]
    det_lens = [p["n_detected"] for p in per_track]
    total_points = max(sum(lens), 1)

    def finite_mean(values: List[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else np.nan

    # point-level (micro) metrics prevent numerous short fragments from receiving
    # the same weight as long trajectories. Macro values are retained for diagnosis.
    candidate_micro = float(sum(p["n_candidate"] for p in per_track) / total_points)
    prebirth_micro = float(sum(p["n_prebirth"] for p in per_track) / total_points)

    # OT probability/cost are intentionally NaN for Center-NN and therefore are
    # only comparable across OT-based variants.
    track_match_cost = pd.to_numeric(
        tracks_df.get("match_cost", pd.Series(dtype=float)), errors="coerce"
    )
    track_ot_prob = pd.to_numeric(
        tracks_df.get("ot_prob", pd.Series(dtype=float)), errors="coerce"
    ).replace(-1.0, np.nan)
    center_dist = pd.to_numeric(
        tracks_df.get("center_match_distance", pd.Series(dtype=float)), errors="coerce"
    )

    return {
        "n_tracks": len(per_track),
        "mean_track_len": float(np.mean(lens)),
        "median_track_len": float(np.median(lens)),
        "mean_detected_len": float(np.mean(det_lens)),
        "median_detected_len": float(np.median(det_lens)),
        "short_track_ratio": float(sum(1 for length in lens if length < short_len) / len(lens)),
        "long_track_ratio": float(sum(1 for length in lens if length >= long_len) / len(lens)),
        "n_short_detected_tracks": int(sum(1 for length in det_lens if length < short_len)),
        "n_long_detected_tracks": int(sum(1 for length in det_lens if length >= long_len)),
        "short_detected_track_ratio": float(
            sum(1 for length in det_lens if length < short_len) / len(det_lens)
        ),
        "long_detected_track_ratio": float(
            sum(1 for length in det_lens if length >= long_len) / len(det_lens)
        ),
        "candidate_ratio": candidate_micro,
        "candidate_ratio_macro": float(np.mean([p["candidate_ratio"] for p in per_track])),
        "prebirth_ratio": prebirth_micro,
        "prebirth_ratio_macro": float(np.mean([p["prebirth_ratio"] for p in per_track])),
        "n_short_tracks": int(sum(1 for length in lens if length < short_len)),
        "n_long_tracks": int(sum(1 for length in lens if length >= long_len)),
        "mean_match_cost": (
            float(track_match_cost.mean(skipna=True))
            if track_match_cost.notna().any()
            else np.nan
        ),
        "mean_ot_prob": (
            float(track_ot_prob.mean(skipna=True))
            if track_ot_prob.notna().any()
            else np.nan
        ),
        "mean_center_match_distance": (
            float(center_dist.mean(skipna=True)) if center_dist.notna().any() else np.nan
        ),
        "n_rescue_matches": int(
            run_stats.get("n_rescue_matches", sum(p["n_rescue_matches"] for p in per_track))
        ),
        "n_gap_candidates": int(
            run_stats.get("n_gap_candidates", sum(p["n_gap_candidates"] for p in per_track))
        ),
        "n_byte_first_stage_matches": int(
            run_stats.get("n_byte_first_stage_matches", 0)
        ),
        "n_byte_second_stage_matches": int(
            run_stats.get("n_byte_second_stage_matches", 0)
        ),
        "n_high_score_detections": int(
            run_stats.get("n_high_score_detections", 0)
        ),
        "n_low_score_detections": int(
            run_stats.get("n_low_score_detections", 0)
        ),
        "n_byte_removed_short_tracks": int(
            run_stats.get("byte_removed_short_tracks", 0)
        ),
        "suspicious_merge_proxy_count": int(
            sum(1 for p in per_track if p["suspicious_merge_proxy"])
        ),
        "large_jump_track_count": int(
            sum(1 for p in per_track if p["n_large_center_jumps"] > 0)
        ),
    }

# ============================================================
# GMM-RTV ignition on crops
# ============================================================

def map_onset_to_frame(crop_meta: pd.DataFrame, crop_idx: int) -> int:
    if crop_idx < 0 or crop_meta.empty or crop_idx >= len(crop_meta):
        return -1
    return int(crop_meta.iloc[crop_idx]["frame_id"])


def make_ignition_summary_row(
    track_id: int,
    n_frames: int,
    *,
    status: str = "ok",
    error: str = "",
    error_reason: str = "",
    n_success_crops: Optional[int] = None,
    n_failed_crops: int = 0,
    crop_status: Optional[str] = None,
    crop_success_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """构造固定列集合，保证正常、部分、跳过和错误记录可直接合并输出。"""
    n_success = n_frames if n_success_crops is None else int(n_success_crops)
    n_failed = int(n_failed_crops)
    if crop_success_rate is None:
        crop_success_rate = float(n_success / max(n_success + n_failed, 1))
    return {
        "track_id": track_id,
        "track_name": f"track_{track_id:04d}",
        "n_frames": n_frames,
        "n_success_crops": n_success,
        "n_failed_crops": n_failed,
        "crop_status": crop_status or status,
        "crop_success_rate": float(crop_success_rate),
        "status": status,
        "error": error,
        "error_reason": error_reason or error,
        "has_ignition": False,
        "ignition_status": "ambiguous" if status in {"ok", "partial"} else "not_evaluated",
        "ignition_idx_candidate": -1,
        "ignition_idx_onset": -1,
        "ignition_idx_confirmed": -1,
        "ignition_onset_frame": -1,
        "ignition_confirmed_frame": -1,
        "ignition_onset_cx": np.nan,
        "ignition_onset_cy": np.nan,
        "ignition_confidence": 0.0,
        "gmm_n_states": 0,
        "gmm_state_sequence": "[]",
    }

def run_ignition_on_track(
    track_id: int,
    crop_paths: List[str],
    crop_meta: pd.DataFrame,
    ignition_mod: Any,
    gmm_rtv_cfg: Any,
    track_output_dir: Path,
    save_plot: bool = False,
    *,
    run_status: str = "ok",
    crop_error_reason: str = "",
    n_success_crops: Optional[int] = None,
    n_failed_crops: int = 0,
    crop_success_rate: Optional[float] = None,
) -> Dict[str, Any]:
    base = make_ignition_summary_row(
        track_id,
        len(crop_paths),
        status=run_status,
        error_reason=crop_error_reason,
        n_success_crops=n_success_crops,
        n_failed_crops=n_failed_crops,
        crop_status=run_status,
        crop_success_rate=crop_success_rate,
    )
    min_len = max(2, int(getattr(gmm_rtv_cfg, "min_sequence_len", 2)))
    if len(crop_paths) < min_len:
        base["status"] = "skip"
        base["error"] = "insufficient_crop_frames"
        base["error_reason"] = "insufficient_crop_frames"
        base["ignition_status"] = "not_evaluated"
        return base

    try:
        df = ignition_mod.extract_sequence_features(crop_paths, gmm_rtv_cfg)
        result = ignition_mod.detect_ignition_gmm_rtv(df, gmm_rtv_cfg)
        onset_crop = int(result.get("ignition_idx_onset", -1))
        confirmed_crop = int(result.get("ignition_idx_confirmed", -1))
        onset_frame = map_onset_to_frame(crop_meta, onset_crop)
        confirmed_frame = map_onset_to_frame(crop_meta, confirmed_crop)

        onset_cx = np.nan
        onset_cy = np.nan
        if 0 <= onset_crop < len(crop_meta):
            onset_cx = float(crop_meta.iloc[onset_crop].get("cx", np.nan))
            onset_cy = float(crop_meta.iloc[onset_crop].get("cy", np.nan))

        base.update({
            "has_ignition": bool(result.get("has_ignition", False)),
            "ignition_status": str(result.get("ignition_status", "")),
            "ignition_idx_candidate": int(result.get("ignition_idx_candidate", -1)),
            "ignition_idx_onset": onset_crop,
            "ignition_idx_confirmed": confirmed_crop,
            "ignition_onset_frame": onset_frame,
            "ignition_confirmed_frame": confirmed_frame,
            "ignition_onset_cx": onset_cx,
            "ignition_onset_cy": onset_cy,
            "ignition_confidence": float(result.get("ignition_confidence", 0.0)),
            "gmm_n_states": int(result.get("gmm_n_states", 0)),
            "gmm_state_sequence": json.dumps(
                to_jsonable(result.get("gmm_state_sequence", [])),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
        })

        if gmm_rtv_cfg.save_features and len(df):
            track_output_dir.mkdir(parents=True, exist_ok=True)
            feat_df = df.copy()
            if "hot_score" in result and len(result["hot_score"]) == len(feat_df):
                feat_df["hot_score"] = result["hot_score"]
            if "change_score" in result and len(result["change_score"]) == len(feat_df):
                feat_df["change_score"] = result["change_score"]
            if result.get("gmm_state_sequence") and len(result["gmm_state_sequence"]) == len(feat_df):
                feat_df["gmm_state_rank"] = result["gmm_state_sequence"]
            if len(crop_meta) == len(feat_df):
                feat_df["source_frame_id"] = crop_meta["frame_id"].values
            save_table(feat_df, track_output_dir / "frame_features", quiet=True)

            result_json = to_jsonable(result)
            with open(track_output_dir / "ignition_result.json", "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False, allow_nan=False)

        if save_plot:
            track_output_dir.mkdir(parents=True, exist_ok=True)
            try:
                ignition_mod.plot_detection_result_gmm_rtv(
                    df,
                    result,
                    save_path=str(track_output_dir / "ignition_detection_curve.png"),
                    show=False,
                )
            except Exception:
                pass

        return base
    except Exception as exc:
        base["status"] = "error"
        base["error"] = str(exc)
        base["error_reason"] = str(exc)
        base["ignition_status"] = "not_evaluated"
        return base

ALREADY_HOT_STATUSES = {"already_hot", "already_hot_at_sequence_start"}

def is_already_hot_status(value: Any) -> bool:
    return str(value or "") in ALREADY_HOT_STATUSES


def is_ok_ignition_row(row: Any) -> bool:
    """ok 与 partial 均表示有足够 crop、已经实际运行过点火判断。"""
    return str(row.get("status", "")) in {"ok", "partial"}

def prediction_has_ignition(row: Any) -> bool:
    return (
        is_ok_ignition_row(row)
        and bool(row.get("has_ignition", False))
        and not is_already_hot_status(row.get("ignition_status", ""))
    )


def aggregate_ignition_metrics(ignition_df: pd.DataFrame) -> Dict[str, Any]:
    if ignition_df.empty:
        return {
            "n_ignition_ok": 0,
            "n_ignition_partial": 0,
            "n_ignition_skip": 0,
            "n_ignition_error": 0,
            "mean_crop_success_rate": np.nan,
            "n_pred_ignition": 0,
            "n_pred_no_ignition": 0,
            "n_pred_ambiguous": 0,
            "n_pred_shape_rejected": 0,
            "n_pred_initially_hot": 0,
            "n_pred_weak_transition_rejected": 0,
            "n_pred_single_state": 0,
            "n_pred_ambiguous_gmm_unstable": 0,
            "n_valid_ignition_tracks": 0,
        }

    run_status = ignition_df["status"].fillna("").astype(str)
    evaluated_df = ignition_df[run_status.isin({"ok", "partial"})].copy()
    status = evaluated_df["ignition_status"].fillna("").astype(str)
    has_ign = evaluated_df.apply(prediction_has_ignition, axis=1)
    initially_hot = status.isin(ALREADY_HOT_STATUSES)
    crop_rates = pd.to_numeric(
        ignition_df.get("crop_success_rate", pd.Series(dtype=float)), errors="coerce"
    )

    return {
        "n_ignition_ok": int((run_status == "ok").sum()),
        "n_ignition_partial": int((run_status == "partial").sum()),
        "n_ignition_skip": int((run_status == "skip").sum()),
        "n_ignition_error": int((run_status == "error").sum()),
        "mean_crop_success_rate": (
            float(crop_rates.mean(skipna=True)) if crop_rates.notna().any() else np.nan
        ),
        "n_pred_ignition": int(has_ign.sum()),
        "n_pred_no_ignition": int((status == "no_ignition").sum()),
        "n_pred_ambiguous": int(status.str.contains("ambiguous", case=False, na=False).sum()),
        "n_pred_shape_rejected": int((status == "rejected_shape_sequence").sum()),
        "n_pred_initially_hot": int(initially_hot.sum()),
        "n_pred_weak_transition_rejected": int((status == "no_ignition_weak_transition").sum()),
        "n_pred_single_state": int((status == "single_state").sum()),
        "n_pred_ambiguous_gmm_unstable": int((status == "ambiguous_gmm_unstable").sum()),
        "n_valid_ignition_tracks": int((has_ign & (status == "ignition_detected")).sum()),
    }

# ============================================================
# Annotations & event evaluation
# ============================================================

def normalize_track_name(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.lower().startswith("track_"):
        suffix = s.split("_", 1)[1]
        try:
            return f"track_{int(float(suffix))}"
        except (TypeError, ValueError):
            return s
    try:
        return f"track_{int(float(s))}"
    except (TypeError, ValueError):
        return s


def read_annotation_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def is_effective_trace_format(df: pd.DataFrame) -> bool:
    cols = {str(c).lower() for c in df.columns}
    has_track = "track_id" in cols or "track_name" in cols
    has_frame = "frame_id" in cols
    has_category = bool(cols & {"ignition_category", "gt_category", "category"})
    has_xy = "x" in cols and "y" in cols
    has_event = "event_id" in cols
    return has_track and has_frame and not has_category and not has_xy and not has_event


def is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return (not s) or s.lower() in {"nan", "none", "null"}


def is_effective_trace_new_format(df: pd.DataFrame) -> bool:
    """effective_trace_new.xlsx：单列未点火 track id，可带 track_id 表头。"""
    if df.shape[1] != 1:
        return False
    cols = {str(c).strip().lower() for c in df.columns}
    incompatible = {
        "frame_id", "ignition_category", "gt_category", "category",
        "event_id", "x", "y", "onset_frame", "onset_frame_idx",
    }
    if cols & incompatible:
        return False

    def _as_int(v: Any) -> Optional[int]:
        if is_blank_value(v):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    non_blank = [v for v in df.iloc[:, 0].tolist() if not is_blank_value(v)]
    parsed = [_as_int(v) for v in non_blank]
    header_id = _as_int(df.columns[0])
    # 无表头且只有一个数值ID时，pandas会把该ID读成列名。
    if not parsed and header_id is not None:
        return True
    # track_id/track_name 等文本表头由 pandas 作为列名读取，不在 values 中。
    return bool(parsed) and all(v is not None for v in parsed)

def read_listed_track_ids(path: Path) -> List[int]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, header=None)
    else:
        df = pd.read_csv(path, header=None, encoding="utf-8-sig")
    ids: List[int] = []
    for v in df.iloc[:, 0].tolist():
        if is_blank_value(v):
            continue
        try:
            ids.append(int(float(v)))
        except (TypeError, ValueError):
            # 允许首行是 track_id/track_name 等文本表头。
            continue
    return sorted(set(ids))


def _is_no_ignition_candidate(value: Any) -> bool:
    try:
        return int(float(value)) < 0
    except (TypeError, ValueError):
        return True


def _no_ignition_gt() -> Dict[str, Any]:
    return {
        "gt_category": "no_ignition",
        "gt_has_ignition": False,
        "gt_onset_frame": -1,
        "gt_onset_start": -1,
        "gt_onset_end": -1,
    }


def build_gt_from_effective_trace_new_and_ref_ignition(
    listed_no_ignition_ids: List[int],
    ref_ignition_df: pd.DataFrame,
    onset_tolerance: int = 2,
    min_crop_success_rate: float = 0.95,
) -> Dict[str, Dict[str, Any]]:
    """
    与 ignition/feature-group 消融一致的 effective_trace_new 语义：
      - xlsx 列出的 = no_ignition
      - 参考 summary 中 ignition_idx_candidate != -1 且不在 xlsx = ignition
      - candidate == -1 / 未检出 = no_ignition
    onset 使用全帧 ignition_onset_frame（非 crop 索引）。
    """
    if onset_tolerance < 0:
        raise ValueError(f"onset_tolerance must be >= 0, got {onset_tolerance}")
    if not 0.0 <= min_crop_success_rate <= 1.0:
        raise ValueError(
            "min_crop_success_rate must be in [0, 1], "
            f"got {min_crop_success_rate}"
        )
    columns = set(ref_ignition_df.columns)
    if "track_id" not in columns:
        raise ValueError("Reference ignition summary missing required field: track_id")
    has_candidate = "ignition_idx_candidate" in columns
    has_ignition_flag = "has_ignition" in columns
    if not has_candidate and not has_ignition_flag:
        raise ValueError(
            "Reference ignition summary must contain ignition_idx_candidate "
            "or has_ignition"
        )
    has_status = "status" in columns

    def _is_valid_reference_row(row: pd.Series) -> bool:
        if has_status and str(row.get("status", "")).strip() not in {"ok", "partial"}:
            return False
        if "crop_success_rate" in columns:
            try:
                rate = float(row.get("crop_success_rate", 0.0))
            except (TypeError, ValueError):
                return False
            if not math.isfinite(rate) or rate < min_crop_success_rate:
                return False
        return True

    valid_reference_ids = {
        int(row["track_id"])
        for _, row in ref_ignition_df.iterrows()
        if _is_valid_reference_row(row)
    }
    listed_no = {
        int(x) for x in listed_no_ignition_ids
        if int(x) in valid_reference_ids
    }
    ann: Dict[str, Dict[str, Any]] = {}
    for tid in sorted(listed_no):
        ann[normalize_track_name(tid)] = _no_ignition_gt()

    n_ign = 0
    n_no_listed = 0
    n_no_candidate = 0
    for _, row in ref_ignition_df.iterrows():
        if not _is_valid_reference_row(row):
            continue
        tid = int(row["track_id"])
        tname = normalize_track_name(tid)
        if tid in listed_no:
            ann[tname] = _no_ignition_gt()
            n_no_listed += 1
            continue
        if is_already_hot_status(row.get("ignition_status", "")):
            is_ign = False
        elif has_candidate and not is_blank_value(
            row.get("ignition_idx_candidate")
        ):
            is_ign = not _is_no_ignition_candidate(row["ignition_idx_candidate"])
        elif has_ignition_flag and not is_blank_value(row.get("has_ignition")):
            # 兼容仅含 has_ignition 的变体 summary
            value = row.get("has_ignition")
            if isinstance(value, str):
                is_ign = value.strip().lower() in {"1", "true", "yes"}
            else:
                is_ign = bool(value)
        else:
            raise ValueError(
                f"Reference track {tid} has no usable ignition_idx_candidate "
                "or has_ignition value"
            )
        if not is_ign:
            ann[tname] = _no_ignition_gt()
            n_no_candidate += 1
            continue
        onset = int(row.get("ignition_onset_frame", -1))
        if onset < 0 and "ignition_idx_onset" in ref_ignition_df.columns:
            # 不应依赖 crop 索引做跨变体对齐；无全帧 onset 时仍标 ignition 但不计区间
            onset = -1
        if onset >= 0:
            ann[tname] = {
                "gt_category": "ignition",
                "gt_has_ignition": True,
                "gt_onset_frame": onset,
                "gt_onset_start": max(0, onset - onset_tolerance),
                "gt_onset_end": onset + onset_tolerance,
            }
        else:
            ann[tname] = {
                "gt_category": "ignition",
                "gt_has_ignition": True,
                "gt_onset_frame": -1,
                "gt_onset_start": -1,
                "gt_onset_end": -1,
            }
        n_ign += 1

    n_no = sum(1 for v in ann.values() if v.get("gt_category") == "no_ignition")
    print(
        f"[INFO] effective_trace_new GT: listed_no={len(listed_no)} | "
        f"ignition={n_ign} no_ignition={n_no} "
        f"(xlsx_in_ref={n_no_listed}, candidate_-1→no_ignition={n_no_candidate}) | "
        f"onset 使用全帧 frame_id ±{onset_tolerance}"
    )
    return ann



def normalize_manual_track_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """兼容 feature/decision ablation 常见 manual_review_gt 列名。"""
    out = df.copy()
    lower = {str(c).strip().lower(): c for c in out.columns}
    rename: Dict[Any, str] = {}
    for src in ("gt_category", "category"):
        if src in lower and "ignition_category" not in lower:
            rename[lower[src]] = "ignition_category"
            break
    if "gt_onset_frame" in lower and "onset_frame" not in lower:
        rename[lower["gt_onset_frame"]] = "onset_frame"
    for src in ("gt_onset_idx", "ignition_idx_onset", "frame_id"):
        if src in lower and "onset_frame_idx" not in lower:
            rename[lower[src]] = "onset_frame_idx"
            break
    if rename:
        out = out.rename(columns=rename)
    return out


def build_gt_from_manual_track_annotations(
    tracks_ann: pd.DataFrame,
    ref_tracks_df: pd.DataFrame,
    *,
    onset_tolerance: int = 2,
    missing_as_no_ignition: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    将绑定 T3 track_id 的 manual GT 转成跨 tracking 可用的参考 ID + global frame GT。

    - ignition/no_ignition 进入 event P/R/F1；already_hot/invalid/ambiguous 等不进入二分类。
    - 若仅给 crop onset index，则通过 T3 tracks_long 的时间顺序映射到 global frame_id。
    - manual_review_gt 默认未出现的 T3 轨迹视为 no_ignition，与 feature ablation 的
      missing_as_no_ignition 口径保持一致。
    """
    ann = normalize_manual_track_annotation_columns(tracks_ann)
    ref_ids = sorted(int(v) for v in ref_tracks_df["track_id"].dropna().unique())
    ref_id_set = set(ref_ids)
    gt: Dict[str, Dict[str, Any]] = {}
    if missing_as_no_ignition:
        for tid in ref_ids:
            gt[normalize_track_name(tid)] = _no_ignition_gt()

    skipped_annotation_ids: List[int] = []
    for _, row in ann.iterrows():
        raw_tid = row.get("track_id", row.get("track_name"))
        tname = normalize_track_name(raw_tid)
        if not tname:
            continue
        tid = int(tname.split("_", 1)[1])
        # GT population is the actually evaluable T3 reference tracks.
        # A positive label whose T3 track/ROI is absent cannot be scored and must not
        # be injected as an artificial FN.
        if tid not in ref_id_set:
            skipped_annotation_ids.append(tid)
            continue
        cat = str(row.get("ignition_category", "")).strip().lower()
        if cat in {"1", "true", "yes"}:
            cat = "ignition"
        elif cat in {"0", "false", "no"}:
            cat = "no_ignition"
        if cat == "no_ignition":
            gt[tname] = _no_ignition_gt()
            continue
        if cat != "ignition":
            # already_hot / invalid / suspicious_merge / defocused / ambiguous 均不计二分类。
            gt.pop(tname, None)
            continue

        onset_frame = -1
        if "onset_frame" in ann.columns and pd.notna(row.get("onset_frame")):
            try:
                onset_frame = int(float(row.get("onset_frame")))
            except (TypeError, ValueError):
                onset_frame = -1
        if onset_frame < 0 and "onset_frame_idx" in ann.columns and pd.notna(row.get("onset_frame_idx")):
            try:
                onset_idx = int(float(row.get("onset_frame_idx")))
            except (TypeError, ValueError):
                onset_idx = -1
            if onset_idx >= 0:
                rg = ref_tracks_df[ref_tracks_df["track_id"] == tid].sort_values("frame_id").reset_index(drop=True)
                if onset_idx < len(rg):
                    onset_frame = int(rg.iloc[onset_idx]["frame_id"])
        if onset_frame >= 0:
            gt[tname] = {
                "gt_category": "ignition",
                "gt_has_ignition": True,
                "gt_onset_frame": onset_frame,
                "gt_onset_start": onset_frame - int(onset_tolerance),
                "gt_onset_end": onset_frame + int(onset_tolerance),
            }
        else:
            gt[tname] = {
                "gt_category": "ignition",
                "gt_has_ignition": True,
                "gt_onset_frame": -1,
                "gt_onset_start": -1,
                "gt_onset_end": -1,
            }
    n_ign = sum(v.get("gt_category") == "ignition" for v in gt.values())
    n_no = sum(v.get("gt_category") == "no_ignition" for v in gt.values())
    print(f"[INFO] manual aligned GT: ignition={n_ign} no_ignition={n_no} total={len(gt)}")
    if skipped_annotation_ids:
        uniq = sorted(set(skipped_annotation_ids))
        print(
            f"[WARN] {len(uniq)} annotated track IDs are absent from the evaluable T3 reference population "
            f"and are excluded from metrics; first IDs={uniq[:20]}"
        )
    return gt


def _row_center(row: pd.Series) -> Tuple[float, float]:
    if "cx" in row and pd.notna(row["cx"]) and "cy" in row and pd.notna(row["cy"]):
        return float(row["cx"]), float(row["cy"])
    return (
        float(row["x1"] + row["x2"]) * 0.5,
        float(row["y1"] + row["y2"]) * 0.5,
    )


def filter_detected_track_points(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """仅保留来自检测器的真实观测点，排除 gap/prebirth 候选点。"""
    if tracks_df.empty or "point_type" not in tracks_df.columns:
        return tracks_df.copy()
    point_type = tracks_df["point_type"].fillna("detected").astype(str)
    return tracks_df[point_type == "detected"].copy()


def build_track_centroids_by_frame(
    tracks_df: pd.DataFrame,
    *,
    detected_only: bool = True,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """track_id -> {global_frame_id -> (cx, cy)}。默认仅用真实检测点。"""
    source_df = filter_detected_track_points(tracks_df) if detected_only else tracks_df
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for tid, group in source_df.groupby("track_id"):
        cents: Dict[int, Tuple[float, float]] = {}
        for _, row in group.iterrows():
            try:
                fid = int(row["frame_id"])
                cx, cy = _row_center(row)
            except Exception:
                continue
            if not math.isfinite(cx) or not math.isfinite(cy):
                continue
            cents[fid] = (cx, cy)
        if cents:
            out[int(tid)] = cents
    return out


def match_tracks_to_reference(
    query_tracks_df: pd.DataFrame,
    ref_tracks_df: pd.DataFrame,
    *,
    max_center_dist: float = 30.0,
    min_shared_frames: int = 3,
    min_ref_coverage: float = 0.20,
    min_query_coverage: float = 0.20,
    min_temporal_iou: float = 0.15,
    max_assignment_cost: float = 1.50,
    detected_only: bool = True,
    return_diagnostics: bool = False,
) -> Any:
    """
    将 query 轨迹一对一匹配到参考轨迹。

    默认仅使用 detected 点，避免 gap fill / prebirth 与 T3 参考轨迹共享
    候选点而造成对齐偏置。使用 Hungarian assignment，并同时约束双向覆盖率、
    时间 IoU 与最终代价。
    """
    for name, value in (
        ("min_ref_coverage", min_ref_coverage),
        ("min_query_coverage", min_query_coverage),
        ("min_temporal_iou", min_temporal_iou),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    if max_center_dist <= 0 or min_shared_frames <= 0 or max_assignment_cost <= 0:
        raise ValueError("track alignment thresholds must be positive")

    q_map = build_track_centroids_by_frame(
        query_tracks_df, detected_only=detected_only
    )
    r_map = build_track_centroids_by_frame(
        ref_tracks_df, detected_only=detected_only
    )
    q_ids = sorted(q_map)
    r_ids = sorted(r_map)
    empty_diag = pd.DataFrame(columns=[
        "query_track_id", "ref_track_id", "n_shared_frames",
        "mean_center_distance", "median_center_distance",
        "p90_center_distance", "query_coverage", "reference_coverage",
        "temporal_iou", "assignment_cost",
    ])
    if not q_ids or not r_ids:
        return ({}, empty_diag) if return_diagnostics else {}

    large = 1e9
    cost = np.full((len(q_ids), len(r_ids)), large, dtype=np.float64)
    info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for qi, qid in enumerate(q_ids):
        qcent = q_map[qid]
        qframes = set(qcent)
        for ri, rid in enumerate(r_ids):
            rcent = r_map[rid]
            rframes = set(rcent)
            shared = qframes & rframes
            n_shared = len(shared)
            if n_shared < min_shared_frames:
                continue
            dists = np.asarray([
                math.hypot(
                    qcent[f][0] - rcent[f][0],
                    qcent[f][1] - rcent[f][1],
                )
                for f in shared
            ], dtype=float)
            mean_d = float(np.mean(dists))
            median_d = float(np.median(dists))
            p90_d = float(np.percentile(dists, 90))
            if mean_d > max_center_dist or p90_d > 1.5 * max_center_dist:
                continue
            query_cov = float(n_shared / max(len(qframes), 1))
            ref_cov = float(n_shared / max(len(rframes), 1))
            temporal_iou = float(n_shared / max(len(qframes | rframes), 1))
            if (
                ref_cov < min_ref_coverage
                or query_cov < min_query_coverage
                or temporal_iou < min_temporal_iou
            ):
                continue
            assignment_cost = (
                mean_d / max(max_center_dist, 1e-6)
                + 0.50 * (1.0 - ref_cov)
                + 0.25 * (1.0 - query_cov)
                + 0.25 * (1.0 - temporal_iou)
            )
            if assignment_cost > max_assignment_cost:
                continue
            cost[qi, ri] = assignment_cost
            info[(qi, ri)] = {
                "query_track_id": int(qid),
                "ref_track_id": int(rid),
                "n_shared_frames": int(n_shared),
                "mean_center_distance": mean_d,
                "median_center_distance": median_d,
                "p90_center_distance": p90_d,
                "query_coverage": query_cov,
                "reference_coverage": ref_cov,
                "temporal_iou": temporal_iou,
                "assignment_cost": float(assignment_cost),
            }

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: Dict[int, int] = {}
    diag_rows: List[Dict[str, Any]] = []
    for qi, ri in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[qi, ri] >= large / 2:
            continue
        record = info[(qi, ri)]
        mapping[int(record["query_track_id"])] = int(record["ref_track_id"])
        diag_rows.append(record)
    diag = pd.DataFrame(diag_rows, columns=empty_diag.columns)
    return (mapping, diag) if return_diagnostics else mapping

def project_ignition_to_ref_ids(
    ignition_df: pd.DataFrame,
    query_to_ref: Dict[int, int],
) -> pd.DataFrame:
    """
    把变体 ignition 预测投影到参考 track_id 空间。
    同一 ref_id 被多条 query 匹配时：优先保留 has_ignition=True，其次置信度更高。
    """
    if ignition_df.empty:
        return pd.DataFrame(columns=[
            "track_id", "status", "has_ignition", "ignition_onset_frame",
            "ignition_status", "ignition_confidence",
        ])
    rows: List[Dict[str, Any]] = []
    for _, row in ignition_df.iterrows():
        if not is_ok_ignition_row(row):
            continue
        qid = int(row["track_id"])
        if qid not in query_to_ref:
            continue
        rid = int(query_to_ref[qid])
        rows.append({
            "track_id": rid,
            "query_track_id": qid,
            "status": "ok",
            "has_ignition": prediction_has_ignition(row),
            "ignition_onset_frame": int(row.get("ignition_onset_frame", -1)),
            "ignition_status": str(row.get("ignition_status", "")),
            "ignition_confidence": float(row.get("ignition_confidence", 0.0) or 0.0),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "track_id", "status", "has_ignition", "ignition_onset_frame",
            "ignition_status", "ignition_confidence",
        ])
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["track_id", "has_ignition", "ignition_confidence"],
        ascending=[True, False, False],
    )
    return df.groupby("track_id", as_index=False).first()


def evaluate_effective_trace_new_aligned(
    annotations: Dict[str, Dict[str, Any]],
    projected_ignition_df: pd.DataFrame,
    *,
    query_ignition_df: Optional[pd.DataFrame] = None,
    query_to_ref: Optional[Dict[int, int]] = None,
    n_query_tracks: int,
    n_matched: int,
    n_reference_tracks: Optional[int] = None,
    count_unmatched_pred_as_fp: bool = True,
) -> Dict[str, Any]:
    """
    在参考 track_id 空间上计算端到端指标。

    主指标覆盖显式 GT：未匹配的 ignition GT 计 FN。
    对 effective_trace_new 可将未匹配预测点火计为 FP；对 effective_trace.xlsx，
    未映射到显式标注 track_id 的预测属于“未标注域”，默认不计 FP。
    同时报告 matched-only 指标，便于区分 tracking coverage 与下游判断误差。
    """
    query_to_ref = query_to_ref or {}
    if not projected_ignition_df.empty:
        projected_ignition_df = projected_ignition_df[
            projected_ignition_df.apply(is_ok_ignition_row, axis=1)
        ].copy()
    pred_by_id = {
        int(row["track_id"]): row
        for _, row in projected_ignition_df.iterrows()
    } if not projected_ignition_df.empty else {}

    gt_by_id: Dict[int, Dict[str, Any]] = {}
    for tname, gt in annotations.items():
        cat = str(gt.get("gt_category", "")).lower()
        if cat not in {"ignition", "no_ignition"}:
            continue
        try:
            tid = int(tname.split("_", 1)[1]) if tname.startswith("track_") else int(float(tname))
        except (TypeError, ValueError):
            continue
        gt_by_id[tid] = gt

    matched_ref_ids = set(int(v) for v in query_to_ref.values())
    unmatched_pred_ignitions = 0
    if query_ignition_df is not None and not query_ignition_df.empty:
        for _, row in query_ignition_df.iterrows():
            try:
                qid = int(row["track_id"])
            except (TypeError, ValueError, KeyError):
                continue
            if qid not in query_to_ref and prediction_has_ignition(row):
                unmatched_pred_ignitions += 1

    def evaluate_ids(ids: List[int], add_unmatched_pred_fp: bool) -> Dict[str, Any]:
        tp = fp = fn = 0
        onset_errors: List[float] = []
        inside_flags: List[bool] = []
        tp_onset_by_id: Dict[int, Dict[str, Any]] = {}
        for tid in ids:
            gt = gt_by_id[tid]
            cat = str(gt.get("gt_category", "")).lower()
            pred = pred_by_id.get(tid)
            pred_has = prediction_has_ignition(pred) if pred is not None else False
            pred_onset = int(pred.get("ignition_onset_frame", -1)) if pred is not None else -1
            if cat == "ignition":
                if pred_has:
                    tp += 1
                    gt_onset = int(gt.get("gt_onset_frame", -1))
                    gt_start = int(gt.get("gt_onset_start", -1))
                    gt_end = int(gt.get("gt_onset_end", -1))
                    if gt_onset >= 0 and pred_onset >= 0:
                        err = abs(pred_onset - gt_onset)
                        onset_errors.append(err)
                        entry: Dict[str, Any] = {"onset_abs_error": float(err)}
                        if gt_start >= 0 and gt_end >= gt_start:
                            inside = bool(gt_start <= pred_onset <= gt_end)
                            inside_flags.append(inside)
                            entry["onset_inside_interval"] = inside
                        tp_onset_by_id[int(tid)] = entry
                else:
                    fn += 1
            elif cat == "no_ignition" and pred_has:
                fp += 1
        if add_unmatched_pred_fp:
            fp += unmatched_pred_ignitions
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "onset_errors": onset_errors,
            "inside_flags": inside_flags,
            "tp_onset_by_id": tp_onset_by_id,
        }

    all_ids = sorted(gt_by_id)
    matched_ids = sorted(set(all_ids) & matched_ref_ids)
    end_to_end = evaluate_ids(all_ids, add_unmatched_pred_fp=bool(count_unmatched_pred_as_fp))
    matched_only = evaluate_ids(matched_ids, add_unmatched_pred_fp=False)
    onset_errors = end_to_end["onset_errors"]
    inside_flags = end_to_end["inside_flags"]
    unmatched_ref_ign = sum(
        1
        for tid, gt in gt_by_id.items()
        if str(gt.get("gt_category", "")).lower() == "ignition"
        and tid not in matched_ref_ids
    )
    reference_den = n_reference_tracks if n_reference_tracks is not None else len(gt_by_id)

    return {
        "n_matched_to_ref": int(n_matched),
        "match_rate": float(n_matched / max(n_query_tracks, 1)),
        "reference_track_coverage": float(len(matched_ref_ids) / max(reference_den, 1)),
        "gt_track_coverage": float(len(matched_ids) / max(len(all_ids), 1)),
        "unmatched_reference_ignition_count": int(unmatched_ref_ign),
        "unmatched_predicted_ignition_count": int(unmatched_pred_ignitions),
        "n_valid_gt": int(len(all_ids)),
        "n_matched_valid_gt": int(len(matched_ids)),
        "end_to_end_tp": end_to_end["tp"],
        "end_to_end_fp": end_to_end["fp"],
        "end_to_end_fn": end_to_end["fn"],
        "ignition_precision": end_to_end["precision"],
        "ignition_recall": end_to_end["recall"],
        "ignition_f1": end_to_end["f1"],
        "matched_only_precision": matched_only["precision"],
        "matched_only_recall": matched_only["recall"],
        "matched_only_f1": matched_only["f1"],
        "n_onset_eval": int(len(onset_errors)),
        "onset_mae": float(np.mean(onset_errors)) if onset_errors else None,
        "onset_median_error": float(np.median(onset_errors)) if onset_errors else None,
        "within_1_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 1) / len(onset_errors))
            if onset_errors else None
        ),
        "within_3_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 3) / len(onset_errors))
            if onset_errors else None
        ),
        "inside_interval_acc": (
            float(sum(inside_flags) / len(inside_flags)) if inside_flags else None
        ),
        # 供跨变体 common onset 使用；写盘前会弹出。
        "tp_onset_by_ref_id": end_to_end["tp_onset_by_id"],
    }


def compute_onset_metrics_on_ids(
    tp_onset_by_id: Dict[int, Dict[str, Any]],
    selected_ids: Set[int],
    *,
    prefix: str = "_common",
) -> Dict[str, Any]:
    """在给定 ref track_id 集合上计算 onset 指标（默认 common 前缀）。"""
    errors: List[float] = []
    insides: List[bool] = []
    for tid in sorted(selected_ids):
        entry = tp_onset_by_id.get(int(tid))
        if not entry:
            continue
        err = entry.get("onset_abs_error")
        if err is None or not np.isfinite(float(err)):
            continue
        errors.append(float(err))
        if "onset_inside_interval" in entry and entry["onset_inside_interval"] is not None:
            insides.append(bool(entry["onset_inside_interval"]))
    n_key = f"n_onset_eval{prefix}"
    return {
        n_key: int(len(selected_ids)),
        f"onset_mae{prefix}": float(np.mean(errors)) if errors else None,
        f"onset_median_error{prefix}": float(np.median(errors)) if errors else None,
        f"within_1_frame_acc{prefix}": (
            float(sum(1 for e in errors if e <= 1) / len(errors)) if errors else None
        ),
        f"within_3_frame_acc{prefix}": (
            float(sum(1 for e in errors if e <= 3) / len(errors)) if errors else None
        ),
        f"inside_interval_acc{prefix}": (
            float(sum(insides) / len(insides)) if insides else None
        ),
    }

def load_event_annotations(path: Path) -> Tuple[str, Any]:
    path = Path(path)
    df_head = read_annotation_table(path)
    if is_effective_trace_new_format(df_head):
        listed = read_listed_track_ids(path)
        print(f"[INFO] 检测到 effective_trace_new：未点火列表 {len(listed)} 条")
        return "effective_trace_new", {"listed_no_ignition_ids": listed, "path": path}

    df = normalize_manual_track_annotation_columns(df_head)
    cols = {str(c).lower(): c for c in df.columns}
    lower_cols = set(cols.keys())

    # effective_trace.xlsx: track_id + frame_id (crop onset idx)
    if is_effective_trace_format(df):
        print(
            "[INFO] 检测到 effective_trace.xlsx：仅 track_id 在表中且 frame_id 非空的轨迹为 ignition；"
            "表内 frame_id 为空以及未出现在表中的 T3 reference tracks 均为 no_ignition。"
        )
        track_col = cols.get("track_id", cols.get("track_name"))
        frame_col = cols["frame_id"]
        rows = []
        for _, row in df.iterrows():
            tname = normalize_track_name(row[track_col])
            if not tname:
                continue
            tid = int(tname.split("_", 1)[1])
            if pd.notna(row[frame_col]):
                onset = int(float(row[frame_col]))
                cat = "ignition"
            else:
                onset = -1
                cat = "no_ignition"
            rows.append({
                "track_name": tname,
                "track_id": tid,
                "ignition_category": cat,
                "onset_frame_idx": onset,
                "onset_frame": np.nan,
            })
        out = pd.DataFrame(rows)
        print(
            f"[INFO] effective_trace track-level 标注: valid={len(out)} "
            f"ignition={(out['ignition_category']=='ignition').sum()} "
            f"no_ignition={(out['ignition_category']=='no_ignition').sum()}"
        )
        return "effective_trace", out

    if "event_id" in lower_cols or ("onset_frame" in lower_cols and "x" in lower_cols and "y" in lower_cols):
        ann_type = "event"
    elif "track_name" in lower_cols or "track_id" in lower_cols:
        ann_type = "track"
        has_onset_frame = "onset_frame" in lower_cols
        has_onset_frame_idx = "onset_frame_idx" in lower_cols or "frame_id" in lower_cols
        if has_onset_frame_idx and not has_onset_frame:
            warnings.warn(
                "track-level 标注仅含 onset_frame_idx/frame_id：该字段表示 crop 序列索引，"
                "不是原始全帧 frame_id，不建议用于跨 tracking 变体对比。"
                "请优先使用 onset_frame（原始全帧 frame_id）或 effective_trace_new。",
                stacklevel=2,
            )
        print(
            "[INFO] 检测到 track-level/manual GT：将以 T3 reference track_id 为标签空间，"
            "其余 tracking 变体通过 detected-only 时空匹配后统一评价。"
        )
        if "frame_id" in lower_cols and "onset_frame_idx" not in lower_cols:
            df = df.rename(columns={cols["frame_id"]: "onset_frame_idx"})
    else:
        raise ValueError(
            f"无法识别标注格式: {path}\n"
            "支持:\n"
            "  - effective_trace_new.xlsx（单列未点火 id，需配合参考变体/summary）\n"
            "  - effective_trace.xlsx (track_id, frame_id；空=未点火)\n"
            "  - event-level (event_id, onset_frame, x, y)\n"
            "  - track-level (track_name/track_id, ignition_category, onset_frame)"
        )
    return ann_type, df


def _event_onset_window(row: pd.Series, default_window: int) -> Tuple[int, int, int]:
    onset = int(float(row.get("onset_frame", row.get("onset_frame_idx", -1))))
    if pd.notna(row.get("onset_start")) and pd.notna(row.get("onset_end")):
        start = int(float(row["onset_start"]))
        end = int(float(row["onset_end"]))
        center = int(round((start + end) / 2))
        return center, start, end
    return onset, onset - default_window, onset + default_window


def evaluate_event_annotations(
    events_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    ignition_df: pd.DataFrame,
    frame_window: int,
    distance_threshold: float,
) -> Dict[str, Any]:
    """使用一对一 Hungarian assignment 评价 event-level tracking + ignition。"""
    gt_events: List[Dict[str, Any]] = []
    capture_flags: List[bool] = []
    for _, ev in events_df.iterrows():
        cat = str(ev.get("category", "ignition")).lower()
        if cat not in {"ignition", ""}:
            continue
        center, win_start, win_end = _event_onset_window(ev, frame_window)
        if "onset_start" not in ev or pd.isna(ev.get("onset_start")):
            win_start = center - frame_window
            win_end = center + frame_window
        ex, ey = float(ev["x"]), float(ev["y"])
        gt_events.append({
            "center": int(center),
            "start": int(win_start),
            "end": int(win_end),
            "x": ex,
            "y": ey,
        })
        captured = any(
            win_start <= int(row["frame_id"]) <= win_end
            and math.hypot(
                float(row.get("cx", (row["x1"] + row["x2"]) / 2)) - ex,
                float(row.get("cy", (row["y1"] + row["y2"]) / 2)) - ey,
            ) <= distance_threshold
            for _, row in tracks_df.iterrows()
        )
        capture_flags.append(bool(captured))

    pred_df = ignition_df[ignition_df.apply(prediction_has_ignition, axis=1)].copy()
    preds: List[Dict[str, Any]] = []
    for idx, row in pred_df.iterrows():
        onset = int(row.get("ignition_onset_frame", -1))
        cx = float(row.get("ignition_onset_cx", np.nan))
        cy = float(row.get("ignition_onset_cy", np.nan))
        if onset < 0 or not np.isfinite(cx) or not np.isfinite(cy):
            continue
        preds.append({"index": idx, "onset": onset, "x": cx, "y": cy})

    n_gt = len(gt_events)
    n_pred = int(len(pred_df))
    matched_pairs: List[Tuple[int, int]] = []
    onset_errors: List[float] = []
    if n_gt and n_pred:
        large = 1e9
        cost = np.full((n_gt, n_pred), large, dtype=np.float64)
        for gi, event in enumerate(gt_events):
            half_window = max(event["end"] - event["start"], 1)
            for pi, pred in enumerate(preds):
                if not (event["start"] <= pred["onset"] <= event["end"]):
                    continue
                distance = math.hypot(pred["x"] - event["x"], pred["y"] - event["y"])
                if distance > distance_threshold:
                    continue
                time_cost = abs(pred["onset"] - event["center"]) / half_window
                dist_cost = distance / max(distance_threshold, 1e-6)
                cost[gi, pi] = time_cost + dist_cost
        rows, cols = linear_sum_assignment(cost)
        for gi, pi in zip(rows.tolist(), cols.tolist()):
            if cost[gi, pi] >= large / 2:
                continue
            matched_pairs.append((gi, pi))
            onset_errors.append(abs(preds[pi]["onset"] - gt_events[gi]["center"]))

    tp = len(matched_pairs)
    fn = n_gt - tp
    fp = n_pred - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "n_gt_events": int(n_gt),
        "n_matched_events": int(tp),
        "event_capture_rate": float(sum(capture_flags) / max(len(capture_flags), 1)) if capture_flags else 0.0,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "false_event_count": int(fp),
        "n_false_like_events": int(fp),
        "onset_mae": float(np.mean(onset_errors)) if onset_errors else None,
        "onset_median_error": float(np.median(onset_errors)) if onset_errors else None,
        "within_1_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 1) / len(onset_errors))
            if onset_errors else None
        ),
        "within_3_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 3) / len(onset_errors))
            if onset_errors else None
        ),
    }

def evaluate_track_annotations(
    tracks_ann: pd.DataFrame,
    ignition_df: pd.DataFrame,
    variant: str,
) -> Dict[str, Any]:
    warnings.warn(
        f"[{variant}] track-level 标注仅适用于当前变体 track_id，跨变体对比结果请谨慎解读。"
        " onset_frame=全帧 frame_id；onset_frame_idx/frame_id=crop 序列索引。",
        stacklevel=2,
    )
    tp = fp = fn = 0
    onset_errors: List[float] = []
    cols = set(tracks_ann.columns)
    use_crop_idx = "onset_frame_idx" in cols and (
        "onset_frame" not in cols or tracks_ann["onset_frame"].isna().all()
    )
    n_valid = 0

    for _, ann in tracks_ann.iterrows():
        tid = ann.get("track_id", ann.get("track_name"))
        if isinstance(tid, str) and str(tid).startswith("track_"):
            tid = int(str(tid).replace("track_", ""))
        else:
            tid = int(tid)
        cat = str(ann.get("ignition_category", "")).lower()
        gt_ign = cat in {"ignition", "1", "true", "yes"}
        gt_no = cat in {"no_ignition", "0", "false", "no"}
        if not (gt_ign or gt_no):
            continue
        n_valid += 1

        if use_crop_idx and pd.notna(ann.get("onset_frame_idx")):
            gt_onset = int(float(ann["onset_frame_idx"]))
        elif (not use_crop_idx) and "onset_frame" in cols and pd.notna(ann.get("onset_frame")):
            gt_onset = int(float(ann["onset_frame"]))
        elif "onset_frame_idx" in cols and pd.notna(ann.get("onset_frame_idx")):
            gt_onset = int(float(ann["onset_frame_idx"]))
            use_crop_idx = True
        else:
            gt_onset = -1

        pred_rows = ignition_df[ignition_df["track_id"] == tid]
        pred_rows = pred_rows[pred_rows.apply(is_ok_ignition_row, axis=1)]
        pred = pred_rows.iloc[0] if not pred_rows.empty else None
        pred_ign = prediction_has_ignition(pred) if pred is not None else False

        if gt_ign and pred_ign:
            tp += 1
            pred_onset = int(
                pred.get("ignition_idx_onset", -1)
                if use_crop_idx
                else pred.get("ignition_onset_frame", -1)
            )
            if gt_onset >= 0 and pred_onset >= 0:
                onset_errors.append(abs(pred_onset - gt_onset))
        elif gt_ign and not pred_ign:
            fn += 1
        elif gt_no and pred_ign:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    result = {
        "n_valid_gt": int(n_valid),
        "ignition_precision": float(precision),
        "ignition_recall": float(recall),
        "ignition_f1": float(f1),
        "track_precision": float(precision),
        "track_recall": float(recall),
        "track_f1": float(f1),
        "onset_mae": float(np.mean(onset_errors)) if onset_errors else None,
        "onset_median_error": float(np.median(onset_errors)) if onset_errors else None,
        "within_1_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 1) / len(onset_errors))
            if onset_errors else None
        ),
        "within_3_frame_acc": (
            float(sum(1 for e in onset_errors if e <= 3) / len(onset_errors))
            if onset_errors else None
        ),
    }
    return result

# ============================================================
# Review plots
# ============================================================

def save_review_plots_for_variant(
    variant_dir: Path,
    ignition_df: pd.DataFrame,
    ignition_mod: Any,
    max_success: int = 3,
    max_failure: int = 3,
) -> None:
    review_dir = variant_dir / "review_plots"
    review_dir.mkdir(parents=True, exist_ok=True)

    success = ignition_df[
        ignition_df["has_ignition"].astype(bool)
        & (ignition_df["ignition_status"] == "ignition_detected")
    ].head(max_success)

    failure = ignition_df[
        ~ignition_df["has_ignition"].astype(bool)
        | ignition_df["ignition_status"].isin(["no_ignition_weak_transition", "ambiguous"])
    ].head(max_failure)

    for label, subset in [("success", success), ("failure", failure)]:
        for _, row in subset.iterrows():
            tid = int(row["track_id"])
            curve = variant_dir / f"track_{tid:04d}" / "ignition_detection_curve.png"
            if curve.is_file():
                dst = review_dir / f"{label}_track_{tid:04d}.png"
                try:
                    import shutil
                    shutil.copy2(curve, dst)
                except Exception:
                    pass


# ============================================================
# Multiprocessing for GMM-RTV
# ============================================================

_GMM_RTV_WORKER_CTX: Dict[str, Any] = {}


def _load_task_crop_meta(task: Dict[str, Any]) -> pd.DataFrame:
    """读取任务的 crop_meta。

    优先从磁盘路径读取（新方式，避免把大体积 DataFrame 塞进进程池管道，
    这是 Windows 上 ProcessPoolExecutor teardown 死锁的根本触发条件）；
    若任务仍带内联 JSON（兼容旧调用），再回退到内联解析。
    """
    crop_meta_path = task.get("crop_meta_path")
    if crop_meta_path:
        return pd.read_json(crop_meta_path, orient="split")
    return pd.read_json(StringIO(task["crop_meta_json"]), orient="split")


def _shutdown_pool_with_timeout(
    pool: ProcessPoolExecutor,
    *,
    label: str = "",
    join_timeout: float = 120.0,
) -> None:
    """安全关闭进程池，规避 Windows 上 ProcessPoolExecutor teardown 偶发死锁。

    标准的 ``with pool:`` 退出会调用 ``shutdown(wait=True)``，其内部依赖
    manager/feeder 线程对 worker 进程的 join；在 Windows(spawn) + multiprocessing
    管道背压下，该 join 可能永远阻塞（表现为主进程 CPU 归零、worker 已退出、
    但 shutdown 不返回）。

    这里改为非阻塞地请求关闭，然后由我们自己带超时地 join，必要时强制
    terminate 残留 worker。调用此函数时所有任务结果均已收集完毕，强制回收
    不影响数据完整性。
    """
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        pool.shutdown(wait=False)

    procs = list((getattr(pool, "_processes", None) or {}).values())
    deadline = time.time() + join_timeout
    for proc in procs:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.join(remaining)
        except Exception:
            pass
    stragglers = [p for p in procs if p.is_alive()]
    if stragglers:
        print(
            f"  {label}进程池 {len(stragglers)} 个 worker 在 {join_timeout:.0f}s 内"
            f"未自行退出，强制回收以解除潜在死锁 ...",
            flush=True,
        )
        for proc in stragglers:
            try:
                proc.terminate()
            except Exception:
                pass
        for proc in stragglers:
            try:
                proc.join(10.0)
            except Exception:
                pass


def _init_gmm_rtv_worker(ctx: Dict[str, Any]) -> None:
    global _GMM_RTV_WORKER_CTX
    _GMM_RTV_WORKER_CTX = dict(ctx)
    ignition_mod = load_module_from_path(Path(ctx["ignition_module_path"]), "ablation_ignition_worker")
    _GMM_RTV_WORKER_CTX["ignition_mod"] = ignition_mod
    _GMM_RTV_WORKER_CTX["gmm_rtv_cfg"] = ignition_mod.IgnitionGMMConfig(**ctx["gmm_rtv_cfg_dict"])


def _gmm_rtv_worker_task(task: Dict[str, Any]) -> Dict[str, Any]:
    crop_meta = _load_task_crop_meta(task)
    return run_ignition_on_track(
        int(task["track_id"]),
        task["crop_paths"],
        crop_meta,
        _GMM_RTV_WORKER_CTX["ignition_mod"],
        _GMM_RTV_WORKER_CTX["gmm_rtv_cfg"],
        Path(task["track_output_dir"]),
        save_plot=bool(task.get("save_plot", False)),
        run_status=str(task.get("run_status", "ok")),
        crop_error_reason=str(task.get("crop_error_reason", "")),
        n_success_crops=int(task.get("n_success_crops", len(task["crop_paths"]))),
        n_failed_crops=int(task.get("n_failed_crops", 0)),
        crop_success_rate=float(task.get("crop_success_rate", 1.0)),
    )

def select_longest_contiguous_crop_segment(
    crop_paths: List[str],
    crop_meta: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame, int]:
    """
    对 partial crop 仅保留原轨迹行索引连续的最长片段，避免裁剪失败后压缩时间轴。
    """
    if not crop_paths or crop_meta.empty:
        return [], crop_meta.iloc[0:0].copy(), len(crop_paths)
    if "track_row_idx" not in crop_meta.columns:
        # 无法判断缺失位置时，宁可跳过而不是把不连续帧压缩成连续序列。
        return [], crop_meta.iloc[0:0].copy(), len(crop_paths)
    meta = crop_meta.copy().reset_index(drop=True)
    meta["__path"] = crop_paths
    meta["track_row_idx"] = pd.to_numeric(
        meta["track_row_idx"], errors="coerce"
    )
    meta = meta.dropna(subset=["track_row_idx"]).sort_values("track_row_idx")
    if meta.empty:
        return [], meta.drop(columns=["__path"], errors="ignore"), len(crop_paths)
    groups = (meta["track_row_idx"].diff().fillna(1) != 1).cumsum()
    candidates = [g for _, g in meta.groupby(groups, sort=False)]
    best = max(candidates, key=lambda g: (len(g), -int(g["track_row_idx"].iloc[0])))
    selected_paths = best["__path"].astype(str).tolist()
    selected_meta = best.drop(columns=["__path"]).reset_index(drop=True)
    selected_meta["crop_idx"] = np.arange(len(selected_meta), dtype=int)
    return selected_paths, selected_meta, len(crop_paths) - len(selected_paths)


def run_gmm_rtv_batch(
    crop_data: Dict[int, Dict[str, Any]],
    variant_dir: Path,
    ignition_mod: Any,
    gmm_rtv_cfg: Any,
    workers: int,
    save_plot: bool,
    ignition_module_path: Path,
    variant: str = "",
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    min_len = max(2, int(getattr(gmm_rtv_cfg, "min_sequence_len", 2)))
    label = f"[{variant}] " if variant else ""
    n_crop_tracks = len(crop_data)
    prep_heartbeat = max(500, n_crop_tracks // 10)
    t_prep = time.time()
    print(
        f"  {label}准备 GMM 任务: 序列化 {n_crop_tracks} 条轨迹裁剪 ...",
        flush=True,
    )
    for prep_idx, (track_id, crop_result) in enumerate(sorted(crop_data.items()), start=1):
        if prep_idx % prep_heartbeat == 0 or prep_idx == n_crop_tracks:
            print(
                f"  {label}准备 GMM 任务 {prep_idx}/{n_crop_tracks} "
                f"| tasks={len(tasks)} | {time.time() - t_prep:.1f}s",
                flush=True,
            )
        crop_paths = list(crop_result.get("crop_paths", []))
        crop_meta = crop_result.get("crop_meta", pd.DataFrame())
        crop_status = str(crop_result.get("status", "error"))
        error_reason = str(crop_result.get("error_reason", ""))
        n_success = int(crop_result.get("n_success_crops", len(crop_paths)))
        n_failed = int(crop_result.get("n_failed_crops", 0))
        success_rate = float(crop_result.get(
            "crop_success_rate", n_success / max(n_success + n_failed, 1)
        ))
        if crop_status == "partial":
            crop_paths, crop_meta, n_trimmed = select_longest_contiguous_crop_segment(
                crop_paths, crop_meta
            )
            detail = f"partial_crop_longest_segment_trimmed={n_trimmed}"
            error_reason = "|".join(
                value for value in (error_reason, detail) if value
            )
        if crop_status not in {"ok", "partial"} or len(crop_paths) < min_len:
            status = "skip" if len(crop_paths) < min_len else "error"
            reason = "insufficient_crop_frames" if len(crop_paths) < min_len else error_reason
            rows.append(make_ignition_summary_row(
                int(track_id),
                len(crop_paths),
                status=status,
                error=reason,
                error_reason=reason,
                n_success_crops=n_success,
                n_failed_crops=n_failed,
                crop_status=crop_status,
                crop_success_rate=success_rate,
            ))
            continue
        track_output_dir = variant_dir / f"track_{track_id:04d}"
        track_output_dir.mkdir(parents=True, exist_ok=True)
        crop_meta_path = track_output_dir / "crop_meta.json"
        crop_meta.to_json(crop_meta_path, orient="split")
        tasks.append({
            "track_id": track_id,
            "crop_paths": crop_paths,
            "crop_meta_path": str(crop_meta_path),
            "track_output_dir": str(track_output_dir),
            "save_plot": save_plot,
            "run_status": crop_status,
            "crop_error_reason": error_reason,
            "n_success_crops": n_success,
            "n_failed_crops": n_failed,
            "crop_success_rate": success_rate,
        })

    if workers <= 1 or len(tasks) <= 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"  [{variant}] GMM-RTV {idx}/{len(tasks)} track_{task['track_id']:04d}")
            try:
                crop_meta = _load_task_crop_meta(task)
                rows.append(run_ignition_on_track(
                    int(task["track_id"]),
                    task["crop_paths"],
                    crop_meta,
                    ignition_mod,
                    gmm_rtv_cfg,
                    Path(task["track_output_dir"]),
                    save_plot=bool(task.get("save_plot", False)),
                    run_status=str(task.get("run_status", "ok")),
                    crop_error_reason=str(task.get("crop_error_reason", "")),
                    n_success_crops=int(task.get("n_success_crops", len(task["crop_paths"]))),
                    n_failed_crops=int(task.get("n_failed_crops", 0)),
                    crop_success_rate=float(task.get("crop_success_rate", 1.0)),
                ))
            except Exception as exc:
                rows.append(make_ignition_summary_row(
                    int(task["track_id"]),
                    len(task["crop_paths"]),
                    status="error",
                    error=str(exc),
                    error_reason=str(exc),
                    n_success_crops=int(task.get("n_success_crops", len(task["crop_paths"]))),
                    n_failed_crops=int(task.get("n_failed_crops", 0)),
                    crop_status=str(task.get("run_status", "ok")),
                    crop_success_rate=float(task.get("crop_success_rate", 1.0)),
                ))
        return sorted(rows, key=lambda row: int(row["track_id"]))

    ctx = {
        "ignition_module_path": str(ignition_module_path.resolve()),
        "gmm_rtv_cfg_dict": asdict(gmm_rtv_cfg),
    }

    def _handle_result(future: Any, task: Dict[str, Any]) -> None:
        try:
            rows.append(future.result())
        except Exception as exc:
            rows.append(make_ignition_summary_row(
                int(task["track_id"]),
                len(task["crop_paths"]),
                status="error",
                error=str(exc),
                error_reason=str(exc),
                n_success_crops=int(task.get("n_success_crops", len(task["crop_paths"]))),
                n_failed_crops=int(task.get("n_failed_crops", 0)),
                crop_status=str(task.get("run_status", "ok")),
                crop_success_rate=float(task.get("crop_success_rate", 1.0)),
            ))

    # 有界窗口分批 submit：避免一次性把数千个大任务塞进进程池，
    # 降低内存峰值与启动压力，同时保持稳定的进度输出。
    n_tasks = len(tasks)
    window = max(workers * 4, workers + 2)
    t_gmm = time.time()
    print(
        f"  {label}启动进程池 workers={workers} | tasks={n_tasks} | window={window}",
        flush=True,
    )
    done = 0
    task_iter = iter(tasks)
    inflight: Dict[Any, Dict[str, Any]] = {}
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_gmm_rtv_worker,
        initargs=(ctx,),
    )
    try:
        for _ in range(min(window, n_tasks)):
            task = next(task_iter)
            inflight[pool.submit(_gmm_rtv_worker_task, task)] = task
        while inflight:
            completed, _ = wait(set(inflight), return_when=FIRST_COMPLETED)
            for future in completed:
                task = inflight.pop(future)
                done += 1
                _handle_result(future, task)
                print(
                    f"  {label}GMM-RTV {done}/{n_tasks} track_{task['track_id']:04d} "
                    f"| {time.time() - t_gmm:.1f}s"
                )
                nxt = next(task_iter, None)
                if nxt is not None:
                    inflight[pool.submit(_gmm_rtv_worker_task, nxt)] = nxt
    finally:
        _shutdown_pool_with_timeout(pool, label=label)
    return sorted(rows, key=lambda row: int(row["track_id"]))

# ============================================================
# Run one variant
# ============================================================

def load_completed_variant_metrics(
    variant: str,
    variant_dir: Path,
) -> Optional[Dict[str, Any]]:
    """若变体已成功完成则返回其磁盘上的指标，否则返回 None（供 --resume 使用）。

    判定条件：variant_metrics.json 存在且 status=ok，且 tracks_long.csv 与
    ignition_summary.csv 均存在（后续 effective_trace_new 对齐依赖这两个产物）。
    """
    metrics_path = variant_dir / "variant_metrics.json"
    tracks_path = variant_dir / "tracks_long.csv"
    ignition_path = variant_dir / "ignition_summary.csv"
    if not (metrics_path.is_file() and tracks_path.is_file() and ignition_path.is_file()):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metrics, dict) or metrics.get("status") != "ok":
        return None
    metrics["variant"] = variant
    for k in SUMMARY_COLUMNS:
        metrics.setdefault(k, None)
    return metrics


def run_one_variant(
    variant: str,
    dets_by_frame: Dict[int, List[Dict[str, Any]]],
    tracker_mod: Any,
    ignition_mod: Any,
    gmm_rtv_cfg: Any,
    args: argparse.Namespace,
    output_root: Path,
    ann_type: Optional[str],
    ann_df: Optional[pd.DataFrame],
    ignition_module_path: Path,
    roi_cache: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    if variant not in VARIANT_ORDER:
        raise ValueError(
            f"Unknown tracking variant: {variant}; allowed: {', '.join(VARIANT_ORDER)}"
        )
    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}\n[{variant}] 开始运行 -> {variant_dir}\n{'=' * 70}")

    run_stats: Dict[str, Any] = {"n_rescue_matches": 0, "n_gap_candidates": 0}

    try:
        use_precomputed_reference = (
            str(getattr(args, "_image_source_mode", "full_frame")) == "roi_cache"
            and variant == str(args.reference_variant)
        )
        if use_precomputed_reference:
            ref_tracks_path = Path(args.roi_tracks_csv)
            if not ref_tracks_path.is_file():
                raise FileNotFoundError(f"Precomputed reference tracks not found: {ref_tracks_path}")
            shutil.copy2(ref_tracks_path, variant_dir / "tracks_long.csv")
            ref_summary_path = Path(args.roi_track_summary_csv) if args.roi_track_summary_csv else None
            if ref_summary_path is not None and ref_summary_path.is_file():
                shutil.copy2(ref_summary_path, variant_dir / "track_summary.csv")
            else:
                ref_df_tmp = pd.read_csv(ref_tracks_path, encoding="utf-8-sig")
                pd.DataFrame({"track_id": sorted(ref_df_tmp["track_id"].dropna().astype(int).unique())}).to_csv(
                    variant_dir / "track_summary.csv", index=False, encoding="utf-8-sig"
                )
            run_stats["precomputed_reference"] = True
            print(f"  [{variant}] ROI-cache 模式：直接复用预计算 T3 tracks_long.csv，不重新执行 prebirth。")
        elif variant == "T0_center_nn":
            tracks, run_stats = run_center_nn_tracker(
                dets_by_frame,
                max_center_dist=args.max_center_dist,
                max_age=args.max_age,
                max_age_tentative=args.max_age_tentative,
                max_age_confirmed=args.max_age_confirmed,
                min_hits=args.min_hits,
            )
            # 与 OT 变体统一确认、连续观测和最少观测点筛选口径。
            tracks = [track for track in tracks if bool(track.is_confirmed)]
            tracks, filter_stats = tracker_mod.postprocess_tracks(
                tracks,
                kf_params={},
                enable_gap_fill=False,
                min_contiguous_observed=args.min_hits,
                min_observed_hits=args.min_hits,
            )
            run_stats["removed_short_tracks"] = int(
                filter_stats.get("removed_short", 0)
            )
            tracker_mod.save_tracks_csv(tracks, variant_dir / "tracks_long.csv", min_track_len=2)
            tracker_mod.save_track_summary_csv(tracks, variant_dir / "track_summary.csv", min_track_len=2)
            augment_tracks_csv_from_history(
                tracks,
                variant_dir / "tracks_long.csv",
                ("center_match_distance", "center_match_distance_norm"),
            )
        elif variant == "B0_byte_track":
            tracks, run_stats = run_byte_track_variant(
                tracker_mod,
                dets_by_frame,
                high_thresh=args.byte_high_thresh,
                low_thresh=args.byte_low_thresh,
                new_track_thresh=args.byte_new_track_thresh,
                match_iou_thresh=args.byte_match_iou,
                second_match_iou_thresh=args.byte_second_match_iou,
                track_buffer=args.byte_track_buffer,
                min_hits=args.min_hits,
            )
            tracker_mod.save_tracks_csv(
                tracks, variant_dir / "tracks_long.csv", min_track_len=2
            )
            tracker_mod.save_track_summary_csv(
                tracks, variant_dir / "track_summary.csv", min_track_len=2
            )
        else:
            tcfg = variant_tracker_config(variant, args)
            image_dir = (
                Path(args.image_dir)
                if str(getattr(args, "_image_source_mode", "full_frame")) == "full_frame" and args.image_dir
                else None
            )
            tracks, run_stats = run_ot_kalman_variant(
                tracker_mod,
                dets_by_frame,
                tcfg,
                image_dir=image_dir,
                image_exts=args.image_exts,
                progress_label=variant,
            )
            tracker_mod.save_tracks_csv(tracks, variant_dir / "tracks_long.csv", min_track_len=2)
            tracker_mod.save_track_summary_csv(tracks, variant_dir / "track_summary.csv", min_track_len=2)

        resave_csv_utf8_sig(variant_dir / "tracks_long.csv")
        resave_csv_utf8_sig(variant_dir / "track_summary.csv")

        tracks_df = pd.read_csv(variant_dir / "tracks_long.csv", encoding="utf-8-sig")
        track_summary_df = pd.read_csv(variant_dir / "track_summary.csv", encoding="utf-8-sig")

        if str(getattr(args, "_image_source_mode", "full_frame")) == "roi_cache":
            if roi_cache is None:
                raise RuntimeError("roi_cache mode selected but ROI cache was not initialized")
            crop_data = build_crop_data_from_roi_cache(
                tracks_df,
                roi_cache,
                bbox_tolerance=float(args.roi_bbox_tolerance),
                min_success_rate=float(args.roi_min_success_rate),
                progress_label=variant,
            )
            reuse_manifest = pd.DataFrame([
                {
                    "track_id": int(tid),
                    "status": data.get("status"),
                    "n_success_crops": data.get("n_success_crops"),
                    "n_failed_crops": data.get("n_failed_crops"),
                    "crop_success_rate": data.get("crop_success_rate"),
                    "error_reason": data.get("error_reason", ""),
                }
                for tid, data in crop_data.items()
            ])
            save_table(reuse_manifest, variant_dir / "roi_cache_reuse_manifest", quiet=True)
        else:
            frame_image_map, image_name_map = build_frame_image_map(
                Path(args.image_dir), args.image_exts, tracks_df
            )
            if args.save_crops:
                crop_root = variant_dir / "track_crop"
            else:
                crop_root = variant_dir / "_tmp_crops"
                warnings.warn(
                    f"[{variant}] --no-save-crops: 裁剪暂存至 {crop_root} 以供 GMM-RTV 使用（不会写入 track_crop/）。",
                    stacklevel=2,
                )
            crop_data = crop_all_tracks(
                tracks_df,
                crop_root,
                frame_image_map,
                image_name_map,
                args.crop_padding,
                crop_format=str(getattr(args, "crop_format", "png")),
                jpeg_quality=int(getattr(args, "jpeg_quality", 90)),
                skip_existing=bool(getattr(args, "skip_existing_crops", False)),
                progress_label=variant,
            )

        n_workers = resolve_workers(args.workers)
        print(f"  [{variant}] 运行 GMM-RTV ({len(crop_data)} tracks, workers={n_workers})")
        ignition_rows = run_gmm_rtv_batch(
            crop_data,
            variant_dir,
            ignition_mod,
            gmm_rtv_cfg,
            workers=n_workers,
            save_plot=args.save_review_plots,
            ignition_module_path=ignition_module_path,
            variant=variant,
        )

        # 裁剪数据后续不再使用，尽早释放以降低内存峰值。
        del crop_data
        gc.collect()

        ignition_df = (
            pd.DataFrame(ignition_rows)
            if ignition_rows
            else pd.DataFrame(
                columns=list(make_ignition_summary_row(0, 0).keys())
            )
        )
        save_table(ignition_df, variant_dir / "ignition_summary", quiet=False)

        tracking_metrics = aggregate_tracking_metrics(tracks_df, track_summary_df, run_stats)
        ignition_metrics = aggregate_ignition_metrics(ignition_df)

        eval_metrics: Dict[str, Any] = {}
        if ann_df is not None and ann_type == "event":
            eval_metrics = evaluate_event_annotations(
                ann_df,
                tracks_df,
                ignition_df,
                args.event_match_frame_window,
                args.event_match_distance,
            )
        elif ann_df is not None and ann_type == "track":
            eval_metrics = evaluate_track_annotations(ann_df, ignition_df, variant)

        metrics = {
            "variant": variant,
            "status": "ok",
            "error": "",
            **tracking_metrics,
            **ignition_metrics,
            **eval_metrics,
        }
        for k in SUMMARY_COLUMNS:
            metrics.setdefault(k, None)

        metrics_path = variant_dir / "variant_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(metrics),
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        metrics_df = pd.DataFrame([metrics])
        save_table(metrics_df, variant_dir / "variant_metrics", quiet=False)
        save_table(metrics_df, variant_dir / "final_variant_summary", quiet=False)

        if args.save_review_plots:
            save_review_plots_for_variant(variant_dir, ignition_df, ignition_mod)

        print(
            f"[{variant}] 完成: tracks={metrics['n_tracks']} "
            f"ignition={metrics.get('n_pred_ignition', 0)} "
            f"event_recall={metrics.get('event_recall', 'N/A')}"
        )
        return metrics

    except Exception as exc:
        print(f"[{variant}] 失败: {exc}")
        traceback.print_exc()
        err_metrics = {"variant": variant, "status": "error", "error": str(exc)}
        for k in SUMMARY_COLUMNS:
            err_metrics.setdefault(k, None)
        err_path = variant_dir / "variant_metrics.json"
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(
                to_jsonable(err_metrics),
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
        err_df = pd.DataFrame([err_metrics])
        save_table(err_df, variant_dir / "variant_metrics", quiet=True)
        save_table(err_df, variant_dir / "final_variant_summary", quiet=True)
        return err_metrics


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tracking 模块辅助消融实验：比较不同 tracking 策略对 GMM-RTV 点火检测的影响。"
    )
    p.add_argument(
        "--detections-txt-dir",
        default='inputs/cleaned_detections_txt',
        help="每帧检测结果 txt 文件夹（排序后 frame_id=0,1,2,...）",
    )
    p.add_argument(
        "--max-txt",
        type=int,
        default=7996,
        help="只处理排序后的前 N 个 .txt；-1 表示全部",
    )
    p.add_argument(
        "--image-source", choices=["auto", "full_frame", "roi_cache"], default="auto",
        help="auto: 原图可用则全帧重裁剪，否则使用既有 T3 ROI cache；full_frame/roi_cache 可强制指定。",
    )
    p.add_argument(
        "--image-dir", default='inputs/MCFF0004-N001',
        help="原始全帧图像目录；roi_cache 模式可不存在。",
    )
    p.add_argument(
        "--roi-root",
        default='inputs/track_crop',
        help="既有完整 T3 tracker 的 track_crop 根目录，用于原图不可用时构建 ROI cache。",
    )
    p.add_argument(
        "--roi-tracks-csv", default=None,
        help="与 --roi-root 对应的 T3 tracks_long.csv；默认取 roi-root 的父目录/tracks_long.csv。",
    )
    p.add_argument(
        "--roi-track-summary-csv", default=None,
        help="可选 T3 track_summary.csv；默认取 roi-root 的父目录/track_summary.csv。",
    )
    p.add_argument("--roi-bbox-tolerance", type=float, default=0.75, help="ROI cache 复用时 bbox 坐标最大允许误差(px)")
    p.add_argument("--roi-min-success-rate", type=float, default=0.95, help="单轨 ROI cache 最低复用率；低于该值则不进入 GMM-RTV")
    p.add_argument("--output-dir", default=None, help="实验输出目录；默认 tracking_ablation_results/run_YYYYMMDD_HHMMSS")
    p.add_argument("--tracking-module-path", default="ot_kalman_tracker.py")
    p.add_argument("--ignition-module-path", default="ignition_detection_gmm_rtv_new.py")
    p.add_argument("--gmm-max-states", type=int, default=None)
    p.add_argument("--gmm-bic-improvement-ratio", type=float, default=None)
    p.add_argument("--gmm-min-state-count", type=int, default=None)
    p.add_argument("--gmm-min-state-fraction", type=float, default=None)
    p.add_argument("--min-state-run", type=int, default=None)
    p.add_argument("--min-cold-run", type=int, default=None)
    p.add_argument("--min-hot-run", type=int, default=None)
    p.add_argument("--min-stable-run", type=int, default=None)
    p.add_argument("--min-sequence-len", type=int, default=None)
    p.add_argument("--refine-radius", type=int, default=None)
    p.add_argument(
        "--no-diff-features",
        dest="use_diff_features",
        action="store_false",
        default=None,
    )
    p.add_argument(
        "--prior-cold-guard",
        dest="require_all_prior_cold",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--refine-onset",
        dest="refine_onset_with_change",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    p.add_argument(
        "--no-shape-sequence-reject",
        dest="enable_shape_sequence_reject",
        action="store_false",
        default=None,
    )
    p.add_argument(
        "--no-initial-hot-reject",
        dest="enable_initial_hot_reject",
        action="store_false",
        default=None,
    )
    p.add_argument(
        "--no-raw-transition-validation",
        dest="enable_raw_transition_validation",
        action="store_false",
        default=None,
    )
    p.add_argument(
        "--variants",
        default=(
            "T0_center_nn,T1_basic_ot_kalman,T2_ot_kalman_rescue,"
            "T2b_ot_kalman_rescue_gap,T3_full_tracker"
        ),
        # default=(
        #     "T3_full_tracker"
        # ),
        help="逗号分隔的变体列表",
    )
    p.add_argument("--crop-padding", type=float, default=4.0)
    p.add_argument("--max-center-dist", type=float, default=45.0)
    p.add_argument("--max-age", type=int, default=5, help="legacy max_age；T0优先使用 tentative/confirmed 分级阈值")
    p.add_argument("--max-age-confirmed", type=int, default=5)
    p.add_argument("--max-age-tentative", type=int, default=1)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--sinkhorn-reg", type=float, default=0.08)
    p.add_argument("--byte-high-thresh", type=float, default=0.6)
    p.add_argument("--byte-low-thresh", type=float, default=0.1)
    p.add_argument("--byte-new-track-thresh", type=float, default=0.7)
    p.add_argument("--byte-match-iou", type=float, default=0.3)
    p.add_argument("--byte-second-match-iou", type=float, default=0.2)
    p.add_argument("--byte-track-buffer", type=int, default=5)
    p.add_argument("--image-exts", default=".png,.jpg,.jpeg,.tif,.tiff")
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument(
        "--event-annotation-csv",
        default=r"effective_trace.xlsx",
        help=(
            "默认 effective_trace.xlsx。仅表内且 frame_id 非空的轨迹为有效 ignition；"
            "表内 frame_id 为空以及未出现在表中的其余 T3 reference tracks 均作为 no_ignition。"
            "其余变体先对齐到 T3 后在同一 reference-ID GT 空间上评价；"
            "仍兼容 manual track GT 与 effective_trace_new，"
            "但后者属于 reference-conditioned evaluation，不应用于和 4.2.2/4.2.3 直接比较 Precision/F1。"
        ),
    )
    p.add_argument(
        "--reference-variant",
        type=str,
        default="T3_full_tracker",
        help="effective_trace_new 的参考变体（完整 tracker）；其余变体的 track 会匹配到该变体的 track_id",
    )
    p.add_argument(
        "--reference-summary",
        type=str,
        default=None,
        help=(
            "可选：外部参考 ignition summary.csv（含 has_ignition / ignition_onset_frame 或 crop onset）。"
            "status 列可省略；若存在则 status=ok/partial 且裁剪成功率达阈值的行有效。"
            "默认使用 --reference-variant 目录下的 ignition_summary.csv"
        ),
    )
    p.add_argument("--gt-onset-tolerance", type=int, default=2,
                   help="effective_trace_new：全帧 onset 容差 [onset-tol, onset+tol]")
    p.add_argument("--track-match-max-dist", type=float, default=30.0,
                   help="变体轨→参考轨匹配：共享帧上平均中心距离上限（像素）")
    p.add_argument("--track-match-min-shared-frames", type=int, default=3,
                   help="变体轨→参考轨匹配：最少共享全帧数")
    p.add_argument(
        "--track-match-min-ref-coverage",
        type=float,
        default=0.20,
        help="变体轨→参考轨匹配：共享帧占参考轨迹帧数的最小比例",
    )
    p.add_argument(
        "--track-match-min-query-coverage",
        type=float,
        default=0.20,
        help="变体轨→参考轨匹配：共享帧占查询轨迹检测帧数的最小比例",
    )
    p.add_argument(
        "--track-match-min-temporal-iou",
        type=float,
        default=0.15,
        help="变体轨→参考轨匹配：检测帧集合的最小时间 IoU",
    )
    p.add_argument(
        "--track-match-max-cost",
        type=float,
        default=1.50,
        help="变体轨→参考轨匹配：Hungarian 候选的最大综合代价",
    )
    p.add_argument(
        "--reference-min-crop-success-rate",
        type=float,
        default=0.95,
        help="effective_trace_new：参考轨迹进入GT所需的最小裁剪成功率",
    )
    p.add_argument(
        "--manual-missing-as-no-ignition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅 generic manual track GT 使用：未出现在标注表中的 T3 reference tracks 是否按 no_ignition 计入。effective_trace.xlsx 固定为 True。",
    )
    p.add_argument("--event-match-frame-window", type=int, default=3)
    p.add_argument("--event-match-distance", type=float, default=20.0)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="GMM-RTV 并行进程数；0 表示自动 min(CPU-1, 16)，1 表示串行",
    )
    p.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True, help="保存 track ROI 裁剪")
    p.add_argument(
        "--crop-format",
        choices=["jpg", "jpeg", "png"],
        default="png",
        help="ROI 裁剪格式；小ROI特征实验默认使用无损 png",
    )
    p.add_argument("--jpeg-quality", type=int, default=90, help="crop-format=jpg 时的质量")
    p.add_argument(
        "--skip-existing-crops",
        action="store_true",
        help="若裁剪文件已存在则跳过重写（断点续跑时有用）",
    )
    p.add_argument("--save-review-plots", action="store_true", help="保存典型点火/失败 review 曲线")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "断点续跑：跳过输出目录中已成功完成的变体"
            "（variant_metrics.json status=ok 且 tracks_long.csv/ignition_summary.csv 均存在）；"
            "被跳过变体的指标从磁盘读取以参与后续对齐与汇总。"
        ),
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    """兼容测试与外部调用的 CLI parser 入口。"""
    return build_arg_parser()


def main() -> None:
    args = build_arg_parser().parse_args()
    variants = validate_variants(args.variants)
    if "B0_byte_track" in variants:
        validate_byte_track_parameters(
            args.byte_low_thresh,
            args.byte_high_thresh,
            args.byte_new_track_thresh,
            args.byte_match_iou,
            args.byte_second_match_iou,
            args.byte_track_buffer,
            args.min_hits,
        )

    output_root = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    output_root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    tracker_path = Path(args.tracking_module_path)
    if not tracker_path.is_file():
        tracker_path = script_dir / args.tracking_module_path
    ignition_path = Path(args.ignition_module_path)
    if not ignition_path.is_file():
        ignition_path = script_dir / args.ignition_module_path

    tracker_mod = load_module_from_path(tracker_path, "ablation_tracker_runtime")
    ignition_mod = load_module_from_path(ignition_path, "ablation_ignition_runtime")
    gmm_rtv_cfg = build_gmm_rtv_config(ignition_mod, args=args)

    args._image_source_mode = resolve_image_source_mode(args)
    roi_cache: Optional[Dict[int, List[Dict[str, Any]]]] = None
    if args._image_source_mode == "roi_cache":
        roi_root = Path(args.roi_root)
        roi_tracks_csv = resolve_roi_tracks_csv(roi_root, args.roi_tracks_csv)
        if args.roi_track_summary_csv:
            roi_track_summary_csv = Path(args.roi_track_summary_csv)
        else:
            roi_track_summary_csv = roi_root.parent / "track_summary.csv"
        args.roi_tracks_csv = str(roi_tracks_csv)
        args.roi_track_summary_csv = str(roi_track_summary_csv) if roi_track_summary_csv.is_file() else None
        ref_tracks_for_cache = pd.read_csv(roi_tracks_csv, encoding="utf-8-sig")
        roi_cache, roi_manifest = build_precomputed_roi_cache(roi_root, ref_tracks_for_cache)
        save_table(roi_manifest, output_root / "roi_cache_source_manifest", quiet=False)
        print(
            f"[INFO] image_source=roi_cache | roi_root={roi_root} | "
            f"cache_frames={len(roi_cache)} | reference_tracks={ref_tracks_for_cache['track_id'].nunique()}"
        )
    else:
        print(f"[INFO] image_source=full_frame | image_dir={args.image_dir}")

    detections_txt_dir = Path(args.detections_txt_dir)
    if not detections_txt_dir.is_dir():
        raise FileNotFoundError(f"Detections txt directory not found: {detections_txt_dir}")

    print(f"[INFO] 加载检测 txt 目录: {detections_txt_dir} (max_txt={args.max_txt})")
    dets_by_frame = load_detections_txt_dir_compat(
        tracker_mod, detections_txt_dir, max_txt=int(args.max_txt)
    )
    print(f"[INFO] 帧数={len(dets_by_frame)} 检测点={sum(len(v) for v in dets_by_frame.values())}")
    print(f"[INFO] GMM-RTV workers={resolve_workers(args.workers)} (multiprocess)")

    ann_type: Optional[str] = None
    ann_payload: Any = None
    if args.event_annotation_csv:
        ann_path = Path(args.event_annotation_csv)
        if not ann_path.is_file():
            ann_path = script_dir / args.event_annotation_csv
        ann_type, ann_payload = load_event_annotations(ann_path)
        if ann_type == "effective_trace_new":
            print(
                f"[INFO] 标注 effective_trace_new: {ann_path} | "
                f"listed_no_ignition={len(ann_payload['listed_no_ignition_ids'])} | "
                f"reference_variant={args.reference_variant}"
            )
            warnings.warn(
                "effective_trace_new 的正类由 reference T3 prediction 补全，因此该模式的 P/R/F1 是 reference-conditioned，"
                "不应与人工 effective_trace.xlsx 的 P/R/F1 直接比较。",
                stacklevel=2,
            )
        elif ann_type == "effective_trace":
            n_pos = int((ann_payload["ignition_category"] == "ignition").sum())
            n_neg = int((ann_payload["ignition_category"] == "no_ignition").sum())
            print(
                f"[INFO] 标注 effective_trace: {ann_path} | explicit_gt={len(ann_payload)} "
                f"ignition={n_pos} no_ignition={n_neg} | reference_variant={args.reference_variant}"
            )
        else:
            print(f"[INFO] 加载标注 ({ann_type}-level): {ann_path} rows={len(ann_payload)}")

    if ann_type in {"effective_trace_new", "effective_trace", "track"} and args.reference_variant not in variants:
        raise ValueError(
            f"aligned track evaluation 需要参考变体 {args.reference_variant} 在 --variants 中，"
            f"当前={variants}"
        )

    # 先跑参考变体，便于后续对齐
    if ann_type in {"effective_trace_new", "effective_trace", "track"} and args.reference_variant in variants:
        variants = [args.reference_variant] + [v for v in variants if v != args.reference_variant]

    all_metrics: List[Dict[str, Any]] = []
    variant_artifacts: Dict[str, Dict[str, Any]] = {}

    for variant in variants:
        vdir = output_root / variant
        # 对 reference-ID 绑定的 track 标注：单变体内不按原 track_id 直接评价，统一在 T3 ID 空间对齐后计算。
        aligned_eval = ann_type in {"effective_trace_new", "effective_trace", "track"}
        pass_ann_type = None if aligned_eval else ann_type
        pass_ann_df = None if aligned_eval else ann_payload

        resumed_metrics = (
            load_completed_variant_metrics(variant, vdir) if args.resume else None
        )
        if resumed_metrics is not None:
            print(
                f"\n[{variant}] --resume：检测到已完成结果，跳过重跑 "
                f"(tracks={resumed_metrics.get('n_tracks')}, "
                f"ignition={resumed_metrics.get('n_pred_ignition')})",
                flush=True,
            )
            metrics = resumed_metrics
        else:
            metrics = run_one_variant(
                variant,
                dets_by_frame,
                tracker_mod,
                ignition_mod,
                gmm_rtv_cfg,
                args,
                output_root,
                pass_ann_type,
                pass_ann_df,
                ignition_path,
                roi_cache=roi_cache,
            )
            # 释放本变体裁剪/中间对象，降低多变体连跑时的内存峰值。
            gc.collect()
        all_metrics.append(metrics)
        variant_artifacts[variant] = {
            "dir": vdir,
            "tracks_path": vdir / "tracks_long.csv",
            "ignition_path": vdir / "ignition_summary.csv",
            "metrics": metrics,
        }

    # effective_trace_new：对齐到参考变体 track_id + 全帧 onset，再算 Pr/Re
    if ann_type == "effective_trace_new":
        tp_onset_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}
        ref_name = args.reference_variant
        ref_art = variant_artifacts.get(ref_name)
        if ref_art is None:
            raise RuntimeError(f"参考变体结果缺失: {ref_name}")
        if not ref_art["tracks_path"].is_file() or not ref_art["ignition_path"].is_file():
            raise FileNotFoundError(
                f"参考变体缺少 tracks_long/ignition_summary: {ref_art['dir']}"
            )
        ref_tracks = pd.read_csv(ref_art["tracks_path"], encoding="utf-8-sig")
        local_ign = pd.read_csv(ref_art["ignition_path"], encoding="utf-8-sig")
        if args.reference_summary:
            ref_sum_path = Path(args.reference_summary)
            if not ref_sum_path.is_file():
                ref_sum_path = script_dir / args.reference_summary
            ref_ign = pd.read_csv(ref_sum_path, encoding="utf-8-sig")
            # 标签可用外部 summary（candidate）；全帧 onset 一律用参考变体本地映射
            onset_cols = ["track_id", "ignition_onset_frame"]
            if "ignition_onset_frame" in local_ign.columns:
                if "ignition_onset_frame" in ref_ign.columns:
                    ref_ign = ref_ign.drop(columns=["ignition_onset_frame"])
                ref_ign = ref_ign.merge(
                    local_ign[onset_cols],
                    on="track_id",
                    how="left",
                )
            print(
                f"[INFO] GT 参考 summary={ref_sum_path} | "
                f"全帧 onset 来自 {ref_name}/ignition_summary.csv"
            )
        else:
            ref_ign = local_ign
            print(f"[INFO] GT 参考 summary={ref_art['ignition_path']}（参考变体本地）")

        gt_ann = build_gt_from_effective_trace_new_and_ref_ignition(
            ann_payload["listed_no_ignition_ids"],
            ref_ign,
            onset_tolerance=int(args.gt_onset_tolerance),
            min_crop_success_rate=float(args.reference_min_crop_success_rate),
        )

        for i, variant in enumerate(variants):
            art = variant_artifacts[variant]
            if not art["tracks_path"].is_file() or not art["ignition_path"].is_file():
                continue
            q_tracks = pd.read_csv(art["tracks_path"], encoding="utf-8-sig")
            q_ign = pd.read_csv(art["ignition_path"], encoding="utf-8-sig")
            if variant == ref_name:
                detected_tracks = filter_detected_track_points(q_tracks)
                query_to_ref = {
                    int(t): int(t) for t in detected_tracks["track_id"].unique()
                }
                alignment_rows = []
                for tid, group in detected_tracks.groupby("track_id"):
                    alignment_rows.append({
                        "query_track_id": int(tid),
                        "ref_track_id": int(tid),
                        "n_shared_frames": int(group["frame_id"].nunique()),
                        "mean_center_distance": 0.0,
                        "median_center_distance": 0.0,
                        "p90_center_distance": 0.0,
                        "query_coverage": 1.0,
                        "reference_coverage": 1.0,
                        "temporal_iou": 1.0,
                        "assignment_cost": 0.0,
                    })
                alignment_df = pd.DataFrame(alignment_rows)
            else:
                query_to_ref, alignment_df = match_tracks_to_reference(
                    q_tracks,
                    ref_tracks,
                    max_center_dist=float(args.track_match_max_dist),
                    min_shared_frames=int(args.track_match_min_shared_frames),
                    min_ref_coverage=float(args.track_match_min_ref_coverage),
                    min_query_coverage=float(args.track_match_min_query_coverage),
                    min_temporal_iou=float(args.track_match_min_temporal_iou),
                    max_assignment_cost=float(args.track_match_max_cost),
                    detected_only=True,
                    return_diagnostics=True,
                )
            projected = project_ignition_to_ref_ids(q_ign, query_to_ref)
            eval_m = evaluate_effective_trace_new_aligned(
                gt_ann,
                projected,
                query_ignition_df=q_ign,
                query_to_ref=query_to_ref,
                n_query_tracks=int(q_tracks["track_id"].nunique()),
                n_matched=len(query_to_ref),
                n_reference_tracks=int(
                    filter_detected_track_points(ref_tracks)["track_id"].nunique()
                ),
            )
            tp_onset_maps[variant] = dict(eval_m.pop("tp_onset_by_ref_id", {}) or {})
            # 写回 metrics
            all_metrics[i].update(eval_m)
            save_table(
                alignment_df,
                art["dir"] / "track_id_alignment_to_ref",
                quiet=True,
            )
            save_table(projected, art["dir"] / "ignition_projected_to_ref", quiet=True)
            # 更新 variant_metrics（common onset 在全部变体对齐后统一回填）
            metrics_df = pd.DataFrame([all_metrics[i]])
            save_table(metrics_df, art["dir"] / "variant_metrics", quiet=True)
            save_table(metrics_df, art["dir"] / "final_variant_summary", quiet=True)
            with open(art["dir"] / "variant_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    to_jsonable(all_metrics[i]),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            print(
                f"[{variant}] aligned_eval: matched={eval_m['n_matched_to_ref']} "
                f"P={eval_m['ignition_precision']:.3f} "
                f"R={eval_m['ignition_recall']:.3f} "
                f"F1={eval_m['ignition_f1']:.3f} "
                f"inside={eval_m.get('inside_interval_acc')}"
            )

        # 各变体 TP（参考 track_id）交集上的 common onset，便于公平对比定帧精度。
        common_tp_ids: Optional[set] = None
        for onset_map in tp_onset_maps.values():
            ids = set(int(k) for k in onset_map.keys())
            common_tp_ids = ids if common_tp_ids is None else (common_tp_ids & ids)
        common_tp_ids = common_tp_ids or set()
        print(
            f"[INFO] common onset 支持集: n_onset_eval_common={len(common_tp_ids)} "
            f"(各变体 TP 轨迹交集)"
        )
        for i, variant in enumerate(variants):
            art = variant_artifacts[variant]
            common_metrics = compute_onset_metrics_on_ids(
                tp_onset_maps.get(variant, {}),
                common_tp_ids,
                prefix="_common",
            )
            all_metrics[i].update(common_metrics)
            metrics_df = pd.DataFrame([all_metrics[i]])
            save_table(metrics_df, art["dir"] / "variant_metrics", quiet=True)
            save_table(metrics_df, art["dir"] / "final_variant_summary", quiet=True)
            with open(art["dir"] / "variant_metrics.json", "w", encoding="utf-8") as f:
                json.dump(
                    to_jsonable(all_metrics[i]),
                    f,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                )


    # effective_trace / generic manual track GT：在 T3 reference-ID 空间跨 tracking 对齐后评价。
    # effective_trace.xlsx 的正类定义为“表内且 frame_id 非空”；其余 T3 reference tracks 均为负类。
    # generic manual GT 仍可通过 CLI 决定未列出轨迹是否按负类。
    if ann_type in {"effective_trace", "track"}:
        tp_onset_maps: Dict[str, Dict[int, Dict[str, Any]]] = {}
        ref_name = args.reference_variant
        ref_art = variant_artifacts.get(ref_name)
        if ref_art is None or not ref_art["tracks_path"].is_file():
            raise RuntimeError(f"manual aligned evaluation 缺少参考变体 tracks: {ref_name}")
        ref_tracks = pd.read_csv(ref_art["tracks_path"], encoding="utf-8-sig")
        gt_ann = build_gt_from_manual_track_annotations(
            ann_payload,
            ref_tracks,
            onset_tolerance=int(args.gt_onset_tolerance),
            missing_as_no_ignition=(
                True if ann_type == "effective_trace"
                else bool(args.manual_missing_as_no_ignition)
            ),
        )
        if ann_type == "effective_trace":
            n_pos = sum(v.get("gt_category") == "ignition" for v in gt_ann.values())
            n_neg = sum(v.get("gt_category") == "no_ignition" for v in gt_ann.values())
            print(
                f"[INFO] effective_trace reference-ID GT: positive_ignition={n_pos} "
                f"negative_no_ignition={n_neg} total={len(gt_ann)}"
            )
        for i, variant in enumerate(variants):
            art = variant_artifacts[variant]
            if not art["tracks_path"].is_file() or not art["ignition_path"].is_file():
                continue
            q_tracks = pd.read_csv(art["tracks_path"], encoding="utf-8-sig")
            q_ign = pd.read_csv(art["ignition_path"], encoding="utf-8-sig")
            if variant == ref_name:
                detected_tracks = filter_detected_track_points(q_tracks)
                query_to_ref = {int(t): int(t) for t in detected_tracks["track_id"].unique()}
                alignment_df = pd.DataFrame([
                    {
                        "query_track_id": int(tid), "ref_track_id": int(tid),
                        "n_shared_frames": int(group["frame_id"].nunique()),
                        "mean_center_distance": 0.0, "median_center_distance": 0.0,
                        "p90_center_distance": 0.0, "query_coverage": 1.0,
                        "reference_coverage": 1.0, "temporal_iou": 1.0,
                        "assignment_cost": 0.0,
                    }
                    for tid, group in detected_tracks.groupby("track_id")
                ])
            else:
                query_to_ref, alignment_df = match_tracks_to_reference(
                    q_tracks, ref_tracks,
                    max_center_dist=float(args.track_match_max_dist),
                    min_shared_frames=int(args.track_match_min_shared_frames),
                    min_ref_coverage=float(args.track_match_min_ref_coverage),
                    min_query_coverage=float(args.track_match_min_query_coverage),
                    min_temporal_iou=float(args.track_match_min_temporal_iou),
                    max_assignment_cost=float(args.track_match_max_cost),
                    detected_only=True,
                    return_diagnostics=True,
                )
            projected = project_ignition_to_ref_ids(q_ign, query_to_ref)
            eval_m = evaluate_effective_trace_new_aligned(
                gt_ann, projected,
                query_ignition_df=q_ign,
                query_to_ref=query_to_ref,
                n_query_tracks=int(q_tracks["track_id"].nunique()),
                n_matched=len(query_to_ref),
                n_reference_tracks=int(filter_detected_track_points(ref_tracks)["track_id"].nunique()),
                count_unmatched_pred_as_fp=True,
            )
            tp_onset_maps[variant] = dict(eval_m.pop("tp_onset_by_ref_id", {}) or {})
            all_metrics[i].update(eval_m)
            save_table(alignment_df, art["dir"] / "track_id_alignment_to_ref", quiet=True)
            save_table(projected, art["dir"] / "ignition_projected_to_ref", quiet=True)
            print(
                f"[{variant}] {ann_type}_aligned_eval: matched={eval_m['n_matched_to_ref']} "
                f"P={eval_m['ignition_precision']:.3f} R={eval_m['ignition_recall']:.3f} "
                f"F1={eval_m['ignition_f1']:.3f}"
            )
        common_tp_ids: Optional[set] = None
        for onset_map in tp_onset_maps.values():
            ids = set(int(k) for k in onset_map.keys())
            common_tp_ids = ids if common_tp_ids is None else (common_tp_ids & ids)
        common_tp_ids = common_tp_ids or set()
        for i, variant in enumerate(variants):
            art = variant_artifacts[variant]
            all_metrics[i].update(compute_onset_metrics_on_ids(
                tp_onset_maps.get(variant, {}), common_tp_ids, prefix="_common"
            ))
            metrics_df = pd.DataFrame([all_metrics[i]])
            save_table(metrics_df, art["dir"] / "variant_metrics", quiet=True)
            save_table(metrics_df, art["dir"] / "final_variant_summary", quiet=True)
            with open(art["dir"] / "variant_metrics.json", "w", encoding="utf-8") as f:
                json.dump(to_jsonable(all_metrics[i]), f, indent=2, ensure_ascii=False, allow_nan=False)

    summary_df = pd.DataFrame(all_metrics)
    summary_columns = list(SUMMARY_COLUMNS)
    if "B0_byte_track" not in set(summary_df.get("variant", pd.Series(dtype=str))):
        byte_only_columns = {
            "n_byte_first_stage_matches", "n_byte_second_stage_matches",
            "n_high_score_detections", "n_low_score_detections",
            "n_byte_removed_short_tracks",
        }
        summary_columns = [c for c in summary_columns if c not in byte_only_columns]
    for col in summary_columns:
        if col not in summary_df.columns:
            summary_df[col] = None
    summary_df = summary_df[[c for c in summary_columns if c in summary_df.columns]]
    order_map = {name: index for index, name in enumerate(VARIANT_ORDER)}
    summary_df["__variant_order"] = summary_df["variant"].map(order_map).fillna(999)
    summary_df = summary_df.sort_values("__variant_order").drop(
        columns=["__variant_order"]
    ).reset_index(drop=True)
    save_table(summary_df, output_root / "tracking_ablation_summary", quiet=False)

    print(f"\n{'=' * 70}")
    print("Tracking 消融实验完成")
    print(f"输出目录: {output_root}")
    print(f"汇总表: {output_root / 'tracking_ablation_summary.csv'}")
    print(f"{'=' * 70}")
    if len(summary_df):
        display_cols = [
            "variant", "n_tracks", "median_track_len", "median_detected_len",
            "short_track_ratio", "short_detected_track_ratio",
            "long_track_ratio", "long_detected_track_ratio",
            "n_matched_to_ref", "match_rate",
            "reference_track_coverage", "unmatched_reference_ignition_count",
            "unmatched_predicted_ignition_count", "n_pred_ignition",
            "ignition_precision", "ignition_recall", "ignition_f1",
            "onset_mae", "inside_interval_acc", "within_3_frame_acc",
            "n_onset_eval", "onset_mae_common", "inside_interval_acc_common",
            "n_onset_eval_common",
            "event_capture_rate", "false_event_count", "suspicious_merge_proxy_count",
        ]
        cols = [c for c in display_cols if c in summary_df.columns]
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
