#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GMM-RTV 点火检测 baseline 对比与模块消融实验。

保留既有脚本的四类 GT 读取、baseline 网格寻优、逐轨特征缓存、多进程、
逐轨/横向对比/汇总表、错误案例图以及 CSV/XLSX 输出。所有状态模型变体
只通过替换核心配置并直接调用核心检测函数实现；默认实验同时包含
shape-sequence rejection 与 initial-hot rejection 的反事实消融。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import run_ignition_ablation_experiments_revised as framework


SCRIPT_DIR = Path(__file__).resolve().parent
# 与 4.2.3 feature-group 实验保持同一 GT：仅 effective_trace 中 frame_id 非空为点火，
# 其余当前 ROI population 中的轨迹均为 no_ignition。
DEFAULT_ANNOTATION_PATH = SCRIPT_DIR / "effective_trace.xlsx"
DEFAULT_REFERENCE_SUMMARY_PATH = (
    SCRIPT_DIR / "ignition_results" / "run_20260715_115437" / "summary.csv"
)

# 论文 4.2.2 的主比较：baseline + decision-module ablation。
METHOD_COMPARISON_METHODS = (
    "gray_p99_threshold",
    "multi_feature_change_score",
    "multi_feature_change_score_rtv",
    "gmm_rtv_full",
)
CORE_ABLATION_METHODS = (
    "gmm_without_rtv",
    "gmm_no_run_length_smoothing",
    "gmm_no_prior_cold_guard",
    "gmm_rtv_full",
)
PRECHECK_ABLATION_METHODS = (
    "gmm_no_shape_precheck",
    "gmm_no_initial_hot_precheck",
    "gmm_rtv_full",
)
# 以下两个只保留作补充诊断，不进入论文 4.2.2 默认链；feature 消融由 4.2.3 专用脚本负责。
DIAGNOSTIC_METHODS = (
    "gmm_no_diff_features",
    "gmm_with_onset_refinement",
    "gmm_rtv_full",
)
DEFAULT_METHODS = (
    "gray_p99_threshold,"
    "multi_feature_change_score,"
    "multi_feature_change_score_rtv,"
    "gmm_without_rtv,"
    "gmm_no_shape_precheck,"
    "gmm_no_initial_hot_precheck,"
    "gmm_no_run_length_smoothing,"
    "gmm_no_prior_cold_guard,"
    "gmm_rtv_full"
)

# 旧名称 gmm_segmentation_only 仅为兼容别名。它实际并非“纯分割”，
# 而是完整 GMM 决策链仅关闭 RTV。
METHOD_ALIASES = {
    "gmm_segmentation_only": "gmm_without_rtv",
}

METHOD_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "gmm_without_rtv": {"enable_raw_transition_validation": False},
    "gmm_no_shape_precheck": {"enable_shape_sequence_reject": False},
    "gmm_no_initial_hot_precheck": {"enable_initial_hot_reject": False},
    "gmm_no_diff_features": {"use_diff_features": False},
    # 仅关闭 GMM state rank 的短 run 平滑；cold/hot persistence 仍保留。
    "gmm_no_run_length_smoothing": {"min_state_run": 1},
    "gmm_no_prior_cold_guard": {"require_all_prior_cold": False},
    "gmm_with_onset_refinement": {"refine_onset_with_change": True},
    "gmm_rtv_full": {},
}

# 与 4.2.3 full_state_plus_diff 的完整 19 维输入保持同一特征集合。
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

PAIRWISE_COMMON_TP_CONTRASTS = (
    ("RTV_on_change_score", "multi_feature_change_score", "multi_feature_change_score_rtv"),
    ("RTV_on_GMM", "gmm_without_rtv", "gmm_rtv_full"),
    ("shape_sequence_precheck", "gmm_no_shape_precheck", "gmm_rtv_full"),
    ("initial_hot_precheck", "gmm_no_initial_hot_precheck", "gmm_rtv_full"),
    ("run_length_smoothing", "gmm_no_run_length_smoothing", "gmm_rtv_full"),
    ("prior_cold_guard", "gmm_no_prior_cold_guard", "gmm_rtv_full"),
)
BASELINE_METHODS = (
    "gray_p99_threshold",
    "multi_feature_change_score",
    "multi_feature_change_score_rtv",
)
TUNABLE_BASELINE_METHODS = (
    "gray_p99_threshold",
    "multi_feature_change_score",
)
METHOD_ORDER = [
    "gray_p99_threshold",
    "multi_feature_change_score",
    "multi_feature_change_score_rtv",
    "gmm_without_rtv",
    "gmm_no_shape_precheck",
    "gmm_no_initial_hot_precheck",
    "gmm_no_run_length_smoothing",
    "gmm_no_prior_cold_guard",
    "gmm_rtv_full",
    # supplemental diagnostics
    "gmm_no_diff_features",
    "gmm_with_onset_refinement",
]
NO_IGNITION_STATUSES = {
    "no_ignition",
    "single_state",
    "no_ignition_weak_transition",
}
UNCERTAIN_STATUSES = {"ambiguous", "ambiguous_gmm_unstable"}
PRIMARY_GT_CATEGORIES = {"ignition", "no_ignition", "already_hot"}
HARD_EXCLUDED_GT_CATEGORIES = {"invalid", "suspicious_merge", "defocused"}
SHAPE_INVALID_GT_CATEGORIES = {"invalid", "suspicious_merge", "defocused"}
OPTIONAL_GT_CATEGORIES = {"ambiguous"}
RAW_THERMAL_FEATURES = (
    "gray_p99",
    "corr_p99",
    "bright_energy_mean",
    "bright_area_ratio",
    "hsv_v_p99",
)
DEFAULT_GRAY_THRESHOLD_GRID = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
DEFAULT_CHANGE_THRESHOLD_GRID = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40]
DEFAULT_MIN_PERSIST_GRID = [2, 3, 4, 5]

# 非 effective_trace_new 格式继续复用既有框架；reference summary 的可选 status
# 规则与 tracking 新脚本保持一致。
lookup_annotation = framework.lookup_annotation
normalize_track_name = framework.normalize_track_name
read_annotation_table = framework.read_annotation_table
read_reference_ignition_summary = framework.read_reference_ignition_summary


def build_annotations_from_effective_trace_new(
    listed_no_ignition_ids: List[int],
    reference_summary: Path,
    onset_tolerance: int = 2,
) -> Dict[str, Dict[str, Any]]:
    if onset_tolerance < 0:
        raise ValueError(f"onset_tolerance must be >= 0, got {onset_tolerance}")
    ref = read_reference_ignition_summary(reference_summary)
    has_status = "status" in ref.columns
    valid_rows = [
        row for _, row in ref.iterrows()
        if not has_status or str(row.get("status", "")).strip() == "ok"
    ]
    parsed_rows = [
        (row, *framework._parse_track_id_from_row(row, ref.columns))
        for row in valid_rows
    ]
    valid_reference_ids = {
        tid for _, _, tid in parsed_rows if tid is not None
    }
    listed_no = {
        int(track_id) for track_id in listed_no_ignition_ids
        if int(track_id) in valid_reference_ids
    }
    annotations: Dict[str, Dict[str, Any]] = {}
    n_ignition = 0
    n_no_listed = 0
    n_no_candidate = 0

    for row, track_name, track_id in parsed_rows:
        if not track_name:
            continue
        if track_id is not None and track_id in listed_no:
            annotations[track_name] = framework._no_ignition_gt()
            n_no_listed += 1
            continue
        if framework._is_no_ignition_candidate(row["ignition_idx_candidate"]):
            annotations[track_name] = framework._no_ignition_gt()
            n_no_candidate += 1
            continue
        onset = int(float(row["ignition_idx_onset"]))
        annotations[track_name] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_idx": onset,
            "gt_onset_start": max(0, onset - onset_tolerance),
            "gt_onset_end": onset + onset_tolerance,
        }
        n_ignition += 1

    n_no_ignition = sum(
        value.get("gt_category") == "no_ignition"
        for value in annotations.values()
    )
    print(
        f"[INFO] effective_trace_new 标注: listed_no_ignition={len(listed_no)} | "
        f"ref={Path(reference_summary).name} | valid_ref_rows={len(valid_rows)} | "
        f"ignition_gt={n_ignition} no_ignition_gt={n_no_ignition} "
        f"(xlsx_in_ref={n_no_listed}, candidate_-1→no_ignition={n_no_candidate}) | "
        f"onset±{onset_tolerance}"
    )
    return annotations


def load_annotations(
    annotation_path: Optional[Path],
    *,
    reference_summary: Optional[Path] = None,
    onset_tolerance: int = 2,
    metadata_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    if annotation_path is None:
        return framework.load_annotations(
            None,
            reference_summary=reference_summary,
            onset_tolerance=onset_tolerance,
            metadata_out=metadata_out,
        )
    annotation_path = Path(annotation_path)
    if annotation_path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(annotation_path, sheet_name=None)
        if any(
            framework.is_manual_review_gt_format(frame)
            and not framework._is_legacy_annotation_format(frame)
            for frame in sheets.values()
        ):
            return framework.load_annotations(
                annotation_path,
                reference_summary=reference_summary,
                onset_tolerance=onset_tolerance,
                metadata_out=metadata_out,
            )
    annotation_table = read_annotation_table(annotation_path)
    if not framework.is_effective_trace_new_format(annotation_table):
        return framework.load_annotations(
            annotation_path,
            reference_summary=reference_summary,
            onset_tolerance=onset_tolerance,
            metadata_out=metadata_out,
        )
    if metadata_out is not None:
        metadata_out.clear()
    if reference_summary is None:
        raise ValueError(
            "effective_trace_new 格式需要同时提供 --reference-summary"
        )
    listed_ids = framework.read_invalid_track_ids(annotation_path)
    warnings.warn(
        "Ignition presence/onset GT is partly derived from reference-summary "
        "predictions. This protocol may introduce circular evaluation and "
        "should not be used as the final manually independent test GT.",
        UserWarning,
    )
    return (
        build_annotations_from_effective_trace_new(
            listed_ids,
            Path(reference_summary),
            onset_tolerance=onset_tolerance,
        ),
        "effective_trace_new",
    )


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_method_name(method: str) -> str:
    method = str(method).strip()
    return METHOD_ALIASES.get(method, method)


def resolve_experiment_methods(
    group: str,
    explicit_methods: Optional[str],
) -> List[str]:
    groups = {
        "default": parse_csv_list(DEFAULT_METHODS),
        "method_comparison": list(METHOD_COMPARISON_METHODS),
        "core_ablation": list(CORE_ABLATION_METHODS),
        "precheck_ablation": list(PRECHECK_ABLATION_METHODS),
        "diagnostic": list(DIAGNOSTIC_METHODS),
        "all": list(METHOD_ORDER),
    }
    if group not in groups:
        raise ValueError(f"Unknown experiment group: {group}")
    raw_methods = parse_csv_list(explicit_methods) if explicit_methods is not None else groups[group]
    methods = [normalize_method_name(method) for method in raw_methods]
    if not methods:
        raise ValueError("methods must not be empty")
    duplicates = sorted({method for method in methods if methods.count(method) > 1})
    if duplicates:
        raise ValueError(f"Duplicate methods after alias normalization: {duplicates}")
    unknown = [method for method in methods if method not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; valid={METHOD_ORDER}")
    return methods


def validate_method_contrasts(methods: List[str], cfg: Any) -> None:
    """防止基础配置消除完整方法与对应反事实消融之间的差异。"""
    if (
        "gmm_no_shape_precheck" in methods
        and not bool(cfg.enable_shape_sequence_reject)
    ):
        raise ValueError(
            "gmm_no_shape_precheck requires shape-sequence rejection to be "
            "enabled in the full base configuration"
        )
    if (
        "gmm_no_initial_hot_precheck" in methods
        and not bool(cfg.enable_initial_hot_reject)
    ):
        raise ValueError(
            "gmm_no_initial_hot_precheck requires initial-hot rejection to be "
            "enabled in the full base configuration"
        )
    if (
        "gmm_no_prior_cold_guard" in methods
        and not bool(cfg.require_all_prior_cold)
    ):
        raise ValueError(
            "gmm_no_prior_cold_guard requires the base prior-cold guard to be "
            "enabled; remove --no-prior-cold-guard or exclude this ablation"
        )
    if (
        "gmm_with_onset_refinement" in methods
        and bool(cfg.refine_onset_with_change)
    ):
        raise ValueError(
            "gmm_with_onset_refinement requires base onset refinement to be "
            "disabled; remove --refine-onset or exclude this ablation"
        )

    full_rtv_methods = {
        "gmm_no_shape_precheck",
        "gmm_no_initial_hot_precheck",
        "gmm_no_diff_features",
        "gmm_no_run_length_smoothing",
        "gmm_no_prior_cold_guard",
        "gmm_with_onset_refinement",
        "gmm_rtv_full",
    }
    if full_rtv_methods.intersection(methods) and not bool(
        cfg.enable_raw_transition_validation
    ):
        raise ValueError(
            "The full GMM-RTV base configuration must enable raw-transition "
            "validation for these module ablations"
        )


def to_py(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_py(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(value) for value in obj]
    return obj


def to_jsonable(obj: Any) -> Any:
    value = to_py(obj)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def write_experiment_config(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def save_table(df: pd.DataFrame, path_without_ext: Path) -> None:
    csv_path = path_without_ext.with_suffix(".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {csv_path}")
    try:
        xlsx_path = path_without_ext.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False)
        print(f"[OK] Saved: {xlsx_path}")
    except Exception as exc:
        warnings.warn(f"Skip Excel export ({exc})")


def make_default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("ignition_ablation_gmm_rtv_results") / f"run_{stamp}"


def resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    import os
    return max(1, min((os.cpu_count() or 4) - 1, 16))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_progress(
    stage: str,
    completed: int,
    total: int,
    started_at: float,
    *,
    ok_count: Optional[int] = None,
    error_count: Optional[int] = None,
    every: int = 25,
    unit: str = "tracks",
) -> None:
    if completed <= 0 or total <= 0:
        return
    if completed != 1 and completed != total and completed % max(1, every) != 0:
        return
    elapsed = max(time.monotonic() - started_at, 1e-9)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else 0.0
    counts = ""
    if ok_count is not None and error_count is not None:
        counts = f" | ok={ok_count} error={error_count}"
    print(
        f"[PROGRESS] {stage}: {completed}/{total} "
        f"({100.0 * completed / total:.1f}%) | "
        f"{rate:.2f} {unit}/s | elapsed={_format_duration(elapsed)} "
        f"| ETA={_format_duration(eta)}{counts}",
        flush=True,
    )


def load_ignition_module(module_path: Path):
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Ignition module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "ignition_gmm_rtv_ablation_core", str(module_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ignition module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = [
        "IgnitionGMMConfig",
        "list_images_in_folder",
        "list_track_folders",
        "track_id_from_name",
        "extract_sequence_features",
        "get_gmm_feature_columns",
        "detect_ignition_gmm_rtv",
        "compute_hot_score",
        "compute_raw_thermal_change_score",
        "raw_transition_has_thermal_increase",
        "check_roi_shape_sequence_quality",
        "is_initially_hot_sequence",
        "normalize_to_01",
        "plot_detection_result_gmm_rtv",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise AttributeError(f"Ignition module missing required symbol: {name}")
    config_fields = getattr(mod.IgnitionGMMConfig, "__dataclass_fields__", {})
    missing_config = [
        name
        for name in (
            "require_all_prior_cold",
            "refine_onset_with_change",
            "enable_shape_sequence_reject",
            "enable_initial_hot_reject",
            "enable_raw_transition_validation",
        )
        if name not in config_fields
    ]
    if missing_config:
        raise RuntimeError(
            "IgnitionGMMConfig missing required configuration: "
            + ", ".join(missing_config)
        )
    return mod


def validate_full_feature_alignment(df: pd.DataFrame, cfg: Any, mod: Any) -> List[str]:
    """验证 4.2.2 full GMM-RTV 与 4.2.3 full_state_plus_diff 使用同一 19 维特征集合。"""
    cols = list(mod.get_gmm_feature_columns(df, cfg))
    expected = list(EXPECTED_FULL_GMM_FEATURES)
    missing = [name for name in expected if name not in cols]
    extra = [name for name in cols if name not in expected]
    if missing or extra or len(cols) != len(expected):
        raise RuntimeError(
            "Full GMM feature set is not aligned with 4.2.3 full_state_plus_diff. "
            f"missing={missing}, extra={extra}, resolved={cols}"
        )
    if not bool(cfg.use_diff_features):
        raise RuntimeError("Full configuration must keep use_diff_features=True")
    print(
        f"[CHECK] Full GMM feature alignment OK: {len(cols)} features "
        f"(13 absolute + 6 temporal-difference).",
        flush=True,
    )
    return cols


def audit_effective_trace_population(
    tracks: List[Tuple[str, str]],
    annotations: Dict[str, Dict[str, Any]],
    mod: Any,
    annotation_format: str,
) -> Dict[str, Any]:
    """打印与 4.2.3 一致的 effective_trace 二分类 population 统计。"""
    available_names = {str(name) for name, _ in tracks}
    available_ids = {
        int(tid) for name, _ in tracks
        for tid in [mod.track_id_from_name(name)] if tid is not None
    }
    positive_names = {
        str(name) for name, gt in annotations.items()
        if str(gt.get("gt_category", "")).lower() == "ignition"
    }
    positive_available = set()
    missing_positive = []
    for name in sorted(positive_names):
        tid = mod.track_id_from_name(name)
        present = name in available_names or (tid is not None and int(tid) in available_ids)
        if present:
            positive_available.add(name)
        else:
            missing_positive.append(name)
    payload = {
        "annotation_format": annotation_format,
        "n_roi_tracks": len(tracks),
        "n_listed_positive": len(positive_names),
        "n_evaluable_positive": len(positive_available),
        "n_implicit_negative": max(0, len(tracks) - len(positive_available)),
        "n_missing_positive": len(missing_positive),
        "missing_positive_tracks": missing_positive,
    }
    if annotation_format == "effective_trace":
        print(
            "[GT] effective_trace: positive iff listed with non-empty frame_id; "
            "all other ROI tracks are no_ignition.",
            flush=True,
        )
        print(
            f"[GT] roi_tracks={payload['n_roi_tracks']} | "
            f"listed_positive={payload['n_listed_positive']} | "
            f"evaluable_positive={payload['n_evaluable_positive']} | "
            f"implicit_negative={payload['n_implicit_negative']} | "
            f"missing_positive={payload['n_missing_positive']}",
            flush=True,
        )
        if missing_positive:
            warnings.warn(
                "Positive GT tracks absent from the current ROI root: "
                + ", ".join(missing_positive[:30])
                + (" ..." if len(missing_positive) > 30 else ""),
                UserWarning,
            )
    return payload


def _gmm_ablation_fields(local_cfg: Any) -> Dict[str, Any]:
    return {
        "ablation_enable_raw_transition_validation": bool(
            local_cfg.enable_raw_transition_validation
        ),
        "ablation_enable_shape_precheck": bool(local_cfg.enable_shape_sequence_reject),
        "ablation_enable_initial_hot_precheck": bool(local_cfg.enable_initial_hot_reject),
        "ablation_enable_diff_features": bool(local_cfg.use_diff_features),
        "ablation_enable_run_length_smoothing": int(local_cfg.min_state_run) > 1,
        "ablation_enable_prior_cold_guard": bool(
            local_cfg.require_all_prior_cold
        ),
        "ablation_enable_onset_refinement": bool(local_cfg.refine_onset_with_change),
        "ablation_min_state_run": int(local_cfg.min_state_run),
        "ablation_min_cold_run": int(local_cfg.min_cold_run),
        "ablation_min_hot_run": int(local_cfg.min_hot_run),
        "ablation_refine_radius": int(local_cfg.refine_radius),
    }


def _baseline_ablation_fields(method: str, cfg: Any) -> Dict[str, Any]:
    return {
        "ablation_enable_raw_transition_validation": (
            method == "multi_feature_change_score_rtv"
        ),
        "ablation_enable_shape_precheck": False,
        "ablation_enable_initial_hot_precheck": bool(
            cfg.enable_initial_hot_reject
        ),
        "ablation_enable_diff_features": False,
        "ablation_enable_run_length_smoothing": False,
        "ablation_enable_prior_cold_guard": False,
        "ablation_enable_onset_refinement": False,
        "ablation_min_state_run": 0,
        "ablation_min_cold_run": 0,
        "ablation_min_hot_run": 0,
        "ablation_refine_radius": 0,
    }


def _empty_result(mod: Any, cfg: Any, n: int, method: str, status: str) -> Dict[str, Any]:
    result = {
        "has_ignition": False,
        "ignition_status": status,
        "ignition_idx_onset": cfg.no_ignition_index,
        "ignition_idx_confirmed": cfg.no_ignition_index,
        "ignition_idx_candidate": cfg.no_ignition_index,
        "stable_combustion_idx": cfg.no_ignition_index,
        "ignition_confidence": 0.0,
        "ignition_amplitude": 0.0,
        "ignition_change_peak": 0.0,
        "gmm_n_states": 0,
        "gmm_state_sequence": [],
        "gmm_feature_cols": [],
        "hot_score": np.zeros(n, dtype=np.float32),
        "change_score": np.zeros(n, dtype=np.float32),
        "ignition_method": method,
        "precheck_diagnostics": {"shape": {}, "initial_hot": {}},
    }
    if method in BASELINE_METHODS:
        result.update(_baseline_ablation_fields(method, cfg))
    elif method in METHOD_CONFIG_OVERRIDES:
        result.update(_gmm_ablation_fields(
            replace(cfg, **METHOD_CONFIG_OVERRIDES[method])
        ))
    else:
        result.update(_gmm_ablation_fields(cfg))
    return result


def _baseline_result(
    mod: Any,
    cfg: Any,
    df: pd.DataFrame,
    method: str,
    status: str,
    score: np.ndarray,
    onset: Optional[int] = None,
    *,
    initial_hot_metrics: Optional[Dict[str, Any]] = None,
    raw_validation: bool = False,
) -> Dict[str, Any]:
    result = _empty_result(mod, cfg, len(df), method, status)
    result["hot_score"] = mod.compute_hot_score(df)
    result["change_score"] = score
    result.update({
        "precheck_diagnostics": {
            "shape": {},
            "initial_hot": initial_hot_metrics or {},
        },
        "ablation_enable_raw_transition_validation": bool(raw_validation),
        "ablation_enable_shape_precheck": False,
        "ablation_enable_initial_hot_precheck": bool(
            cfg.enable_initial_hot_reject
        ),
        "ablation_enable_diff_features": False,
        "ablation_enable_run_length_smoothing": False,
        "ablation_enable_prior_cold_guard": False,
        "ablation_enable_onset_refinement": False,
        "ablation_min_state_run": 0,
        "ablation_min_cold_run": 0,
        "ablation_min_hot_run": 0,
        "ablation_refine_radius": 0,
    })
    if onset is not None:
        idx = int(onset)
        result.update({
            "has_ignition": True,
            "ignition_status": "ignition_detected",
            "ignition_idx_onset": idx,
            "ignition_idx_confirmed": idx,
            "ignition_idx_candidate": idx,
            "stable_combustion_idx": idx,
            "ignition_confidence": float(np.clip(score[idx], 0.0, 1.0)),
            "ignition_change_peak": float(score[idx]),
        })
    return result


def _persistent_onset(score: np.ndarray, threshold: float, count: int) -> Optional[int]:
    if count <= 0:
        raise ValueError("min_persist must be positive")
    for index in range(max(0, len(score) - count + 1)):
        if np.all(score[index:index + count] >= threshold):
            return index
    return None


def detect_gray_p99_threshold(
    df: pd.DataFrame,
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    method = "gray_p99_threshold"
    if len(df) == 0 or "gray_p99" not in df:
        return _empty_result(mod, cfg, len(df), method, "ambiguous")
    score = mod.normalize_to_01(df["gray_p99"].to_numpy(dtype=np.float32))
    initially_hot, initial_hot_metrics = mod.is_initially_hot_sequence(df, cfg)
    if initially_hot:
        return _baseline_result(
            mod, cfg, df, method, "already_hot_at_sequence_start", score,
            initial_hot_metrics=initial_hot_metrics,
        )
    onset = _persistent_onset(
        score, float(params["gray_threshold"]), int(params["gray_min_persist"])
    )
    return _baseline_result(
        mod, cfg, df, method, "no_ignition", score, onset,
        initial_hot_metrics=initial_hot_metrics,
    )


def _detect_change_score_candidate(
    df: pd.DataFrame,
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[int], str, Dict[str, Any]]:
    if not any(column in df.columns for column in RAW_THERMAL_FEATURES):
        raise ValueError(
            "multi_feature_change_score requires at least one raw thermal feature: "
            + ", ".join(RAW_THERMAL_FEATURES)
        )
    if len(df) < cfg.min_sequence_len:
        return np.zeros(len(df), dtype=np.float32), None, "ambiguous", {}
    score = mod.compute_raw_thermal_change_score(df, window=3)
    initially_hot, initial_hot_metrics = mod.is_initially_hot_sequence(df, cfg)
    if initially_hot:
        return score, None, "already_hot_at_sequence_start", initial_hot_metrics
    onset = _persistent_onset(
        score, float(params["change_threshold"]), int(params["change_min_persist"])
    )
    if onset is None:
        return score, None, "no_ignition", initial_hot_metrics
    return score, onset, "candidate", initial_hot_metrics


def detect_multi_feature_change_score(
    df: pd.DataFrame,
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    method = "multi_feature_change_score"
    score, onset, status, initial_hot_metrics = _detect_change_score_candidate(
        df, mod, cfg, params
    )
    return _baseline_result(
        mod, cfg, df, method, status, score,
        onset if status == "candidate" else None,
        initial_hot_metrics=initial_hot_metrics,
    )


def detect_multi_feature_change_score_rtv(
    df: pd.DataFrame,
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    method = "multi_feature_change_score_rtv"
    score, onset, status, initial_hot_metrics = _detect_change_score_candidate(
        df, mod, cfg, params
    )
    if status != "candidate" or onset is None:
        return _baseline_result(
            mod, cfg, df, method, status, score,
            initial_hot_metrics=initial_hot_metrics, raw_validation=True,
        )
    valid, _ = mod.raw_transition_has_thermal_increase(df, onset, cfg)
    if not valid:
        return _baseline_result(
            mod, cfg, df, method, "no_ignition_weak_transition", score,
            initial_hot_metrics=initial_hot_metrics, raw_validation=True,
        )
    return _baseline_result(
        mod, cfg, df, method, "ignition_detected", score, onset,
        initial_hot_metrics=initial_hot_metrics, raw_validation=True,
    )


def _attach_ablation_fields(result: Dict[str, Any], local_cfg: Any) -> Dict[str, Any]:
    result = dict(result)
    result.update(_gmm_ablation_fields(local_cfg))
    return result


def run_method(
    method: str,
    df: pd.DataFrame,
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if method == "gray_p99_threshold":
        return detect_gray_p99_threshold(df, mod, cfg, params)
    if method == "multi_feature_change_score":
        return detect_multi_feature_change_score(df, mod, cfg, params)
    if method == "multi_feature_change_score_rtv":
        return detect_multi_feature_change_score_rtv(df, mod, cfg, params)
    if method in METHOD_CONFIG_OVERRIDES:
        override = METHOD_CONFIG_OVERRIDES[method]
        local_cfg = replace(cfg, **override)
        result = mod.detect_ignition_gmm_rtv(df, local_cfg)
        result = _attach_ablation_fields(result, local_cfg)
        result["ignition_method"] = method
        return result
    raise ValueError(f"Unknown method: {method}")


def evaluate_prediction(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    category = str(gt.get("gt_category", "")).lower()
    pred_has = bool(pred.get("has_ignition", False))
    status = str(pred.get("ignition_status", ""))
    pred_onset = int(pred.get("ignition_idx_onset", -1))
    gt_onset = int(gt.get("gt_onset_idx", -1))
    gt_start = int(gt.get("gt_onset_start", -1))
    gt_end = int(gt.get("gt_onset_end", -1))
    correct: Any = np.nan
    false_ignition = False
    missed = False
    hot_correct = False
    onset_error: Any = np.nan
    inside: Any = np.nan
    if category == "ignition":
        correct = pred_has
        missed = not pred_has
        if pred_has and pred_onset >= 0 and gt_onset >= 0:
            onset_error = abs(pred_onset - gt_onset)
            if gt_start >= 0 and gt_end >= gt_start:
                inside = gt_start <= pred_onset <= gt_end
    elif category == "no_ignition":
        correct = not pred_has
        false_ignition = pred_has
    elif category == "already_hot":
        hot_correct = status in {"already_hot", "already_hot_at_sequence_start"}
        correct = hot_correct
    return {
        "gt_category": category,
        "gt_has_ignition": bool(gt.get("gt_has_ignition", False)),
        "gt_onset_idx": gt_onset,
        "gt_onset_start": gt_start,
        "gt_onset_end": gt_end,
        "onset_abs_error": onset_error,
        "onset_inside_interval": inside,
        "is_correct_presence": correct,
        "is_false_ignition": false_ignition,
        "is_missed_ignition": missed,
        "is_already_hot_correct": hot_correct,
    }


def result_to_row(
    track_name: str,
    track_id: Optional[int],
    method: str,
    n_frames: int,
    pred: Dict[str, Any],
    evaluation: Dict[str, Any],
    error_message: str = "",
) -> Dict[str, Any]:
    sequence = pred.get("gmm_state_sequence", []) or []
    diagnostics = pred.get("precheck_diagnostics", {}) or {}
    shape_diag = diagnostics.get("shape", {}) if isinstance(diagnostics, dict) else {}
    initial_hot_diag = (
        diagnostics.get("initial_hot", {}) if isinstance(diagnostics, dict) else {}
    )
    shape_reasons = (
        pred.get("shape_reject_reasons")
        or shape_diag.get("shape_reject_reasons")
        or shape_diag.get("reject_reasons")
        or []
    )
    if isinstance(shape_reasons, str):
        shape_reasons_text = shape_reasons
    else:
        shape_reasons_text = "|".join(str(value) for value in shape_reasons)
    initial_hot_fraction = initial_hot_diag.get("initial_hot_frame_fraction", np.nan)
    row = {
        "track_name": track_name,
        "track_id": track_id,
        "method": method,
        "n_frames": n_frames,
        "pred_has_ignition": bool(pred.get("has_ignition", False)),
        "pred_ignition_status": str(pred.get("ignition_status", "")),
        "pred_onset_idx": int(pred.get("ignition_idx_onset", -1)),
        "pred_confirmed_idx": int(pred.get("ignition_idx_confirmed", -1)),
        "pred_stable_idx": int(pred.get("stable_combustion_idx", -1)),
        "ignition_confidence": float(pred.get("ignition_confidence", 0.0)),
        "ignition_amplitude": float(pred.get("ignition_amplitude", 0.0)),
        "ignition_change_peak": float(pred.get("ignition_change_peak", 0.0)),
        "gmm_n_states": int(pred.get("gmm_n_states", 0)),
        "gmm_state_sequence_str": ";".join(str(value) for value in sequence),
        "gmm_feature_cols": ";".join(
            str(value) for value in (pred.get("gmm_feature_cols", []) or [])
        ),
        "precheck_rejected": bool(pred.get("precheck_rejected", False)),
        "shape_quality_warning": bool(
            pred.get(
                "shape_quality_warning",
                shape_diag.get("shape_quality_warning", False),
            )
        ),
        "shape_evidence_count": int(
            pred.get(
                "shape_evidence_count",
                shape_diag.get("shape_evidence_count", 0),
            )
            or 0
        ),
        "shape_reject_reasons": shape_reasons_text,
        "initial_hot_frame_fraction": (
            float(initial_hot_fraction)
            if pd.notna(initial_hot_fraction)
            else float("nan")
        ),
        "initial_hot_frames_checked": int(
            initial_hot_diag.get("n_initial_frames_checked", 0) or 0
        ),
        "ablation_enable_raw_transition_validation": bool(
            pred.get("ablation_enable_raw_transition_validation", False)
        ),
        "ablation_enable_shape_precheck": bool(
            pred.get("ablation_enable_shape_precheck", False)
        ),
        "ablation_enable_initial_hot_precheck": bool(
            pred.get("ablation_enable_initial_hot_precheck", False)
        ),
        "ablation_enable_diff_features": bool(
            pred.get("ablation_enable_diff_features", False)
        ),
        "ablation_enable_run_length_smoothing": bool(
            pred.get("ablation_enable_run_length_smoothing", False)
        ),
        "ablation_enable_prior_cold_guard": bool(
            pred.get("ablation_enable_prior_cold_guard", False)
        ),
        "ablation_enable_onset_refinement": bool(
            pred.get("ablation_enable_onset_refinement", False)
        ),
        "ablation_min_state_run": int(pred.get("ablation_min_state_run", 0)),
        "ablation_min_cold_run": int(pred.get("ablation_min_cold_run", 0)),
        "ablation_min_hot_run": int(pred.get("ablation_min_hot_run", 0)),
        "ablation_refine_radius": int(pred.get("ablation_refine_radius", 0)),
        **evaluation,
        "error_message": error_message,
        "plot_error": "",
    }
    return to_py(row)


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def count_statuses(series: pd.Series, statuses: List[str]) -> Dict[str, int]:
    values = series.astype(str)
    return {f"n_status_{status}": int((values == status).sum()) for status in statuses}


def _track_identity_series(df: pd.DataFrame) -> pd.Series:
    """优先 track_name，否则 track_id；用于跨方法对齐同一轨迹。"""
    if "track_name" in df.columns:
        names = df["track_name"].astype(str).str.strip()
        valid = names.ne("") & names.str.lower().ne("nan") & names.str.lower().ne("none")
        if bool(valid.any()):
            return names
    if "track_id" in df.columns:
        return df["track_id"].map(lambda x: str(int(x)) if pd.notna(x) else "")
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _onset_metrics_from_hits(hits: pd.DataFrame, *, prefix: str = "") -> Dict[str, Any]:
    """由 TP hits 计算 onset 指标；prefix='' 为条件口径，prefix='_common' 为统一口径。"""
    n_key = f"n_onset_eval{prefix}"
    mae_key = f"onset_mae{prefix}"
    w1_key = f"within_1_frame_acc{prefix}"
    w3_key = f"within_3_frame_acc{prefix}"
    inside_key = f"inside_interval_acc{prefix}"
    if hits is None or len(hits) == 0:
        return {
            n_key: 0,
            mae_key: float("nan"),
            w1_key: float("nan"),
            w3_key: float("nan"),
            inside_key: float("nan"),
        }
    errors = pd.to_numeric(hits["onset_abs_error"], errors="coerce")
    interval = hits[pd.notna(hits["onset_inside_interval"])]
    return {
        n_key: int(len(hits)),
        mae_key: _safe_mean(errors),
        w1_key: _safe_rate(int((errors <= 1).sum()), len(hits)),
        w3_key: _safe_rate(int((errors <= 3).sum()), len(hits)),
        inside_key: _safe_rate(
            int(interval["onset_inside_interval"].astype(bool).sum()),
            len(interval),
        ),
    }


def _method_tp_onset_hits(
    sub: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    """返回该方法上 GT=ignition 且预测点火的 per-track 行（用于 onset）。"""
    status_errors = sub["pred_ignition_status"].isin(["method_error", "track_error"])
    if "error_message" in sub:
        message_errors = sub["error_message"].fillna("").astype(str).str.strip().ne("")
    else:
        message_errors = pd.Series(False, index=sub.index)
    model_sub = sub[~(status_errors | message_errors)]
    valid = model_sub[model_sub["gt_category"].astype(str).str.strip() != ""]
    valid = valid[~valid["gt_category"].isin(HARD_EXCLUDED_GT_CATEGORIES)]
    allowed = (
        PRIMARY_GT_CATEGORIES | OPTIONAL_GT_CATEGORIES
        if include_ambiguous
        else PRIMARY_GT_CATEGORIES
    )
    valid = valid[valid["gt_category"].isin(allowed)]
    gt_ign = valid[valid["gt_category"].astype(str).str.lower() == "ignition"]
    hits = gt_ign[gt_ign["pred_has_ignition"] == True].copy()  # noqa: E712
    if hits.empty:
        return hits
    hits["__track_identity"] = _track_identity_series(hits)
    hits = hits[hits["__track_identity"].astype(str).str.strip().ne("")]
    return hits


def compute_method_summary(
    per_track_df: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    if per_track_df.empty or "method" not in per_track_df:
        return pd.DataFrame()
    rows = []
    tp_hits_by_method: Dict[str, pd.DataFrame] = {}
    for method, sub in per_track_df.groupby("method", sort=False):
        status_errors = sub["pred_ignition_status"].isin(
            ["method_error", "track_error"]
        )
        if "error_message" in sub:
            message_errors = sub["error_message"].fillna("").astype(str).str.strip().ne("")
        else:
            message_errors = pd.Series(False, index=sub.index)
        errors = status_errors | message_errors
        model_sub = sub[~errors]

        raw_gt = model_sub["gt_category"].astype(str).str.strip().str.lower()
        shape_invalid = model_sub[raw_gt.isin(SHAPE_INVALID_GT_CATEGORIES)]
        shape_valid = model_sub[
            raw_gt.ne("") & ~raw_gt.isin(SHAPE_INVALID_GT_CATEGORIES)
        ]
        shape_invalid_rejected = int(
            (shape_invalid["pred_ignition_status"] == "rejected_shape_sequence").sum()
        )
        shape_valid_rejected = int(
            (shape_valid["pred_ignition_status"] == "rejected_shape_sequence").sum()
        )
        valid = model_sub[
            model_sub["gt_category"].astype(str).str.strip() != ""
        ]
        valid = valid[
            ~valid["gt_category"].isin(HARD_EXCLUDED_GT_CATEGORIES)
        ]
        if not include_ambiguous:
            allowed_categories = PRIMARY_GT_CATEGORIES
        else:
            allowed_categories = PRIMARY_GT_CATEGORIES | OPTIONAL_GT_CATEGORIES
        valid = valid[valid["gt_category"].isin(allowed_categories)]
        gt_ign = valid[valid["gt_category"] == "ignition"]
        gt_no = valid[valid["gt_category"] == "no_ignition"]
        gt_hot = valid[valid["gt_category"] == "already_hot"]
        tp = int(gt_ign["pred_has_ignition"].sum())
        fn = int(len(gt_ign) - tp)
        fp = int(gt_no["pred_has_ignition"].sum())
        hot_fp = int(gt_hot["pred_has_ignition"].sum())
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if pd.notna(precision) and pd.notna(recall) and precision + recall > 0
            else float("nan")
        )
        hits = _method_tp_onset_hits(sub, include_ambiguous=include_ambiguous)
        tp_hits_by_method[str(method)] = hits
        state_counts = model_sub["gmm_n_states"].value_counts().to_dict()
        rows.append({
            "method": method,
            "n_tracks": len(sub),
            "n_valid_gt": len(valid),
            "n_ambiguous_gt": int(
                (valid["gt_category"] == "ambiguous").sum()
            ),
            "n_method_errors": int(errors.sum()),
            "method_error_rate": _safe_rate(int(errors.sum()), len(sub)),
            "n_precheck_rejected": int(
                model_sub.get(
                    "precheck_rejected", pd.Series(False, index=model_sub.index)
                ).astype(bool).sum()
            ),
            "n_shape_sequence_rejected": int(
                (model_sub["pred_ignition_status"] == "rejected_shape_sequence").sum()
            ),
            "n_initial_hot_rejected": int(
                (
                    model_sub["pred_ignition_status"]
                    == "already_hot_at_sequence_start"
                ).sum()
            ),
            "mean_initial_hot_frame_fraction": _safe_mean(
                model_sub.get(
                    "initial_hot_frame_fraction", pd.Series(dtype=float)
                )
            ),
            "mean_gmm_n_states": _safe_mean(model_sub["gmm_n_states"]),
            "n_gmm_k0": int(state_counts.get(0, 0)),
            "n_gmm_k1": int(state_counts.get(1, 0)),
            "n_gmm_k2": int(state_counts.get(2, 0)),
            "n_gmm_k3": int(state_counts.get(3, 0)),
            **count_statuses(sub["pred_ignition_status"], [
                "ignition_detected", "no_ignition", "single_state",
                "no_ignition_weak_transition", "already_hot",
                "already_hot_at_sequence_start", "rejected_shape_sequence",
                "ambiguous", "ambiguous_gmm_unstable", "method_error", "track_error",
            ]),
            "ignition_precision": precision,
            "ignition_recall": recall,
            "ignition_f1": f1,
            "false_ignition_rate": _safe_rate(fp, len(gt_no)),
            "missed_ignition_rate": _safe_rate(fn, len(gt_ign)),
            "already_hot_accuracy": _safe_rate(
                int(gt_hot["is_already_hot_correct"].sum()), len(gt_hot)
            ),
            "already_hot_false_ignition_count": hot_fp,
            "already_hot_false_ignition_rate": _safe_rate(hot_fp, len(gt_hot)),
            **_onset_metrics_from_hits(hits, prefix=""),
            "shape_invalid_rejection_recall": _safe_rate(
                shape_invalid_rejected, len(shape_invalid)
            ),
            "valid_track_retention_rate": _safe_rate(
                len(shape_valid) - shape_valid_rejected, len(shape_valid)
            ),
            "shape_false_rejection_rate": _safe_rate(
                shape_valid_rejected, len(shape_valid)
            ),
        })

    # 统一口径：各方法 TP 轨迹交集上再算一套 onset 指标，便于横向公平对比。
    common_ids: Optional[set] = None
    for hits in tp_hits_by_method.values():
        ids = set(hits["__track_identity"].astype(str)) if len(hits) else set()
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = common_ids or set()
    for row in rows:
        method = str(row["method"])
        hits = tp_hits_by_method.get(method, pd.DataFrame())
        if common_ids and len(hits):
            common_hits = hits[hits["__track_identity"].astype(str).isin(common_ids)]
        else:
            common_hits = hits.iloc[0:0]
        row.update(_onset_metrics_from_hits(common_hits, prefix="_common"))
        # n_onset_eval_common 固定为交集大小（各方法相同），即使某方法行过滤异常也为该值。
        row["n_onset_eval_common"] = int(len(common_ids))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["method"] = pd.Categorical(
            summary["method"], categories=METHOD_ORDER, ordered=True
        )
        summary = summary.sort_values("method").reset_index(drop=True)
    return summary


def build_pairwise_common_tp_summary(
    per_track_df: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    """对每个消融与其直接 reference 做 pairwise Common-TP onset 比较。

    不再用“所有方法 TP 的全局交集”解释单个模块的定位作用，避免 baseline
    或无关消融把 Common-TP 集合不必要地缩小。
    """
    if per_track_df.empty or "method" not in per_track_df.columns:
        return pd.DataFrame()
    available = set(per_track_df["method"].astype(str))
    rows: List[Dict[str, Any]] = []

    def paired_metrics(hits: pd.DataFrame, ids: Set[str]) -> Dict[str, Any]:
        if hits.empty or not ids:
            selected = hits.iloc[0:0]
        else:
            selected = hits[hits["__track_identity"].astype(str).isin(ids)]
        metrics = _onset_metrics_from_hits(selected, prefix="")
        return {
            "n": int(metrics["n_onset_eval"]),
            "mae": metrics["onset_mae"],
            "acc_pm1": metrics["within_1_frame_acc"],
            "acc_pm3": metrics["within_3_frame_acc"],
            "inside_interval_acc": metrics["inside_interval_acc"],
        }

    for contrast, ablated_method, reference_method in PAIRWISE_COMMON_TP_CONTRASTS:
        if ablated_method not in available or reference_method not in available:
            continue
        a = per_track_df[per_track_df["method"].astype(str) == ablated_method]
        b = per_track_df[per_track_df["method"].astype(str) == reference_method]
        hits_a = _method_tp_onset_hits(a, include_ambiguous=include_ambiguous)
        hits_b = _method_tp_onset_hits(b, include_ambiguous=include_ambiguous)
        ids_a = set(hits_a["__track_identity"].astype(str)) if len(hits_a) else set()
        ids_b = set(hits_b["__track_identity"].astype(str)) if len(hits_b) else set()
        common_ids = ids_a & ids_b
        ma = paired_metrics(hits_a, common_ids)
        mb = paired_metrics(hits_b, common_ids)
        rows.append({
            "contrast": contrast,
            "ablated_method": ablated_method,
            "reference_method": reference_method,
            "n_common_tp": len(common_ids),
            "ablated_common_mae": ma["mae"],
            "reference_common_mae": mb["mae"],
            "delta_reference_minus_ablated_mae": (
                float(mb["mae"] - ma["mae"])
                if pd.notna(ma["mae"]) and pd.notna(mb["mae"]) else float("nan")
            ),
            "ablated_common_acc_pm1": ma["acc_pm1"],
            "reference_common_acc_pm1": mb["acc_pm1"],
            "ablated_common_acc_pm3": ma["acc_pm3"],
            "reference_common_acc_pm3": mb["acc_pm3"],
            "ablated_inside_interval_acc": ma["inside_interval_acc"],
            "reference_inside_interval_acc": mb["inside_interval_acc"],
        })
    return pd.DataFrame(rows)


def build_precheck_ablation_contrast(summary: pd.DataFrame) -> pd.DataFrame:
    """比较完整方法与两项预检查反事实消融，正 delta 表示完整方法更高。"""
    required = {
        "gmm_rtv_full",
        "gmm_no_shape_precheck",
        "gmm_no_initial_hot_precheck",
    }
    if summary.empty or "method" not in summary:
        return pd.DataFrame()
    indexed = summary.copy()
    indexed["method"] = indexed["method"].astype(str)
    indexed = indexed.set_index("method", drop=False)
    if not required.issubset(set(indexed.index)):
        return pd.DataFrame()

    full = indexed.loc["gmm_rtv_full"]
    metrics = (
        "ignition_precision",
        "ignition_recall",
        "ignition_f1",
        "false_ignition_rate",
        "missed_ignition_rate",
        "onset_mae",
        "already_hot_accuracy",
        "shape_invalid_rejection_recall",
        "valid_track_retention_rate",
        "shape_false_rejection_rate",
    )
    rows: List[Dict[str, Any]] = []
    for module, method in (
        ("shape_sequence_rejection", "gmm_no_shape_precheck"),
        ("initial_hot_rejection", "gmm_no_initial_hot_precheck"),
    ):
        ablated = indexed.loc[method]
        row: Dict[str, Any] = {
            "module": module,
            "full_method": "gmm_rtv_full",
            "ablated_method": method,
        }
        for metric in metrics:
            full_value = pd.to_numeric(full.get(metric), errors="coerce")
            ablated_value = pd.to_numeric(ablated.get(metric), errors="coerce")
            row[f"full_{metric}"] = full_value
            row[f"ablated_{metric}"] = ablated_value
            row[f"delta_full_minus_ablated_{metric}"] = (
                float(full_value - ablated_value)
                if pd.notna(full_value) and pd.notna(ablated_value)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_method_result_cell(row: pd.Series) -> str:
    status = str(row.get("pred_ignition_status", ""))
    if status in {"method_error", "track_error"}:
        return "ERR"
    if status in NO_IGNITION_STATUSES:
        return "N"
    if status:
        if bool(row.get("pred_has_ignition", False)):
            return f"Y:{int(row.get('pred_onset_idx', -1))}"
        return status
    return "N"


def build_method_track_comparison_table(
    per_track_df: pd.DataFrame,
    methods: List[str],
) -> pd.DataFrame:
    if per_track_df.empty:
        return pd.DataFrame(columns=["track_name", "track_id", *methods])
    rows = []
    for (track_name, track_id), group in per_track_df.groupby(
        ["track_name", "track_id"], dropna=False
    ):
        row: Dict[str, Any] = {"track_name": track_name, "track_id": track_id}
        for method in methods:
            selected = group[group["method"] == method]
            row[method] = (
                format_method_result_cell(selected.iloc[0]) if len(selected) else ""
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["track_id", "track_name"], na_position="last"
    ).reset_index(drop=True)


def is_error_case(row: pd.Series) -> bool:
    category = str(row.get("gt_category", ""))
    if category == "ignition":
        return (
            not bool(row.get("pred_has_ignition", False))
            or (
                pd.notna(row.get("onset_abs_error"))
                and float(row["onset_abs_error"]) > 3
            )
        )
    if category == "no_ignition":
        return bool(row.get("is_false_ignition", False))
    if category == "already_hot":
        return not bool(row.get("is_already_hot_correct", False))
    return False


def plot_ablation_result(
    df: pd.DataFrame,
    result: Dict[str, Any],
    method: str,
    save_path: Path,
) -> None:
    """绘制带当前消融方法标题的错误案例图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(df))
    fig, axis = plt.subplots(figsize=(12, 6))
    for column in ("gray_p99", "bright_area_ratio", "bright_energy_mean"):
        if column in df:
            axis.plot(
                x,
                np.asarray(result.get(f"plot_{column}", []))
                if f"plot_{column}" in result
                else (df[column] - df[column].min())
                / max(float(df[column].max() - df[column].min()), 1e-6),
                label=column,
            )
    for key in ("hot_score", "change_score"):
        values = np.asarray(result.get(key, []))
        if len(values) == len(df):
            axis.plot(x, values, label=key, linewidth=1.5)
    states = np.asarray(result.get("gmm_state_sequence", []))
    if len(states) == len(df):
        denominator = max(int(result.get("gmm_n_states", 1)) - 1, 1)
        axis.step(x, states / denominator, where="mid", label="gmm_state_rank")
    if result.get("has_ignition"):
        onset = int(result.get("ignition_idx_onset", -1))
        if onset >= 0:
            axis.axvline(onset, color="orange", linestyle=":", label=f"onset={onset}")
    axis.set_title(
        f"Method: {method} | status: {result.get('ignition_status', '')} | "
        f"states: K={int(result.get('gmm_n_states', 0))}"
    )
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Normalized score / state rank")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def process_one_track(
    track_name: str,
    folder: str,
    methods: List[str],
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
    annotations: Dict[str, Dict[str, Any]],
    save_error_cases: bool,
    error_cases_dir: Optional[Path],
    missing_as_no_ignition: bool,
    precomputed_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    track_id = mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        annotations, track_name, track_id,
        missing_as_no_ignition=missing_as_no_ignition,
        missing_as_invalid=False,
    )
    rows = []
    try:
        if precomputed_df is None:
            paths = mod.list_images_in_folder(folder)
            if not paths:
                raise RuntimeError("No images found in track folder")
            df = mod.extract_sequence_features(paths, cfg)
        else:
            df = precomputed_df
        for method in dict.fromkeys(methods):
            try:
                pred = run_method(method, df, mod, cfg, params)
                row = result_to_row(
                    track_name, track_id, method, len(df), pred,
                    evaluate_prediction(pred, gt),
                )
                rows.append(row)
                if save_error_cases and error_cases_dir and is_error_case(pd.Series(row)):
                    save_path = error_cases_dir / method / f"{track_name}.png"
                    try:
                        plot_ablation_result(df, pred, method, save_path)
                    except Exception as plot_exc:
                        row["plot_error"] = str(plot_exc)
            except Exception as exc:
                pred = _empty_result(mod, cfg, len(df), method, "method_error")
                rows.append(result_to_row(
                    track_name, track_id, method, len(df), pred,
                    evaluate_prediction(pred, gt), str(exc),
                ))
    except Exception as exc:
        for method in methods:
            pred = _empty_result(mod, cfg, 0, method, "track_error")
            rows.append(result_to_row(
                track_name, track_id, method, 0, pred,
                evaluate_prediction(pred, gt), str(exc),
            ))
    return rows


def build_track_error_rows(
    task: Dict[str, Any],
    methods: List[str],
    mod: Any,
    cfg: Any,
    annotations: Dict[str, Dict[str, Any]],
    missing_as_no_ignition: bool,
    error_message: str,
) -> List[Dict[str, Any]]:
    """为未返回结果的整条轨迹生成完整、同 schema 的错误行。"""
    track_name = task["track_name"]
    track_id = mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        annotations,
        track_name,
        track_id,
        missing_as_no_ignition=missing_as_no_ignition,
        missing_as_invalid=False,
    )
    rows = []
    for method in dict.fromkeys(methods):
        pred = _empty_result(mod, cfg, 0, method, "track_error")
        rows.append(result_to_row(
            track_name, track_id, method, 0, pred,
            evaluate_prediction(pred, gt), error_message,
        ))
    return rows


def complete_result_matrix(
    rows: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    methods: List[str],
    mod: Any,
    cfg: Any,
    annotations: Dict[str, Dict[str, Any]],
    missing_as_no_ignition: bool,
) -> List[Dict[str, Any]]:
    """保证每个 track×method 恰好一行，并显式补齐意外缺失结果。"""
    ordered_keys = [
        (str(task["track_name"]), method)
        for task in tasks
        for method in dict.fromkeys(methods)
    ]
    expected_keys = set(ordered_keys)
    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("track_name", "")), str(row.get("method", "")))
        if key not in expected_keys:
            continue
        if key not in unique:
            unique[key] = row
        else:
            message = str(unique[key].get("error_message", ""))
            unique[key]["error_message"] = (
                f"{message}; duplicate_result_discarded".lstrip("; ")
            )
    for task in tasks:
        for method in dict.fromkeys(methods):
            key = (str(task["track_name"]), method)
            if key not in unique:
                unique[key] = build_track_error_rows(
                    task, [method], mod, cfg, annotations,
                    missing_as_no_ignition, "worker_returned_no_result",
                )[0]
    return [unique[key] for key in ordered_keys]


_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(ctx: Dict[str, Any]) -> None:
    global _WORKER_CTX
    mod = load_ignition_module(Path(ctx["ignition_module_path"]))
    _WORKER_CTX = {
        **ctx,
        "mod": mod,
        "cfg": mod.IgnitionGMMConfig(**ctx["cfg_dict"]),
        "error_cases_dir": (
            Path(ctx["error_cases_dir"]) if ctx["error_cases_dir"] else None
        ),
    }


def _worker_process_track(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return process_one_track(
        task["track_name"], task["folder"], _WORKER_CTX["methods"],
        _WORKER_CTX["mod"], _WORKER_CTX["cfg"], _WORKER_CTX["params"],
        _WORKER_CTX["annotations"], _WORKER_CTX["save_error_cases"],
        _WORKER_CTX["error_cases_dir"], _WORKER_CTX["missing_as_no_ignition"],
    )


def extract_track_feature_cache(
    tasks: List[Dict[str, str]],
    mod: Any,
    cfg: Any,
    annotations: Dict[str, Dict[str, Any]],
    missing_as_no_ignition: bool,
    n_workers: int,
    module_path: Path,
) -> List[Dict[str, Any]]:
    def extract(task: Dict[str, str]) -> Dict[str, Any]:
        name, folder = task["track_name"], task["folder"]
        track_id = mod.track_id_from_name(name)
        gt = lookup_annotation(
            annotations, name, track_id,
            missing_as_no_ignition=missing_as_no_ignition,
            missing_as_invalid=False,
        )
        try:
            paths = mod.list_images_in_folder(folder)
            if not paths:
                raise RuntimeError("No images found in track folder")
            df = mod.extract_sequence_features(paths, cfg)
            return {
                "ok": True, "track_name": name, "track_id": track_id,
                "folder": folder, "df": df, "gt": gt, "error": "",
            }
        except Exception as exc:
            return {
                "ok": False, "track_name": name, "track_id": track_id,
                "folder": folder, "df": None, "gt": gt, "error": str(exc),
            }

    total = len(tasks)
    started_at = time.monotonic()
    print(
        f"[STAGE] 开始提取轨迹特征: total={total}, workers={n_workers}",
        flush=True,
    )
    if n_workers == 1:
        cache = []
        ok_count = 0
        error_count = 0
        for completed, task in enumerate(tasks, start=1):
            item = extract(task)
            cache.append(item)
            ok_count += int(bool(item["ok"]))
            error_count += int(not item["ok"])
            print_progress(
                "特征提取", completed, total, started_at,
                ok_count=ok_count, error_count=error_count,
            )
        return cache
    # 特征提取复用检测 worker 的进程安全配置重建方式；每个任务只返回可缓存数据。
    ctx = {
        "ignition_module_path": str(module_path.resolve()),
        "cfg_dict": asdict(cfg),
        "annotations": annotations,
        "missing_as_no_ignition": missing_as_no_ignition,
    }
    cache = []
    ok_count = 0
    error_count = 0
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_extract_worker, initargs=(ctx,)
    ) as executor:
        futures = {executor.submit(_extract_one_track, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                item = future.result()
            except Exception as exc:
                name = task["track_name"]
                track_id = mod.track_id_from_name(name)
                gt = lookup_annotation(
                    annotations,
                    name,
                    track_id,
                    missing_as_no_ignition=missing_as_no_ignition,
                    missing_as_invalid=False,
                )
                item = {
                    "ok": False,
                    "track_name": name,
                    "track_id": track_id,
                    "folder": task["folder"],
                    "df": None,
                    "gt": gt,
                    "error": f"feature_worker_failed: {exc}",
                }
            cache.append(item)
            ok_count += int(bool(item["ok"]))
            error_count += int(not item["ok"])
            print_progress(
                "特征提取", completed, total, started_at,
                ok_count=ok_count, error_count=error_count,
            )
    return cache


_EXTRACT_CTX: Dict[str, Any] = {}


def _init_extract_worker(ctx: Dict[str, Any]) -> None:
    global _EXTRACT_CTX
    mod = load_ignition_module(Path(ctx["ignition_module_path"]))
    _EXTRACT_CTX = {
        **ctx,
        "mod": mod,
        "cfg": mod.IgnitionGMMConfig(**ctx["cfg_dict"]),
    }


def _extract_one_track(task: Dict[str, str]) -> Dict[str, Any]:
    mod, cfg = _EXTRACT_CTX["mod"], _EXTRACT_CTX["cfg"]
    name, folder = task["track_name"], task["folder"]
    track_id = mod.track_id_from_name(name)
    gt = lookup_annotation(
        _EXTRACT_CTX["annotations"], name, track_id,
        missing_as_no_ignition=_EXTRACT_CTX["missing_as_no_ignition"],
        missing_as_invalid=False,
    )
    try:
        paths = mod.list_images_in_folder(folder)
        if not paths:
            raise RuntimeError("No images found in track folder")
        df = mod.extract_sequence_features(paths, cfg)
        return {
            "ok": True, "track_name": name, "track_id": track_id,
            "folder": folder, "df": df, "gt": gt, "error": "",
        }
    except Exception as exc:
        return {
            "ok": False, "track_name": name, "track_id": track_id,
            "folder": folder, "df": None, "gt": gt, "error": str(exc),
        }


def evaluate_baseline_params_on_cache(
    method: str,
    cache: List[Dict[str, Any]],
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
    names: Set[str],
) -> Dict[str, Any]:
    rows = []
    for item in cache:
        if not item["ok"] or item["track_name"] not in names:
            continue
        pred = run_method(method, item["df"], mod, cfg, params)
        rows.append(result_to_row(
            item["track_name"], item["track_id"], method, len(item["df"]), pred,
            evaluate_prediction(pred, item["gt"]),
        ))
    summary = compute_method_summary(pd.DataFrame(rows))
    return summary.iloc[0].to_dict() if len(summary) else {}


def select_baseline_tune_test_tracks(
    cache: List[Dict[str, Any]],
    test_fraction: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    by_category: Dict[str, List[str]] = {
        category: [] for category in sorted(PRIMARY_GT_CATEGORIES)
    }
    for item in cache:
        category = str(item.get("gt", {}).get("gt_category", "")).lower()
        if category in by_category:
            by_category[category].append(str(item["track_name"]))
    all_names = {
        name for names in by_category.values() for name in names
    }
    if test_fraction <= 0:
        return set(all_names), set(all_names)
    if test_fraction >= 1:
        raise ValueError("baseline_test_fraction must be < 1")
    rng = np.random.default_rng(seed)
    tune: Set[str] = set()
    test: Set[str] = set()
    for category in sorted(by_category):
        names = sorted(by_category[category])
        if len(names) < 2:
            tune.update(names)
            continue
        shuffled = list(np.asarray(names)[rng.permutation(len(names))])
        count = min(len(names) - 1, max(1, round(len(names) * test_fraction)))
        test.update(shuffled[:count])
        tune.update(shuffled[count:])
    return tune, test


def validate_experiment_split(
    cache: List[Dict[str, Any]],
    tune_names: Set[str],
    test_names: Set[str],
    *,
    need_tune: bool,
    test_fraction: float,
    evaluate_all_tracks: bool,
) -> None:
    if need_tune and not any(
        item.get("ok") and str(item["track_name"]) in tune_names
        for item in cache
    ):
        raise RuntimeError(
            "No successfully extracted tracks in the baseline tune split"
        )
    if test_fraction > 0 and not test_names and not evaluate_all_tracks:
        raise RuntimeError(
            "Cannot form a non-empty heldout test split; add more data, "
            "use --evaluate-all-tracks, or set --baseline-test-fraction=0"
        )


def resolve_evaluation_protocol(
    need_tune: bool,
    test_fraction: float,
    evaluate_all_tracks: bool,
) -> str:
    if test_fraction <= 0:
        return (
            "oracle_tuned_full_dataset"
            if need_tune
            else "full_dataset_evaluation"
        )
    if evaluate_all_tracks:
        return "full_dataset_evaluation"
    return "heldout_test"


def build_dataset_split(
    cache: List[Dict[str, Any]],
    tune_names: Set[str],
    test_names: Set[str],
) -> pd.DataFrame:
    rows = []
    for item in sorted(cache, key=lambda value: str(value["track_name"])):
        name = str(item["track_name"])
        in_tune = name in tune_names
        in_test = name in test_names
        if in_tune and in_test:
            split = "tune_and_test_oracle"
        elif in_test:
            split = "test"
        elif in_tune:
            split = "tune"
        else:
            split = "excluded"
        rows.append({
            "track_name": name,
            "track_id": item.get("track_id"),
            "gt_category": item.get("gt", {}).get("gt_category", ""),
            "split": split,
            "in_tune": in_tune,
            "in_test": in_test,
        })
    return pd.DataFrame(rows)


def _metric_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    def value(name: str, fallback: float) -> float:
        raw = row.get(name)
        return float(raw) if pd.notna(raw) else fallback
    return (
        value("ignition_f1", -1.0),
        value("inside_interval_acc", -1.0),
        -value("onset_mae", 1e9),
    )


def tune_baseline_hyperparams(
    cache: List[Dict[str, Any]],
    mod: Any,
    cfg: Any,
    params: Dict[str, Any],
    methods: List[str],
    gray_thresholds: List[float],
    change_thresholds: List[float],
    min_persists: List[int],
    tune_names: Set[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    tuned = dict(params)
    records = []
    grids = {
        "gray_p99_threshold": ("gray_threshold", "gray_min_persist", gray_thresholds),
        "multi_feature_change_score": (
            "change_threshold", "change_min_persist", change_thresholds
        ),
    }
    active_methods = [
        method for method in methods if method in TUNABLE_BASELINE_METHODS
    ]
    total_trials = sum(
        len(grids[method][2]) * len(min_persists) for method in active_methods
    )
    completed_trials = 0
    started_at = time.monotonic()
    if total_trials:
        print(
            f"[STAGE] 开始 baseline 网格调参: trials={total_trials}, "
            f"tune_tracks={len(tune_names)}",
            flush=True,
        )
    for method in methods:
        if method not in TUNABLE_BASELINE_METHODS:
            continue
        threshold_key, persist_key, thresholds = grids[method]
        best: Optional[Dict[str, Any]] = None
        for threshold in thresholds:
            for persist in min_persists:
                trial = dict(tuned)
                trial[threshold_key] = float(threshold)
                trial[persist_key] = int(persist)
                metrics = evaluate_baseline_params_on_cache(
                    method, cache, mod, cfg, trial, tune_names
                )
                row = {
                    "method": method, "threshold": threshold,
                    "min_persist": persist, "n_tune_tracks": len(tune_names),
                    "split": "tune",
                    **metrics,
                }
                records.append(row)
                if best is None or _metric_key(row) > _metric_key(best):
                    best = row
                completed_trials += 1
                print_progress(
                    f"baseline 调参 ({method})",
                    completed_trials,
                    total_trials,
                    started_at,
                    every=1,
                    unit="trials",
                )
        if best is not None:
            tuned[threshold_key] = float(best["threshold"])
            tuned[persist_key] = int(best["min_persist"])
    return tuned, pd.DataFrame(records)


def build_base_cfg(args: argparse.Namespace, mod: Any) -> Any:
    """构造与 4.2.3 feature-group 脚本完全一致的 full decision 配置。"""
    cfg = mod.IgnitionGMMConfig(
        gmm_max_states=args.gmm_max_states,
        gmm_min_state_count=args.gmm_min_state_count,
        gmm_min_state_fraction=args.gmm_min_state_fraction,
        gmm_bic_improvement_ratio=args.gmm_bic_improvement_ratio,
        min_state_run=args.min_state_run,
        min_cold_run=args.min_cold_run,
        min_hot_run=args.min_hot_run,
        min_stable_run=args.min_stable_run,
        min_sequence_len=args.min_sequence_len,
        refine_radius=args.refine_radius,
        random_state=args.random_state,
        use_diff_features=True,
        require_all_prior_cold=args.require_all_prior_cold,
        refine_onset_with_change=args.refine_onset_with_change,
        enable_initial_hot_reject=True,
        enable_shape_sequence_reject=True,
        enable_raw_transition_validation=True,
    )
    return cfg


def _float_list(value: str) -> List[float]:
    return [float(item) for item in parse_csv_list(value)]


def _int_list(value: str) -> List[int]:
    return [int(item) for item in parse_csv_list(value)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GMM-RTV 点火 baseline 对比与模块消融实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--roi-root", "--tracks-root",
        dest="roi_root",
        default='inputs/locate_results'
        r"\run_20260703_104159\postprocessed\tracking_output"
        r"\run_20260706_105300\track_crop",
        help=(
            "逐轨 ROI 根目录：track_crop/track_*/<ROI images>。"
            "不依赖原始全幅 image-dir；--tracks-root 仅作兼容别名。"
        ),
    )
    parser.add_argument(
        "--annotation-csv",
        default=str(DEFAULT_ANNOTATION_PATH),
        help=(
            "人工标注。论文默认 effective_trace.xlsx：仅表内且 frame_id 非空为 ignition，"
            "其余当前 ROI tracks 均为 no_ignition；同时保留 category/manual-review/legacy 支持。"
        ),
    )
    parser.add_argument(
        "--reference-summary",
        default=str(DEFAULT_REFERENCE_SUMMARY_PATH),
        help=(
            "仅旧 effective_trace_new 需要的参考 summary；无 status 列时"
            "所有行有效，有 status 列时仅 status=='ok' 行可生成 GT"
        ),
    )
    parser.add_argument("--gt-onset-tolerance", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--ignition-module-path",
        default="ignition_detection_gmm_rtv_new.py",
        help="正式 GMM-RTV 核心检测模块",
    )
    parser.add_argument("--methods", default=None)
    parser.add_argument(
        "--experiment-group",
        choices=[
            "default", "method_comparison", "core_ablation",
            "precheck_ablation", "diagnostic", "all",
        ],
        default="default",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-error-cases", action="store_true")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--gray-threshold", type=float, default=0.5)
    parser.add_argument("--change-threshold", type=float, default=0.25)
    parser.add_argument("--min-persist", type=int, default=3)
    parser.add_argument(
        "--tune-baselines", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--tune-baselines-only", action="store_true")
    parser.add_argument(
        "--gray-threshold-grid",
        default=",".join(str(value) for value in DEFAULT_GRAY_THRESHOLD_GRID),
    )
    parser.add_argument(
        "--change-threshold-grid",
        default=",".join(str(value) for value in DEFAULT_CHANGE_THRESHOLD_GRID),
    )
    parser.add_argument(
        "--min-persist-grid",
        default=",".join(str(value) for value in DEFAULT_MIN_PERSIST_GRID),
    )
    parser.add_argument("--baseline-test-fraction", type=float, default=0.20)
    parser.add_argument(
        "--baseline-tune-val-fraction",
        dest="baseline_test_fraction",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--baseline-tune-seed", type=int, default=0)
    parser.add_argument(
        "--evaluate-all-tracks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "是否在全部 ROI tracks 上报告最终指标。默认：未调参时=True；"
            "启用 baseline 调参时=False（仅 held-out test），避免测试集泄漏。"
        ),
    )
    # 与 4.2.3 及 IgnitionGMMConfig 的论文默认配置显式对齐。
    parser.add_argument("--gmm-max-states", type=int, default=3)
    parser.add_argument("--gmm-min-state-count", type=int, default=2)
    parser.add_argument("--gmm-min-state-fraction", type=float, default=0.08)
    parser.add_argument("--gmm-bic-improvement-ratio", type=float, default=0.03)
    parser.add_argument("--min-state-run", type=int, default=2)
    parser.add_argument("--min-cold-run", type=int, default=3)
    parser.add_argument("--min-hot-run", type=int, default=3)
    parser.add_argument("--min-stable-run", type=int, default=3)
    parser.add_argument("--min-sequence-len", type=int, default=8)
    parser.add_argument(
        "--prior-cold-guard",
        dest="require_all_prior_cold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--refine-onset",
        dest="refine_onset_with_change",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--refine-radius", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    methods = resolve_experiment_methods(args.experiment_group, args.methods)
    if args.tune_baselines_only:
        methods = list(BASELINE_METHODS)
        args.tune_baselines = True
    module_path = Path(args.ignition_module_path)
    if not module_path.is_file():
        module_path = Path(__file__).resolve().parent / module_path
    module_path = module_path.resolve()
    print(f"[INFO] ignition_module_path={module_path}")
    mod = load_ignition_module(module_path)
    cfg = build_base_cfg(args, mod)
    validate_method_contrasts(methods, cfg)
    gt_metadata: Dict[str, Any] = {}
    annotations, annotation_format = load_annotations(
        Path(args.annotation_csv) if args.annotation_csv else None,
        reference_summary=(
            Path(args.reference_summary) if args.reference_summary else None
        ),
        onset_tolerance=args.gt_onset_tolerance,
        metadata_out=gt_metadata,
    )
    missing_as_no_ignition = annotation_format in {
        "effective_trace", "effective_trace_new"
    }
    tracks = mod.list_track_folders(str(Path(args.roi_root)))
    if not tracks:
        raise RuntimeError(f"No track folders under ROI root: {args.roi_root}")
    gt_population = audit_effective_trace_population(
        tracks, annotations, mod, annotation_format
    )
    if annotation_format == "manual_review_gt":
        gt_metadata = framework.build_manual_review_evaluation_metadata(
            tracks,
            annotations,
            gt_metadata,
        )
        print(
            "[WARN] manual_review_gt: tracks absent from new_gt.xlsx are "
            "evaluated as no_ignition"
        )
    tasks = [{"track_name": name, "folder": folder} for name, folder in tracks]
    output_dir = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    error_dir = output_dir / "error_cases" if args.save_error_cases else None
    workers = resolve_workers(args.workers)
    params = {
        "gray_threshold": args.gray_threshold,
        "change_threshold": args.change_threshold,
        "gray_min_persist": args.min_persist,
        "change_min_persist": args.min_persist,
    }
    # 默认论文协议：baseline 不在最终评价集上自动寻优；所有方法均在同一完整 ROI population 上评价。
    # 若显式开启 --tune-baselines，则默认切换到 held-out test，除非用户显式 --evaluate-all-tracks。
    if args.evaluate_all_tracks is None:
        args.evaluate_all_tracks = not bool(args.tune_baselines)
    print(f"[INFO] methods={methods}")
    print(f"[INFO] roi_root={args.roi_root}")
    print(f"[INFO] tracks={len(tasks)}, workers={workers}, GT={annotation_format}")
    print(
        f"[INFO] tune_baselines={bool(args.tune_baselines)}, "
        f"evaluate_all_tracks={bool(args.evaluate_all_tracks)}"
    )
    if args.tune_baselines and args.evaluate_all_tracks and args.baseline_test_fraction > 0:
        warnings.warn(
            "Baselines are tuned on a subset but final metrics are requested on the full dataset, "
            "which includes tuning tracks. For paper-grade held-out baseline comparison use "
            "--no-evaluate-all-tracks, or keep the default --no-tune-baselines for direct "
            "alignment with 4.2.3.",
            UserWarning,
        )

    if args.baseline_test_fraction < 0 or args.baseline_test_fraction >= 1:
        raise ValueError("baseline_test_fraction must be in [0, 1)")
    cache = extract_track_feature_cache(
        tasks, mod, cfg, annotations, missing_as_no_ignition,
        workers, module_path,
    )
    n_cache_ok = sum(bool(item.get("ok")) for item in cache)
    print(
        f"[STAGE] 特征提取完成: ok={n_cache_ok}, "
        f"error={len(cache) - n_cache_ok}",
        flush=True,
    )
    first_ok = next((item for item in cache if item.get("ok") and item.get("df") is not None), None)
    if first_ok is None:
        raise RuntimeError("No ROI trajectory was successfully converted to features")
    full_feature_cols = validate_full_feature_alignment(first_ok["df"], cfg, mod)
    (output_dir / "gt_population.json").write_text(
        json.dumps(to_jsonable(gt_population), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tune_names, test_names = select_baseline_tune_test_tracks(
        cache, args.baseline_test_fraction, args.baseline_tune_seed
    )
    print(
        f"[STAGE] 数据集划分完成: tune={len(tune_names)}, "
        f"test={len(test_names)}",
        flush=True,
    )
    need_tune = args.tune_baselines and any(
        method in BASELINE_METHODS for method in methods
    )
    evaluation_protocol = resolve_evaluation_protocol(
        need_tune,
        args.baseline_test_fraction,
        args.evaluate_all_tracks,
    )
    validate_experiment_split(
        cache,
        tune_names,
        test_names,
        need_tune=need_tune,
        test_fraction=args.baseline_test_fraction,
        evaluate_all_tracks=args.evaluate_all_tracks,
    )
    if need_tune:
        if not annotations:
            raise RuntimeError("Baseline tuning requires annotations")
        tune_methods = [
            method for method in TUNABLE_BASELINE_METHODS
            if method in methods
            or (
                method == "multi_feature_change_score"
                and "multi_feature_change_score_rtv" in methods
            )
        ]
        params, tune_df = tune_baseline_hyperparams(
            cache, mod, cfg, params, tune_methods,
            _float_list(args.gray_threshold_grid),
            _float_list(args.change_threshold_grid),
            _int_list(args.min_persist_grid),
            tune_names,
        )
        save_table(tune_df, output_dir / "baseline_tune_results")
    best_params_row = {
        **params,
        "n_tune_tracks": len(tune_names),
        "n_test_tracks": len(test_names),
        "baseline_test_fraction": args.baseline_test_fraction,
        "baseline_tune_seed": args.baseline_tune_seed,
        "evaluation_protocol": evaluation_protocol,
    }
    save_table(pd.DataFrame([best_params_row]), output_dir / "baseline_best_params")

    split_df = build_dataset_split(cache, tune_names, test_names)
    split_path = output_dir / "dataset_split.csv"
    split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    summary_splits: List[Tuple[str, Set[str]]] = [("tune", tune_names)]
    if test_names and test_names != tune_names:
        summary_splits.append(("test", test_names))
    if args.evaluate_all_tracks or args.baseline_test_fraction <= 0:
        summary_splits.append((
            "full",
            {str(item["track_name"]) for item in cache if item.get("ok")},
        ))
    baseline_summary_rows = []
    total_summary_jobs = len(summary_splits) * len(BASELINE_METHODS)
    completed_summary_jobs = 0
    summary_started_at = time.monotonic()
    print(
        f"[STAGE] 开始计算 baseline 分组汇总: jobs={total_summary_jobs}",
        flush=True,
    )
    for split_name, names in summary_splits:
        for method in BASELINE_METHODS:
            metrics = evaluate_baseline_params_on_cache(
                method, cache, mod, cfg, params, names
            )
            baseline_summary_rows.append({
                "method": method,
                "evaluation_split": split_name,
                **metrics,
            })
            completed_summary_jobs += 1
            print_progress(
                f"baseline 汇总 ({split_name}/{method})",
                completed_summary_jobs,
                total_summary_jobs,
                summary_started_at,
                every=1,
                unit="jobs",
            )
    save_table(
        pd.DataFrame(baseline_summary_rows),
        output_dir / "baseline_tune_summary",
    )

    all_names = {str(item["track_name"]) for item in cache}
    if args.evaluate_all_tracks or args.baseline_test_fraction <= 0:
        evaluation_names = all_names
    else:
        shape_specialty_names = {
            str(item["track_name"]) for item in cache
            if str(item.get("gt", {}).get("gt_category", "")).lower()
            in SHAPE_INVALID_GT_CATEGORIES
        }
        ambiguous_names = {
            str(item["track_name"]) for item in cache
            if args.include_ambiguous
            and str(item.get("gt", {}).get("gt_category", "")).lower()
            in OPTIONAL_GT_CATEGORIES
        }
        evaluation_names = set(test_names) | shape_specialty_names | ambiguous_names
    selected_tasks = [
        task for task in tasks if str(task["track_name"]) in evaluation_names
    ]
    selected_cache = {
        str(item["track_name"]): item for item in cache
        if str(item["track_name"]) in evaluation_names
    }

    config_payload = {
        "module": str(module_path),
        "roi_root": str(Path(args.roi_root).resolve()),
        "config": asdict(cfg),
        "full_gmm_feature_cols": full_feature_cols,
        "full_gmm_feature_set_aligned_with_4_2_3": True,
        "methods": methods,
        "baseline_best_params": best_params_row,
        "dataset_split": {
            "n_tune": len(tune_names),
            "n_test": len(test_names),
            "n_excluded": int((split_df["split"] == "excluded").sum()),
            "path": str(split_path.resolve()),
        },
        "seed": args.baseline_tune_seed,
        "gt_format": annotation_format,
        "annotation_path": (
            str(Path(args.annotation_csv).resolve()) if args.annotation_csv else None
        ),
        "reference_summary_path": (
            str(Path(args.reference_summary).resolve())
            if args.reference_summary and annotation_format == "effective_trace_new"
            else None
        ),
        "evaluation_protocol": evaluation_protocol,
        "gt_uses_reference_predictions": annotation_format == "effective_trace_new",
        "gt_protocol": gt_metadata,
    }
    write_experiment_config(output_dir / "experiment_config.json", config_payload)
    if args.tune_baselines_only:
        return

    rows: List[Dict[str, Any]] = []
    detection_started_at = time.monotonic()
    detection_ok = 0
    detection_error = 0
    print(
        f"[STAGE] 开始正式消融检测: tracks={len(selected_tasks)}, "
        f"methods={len(methods)}",
        flush=True,
    )
    for completed, task in enumerate(selected_tasks, start=1):
        item = selected_cache[str(task["track_name"])]
        if item["ok"]:
            rows.extend(process_one_track(
                item["track_name"], item["folder"], methods, mod, cfg, params,
                annotations, args.save_error_cases, error_dir,
                missing_as_no_ignition, item["df"],
            ))
            detection_ok += 1
        else:
            rows.extend(build_track_error_rows(
                item, methods, mod, cfg, annotations,
                missing_as_no_ignition, item["error"],
            ))
            detection_error += 1
        print_progress(
            "正式消融检测",
            completed,
            len(selected_tasks),
            detection_started_at,
            ok_count=detection_ok,
            error_count=detection_error,
            every=10,
        )

    print("[STAGE] 检测完成，正在生成对比表并保存结果", flush=True)
    rows = complete_result_matrix(
        rows, selected_tasks, methods, mod, cfg, annotations, missing_as_no_ignition
    )
    per_track = pd.DataFrame(rows)
    if not per_track.empty:
        per_track = per_track.sort_values(["track_id", "method"]).reset_index(drop=True)
    comparison = build_method_track_comparison_table(per_track, methods)
    summary = compute_method_summary(per_track, args.include_ambiguous)
    paired_common_tp = build_pairwise_common_tp_summary(
        per_track, args.include_ambiguous
    )
    precheck_contrast = build_precheck_ablation_contrast(summary)
    save_table(per_track, output_dir / "per_track_results")
    save_table(comparison, output_dir / "method_track_comparison")
    save_table(summary, output_dir / "method_summary")
    if not paired_common_tp.empty:
        save_table(paired_common_tp, output_dir / "paired_common_tp_summary")
    if not precheck_contrast.empty:
        save_table(precheck_contrast, output_dir / "precheck_ablation_contrast")
    print("\n=== Method Summary ===")
    print(summary.to_string(index=False) if len(summary) else "(empty)")
    if not paired_common_tp.empty:
        print("\n=== Pairwise Common-TP Onset Comparison ===")
        print(paired_common_tp.to_string(index=False))
    if not precheck_contrast.empty:
        print("\n=== Precheck Ablation Contrast ===")
        print(precheck_contrast.to_string(index=False))


if __name__ == "__main__":
    main()
