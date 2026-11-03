#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GMM-RTV 点火检测的特征组消融实验（ROI-crop 输入版）。

本脚本直接读取 tracking_output/.../track_crop/track_* 中已经保存的逐轨 ROI 图像，
不依赖原始全幅图像目录（image-dir）。每条轨迹只从 ROI 图像提取一次特征，之后
固定 GMM+BIC、轨迹级预检查、状态平滑和状态解释逻辑，仅改变 GMM 的输入特征组。

论文 4.2.3 对应的默认特征组采用严格的 progressive augmentation：
gray -> +background/intensity -> +radiometric -> +color -> +morphology
-> +temporal differences。最后一步保证仅增加 d1_/d2_ 时间差分特征。

两个默认协议分别为 gmm_without_rtv 与 final_detection；两者使用完全相同的
GMM/时序判定链，唯一差异是 final_detection 启用 raw transition validation (RTV)。
旧名称 segmentation_only 仍作为兼容别名接受，但输出会规范为 gmm_without_rtv。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import run_ignition_ablation_experiments_revised as framework


DEFAULT_FEATURE_GROUPS = (
    "gray_only,gray_bg,radiometric_core,radiometric_color,"
    "radiometric_color_morph,full_state_plus_diff"
)
DEFAULT_PROTOCOLS = "gmm_without_rtv,final_detection"
REFERENCE_FEATURE_GROUP = "full_state_plus_diff"

# 默认论文实验采用 6 级严格递进特征：前 5 级为绝对状态特征的逐级叠加，
# 第 6 级仅在完整绝对状态特征上增加 d1_/d2_ 时间差分特征。
PAPER_FEATURE_GROUPS = (
    "gray_only",
    "gray_bg",
    "radiometric_core",
    "radiometric_color",
    "radiometric_color_morph",
    "full_state_plus_diff",
)

ALL_FEATURE_GROUPS = (
    "gray_only,gray_bg,radiometric_core,radiometric_color,"
    "radiometric_color_morph,full_state_no_diff,full_state_plus_diff,"
    "brightness_only,brightness_bg,brightness_bg_energy,"
    "without_color,without_diff,no_morphology,full_feature_matrix"
)

NO_IGNITION_STATUSES = {
    "no_ignition",
    "no_ignition_weak_transition",
    "single_state",
}
PRIMARY_GT_CATEGORIES = {"ignition", "no_ignition", "already_hot"}
EXCLUDED_GT_CATEGORIES = {"invalid", "suspicious_merge", "defocused", "ambiguous"}

# 论文主实验的绝对状态特征采用同一条累积链，避免旧脚本中
# radiometric_color_morph -> full_state_no_diff 时删/换特征导致无法严格归因。
PAPER_ABSOLUTE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "gray_only": [
        "gray_p99",
    ],
    "gray_bg": [
        "gray_p99",
        "gray_p95",
        "corr_p95", "corr_p99", "corr_max",
    ],
    "radiometric_core": [
        "gray_p99",
        "gray_p95",
        "corr_p95", "corr_p99", "corr_max",
        "bright_energy_mean", "bright_area_ratio", "bright_mean",
        "hsv_v_p95", "hsv_v_p99",
    ],
    "radiometric_color": [
        "gray_p99",
        "gray_p95",
        "corr_p95", "corr_p99", "corr_max",
        "bright_energy_mean", "bright_area_ratio", "bright_mean",
        "hsv_v_p95", "hsv_v_p99",
        "red_ratio", "rg_ratio",
    ],
    "radiometric_color_morph": [
        "gray_p99",
        "gray_p95",
        "corr_p95", "corr_p99", "corr_max",
        "bright_energy_mean", "bright_area_ratio", "bright_mean",
        "hsv_v_p95", "hsv_v_p99",
        "red_ratio", "rg_ratio",
        "largest_area_ratio",
    ],
}

# 保留旧实验组供兼容/补充实验使用；不属于论文主 progressive chain。
LEGACY_EXPLICIT_FEATURE_GROUPS: Dict[str, List[str]] = {
    "brightness_only": [
        "gray_p90", "gray_p95", "gray_p99", "gray_max", "hsv_v_p95", "hsv_v_p99",
    ],
    "brightness_bg": [
        "gray_p90", "gray_p95", "gray_p99", "gray_max",
        "corr_p90", "corr_p95", "corr_p99", "corr_max",
        "hsv_v_p95", "hsv_v_p99",
    ],
    "brightness_bg_energy": [
        "gray_p90", "gray_p95", "gray_p99", "gray_max",
        "corr_p90", "corr_p95", "corr_p99", "corr_max",
        "bright_area_ratio", "bright_energy_mean", "bright_mean", "bright_max",
        "hsv_v_p95", "hsv_v_p99", "largest_area_ratio",
    ],
}

COLOR_RATIO_COLS = {
    "r_mean", "g_mean", "b_mean", "rg_ratio", "rb_ratio", "red_ratio",
    "hsv_s_mean",
}

MORPHOLOGY_BASE_COLS = {"component_count", "largest_area_ratio", "centroid_dist_norm"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def normalize_protocol_name(protocol: str) -> str:
    """将旧名 segmentation_only 规范为更准确的 gmm_without_rtv。"""
    protocol = str(protocol).strip()
    if protocol == "segmentation_only":
        return "gmm_without_rtv"
    return protocol


def _natural_sort_key(value: str) -> List[Any]:
    """对 track_2 / track_10、frame_2 / frame_10 做自然数排序。"""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def list_track_folders_from_roi_root(roi_root: Path) -> List[Tuple[str, str]]:
    """直接枚举 track_crop/track_*，不依赖 ignition module 中的 image-dir 逻辑。"""
    roi_root = Path(roi_root)
    if not roi_root.is_dir():
        raise FileNotFoundError(f"ROI root not found: {roi_root}")
    folders = [p for p in roi_root.iterdir() if p.is_dir() and p.name.lower().startswith("track_")]
    folders.sort(key=lambda p: _natural_sort_key(p.name))
    return [(p.name, str(p)) for p in folders]


def list_roi_images_in_folder(folder: str) -> List[str]:
    """读取单条轨迹文件夹中已裁剪的 ROI 图像，并按文件名自然排序。"""
    root = Path(folder)
    images = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort(key=lambda p: _natural_sort_key(p.name))
    return [str(p) for p in images]


def validate_experiment_matrix(
    feature_groups: List[str],
    protocols: List[str],
) -> None:
    if not feature_groups:
        raise ValueError("feature_groups must not be empty")
    if not protocols:
        raise ValueError("protocols must not be empty")
    duplicate_groups = sorted({
        item for item in feature_groups if feature_groups.count(item) > 1
    })
    if duplicate_groups:
        raise ValueError(f"Duplicate feature groups: {duplicate_groups}")
    duplicate_protocols = sorted({
        item for item in protocols if protocols.count(item) > 1
    })
    if duplicate_protocols:
        raise ValueError(f"Duplicate protocols: {duplicate_protocols}")

    valid_groups = set(parse_csv_list(ALL_FEATURE_GROUPS))
    unknown = [item for item in feature_groups if item not in valid_groups]
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}; valid={sorted(valid_groups)}")
    invalid_protocols = [
        item for item in protocols
        if normalize_protocol_name(item) not in ("gmm_without_rtv", "final_detection")
    ]
    if invalid_protocols:
        raise ValueError(f"Unknown protocols: {invalid_protocols}")


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
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    return obj


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


def make_default_output_dir(base: str = "feature_group_ablation_gmm_rtv_results") -> Path:
    return Path(base) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    import os
    return max(1, min((os.cpu_count() or 4) - 1, 16))


def load_ignition_module(module_path: Path):
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Ignition module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("ignition_fg_gmm_rtv_ablation", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ignition module spec from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = [
        "IgnitionGMMConfig",
        "track_id_from_name",
        "extract_sequence_features",
        "get_gmm_feature_columns",
        "fit_gmm_models_bic",
        "select_gmm_with_occupancy_validation",
        "compute_state_thermal_scores",
        "build_state_rank_mapping",
        "smooth_state_sequence_by_run_length",
        "compute_raw_thermal_change_score",
        "interpret_gmm_state_sequence_for_ignition",
        "compute_hot_score",
        "check_roi_shape_sequence_quality",
        "is_initially_hot_sequence",
        "_make_precheck_result",
        "_attach_shape_fields_to_result",
        "normalize_to_01",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise AttributeError(f"Ignition module missing required symbol: {name}")
    fields = getattr(mod.IgnitionGMMConfig, "__dataclass_fields__", {})
    if "enable_raw_transition_validation" not in fields:
        raise RuntimeError("IgnitionGMMConfig lacks raw transition validation configuration")
    if "require_all_prior_cold" not in fields:
        raise RuntimeError("IgnitionGMMConfig lacks prior-cold guard configuration")
    if "refine_onset_with_change" not in fields:
        raise RuntimeError("IgnitionGMMConfig lacks onset refinement configuration")
    return mod


# 非 effective_trace_new 格式继续复用既有框架；reference summary 的可选 status
# 规则与 tracking 新脚本保持一致。
lookup_annotation = framework.lookup_annotation


def build_annotations_from_effective_trace_new(
    listed_no_ignition_ids: List[int],
    reference_summary: Path,
    onset_tolerance: int = 2,
) -> Dict[str, Dict[str, Any]]:
    if onset_tolerance < 0:
        raise ValueError(f"onset_tolerance must be >= 0, got {onset_tolerance}")
    ref = framework.read_reference_ignition_summary(reference_summary)
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
    annotation_table = framework.read_annotation_table(annotation_path)
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
    return (
        build_annotations_from_effective_trace_new(
            listed_ids,
            Path(reference_summary),
            onset_tolerance=onset_tolerance,
        ),
        "effective_trace_new",
    )


def _is_diff_col(name: str) -> bool:
    return name.startswith("d1_") or name.startswith("d2_")


def _is_morphology_col(name: str) -> bool:
    return any(
        name == base
        or name.startswith(f"d1_{base}")
        or name.startswith(f"d2_{base}")
        or name.startswith(f"roll3_{base}")
        for base in MORPHOLOGY_BASE_COLS
    )


def _is_color_ratio_col(name: str) -> bool:
    base = name[3:] if name.startswith(("d1_", "d2_")) else name
    return base in COLOR_RATIO_COLS


def _require_feature_columns(
    df: pd.DataFrame,
    requested: List[str],
    feature_group: str,
) -> List[str]:
    missing = [c for c in requested if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature_group={feature_group}: ROI feature extraction is missing required columns: {missing}. "
            "The paper ablation uses strict controlled feature sets, so missing columns are not silently skipped."
        )
    return list(requested)


def get_feature_columns_for_group(
    df: pd.DataFrame,
    full_feature_cols: List[str],
    feature_group: str,
) -> List[str]:
    """按 4.2.3 的真实控制变量逻辑构造 GMM 输入特征。

    - gray_only -> gray_bg -> radiometric_core -> radiometric_color
      -> radiometric_color_morph 为严格累积的 absolute-state features；
    - full_state_no_diff 是 radiometric_color_morph 的兼容别名；
    - full_state_plus_diff 只在完整 absolute-state features 上追加 full feature
      matrix 中的 d1_/d2_ temporal-difference columns。
    """
    feature_group = str(feature_group)

    if feature_group in PAPER_ABSOLUTE_FEATURE_GROUPS:
        return _require_feature_columns(
            df, PAPER_ABSOLUTE_FEATURE_GROUPS[feature_group], feature_group
        )

    if feature_group in ("full_state_no_diff", "without_diff"):
        return _require_feature_columns(
            df, PAPER_ABSOLUTE_FEATURE_GROUPS["radiometric_color_morph"], feature_group
        )

    if feature_group == "full_state_plus_diff":
        base = _require_feature_columns(
            df, PAPER_ABSOLUTE_FEATURE_GROUPS["radiometric_color_morph"], feature_group
        )
        diff_cols = [
            c for c in full_feature_cols
            if c in df.columns and _is_diff_col(c) and c not in base
        ]
        if not diff_cols:
            raise ValueError(
                "full_state_plus_diff requested, but no d1_/d2_ temporal-difference "
                "features were returned by get_gmm_feature_columns()."
            )
        return base + diff_cols

    if feature_group == "full_feature_matrix":
        cols = [c for c in full_feature_cols if c in df.columns]
        if not cols:
            raise ValueError("full_feature_matrix is empty")
        return cols

    if feature_group in LEGACY_EXPLICIT_FEATURE_GROUPS:
        requested = LEGACY_EXPLICIT_FEATURE_GROUPS[feature_group]
        missing = [c for c in requested if c not in df.columns]
        if missing:
            warnings.warn(
                f"legacy feature_group={feature_group}: missing columns skipped: {missing[:8]}"
                + ("..." if len(missing) > 8 else ""),
                stacklevel=2,
            )
        return [c for c in requested if c in df.columns]

    if feature_group == "without_color":
        return [c for c in full_feature_cols if c in df.columns and not _is_color_ratio_col(c)]

    if feature_group == "no_morphology":
        return [c for c in full_feature_cols if c in df.columns and not _is_morphology_col(c)]

    raise ValueError(f"Unknown feature_group: {feature_group}")


def validate_progressive_feature_chain(
    df: pd.DataFrame,
    cfg: Any,
    ignition_mod: Any,
) -> Dict[str, List[str]]:
    """启动时验证论文默认特征链是严格嵌套的，并返回实际列名。"""
    full_cols = ignition_mod.get_gmm_feature_columns(df, cfg)
    groups = list(PAPER_FEATURE_GROUPS)
    resolved = {g: get_feature_columns_for_group(df, full_cols, g) for g in groups}
    for prev, curr in zip(groups[:-1], groups[1:]):
        if not set(resolved[prev]).issubset(set(resolved[curr])):
            raise RuntimeError(
                f"Progressive ablation violated: {prev} is not a subset of {curr}. "
                f"missing={sorted(set(resolved[prev]) - set(resolved[curr]))}"
            )
    final_base = resolved["radiometric_color_morph"]
    final_plus = resolved["full_state_plus_diff"]
    added = [c for c in final_plus if c not in final_base]
    if not added or not all(_is_diff_col(c) for c in added):
        raise RuntimeError(
            "The final ablation step must add temporal-difference features only; "
            f"resolved added columns={added}"
        )
    return resolved


def robust_normalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """按核心实现逐列做 median/MAD 稳健标准化。"""
    X = df[feature_cols].values.astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = float(np.nanmedian(col))
        col = np.where(np.isnan(col), med, col)
        mad = float(np.median(np.abs(col - med))) + 1e-6
        X[:, j] = np.clip((col - med) / mad, -6.0, 6.0)
    return X


def build_feature_matrix_for_group(
    df: pd.DataFrame,
    cfg: Any,
    feature_group: str,
    ignition_mod: Any,
) -> Tuple[np.ndarray, List[str]]:
    full_cols = ignition_mod.get_gmm_feature_columns(df, cfg)
    feature_cols = get_feature_columns_for_group(df, full_cols, feature_group)
    if not feature_cols:
        return np.zeros((len(df), 0), dtype=np.float32), []
    return robust_normalize_feature_matrix(df, feature_cols), feature_cols


def _empty_gmm_result(
    ignition_mod: Any,
    df: pd.DataFrame,
    cfg: Any,
    feature_group: str,
    protocol: str,
    status: str = "ambiguous",
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    n = len(df)
    no_idx = cfg.no_ignition_index
    return {
        "has_ignition": False,
        "ignition_status": status,
        "ignition_idx_onset": no_idx,
        "ignition_idx_confirmed": no_idx,
        "ignition_idx_candidate": no_idx,
        "stable_combustion_idx": no_idx,
        "ignition_confidence": 0.0,
        "ignition_amplitude": 0.0,
        "ignition_change_peak": 0.0,
        "gmm_n_states": 0,
        "gmm_state_sequence": [],
        "gmm_state_thermal_scores": {},
        "gmm_bic_by_k": {},
        "gmm_feature_cols": feature_cols or [],
        "hot_score": ignition_mod.compute_hot_score(df) if n else np.zeros(0, dtype=np.float32),
        "change_score": np.zeros(n, dtype=np.float32),
        "precheck_diagnostics": {},
        "feature_group": feature_group,
        "protocol": protocol,
        "ignition_method": f"gmm_rtv_{feature_group}_{protocol}",
        "raw_transition_validation_enabled": normalize_protocol_name(protocol) == "final_detection",
    }


def detect_gmm_rtv_with_feature_group(
    df: pd.DataFrame,
    ignition_mod: Any,
    cfg: Any,
    feature_group: str,
    protocol: str,
) -> Dict[str, Any]:
    """执行核心 GMM-RTV 流程，仅替换用于聚类的特征矩阵。"""
    protocol = normalize_protocol_name(protocol)
    if protocol not in ("gmm_without_rtv", "final_detection"):
        raise ValueError(f"Unknown protocol: {protocol}")
    local_cfg = replace(
        cfg,
        enable_raw_transition_validation=(protocol == "final_detection"),
    )
    n = len(df)
    if n < local_cfg.min_sequence_len:
        return _empty_gmm_result(
            ignition_mod, df, local_cfg, feature_group, protocol, "ambiguous"
        )

    shape_reject, shape_metrics = ignition_mod.check_roi_shape_sequence_quality(df, local_cfg)
    if shape_reject:
        result = ignition_mod._make_precheck_result(
            df,
            local_cfg,
            status="rejected_shape_sequence",
            diagnostics={"shape": shape_metrics},
        )
        result["feature_group"] = feature_group
        result["protocol"] = protocol
        result["ignition_method"] = f"gmm_rtv_{feature_group}_{protocol}"
        return result

    initially_hot, initial_hot_metrics = ignition_mod.is_initially_hot_sequence(df, local_cfg)
    if initially_hot:
        result = ignition_mod._make_precheck_result(
            df,
            local_cfg,
            status="already_hot_at_sequence_start",
            diagnostics={"shape": shape_metrics, "initial_hot": initial_hot_metrics},
        )
        result["feature_group"] = feature_group
        result["protocol"] = protocol
        result["ignition_method"] = f"gmm_rtv_{feature_group}_{protocol}"
        return result

    Xn, feature_cols = build_feature_matrix_for_group(
        df, local_cfg, feature_group, ignition_mod
    )
    if not feature_cols:
        raise ValueError("empty_feature_group")

    gmm, selected_k, gmm_info, all_models = ignition_mod.fit_gmm_models_bic(
        Xn,
        max_states=local_cfg.gmm_max_states,
        covariance_type=local_cfg.gmm_covariance_type,
        reg_covar=local_cfg.gmm_reg_covar,
        n_init=local_cfg.gmm_n_init,
        bic_improvement_ratio=local_cfg.gmm_bic_improvement_ratio,
        random_state=local_cfg.random_state,
    )
    gmm, selected_k, gmm_info = ignition_mod.select_gmm_with_occupancy_validation(
        Xn,
        all_models,
        gmm_info,
        min_state_count=local_cfg.gmm_min_state_count,
        min_state_fraction=local_cfg.gmm_min_state_fraction,
    )
    change_score = ignition_mod.compute_raw_thermal_change_score(df, 3)
    hot_score = ignition_mod.compute_hot_score(df)
    thermal_scores = ignition_mod.compute_state_thermal_scores(gmm, feature_cols)

    if selected_k == 1:
        states_rank = np.zeros(n, dtype=np.int32)
        result = ignition_mod.interpret_gmm_state_sequence_for_ignition(
            states_rank, df, change_score, local_cfg, gmm_info, thermal_scores
        )
    else:
        label_to_rank, _ = ignition_mod.build_state_rank_mapping(thermal_scores)
        labels = gmm.predict(Xn)
        states_rank = np.asarray(
            [label_to_rank[int(label)] for label in labels],
            dtype=np.int32,
        )
        states_rank = ignition_mod.smooth_state_sequence_by_run_length(
            states_rank, local_cfg.min_state_run
        )
        result = ignition_mod.interpret_gmm_state_sequence_for_ignition(
            states_rank, df, change_score, local_cfg, gmm_info, thermal_scores
        )
        result["label_to_rank"] = label_to_rank

    result.update({
        "gmm_state_sequence": states_rank.tolist(),
        "gmm_feature_cols": feature_cols,
        "hot_score": hot_score,
        "change_score": change_score,
        "precheck_diagnostics": {
            "shape": shape_metrics,
            "initial_hot": initial_hot_metrics,
        },
        "feature_group": feature_group,
        "protocol": protocol,
        "ignition_method": f"gmm_rtv_{feature_group}_{protocol}",
        "raw_transition_validation_enabled": bool(
            local_cfg.enable_raw_transition_validation
        ),
    })
    ignition_mod._attach_shape_fields_to_result(result, shape_metrics)
    return result


def evaluate_prediction(pred: Dict[str, Any], gt: Dict[str, Any]) -> Dict[str, Any]:
    gt_cat = str(gt.get("gt_category", "")).lower()
    gt_has = bool(gt.get("gt_has_ignition", False))
    gt_onset = int(gt.get("gt_onset_idx", -1))
    gt_start = int(gt.get("gt_onset_start", -1))
    gt_end = int(gt.get("gt_onset_end", -1))
    pred_has = bool(pred.get("has_ignition", False))
    pred_status = str(pred.get("ignition_status", ""))
    pred_onset = int(pred.get("ignition_idx_onset", -1))

    onset_abs_error = np.nan
    onset_inside_interval = np.nan
    is_correct_presence: Any = np.nan
    is_false_ignition = False
    is_missed_ignition = False
    is_already_hot_correct = False

    if gt_cat == "ignition":
        is_correct_presence = pred_has
        is_missed_ignition = not pred_has
        if pred_has and gt_onset >= 0 and pred_onset >= 0:
            onset_abs_error = abs(pred_onset - gt_onset)
            if gt_start >= 0 and gt_end >= gt_start:
                onset_inside_interval = gt_start <= pred_onset <= gt_end
    elif gt_cat == "no_ignition":
        is_correct_presence = (not pred_has) or pred_status in NO_IGNITION_STATUSES
        is_false_ignition = pred_has and pred_status not in NO_IGNITION_STATUSES
    elif gt_cat == "already_hot":
        is_already_hot_correct = pred_status in {
            "already_hot", "already_hot_at_sequence_start"
        }
        is_correct_presence = is_already_hot_correct

    return {
        "gt_category": gt_cat,
        "gt_has_ignition": gt_has,
        "gt_onset_idx": gt_onset,
        "gt_onset_start": gt_start,
        "gt_onset_end": gt_end,
        "onset_abs_error": onset_abs_error,
        "onset_inside_interval": onset_inside_interval,
        "is_correct_presence": is_correct_presence,
        "is_false_ignition": is_false_ignition,
        "is_missed_ignition": is_missed_ignition,
        "is_already_hot_correct": is_already_hot_correct,
    }


def result_to_row(
    track_name: str,
    track_id: Optional[int],
    protocol: str,
    feature_group: str,
    n_frames: int,
    feature_cols: List[str],
    pred: Dict[str, Any],
    eval_fields: Dict[str, Any],
    error_message: str = "",
) -> Dict[str, Any]:
    states = pred.get("gmm_state_sequence", []) or []
    return to_py({
        "track_name": track_name,
        "track_id": track_id,
        "protocol": protocol,
        "feature_group": feature_group,
        "n_frames": n_frames,
        "n_features": len(feature_cols),
        "feature_cols": ";".join(feature_cols),
        "pred_has_ignition": bool(pred.get("has_ignition", False)),
        "pred_ignition_status": str(pred.get("ignition_status", "")),
        "pred_onset_idx": int(pred.get("ignition_idx_onset", -1)),
        "pred_confirmed_idx": int(pred.get("ignition_idx_confirmed", -1)),
        "pred_stable_idx": int(pred.get("stable_combustion_idx", -1)),
        "ignition_confidence": float(pred.get("ignition_confidence", 0.0)),
        "ignition_amplitude": float(pred.get("ignition_amplitude", 0.0)),
        "ignition_change_peak": float(pred.get("ignition_change_peak", 0.0)),
        "gmm_n_states": int(pred.get("gmm_n_states", 0)),
        "gmm_state_sequence_str": ",".join(str(int(s)) for s in states),
        "gmm_feature_cols": ";".join(pred.get("gmm_feature_cols", []) or []),
        "raw_transition_validation_enabled": bool(
            pred.get("raw_transition_validation_enabled", False)
        ),
        **eval_fields,
        "error_message": error_message,
    })


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else float("nan")


def count_statuses(series: pd.Series, statuses: List[str]) -> Dict[str, int]:
    values = series.astype(str) if len(series) else pd.Series(dtype=str)
    return {f"n_status_{status}": int((values == status).sum()) for status in statuses}


def _track_identity_series(df: pd.DataFrame) -> pd.Series:
    if "track_name" in df.columns:
        names = df["track_name"].astype(str).str.strip()
        valid = names.ne("") & names.str.lower().ne("nan") & names.str.lower().ne("none")
        if bool(valid.any()):
            return names
    if "track_id" in df.columns:
        return df["track_id"].map(lambda x: str(int(x)) if pd.notna(x) else "")
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _onset_metrics_from_hits(
    hits: pd.DataFrame,
    *,
    prefix: str = "",
    include_median: bool = False,
) -> Dict[str, Any]:
    n_key = f"n_onset_eval{prefix}"
    mae_key = f"onset_mae{prefix}"
    w1_key = f"within_1_frame_acc{prefix}"
    w3_key = f"within_3_frame_acc{prefix}"
    inside_key = f"inside_interval_acc{prefix}"
    out: Dict[str, Any] = {
        n_key: 0,
        mae_key: float("nan"),
        w1_key: float("nan"),
        w3_key: float("nan"),
        inside_key: float("nan"),
    }
    if include_median:
        out[f"onset_median_abs_error{prefix}"] = float("nan")
    if hits is None or len(hits) == 0:
        return out
    errors = pd.to_numeric(hits["onset_abs_error"], errors="coerce")
    interval = hits[pd.notna(hits["onset_inside_interval"])]
    out[n_key] = int(len(hits))
    out[mae_key] = _safe_mean(errors)
    out[w1_key] = _safe_rate(int((errors <= 1).sum()), len(hits))
    out[w3_key] = _safe_rate(int((errors <= 3).sum()), len(hits))
    out[inside_key] = _safe_rate(
        int((interval["onset_inside_interval"] == True).sum()),  # noqa: E712
        len(interval),
    )
    if include_median:
        out[f"onset_median_abs_error{prefix}"] = _safe_median(errors)
    return out


def _feature_group_tp_onset_hits(
    sub: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    valid = sub.copy()
    valid = valid[valid["gt_category"].astype(str).str.strip() != ""]
    if not include_ambiguous:
        valid = valid[~valid["gt_category"].isin(EXCLUDED_GT_CATEGORIES)]
    eval_valid = valid[valid["gt_category"].isin(PRIMARY_GT_CATEGORIES)]
    gt_ign = eval_valid[eval_valid["gt_category"] == "ignition"]
    hits = gt_ign[gt_ign["pred_has_ignition"] == True].copy()  # noqa: E712
    if hits.empty:
        return hits
    hits["__track_identity"] = _track_identity_series(hits)
    hits = hits[hits["__track_identity"].astype(str).str.strip().ne("")]
    return hits


def compute_feature_group_summary(
    per_track_df: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if per_track_df.empty:
        return pd.DataFrame()
    tp_hits_by_key: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (protocol, feature_group), sub in per_track_df.groupby(
        ["protocol", "feature_group"], sort=False
    ):
        valid = sub.copy()
        valid = valid[valid["gt_category"].astype(str).str.strip() != ""]
        if not include_ambiguous:
            valid = valid[~valid["gt_category"].isin(EXCLUDED_GT_CATEGORIES)]
        eval_valid = valid[valid["gt_category"].isin(PRIMARY_GT_CATEGORIES)]
        gt_ign = eval_valid[eval_valid["gt_category"] == "ignition"]
        gt_no = eval_valid[eval_valid["gt_category"] == "no_ignition"]
        gt_hot = eval_valid[eval_valid["gt_category"] == "already_hot"]
        tp = int((gt_ign["pred_has_ignition"] == True).sum())  # noqa: E712
        fn = int((gt_ign["pred_has_ignition"] == False).sum())  # noqa: E712
        fp = int((gt_no["pred_has_ignition"] == True).sum())  # noqa: E712
        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = (
            2 * precision * recall / (precision + recall)
            if not np.isnan(precision + recall) and precision + recall > 0
            else float("nan")
        )
        onset_hits = _feature_group_tp_onset_hits(
            sub, include_ambiguous=include_ambiguous
        )
        tp_hits_by_key[(str(protocol), str(feature_group))] = onset_hits
        method_errors = sub[
            sub["pred_ignition_status"].astype(str).isin(["method_error", "track_error"])
            | (sub["error_message"].astype(str).str.len() > 0)
        ]
        k_counts = sub["gmm_n_states"].value_counts(dropna=False).to_dict()
        nonzero_feat = pd.to_numeric(sub["n_features"], errors="coerce")
        rows.append({
            "protocol": protocol,
            "feature_group": feature_group,
            "n_tracks": int(len(sub)),
            "n_valid_gt": int(len(eval_valid)),
            "mean_n_features": float(nonzero_feat.mean()),
            "mean_n_features_nonzero": (
                float(nonzero_feat[nonzero_feat > 0].mean())
                if (nonzero_feat > 0).any() else float("nan")
            ),
            "n_entered_model": int((nonzero_feat > 0).sum()),
            "n_method_errors": int(len(method_errors)),
            "method_error_rate": _safe_rate(len(method_errors), len(sub)),
            "mean_gmm_n_states": _safe_mean(sub["gmm_n_states"]),
            "n_k0": int(k_counts.get(0, 0)),
            "n_k1": int(k_counts.get(1, 0)),
            "n_k2": int(k_counts.get(2, 0)),
            "n_k3": int(k_counts.get(3, 0)),
            **count_statuses(sub["pred_ignition_status"], [
                "ignition_detected",
                "no_ignition",
                "no_ignition_weak_transition",
                "already_hot",
                "already_hot_at_sequence_start",
                "rejected_shape_sequence",
                "single_state",
                "ambiguous",
                "ambiguous_gmm_unstable",
                "method_error",
                "track_error",
            ]),
            "ignition_precision": precision,
            "ignition_recall": recall,
            "ignition_f1": f1,
            "false_ignition_rate": _safe_rate(fp, len(gt_no)),
            "missed_ignition_rate": _safe_rate(fn, len(gt_ign)),
            "already_hot_accuracy": _safe_rate(
                int((gt_hot["is_already_hot_correct"] == True).sum()), len(gt_hot)  # noqa: E712
            ),
            **_onset_metrics_from_hits(
                onset_hits, prefix="", include_median=True
            ),
        })

    # 同一 protocol 下，各 feature_group 的 TP 轨迹交集上计算 common onset。
    common_by_protocol: Dict[str, set] = {}
    for (protocol, _fg), hits in tp_hits_by_key.items():
        ids = set(hits["__track_identity"].astype(str)) if len(hits) else set()
        if protocol not in common_by_protocol:
            common_by_protocol[protocol] = ids
        else:
            common_by_protocol[protocol] &= ids
    for row in rows:
        protocol = str(row["protocol"])
        feature_group = str(row["feature_group"])
        common_ids = common_by_protocol.get(protocol, set())
        hits = tp_hits_by_key.get((protocol, feature_group), pd.DataFrame())
        if common_ids and len(hits):
            common_hits = hits[hits["__track_identity"].astype(str).isin(common_ids)]
        else:
            common_hits = hits.iloc[0:0]
        row.update(
            _onset_metrics_from_hits(
                common_hits, prefix="_common", include_median=True
            )
        )
        row["n_onset_eval_common"] = int(len(common_ids))
    return pd.DataFrame(rows)


def compute_feature_group_change_summary(
    per_track_df: pd.DataFrame,
    reference_group: str = REFERENCE_FEATURE_GROUP,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if per_track_df.empty:
        return pd.DataFrame()
    for protocol in per_track_df["protocol"].unique():
        ref = per_track_df[
            (per_track_df["protocol"] == protocol)
            & (per_track_df["feature_group"] == reference_group)
        ].set_index("track_name")
        if ref.empty:
            continue
        for feature_group in per_track_df["feature_group"].unique():
            if feature_group == reference_group:
                continue
            sub = per_track_df[
                (per_track_df["protocol"] == protocol)
                & (per_track_df["feature_group"] == feature_group)
            ]
            n_presence = n_status = n_onset = 0
            onset_shifts: List[float] = []
            for _, row in sub.iterrows():
                if row["track_name"] not in ref.index:
                    continue
                base = ref.loc[row["track_name"]]
                if bool(row["pred_has_ignition"]) != bool(base["pred_has_ignition"]):
                    n_presence += 1
                if str(row["pred_ignition_status"]) != str(base["pred_ignition_status"]):
                    n_status += 1
                if bool(row["pred_has_ignition"]) and bool(base["pred_has_ignition"]):
                    current = int(row["pred_onset_idx"])
                    reference = int(base["pred_onset_idx"])
                    if current != reference:
                        n_onset += 1
                    if current >= 0 and reference >= 0:
                        onset_shifts.append(abs(current - reference))
            rows.append({
                "protocol": protocol,
                "feature_group": feature_group,
                "reference_feature_group": reference_group,
                "n_tracks": int(len(sub)),
                "n_changed_presence": n_presence,
                "n_changed_status": n_status,
                "n_changed_onset": n_onset,
                "mean_abs_onset_shift_for_common_ignition": (
                    float(np.mean(onset_shifts)) if onset_shifts else float("nan")
                ),
                "median_abs_onset_shift_for_common_ignition": (
                    float(np.median(onset_shifts)) if onset_shifts else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def is_eval_error_case(row: pd.Series) -> bool:
    gt_cat = str(row.get("gt_category", "")).lower()
    if not gt_cat or gt_cat in EXCLUDED_GT_CATEGORIES:
        return False
    if gt_cat == "ignition":
        error = row.get("onset_abs_error")
        inside = row.get("onset_inside_interval")
        return (
            not bool(row.get("pred_has_ignition", False))
            or (pd.notna(error) and float(error) > 3)
            or (pd.notna(inside) and not bool(inside))
        )
    if gt_cat == "no_ignition":
        return bool(row.get("is_false_ignition", False))
    if gt_cat == "already_hot":
        return not bool(row.get("is_already_hot_correct", False))
    return False


def is_changed_vs_reference(row: pd.Series, ref_row: pd.Series) -> bool:
    return (
        bool(row["pred_has_ignition"]) != bool(ref_row["pred_has_ignition"])
        or str(row["pred_ignition_status"]) != str(ref_row["pred_ignition_status"])
        or (
            bool(row["pred_has_ignition"])
            and bool(ref_row["pred_has_ignition"])
            and int(row["pred_onset_idx"]) != int(ref_row["pred_onset_idx"])
        )
    )


def plot_feature_group_result(
    df: pd.DataFrame,
    result: Dict[str, Any],
    save_path: Path,
    feature_group: str,
    protocol: str,
    gt: Optional[Dict[str, Any]] = None,
    ignition_mod: Any = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(df)
    x = np.arange(n)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for col in ("gray_p99", "corr_p99", "bright_energy_mean"):
        if col in df.columns and ignition_mod is not None:
            axes[0].plot(
                x, ignition_mod.normalize_to_01(df[col].values),
                label=col, alpha=0.85,
            )
    axes[0].legend(loc="best", fontsize=7)
    axes[0].set_ylabel("Features (norm)")
    for key in ("hot_score", "change_score"):
        score = result.get(key, [])
        if len(score) == n:
            axes[1].plot(x, score, label=key, linewidth=1.3)
    axes[1].legend(loc="best", fontsize=7)
    axes[1].set_ylabel("Scores")
    seq = result.get("gmm_state_sequence", [])
    if seq and len(seq) == n:
        k = max(int(result.get("gmm_n_states", 1)), 1)
        axes[2].step(
            x, np.asarray(seq) / max(k - 1, 1),
            where="mid", label="gmm_state_rank", linewidth=2,
        )
    onset = int(result.get("ignition_idx_onset", -1))
    if onset >= 0:
        axes[2].axvline(onset, color="orange", linestyle=":", label=f"pred={onset}")
    if gt and int(gt.get("gt_onset_idx", -1)) >= 0:
        gt_onset = int(gt["gt_onset_idx"])
        axes[2].axvline(
            gt_onset, color="green", linestyle="--", label=f"gt={gt_onset}"
        )
    axes[2].legend(loc="best", fontsize=7)
    axes[2].set_ylabel("State")
    axes[2].set_xlabel("Frame index")
    fig.suptitle(
        f"{feature_group} | {protocol} | {result.get('ignition_status', '')} | "
        f"n_feat={len(result.get('gmm_feature_cols', []))}",
        fontsize=10,
    )
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def build_base_cfg(args: argparse.Namespace, mod: Any) -> Any:
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
        use_diff_features=True,
        enable_initial_hot_reject=args.enable_initial_hot_reject,
        enable_shape_sequence_reject=args.enable_shape_sequence_reject,
    )
    overrides = {}
    if args.require_all_prior_cold is not None:
        overrides["require_all_prior_cold"] = args.require_all_prior_cold
    if args.refine_onset_with_change is not None:
        overrides["refine_onset_with_change"] = args.refine_onset_with_change
    return replace(cfg, **overrides)


def process_one_track(
    track_name: str,
    folder: str,
    protocols: List[str],
    feature_groups: List[str],
    ignition_mod: Any,
    cfg: Any,
    annotations: Dict[str, Dict[str, Any]],
    save_error_cases: bool,
    save_changed_cases: bool,
    error_cases_dir: Optional[Path],
    changed_cases_dir: Optional[Path],
    missing_as_no_ignition: bool = False,
    missing_as_invalid: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    track_id = ignition_mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        annotations,
        track_name,
        track_id,
        missing_as_no_ignition=missing_as_no_ignition,
        missing_as_invalid=missing_as_invalid,
    )
    df: Optional[pd.DataFrame] = None
    predictions: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        image_paths = list_roi_images_in_folder(folder)
        if not image_paths:
            raise RuntimeError(f"No ROI images found in track folder: {folder}")
        # 直接从 track_crop/track_* 中的逐轨 ROI 图像提取特征；不需要原始全幅 image-dir。
        # 每条轨迹只提取一次，所有特征组及协议共享同一缓存，保证消融输入一致。
        df = ignition_mod.extract_sequence_features(image_paths, cfg)
        if set(PAPER_FEATURE_GROUPS).issubset(set(feature_groups)):
            validate_progressive_feature_chain(df, cfg, ignition_mod)
        for protocol in protocols:
            for feature_group in feature_groups:
                try:
                    pred = detect_gmm_rtv_with_feature_group(
                        df, ignition_mod, cfg, feature_group, protocol
                    )
                    feature_cols = list(pred.get("gmm_feature_cols", []))
                    row = result_to_row(
                        track_name, track_id, protocol, feature_group, len(df),
                        feature_cols, pred, evaluate_prediction(pred, gt),
                    )
                    rows.append(row)
                    predictions[(protocol, feature_group)] = pred
                except Exception as exc:
                    pred = _empty_gmm_result(
                        ignition_mod, df, cfg, feature_group, protocol, "method_error"
                    )
                    rows.append(result_to_row(
                        track_name, track_id, protocol, feature_group, len(df), [],
                        pred, evaluate_prediction(pred, gt), str(exc),
                    ))

        if save_error_cases or save_changed_cases:
            ref_rows = {
                row["protocol"]: row
                for row in rows
                if row["feature_group"] == REFERENCE_FEATURE_GROUP
            }
            for row in rows:
                protocol = row["protocol"]
                feature_group = row["feature_group"]
                pred = predictions.get((protocol, feature_group))
                if pred is None:
                    continue
                save_path: Optional[Path] = None
                if save_error_cases and error_cases_dir and is_eval_error_case(pd.Series(row)):
                    save_path = error_cases_dir / protocol / feature_group / f"{track_name}.png"
                elif (
                    save_changed_cases
                    and changed_cases_dir
                    and feature_group != REFERENCE_FEATURE_GROUP
                    and protocol in ref_rows
                    and is_changed_vs_reference(pd.Series(row), pd.Series(ref_rows[protocol]))
                ):
                    save_path = changed_cases_dir / protocol / feature_group / f"{track_name}.png"
                if save_path is not None:
                    try:
                        plot_feature_group_result(
                            df, pred, save_path, feature_group, protocol, gt, ignition_mod
                        )
                    except Exception:
                        pass
    except Exception as exc:
        empty_df = pd.DataFrame() if df is None else df
        for protocol in protocols:
            for feature_group in feature_groups:
                pred = _empty_gmm_result(
                    ignition_mod, empty_df, cfg, feature_group, protocol, "track_error"
                )
                rows.append(result_to_row(
                    track_name, track_id, protocol, feature_group, len(empty_df), [],
                    pred, evaluate_prediction(pred, gt), str(exc),
                ))
    return rows


def build_track_error_rows(
    task: Dict[str, Any],
    protocols: List[str],
    feature_groups: List[str],
    ignition_mod: Any,
    cfg: Any,
    annotations: Dict[str, Dict[str, Any]],
    missing_as_no_ignition: bool,
    missing_as_invalid: bool,
    error: BaseException,
) -> List[Dict[str, Any]]:
    track_name = str(task["track_name"])
    track_id = ignition_mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        annotations,
        track_name,
        track_id,
        missing_as_no_ignition=missing_as_no_ignition,
        missing_as_invalid=missing_as_invalid,
    )
    empty_df = pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for protocol in protocols:
        for feature_group in feature_groups:
            pred = _empty_gmm_result(
                ignition_mod,
                empty_df,
                cfg,
                feature_group,
                protocol,
                "track_error",
            )
            rows.append(result_to_row(
                track_name,
                track_id,
                protocol,
                feature_group,
                0,
                [],
                pred,
                evaluate_prediction(pred, gt),
                str(error),
            ))
    return rows


_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(ctx: Dict[str, Any]) -> None:
    global _WORKER_CTX
    mod = load_ignition_module(Path(ctx["ignition_module_path"]))
    _WORKER_CTX = {
        **ctx,
        "mod": mod,
        "cfg": mod.IgnitionGMMConfig(**ctx["cfg_dict"]),
        "error_cases_dir": Path(ctx["error_cases_dir"]) if ctx["error_cases_dir"] else None,
        "changed_cases_dir": Path(ctx["changed_cases_dir"]) if ctx["changed_cases_dir"] else None,
    }


def _worker_process_track(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return process_one_track(
        track_name=task["track_name"],
        folder=task["folder"],
        protocols=_WORKER_CTX["protocols"],
        feature_groups=_WORKER_CTX["feature_groups"],
        ignition_mod=_WORKER_CTX["mod"],
        cfg=_WORKER_CTX["cfg"],
        annotations=_WORKER_CTX["annotations"],
        save_error_cases=_WORKER_CTX["save_error_cases"],
        save_changed_cases=_WORKER_CTX["save_changed_cases"],
        error_cases_dir=_WORKER_CTX["error_cases_dir"],
        changed_cases_dir=_WORKER_CTX["changed_cases_dir"],
        missing_as_no_ignition=_WORKER_CTX["missing_as_no_ignition"],
        missing_as_invalid=_WORKER_CTX["missing_as_invalid"],
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GMM-RTV 特征组消融实验：直接读取 track_crop ROI，按严格 progressive 逻辑改变 GMM 输入特征。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--roi-root", "--tracks-root",
        dest="roi_root",
        type=str,
        default='inputs/run_20260703_104159'
        r"\postprocessed\tracking_output\run_20260706_105300\track_crop",
        help=(
            "逐轨 ROI 图像根目录，目录结构应为 track_crop/track_*/<roi images>. "
            "不需要原始全幅 image-dir；--tracks-root 保留为兼容别名。"
        ),
    )
    p.add_argument(
        "--annotation-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "effective_trace.xlsx"),
        help=(
            "人工标注；默认 effective_trace.xlsx：仅表内且 frame_id 非空的轨迹为 ignition，"
            "其余可评价 track 均为 no_ignition；也支持 manual_review_gt、category、"
            "effective_trace 及旧 effective_trace_new 格式"
        ),
    )
    p.add_argument(
        "--reference-summary", type=str,
        default='inputs/ignition_results'
        r"\run_20260715_115437\summary.csv",
        help=(
            "仅旧 effective_trace_new 需要的参考 summary；无 status 列时"
            "所有行有效，有 status 列时仅 status=='ok' 行可生成 GT"
        ),
    )
    p.add_argument("--gt-onset-tolerance", type=int, default=2)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--ignition-module-path", type=str,
        default="ignition_detection_gmm_rtv_new.py",
    )
    p.add_argument("--feature-groups", type=str, default=DEFAULT_FEATURE_GROUPS)
    p.add_argument("--protocols", type=str, default=DEFAULT_PROTOCOLS)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--include-ambiguous", action="store_true")
    p.add_argument("--save-error-cases", action="store_true")
    p.add_argument("--save-changed-cases", action="store_true")

    # 与 IgnitionGMMConfig 的核心默认值一致。
    p.add_argument("--gmm-max-states", type=int, default=3)
    p.add_argument("--gmm-min-state-count", type=int, default=2)
    p.add_argument("--gmm-min-state-fraction", type=float, default=0.08)
    p.add_argument("--gmm-bic-improvement-ratio", type=float, default=0.03)
    p.add_argument("--min-state-run", type=int, default=2)
    p.add_argument("--min-cold-run", type=int, default=3)
    p.add_argument("--min-hot-run", type=int, default=3)
    p.add_argument("--min-stable-run", type=int, default=3)
    p.add_argument("--min-sequence-len", type=int, default=8)
    p.add_argument("--refine-radius", type=int, default=2)
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
        "--no-initial-hot-reject",
        dest="enable_initial_hot_reject",
        action="store_false",
        default=True,
    )
    p.add_argument(
        "--no-shape-sequence-reject",
        dest="enable_shape_sequence_reject",
        action="store_false",
        default=True,
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    ignition_path = Path(args.ignition_module_path)
    if not ignition_path.is_file():
        ignition_path = script_dir / ignition_path

    feature_groups = parse_csv_list(args.feature_groups)
    protocols = [normalize_protocol_name(p) for p in parse_csv_list(args.protocols)]
    validate_experiment_matrix(feature_groups, protocols)

    output_dir = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    mod = load_ignition_module(ignition_path)
    cfg = build_base_cfg(args, mod)
    gt_metadata: Dict[str, Any] = {}
    annotations, annotation_format = load_annotations(
        Path(args.annotation_csv) if args.annotation_csv else None,
        reference_summary=Path(args.reference_summary) if args.reference_summary else None,
        onset_tolerance=args.gt_onset_tolerance,
        metadata_out=gt_metadata,
    )
    # effective_trace.xlsx 的二分类 GT 语义：
    #   - 只有“表内且 frame_id 非空”的轨迹 = ignition（frame_id 为 ROI 序列 onset 索引）；
    #   - 表内 frame_id 为空，以及未出现在表中的其余 track = no_ignition。
    # 因此 feature/decision ablation 应对完整可评价 track population 计算 P/R/F1。
    missing_as_no_ignition = annotation_format in (
        "effective_trace",
        "effective_trace_new",
        "manual_review_gt",
    )
    missing_as_invalid = False
    if annotation_format == "effective_trace":
        print(
            "[INFO] effective_trace evaluation: positive iff listed with non-empty frame_id; "
            "all other evaluable tracks are treated as no_ignition."
        )
    track_folders = list_track_folders_from_roi_root(Path(args.roi_root))
    if not track_folders:
        raise RuntimeError(f"No track_* folders under ROI root: {args.roi_root}")
    if annotation_format == "effective_trace":
        available_ids = {
            int(mod.track_id_from_name(name))
            for name, _folder in track_folders
            if mod.track_id_from_name(name) is not None
        }
        positive_ids = set()
        for key, value in annotations.items():
            if str(value.get("gt_category", "")).lower() != "ignition":
                continue
            tid = mod.track_id_from_name(str(key))
            if tid is not None:
                positive_ids.add(int(tid))
        evaluable_positive_ids = positive_ids & available_ids
        missing_positive_ids = sorted(positive_ids - available_ids)
        print(
            f"[INFO] effective_trace GT coverage: listed_positive={len(positive_ids)} "
            f"evaluable_positive={len(evaluable_positive_ids)} total_tracks={len(available_ids)} "
            f"implicit_negative={len(available_ids) - len(evaluable_positive_ids)}"
        )
        if missing_positive_ids:
            print(
                f"[WARN] {len(missing_positive_ids)} positive GT tracks are absent from ROI root and are "
                f"excluded from metrics; first IDs={missing_positive_ids[:20]}"
            )
    if annotation_format == "manual_review_gt":
        gt_metadata = framework.build_manual_review_evaluation_metadata(
            track_folders,
            annotations,
            gt_metadata,
        )
        print(
            "[WARN] manual_review_gt: tracks absent from new_gt.xlsx are "
            "evaluated as no_ignition"
        )
    (output_dir / "gt_protocol.json").write_text(
        json.dumps(
            gt_metadata,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # 保存实验设计说明，便于论文结果回溯。
    experiment_design = {
        "input_mode": "per_track_roi_crops",
        "roi_root": str(Path(args.roi_root)),
        "feature_groups": feature_groups,
        "protocols": protocols,
        "protocol_definition": {
            "gmm_without_rtv": "full GMM-based temporal decision chain with RTV disabled",
            "final_detection": "same chain with raw transition validation enabled",
        },
        "progressive_feature_logic": [
            "gray_only",
            "+ expanded grayscale/background-corrected descriptors",
            "+ core radiometric descriptors",
            "+ color descriptors",
            "+ morphology descriptor(s) = complete absolute-state representation",
            "+ temporal-difference descriptors only",
        ],
    }
    (output_dir / "experiment_design.json").write_text(
        json.dumps(experiment_design, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] output_dir:     {output_dir.resolve()}")
    print(f"[INFO] roi_root:       {Path(args.roi_root)}")
    print("[INFO] input_mode:     per-track ROI crops (no full-frame image-dir required)")
    print(f"[INFO] feature_groups: {feature_groups}")
    print(f"[INFO] protocols:      {protocols}")
    print(f"[INFO] tracks:         {len(track_folders)}")
    n_workers = resolve_workers(args.workers)
    print(f"[INFO] workers:        {n_workers}")

    error_dir = output_dir / "error_cases" if args.save_error_cases else None
    changed_dir = output_dir / "changed_cases" if args.save_changed_cases else None
    tasks = [{"track_name": name, "folder": folder} for name, folder in track_folders]
    all_rows: List[Dict[str, Any]] = []

    if n_workers == 1:
        for index, task in enumerate(tasks, 1):
            print(f"[{index}/{len(tasks)}] {task['track_name']}")
            try:
                all_rows.extend(process_one_track(
                    task["track_name"], task["folder"], protocols, feature_groups,
                    mod, cfg, annotations, args.save_error_cases,
                    args.save_changed_cases, error_dir, changed_dir,
                    missing_as_no_ignition, missing_as_invalid,
                ))
            except Exception as exc:
                traceback.print_exc()
                all_rows.extend(build_track_error_rows(
                    task,
                    protocols,
                    feature_groups,
                    mod,
                    cfg,
                    annotations,
                    missing_as_no_ignition,
                    missing_as_invalid,
                    exc,
                ))
    else:
        worker_ctx = {
            "ignition_module_path": str(ignition_path.resolve()),
            "cfg_dict": asdict(cfg),
            "protocols": protocols,
            "feature_groups": feature_groups,
            "annotations": annotations,
            "save_error_cases": args.save_error_cases,
            "save_changed_cases": args.save_changed_cases,
            "error_cases_dir": str(error_dir) if error_dir else "",
            "changed_cases_dir": str(changed_dir) if changed_dir else "",
            "missing_as_no_ignition": missing_as_no_ignition,
            "missing_as_invalid": missing_as_invalid,
        }
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(worker_ctx,),
        ) as executor:
            futures = {executor.submit(_worker_process_track, task): task for task in tasks}
            for done, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    all_rows.extend(future.result())
                except Exception as exc:
                    print(f"[ERROR] {task['track_name']}: {exc}")
                    all_rows.extend(build_track_error_rows(
                        task,
                        protocols,
                        feature_groups,
                        mod,
                        cfg,
                        annotations,
                        missing_as_no_ignition,
                        missing_as_invalid,
                        exc,
                    ))
                print(f"[{done}/{len(tasks)}] {task['track_name']}")

    per_track_df = pd.DataFrame(all_rows)
    if not per_track_df.empty:
        per_track_df = per_track_df.sort_values(
            ["track_id", "protocol", "feature_group"]
        ).reset_index(drop=True)
    save_table(per_track_df, output_dir / "per_track_feature_group_results")
    summary_df = compute_feature_group_summary(
        per_track_df, include_ambiguous=args.include_ambiguous
    )
    save_table(summary_df, output_dir / "feature_group_summary")
    change_df = compute_feature_group_change_summary(per_track_df)
    save_table(change_df, output_dir / "feature_group_change_summary")

    print("\n=== Feature Group Summary (preview) ===")
    if summary_df.empty:
        print("(no summary)")
    else:
        columns = [
            "protocol", "feature_group", "n_tracks", "n_entered_model",
            "mean_n_features_nonzero", "ignition_f1", "onset_mae",
            "within_3_frame_acc", "n_k1", "n_k2", "n_k3", "method_error_rate",
        ]
        print(summary_df[[c for c in columns if c in summary_df]].to_string(index=False))
    print(f"\n=== Done ===\nrows={len(per_track_df)}")


if __name__ == "__main__":
    main()
