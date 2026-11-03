#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点火检测 baseline 对比与模块消融实验。

固定输入为 tracks_root 下 track_* ROI 序列，比较不同方法在
“给定 ROI 时序下是否能正确判断点火及定位点火帧”的能力。

消融规划：
  1) frame-level / change-score baselines
  2) GMM-only trajectory state modeling
  3) GMM-only + raw-transition gate
  4) HMM-GMM temporal regularization
  5) HMM-GMM + low-signal rejection（可选，默认方法列表不含）
  6) HMM-GMM + raw-transition gate
  7) proposed full：HMM-GMM + raw-transition gate（默认关闭 low-signal reject）

说明：GMM-only 与 HMM-GMM 均使用相同的 raw thermal onset score 进行 onset refinement，
避免 HMM/feature ablation 中由不同特征矩阵产生的 change_score 偏置。

Baseline（gray_p99_threshold / multi_feature_change_score）默认开启网格寻优
（--tune-baselines），按 F1（其次 onset interval）选超参后再与 HMM 方法对比。

评估约定：
  - 对 tracks_root 下全部 track 跑检测；
  - effective_trace：frame_id 非空=点火；空/未列出=未点火；
  - effective_trace_new：xlsx 列出的=未点火；参考 summary 中
    ignition_idx_candidate!=-1 且不在 xlsx =点火 GT（onset±tol）；
    candidate==-1 / 未出现 =未点火。
"""

from __future__ import annotations

import argparse
import importlib.util
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

DEFAULT_METHODS = (
    "gray_p99_threshold,"
    "multi_feature_change_score,"
    "gmm_only,"
    "gmm_only_raw_gate,"
    "hmm_gmm,"
    "hmm_gmm_raw_gate,"
    "proposed_full"
)

# 用于规则 baseline 与 HMM/GMM onset refinement 的统一 raw thermal cue。
# 这样 GMM-only/HMM-GMM 的差异主要来自状态解码方式，而不是 onset 微调分数。
CHANGE_RAW_FEATURES = [
    "gray_p99",
    "corr_p99",
    "bright_energy_mean",
    "bright_area_ratio",
    "hsv_v_p99",
]

NO_IGNITION_STATUSES = {
    "no_ignition",
    "no_ignition_black_sequence",
    "no_ignition_weak_transition",
}

PRIMARY_GT_CATEGORIES = {"ignition", "no_ignition", "already_hot"}
EXCLUDED_GT_CATEGORIES = {"invalid", "suspicious_merge", "defocused", "ambiguous"}

BASELINE_METHODS = ("gray_p99_threshold", "multi_feature_change_score")
DEFAULT_GRAY_THRESHOLD_GRID = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
DEFAULT_CHANGE_THRESHOLD_GRID = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40]
DEFAULT_MIN_PERSIST_GRID = [2, 3, 4, 5]


# ============================================================
# Module loading
# ============================================================

def load_ignition_module(module_path: Path):
    module_path = module_path.resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"Ignition module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("ignition_hmm_gmm_ablation", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ignition module spec from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    required = [
        "IgnitionHMMConfig",
        "list_images_in_folder",
        "list_track_folders",
        "track_id_from_name",
        "extract_sequence_features",
        "detect_ignition_hmm_gmm",
        "build_hmm_gmm_feature_matrix",
        "fit_gmm_models_bic",
        "select_gmm_with_occupancy_validation",
        "compute_state_thermal_scores",
        "build_state_rank_mapping",
        "gmm_diag_log_emission",
        "reorder_log_emission_by_rank",
        "build_left_to_right_transition_logprob",
        "make_adaptive_init_logprob",
        "viterbi_decode",
        "smooth_state_sequence_by_run_length",
        "interpret_hmm_state_sequence_for_ignition",
        "compute_hot_score",
        "compute_hmm_change_score",
        "raw_transition_has_thermal_increase",
        "find_first_valid_cold_to_hot_transition",
        "find_stable_combustion_frame",
        "normalize_to_01",
        "plot_detection_result_hmm_gmm",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise AttributeError(f"Ignition module missing required symbol: {name}")
    return mod


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


def make_default_output_dir(base: str = "ignition_ablation_results") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base) / f"run_{stamp}"


def resolve_workers(workers: int) -> int:
    """workers>0 用指定进程数；workers<=0 自动取 min(cpu-1, 16)。"""
    if workers > 0:
        return workers
    import os
    return max(1, min((os.cpu_count() or 4) - 1, 16))


# ============================================================
# Annotations
# ============================================================

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_track_name(raw: Any) -> str:
    """统一 track 命名：'track_6' / '6' / 6 → 'track_6'（不补零，与文件夹一致）。"""
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
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, encoding="utf-8-sig")


MANUAL_REVIEW_GT_ALIASES = {
    "track": ("轨迹", "track_name", "track_id"),
    "valid": ("是否有效", "is_valid", "valid"),
    "manual_onset": ("人为判断点火帧", "manual_onset", "manual_onset_frame"),
    "reason": ("原因", "reason"),
    "model_onset": ("模型判断点火帧", "model_onset", "model_onset_frame"),
}


def _normalized_manual_review_columns(df: pd.DataFrame) -> Dict[str, str]:
    normalized = {str(column).strip(): column for column in df.columns}
    selected: Dict[str, str] = {}
    for field, aliases in MANUAL_REVIEW_GT_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                selected[field] = normalized[alias]
                break
    return selected


def _manual_review_missing_fields(df: pd.DataFrame) -> List[str]:
    columns = _normalized_manual_review_columns(df)
    return [
        field
        for field in ("track", "valid", "manual_onset")
        if field not in columns
    ]


def is_manual_review_gt_format(df: pd.DataFrame) -> bool:
    return not _manual_review_missing_fields(df)


def _is_category_annotation_format(df: pd.DataFrame) -> bool:
    columns = {str(column).strip().lower() for column in df.columns}
    has_track = bool(columns & {"track_name", "track_id"})
    has_category = bool(
        columns & {"ignition_category", "gt_category", "category"}
    )
    return has_track and has_category


def _is_legacy_annotation_format(df: pd.DataFrame) -> bool:
    return is_effective_trace_format(df) or _is_category_annotation_format(df)


def _is_obviously_manual_review_gt(df: pd.DataFrame) -> bool:
    columns = {str(column).strip() for column in df.columns}
    if columns & {"是否有效", "人为判断点火帧"}:
        return True
    has_english_valid = bool(columns & {"valid", "is_valid"})
    has_english_manual_onset = bool(
        columns & {"manual_onset", "manual_onset_frame"}
    )
    return has_english_valid and has_english_manual_onset


def _select_manual_review_gt_sheet(
    path: Path,
    sheets: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    matching = [
        (sheet_name, frame)
        for sheet_name, frame in sheets.items()
        if is_manual_review_gt_format(frame)
        and not _is_legacy_annotation_format(frame)
    ]
    if len(matching) != 1:
        diagnostics = "; ".join(
            f"sheet {sheet_name!r}: columns="
            f"{[str(column).strip() for column in frame.columns]!r}, "
            f"missing required={_manual_review_missing_fields(frame)!r}"
            for sheet_name, frame in sheets.items()
        )
        raise ValueError(
            "Manual-review GT workbook must contain exactly one matching sheet; "
            f"found {len(matching)} in {path.resolve()}. {diagnostics}"
        )

    sheet_name, source = matching[0]
    source_columns = [str(column).strip() for column in source.columns]
    frame = source.copy()
    frame.columns = source_columns
    frame = frame.loc[
        :,
        [
            column
            for column in frame.columns
            if not str(column).strip().lower().startswith("unnamed:")
        ],
    ].copy()
    metadata = {
        "path": str(path.resolve()),
        "sheet_name": str(sheet_name),
        "source_row_count": int(len(source)),
        "source_columns": source_columns,
    }
    return frame, metadata


def read_manual_review_gt_workbook(
    path: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manual-review GT workbook not found: {path}")
    sheets = pd.read_excel(path, sheet_name=None)
    return _select_manual_review_gt_sheet(path, sheets)


def _normalize_manual_track_name(raw: Any) -> str:
    if isinstance(raw, (bool, np.bool_)):
        return ""
    if isinstance(raw, (int, np.integer)):
        value = int(raw)
        return f"track_{value}" if value >= 0 else ""
    if isinstance(raw, (float, np.floating)):
        value = float(raw)
        if not np.isfinite(value) or value < 0 or not value.is_integer():
            return ""
        return f"track_{int(value)}"
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if text.lower().startswith("track_"):
        suffix = text.split("_", 1)[1]
    else:
        suffix = text
    if not suffix.isdigit():
        return ""
    return f"track_{int(suffix)}"


def normalize_manual_track_identity(
    value: Any,
    *,
    source_name: str = "track identity",
) -> str:
    """Return one strict canonical manual-GT identity or raise with context."""
    canonical = _normalize_manual_track_name(value)
    if not canonical:
        raise ValueError(
            f"{source_name}: Invalid track identity {value!r}; "
            "expected a finite non-negative integer track_N identity"
        )
    return canonical


def _parse_manual_onset(value: Any, *, row_number: int, track_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        valid = False
    else:
        try:
            number = float(value)
            valid = np.isfinite(number) and number >= 0 and number.is_integer()
        except (TypeError, ValueError):
            valid = False
    if not valid:
        raise ValueError(
            "Invalid manual_onset at row "
            f"{row_number} for track {track_name}: {value!r}; "
            "expected a finite non-negative integer"
        )
    return int(number)


def build_annotations_from_manual_review_gt(
    df: pd.DataFrame,
    onset_tolerance: int = 2,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if (
        isinstance(onset_tolerance, (bool, np.bool_))
        or not isinstance(onset_tolerance, (int, np.integer))
        or int(onset_tolerance) < 0
    ):
        raise ValueError(
            "onset_tolerance must be a non-negative integer, "
            f"got {onset_tolerance!r}"
        )
    tolerance = int(onset_tolerance)

    columns = _normalized_manual_review_columns(df)
    missing = [
        field
        for field in ("track", "valid", "manual_onset")
        if field not in columns
    ]
    if missing:
        raise ValueError(
            "Manual-review GT missing required fields "
            f"{missing}; columns={list(map(str, df.columns))}"
        )

    annotations: Dict[str, Dict[str, Any]] = {}
    first_rows: Dict[str, int] = {}
    reasons: List[str] = []
    positive_count = 0
    negative_count = 0
    ignored_model_count = 0
    ignored_invalid_manual_count = 0

    for row_offset, (_, row) in enumerate(df.iterrows(), start=2):
        raw_track = row[columns["track"]]
        track_name = _normalize_manual_track_name(raw_track)
        if not track_name:
            raise ValueError(
                f"Invalid track at row {row_offset}: {raw_track!r}; "
                "expected a non-negative track_N identity"
            )
        if track_name in annotations:
            raise ValueError(
                f"duplicate canonical track ID {track_name!r} at row {row_offset}; "
                f"first seen at row {first_rows[track_name]}"
            )
        first_rows[track_name] = row_offset

        raw_validity = row[columns["valid"]]
        validity = str(raw_validity).strip()
        if validity not in {"是", "否"}:
            raise ValueError(
                f"Invalid 是否有效 at row {row_offset} for track {track_name}: "
                f"{raw_validity!r}; expected '是' or '否'"
            )

        if "reason" in columns:
            raw_reason = row[columns["reason"]]
            if pd.notna(raw_reason):
                reason = str(raw_reason).strip()
                if reason:
                    reasons.append(reason)
        if "model_onset" in columns and pd.notna(row[columns["model_onset"]]):
            ignored_model_count += 1

        raw_manual_onset = row[columns["manual_onset"]]
        if validity == "否":
            negative_count += 1
            if pd.notna(raw_manual_onset):
                ignored_invalid_manual_count += 1
            annotations[track_name] = _no_ignition_gt()
            continue

        onset = _parse_manual_onset(
            raw_manual_onset,
            row_number=row_offset,
            track_name=track_name,
        )
        positive_count += 1
        annotations[track_name] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_idx": onset,
            "gt_onset_start": max(0, onset - tolerance),
            "gt_onset_end": onset + tolerance,
        }

    reason_counts = {
        reason: int(count)
        for reason, count in pd.Series(reasons, dtype="object").value_counts().items()
    }
    metadata = {
        "format": "manual_review_gt",
        "source_positive_count": int(positive_count),
        "source_explicit_negative_count": int(negative_count),
        "reason_counts": reason_counts,
        "ignored_model_onset_count": int(ignored_model_count),
        "ignored_invalid_manual_onset_count": int(ignored_invalid_manual_count),
        "onset_tolerance": tolerance,
        "unlisted_track_policy": "no_ignition",
        "uses_reference_predictions": False,
    }
    return annotations, metadata


def _strict_manual_identity_mapping(
    items: List[Tuple[Any, Any]],
    *,
    source_name: str,
) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    raw_identities: Dict[str, Any] = {}
    for raw_identity, value in items:
        track_name = normalize_manual_track_identity(
            raw_identity,
            source_name=source_name,
        )
        if track_name in canonical:
            raise ValueError(
                f"{source_name}: duplicate canonical track ID {track_name!r}: "
                f"{raw_identities[track_name]!r} conflicts with "
                f"{raw_identity!r}"
            )
        canonical[track_name] = value
        raw_identities[track_name] = raw_identity
    return canonical


def build_manual_review_evaluation_metadata(
    track_folders: List[Tuple[Any, Any]],
    annotations: Dict[str, Dict[str, Any]],
    source_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """汇总 manual-review GT 在当前可用轨迹集合上的实际评估支持数。"""
    available = _strict_manual_identity_mapping(
        list(track_folders),
        source_name="track_folders",
    )
    canonical_annotations = _strict_manual_identity_mapping(
        list(annotations.items()),
        source_name="annotations",
    )
    available_names = set(available)
    explicit_names = available_names & set(canonical_annotations)
    evaluated_positive = sum(
        canonical_annotations[name].get("gt_category") == "ignition"
        for name in explicit_names
    )
    evaluated_explicit_negative = sum(
        canonical_annotations[name].get("gt_category") == "no_ignition"
        for name in explicit_names
    )
    evaluated_implicit_negative = len(available_names - set(canonical_annotations))

    metadata = dict(source_metadata)
    metadata.update({
        "evaluated_track_count": int(len(available_names)),
        "evaluated_explicit_track_count": int(len(explicit_names)),
        "evaluated_implicit_track_count": int(evaluated_implicit_negative),
        "evaluated_positive_count": int(evaluated_positive),
        "evaluated_explicit_negative_count": int(evaluated_explicit_negative),
        "evaluated_implicit_negative_count": int(evaluated_implicit_negative),
        "evaluated_negative_count": int(
            evaluated_explicit_negative + evaluated_implicit_negative
        ),
    })
    return metadata


def is_effective_trace_format(df: pd.DataFrame) -> bool:
    """
    effective_trace.xlsx 紧凑格式：
      - 列：track_id（或 track_name）+ frame_id
      - 无 ignition_category
      - frame_id 非空（非 NaN / 非空白）= 点火（onset 为 crop 索引）
      - frame_id 为空，或未出现在表中 = no_ignition
      - 全部 track 均参与检测与指标
    """
    cols = {str(c).lower() for c in df.columns}
    has_track = "track_id" in cols or "track_name" in cols
    has_frame = "frame_id" in cols
    has_category = bool(cols & {"ignition_category", "gt_category", "category"})
    return has_track and has_frame and not has_category


def is_effective_trace_new_format(df: pd.DataFrame) -> bool:
    """
    effective_trace_new.xlsx：单列未点火 track id。
    无标准列名；用 header 读取时首个 id 常被当成列名。
    """
    if is_effective_trace_format(df):
        return False
    cols = {str(c).lower() for c in df.columns}
    named = {
        "track_id", "track_name", "frame_id",
        "ignition_category", "gt_category", "category",
    }
    if cols & named:
        return False
    if df.shape[1] != 1:
        return False

    def _as_int(v: Any) -> Optional[int]:
        if is_blank_frame_id(v):
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    header_id = _as_int(df.columns[0])
    values = [_as_int(v) for v in df.iloc[:, 0].tolist()]
    parsed = [v for v in values if v is not None]
    if header_id is not None:
        return len(parsed) == len([v for v in values if not is_blank_frame_id(v)])
    return bool(parsed) and len(parsed) == len([v for v in values if not is_blank_frame_id(v)])


def is_blank_frame_id(value: Any) -> bool:
    """frame_id 为空：NaN / None / 空白字符串均视为未标注点火帧。"""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return (not s) or s.lower() in {"nan", "none", "null"}


def _no_ignition_gt() -> Dict[str, Any]:
    return {
        "gt_category": "no_ignition",
        "gt_has_ignition": False,
        "gt_onset_idx": -1,
        "gt_onset_start": -1,
        "gt_onset_end": -1,
    }


def _invalid_gt() -> Dict[str, Any]:
    return {
        "gt_category": "invalid",
        "gt_has_ignition": False,
        "gt_onset_idx": -1,
        "gt_onset_start": -1,
        "gt_onset_end": -1,
    }


def read_invalid_track_ids(path: Path) -> List[int]:
    """读取 effective_trace_new 单列未点火 track id（无表头）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Invalid-track list not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, header=None)
    else:
        df = pd.read_csv(path, header=None, encoding="utf-8-sig")
    if df.shape[1] < 1:
        raise ValueError(f"invalid-track list is empty: {path}")
    ids: List[int] = []
    for v in df.iloc[:, 0].tolist():
        if is_blank_frame_id(v):
            continue
        ids.append(int(float(v)))
    return ids


def read_reference_ignition_summary(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference summary not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"has_ignition", "ignition_idx_onset", "ignition_idx_candidate"}
    missing = need - {str(c) for c in df.columns}
    if missing:
        raise ValueError(f"reference summary missing columns {sorted(missing)}: {path}")
    if "track_name" not in df.columns and "track_id" not in df.columns:
        raise ValueError(f"reference summary needs track_name or track_id: {path}")
    return df


def _parse_track_id_from_row(row: pd.Series, ref_columns) -> Tuple[str, Optional[int]]:
    raw_name = row["track_name"] if "track_name" in ref_columns and pd.notna(row["track_name"]) else None
    raw_id = row["track_id"] if "track_id" in ref_columns and pd.notna(row["track_id"]) else None
    track_name = normalize_track_name(raw_name if raw_name is not None else raw_id)
    tid: Optional[int] = None
    if raw_id is not None:
        try:
            tid = int(float(raw_id))
        except (TypeError, ValueError):
            tid = None
    if tid is None and track_name.startswith("track_"):
        try:
            tid = int(track_name.split("_", 1)[1])
        except (TypeError, ValueError):
            tid = None
    return track_name, tid


def _is_no_ignition_candidate(value: Any) -> bool:
    """ignition_idx_candidate < 0（通常为 -1）视为未点火。"""
    if is_blank_frame_id(value):
        return True
    try:
        return int(float(value)) < 0
    except (TypeError, ValueError):
        return True


def build_annotations_from_effective_trace_new(
    listed_no_ignition_ids: List[int],
    reference_summary: Path,
    onset_tolerance: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """
    effective_trace_new GT：
      - xlsx 列出的 = no_ignition（人工确认未点火，即使参考检出）
      - 参考 summary 中 ignition_idx_candidate != -1 且不在 xlsx = ignition（onset±tol）
      - 参考 summary 中 ignition_idx_candidate == -1 = no_ignition
    """
    if onset_tolerance < 0:
        raise ValueError(f"onset_tolerance must be >= 0, got {onset_tolerance}")
    listed_no = {int(x) for x in listed_no_ignition_ids}
    ref = read_reference_ignition_summary(reference_summary)
    ann: Dict[str, Dict[str, Any]] = {}

    for tid in sorted(listed_no):
        ann[normalize_track_name(tid)] = _no_ignition_gt()

    n_ign = 0
    n_no_listed = 0
    n_no_candidate = 0
    for _, row in ref.iterrows():
        track_name, tid = _parse_track_id_from_row(row, ref.columns)
        if not track_name:
            continue
        if tid is not None and tid in listed_no:
            ann[track_name] = _no_ignition_gt()
            n_no_listed += 1
            continue
        if _is_no_ignition_candidate(row["ignition_idx_candidate"]):
            ann[track_name] = _no_ignition_gt()
            n_no_candidate += 1
            continue
        onset = int(float(row["ignition_idx_onset"]))
        ann[track_name] = {
            "gt_category": "ignition",
            "gt_has_ignition": True,
            "gt_onset_idx": onset,
            "gt_onset_start": max(0, onset - onset_tolerance),
            "gt_onset_end": onset + onset_tolerance,
        }
        n_ign += 1

    n_no = sum(1 for v in ann.values() if v.get("gt_category") == "no_ignition")
    print(
        f"[INFO] effective_trace_new 标注: listed_no_ignition={len(listed_no)} | "
        f"ref={Path(reference_summary).name} | "
        f"ignition_gt={n_ign} no_ignition_gt={n_no} "
        f"(xlsx_in_ref={n_no_listed}, candidate_-1→no_ignition={n_no_candidate}) | "
        f"onset±{onset_tolerance}"
    )
    return ann


def load_annotations(
    annotation_path: Optional[Path],
    *,
    reference_summary: Optional[Path] = None,
    onset_tolerance: int = 2,
    metadata_out: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    加载人工标注。

    Returns:
        (annotations_dict, format_name)
        format_name: "none" | "manual_review_gt" | "effective_trace" |
                     "effective_trace_new" | "category"
    """
    if metadata_out is not None:
        metadata_out.clear()
    if annotation_path is None:
        return {}, "none"
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    workbook_metadata: Dict[str, Any] = {}
    if annotation_path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(annotation_path, sheet_name=None)
        df = next(iter(sheets.values()))
        if not (
            _is_legacy_annotation_format(df)
            or is_effective_trace_new_format(df)
        ):
            matching_count = sum(
                is_manual_review_gt_format(frame)
                and not _is_legacy_annotation_format(frame)
                for frame in sheets.values()
            )
            malformed_manual = any(
                _is_obviously_manual_review_gt(frame)
                and not _is_legacy_annotation_format(frame)
                for frame in sheets.values()
            )
            if matching_count or malformed_manual:
                df, workbook_metadata = _select_manual_review_gt_sheet(
                    annotation_path,
                    sheets,
                )
    else:
        df = read_annotation_table(annotation_path)

    legacy_format = _is_legacy_annotation_format(df)
    if not legacy_format and is_manual_review_gt_format(df):
        try:
            ann, mapping_metadata = build_annotations_from_manual_review_gt(
                df,
                onset_tolerance=onset_tolerance,
            )
        except ValueError as exc:
            if workbook_metadata:
                raise ValueError(
                    "Manual-review GT validation failed for workbook "
                    f"{workbook_metadata['path']}, "
                    f"sheet {workbook_metadata['sheet_name']!r}: {exc}"
                ) from exc
            raise
        if metadata_out is not None:
            metadata_out.update(workbook_metadata)
            metadata_out.update(mapping_metadata)
        return ann, "manual_review_gt"

    if not legacy_format and _is_obviously_manual_review_gt(df):
        raise ValueError(
            f"Incomplete manual-review GT in {annotation_path.resolve()}: "
            f"columns={[str(column).strip() for column in df.columns]!r}, "
            f"missing required={_manual_review_missing_fields(df)!r}"
        )

    if is_effective_trace_new_format(df):
        if reference_summary is None:
            raise ValueError(
                "effective_trace_new 格式需要同时提供 --reference-summary "
                "(ignition_detection_hmm_gmm_shape_hot_prefilter_new 的 summary.csv/xlsx)"
            )
        invalid_ids = read_invalid_track_ids(annotation_path)
        ann = build_annotations_from_effective_trace_new(
            invalid_ids,
            Path(reference_summary),
            onset_tolerance=onset_tolerance,
        )
        return ann, "effective_trace_new"

    track_name_col = _pick_col(df, ["track_name"])
    track_id_col = _pick_col(df, ["track_id"])
    if track_name_col is None and track_id_col is None:
        raise ValueError(
            f"annotation must contain track_name/track_id，或为 effective_trace_new 单列无效 id: "
            f"{annotation_path}"
        )

    ann: Dict[str, Dict[str, Any]] = {}

    if is_effective_trace_format(df):
        frame_col = _pick_col(df, ["frame_id"])
        assert frame_col is not None
        n_ign = n_no = 0
        for _, row in df.iterrows():
            raw = row[track_name_col] if track_name_col else row[track_id_col]
            track_name = normalize_track_name(raw)
            if not track_name:
                continue
            if not is_blank_frame_id(row[frame_col]):
                onset = int(float(row[frame_col]))
                cat = "ignition"
                n_ign += 1
                ann[track_name] = {
                    "gt_category": cat,
                    "gt_has_ignition": True,
                    "gt_onset_idx": onset,
                    "gt_onset_start": -1,
                    "gt_onset_end": -1,
                }
            else:
                n_no += 1
                ann[track_name] = _no_ignition_gt()
        print(
            f"[INFO] effective_trace 标注: {annotation_path.name} | "
            f"listed={len(ann)} ignition={n_ign} listed_no_ignition={n_no} | "
            f"未列出的 track 与空 frame_id 一样按 no_ignition；全部 track 计入指标"
        )
        return ann, "effective_trace"

    category_col = _pick_col(df, ["ignition_category", "gt_category", "category"])
    onset_col = _pick_col(df, ["onset_frame_idx", "onset_frame", "gt_onset_idx", "frame_id"])
    start_col = _pick_col(df, ["onset_start", "gt_onset_start"])
    end_col = _pick_col(df, ["onset_end", "gt_onset_end"])

    if category_col is None:
        raise ValueError(
            "annotation 需含 ignition_category，或使用 effective_trace 格式 "
            "(track_id + frame_id；非空=点火，空/未列出=未点火)，"
            "或 effective_trace_new 单列无效 id + --reference-summary"
        )

    for _, row in df.iterrows():
        raw = row[track_name_col] if track_name_col is not None and pd.notna(row[track_name_col]) else (
            row[track_id_col] if track_id_col is not None else None
        )
        track_name = normalize_track_name(raw)
        if not track_name:
            continue

        cat = str(row[category_col]).strip().lower()
        onset = -1
        if onset_col is not None and pd.notna(row[onset_col]):
            onset = int(float(row[onset_col]))
        onset_start = -1
        onset_end = -1
        if start_col is not None and pd.notna(row[start_col]):
            onset_start = int(float(row[start_col]))
        if end_col is not None and pd.notna(row[end_col]):
            onset_end = int(float(row[end_col]))

        ann[track_name] = {
            "gt_category": cat,
            "gt_has_ignition": cat == "ignition",
            "gt_onset_idx": onset,
            "gt_onset_start": onset_start,
            "gt_onset_end": onset_end,
        }
    return ann, "category"


def lookup_annotation(
    annotations: Dict[str, Dict[str, Any]],
    track_name: str,
    track_id: Optional[int],
    *,
    missing_as_no_ignition: bool = False,
    missing_as_invalid: bool = False,
) -> Dict[str, Any]:
    name = normalize_track_name(track_name)
    if name in annotations:
        return dict(annotations[name])
    if track_id is not None:
        alt = normalize_track_name(track_id)
        if alt in annotations:
            return dict(annotations[alt])
        alt2 = f"track_{track_id}"
        if alt2 in annotations:
            return dict(annotations[alt2])
    if missing_as_invalid:
        return _invalid_gt()
    if missing_as_no_ignition:
        return _no_ignition_gt()
    return {
        "gt_category": "",
        "gt_has_ignition": False,
        "gt_onset_idx": -1,
        "gt_onset_start": -1,
        "gt_onset_end": -1,
    }


# ============================================================
# Result helpers
# ============================================================

def _empty_result(mod, cfg, n: int, method: str, status: str = "ambiguous") -> Dict[str, Any]:
    no_idx = cfg.no_ignition_index
    hot = mod.compute_hot_score(pd.DataFrame({"gray_p99": np.zeros(n)})) if n == 0 else np.zeros(n, dtype=np.float32)
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
        "hmm_n_states": 0,
        "hmm_state_sequence": [],
        "hot_score": hot,
        "change_score": np.zeros(n, dtype=np.float32),
        "ignition_method": method,
    }


def _pack_onset_result(
    mod,
    cfg,
    df: pd.DataFrame,
    method: str,
    status: str,
    has_ignition: bool,
    onset_idx: int,
    confirmed_idx: int,
    change_score: np.ndarray,
    hot_score: np.ndarray,
    confidence: float = 0.0,
    amplitude: float = 0.0,
    change_peak: float = 0.0,
    hmm_n_states: int = 0,
    state_seq: Optional[List[int]] = None,
) -> Dict[str, Any]:
    no_idx = cfg.no_ignition_index
    stable_idx = no_idx
    if has_ignition and onset_idx >= 0 and state_seq:
        stable = mod.find_stable_combustion_frame(
            np.asarray(state_seq, dtype=np.int32),
            onset_idx,
            cfg.hmm_min_stable_run,
        )
        if stable is not None:
            stable_idx = int(stable)

    filename_at = lambda idx: str(df.loc[idx, "filename"]) if 0 <= idx < len(df) and "filename" in df.columns else ""

    return {
        "has_ignition": bool(has_ignition),
        "ignition_status": status,
        "ignition_idx_onset": int(onset_idx),
        "ignition_idx_confirmed": int(confirmed_idx),
        "ignition_idx_candidate": int(onset_idx),
        "stable_combustion_idx": int(stable_idx),
        "ignition_confidence": float(confidence),
        "ignition_amplitude": float(amplitude),
        "ignition_change_peak": float(change_peak),
        "hmm_n_states": int(hmm_n_states),
        "hmm_state_sequence": state_seq or [],
        "hot_score": hot_score,
        "change_score": change_score,
        "ignition_method": method,
        "candidate_filename": filename_at(onset_idx) if has_ignition else "",
        "confirmed_filename": filename_at(confirmed_idx) if has_ignition else "",
    }


def build_raw_feature_matrix(df: pd.DataFrame, feature_cols: List[str], mod) -> np.ndarray:
    cols = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(np.float32)
        cols.append(mod.normalize_to_01(vals))
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


def compute_common_raw_change_score(df: pd.DataFrame, mod, window: int = 3) -> np.ndarray:
    """统一 onset refinement 使用的 raw thermal change score。

    该分数仅由原始物理热特征构成，不随 GMM/HMM 输入特征组改变。
    因此模块消融中 GMM-only 与 HMM-GMM 的 onset 差异不会被特征维度偏置放大。
    """
    X = build_raw_feature_matrix(df, CHANGE_RAW_FEATURES, mod)
    return mod.compute_hmm_change_score(X, window=window)


def count_statuses(series: pd.Series, statuses: List[str]) -> Dict[str, int]:
    values = series.astype(str) if len(series) else pd.Series(dtype=str)
    return {f"n_status_{st}": int((values == st).sum()) for st in statuses}


def find_earliest_persistent_onset(
    score: np.ndarray,
    threshold: float,
    min_persist: int,
) -> Optional[int]:
    n = len(score)
    if n < min_persist:
        return None
    for t in range(n - min_persist + 1):
        if np.all(score[t : t + min_persist] >= threshold):
            return t
    return None


def is_already_hot_by_raw_initial_window(
    df: pd.DataFrame,
    init_n: int,
    gray_p99_threshold: float,
    hsv_v_p99_threshold: float,
) -> bool:
    """用初始窗口 raw gray_p99 / hsv_v_p99 均值判断 already_hot（非局部归一化）。"""
    if init_n <= 0 or len(df) == 0:
        return False
    n = min(init_n, len(df))
    checks: List[bool] = []
    if "gray_p99" in df.columns:
        gray_mean = float(np.mean(df["gray_p99"].values[:n].astype(np.float32)))
        checks.append(gray_mean >= gray_p99_threshold)
    if "hsv_v_p99" in df.columns:
        hsv_mean = float(np.mean(df["hsv_v_p99"].values[:n].astype(np.float32)))
        checks.append(hsv_mean >= hsv_v_p99_threshold)
    if not checks:
        return False
    return any(checks)


# ============================================================
# Method variants
# ============================================================

def detect_gray_p99_threshold(
    df: pd.DataFrame,
    mod,
    cfg,
    *,
    threshold: float = 0.5,
    min_persist: int = 3,
    already_hot_gray_p99_threshold: float = 120.0,
    already_hot_hsv_v_p99_threshold: float = 120.0,
) -> Dict[str, Any]:
    n = len(df)
    method = "gray_p99_threshold"
    if n == 0:
        return _empty_result(mod, cfg, 0, method)

    hot_score = mod.compute_hot_score(df)
    seq = mod.normalize_to_01(df["gray_p99"].values.astype(np.float32))
    change_score = np.zeros(n, dtype=np.float32)
    init_n = min(cfg.initial_hot_check_frames, n)

    if is_already_hot_by_raw_initial_window(
        df, init_n, already_hot_gray_p99_threshold, already_hot_hsv_v_p99_threshold
    ):
        return _pack_onset_result(
            mod, cfg, df, method, "already_hot", False, cfg.no_ignition_index,
            cfg.no_ignition_index, change_score, hot_score,
        )

    onset = find_earliest_persistent_onset(seq, threshold, min_persist)
    if onset is None:
        return _pack_onset_result(
            mod, cfg, df, method, "no_ignition", False, cfg.no_ignition_index,
            cfg.no_ignition_index, change_score, hot_score,
        )

    pre_mean = float(np.mean(hot_score[max(0, onset - 3):onset])) if onset > 0 else 0.0
    post_mean = float(np.mean(hot_score[onset:min(n, onset + 3)]))
    return _pack_onset_result(
        mod, cfg, df, method, "ignition_detected", True, onset, onset,
        change_score, hot_score,
        confidence=float(np.clip(seq[onset], 0, 1)),
        amplitude=post_mean - pre_mean,
        change_peak=float(seq[onset]),
    )


def detect_multi_feature_change_score(
    df: pd.DataFrame,
    mod,
    cfg,
    *,
    change_threshold: float = 0.25,
    min_persist: int = 3,
    already_hot_gray_p99_threshold: float = 120.0,
    already_hot_hsv_v_p99_threshold: float = 120.0,
) -> Dict[str, Any]:
    n = len(df)
    method = "multi_feature_change_score"
    if n < cfg.hmm_min_sequence_len:
        return _empty_result(mod, cfg, n, method, "ambiguous")

    hot_score = mod.compute_hot_score(df)
    X = build_raw_feature_matrix(df, CHANGE_RAW_FEATURES, mod)
    change_score = mod.compute_hmm_change_score(X, window=3)

    init_n = min(cfg.initial_hot_check_frames, n)
    if is_already_hot_by_raw_initial_window(
        df, init_n, already_hot_gray_p99_threshold, already_hot_hsv_v_p99_threshold
    ):
        return _pack_onset_result(
            mod, cfg, df, method, "already_hot", False, cfg.no_ignition_index,
            cfg.no_ignition_index, change_score, hot_score,
        )

    candidate = find_earliest_persistent_onset(change_score, change_threshold, min_persist)
    if candidate is None:
        return _pack_onset_result(
            mod, cfg, df, method, "no_ignition", False, cfg.no_ignition_index,
            cfg.no_ignition_index, change_score, hot_score,
        )

    gate_cfg = replace(cfg, enable_raw_transition_gate=True)
    has_raw, _ = mod.raw_transition_has_thermal_increase(df, candidate, gate_cfg)
    if not has_raw:
        return _pack_onset_result(
            mod, cfg, df, method, "no_ignition_weak_transition", False,
            cfg.no_ignition_index, cfg.no_ignition_index, change_score, hot_score,
        )

    pre_mean = float(np.mean(hot_score[max(0, candidate - 3):candidate])) if candidate > 0 else 0.0
    post_mean = float(np.mean(hot_score[candidate:min(n, candidate + 3)]))
    return _pack_onset_result(
        mod, cfg, df, method, "ignition_detected", True, candidate, candidate,
        change_score, hot_score,
        confidence=float(change_score[candidate]),
        amplitude=post_mean - pre_mean,
        change_peak=float(change_score[candidate]),
    )


def detect_gmm_only(
    df: pd.DataFrame,
    mod,
    cfg,
    *,
    method: str = "gmm_only",
) -> Dict[str, Any]:
    """GMM + BIC + thermal ranking，无 HMM Viterbi。gate 由 cfg 控制。"""
    n = len(df)
    if n < cfg.hmm_min_sequence_len:
        return _empty_result(mod, cfg, n, method, "ambiguous")

    if cfg.enable_low_signal_reject:
        is_black, low_signal_metrics = mod.is_low_signal_black_sequence(df, cfg)
        if is_black:
            r = _empty_result(mod, cfg, n, method, "no_ignition_black_sequence")
            r["low_signal_metrics"] = low_signal_metrics
            return r

    Xn, feature_cols = mod.build_hmm_gmm_feature_matrix(df, cfg)
    gmm, selected_k, gmm_info, all_models = mod.fit_gmm_models_bic(
        Xn,
        max_states=cfg.hmm_max_states,
        covariance_type=cfg.hmm_covariance_type,
        reg_covar=cfg.hmm_reg_covar,
        n_init=cfg.hmm_n_init,
        bic_improvement_ratio=cfg.hmm_bic_improvement_ratio,
        random_state=cfg.random_state,
    )
    gmm, selected_k, gmm_info = mod.select_gmm_with_occupancy_validation(
        Xn,
        all_models,
        gmm_info,
        min_state_count=cfg.hmm_min_state_count,
        min_state_fraction=cfg.hmm_min_state_fraction,
    )
    # 使用统一 raw thermal cue 进行 onset refinement，避免高维/低维特征组自身 change_score 造成偏置。
    change_score = compute_common_raw_change_score(df, mod, window=3)
    hot_score = mod.compute_hot_score(df)

    if selected_k == 1:
        thermal_scores = mod.compute_state_thermal_scores(gmm, feature_cols)
        states_zero = np.zeros(n, dtype=np.int32)
        result = mod.interpret_hmm_state_sequence_for_ignition(
            states_zero, df, change_score, cfg, gmm_info, thermal_scores
        )
        result["ignition_method"] = method
        result["hot_score"] = hot_score
        result["change_score"] = change_score
        result["hmm_state_sequence"] = [0] * n
        result["hmm_feature_cols"] = feature_cols
        result["ablation_enable_low_signal_reject"] = bool(cfg.enable_low_signal_reject)
        result["ablation_enable_raw_transition_gate"] = bool(cfg.enable_raw_transition_gate)
        return result

    thermal_scores = mod.compute_state_thermal_scores(gmm, feature_cols)
    label_to_rank, _ = mod.build_state_rank_mapping(thermal_scores)
    labels = gmm.predict(Xn)
    states_rank = np.array([label_to_rank[int(l)] for l in labels], dtype=np.int32)
    states_rank = mod.smooth_state_sequence_by_run_length(states_rank, cfg.hmm_min_state_run)

    result = mod.interpret_hmm_state_sequence_for_ignition(
        states_rank, df, change_score, cfg, gmm_info, thermal_scores
    )
    result["ignition_method"] = method
    result["hmm_state_sequence"] = states_rank.tolist()
    result["hot_score"] = hot_score
    result["change_score"] = change_score
    result["hmm_feature_cols"] = feature_cols
    result["ablation_enable_low_signal_reject"] = bool(cfg.enable_low_signal_reject)
    result["ablation_enable_raw_transition_gate"] = bool(cfg.enable_raw_transition_gate)
    return result

def detect_hmm_variant(
    df: pd.DataFrame,
    mod,
    cfg,
    *,
    method: str,
    enable_low_signal_reject: bool,
    enable_raw_transition_gate: bool,
) -> Dict[str, Any]:
    """HMM-GMM variant used in module ablation.

    This reimplements mod.detect_ignition_hmm_gmm with one deliberate change:
    all HMM-GMM variants use the same raw thermal change score for onset
    refinement. The GMM/HMM feature matrix is still the default feature matrix
    from ignition_detection_hmm_gmm.py.
    """
    local_cfg = replace(
        cfg,
        enable_low_signal_reject=enable_low_signal_reject,
        enable_raw_transition_gate=enable_raw_transition_gate,
    )
    n = len(df)
    if n < local_cfg.hmm_min_sequence_len:
        return _empty_result(mod, local_cfg, n, method, "ambiguous")

    if local_cfg.enable_low_signal_reject:
        is_black, low_signal_metrics = mod.is_low_signal_black_sequence(df, local_cfg)
        if is_black:
            r = _empty_result(mod, local_cfg, n, method, "no_ignition_black_sequence")
            r["low_signal_metrics"] = low_signal_metrics
            return r

    Xn, feature_cols = mod.build_hmm_gmm_feature_matrix(df, local_cfg)
    gmm, selected_k, gmm_info, all_models = mod.fit_gmm_models_bic(
        Xn,
        max_states=local_cfg.hmm_max_states,
        covariance_type=local_cfg.hmm_covariance_type,
        reg_covar=local_cfg.hmm_reg_covar,
        n_init=local_cfg.hmm_n_init,
        bic_improvement_ratio=local_cfg.hmm_bic_improvement_ratio,
        random_state=local_cfg.random_state,
    )
    gmm, selected_k, gmm_info = mod.select_gmm_with_occupancy_validation(
        Xn,
        all_models,
        gmm_info,
        min_state_count=local_cfg.hmm_min_state_count,
        min_state_fraction=local_cfg.hmm_min_state_fraction,
    )

    # 同一 raw onset score 用于所有 HMM/GMM 模块消融，保证公平比较。
    change_score = compute_common_raw_change_score(df, mod, window=3)
    hot_score = mod.compute_hot_score(df)
    thermal_scores = mod.compute_state_thermal_scores(gmm, feature_cols)

    if selected_k == 1:
        states_zero = np.zeros(n, dtype=np.int32)
        result = mod.interpret_hmm_state_sequence_for_ignition(
            states_zero, df, change_score, local_cfg, gmm_info, thermal_scores
        )
        result["hmm_state_sequence"] = [0] * n
    else:
        _label_to_rank, rank_to_label = mod.build_state_rank_mapping(thermal_scores)
        log_em_label = mod.gmm_diag_log_emission(Xn, gmm)
        log_em_rank = mod.reorder_log_emission_by_rank(log_em_label, rank_to_label)
        log_trans = mod.build_left_to_right_transition_logprob(
            selected_k,
            local_cfg.hmm_self_prob,
            local_cfg.hmm_forward_prob,
            local_cfg.hmm_backward_eps,
        )
        init_logprob = mod.make_adaptive_init_logprob(log_em_rank[0])
        states_rank = mod.viterbi_decode(log_em_rank, log_trans, init_logprob=init_logprob)
        states_rank = mod.smooth_state_sequence_by_run_length(states_rank, local_cfg.hmm_min_state_run)
        result = mod.interpret_hmm_state_sequence_for_ignition(
            states_rank, df, change_score, local_cfg, gmm_info, thermal_scores
        )
        result["hmm_state_sequence"] = states_rank.tolist()

    result["ignition_method"] = method
    result["hot_score"] = hot_score
    result["change_score"] = change_score
    result["hmm_feature_cols"] = feature_cols
    result["ablation_enable_low_signal_reject"] = enable_low_signal_reject
    result["ablation_enable_raw_transition_gate"] = enable_raw_transition_gate
    return result

METHOD_DISPATCH = {
    "gray_p99_threshold": "baseline_gray",
    "multi_feature_change_score": "baseline_change",
    "gmm_only": "gmm_only",
    "gmm_only_raw_gate": "gmm_only_raw_gate",
    "hmm_gmm": "hmm_gmm",
    "hmm_gmm_low_signal": "hmm_gmm_low_signal",
    "hmm_gmm_raw_gate": "hmm_gmm_raw_gate",
    "proposed_full": "proposed_full",
}


def run_method(
    method: str,
    df: pd.DataFrame,
    mod,
    cfg,
    exp_params: Dict[str, Any],
) -> Dict[str, Any]:
    if method == "gray_p99_threshold":
        return detect_gray_p99_threshold(
            df, mod, cfg,
            threshold=float(exp_params["gray_threshold"]),
            min_persist=int(exp_params.get("gray_min_persist", exp_params["min_persist"])),
            already_hot_gray_p99_threshold=float(exp_params["already_hot_gray_p99_threshold"]),
            already_hot_hsv_v_p99_threshold=float(exp_params["already_hot_hsv_v_p99_threshold"]),
        )
    if method == "multi_feature_change_score":
        return detect_multi_feature_change_score(
            df, mod, cfg,
            change_threshold=float(exp_params["change_threshold"]),
            min_persist=int(exp_params.get("change_min_persist", exp_params["min_persist"])),
            already_hot_gray_p99_threshold=float(exp_params["already_hot_gray_p99_threshold"]),
            already_hot_hsv_v_p99_threshold=float(exp_params["already_hot_hsv_v_p99_threshold"]),
        )
    if method == "gmm_only":
        local_cfg = replace(
            cfg,
            enable_low_signal_reject=False,
            enable_raw_transition_gate=False,
        )
        return detect_gmm_only(df, mod, local_cfg, method=method)
    if method == "gmm_only_raw_gate":
        local_cfg = replace(
            cfg,
            enable_low_signal_reject=False,
            enable_raw_transition_gate=True,
        )
        return detect_gmm_only(df, mod, local_cfg, method=method)
    if method == "hmm_gmm":
        return detect_hmm_variant(
            df, mod, cfg, method=method,
            enable_low_signal_reject=False,
            enable_raw_transition_gate=False,
        )
    if method == "hmm_gmm_low_signal":
        return detect_hmm_variant(
            df, mod, cfg, method=method,
            enable_low_signal_reject=True,
            enable_raw_transition_gate=False,
        )
    if method == "hmm_gmm_raw_gate":
        return detect_hmm_variant(
            df, mod, cfg, method=method,
            enable_low_signal_reject=False,
            enable_raw_transition_gate=True,
        )
    if method == "proposed_full":
        return detect_hmm_variant(
            df, mod, cfg, method=method,
            enable_low_signal_reject=False,
            enable_raw_transition_gate=True,
        )
    raise ValueError(f"Unknown method: {method}")


# ============================================================
# Evaluation
# ============================================================

def evaluate_prediction(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
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
    is_correct_presence = np.nan
    is_false_ignition = False
    is_missed_ignition = False
    is_already_hot_correct = False

    if gt_cat == "ignition":
        is_correct_presence = pred_has
        is_missed_ignition = not pred_has
        if pred_has and gt_onset >= 0 and pred_onset >= 0:
            onset_abs_error = abs(pred_onset - gt_onset)
            if gt_start >= 0 and gt_end >= 0 and gt_end >= gt_start:
                onset_inside_interval = gt_start <= pred_onset <= gt_end
            else:
                onset_inside_interval = np.nan
    elif gt_cat == "no_ignition":
        is_correct_presence = (not pred_has) or (pred_status in NO_IGNITION_STATUSES)
        is_false_ignition = pred_has and pred_status not in NO_IGNITION_STATUSES
    elif gt_cat == "already_hot":
        is_already_hot_correct = pred_status == "already_hot"
        is_correct_presence = is_already_hot_correct
    elif gt_cat:
        is_correct_presence = np.nan

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
    method: str,
    n_frames: int,
    pred: Dict[str, Any],
    eval_fields: Dict[str, Any],
    error_message: str = "",
) -> Dict[str, Any]:
    return to_py({
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
        "hmm_n_states": int(pred.get("hmm_n_states", 0)),
        "hmm_feature_cols": ";".join(str(c) for c in pred.get("hmm_feature_cols", []) or []),
        "ablation_enable_low_signal_reject": bool(pred.get("ablation_enable_low_signal_reject", False)),
        "ablation_enable_raw_transition_gate": bool(pred.get("ablation_enable_raw_transition_gate", False)),
        **eval_fields,
        "error_message": error_message,
    })


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.mean())


def _safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.median())


def compute_method_summary(
    per_track_df: pd.DataFrame,
    include_ambiguous: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, sub in per_track_df.groupby("method", sort=False):
        valid = sub.copy()
        if "gt_category" in valid.columns:
            valid = valid[valid["gt_category"].astype(str).str.strip() != ""]
            if not include_ambiguous:
                valid = valid[~valid["gt_category"].isin(EXCLUDED_GT_CATEGORIES)]
            eval_valid = valid[valid["gt_category"].isin(PRIMARY_GT_CATEGORIES)]
        else:
            eval_valid = valid.iloc[0:0]

        gt_ign = eval_valid[eval_valid["gt_category"] == "ignition"]
        gt_no = eval_valid[eval_valid["gt_category"] == "no_ignition"]
        gt_hot = eval_valid[eval_valid["gt_category"] == "already_hot"]

        tp = int((gt_ign["pred_has_ignition"] == True).sum())  # noqa: E712
        fn = int((gt_ign["pred_has_ignition"] == False).sum())  # noqa: E712
        fp = int((gt_no["pred_has_ignition"] == True).sum())  # noqa: E712

        precision = _safe_rate(tp, tp + fp)
        recall = _safe_rate(tp, tp + fn)
        f1 = float("nan")
        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        onset_hits = gt_ign[gt_ign["pred_has_ignition"] == True]  # noqa: E712
        interval_subset = onset_hits[pd.notna(onset_hits["onset_inside_interval"])]

        method_errors = sub[
            sub["pred_ignition_status"].astype(str).isin(["method_error", "track_error"])
        ]
        n_method_errors = int(len(method_errors))
        n_tracks = int(len(sub))
        method_error_rate = float(n_method_errors / n_tracks) if n_tracks > 0 else float("nan")
        status_stats = count_statuses(
            sub["pred_ignition_status"],
            [
                "ignition_detected",
                "no_ignition",
                "no_ignition_black_sequence",
                "no_ignition_weak_transition",
                "already_hot",
                "single_state",
                "ambiguous",
                "ambiguous_gmm_unstable",
                "method_error",
                "track_error",
            ],
        )
        k_counts = sub["hmm_n_states"].value_counts(dropna=False).to_dict() if "hmm_n_states" in sub else {}

        rows.append({
            "method": method,
            "n_tracks": n_tracks,
            "n_valid_gt": int(len(eval_valid)),
            "n_method_errors": n_method_errors,
            "method_error_rate": method_error_rate,
            "mean_hmm_n_states": _safe_mean(sub["hmm_n_states"]),
            "n_k0": int(k_counts.get(0, 0)),
            "n_k1": int(k_counts.get(1, 0)),
            "n_k2": int(k_counts.get(2, 0)),
            "n_k3": int(k_counts.get(3, 0)),
            **status_stats,
            "ignition_precision": precision,
            "ignition_recall": recall,
            "ignition_f1": f1,
            "false_ignition_rate": _safe_rate(fp, len(gt_no)),
            "missed_ignition_rate": _safe_rate(fn, len(gt_ign)),
            "already_hot_accuracy": _safe_rate(
                int((gt_hot["is_already_hot_correct"] == True).sum()), len(gt_hot)  # noqa: E712
            ),
            "no_ignition_false_alarm_rate": _safe_rate(fp, len(gt_no)),
            "onset_mae": _safe_mean(onset_hits["onset_abs_error"]),
            "onset_median_abs_error": _safe_median(onset_hits["onset_abs_error"]),
            "within_1_frame_acc": _safe_rate(
                int((onset_hits["onset_abs_error"] <= 1).sum()), len(onset_hits)
            ),
            "within_3_frame_acc": _safe_rate(
                int((onset_hits["onset_abs_error"] <= 3).sum()), len(onset_hits)
            ),
            "inside_interval_acc": _safe_rate(
                int((interval_subset["onset_inside_interval"] == True).sum()), len(interval_subset)  # noqa: E712
            ),
        })

    summary = pd.DataFrame(rows)
    method_order = list(METHOD_DISPATCH.keys())
    summary["method"] = pd.Categorical(summary["method"], categories=method_order, ordered=True)
    return summary.sort_values("method").reset_index(drop=True)


def format_method_result_cell(row: pd.Series) -> str:
    """紧凑格式：Y:12 / N / already_hot / ERR / ambiguous 等。"""
    status = str(row.get("pred_ignition_status", ""))
    if status in ("method_error", "track_error"):
        return "ERR"
    if status == "already_hot":
        return "already_hot"
    if status in NO_IGNITION_STATUSES:
        return "N"
    has_ign = bool(row.get("pred_has_ignition", False))
    onset = int(row.get("pred_onset_idx", -1))
    if has_ign and onset >= 0:
        return f"Y:{onset}"
    if has_ign:
        return "Y"
    if status and status not in ("ignition_detected", ""):
        return status
    return "N"


def build_method_track_comparison_table(
    per_track_df: pd.DataFrame,
    methods: List[str],
) -> pd.DataFrame:
    """
    按 track 横向对比各方法的点火判定与 onset 位置。

    每个 method 列格式：
        Y:12  — 检测到点火，onset 在 crop 序列第 12 帧（0-based）
        N     — 未检测到点火
        already_hot / ambiguous / ERR 等 — 其他判定
    """
    if per_track_df.empty:
        return pd.DataFrame(columns=["track_name", "track_id"] + list(methods))

    rows: List[Dict[str, Any]] = []
    group_cols = ["track_name", "track_id"]
    for key, group in per_track_df.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            track_name, track_id = key
        else:
            track_name, track_id = key, group["track_id"].iloc[0]
        row: Dict[str, Any] = {"track_name": track_name, "track_id": track_id}
        for method in methods:
            sub = group[group["method"] == method]
            row[method] = format_method_result_cell(sub.iloc[0]) if len(sub) else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["track_id", "track_name"], na_position="last").reset_index(drop=True)
    col_order = ["track_name", "track_id"] + [m for m in methods if m in df.columns]
    return df[col_order]


def is_error_case(row: pd.Series) -> bool:
    gt_cat = str(row.get("gt_category", "")).lower()
    if not gt_cat or gt_cat in EXCLUDED_GT_CATEGORIES:
        return False
    if gt_cat == "ignition":
        if not bool(row.get("pred_has_ignition", False)):
            return True
        err = row.get("onset_abs_error")
        if pd.notna(err) and float(err) > 3:
            return True
        inside = row.get("onset_inside_interval")
        if pd.notna(inside) and not bool(inside):
            return True
    elif gt_cat == "no_ignition":
        if bool(row.get("is_false_ignition", False)):
            return True
    elif gt_cat == "already_hot":
        if not bool(row.get("is_already_hot_correct", False)):
            return True
    return False


# ============================================================
# Track processing
# ============================================================

def build_base_cfg(args, mod) -> Any:
    return mod.IgnitionHMMConfig(
        hmm_max_states=args.hmm_max_states,
        hmm_self_prob=args.hmm_self_prob,
        hmm_forward_prob=args.hmm_forward_prob,
        hmm_min_state_run=args.hmm_min_state_run,
        hmm_min_cold_run=args.hmm_min_cold_run,
        hmm_min_hot_run=args.hmm_min_hot_run,
        hmm_min_state_count=args.hmm_min_state_count,
        hmm_min_state_fraction=args.hmm_min_state_fraction,
        hmm_bic_improvement_ratio=args.hmm_bic_improvement_ratio,
        enable_low_signal_reject=False,
    )


def process_one_track(
    track_name: str,
    folder: str,
    methods: List[str],
    mod,
    cfg,
    exp_params: Dict[str, Any],
    annotations: Dict[str, Dict[str, Any]],
    save_error_cases: bool,
    error_cases_dir: Optional[Path],
    missing_as_no_ignition: bool = False,
    missing_as_invalid: bool = False,
    precomputed_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    track_id = mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        annotations,
        track_name,
        track_id,
        missing_as_no_ignition=missing_as_no_ignition,
        missing_as_invalid=missing_as_invalid,
    )

    try:
        if precomputed_df is not None:
            df = precomputed_df
            n_frames = len(df)
        else:
            image_paths = mod.list_images_in_folder(folder)
            if not image_paths:
                raise RuntimeError("No images found in track folder")
            df = mod.extract_sequence_features(image_paths, cfg)
            n_frames = len(df)

        for method in methods:
            try:
                pred = run_method(method, df, mod, cfg, exp_params)
                eval_fields = evaluate_prediction(pred, gt)
                rows.append(result_to_row(track_name, track_id, method, n_frames, pred, eval_fields))

                if save_error_cases and error_cases_dir is not None:
                    row = rows[-1]
                    if is_error_case(pd.Series(row)):
                        out_dir = error_cases_dir / method
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_path = str(out_dir / f"{track_name}.png")
                        try:
                            mod.plot_detection_result_hmm_gmm(
                                df, pred, save_path=save_path, show=False
                            )
                        except Exception as plot_exc:
                            rows[-1]["error_message"] = f"plot_failed: {plot_exc}"
            except Exception as exc:
                eval_fields = evaluate_prediction({}, gt)
                rows.append(result_to_row(
                    track_name, track_id, method, n_frames,
                    _empty_result(mod, cfg, n_frames, method, "method_error"),
                    eval_fields,
                    error_message=str(exc),
                ))
    except Exception as exc:
        for method in methods:
            eval_fields = evaluate_prediction({}, gt)
            rows.append(result_to_row(
                track_name, track_id, method, 0,
                _empty_result(mod, cfg, 0, method, "track_error"),
                eval_fields,
                error_message=str(exc),
            ))
    return rows


_WORKER_CTX: Dict[str, Any] = {}


def _init_worker(ctx: Dict[str, Any]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = {
        "mod": load_ignition_module(Path(ctx["ignition_module_path"])),
        "cfg": None,
        "exp_params": ctx["exp_params"],
        "annotations": ctx["annotations"],
        "methods": ctx["methods"],
        "save_error_cases": ctx["save_error_cases"],
        "error_cases_dir": Path(ctx["error_cases_dir"]) if ctx["error_cases_dir"] else None,
        "missing_as_no_ignition": bool(ctx.get("missing_as_no_ignition", False)),
        "missing_as_invalid": bool(ctx.get("missing_as_invalid", False)),
    }
    mod = _WORKER_CTX["mod"]
    _WORKER_CTX["cfg"] = mod.IgnitionHMMConfig(**ctx["cfg_dict"])


def _worker_process_track(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    return process_one_track(
        track_name=task["track_name"],
        folder=task["folder"],
        methods=_WORKER_CTX["methods"],
        mod=_WORKER_CTX["mod"],
        cfg=_WORKER_CTX["cfg"],
        exp_params=_WORKER_CTX["exp_params"],
        annotations=_WORKER_CTX["annotations"],
        save_error_cases=_WORKER_CTX["save_error_cases"],
        error_cases_dir=_WORKER_CTX["error_cases_dir"],
        missing_as_no_ignition=_WORKER_CTX["missing_as_no_ignition"],
        missing_as_invalid=_WORKER_CTX["missing_as_invalid"],
    )


# ============================================================
# Baseline hyperparameter tuning
# ============================================================

def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def _metric_sort_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    """越大越好：F1 → onset interval acc → -onset MAE。"""
    f1 = row.get("ignition_f1", float("nan"))
    inside = row.get("inside_interval_acc", float("nan"))
    mae = row.get("onset_mae", float("nan"))
    f1_v = float(f1) if pd.notna(f1) else -1.0
    inside_v = float(inside) if pd.notna(inside) else -1.0
    mae_v = float(mae) if pd.notna(mae) else 1e9
    return (f1_v, inside_v, -mae_v)


def _summary_row_to_dict(summary_df: pd.DataFrame, method: str) -> Dict[str, Any]:
    if summary_df.empty:
        return {
            "method": method,
            "ignition_f1": float("nan"),
            "ignition_precision": float("nan"),
            "ignition_recall": float("nan"),
            "inside_interval_acc": float("nan"),
            "onset_mae": float("nan"),
            "n_valid_gt": 0,
        }
    sub = summary_df[summary_df["method"].astype(str) == method]
    if sub.empty:
        return {
            "method": method,
            "ignition_f1": float("nan"),
            "ignition_precision": float("nan"),
            "ignition_recall": float("nan"),
            "inside_interval_acc": float("nan"),
            "onset_mae": float("nan"),
            "n_valid_gt": 0,
        }
    return sub.iloc[0].to_dict()


def evaluate_baseline_params_on_cache(
    method: str,
    cache: List[Dict[str, Any]],
    mod,
    cfg,
    exp_params: Dict[str, Any],
    tune_track_names: Optional[set] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for item in cache:
        if not item.get("ok"):
            continue
        name = item["track_name"]
        if tune_track_names is not None and name not in tune_track_names:
            continue
        gt = item["gt"]
        gt_cat = str(gt.get("gt_category", ""))
        if gt_cat in EXCLUDED_GT_CATEGORIES or gt_cat not in PRIMARY_GT_CATEGORIES:
            # 仍写入以便 n_tracks 统计，但 summary 会按类别过滤
            pass
        try:
            pred = run_method(method, item["df"], mod, cfg, exp_params)
            eval_fields = evaluate_prediction(pred, gt)
            rows.append(result_to_row(
                name, item.get("track_id"), method, int(item.get("n_frames", len(item["df"]))),
                pred, eval_fields,
            ))
        except Exception as exc:
            eval_fields = evaluate_prediction({}, gt)
            n = int(item.get("n_frames", 0))
            rows.append(result_to_row(
                name, item.get("track_id"), method, n,
                _empty_result(mod, cfg, n, method, "method_error"),
                eval_fields,
                error_message=str(exc),
            ))
    per_track = pd.DataFrame(rows)
    summary = compute_method_summary(per_track, include_ambiguous=False)
    return _summary_row_to_dict(summary, method)


_EXTRACT_CTX: Dict[str, Any] = {}


def _init_extract_worker(ctx: Dict[str, Any]) -> None:
    global _EXTRACT_CTX
    mod = load_ignition_module(Path(ctx["ignition_module_path"]))
    _EXTRACT_CTX = {
        "mod": mod,
        "cfg": mod.IgnitionHMMConfig(**ctx["cfg_dict"]),
        "annotations": ctx["annotations"],
        "missing_as_no_ignition": bool(ctx.get("missing_as_no_ignition", False)),
        "missing_as_invalid": bool(ctx.get("missing_as_invalid", False)),
    }


def _extract_one_track_task(task: Dict[str, Any]) -> Dict[str, Any]:
    mod = _EXTRACT_CTX["mod"]
    cfg = _EXTRACT_CTX["cfg"]
    track_name = task["track_name"]
    folder = task["folder"]
    track_id = mod.track_id_from_name(track_name)
    gt = lookup_annotation(
        _EXTRACT_CTX["annotations"],
        track_name,
        track_id,
        missing_as_no_ignition=_EXTRACT_CTX["missing_as_no_ignition"],
        missing_as_invalid=_EXTRACT_CTX["missing_as_invalid"],
    )
    try:
        image_paths = mod.list_images_in_folder(folder)
        if not image_paths:
            raise RuntimeError("No images found in track folder")
        df = mod.extract_sequence_features(image_paths, cfg)
        return {
            "ok": True,
            "track_name": track_name,
            "track_id": track_id,
            "folder": folder,
            "df": df,
            "gt": gt,
            "n_frames": len(df),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "track_name": track_name,
            "track_id": track_id,
            "folder": folder,
            "df": None,
            "gt": gt,
            "n_frames": 0,
            "error": str(exc),
        }


def extract_track_feature_cache(
    tasks: List[Dict[str, Any]],
    mod,
    cfg,
    annotations: Dict[str, Dict[str, Any]],
    *,
    missing_as_no_ignition: bool,
    missing_as_invalid: bool,
    n_workers: int,
    ignition_module_path: Path,
) -> List[Dict[str, Any]]:
    cache: List[Dict[str, Any]] = []
    if n_workers == 1:
        for idx, task in enumerate(tasks, start=1):
            if idx == 1 or idx % 200 == 0 or idx == len(tasks):
                print(f"[INFO] feature extract [{idx}/{len(tasks)}]")
            track_name = task["track_name"]
            folder = task["folder"]
            track_id = mod.track_id_from_name(track_name)
            gt = lookup_annotation(
                annotations, track_name, track_id,
                missing_as_no_ignition=missing_as_no_ignition,
                missing_as_invalid=missing_as_invalid,
            )
            try:
                image_paths = mod.list_images_in_folder(folder)
                if not image_paths:
                    raise RuntimeError("No images found in track folder")
                df = mod.extract_sequence_features(image_paths, cfg)
                cache.append({
                    "ok": True, "track_name": track_name, "track_id": track_id,
                    "folder": folder, "df": df, "gt": gt, "n_frames": len(df), "error": "",
                })
            except Exception as exc:
                cache.append({
                    "ok": False, "track_name": track_name, "track_id": track_id,
                    "folder": folder, "df": None, "gt": gt, "n_frames": 0, "error": str(exc),
                })
        return cache

    ctx = {
        "ignition_module_path": str(Path(ignition_module_path).resolve()),
        "cfg_dict": asdict(cfg),
        "annotations": annotations,
        "missing_as_no_ignition": missing_as_no_ignition,
        "missing_as_invalid": missing_as_invalid,
    }
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_extract_worker, initargs=(ctx,)) as ex:
        futures = {ex.submit(_extract_one_track_task, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                cache.append(future.result())
            except Exception as exc:
                task = futures[future]
                cache.append({
                    "ok": False,
                    "track_name": task["track_name"],
                    "track_id": None,
                    "folder": task["folder"],
                    "df": None,
                    "gt": {},
                    "n_frames": 0,
                    "error": str(exc),
                })
            if done == 1 or done % 200 == 0 or done == len(tasks):
                print(f"[INFO] feature extract [{done}/{len(tasks)}]")
    return cache


def select_baseline_tune_tracks(
    cache: List[Dict[str, Any]],
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[set, set]:
    """返回 (tune_names, heldout_names)。val_fraction=0 表示全部用于寻优。"""
    valid_names = []
    for item in cache:
        if not item.get("ok"):
            continue
        cat = str(item.get("gt", {}).get("gt_category", ""))
        if cat in PRIMARY_GT_CATEGORIES and cat not in EXCLUDED_GT_CATEGORIES:
            valid_names.append(item["track_name"])
    valid_names = sorted(valid_names)
    if not valid_names or val_fraction <= 0:
        return set(valid_names), set()
    rng = np.random.default_rng(seed)
    n_hold = int(round(len(valid_names) * float(val_fraction)))
    n_hold = max(0, min(len(valid_names) - 1, n_hold)) if len(valid_names) > 1 else 0
    hold = set(rng.choice(valid_names, size=n_hold, replace=False).tolist()) if n_hold else set()
    tune = set(valid_names) - hold
    return tune, hold


def tune_baseline_hyperparams(
    cache: List[Dict[str, Any]],
    mod,
    cfg,
    base_exp_params: Dict[str, Any],
    methods: List[str],
    *,
    gray_thresholds: List[float],
    change_thresholds: List[float],
    min_persists: List[int],
    tune_track_names: set,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """对 baseline 网格寻优；返回更新后的 exp_params 与寻优记录表。"""
    records: List[Dict[str, Any]] = []
    exp_params = dict(base_exp_params)
    exp_params.setdefault("gray_min_persist", exp_params["min_persist"])
    exp_params.setdefault("change_min_persist", exp_params["min_persist"])

    if "gray_p99_threshold" in methods:
        best_row = None
        best_key = None
        for th in gray_thresholds:
            for persist in min_persists:
                trial = dict(exp_params)
                trial["gray_threshold"] = float(th)
                trial["gray_min_persist"] = int(persist)
                metrics = evaluate_baseline_params_on_cache(
                    "gray_p99_threshold", cache, mod, cfg, trial, tune_track_names
                )
                row = {
                    "method": "gray_p99_threshold",
                    "gray_threshold": float(th),
                    "change_threshold": np.nan,
                    "min_persist": int(persist),
                    "n_tune_tracks": len(tune_track_names),
                    **{k: metrics.get(k) for k in (
                        "ignition_precision", "ignition_recall", "ignition_f1",
                        "inside_interval_acc", "onset_mae", "n_valid_gt",
                    )},
                }
                records.append(row)
                key = _metric_sort_key(row)
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
        if best_row is not None:
            exp_params["gray_threshold"] = float(best_row["gray_threshold"])
            exp_params["gray_min_persist"] = int(best_row["min_persist"])
            print(
                f"[INFO] tuned gray_p99_threshold: "
                f"threshold={exp_params['gray_threshold']}, "
                f"min_persist={exp_params['gray_min_persist']}, "
                f"F1={best_row.get('ignition_f1')}"
            )

    if "multi_feature_change_score" in methods:
        best_row = None
        best_key = None
        for th in change_thresholds:
            for persist in min_persists:
                trial = dict(exp_params)
                trial["change_threshold"] = float(th)
                trial["change_min_persist"] = int(persist)
                metrics = evaluate_baseline_params_on_cache(
                    "multi_feature_change_score", cache, mod, cfg, trial, tune_track_names
                )
                row = {
                    "method": "multi_feature_change_score",
                    "gray_threshold": np.nan,
                    "change_threshold": float(th),
                    "min_persist": int(persist),
                    "n_tune_tracks": len(tune_track_names),
                    **{k: metrics.get(k) for k in (
                        "ignition_precision", "ignition_recall", "ignition_f1",
                        "inside_interval_acc", "onset_mae", "n_valid_gt",
                    )},
                }
                records.append(row)
                key = _metric_sort_key(row)
                if best_key is None or key > best_key:
                    best_key = key
                    best_row = row
        if best_row is not None:
            exp_params["change_threshold"] = float(best_row["change_threshold"])
            exp_params["change_min_persist"] = int(best_row["min_persist"])
            print(
                f"[INFO] tuned multi_feature_change_score: "
                f"threshold={exp_params['change_threshold']}, "
                f"min_persist={exp_params['change_min_persist']}, "
                f"F1={best_row.get('ignition_f1')}"
            )

    # 兼容旧字段：若未分别设置，保持 min_persist
    if "gray_min_persist" in exp_params and "change_min_persist" in exp_params:
        if exp_params["gray_min_persist"] == exp_params["change_min_persist"]:
            exp_params["min_persist"] = int(exp_params["gray_min_persist"])

    return exp_params, pd.DataFrame(records)


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="点火检测 baseline 对比与模块消融实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tracks-root", type=str, default='inputs/track_crop',
                   help="包含 track_1, track_2, ... 的 ROI 序列根目录")
    p.add_argument(
        "--annotation-csv",
        type=str,
        default=r'effective_trace.xlsx',
        help=(
            "人工标注 CSV/XLSX。"
            "effective_trace: track_id+frame_id（非空=点火，空/未列出=未点火）；"
            "effective_trace_new: 单列未点火 track id，需配合 --reference-summary；"
            "candidate!=-1 且不在 xlsx=点火；xlsx 与 candidate==-1 / 未列出=未点火；"
            "或含 ignition_category / onset_frame_idx 的完整格式。"
        ),
    )
    p.add_argument(
        "--reference-summary",
        type=str,
        default='inputs/summary.csv',
        help=(
            "effective_trace_new 用：参考 summary。"
            "candidate!=-1 且不在 xlsx → 点火；xlsx 与 candidate==-1 → 未点火。"
        ),
    )
    p.add_argument(
        "--gt-onset-tolerance",
        type=int,
        default=2,
        help="effective_trace_new：GT onset 容差，区间为 [onset-tol, onset+tol]",
    )
    p.add_argument("--output-dir", type=str, default=None,
                   help="实验输出目录")
    p.add_argument("--ignition-module-path", type=str, default="ignition_detection_hmm_gmm_shape_hot_prefilter_new.py",
                   help="点火检测模块路径")
    p.add_argument("--methods", type=str, default=DEFAULT_METHODS,
                   help="逗号分隔的方法列表")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行进程数；0 表示自动 min(CPU-1, 16)，1 表示串行",
    )
    p.add_argument("--save-error-cases", action="store_true",
                   help="为错误样本保存曲线图")
    p.add_argument("--include-ambiguous", action="store_true",
                   help="统计时纳入 ambiguous/invalid 等样本")

    p.add_argument("--gray-threshold", type=float, default=0.5,
                   help="gray_p99_threshold 默认阈值；--tune-baselines 时作为初值，会被网格最优覆盖")
    p.add_argument("--change-threshold", type=float, default=0.25,
                   help="multi_feature_change_score 默认阈值；寻优开启时会被覆盖")
    p.add_argument("--min-persist", type=int, default=3,
                   help="baseline 默认持续帧数；寻优开启时两个方法可分别取最优")
    p.add_argument(
        "--tune-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对 gray_p99_threshold / multi_feature_change_score 做网格寻优后再对比（默认开启）",
    )
    p.add_argument(
        "--tune-baselines-only",
        default=False,
        help=(
            "只运行两个 baseline 的超参寻优并写出 baseline_tune_results / baseline_best_params，"
            "不跑完整消融评估"
        ),
    )
    p.add_argument(
        "--gray-threshold-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GRAY_THRESHOLD_GRID),
        help="gray_p99_threshold 寻优网格（逗号分隔）",
    )
    p.add_argument(
        "--change-threshold-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_CHANGE_THRESHOLD_GRID),
        help="multi_feature_change_score 寻优网格（逗号分隔）",
    )
    p.add_argument(
        "--min-persist-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_MIN_PERSIST_GRID),
        help="baseline min_persist 寻优网格（逗号分隔）",
    )
    p.add_argument(
        "--baseline-tune-val-fraction",
        type=float,
        default=0.0,
        help="寻优时从有效 GT 中留出的验证比例；0=在全部有效 GT 上寻优（偏乐观）",
    )
    p.add_argument("--baseline-tune-seed", type=int, default=0,
                   help="baseline 寻优划分随机种子")
    p.add_argument("--already-hot-gray-p99-threshold", type=float, default=120.0,
                   help="baseline already_hot 判断：初始窗口 raw gray_p99 均值阈值")
    p.add_argument("--already-hot-hsv-v-p99-threshold", type=float, default=120.0,
                   help="baseline already_hot 判断：初始窗口 raw hsv_v_p99 均值阈值")

    p.add_argument("--hmm-max-states", type=int, default=3)
    p.add_argument("--hmm-self-prob", type=float, default=0.98)
    p.add_argument("--hmm-forward-prob", type=float, default=0.02)
    p.add_argument("--hmm-min-state-run", type=int, default=3)
    p.add_argument("--hmm-min-cold-run", type=int, default=4)
    p.add_argument("--hmm-min-hot-run", type=int, default=4)
    p.add_argument("--hmm-min-state-count", type=int, default=4)
    p.add_argument("--hmm-min-state-fraction", type=float, default=0.15)
    p.add_argument("--hmm-bic-improvement-ratio", type=float, default=0.08)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    tracks_root = Path(args.tracks_root)
    output_dir = Path(args.output_dir) if args.output_dir else make_default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.tune_baselines_only:
        methods = list(BASELINE_METHODS)
        args.tune_baselines = True
    unknown = [m for m in methods if m not in METHOD_DISPATCH]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {list(METHOD_DISPATCH.keys())}")

    mod = load_ignition_module(Path(args.ignition_module_path))
    cfg = build_base_cfg(args, mod)
    annotations, ann_format = load_annotations(
        Path(args.annotation_csv) if args.annotation_csv else None,
        reference_summary=Path(args.reference_summary) if args.reference_summary else None,
        onset_tolerance=int(args.gt_onset_tolerance),
    )
    missing_as_no_ignition = ann_format in ("effective_trace", "effective_trace_new")
    missing_as_invalid = False
    exp_params = {
        "gray_threshold": args.gray_threshold,
        "change_threshold": args.change_threshold,
        "min_persist": args.min_persist,
        "gray_min_persist": args.min_persist,
        "change_min_persist": args.min_persist,
        "already_hot_gray_p99_threshold": args.already_hot_gray_p99_threshold,
        "already_hot_hsv_v_p99_threshold": args.already_hot_hsv_v_p99_threshold,
    }

    track_folders = mod.list_track_folders(str(tracks_root))
    if not track_folders:
        raise RuntimeError(f"No track folders found under: {tracks_root}")

    n_gt_ign = sum(1 for v in annotations.values() if v.get("gt_category") == "ignition")
    n_gt_no = sum(1 for v in annotations.values() if v.get("gt_category") == "no_ignition")
    n_gt_inv = sum(1 for v in annotations.values() if v.get("gt_category") == "invalid")
    print(f"[INFO] tracks_root: {tracks_root.resolve()}")
    print(f"[INFO] output_dir:  {output_dir.resolve()}")
    print(f"[INFO] methods:     {methods}")
    if args.tune_baselines_only:
        print("[INFO] mode:       tune-baselines-only（仅超参寻优，不跑完整评估）")
    print(f"[INFO] tracks:      {len(track_folders)} (全部检测)")
    missing_policy = (
        "未列出→no_ignition" if missing_as_no_ignition
        else ("未列出/未输出→invalid" if missing_as_invalid else "未列出→空")
    )
    print(
        f"[INFO] annotations: {len(annotations)} (format={ann_format}; "
        f"ignition={n_gt_ign}, no_ignition={n_gt_no}, invalid={n_gt_inv}; "
        f"{missing_policy})"
    )

    error_cases_dir = output_dir / "error_cases" if args.save_error_cases else None
    tasks = [{"track_name": name, "folder": folder} for name, folder in track_folders]

    all_rows: List[Dict[str, Any]] = []
    n_workers = resolve_workers(args.workers)
    print(f"[INFO] workers:     {n_workers} (multiprocess)")

    need_baseline_tune = (
        bool(args.tune_baselines_only)
        or (bool(args.tune_baselines) and any(m in BASELINE_METHODS for m in methods))
    )
    feature_cache: Optional[List[Dict[str, Any]]] = None

    if need_baseline_tune:
        if not annotations:
            raise RuntimeError("--tune-baselines / --tune-baselines-only 需要有效 annotation")
        print("[INFO] extracting features for baseline tuning ...")
        feature_cache = extract_track_feature_cache(
            tasks, mod, cfg, annotations,
            missing_as_no_ignition=missing_as_no_ignition,
            missing_as_invalid=missing_as_invalid,
            n_workers=n_workers,
            ignition_module_path=Path(args.ignition_module_path),
        )
        tune_names, hold_names = select_baseline_tune_tracks(
            feature_cache,
            val_fraction=float(args.baseline_tune_val_fraction),
            seed=int(args.baseline_tune_seed),
        )
        print(
            f"[INFO] baseline tune tracks={len(tune_names)} "
            f"heldout={len(hold_names)} "
            f"(val_fraction={args.baseline_tune_val_fraction})"
        )
        tune_methods = list(BASELINE_METHODS) if args.tune_baselines_only else [
            m for m in methods if m in BASELINE_METHODS
        ]
        exp_params, tune_df = tune_baseline_hyperparams(
            feature_cache, mod, cfg, exp_params, tune_methods,
            gray_thresholds=_parse_float_list(args.gray_threshold_grid),
            change_thresholds=_parse_float_list(args.change_threshold_grid),
            min_persists=_parse_int_list(args.min_persist_grid),
            tune_track_names=tune_names,
        )
        if not tune_df.empty:
            save_table(tune_df, output_dir / "baseline_tune_results")
        best_params_df = pd.DataFrame([{
            "gray_threshold": exp_params.get("gray_threshold"),
            "gray_min_persist": exp_params.get("gray_min_persist"),
            "change_threshold": exp_params.get("change_threshold"),
            "change_min_persist": exp_params.get("change_min_persist"),
            "min_persist": exp_params.get("min_persist"),
            "n_tune_tracks": len(tune_names),
            "n_heldout_tracks": len(hold_names),
            "val_fraction": float(args.baseline_tune_val_fraction),
            "seed": int(args.baseline_tune_seed),
        }])
        save_table(best_params_df, output_dir / "baseline_best_params")
        print(
            f"[INFO] baseline params after tune: "
            f"gray_th={exp_params.get('gray_threshold')}, "
            f"gray_persist={exp_params.get('gray_min_persist')}, "
            f"change_th={exp_params.get('change_threshold')}, "
            f"change_persist={exp_params.get('change_min_persist')}"
        )

        if args.tune_baselines_only:
            # 可选：在 heldout / 全量有效 GT 上报告最优超参指标
            report_names = hold_names if hold_names else tune_names
            report_tag = "heldout" if hold_names else "tune_set"
            summary_rows = []
            for method in BASELINE_METHODS:
                metrics = evaluate_baseline_params_on_cache(
                    method, feature_cache, mod, cfg, exp_params, report_names
                )
                summary_rows.append({
                    "split": report_tag,
                    "method": method,
                    **{k: metrics.get(k) for k in (
                        "ignition_precision", "ignition_recall", "ignition_f1",
                        "inside_interval_acc", "onset_mae", "n_valid_gt",
                    )},
                })
            summary_df = pd.DataFrame(summary_rows)
            save_table(summary_df, output_dir / "baseline_tune_summary")
            print("\n=== Baseline Tune Summary ===")
            print(summary_df.to_string(index=False))
            print(f"\n=== Done (tune-baselines-only) ===")
            print(f"tune_grid -> {output_dir / 'baseline_tune_results.csv'}")
            print(f"best_params -> {output_dir / 'baseline_best_params.csv'}")
            print(f"summary -> {output_dir / 'baseline_tune_summary.csv'}")
            return

    if feature_cache is not None:
        print("[INFO] evaluating all methods on cached features ...")
        ok_cache = [x for x in feature_cache if x.get("ok") and x.get("df") is not None]
        fail_cache = [x for x in feature_cache if not x.get("ok")]
        for idx, item in enumerate(ok_cache, start=1):
            if idx == 1 or idx % 200 == 0 or idx == len(ok_cache):
                print(f"[{idx}/{len(ok_cache)}] {item['track_name']}")
            all_rows.extend(process_one_track(
                track_name=item["track_name"],
                folder=item.get("folder", ""),
                methods=methods,
                mod=mod,
                cfg=cfg,
                exp_params=exp_params,
                annotations=annotations,
                save_error_cases=args.save_error_cases,
                error_cases_dir=error_cases_dir,
                missing_as_no_ignition=missing_as_no_ignition,
                missing_as_invalid=missing_as_invalid,
                precomputed_df=item["df"],
            ))
        for item in fail_cache:
            gt = item.get("gt") or {}
            track_name = item["track_name"]
            track_id = item.get("track_id")
            for method in methods:
                eval_fields = evaluate_prediction({}, gt)
                all_rows.append(result_to_row(
                    track_name, track_id, method, 0,
                    _empty_result(mod, cfg, 0, method, "track_error"),
                    eval_fields,
                    error_message=str(item.get("error", "feature_extract_failed")),
                ))
    elif n_workers == 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"[{idx}/{len(tasks)}] {task['track_name']}")
            rows = process_one_track(
                track_name=task["track_name"],
                folder=task["folder"],
                methods=methods,
                mod=mod,
                cfg=cfg,
                exp_params=exp_params,
                annotations=annotations,
                save_error_cases=args.save_error_cases,
                error_cases_dir=error_cases_dir,
                missing_as_no_ignition=missing_as_no_ignition,
                missing_as_invalid=missing_as_invalid,
            )
            all_rows.extend(rows)
    else:
        worker_ctx = {
            "ignition_module_path": str(Path(args.ignition_module_path).resolve()),
            "cfg_dict": asdict(cfg),
            "exp_params": exp_params,
            "annotations": annotations,
            "methods": methods,
            "save_error_cases": args.save_error_cases,
            "error_cases_dir": str(error_cases_dir) if error_cases_dir else "",
            "missing_as_no_ignition": missing_as_no_ignition,
            "missing_as_invalid": missing_as_invalid,
        }
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(worker_ctx,)) as ex:
            futures = {ex.submit(_worker_process_track, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                done += 1
                try:
                    all_rows.extend(future.result())
                except Exception as exc:
                    print(f"[ERROR] {task['track_name']}: {exc}")
                    traceback.print_exc()
                print(f"[{done}/{len(tasks)}] {task['track_name']}")

    per_track_df = pd.DataFrame(all_rows)
    if not per_track_df.empty:
        per_track_df = per_track_df.sort_values(["track_id", "method"]).reset_index(drop=True)

    save_table(per_track_df, output_dir / "per_track_results")

    comparison_df = build_method_track_comparison_table(per_track_df, methods)
    save_table(comparison_df, output_dir / "method_track_comparison")

    summary_df = compute_method_summary(per_track_df, include_ambiguous=args.include_ambiguous)
    save_table(summary_df, output_dir / "method_summary")

    if not summary_df.empty and "n_method_errors" in summary_df.columns:
        for _, row in summary_df.iterrows():
            n_tracks = int(row.get("n_tracks", 0))
            n_errors = int(row.get("n_method_errors", 0))
            if n_tracks > 0 and n_errors >= n_tracks:
                print(
                    f"\n*** WARNING: method '{row['method']}' failed on ALL tracks "
                    f"(n_method_errors={n_errors}/{n_tracks}, method_error_rate=1.0) ***\n"
                )

    print("\n=== Method Summary ===")
    if summary_df.empty:
        print("No annotation-based metrics (provide --annotation-csv for evaluation).")
    else:
        print(summary_df.to_string(index=False))

    print("\n=== Method × Track Comparison (preview) ===")
    if comparison_df.empty:
        print("(empty)")
    else:
        preview = comparison_df.head(20)
        print(preview.to_string(index=False))
        if len(comparison_df) > len(preview):
            print(f"... ({len(comparison_df) - len(preview)} more rows in method_track_comparison.csv)")

    print(f"\n=== Done ===")
    print(f"per_track_rows={len(per_track_df)}")
    print(f"results -> {output_dir / 'per_track_results.csv'}")
    print(f"comparison -> {output_dir / 'method_track_comparison.csv'}")


if __name__ == "__main__":
    main()
