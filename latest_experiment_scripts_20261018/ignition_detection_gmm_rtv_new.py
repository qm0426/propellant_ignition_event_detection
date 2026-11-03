#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 GMM 热状态划分和原始热转移验证的无监督点火检测。

流程：
ROI 几何/初始高亮预检查
→ 热特征提取
→ GMM+BIC 状态划分
→ 状态热度排序
→ 短状态段平滑
→ 带全历史冷态约束的连续 cold-to-hot 候选转移
→ 原始热特征增量验证
→ 可选的点火时刻微调。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class IgnitionGMMConfig:
    bright_lambda: float = 6.0
    border_width: int = 2
    smooth_window: int = 5
    save_features: bool = True

    gmm_max_states: int = 3
    gmm_covariance_type: str = "diag"
    gmm_reg_covar: float = 1e-4
    gmm_n_init: int = 5
    gmm_bic_improvement_ratio: float = 0.03
    min_state_run: int = 2
    min_cold_run: int = 3
    min_hot_run: int = 3
    min_stable_run: int = 3
    min_sequence_len: int = 8
    gmm_min_state_count: int = 2
    gmm_min_state_fraction: float = 0.08
    use_diff_features: bool = True
    require_all_prior_cold: bool = True
    refine_onset_with_change: bool = False
    refine_radius: int = 2
    random_state: int = 42

    # 序列级预检查：第一帧起已处于高亮/燃烧态
    enable_initial_hot_reject: bool = True
    initial_hot_check_frames: int = 5
    initial_hot_min_frame_fraction: float = 0.80
    initial_hot_min_feature_votes: int = 3
    initial_hot_gray_p99_min: float = 160.0
    initial_hot_hsv_v_p99_min: float = 160.0
    initial_hot_corr_p99_min: float = 25.0
    initial_hot_bright_area_min: float = 0.03
    initial_hot_bright_energy_min: float = 6.0

    # 序列级 ROI 几何质量预检查。所有比值均采用 fold-change，天然兼容不同绝对尺寸。
    # 拒绝规则：极端突变直接拒绝；否则至少两类独立证据才拒绝。
    enable_shape_sequence_reject: bool = True
    shape_area_jump_ratio_max: float = 2.5
    shape_aspect_jump_ratio_max: float = 1.8
    shape_hard_area_jump_ratio: float = 5.0
    shape_hard_aspect_jump_ratio: float = 3.0
    shape_global_area_ratio_max: float = 5.0
    shape_global_aspect_ratio_max: float = 2.5
    shape_bad_jump_fraction_max: float = 0.30
    shape_invalid_fraction_max: float = 0.10
    shape_extreme_area_jump_ratio: float = 8.0
    shape_extreme_aspect_jump_ratio: float = 5.0

    no_ignition_index: int = -1

    enable_raw_transition_validation: bool = True

    min_raw_delta_gray_p99: float = 25.0
    min_raw_delta_corr_p99: float = 20.0
    min_raw_delta_bright_energy: float = 8.0
    min_raw_delta_bright_area: float = 0.04
    min_raw_delta_hsv_v_p99: float = 25.0


# ============================================================
# 2. 通用 IO / 工具
# ============================================================

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def natural_sort_key(path: str) -> list:
    name = os.path.basename(path)
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def list_images_in_folder(folder: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths), key=natural_sort_key)


def list_track_folders(
    root_dir: str,
    track_prefix: str = "track_",
    start_from: int = 1,
) -> List[Tuple[str, str]]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Track root not found: {root_dir}")
    tracks: List[Tuple[str, str]] = []
    for p in sorted(root.iterdir(), key=lambda x: natural_sort_key(x.name)):
        if not p.is_dir() or not p.name.startswith(track_prefix):
            continue
        suffix = p.name[len(track_prefix) :]
        if suffix.isdigit() and int(suffix) < start_from:
            continue
        tracks.append((p.name, str(p)))
    return tracks


def track_id_from_name(track_name: str, track_prefix: str = "track_") -> Optional[int]:
    if not track_name.startswith(track_prefix):
        return None
    suffix = track_name[len(track_prefix) :]
    return int(suffix) if suffix.isdigit() else None


def make_default_batch_output_dir(base_name: str = "ignition_results") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(base_name) / f"run_{stamp}")


def save_table(df: pd.DataFrame, path_without_ext: str, quiet: bool = False) -> None:
    csv_path = f"{path_without_ext}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if not quiet:
        print(f"[OK] Saved: {csv_path}")
    try:
        xlsx_path = f"{path_without_ext}.xlsx"
        df.to_excel(xlsx_path, index=False)
        if not quiet:
            print(f"[OK] Saved: {xlsx_path}")
    except Exception as exc:
        if not quiet:
            warnings.warn(f"Skip Excel export ({exc}); CSV is available at {csv_path}")


def read_image(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot read image: {path}")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def robust_mad(x: np.ndarray, eps: float = 1e-6) -> float:
    x = x.astype(np.float32).ravel()
    return float(np.median(np.abs(x - np.median(x))) + eps)


def get_border_pixels(gray: np.ndarray, border_width: int = 2) -> np.ndarray:
    h, w = gray.shape
    bw = max(1, min(border_width, h // 3, w // 3))
    parts = [gray[:bw, :].ravel(), gray[-bw:, :].ravel(), gray[:, :bw].ravel(), gray[:, -bw:].ravel()]
    return np.concatenate(parts)


def safe_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    return float(a / (b + eps))


def normalize_to_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    return np.clip((x - lo) / (hi - lo + eps), 0, 1)


def smooth_array(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n < 5:
        return x
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return x
    if window % 2 == 0:
        window -= 1
    return savgol_filter(x, window_length=window, polyorder=2)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def _filename_at(df: pd.DataFrame, idx: int) -> str:
    if idx < 0 or idx >= len(df):
        return ""
    return str(df.loc[idx, "filename"])


# ============================================================
# 3. 特征提取
# ============================================================

def extract_frame_features(img_rgb: np.ndarray, cfg: IgnitionGMMConfig) -> Dict[str, float]:
    img = img_rgb.astype(np.float32)
    h, w, _ = img.shape
    area_total = h * w
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    bg_pixels = get_border_pixels(gray, cfg.border_width)
    bg_med = float(np.median(bg_pixels))
    bg_mad = robust_mad(bg_pixels)
    bright_th = float(np.clip(bg_med + cfg.bright_lambda * bg_mad, 0, 255))
    gray_corr = gray - bg_med

    p50 = float(np.percentile(gray, 50))
    p75 = float(np.percentile(gray, 75))
    p90 = float(np.percentile(gray, 90))
    p95 = float(np.percentile(gray, 95))
    p99 = float(np.percentile(gray, 99))
    gmax = float(np.max(gray))
    gmean = float(np.mean(gray))
    gstd = float(np.std(gray))

    c_p90 = float(np.percentile(gray_corr, 90))
    c_p95 = float(np.percentile(gray_corr, 95))
    c_p99 = float(np.percentile(gray_corr, 99))
    c_max = float(np.max(gray_corr))

    bright_mask = gray > bright_th
    if min(h, w) >= 5:
        bright_mask = binary_opening(bright_mask, structure=np.ones((2, 2)))
        bright_mask = binary_closing(bright_mask, structure=np.ones((2, 2)))

    bright_area = float(np.sum(bright_mask))
    bright_area_ratio = bright_area / max(area_total, 1)
    excess = np.maximum(gray - bright_th, 0)
    bright_energy = float(np.sum(excess))
    bright_energy_mean = bright_energy / max(area_total, 1)

    if bright_area > 0:
        bright_mean = float(np.mean(gray[bright_mask]))
        bright_max = float(np.max(gray[bright_mask]))
    else:
        bright_mean = bright_max = 0.0

    r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    rg_ratio = safe_ratio(r_mean, g_mean)
    rb_ratio = safe_ratio(r_mean, b_mean)
    red_ratio = safe_ratio(r_mean, r_mean + g_mean + b_mean)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    hsv_s_mean = float(np.mean(sat))
    hsv_v_mean = float(np.mean(val))
    hsv_v_p95 = float(np.percentile(val, 95))
    hsv_v_p99 = float(np.percentile(val, 99))

    mask_uint8 = bright_mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(component_areas)) + 1
        largest_area_ratio = float(stats[largest_idx, cv2.CC_STAT_AREA]) / max(area_total, 1)
        cx, cy = centroids[largest_idx]
        center_x, center_y = (w - 1) / 2.0, (h - 1) / 2.0
        centroid_dist_norm = float(np.hypot(cx - center_x, cy - center_y)) / max(np.hypot(h, w), 1.0)
        comp_count = float(num_labels - 1)
    else:
        largest_area_ratio = centroid_dist_norm = comp_count = 0.0

    return {
        "bg_med": bg_med, "bg_mad": bg_mad, "bright_th": bright_th,
        "gray_mean": gmean, "gray_std": gstd, "gray_p50": p50, "gray_p75": p75,
        "gray_p90": p90, "gray_p95": p95, "gray_p99": p99, "gray_max": gmax,
        "corr_p90": c_p90, "corr_p95": c_p95, "corr_p99": c_p99, "corr_max": c_max,
        "bright_area_ratio": bright_area_ratio, "bright_energy": bright_energy,
        "bright_energy_mean": bright_energy_mean, "bright_mean": bright_mean, "bright_max": bright_max,
        "r_mean": r_mean, "g_mean": g_mean, "b_mean": b_mean,
        "rg_ratio": rg_ratio, "rb_ratio": rb_ratio, "red_ratio": red_ratio,
        "hsv_s_mean": hsv_s_mean, "hsv_v_mean": hsv_v_mean, "hsv_v_p95": hsv_v_p95, "hsv_v_p99": hsv_v_p99,
        "component_count": comp_count, "largest_area_ratio": largest_area_ratio,
        "centroid_dist_norm": centroid_dist_norm,
    }


def extract_sequence_features(image_paths: List[str], cfg: IgnitionGMMConfig) -> pd.DataFrame:
    rows = []
    for idx, path in enumerate(image_paths):
        img_rgb = read_image(path)
        feat = extract_frame_features(img_rgb, cfg)
        h, w = img_rgb.shape[:2]
        feat["roi_width"] = float(w)
        feat["roi_height"] = float(h)
        feat["roi_area"] = float(max(w * h, 1))
        feat["roi_aspect_ratio"] = float(w / max(h, 1))
        feat["frame_idx"] = idx
        feat["filename"] = os.path.basename(path)
        feat["image_path"] = path
        rows.append(feat)
    df = pd.DataFrame(rows)

    diff_cols = [
        "gray_p95", "gray_p99", "gray_max", "corr_p95", "corr_p99", "corr_max",
        "bright_area_ratio", "bright_energy_mean", "bright_mean", "red_ratio",
        "rg_ratio", "hsv_v_p95", "hsv_v_p99", "largest_area_ratio",
    ]
    for col in diff_cols:
        df[f"d1_{col}"] = df[col].diff().fillna(0.0)
        df[f"d2_{col}"] = df[f"d1_{col}"].diff().fillna(0.0)

    rolling_cols = [
        "gray_p99", "corr_p99", "bright_area_ratio", "bright_energy_mean",
        "red_ratio", "hsv_v_p99", "largest_area_ratio",
    ]
    for col in rolling_cols:
        df[f"roll3_{col}"] = df[col].rolling(window=3, center=True, min_periods=1).mean()
    return df


def compute_hot_score(df: pd.DataFrame) -> np.ndarray:
    """仅用于绘图辅助。"""
    keys = [
        "gray_p99", "corr_p99", "bright_area_ratio", "bright_energy_mean",
        "red_ratio", "hsv_v_p99", "largest_area_ratio",
    ]
    scores = [normalize_to_01(df[k].values) for k in keys if k in df.columns]
    if not scores:
        return np.zeros(len(df), dtype=np.float32)
    return np.mean(np.stack(scores, axis=1), axis=1)


def _raw_window_mean(df: pd.DataFrame, col: str, start: int, end: int) -> float:
    if col not in df.columns or end <= start:
        return 0.0
    vals = pd.to_numeric(df[col].iloc[start:end], errors="coerce")
    if vals.notna().any():
        return float(np.nanmean(vals.values))
    return 0.0



def _fold_change(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对称倍数变化：2→4 与 4→2 均返回 2。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.maximum((a + eps) / (b + eps), (b + eps) / (a + eps))


def symmetric_ratio(a: float, b: float, eps: float = 1e-6) -> float:
    """单对对称倍数；无效输入返回 NaN。"""
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("nan")
    return float(max(a / max(b, eps), b / max(a, eps)))


def safe_quantile_ratio(
    values: np.ndarray,
    low_q: float = 5,
    high_q: float = 95,
    eps: float = 1e-6,
) -> Tuple[float, float, float, float]:
    """
    对正有限值计算分位数与 P_high/P_low 比值。

    Returns:
        p_low, p50, p_high, p_high_over_p_low
    """
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 1.0
    p_low, p50, p_high = np.percentile(vals, [low_q, 50, high_q])
    ratio = float((float(p_high) + eps) / (max(float(p_low), 0.0) + eps))
    return float(p_low), float(p50), float(p_high), ratio


def _empty_shape_diagnostics(
    *,
    enabled: bool,
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "reason": reason,
        "shape_quality_warning": False,
        "shape_warning_reasons": [],
        "shape_reject_reasons": [],
        "shape_evidence_count": 0,
        "shape_jump_failure": False,
        "shape_global_failure": False,
        "shape_frequent_failure": False,
        "shape_invalid_failure": False,
        "shape_extreme_failure": False,
        "max_area_jump_ratio": 1.0,
        "max_aspect_jump_ratio": 1.0,
        "global_area_ratio_q95_q05": 1.0,
        "global_aspect_ratio_q95_q05": 1.0,
        "bad_area_jump_fraction": 0.0,
        "bad_aspect_jump_fraction": 0.0,
        "bad_jump_fraction": 0.0,
        "invalid_roi_count": 0,
        "invalid_roi_fraction": 0.0,
        "roi_area_p05": 0.0,
        "roi_area_p50": 0.0,
        "roi_area_p95": 0.0,
        "roi_aspect_p05": 0.0,
        "roi_aspect_p50": 0.0,
        "roi_aspect_p95": 0.0,
        # 兼容旧字段
        "reject_reasons": [],
        "invalid_geometry_fraction": 0.0,
    }


def check_roi_shape_sequence_quality(
    df: pd.DataFrame,
    cfg: IgnitionGMMConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    在进入 GMM 状态建模前检查 crop 的宽、高、面积及宽高比时序。

    拒绝规则：
      - 极端相邻突变：直接拒绝；
      - 否则按四类证据（jump / global / frequent / invalid）计数，
        至少两类独立证据才拒绝；
      - 单一轻度异常只 warning，不排除。

    当 enable_shape_sequence_reject=False 时仍可计算 diagnostics，但永不拒绝。
    """
    required = ["roi_width", "roi_height", "roi_area", "roi_aspect_ratio"]
    compute_only = not bool(cfg.enable_shape_sequence_reject)

    if len(df) < 2 or any(c not in df.columns for c in required):
        diag = _empty_shape_diagnostics(
            enabled=not compute_only,
            reason="insufficient_geometry_data",
        )
        return False, diag

    area = pd.to_numeric(df["roi_area"], errors="coerce").to_numpy(dtype=np.float64)
    aspect = pd.to_numeric(df["roi_aspect_ratio"], errors="coerce").to_numpy(dtype=np.float64)
    width = pd.to_numeric(df["roi_width"], errors="coerce").to_numpy(dtype=np.float64)
    height = pd.to_numeric(df["roi_height"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(area)

    finite = (
        np.isfinite(area)
        & np.isfinite(aspect)
        & np.isfinite(width)
        & np.isfinite(height)
    )
    positive = (area > 0) & (aspect > 0) & (width > 0) & (height > 0)
    valid = finite & positive
    invalid_count = int(np.sum(~valid))
    invalid_fraction = float(invalid_count / max(n, 1))

    area_p05, area_p50, area_p95, global_area_ratio = safe_quantile_ratio(area[valid])
    aspect_p05, aspect_p50, aspect_p95, global_aspect_ratio = safe_quantile_ratio(aspect[valid])

    # 仅统计“原序列相邻且两侧均有效”的帧对
    pair_area_jumps: List[float] = []
    pair_aspect_jumps: List[float] = []
    for t in range(1, n):
        if not (valid[t] and valid[t - 1]):
            continue
        aj = symmetric_ratio(float(area[t]), float(area[t - 1]))
        sj = symmetric_ratio(float(aspect[t]), float(aspect[t - 1]))
        if np.isfinite(aj):
            pair_area_jumps.append(float(aj))
        if np.isfinite(sj):
            pair_aspect_jumps.append(float(sj))

    area_jump = np.asarray(pair_area_jumps, dtype=np.float64)
    aspect_jump = np.asarray(pair_aspect_jumps, dtype=np.float64)

    if area_jump.size:
        bad_area_jump = area_jump >= float(cfg.shape_area_jump_ratio_max)
        max_area_jump_ratio = float(np.nanmax(area_jump))
        bad_area_jump_fraction = float(np.mean(bad_area_jump))
    else:
        bad_area_jump = np.zeros(0, dtype=bool)
        max_area_jump_ratio = 1.0
        bad_area_jump_fraction = 0.0

    if aspect_jump.size:
        bad_aspect_jump = aspect_jump >= float(cfg.shape_aspect_jump_ratio_max)
        max_aspect_jump_ratio = float(np.nanmax(aspect_jump))
        bad_aspect_jump_fraction = float(np.mean(bad_aspect_jump))
    else:
        bad_aspect_jump = np.zeros(0, dtype=bool)
        max_aspect_jump_ratio = 1.0
        bad_aspect_jump_fraction = 0.0

    if area_jump.size and aspect_jump.size and len(area_jump) == len(aspect_jump):
        bad_geometry_jump = bad_area_jump | bad_aspect_jump
        bad_jump_fraction = float(np.mean(bad_geometry_jump))
    elif area_jump.size:
        bad_geometry_jump = bad_area_jump
        bad_jump_fraction = bad_area_jump_fraction
    elif aspect_jump.size:
        bad_geometry_jump = bad_aspect_jump
        bad_jump_fraction = bad_aspect_jump_fraction
    else:
        bad_geometry_jump = np.zeros(0, dtype=bool)
        bad_jump_fraction = 0.0

    jump_failure = bool(
        max_area_jump_ratio >= float(cfg.shape_hard_area_jump_ratio)
        or max_aspect_jump_ratio >= float(cfg.shape_hard_aspect_jump_ratio)
    )
    global_failure = bool(
        global_area_ratio >= float(cfg.shape_global_area_ratio_max)
        or global_aspect_ratio >= float(cfg.shape_global_aspect_ratio_max)
    )
    frequent_failure = bool(bad_jump_fraction > float(cfg.shape_bad_jump_fraction_max))
    # 单次尖峰（连续最多 2 个坏跳变对）由 jump_failure 覆盖，不单独记 frequent。
    if frequent_failure and bad_geometry_jump.size:
        n_bad = int(np.sum(bad_geometry_jump))
        n_runs = 0
        prev = False
        for flag in bad_geometry_jump.tolist():
            if flag and not prev:
                n_runs += 1
            prev = bool(flag)
        if n_runs <= 1 and n_bad <= 2:
            frequent_failure = False

    invalid_failure = bool(invalid_fraction > float(cfg.shape_invalid_fraction_max))
    extreme_failure = bool(
        max_area_jump_ratio >= float(cfg.shape_extreme_area_jump_ratio)
        or max_aspect_jump_ratio >= float(cfg.shape_extreme_aspect_jump_ratio)
    )

    # 同源尖峰去重：去掉最大相邻突变涉及的离群帧后，若全局范围恢复，则不把 global 算作独立证据。
    if jump_failure and global_failure and np.any(valid):
        drop_mask = np.zeros(n, dtype=bool)
        best_pair = (-1, -1)
        best_score = -1.0
        for t in range(1, n):
            if not (valid[t] and valid[t - 1]):
                continue
            score = max(
                symmetric_ratio(float(area[t]), float(area[t - 1])),
                symmetric_ratio(float(aspect[t]), float(aspect[t - 1])),
            )
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_pair = (t - 1, t)
        if best_pair[0] >= 0:
            # 去掉尖峰帧（取面积更大的一侧，并带上配对帧）
            i0, i1 = best_pair
            spike_idx = i0 if float(area[i0]) >= float(area[i1]) else i1
            drop_mask[spike_idx] = True
            drop_mask[i0] = True
            drop_mask[i1] = True
            remain = valid & (~drop_mask)
            if int(np.sum(remain)) >= 2:
                _, _, _, ga2 = safe_quantile_ratio(area[remain])
                _, _, _, gs2 = safe_quantile_ratio(aspect[remain])
                if (
                    ga2 < float(cfg.shape_global_area_ratio_max)
                    and gs2 < float(cfg.shape_global_aspect_ratio_max)
                ):
                    global_failure = False

    evidence_flags = {
        "jump_failure": bool(jump_failure),
        "global_failure": bool(global_failure),
        "frequent_failure": bool(frequent_failure),
        "invalid_failure": bool(invalid_failure),
    }
    evidence_count = int(sum(int(v) for v in evidence_flags.values()))

    warning_reasons: List[str] = []
    if jump_failure:
        warning_reasons.append("large_geometry_jump")
    if global_failure:
        warning_reasons.append("large_global_geometry_variation")
    if frequent_failure:
        warning_reasons.append("frequent_geometry_jumps")
    if invalid_failure:
        warning_reasons.append("too_many_invalid_roi_geometry")
    if extreme_failure:
        warning_reasons.append("extreme_geometry_jump")

    reject_sequence = bool(extreme_failure or evidence_count >= 2)
    if compute_only:
        reject_sequence = False

    reject_reasons: List[str] = []
    if reject_sequence:
        if extreme_failure:
            reject_reasons.append("extreme_geometry_jump")
        if evidence_count >= 2:
            reject_reasons.extend(name for name, flag in evidence_flags.items() if flag)
        # 去重且保序
        seen = set()
        reject_reasons = [r for r in reject_reasons if not (r in seen or seen.add(r))]

    diagnostics: Dict[str, Any] = {
        "enabled": not compute_only,
        "reason": "compute_only" if compute_only else "",
        "shape_quality_warning": bool(warning_reasons),
        "shape_warning_reasons": list(warning_reasons),
        "shape_reject_reasons": list(reject_reasons),
        "shape_evidence_count": int(evidence_count),
        "shape_jump_failure": bool(jump_failure),
        "shape_global_failure": bool(global_failure),
        "shape_frequent_failure": bool(frequent_failure),
        "shape_invalid_failure": bool(invalid_failure),
        "shape_extreme_failure": bool(extreme_failure),
        "max_area_jump_ratio": float(max_area_jump_ratio),
        "max_aspect_jump_ratio": float(max_aspect_jump_ratio),
        "global_area_ratio_q95_q05": float(global_area_ratio),
        "global_aspect_ratio_q95_q05": float(global_aspect_ratio),
        "bad_area_jump_fraction": float(bad_area_jump_fraction),
        "bad_aspect_jump_fraction": float(bad_aspect_jump_fraction),
        "bad_jump_fraction": float(bad_jump_fraction),
        "invalid_roi_count": int(invalid_count),
        "invalid_roi_fraction": float(invalid_fraction),
        "roi_area_p05": float(area_p05),
        "roi_area_p50": float(area_p50),
        "roi_area_p95": float(area_p95),
        "roi_aspect_p05": float(aspect_p05),
        "roi_aspect_p50": float(aspect_p50),
        "roi_aspect_p95": float(aspect_p95),
        "n_valid_frame_pairs": int(len(pair_area_jumps)),
        "median_roi_width": float(np.median(width[valid])) if np.any(valid) else 0.0,
        "median_roi_height": float(np.median(height[valid])) if np.any(valid) else 0.0,
        "median_roi_area": float(area_p50),
        "median_roi_aspect_ratio": float(aspect_p50),
        # 兼容旧字段名
        "reject_reasons": list(reject_reasons),
        "invalid_geometry_fraction": float(invalid_fraction),
        "bad_geometry_jump_fraction": float(bad_jump_fraction),
    }
    return bool(reject_sequence), diagnostics


def is_initially_hot_sequence(
    df: pd.DataFrame,
    cfg: IgnitionGMMConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    用原始绝对亮度检查序列起点是否已经处于高温/燃烧态。

    不使用序列内归一化或 GMM rank，避免“整段都亮但被相对分成冷/热状态”。
    每帧由 5 项绝对特征投票，前 N 帧达到指定热帧比例时判为 already_hot。
    """
    n = len(df)
    if not cfg.enable_initial_hot_reject:
        return False, {"enabled": False}
    if n == 0:
        return False, {"enabled": True, "reason": "empty_sequence"}

    n_check = max(1, min(int(cfg.initial_hot_check_frames), n))
    head = df.iloc[:n_check]

    specs = [
        ("gray_p99", cfg.initial_hot_gray_p99_min),
        ("hsv_v_p99", cfg.initial_hot_hsv_v_p99_min),
        ("corr_p99", cfg.initial_hot_corr_p99_min),
        ("bright_area_ratio", cfg.initial_hot_bright_area_min),
        ("bright_energy_mean", cfg.initial_hot_bright_energy_min),
    ]
    votes = np.zeros(n_check, dtype=np.int32)
    medians: Dict[str, float] = {}
    for col, threshold in specs:
        vals = pd.to_numeric(head[col], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
        votes += vals >= threshold
        finite_vals = vals[np.isfinite(vals)]
        medians[f"initial_median_{col}"] = float(np.median(finite_vals)) if len(finite_vals) else 0.0

    hot_mask = votes >= int(cfg.initial_hot_min_feature_votes)
    hot_fraction = float(np.mean(hot_mask))
    initially_hot = hot_fraction >= float(cfg.initial_hot_min_frame_fraction)

    diagnostics: Dict[str, Any] = {
        "enabled": True,
        "n_initial_frames_checked": n_check,
        "initial_hot_frame_fraction": hot_fraction,
        "initial_hot_vote_counts": votes.tolist(),
        "min_feature_votes": int(cfg.initial_hot_min_feature_votes),
        "min_hot_frame_fraction": float(cfg.initial_hot_min_frame_fraction),
        **medians,
    }
    return initially_hot, diagnostics


def _make_precheck_result(
    df: pd.DataFrame,
    cfg: IgnitionGMMConfig,
    status: str,
    diagnostics: Dict[str, Any],
) -> Dict:
    """构造与正常检测结果字段兼容的预检查拒绝结果。"""
    n = len(df)
    no_idx = cfg.no_ignition_index
    result = {
        "has_ignition": False,
        "ignition_status": status,
        "ignition_idx_onset": no_idx,
        "ignition_idx_confirmed": no_idx,
        "ignition_idx_candidate": no_idx,
        "candidate_filename": "",
        "confirmed_filename": "",
        "ignition_confidence": 0.0,
        "ignition_amplitude": 0.0,
        "ignition_change_peak": 0.0,
        "n_ignition_candidates": 0,
        "all_ignition_candidates": [],
        "gmm_n_states": 0,
        "gmm_state_sequence": [],
        "gmm_state_thermal_scores": {},
        "gmm_bic_by_k": {},
        "gmm_feature_cols": [],
        "ignition_method": "gmm_rtv",
        "raw_transition_validation_enabled": bool(cfg.enable_raw_transition_validation),
        "precheck_rejected": True,
        "precheck_diagnostics": diagnostics,
        "hot_score": compute_hot_score(df),
        "change_score": np.zeros(n, dtype=np.float32),
        **_empty_keyframe_fields(no_idx),
    }
    shape_metrics = diagnostics.get("shape", {}) if isinstance(diagnostics, dict) else {}
    _attach_shape_fields_to_result(result, shape_metrics)
    return result


def _attach_shape_fields_to_result(result: Dict[str, Any], shape_metrics: Dict[str, Any]) -> None:
    """把 shape diagnostics 扁平化写入 result，供 summary 读取；不改变 ignition_status。"""
    shape_metrics = shape_metrics or {}
    warning_reasons = list(shape_metrics.get("shape_warning_reasons", []) or [])
    reject_reasons = list(shape_metrics.get("shape_reject_reasons", []) or shape_metrics.get("reject_reasons", []) or [])
    result["precheck_rejected"] = bool(result.get("precheck_rejected", False))
    result["shape_quality_warning"] = bool(shape_metrics.get("shape_quality_warning", bool(warning_reasons)))
    result["shape_warning_reasons"] = warning_reasons
    result["shape_reject_reasons"] = reject_reasons
    result["shape_evidence_count"] = int(shape_metrics.get("shape_evidence_count", 0) or 0)
    result["shape_jump_failure"] = bool(shape_metrics.get("shape_jump_failure", False))
    result["shape_global_failure"] = bool(shape_metrics.get("shape_global_failure", False))
    result["shape_frequent_failure"] = bool(shape_metrics.get("shape_frequent_failure", False))
    result["shape_invalid_failure"] = bool(shape_metrics.get("shape_invalid_failure", False))
    result["shape_extreme_failure"] = bool(shape_metrics.get("shape_extreme_failure", False))
    result["max_area_jump_ratio"] = float(shape_metrics.get("max_area_jump_ratio", 1.0) or 1.0)
    result["max_aspect_jump_ratio"] = float(shape_metrics.get("max_aspect_jump_ratio", 1.0) or 1.0)
    result["global_area_ratio_q95_q05"] = float(shape_metrics.get("global_area_ratio_q95_q05", 1.0) or 1.0)
    result["global_aspect_ratio_q95_q05"] = float(shape_metrics.get("global_aspect_ratio_q95_q05", 1.0) or 1.0)
    result["bad_jump_fraction"] = float(
        shape_metrics.get("bad_jump_fraction", shape_metrics.get("bad_geometry_jump_fraction", 0.0)) or 0.0
    )
    result["invalid_roi_fraction"] = float(
        shape_metrics.get("invalid_roi_fraction", shape_metrics.get("invalid_geometry_fraction", 0.0)) or 0.0
    )


def raw_transition_has_thermal_increase(
    df: pd.DataFrame,
    transition_idx: int,
    cfg: IgnitionGMMConfig,
) -> Tuple[bool, Dict[str, float]]:
    """
    比较候选 GMM 状态转移点前后 raw 特征增量。
    只要至少一类原始热特征增量达到阈值，则候选状态转移通过验证。
    """
    n = len(df)
    if transition_idx <= 0 or transition_idx >= n:
        return False, {"reason": "invalid_transition_idx"}

    pre_start = max(0, transition_idx - cfg.min_cold_run)
    pre_end = transition_idx
    post_start = transition_idx
    post_end = min(n, transition_idx + cfg.min_hot_run)

    specs = [
        ("gray_p99", "delta_gray_p99", cfg.min_raw_delta_gray_p99),
        ("corr_p99", "delta_corr_p99", cfg.min_raw_delta_corr_p99),
        ("bright_energy_mean", "delta_bright_energy_mean", cfg.min_raw_delta_bright_energy),
        ("bright_area_ratio", "delta_bright_area_ratio", cfg.min_raw_delta_bright_area),
        ("hsv_v_p99", "delta_hsv_v_p99", cfg.min_raw_delta_hsv_v_p99),
    ]

    diag: Dict[str, float] = {"transition_idx": float(transition_idx)}
    any_pass = False
    for col, delta_key, min_delta in specs:
        pre_val = _raw_window_mean(df, col, pre_start, pre_end)
        post_val = _raw_window_mean(df, col, post_start, post_end)
        delta = post_val - pre_val
        diag[f"pre_{col}"] = pre_val
        diag[f"post_{col}"] = post_val
        diag[delta_key] = delta
        diag[f"min_{delta_key}"] = float(min_delta)
        if delta >= min_delta:
            any_pass = True

    diag["any_thermal_increase"] = float(any_pass)
    return any_pass, diag


# ============================================================
# 4. GMM 特征矩阵
# ============================================================

BASE_GMM_FEATURES = [
    "gray_p95", "gray_p99", "corr_p95", "corr_p99", "corr_max",
    "bright_energy_mean", "bright_area_ratio", "bright_mean",
    "red_ratio", "rg_ratio", "hsv_v_p95", "hsv_v_p99", "largest_area_ratio",
]

DIFF_GMM_FEATURES = [
    "d1_gray_p99", "d1_corr_p99", "d1_bright_energy_mean",
    "d1_bright_area_ratio", "d1_red_ratio", "d1_hsv_v_p99",
]

THERMAL_FEATURE_NAMES = [
    "gray_p99", "corr_p99", "corr_max", "bright_energy_mean",
    "bright_area_ratio", "red_ratio", "hsv_v_p99", "largest_area_ratio",
]

RAW_THERMAL_CHANGE_FEATURES = [
    "gray_p99",
    "corr_p99",
    "bright_energy_mean",
    "bright_area_ratio",
    "hsv_v_p99",
]


def get_gmm_feature_columns(df: pd.DataFrame, cfg: IgnitionGMMConfig) -> List[str]:
    cols = [c for c in BASE_GMM_FEATURES if c in df.columns]
    if cfg.use_diff_features:
        cols.extend(c for c in DIFF_GMM_FEATURES if c in df.columns)
    return cols


def build_gmm_feature_matrix(
    df: pd.DataFrame,
    cfg: IgnitionGMMConfig,
) -> Tuple[np.ndarray, List[str]]:
    feature_cols = get_gmm_feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError("No GMM feature columns available in dataframe.")

    X = df[feature_cols].values.astype(np.float32)
    X = np.where(np.isfinite(X), X, np.nan)
    for j in range(X.shape[1]):
        col = X[:, j]
        med = float(np.nanmedian(col))
        col = np.where(np.isnan(col), med, col)
        mad = float(np.median(np.abs(col - med))) + 1e-6
        X[:, j] = np.clip((col - med) / mad, -6.0, 6.0)
    return X, feature_cols


# ============================================================
# 5. GMM + BIC
# ============================================================

def fit_gmm_models_bic(
    X: np.ndarray,
    max_states: int,
    covariance_type: str,
    reg_covar: float,
    n_init: int,
    bic_improvement_ratio: float,
    random_state: int = 42,
) -> Tuple[GaussianMixture, int, Dict, Dict[int, GaussianMixture]]:
    n_frames, n_feat = X.shape
    max_k = min(max_states, n_frames)
    bic_by_k: Dict[int, float] = {}
    models: Dict[int, GaussianMixture] = {}

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=random_state,
        )
        gmm.fit(X)
        bic_by_k[k] = float(gmm.bic(X))
        models[k] = gmm

    best_k = 1
    best_bic = bic_by_k[1]
    for k in range(2, max_k + 1):
        prev_bic = bic_by_k[k - 1]
        cur_bic = bic_by_k[k]
        improvement = (prev_bic - cur_bic) / max(abs(prev_bic), 1e-6)
        if improvement >= bic_improvement_ratio and cur_bic < best_bic:
            best_k = k
            best_bic = cur_bic
        elif cur_bic < best_bic and improvement >= bic_improvement_ratio * 0.5:
            best_k = k
            best_bic = cur_bic

    info = {
        "bic_by_k": bic_by_k,
        "selected_k": best_k,
        "best_bic": best_bic,
        "bic_improvement_ratio": bic_improvement_ratio,
    }
    return models[best_k], best_k, info, models


def validate_gmm_state_occupancy(
    gmm: GaussianMixture,
    X: np.ndarray,
    min_state_count: int = 2,
    min_state_fraction: float = 0.08,
) -> Tuple[bool, Dict]:
    """检查 GMM 各分量在序列上的占比是否足够。"""
    if gmm.n_components <= 1:
        return True, {"counts": {0: len(X)}, "fractions": {0: 1.0}}

    labels = gmm.predict(X)
    n = max(len(labels), 1)
    counts = {k: int(np.sum(labels == k)) for k in range(gmm.n_components)}
    fractions = {k: counts[k] / n for k in counts}
    ok = all(
        counts[k] >= min_state_count and fractions[k] >= min_state_fraction
        for k in counts
    )
    return ok, {"counts": counts, "fractions": fractions}


def select_gmm_with_occupancy_validation(
    X: np.ndarray,
    models: Dict[int, GaussianMixture],
    gmm_info: Dict,
    min_state_count: int,
    min_state_fraction: float,
) -> Tuple[GaussianMixture, int, Dict]:
    """BIC 选 K 后做占比检查，不稳定则退回较小 K。"""
    original_k = int(gmm_info["selected_k"])
    selected_k = original_k
    info = dict(gmm_info)

    while selected_k >= 1:
        gmm = models[selected_k]
        ok, occ = validate_gmm_state_occupancy(
            gmm, X, min_state_count, min_state_fraction
        )
        info["occupancy"] = occ
        info["occupancy_valid"] = ok
        info["selected_k"] = selected_k
        info["bic_selected_k"] = original_k
        if ok:
            info["gmm_unstable"] = original_k >= 2 and selected_k < original_k
            return gmm, selected_k, info
        selected_k -= 1

    gmm = models[1]
    info["selected_k"] = 1
    info["occupancy_valid"] = True
    info["bic_selected_k"] = original_k
    info["gmm_unstable"] = original_k >= 2
    return gmm, 1, info


# ============================================================
# 6. 状态热度排序
# ============================================================

def compute_state_thermal_scores(
    gmm: GaussianMixture,
    feature_cols: List[str],
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    col_idx = {c: i for i, c in enumerate(feature_cols)}
    for label in range(gmm.n_components):
        vals = []
        for fname in THERMAL_FEATURE_NAMES:
            if fname in col_idx:
                vals.append(float(gmm.means_[label, col_idx[fname]]))
        scores[label] = float(np.mean(vals)) if vals else float(label)
    return scores


def build_state_rank_mapping(
    state_thermal_scores: Dict[int, float],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    sorted_labels = sorted(state_thermal_scores.keys(), key=lambda l: state_thermal_scores[l])
    label_to_rank = {lbl: rank for rank, lbl in enumerate(sorted_labels)}
    rank_to_label = {rank: lbl for lbl, rank in label_to_rank.items()}
    return label_to_rank, rank_to_label


# ============================================================
# 7. 状态持续性平滑与 change_score
# ============================================================

def smooth_state_sequence_by_run_length(states: np.ndarray, min_run: int) -> np.ndarray:
    if len(states) == 0 or min_run <= 1:
        return states.copy()

    out = states.copy()
    i = 0
    n = len(out)
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_run:
            if i > 0:
                out[i:j] = out[i - 1]
            elif j < n:
                out[i:j] = out[j]
        i = j
    return out


def compute_change_score(X: np.ndarray, window: int = 3) -> np.ndarray:
    n = len(X)
    scores = np.zeros(n, dtype=np.float32)
    if n < 2 * window + 1:
        return scores
    for t in range(window, n - window):
        pre = X[t - window : t]
        post = X[t : t + window]
        diff = np.mean(post, axis=0) - np.mean(pre, axis=0)
        diff_pos = np.maximum(diff, 0)
        scores[t] = float(np.sqrt(np.sum(diff_pos ** 2)))
    if np.max(scores) > np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-6)
    return scores


def build_raw_thermal_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """构建固定原始热特征矩阵，每列做序列内 normalize_to_01。"""
    cols = []
    for col in RAW_THERMAL_CHANGE_FEATURES:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(np.float32)
        cols.append(normalize_to_01(vals))
    if not cols:
        return np.zeros((len(df), 1), dtype=np.float32)
    return np.stack(cols, axis=1).astype(np.float32)


def compute_raw_thermal_change_score(
    df: pd.DataFrame,
    window: int = 3,
) -> np.ndarray:
    """用固定原始热特征生成统一 change score，供 onset refinement 使用。"""
    X = build_raw_thermal_feature_matrix(df)
    return compute_change_score(X, window=window)


# ============================================================
# 8. 点火候选解释
# ============================================================

def _state_separation_score(state_thermal_scores: Dict[int, float], n_ranks: int) -> float:
    if n_ranks < 2:
        return 0.0
    sorted_vals = sorted(state_thermal_scores.values())
    if len(sorted_vals) < 2:
        return 0.0
    gaps = [sorted_vals[i + 1] - sorted_vals[i] for i in range(len(sorted_vals) - 1)]
    return float(np.clip(np.mean(gaps) / (np.std(sorted_vals) + 1e-6), 0, 1))


def _segment_stability_score(states: np.ndarray, transition_idx: int) -> float:
    if transition_idx <= 0 or transition_idx >= len(states):
        return 0.5
    pre_run = transition_idx
    post_run = 0
    for t in range(transition_idx, len(states)):
        if states[t] == states[transition_idx]:
            post_run += 1
        else:
            break
    return float(np.clip(min(pre_run, post_run) / 5.0, 0, 1))


def find_stable_combustion_frame(
    states: np.ndarray,
    onset_idx: int,
    min_stable_run: int,
) -> Optional[int]:
    """
    在点火后寻找稳定燃烧帧：点火后连续 min_stable_run 帧处于最高热态 rank。
    返回该稳定段的中点帧索引；找不到则尝试最长热态 run 的中点。
    """
    n = len(states)
    if onset_idx < 0 or onset_idx >= n or min_stable_run <= 0:
        return None

    post = states[onset_idx:]
    if len(post) < min_stable_run:
        return None

    max_rank = int(np.max(post))
    if max_rank <= 0:
        return None

    start = onset_idx + min_stable_run - 1
    for t in range(start, n):
        window = states[t - min_stable_run + 1 : t + 1]
        if np.all(window == max_rank):
            run_start = t - min_stable_run + 1
            return int(run_start + min_stable_run // 2)

    best_t: Optional[int] = None
    best_len = 0
    i = onset_idx
    while i < n:
        j = i + 1
        while j < n and states[j] == states[i]:
            j += 1
        run_len = j - i
        if states[i] > 0 and run_len >= min_stable_run and run_len >= best_len:
            best_len = run_len
            best_t = int(i + run_len // 2)
        i = j
    return best_t


def _empty_keyframe_fields(no_idx: int) -> Dict:
    return {
        "stable_combustion_idx": no_idx,
        "stable_combustion_filename": "",
        "ignition_onset_image": "",
        "stable_combustion_image": "",
    }


def save_keyframe_images(
    image_paths: List[str],
    result: Dict,
    output_dir: str,
    quiet: bool = False,
) -> Dict[str, str]:
    """仅 has_ignition 时保存点火帧与稳定燃烧帧到 track 输出目录。"""
    saved = {"ignition_onset_image": "", "stable_combustion_image": ""}
    if not result.get("has_ignition"):
        return saved

    onset = int(result.get("ignition_idx_onset", -1))
    stable = int(result.get("stable_combustion_idx", -1))

    if 0 <= onset < len(image_paths):
        dst = os.path.join(output_dir, "ignition_onset.png")
        shutil.copy2(image_paths[onset], dst)
        saved["ignition_onset_image"] = dst
        if not quiet:
            print(f"[OK] Saved ignition frame: {dst}")

    if 0 <= stable < len(image_paths):
        dst = os.path.join(output_dir, "stable_combustion.png")
        shutil.copy2(image_paths[stable], dst)
        saved["stable_combustion_image"] = dst
        if not quiet:
            print(f"[OK] Saved stable combustion frame: {dst}")

    return saved


def find_first_valid_cold_to_hot_transition(
    states: np.ndarray,
    min_cold_run: int,
    min_hot_run: int,
    require_all_prior_cold: bool = True,
) -> Optional[int]:
    """
    找第一次有效 cold→hot GMM thermal state 转移：
    转移前连续 min_cold_run 帧为 rank 0，转移后连续 min_hot_run 帧 rank > 0。
    require_all_prior_cold=True 时，候选冷态段之前的全部历史也必须为 rank 0，
    防止把冷却后的再次升温误判为首次点火。
    """
    n = len(states)
    if n < min_cold_run + min_hot_run:
        return None

    for t in range(min_cold_run, n - min_hot_run + 1):
        pre = states[t - min_cold_run : t]
        post = states[t : t + min_hot_run]
        prior = states[: t - min_cold_run]
        prior_is_cold = (
            not require_all_prior_cold
            or len(prior) == 0
            or np.all(prior == 0)
        )
        if prior_is_cold and np.all(pre == 0) and np.all(post > 0):
            return t
    return None


def interpret_gmm_state_sequence_for_ignition(
    states_rank: np.ndarray,
    df: pd.DataFrame,
    change_score: np.ndarray,
    cfg: IgnitionGMMConfig,
    gmm_info: Dict,
    state_thermal_scores: Dict[int, float],
) -> Dict:
    no_idx = cfg.no_ignition_index
    n = len(states_rank)
    selected_k = int(gmm_info.get("selected_k", 1))

    base = {
        "has_ignition": False,
        "ignition_status": "ambiguous",
        "ignition_idx_onset": no_idx,
        "ignition_idx_confirmed": no_idx,
        "ignition_idx_candidate": no_idx,
        "candidate_filename": "",
        "confirmed_filename": "",
        "ignition_confidence": 0.0,
        "ignition_amplitude": 0.0,
        "ignition_change_peak": 0.0,
        "n_ignition_candidates": 0,
        "all_ignition_candidates": [],
        "gmm_n_states": selected_k,
        "gmm_bic_by_k": gmm_info.get("bic_by_k", {}),
        "gmm_state_thermal_scores": state_thermal_scores,
        "ignition_method": "gmm_rtv",
        "raw_transition_validation_enabled": bool(cfg.enable_raw_transition_validation),
        **_empty_keyframe_fields(no_idx),
    }

    if n < cfg.min_sequence_len:
        base["ignition_status"] = "ambiguous"
        return base

    if gmm_info.get("gmm_unstable") and selected_k == 1:
        base["ignition_status"] = "ambiguous_gmm_unstable"
        return base

    if selected_k == 1:
        base["ignition_status"] = "single_state"
        return base

    if gmm_info.get("occupancy_valid") is False:
        base["ignition_status"] = "ambiguous_gmm_unstable"
        return base

    has_cold = bool(np.any(states_rank == 0))
    starts_hot = int(states_rank[0]) > 0

    if starts_hot and not has_cold:
        base["ignition_status"] = "already_hot"
        return base

    transition_idx = find_first_valid_cold_to_hot_transition(
        states_rank,
        cfg.min_cold_run,
        cfg.min_hot_run,
        require_all_prior_cold=cfg.require_all_prior_cold,
    )

    if transition_idx is None:
        if np.all(states_rank == 0):
            base["ignition_status"] = "no_ignition"
        elif not has_cold:
            base["ignition_status"] = "already_hot"
        else:
            base["ignition_status"] = "ambiguous"
        return base

    confirmed_idx = transition_idx
    onset_idx = transition_idx

    if cfg.refine_onset_with_change and len(change_score) == n:
        left = max(0, transition_idx - cfg.refine_radius)
        right = min(n, transition_idx + cfg.refine_radius + 1)
        local = change_score[left:right]
        if len(local) > 0:
            onset_idx = int(left + np.argmax(local))

    has_raw_increase = True
    raw_transition_metrics: Dict[str, float] = {}
    if cfg.enable_raw_transition_validation:
        has_raw_increase, raw_transition_metrics = raw_transition_has_thermal_increase(
            df, transition_idx, cfg
        )
        if not has_raw_increase:
            base["ignition_status"] = "no_ignition_weak_transition"
            base["raw_transition_metrics"] = raw_transition_metrics
            return base

    if raw_transition_metrics:
        base["raw_transition_metrics"] = raw_transition_metrics

    local_change = float(change_score[onset_idx]) if onset_idx < len(change_score) else 0.0
    sep_score = _state_separation_score(state_thermal_scores, selected_k)
    stab_score = _segment_stability_score(states_rank, transition_idx)
    confidence = float(np.clip(
        0.4 * local_change + 0.3 * sep_score + 0.3 * stab_score, 0, 1
    ))

    hot_score = compute_hot_score(df)
    pre_mean = float(np.mean(hot_score[max(0, transition_idx - 3) : transition_idx])) if transition_idx > 0 else 0.0
    post_mean = float(np.mean(hot_score[transition_idx : min(n, transition_idx + 3)]))
    amplitude = post_mean - pre_mean

    stable_idx = find_stable_combustion_frame(
        states_rank, onset_idx, cfg.min_stable_run
    )
    if stable_idx is None:
        stable_idx = no_idx

    candidate = {
        "onset_idx": onset_idx,
        "confirmed_idx": confirmed_idx,
        "transition_idx": transition_idx,
        "stable_combustion_idx": stable_idx,
        "confidence": confidence,
        "amplitude": amplitude,
        "change_peak": local_change,
    }

    base.update({
        "has_ignition": True,
        "ignition_status": "ignition_detected",
        "ignition_idx_onset": onset_idx,
        "ignition_idx_confirmed": confirmed_idx,
        "ignition_idx_candidate": onset_idx,
        "candidate_filename": _filename_at(df, onset_idx),
        "confirmed_filename": _filename_at(df, confirmed_idx),
        "stable_combustion_idx": stable_idx,
        "stable_combustion_filename": _filename_at(df, stable_idx),
        "ignition_confidence": confidence,
        "ignition_amplitude": amplitude,
        "ignition_change_peak": local_change,
        "n_ignition_candidates": 1,
        "all_ignition_candidates": [candidate],
    })
    return base


# ============================================================
# 9. GMM-RTV 主检测
# ============================================================

def detect_ignition_gmm_rtv(df: pd.DataFrame, cfg: IgnitionGMMConfig) -> Dict:
    n = len(df)
    no_idx = cfg.no_ignition_index

    if n < cfg.min_sequence_len:
        hot_score = compute_hot_score(df)
        return {
            "has_ignition": False,
            "ignition_status": "ambiguous",
            "ignition_idx_onset": no_idx,
            "ignition_idx_confirmed": no_idx,
            "ignition_idx_candidate": no_idx,
            "candidate_filename": "",
            "confirmed_filename": "",
            "ignition_confidence": 0.0,
            "ignition_amplitude": 0.0,
            "ignition_change_peak": 0.0,
            "n_ignition_candidates": 0,
            "all_ignition_candidates": [],
            "gmm_n_states": 0,
            "gmm_state_sequence": [],
            "gmm_state_thermal_scores": {},
            "gmm_bic_by_k": {},
            "gmm_feature_cols": [],
            "ignition_method": "gmm_rtv",
            "raw_transition_validation_enabled": bool(cfg.enable_raw_transition_validation),
            "hot_score": hot_score,
            "change_score": np.zeros(n, dtype=np.float32),
            **_empty_keyframe_fields(no_idx),
        }

    # 先检查 ROI 几何质量；仅在达到拒绝条件时才排除，单一轻度异常只 warning。
    shape_reject, shape_metrics = check_roi_shape_sequence_quality(df, cfg)
    if shape_reject:
        return _make_precheck_result(
            df, cfg,
            status="rejected_shape_sequence",
            diagnostics={"shape": shape_metrics},
        )

    # 再用绝对亮度检查起点已经燃烧的序列；必须在稳健归一化和 GMM 之前完成。
    initially_hot, initial_hot_metrics = is_initially_hot_sequence(df, cfg)
    if initially_hot:
        return _make_precheck_result(
            df, cfg,
            status="already_hot_at_sequence_start",
            diagnostics={"shape": shape_metrics, "initial_hot": initial_hot_metrics},
        )

    Xn, feature_cols = build_gmm_feature_matrix(df, cfg)
    gmm, selected_k, gmm_info, all_models = fit_gmm_models_bic(
        Xn,
        max_states=cfg.gmm_max_states,
        covariance_type=cfg.gmm_covariance_type,
        reg_covar=cfg.gmm_reg_covar,
        n_init=cfg.gmm_n_init,
        bic_improvement_ratio=cfg.gmm_bic_improvement_ratio,
        random_state=cfg.random_state,
    )
    gmm, selected_k, gmm_info = select_gmm_with_occupancy_validation(
        Xn,
        all_models,
        gmm_info,
        min_state_count=cfg.gmm_min_state_count,
        min_state_fraction=cfg.gmm_min_state_fraction,
    )

    # 统一 raw thermal change score，避免 onset refinement 随 GMM 特征维度偏置。
    change_score = compute_raw_thermal_change_score(df, window=3)
    hot_score = compute_hot_score(df)

    if selected_k == 1:
        thermal_scores = compute_state_thermal_scores(gmm, feature_cols)
        states_zero = np.zeros(n, dtype=np.int32)
        result = interpret_gmm_state_sequence_for_ignition(
            states_zero, df, change_score, cfg, gmm_info, thermal_scores
        )
        result["gmm_state_sequence"] = [0] * n
        result["gmm_feature_cols"] = feature_cols
        result["hot_score"] = hot_score
        result["change_score"] = change_score
        result["raw_transition_validation_enabled"] = bool(cfg.enable_raw_transition_validation)
        result["precheck_diagnostics"] = {
            "shape": shape_metrics,
            "initial_hot": initial_hot_metrics,
        }
        _attach_shape_fields_to_result(result, shape_metrics)
        return result

    thermal_scores = compute_state_thermal_scores(gmm, feature_cols)
    label_to_rank, _rank_to_label = build_state_rank_mapping(thermal_scores)

    gmm_labels = gmm.predict(Xn)
    states_rank = np.asarray(
        [label_to_rank[int(label)] for label in gmm_labels],
        dtype=np.int32,
    )
    states_rank = smooth_state_sequence_by_run_length(
        states_rank,
        cfg.min_state_run,
    )

    result = interpret_gmm_state_sequence_for_ignition(
        states_rank, df, change_score, cfg, gmm_info, thermal_scores
    )
    result["gmm_state_sequence"] = states_rank.tolist()
    result["gmm_feature_cols"] = feature_cols
    result["hot_score"] = hot_score
    result["change_score"] = change_score
    result["label_to_rank"] = label_to_rank
    result["raw_transition_validation_enabled"] = bool(cfg.enable_raw_transition_validation)
    result["precheck_diagnostics"] = {
        "shape": shape_metrics,
        "initial_hot": initial_hot_metrics,
    }
    _attach_shape_fields_to_result(result, shape_metrics)
    return result


# ============================================================
# 10. 单 track / 批处理
# ============================================================

def _summary_from_gmm_result(result: Dict, track_name: str, folder: str, n_frames: int) -> Dict:
    shape = result.get("precheck_diagnostics", {}).get("shape", {}) if isinstance(result.get("precheck_diagnostics"), dict) else {}
    warning_reasons = result.get("shape_warning_reasons")
    if warning_reasons is None:
        warning_reasons = shape.get("shape_warning_reasons", [])
    reject_reasons = result.get("shape_reject_reasons")
    if reject_reasons is None:
        reject_reasons = shape.get("shape_reject_reasons", shape.get("reject_reasons", []))

    def _join_reasons(vals: Any) -> str:
        if not vals:
            return ""
        if isinstance(vals, str):
            return vals
        return "|".join(str(v) for v in vals if str(v))

    return {
        "track_name": track_name,
        "track_id": track_id_from_name(track_name),
        "track_folder": folder,
        "n_frames": n_frames,
        "mode": "gmm_rtv",
        "status": "ok",
        "error": "",
        "has_ignition": bool(result.get("has_ignition", False)),
        "ignition_status": str(result.get("ignition_status", "")),
        "ignition_idx_onset": int(result.get("ignition_idx_onset", -1)),
        "ignition_idx_confirmed": int(result.get("ignition_idx_confirmed", -1)),
        "ignition_confidence": float(result.get("ignition_confidence", 0.0)),
        "ignition_amplitude": float(result.get("ignition_amplitude", 0.0)),
        "ignition_change_peak": float(result.get("ignition_change_peak", 0.0)),
        "gmm_n_states": int(result.get("gmm_n_states", 0)),
        "n_ignition_candidates": int(result.get("n_ignition_candidates", 0)),
        "raw_transition_validation_enabled": bool(
            result.get("raw_transition_validation_enabled", False)
        ),
        "ignition_idx_candidate": int(result.get("ignition_idx_candidate", -1)),
        "candidate_filename": str(result.get("candidate_filename", "")),
        "confirmed_filename": str(result.get("confirmed_filename", "")),
        "stable_combustion_idx": int(result.get("stable_combustion_idx", -1)),
        "stable_combustion_filename": str(result.get("stable_combustion_filename", "")),
        "ignition_onset_image": str(result.get("ignition_onset_image", "")),
        "stable_combustion_image": str(result.get("stable_combustion_image", "")),
        "precheck_rejected": bool(result.get("precheck_rejected", False)),
        "shape_quality_warning": bool(result.get("shape_quality_warning", shape.get("shape_quality_warning", False))),
        "shape_warning_reasons": _join_reasons(warning_reasons),
        "shape_reject_reasons": _join_reasons(reject_reasons),
        "shape_evidence_count": int(result.get("shape_evidence_count", shape.get("shape_evidence_count", 0)) or 0),
        "shape_jump_failure": bool(result.get("shape_jump_failure", shape.get("shape_jump_failure", False))),
        "shape_global_failure": bool(result.get("shape_global_failure", shape.get("shape_global_failure", False))),
        "shape_frequent_failure": bool(result.get("shape_frequent_failure", shape.get("shape_frequent_failure", False))),
        "shape_invalid_failure": bool(result.get("shape_invalid_failure", shape.get("shape_invalid_failure", False))),
        "shape_extreme_failure": bool(result.get("shape_extreme_failure", shape.get("shape_extreme_failure", False))),
        "max_area_jump_ratio": float(result.get("max_area_jump_ratio", shape.get("max_area_jump_ratio", 1.0)) or 1.0),
        "max_aspect_jump_ratio": float(result.get("max_aspect_jump_ratio", shape.get("max_aspect_jump_ratio", 1.0)) or 1.0),
        "global_area_ratio_q95_q05": float(
            result.get("global_area_ratio_q95_q05", shape.get("global_area_ratio_q95_q05", 1.0)) or 1.0
        ),
        "global_aspect_ratio_q95_q05": float(
            result.get("global_aspect_ratio_q95_q05", shape.get("global_aspect_ratio_q95_q05", 1.0)) or 1.0
        ),
        "bad_jump_fraction": float(
            result.get(
                "bad_jump_fraction",
                shape.get("bad_jump_fraction", shape.get("bad_geometry_jump_fraction", 0.0)),
            )
            or 0.0
        ),
        "invalid_roi_fraction": float(
            result.get(
                "invalid_roi_fraction",
                shape.get("invalid_roi_fraction", shape.get("invalid_geometry_fraction", 0.0)),
            )
            or 0.0
        ),
        "initial_hot_frame_fraction": float(
            result.get("precheck_diagnostics", {}).get("initial_hot", {}).get("initial_hot_frame_fraction", 0.0)
            if isinstance(result.get("precheck_diagnostics"), dict)
            else 0.0
        ),
    }


def detect_folder_gmm_rtv(
    folder: str,
    output_dir: str,
    cfg: Optional[IgnitionGMMConfig] = None,
    track_name: Optional[str] = None,
    save_plot: bool = True,
    quiet: bool = False,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    if cfg is None:
        cfg = IgnitionGMMConfig()

    track_name = track_name or Path(folder).name
    image_paths = list_images_in_folder(folder)
    if not image_paths:
        raise RuntimeError(f"No images found in folder: {folder}")

    if not quiet:
        print(f"[INFO] Loaded {len(image_paths)} images from {folder}")

    df = extract_sequence_features(image_paths, cfg)
    result = detect_ignition_gmm_rtv(df, cfg)

    df["hot_score"] = result["hot_score"]
    df["change_score"] = result["change_score"]
    if result.get("gmm_state_sequence"):
        df["gmm_state_rank"] = result["gmm_state_sequence"]

    summary = _summary_from_gmm_result(result, track_name, folder, len(df))

    if result.get("has_ignition"):
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = save_keyframe_images(image_paths, result, output_dir, quiet=quiet)
        result.update(saved_paths)
        summary.update(saved_paths)

        if cfg.save_features:
            save_table(df, os.path.join(output_dir, "frame_features"), quiet=quiet)

        result_for_json = {k: to_jsonable(v) for k, v in result.items() if not isinstance(v, np.ndarray)}
        result_for_json["mode"] = "gmm_rtv"
        result_for_json.update({k: summary[k] for k in summary if k not in ("track_folder", "error", "status")})
        result_json = os.path.join(output_dir, "ignition_result.json")
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(result_for_json, f, indent=2, ensure_ascii=False)
        if not quiet:
            print(f"[OK] Result saved to: {result_json}")

        if save_plot:
            plot_detection_result_gmm_rtv(
                df, result,
                save_path=os.path.join(output_dir, "ignition_detection_curve.png"),
                show=False,
            )
    elif not quiet:
        print(f"[SKIP] No ignition in {track_name}, track output folder not created.")

    if not quiet:
        print(f"\n========== Ignition Detection Result ==========")
        print(f"Track: {track_name}")
        print(f"Ignition status: {summary['ignition_status']}")
        print(f"Has ignition: {summary['has_ignition']}")
        print(f"Onset index: {summary['ignition_idx_onset']}")
        print(f"Confirmed index: {summary['ignition_idx_confirmed']}")
        if summary["has_ignition"]:
            print(f"Candidate filename: {summary['candidate_filename']}")
            print(f"Confirmed filename: {summary['confirmed_filename']}")
            print(f"Stable combustion index: {summary['stable_combustion_idx']}")
            print(f"Stable combustion filename: {summary['stable_combustion_filename']}")
            print(f"Confidence: {summary['ignition_confidence']:.3f}")

    return df, result, summary


def resolve_workers(workers: int) -> int:
    if workers > 0:
        return workers
    cpu = os.cpu_count() or 4
    return max(1, cpu - 1)


def _error_summary_gmm(track_name: str, folder: str, exc: Exception) -> Dict:
    return {
        "track_name": track_name,
        "track_id": track_id_from_name(track_name),
        "track_folder": folder,
        "n_frames": 0,
        "mode": "gmm_rtv",
        "status": "error",
        "error": str(exc),
        "has_ignition": False,
        "ignition_status": "error",
        "ignition_idx_onset": "",
        "ignition_idx_confirmed": "",
        "ignition_confidence": "",
        "ignition_amplitude": "",
        "ignition_change_peak": "",
        "gmm_n_states": "",
        "n_ignition_candidates": "",
        "raw_transition_validation_enabled": "",
        "ignition_idx_candidate": "",
        "candidate_filename": "",
        "confirmed_filename": "",
        "stable_combustion_idx": "",
        "stable_combustion_filename": "",
        "ignition_onset_image": "",
        "stable_combustion_image": "",
        "precheck_rejected": False,
        "shape_quality_warning": False,
        "shape_warning_reasons": "",
        "shape_reject_reasons": "",
        "shape_evidence_count": "",
        "shape_jump_failure": "",
        "shape_global_failure": "",
        "shape_frequent_failure": "",
        "shape_invalid_failure": "",
        "shape_extreme_failure": "",
        "max_area_jump_ratio": "",
        "max_aspect_jump_ratio": "",
        "global_area_ratio_q95_q05": "",
        "global_aspect_ratio_q95_q05": "",
        "bad_jump_fraction": "",
        "invalid_roi_fraction": "",
        "initial_hot_frame_fraction": "",
    }


def _process_one_track_gmm(task: Dict) -> Dict:
    try:
        cfg = IgnitionGMMConfig(**task["cfg_dict"])
        _, _, summary = detect_folder_gmm_rtv(
            folder=task["folder"],
            output_dir=task["output_dir"],
            cfg=cfg,
            track_name=task["track_name"],
            save_plot=task["save_plot"],
            quiet=True,
        )
        return summary
    except Exception as exc:
        return _error_summary_gmm(task["track_name"], task["folder"], exc)


def detect_tracks_batch_gmm_rtv(
    tracks_root: str,
    output_dir: str,
    cfg: Optional[IgnitionGMMConfig] = None,
    start_from: int = 1,
    track_prefix: str = "track_",
    save_plot: bool = True,
    workers: int = 0,
) -> pd.DataFrame:
    """
    批处理 tracks_root 下所有 track_* 文件夹。

    输出结构:
        output_dir/
            summary.csv / summary.xlsx
            track_N/   （仅 has_ignition=True 的 track）
                frame_features.csv, ignition_result.json,
                ignition_detection_curve.png, ignition_onset.png, stable_combustion.png
    """
    if cfg is None:
        cfg = IgnitionGMMConfig()

    os.makedirs(output_dir, exist_ok=True)
    track_folders = list_track_folders(tracks_root, track_prefix=track_prefix, start_from=start_from)
    if not track_folders:
        raise RuntimeError(f"No track folders found under: {tracks_root}")

    print(f"[INFO] Batch root: {tracks_root}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Found {len(track_folders)} track folder(s)")

    n_workers = resolve_workers(workers)
    print(f"[INFO] Workers: {n_workers}")

    cfg_dict = asdict(cfg)
    tasks = [
        {
            "track_name": track_name,
            "folder": folder,
            "output_dir": os.path.join(output_dir, track_name),
            "cfg_dict": cfg_dict,
            "save_plot": save_plot,
        }
        for track_name, folder in track_folders
    ]

    summary_rows: List[Dict] = []
    if n_workers == 1:
        for idx, task in enumerate(tasks, start=1):
            print(f"\n{'=' * 60}")
            print(f"[{idx}/{len(tasks)}] {task['track_name']}")
            try:
                summary = _process_one_track_gmm(task)
            except Exception as exc:
                summary = _error_summary_gmm(task["track_name"], task["folder"], exc)
            summary_rows.append(summary)
            print(
                f"[{idx}/{len(tasks)}] {summary['track_name']} "
                f"status={summary['status']} "
                f"ignition={summary.get('ignition_status', '')} "
                f"onset={summary.get('ignition_idx_onset', '')} "
                f"confirmed={summary.get('ignition_idx_confirmed', '')}"
            )
    else:
        done = 0
        total = len(tasks)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {executor.submit(_process_one_track_gmm, t): t for t in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                done += 1
                try:
                    summary = future.result()
                except Exception as exc:
                    summary = _error_summary_gmm(task["track_name"], task["folder"], exc)
                summary_rows.append(summary)
                print(
                    f"[{done}/{total}] {summary['track_name']} "
                    f"status={summary['status']} "
                    f"ignition={summary.get('ignition_status', '')} "
                    f"onset={summary.get('ignition_idx_onset', '')} "
                    f"confirmed={summary.get('ignition_idx_confirmed', '')}"
                )

    summary_df = pd.DataFrame(summary_rows)
    if "track_id" in summary_df.columns:
        summary_df = summary_df.sort_values(
            by=["track_id", "track_name"],
            na_position="last",
        ).reset_index(drop=True)
    save_table(summary_df, os.path.join(output_dir, "summary"))

    ok_count = int((summary_df["status"] == "ok").sum()) if len(summary_df) else 0
    print(f"\n=== Batch Done ===")
    print(f"tracks={len(summary_df)} success={ok_count} failed={len(summary_df) - ok_count}")
    print(f"summary -> {os.path.join(output_dir, 'summary.csv')}")
    return summary_df


# ============================================================
# 11. 绘图
# ============================================================

def plot_detection_result_gmm_rtv(
    df: pd.DataFrame,
    result: Dict,
    save_path: Optional[str] = None,
    show: bool = True,
):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(df))
    plt.figure(figsize=(12, 6))

    if "gray_p99" in df.columns:
        plt.plot(x, normalize_to_01(df["gray_p99"].values), label="gray_p99")
    if "bright_area_ratio" in df.columns:
        plt.plot(x, normalize_to_01(df["bright_area_ratio"].values), label="bright_area_ratio")
    if "bright_energy_mean" in df.columns:
        plt.plot(x, normalize_to_01(df["bright_energy_mean"].values), label="bright_energy_mean")
    if "hot_score" in result:
        plt.plot(x, result["hot_score"], label="hot_score", linewidth=2, alpha=0.7)
    if "change_score" in result:
        plt.plot(x, result["change_score"], label="change_score", linewidth=1.5, alpha=0.7)

    seq = result.get("gmm_state_sequence", [])
    if seq and len(seq) == len(df):
        k = max(int(result.get("gmm_n_states", 1)), 1)
        plt.step(x, np.array(seq) / max(k - 1, 1), where="mid", label="gmm_state_rank", linewidth=2)

    status = result.get("ignition_status", "")
    k_states = result.get("gmm_n_states", 0)
    if result.get("has_ignition"):
        onset = int(result.get("ignition_idx_onset", -1))
        confirmed = int(result.get("ignition_idx_confirmed", -1))
        if onset >= 0:
            plt.axvline(onset, linestyle=":", color="orange", linewidth=1.5, label=f"onset={onset}")
        if confirmed >= 0 and confirmed != onset:
            plt.axvline(confirmed, linestyle="--", color="red", linewidth=1.5, label=f"confirmed={confirmed}")
        stable = int(result.get("stable_combustion_idx", -1))
        if stable >= 0 and stable not in (onset, confirmed):
            plt.axvline(stable, linestyle="-.", color="green", linewidth=1.5, label=f"stable={stable}")

    plt.xlabel("Frame index")
    plt.ylabel("Normalized score / state rank")
    plt.title(f"Ignition status: {status} | method: gmm_rtv | states: K={k_states}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        if show:
            print(f"[OK] Plot saved to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# 12. CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GMM 状态分割与原始热转移验证点火检测")
    p.add_argument(
        "--tracks-root",
        type=str,
        default='inputs/track_crop',
        help="包含 track_1, track_2, ... 的根目录",
    )
    p.add_argument(
        "--track-folder",
        type=str,
        default=None,
        help="单 track 模式：指定单个 track 文件夹",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；批处理默认 ignition_results/run_<timestamp>",
    )
    p.add_argument("--start-from", type=int, default=1, help="从 track_N 开始处理（含 N）")
    p.add_argument("--track-prefix", type=str, default="track_")
    p.add_argument(
        "--save-plot",
        dest="save_plot",
        action="store_true",
        default=True,
        help="为每个 track 保存 ignition_detection_curve.png（默认开启）",
    )
    p.add_argument(
        "--no-save-plot",
        dest="save_plot",
        action="store_false",
        help="不保存曲线图",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="单 track 模式下弹窗显示曲线（批处理时忽略）",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行进程数；0 表示自动（CPU 核心数 - 1），1 表示串行",
    )

    p.add_argument("--bright-lambda", type=float, default=6.0)
    p.add_argument("--smooth-window", type=int, default=5)
    p.add_argument("--gmm-max-states", type=int, default=3)
    p.add_argument("--gmm-min-state-count", type=int, default=2)
    p.add_argument("--gmm-min-state-fraction", type=float, default=0.08)
    p.add_argument("--gmm-bic-improvement-ratio", type=float, default=0.03)
    p.add_argument("--min-state-run", type=int, default=2)
    p.add_argument("--min-cold-run", type=int, default=3)
    p.add_argument("--min-hot-run", type=int, default=3)
    p.add_argument("--min-stable-run", type=int, default=3, help="稳定燃烧段最少连续热态帧数")
    p.add_argument("--min-sequence-len", type=int, default=8)
    p.add_argument("--refine-radius", type=int, default=2)
    p.add_argument("--no-diff-features", dest="use_diff_features", action="store_false", default=True)
    p.add_argument(
        "--prior-cold-guard",
        dest="require_all_prior_cold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="要求候选冷态段之前的全部状态历史均为冷态",
    )
    p.add_argument(
        "--refine-onset",
        dest="refine_onset_with_change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="在候选转移附近用原始热变化分数微调 onset",
    )
    p.add_argument(
        "--no-initial-hot-reject",
        dest="enable_initial_hot_reject",
        action="store_false",
        default=True,
        help="关闭序列起点已燃烧预检查（默认开启）",
    )
    p.add_argument("--initial-hot-check-frames", type=int, default=5)
    p.add_argument("--initial-hot-min-frame-fraction", type=float, default=0.80)
    p.add_argument("--initial-hot-min-feature-votes", type=int, default=3)
    p.add_argument("--initial-hot-gray-p99-min", type=float, default=160.0)
    p.add_argument("--initial-hot-hsv-v-p99-min", type=float, default=160.0)
    p.add_argument("--initial-hot-corr-p99-min", type=float, default=25.0)
    p.add_argument("--initial-hot-bright-area-min", type=float, default=0.03)
    p.add_argument("--initial-hot-bright-energy-min", type=float, default=6.0)

    p.add_argument(
        "--no-shape-sequence-reject",
        dest="enable_shape_sequence_reject",
        action="store_false",
        default=True,
        help="关闭 ROI 宽高/面积/宽高比序列拒绝（仍计算 diagnostics，并继续 GMM-RTV）",
    )
    p.add_argument("--shape-area-jump-ratio-max", type=float, default=2.5)
    p.add_argument("--shape-aspect-jump-ratio-max", type=float, default=1.8)
    p.add_argument("--shape-hard-area-jump-ratio", type=float, default=5.0)
    p.add_argument("--shape-hard-aspect-jump-ratio", type=float, default=3.0)
    p.add_argument("--shape-global-area-ratio-max", type=float, default=5.0)
    p.add_argument("--shape-global-aspect-ratio-max", type=float, default=2.5)
    p.add_argument("--shape-bad-jump-fraction-max", type=float, default=0.30)
    p.add_argument("--shape-invalid-fraction-max", type=float, default=0.10)
    p.add_argument("--shape-extreme-area-jump-ratio", type=float, default=8.0)
    p.add_argument("--shape-extreme-aspect-jump-ratio", type=float, default=5.0)

    p.add_argument(
        "--no-raw-transition-validation",
        dest="enable_raw_transition_validation",
        action="store_false",
        default=True,
        help=(
            "关闭原始热转移验证，仅依据 GMM 状态转移判断点火；"
            "用于 segmentation-only 消融"
        ),
    )
    p.add_argument("--min-raw-delta-gray-p99", type=float, default=25.0)
    p.add_argument("--min-raw-delta-corr-p99", type=float, default=20.0)
    p.add_argument("--min-raw-delta-bright-energy", type=float, default=8.0)
    p.add_argument("--min-raw-delta-bright-area", type=float, default=0.04)
    p.add_argument("--min-raw-delta-hsv-v-p99", type=float, default=25.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = IgnitionGMMConfig(
        bright_lambda=args.bright_lambda,
        smooth_window=args.smooth_window,
        gmm_max_states=args.gmm_max_states,
        gmm_min_state_count=args.gmm_min_state_count,
        gmm_min_state_fraction=args.gmm_min_state_fraction,
        gmm_bic_improvement_ratio=args.gmm_bic_improvement_ratio,
        min_state_run=args.min_state_run,
        min_cold_run=args.min_cold_run,
        min_hot_run=args.min_hot_run,
        min_stable_run=args.min_stable_run,
        min_sequence_len=args.min_sequence_len,
        require_all_prior_cold=args.require_all_prior_cold,
        refine_radius=args.refine_radius,
        use_diff_features=args.use_diff_features,
        refine_onset_with_change=args.refine_onset_with_change,
        enable_initial_hot_reject=args.enable_initial_hot_reject,
        initial_hot_check_frames=args.initial_hot_check_frames,
        initial_hot_min_frame_fraction=args.initial_hot_min_frame_fraction,
        initial_hot_min_feature_votes=args.initial_hot_min_feature_votes,
        initial_hot_gray_p99_min=args.initial_hot_gray_p99_min,
        initial_hot_hsv_v_p99_min=args.initial_hot_hsv_v_p99_min,
        initial_hot_corr_p99_min=args.initial_hot_corr_p99_min,
        initial_hot_bright_area_min=args.initial_hot_bright_area_min,
        initial_hot_bright_energy_min=args.initial_hot_bright_energy_min,
        enable_shape_sequence_reject=args.enable_shape_sequence_reject,
        shape_area_jump_ratio_max=args.shape_area_jump_ratio_max,
        shape_aspect_jump_ratio_max=args.shape_aspect_jump_ratio_max,
        shape_hard_area_jump_ratio=args.shape_hard_area_jump_ratio,
        shape_hard_aspect_jump_ratio=args.shape_hard_aspect_jump_ratio,
        shape_global_area_ratio_max=args.shape_global_area_ratio_max,
        shape_global_aspect_ratio_max=args.shape_global_aspect_ratio_max,
        shape_bad_jump_fraction_max=args.shape_bad_jump_fraction_max,
        shape_invalid_fraction_max=args.shape_invalid_fraction_max,
        shape_extreme_area_jump_ratio=args.shape_extreme_area_jump_ratio,
        shape_extreme_aspect_jump_ratio=args.shape_extreme_aspect_jump_ratio,
        enable_raw_transition_validation=args.enable_raw_transition_validation,
        min_raw_delta_gray_p99=args.min_raw_delta_gray_p99,
        min_raw_delta_corr_p99=args.min_raw_delta_corr_p99,
        min_raw_delta_bright_energy=args.min_raw_delta_bright_energy,
        min_raw_delta_bright_area=args.min_raw_delta_bright_area,
        min_raw_delta_hsv_v_p99=args.min_raw_delta_hsv_v_p99,
    )

    if args.track_folder:
        output_dir = args.output_dir or "./output_ignition"
        df, result, _ = detect_folder_gmm_rtv(
            folder=args.track_folder,
            output_dir=output_dir,
            cfg=cfg,
            save_plot=args.save_plot,
        )
        if args.show:
            plot_detection_result_gmm_rtv(df, result, show=True)
        return

    output_dir = args.output_dir or make_default_batch_output_dir()
    detect_tracks_batch_gmm_rtv(
        tracks_root=args.tracks_root,
        output_dir=output_dir,
        cfg=cfg,
        start_from=args.start_from,
        track_prefix=args.track_prefix,
        save_plot=args.save_plot,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
