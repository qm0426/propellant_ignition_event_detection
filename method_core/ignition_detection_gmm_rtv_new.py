from __future__ import annotations









from dataclasses import dataclass



from typing import Any, Dict, List, Tuple

import cv2

import numpy as np

import pandas as pd


from scipy.ndimage import binary_closing, binary_opening


from sklearn.mixture import GaussianMixture

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

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

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
