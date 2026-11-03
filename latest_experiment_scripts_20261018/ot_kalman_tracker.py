#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OT + Kalman filter tracker for particle/droplet detections.

Input:
    cleaned_all.csv from postprocess_detection_txt.py, with at least:
        frame_id,image_name,x1,y1,x2,y2,final_quality_score
    or per-frame txt detections:
        x1 y1 x2 y2 score
        particle x1 y1 x2 y2 score

Output:
    tracks.csv
    tracks_long.csv
    optional visualization video/images

Core idea:
    1. Use a constant-velocity Kalman filter for each active track.
    2. Predict track positions at the current frame.
    3. Build a cost matrix between predicted tracks and current detections.
    4. Convert the cost matrix into an entropic OT/Sinkhorn transport plan.
    5. Extract high-probability one-to-one matches.
    6. Update matched Kalman filters, create new tracks, mark unmatched tracks as missed.

This is class-agnostic. All detections are treated as particle candidates.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# IO
# -----------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect_image_paths(image_dir: Path, image_exts: str = ".png,.jpg,.jpeg,.tif,.tiff") -> List[Path]:
    exts = {e if e.startswith(".") else "." + e for e in image_exts.lower().split(",") if e.strip()}
    return [p for p in sorted(image_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def build_frame_image_map(
    image_dir: Optional[Path],
    image_exts: str,
) -> Dict[int, Path]:
    """建立 frame_id -> image_path 映射（与 visualize_tracks 的 enumerate 一致）。"""
    if image_dir is None or not image_dir.is_dir():
        return {}
    paths = collect_image_paths(image_dir, image_exts)
    return {i: p for i, p in enumerate(paths)}


def make_tracking_run_dir(base_dir: Path) -> Path:
    """在 base_dir 下创建带时间戳的子目录，避免重复运行覆盖结果。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"run_{stamp}"


def resolve_tracking_output_dir(
    output_dir: Path,
    use_run_subdir: bool = True,
) -> Path:
    """解析实际输出目录；默认在 output_dir 下追加 run_YYYYMMDD_HHMMSS。"""
    if not use_run_subdir:
        return output_dir
    return make_tracking_run_dir(output_dir)


def read_gray_image(path: Path) -> Optional[np.ndarray]:
    """安全读取灰度图，返回 float32 2D array；失败返回 None。"""
    try:
        from PIL import Image

        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        return gray
    except Exception:
        return None


def load_detections_csv(csv_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load cleaned_all.csv style detections.

    Expected fields:
        frame_id, image_name, x1, y1, x2, y2, final_quality_score
    """
    if csv_path.is_dir():
        raise ValueError(
            f"期望 CSV 文件，但得到目录: {csv_path}\n"
            f"请指定 .../postprocessed/cleaned_all.csv，"
            f"或使用 --input-format txt-dir 指向 cleaned_detections_txt/"
        )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Detections CSV not found: {csv_path}")

    dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame_id = int(float(row["frame_id"]))
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                score = float(row.get("final_quality_score", row.get("score", 1.0)) or 1.0)
            except (KeyError, ValueError, TypeError):
                continue

            w = max(1e-6, x2 - x1)
            h = max(1e-6, y2 - y1)
            det = {
                "frame_id": frame_id,
                "image_name": row.get("image_name", ""),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "w": w,
                "h": h,
                "area": w * h,
                "score": score,
            }
            dets_by_frame.setdefault(frame_id, []).append(det)
    return dets_by_frame


def load_detections_txt_dir(txt_dir: Path, max_txt: int = -1) -> Dict[int, List[Dict[str, Any]]]:
    """Load per-frame txt detections.

    The parser is intentionally simple:
        x1 y1 x2 y2 score
        label x1 y1 x2 y2 score

    Frame id is inferred from sorted txt files.
    max_txt=-1 表示全部；>=0 时只加载前 max_txt 个 txt。
    """
    dets_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    txt_dir = Path(txt_dir)
    if not txt_dir.is_dir():
        raise FileNotFoundError(f"Detections txt directory not found: {txt_dir}")
    txt_paths = sorted([p for p in txt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    total_available = len(txt_paths)
    if max_txt is not None and int(max_txt) >= 0:
        txt_paths = txt_paths[: int(max_txt)]
    print(
        f"[INFO] detections txt: using {len(txt_paths)}/{total_available} "
        f"(max_txt={max_txt})"
    )
    for frame_id, path in enumerate(txt_paths):
        frame_dets: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            nums: List[float] = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) < 4:
                continue
            if len(nums) >= 5 and len(parts) >= 6:
                x1, y1, x2, y2 = nums[:4]
                score = nums[4]
            elif len(nums) >= 5 and len(parts) == 5:
                x1, y1, x2, y2, score = nums[:5]
            else:
                x1, y1, x2, y2 = nums[:4]
                score = 1.0
            w = max(1e-6, x2 - x1)
            h = max(1e-6, y2 - y1)
            frame_dets.append(
                {
                    "frame_id": frame_id,
                    "image_name": path.stem,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "w": w,
                    "h": h,
                    "area": w * h,
                    "score": score,
                }
            )
        dets_by_frame[frame_id] = frame_dets
    return dets_by_frame


# -----------------------------
# Geometry
# -----------------------------

def xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def det_to_xyxy(det: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])


# -----------------------------
# Kalman filter
# -----------------------------

class KalmanBoxFilter:
    """Constant-velocity Kalman filter for bbox center and size.

    State:
        x = [cx, cy, vx, vy, w, h]^T

    Measurement:
        z = [cx, cy, w, h]^T
    """

    def __init__(
        self,
        cx: float,
        cy: float,
        w: float,
        h: float,
        dt: float = 1.0,
        process_pos_std: float = 2.0,
        process_vel_std: float = 8.0,
        process_size_std: float = 2.0,
        meas_pos_std: float = 6.0,
        meas_size_std: float = 8.0,
    ):
        self.dt = float(dt)

        self.x = np.array([cx, cy, 0.0, 0.0, w, h], dtype=np.float64)

        # Large initial uncertainty on velocity.
        self.P = np.diag([25.0, 25.0, 200.0, 200.0, 25.0, 25.0]).astype(np.float64)

        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt

        self.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.Q_base = np.diag(
            [
                process_pos_std**2,
                process_pos_std**2,
                process_vel_std**2,
                process_vel_std**2,
                process_size_std**2,
                process_size_std**2,
            ]
        ).astype(np.float64)

        self.R = np.diag(
            [meas_pos_std**2, meas_pos_std**2, meas_size_std**2, meas_size_std**2]
        ).astype(np.float64)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        if abs(dt - self.dt) > 1e-9:
            self.dt = float(dt)
            self.F = np.eye(6, dtype=np.float64)
            self.F[0, 2] = self.dt
            self.F[1, 3] = self.dt

        # Scale process noise slightly by dt.
        Q = self.Q_base.copy()
        Q[0, 0] *= dt**2
        Q[1, 1] *= dt**2
        Q[2, 2] *= max(dt, 1e-6)
        Q[3, 3] *= max(dt, 1e-6)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

        # Keep size physically valid.
        self.x[4] = max(1.0, self.x[4])
        self.x[5] = max(1.0, self.x[5])
        return self.x.copy()

    def predict_backward(self, dt: float = 1.0) -> np.ndarray:
        """按恒定速度模型向过去外推（用于后向卡尔曼插值）。"""
        dt = max(1.0, float(dt))
        if abs(dt - self.dt) > 1e-9:
            self.dt = float(dt)

        Q = self.Q_base.copy()
        Q[0, 0] *= dt**2
        Q[1, 1] *= dt**2
        Q[2, 2] *= max(dt, 1e-6)
        Q[3, 3] *= max(dt, 1e-6)

        self.x[0] -= self.x[2] * dt
        self.x[1] -= self.x[3] * dt
        self.P = self.P + Q

        self.x[4] = max(1.0, self.x[4])
        self.x[5] = max(1.0, self.x[5])
        return self.x.copy()

    def update(self, cx: float, cy: float, w: float, h: float) -> np.ndarray:
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        self.x[4] = max(1.0, self.x[4])
        self.x[5] = max(1.0, self.x[5])
        return self.x.copy()

    def measurement_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        z_pred = self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return z_pred, S

    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        cx, cy, _, _, w, h = self.x
        return xywh_to_xyxy(cx, cy, max(1.0, w), max(1.0, h))


# -----------------------------
# Track container
# -----------------------------

@dataclass
class Track:
    track_id: int
    kf: KalmanBoxFilter
    start_frame: int
    last_frame: int
    age: int = 1
    hits: int = 1
    misses: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_confirmed: bool = False
    track_state: str = "tentative"
    birth_reason: str = "unmatched_detection"
    birth_frame: int = 0
    birth_x: float = 0.0
    birth_y: float = 0.0
    end_reason: str = ""
    prebirth_recovered: bool = False

    def predict(self, frame_id: int) -> None:
        dt = max(1, frame_id - self.last_frame)
        self.kf.predict(dt=dt)
        self.age += 1
        self.misses += 1

    def update(self, det: Dict[str, Any], frame_id: int, ot_prob: float = 0.0, match_cost: float = 0.0) -> None:
        self.kf.update(det["cx"], det["cy"], det["w"], det["h"])
        self.last_frame = frame_id
        self.hits += 1
        self.misses = 0
        self.history.append(
            {
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
                "ot_prob": ot_prob,
                "match_cost": match_cost,
                "pred_x1": self.kf.bbox_xyxy()[0],
                "pred_y1": self.kf.bbox_xyxy()[1],
                "pred_x2": self.kf.bbox_xyxy()[2],
                "pred_y2": self.kf.bbox_xyxy()[3],
                "point_type": "detected",
                "gap_fill_method": "",
            }
        )


# -----------------------------
# OT / Sinkhorn matching
# -----------------------------

def sinkhorn_transport(
    cost: np.ndarray,
    reg: float = 0.08,
    num_iters: int = 100,
    eps: float = 1e-9,
) -> np.ndarray:
    """Balanced entropic OT using Sinkhorn scaling.

    cost: [N, M], preferably normalized to [0, 1].
    returns transport matrix [N, M] with row/column masses approximately uniform.

    For tracking, dummy rows/cols are appended before this function so unmatched
    tracks/detections are allowed.
    """
    if cost.size == 0:
        return np.zeros_like(cost, dtype=np.float64)

    c = np.asarray(cost, dtype=np.float64)
    c = np.nan_to_num(c, nan=1e6, posinf=1e6, neginf=1e6)

    # Numerical stabilization.
    c_min = np.min(c)
    c = c - c_min

    K = np.exp(-c / max(reg, eps))
    K = np.maximum(K, eps)

    n, m = K.shape
    a = np.ones(n, dtype=np.float64) / n
    b = np.ones(m, dtype=np.float64) / m

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)

    for _ in range(num_iters):
        u = a / (K @ v + eps)
        v = b / (K.T @ u + eps)

    P = (u[:, None] * K) * v[None, :]
    return P


def mahalanobis_distance(z: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    d = z - mean
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov)
    return float(math.sqrt(max(0.0, d.T @ inv @ d)))


def observed_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [h for h in history if h.get("point_type", "detected") == "detected"]


def max_contiguous_run(frame_ids: List[int]) -> int:
    if not frame_ids:
        return 0
    ids = sorted(int(f) for f in frame_ids)
    best = cur = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def classify_birth_reason(
    cx: float,
    cy: float,
    image_w: Optional[int],
    image_h: Optional[int],
    border_margin: float,
) -> str:
    """新观测轨迹的 birth 分类（非物理意义上的新进入视场）。"""
    if image_w is None or image_h is None:
        return "unmatched_detection"
    near_border = (
        cx < border_margin
        or cx > image_w - border_margin
        or cy < border_margin
        or cy > image_h - border_margin
    )
    if near_border:
        return "enter_from_border"
    return "appear_inside_fov"


def classify_track_end(
    track: Track,
    image_w: Optional[int],
    image_h: Optional[int],
    border_margin: float,
) -> str:
    """根据最后位置与速度方向判断轨迹终止原因。"""
    if image_w is None or image_h is None:
        return "unmatched_timeout"

    hist = track.history
    if not hist:
        return "unmatched_timeout"

    detected = observed_history(hist)
    points = detected if detected else sorted(hist, key=lambda h: int(h["frame_id"]))
    if not points:
        return "unmatched_timeout"

    points = sorted(points, key=lambda h: int(h["frame_id"]))
    last = points[-1]
    cx = float(last["cx"])
    cy = float(last["cy"])

    vx, vy = 0.0, 0.0
    if len(points) >= 2:
        prev = points[-2]
        vx = cx - float(prev["cx"])
        vy = cy - float(prev["cy"])

    margin = border_margin
    near_left = cx < margin
    near_right = cx > image_w - margin
    near_top = cy < margin
    near_bottom = cy > image_h - margin
    near_border = near_left or near_right or near_top or near_bottom

    moving_out = (
        (near_left and vx < 0)
        or (near_right and vx > 0)
        or (near_top and vy < 0)
        or (near_bottom and vy > 0)
    )

    if near_border and moving_out:
        return "exit_fov"
    if near_border:
        return "near_border_lost"
    return "lost_inside"


def build_cost_matrix(
    tracks: List[Track],
    detections: List[Dict[str, Any]],
    max_center_dist: float,
    max_mahalanobis: float,
    w_maha: float,
    w_center: float,
    w_iou: float,
    w_size: float,
    w_direction: float = 0.05,
    min_direction_speed: float = 1.0,
) -> Tuple[np.ndarray, List[List[Dict[str, float]]]]:
    """Build track-detection cost matrix.

    Lower cost is better. gate_penalty 仅由 center_dist 和 mahalanobis 触发。
    """
    n = len(tracks)
    m = len(detections)
    cost = np.zeros((n, m), dtype=np.float64)
    details: List[List[Dict[str, float]]] = [[{} for _ in range(m)] for _ in range(n)]

    for i, tr in enumerate(tracks):
        z_pred, S = tr.kf.measurement_prediction()
        pred_box = tr.kf.bbox_xyxy()
        pcx, pcy, vx, vy, pw, ph = tr.kf.x

        for j, det in enumerate(detections):
            z = np.array([det["cx"], det["cy"], det["w"], det["h"]], dtype=np.float64)

            maha = mahalanobis_distance(z, z_pred, S)
            center_dist = float(math.hypot(det["cx"] - pcx, det["cy"] - pcy))

            det_box = det_to_xyxy(det)
            iou = box_iou_xyxy(pred_box, det_box)

            size_rel = abs(math.log((det["w"] + 1e-6) / (pw + 1e-6))) + abs(
                math.log((det["h"] + 1e-6) / (ph + 1e-6))
            )
            size_cost = min(1.0, size_rel / 2.0)

            maha_cost = min(1.0, maha / max(max_mahalanobis, 1e-6))
            center_cost = min(1.0, center_dist / max(max_center_dist, 1e-6))
            iou_cost = 1.0 - iou

            speed = float(math.hypot(vx, vy))
            if speed < min_direction_speed:
                direction_cost = 0.0
            else:
                dx = float(det["cx"]) - pcx
                dy = float(det["cy"]) - pcy
                d_norm = math.hypot(dx, dy)
                if d_norm < 1e-6:
                    direction_cost = 0.0
                else:
                    cos_sim = (vx * dx + vy * dy) / (speed * d_norm)
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    direction_cost = 0.5 * (1.0 - cos_sim)

            # Gate very unlikely pairs by adding a large penalty (motion only).
            gate_penalty = 0.0
            if center_dist > max_center_dist:
                gate_penalty += 2.0
            if maha > max_mahalanobis:
                gate_penalty += 2.0

            c = (
                w_maha * maha_cost
                + w_center * center_cost
                + w_iou * iou_cost
                + w_size * size_cost
                + w_direction * direction_cost
                + gate_penalty
            )
            cost[i, j] = c
            details[i][j] = {
                "maha": maha,
                "center_dist": center_dist,
                "iou": iou,
                "size_cost": size_cost,
                "direction_cost": direction_cost,
                "cost": c,
            }

    return cost, details


def append_dustbins(cost: np.ndarray, dummy_cost: float) -> np.ndarray:
    """Append dummy columns/rows to allow unmatched tracks and births.

    If cost is [N, M], return [N+M, M+N]:
      - first N rows are real tracks;
      - first M cols are real detections;
      - last N cols are dummy deaths;
      - last M rows are dummy births.
    """
    n, m = cost.shape
    out = np.full((n + m, m + n), dummy_cost, dtype=np.float64)
    out[:n, :m] = cost

    # Track i can go to its own dummy-death column with dummy_cost.
    # Detection j can be explained by its own dummy-birth row with dummy_cost.
    # Other dummy couplings are also allowed but not used for extraction.
    return out


def extract_matches_from_transport(
    P: np.ndarray,
    cost: np.ndarray,
    n_tracks: int,
    n_dets: int,
    min_ot_prob: float,
    max_match_cost: float,
) -> List[Tuple[int, int, float, float]]:
    """Extract one-to-one matches from real-real block of transport plan."""
    if n_tracks == 0 or n_dets == 0:
        return []

    real_P = P[:n_tracks, :n_dets].copy()

    # Convert probability scale to a more intuitive relative confidence.
    # Since balanced OT row mass is about 1 / (n_tracks + n_dets),
    # normalize each row by its maximum possible row mass.
    row_mass = P[:n_tracks, :].sum(axis=1, keepdims=True) + 1e-12
    rel_P = real_P / row_mass

    candidates: List[Tuple[int, int, float, float]] = []
    for i in range(n_tracks):
        for j in range(n_dets):
            prob = float(rel_P[i, j])
            c = float(cost[i, j])
            if prob >= min_ot_prob and c <= max_match_cost:
                candidates.append((i, j, prob, c))

    # Greedy one-to-one by high probability then low cost.
    candidates.sort(key=lambda x: (-x[2], x[3]))
    used_tracks = set()
    used_dets = set()
    matches: List[Tuple[int, int, float, float]] = []
    for i, j, prob, c in candidates:
        if i in used_tracks or j in used_dets:
            continue
        used_tracks.add(i)
        used_dets.add(j)
        matches.append((i, j, prob, c))
    return matches


def rescue_unmatched_matches(
    tracks: List[Track],
    detections: List[Dict[str, Any]],
    unmatched_track_indices: List[int],
    unmatched_det_indices: List[int],
    max_center_dist: float,
    max_mahalanobis: float,
    max_age_confirmed: int,
    rescue_center_factor: float = 1.3,
    rescue_maha_factor: float = 1.3,
) -> Tuple[List[Tuple[int, int, float, float]], int]:
    """将短时 lost 的 confirmed track 与 unmatched detection 做 gap-aware 重关联。"""
    if not tracks or not detections or not unmatched_track_indices or not unmatched_det_indices:
        return [], 0

    candidates: List[Tuple[int, int, float]] = []
    rescue_candidates_count = 0

    for ti in unmatched_track_indices:
        tr = tracks[ti]
        if not tr.is_confirmed or tr.misses <= 0 or tr.misses > max_age_confirmed:
            continue

        z_pred, S = tr.kf.measurement_prediction()
        pcx = float(z_pred[0])
        pcy = float(z_pred[1])
        miss_factor = min(1.0 + 0.25 * tr.misses, 1.8)

        for di in unmatched_det_indices:
            det = detections[di]
            dcx = float(det["cx"])
            dcy = float(det["cy"])
            center_dist = math.hypot(dcx - pcx, dcy - pcy)

            z = np.array([det["cx"], det["cy"], det["w"], det["h"]], dtype=np.float64)
            maha = mahalanobis_distance(z, z_pred, S)

            effective_center_dist = rescue_center_factor * miss_factor * max_center_dist
            effective_maha = rescue_maha_factor * max_mahalanobis

            if center_dist > effective_center_dist or maha > effective_maha:
                continue

            rescue_candidates_count += 1
            rescue_score = (
                0.6 * center_dist / max(effective_center_dist, 1e-6)
                + 0.4 * maha / max(effective_maha, 1e-6)
            )
            candidates.append((ti, di, rescue_score))

    if not candidates:
        return [], rescue_candidates_count

    candidates.sort(key=lambda x: x[2])
    used_tracks = set()
    used_dets = set()
    rescue_matches: List[Tuple[int, int, float, float]] = []

    for ti, di, rescue_score in candidates:
        if ti in used_tracks or di in used_dets:
            continue
        used_tracks.add(ti)
        used_dets.add(di)
        rescue_matches.append((ti, di, -1.0, rescue_score))

    return rescue_matches, rescue_candidates_count


# -----------------------------
# Tracker
# -----------------------------

class OTKalmanTracker:
    def __init__(
        self,
        max_age: int = 4,
        max_age_tentative: int = 1,
        max_age_confirmed: int = 5,
        min_hits: int = 3,
        min_contiguous_observed: int = 3,
        max_center_dist: float = 45.0,
        max_mahalanobis: float = 12.0,
        ot_reg: float = 0.08,
        ot_iters: int = 100,
        dummy_cost: float = 0.65,
        min_ot_prob: float = 0.15,
        max_match_cost: float = 0.80,
        w_maha: float = 0.45,
        w_center: float = 0.45,
        w_iou: float = 0.15,
        w_size: float = 0.15,
        w_direction: float = 0.05,
        min_direction_speed: float = 1.0,
        kf_process_pos_std: float = 2.0,
        kf_process_vel_std: float = 12.0,
        kf_process_size_std: float = 2.0,
        kf_meas_pos_std: float = 5.0,
        kf_meas_size_std: float = 12.0,
        min_det_score: float = 0.0,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        border_margin: float = 20.0,
        rescue_center_factor: float = 1.3,
        rescue_maha_factor: float = 1.3,
    ):
        self.max_age = max_age
        self.max_age_tentative = max_age_tentative
        self.max_age_confirmed = max_age_confirmed
        self.min_hits = min_hits
        self.min_contiguous_observed = min_contiguous_observed
        self.max_center_dist = max_center_dist
        self.max_mahalanobis = max_mahalanobis
        self.ot_reg = ot_reg
        self.ot_iters = ot_iters
        self.dummy_cost = dummy_cost
        self.min_ot_prob = min_ot_prob
        self.max_match_cost = max_match_cost
        self.w_maha = w_maha
        self.w_center = w_center
        self.w_iou = w_iou
        self.w_size = w_size
        self.w_direction = w_direction
        self.min_direction_speed = min_direction_speed

        self.kf_process_pos_std = kf_process_pos_std
        self.kf_process_vel_std = kf_process_vel_std
        self.kf_process_size_std = kf_process_size_std
        self.kf_meas_pos_std = kf_meas_pos_std
        self.kf_meas_size_std = kf_meas_size_std

        self.min_det_score = min_det_score
        self.image_width = image_width
        self.image_height = image_height
        self.border_margin = border_margin
        self.rescue_center_factor = rescue_center_factor
        self.rescue_maha_factor = rescue_maha_factor

        self.next_track_id = 1
        self.active_tracks: List[Track] = []
        self.finished_tracks: List[Track] = []

    def _maybe_confirm(self, tr: Track) -> None:
        if tr.hits < self.min_hits:
            return
        obs_frames = [int(h["frame_id"]) for h in observed_history(tr.history)]
        if max_contiguous_run(obs_frames) >= self.min_contiguous_observed:
            tr.is_confirmed = True
            tr.track_state = "confirmed"

    def _new_track(self, det: Dict[str, Any], frame_id: int) -> Track:
        kf = KalmanBoxFilter(
            det["cx"],
            det["cy"],
            det["w"],
            det["h"],
            process_pos_std=self.kf_process_pos_std,
            process_vel_std=self.kf_process_vel_std,
            process_size_std=self.kf_process_size_std,
            meas_pos_std=self.kf_meas_pos_std,
            meas_size_std=self.kf_meas_size_std,
        )
        birth_reason = classify_birth_reason(
            det["cx"],
            det["cy"],
            self.image_width,
            self.image_height,
            self.border_margin,
        )
        tr = Track(
            track_id=self.next_track_id,
            kf=kf,
            start_frame=frame_id,
            last_frame=frame_id,
            age=1,
            hits=1,
            misses=0,
            history=[],
            is_confirmed=False,
            track_state="tentative",
            birth_reason=birth_reason,
            birth_frame=frame_id,
            birth_x=float(det["cx"]),
            birth_y=float(det["cy"]),
            end_reason="",
        )
        tr.history.append(
            {
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
                "ot_prob": 1.0,
                "match_cost": 0.0,
                "pred_x1": det["x1"],
                "pred_y1": det["y1"],
                "pred_x2": det["x2"],
                "pred_y2": det["y2"],
                "point_type": "detected",
                "gap_fill_method": "",
            }
        )
        self.next_track_id += 1
        return tr

    def step(self, frame_id: int, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        detections = [d for d in detections if float(d.get("score", 1.0)) >= self.min_det_score]

        # Predict all active tracks to current frame.
        for tr in self.active_tracks:
            tr.predict(frame_id)

        n = len(self.active_tracks)
        m = len(detections)

        matches: List[Tuple[int, int, float, float]] = []

        if n > 0 and m > 0:
            cost, details = build_cost_matrix(
                self.active_tracks,
                detections,
                max_center_dist=self.max_center_dist,
                max_mahalanobis=self.max_mahalanobis,
                w_maha=self.w_maha,
                w_center=self.w_center,
                w_iou=self.w_iou,
                w_size=self.w_size,
                w_direction=self.w_direction,
                min_direction_speed=self.min_direction_speed,
            )
            cost_aug = append_dustbins(cost, dummy_cost=self.dummy_cost)
            P = sinkhorn_transport(cost_aug, reg=self.ot_reg, num_iters=self.ot_iters)
            matches = extract_matches_from_transport(
                P,
                cost,
                n_tracks=n,
                n_dets=m,
                min_ot_prob=self.min_ot_prob,
                max_match_cost=self.max_match_cost,
            )

        matched_tracks = set()
        matched_dets = set()
        for ti, di, prob, c in matches:
            tr = self.active_tracks[ti]
            det = detections[di]
            tr.update(det, frame_id, ot_prob=prob, match_cost=c)
            self._maybe_confirm(tr)
            if tr.track_state == "lost":
                tr.track_state = "confirmed"
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Gap-aware rescue: reconnect confirmed lost tracks before creating new ones.
        unmatched_track_indices = [i for i in range(n) if i not in matched_tracks]
        unmatched_det_indices = [j for j in range(m) if j not in matched_dets]

        rescue_matches, rescue_candidates = rescue_unmatched_matches(
            tracks=self.active_tracks,
            detections=detections,
            unmatched_track_indices=unmatched_track_indices,
            unmatched_det_indices=unmatched_det_indices,
            max_center_dist=self.max_center_dist,
            max_mahalanobis=self.max_mahalanobis,
            max_age_confirmed=self.max_age_confirmed,
            rescue_center_factor=self.rescue_center_factor,
            rescue_maha_factor=self.rescue_maha_factor,
        )

        for ti, di, prob, c in rescue_matches:
            tr = self.active_tracks[ti]
            det = detections[di]
            tr.update(det, frame_id, ot_prob=prob, match_cost=c)
            self._maybe_confirm(tr)
            if tr.is_confirmed:
                tr.track_state = "confirmed"
            matched_tracks.add(ti)
            matched_dets.add(di)

        # Unmatched detections start new observable tracks.
        new_tracks = []
        for j, det in enumerate(detections):
            if j not in matched_dets:
                new_tracks.append(self._new_track(det, frame_id))

        # Update state and move dead tracks to finished.
        still_active = []
        for i, tr in enumerate(self.active_tracks):
            if i in matched_tracks:
                still_active.append(tr)
                continue
            allowed_age = self.max_age_confirmed if tr.is_confirmed else self.max_age_tentative
            if tr.misses > allowed_age:
                tr.track_state = "finished"
                tr.end_reason = classify_track_end(
                    tr, self.image_width, self.image_height, self.border_margin
                )
                self.finished_tracks.append(tr)
            else:
                if tr.is_confirmed:
                    tr.track_state = "lost"
                else:
                    tr.track_state = "tentative"
                still_active.append(tr)

        self.active_tracks = still_active + new_tracks

        tentative = sum(1 for t in self.active_tracks if t.track_state == "tentative")
        confirmed = sum(1 for t in self.active_tracks if t.track_state == "confirmed")
        lost = sum(1 for t in self.active_tracks if t.track_state == "lost")

        return {
            "frame_id": frame_id,
            "detections": m,
            "matches": len(matches),
            "rescue_matches": len(rescue_matches),
            "rescue_candidates": rescue_candidates,
            "new_tracks": len(new_tracks),
            "unmatched_detections": m - len(matched_dets),
            "unmatched_tracks": n - len(matched_tracks),
            "tentative_tracks": tentative,
            "confirmed_tracks": confirmed,
            "lost_tracks": lost,
            "active_tracks": len(self.active_tracks),
            "finished_tracks": len(self.finished_tracks),
        }

    def finish(self) -> List[Track]:
        for tr in self.active_tracks:
            tr.track_state = "finished"
            if not tr.end_reason:
                tr.end_reason = classify_track_end(
                    tr, self.image_width, self.image_height, self.border_margin
                )
                if tr.end_reason == "unmatched_timeout" and tr.misses == 0:
                    tr.end_reason = "end_of_sequence"
        self.finished_tracks.extend(self.active_tracks)
        self.active_tracks = []
        return self.finished_tracks


# -----------------------------
# Gap fill & track filtering
# -----------------------------

def kf_params_from_tracker(tracker: OTKalmanTracker) -> Dict[str, float]:
    return {
        "process_pos_std": tracker.kf_process_pos_std,
        "process_vel_std": tracker.kf_process_vel_std,
        "process_size_std": tracker.kf_process_size_std,
        "meas_pos_std": tracker.kf_meas_pos_std,
        "meas_size_std": tracker.kf_meas_size_std,
    }


def _history_item_from_bbox(
    frame_id: int,
    image_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    point_type: str,
    pred_box: Tuple[float, float, float, float],
) -> Dict[str, Any]:
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return {
        "frame_id": frame_id,
        "image_name": image_name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "w": w,
        "h": h,
        "det_score": 0.0,
        "ot_prob": 0.0,
        "match_cost": "",
        "pred_x1": pred_box[0],
        "pred_y1": pred_box[1],
        "pred_x2": pred_box[2],
        "pred_y2": pred_box[3],
        "point_type": point_type,
        "gap_fill_method": "kf_bidirectional" if point_type == "candidate" else "",
    }


def interpolate_gap_bidirectional(
    left: Dict[str, Any],
    right: Dict[str, Any],
    missing_frames: List[int],
    kf_params: Dict[str, float],
) -> List[Dict[str, Any]]:
    """对缺失帧做前向+后向卡尔曼外推，并按时间权重融合。"""
    if not missing_frames:
        return []

    fwd_boxes: Dict[int, Tuple[float, float, float, float]] = {}
    kf_fwd = KalmanBoxFilter(
        left["cx"], left["cy"], left["w"], left["h"], **kf_params
    )
    kf_fwd.update(left["cx"], left["cy"], left["w"], left["h"])
    prev_f = int(left["frame_id"])
    for f in missing_frames:
        dt = max(1, f - prev_f)
        kf_fwd.predict(dt=dt)
        fwd_boxes[f] = kf_fwd.bbox_xyxy()
        prev_f = f

    bwd_boxes: Dict[int, Tuple[float, float, float, float]] = {}
    kf_bwd = KalmanBoxFilter(
        right["cx"], right["cy"], right["w"], right["h"], **kf_params
    )
    kf_bwd.update(right["cx"], right["cy"], right["w"], right["h"])
    prev_f = int(right["frame_id"])
    for f in reversed(missing_frames):
        dt = max(1, prev_f - f)
        kf_bwd.predict_backward(dt=dt)
        bwd_boxes[f] = kf_bwd.bbox_xyxy()
        prev_f = f

    f_left = int(left["frame_id"])
    f_right = int(right["frame_id"])
    span = max(1, f_right - f_left)
    image_name = right.get("image_name") or left.get("image_name", "")

    filled: List[Dict[str, Any]] = []
    for f in missing_frames:
        t = (f - f_left) / span
        x1f, y1f, x2f, y2f = fwd_boxes[f]
        x1b, y1b, x2b, y2b = bwd_boxes[f]
        cx = (1.0 - t) * (x1f + x2f) / 2.0 + t * (x1b + x2b) / 2.0
        cy = (1.0 - t) * (y1f + y2f) / 2.0 + t * (y1b + y2b) / 2.0
        wf = max(1e-6, x2f - x1f)
        hf = max(1e-6, y2f - y1f)
        wb = max(1e-6, x2b - x1b)
        hb = max(1e-6, y2b - y1b)
        w = (1.0 - t) * wf + t * wb
        h = (1.0 - t) * hf + t * hb
        x1, y1, x2, y2 = xywh_to_xyxy(cx, cy, w, h)
        pred_box = fwd_boxes[f]
        filled.append(
            _history_item_from_bbox(
                frame_id=f,
                image_name=image_name,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                point_type="candidate",
                pred_box=pred_box,
            )
        )
    return filled


def fill_track_gaps(
    track: Track,
    kf_params: Dict[str, float],
    max_gap_fill: int,
) -> int:
    """在相邻检测之间插值缺失帧，返回新增 candidate 点数。"""
    detected = sorted(observed_history(track.history), key=lambda h: int(h["frame_id"]))
    if len(detected) < 2:
        return 0

    candidates: List[Dict[str, Any]] = []
    for left, right in zip(detected[:-1], detected[1:]):
        f_left = int(left["frame_id"])
        f_right = int(right["frame_id"])
        gap = f_right - f_left - 1
        if gap <= 0 or gap > max_gap_fill:
            continue
        missing = list(range(f_left + 1, f_right))
        candidates.extend(
            interpolate_gap_bidirectional(left, right, missing, kf_params)
        )

    if not candidates:
        return 0

    merged = {int(h["frame_id"]): h for h in track.history}
    for item in candidates:
        merged[int(item["frame_id"])] = item
    track.history = [merged[f] for f in sorted(merged.keys())]
    track.start_frame = int(track.history[0]["frame_id"])
    track.last_frame = int(track.history[-1]["frame_id"])
    return len(candidates)


def postprocess_tracks(
    tracks: List[Track],
    kf_params: Dict[str, float],
    enable_gap_fill: bool = True,
    max_gap_fill: int = 10,
    min_contiguous_observed: int = 3,
    min_observed_hits: int = 3,
) -> Tuple[List[Track], Dict[str, int]]:
    """缺口卡尔曼插值 + 剔除过短轨迹。仅对已 confirmed 轨迹插值。"""
    stats = {
        "input_tracks": len(tracks),
        "kept_tracks": 0,
        "removed_short": 0,
        "gap_filled_points": 0,
    }
    kept: List[Track] = []

    for tr in tracks:
        for h in tr.history:
            h.setdefault("point_type", "detected")
            h.setdefault("gap_fill_method", "")

        obs = observed_history(tr.history)
        obs_frames = [int(h["frame_id"]) for h in obs]

        if len(obs) < min_observed_hits:
            stats["removed_short"] += 1
            continue
        if max_contiguous_run(obs_frames) < min_contiguous_observed:
            stats["removed_short"] += 1
            continue

        if enable_gap_fill and tr.is_confirmed:
            stats["gap_filled_points"] += fill_track_gaps(tr, kf_params, max_gap_fill)

        kept.append(tr)

    stats["kept_tracks"] = len(kept)
    return kept, stats


# -----------------------------
# Pre-birth backward dark-particle recovery
# -----------------------------

def _clip_roi_to_image(
    x1: float, y1: float, x2: float, y2: float, img_h: int, img_w: int
) -> Optional[Tuple[int, int, int, int]]:
    left = max(0, int(min(x1, x2)))
    top = max(0, int(min(y1, y2)))
    right = min(img_w, int(max(x1, x2)))
    bottom = min(img_h, int(max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def is_first_detection_bright(
    first_item: Dict[str, Any],
    gray: np.ndarray,
    p95_threshold: float,
) -> bool:
    """判断轨迹第一帧检测 ROI 是否已明显高亮。"""
    if gray.ndim != 2:
        return False
    img_h, img_w = gray.shape
    roi = _clip_roi_to_image(
        float(first_item["x1"]),
        float(first_item["y1"]),
        float(first_item["x2"]),
        float(first_item["y2"]),
        img_h,
        img_w,
    )
    if roi is None:
        return False
    left, top, right, bottom = roi
    patch = gray[top:bottom, left:right]
    if patch.size == 0:
        return False
    p95 = float(np.percentile(patch, 95))
    return p95 >= p95_threshold


def estimate_initial_velocity(
    track: Track,
    max_points: int = 5,
) -> Tuple[float, float, bool]:
    """根据轨迹早期 detected 点估计初始速度 (vx, vy)。"""
    detected = sorted(observed_history(track.history), key=lambda h: int(h["frame_id"]))
    if len(detected) < 2:
        return 0.0, 0.0, False

    early = detected[:max_points]
    vxs: List[float] = []
    vys: List[float] = []
    for a, b in zip(early[:-1], early[1:]):
        dt = max(1, int(b["frame_id"]) - int(a["frame_id"]))
        vxs.append((float(b["cx"]) - float(a["cx"])) / dt)
        vys.append((float(b["cy"]) - float(a["cy"])) / dt)

    if not vxs:
        return 0.0, 0.0, False
    return float(np.median(vxs)), float(np.median(vys)), True


def search_dark_blob_near_prediction(
    gray: np.ndarray,
    pred_cx: float,
    pred_cy: float,
    search_radius: float,
    cold_box_w: float,
    cold_box_h: float,
    dark_threshold_mad: float,
    min_blob_area: int,
    max_blob_area: int,
) -> Optional[Dict[str, Any]]:
    """在预测位置附近搜索暗小 blob，返回最佳候选。"""
    if gray.ndim != 2 or search_radius <= 0:
        return None

    img_h, img_w = gray.shape
    cx_i = int(round(pred_cx))
    cy_i = int(round(pred_cy))
    r = int(math.ceil(search_radius))

    x0 = max(0, cx_i - r)
    y0 = max(0, cy_i - r)
    x1 = min(img_w, cx_i + r + 1)
    y1 = min(img_h, cy_i + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    window = gray[y0:y1, x0:x1].astype(np.float32)
    local_median = float(np.median(window))
    local_mad = float(np.median(np.abs(window - local_median))) + 1e-6

    dark_mask = (window < local_median - dark_threshold_mad * local_mad).astype(np.uint8)
    if not np.any(dark_mask):
        return None

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    if n_labels <= 1:
        return None

    win_cx = pred_cx - x0
    win_cy = pred_cy - y0
    area_mid = 0.5 * (min_blob_area + max_blob_area)

    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_blob_area or area > max_blob_area:
            continue

        blob_mask = labels == label_id
        blob_vals = window[blob_mask]
        blob_mean = float(np.mean(blob_vals))
        dark_contrast = local_median - blob_mean
        contrast_score = min(2.0, dark_contrast / (dark_threshold_mad * local_mad + 1e-6)) / 2.0

        bcx = float(centroids[label_id, 0])
        bcy = float(centroids[label_id, 1])
        distance = math.hypot(bcx - win_cx, bcy - win_cy)
        distance_score = max(0.0, 1.0 - distance / max(search_radius, 1e-6))
        area_score = max(0.0, 1.0 - abs(area - area_mid) / max(area_mid, 1.0))

        recovery_score = 0.55 * contrast_score + 0.35 * distance_score + 0.10 * area_score
        if recovery_score > best_score:
            best_score = recovery_score
            gcx = bcx + x0
            gcy = bcy + y0
            bx1, by1, bx2, by2 = xywh_to_xyxy(gcx, gcy, cold_box_w, cold_box_h)
            bx1 = max(0.0, min(bx1, img_w - 1.0))
            by1 = max(0.0, min(by1, img_h - 1.0))
            bx2 = max(bx1 + 1.0, min(bx2, float(img_w)))
            by2 = max(by1 + 1.0, min(by2, float(img_h)))
            best = {
                "cx": gcx,
                "cy": gcy,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "w": max(1e-6, bx2 - bx1),
                "h": max(1e-6, by2 - by1),
                "recovery_score": recovery_score,
                "dark_contrast": dark_contrast,
                "blob_area": area,
                "search_radius": search_radius,
            }

    return best


def _make_prebirth_history_item(
    frame_id: int,
    image_name: str,
    blob: Dict[str, Any],
    pred_cx: float,
    pred_cy: float,
    cold_w: float,
    cold_h: float,
) -> Dict[str, Any]:
    pred_x1, pred_y1, pred_x2, pred_y2 = xywh_to_xyxy(pred_cx, pred_cy, cold_w, cold_h)
    recovery_score = float(blob["recovery_score"])
    return {
        "frame_id": frame_id,
        "image_name": image_name,
        "x1": blob["x1"],
        "y1": blob["y1"],
        "x2": blob["x2"],
        "y2": blob["y2"],
        "cx": blob["cx"],
        "cy": blob["cy"],
        "w": blob["w"],
        "h": blob["h"],
        "det_score": 0.0,
        "ot_prob": -2.0,
        "match_cost": recovery_score,
        "pred_x1": pred_x1,
        "pred_y1": pred_y1,
        "pred_x2": pred_x2,
        "pred_y2": pred_y2,
        "point_type": "prebirth_candidate",
        "gap_fill_method": "backward_dark_search",
        "recovery_score": recovery_score,
        "dark_contrast": blob["dark_contrast"],
        "blob_area": blob["blob_area"],
        "search_radius": blob["search_radius"],
    }


def recover_prebirth_dark_particles(
    tracks: List[Track],
    frame_image_map: Dict[int, Path],
    max_backward_frames: int = 8,
    max_failures: int = 2,
    search_radius_base: float = 12.0,
    radius_growth: float = 2.0,
    dark_threshold_mad: float = 2.5,
    min_blob_area: int = 3,
    max_blob_area: int = 80,
    score_threshold: float = 0.45,
    bright_p95_threshold: float = 150.0,
    box_scale: float = 0.45,
    min_box_size: float = 4.0,
) -> Tuple[List[Track], Dict[str, int]]:
    """轨迹起点反向暗颗粒补全（后处理）。"""
    stats = {
        "tracks_checked": 0,
        "tracks_recovered": 0,
        "prebirth_points_added": 0,
        "skipped_no_images": 0,
        "skipped_not_bright": 0,
        "skipped_no_velocity": 0,
    }

    if not frame_image_map:
        stats["skipped_no_images"] = len(tracks)
        return tracks, stats

    gray_cache: Dict[int, Optional[np.ndarray]] = {}

    def get_gray(frame_id: int) -> Optional[np.ndarray]:
        if frame_id not in gray_cache:
            path = frame_image_map.get(frame_id)
            gray_cache[frame_id] = read_gray_image(path) if path else None
        return gray_cache[frame_id]

    for tr in tracks:
        if not tr.is_confirmed:
            continue

        detected = sorted(observed_history(tr.history), key=lambda h: int(h["frame_id"]))
        if len(detected) < 2:
            continue

        stats["tracks_checked"] += 1
        first = detected[0]
        t0 = int(first["frame_id"])

        if t0 <= 0 or t0 not in frame_image_map:
            stats["skipped_no_images"] += 1
            continue

        if tr.birth_reason == "enter_from_border":
            continue

        gray_t0 = get_gray(t0)
        if gray_t0 is None:
            stats["skipped_no_images"] += 1
            continue

        if not is_first_detection_bright(first, gray_t0, bright_p95_threshold):
            stats["skipped_not_bright"] += 1
            continue

        vx, vy, ok_vel = estimate_initial_velocity(tr)
        if not ok_vel:
            stats["skipped_no_velocity"] += 1
            continue

        first_w = float(first["w"])
        first_h = float(first["h"])
        cold_w = max(min_box_size, first_w * box_scale)
        cold_h = max(min_box_size, first_h * box_scale)
        first_cx = float(first["cx"])
        first_cy = float(first["cy"])

        existing_frames = {int(h["frame_id"]) for h in tr.history}
        prebirth_items: List[Dict[str, Any]] = []
        failures = 0
        track_recovered = False

        for k in range(1, max_backward_frames + 1):
            t = t0 - k
            if t < 0 or t not in frame_image_map:
                break

            pred_cx = first_cx - vx * k
            pred_cy = first_cy - vy * k

            gray_t = get_gray(t)
            if gray_t is None:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            img_h, img_w = gray_t.shape
            if pred_cx < 0 or pred_cy < 0 or pred_cx >= img_w or pred_cy >= img_h:
                break

            if t in existing_frames:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            search_radius = search_radius_base + radius_growth * k
            blob = search_dark_blob_near_prediction(
                gray_t,
                pred_cx,
                pred_cy,
                search_radius,
                cold_w,
                cold_h,
                dark_threshold_mad,
                min_blob_area,
                max_blob_area,
            )

            if blob is None or float(blob["recovery_score"]) < score_threshold:
                failures += 1
                if failures >= max_failures:
                    break
                continue

            failures = 0
            image_path = frame_image_map[t]
            prebirth_items.append(
                _make_prebirth_history_item(
                    frame_id=t,
                    image_name=image_path.stem,
                    blob=blob,
                    pred_cx=pred_cx,
                    pred_cy=pred_cy,
                    cold_w=cold_w,
                    cold_h=cold_h,
                )
            )
            track_recovered = True

        if not prebirth_items:
            continue

        merged = {int(h["frame_id"]): h for h in tr.history}
        for item in prebirth_items:
            fid = int(item["frame_id"])
            if fid not in merged:
                merged[fid] = item

        tr.history = [merged[f] for f in sorted(merged.keys())]
        tr.start_frame = int(tr.history[0]["frame_id"])
        tr.prebirth_recovered = True
        stats["tracks_recovered"] += 1
        stats["prebirth_points_added"] += len(prebirth_items)

    return tracks, stats


# -----------------------------
# Output
# -----------------------------

def save_tracks_csv(tracks: List[Track], out_path: Path, min_track_len: int = 2) -> None:
    fields = [
        "track_id",
        "frame_id",
        "image_name",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "w",
        "h",
        "det_score",
        "ot_prob",
        "match_cost",
        "pred_x1",
        "pred_y1",
        "pred_x2",
        "pred_y2",
        "track_len",
        "start_frame",
        "end_frame",
        "is_confirmed",
        "track_state",
        "birth_reason",
        "end_reason",
        "point_type",
        "gap_fill_method",
        "recovery_score",
        "dark_contrast",
        "blob_area",
        "search_radius",
    ]
    rows = []
    for tr in tracks:
        if len(tr.history) < min_track_len:
            continue
        track_len = len(tr.history)
        for item in tr.history:
            row = {k: item.get(k, "") for k in fields if k not in (
                "track_id", "track_len", "start_frame", "end_frame", "is_confirmed",
                "track_state", "birth_reason", "end_reason",
            )}
            row["track_id"] = tr.track_id
            row["track_len"] = track_len
            row["start_frame"] = tr.start_frame
            row["end_frame"] = tr.last_frame
            row["is_confirmed"] = int(tr.is_confirmed)
            row["track_state"] = tr.track_state
            row["birth_reason"] = tr.birth_reason
            row["end_reason"] = tr.end_reason
            row.setdefault("point_type", item.get("point_type", "detected"))
            row.setdefault("gap_fill_method", item.get("gap_fill_method", ""))
            rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_track_summary_csv(tracks: List[Track], out_path: Path, min_track_len: int = 2) -> None:
    fields = [
        "track_id",
        "start_frame",
        "end_frame",
        "track_len",
        "duration",
        "is_confirmed",
        "track_state",
        "birth_reason",
        "end_reason",
        "mean_det_score",
        "mean_ot_prob",
        "mean_match_cost",
        "mean_speed_px_per_frame",
        "n_detected",
        "n_candidate",
        "n_prebirth_candidate",
        "max_contiguous_observed",
    ]
    rows = []
    for tr in tracks:
        if len(tr.history) < min_track_len:
            continue

        obs = observed_history(tr.history)
        obs_frames = [int(h["frame_id"]) for h in obs]
        n_candidate = sum(
            1 for h in tr.history if h.get("point_type") == "candidate"
        )
        n_prebirth = sum(
            1 for h in tr.history if h.get("point_type") == "prebirth_candidate"
        )
        duration = int(tr.last_frame) - int(tr.start_frame) + 1

        speeds = []
        for a, b in zip(tr.history[:-1], tr.history[1:]):
            dt = max(1, int(b["frame_id"]) - int(a["frame_id"]))
            speeds.append(math.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) / dt)

        rows.append(
            {
                "track_id": tr.track_id,
                "start_frame": tr.start_frame,
                "end_frame": tr.last_frame,
                "track_len": len(tr.history),
                "duration": duration,
                "is_confirmed": int(tr.is_confirmed),
                "track_state": tr.track_state,
                "birth_reason": tr.birth_reason,
                "end_reason": tr.end_reason,
                "mean_det_score": np.mean([h.get("det_score", 1.0) for h in obs]) if obs else 0.0,
                "mean_ot_prob": np.mean([h.get("ot_prob", 0.0) for h in obs]) if obs else 0.0,
                "mean_match_cost": np.mean([h.get("match_cost", 0.0) for h in obs if h.get("match_cost") != ""]) if obs else 0.0,
                "mean_speed_px_per_frame": np.mean(speeds) if speeds else 0.0,
                "n_detected": len(obs),
                "n_candidate": n_candidate,
                "n_prebirth_candidate": n_prebirth,
                "max_contiguous_observed": max_contiguous_run(obs_frames),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def track_color(track_id: int) -> Tuple[int, int, int]:
    return (
        int((37 * track_id + 80) % 206) + 30,
        int((91 * track_id + 40) % 206) + 30,
        int((157 * track_id + 20) % 206) + 30,
    )


def tracks_from_track_objects(
    tracks: List[Track], min_track_len: int
) -> Dict[int, List[Dict[str, Any]]]:
    tracks_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for tr in tracks:
        if len(tr.history) < min_track_len:
            continue
        items = []
        for item in tr.history:
            items.append(
                {
                    "track_id": tr.track_id,
                    "frame_id": int(item["frame_id"]),
                    "image_name": item.get("image_name", ""),
                    "x1": float(item["x1"]),
                    "y1": float(item["y1"]),
                    "x2": float(item["x2"]),
                    "y2": float(item["y2"]),
                    "cx": float(item["cx"]),
                    "cy": float(item["cy"]),
                    "is_confirmed": int(tr.is_confirmed),
                    "track_state": tr.track_state,
                    "point_type": item.get("point_type", "detected"),
                }
            )
        tracks_by_id[tr.track_id] = sorted(items, key=lambda x: x["frame_id"])
    return tracks_by_id


def load_tracks_csv(
    csv_path: Path, min_track_len: int = 1
) -> Dict[int, List[Dict[str, Any]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"tracks.csv not found: {csv_path}")

    tracks_by_id: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tid = int(float(row["track_id"]))
                frame_id = int(float(row["frame_id"]))
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                cx = float(row.get("cx") or (x1 + x2) / 2.0)
                cy = float(row.get("cy") or (y1 + y2) / 2.0)
            except (KeyError, ValueError, TypeError):
                continue
            tracks_by_id.setdefault(tid, []).append(
                {
                    "track_id": tid,
                    "frame_id": frame_id,
                    "image_name": row.get("image_name", ""),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "is_confirmed": int(float(row.get("is_confirmed", 1) or 1)),
                    "track_state": row.get("track_state", "confirmed"),
                    "point_type": row.get("point_type", "detected"),
                }
            )

    filtered: Dict[int, List[Dict[str, Any]]] = {}
    for tid, items in tracks_by_id.items():
        items = sorted(items, key=lambda x: x["frame_id"])
        if len(items) >= min_track_len:
            filtered[tid] = items
    return filtered


def _trail_points(
    history: List[Dict[str, Any]], current_frame: int, trail_length: int
) -> List[Tuple[float, float]]:
    pts = [
        (float(p["cx"]), float(p["cy"]))
        for p in history
        if int(p["frame_id"]) <= current_frame
    ]
    if trail_length > 0 and len(pts) > trail_length:
        pts = pts[-trail_length:]
    return pts


def _draw_polyline(draw, points: List[Tuple[float, float]], color: Tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width)


def _draw_track_marker(
    draw,
    x: float,
    y: float,
    color: Tuple[int, int, int],
    radius: int = 4,
    outline: Optional[Tuple[int, int, int]] = None,
) -> None:
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline=outline or color,
        width=1,
    )


def _save_pil_image(img: "Image.Image", path: Path, fmt: str = "PNG", **kwargs) -> None:
    """可靠保存 PIL 图像（避免 Windows 下 Path/fileno 相关问题）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"format": fmt, **kwargs}
    with open(path, "wb") as f:
        img.save(f, **save_kwargs)


def visualize_tracks(
    tracks_by_id: Dict[int, List[Dict[str, Any]]],
    image_dir: Path,
    output_dir: Path,
    image_exts: str = ".png,.jpg,.jpeg,.tif,.tiff",
    trail_length: int = 0,
    min_track_len: int = 1,
    only_confirmed: bool = False,
    overlay_only: bool = False,
    frame_step: int = 1,
    max_label_tracks_per_frame: int = 80,
) -> Path:
    """根据 tracks.csv 数据生成轨迹可视化。

    输出:
        track_vis/{frame}.png      每帧叠加轨迹线、当前框、track id
        track_vis/tracks_overlay.png  全序列轨迹总览（首帧作底图）
    """
    from PIL import Image, ImageDraw, ImageFont

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = collect_image_paths(image_dir, image_exts)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if only_confirmed:
        tracks_by_id = {
            tid: items
            for tid, items in tracks_by_id.items()
            if items and int(items[0].get("is_confirmed", 1)) == 1
        }

    tracks_by_id = {
        tid: items
        for tid, items in tracks_by_id.items()
        if len(items) >= min_track_len
    }
    if not tracks_by_id:
        raise ValueError("No tracks to visualize after filtering.")

    frame_to_items: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for tid, history in tracks_by_id.items():
        for item in history:
            frame_to_items.setdefault(int(item["frame_id"]), []).append((tid, item))

    vis_dir = output_dir / "track_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    # 全轨迹总览
    base_img = Image.open(image_paths[0]).convert("RGB")
    overlay = base_img.copy()
    overlay_draw = ImageDraw.Draw(overlay)
    for tid, history in sorted(tracks_by_id.items()):
        color = track_color(tid)
        pts = [(p["cx"], p["cy"]) for p in history]
        _draw_polyline(overlay_draw, pts, color, width=2)
        if pts:
            _draw_track_marker(overlay_draw, pts[0][0], pts[0][1], color, radius=5)
            _draw_track_marker(
                overlay_draw,
                pts[-1][0],
                pts[-1][1],
                color,
                radius=5,
                outline=(255, 255, 255),
            )
            overlay_draw.text(
                (pts[-1][0] + 6, pts[-1][1] + 6),
                f"ID{tid}{'T' if history[0].get('track_state') == 'tentative' else ''}",
                fill=color,
                font=font_small,
            )
    _save_pil_image(overlay, vis_dir / "tracks_overlay.png", compress_level=3)
    overlay.close()
    base_img.close()

    if overlay_only:
        print(
            f"Saved tracks_overlay.png ({len(tracks_by_id)} tracks) -> {vis_dir}"
        )
        return vis_dir

    frame_step = max(1, int(frame_step))
    n_frames = len(image_paths)
    saved_frames = 0

    # 逐帧可视化
    for frame_id, image_path in enumerate(image_paths):
        if frame_id % frame_step != 0:
            continue

        with Image.open(image_path) as src:
            img = src.convert("RGB")
        draw = ImageDraw.Draw(img)

        for tid, history in tracks_by_id.items():
            if int(history[0]["frame_id"]) > frame_id:
                continue
            color = track_color(tid)
            trail = _trail_points(history, frame_id, trail_length)
            if len(trail) >= 2:
                _draw_polyline(draw, trail, color, width=2)

        items = frame_to_items.get(frame_id, [])
        draw_labels = len(items) <= max_label_tracks_per_frame
        for tid, item in items:
            color = track_color(tid)
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            cx, cy = item["cx"], item["cy"]
            pt = item.get("point_type", "detected")
            is_tentative = item.get("track_state") == "tentative"
            is_prebirth = pt == "prebirth_candidate"
            if is_prebirth:
                outline = (120, 180, 255)
                width = 1
            elif is_tentative:
                outline = (200, 200, 200)
                width = 2
            else:
                outline = color
                width = 2
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
            _draw_track_marker(draw, cx, cy, color, radius=3)
            if draw_labels:
                label = f"ID {tid}"
                if is_tentative:
                    label += " T"
                if is_prebirth:
                    label += " PB"
                draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color, font=font)

        draw.text((8, 8), f"frame {frame_id}", fill=(255, 255, 0), font=font)
        out_path = vis_dir / f"{image_path.stem}.png"
        _save_pil_image(img, out_path, compress_level=3)
        img.close()
        saved_frames += 1

        if saved_frames % 50 == 0 or frame_id == n_frames - 1:
            print(f"  vis progress: {saved_frames}/{(n_frames + frame_step - 1) // frame_step} frames")

    print(
        f"Saved {saved_frames} frame visualizations and tracks_overlay.png "
        f"({len(tracks_by_id)} tracks) -> {vis_dir}"
    )
    return vis_dir


def maybe_visualize_tracks(
    tracks: List[Track],
    image_dir: Optional[Path],
    output_dir: Path,
    image_exts: str,
    min_track_len: int,
    trail_length: int = 0,
    only_confirmed: bool = False,
    overlay_only: bool = False,
    frame_step: int = 1,
) -> None:
    if image_dir is None or not image_dir.is_dir():
        print("Skip visualization: image-dir not set or missing.")
        return
    tracks_by_id = tracks_from_track_objects(tracks, min_track_len=min_track_len)
    visualize_tracks(
        tracks_by_id,
        image_dir=image_dir,
        output_dir=output_dir,
        image_exts=image_exts,
        trail_length=trail_length,
        min_track_len=min_track_len,
        only_confirmed=only_confirmed,
        overlay_only=overlay_only,
        frame_step=frame_step,
    )


def run_visualize_only(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    tracks_csv = Path(args.tracks_csv) if args.tracks_csv else output_dir / "tracks.csv"
    tracks_by_id = load_tracks_csv(tracks_csv, min_track_len=args.min_track_len)
    visualize_tracks(
        tracks_by_id,
        image_dir=Path(args.image_dir),
        output_dir=output_dir,
        image_exts=args.image_exts,
        trail_length=args.trail_length,
        min_track_len=args.min_track_len,
        only_confirmed=args.only_confirmed,
        overlay_only=args.vis_overlay_only,
        frame_step=args.vis_frame_step,
    )


# -----------------------------
# CLI
# -----------------------------

def resolve_detections_input(
    det_path: Path, input_format: str
) -> Tuple[Path, str]:
    """根据路径自动解析检测输入（CSV 或 txt 目录）。"""
    if not det_path.exists():
        raise FileNotFoundError(f"Detections path not found: {det_path}")

    if det_path.is_file():
        if det_path.suffix.lower() != ".csv":
            raise ValueError(f"CSV 模式需要 .csv 文件: {det_path}")
        return det_path, "csv"

    # 目录
    if input_format == "txt-dir":
        return det_path, "txt-dir"

    cleaned_csv = det_path / "cleaned_all.csv"
    if cleaned_csv.is_file():
        print(f"Auto-resolved detections CSV: {cleaned_csv}")
        return cleaned_csv, "csv"

    if det_path.name == "cleaned_detections_txt" or list(det_path.glob("*.txt")):
        print(f"Using txt-dir mode for: {det_path}")
        return det_path, "txt-dir"

    txt_subdir = det_path / "cleaned_detections_txt"
    if txt_subdir.is_dir() and list(txt_subdir.glob("*.txt")):
        print(f"Auto-resolved detections txt dir: {txt_subdir}")
        return txt_subdir, "txt-dir"

    raise ValueError(
        f"无法在目录中找到 cleaned_all.csv 或 txt 检测文件: {det_path}\n"
        f"请指定:\n"
        f"  --detections .../postprocessed/cleaned_all.csv --input-format csv\n"
        f"  或 --detections .../postprocessed/cleaned_detections_txt --input-format txt-dir"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Class-agnostic OT + Kalman tracker for particle detections."
    )

    p.add_argument(
        "--detections",
        default='inputs/cleaned_detections_txt',
        help="cleaned_all.csv，或 postprocessed/ 目录（自动查找），或 per-frame txt 文件夹",
    )
    p.add_argument(
        "--output-dir",
        default=r"locate_results/run_20260703_104159/postprocessed/tracking_output",
        help="输出根目录；默认在其下自动创建 run_YYYYMMDD_HHMMSS 子目录",
    )
    p.add_argument(
        "--no-run-subdir",
        action="store_true",
        help="直接写入 --output-dir，不追加时间戳子目录",
    )
    p.add_argument("--image-dir", default='inputs/MCFF0004-N001', help="Optional image folder for visualization")
    p.add_argument("--image-exts", default=".png,.jpg,.jpeg,.tif,.tiff")
    p.add_argument("--input-format", choices=["csv", "txt-dir"], default="txt-dir")

    p.add_argument("--max-age", type=int, default=4, help="Frames to keep unmatched tracks alive (legacy, kept for compatibility)")
    p.add_argument(
        "--max-age-tentative",
        type=int,
        default=1,
        help="Frames to keep unmatched tentative tracks alive",
    )
    p.add_argument(
        "--max-age-confirmed",
        type=int,
        default=3,
        help="Frames to keep unmatched confirmed tracks alive",
    )
    p.add_argument(
        "--rescue-center-factor",
        type=float,
        default=1.,
        help="Center-distance relaxation factor for rescue matching",
    )
    p.add_argument(
        "--rescue-maha-factor",
        type=float,
        default=1.,
        help="Mahalanobis relaxation factor for rescue matching",
    )
    p.add_argument("--min-hits", type=int, default=3, help="Hits needed to mark track confirmed")
    p.add_argument("--min-track-len", type=int, default=2, help="保存时轨迹最少点数（含插值）")
    p.add_argument(
        "--min-contiguous-observed",
        type=int,
        default=3,
        help="剔除最长连续检测帧数小于此值的轨迹",
    )
    p.add_argument(
        "--min-observed-hits",
        type=int,
        default=3,
        help="剔除实际检测点数少于此值的轨迹",
    )
    p.add_argument(
        "--enable-gap-fill",
        dest="enable_gap_fill",
        action="store_true",
        default=True,
        help="对前后有检测但中间缺帧的区段做卡尔曼插值",
    )
    p.add_argument(
        "--disable-gap-fill",
        dest="enable_gap_fill",
        action="store_false",
    )
    p.add_argument(
        "--max-gap-fill",
        type=int,
        default=10,
        help="单次缺口最大插值帧数",
    )
    p.add_argument("--min-det-score", type=float, default=0.0)

    p.add_argument("--max-center-dist", type=float, default=35.0)
    p.add_argument("--max-mahalanobis", type=float, default=10.0)
    p.add_argument("--ot-reg", type=float, default=0.08)
    p.add_argument("--ot-iters", type=int, default=100)
    p.add_argument("--dummy-cost", type=float, default=0.65)
    p.add_argument("--min-ot-prob", type=float, default=0.15)
    p.add_argument("--max-match-cost", type=float, default=0.80)

    p.add_argument("--w-maha", type=float, default=0.45)
    p.add_argument("--w-center", type=float, default=0.45)
    p.add_argument("--w-iou", type=float, default=0.05)
    p.add_argument("--w-size", type=float, default=0.05)
    p.add_argument("--w-direction", type=float, default=0.05)
    p.add_argument("--min-direction-speed", type=float, default=1.0)

    p.add_argument("--kf-process-pos-std", type=float, default=2.0)
    p.add_argument("--kf-process-vel-std", type=float, default=12.0)
    p.add_argument("--kf-process-size-std", type=float, default=2.0)
    p.add_argument("--kf-meas-pos-std", type=float, default=5.0)
    p.add_argument("--kf-meas-size-std", type=float, default=12.0)

    p.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="图像宽度（用于 birth/end 边界判断）",
    )
    p.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="图像高度（用于 birth/end 边界判断）",
    )
    p.add_argument(
        "--border-margin",
        type=float,
        default=20.0,
        help="距图像边界小于此距离视为 near_border",
    )

    p.add_argument(
        "--save-vis",
        dest="save_vis",
        action="store_true",
        default=False,
        help="跟踪结束后生成轨迹可视化（默认开启）",
    )
    p.add_argument(
        "--no-save-vis",
        dest="save_vis",
        action="store_false",
        help="不生成轨迹可视化",
    )
    p.add_argument(
        "--vis-only",
        action="store_true",
        help="仅从已有 tracks.csv 生成轨迹可视化，不重新跟踪",
    )
    p.add_argument(
        "--tracks-csv",
        default=None,
        help="可视化用的 tracks.csv（默认 output-dir/tracks.csv）",
    )
    p.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="每帧显示的轨迹尾迹长度，0 表示从轨迹起点到当前帧",
    )
    p.add_argument(
        "--vis-overlay-only",
        action="store_true",
        help="仅保存 tracks_overlay.png，不逐帧导出（轨迹多时推荐）",
    )
    p.add_argument(
        "--vis-frame-step",
        type=int,
        default=1,
        help="逐帧可视化时每隔 N 帧保存一张，默认 1",
    )
    p.add_argument(
        "--only-confirmed",
        action="store_true",
        help="可视化时只显示 is_confirmed=1 的轨迹",
    )

    p.add_argument(
        "--enable-prebirth-recovery",
        action="store_true",
        default=False,
        help="Enable backward dark-particle recovery before the first detected frame",
    )
    p.add_argument("--prebirth-max-frames", type=int, default=8)
    p.add_argument("--prebirth-max-failures", type=int, default=2)
    p.add_argument("--prebirth-search-radius", type=float, default=12.0)
    p.add_argument("--prebirth-radius-growth", type=float, default=2.0)
    p.add_argument("--prebirth-dark-threshold", type=float, default=2.5)
    p.add_argument("--prebirth-min-blob-area", type=int, default=3)
    p.add_argument("--prebirth-max-blob-area", type=int, default=80)
    p.add_argument("--prebirth-score-threshold", type=float, default=0.45)
    p.add_argument("--prebirth-bright-p95-threshold", type=float, default=150.0)
    p.add_argument("--prebirth-box-scale", type=float, default=0.45)
    p.add_argument("--prebirth-min-box-size", type=float, default=4.0)
    p.add_argument(
        "--prebirth-use-grayscale-invert",
        dest="prebirth_use_grayscale_invert",
        action="store_true",
        default=True,
        help="Use inverted grayscale local thresholding for dark blob search",
    )
    p.add_argument(
        "--no-prebirth-use-grayscale-invert",
        dest="prebirth_use_grayscale_invert",
        action="store_false",
    )

    return p


def run_tracker(args: argparse.Namespace) -> None:
    det_path, resolved_format = resolve_detections_input(
        Path(args.detections), args.input_format
    )
    output_base = Path(args.output_dir)
    output_dir = resolve_tracking_output_dir(
        output_base, use_run_subdir=not args.no_run_subdir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_run_subdir:
        print(f"Tracking output -> {output_dir.resolve()}")

    if resolved_format == "csv":
        dets_by_frame = load_detections_csv(det_path)
    else:
        dets_by_frame = load_detections_txt_dir(det_path)

    if not dets_by_frame:
        print("No detections loaded.")
        return

    tracker = OTKalmanTracker(
        max_age=args.max_age,
        max_age_tentative=args.max_age_tentative,
        max_age_confirmed=args.max_age_confirmed,
        min_hits=args.min_hits,
        min_contiguous_observed=args.min_contiguous_observed,
        max_center_dist=args.max_center_dist,
        max_mahalanobis=args.max_mahalanobis,
        ot_reg=args.ot_reg,
        ot_iters=args.ot_iters,
        dummy_cost=args.dummy_cost,
        min_ot_prob=args.min_ot_prob,
        max_match_cost=args.max_match_cost,
        w_maha=args.w_maha,
        w_center=args.w_center,
        w_iou=args.w_iou,
        w_size=args.w_size,
        w_direction=args.w_direction,
        min_direction_speed=args.min_direction_speed,
        kf_process_pos_std=args.kf_process_pos_std,
        kf_process_vel_std=args.kf_process_vel_std,
        kf_process_size_std=args.kf_process_size_std,
        kf_meas_pos_std=args.kf_meas_pos_std,
        kf_meas_size_std=args.kf_meas_size_std,
        min_det_score=args.min_det_score,
        image_width=args.image_width,
        image_height=args.image_height,
        border_margin=args.border_margin,
        rescue_center_factor=args.rescue_center_factor,
        rescue_maha_factor=args.rescue_maha_factor,
    )

    frame_ids = sorted(dets_by_frame.keys())
    for frame_id in range(frame_ids[0], frame_ids[-1] + 1):
        detections = dets_by_frame.get(frame_id, [])
        stats = tracker.step(frame_id, detections)
        print(
            f"[{frame_id:05d}] det={stats['detections']} "
            f"match={stats['matches']} "
            f"rescue={stats['rescue_matches']} "
            f"rescue_cand={stats['rescue_candidates']} "
            f"new={stats['new_tracks']} "
            f"unmatched_det={stats['unmatched_detections']} "
            f"unmatched_trk={stats['unmatched_tracks']} "
            f"tentative={stats['tentative_tracks']} "
            f"confirmed={stats['confirmed_tracks']} "
            f"lost={stats['lost_tracks']} "
            f"active={stats['active_tracks']} "
            f"finished={stats['finished_tracks']}"
        )

    tracks = tracker.finish()
    kf_params = kf_params_from_tracker(tracker)
    tracks, pp_stats = postprocess_tracks(
        tracks,
        kf_params=kf_params,
        enable_gap_fill=args.enable_gap_fill,
        max_gap_fill=args.max_gap_fill,
        min_contiguous_observed=args.min_contiguous_observed,
        min_observed_hits=args.min_observed_hits,
    )
    print(
        f"\nPostprocess: input={pp_stats['input_tracks']} kept={pp_stats['kept_tracks']} "
        f"removed_short={pp_stats['removed_short']} "
        f"gap_filled_points={pp_stats['gap_filled_points']}"
    )

    if args.enable_prebirth_recovery:
        image_dir = Path(args.image_dir) if args.image_dir else None
        frame_image_map = build_frame_image_map(image_dir, args.image_exts)
        if not frame_image_map:
            print(
                "Prebirth recovery skipped: image_dir missing or no images found."
            )
        else:
            tracks, prebirth_stats = recover_prebirth_dark_particles(
                tracks,
                frame_image_map=frame_image_map,
                max_backward_frames=args.prebirth_max_frames,
                max_failures=args.prebirth_max_failures,
                search_radius_base=args.prebirth_search_radius,
                radius_growth=args.prebirth_radius_growth,
                dark_threshold_mad=args.prebirth_dark_threshold,
                min_blob_area=args.prebirth_min_blob_area,
                max_blob_area=args.prebirth_max_blob_area,
                score_threshold=args.prebirth_score_threshold,
                bright_p95_threshold=args.prebirth_bright_p95_threshold,
                box_scale=args.prebirth_box_scale,
                min_box_size=args.prebirth_min_box_size,
            )
            print(
                f"Prebirth recovery: checked={prebirth_stats['tracks_checked']} "
                f"recovered={prebirth_stats['tracks_recovered']} "
                f"points_added={prebirth_stats['prebirth_points_added']} "
                f"skipped_not_bright={prebirth_stats['skipped_not_bright']} "
                f"skipped_no_velocity={prebirth_stats['skipped_no_velocity']} "
                f"skipped_no_images={prebirth_stats['skipped_no_images']}"
            )

    tracks_path = output_dir / "tracks.csv"
    summary_path = output_dir / "track_summary.csv"
    save_tracks_csv(tracks, tracks_path, min_track_len=args.min_track_len)
    save_track_summary_csv(tracks, summary_path, min_track_len=args.min_track_len)

    n_saved = sum(1 for tr in tracks if len(tr.history) >= args.min_track_len)
    n_confirmed = sum(
        1 for tr in tracks
        if len(tr.history) >= args.min_track_len and tr.is_confirmed
    )

    if args.save_vis:
        maybe_visualize_tracks(
            tracks,
            Path(args.image_dir) if args.image_dir else None,
            output_dir,
            args.image_exts,
            min_track_len=args.min_track_len,
            trail_length=args.trail_length,
            only_confirmed=args.only_confirmed,
            overlay_only=args.vis_overlay_only,
            frame_step=args.vis_frame_step,
        )

    print("\n=== Done ===")
    print(f"共输出轨迹: {n_saved} 条（confirmed: {n_confirmed}）")
    print(f"tracks.csv -> {tracks_path}")
    print(f"track_summary.csv -> {summary_path}")
    if args.save_vis:
        print(f"track_vis -> {output_dir / 'track_vis'}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.vis_only:
        run_visualize_only(args)
        return
    run_tracker(args)


if __name__ == "__main__":
    main()
