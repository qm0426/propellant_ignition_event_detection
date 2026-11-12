from __future__ import annotations



import math

from dataclasses import dataclass, field


from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import cv2

import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

def read_gray_image(path: Path) -> Optional[np.ndarray]:
    """安全读取灰度图，返回 float32 2D array；失败返回 None。"""
    try:
        from PIL import Image

        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        return gray
    except Exception:
        return None

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
