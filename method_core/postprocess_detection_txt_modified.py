from __future__ import annotations

import argparse


import logging

import re


from typing import Any, Dict, List, Tuple

import numpy as np

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_RE_LOCATE_BOX = re.compile(
    r"label=(?P<label>[^\s]+)\s+"
    r"x1=(?P<x1>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"y1=(?P<y1>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"x2=(?P<x2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"y2=(?P<y2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)

def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "t", "1"):
        return True
    if text in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")

def clip_box_xyxy(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(x1), float(img_w)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    y2 = max(0.0, min(float(y2), float(img_h)))
    return x1, y1, x2, y2

def box_area(box: Dict[str, Any]) -> float:
    w = max(0.0, box["x2"] - box["x1"])
    h = max(0.0, box["y2"] - box["y1"])
    return w * h

def box_intersection_area(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_aspect_ratio(box: Dict[str, Any]) -> float:
    w = max(1e-6, box["x2"] - box["x1"])
    h = max(1e-6, box["y2"] - box["y1"])
    return max(w / h, h / w)

def compute_box_content_metrics(
    image_rgb: Image.Image,
    box: Dict[str, Any],
    ring_scale: float = 1.5,
    bg_diff_thresh: float = 20.0,
    dark_thresh: float = 55.0,
    bright_thresh: float = 210.0,
    orange_thresh: float = 25.0,
) -> Dict[str, float]:
    img = np.asarray(image_rgb.convert("RGB"), dtype=np.float32)
    h_img, w_img = img.shape[:2]

    x1, y1, x2, y2 = clip_box_xyxy(
        box["x1"], box["y1"], box["x2"], box["y2"], w_img, h_img
    )
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    area = bw * bh

    empty = {
        "box_w": bw,
        "box_h": bh,
        "area": area,
        "aspect_ratio": box_aspect_ratio({"x1": x1, "y1": y1, "x2": x2, "y2": y2}),
        "gray_mean": 0.0,
        "gray_p95": 0.0,
        "gray_p05": 0.0,
        "gray_std": 0.0,
        "local_contrast": 0.0,
        "orange_p90": 0.0,
        "dark_pixel_ratio": 0.0,
        "bright_pixel_ratio": 0.0,
        "orange_pixel_ratio": 0.0,
        "non_background_ratio": 0.0,
    }
    if area <= 0:
        return empty

    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    if ix2 <= ix1:
        ix2 = min(w_img, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(h_img, iy1 + 1)

    gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    roi_gray = gray[iy1:iy2, ix1:ix2]
    roi_rgb = img[iy1:iy2, ix1:ix2]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ow = bw * ring_scale
    oh = bh * ring_scale
    ox1 = max(0, int(cx - ow / 2.0))
    oy1 = max(0, int(cy - oh / 2.0))
    ox2 = min(w_img, int(cx + ow / 2.0))
    oy2 = min(h_img, int(cy + oh / 2.0))

    ring_vals: List[np.ndarray] = []
    if ox2 > ox1 and oy2 > oy1:
        outer = gray[oy1:oy2, ox1:ox2]
        mask = np.ones(outer.shape, dtype=bool)
        inner_x1 = max(ix1, ox1) - ox1
        inner_y1 = max(iy1, oy1) - oy1
        inner_x2 = min(ix2, ox2) - ox1
        inner_y2 = min(iy2, oy2) - oy1
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
        ring_pixels = outer[mask]
        if ring_pixels.size >= 16:
            ring_vals.append(ring_pixels)

    if ring_vals:
        bg_gray_median = float(np.median(np.concatenate(ring_vals)))
    else:
        margin = max(8, int(max(bw, bh) * 0.5))
        lx1 = max(0, ix1 - margin)
        ly1 = max(0, iy1 - margin)
        lx2 = min(w_img, ix2 + margin)
        ly2 = min(h_img, iy2 + margin)
        local = gray[ly1:ly2, lx1:lx2]
        bg_gray_median = float(np.median(local)) if local.size else float(np.median(gray))

    r = roi_rgb[:, :, 0]
    g = roi_rgb[:, :, 1]
    b = roi_rgb[:, :, 2]
    orange_score = r - np.maximum(g, b)

    fg = (
        (np.abs(roi_gray - bg_gray_median) > bg_diff_thresh)
        | (roi_gray < dark_thresh)
        | (roi_gray > bright_thresh)
        | (orange_score > orange_thresh)
    )
    pixel_count = roi_gray.size
    dark_pixel_ratio = float((roi_gray < dark_thresh).sum() / pixel_count)
    bright_pixel_ratio = float((roi_gray > bright_thresh).sum() / pixel_count)
    orange_pixel_ratio = float((orange_score > orange_thresh).sum() / pixel_count)
    non_background_ratio = float(fg.sum() / pixel_count)

    return {
        "box_w": bw,
        "box_h": bh,
        "area": area,
        "aspect_ratio": box_aspect_ratio({"x1": x1, "y1": y1, "x2": x2, "y2": y2}),
        "gray_mean": float(roi_gray.mean()),
        "gray_p95": float(np.percentile(roi_gray, 95)),
        "gray_p05": float(np.percentile(roi_gray, 5)),
        "gray_std": float(np.std(roi_gray)),
        "local_contrast": float((np.percentile(roi_gray, 95) - np.percentile(roi_gray, 5)) / 255.0),
        "orange_p90": float(np.percentile(orange_score, 90)),
        "dark_pixel_ratio": dark_pixel_ratio,
        "bright_pixel_ratio": bright_pixel_ratio,
        "orange_pixel_ratio": orange_pixel_ratio,
        "non_background_ratio": non_background_ratio,
    }

def compute_size_score(
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    min_box_side: float,
    max_box_side_frac: float,
    max_box_area_frac: float,
) -> float:
    side = min(w, h)
    if side < min_box_side:
        return 0.0
    area_frac = (w * h) / max(1.0, img_w * img_h)
    side_frac = max(w / img_w, h / img_h)
    score = 1.0
    if side < min_box_side * 2:
        score = min(score, 0.5 + 0.5 * (side - min_box_side) / min_box_side)
    if area_frac > max_box_area_frac:
        score = min(
            score,
            max(0.0, 1.0 - (area_frac - max_box_area_frac) / max(max_box_area_frac, 1e-6)),
        )
    if side_frac > max_box_side_frac:
        score = min(
            score,
            max(0.0, 1.0 - (side_frac - max_box_side_frac) / max(max_box_side_frac, 1e-6)),
        )
    return float(np.clip(score, 0.0, 1.0))

def compute_final_quality_score(
    metrics: Dict[str, float],
    max_aspect_ratio: float,
) -> float:
    signal_ratio = (
        metrics["dark_pixel_ratio"]
        + metrics["bright_pixel_ratio"]
        + metrics["orange_pixel_ratio"]
    )
    aspect = max(metrics["aspect_ratio"], 1e-6)
    compactness_score = min(1.0, max_aspect_ratio / aspect)
    return float(
        0.5 * metrics["non_background_ratio"]
        + 0.3 * signal_ratio
        + 0.2 * compactness_score
    )

def enrich_box_metrics(box: Dict[str, Any], image: Image.Image, args: argparse.Namespace) -> None:
    metrics = compute_box_content_metrics(
        image,
        box,
        bg_diff_thresh=args.bg_diff_thresh,
        dark_thresh=args.dark_thresh,
        bright_thresh=args.bright_thresh,
        orange_thresh=args.orange_thresh,
    )
    box.update(metrics)
    box["size_score"] = compute_size_score(
        metrics["box_w"],
        metrics["box_h"],
        image.size[0],
        image.size[1],
        args.min_box_side,
        args.max_box_side_frac,
        args.max_box_area_frac,
    )
    box["final_quality_score"] = compute_final_quality_score(
        metrics, args.max_aspect_ratio
    )
    box["keep"] = True
    box["remove_reason"] = ""

def apply_rule1_background(box: Dict[str, Any], args: argparse.Namespace) -> None:
    if not box["keep"]:
        return
    if box["non_background_ratio"] < args.min_non_background_ratio:
        box["keep"] = False
        box["remove_reason"] = "background_dominated"

def apply_rule2_geometry(
    box: Dict[str, Any], img_w: int, img_h: int, args: argparse.Namespace
) -> None:
    if not box["keep"]:
        return

    w = box["box_w"]
    h = box["box_h"]
    area = box["area"]
    img_area = img_w * img_h
    side = min(w, h)

    if side < args.min_box_side:
        box["keep"] = False
        box["remove_reason"] = "too_small"
        return

    if box["aspect_ratio"] > args.max_aspect_ratio:
        box["keep"] = False
        box["remove_reason"] = "elongated_box"
        return

    if area > args.extreme_max_box_area_frac * img_area:
        box["keep"] = False
        box["remove_reason"] = "extremely_large_box"

def apply_optional_defocus_filter(box: Dict[str, Any], args: argparse.Namespace) -> None:
    """可选：删除低对比、低信号、疑似离焦/虚影框。默认关闭。

    这个规则较保守，因为未燃烧小颗粒也可能低亮、低对比；建议先通过时间连续性过滤。
    """
    if not box.get("keep") or not getattr(args, "enable_defocus_filter", False):
        return
    if (
        box.get("local_contrast", 0.0) < args.defocus_max_local_contrast
        and box.get("non_background_ratio", 0.0) < args.defocus_max_non_background_ratio
        and box_signal_ratio(box) < args.defocus_max_signal_ratio
    ):
        box["keep"] = False
        box["remove_reason"] = "defocused_low_contrast"

def box_signal_ratio(box: Dict[str, Any]) -> float:
    return float(
        box.get("dark_pixel_ratio", 0.0)
        + box.get("bright_pixel_ratio", 0.0)
        + box.get("orange_pixel_ratio", 0.0)
    )

def box_center(box: Dict[str, Any]) -> Tuple[float, float]:
    return (float(box["x1"] + box["x2"]) / 2.0, float(box["y1"] + box["y2"]) / 2.0)

def center_inside(inner: Dict[str, Any], outer: Dict[str, Any]) -> bool:
    cx, cy = box_center(inner)
    return outer["x1"] <= cx <= outer["x2"] and outer["y1"] <= cy <= outer["y2"]

def containment_ratio(inner: Dict[str, Any], outer: Dict[str, Any]) -> float:
    area = box_area(inner)
    if area <= 0:
        return 0.0
    return box_intersection_area(inner, outer) / area

def is_valid_large_box(box: Dict[str, Any], args: argparse.Namespace) -> bool:
    signal_ratio = box_signal_ratio(box)
    quality_ok = box.get("final_quality_score", 0.0) >= args.nested_large_good_score
    content_ok = (
        box.get("non_background_ratio", 0.0) > args.large_box_min_non_background_ratio
        or signal_ratio > args.large_box_min_signal_ratio
    )
    return content_ok and quality_ok

def is_good_small_box(box: Dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        box.get("final_quality_score", 0.0) >= args.nested_small_good_score
        or box.get("non_background_ratio", 0.0) >= args.nested_small_min_non_background_ratio
        or box_signal_ratio(box) >= args.nested_small_min_signal_ratio
    )

def is_nested_pair(big: Dict[str, Any], small: Dict[str, Any], args: argparse.Namespace) -> bool:
    big_area = box_area(big)
    small_area = box_area(small)
    if small_area <= 0 or big_area <= small_area * args.nested_area_ratio:
        return False
    contain = containment_ratio(small, big)
    if contain >= args.nested_containment_thresh:
        return True
    # LocateAnything 的大框有时会偏移，导致 containment 不到 0.85。
    # 对中心落在大框内部、且有明显交叠的情况，使用较宽松阈值辅助判断。
    return center_inside(small, big) and contain >= args.nested_soft_containment_thresh

def apply_rule3_nested(boxes: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """处理大框/小框嵌套关系。

    修正两类常见情况：
    1) 质量较差的大框包住质量较好的小框：删除大框。
    2) 质量较好的大框包住质量也不错的小框：保留大框，删除小框。

    该规则仍然是 class-agnostic，服务于后续轨迹连接前的候选框清洗。
    """
    active = [b for b in boxes if b.get("keep")]

    # 先删除“坏大框包好小框”，避免坏大框反过来压制小框。
    for big in sorted(active, key=lambda b: box_area(b), reverse=True):
        if not big.get("keep"):
            continue
        contained_smalls = [
            s
            for s in active
            if s is not big and s.get("keep") and is_nested_pair(big, s, args)
        ]
        if not contained_smalls:
            continue

        good_smalls = [s for s in contained_smalls if is_good_small_box(s, args)]
        if not good_smalls:
            continue

        best_small = max(good_smalls, key=lambda s: s.get("final_quality_score", 0.0))
        big_score = float(big.get("final_quality_score", 0.0))
        best_small_score = float(best_small.get("final_quality_score", 0.0))
        big_good = is_valid_large_box(big, args)

        if (not big_good) or (big_score + args.nested_large_score_margin < best_small_score):
            big["keep"] = False
            big["remove_reason"] = "poor_large_contains_good_small"

    # 再处理“好大框包小框”：大框保留，小框视为重复检测删除。
    active = [b for b in boxes if b.get("keep")]
    for big in sorted(active, key=lambda b: box_area(b), reverse=True):
        if not big.get("keep") or not is_valid_large_box(big, args):
            continue
        for small in active:
            if small is big or not small.get("keep"):
                continue
            if not is_nested_pair(big, small, args):
                continue
            # 两个框质量都可以时，也按“更完整的大框”保留，删除内部小框。
            small["keep"] = False
            small["remove_reason"] = "inside_valid_large_box"

def nms_score(box: Dict[str, Any]) -> float:
    if box.get("score") is not None:
        return float(box["score"])
    if box.get("final_quality_score") is not None:
        return float(box["final_quality_score"])
    return (
        box.get("non_background_ratio", 0.0)
        + box.get("bright_pixel_ratio", 0.0)
        + box.get("orange_pixel_ratio", 0.0)
        + box.get("dark_pixel_ratio", 0.0)
    )

def box_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = box_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0

def suppress_lower_quality_pair(
    a: Dict[str, Any],
    b: Dict[str, Any],
    reason: str,
) -> None:
    if nms_score(a) >= nms_score(b):
        b["keep"] = False
        b["remove_reason"] = reason
    else:
        a["keep"] = False
        a["remove_reason"] = reason

def apply_duplicate_overlap_suppression(boxes: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """删除同一区域内高度重叠的重复好框。

    与 Rule 3 不同，这里主要处理面积相近、IoU 较大或相互覆盖度较高的框。
    """
    candidates = [b for b in boxes if b.get("keep")]
    candidates.sort(key=lambda x: (-nms_score(x), x.get("_idx", 0)))
    for i, a in enumerate(candidates):
        if not a.get("keep"):
            continue
        for b in candidates[i + 1 :]:
            if not b.get("keep"):
                continue
            area_a = box_area(a)
            area_b = box_area(b)
            if area_a <= 0 or area_b <= 0:
                continue
            size_ratio = max(area_a, area_b) / max(min(area_a, area_b), 1e-6)
            iou = box_iou(a, b)
            contain_ab = containment_ratio(a, b)
            contain_ba = containment_ratio(b, a)
            high_containment = max(contain_ab, contain_ba) >= args.duplicate_containment_thresh
            comparable_size = size_ratio <= args.duplicate_size_ratio

            if iou >= args.duplicate_iou_thresh or (comparable_size and high_containment):
                suppress_lower_quality_pair(a, b, "overlap_duplicate")
                if not a.get("keep"):
                    break

def apply_nms(boxes: List[Dict[str, Any]], iou_threshold: float) -> None:
    candidates = [b for b in boxes if b["keep"]]
    candidates.sort(key=lambda x: (-nms_score(x), x.get("_idx", 0)))
    suppressed = set()
    for i, best in enumerate(candidates):
        if id(best) in suppressed:
            continue
        for cand in candidates[i + 1 :]:
            if id(cand) in suppressed:
                continue
            if box_iou(best, cand) >= iou_threshold:
                cand["keep"] = False
                cand["remove_reason"] = "nms_duplicate"
                suppressed.add(id(cand))

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="清洗图像序列开放词汇检测 txt，过滤背景框、细长框和嵌套重复框。"
    )
    p.add_argument("--image-dir", default=r'test_images', help="图像序列文件夹")
    p.add_argument("--txt-dir", default='inputs/detections_txt', help="检测 txt 文件夹，或单个 txt/csv 文件")
    p.add_argument("--output-dir", default='outputs/cleaning', help="输出根目录")
    p.add_argument(
        "--image-exts",
        default=".png,.jpg,.jpeg,.tif,.tiff",
        help="图像后缀，逗号分隔",
    )
    p.add_argument(
        "--box-format",
        choices=["xyxy", "yolo"],
        default="xyxy",
        help="输入框格式",
    )
    p.add_argument(
        "--normalized",
        type=str2bool,
        default=False,
        help="坐标是否为归一化 (0-1)",
    )
    p.add_argument(
        "--output-with-label",
        type=str2bool,
        default=True,
        help="输出 txt 是否带 label 列",
    )
    p.add_argument(
        "--draw-removed",
        type=str2bool,
        default=False,
        help="可视化中是否绘制被删除的框",
    )

    p.add_argument("--bg-diff-thresh", type=float, default=20.0)
    p.add_argument("--dark-thresh", type=float, default=55.0)
    p.add_argument("--bright-thresh", type=float, default=210.0)
    p.add_argument("--orange-thresh", type=float, default=25.0)
    p.add_argument("--min-non-background-ratio", type=float, default=0.08)
    p.add_argument("--max-aspect-ratio", type=float, default=2.8)
    p.add_argument("--min-box-side", type=float, default=3.0)
    p.add_argument("--max-box-side-frac", type=float, default=0.18)
    p.add_argument("--max-box-area-frac", type=float, default=0.03)
    p.add_argument("--extreme-max-box-area-frac", type=float, default=0.12)

    p.add_argument("--nested-area-ratio", type=float, default=2.5)
    p.add_argument("--nested-containment-thresh", type=float, default=0.85)
    p.add_argument(
        "--nested-soft-containment-thresh",
        type=float,
        default=0.45,
        help="中心落入大框时使用的宽松 containment 阈值，用于处理偏移大框",
    )
    p.add_argument(
        "--large-box-min-non-background-ratio", type=float, default=0.12
    )
    p.add_argument("--large-box-min-signal-ratio", type=float, default=0.10)
    p.add_argument(
        "--nested-small-good-score",
        type=float,
        default=0.45,
        help="小框被认为质量较好时的 final_quality_score 阈值",
    )
    p.add_argument("--nested-small-min-non-background-ratio", type=float, default=0.10)
    p.add_argument("--nested-small-min-signal-ratio", type=float, default=0.08)
    p.add_argument(
        "--nested-large-good-score",
        type=float,
        default=0.55,
        help="大框被认为质量足够好并可压制内部小框时的分数阈值",
    )
    p.add_argument(
        "--nested-large-score-margin",
        type=float,
        default=0.08,
        help="若大框分数比内部最好小框低超过该值，则删除大框",
    )

    p.add_argument(
        "--enable-duplicate-suppression",
        dest="enable_duplicate_suppression",
        action="store_true",
        default=True,
        help="启用同区域高度重叠框抑制，默认启用",
    )
    p.add_argument(
        "--disable-duplicate-suppression",
        dest="enable_duplicate_suppression",
        action="store_false",
        help="关闭同区域高度重叠框抑制",
    )
    p.add_argument("--duplicate-iou-thresh", type=float, default=0.35)
    p.add_argument("--duplicate-containment-thresh", type=float, default=0.70)
    p.add_argument("--duplicate-size-ratio", type=float, default=2.2)

    p.add_argument(
        "--enable-nms",
        dest="enable_nms",
        action="store_true",
        default=True,
        help="启用 IoU NMS，默认启用",
    )
    p.add_argument(
        "--disable-nms",
        dest="enable_nms",
        action="store_false",
        help="关闭 IoU NMS",
    )
    p.add_argument("--nms-iou", type=float, default=0.35)

    p.add_argument(
        "--enable-defocus-filter",
        action="store_true",
        help="可选：删除低对比、低信号的疑似离焦框；默认关闭，避免误删未燃烧小颗粒",
    )
    p.add_argument("--defocus-max-local-contrast", type=float, default=0.06)
    p.add_argument("--defocus-max-non-background-ratio", type=float, default=0.12)
    p.add_argument("--defocus-max-signal-ratio", type=float, default=0.08)
    return p
