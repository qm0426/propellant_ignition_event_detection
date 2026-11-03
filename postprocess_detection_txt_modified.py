#!/usr/bin/env python3
"""
轻量级检测结果后处理：清洗开放词汇检测 txt，删除背景占比过高、细长框和嵌套重复框。

不做轨迹连接，仅做 detection box cleaning / filtering。
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# LocateAnything verbose: box_0: label=foo x1=1.0 y1=2.0 x2=3.0 y2=4.0
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


def parse_image_exts(exts: str) -> set:
    parts = [p.strip().lower() for p in exts.split(",") if p.strip()]
    return {p if p.startswith(".") else f".{p}" for p in parts}


def collect_image_paths(image_dir: Path, exts: set) -> List[Path]:
    paths = []
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            paths.append(path)
    return paths


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


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
    normalized: bool,
) -> Tuple[float, float, float, float]:
    if normalized:
        cx *= img_w
        cy *= img_h
        w *= img_w
        h *= img_h
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)


def _try_floats(tokens: List[str]) -> Optional[List[float]]:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _parse_line_xyxy(
    line: str,
    img_w: int,
    img_h: int,
    normalized: bool,
) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    m = _RE_LOCATE_BOX.search(stripped)
    if m:
        x1, y1, x2, y2 = (
            float(m.group("x1")),
            float(m.group("y1")),
            float(m.group("x2")),
            float(m.group("y2")),
        )
        if normalized:
            x1, x2 = x1 * img_w, x2 * img_w
            y1, y2 = y1 * img_h, y2 * img_h
        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)
        return {
            "label": m.group("label"),
            "score": None,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "source_line": stripped,
        }

    if "\t" in stripped:
        parts = [p.strip() for p in stripped.split("\t")]
        delim = "tab"
    elif "," in stripped and not re.match(r"^[\d.\-+eE,\s]+$", stripped):
        parts = [p.strip() for p in stripped.split(",")]
        delim = "csv"
    else:
        parts = stripped.split()
        delim = "space"

    if not parts:
        return None

    label = ""
    score = None
    coords: Optional[List[float]] = None

    if delim in ("tab", "csv"):
        header_like = parts[0].lower() in (
            "label",
            "class",
            "name",
            "x1",
            "image_name",
            "image_path",
        )
        if header_like:
            return None
        if len(parts) >= 5:
            label = parts[0]
            nums = _try_floats(parts[1:5])
            if nums is not None:
                coords = nums
                if len(parts) >= 6:
                    score = float(parts[5])
        elif len(parts) == 4:
            nums = _try_floats(parts)
            if nums is not None:
                coords = nums
    else:
        nums = _try_floats(parts)
        if nums is None:
            return None
        if len(nums) == 4:
            coords = nums
        elif len(nums) == 5:
            label = str(int(nums[0])) if nums[0] == int(nums[0]) else str(nums[0])
            coords = nums[1:5]
        elif len(nums) == 6:
            label = str(int(nums[0])) if nums[0] == int(nums[0]) else str(nums[0])
            coords = nums[1:5]
            score = nums[5]
        else:
            return None

    if coords is None:
        if len(parts) >= 4:
            if not _try_floats(parts[:1]):
                label = parts[0]
                coords = _try_floats(parts[1:5])
                if coords and len(parts) >= 6:
                    try:
                        score = float(parts[5])
                    except ValueError:
                        pass
            else:
                coords = _try_floats(parts[:4])
                if coords and len(parts) >= 5:
                    try:
                        score = float(parts[4])
                    except ValueError:
                        pass

    if coords is None or len(coords) != 4:
        return None

    x1, y1, x2, y2 = coords
    if normalized:
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h
    x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)
    return {
        "label": label,
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "source_line": stripped,
    }


def _parse_line_yolo(
    line: str,
    img_w: int,
    img_h: int,
    normalized: bool,
) -> Optional[Dict[str, Any]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if "\t" in stripped:
        parts = [p.strip() for p in stripped.split("\t")]
    elif "," in stripped:
        parts = [p.strip() for p in stripped.split(",")]
    else:
        parts = stripped.split()

    if len(parts) < 4:
        return None
    if parts[0].lower() in ("class", "label", "name"):
        return None

    nums = _try_floats(parts[1:]) if not _try_floats(parts[:1]) else _try_floats(parts)
    label = ""
    score = None

    if nums is None:
        label = parts[0]
        nums = _try_floats(parts[1:])
    elif len(parts) >= 5 and not _try_floats(parts[:1]):
        label = parts[0]
        nums = _try_floats(parts[1:])

    if nums is None or len(nums) < 4:
        return None

    cx, cy, w, h = nums[:4]
    if len(nums) >= 5:
        score = nums[4]
    if not label and len(parts) >= 5 and not _try_floats(parts[:1]):
        label = parts[0]

    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h, normalized)
    return {
        "label": label,
        "score": score,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "source_line": stripped,
    }


def parse_detection_lines(
    lines: List[str],
    box_format: str,
    normalized: bool,
    img_w: int,
    img_h: int,
    image_name: str = "",
    frame_id: int = 0,
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    parser = _parse_line_yolo if box_format == "yolo" else _parse_line_xyxy

    for line in lines:
        parsed = parser(line, img_w, img_h, normalized)
        if parsed is None:
            if line.strip() and not line.strip().startswith("#"):
                lower = line.strip().lower()
                if not any(
                    skip in lower
                    for skip in (
                        "image_path:",
                        "total_boxes:",
                        "prompt:",
                        "label\tx1",
                        "label,x1",
                        "image_name",
                    )
                ):
                    logger.warning("无法解析行 [%s]: %s", image_name, line.strip())
            continue
        parsed["frame_id"] = frame_id
        parsed["image_name"] = image_name
        boxes.append(parsed)
    return boxes


def parse_detection_file(
    txt_path: Path,
    box_format: str,
    normalized: bool,
    img_w: int,
    img_h: int,
    image_name: str,
    frame_id: int,
    image_stem: str = "",
) -> List[Dict[str, Any]]:
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    suffix = txt_path.suffix.lower()

    if suffix == ".csv":
        return _parse_csv_file(
            txt_path, box_format, normalized, img_w, img_h, image_name, frame_id, image_stem
        )

    return parse_detection_lines(
        text.splitlines(),
        box_format=box_format,
        normalized=normalized,
        img_w=img_w,
        img_h=img_h,
        image_name=image_name,
        frame_id=frame_id,
    )


def _parse_csv_file(
    csv_path: Path,
    box_format: str,
    normalized: bool,
    img_w: int,
    img_h: int,
    image_name: str,
    frame_id: int,
    image_stem: str,
) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return boxes
        fields = {c.lower(): c for c in reader.fieldnames}
        for row in reader:
            row_image = ""
            for key in ("image_name", "image_path", "filename", "frame"):
                if key in fields:
                    row_image = Path(row[fields[key]]).stem
                    break
            if image_stem and row_image and row_image != image_stem:
                continue

            try:
                label = row.get(fields.get("label", "label"), "") or ""
                score_val = row.get(fields.get("score", "score"), "")
                score = float(score_val) if score_val not in ("", None) else None

                if box_format == "yolo":
                    cx = float(row[fields.get("cx", "cx")])
                    cy = float(row[fields.get("cy", "cy")])
                    w = float(row[fields.get("w", "w")])
                    h = float(row[fields.get("h", "h")])
                    x1, y1, x2, y2 = yolo_to_xyxy(
                        cx, cy, w, h, img_w, img_h, normalized
                    )
                else:
                    x1 = float(row[fields.get("x1", "x1")])
                    y1 = float(row[fields.get("y1", "y1")])
                    x2 = float(row[fields.get("x2", "x2")])
                    y2 = float(row[fields.get("y2", "y2")])
                    if normalized:
                        x1, x2 = x1 * img_w, x2 * img_w
                        y1, y2 = y1 * img_h, y2 * img_h
                    x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)

                boxes.append(
                    {
                        "frame_id": frame_id,
                        "image_name": image_name,
                        "label": str(label) if label else "",
                        "score": score,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "source_line": str(row),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("CSV 行解析失败 [%s]: %s", image_name, exc)
    return boxes


def find_txt_for_image(image_path: Path, txt_dir: Path) -> Optional[Path]:
    stem = image_path.stem
    for ext in (".txt", ".csv"):
        candidate = txt_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


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


def save_cleaned_txt(
    boxes: List[Dict[str, Any]], out_path: Path, output_with_label: bool
) -> None:
    lines: List[str] = []
    for b in boxes:
        if not b["keep"]:
            continue
        score = b.get("final_quality_score", 0.0)
        if output_with_label:
            lines.append(
                f"particle {b['x1']:.2f} {b['y1']:.2f} {b['x2']:.2f} {b['y2']:.2f} {score:.4f}"
            )
        else:
            lines.append(
                f"{b['x1']:.2f} {b['y1']:.2f} {b['x2']:.2f} {b['y2']:.2f} {score:.4f}"
            )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_visualization(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    draw_removed: bool,
    out_path: Path,
) -> None:
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, b in enumerate(boxes):
        kept = b["keep"]
        if not kept and not draw_removed:
            continue
        color = (0, 200, 0) if kept else (220, 60, 60)
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if kept:
            text = f"{i}:{b.get('final_quality_score', 0):.2f}"
        else:
            text = f"{i}:{b.get('remove_reason', '')}"
        draw.text((x1 + 2, max(0, y1 - 14)), text, fill=color, font=font)

    vis.save(out_path)


def box_to_summary_row(box: Dict[str, Any], source_txt: str) -> Dict[str, Any]:
    return {
        "frame_id": box.get("frame_id", ""),
        "image_name": box.get("image_name", ""),
        "source_txt": source_txt,
        "source_line": box.get("source_line", ""),
        "label": box.get("label", ""),
        "raw_score": box.get("score", ""),
        "x1": f"{box['x1']:.2f}",
        "y1": f"{box['y1']:.2f}",
        "x2": f"{box['x2']:.2f}",
        "y2": f"{box['y2']:.2f}",
        "w": f"{box.get('box_w', 0):.2f}",
        "h": f"{box.get('box_h', 0):.2f}",
        "area": f"{box.get('area', 0):.2f}",
        "aspect_ratio": f"{box.get('aspect_ratio', 0):.4f}",
        "gray_mean": f"{box.get('gray_mean', 0):.2f}",
        "gray_p05": f"{box.get('gray_p05', 0):.2f}",
        "gray_p95": f"{box.get('gray_p95', 0):.2f}",
        "gray_std": f"{box.get('gray_std', 0):.2f}",
        "local_contrast": f"{box.get('local_contrast', 0):.4f}",
        "signal_ratio": f"{box_signal_ratio(box):.4f}",
        "dark_pixel_ratio": f"{box.get('dark_pixel_ratio', 0):.4f}",
        "bright_pixel_ratio": f"{box.get('bright_pixel_ratio', 0):.4f}",
        "orange_pixel_ratio": f"{box.get('orange_pixel_ratio', 0):.4f}",
        "non_background_ratio": f"{box.get('non_background_ratio', 0):.4f}",
        "final_quality_score": f"{box.get('final_quality_score', 0):.4f}",
        "keep": int(bool(box.get("keep"))),
        "remove_reason": box.get("remove_reason", ""),
    }


def box_to_cleaned_row(box: Dict[str, Any]) -> Dict[str, Any]:
    w = box["x2"] - box["x1"]
    h = box["y2"] - box["y1"]
    return {
        "frame_id": box.get("frame_id", ""),
        "image_name": box.get("image_name", ""),
        "x1": f"{box['x1']:.2f}",
        "y1": f"{box['y1']:.2f}",
        "x2": f"{box['x2']:.2f}",
        "y2": f"{box['y2']:.2f}",
        "cx": f"{(box['x1'] + box['x2']) / 2:.2f}",
        "cy": f"{(box['y1'] + box['y2']) / 2:.2f}",
        "w": f"{w:.2f}",
        "h": f"{h:.2f}",
        "final_quality_score": f"{box.get('final_quality_score', 0):.4f}",
    }


def new_postprocess_stats() -> Dict[str, int]:
    return {
        "raw": 0,
        "kept": 0,
        "background_dominated": 0,
        "elongated_box": 0,
        "inside_valid_large_box": 0,
        "poor_large_contains_good_small": 0,
        "overlap_duplicate": 0,
        "nms_duplicate": 0,
        "defocused_low_contrast": 0,
        "others": 0,
    }


def tally_remove_stats(boxes: List[Dict[str, Any]], stats: Dict[str, int]) -> None:
    stats["kept"] = sum(1 for b in boxes if b.get("keep"))
    for b in boxes:
        if b.get("keep"):
            continue
        reason = b.get("remove_reason", "")
        if reason == "background_dominated":
            stats["background_dominated"] += 1
        elif reason == "elongated_box":
            stats["elongated_box"] += 1
        elif reason == "inside_valid_large_box":
            stats["inside_valid_large_box"] += 1
        elif reason == "poor_large_contains_good_small":
            stats["poor_large_contains_good_small"] += 1
        elif reason == "overlap_duplicate":
            stats["overlap_duplicate"] += 1
        elif reason == "nms_duplicate":
            stats["nms_duplicate"] += 1
        elif reason == "defocused_low_contrast":
            stats["defocused_low_contrast"] += 1
        else:
            stats["others"] += 1


def run_postprocess_pipeline(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    img_w, img_h = image.size
    for idx, box in enumerate(boxes):
        box["_idx"] = idx
        x1, y1, x2, y2 = clip_box_xyxy(
            box["x1"], box["y1"], box["x2"], box["y2"], img_w, img_h
        )
        box["x1"], box["y1"], box["x2"], box["y2"] = x1, y1, x2, y2
        enrich_box_metrics(box, image, args)
        apply_rule1_background(box, args)
        apply_rule2_geometry(box, img_w, img_h, args)
        apply_optional_defocus_filter(box, args)

    apply_rule3_nested(boxes, args)

    if args.enable_duplicate_suppression:
        apply_duplicate_overlap_suppression(boxes, args)

    if args.enable_nms:
        apply_nms(boxes, args.nms_iou)


def postprocess_boxes_in_memory(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    frame_id: int,
    image_name: str,
    stem: str,
    args: argparse.Namespace,
    postprocess_dir: Path,
    source_tag: str = "inline_detection",
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """对内存中的检测框执行后处理，并写入 postprocess_dir 子目录。"""
    import copy

    working = copy.deepcopy(boxes)
    stats = new_postprocess_stats()
    stats["raw"] = len(working)

    for i, box in enumerate(working):
        box.setdefault("frame_id", frame_id)
        box.setdefault("image_name", image_name)
        box.setdefault("source_line", f"{source_tag} box_{i}")
        box.setdefault("score", None)

    run_postprocess_pipeline(image, working, args)
    tally_remove_stats(working, stats)

    cleaned_txt_dir = postprocess_dir / "cleaned_detections_txt"
    cleaned_vis_dir = postprocess_dir / "cleaned_vis"
    cleaned_txt_dir.mkdir(parents=True, exist_ok=True)
    cleaned_vis_dir.mkdir(parents=True, exist_ok=True)

    save_cleaned_txt(working, cleaned_txt_dir / f"{stem}.txt", args.output_with_label)
    draw_visualization(
        image,
        working,
        draw_removed=args.draw_removed,
        out_path=cleaned_vis_dir / f"{stem}.png",
    )

    print(
        f"  postprocess | raw={stats['raw']} kept={stats['kept']} "
        f"removed: bg={stats['background_dominated']} elongated={stats['elongated_box']} "
        f"nested={stats['inside_valid_large_box']} bad_large={stats['poor_large_contains_good_small']} "
        f"overlap={stats['overlap_duplicate']} nms={stats['nms_duplicate']} "
        f"defocus={stats['defocused_low_contrast']} others={stats['others']}"
    )

    summary_rows = [box_to_summary_row(b, source_tag) for b in working]
    return summary_rows, stats


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    """向 ArgumentParser 注册后处理相关参数。"""
    group = parser.add_argument_group("postprocess")
    group.add_argument(
        "--output-with-label",
        type=str2bool,
        default=True,
        help="后处理输出 txt 是否带 label 列",
    )
    group.add_argument(
        "--draw-removed",
        type=str2bool,
        default=False,
        help="后处理可视化是否绘制被删除的框",
    )
    group.add_argument("--bg-diff-thresh", type=float, default=20.0)
    group.add_argument("--dark-thresh", type=float, default=55.0)
    group.add_argument("--bright-thresh", type=float, default=210.0)
    group.add_argument("--orange-thresh", type=float, default=25.0)
    group.add_argument("--min-non-background-ratio", type=float, default=0.08)
    group.add_argument("--max-aspect-ratio", type=float, default=2.8)
    group.add_argument("--min-box-side", type=float, default=3.0)
    group.add_argument("--max-box-side-frac", type=float, default=0.18)
    group.add_argument("--max-box-area-frac", type=float, default=0.03)
    group.add_argument("--extreme-max-box-area-frac", type=float, default=0.12)
    group.add_argument("--nested-area-ratio", type=float, default=2.5)
    group.add_argument("--nested-containment-thresh", type=float, default=0.85)
    group.add_argument("--nested-soft-containment-thresh", type=float, default=0.45)
    group.add_argument("--large-box-min-non-background-ratio", type=float, default=0.12)
    group.add_argument("--large-box-min-signal-ratio", type=float, default=0.10)
    group.add_argument("--nested-small-good-score", type=float, default=0.45)
    group.add_argument("--nested-small-min-non-background-ratio", type=float, default=0.10)
    group.add_argument("--nested-small-min-signal-ratio", type=float, default=0.08)
    group.add_argument("--nested-large-good-score", type=float, default=0.55)
    group.add_argument("--nested-large-score-margin", type=float, default=0.08)
    group.add_argument(
        "--enable-duplicate-suppression",
        dest="enable_duplicate_suppression",
        action="store_true",
        default=True,
    )
    group.add_argument(
        "--disable-duplicate-suppression",
        dest="enable_duplicate_suppression",
        action="store_false",
    )
    group.add_argument("--duplicate-iou-thresh", type=float, default=0.35)
    group.add_argument("--duplicate-containment-thresh", type=float, default=0.70)
    group.add_argument("--duplicate-size-ratio", type=float, default=2.2)
    group.add_argument(
        "--enable-post-nms",
        dest="enable_nms",
        action="store_true",
        default=True,
    )
    group.add_argument(
        "--disable-post-nms",
        dest="enable_nms",
        action="store_false",
    )
    group.add_argument("--post-nms-iou", type=float, default=0.35, dest="nms_iou")
    group.add_argument("--enable-defocus-filter", action="store_true")
    group.add_argument("--defocus-max-local-contrast", type=float, default=0.06)
    group.add_argument("--defocus-max-non-background-ratio", type=float, default=0.12)
    group.add_argument("--defocus-max-signal-ratio", type=float, default=0.08)


def process_frame(
    frame_id: int,
    image_path: Path,
    txt_path: Optional[Path],
    args: argparse.Namespace,
    output_dir: Path,
    consolidated_rows: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    image_name = image_path.name
    stem = image_path.stem
    stats = {
        "raw": 0,
        "kept": 0,
        "background_dominated": 0,
        "elongated_box": 0,
        "inside_valid_large_box": 0,
        "poor_large_contains_good_small": 0,
        "overlap_duplicate": 0,
        "nms_duplicate": 0,
        "defocused_low_contrast": 0,
        "others": 0,
    }

    image = Image.open(image_path)
    img_w, img_h = image.size

    boxes: List[Dict[str, Any]] = []
    source_txt = ""

    if txt_path is None and consolidated_rows is not None:
        for row in consolidated_rows:
            row_stem = ""
            for key in ("image_name", "image_path", "filename", "frame"):
                if key in row and row[key]:
                    row_stem = Path(str(row[key])).stem
                    break
            if row_stem == stem:
                try:
                    if args.box_format == "yolo":
                        cx, cy, bw, bh = (
                            float(row["cx"]),
                            float(row["cy"]),
                            float(row["w"]),
                            float(row["h"]),
                        )
                        x1, y1, x2, y2 = yolo_to_xyxy(
                            cx, cy, bw, bh, img_w, img_h, args.normalized
                        )
                    else:
                        x1, y1, x2, y2 = (
                            float(row["x1"]),
                            float(row["y1"]),
                            float(row["x2"]),
                            float(row["y2"]),
                        )
                        if args.normalized:
                            x1, x2 = x1 * img_w, x2 * img_w
                            y1, y2 = y1 * img_h, y2 * img_h
                        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, img_w, img_h)
                    boxes.append(
                        {
                            "frame_id": frame_id,
                            "image_name": image_name,
                            "label": row.get("label", ""),
                            "score": float(row["score"]) if row.get("score") not in ("", None) else None,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "source_line": str(row),
                        }
                    )
                except (KeyError, ValueError, TypeError):
                    pass
        source_txt = str(args.txt_dir)
    elif txt_path is not None:
        source_txt = str(txt_path)
        boxes = parse_detection_file(
            txt_path,
            box_format=args.box_format,
            normalized=args.normalized,
            img_w=img_w,
            img_h=img_h,
            image_name=image_name,
            frame_id=frame_id,
            image_stem=stem,
        )
    else:
        logger.warning("未找到对应 txt: %s", image_name)

    stats["raw"] = len(boxes)

    for idx, box in enumerate(boxes):
        box["_idx"] = idx
        x1, y1, x2, y2 = clip_box_xyxy(
            box["x1"], box["y1"], box["x2"], box["y2"], img_w, img_h
        )
        box["x1"], box["y1"], box["x2"], box["y2"] = x1, y1, x2, y2
        enrich_box_metrics(box, image, args)
        apply_rule1_background(box, args)
        apply_rule2_geometry(box, img_w, img_h, args)
        apply_optional_defocus_filter(box, args)

    apply_rule3_nested(boxes, args)

    if args.enable_duplicate_suppression:
        apply_duplicate_overlap_suppression(boxes, args)

    if args.enable_nms:
        apply_nms(boxes, args.nms_iou)

    kept = [b for b in boxes if b["keep"]]
    stats["kept"] = len(kept)
    for b in boxes:
        if b["keep"]:
            continue
        reason = b.get("remove_reason", "")
        if reason == "background_dominated":
            stats["background_dominated"] += 1
        elif reason == "elongated_box":
            stats["elongated_box"] += 1
        elif reason == "inside_valid_large_box":
            stats["inside_valid_large_box"] += 1
        elif reason == "poor_large_contains_good_small":
            stats["poor_large_contains_good_small"] += 1
        elif reason == "overlap_duplicate":
            stats["overlap_duplicate"] += 1
        elif reason == "nms_duplicate":
            stats["nms_duplicate"] += 1
        elif reason == "defocused_low_contrast":
            stats["defocused_low_contrast"] += 1
        else:
            stats["others"] += 1

    cleaned_txt_dir = output_dir / "cleaned_detections_txt"
    cleaned_vis_dir = output_dir / "cleaned_vis"
    cleaned_txt_dir.mkdir(parents=True, exist_ok=True)
    cleaned_vis_dir.mkdir(parents=True, exist_ok=True)

    save_cleaned_txt(boxes, cleaned_txt_dir / f"{stem}.txt", args.output_with_label)
    draw_visualization(
        image,
        boxes,
        draw_removed=args.draw_removed,
        out_path=cleaned_vis_dir / f"{stem}.png",
    )

    print(
        f"[{frame_id:04d}] {image_name} | raw={stats['raw']} kept={stats['kept']} "
        f"removed: bg={stats['background_dominated']} elongated={stats['elongated_box']} "
        f"nested={stats['inside_valid_large_box']} bad_large={stats['poor_large_contains_good_small']} "
        f"overlap={stats['overlap_duplicate']} nms={stats['nms_duplicate']} "
        f"defocus={stats['defocused_low_contrast']} others={stats['others']}"
    )

    summary_rows = [box_to_summary_row(b, source_txt) for b in boxes]
    return summary_rows, stats


def load_consolidated_file(txt_path: Path) -> List[Dict[str, Any]]:
    suffix = txt_path.suffix.lower()
    if suffix == ".csv":
        with open(txt_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in lines:
        if not line.strip():
            continue
        rows.append({"raw_line": line})
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="清洗图像序列开放词汇检测 txt，过滤背景框、细长框和嵌套重复框。"
    )
    p.add_argument("--image-dir", default=r'test_images', help="图像序列文件夹")
    p.add_argument("--txt-dir", default='inputs/detections_txt', help="检测 txt 文件夹，或单个 txt/csv 文件")
    p.add_argument("--output-dir", default='inputs/postprocessed', help="输出根目录")
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


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    image_dir = Path(args.image_dir)
    txt_dir = Path(args.txt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = parse_image_exts(args.image_exts)
    image_paths = collect_image_paths(image_dir, exts)
    if not image_paths:
        logger.error("图像目录为空: %s", image_dir)
        return

    consolidated_rows: Optional[List[Dict[str, Any]]] = None
    per_frame_mode = True
    if txt_dir.is_file():
        per_frame_mode = False
        consolidated_rows = load_consolidated_file(txt_dir)
        logger.info("使用单文件模式: %s", txt_dir)

    all_summary: List[Dict[str, Any]] = []
    all_cleaned: List[Dict[str, Any]] = []
    total_stats = {
        "raw": 0,
        "kept": 0,
        "background_dominated": 0,
        "elongated_box": 0,
        "inside_valid_large_box": 0,
        "poor_large_contains_good_small": 0,
        "overlap_duplicate": 0,
        "nms_duplicate": 0,
        "defocused_low_contrast": 0,
        "others": 0,
    }

    for frame_id, image_path in enumerate(image_paths):
        txt_path = None
        if per_frame_mode:
            txt_path = find_txt_for_image(image_path, txt_dir)

        summary_rows, stats = process_frame(
            frame_id,
            image_path,
            txt_path,
            args,
            output_dir,
            consolidated_rows=consolidated_rows,
        )
        all_summary.extend(summary_rows)
        for row in summary_rows:
            if int(row["keep"]):
                box = {
                    "frame_id": row["frame_id"],
                    "image_name": row["image_name"],
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "final_quality_score": float(row["final_quality_score"]),
                }
                all_cleaned.append(box_to_cleaned_row(box))

        for k in total_stats:
            total_stats[k] += stats[k]

    summary_fields = [
        "frame_id",
        "image_name",
        "source_txt",
        "source_line",
        "label",
        "raw_score",
        "x1",
        "y1",
        "x2",
        "y2",
        "w",
        "h",
        "area",
        "aspect_ratio",
        "gray_mean",
        "gray_p05",
        "gray_p95",
        "gray_std",
        "local_contrast",
        "signal_ratio",
        "dark_pixel_ratio",
        "bright_pixel_ratio",
        "orange_pixel_ratio",
        "non_background_ratio",
        "final_quality_score",
        "keep",
        "remove_reason",
    ]
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(all_summary)

    cleaned_fields = [
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
        "final_quality_score",
    ]
    cleaned_path = output_dir / "cleaned_all.csv"
    with open(cleaned_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cleaned_fields)
        writer.writeheader()
        writer.writerows(all_cleaned)

    print("\n=== 总计 ===")
    print(
        f"frames={len(image_paths)} raw={total_stats['raw']} kept={total_stats['kept']} "
        f"removed: bg={total_stats['background_dominated']} elongated={total_stats['elongated_box']} "
        f"nested={total_stats['inside_valid_large_box']} bad_large={total_stats['poor_large_contains_good_small']} "
        f"overlap={total_stats['overlap_duplicate']} nms={total_stats['nms_duplicate']} "
        f"defocus={total_stats['defocused_low_contrast']} others={total_stats['others']}"
    )
    print(f"summary.csv -> {summary_path}")
    print(f"cleaned_all.csv -> {cleaned_path}")
    print(f"cleaned txt -> {output_dir / 'cleaned_detections_txt'}")
    print(f"visualizations -> {output_dir / 'cleaned_vis'}")


if __name__ == "__main__":
    main()
