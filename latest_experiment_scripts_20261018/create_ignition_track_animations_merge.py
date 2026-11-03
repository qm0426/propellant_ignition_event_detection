#!/usr/bin/env python3
"""Create grouped annotated GIFs for manually labeled ignition trajectories.

The tracking table is expected to contain one row per track observation with
columns equivalent to

    track_id, frame_id, image_name, x1, y1, x2, y2

The ignition table is expected to contain a track identifier and the ordinal
position of its manually labeled ignition frame within that track. Track
identifiers such as ``6`` and ``track_6`` are treated as the same identifier.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


TRACK_ALIASES = ("track_id", "track", "trajectory_id", "trajectory")
FRAME_ALIASES = ("frame_id", "frame", "frame_index")
IMAGE_ALIASES = ("image_name", "image_path", "frame_path", "file_name")
ONSET_ALIASES = ("ignition_frame", "onset_frame", "gt_frame", "frame_id", "frame")
BBOX_ALIASES = {
    "x1": ("x1", "xmin", "left"),
    "y1": ("y1", "ymin", "top"),
    "x2": ("x2", "xmax", "right"),
    "y2": ("y2", "ymax", "bottom"),
}


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    sequence_position: int
    image_path: Path
    bbox: tuple[float, float, float, float]
    status: str


@dataclass
class EventPlan:
    event_index: int
    track_key: str
    track_display: str
    onset_position: int
    onset_frame_id: int
    frame_records: list[FrameRecord]
    summary: dict[str, Any]


TRACK_COLORS: tuple[tuple[int, int, int, int], ...] = (
    (0, 229, 255, 255),    # cyan
    (255, 0, 255, 255),    # magenta
    (255, 155, 0, 255),    # orange
    (90, 150, 255, 255),   # blue
    (160, 90, 255, 255),   # violet
    (0, 210, 170, 255),    # turquoise
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate grouped annotated GIFs for manually labeled ignition "
            "events using the image names and particle boxes in a tracking table."
        )
    )
    parser.add_argument(
        "--track-table",
        default=(
            'inputs/run_20260703_104159'
            r"\postprocessed\tracking_output\run_20260706_105300"
            r"\track_crop\export_summary.csv"
        ),
        help="Tracking .xlsx/.xls/.csv file",
    )
    parser.add_argument(
        "--ignition-table",
        default='inputs/effective_trace.xlsx',
        help="Manual ignition .xlsx/.xls/.csv file",
    )
    parser.add_argument(
        "--image-root",
        default='inputs/MCFF0004-N001',
        help="Root directory containing the original images",
    )
    parser.add_argument("--output-dir", default="ignition_track_animations", help="Output directory")
    parser.add_argument("--track-sheet", default="0", help="Tracking Excel sheet name or zero-based index")
    parser.add_argument("--ignition-sheet", default="0", help="Ignition Excel sheet name or zero-based index")
    parser.add_argument("--before", type=int, default=10, help="Frames before the GT ignition frame")
    parser.add_argument("--after", type=int, default=10, help="Frames after the GT ignition frame")
    parser.add_argument(
        "--gt-frame-base",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Index base used by frame_id in the manual ignition table: "
            "0 means the first row of a track is position 0; 1 means it is position 1"
        ),
    )
    parser.add_argument("--duration-ms", type=int, default=160, help="Normal GIF frame duration in ms")
    parser.add_argument(
        "--ignition-hold",
        type=float,
        default=4.0,
        help="Multiplier applied to the ignition-frame display duration",
    )
    parser.add_argument(
        "--view",
        choices=("context", "full"),
        default="full",
        help="Use one fixed particle-centered context window or the full frame",
    )
    parser.add_argument("--context-margin", type=int, default=80, help="Pixels around the union of boxes")
    parser.add_argument("--min-context-size", type=int, default=256, help="Minimum width and height of context view")
    parser.add_argument("--max-output-size", type=int, default=1000, help="Maximum GIF width or height; 0 disables resizing")
    parser.add_argument("--box-width", type=int, default=3, help="Normal bounding-box line width")
    parser.add_argument("--ignition-box-width", type=int, default=4, help="Ignition-frame box line width")
    parser.add_argument(
        "--box-padding",
        type=int,
        default=5,
        help="Outward padding around the normal detection box in pixels",
    )
    parser.add_argument(
        "--ignition-box-padding",
        type=int,
        default=10,
        help="Outward padding around the ignition-frame box in pixels",
    )
    parser.add_argument(
        "--group-by",
        choices=("shared-images",),
        default="shared-images",
        help=(
            "Merge events whose pre/post-onset windows share one or more source images; "
            "each connected group is rendered as one GIF without duplicate frames"
        ),
    )
    parser.add_argument(
        "--center-radius",
        type=int,
        default=None,
        help="Deprecated compatibility option; current-frame center markers are no longer drawn",
    )
    parser.add_argument(
        "--ignition-center-radius",
        type=int,
        default=None,
        help="Deprecated compatibility option; ignition-frame center markers are no longer drawn",
    )
    parser.add_argument("--history-radius", type=int, default=4, help="Filled marker radius for earlier non-ignition trajectory positions")
    parser.add_argument(
        "--ignition-history-radius",
        type=int,
        default=8,
        help="Radius of the red ignition marker after the ignition frame becomes part of the history",
    )
    parser.add_argument("--trajectory-width", type=int, default=3, help="Width of the accumulated trajectory line")
    parser.add_argument("--font-size", type=int, default=24, help="Overlay font size")
    parser.add_argument("--font-path", default=None, help="Optional TrueType/OpenType font path")
    parser.add_argument(
        "--tone-percentiles",
        default="0.5,99.7",
        help="Shared low,high percentiles for 16-bit/float image display; use 'none' to clip directly",
    )
    parser.add_argument(
        "--gif-quantizer",
        choices=("maxcoverage", "mediancut"),
        default="maxcoverage",
        help=(
            "GIF palette quantizer. maxcoverage better preserves small saturated "
            "orange/red particles; mediancut may better preserve the dominant background"
        ),
    )
    parser.add_argument(
        "--gif-dither",
        choices=("floyd", "none"),
        default="floyd",
        help="GIF color dithering; floyd reduces 256-color banding",
    )
    parser.add_argument(
        "--allowed-status",
        action="append",
        default=None,
        help="Optional accepted status value; repeat for multiple values. Default: use every valid row.",
    )
    parser.add_argument("--no-loop", action="store_true", help="Play each GIF once instead of looping")
    parser.add_argument("--dry-run", action="store_true", help="Validate tables and list events without reading images")
    return parser


def parse_sheet(value: str) -> str | int:
    text = str(value).strip()
    return int(text) if re.fullmatch(r"\d+", text) else text


def read_table(path: Path, sheet: str | int) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gb18030")
    raise ValueError(f"Unsupported table format: {path}")


def normalized_columns(df: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "_")
        result.setdefault(key, str(column))
    return result


def find_column(df: pd.DataFrame, aliases: Iterable[str], purpose: str) -> str:
    columns = normalized_columns(df)
    for alias in aliases:
        key = alias.strip().lower().replace(" ", "_")
        if key in columns:
            return columns[key]
    raise ValueError(
        f"Missing {purpose} column. Accepted names: {', '.join(aliases)}. "
        f"Available columns: {', '.join(map(str, df.columns))}"
    )


def normalize_track_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    text = str(value).strip()
    numeric = re.fullmatch(r"(?:track[_\- ]*)?(\d+)(?:\.0+)?", text, flags=re.IGNORECASE)
    if numeric:
        return str(int(numeric.group(1)))
    return text.casefold()


def safe_track_label(track_key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", track_key).strip("_")
    return cleaned or "unknown"


def clean_relative_path(value: Any) -> Optional[Path]:
    if pd.isna(value):
        return None
    text = str(value).strip().strip('"')
    if not text:
        return None
    windows = PureWindowsPath(text)
    if windows.drive:
        return Path(text)
    return Path(*windows.parts)


def resolve_image_path(value: Any, image_root: Path, table_dir: Path) -> Optional[Path]:
    relative = clean_relative_path(value)
    if relative is None:
        return None
    candidates: list[Path] = []
    if relative.is_absolute():
        candidates.append(relative)
    candidates.extend((image_root / relative, table_dir / relative))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0] if candidates else None


def numeric_frame(value: Any) -> Optional[int]:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not math.isfinite(float(number)):
        return None
    rounded = int(round(float(number)))
    if not math.isclose(float(number), rounded, abs_tol=1e-6):
        return None
    return rounded


def ordered_bbox(row: pd.Series, columns: dict[str, str]) -> Optional[tuple[float, float, float, float]]:
    values = [pd.to_numeric(pd.Series([row[columns[name]]]), errors="coerce").iloc[0] for name in ("x1", "y1", "x2", "y2")]
    if any(pd.isna(value) or not math.isfinite(float(value)) for value in values):
        return None
    x1, y1, x2, y2 = map(float, values)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def parse_percentiles(value: str) -> Optional[tuple[float, float]]:
    if value.strip().casefold() == "none":
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--tone-percentiles must be 'low,high' or 'none'")
    low, high = map(float, parts)
    if not (0 <= low < high <= 100):
        raise ValueError("--tone-percentiles must satisfy 0 <= low < high <= 100")
    return low, high


def raw_image_array(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def compute_tone_range(paths: Sequence[Path], percentiles: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
    if percentiles is None:
        return None
    samples: list[np.ndarray] = []
    needs_mapping = False
    for path in paths:
        array = raw_image_array(path)
        if array.dtype.itemsize > 1 or np.issubdtype(array.dtype, np.floating):
            needs_mapping = True
        if array.ndim == 3:
            array = array[..., :3].mean(axis=2)
        flat = np.asarray(array).reshape(-1)
        if flat.size > 200_000:
            step = max(1, flat.size // 200_000)
            flat = flat[::step]
        flat = flat[np.isfinite(flat)]
        if flat.size:
            samples.append(flat.astype(np.float64, copy=False))
    if not needs_mapping or not samples:
        return None
    merged = np.concatenate(samples)
    low, high = np.percentile(merged, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.nanmin(merged)), float(np.nanmax(merged))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def load_display_image(path: Path, tone_range: Optional[tuple[float, float]]) -> Image.Image:
    with Image.open(path) as image:
        array = np.asarray(image)
        if tone_range is None or (array.dtype == np.uint8 and not np.issubdtype(array.dtype, np.floating)):
            return image.convert("RGB")
    low, high = tone_range
    values = np.asarray(array, dtype=np.float32)
    values = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    if values.ndim == 2:
        return Image.fromarray(values, mode="L").convert("RGB")
    if values.ndim == 3:
        if values.shape[2] == 1:
            return Image.fromarray(values[..., 0], mode="L").convert("RGB")
        return Image.fromarray(values[..., :3], mode="RGB")
    raise ValueError(f"Unsupported image array shape {values.shape} for {path}")


def load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    candidates = [font_path] if font_path else []
    candidates.extend(
        [
            'inputs/arial.ttf',
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def fixed_context_box(
    records: Sequence[FrameRecord],
    image_size: tuple[int, int],
    margin: int,
    minimum: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left = min(record.bbox[0] for record in records) - margin
    top = min(record.bbox[1] for record in records) - margin
    right = max(record.bbox[2] for record in records) + margin
    bottom = max(record.bbox[3] for record in records) + margin
    target_width = max(float(minimum), right - left)
    target_height = max(float(minimum), bottom - top)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    target_width = min(target_width, float(width))
    target_height = min(target_height, float(height))
    left = max(0.0, min(float(width) - target_width, center_x - target_width / 2.0))
    top = max(0.0, min(float(height) - target_height, center_y - target_height / 2.0))
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(left + target_width)),
        int(math.ceil(top + target_height)),
    )


def draw_label(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int, ignition: bool) -> None:
    padding = 7
    try:
        box = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, stroke_width=0)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
    except AttributeError:
        text_width, text_height = draw.multiline_textsize(text, font=font, spacing=2)
    panel_width = min(width, text_width + 2 * padding)
    panel_height = text_height + 2 * padding
    fill = (190, 0, 0, 220) if ignition else (0, 0, 0, 185)
    draw.rectangle((0, 0, panel_width, panel_height), fill=fill)
    draw.multiline_text(
        (padding, padding),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        spacing=2,
    )


def center_in_context(
    record: FrameRecord,
    offset_x: int,
    offset_y: int,
) -> tuple[float, float]:
    return (
        0.5 * (record.bbox[0] + record.bbox[2]) - offset_x,
        0.5 * (record.bbox[1] + record.bbox[3]) - offset_y,
    )


def render_group_frame(
    image: Image.Image,
    image_path: Path,
    frame_id: int,
    group_index: int,
    group_events: Sequence[EventPlan],
    context: Optional[tuple[int, int, int, int]],
    font: ImageFont.ImageFont,
    box_width: int,
    ignition_box_width: int,
    box_padding: int,
    ignition_box_padding: int,
    history_radius: int,
    ignition_history_radius: int,
    trajectory_width: int,
    max_output_size: int,
) -> tuple[Image.Image, bool]:
    canvas = image.convert("RGBA")
    offset_x = context[0] if context else 0
    offset_y = context[1] if context else 0
    if context:
        canvas = canvas.crop(context)
    draw = ImageDraw.Draw(canvas, mode="RGBA")

    current_key = image_path.resolve()
    ignition_tracks: list[str] = []
    for color_index, event in enumerate(group_events):
        track_color = TRACK_COLORS[color_index % len(TRACK_COLORS)]
        history = [
            record
            for record in event.frame_records
            if record.frame_id < frame_id
        ]
        history_centers = [
            center_in_context(record, offset_x, offset_y)
            for record in history
        ]
        # Only historical positions are marked. The current particle is left
        # unobscured; its enlarged box supplies the current-frame location.
        if len(history_centers) >= 2:
            draw.line(
                history_centers,
                fill=track_color,
                width=max(1, trajectory_width),
                joint="curve",
            )
        for past_record, (past_x, past_y) in zip(history, history_centers):
            is_ignition_marker = past_record.sequence_position == event.onset_position
            marker_radius = ignition_history_radius if is_ignition_marker else history_radius
            draw.ellipse(
                (
                    past_x - marker_radius,
                    past_y - marker_radius,
                    past_x + marker_radius,
                    past_y + marker_radius,
                ),
                fill=(235, 0, 0, 255) if is_ignition_marker else track_color,
                outline=(255, 255, 255, 255) if is_ignition_marker else None,
                width=2 if is_ignition_marker else 1,
            )

        current_records = [
            record
            for record in event.frame_records
            if record.image_path.resolve() == current_key
        ]
        for record in current_records:
            ignition = record.sequence_position == event.onset_position
            if ignition:
                ignition_tracks.append(event.track_display)
            left, top, right, bottom = record.bbox
            left -= offset_x
            right -= offset_x
            top -= offset_y
            bottom -= offset_y
            padding = ignition_box_padding if ignition else box_padding
            draw.rectangle(
                (left - padding, top - padding, right + padding, bottom + padding),
                outline=(255, 35, 35, 255) if ignition else track_color,
                width=ignition_box_width if ignition else box_width,
            )

    ignition_now = bool(ignition_tracks)
    label = (
        f"Group {group_index:03d} | tracks={len(group_events)} | frame={frame_id}\n"
        + (
            "GT ignition: " + ", ".join(ignition_tracks)
            if ignition_now
            else "Filled history; red point = past GT ignition"
        )
    )
    draw_label(draw, label, font, canvas.width, ignition_now)
    if ignition_now:
        border = max(3, ignition_box_width)
        draw.rectangle((1, 1, canvas.width - 2, canvas.height - 2), outline=(255, 0, 0, 255), width=border)

    frame = canvas.convert("RGB")
    if max_output_size > 0 and max(frame.size) > max_output_size:
        scale = max_output_size / max(frame.size)
        target = (max(1, round(frame.width * scale)), max(1, round(frame.height * scale)))
        frame = frame.resize(target, Image.Resampling.LANCZOS)
    return frame, ignition_now


def save_gif(
    frames: Sequence[Image.Image],
    durations: Sequence[int],
    output_path: Path,
    loop: int,
    quantizer: str,
    dither: str,
) -> None:
    if not frames:
        raise ValueError("No rendered frames")
    first_size = frames[0].size
    normalized: list[Image.Image] = []
    for frame in frames:
        if frame.size != first_size:
            padded = Image.new("RGB", first_size, color=(0, 0, 0))
            copy = frame.copy()
            copy.thumbnail(first_size, Image.Resampling.LANCZOS)
            position = ((first_size[0] - copy.width) // 2, (first_size[1] - copy.height) // 2)
            padded.paste(copy, position)
            frame = padded
        normalized.append(frame.convert("RGB"))

    # GIF is limited to 256 colors. Pillow's implicit median-cut conversion
    # allocates most colors to the large gray background and can turn the
    # small orange/red burning particles into flat white or brown patches.
    # Explicit maximum-coverage quantization preserves the full RGB color
    # extent, while Floyd--Steinberg dithering reduces visible banding.
    method = (
        Image.Quantize.MAXCOVERAGE
        if quantizer == "maxcoverage"
        else Image.Quantize.MEDIANCUT
    )
    dither_mode = (
        Image.Dither.FLOYDSTEINBERG
        if dither == "floyd"
        else Image.Dither.NONE
    )
    paletted = [
        frame.quantize(colors=256, method=method, dither=dither_mode)
        for frame in normalized
    ]
    paletted[0].save(
        output_path,
        save_all=True,
        append_images=paletted[1:],
        duration=list(durations),
        loop=loop,
        disposal=2,
        optimize=False,
    )


def prepare_tracking_table(df: pd.DataFrame, image_root: Path, table_dir: Path, allowed_status: Optional[set[str]]) -> tuple[pd.DataFrame, dict[str, str]]:
    columns = {
        "track": find_column(df, TRACK_ALIASES, "track ID"),
        "frame": find_column(df, FRAME_ALIASES, "frame ID"),
        "image": find_column(df, IMAGE_ALIASES, "image name/path"),
    }
    for name, aliases in BBOX_ALIASES.items():
        columns[name] = find_column(df, aliases, f"bounding-box {name}")
    status_column = normalized_columns(df).get("status")
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        track_key = normalize_track_id(row[columns["track"]])
        frame_id = numeric_frame(row[columns["frame"]])
        bbox = ordered_bbox(row, columns)
        status = str(row[status_column]).strip() if status_column and not pd.isna(row[status_column]) else ""
        if not track_key or frame_id is None or bbox is None:
            continue
        if allowed_status is not None and status.casefold() not in allowed_status:
            continue
        image_path = resolve_image_path(row[columns["image"]], image_root, table_dir)
        rows.append(
            {
                "track_key": track_key,
                "frame_id": frame_id,
                "image_path": image_path,
                "bbox": bbox,
                "status": status,
            }
        )
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError("No valid tracking rows remained after column parsing and filtering")
    prepared.sort_values(["track_key", "frame_id"], inplace=True, kind="stable")
    duplicate_count = int(prepared.duplicated(["track_key", "frame_id"]).sum())
    if duplicate_count:
        print(f"[WARN] duplicate track/frame rows={duplicate_count}; keeping the first row", flush=True)
        prepared.drop_duplicates(["track_key", "frame_id"], keep="first", inplace=True)
    return prepared, columns


def prepare_ignition_table(df: pd.DataFrame) -> pd.DataFrame:
    track_column = find_column(df, TRACK_ALIASES, "GT track ID")
    onset_column = find_column(df, ONSET_ALIASES, "GT ignition/onset frame")
    rows: list[dict[str, Any]] = []
    for source_index, row in df.iterrows():
        track_key = normalize_track_id(row[track_column])
        onset_position = numeric_frame(row[onset_column])
        if track_key and onset_position is not None:
            rows.append(
                {
                    "source_row": int(source_index) + 2,
                    "track_key": track_key,
                    "track_display": str(row[track_column]).strip(),
                    "onset_position": onset_position,
                }
            )
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError("No valid manually labeled ignition rows were found")
    return prepared


def write_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "event_index",
        "track_id",
        "annotated_position",
        "position_base",
        "resolved_frame_id",
        "track_length",
        "requested_position_start",
        "requested_position_end",
        "selected_position_start",
        "selected_position_end",
        "tracking_rows_in_window",
        "rendered_frames",
        "missing_image_frame_ids",
        "ignition_frame_rendered",
        "group_id",
        "group_tracks",
        "group_source_images",
        "status",
        "output_file",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_group_summary(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "group_id",
        "track_ids",
        "event_count",
        "source_image_count",
        "first_frame_id",
        "last_frame_id",
        "output_file",
        "status",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_event_plans(
    tracks: pd.DataFrame,
    ignitions: pd.DataFrame,
    args: argparse.Namespace,
    verify_images: bool,
) -> tuple[list[EventPlan], list[dict[str, Any]]]:
    plans: list[EventPlan] = []
    summary: list[dict[str, Any]] = []
    for event_index, event in ignitions.reset_index(drop=True).iterrows():
        track_key = str(event["track_key"])
        track_display = str(event["track_display"])
        annotated_position = int(event["onset_position"])
        requested_start = annotated_position - args.before
        requested_end = annotated_position + args.after
        record: dict[str, Any] = {
            "event_index": event_index + 1,
            "track_id": track_display,
            "annotated_position": annotated_position,
            "position_base": args.gt_frame_base,
            "resolved_frame_id": "",
            "track_length": 0,
            "requested_position_start": requested_start,
            "requested_position_end": requested_end,
            "selected_position_start": "",
            "selected_position_end": "",
            "tracking_rows_in_window": 0,
            "rendered_frames": 0,
            "missing_image_frame_ids": "",
            "ignition_frame_rendered": False,
            "group_id": "",
            "group_tracks": "",
            "group_source_images": 0,
            "status": "failed",
            "output_file": "",
            "message": "",
        }
        try:
            track_rows = (
                tracks[tracks["track_key"] == track_key]
                .sort_values("frame_id", kind="stable")
                .reset_index(drop=True)
            )
            track_length = len(track_rows)
            record["track_length"] = track_length
            if track_length == 0:
                raise ValueError("Track is absent from the tracking table")
            onset_index = annotated_position - args.gt_frame_base
            if onset_index < 0 or onset_index >= track_length:
                first_position = args.gt_frame_base
                last_position = args.gt_frame_base + track_length - 1
                raise ValueError(
                    f"Annotated position {annotated_position} is outside the valid "
                    f"track range {first_position}..{last_position}"
                )

            start_index = max(0, onset_index - args.before)
            end_index = min(track_length - 1, onset_index + args.after)
            track_rows["sequence_index"] = np.arange(track_length, dtype=int)
            selected = track_rows.iloc[start_index : end_index + 1].copy()
            actual_onset_frame = int(track_rows.iloc[onset_index]["frame_id"])
            record["resolved_frame_id"] = actual_onset_frame
            record["selected_position_start"] = start_index + args.gt_frame_base
            record["selected_position_end"] = end_index + args.gt_frame_base
            record["tracking_rows_in_window"] = len(selected)

            frame_records: list[FrameRecord] = []
            missing_images: list[int] = []
            for _, row in selected.iterrows():
                image_path = row["image_path"]
                if image_path is None or (verify_images and not Path(image_path).is_file()):
                    missing_images.append(int(row["frame_id"]))
                    continue
                frame_records.append(
                    FrameRecord(
                        frame_id=int(row["frame_id"]),
                        sequence_position=int(row["sequence_index"]) + args.gt_frame_base,
                        image_path=Path(image_path),
                        bbox=tuple(row["bbox"]),
                        status=str(row["status"]),
                    )
                )
            record["missing_image_frame_ids"] = ";".join(map(str, missing_images))
            if not frame_records:
                raise ValueError("None of the image files in the requested window could be found")
            frame_records.sort(key=lambda item: (item.frame_id, item.sequence_position))
            plans.append(
                EventPlan(
                    event_index=event_index + 1,
                    track_key=track_key,
                    track_display=track_display,
                    onset_position=annotated_position,
                    onset_frame_id=actual_onset_frame,
                    frame_records=frame_records,
                    summary=record,
                )
            )
            record["status"] = "planned"
        except Exception as exc:
            record["message"] = str(exc)
        summary.append(record)
    return plans, summary


def group_event_plans(plans: Sequence[EventPlan]) -> list[list[EventPlan]]:
    """Return connected groups of events whose source-image windows overlap."""
    if not plans:
        return []
    parent = list(range(len(plans)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_owner: dict[Path, int] = {}
    for index, event in enumerate(plans):
        for record in event.frame_records:
            image_key = record.image_path.resolve()
            previous = first_owner.get(image_key)
            if previous is None:
                first_owner[image_key] = index
            else:
                union(index, previous)

    grouped: dict[int, list[EventPlan]] = {}
    for index, event in enumerate(plans):
        grouped.setdefault(find(index), []).append(event)
    return sorted(
        grouped.values(),
        key=lambda group: min(record.frame_id for event in group for record in event.frame_records),
    )


def group_source_images(group: Sequence[EventPlan]) -> list[tuple[int, Path]]:
    images: dict[Path, int] = {}
    for event in group:
        for record in event.frame_records:
            image_path = record.image_path.resolve()
            images[image_path] = min(images.get(image_path, record.frame_id), record.frame_id)
    return sorted(((frame_id, path) for path, frame_id in images.items()), key=lambda item: (item[0], str(item[1])))


def render_group(
    group_index: int,
    group: Sequence[EventPlan],
    output_dir: Path,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
    tone_percentiles: Optional[tuple[float, float]],
) -> dict[str, Any]:
    images = group_source_images(group)
    if not images:
        raise ValueError("The group has no source images")
    frame_ids, image_paths = zip(*images)
    tone_range = compute_tone_range(image_paths, tone_percentiles)
    first_image = load_display_image(image_paths[0], tone_range)
    all_records = [record for event in group for record in event.frame_records]
    context = None
    if args.view == "context":
        context = fixed_context_box(
            all_records,
            first_image.size,
            args.context_margin,
            args.min_context_size,
        )

    frames: list[Image.Image] = []
    durations: list[int] = []
    for frame_id, image_path in images:
        image = load_display_image(image_path, tone_range)
        frame, ignition_now = render_group_frame(
            image=image,
            image_path=image_path,
            frame_id=frame_id,
            group_index=group_index,
            group_events=group,
            context=context,
            font=font,
            box_width=args.box_width,
            ignition_box_width=args.ignition_box_width,
            box_padding=args.box_padding,
            ignition_box_padding=args.ignition_box_padding,
            history_radius=args.history_radius,
            ignition_history_radius=args.ignition_history_radius,
            trajectory_width=args.trajectory_width,
            max_output_size=args.max_output_size,
        )
        frames.append(frame)
        durations.append(
            max(1, round(args.duration_ms * args.ignition_hold))
            if ignition_now
            else args.duration_ms
        )

    track_count = len({event.track_key for event in group})
    output_path = output_dir / (
        f"group_{group_index:03d}_frames_{frame_ids[0]:06d}_{frame_ids[-1]:06d}"
        f"_tracks_{track_count}_events_{len(group)}.gif"
    )
    save_gif(
        frames,
        durations,
        output_path,
        loop=1 if args.no_loop else 0,
        quantizer=args.gif_quantizer,
        dither=args.gif_dither,
    )
    return {
        "group_id": group_index,
        "track_ids": ";".join(event.track_display for event in group),
        "event_count": len(group),
        "source_image_count": len(images),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "output_file": str(output_path),
        "status": "success",
        "message": "",
    }


def run(args: argparse.Namespace) -> int:
    track_path = Path(args.track_table).expanduser().resolve()
    ignition_path = Path(args.ignition_table).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.before < 0 or args.after < 0:
        raise ValueError("--before and --after must be non-negative")
    if args.duration_ms <= 0 or args.ignition_hold <= 0:
        raise ValueError("--duration-ms and --ignition-hold must be positive")
    if args.box_padding < 0 or args.ignition_box_padding < 0:
        raise ValueError("--box-padding and --ignition-box-padding must be non-negative")
    if args.box_width < 1 or args.ignition_box_width < 1 or args.trajectory_width < 1:
        raise ValueError("Box and trajectory line widths must be positive")
    if args.history_radius < 1 or args.ignition_history_radius < 1:
        raise ValueError("--history-radius and --ignition-history-radius must be positive")
    for path, label in ((track_path, "track table"), (ignition_path, "ignition table")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not image_root.is_dir() and not args.dry_run:
        raise FileNotFoundError(f"Missing image root: {image_root}")

    allowed_status = None
    if args.allowed_status:
        allowed_status = {str(value).strip().casefold() for value in args.allowed_status}
    tone_percentiles = parse_percentiles(args.tone_percentiles)
    track_df = read_table(track_path, parse_sheet(args.track_sheet))
    ignition_df = read_table(ignition_path, parse_sheet(args.ignition_sheet))
    tracks, _ = prepare_tracking_table(track_df, image_root, track_path.parent, allowed_status)
    ignitions = prepare_ignition_table(ignition_df)

    available_tracks = set(tracks["track_key"])
    matched_events = int(ignitions["track_key"].isin(available_tracks).sum())
    print(
        f"[INFO] tracking rows={len(tracks)} tracks={len(available_tracks)}; "
        f"GT ignition events={len(ignitions)} matched tracks={matched_events}",
        flush=True,
    )
    print(
        "[INFO] image colors: Pillow RGB input; "
        f"GIF quantizer={args.gif_quantizer}, dither={args.gif_dither}; "
        "trajectory history=enabled",
        flush=True,
    )
    plans, summary = build_event_plans(
        tracks,
        ignitions,
        args,
        verify_images=not args.dry_run,
    )
    groups = group_event_plans(plans)
    print(
        f"[INFO] grouping={args.group_by}; renderable events={len(plans)}; "
        f"shared-image groups={len(groups)}",
        flush=True,
    )
    for group_index, group in enumerate(groups, start=1):
        images = group_source_images(group)
        labels = ", ".join(event.track_display for event in group)
        print(
            f"[GROUP {group_index:03d}] tracks={labels}; events={len(group)}; "
            f"unique source images={len(images)}; frames={images[0][0]}..{images[-1][0]}",
            flush=True,
        )

    if args.dry_run:
        failed = [row for row in summary if row["status"] == "failed"]
        if failed:
            print(f"[WARN] invalid or unmatched events={len(failed)}", flush=True)
        print("[INFO] dry run complete; no image pixels were read and no GIFs were created", flush=True)
        return 0 if not failed else 1

    output_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(args.font_path, args.font_size)
    group_summary: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        try:
            group_record = render_group(
                group_index,
                group,
                output_dir,
                args,
                font,
                tone_percentiles,
            )
            group_summary.append(group_record)
            for event in group:
                row = event.summary
                row["group_id"] = group_index
                row["group_tracks"] = group_record["track_ids"]
                row["group_source_images"] = group_record["source_image_count"]
                row["rendered_frames"] = group_record["source_image_count"]
                row["ignition_frame_rendered"] = any(
                    record.sequence_position == event.onset_position
                    for record in event.frame_records
                )
                row["status"] = "success"
                row["output_file"] = group_record["output_file"]
                if not row["ignition_frame_rendered"]:
                    row["message"] = "The annotated ignition row has no renderable image"
            print(
                f"[OK] group {group_index:03d}/{len(groups)}: events={len(group)} "
                f"frames={group_record['source_image_count']} -> "
                f"{Path(group_record['output_file']).name}",
                flush=True,
            )
        except Exception as exc:
            message = str(exc)
            group_summary.append(
                {
                    "group_id": group_index,
                    "track_ids": ";".join(event.track_display for event in group),
                    "event_count": len(group),
                    "source_image_count": 0,
                    "first_frame_id": "",
                    "last_frame_id": "",
                    "output_file": "",
                    "status": "failed",
                    "message": message,
                }
            )
            for event in group:
                event.summary["group_id"] = group_index
                event.summary["status"] = "failed"
                event.summary["message"] = message
            print(f"[ERROR] group {group_index:03d}: {message}", file=sys.stderr, flush=True)

    summary_path = output_dir / "animation_summary.csv"
    group_summary_path = output_dir / "animation_groups.csv"
    write_summary(summary, summary_path)
    write_group_summary(group_summary, group_summary_path)
    successful_groups = sum(row["status"] == "success" for row in group_summary)
    successful_events = sum(row["status"] == "success" for row in summary)
    visible_ignitions = sum(bool(row["ignition_frame_rendered"]) for row in summary)
    print(
        f"[DONE] groups={successful_groups}/{len(group_summary)}; "
        f"events={successful_events}/{len(summary)}; "
        f"ignition frames={visible_ignitions}/{len(summary)}; "
        f"event summary={summary_path}; group summary={group_summary_path}",
        flush=True,
    )
    return 0 if successful_events == len(summary) and visible_ignitions == len(summary) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return run(args)
    except (ValueError, FileNotFoundError, ImportError, OSError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
