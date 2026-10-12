#!/usr/bin/env python3
"""Visually audit camera/object coordinates against decoded RGB frames."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from camera_geometry import euler_xyz_matrix


SOURCES = ("real_arkit", "real_scannet", "real_scannetpp", "sim_front3d")
SKIP_CATEGORIES = {
    "ceiling",
    "floor",
    "other",
    "unknown",
    "wall",
    "window",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--metadata-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes-per-source", type=int, default=5)
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument("--candidate-episodes-per-source", type=int, default=12)
    parser.add_argument("--focal-length-mm", type=float, default=24.0)
    parser.add_argument("--sensor-width-mm", type=float, default=36.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def choose_episode_ids(args: argparse.Namespace) -> tuple[dict[str, list[str]], set[Path]]:
    selected: dict[str, list[str]] = defaultdict(list)
    output_files: set[Path] = set()
    for row in read_jsonl(args.task_manifest):
        source = str(row.get("source", ""))
        if str(row.get("split", "")) != args.split or source not in SOURCES:
            continue
        if len(selected[source]) >= args.candidate_episodes_per_source:
            continue
        selected[source].append(str(row["episode_id"]))
        output_files.add(Path(str(row["output_file"])))
    return selected, output_files


def load_records(
    selected: Mapping[str, Sequence[str]], output_files: Iterable[Path]
) -> dict[str, dict[str, Any]]:
    wanted = {episode_id for values in selected.values() for episode_id in values}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(output_files):
        for row in read_jsonl(path):
            episode_id = str(row.get("episode_id", ""))
            if episode_id in wanted:
                records[episode_id] = row
                if len(records) == len(wanted):
                    return records
    return records


def visible_by_frame(record: Mapping[str, Any]) -> dict[int, list[str]]:
    visibility = record.get("visibility") if isinstance(record.get("visibility"), Mapping) else {}
    return {
        int(row["frame_id"]): [str(value) for value in row.get("visible_objects", [])]
        for row in visibility.get("frames", []) or []
        if isinstance(row, Mapping) and isinstance(row.get("frame_id"), int)
    }


def pose_by_frame(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    camera = record.get("camera") if isinstance(record.get("camera"), Mapping) else {}
    return {
        int(row["frame_id"]): row
        for row in camera.get("pose_lookup", []) or []
        if isinstance(row, Mapping) and isinstance(row.get("frame_id"), int)
    }


def usable_objects(record: Mapping[str, Any], object_ids: Sequence[str]) -> list[str]:
    objects = record.get("objects") if isinstance(record.get("objects"), Mapping) else {}
    usable = []
    for object_id in object_ids:
        item = objects.get(object_id)
        if not isinstance(item, Mapping):
            continue
        category = str(item.get("category", "")).lower()
        if category in SKIP_CATEGORIES or not item.get("world_center"):
            continue
        usable.append(object_id)
    return usable


def select_frames(record: Mapping[str, Any], count: int) -> list[int]:
    visibility = visible_by_frame(record)
    poses = pose_by_frame(record)
    candidates = [
        frame_id
        for frame_id, object_ids in visibility.items()
        if frame_id in poses and usable_objects(record, object_ids)
    ]
    if not candidates:
        return []
    candidates.sort()
    if len(candidates) <= count:
        return candidates
    chosen = []
    for index in range(count):
        position = round((index + 0.5) * len(candidates) / count - 0.5)
        chosen.append(candidates[max(0, min(len(candidates) - 1, position))])
    return sorted(set(chosen))


def world_to_camera(
    point: Sequence[float], location: Sequence[float], rotation: Sequence[float]
) -> np.ndarray:
    matrix = np.asarray(euler_xyz_matrix(rotation), dtype=np.float64)
    return matrix.T @ (np.asarray(point, dtype=np.float64) - np.asarray(location, dtype=np.float64))


def project(
    point: Sequence[float],
    location: Sequence[float],
    rotation: Sequence[float],
    width: int,
    height: int,
    focal_length_mm: float,
    sensor_width_mm: float,
    forward_sign: int,
    image_y_sign: int,
) -> tuple[float, float, float] | None:
    local = world_to_camera(point, location, rotation)
    depth = forward_sign * float(local[2])
    if not math.isfinite(depth) or depth <= 1e-6:
        return None
    focal_pixels = focal_length_mm * width / sensor_width_mm
    return (
        width / 2.0 + focal_pixels * float(local[0]) / depth,
        height / 2.0 + image_y_sign * focal_pixels * float(local[1]) / depth,
        depth,
    )


def bbox_corners(item: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    minimum = item.get("bbox_min")
    maximum = item.get("bbox_max")
    if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
        return []
    try:
        return [
            (float(x_value), float(y_value), float(z_value))
            for x_value in (minimum[0], maximum[0])
            for y_value in (minimum[1], maximum[1])
            for z_value in (minimum[2], maximum[2])
        ]
    except (IndexError, TypeError, ValueError):
        return []


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(
    frame: np.ndarray,
    record: Mapping[str, Any],
    frame_id: int,
    forward_sign: int,
    image_y_sign: int,
    label: str,
    color: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[Image.Image, Counter]:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    pose = pose_by_frame(record)[frame_id]
    location = pose.get("location")
    rotation = pose.get("rotation_euler_deg")
    object_ids = usable_objects(record, visible_by_frame(record).get(frame_id, []))
    objects = record.get("objects") if isinstance(record.get("objects"), Mapping) else {}
    stats = Counter(visible=len(object_ids))
    draw.line((width / 2 - 8, height / 2, width / 2 + 8, height / 2), fill=(255, 220, 40), width=2)
    draw.line((width / 2, height / 2 - 8, width / 2, height / 2 + 8), fill=(255, 220, 40), width=2)
    for object_id in object_ids:
        item = objects[object_id]
        center = project(
            item["world_center"], location, rotation, width, height,
            args.focal_length_mm, args.sensor_width_mm, forward_sign, image_y_sign,
        )
        if center is None:
            stats["behind"] += 1
            continue
        x_value, y_value, _ = center
        if 0 <= x_value < width and 0 <= y_value < height:
            stats["center_in_frame"] += 1
        else:
            stats["center_outside"] += 1
        projected = [
            value
            for corner in bbox_corners(item)
            if (value := project(
                corner, location, rotation, width, height,
                args.focal_length_mm, args.sensor_width_mm, forward_sign, image_y_sign,
            )) is not None
        ]
        if projected:
            x_values = [value[0] for value in projected]
            y_values = [value[1] for value in projected]
            if max(x_values) >= 0 and min(x_values) < width and max(y_values) >= 0 and min(y_values) < height:
                stats["bbox_intersects_frame"] += 1
            else:
                stats["bbox_outside"] += 1
            box = (
                max(0.0, min(x_values)), max(0.0, min(y_values)),
                min(width - 1.0, max(x_values)), min(height - 1.0, max(y_values)),
            )
            if box[0] <= box[2] and box[1] <= box[3]:
                draw.rectangle(box, outline=color, width=2)
        radius = 6
        draw.ellipse((x_value - radius, y_value - radius, x_value + radius, y_value + radius), fill=color, outline=(0, 0, 0), width=1)
        object_label = f"{item.get('category', '')}:{object_id}"
        draw.text((x_value + 8, y_value - 8), object_label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0), font=font(13))
    label = f"{label} | visible={stats['visible']} in={stats['center_in_frame']} behind={stats['behind']}"
    draw.rectangle((0, 0, width, 28), fill=(0, 0, 0))
    draw.text((8, 5), label, fill=color, font=font(15))
    return image, stats


def contact_sheet(
    images: Sequence[Image.Image], columns: int = 3, max_size: tuple[int, int] = (720, 360)
) -> Image.Image:
    thumbnails = []
    for image in images:
        copy = image.copy()
        copy.thumbnail(max_size)
        thumbnails.append(copy)
    width = max(image.width for image in thumbnails)
    height = max(image.height for image in thumbnails)
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * width, rows * height), (25, 25, 25))
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    return sheet




