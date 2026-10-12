"""Contracts shared by the Plan 25 transition-first active probe."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


TRANSITION_SCHEMA_VERSION = "s3_plan25_visibility_transition_v1.0"
PIXEL_TRAJECTORY_SCHEMA_VERSION = "s3_plan25_pixel_trajectory_v1.0"
ACTIVE_SCHEMA_VERSION = "s3_plan25_transition_active_v1.0"
CANDIDATE_SCHEMA_VERSION = "s3_plan25_candidate_pose_v1.0"
RENDER_SCHEMA_VERSION = "s3_plan25_pixel_visibility_v1.0"
RENDER_CONTRACT_VERSION = "s3_plan25_pixel_render_v1.1"
QA_CONTRACT_VERSION = "s3_plan25_invariant_answer_v1"
SOURCE_CAMERA_FOCAL_LENGTH_MM = 24.0
CAMERA_SENSOR_WIDTH_MM = 36.0

ACTION_TYPES = (
    "rotate_left_45",
    "rotate_right_45",
    "scan_forward_90",
    "sweep_360",
    "move_forward_d",
    "move_left_d",
    "move_right_d",
)

ACTION_COSTS = {
    "rotate_left_45": 0.55,
    "rotate_right_45": 0.55,
    "scan_forward_90": 0.80,
    "sweep_360": 1.60,
    "move_forward_d": 1.00,
    "move_left_d": 1.10,
    "move_right_d": 1.10,
}

BAD_CATEGORY_RE = re.compile(
    r"(?:^|[._ -])(unknown|other|others|generic|object|instance|thing)(?:$|[._ -])",
    re.IGNORECASE,
)
BAD_CATEGORY_WORDS = {"build element", "recreation", "electronics", "appliance"}


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fov_from_focal_length(focal_length_mm: float, sensor_width_mm: float = CAMERA_SENSOR_WIDTH_MM) -> float:
    """Match the historical Blender camera contract used for sim visibility."""

    focal = float(focal_length_mm)
    sensor = float(sensor_width_mm)
    if not math.isfinite(focal) or not math.isfinite(sensor) or focal <= 0.0 or sensor <= 0.0:
        raise ValueError("focal length and sensor width must be positive finite values")
    return math.degrees(2.0 * math.atan(sensor / (2.0 * focal)))


def stable_id(*parts: Any, size: int = 20) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:size]


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def clean_category(value: Any) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not text or text in BAD_CATEGORY_WORDS or BAD_CATEGORY_RE.search(text):
        return None
    if re.fullmatch(r"\d{3}\s*-\s*.*", text):
        return None
    if re.search(r"(?:^|[._ -])\d{2,}$", text):
        return None
    return text


def object_category(obj: Mapping[str, Any]) -> Optional[str]:
    return clean_category(obj.get("category") or obj.get("raw_name"))


def object_center(obj: Mapping[str, Any]) -> Optional[list[float]]:
    value = obj.get("world_center") or obj.get("location_blender") or obj.get("location")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    out = [finite_float(x) for x in value[:3]]
    return [float(x) for x in out] if all(x is not None for x in out) else None  # type: ignore[arg-type]


def longest_bbox_side(obj: Mapping[str, Any]) -> Optional[float]:
    value = obj.get("bbox_size") or obj.get("size") or obj.get("bbox")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    values = [finite_float(x) for x in value[:3]]
    if any(x is None or x <= 0 for x in values):
        return None
    return float(max(values))  # type: ignore[arg-type]


def visible_ids(frame: Mapping[str, Any]) -> set[str]:
    values = frame.get("visible_objects") or frame.get("visible_object_ids") or []
    output: set[str] = set()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return output
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("object_id") or value.get("object_name")
        if value:
            output.add(str(value))
    return output


def object_pixels(record: Mapping[str, Any], object_id: str) -> int:
    for item in record.get("visible_objects", []) or []:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("object_id") or item.get("object_name") or "") == object_id:
            try:
                return max(0, int(item.get("pixel_count", 0)))
            except (TypeError, ValueError):
                return 0
    return 0


def action_contract_hash(actions: Iterable[str] = ACTION_TYPES) -> str:
    return sha256_json({"version": ACTIVE_SCHEMA_VERSION, "actions": list(actions), "render": RENDER_CONTRACT_VERSION})
