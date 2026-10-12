#!/usr/bin/env python3
"""Geometry helpers for episode metadata cache and downstream S3 generators."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple


Point3 = Sequence[float]
ARKIT_TO_BLENDER_CONTRACT = "arkit_asset_xyz_to_blender_x_negz_y_v1"
IDENTITY_COORDINATE_CONTRACT = "source_world_identity_v1"


def finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def float_list(value: Any, *, min_len: int = 0, max_len: Optional[int] = None) -> Optional[List[float]]:
    if not isinstance(value, list):
        return None
    if len(value) < min_len:
        return None
    if max_len is not None and len(value) > max_len:
        value = value[:max_len]
    out: List[float] = []
    for item in value:
        x = finite_float(item)
        if x is None:
            return None
        out.append(x)
    return out


def vector_distance(a: Optional[Point3], b: Optional[Point3]) -> Optional[float]:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    total = 0.0
    for x, y in zip(a, b):
        dx = finite_float(x)
        dy = finite_float(y)
        if dx is None or dy is None:
            return None
        total += (dx - dy) ** 2
    return math.sqrt(total)


def bbox_from_values(
    *,
    center: Optional[Sequence[float]],
    bbox_size: Optional[Sequence[float]],
    bbox_minmax: Optional[Sequence[float]],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """Return center, bbox_min, bbox_max, bbox_size from common S3 bbox encodings."""

    clean_center = float_list(center, min_len=3, max_len=3) if center is not None else None
    clean_size = float_list(bbox_size, min_len=3, max_len=3) if bbox_size is not None else None
    clean_minmax = float_list(bbox_minmax, min_len=6, max_len=6) if bbox_minmax is not None else None

    bbox_min: Optional[List[float]] = None
    bbox_max: Optional[List[float]] = None
    if clean_minmax is not None:
        mins = clean_minmax[:3]
        maxs = clean_minmax[3:6]
        bbox_min = [min(a, b) for a, b in zip(mins, maxs)]
        bbox_max = [max(a, b) for a, b in zip(mins, maxs)]
        clean_size = [bbox_max[i] - bbox_min[i] for i in range(3)]
        clean_center = [(bbox_min[i] + bbox_max[i]) / 2.0 for i in range(3)]
    elif clean_center is not None and clean_size is not None:
        bbox_min = [clean_center[i] - clean_size[i] / 2.0 for i in range(3)]
        bbox_max = [clean_center[i] + clean_size[i] / 2.0 for i in range(3)]

    return clean_center, bbox_min, bbox_max, clean_size


def near_zero_vector(value: Optional[Sequence[float]], *, tolerance: float = 1e-8) -> bool:
    clean = float_list(value, min_len=3, max_len=3) if value is not None else None
    return clean is not None and all(abs(component) <= tolerance for component in clean)


def source_point_to_world(value: Optional[Sequence[float]], *, source: str) -> Optional[List[float]]:
    point = float_list(value, min_len=3, max_len=3) if value is not None else None
    if point is None:
        return None
    if str(source) == "real_arkit":
        return [point[0], -point[2], point[1]]
    return point


def source_aabb_to_world(
    minimum: Optional[Sequence[float]],
    maximum: Optional[Sequence[float]],
    *,
    source: str,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    low = float_list(minimum, min_len=3, max_len=3) if minimum is not None else None
    high = float_list(maximum, min_len=3, max_len=3) if maximum is not None else None
    if low is None or high is None:
        return low, high
    if str(source) == "real_arkit":
        return [low[0], -high[2], low[1]], [high[0], -low[2], high[1]]
    return low, high


def physical_bbox_size(item: Mapping[str, Any]) -> Optional[List[float]]:
    size = float_list(item.get("bbox"), min_len=3, max_len=3)
    if size is None:
        return None
    size = [abs(value) for value in size]
    return size if all(value > 0.0 for value in size) else None


def oriented_bbox_to_world_aabb(
    *,
    center: Optional[Sequence[float]],
    bbox_size: Optional[Sequence[float]],
    rotation_euler: Optional[Sequence[float]],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]]]:
    """Convert an XYZ-Euler OBB in radians to a world-space axis-aligned bbox."""

    clean_center = float_list(center, min_len=3, max_len=3) if center is not None else None
    clean_size = float_list(bbox_size, min_len=3, max_len=3) if bbox_size is not None else None
    if clean_center is None or clean_size is None:
        return clean_center, None, None, clean_size

    clean_size = [abs(component) for component in clean_size]
    rotation = float_list(rotation_euler, min_len=3, max_len=3) if rotation_euler is not None else None
    x_angle, y_angle, z_angle = rotation or [0.0, 0.0, 0.0]
    sx, cx = math.sin(x_angle), math.cos(x_angle)
    sy, cy = math.sin(y_angle), math.cos(y_angle)
    sz, cz = math.sin(z_angle), math.cos(z_angle)
    rotation_matrix = (
        (cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz),
        (cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz),
        (-sy, sx * cy, cx * cy),
    )
    local_half = [component / 2.0 for component in clean_size]
    world_half = [
        sum(abs(rotation_matrix[row][column]) * local_half[column] for column in range(3))
        for row in range(3)
    ]
    bbox_min = [clean_center[index] - world_half[index] for index in range(3)]
    bbox_max = [clean_center[index] + world_half[index] for index in range(3)]
    world_size = [2.0 * component for component in world_half]
    return clean_center, bbox_min, bbox_max, world_size


def resolve_object_geometry(
    item: Mapping[str, Any],
    *,
    source: str = "",
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[List[float]], str]:
    """Resolve mixed asset conventions without treating local placeholders as world geometry."""

    asset_center = float_list(item.get("location"), min_len=3, max_len=3)
    blender_center = float_list(item.get("location_blender"), min_len=3, max_len=3)
    blender_bbox = float_list(item.get("bbox_blender"), min_len=6, max_len=6)

    blender_bbox_center = None
    if blender_bbox is not None:
        blender_bbox_center = [(blender_bbox[index] + blender_bbox[index + 3]) / 2.0 for index in range(3)]
    placeholder_blender_geometry = (
        near_zero_vector(blender_center)
        and asset_center is not None
        and not near_zero_vector(asset_center)
        and (blender_bbox is None or near_zero_vector(blender_bbox_center))
    )

    if blender_bbox is not None and not placeholder_blender_geometry:
        center, bbox_min, bbox_max, bbox_size = bbox_from_values(
            center=blender_center,
            bbox_size=item.get("bbox"),
            bbox_minmax=blender_bbox,
        )
        return center, bbox_min, bbox_max, bbox_size, "blender_world_aabb"

    if blender_center is not None and not placeholder_blender_geometry:
        center, bbox_min, bbox_max, bbox_size = bbox_from_values(
            center=blender_center,
            bbox_size=item.get("bbox"),
            bbox_minmax=None,
        )
        return center, bbox_min, bbox_max, bbox_size, "blender_center_size"

    center, bbox_min, bbox_max, bbox_size = oriented_bbox_to_world_aabb(
        center=asset_center,
        bbox_size=item.get("bbox"),
        rotation_euler=item.get("rotation_euler"),
    )
    geometry_source = "asset_oriented_bbox" if bbox_min is not None else "asset_center_only" if center is not None else "missing"
    center = source_point_to_world(center, source=source)
    bbox_min, bbox_max = source_aabb_to_world(bbox_min, bbox_max, source=source)
    if bbox_min is not None and bbox_max is not None:
        bbox_size = [round(bbox_max[index] - bbox_min[index], 12) for index in range(3)]
    if str(source) == "real_arkit" and geometry_source != "missing":
        geometry_source += "+arkit_to_blender"
    return center, bbox_min, bbox_max, bbox_size, geometry_source


def bbox_closest_point(point: Optional[Point3], bbox_min: Optional[Point3], bbox_max: Optional[Point3]) -> Optional[List[float]]:
    if point is None or bbox_min is None or bbox_max is None:
        return None
    if len(point) < 3 or len(bbox_min) < 3 or len(bbox_max) < 3:
        return None
    out: List[float] = []
    for i in range(3):
        x = finite_float(point[i])
        lo = finite_float(bbox_min[i])
        hi = finite_float(bbox_max[i])
        if x is None or lo is None or hi is None:
            return None
        if lo > hi:
            lo, hi = hi, lo
        out.append(max(lo, min(hi, x)))
    return out


def distance_to_bbox(point: Optional[Point3], bbox_min: Optional[Point3], bbox_max: Optional[Point3]) -> Optional[float]:
    closest = bbox_closest_point(point, bbox_min, bbox_max)
    return vector_distance(point, closest) if closest is not None else None


def normalize_polygon(points: Any) -> List[List[float]]:
    if not isinstance(points, list):
        return []
    out: List[List[float]] = []
    for item in points:
        xy = float_list(item, min_len=2, max_len=2)
        if xy is not None:
            out.append(xy)
    if len(out) >= 2 and out[0] == out[-1]:
        out = out[:-1]
    return out


def point_in_polygon(point_xy: Optional[Sequence[float]], polygon: Any) -> bool:
    point = float_list(point_xy, min_len=2, max_len=2) if point_xy is not None else None
    poly = normalize_polygon(polygon)
    if point is None or len(poly) < 3:
        return False

    x, y = point
    inside = False
    j = len(poly) - 1
    for i, pi in enumerate(poly):
        xi, yi = pi
        xj, yj = poly[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            denom = yj - yi
            if abs(denom) < 1e-12:
                j = i
                continue
            x_at_y = (xj - xi) * (y - yi) / denom + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def infer_room_id_for_point(point: Optional[Sequence[float]], rooms: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    if point is None or len(point) < 2:
        return None
    point_xy = [point[0], point[1]]
    matches = []
    for room_id, room in rooms.items():
        polygon = room.get("outline") or room.get("hull_verts")
        if point_in_polygon(point_xy, polygon):
            matches.append(str(room_id))
    return matches[0] if len(matches) == 1 else None


def compact_room_visit_sequence(frame_rooms: Iterable[Tuple[int, Optional[str]]], *, source: str) -> List[dict]:
    visits: List[dict] = []
    current_room: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None

    def flush() -> None:
        nonlocal current_room, start, end
        if current_room is not None and start is not None and end is not None:
            visits.append(
                {
                    "room_id": current_room,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "frame_count": int(end - start + 1),
                    "source": source,
                }
            )
        current_room = None
        start = None
        end = None

    for frame_id, room_id in frame_rooms:
        if room_id != current_room:
            flush()
            current_room = room_id
            start = int(frame_id) if room_id is not None else None
        if room_id is not None:
            end = int(frame_id)
    flush()
    return visits
