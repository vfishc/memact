"""Shared camera-coordinate geometry for passive and active S3 supervision."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple


CAMERA_GEOMETRY_CONTRACT_VERSION = "s3_camera_geometry_v1"
DIRECTION_CONTRACT_VERSION = "s3_camera_direction_five_way_v1"
DIRECTION_FRONT_MAX_DEG = 10.0
DIRECTION_SIDE_MIN_DEG = 20.0
DIRECTION_OPTIONS = ("in front", "to the left", "to the right")


def _finite_vector(value: Any, *, length: int = 3) -> Optional[Tuple[float, ...]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < length:
        return None
    try:
        result = tuple(float(component) for component in value[:length])
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(component) for component in result) else None


def euler_xyz_matrix(rotation_euler_deg: Sequence[float]) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """Return Blender-compatible XYZ Euler rotation as a 3x3 matrix."""

    rotation = _finite_vector(rotation_euler_deg)
    if rotation is None:
        return None
    x_angle, y_angle, z_angle = (math.radians(value) for value in rotation)
    sx, cx = math.sin(x_angle), math.cos(x_angle)
    sy, cy = math.sin(y_angle), math.cos(y_angle)
    sz, cz = math.sin(z_angle), math.cos(z_angle)
    return (
        (cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz),
        (cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz),
        (-sy, sx * cy, cx * cy),
    )


def rotate_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def rotation_geodesic_deg(
    first_euler_deg: Sequence[float],
    second_euler_deg: Sequence[float],
) -> Optional[float]:
    """Return the endpoint SO(3) geodesic angle in degrees."""

    first = euler_xyz_matrix(first_euler_deg)
    second = euler_xyz_matrix(second_euler_deg)
    if first is None or second is None:
        return None
    trace = sum(
        sum(float(first[row][column]) * float(second[row][column]) for row in range(3))
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def camera_local_axes(
    rotation_euler_deg: Sequence[float],
    *,
    source: str,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Return world-space (forward, right) for a source camera convention."""

    matrix = euler_xyz_matrix(rotation_euler_deg)
    if matrix is None:
        return None
    forward_local = (0.0, 0.0, -1.0) if str(source) == "sim_front3d" else (0.0, 0.0, 1.0)
    return rotate_vector(matrix, forward_local), rotate_vector(matrix, (1.0, 0.0, 0.0))


def camera_relative_components(
    camera_location: Sequence[float],
    rotation_euler_deg: Sequence[float],
    object_location: Sequence[float],
    *,
    source: str,
) -> Optional[Tuple[float, float]]:
    location = _finite_vector(camera_location)
    target = _finite_vector(object_location)
    axes = camera_local_axes(rotation_euler_deg, source=source)
    if location is None or target is None or axes is None:
        return None
    forward_axis, right_axis = axes
    delta = tuple(target[index] - location[index] for index in range(3))
    forward = sum(delta[index] * forward_axis[index] for index in range(3))
    right = sum(delta[index] * right_axis[index] for index in range(3))
    return forward, right


def three_way_direction_from_components(
    forward: float,
    right: float,
    *,
    front_max_deg: float = DIRECTION_FRONT_MAX_DEG,
    side_min_deg: float = DIRECTION_SIDE_MIN_DEG,
) -> Optional[str]:
    """Classify a target with an explicit front/side gray zone and no behind class."""

    if not math.isfinite(forward) or not math.isfinite(right) or forward <= 0.0:
        return None
    angle_deg = abs(math.degrees(math.atan2(right, forward)))
    if angle_deg <= front_max_deg:
        return "in front"
    if angle_deg < side_min_deg:
        return None
    return "to the right" if right > 0.0 else "to the left"


def three_way_camera_direction(
    camera_location: Sequence[float],
    rotation_euler_deg: Sequence[float],
    object_location: Sequence[float],
    *,
    source: str,
) -> Optional[str]:
    components = camera_relative_components(
        camera_location,
        rotation_euler_deg,
        object_location,
        source=source,
    )
    return three_way_direction_from_components(*components) if components is not None else None


def five_way_direction_from_components(forward: float, right: float) -> Optional[str]:
    if not math.isfinite(forward) or not math.isfinite(right) or forward <= 0.0:
        return None
    angle = math.degrees(math.atan2(right, forward))
    absolute = abs(angle)
    if absolute <= 10.0:
        return "in front"
    if 20.0 <= absolute <= 40.0:
        return "slightly to the right" if angle > 0.0 else "slightly to the left"
    if absolute >= 50.0:
        return "far to the right" if angle > 0.0 else "far to the left"
    return None


def five_way_camera_direction(
    camera_location: Sequence[float],
    rotation_euler_deg: Sequence[float],
    object_location: Sequence[float],
    *,
    source: str,
) -> Optional[str]:
    components = camera_relative_components(
        camera_location,
        rotation_euler_deg,
        object_location,
        source=source,
    )
    return five_way_direction_from_components(*components) if components is not None else None


def horizontal_action_axes(rotation_euler_deg: Sequence[float], *, source: str) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    axes = camera_local_axes(rotation_euler_deg, source=source)
    if axes is None:
        return None

    def horizontal(axis: Sequence[float]) -> Optional[Tuple[float, float, float]]:
        norm = math.hypot(float(axis[0]), float(axis[1]))
        if norm <= 1e-9:
            return None
        return float(axis[0]) / norm, float(axis[1]) / norm, 0.0

    forward = horizontal(axes[0])
    right = horizontal(axes[1])
    return (forward, right) if forward is not None and right is not None else None


def physical_pose_key(
    pose: Mapping[str, Any],
    *,
    renderer_contract: str = "occupancy_blender_visibility:320x240:pixel10",
) -> Optional[Tuple[Any, ...]]:
    """Identify renderer-equivalent poses without action or candidate IDs."""

    location = _finite_vector(pose.get("location"))
    rotation = _finite_vector(pose.get("rotation_euler_deg"))
    scene_id = str(pose.get("scene_id", ""))
    if not scene_id or location is None or rotation is None:
        return None
    try:
        fov = float(pose.get("fov_deg"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fov):
        return None
    return (
        scene_id,
        tuple(round(value, 5) for value in location),
        tuple(round(value % 360.0, 5) for value in rotation),
        round(fov, 5),
        str(renderer_contract),
    )


def physical_pose_hash(pose: Mapping[str, Any], *, renderer_contract: str = "occupancy_blender_visibility:320x240:pixel10") -> str:
    key = physical_pose_key(pose, renderer_contract=renderer_contract)
    if key is None:
        return ""
    payload = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("ascii")).hexdigest()[:24]
