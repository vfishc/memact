"""Recover private camera intrinsics from existing render settings or raw headers."""

import ast
import json
import struct
from pathlib import Path

import cv2
import numpy as np

from routed_facts import file_hash


def video_size(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("missing calibration video")
        return [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    finally:
        capture.release()


def resize_intrinsics(matrix, original_size, output_size):
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or min(*original_size, *output_size) <= 0:
        raise ValueError("invalid calibration")
    return (np.diag([output_size[0] / original_size[0], output_size[1] / original_size[1], 1]) @ matrix).tolist()


def sim_intrinsics(settings, output_size):
    constants = {}
    for node in ast.parse(Path(settings).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    native = [constants.get("RES_X"), constants.get("RES_Y")]
    lens = constants.get("FOCAL_LENGTH")
    if native != list(output_size) or native[0] != native[1] or not isinstance(lens, (float, int)) or lens <= 0:
        raise ValueError("render dimensions mismatch or unsupported display transform")
    focal = lens / 36.0 * native[0]
    matrix = [[focal, 0, native[0] / 2], [0, focal, native[1] / 2], [0, 0, 1]]
    return matrix, {"kind": "archived_sim_batch_settings", "script": str(Path(settings).resolve()),
                    "script_sha256": file_hash(settings), "lens_mm": lens, "native_size": native,
                    "camera_defaults": "36mm sensor, AUTO fit, square pixels, zero shift",
                    "per_episode_job_receipt": False}


def scannet_intrinsics(path, output_size):
    with Path(path).open("rb") as handle:
        if struct.unpack("<I", handle.read(4))[0] != 4:
            raise ValueError("unsupported ScanNet sensor version")
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > 100000:
            raise ValueError("invalid sensor name length")
        handle.read(length)
        matrix = np.frombuffer(handle.read(64), dtype="<f4").reshape(4, 4)[:3, :3]
        handle.read(64 * 3)
        metadata = struct.unpack("<iiIIIIf", handle.read(28))
    return resize_intrinsics(matrix, metadata[2:4], output_size)


def scannetpp_intrinsics(path, source_times, original_size, output_size, source_fps=30, max_time_error=0.05):
    data = json.loads(Path(path).read_text())
    timed = []
    for key, row in data.items():
        try:
            timestamp = float(key)
        except ValueError:
            timestamp = float(key.split("_")[-1]) / source_fps if "frame" in key.lower() else float(key.split("_")[-1])
        timed.append((timestamp, row))
    timed.sort(key=lambda item: item[0])
    if not timed:
        raise ValueError("empty ScanNet++ calibration")
    times = np.array([item[0] for item in timed])
    matrices = []
    for source_time in source_times:
        position = int(np.argmin(abs(times - source_time)))
        if abs(times[position] - source_time) > max_time_error:
            raise ValueError("ScanNet++ timestamp correspondence missing")
        matrices.append(resize_intrinsics(timed[position][1]["intrinsic"], original_size, output_size))
    return matrices


def attach_intrinsics(record, sim_settings=None, raw_root=None):
    source = record["source"]
    resolution = video_size(record["paths"]["video"])
    record["resolution"] = resolution
    if source == "sim_front3d" and sim_settings:
        record["K"], record["intrinsics_provenance"] = sim_intrinsics(sim_settings, resolution)
    elif source in ("real_scannet", "real_scannetpp") and raw_root:
        base_scene = record["scene_id"].split("_part")[0].split("_chunk")[0]
        root = Path(raw_root) / base_scene
        if source == "real_scannet":
            path = root / (base_scene + ".sens")
            record["K"] = scannet_intrinsics(path, resolution)
        else:
            path = root / "iphone" / "pose_intrinsic_imu.json"
            original_size = video_size(root / "iphone" / "rgb.mkv")
            poses = json.loads(Path(record["paths"]["camera_path"]).read_text())
            if any("timestamp" not in pose for pose in poses):
                raise ValueError("original source timestamps missing; do not infer from chunk-local frame numbers")
            matrices = scannetpp_intrinsics(path, [pose["timestamp"] for pose in poses], original_size, resolution)
            for pose, matrix in zip(record["camera"]["pose_lookup"], matrices):
                pose["K"] = matrix
        record["intrinsics_provenance"] = {"kind": "source_calibration_resized", "path": str(path.resolve()), "sha256": file_hash(path)}
    elif source in ("real_scannet", "real_scannetpp") and "K" not in record:
        width, height = resolution
        focal = width * 24 / 36
        record["K"] = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
        record["intrinsics_provenance"] = {"kind": "user_approved_approximation", "decision": "D-026",
                                           "focal_ratio": 24 / 36, "resolution": resolution,
                                           "principal_point": "image_center", "official_calibration": False}
    elif "K" not in record:
        record["intrinsics_provenance"] = {"kind": "unavailable", "source": source}
    return record
