#!/usr/bin/env python3
"""Shared helpers for the S3 streaming spatial data pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from metadata_geometry import resolve_object_geometry


DEFAULT_ROOT = Path(os.environ.get("MEMACT_DATA_ROOT", "data")).resolve()
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "outputs" / "s3_v0"
DEFAULT_FPS = 24.0
REGISTRY_VERSION = "s3_registry_v0.1"
SAMPLE_SCHEMA_VERSION = "s3_sample_v0.1"

SIM_SPLIT_QA_DIR = {
    "train": "train_wo_neg",
    "eval": "eval_wo_neg",
}

REAL_MERGED_SOURCES = {
    "path_strat_arkit": {
        "domain": "real",
        "source": "real_arkit",
        "source_family": "arkit",
        "asset_root": "data/real/arkit/processed_assets",
        "planned_root": "data/real/arkit/planned_path",
        "render_visibility_root": "data/real/arkit/render_output_with_visibility",
        "path_strat": "path_strat_1",
    },
    "path_strat_scannet": {
        "domain": "real",
        "source": "real_scannet",
        "source_family": "scannet",
        "asset_root": "data/real/scannet/processed_assets",
        "planned_root": "data/real/scannet/planned_path",
        "render_visibility_root": "data/real/scannet/render_output_with_visibility",
        "path_strat": "path_strat_1",
    },
    "path_strat_scannetpp": {
        "domain": "real",
        "source": "real_scannetpp",
        "source_family": "scannetpp",
        "asset_root": "data/real/scannetpp/processed_assets",
        "planned_root": "data/real/scannetpp/planned_path",
        "render_visibility_root": "data/real/scannetpp/render_output_with_visibility",
        "path_strat": "path_strat_1",
    },
}

NOW_TYPES = {
    "area",
    "attribute",
    "cam_obj_distance",
    "cam_obj_relative",
    "identification_closest",
    "relative_orientation",
    "spatial_distance",
    "spatial_proximity",
}

DELTA_TYPES = {
    "camera_displacement",
    "camera_motion_target",
}

HISTORY_TYPES = {
    "count",
    "sequence",
    "sequence_identification",
}

REFCHAIN_TYPES = {
    "tem_cam_obj_distance_ref",
    "tem_horizontal_direction",
    "tem_spatial_distance_ref",
}

TAG_TO_SUBSET = {
    "dep_q1_closest_id": "S3-RefChain",
    "dep_q2_rel_dist_ref": "S3-RefChain",
    "dep_q3_first_seen": "S3-History",
    "dep_q4_horizontal_dir": "S3-RefChain",
    "dep_q5_motion_target": "S3-Delta",
    "dep_q6_curr_dist_ref": "S3-RefChain",
    "dep_q7_rel_dir_chain": "S3-RefChain",
    "dep_q8_relay_dist": "S3-RefChain",
    "new_cam_displacement": "S3-Delta",
    "new_cam_obj_abs_dist": "S3-Now",
    "new_cam_obj_rel_dist": "S3-Now",
    "q01_count": "S3-History",
    "q03_sequence": "S3-History",
    "q05_total_area": "S3-History",
    "q06_closest_obj": "S3-Now",
    "q07_orientation": "S3-Now",
}


def json_dump(obj: Any, path: Path, *, indent: Optional[int] = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def json_load(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def rel_to(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_or_empty(path: Path) -> str:
    return str(path) if path else ""


def exists(path_str: str) -> bool:
    return bool(path_str) and Path(path_str).exists()


def stable_id(*parts: Any, size: int = 20) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:size]


def safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def answer_type(answer: Any, options: Any = None) -> str:
    if options:
        return "mc"
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in "ABCDEF":
        return "mc"
    if safe_float(text) is not None:
        return "number"
    if text.startswith("{") or text.startswith("["):
        return "json"
    return "text"


def infer_subset(question_type: str, tag: str) -> str:
    if tag in TAG_TO_SUBSET:
        return TAG_TO_SUBSET[tag]
    if question_type in DELTA_TYPES:
        return "S3-Delta"
    if question_type in HISTORY_TYPES:
        return "S3-History"
    if question_type in REFCHAIN_TYPES:
        return "S3-RefChain"
    if question_type in NOW_TYPES:
        return "S3-Now"
    if question_type.startswith("tem_") or tag.startswith("dep_"):
        return "S3-RefChain"
    return "S3-Now"


def required_evidence_for(subset: str, question_type: str) -> List[str]:
    if subset == "S3-Delta":
        return ["camera_pose_window", "object_visibility_window", "object_3d_metadata"]
    if subset == "S3-History":
        return ["visibility_history_prefix", "camera_trace_prefix", "room_or_object_metadata"]
    if subset == "S3-RefChain":
        return ["previous_answer_reference", "visibility_history_prefix", "current_camera_pose"]
    if question_type == "area":
        return ["current_room_id", "room_geometry"]
    return ["current_camera_pose", "visible_or_remembered_objects", "object_3d_metadata"]


def recent_frame_ids(frame_id: int, fps: float, window_sec: float = 8.0, max_frames: int = 16) -> List[int]:
    start = max(0, int(round(frame_id - window_sec * fps)))
    end = max(0, int(frame_id))
    if end <= start:
        return [end]
    span = end - start
    step = max(1, int(math.ceil(span / max(1, max_frames - 1))))
    frames = list(range(start, end + 1, step))
    if frames[-1] != end:
        frames.append(end)
    return frames[-max_frames:]


def summarize_counter(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def nested_count(records: Iterable[Mapping[str, Any]], keys: Tuple[str, ...]) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for rec in records:
        node = root
        for key in keys[:-1]:
            val = str(rec.get(key, ""))
            node = node.setdefault(val, {})
        leaf = str(rec.get(keys[-1], ""))
        node[leaf] = int(node.get(leaf, 0)) + 1
    return root


def count_by(records: Iterable[Mapping[str, Any]], *keys: str) -> Dict[str, int]:
    c = Counter()
    for rec in records:
        c["|".join(str(rec.get(k, "")) for k in keys)] += 1
    return summarize_counter(c)


def scene_object_metadata(asset_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json_load(asset_path, default={}) or {}
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    for room_id, objects in data.items():
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_id = str(obj.get("name") or obj.get("uid") or "")
            if not obj_id:
                continue
            center, bbox_min, bbox_max, bbox_size, geometry_source = resolve_object_geometry(obj)
            out[obj_id] = {
                "slot_id": f"obj_{stable_id(obj_id, size=10)}",
                "object_id": obj_id,
                "category": str(obj.get("name", obj_id)).lower(),
                "room_id": room_id,
                "world_center": center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_size": bbox_size,
                "geometry_source": geometry_source,
            }
    return out


def room_metadata(room_info_path: Path) -> Dict[str, Dict[str, Any]]:
    data = json_load(room_info_path, default={}) or {}
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return out
    for room_id, info in data.items():
        if not isinstance(info, dict):
            continue
        out[room_id] = {
            "room_id": room_id,
            "area": info.get("area"),
            "center": info.get("center"),
            "connected_rooms": info.get("connected_rooms", []),
        }
    return out


def visibility_prefix(visibility_path: Path, frame_id: int) -> Dict[str, Dict[str, int]]:
    data = json_load(visibility_path, default=[]) or []
    seen: Dict[str, Dict[str, int]] = {}
    if not isinstance(data, list):
        return seen
    for item in data:
        if not isinstance(item, dict):
            continue
        fid = item.get("frame_id")
        if not isinstance(fid, int) or fid > frame_id:
            continue
        for obj in item.get("visible_objects", []) or []:
            obj_id = str(obj)
            rec = seen.setdefault(obj_id, {"first_seen_frame": fid, "last_seen_frame": fid})
            rec["last_seen_frame"] = fid
    return seen


def light_memory_input(entry: Mapping[str, Any], frame_id: int, *, max_object_slots: int = 20) -> Dict[str, Any]:
    visibility_path = Path(entry["visibility_path"]) if entry.get("visibility_path") else None
    if not visibility_path or not visibility_path.exists():
        return {"object_slots": [], "room_slots": [], "event_slots": [], "status": "no_visibility"}

    object_meta = scene_object_metadata(Path(entry["asset_path"])) if entry.get("asset_path") else {}
    room_meta = room_metadata(Path(entry["room_info_path"])) if entry.get("room_info_path") else {}
    seen = visibility_prefix(visibility_path, frame_id)

    slots = []
    for obj_id, times in sorted(seen.items(), key=lambda kv: kv[1]["last_seen_frame"], reverse=True):
        meta = object_meta.get(obj_id, {})
        slots.append({
            "slot_id": meta.get("slot_id", f"obj_{stable_id(obj_id, size=10)}"),
            "object_id": obj_id,
            "category": meta.get("category", obj_id),
            "first_seen_t": round(times["first_seen_frame"] / DEFAULT_FPS, 3),
            "last_seen_t": round(times["last_seen_frame"] / DEFAULT_FPS, 3),
            "best_frame_t": round(times["last_seen_frame"] / DEFAULT_FPS, 3),
            "room_id": meta.get("room_id"),
            "world_center": meta.get("world_center"),
            "bbox_size": meta.get("bbox_size"),
            "visibility_score": 1.0,
            "confidence": 1.0,
        })
        if len(slots) >= max_object_slots:
            break

    room_slots = []
    rooms = {s.get("room_id") for s in slots if s.get("room_id")}
    for room_id in sorted(rooms):
        if room_id in room_meta:
            room_slots.append(room_meta[room_id])

    events = []
    for slot in slots[:10]:
        events.append({
            "event_type": "object_seen",
            "time": slot["last_seen_t"],
            "object_slot": slot["slot_id"],
        })

    return {
        "object_slots": slots,
        "room_slots": room_slots,
        "event_slots": events,
        "status": "light_from_visibility",
    }


def add_counter(summary: Dict[str, Any], name: str, key: Tuple[Any, ...]) -> None:
    node = summary.setdefault(name, {})
    for part in key[:-1]:
        node = node.setdefault(str(part), {})
    leaf = str(key[-1])
    node[leaf] = int(node.get(leaf, 0)) + 1
