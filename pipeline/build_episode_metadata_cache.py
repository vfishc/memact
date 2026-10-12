#!/usr/bin/env python3
"""Build a reusable per-episode metadata cache for S3 v1 generators."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from common import DEFAULT_OUTPUT_ROOT, DEFAULT_ROOT, json_dump, json_load, read_jsonl, stable_id
from metadata_geometry import (
    ARKIT_TO_BLENDER_CONTRACT,
    IDENTITY_COORDINATE_CONTRACT,
    compact_room_visit_sequence,
    finite_float,
    float_list,
    infer_room_id_for_point,
    normalize_polygon,
    physical_bbox_size,
    resolve_object_geometry,
)
from qa_semantic_contract import normalize_semantic_label


SCHEMA_VERSION = "s3_episode_metadata_cache_v1.1"
DEFAULT_V1_ROOT = DEFAULT_OUTPUT_ROOT.parent / "s3_v1"


def parse_csv(value: str) -> Set[str]:
    value = value.strip()
    if not value or value.lower() == "all":
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def clean_category(object_id: str) -> str:
    text = str(object_id).strip().lower()
    if "." in text:
        base, suffix = text.rsplit(".", 1)
        if suffix.isdigit():
            return normalize_semantic_label(base)
    return normalize_semantic_label(text)


def compact_counts(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path, *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def output_file_for(output_dir: Path, split: str, *, num_shards: int, shard_index: int) -> Path:
    return output_dir / split / f"episode_metadata_shard_{shard_index:05d}_of_{num_shards:05d}.jsonl"


def default_summary_path(output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards > 1:
        return output_dir / f"s3_metadata_cache_summary_shard_{shard_index:05d}_of_{num_shards:05d}.json"
    return output_dir / "s3_metadata_cache_summary.json"


def default_task_manifest_path(log_dir: Optional[Path], output_dir: Path, *, num_shards: int, shard_index: int) -> Path:
    base = log_dir or output_dir
    return base / f"metadata_cache_tasks_shard_{shard_index:05d}_of_{num_shards:05d}.jsonl"


def load_split_manifest(path: Optional[Path]) -> Tuple[Dict[str, Mapping[str, Any]], Counter]:
    stats = Counter()
    if not path:
        return {}, stats
    out: Dict[str, Mapping[str, Any]] = {}
    for rec in read_jsonl(path):
        episode_id = str(rec.get("episode_id", ""))
        if not episode_id:
            stats["missing_episode_id"] += 1
            continue
        if episode_id in out:
            stats["duplicate_episode_id"] += 1
        out[episode_id] = rec
    stats["loaded"] = len(out)
    return out, stats


def apply_split_manifest(
    entry: Mapping[str, Any],
    split_manifest: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    out = dict(entry)
    out["original_split"] = entry.get("split", "")
    if not split_manifest:
        out.setdefault("scene_assignment_id", "")
        out.setdefault("split_manifest_version", "")
        return out

    rec = split_manifest.get(str(entry.get("episode_id", "")))
    if not rec:
        return None
    out["split"] = rec.get("s3_split", "")
    out["scene_assignment_id"] = rec.get("scene_assignment_id", "")
    out["split_manifest_version"] = rec.get("split_manifest_version", "")
    out["split_manifest_original_split"] = rec.get("original_split", entry.get("split", ""))
    return out


def resolve_visibility_path(entry: Mapping[str, Any], sim_visibility_root: Optional[Path]) -> Tuple[str, str]:
    registry_path = str(entry.get("visibility_path") or "")
    if entry.get("source") == "sim_front3d" and sim_visibility_root is not None:
        original_split = str(entry.get("original_split") or entry.get("split") or "")
        candidate = sim_visibility_root / original_split / str(entry.get("path_strat", "")) / str(entry.get("scene_id", "")) / "visibility_results.json"
        if candidate.exists():
            return str(candidate), "sim_visibility_root"
    if registry_path and Path(registry_path).exists():
        return registry_path, "registry"
    return "", "missing"


def required_input_status(entry: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "asset_path": bool(entry.get("asset_path")) and Path(str(entry.get("asset_path"))).exists(),
        "room_info_path": bool(entry.get("room_info_path")) and Path(str(entry.get("room_info_path"))).exists(),
        "camera_path": bool(entry.get("camera_path")) and Path(str(entry.get("camera_path"))).exists(),
    }


def load_entries(
    *,
    registry: Path,
    split_manifest: Mapping[str, Mapping[str, Any]],
    splits: Set[str],
    sources: Set[str],
    domains: Set[str],
    sim_visibility_root: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Counter]:
    entries: List[Dict[str, Any]] = []
    stats = Counter()
    for entry in read_jsonl(registry):
        stats["registry_entries"] += 1
        applied = apply_split_manifest(entry, split_manifest)
        if applied is None:
            stats["skipped_no_split_manifest"] += 1
            continue

        if splits and str(applied.get("split", "")) not in splits:
            stats["skipped_split_filter"] += 1
            continue
        if sources and str(applied.get("source", "")) not in sources:
            stats["skipped_source_filter"] += 1
            continue
        if domains and str(applied.get("domain", "")) not in domains:
            stats["skipped_domain_filter"] += 1
            continue

        visibility_path, visibility_source = resolve_visibility_path(applied, sim_visibility_root)
        applied["visibility_path"] = visibility_path
        applied["visibility_source"] = visibility_source
        applied["has_visibility"] = bool(visibility_path)
        entries.append(applied)

    entries.sort(
        key=lambda e: (
            str(e.get("split", "")),
            str(e.get("domain", "")),
            str(e.get("source", "")),
            str(e.get("scene_id", "")),
            str(e.get("path_strat", "")),
        )
    )
    stats["after_filters"] = len(entries)
    return entries, stats


def apply_max_per_source_split(entries: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return list(entries)
    counts = Counter()
    out: List[Dict[str, Any]] = []
    for entry in entries:
        key = (str(entry.get("source", "")), str(entry.get("split", "")))
        if counts[key] >= limit:
            continue
        counts[key] += 1
        out.append(entry)
    return out


def apply_shard(entries: Sequence[Dict[str, Any]], num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return list(entries)
    return [entry for i, entry in enumerate(entries) if i % num_shards == shard_index]


def apply_limits(entries: Sequence[Dict[str, Any]], *, max_entries: int, max_episodes: int) -> List[Dict[str, Any]]:
    limit = 0
    for value in (max_entries, max_episodes):
        if value and value > 0:
            limit = value if limit <= 0 else min(limit, value)
    return list(entries[:limit]) if limit > 0 else list(entries)


def clean_room_info(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json_load(path, default={}) or {}
    rooms: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return rooms
    for room_id, info in data.items():
        if not isinstance(info, dict):
            continue
        area = finite_float(info.get("area"))
        center = float_list(info.get("center"), min_len=2, max_len=3) if info.get("center") is not None else None
        outline = normalize_polygon(info.get("outline"))
        hull = normalize_polygon(info.get("hull_verts"))
        connected = info.get("connected_rooms", [])
        rooms[str(room_id)] = {
            "room_id": str(room_id),
            "area": area,
            "center": center,
            "outline": outline,
            "hull_verts": hull,
            "connected_rooms": [str(x) for x in connected] if isinstance(connected, list) else [],
            "object_ids": [],
        }
    return rooms


def iter_scene_objects(asset_data: Any) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    if not isinstance(asset_data, dict):
        return
    for room_id, objects in asset_data.items():
        if not isinstance(objects, list):
            continue
        for item in objects:
            if not isinstance(item, dict):
                continue
            yield str(room_id), item


def clean_object_info(path: Path, rooms: Dict[str, Dict[str, Any]], *, source: str) -> Tuple[Dict[str, Dict[str, Any]], Counter]:
    data = json_load(path, default={}) or {}
    objects: Dict[str, Dict[str, Any]] = {}
    stats = Counter()
    for room_id, item in iter_scene_objects(data):
        name = str(item.get("name") or item.get("uid") or "").strip()
        if not name:
            stats["skipped_missing_name"] += 1
            continue

        future_type = str(item.get("3D_future_type") or "")
        if future_type and future_type.lower() != "object" and item.get("bbox") is None:
            stats["skipped_non_object"] += 1
            continue

        center, bbox_min, bbox_max, bbox_size, geometry_source = resolve_object_geometry(item, source=source)
        local_bbox_size = physical_bbox_size(item)
        world_center = center
        if not world_center:
            stats["missing_world_center"] += 1
        if not bbox_size:
            stats["missing_bbox"] += 1

        if name in objects:
            stats["duplicate_object_id"] += 1

        category = clean_category(name)
        rec = {
            "object_id": name,
            "uid": str(item.get("uid", "")),
            "category": category,
            "raw_name": str(item.get("name", name)),
            "category_id": item.get("category_id"),
            "coarse_grained_class": item.get("coarse_grained_class"),
            "future_type": future_type,
            "room_id": room_id,
            "world_center": world_center,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_size": bbox_size,
            "physical_bbox_size": local_bbox_size,
            "has_bbox": bool(bbox_min and bbox_max and bbox_size),
            "geometry_source": geometry_source,
            "geometry_coordinate_contract": (
                ARKIT_TO_BLENDER_CONTRACT if source == "real_arkit" and "arkit_to_blender" in geometry_source
                else IDENTITY_COORDINATE_CONTRACT
            ),
            "physical_size_source": "asset_local_bbox" if local_bbox_size else "missing",
            "distance_ref": {
                "center": world_center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            },
        }
        objects[name] = rec
        rooms.setdefault(
            room_id,
            {
                "room_id": room_id,
                "area": None,
                "center": None,
                "outline": [],
                "hull_verts": [],
                "connected_rooms": [],
                "object_ids": [],
            },
        )
        rooms[room_id].setdefault("object_ids", []).append(name)
    for room in rooms.values():
        room["object_ids"] = sorted(set(str(x) for x in room.get("object_ids", [])))
    stats["objects"] = len(objects)
    return objects, stats


def clean_camera_info(path: Path, rooms: Mapping[str, Mapping[str, Any]], fps: float) -> Tuple[Dict[str, Any], Counter]:
    data = json_load(path, default=[]) or []
    stats = Counter()
    pose_lookup: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        stats["camera_not_list"] += 1
        data = []
    for frame_id, item in enumerate(data):
        if not isinstance(item, dict):
            stats["invalid_camera_frame"] += 1
            continue
        location = float_list(item.get("location"), min_len=3, max_len=3)
        rotation = float_list(item.get("rotation_euler_deg"), min_len=3, max_len=3)
        yaw = rotation[-1] if rotation else None
        room_id = infer_room_id_for_point(location, rooms) if location else None
        if location and room_id is None:
            stats["camera_room_unresolved"] += 1
        pose_lookup.append(
            {
                "frame_id": frame_id,
                "name": str(item.get("name") or f"cam_{frame_id:05d}"),
                "location": location,
                "rotation_euler_deg": rotation,
                "yaw_deg": yaw,
                "room_id": room_id,
            }
        )
    room_frames = [(pose["frame_id"], pose.get("room_id")) for pose in pose_lookup]
    return (
        {
            "frame_count": len(data),
            "fps": fps,
            "first_frame_id": 0 if data else None,
            "last_frame_id": len(data) - 1 if data else None,
            "pose_lookup": pose_lookup,
            "room_visit_sequence": compact_room_visit_sequence(room_frames, source="camera_point_in_room"),
            "clamp_rule": "Cache stores camera frames [0, frame_count-1]. Terminal frame_id == frame_count must be clamped explicitly by callers.",
        },
        stats,
    )


def load_visibility(
    path: str,
    objects: Mapping[str, Mapping[str, Any]],
    camera_frame_count: int,
) -> Tuple[Dict[str, Any], Counter]:
    stats = Counter()
    if not path:
        stats["missing_visibility"] += 1
        return (
            {
                "status": "missing",
                "path": "",
                "frame_count": 0,
                "frame_id_min": None,
                "frame_id_max": None,
                "frames": [],
                "unknown_objects": [],
                "object_first_last": {},
                "visible_frame_count_by_object": {},
                "observed_object_ids": [],
            },
            stats,
        )

    data = json_load(Path(path), default=None)
    if not isinstance(data, list):
        stats["invalid_visibility_json"] += 1
        return (
            {
                "status": "invalid",
                "path": path,
                "frame_count": 0,
                "frame_id_min": None,
                "frame_id_max": None,
                "frames": [],
                "unknown_objects": [],
                "object_first_last": {},
                "visible_frame_count_by_object": {},
                "observed_object_ids": [],
            },
            stats,
        )

    frames: List[Dict[str, Any]] = []
    first_last: Dict[str, Dict[str, int]] = {}
    counts = Counter()
    unknown = Counter()
    frame_ids: List[int] = []
    object_ids = set(objects.keys())

    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("frame_id"), int):
            stats["invalid_visibility_frame"] += 1
            continue
        frame_id = int(item["frame_id"])
        frame_ids.append(frame_id)
        if camera_frame_count > 0 and (frame_id < 0 or frame_id >= camera_frame_count):
            stats["frame_id_out_of_range"] += 1
        visible = sorted(set(str(x) for x in (item.get("visible_objects") or []) if str(x)))
        for obj_id in visible:
            if obj_id not in object_ids:
                unknown[obj_id] += 1
                continue
            counts[obj_id] += 1
            if obj_id not in first_last:
                first_last[obj_id] = {"first_seen_frame": frame_id, "last_seen_frame": frame_id}
            else:
                first_last[obj_id]["last_seen_frame"] = frame_id
        frames.append({"frame_id": frame_id, "visible_objects": visible})

    frames.sort(key=lambda x: int(x["frame_id"]))
    frame_ids = sorted(frame_ids)
    status = "ok" if frames else "empty"
    if unknown:
        stats["unknown_visible_objects"] += sum(unknown.values())
    return (
        {
            "status": status,
            "path": path,
            "frame_count": len(frames),
            "frame_id_min": frame_ids[0] if frame_ids else None,
            "frame_id_max": frame_ids[-1] if frame_ids else None,
            "frames": frames,
            "unknown_objects": compact_counts(unknown),
            "object_first_last": {k: v for k, v in sorted(first_last.items())},
            "visible_frame_count_by_object": compact_counts(counts),
            "observed_object_ids": sorted(first_last.keys()),
        },
        stats,
    )


def fallback_room_visit_from_visibility(
    visibility_frames: Sequence[Mapping[str, Any]],
    objects: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    frame_rooms: List[Tuple[int, Optional[str]]] = []
    for frame in visibility_frames:
        frame_id = int(frame.get("frame_id", 0))
        rooms = sorted(
            {
                str(objects[obj_id].get("room_id"))
                for obj_id in frame.get("visible_objects", [])
                if obj_id in objects and objects[obj_id].get("room_id")
            }
        )
        frame_rooms.append((frame_id, rooms[0] if len(rooms) == 1 else None))
    return compact_room_visit_sequence(frame_rooms, source="visible_object_room")


def category_seen_counts_by_prefix(
    visibility_frames: Sequence[Mapping[str, Any]],
    objects: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    final_counts = Counter()
    for frame in visibility_frames:
        frame_id = int(frame.get("frame_id", 0))
        for obj_id in frame.get("visible_objects", []):
            if obj_id in objects:
                seen.add(str(obj_id))
        counts = Counter(str(objects[obj_id].get("category") or obj_id) for obj_id in seen if obj_id in objects)
        final_counts = counts
        out.append(
            {
                "frame_id": frame_id,
                "seen_object_count": len(seen),
                "category_counts": compact_counts(counts),
            }
        )
    return out, compact_counts(final_counts)


def build_record(entry: Mapping[str, Any]) -> Dict[str, Any]:
    required = required_input_status(entry)
    missing_required = [key for key, ok in required.items() if not ok]
    if missing_required:
        return {
            "status": "missing_inputs",
            "episode_id": entry.get("episode_id", ""),
            "missing_inputs": missing_required,
            "entry": dict(entry),
        }

    fps = float(entry.get("fps") or 24.0)
    rooms = clean_room_info(Path(str(entry["room_info_path"])))
    objects, object_stats = clean_object_info(
        Path(str(entry["asset_path"])), rooms, source=str(entry.get("source", ""))
    )
    camera, camera_stats = clean_camera_info(Path(str(entry["camera_path"])), rooms, fps)
    visibility, visibility_stats = load_visibility(str(entry.get("visibility_path") or ""), objects, int(camera["frame_count"]))
    category_prefix, final_category_counts = category_seen_counts_by_prefix(visibility["frames"], objects)

    room_visit_sequence = camera.get("room_visit_sequence") or []

    issues = Counter()
    issues.update({f"object:{k}": v for k, v in object_stats.items() if k != "objects"})
    issues.update({f"camera:{k}": v for k, v in camera_stats.items()})
    issues.update({f"visibility:{k}": v for k, v in visibility_stats.items()})

    record = {
        "dataset": "s3_streaming_spatial",
        "schema_version": SCHEMA_VERSION,
        "cache_id": stable_id(entry.get("episode_id"), SCHEMA_VERSION),
        "episode_id": entry.get("episode_id", ""),
        "domain": entry.get("domain", ""),
        "source": entry.get("source", ""),
        "source_family": entry.get("source_family", ""),
        "split": entry.get("split", ""),
        "original_split": entry.get("original_split", entry.get("split", "")),
        "scene_assignment_id": entry.get("scene_assignment_id", ""),
        "split_manifest_version": entry.get("split_manifest_version", ""),
        "scene_id": entry.get("scene_id", ""),
        "traj_id": entry.get("traj_id", ""),
        "path_strat": entry.get("path_strat", ""),
        "fps": fps,
        "paths": {
            "video": entry.get("video_path", ""),
            "qa": entry.get("qa_path", ""),
            "asset": entry.get("asset_path", ""),
            "room_info": entry.get("room_info_path", ""),
            "camera_path": entry.get("camera_path", ""),
            "segments": entry.get("segments_path", ""),
            "visibility": entry.get("visibility_path", ""),
            "visibility_source": entry.get("visibility_source", ""),
        },
        "camera": camera,
        "objects": objects,
        "rooms": rooms,
        "visibility": visibility,
        "derived": {
            "room_visit_sequence": room_visit_sequence,
            "category_seen_counts_by_prefix": category_prefix,
            "final_category_seen_counts": final_category_counts,
        },
        "counts": {
            "camera_frames": int(camera["frame_count"]),
            "objects": len(objects),
            "rooms": len(rooms),
            "visibility_frames": int(visibility["frame_count"]),
            "observed_objects": len(visibility["observed_object_ids"]),
            "room_visits": len(room_visit_sequence),
        },
        "generation": {
            "status": "ok",
            "notes": [
                "This cache may contain GT metadata for data generation and auditing.",
                "Model inference inputs must not directly read these metadata fields.",
            ],
        },
        "build_issues": compact_counts(issues),
    }
    return {"status": "generated", "record": record}


def load_existing_episode_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_id = str(rec.get("episode_id", ""))
            if episode_id:
                seen.add(episode_id)
    return seen


def task_record(entry: Mapping[str, Any], output_dir: Path, num_shards: int, shard_index: int) -> Dict[str, Any]:
    split = str(entry.get("split", ""))
    return {
        "episode_id": entry.get("episode_id", ""),
        "split": split,
        "domain": entry.get("domain", ""),
        "source": entry.get("source", ""),
        "scene_id": entry.get("scene_id", ""),
        "path_strat": entry.get("path_strat", ""),
        "output_file": str(output_file_for(output_dir, split, num_shards=num_shards, shard_index=shard_index)),
        "visibility_path": entry.get("visibility_path", ""),
        "visibility_source": entry.get("visibility_source", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_OUTPUT_ROOT / "registry" / "s3_registry.jsonl")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_OUTPUT_ROOT / "splits" / "s3_scene_disjoint_episodes.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_V1_ROOT / "metadata_cache")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--task-manifest", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--sim-visibility-root", type=Path, default=None)
    parser.add_argument("--splits", default="all", help="Comma-separated final S3 splits, e.g. train,val,test")
    parser.add_argument("--sources", default="all")
    parser.add_argument("--domains", default="all")
    parser.add_argument("--max-entries", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-per-source-split", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    output_dir = args.output_dir.resolve()
    summary_path = args.summary or default_summary_path(output_dir, num_shards=args.num_shards, shard_index=args.shard_index)
    task_manifest = args.task_manifest or default_task_manifest_path(args.log_dir, output_dir, num_shards=args.num_shards, shard_index=args.shard_index)

    splits = parse_csv(args.splits)
    sources = parse_csv(args.sources)
    domains = parse_csv(args.domains)

    split_manifest, manifest_stats = load_split_manifest(args.split_manifest)
    entries, load_stats = load_entries(
        registry=args.registry,
        split_manifest=split_manifest,
        splits=splits,
        sources=sources,
        domains=domains,
        sim_visibility_root=args.sim_visibility_root,
    )
    after_filter = len(entries)
    entries = apply_max_per_source_split(entries, args.max_per_source_split)
    after_max_per = len(entries)
    sharded = apply_shard(entries, args.num_shards, args.shard_index)
    after_shard = len(sharded)
    selected = apply_limits(sharded, max_entries=args.max_entries, max_episodes=args.max_episodes)

    task_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        (task_record(e, output_dir, args.num_shards, args.shard_index) for e in selected),
        task_manifest,
        append=False,
    )

    output_files = {
        split: output_file_for(output_dir, split, num_shards=args.num_shards, shard_index=args.shard_index)
        for split in sorted({str(e.get("split", "")) for e in selected})
    }
    existing_ids: Dict[str, Set[str]] = {split: set() for split in output_files}
    if args.resume:
        for split, path in output_files.items():
            existing_ids[split] = load_existing_episode_ids(path)
    elif not args.dry_run:
        for path in output_files.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    status_counts = Counter()
    visibility_status_counts = Counter()
    source_split_counts = Counter()
    issue_counts = Counter()
    generated_records = 0
    failure_records: List[Mapping[str, Any]] = []

    to_build: List[Dict[str, Any]] = []
    for entry in selected:
        split = str(entry.get("split", ""))
        episode_id = str(entry.get("episode_id", ""))
        if args.resume and episode_id in existing_ids.get(split, set()):
            status_counts["skipped_existing"] += 1
            continue
        to_build.append(entry)

    if args.dry_run:
        status_counts["planned"] = len(to_build)
    else:
        max_workers = max(1, int(args.workers))
        if max_workers == 1:
            iterable = ((entry, build_record(entry)) for entry in to_build)
            for entry, result in iterable:
                status = str(result.get("status", "unknown"))
                status_counts[status] += 1
                if status == "generated":
                    record = result["record"]
                    split = str(record.get("split", ""))
                    write_jsonl([record], output_files[split], append=True)
                    generated_records += 1
                    visibility_status_counts[str(record.get("visibility", {}).get("status", ""))] += 1
                    source_split_counts[(record.get("domain", ""), record.get("source", ""), split)] += 1
                    for issue, count in record.get("build_issues", {}).items():
                        issue_counts[str(issue)] += int(count)
                else:
                    failure_records.append(result)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_entry = {executor.submit(build_record, entry): entry for entry in to_build}
                for future in as_completed(future_to_entry):
                    result = future.result()
                    status = str(result.get("status", "unknown"))
                    status_counts[status] += 1
                    if status == "generated":
                        record = result["record"]
                        split = str(record.get("split", ""))
                        write_jsonl([record], output_files[split], append=True)
                        generated_records += 1
                        visibility_status_counts[str(record.get("visibility", {}).get("status", ""))] += 1
                        source_split_counts[(record.get("domain", ""), record.get("source", ""), split)] += 1
                        for issue, count in record.get("build_issues", {}).items():
                            issue_counts[str(issue)] += int(count)
                    else:
                        failure_records.append(result)

    failure_path = None
    if failure_records:
        base = args.log_dir or output_dir
        failure_path = base / f"metadata_cache_failures_shard_{args.shard_index:05d}_of_{args.num_shards:05d}.jsonl"
        write_jsonl(failure_records, failure_path, append=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "root": str(args.root),
        "registry": str(args.registry),
        "split_manifest": str(args.split_manifest),
        "output_dir": str(output_dir),
        "summary": str(summary_path),
        "task_manifest": str(task_manifest),
        "log_dir": str(args.log_dir) if args.log_dir else "",
        "sim_visibility_root": str(args.sim_visibility_root) if args.sim_visibility_root else "",
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "seed": int(args.seed),
        "filters": {
            "splits": sorted(splits) if splits else ["all"],
            "sources": sorted(sources) if sources else ["all"],
            "domains": sorted(domains) if domains else ["all"],
            "max_entries": int(args.max_entries),
            "max_episodes": int(args.max_episodes),
            "max_per_source_split": int(args.max_per_source_split),
        },
        "sharding": {
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "total_after_filters": after_filter,
            "total_after_max_per_source_split": after_max_per,
            "total_after_shard": after_shard,
        },
        "workers": int(args.workers),
        "entries_selected": len(selected),
        "entries_to_build": len(to_build),
        "records_generated": generated_records,
        "output_files": {k: str(v) for k, v in sorted(output_files.items())},
        "status_counts": compact_counts(status_counts),
        "visibility_status_counts": compact_counts(visibility_status_counts),
        "source_split_counts": {"|".join(str(x) for x in k): int(v) for k, v in sorted(source_split_counts.items())},
        "issue_counts": compact_counts(issue_counts),
        "failure_path": str(failure_path) if failure_path else "",
        "manifest_stats": compact_counts(manifest_stats),
        "load_stats": compact_counts(load_stats),
        "elapsed_sec": round(time.time() - start, 3),
        "pid": os.getpid(),
        "status": "ok" if not failure_records else "needs_attention",
    }
    json_dump(summary, summary_path)
    print(f"Wrote task manifest to {task_manifest}")
    print(f"Wrote summary to {summary_path}")
    print(summary["status"])


if __name__ == "__main__":
    main()
