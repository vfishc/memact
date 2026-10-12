#!/usr/bin/env python3
"""Shared helpers for metadata-derived S3 v1 QA generators and validators."""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from common import DEFAULT_FPS, answer_type, recent_frame_ids, stable_id
from camera_geometry import five_way_camera_direction, rotation_geodesic_deg
from metadata_geometry import distance_to_bbox, vector_distance
from qa_observability import QUALITY_POLICY
from qa_semantic_contract import (
    ANSWER_CONTRACT_HASH,
    ANSWER_CONTRACT_VERSION,
    VISUAL_CATEGORY_CONTRACT_HASH,
    VISUAL_CATEGORY_CONTRACT_VERSION,
)
from qa_difficulty_contract import (
    QA_DIFFICULTY_CONTRACT_HASH,
    QA_DIFFICULTY_CONTRACT_VERSION,
    question_wording,
)


SAMPLE_SCHEMA_VERSION = "s3_sample_v1.0"
QA_SEMANTICS_CONTRACT_VERSION = "s3_qa_geometry_semantics_v4"
BALANCED_OPTION_ORDER_POLICY = "s3_balanced_option_order_v1"
SHARD_FILE_RE = re.compile(r"_shard_(\d{5})_of_(\d{5})\.jsonl$")
JsonlEpisodeSpan = Tuple[Path, int, int]


def parse_csv(value: str) -> Set[str]:
    value = value.strip()
    if not value or value.lower() == "all":
        return set()
    return {x.strip() for x in value.split(",") if x.strip()}


def parse_float_csv(value: str) -> List[float]:
    out: List[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def format_seconds(value: float) -> str:
    """Format a duration with the correct singular or plural unit."""

    numeric = float(value)
    unit = "second" if math.isclose(numeric, 1.0, rel_tol=0.0, abs_tol=1e-9) else "seconds"
    return f"{numeric:g} {unit}"


def compact_counts(counter: Counter) -> Dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def balanced_option_order(
    options: Sequence[Any],
    answer: Any,
    *,
    identity_parts: Sequence[Any] = (),
    seed: int = 42,
) -> List[Any]:
    """Return a stable permutation with a balanced correct-answer position.

    This deliberately changes only option order.  The answer value and the set
    of distractors remain untouched, so the generated QA semantics do not
    depend on the balancing policy.  A hash-based position is reproducible
    across shards and does not require a global counter or a second pass.
    """
    values = list(options)
    if len(values) < 2:
        return values
    normalized = [str(value).strip().casefold() for value in values]
    if len(set(normalized)) != len(values):
        return values
    answer_text = str(answer).strip().casefold()
    answer_index = next((index for index, value in enumerate(normalized) if value == answer_text), None)
    if answer_index is None:
        return values
    identity = "\0".join(str(part) for part in identity_parts)

    def digest(*parts: Any) -> bytes:
        payload = "\0".join((str(seed), identity, *(str(part) for part in parts)))
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).digest()

    target_position = int.from_bytes(digest("answer-position"), "big") % len(values)
    distractors = [value for index, value in enumerate(values) if index != answer_index]
    distractors.sort(key=lambda value: (digest("distractor", value), str(value)))
    ordered: List[Any] = []
    distractor_index = 0
    for position in range(len(values)):
        if position == target_position:
            ordered.append(values[answer_index])
        else:
            ordered.append(distractors[distractor_index])
            distractor_index += 1
    return ordered


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_jsonl_episode_span_index(
    paths: Sequence[Path],
    *,
    max_records: int = 0,
) -> Tuple[Dict[str, List[JsonlEpisodeSpan]], Counter]:
    """Index contiguous episode ranges without retaining decoded JSON records."""
    index: Dict[str, List[JsonlEpisodeSpan]] = defaultdict(list)
    counts = Counter()
    stop = False

    def append_span(episode_id: str, path: Path, start: int, end: int) -> None:
        if not episode_id or end <= start:
            return
        index[episode_id].append((path, start, end))
        if len(index[episode_id]) > 1:
            counts["noncontiguous_episode_span"] += 1

    for path in paths:
        current_episode = ""
        span_start = 0
        span_end = 0
        with path.open("rb") as f:
            while True:
                line_start = f.tell()
                line = f.readline()
                line_end = f.tell()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    append_span(current_episode, path, span_start, span_end)
                    current_episode = ""
                    counts["json_decode_error"] += 1
                    continue
                episode_id = str(record.get("episode_id", ""))
                if not episode_id:
                    append_span(current_episode, path, span_start, span_end)
                    current_episode = ""
                    counts["missing_episode_id"] += 1
                    continue
                if current_episode and episode_id != current_episode:
                    append_span(current_episode, path, span_start, span_end)
                    current_episode = ""
                if not current_episode:
                    current_episode = episode_id
                    span_start = line_start
                span_end = line_end
                counts["records_indexed"] += 1
                if max_records and counts["records_indexed"] >= max_records:
                    stop = True
                    break
        if current_episode:
            append_span(current_episode, path, span_start, span_end)
        if stop:
            break
    counts["episodes_indexed"] = len(index)
    counts["files_indexed"] = len(paths)
    return dict(index), counts


def read_jsonl_episode_spans(
    spans: Sequence[JsonlEpisodeSpan],
    *,
    handles: Optional[Dict[Path, BinaryIO]] = None,
) -> Iterator[Dict[str, Any]]:
    for path, start, end in spans:
        close_after = handles is None
        if handles is None:
            handle = path.open("rb")
        else:
            handle = handles.get(path)
            if handle is None:
                handle = path.open("rb")
                handles[path] = handle
        try:
            handle.seek(start)
            while handle.tell() < end:
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    yield json.loads(line)
        finally:
            if close_after:
                handle.close()


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path, *, append: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def cache_files(metadata_cache: Path, splits: Set[str]) -> List[Path]:
    files: List[Path] = []
    if splits:
        for split in sorted(splits):
            files.extend(sorted((metadata_cache / split).glob("episode_metadata_*.jsonl")))
    else:
        files.extend(sorted(metadata_cache.glob("*/episode_metadata_*.jsonl")))
    return files


def iter_cache_records(
    metadata_cache: Path,
    *,
    splits: Set[str],
    sources: Set[str],
    domains: Set[str],
    num_shards: int = 1,
    shard_index: int = 0,
) -> Iterator[Dict[str, Any]]:
    paths, read_mode = cache_files_for_shard(metadata_cache, splits, num_shards, shard_index)
    filtered_index = 0
    for path in paths:
        for rec in read_jsonl(path):
            if sources and str(rec.get("source", "")) not in sources:
                continue
            if domains and str(rec.get("domain", "")) not in domains:
                continue
            if read_mode == "stream_modulo":
                selected = filtered_index % num_shards == shard_index
                filtered_index += 1
                if not selected:
                    continue
            rec = dict(rec)
            rec["_cache_file"] = str(path)
            yield rec


def cache_files_for_shard(
    metadata_cache: Path,
    splits: Set[str],
    num_shards: int,
    shard_index: int,
) -> Tuple[List[Path], str]:
    paths = cache_files(metadata_cache, splits)
    return files_for_shard(paths, num_shards, shard_index)


def files_for_shard(paths: Sequence[Path], num_shards: int, shard_index: int) -> Tuple[List[Path], str]:
    paths = list(paths)
    if num_shards <= 1:
        return paths, "all_files"

    parsed = [SHARD_FILE_RE.search(path.name) for path in paths]
    if paths and all(match and int(match.group(2)) == num_shards for match in parsed):
        selected = [path for path, match in zip(paths, parsed) if int(match.group(1)) == shard_index]
        return selected, "physical_shard_files"
    return paths, "stream_modulo"


def cache_shard_read_mode(metadata_cache: Path, splits: Set[str], num_shards: int, shard_index: int) -> str:
    return cache_files_for_shard(metadata_cache, splits, num_shards, shard_index)[1]


def load_cache_index(
    metadata_cache: Path,
    *,
    paths: Optional[Sequence[Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in paths if paths is not None else cache_files(metadata_cache, set()):
        for rec in read_jsonl(path):
            episode_id = str(rec.get("episode_id", ""))
            if episode_id:
                rec = dict(rec)
                rec["_cache_file"] = str(path)
                index[episode_id] = rec
    return index


def shard_records(records: Sequence[Dict[str, Any]], num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return list(records)
    return [rec for i, rec in enumerate(records) if i % num_shards == shard_index]


def limit_records(records: Sequence[Dict[str, Any]], max_entries: int, max_episodes: int) -> List[Dict[str, Any]]:
    limit = 0
    for value in (max_entries, max_episodes):
        if value and value > 0:
            limit = value if limit <= 0 else min(limit, value)
    return list(records[:limit]) if limit > 0 else list(records)


def limit_record_iterator(records: Iterable[Dict[str, Any]], max_entries: int, max_episodes: int) -> Iterator[Dict[str, Any]]:
    limit = 0
    for value in (max_entries, max_episodes):
        if value and value > 0:
            limit = value if limit <= 0 else min(limit, value)
    return islice(records, limit) if limit > 0 else iter(records)


def output_file(output_dir: Path, prefix: str, split: str, *, num_shards: int, shard_index: int) -> Path:
    if num_shards > 1:
        return output_dir / f"{prefix}_{split}_shard_{shard_index:05d}_of_{num_shards:05d}.jsonl"
    return output_dir / f"{prefix}_{split}.jsonl"


def output_files_for(records: Sequence[Mapping[str, Any]], output_dir: Path, prefix: str, *, num_shards: int, shard_index: int) -> Dict[str, Path]:
    return {
        split: output_file(output_dir, prefix, split, num_shards=num_shards, shard_index=shard_index)
        for split in sorted({str(rec.get("split", "")) for rec in records if rec.get("split")})
    }


def output_files_for_splits(splits: Iterable[str], output_dir: Path, prefix: str, *, num_shards: int, shard_index: int) -> Dict[str, Path]:
    return {
        split: output_file(output_dir, prefix, split, num_shards=num_shards, shard_index=shard_index)
        for split in sorted({str(split) for split in splits if split})
    }


def sample_files(samples_dir: Path, prefix: str, splits: Set[str]) -> List[Path]:
    files: List[Path] = []
    if splits:
        for split in sorted(splits):
            files.extend(sorted(samples_dir.glob(f"{prefix}_{split}*.jsonl")))
    else:
        files.extend(sorted(samples_dir.glob(f"{prefix}_*.jsonl")))
    return files


def load_existing_sample_ids(paths: Iterable[Path]) -> Set[str]:
    seen: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for rec in read_jsonl(path):
            sample_id = str(rec.get("sample_id", ""))
            if sample_id:
                seen.add(sample_id)
    return seen


def clear_output_files(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def frame_ids_with_visibility(record: Mapping[str, Any]) -> List[int]:
    visibility = record.get("visibility", {})
    frames = visibility.get("frames", []) if isinstance(visibility, dict) else []
    out = [int(f["frame_id"]) for f in frames if isinstance(f, dict) and isinstance(f.get("frame_id"), int)]
    camera_count = int((record.get("camera") or {}).get("frame_count") or 0)
    if camera_count > 0:
        out = [x for x in out if 0 <= x < camera_count]
    return sorted(set(out))


def select_query_frames(record: Mapping[str, Any], *, frame_step_sec: float, max_frames: int) -> List[int]:
    frames = frame_ids_with_visibility(record)
    if not frames:
        return []
    fps = float(record.get("fps") or DEFAULT_FPS)
    step = max(1, int(round(frame_step_sec * fps)))
    selected: List[int] = []
    last = -10**12
    for frame_id in frames:
        if frame_id - last >= step:
            selected.append(frame_id)
            last = frame_id
    if not selected:
        selected = [frames[0]]
    if frames[-1] not in selected:
        selected.append(frames[-1])
    if max_frames and len(selected) > max_frames:
        if max_frames == 1:
            return [selected[-1]]
        stride = (len(selected) - 1) / float(max_frames - 1)
        idxs = sorted({round(i * stride) for i in range(max_frames)})
        return [selected[int(i)] for i in idxs]
    return selected


def pose_at(record: Mapping[str, Any], frame_id: int) -> Optional[Mapping[str, Any]]:
    poses = ((record.get("camera") or {}).get("pose_lookup") or [])
    if not isinstance(poses, list) or frame_id < 0 or frame_id >= len(poses):
        return None
    pose = poses[frame_id]
    return pose if isinstance(pose, dict) else None


def camera_location(record: Mapping[str, Any], frame_id: int) -> Optional[List[float]]:
    pose = pose_at(record, frame_id)
    loc = pose.get("location") if pose else None
    if not isinstance(loc, list) or len(loc) < 3:
        return None
    try:
        return [float(loc[0]), float(loc[1]), float(loc[2])]
    except (TypeError, ValueError):
        return None


def camera_yaw(record: Mapping[str, Any], frame_id: int) -> Optional[float]:
    pose = pose_at(record, frame_id)
    if not pose:
        return None
    yaw = pose.get("yaw_deg")
    try:
        value = float(yaw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def visibility_map(record: Mapping[str, Any]) -> Dict[int, Set[str]]:
    visibility = record.get("visibility", {})
    frames = visibility.get("frames", []) if isinstance(visibility, dict) else []
    out: Dict[int, Set[str]] = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), int):
            continue
        out[int(frame["frame_id"])] = {str(x) for x in (frame.get("visible_objects") or [])}
    return out


def visible_objects_at(record: Mapping[str, Any], frame_id: int) -> Set[str]:
    return visibility_map(record).get(int(frame_id), set())


def visible_objects_window(record: Mapping[str, Any], frame_id: int, stable_frames: int) -> Set[str]:
    stable_frames = max(1, int(stable_frames))
    vis = visibility_map(record)
    sets: List[Set[str]] = []
    for offset in range(stable_frames):
        sets.append(vis.get(frame_id - offset, set()))
    if not sets:
        return set()
    out = set(sets[0])
    for items in sets[1:]:
        out &= items
    return out


def objects(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    data = record.get("objects", {})
    return data if isinstance(data, dict) else {}


def rooms(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    data = record.get("rooms", {})
    return data if isinstance(data, dict) else {}


def object_display_names(record: Mapping[str, Any]) -> Dict[str, str]:
    by_category: Dict[str, List[str]] = defaultdict(list)
    for object_id, obj in objects(record).items():
        category = str(obj.get("category") or object_id).replace("_", " ")
        by_category[category].append(str(object_id))
    names: Dict[str, str] = {}
    for category, ids in by_category.items():
        ids = sorted(ids)
        if len(ids) == 1:
            names[ids[0]] = category
        else:
            for idx, object_id in enumerate(ids, 1):
                names[object_id] = f"{category} {idx}"
    return names


def display_object(record: Mapping[str, Any], object_id: str) -> str:
    return object_display_names(record).get(object_id, str(object_id))


def center_of(record: Mapping[str, Any], object_id: str) -> Optional[List[float]]:
    obj = objects(record).get(object_id)
    center = obj.get("world_center") if obj else None
    if not isinstance(center, list) or len(center) < 3:
        return None
    try:
        return [float(center[0]), float(center[1]), float(center[2])]
    except (TypeError, ValueError):
        return None


def bbox_minmax(record: Mapping[str, Any], object_id: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    obj = objects(record).get(object_id)
    if not obj:
        return None, None
    out = []
    for key in ("bbox_min", "bbox_max"):
        value = obj.get(key)
        if not isinstance(value, list) or len(value) < 3:
            out.append(None)
            continue
        try:
            out.append([float(value[0]), float(value[1]), float(value[2])])
        except (TypeError, ValueError):
            out.append(None)
    return out[0], out[1]


def bbox_longest_edge(record: Mapping[str, Any], object_id: str) -> Optional[float]:
    """Return the longest physical/local bbox side for an object."""

    obj = objects(record).get(object_id)
    if not isinstance(obj, Mapping):
        return None
    size = obj.get("physical_bbox_size")
    if not isinstance(size, list) or len(size) < 3:
        return None
    try:
        values = [abs(float(value)) for value in size[:3]]
    except (TypeError, ValueError):
        return None
    longest = max(values, default=0.0)
    return longest if math.isfinite(longest) and longest > 0.0 else None


def object_size_m(record: Mapping[str, Any], object_id: str) -> Optional[float]:
    return bbox_longest_edge(record, object_id)


def camera_to_object_distance(record: Mapping[str, Any], frame_id: int, object_id: str, *, mode: str = "center") -> Optional[float]:
    cam = camera_location(record, frame_id)
    if cam is None:
        return None
    if mode == "center":
        return vector_distance(cam, center_of(record, object_id))
    bmin, bmax = bbox_minmax(record, object_id)
    dist = distance_to_bbox(cam, bmin, bmax)
    if dist is not None:
        return dist
    return vector_distance(cam, center_of(record, object_id))


def _rotation_at(record: Mapping[str, Any], frame_id: int) -> Optional[List[float]]:
    pose = pose_at(record, frame_id)
    rotation = pose.get("rotation_euler_deg") if pose else None
    if not isinstance(rotation, list) or len(rotation) < 3:
        return None
    try:
        values = [float(value) for value in rotation[:3]]
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(value) for value in values) else None


def camera_displacement_m(record: Mapping[str, Any], before_frame: int, after_frame: int) -> Optional[float]:
    if after_frame < before_frame:
        return None
    return vector_distance(camera_location(record, before_frame), camera_location(record, after_frame))


def camera_path_length_m(record: Mapping[str, Any], before_frame: int, after_frame: int) -> Optional[float]:
    if after_frame < before_frame:
        return None
    locations = [camera_location(record, frame_id) for frame_id in range(before_frame, after_frame + 1)]
    if any(location is None for location in locations):
        return None
    total = 0.0
    for first, second in zip(locations, locations[1:]):
        distance = vector_distance(first, second)
        if distance is None:
            return None
        total += distance
    return total


def camera_rotation_deg(record: Mapping[str, Any], before_frame: int, after_frame: int) -> Optional[float]:
    first = _rotation_at(record, before_frame)
    second = _rotation_at(record, after_frame)
    if first is None or second is None:
        return None
    return rotation_geodesic_deg(first, second)


def first_seen_timestamp_s(record: Mapping[str, Any], object_id: str) -> Optional[float]:
    values = object_first_last(record).get(object_id)
    if not isinstance(values, Mapping):
        return None
    first = values.get("first_seen_frame")
    if not isinstance(first, int) or first < 0:
        return None
    fps = float(record.get("fps") or DEFAULT_FPS)
    return first / fps if fps > 0 else None


def object_to_object_distance(record: Mapping[str, Any], a: str, b: str) -> Optional[float]:
    return vector_distance(center_of(record, a), center_of(record, b))


def relative_direction(record: Mapping[str, Any], frame_id: int, object_id: str) -> Optional[str]:
    cam = camera_location(record, frame_id)
    obj = center_of(record, object_id)
    pose = pose_at(record, frame_id)
    rotation = pose.get("rotation_euler_deg") if pose else None
    if cam is None or obj is None or not isinstance(rotation, list):
        return None
    return five_way_camera_direction(cam, rotation, obj, source=str(record.get("source", "")))


def visible_candidates(record: Mapping[str, Any], frame_id: int, *, require_center: bool = False) -> List[str]:
    out = []
    for object_id in sorted(visible_objects_at(record, frame_id)):
        if object_id not in objects(record):
            continue
        if require_center and center_of(record, object_id) is None:
            continue
        out.append(object_id)
    return out


def closest_visible_object(record: Mapping[str, Any], frame_id: int) -> Optional[Tuple[str, float]]:
    candidates = []
    for object_id in visible_candidates(record, frame_id):
        dist = camera_to_object_distance(record, frame_id, object_id)
        if dist is not None:
            candidates.append((object_id, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[1], display_object(record, x[0]), x[0]))
    return candidates[0]


def closest_pair(record: Mapping[str, Any], frame_id: int) -> Optional[Tuple[str, str, float]]:
    ids = visible_candidates(record, frame_id, require_center=True)
    best: Optional[Tuple[str, str, float]] = None
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            dist = object_to_object_distance(record, a, b)
            if dist is None:
                continue
            item = (a, b, dist)
            if best is None or (dist, display_object(record, a), display_object(record, b)) < (
                best[2],
                display_object(record, best[0]),
                display_object(record, best[1]),
            ):
                best = item
    return best


def current_room_id(record: Mapping[str, Any], frame_id: int) -> Optional[str]:
    pose = pose_at(record, frame_id)
    room_id = pose.get("room_id") if pose else None
    if room_id:
        return str(room_id)
    return None


def room_area(record: Mapping[str, Any], room_id: str) -> Optional[float]:
    room = rooms(record).get(room_id)
    if not room:
        return None
    try:
        area = float(room.get("area"))
    except (TypeError, ValueError):
        return None
    return area if math.isfinite(area) else None


def object_first_last(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, int]]:
    visibility = record.get("visibility", {})
    data = visibility.get("object_first_last", {}) if isinstance(visibility, dict) else {}
    return data if isinstance(data, dict) else {}


def prefix_seen_objects(record: Mapping[str, Any], frame_id: int) -> List[str]:
    out = []
    for object_id, times in object_first_last(record).items():
        if not isinstance(times, dict):
            continue
        first = times.get("first_seen_frame")
        if isinstance(first, int) and first <= frame_id and object_id in objects(record):
            out.append(str(object_id))
    return sorted(out)


def first_seen_object(record: Mapping[str, Any], frame_id: int) -> Optional[str]:
    candidates = []
    for object_id, times in object_first_last(record).items():
        if object_id not in objects(record) or not isinstance(times, dict):
            continue
        first = times.get("first_seen_frame")
        if isinstance(first, int) and first <= frame_id:
            candidates.append((first, display_object(record, str(object_id)), str(object_id)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def seen_count_by_category(record: Mapping[str, Any], frame_id: int) -> Counter:
    counts = Counter()
    for object_id in prefix_seen_objects(record, frame_id):
        category = observable_object_category(record, object_id)
        if category:
            counts[category] += 1
    return counts


def choose_count_category(record: Mapping[str, Any], frame_id: int) -> Optional[Tuple[str, int]]:
    counts = seen_count_by_category(record, frame_id)
    if not counts:
        return None
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return str(items[0][0]), int(items[0][1])


def sequence_pair(record: Mapping[str, Any], frame_id: int) -> Optional[Tuple[str, str, str]]:
    seen = []
    for object_id, times in object_first_last(record).items():
        if object_id not in objects(record) or not isinstance(times, dict):
            continue
        first = times.get("first_seen_frame")
        if isinstance(first, int) and first <= frame_id:
            seen.append((first, display_object(record, str(object_id)), str(object_id)))
    if len(seen) < 2:
        return None
    seen.sort()
    if seen[0][0] == seen[-1][0]:
        return None
    return seen[0][2], seen[-1][2], seen[0][2]


def room_visits_so_far(record: Mapping[str, Any], frame_id: int) -> List[Mapping[str, Any]]:
    derived = record.get("derived", {})
    visits = derived.get("room_visit_sequence", []) if isinstance(derived, dict) else []
    out = []
    for visit in visits:
        if not isinstance(visit, dict):
            continue
        start = visit.get("start_frame")
        if isinstance(start, int) and start <= frame_id and visit.get("room_id"):
            out.append(visit)
    out.sort(key=lambda x: int(x.get("start_frame", 0)))
    return out


def first_visited_room(record: Mapping[str, Any], frame_id: int) -> Optional[str]:
    visits = room_visits_so_far(record, frame_id)
    return str(visits[0]["room_id"]) if visits else None


def visited_area_so_far(record: Mapping[str, Any], frame_id: int) -> Optional[float]:
    room_ids = []
    for visit in room_visits_so_far(record, frame_id):
        room_id = str(visit.get("room_id"))
        if room_id not in room_ids:
            room_ids.append(room_id)
    total = 0.0
    used = 0
    for room_id in room_ids:
        area = room_area(record, room_id)
        if area is not None:
            total += area
            used += 1
    return total if used else None


def round_answer(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def required_evidence_for_v1(subset: str, question_type: str) -> List[str]:
    if subset == "S3-Delta":
        return ["metadata_cache.camera.pose_lookup", "metadata_cache.visibility.frames", "window_features"]
    if subset == "S3-History":
        return ["metadata_cache.visibility.object_first_last", "metadata_cache.derived.room_visit_sequence"]
    if subset == "S3-RefChain":
        return ["metadata_cache.visibility.frames", "reference_chain.labels", "metadata_cache.objects"]
    if question_type in {"current_room_area"}:
        return ["metadata_cache.camera.pose_lookup", "metadata_cache.rooms"]
    return ["metadata_cache.camera.pose_lookup", "metadata_cache.visibility.frames", "metadata_cache.objects"]


def make_sample(
    record: Mapping[str, Any],
    *,
    frame_id: int,
    subset: str,
    question_type: str,
    template_id: str,
    question: str,
    answer: Any,
    rule_id: str,
    meta: Mapping[str, Any],
    options: Optional[List[str]] = None,
    recent_window_sec: float = 8.0,
    turns: Optional[List[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    fps = float(record.get("fps") or DEFAULT_FPS)
    sample_identity = (
        record.get("source", ""),
        record.get("split", ""),
        template_id,
        frame_id,
        json.dumps(meta, sort_keys=True, ensure_ascii=False),
    )
    question, wording_family = question_wording(
        question,
        split=str(record.get("split", "")),
        identity=sample_identity,
    )
    canonical_options = balanced_option_order(
        options or [],
        answer,
        identity_parts=sample_identity,
    )
    canonical_turns: Optional[List[Dict[str, Any]]] = None
    if turns is not None:
        canonical_turns = []
        for turn in turns:
            turn_row = dict(turn)
            turn_row["question"], turn_wording_family = question_wording(
                str(turn_row.get("question", "")),
                split=str(record.get("split", "")),
                identity=(*sample_identity, "turn", turn_row.get("turn_id", 0)),
            )
            turn_row["wording_family"] = turn_wording_family
            turn_options = turn_row.get("options", [])
            turn_row["options"] = balanced_option_order(
                turn_options if isinstance(turn_options, list) else [],
                turn_row.get("answer"),
                identity_parts=(*sample_identity, "turn", turn_row.get("turn_id", 0)),
            )
            canonical_turns.append(turn_row)
    atype = answer_type(answer, canonical_options)
    sample_id = stable_id(
        record.get("episode_id"),
        subset,
        template_id,
        frame_id,
        json.dumps(meta, sort_keys=True),
        answer,
        QA_SEMANTICS_CONTRACT_VERSION,
    )
    out = {
        "dataset": "s3_streaming_spatial",
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "episode_id": record.get("episode_id", ""),
        "source": record.get("source", ""),
        "domain": record.get("domain", ""),
        "source_family": record.get("source_family", ""),
        "split": record.get("split", ""),
        "original_split": record.get("original_split", record.get("split", "")),
        "scene_assignment_id": record.get("scene_assignment_id", ""),
        "scene_id": record.get("scene_id", ""),
        "traj_id": record.get("traj_id", ""),
        "path_strat": record.get("path_strat", ""),
        "timestamp": round(frame_id / fps, 4),
        "frame_id": int(frame_id),
        "subset": subset,
        "question_type": question_type,
        "template_id": template_id,
        "legacy_tag": "",
        "question": question,
        "answer_type": atype,
        "recent_frames": recent_frame_ids(frame_id, fps, window_sec=recent_window_sec, max_frames=16),
        "recent_frame_window": {
            "start_frame": max(0, int(round(frame_id - recent_window_sec * fps))),
            "end_frame": int(frame_id),
            "fps": fps,
        },
        "memory_input": {
            "status": "metadata_cache_reference",
            "cache_id": record.get("cache_id", ""),
            "cache_file": record.get("_cache_file", ""),
        },
        "required_evidence": required_evidence_for_v1(subset, question_type),
        "labels": {
            "answer": answer,
            "answer_type": atype,
            "answerable": True,
            "should_act": False,
            "oracle_action": {"type": "answer_now"},
            "evidence_gain": 0.0,
            "source_qa": {
                "template_id": template_id,
                "type": question_type,
                "rule_id": rule_id,
                "meta": dict(meta),
                "options": canonical_options,
            },
        },
        "paths": {
            "metadata_cache": record.get("_cache_file", ""),
            **(record.get("paths", {}) if isinstance(record.get("paths"), dict) else {}),
        },
        "generation": {
            "mode": "metadata_derived_v1",
            "rule_id": rule_id,
            "quality_policy": QUALITY_POLICY,
            "visual_category_contract": VISUAL_CATEGORY_CONTRACT_VERSION,
            "visual_category_contract_hash": VISUAL_CATEGORY_CONTRACT_HASH,
            "answer_contract": ANSWER_CONTRACT_VERSION,
            "answer_contract_hash": ANSWER_CONTRACT_HASH,
            "options_order_policy": BALANCED_OPTION_ORDER_POLICY,
            "qa_semantics_contract": QA_SEMANTICS_CONTRACT_VERSION,
            "qa_difficulty_contract": QA_DIFFICULTY_CONTRACT_VERSION,
            "qa_difficulty_contract_hash": QA_DIFFICULTY_CONTRACT_HASH,
            "question_wording_family": wording_family,
            "metadata_cache_schema": record.get("schema_version", ""),
            "notes": [
                "Answers are deterministically generated from metadata cache for training/auditing.",
                "Inference inputs must not directly read metadata cache internals.",
            ],
        },
    }
    if canonical_options:
        out["options"] = canonical_options
    if canonical_turns is not None:
        out["turns"] = canonical_turns
        out["labels"]["source_qa"]["turns"] = canonical_turns
    return out


def answers_match(expected: Any, actual: Any, *, answer_type_name: str, tolerance: float = 0.05) -> bool:
    if answer_type_name == "number":
        try:
            return abs(float(expected) - float(actual)) <= tolerance
        except (TypeError, ValueError):
            return False
    return str(expected).strip().lower() == str(actual).strip().lower()


def compare_sample_answer(sample: Mapping[str, Any], expected: Any, *, tolerance: float = 0.05) -> bool:
    labels = sample.get("labels", {})
    actual = labels.get("answer") if isinstance(labels, dict) else None
    atype = str(sample.get("answer_type") or (labels.get("answer_type") if isinstance(labels, dict) else ""))
    return answers_match(expected, actual, answer_type_name=atype, tolerance=tolerance)
