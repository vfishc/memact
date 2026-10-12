"""Bounded four-source RGB projection audit using the existing geometry chain."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT))

from coordinate_alignment import bbox_corners, contact_sheet, pose_by_frame, select_frames, usable_objects, visible_by_frame, world_to_camera
from camera_geometry import euler_xyz_matrix
from metadata_geometry import resolve_object_geometry, source_point_to_world

CONVENTIONS = {
    "real_arkit": (1, -1),
    "real_scannet": (1, 1),
    "real_scannetpp": (1, 1),
    "sim_front3d": (-1, -1),
}
EDGES = [(first, second) for first in range(8) for second in range(first + 1, 8) if (first ^ second) in (1, 2, 4)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_local(point, width, height, forward, image_y, focal_ratio):
    depth = forward * float(point[2])
    if depth < 1e-4:
        return None
    return (width / 2 + focal_ratio * width * float(point[0]) / depth,
            height / 2 + image_y * focal_ratio * width * float(point[1]) / depth)


def clipped_edge(first, second, forward):
    first, second = np.asarray(first, dtype=float).copy(), np.asarray(second, dtype=float).copy()
    depth_first, depth_second = forward * first[2], forward * second[2]
    near = 1e-3
    if depth_first < near and depth_second < near:
        return None
    if depth_first < near:
        first += (second - first) * ((near - depth_first) / (depth_second - depth_first))
    elif depth_second < near:
        second += (first - second) * ((near - depth_second) / (depth_first - depth_second))
    return first, second


def object_corners(cached, raw, source):
    if cached.get("geometry_source", "").startswith("asset_oriented_bbox"):
        matrix = np.asarray(euler_xyz_matrix(np.degrees(raw.get("rotation_euler", [0, 0, 0])).tolist()))
        center = np.asarray(raw["location"], dtype=float)
        half = np.abs(np.asarray(raw["bbox"], dtype=float)) / 2
        corners = [source_point_to_world((center + matrix @ (half * signs)).tolist(), source=source)
                   for signs in itertools.product((-1, 1), repeat=3)]
        return corners, "source_OBB"
    return bbox_corners(cached), "cached_world_AABB"


def run(args):
    from decord import VideoReader, cpu

    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "requires_visual_review", "settings": vars(args).copy(),
              "intrinsics_status": "historical_24mm_36mm_assumption_not_independently_verified",
              "code_hashes": {}, "samples": [], "failures": [], "selected_scenes": {}}
    report["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in report["settings"].items()}
    for name in ("camera_geometry.py", "metadata_geometry.py", "build_episode_metadata_cache.py", "coordinate_alignment.py"):
        report["code_hashes"][name] = digest(PARENT / name)
    report["code_hashes"]["projection_audit.py"] = digest(__file__)
    seen = {source: set() for source in args.sources}
    sheets = {source: [] for source in args.sources}
    for shard in sorted((args.cache / args.split).glob("*.jsonl")):
        with shard.open() as handle:
            for line in handle:
                record = json.loads(line)
                source, scene = record["source"], record["scene_id"]
                if source not in seen or scene in seen[source] or len(seen[source]) >= args.scenes:
                    continue
                try:
                    frame_ids = select_frames(record, args.frames)
                    if not frame_ids:
                        raise ValueError("no usable frames")
                    paths = record["paths"]
                    raw_asset = json.loads(Path(paths["asset"]).read_text())
                    raw_objects = {str(item.get("name") or item.get("uid")): item for values in raw_asset.values()
                                   if isinstance(values, list) for item in values if isinstance(item, dict)}
                    trajectory = json.loads(Path(paths["camera_path"]).read_text())
                    reader = VideoReader(paths["video"], ctx=cpu(0), num_threads=1)
                    poses, visible = pose_by_frame(record), visible_by_frame(record)
                    forward, image_y = CONVENTIONS[source]
                    provenance = {name: digest(paths[name]) for name in ("asset", "camera_path", "visibility")}
                    for frame_id in frame_ids:
                        pose = poses[frame_id]
                        image = Image.fromarray(reader[frame_id].asnumpy()).convert("RGB")
                        overlay = image.copy()
                        draw = ImageDraw.Draw(overlay)
                        width, height = image.size
                        objects = []
                        for object_id in usable_objects(record, visible[frame_id])[:args.objects]:
                            cached, raw = record["objects"][object_id], raw_objects[object_id]
                            resolved = resolve_object_geometry(raw, source=source)
                            geometry_match = all(np.allclose(resolved[index], cached[key], atol=1e-6, rtol=0)
                                                 for index, key in enumerate(("world_center", "bbox_min", "bbox_max")))
                            corners, kind = object_corners(cached, raw, source)
                            local = [world_to_camera(point, pose["location"], pose["rotation_euler_deg"]) for point in corners]
                            color = tuple(70 + value % 186 for value in hashlib.sha256(object_id.encode()).digest()[:3])
                            for first, second in EDGES if len(local) == 8 else []:
                                edge = clipped_edge(local[first], local[second], forward)
                                if edge is not None:
                                    points = [project_local(point, width, height, forward, image_y, args.focal_ratio) for point in edge]
                                    draw.line([tuple(max(-100000, min(100000, value)) for value in point) for point in points], fill=color, width=2)
                            center = project_local(world_to_camera(cached["world_center"], pose["location"], pose["rotation_euler_deg"]), width, height, forward, image_y, args.focal_ratio)
                            if center and 0 <= center[0] < width and 0 <= center[1] < height:
                                draw.ellipse((center[0]-3, center[1]-3, center[0]+3, center[1]+3), fill=color)
                                draw.text(center, f"{cached['category']} [{object_id}]", fill=color, stroke_width=1, stroke_fill="black")
                            objects.append({"id": object_id, "category": cached["category"], "box_kind": kind,
                                            "geometry_match": bool(geometry_match), "center_pixel": center,
                                            "geometry_source": cached["geometry_source"]})
                        pair = Image.new("RGB", (width * 2, height + 30))
                        pair.paste(image, (0, 30)); pair.paste(overlay, (width, 30))
                        ImageDraw.Draw(pair).text((5, 5), f"{source} {scene} frame={frame_id} | historical K assumption | original / projection", fill="white")
                        name = f"{source}_{hashlib.sha256(record['episode_id'].encode()).hexdigest()[:12]}_{frame_id}.jpg"
                        pair.save(args.output / name)
                        sheets[source].append(pair)
                        pose_match = all(np.allclose(pose[key], trajectory[frame_id][key], atol=1e-8, rtol=0) for key in ("location", "rotation_euler_deg"))
                        report["samples"].append({"source": source, "scene": scene, "episode": record["episode_id"],
                            "frame_id": frame_id, "timestamp_s": frame_id / record["fps"], "pose": pose,
                            "pose_matches_raw": bool(pose_match), "video_fps": reader.get_avg_fps(),
                            "cache_fps": record["fps"], "video_frames": len(reader), "trajectory_frames": len(trajectory),
                            "paths": paths, "input_hashes": provenance, "cache_record_hash": hashlib.sha256(line.encode()).hexdigest(),
                            "cache_shard": str(shard), "convention": [forward, image_y], "objects": objects, "overlay": name})
                    seen[source].add(scene)
                except Exception as error:
                    report["failures"].append({"source": source, "scene": scene, "error": repr(error)})
                if all(len(values) >= args.scenes for values in seen.values()):
                    break
        if all(len(values) >= args.scenes for values in seen.values()):
            break
    for source, images in sheets.items():
        if images:
            contact_sheet(images, columns=2, max_size=(960, 400)).save(args.output / f"sheet_{source}.jpg")
    report["selected_scenes"] = {source: sorted(values) for source, values in seen.items()}
    report["coverage_complete"] = all(len(values) == args.scenes for values in seen.values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"coverage_complete": report["coverage_complete"], "frames": len(report["samples"]), "failures": report["failures"]}))
    if not report["coverage_complete"] or report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenes", type=int, default=2)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--objects", type=int, default=8)
    parser.add_argument("--focal-ratio", type=float, default=24 / 36)
    parser.add_argument("--sources", nargs="+", choices=tuple(CONVENTIONS), default=list(CONVENTIONS))
    arguments = parser.parse_args()
    if min(arguments.scenes, arguments.frames, arguments.objects, arguments.focal_ratio) <= 0:
        parser.error("counts and focal ratio must be positive")
    run(arguments)
