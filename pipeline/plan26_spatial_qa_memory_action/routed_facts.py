"""Private, sampled episode facts; never pass this object to the runtime router."""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_geometry import euler_xyz_matrix, five_way_camera_direction, rotation_geodesic_deg
from build_episode_metadata_cache import clean_camera_info, clean_object_info, clean_room_info, load_visibility
from qa_semantic_contract import visual_category


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def rebuild_record(record, visibility_path=None):
    if record["source"] == "real_arkit" and record.get("coordinate_version") != "arkit_official_native_v2":
        raise ValueError("Legacy ARKit cache import disabled; use checked official normalized facts")
    paths = dict(record["paths"])
    if visibility_path:
        paths["visibility"] = str(visibility_path)
    rooms = clean_room_info(Path(paths["room_info"]))
    objects, object_stats = clean_object_info(Path(paths["asset"]), rooms, source=record["source"])
    camera, camera_stats = clean_camera_info(Path(paths["camera_path"]), rooms, float(record["fps"]))
    if record["source"] == "real_arkit":
        original_poses = json.loads(Path(paths["camera_path"]).read_text())
        for pose, original in zip(camera["pose_lookup"], original_poses):
            pose["K"] = original["intrinsics"]
    visibility, visibility_stats = load_visibility(paths["visibility"], objects, camera["frame_count"])
    if visibility["status"] != "ok" or visibility_stats.get("unknown_visible_objects"):
        raise ValueError("Incomplete visibility or unknown object identifiers")
    return {**record, "paths": paths, "objects": objects, "camera": camera, "visibility": visibility,
            "p26_rebuild": {"object_stats": dict(object_stats), "camera_stats": dict(camera_stats),
                            "visibility_stats": dict(visibility_stats),
                            "hashes": {key: file_hash(paths[key]) for key in ("asset", "camera_path", "visibility", "video")}}}


def official_arkit_record(root):
    root = Path(root)
    receipt = json.loads((root / ".complete.json").read_text())
    if receipt.get("official_orientation") is not True or receipt.get("fps") != 24 or receipt.get("size") != [640, 480]:
        raise ValueError("official ARKit completion contract mismatch")
    names = {"video": "output.mp4", "camera_path": "path_strat_1.json", "asset": "front3d_scene_processed.json",
             "room_info": "room_info.json", "visibility": "visibility_results.json"}
    for name in list(names.values()) + ["frame_mapping.json"]:
        if file_hash(root / name) != receipt["output_sha256"].get(name):
            raise ValueError("official asset hash mismatch")
    record = {"source": "real_arkit", "episode_id": receipt["episode"], "scene_id": receipt["source_scene"],
              "split": "fixture_only", "fps": receipt["fps"], "resolution": receipt["size"],
              "coordinate_version": "arkit_official_native_v2", "paths": {key: str((root / name).resolve()) for key, name in names.items()}}
    return rebuild_record(record)


class EpisodeFacts:
    def __init__(self, record, sample_fps=1.0, max_seconds=90.0):
        if sample_fps <= 0 or max_seconds <= 0:
            raise ValueError("positive sample rate and duration required")
        self.record = record
        self.source = record["source"]
        if self.source not in ("sim_front3d", "real_scannet", "real_scannetpp", "real_arkit"):
            raise ValueError("unknown coordinate source")
        if self.source == "real_arkit" and record.get("coordinate_version") != "arkit_official_native_v2":
            raise ValueError("official ARKit coordinate version required")
        self.fps = float(record["fps"])
        if not math.isfinite(self.fps) or not 0 < sample_fps <= self.fps:
            raise ValueError("invalid FPS")
        self.sample_fps = sample_fps
        self.poses = record["camera"]["pose_lookup"]
        self.objects = record["objects"]
        self.categories = {key: visual_category(value.get("category", "")) for key, value in self.objects.items()}
        self.categories = {key: value for key, value in self.categories.items() if value}
        raw = {row["frame_id"]: set(row["visible_objects"]) for row in record["visibility"]["frames"]}
        self.end = min(max_seconds, len(self.poses) / self.fps)
        self.frames = []
        for index in range(math.ceil(self.end * sample_fps)):
            frame_id = round(index * self.fps / sample_fps)
            if frame_id / self.fps >= self.end:
                break
            if frame_id not in raw:
                raise ValueError(f"missing sampled visibility at {frame_id}; no invisible fallback")
            if raw[frame_id] - self.objects.keys():
                raise ValueError("unknown visible instance")
            self.frames.append({"frame": f"f{index:06d}", "timestamp": frame_id / self.fps,
                                "source_frame": frame_id, "visible": raw[frame_id] & self.categories.keys()})
        if not self.frames:
            raise ValueError("empty episode")

    def center(self, object_id):
        point = self.objects[object_id].get("world_center")
        if not isinstance(point, list) or len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("missing geometry")
        return point

    def distance(self, index, object_id):
        return math.dist(self.poses[self.frames[index]["source_frame"]]["location"], self.center(object_id))

    def direction(self, index, object_id):
        pose = self.poses[self.frames[index]["source_frame"]]
        return five_way_camera_direction(pose["location"], pose["rotation_euler_deg"], self.center(object_id), source=self.source)

    def counts(self, index):
        return Counter(self.categories[object_id] for object_id in self.frames[index]["visible"])

    def prefix(self, index):
        return set().union(*(row["visible"] for row in self.frames[:index + 1]))

    def first_categories(self, index):
        first = {}
        for position, row in enumerate(self.frames[:index + 1]):
            for object_id in sorted(row["visible"]):
                first.setdefault(self.categories[object_id], position)
        return sorted((position, category) for category, position in first.items())

    def midpoint(self, indices):
        if not indices:
            raise ValueError("no evidence")
        runs = []
        for index in sorted(set(indices)):
            if not runs or index != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(index)
        run = max(runs, key=lambda values: (len(values), -values[0]))
        return run[(len(run) - 1) // 2]

    def motion(self, before, after):
        first = self.frames[before]["source_frame"]
        last = self.frames[after]["source_frame"]
        poses = self.poses[first:last + 1]
        distance = math.dist(poses[0]["location"], poses[-1]["location"])
        length = sum(math.dist(start["location"], end["location"]) for start, end in zip(poses, poses[1:]))
        rotation = rotation_geodesic_deg(poses[0]["rotation_euler_deg"], poses[-1]["rotation_euler_deg"])
        return distance, length, rotation

    def regions(self, index):
        pose = self.poses[self.frames[index]["source_frame"]]
        intrinsics = pose.get("K", self.record.get("K"))
        resolution = self.record.get("resolution")
        if intrinsics is None or resolution is None:
            return {}
        rotation = np.array(euler_xyz_matrix(pose["rotation_euler_deg"]))
        regions = {}
        for object_id in sorted(self.frames[index]["visible"]):
            local = rotation.T @ (np.array(self.center(object_id)) - pose["location"])
            if self.source == "sim_front3d":
                local = local * [1, -1, -1]
            if local[2] <= 0:
                continue
            projected = np.array(intrinsics) @ local
            horizontal, vertical = projected[:2] / projected[2] / np.array(resolution)
            if not (0 < horizontal < 1 and 0 < vertical < 1):
                continue
            if min(abs(value - boundary) for value in (horizontal, vertical) for boundary in (1 / 3, 2 / 3)) < 0.03:
                continue
            column, row = int(horizontal * 3), int(vertical * 3)
            label = f"{('top', 'middle', 'bottom')[row]}_{('left', 'center', 'right')[column]}"
            regions[object_id] = ("center" if label == "middle_center" else label, column, row)
        return regions

    def teacher(self, indices):
        observations, relations = [], []
        for index in indices:
            frame = self.frames[index]["frame"]
            observations.append({"frame": frame, "objects": [{"category": category, "count": count}
                                  for category, count in sorted(self.counts(index).items())]})
            regions = self.regions(index)
            multiplicity = Counter((self.categories[object_id], region[0]) for object_id, region in regions.items())
            eligible = [object_id for object_id, region in regions.items() if multiplicity[self.categories[object_id], region[0]] == 1]
            for subject in eligible:
                for target in eligible:
                    if subject >= target:
                        continue
                    source_region, column, row = regions[subject]
                    target_region, target_column, target_row = regions[target]
                    predicates = []
                    if column != target_column:
                        predicates.append("left_of_in_image" if column < target_column else "right_of_in_image")
                    if row != target_row:
                        predicates.append("above_in_image" if row < target_row else "below_in_image")
                    for predicate in predicates:
                        relations.append({"frame": frame, "subject": {"category": self.categories[subject], "region": source_region},
                                          "predicate": predicate, "object": {"category": self.categories[target], "region": target_region}})
        return {"type": "memory_write", "observations": observations, "relations": relations}
