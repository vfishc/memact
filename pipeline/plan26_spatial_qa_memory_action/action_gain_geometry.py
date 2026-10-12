"""Compare pixel gain with geometry, clipping and joint-object evidence."""

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_geometry import euler_xyz_matrix


def area(points):
    return abs(sum(first[0] * second[1] - second[0] * first[1]
                   for first, second in zip(points, points[1:] + points[:1]))) / 2 if len(points) >= 3 else 0


def convex_hull(points):
    ordered = sorted(set(tuple(point) for point in points))
    def cross(first, second, third):
        return ((second[0] - first[0]) * (third[1] - first[1])
                - (second[1] - first[1]) * (third[0] - first[0]))
    halves = []
    for sequence in [ordered, ordered[::-1]]:
        half = []
        for point in sequence:
            while len(half) >= 2 and cross(half[-2], half[-1], point) <= 0:
                half.pop()
            half.append(point)
        halves.append(half[:-1])
    return halves[0] + halves[1]


def clip_rectangle(points, width=640, height=480):
    polygon = list(points)
    for axis, bound, sign in [(0, 0, 1), (0, width, -1), (1, 0, 1), (1, height, -1)]:
        result = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            start_inside = sign * (start[axis] - bound) >= 0
            end_inside = sign * (end[axis] - bound) >= 0
            if start_inside != end_inside:
                fraction = (bound - start[axis]) / (end[axis] - start[axis])
                result.append(tuple(start[index] + fraction * (end[index] - start[index]) for index in range(2)))
            if end_inside:
                result.append(end)
        polygon = result
    return polygon


def project_box(bounds, pose, lens):
    corners = np.asarray(list(itertools.product(*zip(bounds[:3], bounds[3:]))))
    rotation = np.asarray(euler_xyz_matrix(pose['rotation_euler_deg']))
    local = (corners - np.asarray(pose['location'])) @ rotation
    depth = -local[:, 2]
    if min(depth) <= 0.01:
        return {'status': 'near_plane_intersection_or_behind'}
    focal = lens / 36 * 640
    points = np.column_stack([320 + focal * local[:, 0] / depth, 240 - focal * local[:, 1] / depth])
    hull = convex_hull(points.tolist())
    full_area = area(hull)
    clipped_area = area(clip_rectangle(hull))
    return {'status': 'ok', 'projected_box_area': full_area,
            'in_frame_box_area': clipped_area,
            'box_in_frame_fraction': clipped_area / full_area if full_area else 0}


def object_metrics(object_id, state, branch, raw_objects):
    if object_id is None or object_id not in raw_objects:
        return None
    raw = raw_objects[object_id]
    bounds = raw.get('bbox_blender')
    if not bounds or len(bounds) != 6:
        return None
    pose = branch['pose']
    center = np.array(state['objects'][object_id]['center'])
    relative = center - np.array(pose['location'])
    rotation = np.asarray(euler_xyz_matrix(pose['rotation_euler_deg']))
    local = relative @ rotation
    range_m = float(np.linalg.norm(relative))
    bearing = math.degrees(math.acos(float(np.clip(-local[2] / max(range_m, 1e-8), -1, 1))))
    projection = project_box(bounds, pose, branch['lens_mm'])
    pixels = branch['objects'].get(object_id, {}).get('pixels', 0)
    full_area = projection.get('projected_box_area', 0)
    projection.update(range_m=range_m, bearing_deg=bearing, pixels=pixels,
                      pixel_over_full_projected_box=pixels / full_area if full_area else None)
    return projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    output = args.run / 'geometry_gain'
    output.mkdir(exist_ok=False)
    states = {row['state_id']: row for row in json.loads((args.run / 'states.json').read_text())}
    audits = {row['state_id']: row for row in json.loads((args.run / 'audit/state_audits.json').read_text())}
    rows = []
    counts = Counter()
    for path in sorted((args.run / 'scenes').glob('*/results.json')):
        rendered = json.loads(path.read_text())
        asset = json.loads((args.assets / rendered['scene_id'] / 'front3d_scene_processed.json').read_text())
        raw_objects = {item['name']: item for room in asset.values() for item in room if isinstance(item, dict) and item.get('name')}
        for result in rendered['states']:
            state = states[result['state_id']]
            base = result['branches']['base']
            before = object_metrics(state['target_id'], state, base, raw_objects)
            for audit in audits[state['state_id']]['branches']:
                action = audit['action']
                branch = result['branches'][action]
                after = object_metrics(state['target_id'], state, branch, raw_objects)
                anchor = object_metrics(state['anchor_id'], state, branch, raw_objects)
                baseline = math.dist(base['pose']['location'], branch['pose']['location'])
                target_center = np.asarray(state['objects'][state['target_id']]['center'])
                vectors = [np.asarray(observation['pose']['location']) - target_center for observation in [base, branch]]
                angle = math.degrees(math.acos(float(np.clip(np.dot(*vectors) / max(np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]), 1e-8), -1, 1))))
                row = {'state_id': state['state_id'], 'action': action, 'pixel_pass': audit['pixel_pass'],
                       'before': before, 'after': after, 'anchor_after': anchor,
                       'translation_baseline_m': baseline, 'target_viewpoint_change_deg': angle,
                       'amodal_in_frame': audit['visible_over_in_frame_amodal'],
                       'after_border': audit['after_border'],
                       'metric_scale_from_pure_rotation': False,
                       'full_body_collision_verified': False}
                if before and after:
                    row['range_change_m'] = before['range_m'] - after['range_m']
                    row['centering_gain_deg'] = before['bearing_deg'] - after['bearing_deg']
                    box_before, box_after = before.get('projected_box_area'), after.get('projected_box_area')
                    if box_before and box_after and before['pixels']:
                        row['raw_pixel_ratio'] = after['pixels'] / before['pixels']
                        row['projected_box_area_ratio'] = box_after / box_before
                        row['area_corrected_pixel_ratio'] = row['raw_pixel_ratio'] / row['projected_box_area_ratio']
                    if after.get('box_in_frame_fraction') is not None:
                        counts['projected_box_coverage_available'] += 1
                        if audit['pixel_pass'] and after['box_in_frame_fraction'] >= 0.8:
                            counts['pixel_pass_and_box_in_frame_80pct'] += 1
                    if anchor and anchor.get('pixel_over_full_projected_box') is not None and after.get('pixel_over_full_projected_box') is not None:
                        row['joint_box_visibility_proxy'] = min(after['pixel_over_full_projected_box'], anchor['pixel_over_full_projected_box'])
                if audit['pixel_pass']:
                    counts['pixel_pass'] += 1
                    counts['pixel_pass_translation' if baseline else 'pixel_pass_rotation'] += 1
                rows.append(row)
    (output / 'comparisons.json').write_text(json.dumps(rows, indent=2))
    (output / 'summary.json').write_text(json.dumps({'counts': dict(counts),
        'limitations': ['World AABB projected hull is a geometric proxy, not an amodal silhouette.',
                       'Near-plane intersections are unknown; no invalid projection used.',
                       'Metric translation baseline enables scale only with valid correspondence/parallax.',
                       'No score automatically certifies QA answerability.']}, indent=2))
    print(json.dumps(dict(counts), indent=2))


if __name__ == '__main__':
    main()
