"""Prepare and run the approved, bounded five-action rendering probe."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from camera_geometry import horizontal_action_axes
from plan25_contract import clean_category
from qa_v1_common import iter_cache_records, object_size_m


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def action_poses(base):
    location = list(base['location'])
    rotation = list(base['rotation_euler_deg'])
    forward, right = horizontal_action_axes(rotation, source='sim_front3d')
    result = {'base': {'location': location, 'rotation_euler_deg': rotation}}
    for action, angle in [('rotate_left_45', 45), ('rotate_right_45', -45)]:
        result[action] = {'location': location, 'rotation_euler_deg': rotation[:2] + [rotation[2] + angle]}
    for action, axis, sign in [('move_forward_d', forward, 1), ('move_left_d', right, -1),
                               ('move_right_d', right, 1)]:
        result[action] = {'location': [value + 0.5 * direction * sign
                                       for value, direction in zip(location, axis)],
                          'rotation_euler_deg': rotation}
    return result


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    trajectories = read_rows(args.trajectory)
    nominations = read_rows(args.nominations)
    by_frame = {(row['episode_id'], row['frame_id']): row for row in trajectories}
    needed = {row['episode_id'] for row in nominations}
    metadata = {}
    for record in iter_cache_records(args.metadata, splits={'test'}, sources={'sim_front3d'}, domains={'sim'}):
        if record['episode_id'] in needed:
            metadata[record['episode_id']] = record
        if needed <= metadata.keys():
            break
    rejection = Counter()
    scenes = defaultdict(list)
    for nomination in nominations:
        episode = nomination['episode_id']
        record = metadata.get(episode)
        if record is None:
            rejection['missing_repaired_metadata'] += 1
            continue
        base = by_frame[episode, nomination['base_frame']]
        objects = {}
        for object_id, item in record['objects'].items():
            category = clean_category(item.get('category'))
            center = item.get('world_center') or item.get('center')
            size = object_size_m(record, object_id)
            if category and center and size:
                objects[object_id] = {'category': category, 'center': center, 'size_m': size,
                                      'physical_bbox_size': item.get('physical_bbox_size')}
        target = nomination['target_id']
        if target not in objects:
            rejection['target_geometry_or_category'] += 1
            continue
        visible = {item['object_id']: item['pixel_count'] for item in base['visible_objects']
                   if item['object_id'] in objects and item['pixel_count'] / base['pixel_area'] >= 0.0002}
        if target not in visible:
            rejection['target_below_probe_floor'] += 1
            continue
        anchors = [object_id for object_id in visible if object_id != target]
        anchors.sort(key=lambda object_id: (-visible[object_id], object_id))
        anchor = anchors[0] if anchors else None
        identity = f'{episode}:{base["frame_id"]}:{target}'
        state_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        scenes[base['scene_id']].append({
            'state_id': state_id, 'episode_id': episode, 'scene_id': base['scene_id'],
            'split': base['split'], 'frame_id': base['frame_id'], 'timestamp_sec': base['timestamp_sec'],
            'target_id': target, 'anchor_id': anchor, 'objects': objects,
            'poses': action_poses(base), 'focal_length_mm': base['focal_length_mm'],
            'source_render_contract': base['render_contract'],
            'source_pixel_counts': visible, 'source_pixel_area': base['pixel_area'],
            'nomination_gain': nomination['future_fraction'] - nomination['before_fraction'],
            'memory_gate_applied': False, 'video_path': record.get('video_path') or record.get('video'),
        })
    for values in scenes.values():
        values.sort(key=lambda row: (-row['nomination_gain'], row['frame_id'], row['target_id']))
    selected = []
    used_frames = set()
    used_targets = Counter()
    while len(selected) < args.max_states:
        progress = False
        for scene_id in sorted(scenes):
            candidates = scenes[scene_id]
            while candidates:
                candidate = candidates.pop(0)
                frame_key = (candidate['episode_id'], candidate['frame_id'])
                target_key = (scene_id, candidate['target_id'])
                if frame_key in used_frames or used_targets[target_key] >= 2:
                    continue
                used_frames.add(frame_key)
                used_targets[target_key] += 1
                selected.append(candidate)
                progress = True
                break
            if len(selected) >= args.max_states:
                break
        if not progress:
            break
    write_json(args.output / 'states.json', selected)
    write_json(args.output / 'prepare_summary.json', {
        'states': len(selected), 'scenes': len({row['scene_id'] for row in selected}),
        'metadata_episodes': len(metadata), 'rejection': dict(rejection),
        'metadata_root': str(args.metadata), 'memory_gate_applied': False,
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.trajectory, args.nominations]},
        'camera_collision_proxy_radius_m': 0.1, 'status': 'probe_prepared',
    })
    print(json.dumps({'states': len(selected), 'rejection': dict(rejection)}), flush=True)


def render(args):
    states = json.loads((args.output / 'states.json').read_text())
    scenes = defaultdict(list)
    for state in states:
        scenes[state['scene_id']].append(state)
    for scene_id, values in sorted(scenes.items()):
        destination = args.output / 'scenes' / scene_id
        destination.mkdir(parents=True, exist_ok=False)
        manifest = destination / 'states.json'
        write_json(manifest, values)
        command = [str(args.blender), '-b', '-P', str(Path(__file__).with_name('action_branch_render.py')),
                   '--', '--states', str(manifest), '--assets', str(args.assets),
                   '--output', str(destination), '--gpu', str(args.gpu),
                   '--samples', str(args.samples), '--bounces', str(args.bounces),
                   '--exposure', str(args.exposure)]
        if args.denoise:
            command.append('--denoise')
        if args.reuse_run:
            command.extend(['--reuse-results', str(args.reuse_run / 'scenes' / scene_id / 'results.json')])
        with (destination / 'render.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (destination / 'results.json').exists():
            raise RuntimeError(f'Render failed: {destination / "render.log"}')
        print(json.dumps({'scene_complete': scene_id, 'states': len(values)}), flush=True)
    write_json(args.output / 'render.complete.json', {'scenes': len(scenes), 'states': len(states)})




