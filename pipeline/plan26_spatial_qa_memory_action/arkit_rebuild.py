"""Resumable official ARKit downloads and isolated legacy-video annotation rebuild."""

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import zipfile

BASE = 'https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw'
ASSETS = ('{scene}_3dod_annotation.json', 'lowres_wide.traj', 'lowres_wide_intrinsics.zip')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def fetch(url, path, attempts=4):
    path = Path(path)
    record_path = path.with_name(path.name + '.download.json')
    if path.exists() and record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['url'] == url and previous['sha256'] == sha(path):
            return previous
        raise ValueError(f'Existing asset checksum or URL differs: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    start = time.monotonic()
    for attempt in range(attempts):
        attempt_start = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Plan26-ARKit-Recovery/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, path.with_name(path.name + '.part').open('wb') as output:
                expected = response.headers.get('Content-Length')
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(chunk)
            temporary = path.with_name(path.name + '.part')
            if expected and temporary.stat().st_size != int(expected):
                raise ValueError('Incomplete HTTP body')
            if path.suffix == '.zip':
                with zipfile.ZipFile(temporary) as archive:
                    if archive.testzip():
                        raise ValueError('ZIP CRC failure')
            if path.suffix == '.json':
                json.loads(temporary.read_text())
            temporary.replace(path)
            elapsed = time.monotonic() - attempt_start
            record = {'url': url, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      'attempt_seconds': elapsed, 'total_seconds': time.monotonic() - start,
                      'MB_per_second': path.stat().st_size / max(elapsed, 1e-9) / 1e6,
                      'attempts': attempt + 1, 'previous_errors': errors,
                      'timing_scope': 'connection + transfer + disk write + format/CRC check'}
            write_json(record_path, record)
            return record
        except Exception as error:
            errors.append(repr(error))
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f'{url}: {errors}')


def inventory(args):
    args.output.mkdir(parents=True, exist_ok=False)
    fetch(BASE + '/metadata.csv', args.output / 'metadata.csv')
    metadata = {row['video_id']: row for row in csv.DictReader((args.output / 'metadata.csv').open())}
    episodes = []
    rejected = []
    for video in sorted((args.data / 'merged_videos').glob('*/path_strat_arkit/*/output.mp4')):
        episode = video.parent.name
        scene = re.sub(r'_part\d+$', '', episode)
        camera = args.data / 'arkit/planned_path' / episode / 'paths/path_strat_1.json'
        if scene not in metadata or not camera.is_file():
            rejected.append({'episode': episode, 'reason': 'missing official metadata or camera path'})
            continue
        episodes.append({'episode': episode, 'scene': scene, 'official_fold': metadata[scene]['fold'],
                         'sky_direction': metadata[scene]['sky_direction'],
                         'video': str(video), 'video_group': video.parents[2].name,
                         'camera_path': str(camera), 'camera_sha256': sha(camera)})
    if not episodes:
        raise ValueError('No existing video episodes found')
    write_json(args.output / 'manifest.json', {'version': 1, 'episodes': episodes, 'rejected': rejected,
               'scene_count': len({episode['scene'] for episode in episodes}),
               'assets': ASSETS, 'timing': 'legacy_preserved',
               'contract_sha256': sha(Path(__file__).parent / 'contracts/coordinate_contract.md')})
    print(json.dumps({'episodes': len(episodes), 'scenes': len({e['scene'] for e in episodes}), 'rejected': len(rejected)}))


def download(args):
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / '.download.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.monotonic()
    state = {'status': 'running', 'pid': os.getpid(), 'manifest_sha256': sha(args.manifest),
             'script_sha256': sha(__file__), 'completed_scenes': [], 'failures': [],
             'workers': args.workers, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    scenes = {item['scene']: item['official_fold'] for item in manifest['episodes']}

    def one_scene(item):
        scene, fold = item
        records = [fetch(f'{BASE}/{fold}/{scene}/{template.format(scene=scene)}',
                         args.output / scene / template.format(scene=scene)) for template in ASSETS]
        write_json(args.output / scene / 'download.complete.json', {'scene': scene, 'records': records})
        return scene

    try:
        write_json(args.output / 'download_status.json', state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(one_scene, item): item[0] for item in sorted(scenes.items())}
            for future in concurrent.futures.as_completed(pending):
                try:
                    state['completed_scenes'].append(future.result())
                except Exception as error:
                    state['failures'].append({'scene': pending[future], 'error': repr(error)})
                state['elapsed_seconds'] = time.monotonic() - started
                write_json(args.output / 'download_status.json', state)
                print(f"complete={len(state['completed_scenes'])}/{len(scenes)} failures={len(state['failures'])}", flush=True)
        state['status'] = 'complete' if not state['failures'] else 'complete_with_failures'
        write_json(args.output / 'download_status.json', state)
    finally:
        lock.unlink(missing_ok=True)


def proxy_visibility(corners, pose, intrinsic, width, height, stride):
    import numpy as np

    horizontal, vertical = np.meshgrid(np.arange(0, width, stride) + .5, np.arange(0, height, stride) + .5)
    rays = np.stack([horizontal.ravel(), vertical.ravel(), np.ones(horizontal.size)], axis=1)
    rays = rays @ np.linalg.inv(intrinsic).T @ pose[:3, :3].T
    origin = pose[:3, 3]
    nearest = np.full(len(rays), np.inf)
    labels = np.full(len(rays), -1, dtype=int)
    for index, box in enumerate(corners):
        center = box.mean(axis=0)
        edges = np.array([box[4] - box[0], box[2] - box[0], box[1] - box[0]])
        lengths = np.linalg.norm(edges, axis=1)
        axes = edges / lengths[:, None]
        local_origin = axes @ (origin - center)
        directions = rays @ axes.T
        parallel = np.abs(directions) < 1e-12
        safe = np.where(parallel, 1, directions)
        first = (-lengths / 2 - local_origin) / safe
        second = (lengths / 2 - local_origin) / safe
        entry = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
        leave = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
        outside = (parallel & (np.abs(local_origin) > lengths / 2)).any(axis=1)
        depth = np.where(entry > 1e-4, entry, leave)
        valid = (~outside) & (leave >= np.maximum(entry, 1e-4)) & (depth < nearest)
        nearest[valid] = depth[valid]
        labels[valid] = index
    counts = {int(index): int((labels == index).sum()) for index in np.unique(labels) if index >= 0}
    return counts, len(labels)


def generate(args):
    import numpy as np
    from decord import VideoReader, cpu
    from arkit_annotations import official_corners, read_trajectory, timestamp
    from projection_audit import euler_xyz_matrix

    manifest = json.loads(args.manifest.read_text())
    rotation_overrides = json.loads(args.rotations.read_text()) if args.rotations else {}
    args.output.mkdir(parents=True, exist_ok=True)
    config = {'manifest_sha256': sha(args.manifest), 'script_sha256': sha(__file__),
              'sample_fps': args.sample_fps, 'pixel_stride': args.pixel_stride,
              'rotations': rotation_overrides, 'timing': 'legacy_preserved',
              'visibility_kind': 'official_obb_nearest_ray_proxy_not_rgb_segmentation'}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output configuration differs; use new output directory')
    write_json(config_path, config)
    status = {'complete': [], 'failed': [], 'config_sha256': config_hash}
    world_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    selected = [item for item in manifest['episodes'] if not args.episode or item['episode'] == args.episode]
    if args.limit:
        selected = selected[:args.limit]
    for item in selected:
        destination = args.output / item['video_group'] / item['episode']
        destination.mkdir(parents=True, exist_ok=True)
        try:
            if (destination / '.complete.json').exists():
                previous = json.loads((destination / '.complete.json').read_text())
                if previous['config_sha256'] != config_hash or any(sha(destination / name) != value for name, value in previous['output_sha256'].items()):
                    raise ValueError('Existing complete artifact mismatch')
                status['complete'].append(item['episode'])
                continue
            assets = args.assets / item['scene']
            completion = json.loads((assets / 'download.complete.json').read_text())
            for record in completion['records']:
                if sha(assets / record['url'].rsplit('/', 1)[-1]) != record['sha256']:
                    raise ValueError('Downloaded asset checksum mismatch')
            annotation = json.loads((assets / f"{item['scene']}_3dod_annotation.json").read_text())
            if annotation.get('skipped'):
                raise ValueError('Official annotation skipped')
            objects = []
            boxes = []
            for obj in annotation['data']:
                geometry = obj['segments']['obbAligned']
                axes = np.asarray(geometry['normalizedAxes']).reshape(3, 3)
                if not np.isfinite(axes).all() or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-4) or np.any(np.asarray(geometry['axesLengths']) <= 0):
                    raise ValueError('Invalid official OBB')
                box = official_corners(geometry)
                if not np.isfinite(box).all():
                    raise ValueError('Nonfinite official geometry')
                boxes.append(box)
                objects.append({'uid': obj['uid'], 'category': obj['label'], 'center': geometry['centroid'],
                                'size_m': geometry['axesLengths'], 'corners': box.tolist(),
                                'bbox_min': box.min(axis=0).tolist(), 'bbox_max': box.max(axis=0).tolist()})
            if not objects or len({obj['uid'] for obj in objects}) != len(objects):
                raise ValueError('Empty/duplicate official identities')
            times, poses = read_trajectory(assets / 'lowres_wide.traj')
            pose_lookup = {}
            for index, official_pose in enumerate(poses):
                pose_lookup.setdefault(tuple(np.round(official_pose[:3].ravel(), 5)), []).append(index)
            if sha(item['camera_path']) != item['camera_sha256']:
                raise ValueError('Camera input changed since inventory')
            legacy = json.loads(Path(item['camera_path']).read_text())
            video = VideoReader(item['video'], ctx=cpu(0), num_threads=1)
            width, height = (int(video[0].shape[1]), int(video[0].shape[0]))
            if len(video) != len(legacy):
                raise ValueError('Video/pose frame count differs')
            video_fps = float(video.get_avg_fps())
            if not np.isfinite(video_fps) or video_fps <= 0:
                raise ValueError('Invalid video FPS')
            step = max(1, round(video_fps / args.sample_fps))
            frame_ids = range(len(video))
            with zipfile.ZipFile(assets / 'lowres_wide_intrinsics.zip') as archive:
                names = sorted([name for name in archive.namelist() if name.endswith('.pincam')], key=timestamp)
                intrinsic_times = np.array([timestamp(name) for name in names])
                turn = rotation_overrides.get(item['episode'], 0)
                if turn not in (0, 90, 180, 270):
                    raise ValueError('Rotation override must be CCW quarter turn')
                snapshots = []
                camera_frames = []
                intrinsic_cache = {}
                for frame_id in frame_ids:
                    pose = np.eye(4)
                    pose[:3, 3] = world_rotation.T @ np.array(legacy[frame_id]['location'])
                    pose[:3, :3] = world_rotation.T @ np.array(euler_xyz_matrix(legacy[frame_id]['rotation_euler_deg']))
                    candidates = pose_lookup.get(tuple(np.round(pose[:3].ravel(), 5)))
                    if candidates:
                        expected_time = camera_frames[-1]['official_pose_timestamp'] if camera_frames else times[0]
                        best = min(candidates, key=lambda index: abs(times[index] - expected_time))
                        error = float(np.linalg.norm(poses[best, :3] - pose[:3]))
                    else:
                        errors = np.linalg.norm(poses[:, :3, 3] - pose[:3, 3], axis=1) + np.linalg.norm(poses[:, :3, :3] - pose[:3, :3], axis=(1, 2))
                        best = int(np.argmin(errors))
                        error = float(errors[best])
                    if error > 1e-5:
                        raise ValueError('Legacy pose does not match official trajectory')
                    intrinsic_index = int(np.argmin(np.abs(intrinsic_times - times[best])))
                    if abs(intrinsic_times[intrinsic_index] - times[best]) > .2:
                        raise ValueError('Missing nearby intrinsics')
                    if intrinsic_index not in intrinsic_cache:
                        intrinsic_cache[intrinsic_index] = np.loadtxt(io.BytesIO(archive.read(names[intrinsic_index])))
                    raw_width, raw_height, focal_x, focal_y, center_x, center_y = intrinsic_cache[intrinsic_index]
                    intrinsic = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]])
                    if turn == 0:
                        display = np.eye(3)
                    elif turn == 90:
                        display = np.array([[0, 1, 0], [-1, 0, raw_width - 1], [0, 0, 1]])
                    elif turn == 180:
                        display = np.array([[-1, 0, raw_width - 1], [0, -1, raw_height - 1], [0, 0, 1]])
                    else:
                        display = np.array([[0, -1, raw_height - 1], [1, 0, 0], [0, 0, 1]])
                    target_width, target_height = (raw_height, raw_width) if turn in (90, 270) else (raw_width, raw_height)
                    if abs(width / height - target_width / target_height) > .01:
                        raise ValueError('Video orientation/aspect requires explicit rotation override')
                    display = np.diag([width / target_width, height / target_height, 1]) @ display
                    intrinsic_video = display @ intrinsic
                    camera_frames.append({'frame_id': frame_id, 'timestamp_s': frame_id / video_fps,
                                          'official_pose_timestamp': float(times[best]), 'c2w': pose.tolist(),
                                          'intrinsics': intrinsic_video.tolist(), 'display_transform': display.tolist(),
                                          'orientation_status': 'explicit_override' if item['episode'] in rotation_overrides else 'same_aspect_zero_rotation_assumption',
                                          'pose_match_error': error})
                    if frame_id % step:
                        continue
                    counts, ray_count = proxy_visibility(boxes, pose, intrinsic_video, width, height, args.pixel_stride)
                    snapshots.append({'frame_id': frame_id, 'timestamp_s': frame_id / video_fps,
                                      'visible_objects': [objects[index]['uid'] for index in counts],
                                      'pixel_counts_sampled': {objects[index]['uid']: count for index, count in counts.items()},
                                      'ray_count': ray_count, 'pixel_stride': args.pixel_stride})
            official_seconds = np.array([frame['official_pose_timestamp'] for frame in camera_frames])
            video_seconds = np.array([frame['timestamp_s'] for frame in camera_frames])
            if np.any(np.diff(official_seconds) < -1e-6):
                raise ValueError('Segment source timestamps are not monotonic')
            slope = float(np.polyfit(video_seconds, official_seconds, 1)[0]) if len(video_seconds) > 1 else None
            segment_alignment = {'episode': item['episode'], 'source_scene': item['scene'],
                                 'source_pose_start_s': float(official_seconds[0]),
                                 'source_pose_end_sampled_s': float(official_seconds[-1]),
                                 'video_fps_actual': video_fps, 'video_frame_count': len(video),
                                 'source_seconds_per_video_second': slope,
                                 'time_scale_warning': slope is None or abs(slope - 1) > .05,
                                 'source_timestamps_basis': 'matched_preserved_pose_not_corrected_RGB_time'}
            write_json(destination / 'segment_alignment.json', segment_alignment)
            observations = {obj['uid']: [] for obj in objects}
            for snapshot in snapshots:
                for uid in snapshot['visible_objects']:
                    observations[uid].append(snapshot['frame_id'])
            for obj in objects:
                seen_frames = observations[obj['uid']]
                obj['observed_frame_ids'] = seen_frames
                obj['first_seen_frame'] = seen_frames[0] if seen_frames else None
                obj['last_seen_frame'] = seen_frames[-1] if seen_frames else None
                obj['observed_sample_count'] = len(seen_frames)
                obj['observation_evidence'] = config['visibility_kind']
            write_json(destination / 'objects.json', objects)
            write_json(destination / 'camera_frames.json', camera_frames)
            write_json(destination / 'visibility_results.json', snapshots)
            write_json(destination / 'episode.json', {**item, **config, 'video_frames': len(video), 'video_fps': video_fps,
                       'sample_frame_step': step, 'room_annotations': 'unavailable_not_inferred',
                       'objects': 'objects.json', 'camera_frames': 'camera_frames.json', 'visibility': 'visibility_results.json',
                       'downstream_status': 'requires_QA_evidence_gate_not_formal_training_data'})
            outputs = ['objects.json', 'camera_frames.json', 'visibility_results.json', 'episode.json', 'segment_alignment.json']
            write_json(destination / '.complete.json', {'config_sha256': config_hash,
                       'output_sha256': {name: sha(destination / name) for name in outputs},
                       'input_records': completion['records']})
            status['complete'].append(item['episode'])
        except Exception as error:
            failure = {'episode': item['episode'], 'error': repr(error)}
            write_json(destination / 'failure.json', failure)
            status['failed'].append(failure)
        write_json(args.output / 'generation_status.json', status)
        print(f"generated={len(status['complete'])} failed={len(status['failed'])}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('inventory')
    prepare.add_argument('--data', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    transfer = subparsers.add_parser('download')
    transfer.add_argument('--manifest', type=Path, required=True)
    transfer.add_argument('--output', type=Path, required=True)
    transfer.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    rebuild = subparsers.add_parser('generate')
    rebuild.add_argument('--manifest', type=Path, required=True)
    rebuild.add_argument('--assets', type=Path, required=True)
    rebuild.add_argument('--output', type=Path, required=True)
    rebuild.add_argument('--rotations', type=Path)
    rebuild.add_argument('--episode')
    rebuild.add_argument('--limit', type=int)
    rebuild.add_argument('--sample-fps', type=float, default=1)
    rebuild.add_argument('--pixel-stride', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'generate' and (args.sample_fps <= 0 or args.pixel_stride < 1):
        parser.error('sample-fps and pixel-stride must be positive')
    {'inventory': inventory, 'download': download, 'generate': generate}[args.command](args)
