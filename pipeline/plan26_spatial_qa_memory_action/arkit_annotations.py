"""Project official ARKit OBBs and match timestamped RGB to an existing video."""

import argparse
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from projection_audit import EDGES, clipped_edge, contact_sheet, digest, euler_xyz_matrix


def official_corners(geometry):
    center = np.asarray(geometry['centroid'], dtype=float)
    dimensions = np.asarray(geometry['axesLengths'], dtype=float)
    axes = np.asarray(geometry['normalizedAxes'], dtype=float).reshape(3, 3)
    local = np.array(list(itertools.product((-1, 1), repeat=3))) * dimensions / 2
    return local @ axes + center


def read_trajectory(path):
    values = np.loadtxt(path)
    poses = []
    for row in values:
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = axis_angle_matrix(row[1:4])
        world_to_camera[:3, 3] = row[4:7]
        poses.append(np.linalg.inv(world_to_camera))
    return values[:, 0], np.asarray(poses)


def axis_angle_matrix(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-15:
        return np.eye(3)
    axis = np.asarray(vector) / angle
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def timestamp(name):
    return float(Path(name).stem.split('_')[-1])


def descriptor(image):
    small = np.asarray(image.convert('L').resize((32, 24), Image.Resampling.BOX), dtype=np.float32).ravel().copy()
    small -= small.mean()
    return small / max(float(np.linalg.norm(small)), 1e-8)


def overlay(image, objects, camera_to_world, intrinsic):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    world_to_camera = np.linalg.inv(camera_to_world)
    projected = []
    for index, obj in enumerate(objects):
        corners = official_corners(obj['segments']['obbAligned'])
        local = corners @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        center = local.mean(axis=0)
        if center[2] <= 0:
            continue
        pixel = intrinsic @ center
        pixel = pixel[:2] / pixel[2]
        color = ((index * 67 + 90) % 200 + 55, (index * 103 + 50) % 200 + 55,
                 (index * 139 + 20) % 200 + 55)
        for first, second in EDGES:
            edge = clipped_edge(local[first], local[second], 1)
            if edge is None:
                continue
            pixels = []
            for point in edge:
                screen = intrinsic @ point
                pixels.append(tuple(np.clip(screen[:2] / screen[2], -10000, 10000)))
            draw.line(pixels, fill=color, width=2)
        if 0 <= pixel[0] < image.width and 0 <= pixel[1] < image.height:
            draw.text(tuple(pixel), f"{obj['label']} {index}", fill=color, stroke_width=1, stroke_fill='black')
        projected.append({'uid': obj['uid'], 'label': obj['label'], 'center_px': pixel.tolist()})
    return canvas, projected


def rotated(image, quarter_turns):
    return Image.fromarray(np.rot90(np.asarray(image), quarter_turns).copy())


def run(args):
    from decord import VideoReader, cpu

    args.output.mkdir(parents=True, exist_ok=False)
    annotation_path = args.assets / f'{args.scene}_3dod_annotation.json'
    annotations = json.loads(annotation_path.read_text())
    if annotations.get('skipped'):
        raise ValueError('Official annotation marked skipped')
    times, poses = read_trajectory(args.assets / 'lowres_wide.traj')
    legacy_poses = json.loads(args.camera_path.read_text())
    video = VideoReader(str(args.video), ctx=cpu(0), num_threads=1)
    frame_ids = sorted(set(np.linspace(0, len(video) - 1, args.samples, dtype=int).tolist() + args.frames))
    frames = [Image.fromarray(video[frame_id].asnumpy()) for frame_id in frame_ids]
    targets = np.stack([descriptor(frame) for frame in frames])
    report = {'status': 'requires_visual_review', 'scene': args.scene,
              'code_sha256': digest(__file__), 'video': str(args.video),
              'camera_path_sha256': digest(args.camera_path),
              'video_fps': float(video.get_avg_fps()), 'video_frames': len(video),
              'video_size': list(frames[0].size), 'trajectory_frames': len(legacy_poses),
              'downloads': json.loads((args.assets / 'downloads.json').read_text()),
              'projection': 'official obbAligned; local @ normalizedAxes + centroid; CV positive Z/Y-down',
              'matching': 'global normalized grayscale correlation; 4 rotations; 32x24 thumbnails',
              'samples': []}
    with zipfile.ZipFile(args.assets / 'lowres_wide.zip') as rgb_zip, zipfile.ZipFile(args.assets / 'lowres_wide_intrinsics.zip') as intrinsic_zip:
        for archive in (rgb_zip, intrinsic_zip):
            bad = archive.testzip()
            if bad:
                raise ValueError(f'ZIP CRC failed: {bad}')
        names = sorted([name for name in rgb_zip.namelist() if name.endswith('.png')], key=timestamp)
        intrinsic_names = sorted([name for name in intrinsic_zip.namelist() if name.endswith('.pincam')], key=timestamp)
        intrinsic_times = np.array([timestamp(name) for name in intrinsic_names])
        features = []
        for index, name in enumerate(names):
            image = Image.open(io.BytesIO(rgb_zip.read(name))).convert('RGB')
            features.append(np.stack([descriptor(rotated(image, turn)) for turn in range(4)]))
            if index % 1000 == 0:
                print(f'Indexed {index}/{len(names)} official frames', flush=True)
        features = np.stack(features)
        scores = np.einsum('nkd,sd->snk', features, targets)
        report['official_rgb_frames'] = len(names)
        report['official_time_range'] = [timestamp(names[0]), timestamp(names[-1])]
        panels = []
        world_rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        for sample_index, (frame_id, video_frame) in enumerate(zip(frame_ids, frames)):
            best_frame, turn = np.unravel_index(np.argmax(scores[sample_index]), scores[sample_index].shape)
            name = names[best_frame]
            matched_time = timestamp(name)
            pose_index = int(np.argmin(np.abs(times - matched_time)))
            intrinsic_index = int(np.argmin(np.abs(intrinsic_times - matched_time)))
            width, height, focal_x, focal_y, center_x, center_y = np.loadtxt(io.BytesIO(intrinsic_zip.read(intrinsic_names[intrinsic_index])))
            intrinsic = np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]])
            raw_image = Image.open(io.BytesIO(rgb_zip.read(name))).convert('RGB')
            if raw_image.size != (int(width), int(height)):
                raise ValueError('RGB/intrinsics dimensions differ')
            official_overlay, projected = overlay(raw_image, annotations['data'], poses[pose_index], intrinsic)
            if turn == 0:
                video_intrinsic = np.diag([video_frame.width / width, video_frame.height / height, 1]) @ intrinsic
                video_overlay, _ = overlay(video_frame, annotations['data'], poses[pose_index], video_intrinsic)
                video_overlay.save(args.output / f'video_projection_{frame_id}.jpg')
            transformed_rgb = rotated(raw_image, int(turn)).resize(video_frame.size)
            transformed_overlay = rotated(official_overlay, int(turn)).resize(video_frame.size)
            legacy_pose = legacy_poses[frame_id]
            recovered_center = world_rotation.T @ np.array(legacy_pose['location'])
            recovered_rotation = world_rotation.T @ np.array(euler_xyz_matrix(legacy_pose['rotation_euler_deg']))
            pose_errors = np.linalg.norm(poses[:, :3, 3] - recovered_center, axis=1) + np.linalg.norm(poses[:, :3, :3] - recovered_rotation, axis=(1, 2))
            legacy_index = int(np.argmin(pose_errors))
            fitted_overlay, _ = overlay(raw_image, annotations['data'], poses[legacy_index], intrinsic)
            fitted_overlay = rotated(fitted_overlay, int(turn)).resize(video_frame.size)
            comparison = Image.new('RGB', (video_frame.width * 2, video_frame.height * 2 + 48))
            labels = ['Existing video', 'Official RGB: matched + rotated + resized',
                      'Official OBB + matched official pose/K', 'Official OBB/K + pose recovered from existing trajectory']
            for index, panel in enumerate([video_frame, transformed_rgb, transformed_overlay, fitted_overlay]):
                left, top = (index % 2) * video_frame.width, (index // 2) * (video_frame.height + 24)
                comparison.paste(panel, (left, top + 24))
                ImageDraw.Draw(comparison).text((left + 3, top + 4), labels[index], fill='white')
            output_name = f'{args.scene}_{frame_id}.jpg'
            comparison.save(args.output / output_name)
            official_overlay.resize((raw_image.width * 3, raw_image.height * 3)).save(args.output / f'official_{frame_id}.jpg')
            panels.append(comparison)
            report['samples'].append({'video_frame': frame_id, 'video_time': frame_id / video.get_avg_fps(),
                'official_rgb': name, 'official_timestamp': matched_time, 'official_size': list(raw_image.size),
                'rotation_ccw_degrees': int(turn) * 90, 'correlation': float(scores[sample_index, best_frame, turn]),
                'best_score_per_rotation': scores[sample_index].max(axis=0).tolist(),
                'rgb_mae_255': float(np.abs(np.asarray(transformed_rgb, dtype=float) - np.asarray(video_frame, dtype=float)).mean()),
                'pose_timestamp_error_s': float(times[pose_index] - matched_time),
                'intrinsic_timestamp_error_s': float(intrinsic_times[intrinsic_index] - matched_time),
                'intrinsic_matrix': intrinsic.tolist(), 'official_c2w': poses[pose_index].tolist(),
                'legacy_pose_nearest_official_time': float(times[legacy_index]),
                'legacy_pose_center_error_m': float(np.linalg.norm(poses[legacy_index, :3, 3] - recovered_center)),
                'legacy_pose_rotation_matrix_error': float(np.linalg.norm(poses[legacy_index, :3, :3] - recovered_rotation)),
                'legacy_pose_vs_rgb_time_error_s': float(times[legacy_index] - matched_time),
                'projected_objects': projected, 'comparison': output_name})
    video_times = np.array([sample['video_time'] for sample in report['samples']])
    raw_times = np.array([sample['official_timestamp'] for sample in report['samples']])
    slope, intercept = np.polyfit(video_times, raw_times, 1)
    report['time_fit'] = {'official_time_per_video_second': float(slope), 'official_start_time': float(intercept),
                          'residual_seconds': (raw_times - (video_times * slope + intercept)).tolist()}
    contact_sheet(panels, columns=2, max_size=(960, 800)).save(args.output / 'comparison.jpg')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'time_fit': report['time_fit'], 'samples': len(frame_ids)}), flush=True)


