"""Blender worker: fixed-intrinsic branches, camera clearance, RGB and masks."""

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time


def main():
    import bpy
    import numpy as np
    from mathutils import Euler, Vector
    from mathutils.bvhtree import BVHTree

    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--bounces', type=int, default=1)
    parser.add_argument('--exposure', type=float, default=0)
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--reuse-results', type=Path)
    parser.add_argument('--passive-only', action='store_true')
    parser.add_argument('--all-isolated', action='store_true')
    parser.add_argument('--isolated-samples', type=int)
    parser.add_argument('--mask-only-isolated', action='store_true')
    parser.add_argument('--prefilter', action='store_true')
    parser.add_argument('--reuse-passive', type=Path)
    parser.add_argument('--fixed-frame', action='store_true')
    parser.add_argument('--resident-objects', type=Path)
    parser.add_argument('--expand-pairs', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plan26_spatial_qa_memory_action.action_render_optimization import (
        after_possible, before_possible, isolated_requests)
    states = json.loads(args.states.read_text())
    scene_id = states[0]['scene_id']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.max_bounces = args.bounces
    scene.cycles.diffuse_bounces = args.bounces
    scene.cycles.glossy_bounces = 0
    scene.cycles.transparent_max_bounces = 1
    scene.cycles.use_denoising = args.denoise
    scene.view_settings.exposure = args.exposure
    scene.cycles.seed = 25
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'CUDA'
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == 'CUDA']
    if args.gpu >= len(devices):
        raise RuntimeError('Requested CUDA device unavailable')
    for device in preferences.devices:
        device.use = device == devices[args.gpu]
    scene.cycles.device = 'GPU'
    bpy.ops.import_scene.gltf(filepath=str(args.assets / scene_id / 'front3d_scene.glb'))
    contract = states[0]['source_render_contract']
    world = bpy.data.worlds.new('P26World')
    world.use_nodes = True
    scene.world = world
    nodes = world.node_tree.nodes
    nodes.clear()
    environment = nodes.new('ShaderNodeTexEnvironment')
    environment.image = bpy.data.images.load(contract['hdri'])
    background = nodes.new('ShaderNodeBackground')
    output = nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(environment.outputs['Color'], background.inputs['Color'])
    world.node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

    def material(name, image_path):
        value = bpy.data.materials.new(name)
        value.use_nodes = True
        shader = value.node_tree.nodes.get('Principled BSDF')
        texture = value.node_tree.nodes.new('ShaderNodeTexImage')
        texture.image = bpy.data.images.load(image_path, check_existing=True)
        value.node_tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
        return value

    floor = material('P26Floor', contract['floor_texture'])
    wall = material('P26Wall', contract['wall_texture'])
    ceiling = bpy.data.materials.new('P26CeilingPlain')
    ceiling.use_nodes = True
    ceiling_shader = ceiling.node_tree.nodes.get('Principled BSDF')
    ceiling_shader.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    ceiling_shader.inputs['Roughness'].default_value = 0.8
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    for obj in meshes:
        name = obj.name.lower()
        surface = ceiling if 'ceiling' in name else (
            floor if 'floor' in name or 'ground' in name else (
                wall if 'wall' in name else None))
        if surface:
            obj.data.materials.clear()
            obj.data.materials.append(surface)
    processed = json.loads((args.assets / scene_id / 'front3d_scene_processed.json').read_text())
    names = {item['name'] for room in processed.values() for item in room
             if isinstance(item, dict) and item.get('name') and item.get('bbox')}
    id_to_name = {}
    for obj in meshes:
        obj.pass_index = 0
        if obj.name in names:
            obj.pass_index = len(id_to_name) + 1
            id_to_name[obj.pass_index] = obj.name
    name_to_id = {name: index for index, name in id_to_name.items()}
    geometry = {}
    vertices = []
    polygons = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        offset = len(vertices)
        vertices.extend(evaluated.matrix_world @ vertex.co for vertex in mesh.vertices)
        polygons.extend(tuple(offset + index for index in polygon.vertices) for polygon in mesh.polygons)
        evaluated.to_mesh_clear()
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        center = sum(corners, Vector()) / 8
        geometry[obj.name] = {'center': list(center), 'dimensions': list(obj.dimensions),
                              'world_obb_corners': [list(corner) for corner in corners],
                              'obb_source': 'mesh_local_bound_box_transformed_by_matrix_world'}
    tree = BVHTree.FromPolygons(vertices, polygons)
    del vertices, polygons

    def collision(start, end):
        start, end = Vector(start), Vector(end)
        travel = end - start
        distance = travel.length
        steps = max(1, math.ceil(distance / 0.025))
        clearance = float('inf')
        for step in range(steps + 1):
            position = start.lerp(end, step / steps)
            nearest = tree.find_nearest(position)
            if nearest[0] is None:
                return {'valid': False, 'reason': 'missing_collision_geometry'}
            clearance = min(clearance, nearest[3])
            if nearest[3] < 0.125:
                return {'valid': False, 'reason': 'camera_clearance', 'clearance_m': clearance}
            floor_hit = tree.ray_cast(position, Vector((0, 0, -1)), 2.0)
            if floor_hit[0] is None:
                return {'valid': False, 'reason': 'no_support_below_camera'}
        if distance:
            hit = tree.ray_cast(start, travel.normalized(), distance)
            if hit[0] is not None:
                return {'valid': False, 'reason': 'path_surface_intersection'}
        return {'valid': True, 'clearance_m': clearance, 'proxy': 'camera_sphere_0.1m',
                'sample_step_m': 0.025, 'clearance_gate_m': 0.125,
                'full_robot_body_verified': False}

    camera_data = bpy.data.cameras.new('P26Camera')
    camera_data.sensor_width = 36
    camera_data.sensor_fit = 'AUTO'
    camera = bpy.data.objects.new('P26Camera', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.use_nodes = True
    scene.view_layers[0].use_pass_object_index = True
    nodes = scene.node_tree.nodes
    nodes.clear()
    layers = nodes.new('CompositorNodeRLayers')
    composite = nodes.new('CompositorNodeComposite')
    scene.node_tree.links.new(layers.outputs['Image'], composite.inputs['Image'])
    index_output = nodes.new('CompositorNodeOutputFile')
    index_output.format.file_format = 'OPEN_EXR'
    index_output.format.color_depth = '32'
    index_output.file_slots[0].path = 'index'
    scene.node_tree.links.new(layers.outputs['IndexOB'], index_output.inputs[0])
    serial = 0
    render_cache = {}
    reuse_count = 0
    passive_by_frame = {}
    if args.reuse_passive:
        passive_data = json.loads(args.reuse_passive.read_text())
        if passive_data['scene_id'] != scene_id or passive_data['source_environment'] != contract:
            raise ValueError('Passive scene/material contract mismatch')
        if passive_data['geometry'] != geometry:
            raise ValueError('Passive geometry mismatch')
        for key, expected in [('samples', args.samples), ('bounces', args.bounces),
                              ('exposure', args.exposure), ('denoise', args.denoise)]:
            if passive_data.get(key) != expected:
                raise ValueError(f'Passive rendering setting mismatch: {key}')
        passive_states = json.loads((args.reuse_passive.parent / 'states.json').read_text())
        if len(passive_states) != len(passive_data['states']):
            raise ValueError('Incomplete passive results')
        for state, result in zip(passive_states, passive_data['states']):
            if state['state_id'] != result['state_id']:
                raise ValueError('Passive state ordering mismatch')
            passive_by_frame[state['frame_id']] = result['branches']['base']
    results = []
    started = time.time()
    resident_passive_renders = 0

    def result_payload():
        return {'scene_id': scene_id, 'states': list(results), 'geometry': geometry,
            'id_to_name': id_to_name, 'elapsed_sec': time.time() - started,
            'renders': serial - resident_passive_renders, 'samples': args.samples,
            'bounces': args.bounces, 'exposure': args.exposure, 'denoise': args.denoise,
            'source_environment': contract,
            'optimizations': {'isolated_samples': args.isolated_samples,
                'mask_only_isolated': args.mask_only_isolated, 'prefilter': args.prefilter,
                'fixed_frame': args.fixed_frame, 'reused_base_states': reuse_count,
                'resident': bool(args.resident_objects)}}

    def state_stream():
        nonlocal resident_passive_renders, started
        yield from states
        if not args.resident_objects:
            return
        from plan26_spatial_qa_memory_action.action_state_mining import select_states
        passive = result_payload()
        (args.output / 'results.json').write_text(json.dumps(passive, indent=2))
        objects = json.loads(args.resident_objects.read_text())
        selected, counts, frame_set = select_states(states, passive, objects, args.expand_pairs)
        directory = args.output.parent
        (directory / 'states.json').write_text(json.dumps(selected, indent=2))
        (directory / 'mining.json').write_text(json.dumps({'nomination_stats': counts,
            'candidate_frames': len(frame_set), 'target_pair_states': len(selected)}))
        for state, result in zip(states, results):
            passive_by_frame[state['frame_id']] = result['branches']['base']
        results.clear()
        resident_passive_renders = serial
        started = time.time()
        args.output = directory / 'scenes' / scene_id
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'states.json').write_text(json.dumps(selected, indent=2))
        args.passive_only = False
        args.all_isolated = True
        yield from selected
    reused = {row['state_id']: row for row in json.loads(args.reuse_results.read_text())['states']} if args.reuse_results else {}
    with tempfile.TemporaryDirectory(prefix='p26_action_') as temporary:
        index_output.base_path = temporary

        def render(pose, lens, directory, isolated=None):
            nonlocal serial
            cache_key = (tuple(pose['location']), tuple(pose['rotation_euler_deg']), lens, isolated)
            if cache_key in render_cache:
                return render_cache[cache_key]
            serial += 1
            directory.mkdir(parents=True, exist_ok=False)
            camera.location = Vector(pose['location'])
            camera.rotation_euler = Euler(tuple(math.radians(value) for value in pose['rotation_euler_deg']), 'XYZ')
            camera_data.lens = lens
            for obj in meshes:
                obj.hide_render = isolated is not None and obj.name != isolated
            render_frame = 1 if args.fixed_frame else serial
            scene.frame_set(render_frame)
            mask_only = isolated is not None and args.mask_only_isolated
            scene.cycles.samples = args.isolated_samples if isolated is not None and args.isolated_samples else args.samples
            scene.cycles.use_denoising = args.denoise and not mask_only
            scene.cycles.max_bounces = 0 if mask_only else args.bounces
            scene.cycles.diffuse_bounces = 0 if mask_only else args.bounces
            scene.render.filepath = str(directory / 'rgb.png')
            bpy.ops.render.render(write_still=not mask_only)
            exr = Path(temporary) / f'index{render_frame:04d}.exr'
            image = bpy.data.images.load(str(exr), check_existing=False)
            width, height = image.size
            buffer = np.empty(width * height * 4, dtype=np.float32)
            image.pixels.foreach_get(buffer)
            bpy.data.images.remove(image)
            labels = np.rint(buffer[::4]).astype(np.uint16).reshape(height, width)[::-1].copy()
            exr.unlink()
            np.savez_compressed(directory / 'mask.npz', labels=labels)
            evidence = {}
            for index in np.unique(labels):
                if int(index) not in id_to_name:
                    continue
                vertical, horizontal = np.where(labels == index)
                evidence[id_to_name[int(index)]] = {
                    'pixels': int(len(vertical)), 'fraction': float(len(vertical) / labels.size),
                    'bbox_xyxy': [int(horizontal.min()), int(vertical.min()),
                                  int(horizontal.max()), int(vertical.max())],
                    'touches_border': bool(horizontal.min() == 0 or horizontal.max() == width - 1
                                           or vertical.min() == 0 or vertical.max() == height - 1),
                }
            observation = {'rgb': None if mask_only else str(directory / 'rgb.png'), 'mask': str(directory / 'mask.npz'),
                    'objects': evidence, 'pose': pose, 'lens_mm': lens, 'resolution': [width, height],
                    'K': [[lens / 36 * width, 0, width / 2], [0, lens / 36 * width, height / 2], [0, 0, 1]]}
            render_cache[cache_key] = observation
            return observation

        for state in state_stream():
            base = state['poses']['base']
            state_dir = args.output / state['state_id']
            branches = {}
            for action, pose in state['poses'].items():
                if action == 'base' and state['frame_id'] in passive_by_frame:
                    observation = passive_by_frame[state['frame_id']]
                    if observation['pose'] != pose or observation['lens_mm'] != state['focal_length_mm']:
                        raise ValueError('Passive pose/intrinsics mismatch')
                    if observation['resolution'] != [640, 480] or not Path(observation['rgb']).is_file() or not Path(observation['mask']).is_file():
                        raise ValueError('Passive media missing or incompatible')
                    branches[action] = observation
                    reuse_count += 1
                    continue
                if reused:
                    branches[action] = reused[state['state_id']]['branches'][action]
                    continue
                if args.prefilter and action != 'base':
                    before = branches['base']
                    projected = {**before, 'pose': pose}
                    if not before_possible(state, before, geometry, state['target_id']) or not after_possible(
                            before, projected, geometry, state['target_id'], require_pixels=False):
                        branches[action] = {'rendered': False, 'prefilter': 'C_necessary_geometry_or_referent'}
                        continue
                check = {'valid': True, 'status': 'recorded_passive_pose_not_action_checked'} if args.passive_only else collision(base['location'], pose['location'])
                if not check['valid'] and action != 'base':
                    branches[action] = {'collision': check, 'rendered': False}
                    continue
                observation = render(pose, state['focal_length_mm'], state_dir / action)
                branches[action] = {'collision': check, 'rendered': True, **observation}
            target = state['target_id']
            valid = [action for action, branch in branches.items()
                     if action != 'base' and branch['rendered'] and branch['collision']['valid']]
            best = max(valid, key=lambda action: branches[action]['objects'].get(target, {}).get('pixels', 0), default=None)
            amodal = {}
            for action in ([] if reused or args.passive_only or args.all_isolated else ['base'] + ([best] if best else [])):
                amodal[action] = render(state['poses'][action], state['focal_length_mm'],
                                         state_dir / f'amodal_{action}', isolated=target)
            isolated_objects = {}
            requests = isolated_requests(state, branches, geometry) if args.prefilter else None
            if reused or args.all_isolated:
                for action, branch in branches.items():
                    if not branch['rendered']:
                        continue
                    isolated_objects[action] = {}
                    for role in ['target_id', 'anchor_id']:
                        object_id = state.get(role)
                        if not object_id:
                            continue
                        if requests is not None and object_id not in requests.get(action, set()):
                            continue
                        existing = reused[state['state_id']].get('amodal_target', {}).get(action) if reused and role == 'target_id' else None
                        isolated_objects[action][object_id] = existing or render(
                            state['poses'][action], state['focal_length_mm'],
                            state_dir / f'isolated_{action}_{role}', isolated=object_id)
                if reused:
                    amodal = reused[state['state_id']]['amodal_target']
            results.append({'state_id': state['state_id'], 'branches': branches,
                            'amodal_target': amodal, 'isolated_objects': isolated_objects,
                            'best_pixel_action': best})
            (args.output / 'progress.json').write_text(json.dumps({'states_done': len(results), 'renders': serial}))
    (args.output / 'results.json').write_text(json.dumps(result_payload(), indent=2))


if __name__ == '__main__':
    main()
