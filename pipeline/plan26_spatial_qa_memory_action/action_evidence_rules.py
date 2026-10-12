"""Versioned Action-only evidence filters and review-only causal QA bundles."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from camera_geometry import euler_xyz_matrix
from plan26_spatial_qa_memory_action.action_gain_geometry import area, clip_rectangle, convex_hull
from plan26_spatial_qa_memory_action.action_branch_audit import referent
from qa_observability import OBJECT_DISTANCE_OPTIONS, object_distance_bucket
from plan26_spatial_qa_memory_action.action_size_truth import mesh_size_truth

CONFIG = {'version': 'p26_action_evidence_proposed_v0.2', 'before_p': 0.0005,
          'after_p': 0.005, 'after_f': 0.8, 'after_o': 0.7, 'ratio': 2.0,
          'absolute_gain': 0.003, 'coverage_gain': 0.2, 'nonregression': 0.1,
          'near_m': 0.01, 'memory_gate': False}


def framing(corners, pose, intrinsic, resolution, near=0.01):
    rotation = np.asarray(euler_xyz_matrix(pose['rotation_euler_deg']))
    local = (np.asarray(corners, dtype=float) - np.asarray(pose['location'])) @ rotation
    depth = -local[:, 2]
    if np.max(depth) <= near:
        return {'status': 'behind_near_plane', 'fraction': None}
    clipped = bool(np.min(depth) <= near)
    points = [point for point in local if -point[2] >= near]
    if clipped:
        for first, second in itertools.combinations(local, 2):
            if (-first[2] < near) != (-second[2] < near):
                fraction = (-near - first[2]) / (second[2] - first[2])
                points.append(first + fraction * (second - first))
    projected = [(intrinsic[0][2] + intrinsic[0][0] * point[0] / -point[2],
                  intrinsic[1][2] - intrinsic[1][1] * point[1] / -point[2]) for point in points]
    hull = convex_hull(projected)
    full = area(hull)
    inside = area(clip_rectangle(hull, *resolution))
    return {'status': 'near_clipped_diagnostic' if clipped else 'ok',
            'fraction': max(0.0, min(1.0, inside / full)) if full else None,
            'full_area': full, 'inside_area': inside, 'source': 'transformed_mesh_local_bbox'}


def metric(state, branch, isolated, geometry, object_id):
    if not object_id or object_id not in geometry:
        return None
    visible = branch['objects'].get(object_id, {})
    denominator = isolated.get(object_id, {}).get('objects', {}).get(object_id, {}).get('pixels', 0)
    count = visible.get('pixels', 0)
    projection = framing(geometry[object_id]['world_obb_corners'], branch['pose'], branch['K'], branch['resolution'])
    occlusion = count / denominator if denominator else None
    status = 'ok' if occlusion is not None and occlusion <= 1.01 else 'invalid_or_missing_silhouette'
    return {'p': visible.get('fraction', 0),
            'f': projection['fraction'] if projection['status'] == 'ok' else None,
            'o': min(1.0, occlusion) if status == 'ok' else None,
            'occlusion_status': status, 'framing': projection,
            'pixels': count, 'isolated_pixels': denominator,
            'border': visible.get('touches_border'),
            'range_m': math.dist(state['objects'][object_id]['center'], branch['pose']['location'])}


def quality(value, config):
    return bool(value and value['p'] >= config['after_p'] and value['f'] is not None
                and value['f'] >= config['after_f'] and value['o'] is not None and value['o'] >= config['after_o'])


def decide(before, after, config=CONFIG):
    if not before or not after:
        return {'A': False, 'B': False, 'reasons': ['missing_object_evidence']}
    present = before['p'] >= config['before_p']
    pixel_gain = (after['p'] >= before['p'] * config['ratio']
                  and after['p'] - before['p'] >= config['absolute_gain'])
    rule_a = present and after['p'] >= config['after_p'] and pixel_gain
    coverage_gain = any(before[key] is not None and after[key] is not None
                        and after[key] - before[key] >= config['coverage_gain'] for key in ['f', 'o'])
    nonregression = all(before[key] is None or after[key] is None
                        or after[key] >= before[key] - config['nonregression'] for key in ['f', 'o'])
    rule_b = present and quality(after, config) and (pixel_gain or coverage_gain) and nonregression
    reasons = []
    if not present:
        reasons.append('before_below_identification_nomination_floor')
    for key, threshold in [('p', config['after_p']), ('f', config['after_f']), ('o', config['after_o'])]:
        if after[key] is None:
            reasons.append(f'after_{key}_unknown')
        elif after[key] < threshold:
            reasons.append(f'after_{key}_insufficient')
    if not pixel_gain and not coverage_gain:
        reasons.append('no_significant_gain')
    if not nonregression:
        reasons.append('coverage_regression')
    return {'A': bool(rule_a), 'B': bool(rule_b), 'before_quality': quality(before, config),
            'pixel_gain': pixel_gain, 'coverage_gain': coverage_gain, 'reasons': reasons}


def export_bundle(state, result, action, template, target_ref, anchor_ref, geometry=None):
    target = state['target_id']
    anchor = state['anchor_id']
    pair = template != 'object_size_m'
    if pair:
        distance = math.dist(state['objects'][target]['center'], state['objects'][anchor]['center'])
        answer = round(distance, 3) if template.endswith('_m') else object_distance_bucket(distance)
        question = (f'What is the 3D center-to-center distance between {target_ref} and {anchor_ref}, in meters?'
                    if template.endswith('_m') else
                    f'Which 3D center-to-center distance range describes {target_ref} and {anchor_ref}?')
    else:
        if geometry is None:
            raise ValueError('Size QA requires rendered mesh geometry')
        answer = round(mesh_size_truth(geometry[target])['longest_side_m'], 3)
        question = f'What is the longest physical side of the 3D bounding box of {target_ref}, in meters?'
    if answer is None:
        return None
    action_args = {'angle_deg': 45} if action.startswith('rotate_') else {'distance_m': 0.5}
    base = result['branches']['base']
    after = result['branches'][action]
    row_id = hashlib.sha256(f'{state["state_id"]}:{template}'.encode()).hexdigest()[:20]
    options = None if template.endswith('_m') else list(OBJECT_DISTANCE_OPTIONS)
    return {'sample_id': row_id, 'state_id': state['state_id'], 'split': state['split'],
            'template_id': template, 'training_eligible': False,
            'status': 'evidence_pass_pending_visual_and_scale_review',
            'events': [
                {'type': 'observation', 'media_ref': base['rgb']},
                {'type': 'qa_turn', 'question': question, 'options': options},
                {'type': 'assistant_action', 'name': action, 'arguments': action_args},
                {'type': 'action_result', 'media_ref': after['rgb']},
                {'type': 'assistant_answer', 'answer': answer}],
            'audit': {'episode_id': state['episode_id'], 'frame_id': state['frame_id'],
                      'timestamp_sec': state['timestamp_sec'], 'target_id': target,
                      'anchor_id': anchor if pair else None,
                      'geometry_truth_only': True, 'scale_verified': False,
                      'referent_visual_verified': False, 'before_answerability': 'unreviewed',
                      'after_answerability': 'unreviewed', 'memory_gate_applied': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / 'render.complete.json').exists():
        raise RuntimeError('Missing render completion receipt')
    args.output.mkdir(parents=True, exist_ok=False)
    states = {row['state_id']: row for row in json.loads((args.run / 'states.json').read_text())}
    records = []
    bundles = []
    statistics = Counter()
    rule_states = {rule: set() for rule in ['A', 'B', 'C_size', 'C_pair']}
    rule_actions = Counter()
    prefiltered = False
    skipped_before_render = 0
    for path in sorted((args.run / 'scenes').glob('*/results.json')):
        rendered = json.loads(path.read_text())
        prefiltered = prefiltered or rendered.get('optimizations', {}).get('prefilter', False)
        for result in rendered['states']:
            skipped_before_render += sum('prefilter' in branch for branch in result['branches'].values())
            state = states[result['state_id']]
            before_branch = result['branches']['base']
            target_ref = referent(state, before_branch, state['target_id'])
            anchor_ref = referent(state, before_branch, state['anchor_id'])
            options = []
            for action, branch in result['branches'].items():
                if action == 'base' or not branch['rendered'] or not branch['collision']['valid']:
                    continue
                metrics = {}
                for role in ['target_id', 'anchor_id']:
                    object_id = state[role]
                    metrics[role] = {name: metric(state, observation,
                        result['isolated_objects'].get(key, {}), rendered['geometry'], object_id)
                        for name, key, observation in [('before', 'base', before_branch), ('after', action, branch)]}
                before, after = metrics['target_id']['before'], metrics['target_id']['after']
                decision = decide(before, after)
                anchor_before, anchor_after = metrics['anchor_id']['before'], metrics['anchor_id']['after']
                deficient_before = before is not None and before['f'] is not None and before['o'] is not None and not quality(before, CONFIG)
                size_pass = decision['B'] and bool(target_ref) and deficient_before
                anchor_nonregression = bool(anchor_before and anchor_after and all(
                    anchor_before[key] is not None and anchor_after[key] is not None
                    and anchor_after[key] >= anchor_before[key] - CONFIG['nonregression'] for key in ['f', 'o']))
                pair_pass = (decision['B'] and bool(target_ref and anchor_ref) and anchor_before is not None
                             and anchor_before['p'] >= CONFIG['before_p'] and quality(anchor_after, CONFIG)
                             and anchor_nonregression and deficient_before)
                geometry_errors = {object_id: math.dist(state['objects'][object_id]['center'],
                    rendered['geometry'][object_id]['center']) for object_id in [state['target_id'], state['anchor_id']]
                    if object_id and object_id in rendered['geometry']}
                size_pass = size_pass and geometry_errors.get(state['target_id'], 1) < 0.05
                pair_pass = pair_pass and all(geometry_errors.get(object_id, 1) < 0.05
                                               for object_id in [state['target_id'], state['anchor_id']])
                row = {'state_id': state['state_id'], 'action': action, 'metrics': metrics,
                       'rules': {**decision, 'C_size': bool(size_pass), 'C_pair': bool(pair_pass)},
                       'geometry_errors_m': geometry_errors,
                       'meaning': 'C is evidence proxy, not final answerability or metric-scale acceptance'}
                records.append(row)
                options.append(row)
                statistics.update(decision['reasons'])
                for rule in rule_states:
                    if row['rules'][rule]:
                        rule_states[rule].add(state['state_id'])
                        rule_actions[rule] += 1
            for template in ['object_size_m', 'object_to_object_distance_m',
                             'object_to_object_distance_bucket', 'object_spatial_proximity']:
                valid = [row for row in options if row['rules']['C_size' if template == 'object_size_m' else 'C_pair']]
                if not valid:
                    continue
                chosen = max(valid, key=lambda row: (
                    min(row['metrics'][role]['after'][key] for role in
                        (['target_id'] if template == 'object_size_m' else ['target_id', 'anchor_id'])
                        for key in ['f', 'o']), row['metrics']['target_id']['after']['p']))
                bundle = export_bundle(state, result, chosen['action'], template, target_ref, anchor_ref, rendered['geometry'])
                if bundle:
                    if template == 'object_size_m':
                        bundle['audit']['size_truth'] = mesh_size_truth(rendered['geometry'][state['target_id']])
                    bundle['audit']['acceptable_evidence_actions'] = [row['action'] for row in valid]
                    bundles.append(bundle)
    grid = []
    for minimum_f, minimum_o in itertools.product([0.6, 0.8, 0.9], [0.5, 0.7, 0.9]):
        config = {**CONFIG, 'after_f': minimum_f, 'after_o': minimum_o}
        passing = [row for row in records if decide(row['metrics']['target_id']['before'], row['metrics']['target_id']['after'], config)['B']]
        grid.append({'f': minimum_f, 'o': minimum_o, 'B_actions': len(passing),
                     'B_states': len({row['state_id'] for row in passing})})
    summary = {'config': CONFIG, 'states': len(states), 'legal_branches': len(records),
               'diagnostic_scope': 'C_candidate_prefiltered' if prefiltered else 'all_rendered_branches',
               'skipped_before_render': skipped_before_render,
               'rules': {rule: {'actions': rule_actions[rule], 'states': len(values)} for rule, values in rule_states.items()},
               'rejections': dict(statistics), 'review_bundles': len(bundles),
               'templates': dict(Counter(row['template_id'] for row in bundles)),
               'accepted_training_pairs': 0, 'grid': grid,
               'input_sha256': hashlib.sha256((args.run / 'states.json').read_bytes()).hexdigest(),
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'rule_comparisons.json').write_text(json.dumps(records, indent=2))
    with (args.output / 'review_bundles.jsonl').open('w') as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle) + '\n')
    (args.output / '.complete.json').write_text(json.dumps({'status': 'probe_complete', 'formal_acceptance': False}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
