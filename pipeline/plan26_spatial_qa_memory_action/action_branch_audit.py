"""Audit pixel thresholds and export reviewable QA drafts and image cards."""

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
from qa_observability import OBJECT_DISTANCE_OPTIONS, object_distance_bucket


def evidence(branch, object_id):
    return branch.get('objects', {}).get(object_id, {'pixels': 0, 'fraction': 0})


def qualifies(before, after, minimum_before=0.0002, minimum_after=0.005, ratio=2, absolute_gain=0.003):
    return (before >= minimum_before and after >= minimum_after
            and after >= before * ratio and after - before >= absolute_gain)


def referent(state, before, object_id):
    objects = state['objects']
    if object_id not in objects or evidence(before, object_id)['fraction'] < 0.0002:
        return None
    category = objects[object_id]['category']
    peers = [name for name in before['objects'] if name in objects
             and objects[name]['category'] == category
             and evidence(before, name)['fraction'] >= 0.0002]
    if len(peers) == 1:
        return f'the {category} visible in the initial view'
    def cell(name):
        left, top, right, bottom = evidence(before, name)['bbox_xyxy']
        return min(2, int((left + right) / 2 / 640 * 3)), min(2, int((top + bottom) / 2 / 480 * 3))
    target_cell = cell(object_id)
    if sum(cell(name) == target_cell for name in peers) != 1:
        return None
    horizontal = ['left', 'center', 'right'][target_cell[0]]
    vertical = ['upper', 'middle', 'lower'][target_cell[1]]
    return f'the {category} in the {vertical} {horizontal} of the initial view'


def overlay(branch, id_to_name, selected):
    from PIL import Image, ImageDraw
    image = Image.open(branch['rgb']).convert('RGB')
    labels = np.load(branch['mask'])['labels']
    array = np.array(image).astype(float)
    colors = [(255, 70, 70), (40, 220, 255)]
    name_to_id = {name: int(index) for index, name in id_to_name.items()}
    for name, color in zip(selected, colors):
        if name is None or name not in name_to_id:
            continue
        mask = labels == name_to_id[name]
        array[mask] = array[mask] * 0.5 + np.array(color) * 0.5
    result = Image.fromarray(np.uint8(array))
    draw = ImageDraw.Draw(result)
    for name, color in zip(selected, colors):
        value = evidence(branch, name)
        if value.get('bbox_xyxy'):
            draw.rectangle(value['bbox_xyxy'], outline=color, width=2)
    return result


def card(path, state, result, rendered):
    from PIL import Image, ImageDraw
    action = result['best_pixel_action']
    if action is None:
        return
    before = result['branches']['base']
    after = result['branches'][action]
    canvas = Image.new('RGB', (1280, 1040), 'white')
    canvas.paste(Image.open(before['rgb']).convert('RGB'), (0, 80))
    canvas.paste(Image.open(after['rgb']).convert('RGB'), (640, 80))
    selected = [state['target_id'], state['anchor_id']]
    canvas.paste(overlay(before, rendered['id_to_name'], selected), (0, 560))
    canvas.paste(overlay(after, rendered['id_to_name'], selected), (640, 560))
    draw = ImageDraw.Draw(canvas)
    target = state['target_id']
    draw.text((12, 10), f'{state["state_id"]} | frame {state["frame_id"]} | target(red): {target} | anchor(cyan): {state["anchor_id"]}', fill='black')
    draw.text((12, 30), f'BEFORE {evidence(before,target)["pixels"]}px   -->   {action}   AFTER {evidence(after,target)["pixels"]}px', fill='black')
    draw.text((12, 50), 'Top: unmodified RGB. Bottom: visible instance masks and boxes. Pixel gain is not QA acceptance.', fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if not (args.run / 'render.complete.json').exists():
        raise RuntimeError('Render incomplete')
    output = args.run / 'audit'
    output.mkdir(exist_ok=False)
    states = {state['state_id']: state for state in json.loads((args.run / 'states.json').read_text())}
    audits = []
    drafts = []
    outcomes = Counter()
    template_counts = Counter()
    for path in sorted((args.run / 'scenes').glob('*/results.json')):
        rendered = json.loads(path.read_text())
        for result in rendered['states']:
            state = states[result['state_id']]
            target, anchor = state['target_id'], state['anchor_id']
            before = result['branches']['base']
            target_ref = referent(state, before, target)
            anchor_ref = referent(state, before, anchor)
            center_errors = {}
            for object_id in [target, anchor]:
                if object_id and object_id in rendered['geometry']:
                    center_errors[object_id] = math.dist(state['objects'][object_id]['center'],
                                                          rendered['geometry'][object_id]['center'])
            geometry_ok = target in center_errors and center_errors[target] < 0.05
            pair_geometry_ok = geometry_ok and anchor in center_errors and center_errors[anchor] < 0.05
            branch_audits = []
            for action, branch in result['branches'].items():
                if action == 'base':
                    continue
                if not branch['rendered'] or not branch['collision']['valid']:
                    outcomes['collision_rejected'] += 1
                    continue
                before_fraction = evidence(before, target)['fraction']
                after_fraction = evidence(branch, target)['fraction']
                passed = qualifies(before_fraction, after_fraction)
                outcomes['legal_rendered_actions'] += 1
                outcomes['default_pixel_pass_actions'] += int(passed)
                complete = {}
                for role in ['base', action]:
                    if role in result['amodal_target']:
                        isolated = evidence(result['amodal_target'][role], target)['pixels']
                        visible = evidence(result['branches'][role], target)['pixels']
                        complete[role] = visible / isolated if isolated else None
                branch_audits.append({'action': action, 'before_fraction': before_fraction,
                    'after_fraction': after_fraction, 'pixel_pass': passed,
                    'before_border': evidence(before, target).get('touches_border'),
                    'after_border': evidence(branch, target).get('touches_border'),
                    'visible_over_in_frame_amodal': complete,
                    'target_ref': target_ref, 'anchor_ref': anchor_ref,
                    'anchor_after_fraction': evidence(branch, anchor)['fraction']})
            valid = [entry for entry in branch_audits if entry['pixel_pass']]
            outcomes['states_with_pixel_pass'] += int(bool(valid))
            audits.append({'state_id': state['state_id'], 'target_id': target,
                           'center_error_m': center_errors, 'branches': branch_audits,
                           'best_pixel_action': result['best_pixel_action']})
            card(output / f'{state["state_id"]}.jpg', state, result, rendered)
            for template in ['object_size_m', 'object_to_object_distance_m',
                             'object_to_object_distance_bucket', 'object_spatial_proximity']:
                pair = template != 'object_size_m'
                if not target_ref or not geometry_ok or (pair and (not anchor_ref or not pair_geometry_ok)):
                    continue
                actions = [entry for entry in valid if not pair or entry['anchor_after_fraction'] >= 0.002]
                if not actions:
                    continue
                best = max(actions, key=lambda entry: entry['after_fraction'])
                if pair:
                    distance = math.dist(state['objects'][target]['center'], state['objects'][anchor]['center'])
                    answer = round(distance, 3) if template.endswith('_m') else object_distance_bucket(distance)
                    if answer is None:
                        outcomes['distance_bin_gap'] += 1
                        continue
                    question = (f'What is the 3D center-to-center distance between {target_ref} and {anchor_ref}, in meters?'
                                if template.endswith('_m') else
                                f'Which 3D center-to-center distance range describes {target_ref} and {anchor_ref}?')
                else:
                    answer = round(state['objects'][target]['size_m'], 3)
                    question = f'What is the longest physical side of the 3D bounding box of {target_ref}, in meters?'
                selected_branch = result['branches'][best['action']]
                drafts.append({'state_id': state['state_id'], 'template_id': template,
                    'split': 'test', 'status': 'pixel_and_geometry_pass_visual_review_required',
                    'model_before': {'question': question, 'rgb': before['rgb']},
                    'supervision': {'action': best['action'], 'answer': answer,
                                   'options': list(OBJECT_DISTANCE_OPTIONS) if not template.endswith('_m') else None},
                    'action_result': {'rgb': selected_branch['rgb']},
                    'audit': {'target_id': target, 'anchor_id': anchor if pair else None,
                              'geometry_definition': 'physical_local_bbox_longest_side' if not pair else 'world_center_euclidean',
                              'pixel_metrics': best, 'center_error_m': center_errors,
                              'memory_gate_applied': False, 'metric_scale_verified': False,
                              'visual_referent_verified': False, 'answerable_before': None,
                              'answerable_after': None, 'training_eligible': False,
                              'semantic_family': 'object_distance_bucket' if not template.endswith('_m') else template}})
                template_counts[template] += 1
    sweep = []
    for minimum_before, minimum_after, ratio in itertools.product([0.0002, 0.0005, 0.001], [0.002, 0.005, 0.01], [1.5, 2, 3]):
        passing = [(row['state_id'], branch['action']) for row in audits for branch in row['branches']
                   if qualifies(branch['before_fraction'], branch['after_fraction'],
                                minimum_before, minimum_after, ratio, 0.001)]
        sweep.append({'minimum_before': minimum_before, 'minimum_after': minimum_after,
                      'ratio': ratio, 'minimum_absolute_gain': 0.001,
                      'actions': len(passing), 'states': len({pair[0] for pair in passing})})
    summary = {'states': len(states), 'counts': dict(outcomes), 'qa_drafts': len(drafts),
               'templates': dict(template_counts), 'accepted_training_qa': 0,
               'memory_gate_applied': False, 'threshold_sweep': sweep,
               'default_thresholds': {'before': 0.0002, 'after': 0.005, 'ratio': 2, 'absolute_gain': 0.003},
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name, value in [('summary', summary), ('state_audits', audits)]:
        (output / f'{name}.json').write_text(json.dumps(value, indent=2))
    with (output / 'qa_drafts.jsonl').open('w') as handle:
        for row in drafts:
            handle.write(json.dumps(row) + '\n')
    cards = ''.join(f'<p>{row["state_id"]}</p><img loading="lazy" width="1280" src="{row["state_id"]}.jpg">'
                    for row in audits if row['best_pixel_action'])
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P26 Action probe</title>'
        '<h1>Action-only probe: pixel evidence and QA drafts</h1><p>Red: target. Cyan: anchor. '
        'Top: RGB. Bottom: instance mask overlay. Not accepted training data.</p>' + cards)
    print(json.dumps({key: value for key, value in summary.items() if key != 'threshold_sweep'}, indent=2))


if __name__ == '__main__':
    main()
