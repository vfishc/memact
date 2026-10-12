"""Shared bounded candidate selection for separate and resident render workers."""

from collections import Counter
import hashlib
import json
import math

from plan26_spatial_qa_memory_action.camera_transitions import action_poses
from plan26_spatial_qa_memory_action.action_branch_audit import referent
from plan26_spatial_qa_memory_action.pixel_evidence import nominate, PROFILES


def select_states(observations, passive, objects, expand_pairs=False):
    by_frame = {state['frame_id']: result['branches']['base'] for state, result in zip(observations, passive['states'])}
    trajectory = []
    for state in observations:
        observation = by_frame[state['frame_id']]
        environment = state['source_render_contract']
        trajectory.append({**state, 'domain': 'sim', 'render_status': 'ok',
            'visibility_source': 'blender_object_index_pixel_count', 'pixel_area': 640 * 480,
            'render_contract_hash': hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest(),
            'visible_objects': [{'object_id': object_id, 'pixel_count': value['pixels']}
                                for object_id, value in observation['objects'].items() if object_id in objects]})
    nominations, counts = nominate(trajectory, PROFILES['moderate'])
    nominations.sort(key=lambda row: (-(row['future_fraction'] - row['before_fraction']), row['base_frame'], row['target_id']))
    used = Counter()
    frame_set = set()
    selected = []
    source_states = {state['frame_id']: state for state in observations}
    for candidate in nominations:
        frame_id, target = candidate['base_frame'], candidate['target_id']
        if used[target] >= 2 or len(frame_set) >= 10 and frame_id not in frame_set:
            continue
        if sum(row['frame_id'] == frame_id for row in selected) >= 4:
            continue
        before = by_frame[frame_id]
        if before['objects'][target]['fraction'] < 0.0005:
            continue
        anchors = [object_id for object_id, value in before['objects'].items()
                   if object_id != target and object_id in objects and value['fraction'] >= 0.0005]
        anchors.sort(key=lambda name: (-before['objects'][name]['fraction'], name))
        frame_set.add(frame_id)
        used[target] += 1
        for anchor in anchors[:2] or [None]:
            state = source_states[frame_id]
            state_id = hashlib.sha256(f'{state["episode_id"]}:{frame_id}:{target}:{anchor}'.encode()).hexdigest()[:20]
            selected.append({**state, 'state_id': state_id, 'split': 'train',
                'target_id': target, 'anchor_id': anchor, 'objects': objects,
                'poses': action_poses(before['pose'])})
    if expand_pairs:
        extra = []
        visited = set()
        existing = {(state['frame_id'], state['target_id'], state['anchor_id']) for state in selected}
        for state in selected:
            frame_id, target = state['frame_id'], state['target_id']
            if (frame_id, target) in visited:
                continue
            visited.add((frame_id, target))
            before = by_frame[frame_id]
            if not referent(state, before, target):
                continue
            anchors = [name for name, value in before['objects'].items()
                       if name != target and name in objects and value['fraction'] >= 0.0005
                       and (frame_id, target, name) not in existing and referent(state, before, name)]
            anchors.sort(key=lambda name: (math.dist(objects[target]['center'], objects[name]['center']), name))
            for anchor in anchors[:2]:
                state_id = hashlib.sha256(f'{state["episode_id"]}:{frame_id}:{target}:{anchor}'.encode()).hexdigest()[:20]
                extra.append({**state, 'state_id': state_id, 'anchor_id': anchor})
        selected.extend(extra)
    return selected, counts, frame_set
