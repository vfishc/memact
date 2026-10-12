"""Necessary conditions for C-template mining; never relax evidence gates."""

import math

from plan26_spatial_qa_memory_action.action_branch_audit import referent
from plan26_spatial_qa_memory_action.action_evidence_rules import CONFIG, framing


def object_framing(branch, geometry, object_id):
    if object_id not in geometry:
        return None
    result = framing(geometry[object_id]['world_obb_corners'], branch['pose'],
                     branch['K'], branch['resolution'])
    return result['fraction'] if result['status'] == 'ok' else None


def before_possible(state, base, geometry, object_id):
    if not object_id or object_id not in state['objects'] or object_id not in geometry:
        return False
    return bool(base['objects'].get(object_id, {}).get('fraction', 0) >= CONFIG['before_p']
                and referent(state, base, object_id)
                and object_framing(base, geometry, object_id) is not None
                and math.dist(state['objects'][object_id]['center'],
                              geometry[object_id]['center']) < 0.05)


def after_possible(base, branch, geometry, object_id, require_pixels=True):
    before_f = object_framing(base, geometry, object_id)
    after_f = object_framing(branch, geometry, object_id)
    return bool(before_f is not None and after_f is not None
                and after_f >= CONFIG['after_f']
                and after_f >= before_f - CONFIG['nonregression']
                and (not require_pixels or
                     branch['objects'].get(object_id, {}).get('fraction', 0) >= CONFIG['after_p']))


def isolated_requests(state, branches, geometry):
    base = branches['base']
    target = state['target_id']
    anchor = state['anchor_id']
    requests = {}
    if not before_possible(state, base, geometry, target):
        return requests
    for action, branch in branches.items():
        if action == 'base' or not branch.get('rendered') or not branch['collision']['valid']:
            continue
        if not after_possible(base, branch, geometry, target):
            continue
        requests.setdefault('base', set()).add(target)
        requests.setdefault(action, set()).add(target)
        if before_possible(state, base, geometry, anchor) and after_possible(base, branch, geometry, anchor):
            requests['base'].add(anchor)
            requests[action].add(anchor)
    return requests
