"""Nominate diagnostic pixel transitions; never export accepted Action QA."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


PROFILES = {
    'conservative': {'min_before': 0.0002, 'min_after': 0.005, 'min_gain': 0.003,
                     'min_ratio': 2.0, 'window_sec': 2.0},
    'moderate': {'min_before': 0.0001, 'min_after': 0.002, 'min_gain': 0.001,
                 'min_ratio': 1.5, 'window_sec': 5.0},
}


def pixels(row):
    area = int(row['pixel_area'])
    if area <= 0:
        raise ValueError('Nonpositive pixel area')
    result = {}
    for item in row['visible_objects']:
        count = int(item['pixel_count'])
        if count < 0 or count > area:
            raise ValueError('Invalid object pixel count')
        result[str(item.get('object_id') or item['object_name'])] = count / area
    return result


def nominate(rows, config):
    groups = defaultdict(list)
    stats = Counter(input_frames=len(rows))
    for row in rows:
        if row.get('render_status') != 'ok' or row.get('domain') != 'sim':
            stats['rejected_frame_status_or_domain'] += 1
            continue
        if row.get('visibility_source') != 'blender_object_index_pixel_count':
            stats['rejected_non_instance_evidence'] += 1
            continue
        groups[str(row['episode_id'])].append(row)
    output = []
    for episode, frames in sorted(groups.items()):
        frames.sort(key=lambda row: float(row['timestamp_sec']))
        keys = {(row['scene_id'], row['split'], row.get('render_contract_hash'),
                 row['pixel_area'], row.get('focal_length_mm')) for row in frames}
        if len(keys) != 1 or not frames[0].get('render_contract_hash'):
            raise ValueError(f'Mixed or missing render provenance: {episode}')
        times = [float(row['timestamp_sec']) for row in frames]
        if len(set(times)) != len(times):
            raise ValueError(f'Duplicate timestamps: {episode}')
        maps = [pixels(row) for row in frames]
        history_best = {}
        for base_index, base in enumerate(frames):
            for target, before in sorted(maps[base_index].items()):
                if before <= 0 or before < config['min_before']:
                    stats['below_before_floor'] += 1
                    continue
                choices = []
                for after_index in range(base_index + 1, len(frames)):
                    elapsed = times[after_index] - times[base_index]
                    if elapsed > config['window_sec']:
                        break
                    after = maps[after_index].get(target, 0)
                    if (after >= config['min_after'] and after - before >= config['min_gain']
                            and after / before >= config['min_ratio']):
                        choices.append((after, after_index))
                if not choices:
                    stats['no_gain_in_window'] += 1
                    continue
                after, after_index = max(choices, key=lambda item: (item[0], -item[1]))
                previous_best = history_best.get(target, 0)
                recent_best = max((maps[index].get(target, 0)
                                   for index in range(base_index + 1)
                                   if times[base_index] - times[index] <= 5), default=0)
                history_competitive = previous_best >= after * 0.8
                stats['prior_sampled_history_competitive'] += int(history_competitive)
                output.append({
                    'schema_version': 'p26_action_pixel_nomination_v0.1',
                    'status': 'diagnostic_only', 'episode_id': episode,
                    'scene_id': base['scene_id'], 'split': base['split'],
                    'target_id': target, 'base_frame': base['frame_id'],
                    'base_time_sec': times[base_index],
                    'mining_future_frame': frames[after_index]['frame_id'],
                    'mining_future_time_sec': times[after_index],
                    'before_fraction': before, 'future_fraction': after,
                    'history_best_sampled_fraction': previous_best,
                    'recent_5s_best_sampled_fraction': recent_best,
                    'history_competitive_diagnostic': history_competitive,
                    'render_contract_hash': base['render_contract_hash'],
                    'joint_visible_other_ids': sorted(
                        target_id for target_id, fraction in maps[after_index].items()
                        if target_id != target and fraction >= config['min_after']
                        and maps[base_index].get(target_id, 0) >= config['min_before']),
                    'legal_action_verified': False, 'referent_verified': False,
                    'template_answerability_verified': False,
                    'model_input_export_allowed': False,
                })
            for target, fraction in maps[base_index].items():
                history_best[target] = max(history_best.get(target, 0), fraction)
    stats.update(nominations=len(output), episodes=len(groups),
                 nominated_states=len({(row['episode_id'], row['base_frame']) for row in output}),
                 nominated_objects=len({(row['episode_id'], row['target_id']) for row in output}),
                 accepted_qa_pairs=0)
    return output, dict(stats)




