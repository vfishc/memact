"""Convert rendered task records into the private training graph.

Answers and evidence come from the task binding, never the source graph's
original question. Public observations contain no geometry or reward fields.
"""
import json
from pathlib import Path
from core import Graph, Config


def load_task(path, path_map=(), check_images=True):
    path = Path(path)
    task = json.loads(path.read_text())

    def resolve(value):
        for old, new in path_map:
            if value == old or value.startswith(old.rstrip('/') + '/'):
                value = new.rstrip('/') + value[len(old.rstrip('/')):]
                break
        p = Path(value)
        return p if p.is_absolute() else path.parent / p

    raw = json.loads(resolve(task['graph']).read_text())
    targets = task['targets']

    def node(image, evidence, frame=None, timestamp=None):
        image = resolve(image)
        if check_images and not image.is_file():
            raise FileNotFoundError(image)
        obs = {'image': str(image)}
        if frame is not None:
            obs['frame'] = frame
        if timestamp is not None:
            obs['timestamp'] = timestamp
        measurements = {}
        for obj in targets:
            m = evidence['objects'][obj]
            p, f = m['p'], m['f']
            # A measured zero pixel count has zero reward in every variant.
            # Keep unavailable framing unavailable in source provenance.
            if f is None:
                if p != 0:
                    raise ValueError('Visible target has unavailable framing')
                f = 0.0
            measurements[obj] = {'pixel_fraction': p, 'framing': f}
        return {'observation': obs, 'object_evidence': measurements}

    nodes = {key: node(n['image'], task['node_evidence'][key])
             for key, n in raw['nodes'].items()}
    history = []
    if len(task['history']) != len(raw['history_frames']):
        raise ValueError('History frame mapping length mismatch')
    last_time = None
    for i, (h, meta) in enumerate(zip(task['history'], raw['history_frames'])):
        timestamp = meta['timestamp_sec']
        if last_time is not None and timestamp < last_time:
            raise ValueError('History is not chronological')
        last_time = timestamp
        key = '__history_' + str(i)
        if key in nodes:
            raise ValueError('History key collision')
        nodes[key] = node(h['image'], h['evidence'], meta['frame_id'], timestamp)
        history.append(key)
    question = task['question']
    if task.get('options'):
        question += '\n' + '\n'.join(k + '. ' + v for k, v in task['options'].items())
        if task['answer'] not in task['options']:
            raise ValueError('Answer is not an option letter')
    graph = Graph(dict(root=raw['root'], question=question, answer=task['answer'],
                       required_objects=targets, nodes=nodes, history=history,
                       edges=raw['edges']))
    graph.preflight(Config(reward_mode='historical'))
    return graph
