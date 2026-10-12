"""Apply a frozen question mask identically to streaming and offline schedules."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from joint_episode import load_episode


def subset_schedule(schedule, answers, samples):
    samples=set(samples)
    groups=defaultdict(list)
    by_index={}
    for a in answers:
        index=a['schedule_index']
        if index in by_index or index>=len(schedule) or index<0 or schedule[index]['kind']!='question':
            raise ValueError('invalid answer/question mapping')
        by_index[index]=a;groups[a['sample']].append(a)
    if samples-groups.keys():raise ValueError('unknown selected sample')
    if {i for i,e in enumerate(schedule) if e['kind']=='question'}!=set(by_index):
        raise ValueError('unmapped question')
    for group in groups.values():
        turns=[a['turn'] for a in sorted(group,key=lambda a:a['schedule_index'])]
        if turns!=list(range(len(turns))):raise ValueError('incomplete chain')
    new_schedule=[];new_answers=[];indices=[]
    for i,event in enumerate(schedule):
        if event['kind']=='question':
            answer=by_index[i]
            if answer['sample'] not in samples:continue
            new_answers.append({**answer,'schedule_index':len(new_schedule)})
            indices.append(i)
        new_schedule.append(event)
    return new_schedule,new_answers,indices


def load_selected_episode(row):
    root=Path(row.get('root',row.get('destination')))
    data=load_episode(root)
    if row.get('action_graph_bundle'):
        from joint_episode import load_episode as load_graph_bundle
        bundle_root = Path(row['action_graph_bundle'])
        if not bundle_root.is_absolute():
            bundle_root = root / bundle_root
        extra = load_graph_bundle(bundle_root)
        # Imported graph images use a separate namespace; original episode
        # history (including its codec and frame hashes) remains unchanged.
        graph = extra['action_graph']
        remap = {}
        prefix = 'f' + str(int(hashlib.sha256(str(bundle_root.resolve()).encode()).hexdigest()[:16],16))
        for i, frame in enumerate(sorted({n['frame'] for n in graph['nodes'].values()})):
            new = prefix + str(i).zfill(6)
            if new in data['media']:raise ValueError('action frame namespace collision')
            data['media'][new] = extra['media'][frame]
            remap[frame] = new
        for node in graph['nodes'].values():node['frame'] = remap[node['frame']]
        if row.get('action_question_roots') is not None:
            graph['question_roots'] = row['action_question_roots']
        data['action_graph'] = graph
    selection=row.get('evaluation_selection')
    if not selection:return data
    if selection['schema']!='sparse_test_v2':raise ValueError('unknown selection schema')
    for name,digest in selection['source_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('frozen source changed')
    schedule,answers,indices=subset_schedule(data['schedule'],data['answers'],selection['samples'])
    if indices!=selection['question_indices'] or len(answers)!=selection['qa_turns']:
        raise ValueError('frozen selection mismatch')
    if data.get('action_graph', {}).get('question_roots'):
        old_roots = data['action_graph']['question_roots']
        data['action_graph']['question_roots'] = {
            str(a['schedule_index']): old_roots[str(old)]
            for old, a in zip(indices, answers) if str(old) in old_roots}
    data={**data,'schedule':schedule,'answers':answers}
    return data
