"""Render a complete bounded camera graph and export portable evaluation/RL/SFT data.

Input JSON: state (renderer state), history (causal renderer states), assets
(scene directory parent), and question (template, targets, referents). Paths are
relative to the input file. No source-machine directories are assumed.
"""
import argparse, copy, gzip, hashlib, json, math, os, shutil, subprocess, sys, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action')]
from camera_transitions import action_poses
from action_graph import COMMANDS, GraphActionExecutor
from action_evidence_rules import metric
from action_size_truth import mesh_size_truth
from qa_observability import OBJECT_DISTANCE_OPTIONS, object_distance_bucket
from action_branch_audit import referent
from qa_eight_contract import full_instruction
from routed_stream import ACTION_SYSTEM_INSTRUCTION, check_public


def read(p): return json.loads(Path(p).read_text())
def save(p, value):
    p=Path(p); tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def key(pose):
    # Only numerical roundoff is merged, not action tolerances.
    p={k:[round(float(v),9) for v in vals] for k,vals in pose.items()}
    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:20]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def generate(config_path, output, blender, gpu=0, steps=3, resume=False):
    config_path=Path(config_path).resolve();out=Path(output).resolve();cfg=read(config_path)
    def resolve(p):
        p=Path(p);return p if p.is_absolute() else (config_path.parent/p).resolve()
    state=copy.deepcopy(cfg['state']);assets=resolve(cfg['assets'])
    for k in ('floor_texture','wall_texture','hdri'):
        state['source_render_contract'][k]=str(resolve(state['source_render_contract'][k]))
        expected=state['source_render_contract'].get(k+'_sha256')
        if expected and sha(state['source_render_contract'][k])!=expected:raise ValueError('asset hash mismatch: '+k)
    settings=state['source_render_contract']['actual_render_settings']
    if settings.get('width',640)!=640 or settings.get('height',480)!=480:raise ValueError('renderer requires 640x480')
    question=cfg['question'];targets=question['targets']
    if not 1<=len(targets)<=2 or len(set(targets))!=len(targets):raise ValueError('one or two distinct targets required')
    state['target_id']=targets[0];state['anchor_id']=targets[1] if len(targets)==2 else None
    fingerprint={'config':sha(config_path),'steps':steps,'renderer':sha(ROOT/'pipeline/plan26_spatial_qa_memory_action/action_branch_render.py'),
                 'scene':sha(assets/state['scene_id']/'front3d_scene.glb')}
    if out.exists():
        previous=read(out/'input_receipt.json');previous.pop('generator',None)
        if not resume or previous!=fingerprint:raise ValueError('output exists or resume inputs changed')
    else:
        out.mkdir(parents=True);save(out/'input_receipt.json',fingerprint)
    # A unique attempt directory preserves interrupted render outputs. Completed
    # layer receipts are immutable and reused after every referenced file is checked.
    def render(states, name, passive=False):
        receipt=out/(name+'.complete.json')
        if receipt.exists():
            r=read(receipt)
            for p,h in r['files'].items():
                if sha(out/p)!=h:raise ValueError('cached render changed: '+p)
            cached_states=(out/r['result']).parent.parent/'states.json'
            if read(cached_states)!=states:raise ValueError('cached camera requests differ')
            return read(out/r['result'])
        d=out/(name+'_'+uuid.uuid4().hex[:8]);d.mkdir();save(d/'states.json',states)
        rd=d/'render';rd.mkdir()
        cmd=[str(blender),'-b','--threads','4','--python-exit-code','1','--python',str(ROOT/'pipeline/plan26_spatial_qa_memory_action/action_branch_render.py'),
             '--','--states',str(d/'states.json'),'--assets',str(assets),'--output',str(rd),'--gpu',str(gpu),
             '--samples',str(settings['samples']),'--bounces',str(settings['bounces']),'--exposure',str(settings['exposure']),
             '--fixed-frame','--all-isolated','--mask-only-isolated']
        if settings['denoise']:cmd.append('--denoise')
        if passive:cmd.append('--passive-only')
        env=dict(os.environ,TMPDIR=str(out/'tmp'));(out/'tmp').mkdir(exist_ok=True)
        with (d/'render.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        result=rd/'results.json';value=read(result)
        files={str(p.relative_to(out)):sha(p) for p in d.rglob('*') if p.is_file() and p.name!='render.log'}
        save(receipt,{'result':str(result.relative_to(out)),'files':files})
        return value
    def node(s,res,geometry,action):
        b=res['branches'][action]
        ms={j:metric(s,b,res['isolated_objects'].get(action,{}),geometry,j) for j in targets}
        if any(v is None for v in ms.values()):raise ValueError('missing required geometry')
        return {'image':str(Path(b['rgb']).relative_to(out)),'pose':b['pose'],'metrics':ms}
    history=[];history_frames=[];last=-float('inf')
    for i,h in enumerate(cfg['history']):
        t=h['timestamp_sec']
        if t<=last or t>state['timestamp_sec']:raise ValueError('noncausal history')
        last=t;s=copy.deepcopy(state);s.update(state_id='history_'+str(i),poses={'base':h['poses']['base']},frame_id=h['frame_id'],timestamp_sec=t)
        history.append(s);history_frames.append({'frame_id':h['frame_id'],'timestamp_sec':t})
    if not history or key(history[-1]['poses']['base'])!=key(state['poses']['base']) or last!=state['timestamp_sec']:
        raise ValueError('history must end at the question anchor')
    hr=render(history,'history',True)
    hist=[node(s,r,hr['geometry'],'base') for s,r in zip(history,hr['states'])]
    root=key(state['poses']['base']);nodes={};edges=[];done=set();frontier={root:state['poses']['base']}
    for depth in range(steps):
        states=[]
        for k,p in sorted(frontier.items()):
            if k not in done:
                s=copy.deepcopy(state);s.update(state_id=k,poses=action_poses(p));states.append(s)
        if not states:break
        rendered=render(states,'depth_'+str(depth));nxt={}
        for s,r in zip(states,rendered['states']):
            if s['state_id']!=r['state_id']:raise ValueError('renderer state mismatch')
            source=s['state_id'];done.add(source);nodes.setdefault(source,node(s,r,rendered['geometry'],'base'))
            for c in COMMANDS:
                name=c['name'];b=r['branches'][name];valid=b['collision']['valid']
                target=key(b['pose']) if valid else source
                edges.append(dict(source=source,target=target,action=name,valid=bool(valid)))
                if valid:
                    nodes.setdefault(target,node(s,r,rendered['geometry'],name));nxt[target]=b['pose']
        frontier=nxt;save(out/'progress.json',dict(depth=depth+1,nodes=len(nodes),edges=len(edges)))
    media={}
    def bind(frame,path):
        media[frame]={'path':path,'sha256':sha(out/path)}
    public_nodes={}
    for i,(k,n) in enumerate(nodes.items()):
        frame=f'f{len(hist)+i:06d}';bind(frame,n['image']);public_nodes[k]={'frame':frame}
    public=dict(root=root,nodes=public_nodes,edges=edges)
    check=GraphActionExecutor(public,steps).preflight()
    template=question['template']
    base=hr['states'][-1]['branches']['base']
    refs=[referent(state,base,j) for j in targets]
    if any(not r for r in refs):raise ValueError('ambiguous initial referent')
    options=None
    if template=='object_size_m' and len(targets)==1:
        answer=round(mesh_size_truth(hr['geometry'][targets[0]])['longest_side_m'],3)
        text=f'Estimate the longest physical side of the 3D bounding box of {refs[0]}. Answer in meters.'
    elif template in ('object_to_object_distance_m','object_to_object_distance_bucket','object_spatial_proximity') and len(targets)==2:
        distance=math.dist(state['objects'][targets[0]]['center'],state['objects'][targets[1]]['center'])
        if template.endswith('_m'):
            answer=round(distance,3)
            text=f'Estimate the 3D center-to-center distance between {refs[0]} and {refs[1]}. Answer in meters.'
        else:
            semantic=object_distance_bucket(distance)
            if semantic is None:raise ValueError('ambiguous distance interval')
            options={chr(65+i):v for i,v in enumerate(OBJECT_DISTANCE_OPTIONS)}
            answer=next(k for k,v in options.items() if v==semantic)
            text=f'Which range contains the 3D center-to-center distance between {refs[0]} and {refs[1]}? Return the option letter.'
    else:raise ValueError('unsupported template or target count')
    q={'mode':'[QA]','question':text,'answer_type':'choice' if options else 'numeric','actions':COMMANDS}
    if options:q['options']=options
    else:q['unit']='meters'
    if question.get('payload'):
        # Optional frozen evaluation wording/option order is kept verbatim.
        frozen=copy.deepcopy(question['payload']);truth=question['answer']
        if options:
            if frozen.get('options',{}).get(truth)!=options[answer]:raise ValueError('frozen choice disagrees with geometry')
        elif type(truth) not in (int,float) or abs(truth-answer)>.000501:
            raise ValueError('frozen answer disagrees with geometry')
        q=frozen;q['actions']=COMMANDS;answer=truth;text=q['question'];options=q.get('options')
    check_public(q)
    schedule=[{'kind':'system_prompt','payload':{'instruction':full_instruction(ACTION_SYSTEM_INSTRUCTION)}}]
    for i,h in enumerate(hist):
        frame=f'f{i:06d}';bind(frame,h['image']);schedule.append({'kind':'observation','payload':{'frame':frame,'timestamp':history_frames[i]['timestamp_sec']}})
    schedule.append({'kind':'question','payload':q});qi=len(schedule)-1
    bundle={'events':[],'schedule':schedule,'answers':[dict(schedule_index=qi,answer=answer,sample=0,turn=0,template=template,answer_type=q['answer_type'],unit=q.get('unit'),options=q.get('options'))],'media':media,'actions':[],'action_graph':public}
    save(out/'bundle.json',bundle)
    graph=dict(root=root,nodes=nodes,edges=edges,history_frames=history_frames)
    save(out/'graph.private.json',graph)
    evidence=lambda n:{'objects':n['metrics']}
    task=dict(graph='graph.private.json',targets=targets,question=text,answer=answer,
              options=options,node_evidence={k:evidence(n) for k,n in nodes.items()},history=[{'image':n['image'],'evidence':evidence(n)} for n in hist])
    save(out/'task.json',task)
    baseline=max(min(n['metrics'][j]['p'] for j in targets) for n in hist)
    def sufficient(n):
        return all(m['p']>=.005 and m['f'] is not None and m['f']>=.8 and m['o'] is not None and m['o']>=.7 for m in n['metrics'].values()) and min(m['p'] for m in n['metrics'].values())>=max(1.5*baseline,baseline+.002)
    by={(e['source'],e['action']):e for e in edges};paths=[]
    def walk(k,path):
        if path and sufficient(nodes[k]):paths.append(path);return
        if len(path)==steps:return
        for c in COMMANDS:
            e=by[k,c['name']]
            if e['valid']:walk(e['target'],path+[e])
    walk(root,[])
    demos=[]
    for i,path in enumerate(paths):
        events=[{'role':'system','kind':'system_prompt','payload':schedule[0]['payload'],'loss':False}]
        events += [{'role':'user','kind':s['kind'],'payload':s['payload'],'loss':False} for s in schedule[1:]]
        for e in path:
            c=next(c for c in COMMANDS if c['name']==e['action'])
            events.extend([{'role':'assistant','kind':'action','payload':c,'loss':True},
                           {'role':'tool','kind':'action_result','payload':{'status':'ok','action':c,'frame':public_nodes[e['target']]['frame']},'loss':False}])
        events.append({'role':'assistant','kind':'answer','payload':{'type':'answer','answer':answer},'loss':True})
        for event in events:check_public(event['payload'])
        demos.append({'id':str(i),'events':events,'media':media})
    save(out/'demonstrations.json',demos)
    temp=out/'demonstrations.jsonl.gz.tmp'
    with gzip.open(temp,'wt') as f:
        for row in demos:f.write(json.dumps(row)+'\n')
    temp.replace(out/'demonstrations.jsonl.gz')
    save(out/'complete.json',{**check,'demonstrations':len(demos),'history_frames':len(hist),'question':text,'answer':answer,
                              'geometry_labels':True,'rgb_answerability_guaranteed':False})
    return read(out/'complete.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blender',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--steps',type=int,choices=(1,2,3),default=3);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(generate(a.config,a.output,a.blender,a.gpu,a.steps,a.resume)))
