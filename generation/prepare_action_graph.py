"""Convert a packaged RL task into a private state graph usable by native evaluation.

The task must belong to the question being evaluated. This converter never
changes the question or answer and never substitutes training tasks for a test.
"""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1')]
from task_adapter import load_task
from action_graph import GraphActionExecutor,COMMANDS
from qa_eight_contract import full_instruction
from routed_stream import ACTION_SYSTEM_INSTRUCTION


def prepare(task_path,output,steps=3):
    source_task=json.loads(Path(task_path).read_text());options=source_task.get('options')
    graph=load_task(task_path);data=graph.data;out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    media={};nodes={}
    for i,(k,n) in enumerate(data['nodes'].items()):
        source=Path(n['observation']['image']);digest=hashlib.sha256(source.read_bytes()).hexdigest();rel='images/'+digest+source.suffix
        if not (out/rel).exists():shutil.copyfile(source,out/rel)
        frame=f'f{i:06d}'
        media[frame]={'path':rel,'sha256':digest};nodes[k]={'frame':frame}
    public=dict(root=data['root'],nodes=nodes,edges=data['edges'])
    receipt=GraphActionExecutor(public,steps).preflight()
    schedule=[{'kind':'system_prompt','payload':{'instruction':full_instruction(ACTION_SYSTEM_INSTRUCTION)}}]
    for k in data.get('history',[]):
        n=data['nodes'][k]['observation'];schedule.append({'kind':'observation','payload':{'frame':nodes[k]['frame'],'timestamp':n['timestamp']}})
    if not schedule or len(schedule)==1:raise ValueError('task requires timestamped causal history')
    schedule.append({'kind':'question','payload':{'mode':'[QA]','question':data['question'],'actions':COMMANDS,'answer_type':'choice' if options else 'numeric','options':options,'unit':None if options else 'meters'}})
    bundle=dict(events=[],schedule=schedule,answers=[dict(schedule_index=len(schedule)-1,sample=0,turn=0,answer=data['answer'],answer_type='choice' if options else 'numeric',options=options,unit=None if options else 'meters')],media=media,actions=[],action_graph=public)
    (out/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');(out/'coverage.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    a=p.parse_args();print(json.dumps(prepare(a.task,a.output,a.steps)))
