import copy
import pytest
from action_graph import COMMANDS, GraphActionExecutor, ActionCoverageError, make_executor
from routed_stream import run_session
from offline_uniform32 import run_offline


def graph():
    return dict(root='0',nodes={str(i):{'frame':'a'+str(i)} for i in range(4)},edges=[
        dict(source=str(i),target=str(i+1) if c==COMMANDS[0] else str(i),action=c['name'],valid=c==COMMANDS[0])
        for i in range(3) for c in COMMANDS])


def test_chain_invalid_reset_and_private_boundary():
    g=graph();g['nodes']['1']['reward']=123
    e=GraphActionExecutor(g);e.preflight()
    a=e(COMMANDS[0]);assert a['frame']=='a1' and 'reward' not in a
    assert e(COMMANDS[1])['status']=='invalid_action' and e.node=='1'
    assert e(COMMANDS[0])['frame']=='a2'
    with pytest.raises(ValueError):e(COMMANDS[0])
    e.begin_question(5,{});assert e.node=='0' and e.attempts==0


def test_missing_branch_is_coverage_error_not_invalid_action():
    g=graph();g['edges'].pop()
    with pytest.raises(ActionCoverageError):GraphActionExecutor(g).preflight()
    with pytest.raises(ActionCoverageError):make_executor({'actions':[{}]},3)
    with pytest.raises(ActionCoverageError):make_executor({'actions':[{'action':COMMANDS[0],'result':{'status':'ok'}}]},1)


def test_three_actions_plus_memory_separate_budgets():
    e=GraphActionExecutor(graph());calls=[]
    outputs=iter([{'type':'use_memory','categories':[],'with_frames':True}]+[COMMANDS[0]]*3+[{'type':'answer','answer':1}])
    def generate(ctx):calls.append(ctx);return copy.deepcopy(next(outputs))
    pred,ctx,_=run_session([{'kind':'system_prompt','payload':{'instruction':'answer or use tools'}},
                           {'kind':'question','payload':{'question':'q'}}],generate,max_calls=1,max_actions=3,action_executor=e)
    assert pred[0]['answer']==1 and e.attempts==3
    assert calls[-1][-1]['content']['instruction']=='Return a final answer now.'
    frames=[m['content']['frame'] for m in ctx if m['role']=='tool' and 'frame' in m['content']]
    assert frames==['a1','a2','a3']
    assert 'a3' not in str(calls[2])


def test_invalid_attempt_counts_and_force_answer():
    e=GraphActionExecutor(graph(),1);outs=iter([COMMANDS[1],{'type':'answer','answer':0}])
    pred,ctx,_=run_session([{'kind':'question','payload':{'question':'q'}}],lambda ctx:next(outs),action_executor=e,max_actions=1)
    assert e.attempts==1 and e.node=='0' and pred[0]['answer']==0


def test_offline_active_three_steps_and_new_question_reset():
    e=GraphActionExecutor(graph());contexts=[]
    outputs=iter([COMMANDS[0]]*3+[{'type':'answer','answer':4}]+[COMMANDS[0],{'type':'answer','answer':2}])
    def gen(ctx):contexts.append(copy.deepcopy(ctx));return copy.deepcopy(next(outputs))
    req=[{'schedule_index':i,'question':{'question':'q','answer_type':'numeric'},'previous':[],
          'frames':[{'frame':'h','timestamp':0}]} for i in (1,2)]
    pred,res=run_offline(req,gen,action_executor=e)
    assert [p['answer'] for p in pred]==[4,2]
    assert [r['action_attempts'] for r in res]==[3,1]
    assert e.node=='1' and 'a3' not in str(contexts[4])


def test_graph_sidecar_preserves_original_history_on_frame_collision(tmp_path):
    import json
    from evaluation_selection import load_selected_episode
    from joint_episode import load_episode
    root=tmp_path/'episode';root.mkdir();side=tmp_path/'side';side.mkdir()
    original=dict(events=[],schedule=[],answers=[],media={'f000000':{'path':'original.png'}},actions=[])
    (root/'bundle.json').write_text(json.dumps(original))
    g=graph()
    for i,n in enumerate(g['nodes'].values()):n['frame']=f'f{i:06d}'
    extra=dict(events=[],schedule=[],answers=[],media={n['frame']:{'path':str(i)+'.png'} for i,n in enumerate(g['nodes'].values())},actions=[],action_graph=g)
    (side/'bundle.json').write_text(json.dumps(extra))
    data=load_selected_episode({'root':str(root),'action_graph_bundle':str(side)})
    assert data['media']['f000000']['path']==str(root/'original.png')
    assert all(n['frame']!='f000000' for n in data['action_graph']['nodes'].values())
    make_executor(data).preflight()
