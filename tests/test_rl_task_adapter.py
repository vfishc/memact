import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1'))
from core import ACTIONS,Config,CoverageError
from task_adapter import load_task

def fixture(tmp_path):
 image=tmp_path/'image.png';image.write_bytes(b'fixture: existence only')
 evidence={'objects':{'chair':{'p':.03,'f':1.}}}
 raw={'root':'r','question':'wrong source question','answer':999,'nodes':{'r':{'image':str(image),'pose':{'private':1}}},'edges':[dict(source='r',target='r',action=a,valid=False) for a in ACTIONS[1:]],'history_frames':[{'frame_id':0,'timestamp_sec':0.}]}
 graph=tmp_path/'graph.json';graph.write_text(json.dumps(raw))
 task={'graph':str(graph),'targets':['chair'],'node_evidence':{'r':evidence},'history':[{'image':str(image),'evidence':evidence}],'question':'Bound task question','answer':2.5}
 path=tmp_path/'task.json';path.write_text(json.dumps(task));return path,graph,task,raw

def test_task_binding_and_private_boundary(tmp_path):
 p,_,_,_=fixture(tmp_path);g=load_task(p);cfg=Config(reward_mode='historical');s=g.initial(cfg)
 assert g.data['answer']==2.5
 view=g.public_view(s,cfg);assert view['question']=='Bound task question'
 assert not any(k in json.dumps(view) for k in ['pixel_fraction','object_evidence','wrong source question'])
 assert all(set(e['value']) <= {'image','frame','timestamp','text'} for e in view['events'])
 ns,r=g.advance(s,ACTIONS[1],cfg);assert ns.steps==1;assert r==pytest.approx(-.23)

def test_missing_branch_is_not_invalid_action(tmp_path):
 p,g,_,raw=fixture(tmp_path);raw['edges'].pop();g.write_text(json.dumps(raw))
 with pytest.raises(CoverageError):load_task(p)

def test_missing_media_and_visible_unknown_framing(tmp_path):
 p,_,task,_=fixture(tmp_path);task['node_evidence']['r']['objects']['chair']['f']=None;p.write_text(json.dumps(task))
 with pytest.raises(ValueError,match='framing'):load_task(p)
 task['node_evidence']['r']['objects']['chair']['p']=0;p.write_text(json.dumps(task));load_task(p)
 (tmp_path/'image.png').unlink()
 with pytest.raises(FileNotFoundError):load_task(p)
