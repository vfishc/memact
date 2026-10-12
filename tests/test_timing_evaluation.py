import copy
import pytest
from timing_evaluation import run_timing,score_timing

def sample():
    r=dict(id='q',public=dict(question='When?',answer_type='numeric',unit='seconds'),answer=3,
           issue_after_observation=0,first_answerable_seconds=3,deadline_seconds=8,checkpoints=list(range(9)))
    s=[dict(kind='observation',payload=dict(frame=f'f{i:06d}',timestamp=i)) for i in range(9)]
    return r,s

def test_causal_checkpoints_and_inclusive_deadline():
    r,s=sample();calls=[]
    def gen(ctx):
        calls.append(ctx);t=max(m['content'].get('timestamp',-1) for m in ctx)
        assert all(k not in str(ctx) for k in ('first_answerable','deadline','expected_policy'))
        assert 'f000008' not in str(ctx) if t<8 else True
        return {'type':'wait'} if t<8 else {'type':'answer','answer':3}
    p=run_timing(r,s,gen);assert p['answer_time']==8
    assert score_timing([r],[p])['JM']==100
    p=run_timing(r,s,lambda ctx:{'type':'wait'});assert score_timing([r],[p])['T']==0

def test_early_stops_and_failed_answers_keep_denominator():
    r,s=sample();p=run_timing(r,s,lambda ctx:{'type':'answer','answer':3});assert len(p['events'])==1
    r2={**r,'id':'other'};m=score_timing([r,r2],[p,{'id':'other','status':'failed'}]);assert m['T']==0 and m['counts']['all']==2
    with pytest.raises(ValueError):run_timing(r,s,lambda ctx:{'type':'action','name':'move_forward_d'})

def test_zero_numeric_and_choice_denominators():
    r,_=sample();r['answer']=0
    c={**r,'id':'choice','answer':'A','public':dict(question='Which?',answer_type='choice',options={'A':'x','B':'y'})}
    pred=[dict(id='q',status='answered',answer_time=3,answer=0),dict(id='choice',status='answered',answer_time=4,answer='B')]
    m=score_timing([r,c],pred);assert m['T']==100 and m['JM']==100 and m['JA']==0


def test_question_before_first_frame_has_no_future_observation():
    r,s=sample();r.update(issue_after_observation=-1,checkpoints=list(range(-1,9)))
    seen=[]
    def gen(ctx):
        seen.append(ctx)
        return {'type':'wait'} if len(seen)==1 else {'type':'answer','answer':3}
    p=run_timing(r,s,gen)
    assert not any('frame' in m['content'] for m in seen[0])
    assert p['answer_time']==0 and score_timing([r],[p])['T']==0


def test_text_only_visual_batch():
    import torch
    from types import SimpleNamespace
    from joint_episode import check_visual_batch
    proc=SimpleNamespace(image_token='<image>',tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _:99))
    assert check_visual_batch({'input_ids':torch.tensor([[1,2]])},[],proc)['decoded_images']==0
    with pytest.raises(ValueError):check_visual_batch({'input_ids':torch.tensor([[99]])},[],proc)
