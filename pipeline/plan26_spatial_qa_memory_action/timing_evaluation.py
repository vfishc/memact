"""Causal wait/answer evaluation for the released Timing bank."""
import copy
import math
from qa_eight_contract import valid_answer
from routed_stream import check_public, validate_answer

INSTRUCTION = ('Observe chronological video frames. At each checkpoint return exactly '
    '{"type":"wait"} while evidence is insufficient, or '
    '{"type":"answer","answer":value} once ready. Your first answer is final. '
    'For labeled options use a single uppercase option letter. For numerical '
    'questions use a JSON number in the stated unit; counts must be integers. '
    'Do not call memory or action tools. Times are episode-relative.')


def run_timing(row, schedule, generate):
    observations=[copy.deepcopy(e['payload']) for e in schedule if e['kind']=='observation']
    points=row['checkpoints']
    if not points or points!=sorted(set(points)) or min(points)<-1 or max(points)>=len(observations):
        raise ValueError('invalid Timing checkpoints')
    times=[o['timestamp'] for o in observations]
    if any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('invalid observation chronology')
    if points[0]!=row['issue_after_observation']:
        raise ValueError('question arrival/checkpoint mismatch')
    # Private timing indices and expected decisions are never serialized here.
    question=copy.deepcopy(row['public']);check_public(question)
    context=[{'role':'system','content':{'instruction':INSTRUCTION}}]
    previous=-1;events=[];issued=False
    for index in points:
        checkpoint_time = times[index] if index >= 0 else times[0]-1.0
        for obs in observations[previous+1:index+1]:
            check_public(obs);context.append({'role':'user','content':obs})
        if not issued:
            context.append({'role':'user','content':question});issued=True
        else:
            context.append({'role':'user','content':{'mode':'[TIMING_CHECK]'}})
        prediction=generate(copy.deepcopy(context));check_public(prediction)
        if prediction=={'type':'wait'}:
            events.append({'timestamp':checkpoint_time,'type':'wait'})
            context.append({'role':'assistant','content':prediction});previous=index;continue
        validate_answer(prediction)
        events.append({'timestamp':checkpoint_time,**prediction})
        return {'id':row['id'],'status':'answered','answer_time':checkpoint_time,
                'answer':prediction['answer'],'events':events}
    return {'id':row['id'],'status':'unanswered','events':events}


def score_timing(rows, predictions):
    by={p['id']:p for p in predictions}
    if len(by)!=len(predictions):raise ValueError('duplicate Timing predictions')
    counts={'all':len(rows),'choice':0,'numeric':0};totals={'on_time':0,'joint_choice':0,'joint_numeric':0.0}
    for row in rows:
        kind=row['public']['answer_type']
        if kind not in ('choice','numeric'):raise ValueError('unknown answer type')
        counts[kind]+=1;p=by.get(row['id'],{});t=p.get('answer_time')
        timely=p.get('status')=='answered' and type(t) in (int,float) and row['first_answerable_seconds']<=t<=row['deadline_seconds']
        totals['on_time']+=int(timely)
        value=p.get('answer')
        if not timely or not valid_answer(value,row['public']):continue
        if kind=='choice':totals['joint_choice']+=int(value==row['answer'])
        else:
            y=row['answer'];score=float(value==0) if y==0 else sum(abs(value-y)/abs(y)<=k/100 for k in range(5,51,5))/10
            totals['joint_numeric']+=score
    percent=lambda x,n:100*x/n if n else None
    return {'counts':counts,'T':percent(totals['on_time'],counts['all']),
            'JA':percent(totals['joint_choice'],counts['choice']),
            'JM':percent(totals['joint_numeric'],counts['numeric'])}
