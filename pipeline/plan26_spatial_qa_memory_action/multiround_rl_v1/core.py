"""Causal finite-graph cumulative-return candidate optimization.

Policy boundary is public_view only. Graph geometry, evidence and GT stay private.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import copy, math, random
from rewards import quality, transition_reward, metrics, validate_config

ACTIONS=('answer_now','rotate_left_45','rotate_right_45','move_forward_d','move_left_d','move_right_d')
class CoverageError(ValueError): pass

@dataclass(frozen=True)
class Config:
    max_steps:int=3
    gamma:float=1.0
    evidence_weight:float=0.1
    action_cost:float=0.03
    invalid_penalty:float=0.2
    rollouts:int=4
    reward_mode:str="legacy"
    pixel_reference:float=0.10
    effective_gain_threshold:float=0.10
    def __post_init__(self):
        validate_config(self.reward_mode,self.pixel_reference,self.effective_gain_threshold)
        if self.max_steps<1 or self.rollouts<1 or not 0<self.gamma<=1: raise ValueError('invalid budget/discount')
        if any(not math.isfinite(v) or v<0 for v in (self.evidence_weight,self.action_cost,self.invalid_penalty)):raise ValueError('invalid reward coefficient')

@dataclass(frozen=True)
class State:
    node:str
    events:tuple
    best_evidence:float
    steps:int=0
    reward_signature:tuple=("legacy",0.10)
    visited_nodes:tuple=()

class Graph:
    def __init__(self,data):
        self.data=copy.deepcopy(data)
        self.nodes=self.data['nodes'];self.edges={}
        for node in self.nodes.values():
            if 'evidence' in node:quality(node,[], 'legacy',.1)
            if set(node['observation'])-{'frame','timestamp','image','text'}:raise ValueError('nonpublic observation fields')
        for e in self.data['edges']:
            k=(e['source'],e['action'])
            if k in self.edges or e['action'] not in ACTIONS[1:]:raise ValueError('duplicate/unknown edge')
            if e['source'] not in self.nodes or e['target'] not in self.nodes:raise ValueError('unknown node')
            if type(e['valid']) is not bool:raise ValueError('valid must be boolean')
            if not e['valid'] and e['target']!=e['source']:raise ValueError('invalid action must stay in place')
            if not math.isfinite(e.get('cost',1.0)) or e.get('cost',1.0)<0:raise ValueError('invalid cost')
            self.edges[k]=e
        if self.data['root'] not in self.nodes:raise ValueError('unknown root')
    def score(self,node,cfg,full=False):
        mode=("joint" if full and cfg.reward_mode!="legacy" else cfg.reward_mode)
        return quality(self.nodes[node],self.data.get("required_objects"),mode,cfg.pixel_reference)
    def initial(self,cfg=None):
        cfg=cfg or Config()
        hist=self.data.get('history',[])
        observations=[copy.deepcopy(self.nodes[n]['observation']) for n in hist+[self.data['root']]]
        return State(self.data['root'],tuple({'kind':'observation','value':o} for o in observations),
                     max(self.score(n,cfg) for n in hist+[self.data['root']]),
                     reward_signature=(cfg.reward_mode,cfg.pixel_reference),visited_nodes=(self.data['root'],))
    def public_view(self,state,cfg):
        return copy.deepcopy({'question':self.data['question'],'events':list(state.events),
                              'remaining_actions':cfg.max_steps-state.steps,'candidates':list(ACTIONS)})
    def advance(self,state,action,cfg):
        if state.reward_signature!=(cfg.reward_mode,cfg.pixel_reference):raise ValueError('state/reward configuration mismatch')
        if state.steps>=cfg.max_steps:raise ValueError('budget exhausted')
        if (state.node,action) not in self.edges:raise CoverageError(f'missing edge {state.node}/{action}')
        e=self.edges[state.node,action];n=self.nodes[e['target']]
        best,reward=transition_reward(self.score(state.node,cfg),self.score(e['target'],cfg),state.best_evidence,cfg,e['valid'],e.get('cost',1.0))
        events=state.events+({'kind':'action','value':action},{'kind':'tool','value':{
            'status':'ok' if e['valid'] else 'invalid_action',
            'observation':copy.deepcopy(n['observation'])}})
        return State(e['target'],events,best,state.steps+1,state.reward_signature,state.visited_nodes+(e['target'],)),reward
    def evidence_metrics(self,state,cfg):
        initial=self.data.get("history",[])+[self.data["root"]]
        baseline=max(self.score(n,cfg,True) for n in initial)
        final=max(self.score(n,cfg,True) for n in initial+list(state.visited_nodes))
        return metrics(baseline,final,cfg)
    def preflight(self,cfg):
        for n in self.nodes:self.score(n,cfg)
        frontier={self.data['root']}
        for _ in range(cfg.max_steps):
            nxt=set()
            for node in frontier:
                for a in ACTIONS[1:]:
                    if (node,a) not in self.edges:raise CoverageError(f'missing edge {node}/{a}')
                    nxt.add(self.edges[node,a]['target'])
            frontier=nxt

def qa_score(pred,truth):
    if type(truth) in (int,float):
        if type(pred) not in (int,float) or not math.isfinite(pred) or not math.isfinite(truth):return 0.0
        if truth==0:return float(pred==0)
        err=abs(pred-truth)/abs(truth)
        return sum(err<=t/100+1e-12 for t in range(5,51,5))/10
    return float(type(pred) is type(truth) and pred==truth)

def sample_action(policy,view,rng):
    import torch
    with torch.no_grad():p=torch.softmax(policy.scores(view).float(),-1).cpu().tolist()
    if len(p)!=len(ACTIONS) or any(not math.isfinite(x) for x in p):raise ValueError('invalid candidate scores')
    return rng.choices(ACTIONS,weights=p,k=1)[0]

def rollout(graph,state,first,policy,cfg,seed):
    rng=random.Random(seed);a=first;total=0.0;discount=1.0;visited=[];trace=[]
    while True:
        if a=='answer_now' or state.steps>=cfg.max_steps:
            answer=policy.answer(graph.public_view(state,cfg),rng)
            score=qa_score(answer,graph.data['answer']);total+=discount*score
            return {'return':total,'answer':answer,'qa_score':score,'trace':trace,'visited':visited,
                    'termination':'budget' if state.steps>=cfg.max_steps else 'answer','steps':state.steps,
                    **graph.evidence_metrics(state,cfg)}
        state,r=graph.advance(state,a,cfg);total+=discount*r;discount*=cfg.gamma
        trace.append({'action':a,'reward':r});visited.append(state)
        if state.steps>=cfg.max_steps:a='answer_now'
        else:a=sample_action(policy,graph.public_view(state,cfg),rng)

def candidate_values(graph,state,policy,cfg,seed=0):
    # Equal seed schedule across branches; stochastic adapters must use supplied RNG.
    import torch
    values=[];records=[]
    for a in ACTIONS:
        rr=[rollout(graph,state,a,policy,cfg,seed+b) for b in range(cfg.rollouts)]
        values.append(sum(r['return'] for r in rr)/len(rr));records.append(rr)
    return torch.tensor(values,dtype=torch.float32),records

def objective(scores,reference,values,beta=.1,entropy_weight=0.):
    import torch
    if scores.shape!=reference.shape or scores.shape!=values.shape or scores.numel()!=len(ACTIONS):raise ValueError('candidate shape mismatch')
    if not all(bool(torch.isfinite(x).all()) for x in (scores,reference,values)):raise ValueError('nonfinite objective')
    lp=torch.log_softmax(scores.float(),-1);p=lp.exp();lr=torch.log_softmax(reference.detach().float(),-1)
    reward=(p*values.detach().to(p.device)).sum();kl=(p*(lp-lr)).sum();entropy=-(p*lp).sum()
    return -reward+beta*kl-entropy_weight*entropy

def train(graph,policy,cfg,steps=60,lr=.1,beta=.05,seed=26):
    """Bounded adapter trainer; native QA/format replay supplied by caller, not fabricated."""
    import torch
    graph.preflight(cfg);torch.manual_seed(seed)
    ref=copy.deepcopy(policy).eval();opt=torch.optim.Adam(policy.parameters(),lr=lr);logs=[]
    for step in range(steps):
        mu=copy.deepcopy(policy).eval()  # Refresh frozen continuation snapshot per iteration.
        initial=graph.initial(cfg); first=sample_action(mu,graph.public_view(initial,cfg),random.Random(seed+step))
        sampled=rollout(graph,initial,first,mu,cfg,seed+step)
        states=[initial]+[s for s in sampled['visited'] if s.steps<cfg.max_steps]
        opt.zero_grad();losses=[]
        for i,s in enumerate(states):
            view=graph.public_view(s,cfg);values,_=candidate_values(graph,s,mu,cfg,seed+step*100+i*10)
            with torch.no_grad():rs=ref.scores(view)
            losses.append(objective(policy.scores(view),rs,values,beta))
        loss=torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);opt.step()
        logs.append({'step':step,'loss':float(loss.detach()),'states':len(states),'sampled_return':sampled['return']})
    return logs

def chain_continuation_values(stop_rewards, action_rewards, probabilities, gamma=1.0):
    """Exact stop/continue Q on a finite recorded chain, fixed terminal answers.

    This explicitly restricted support does not fill absent camera branches.
    Policy inputs never receive these private reward arrays.
    """
    if len(stop_rewards)!=len(action_rewards)+1 or len(probabilities)!=len(action_rewards):
        raise ValueError('chain length mismatch')
    if not 0<gamma<=1:raise ValueError('invalid discount')
    if any(not math.isfinite(float(x)) for x in list(stop_rewards)+list(action_rewards)):
        raise ValueError('nonfinite reward')
    values=[None]*len(action_rewards);v=float(stop_rewards[-1])
    for d in reversed(range(len(action_rewards))):
        p=probabilities[d]
        if len(p)!=2 or any(not math.isfinite(float(x)) or x<0 for x in p) or abs(sum(p)-1)>1e-5:
            raise ValueError('invalid stop/continue probability')
        q=[float(stop_rewards[d]),float(action_rewards[d])+gamma*v]
        values[d]=q;v=sum(float(pi)*qi for pi,qi in zip(p,q))
    return values,v
