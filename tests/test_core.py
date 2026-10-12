import unittest,copy,random,json
import torch
from core import *

def fixture():
 nodes={n:{'observation':{'frame':n,'text':t},'evidence':e} for n,t,e in [('start','blocked',0),('middle','corner',0),('goal','answer is 2',1)]}
 edges=[]
 for n in nodes:
  for a in ACTIONS[1:]:
   target='middle' if n=='start' and a=='move_left_d' else 'goal' if n=='middle' and a=='rotate_right_45' else n
   edges.append({'source':n,'action':a,'target':target,'valid':target!=n})
 return Graph({'question':'How many items?','answer':2,'root':'start','nodes':nodes,'edges':edges})

class TinyPolicy(torch.nn.Module):
    """CPU algorithm fixture, NOT a visual model. Answer reads public fixture text."""
    def __init__(self):
        super().__init__();self.logits=torch.nn.Parameter(torch.zeros(3,len(ACTIONS)))
    def index(self,v):
        texts=[e['value'].get('text','') if e['kind']=='observation' else e['value'].get('observation',{}).get('text','') for e in v['events'] if e['kind']!='action']
        return 2 if 'answer is 2' in texts else 1 if 'corner' in texts else 0
    def scores(self,v):return self.logits[self.index(v)]
    def answer(self,v,rng):return 2 if self.index(v)==2 else 0

class Teacher(TinyPolicy):
    def scores(self,v):
        out=torch.full((len(ACTIONS),),-100.)
        out[{0:4,1:2,2:0}[self.index(v)]]=100
        return out

class Tests(unittest.TestCase):
 def test_delayed_reward(self):
  g=fixture();c=Config(max_steps=3,rollouts=1);s=g.initial();q,_=candidate_values(g,s,Teacher(),c)
  self.assertAlmostEqual(float(q[4]),1.04,places=5);self.assertEqual(float(q[0]),0);self.assertEqual(s,g.initial())
 def test_discount(self):
  g=fixture();r=rollout(g,g.initial(),'move_left_d',Teacher(),Config(gamma=.5),0)
  self.assertAlmostEqual(r['return'],-.03+.5*.07+.25)
 def test_invalid_budget(self):
  g=fixture();s,r=g.advance(g.initial(),'move_right_d',Config());self.assertEqual(s.node,'start');self.assertEqual(s.steps,1);self.assertAlmostEqual(r,-.23)
 def test_missing_not_collision(self):
  g=fixture();del g.edges['start','move_right_d']
  with self.assertRaises(CoverageError):g.preflight(Config())
 def test_history_no_gain_again(self):
  d=fixture().data;d['history']=['goal'];g=Graph(d);s=g.initial();s,r=g.advance(s,'move_left_d',Config());s,r=g.advance(s,'rotate_right_45',Config());self.assertAlmostEqual(r,-.03)
 def test_privacy_and_copy(self):
  g=fixture();s=g.initial();v=g.public_view(s,Config());self.assertEqual(set(v),{'question','events','remaining_actions','candidates'});v['events'].clear();self.assertTrue(s.events)
 def test_answer_score(self):
  self.assertEqual(qa_score(True,1),0);self.assertEqual(qa_score(float('nan'),1),0);self.assertEqual(qa_score(0,0),1);self.assertEqual(qa_score(1,0),0)
 def test_gradient_q_detached(self):
  s=torch.zeros(6,requires_grad=True);q=torch.arange(6.,requires_grad=True);loss=objective(s,s.detach(),q);loss.backward();self.assertIsNone(q.grad);self.assertLess(float(s.grad[-1]),0)
 def test_terminal_forced_answer(self):
  g=fixture();r=rollout(g,g.initial(),'move_left_d',Teacher(),Config(max_steps=1),0);self.assertEqual(r['termination'],'budget');self.assertEqual(r['qa_score'],0)
 def test_branch_reproducibility(self):
  g=fixture();p=TinyPolicy();c=Config();a,_=candidate_values(g,g.initial(),p,c,19);b,_=candidate_values(g,g.initial(),p,c,19);self.assertTrue(torch.equal(a,b))
 def test_training_end_to_end(self):
  g=fixture();p=TinyPolicy();cfg=Config(rollouts=16);logs=train(g,p,cfg,steps=100,lr=.12)
  s=g.initial();a=ACTIONS[int(p.scores(g.public_view(s,cfg)).argmax())];self.assertEqual(a,'move_left_d')
  s,_=g.advance(s,a,cfg);a=ACTIONS[int(p.scores(g.public_view(s,cfg)).argmax())];self.assertEqual(a,'rotate_right_45')
  s,_=g.advance(s,a,cfg);self.assertEqual(ACTIONS[int(p.scores(g.public_view(s,cfg)).argmax())],'answer_now')

if __name__=='__main__':unittest.main()
