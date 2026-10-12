import unittest,copy,json,math
import torch
from core import Config,Graph,train
from rewards import quality,transition_reward,metrics
from test_core import fixture,TinyPolicy

def raw_fixture():
    data=fixture().data;data['required_objects']=['a','b']
    for name,node in data['nodes'].items():
        p={'start':.01,'middle':.02,'goal':.10}[name]
        node['object_evidence']={'a':{'pixel_fraction':p,'framing':1.},'b':{'pixel_fraction':p/2,'framing':.8}}
    return Graph(data)

class RewardTests(unittest.TestCase):
    def test_components(self):
        n={'object_evidence':{'a':{'pixel_fraction':.1,'framing':.5},'b':{'pixel_fraction':.02,'framing':1.}}}
        self.assertAlmostEqual(quality(n,['a','b'],'pixel',.1),.6)
        self.assertAlmostEqual(quality(n,['a','b'],'framing',.1),.35)
        self.assertAlmostEqual(quality(n,['a','b'],'joint',.1),.2)
        self.assertEqual(quality(n,['a'],'framing',.1),quality(n,['a'],'joint',.1))
    def test_saturation_and_zero(self):
        n={'object_evidence':{'a':{'pixel_fraction':.8,'framing':.25}}}
        self.assertEqual(quality(n,['a'],'pixel',.1),1)
        self.assertEqual(quality(n,['a'],'historical',.1),.25)
        n['object_evidence']['a']['pixel_fraction']=0
        self.assertEqual(quality(n,['a'],'joint',.1),0)
    def test_historical_revisit(self):
        local=Config(reward_mode='joint');hist=Config(reward_mode='historical')
        self.assertAlmostEqual(transition_reward(.1,.6,.8,local)[1],.02)
        self.assertAlmostEqual(transition_reward(.1,.6,.8,hist)[1],-.03)
    def test_invalid_no_evidence_gain(self):
        for m in ['pixel','framing','joint','historical']:
            self.assertAlmostEqual(transition_reward(0,0,0,Config(reward_mode=m),False)[1],-.23)
    def test_missing_and_nonfinite(self):
        n={'object_evidence':{'a':{'pixel_fraction':.1,'framing':None}}}
        with self.assertRaises(ValueError):quality(n,['a'],'historical',.1)
        with self.assertRaises(ValueError):quality(n,['b'],'pixel',.1)
        n['object_evidence']['a']['framing']=float('nan')
        with self.assertRaises(ValueError):quality(n,['a'],'framing',.1)
        for kwargs in [{'pixel_reference':0},{'reward_mode':'typo'},{'effective_gain_threshold':1.1}]:
            with self.assertRaises(ValueError):Config(**kwargs)
    def test_new_mode_rejects_scalar_only(self):
        with self.assertRaises(ValueError):fixture().preflight(Config(reward_mode='historical'))
    def test_state_config_guard(self):
        g=raw_fixture();cfg=Config(reward_mode='historical')
        with self.assertRaises(ValueError):g.advance(g.initial(), 'move_left_d',cfg)
    def test_common_eval_and_privacy(self):
        results=[]
        for mode in ['pixel','framing','joint','historical']:
            g=raw_fixture();cfg=Config(reward_mode=mode);s=g.initial(cfg)
            self.assertEqual(g.evidence_metrics(s,cfg)['evidence_gain_points'],0)
            for a in ['move_left_d','rotate_right_45']:s,_=g.advance(s,a,cfg)
            results.append(g.evidence_metrics(s,cfg))
            v=json.dumps(g.public_view(s,cfg));self.assertNotIn('object_evidence',v);self.assertNotIn('pixel_fraction',v);self.assertNotIn('required_objects',v)
        self.assertTrue(all(r==results[0] for r in results))
        self.assertAlmostEqual(results[0]['evidence_gain_points'],36.)
        self.assertTrue(results[0]['effective_gain'])
    def test_all_modes_optimizer_update(self):
        for mode in ['pixel','framing','joint','historical']:
            g=raw_fixture();p=TinyPolicy();before=p.logits.detach().clone()
            logs=train(g,p,Config(reward_mode=mode,rollouts=2),steps=8,lr=.1)
            self.assertTrue(all(math.isfinite(x['loss']) for x in logs))
            self.assertFalse(torch.equal(before,p.logits.detach()))
    def test_threshold_strict(self):
        cfg=Config();self.assertFalse(metrics(0,.1,cfg)['effective_gain']);self.assertTrue(metrics(0,.10001,cfg)['effective_gain'])
if __name__=='__main__':unittest.main()
