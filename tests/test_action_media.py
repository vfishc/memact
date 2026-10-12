import pathlib,sys,tempfile,unittest
from PIL import Image
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'pipeline/plan26_spatial_qa_memory_action'))
from evaluate_routed_memory_qa import FrameResolver

class ActionMediaTests(unittest.TestCase):
 def test_new_observation_after_action_is_materialized(self):
  with tempfile.TemporaryDirectory() as directory:
   p=pathlib.Path(directory)/'image.png';Image.new('RGB',(8,8)).save(p)
   media={f:{'path':str(p)} for f in ('before','after')}
   context=[{'role':'user','content':{'frame':'before','timestamp':0}},
            {'role':'assistant','content':{'type':'action','name':'move_forward_d'}},
            {'role':'tool','content':{'type':'action_result','frame':'after'}}]
   messages=FrameResolver(media).materialize(context)
   self.assertEqual([[i['frame'] for i in m['images']] for m in messages],[['before'],[],['after']])
   with self.assertRaises(ValueError):FrameResolver(media).materialize(context[-1:])
   with self.assertRaises(ValueError):FrameResolver({'before':media['before']}).materialize(context)

if __name__=='__main__':unittest.main()
