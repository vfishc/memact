"""Public-history Qwen policy; private labels are never passed to this adapter."""
import json
import torch
from PIL import Image
from core import ACTIONS
class NativePolicy:
 def __init__(self,model,processor,max_tokens=180000):
  self.model=model;self.processor=processor;self.max_tokens=max_tokens;self.answer_cache={}
 def messages(self,view,answer=False):
  content=[]
  for e in view['events']:
   if e['kind']=='action':content.append({'type':'text','text':'Camera command: '+e['value']});continue
   value=e['value'];o=value.get('observation',value)
   if 'status' in value:content.append({'type':'text','text':'Tool status: '+value['status']})
   if 'timestamp' in o:content.append({'type':'text','text':'Timestamp (s): '+str(o['timestamp'])})
   if o.get('image'):
    with Image.open(o['image']) as image:rgb=image.convert('RGB').copy()
    content.append({'type':'image','image':rgb})
  text=view['question']+'\nRemaining camera attempts: '+str(view['remaining_actions'])
  if answer:text+='\nReturn only a JSON object {"type":"answer","answer":value}. Use the specified numerical unit or one uppercase option letter.'
  else:text+='\nSelect one camera command or answer_now. Return only its name. Commands: '+', '.join(ACTIONS)+'. Translations are 0.5 m without rotation; rotations are 45 degrees in place.'
  content.append({'type':'text','text':text});return [{'role':'user','content':content}]
 def batch(self,messages):
  b=self.processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
  if b['input_ids'].shape[1]>self.max_tokens:raise ValueError('Smoke context exceeds token limit; no truncation')
  return {k:v.cuda() if hasattr(v,'cuda') else v for k,v in b.items()}
 def one_score(self,view,action):
  b=self.batch(self.messages(view));ids=self.processor.tokenizer.encode(action,add_special_tokens=False);prefix=b['input_ids'].shape[1]
  b['input_ids']=torch.cat([b['input_ids'],torch.tensor([ids],device='cuda')],1)
  b['attention_mask']=torch.ones_like(b['input_ids'])
  if 'mm_token_type_ids' in b:
   b['mm_token_type_ids']=torch.cat([b['mm_token_type_ids'],torch.zeros((1,len(ids)),dtype=b['mm_token_type_ids'].dtype,device='cuda')],1)
  base=self.model.get_base_model();hidden=base.model(**b,use_cache=False).last_hidden_state
  logits=base.lm_head(hidden[:,prefix-1:prefix+len(ids)-1]).float()
  target=torch.tensor(ids,device='cuda').view(1,-1,1)
  return logits.log_softmax(-1).gather(-1,target).sum()
 def scores(self,view):return torch.stack([self.one_score(view,act) for act in ACTIONS])
 def answer(self,view,rng):
  key=json.dumps(view,sort_keys=True)
  if key in self.answer_cache:return self.answer_cache[key]
  b=self.batch(self.messages(view,True))
  with torch.no_grad():out=self.model.generate(**b,max_new_tokens=64,do_sample=False)
  text=self.processor.tokenizer.decode(out[0,b['input_ids'].shape[1]:],skip_special_tokens=True).strip()
  try:
   obj=json.loads(text);value=obj['answer'] if obj.get('type')=='answer' else None;self.answer_cache[key]=value;return value
  except (ValueError,KeyError,AttributeError):return None
