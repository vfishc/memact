"""Real-image six-candidate RL update; explicitly a smoke, not full training.
Uses the same frozen policy for rollout and reference at the first update.
--initial evaluates full remaining-horizon continuations; default tests a
reachable final decision to bound runtime. No checkpoint is written.
"""
import argparse,json,pathlib,sys,time,random
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1'))
from core import ACTIONS,Config,candidate_values,objective
from task_adapter import load_task

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--task',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--initial',action='store_true');p.add_argument('--max-tokens',type=int,default=24000)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 import torch
 from PIL import Image
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from peft import LoraConfig,get_peft_model
 start=time.time();torch.manual_seed(26)
 graph=load_task(a.task);cfg=Config(reward_mode='historical');state=graph.initial(cfg)
 if not a.initial:
  for _ in range(2):
   action=next((act for act in ACTIONS[1:] if graph.edges[state.node,act]['valid']),ACTIONS[1])
   state,_=graph.advance(state,action,cfg)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
 for name,param in model.named_parameters():
  if 'visual' in name:param.requires_grad_(False)
 model.eval()
 class Policy:
  answer_cache={}
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
   b=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=False)
   if b['input_ids'].shape[1]>a.max_tokens:raise ValueError('Smoke context exceeds token limit; no truncation')
   return {k:v.cuda() if hasattr(v,'cuda') else v for k,v in b.items()}
  def one_score(self,view,action):
   b=self.batch(self.messages(view));ids=proc.tokenizer.encode(action,add_special_tokens=False);prefix=b['input_ids'].shape[1]
   b['input_ids']=torch.cat([b['input_ids'],torch.tensor([ids],device='cuda')],1)
   b['attention_mask']=torch.ones_like(b['input_ids'])
   if 'mm_token_type_ids' in b:
    b['mm_token_type_ids']=torch.cat([b['mm_token_type_ids'],torch.zeros((1,len(ids)),dtype=b['mm_token_type_ids'].dtype,device='cuda')],1)
   base=model.get_base_model();hidden=base.model(**b,use_cache=False).last_hidden_state
   logits=base.lm_head(hidden[:,prefix-1:prefix+len(ids)-1]).float()
   target=torch.tensor(ids,device='cuda').view(1,-1,1)
   return logits.log_softmax(-1).gather(-1,target).sum()
  def scores(self,view):return torch.stack([self.one_score(view,act) for act in ACTIONS])
  def answer(self,view,rng):
   key=json.dumps(view,sort_keys=True)
   if key in self.answer_cache:return self.answer_cache[key]
   b=self.batch(self.messages(view,True))
   with torch.no_grad():out=model.generate(**b,max_new_tokens=64,do_sample=False)
   text=proc.tokenizer.decode(out[0,b['input_ids'].shape[1]:],skip_special_tokens=True).strip()
   try:
    obj=json.loads(text);value=obj['answer'] if obj.get('type')=='answer' else None;self.answer_cache[key]=value;return value
   except (ValueError,KeyError,AttributeError):return None
 policy=Policy();view=graph.public_view(state,cfg)
 # No optimizer mutation occurs until all sampled targets and reference scores exist.
 with torch.no_grad():values,records=candidate_values(graph,state,policy,cfg,seed=26);scores=policy.scores(view)
 detached=scores.detach().clone().requires_grad_(True);loss=objective(detached,scores,values,beta=.05)
 coeff,=torch.autograd.grad(loss,detached)
 params=[v for v in model.parameters() if v.requires_grad];before=[v.detach().clone() for v in params];opt=torch.optim.AdamW(params,lr=1e-5);opt.zero_grad()
 # Recompute each score: identical objective gradient without retaining six VLM graphs.
 for i,act in enumerate(ACTIONS):(coeff[i].detach()*policy.one_score(view,act)).backward()
 norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
 delta=max(float((v.detach()-old).abs().max()) for v,old in zip(params,before))
 if not delta>0:raise ValueError('No parameter update')
 report=dict(status='pass',scope='real_rgb_six_candidate_native_rl_smoke',initial_history=a.initial,remaining_actions=cfg.max_steps-state.steps,rollouts=cfg.rollouts,values=values.tolist(),loss=float(loss.detach()),gradient_norm=float(norm),parameter_max_delta=delta,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.time()-start,limitations=['Fresh LoRA on base model, not experiment checkpoint','Replay mixture and distributed training not exercised','One update only; no saved weights'])
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
