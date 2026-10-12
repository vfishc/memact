"""Six-candidate cumulative-return RL with 50/25/25 supervised replay.
Launch with python or torchrun. Input manifests must be prevalidated.
No automatic truncation, missing-branch substitution, or fabricated QA rewards.
"""
import argparse,json,os,pathlib,random,sys,time
from contextlib import contextmanager
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action'),str(ROOT/'pipeline')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['model','action-manifest','qa-manifest','memory-manifest','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--sft-adapter');p.add_argument('--updates',type=int,required=True);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--max-tokens',type=int,default=180000);p.add_argument('--seed',type=int,default=26);p.add_argument('--reward-mode',choices=['pixel','framing','joint','historical'],default='historical');p.add_argument('--no-save',action='store_true');p.add_argument('--smoke-base',action='store_true');p.add_argument('--smoke-cycle',action='store_true');p.add_argument('--smoke-final-decision',action='store_true')
 a=p.parse_args()
 if a.updates<1 or a.accumulation<1:p.error('Positive update and accumulation counts required')
 if not a.sft_adapter and not a.smoke_base:p.error('Full RL requires --sft-adapter; --smoke-base is only for runtime validation')
 if (a.smoke_cycle or a.smoke_final_decision) and not a.smoke_base:p.error('Smoke overrides require --smoke-base')
 import torch,torch.distributed as dist
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from peft import PeftModel,LoraConfig,get_peft_model
 from core import ACTIONS,Config,candidate_values,objective,rollout,sample_action
 from task_adapter import load_task
 from native_policy import NativePolicy
 from joint_episode import load_episode,event_context,materialize,check_visual_batch
 from assistant_targets import assistant_labels
 from chunked_target_loss import chunked_target_loss
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));device=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(device)
 if world>1:dist.init_process_group('nccl')
 output=pathlib.Path(a.output)
 if rank==0:output.mkdir(parents=True,exist_ok=False)
 if world>1:dist.barrier()
 def manifest(path,key):
  path=pathlib.Path(path);rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  if not rows:raise ValueError('Empty manifest: '+str(path))
  return [str((path.parent/pathlib.Path(r[key])).resolve()) for r in rows]
 pools=[manifest(a.action_manifest,'task'),manifest(a.qa_manifest,'directory'),manifest(a.memory_manifest,'directory')]
 torch.manual_seed(a.seed);rng=random.Random(a.seed+rank)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 if a.sft_adapter:model=PeftModel.from_pretrained(model,a.sft_adapter,is_trainable=True)
 else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
 for name,v in model.named_parameters():
  if 'visual' in name:v.requires_grad_(False)
 model.eval();params=[v for v in model.parameters() if v.requires_grad];reference=[v.detach().cpu().clone() for v in params]
 opt=torch.optim.AdamW(params,lr=1e-5);policy=NativePolicy(model,proc,a.max_tokens);cfg=Config(reward_mode=a.reward_mode);counts=[0,0,0];started=time.time()
 @contextmanager
 def reference_weights():
  current=[v.detach().cpu().clone() for v in params]
  try:
   with torch.no_grad():
    for v,frozen in zip(params,reference):v.copy_(frozen)
   yield
  finally:
   with torch.no_grad():
    for v,saved in zip(params,current):v.copy_(saved)
 def replay_loss(directory,kind):
  data=load_episode(directory);events=data['events'];last=max(i for i,e in enumerate(events) if e['role']=='assistant');events=events[:last+1]
  messages,receipts=materialize(event_context(events),data['media']);b=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=False,enable_thinking=False);check_visual_batch(b,receipts,proc)
  if b['input_ids'].numel()>a.max_tokens:raise ValueError('Replay exceeds token budget')
  labels,_=assistant_labels(b['input_ids'],proc.tokenizer)
  spans=[];start=None
  for i,value in enumerate(labels[0].tolist()+[-100]):
   if value!=-100 and start is None:start=i
   if value==-100 and start is not None:spans.append((start,i));start=None
  assistants=[e for e in events if e['role']=='assistant']
  if len(spans)!=len(assistants):raise ValueError('Assistant span/event alignment mismatch')
  for (start,end),e in zip(spans,assistants):
   memory=e['payload'].get('type')=='memory_write'
   if not e.get('loss',False) or memory!=(kind==2):labels[:,start:end]=-100
  if not bool((labels!=-100).any()):raise ValueError('Replay item has no targets for requested family')
  b={k:v.cuda() if hasattr(v,'cuda') else v for k,v in b.items()};base=model.get_base_model();hidden=base.model(**b,use_cache=False).last_hidden_state
  return chunked_target_loss(hidden,base.lm_head,labels.cuda(),256)
 for update in range(a.updates):
  opt.zero_grad(set_to_none=True);total=0.;policy.answer_cache.clear()
  for micro in range(a.accumulation):
   kind=(update*a.accumulation+micro)%3 if a.smoke_cycle else rng.choices(range(3),weights=[.5,.25,.25])[0];counts[kind]+=1;item=rng.choice(pools[kind])
   if kind:
    loss=replay_loss(item,kind)
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite replay loss')
    (loss/a.accumulation).backward();total+=float(loss.detach())/a.accumulation;continue
   graph=load_task(item);initial=graph.initial(cfg)
   # Parameters remain unchanged for every rollout and microbatch in this update.
   # eval() plus no_grad() therefore provides a fixed snapshot without a second VLM.
   if a.smoke_final_decision:
    for _ in range(2):initial,_=graph.advance(initial,ACTIONS[1],cfg)
    states=[initial]
   else:
    first=sample_action(policy,graph.public_view(initial,cfg),rng)
    sampled=rollout(graph,initial,first,policy,cfg,rng.randrange(2**31));states=[initial]+[s for s in sampled['visited'] if s.steps<cfg.max_steps]
   for state in states:
    view=graph.public_view(state,cfg)
    with torch.no_grad():values,_=candidate_values(graph,state,policy,cfg,rng.randrange(2**31));scores=policy.scores(view)
    with reference_weights(),torch.no_grad():ref_scores=policy.scores(view)
    probe=scores.detach().clone().requires_grad_(True);loss=objective(probe,ref_scores,values,beta=.05)/(a.accumulation*len(states));coeff,=torch.autograd.grad(loss,probe)
    for i,act in enumerate(ACTIONS):(coeff[i].detach()*policy.one_score(view,act)).backward()
    total+=float(loss.detach())
  if world>1:
   for v in params:
    if v.grad is None:v.grad=torch.zeros_like(v)
    dist.all_reduce(v.grad);v.grad.div_(world)
  norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
  record=dict(update=update+1,rank=rank,loss=total,gradient_norm=float(norm),item_counts=counts.copy(),seconds=time.time()-started)
  with (output/f'updates_rank{rank}.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
 if rank==0:
  if not a.no_save:model.save_pretrained(output/'adapter');torch.save(opt.state_dict(),output/'optimizer.pt')
  (output/'report.json').write_text(json.dumps(dict(status='pass',updates=a.updates,world_size=world,effective_batch=world*a.accumulation,smoke=a.smoke_base,smoke_cycle=a.smoke_cycle,final_decision_only=a.smoke_final_decision,replay_probabilities=[.5,.25,.25],rank0_counts=counts,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,resume_supported=False),indent=2))
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
