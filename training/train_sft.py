"""Whole-sequence, assistant-only LoRA SFT; launch with python or torchrun.
Manifest JSONL rows contain directory, relative to the manifest or absolute.
Use prevalidated episodes. Oversize sequences fail; nothing is truncated.
"""
import argparse,json,os,random,sys,time
from pathlib import Path
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline/plan26_spatial_qa_memory_action'),str(ROOT/'pipeline')]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--epochs',type=int,default=1);p.add_argument('--accumulation',type=int,default=8);p.add_argument('--max-updates',type=int,default=0);p.add_argument('--max-tokens',type=int,default=180000);p.add_argument('--seed',type=int,default=26);p.add_argument('--no-save',action='store_true')
 a=p.parse_args()
 if a.accumulation<1 or a.epochs<1 or a.max_tokens<1 or a.max_updates<0:p.error('invalid budgets')
 import torch,torch.distributed as dist
 from torch.nn.parallel import DistributedDataParallel as DDP
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from peft import LoraConfig,get_peft_model
 from joint_episode import load_episode,materialize,event_context,check_visual_batch
 from assistant_targets import assistant_labels
 from chunked_target_loss import chunked_target_loss
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local)
 if world>1:dist.init_process_group('nccl')
 if rank==0:a.output.mkdir(parents=True,exist_ok=False)
 if world>1:dist.barrier()
 rows=[json.loads(l) for l in a.manifest.open() if l.strip()];batch_size=world*a.accumulation
 if len(rows)<batch_size:raise ValueError('Need at least world_size * accumulation manifest items; for smoke use --accumulation 1')
 torch.manual_seed(a.seed)
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
 for name,v in model.named_parameters():
  if 'visual' in name:v.requires_grad_(False)
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.train()
 class LossModel(torch.nn.Module):
  def __init__(self,vlm):super().__init__();self.vlm=vlm
  def forward(self,batch,labels):
   base=self.vlm.get_base_model();hidden=base.model(**batch,use_cache=False).last_hidden_state
   return chunked_target_loss(hidden,base.lm_head,labels,256)
 wrapped=LossModel(model)
 if world>1:wrapped=DDP(wrapped,device_ids=[local])
 params=[v for v in model.parameters() if v.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5)
 updates=0;started=time.time();last_loss=None
 for epoch in range(a.epochs):
  order=list(range(len(rows)));random.Random(a.seed+epoch).shuffle(order)
  for offset in range(0,len(order)-batch_size+1,batch_size):
   opt.zero_grad(set_to_none=True);mean_loss=0.
   for micro in range(a.accumulation):
    row=rows[order[offset+micro*world+rank]];episode=Path(row['directory'])
    if not episode.is_absolute():episode=a.manifest.resolve().parent/episode
    data=load_episode(episode);events=data['events'];last=max(i for i,e in enumerate(events) if e['role']=='assistant');events=events[:last+1]
    if any(e['role']=='assistant' and not e.get('loss',False) for e in events):raise ValueError('Assistant loss-masked events require an event-aware mask; unsupported by this entrypoint')
    messages,receipts=materialize(event_context(events),data['media']);batch=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=False,enable_thinking=False)
    labels,_=assistant_labels(batch['input_ids'],proc.tokenizer);check_visual_batch(batch,receipts,proc)
    if batch['input_ids'].numel()>a.max_tokens:raise ValueError('Unfiltered over-budget episode; prepare a filtered manifest, do not truncate')
    batch={k:v.cuda() if hasattr(v,'cuda') else v for k,v in batch.items()};labels=labels.cuda()
    context=wrapped.no_sync() if world>1 and micro<a.accumulation-1 else nullcontext()
    with context:
     loss=wrapped(batch,labels)
     if not torch.isfinite(loss):raise ValueError('nonfinite loss')
     (loss/a.accumulation).backward()
    mean_loss+=float(loss.detach())/a.accumulation
   norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();updates+=1;last_loss=mean_loss
   if rank==0:
    record={'update':updates,'epoch':epoch,'loss_rank0':mean_loss,'gradient_norm':float(norm),'effective_batch_size':batch_size}
    with (a.output/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
   if a.max_updates and updates>=a.max_updates:break
  if a.max_updates and updates>=a.max_updates:break
 if rank==0:
  if not a.no_save:
   model.save_pretrained(a.output/'adapter');torch.save(opt.state_dict(),a.output/'optimizer.pt')
  (a.output/'report.json').write_text(json.dumps({'status':'pass','updates':updates,'seconds':time.time()-started,'world_size':world,'accumulation':a.accumulation,'effective_batch_size':batch_size,'loss_rank0':last_loss,'dropped_remainder_per_epoch':len(rows)%batch_size,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'resume_supported':False},indent=2))
 if world>1:dist.destroy_process_group()
if __name__=='__main__':main()
