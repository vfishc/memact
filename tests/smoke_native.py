"""One real-RGB LoRA update. A smoke test, not an experiment reproduction."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline/plan26_spatial_qa_memory_action'),str(ROOT/'pipeline')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 import torch,transformers,peft
 from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
 from peft import LoraConfig,get_peft_model
 from joint_episode import load_episode,materialize,event_context,check_visual_batch
 from assistant_targets import assistant_labels
 from chunked_target_loss import chunked_target_loss
 torch.manual_seed(26);started=time.time()
 data=load_episode(a.episode);events=data['events'];last=max(i for i,e in enumerate(events) if e['role']=='assistant');events=events[:last+1]
 messages,receipts=materialize(event_context(events),data['media'])
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 batch=proc.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=False,enable_thinking=False)
 labels,targets=assistant_labels(batch['input_ids'],proc.tokenizer)
 vision=check_visual_batch(batch,receipts,proc)
 if batch['input_ids'].numel()>12000:raise ValueError('Smoke input exceeds 12000 tokens')
 model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
 model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
 for name,v in model.named_parameters():
  if 'visual' in name:v.requires_grad_(False)
 model.gradient_checkpointing_enable();model.enable_input_require_grads();model.train()
 params=[v for v in model.parameters() if v.requires_grad];before=[v.detach().clone() for v in params]
 opt=torch.optim.AdamW(params,lr=1e-5);batch={k:v.cuda() if hasattr(v,'cuda') else v for k,v in batch.items()}
 base=model.get_base_model();hidden=base.model(**batch,use_cache=False).last_hidden_state
 loss=chunked_target_loss(hidden,base.lm_head,labels.cuda(),256)
 if not torch.isfinite(loss):raise ValueError('nonfinite loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
 delta=max(float((v.detach()-b).abs().max()) for v,b in zip(params,before))
 if not delta>0:raise ValueError('no parameter update')
 report={'status':'pass','scope':'one_real_rgb_sft_update','tokens':batch['input_ids'].numel(),'target_tokens':targets,'images':len(receipts),'loss':float(loss.detach()),'gradient_norm':float(norm),'parameter_max_delta':delta,'seconds':time.time()-started,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'supervision_types':sorted({e['kind'] for e in events if e['role']=='assistant'}),'limitations':['Not full training','Not RL validation','Not an 180000-token stress test','No checkpoint included']}
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
