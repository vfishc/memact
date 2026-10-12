"""Run native Timing or score saved predictions; labels stay outside prompts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action')]
from timing_evaluation import run_timing,score_timing
from joint_episode import load_episode

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scoring',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predictions',type=Path);p.add_argument('--model');p.add_argument('--adapter',type=Path);p.add_argument('--limit',type=int);p.add_argument('--max-new-tokens',type=int,default=128)
    a=p.parse_args();rows=[json.loads(l) for l in a.scoring.read_text().splitlines() if l.strip()]
    if a.limit is not None:
        if a.limit<1:raise ValueError('limit must be positive')
        rows=rows[:a.limit]
    a.output.mkdir(parents=True,exist_ok=False)
    if a.predictions:
        predictions=[json.loads(l) for l in a.predictions.read_text().splitlines() if l.strip()]
    else:
        if not a.model:raise ValueError('--model or --predictions required')
        import torch
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        from model_inference import NativeModelCallback
        processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
        model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
        if a.adapter:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,a.adapter)
        model.eval();predictions=[]
        with (a.output/'predictions.jsonl').open('w') as f:
            for row in rows:
                try:
                    path=Path(row['episode_bundle'])
                    if not path.is_absolute():path=a.scoring.parent/path
                    data=load_episode(path.parent)
                    cb=NativeModelCallback(model,processor,data['media'],max_new_tokens=a.max_new_tokens)
                    result=run_timing(row,data['schedule'],cb)
                except Exception as e:result={'id':row['id'],'status':'failed','error':repr(e)}
                predictions.append(result);f.write(json.dumps(result)+'\n');f.flush()
    metrics=score_timing(rows,predictions);metrics['failed_questions']=sum(p.get('status')=='failed' for p in predictions)
    (a.output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics))
if __name__=='__main__':main()
