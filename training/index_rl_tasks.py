"""Incrementally validate completed RL tasks. Does not certify text polishing."""
import argparse, json, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pipeline/plan26_spatial_qa_memory_action/multiround_rl_v1'))
from task_adapter import load_task

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--watch',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    while True:
        ready=[];errors=[]
        for path in sorted(a.tasks.glob('*.json')):
            if path.name.startswith(('receipt_','error_')):continue
            receipt=a.output/(path.stem+'.json')
            if receipt.exists():
                row=json.loads(receipt.read_text())
                if row.get('status')=='structure_and_media_pass':ready.append(row);continue
            try:
                graph=load_task(path)
                row=dict(task=str(path),status='structure_and_media_pass',nodes=len(graph.nodes),
                         edges=len(graph.edges),text_polish_verified=False)
                tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(receipt)
                ready.append(row)
            except Exception as e:errors.append(dict(task=str(path),error=repr(e)))
        tmp=a.output/'manifest.tmp';tmp.write_text(''.join(json.dumps(r)+'\n' for r in ready));tmp.replace(a.output/'manifest.jsonl')
        tmp=a.output/'progress.tmp';tmp.write_text(json.dumps(dict(accepted=len(ready),errors=errors,updated=time.time())));tmp.replace(a.output/'progress.json')
        if not a.watch:break
        time.sleep(60)

if __name__=='__main__':main()
