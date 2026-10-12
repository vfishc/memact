"""Check every reachable action branch before running a costly model evaluation."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pipeline'),str(ROOT/'pipeline/plan26_spatial_qa_memory_action')]
from evaluation_selection import load_selected_episode
from action_graph import make_executor

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--steps',type=int,choices=(1,2,3),default=3);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();text=a.manifest.read_text();rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()];reports=[]
    for row in rows:
        record={'root':row.get('root',row.get('destination'))}
        try:
            data=load_selected_episode(row);ex=make_executor(data,a.steps)
            if ex is None and any(e['kind']=='question' and e['payload'].get('actions') for e in data['schedule']):raise ValueError('Action questions have no executable graph')
            if ex is not None:
                for m in data['media'].values():
                    if not Path(m['path']).is_file():raise FileNotFoundError(m['path'])
            record.update(status='pass',action_graph=ex is not None)
        except Exception as e:record.update(status='failed',error=str(e))
        reports.append(record)
    result=dict(status='pass' if all(r['status']=='pass' for r in reports) else 'failed',steps=a.steps,episodes=reports)
    with a.report.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'episodes':len(reports)}));sys.exit(0 if result['status']=='pass' else 2)
