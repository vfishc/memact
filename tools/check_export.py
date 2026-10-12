"""Validate annotation shard integrity/counts; does not certify referenced media."""
import argparse,gzip,json,collections,concurrent.futures,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
def check(path):
 n=q=0;families=collections.Counter();types=collections.Counter()
 with gzip.open(path,'rt') as f:
  for line in f:
   d=json.loads(line);events=d['payload']['events'];actual=sum(e['role']=='assistant' and e['kind']=='answer' for e in events)
   if actual!=d['qa_turns']:raise ValueError('QA count mismatch')
   n+=1;q+=actual;families.update(d['families']);types.update(e['kind'] for e in events if e['role']=='assistant')
 digest=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
 return {'file':path.name,'streams':n,'qa_turns':q,'families':dict(families),'assistant_events':dict(types),'sha256':digest.hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(check,sorted(a.input.glob('annotations-*.jsonl.gz'))))
if not results:raise ValueError('no shards')
total={'status':'shard_integrity_pass','streams':sum(r['streams'] for r in results),'qa_turns':sum(r['qa_turns'] for r in results),'media_verified':False,'shards':results};a.report.write_text(json.dumps(total,indent=2));print(json.dumps({k:v for k,v in total.items() if k!='shards'}))
