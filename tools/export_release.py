"""Export existing manifests to portable annotation shards without changing sources.
Media are referenced by content hash and copied separately after provenance review.
"""
import argparse,concurrent.futures,gzip,hashlib,json,pathlib,collections,time

def load(row):
 p=pathlib.Path(row['directory']);raw=(p/'events.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=row['events_sha256']:raise ValueError(f'events checksum mismatch: {p.name}')
 events=json.loads(raw)
 if (p/'bundle.json').exists():
  data=json.loads((p/'bundle.json').read_text())
 else:
  data={'events':events,'media':json.loads((p/'media.json').read_text())}
  for key,name in [('schedule','schedule.json'),('answers','answers.json'),('actions','cached_actions.json')]:
   if (p/name).exists():data[key]=json.loads((p/name).read_text())
 if data['events']!=events:raise ValueError('bundle and events differ')
 qa=sum(e['role']=='assistant' and e.get('kind',e.get('payload',{}).get('type'))=='answer' for e in events)
 expected=row.get('qa',row.get('budget',{}).get('qa_turns'))
 if expected is None or qa!=expected:raise ValueError('QA count mismatch or missing declared count')
 for media in data.get('media',{}).values():
  path=pathlib.Path(media['path'])
  if not path.is_absolute():
   path=(p/path).resolve()
   if not path.is_relative_to(p.resolve()):raise ValueError('media escapes episode')
   media['path']=str(path)
 locators={}
 def portable(v):
  if isinstance(v,dict):
   result={k:portable(x) for k,x in v.items()}
   if isinstance(v.get('path'),str) and v['path'].startswith('/'):
    path=v['path'];digest=v.get('sha256');key=digest or hashlib.sha256(path.encode()).hexdigest()
    rel='media/'+key+pathlib.Path(path).suffix
    locators[rel]={'asset':rel,'source_path':path,'sha256':digest,'content_verified':False}
    result['path']=rel
   return result
  if isinstance(v,list):return [portable(x) for x in v]
  if isinstance(v,str) and v.startswith(('/mnt/','/home/','/data/','/root/')):
   key=hashlib.sha256(v.encode()).hexdigest();rel='source_reference/'+key
   locators.setdefault(rel,{'asset':rel,'source_path':v,'sha256':None,'content_verified':False});return rel
  return v
 is_action='action_protocol' in row
 result={'id':p.name,'source':row.get('source','sim_front3d' if is_action else None),'scene':row.get('scene'),'split':row.get('split'),'legacy_split':row.get('legacy_split',row.get('split')),'qa_turns':qa,'families':row.get('families',{'action':qa} if is_action else {}),'payload':portable(data)}
 if is_action:
  result['action_metadata']={k:row.get(k) for k in ['template','steps','action_protocol','evidence_type','evidence_provenance']}
 return result,list(locators.values())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=pathlib.Path,action='append',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--shard-size',type=int,default=10000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 rows=(json.loads(l) for m in a.manifest for l in m.open());count=qa=0;families=collections.Counter();shard=None;start=time.time()
 with gzip.open(a.output/'media_locators.private.jsonl.gz','wt') as loc,concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  while True:
   batch=[]
   for _ in range(256):
    try:batch.append(next(rows))
    except StopIteration:break
   if not batch:break
   for result,assets in pool.map(load,batch):
    if count%a.shard_size==0:
     if shard:shard.close()
     shard=gzip.open(a.output/f'annotations-{count//a.shard_size:05d}.jsonl.gz','wt')
    shard.write(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    for asset in assets:loc.write(json.dumps(asset,separators=(',',':'))+'\n')
    count+=1;qa+=result['qa_turns'];families.update(result['families'])
   (a.output/'progress.json').write_text(json.dumps({'streams':count,'qa_turns':qa,'seconds':time.time()-start,'status':'running'}))
 if shard:shard.close()
 report={'status':'annotation_export_complete','streams':count,'qa_turns':qa,'families':dict(families),'seconds':time.time()-start,'media_copied':False,'membership':'as specified by input manifests'}
 (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
