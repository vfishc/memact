"""Copy completed passive streams and content-addressed images into a portable tree."""
import argparse,json,pathlib,hashlib,shutil,time,concurrent.futures,os

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--watch',action='store_true');a=p.parse_args()
 (a.output/'episodes').mkdir(parents=True,exist_ok=True);(a.output/'assets').mkdir(exist_ok=True)
 def process(receipt):
  src=receipt.parent;dest=a.output/'episodes'/src.name
  if (dest/'receipt.json').exists():return json.loads((dest/'receipt.json').read_text())
  r=json.loads(receipt.read_text());assert r['status']=='pass';assert r['budget']['tokens']<=180000
  events=src/'events.jsonl'
  if not events.exists():events=src/'events.json'
  raw=events.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['events_sha256']
  media=json.loads((src/'media.json').read_text());out={}
  (dest/'frames').mkdir(parents=True,exist_ok=True)
  for frame,m in media.items():
   source=pathlib.Path(m['path']);source=source if source.is_absolute() else src/source
   digest=m['sha256'];suffix=source.suffix.lower();target=a.output/'assets'/(digest+suffix)
   if not target.exists():
    data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
    tmp=target.with_name(target.name+'.'+src.name+'.tmp');tmp.write_bytes(data);tmp.replace(target)
   local=dest/'frames'/target.name
   if not local.exists():os.link(target,local)
   out[frame]={**m,'path':'frames/'+target.name}
  dest.mkdir(exist_ok=True);(dest/events.name).write_bytes(raw)
  if (src/'bundle.json').exists():
   bundle=json.loads((src/'bundle.json').read_text());bundle['media']=out;(dest/'bundle.json').write_text(json.dumps(bundle))
  (dest/'media.json').write_text(json.dumps(out));result={**r,'directory':str(dest),'source_directory':str(src),'portable_media':True,'media_count':len(out)}
  temporary=dest/'receipt.tmp';temporary.write_text(json.dumps(result));temporary.replace(dest/'receipt.json');return result
 while True:
  receipts=[p for p in sorted(a.source.glob('*/receipt.json')) if json.loads(p.read_text()).get('status')=='pass'];rows=[];errors=[]
  with concurrent.futures.ThreadPoolExecutor(2) as pool:
   futures={pool.submit(process,p):p for p in receipts}
   for f in concurrent.futures.as_completed(futures):
    try:rows.append(f.result())
    except Exception as e:errors.append({'source':str(futures[f]),'error':repr(e)})
    if len(rows)%50==0:(a.output/'progress.json').write_text(json.dumps(dict(streams=len(rows),qa=sum(r['qa'] for r in rows),errors=errors,updated=time.time())))
  tmp=a.output/'manifest.tmp';tmp.write_text(''.join(json.dumps(dict(directory='episodes/'+pathlib.Path(r['directory']).name,qa=r['qa']))+'\n' for r in sorted(rows,key=lambda r:r['directory'])));tmp.replace(a.output/'manifest.jsonl')
  (a.output/'progress.json').write_text(json.dumps(dict(streams=len(rows),qa=sum(r['qa'] for r in rows),errors=errors,updated=time.time())))
  if not a.watch:break
  time.sleep(60)
if __name__=='__main__':main()
