"""Materialize released evaluation episodes and keep scoring labels separate.

Multistep Action evaluation additionally requires complete action graphs.
"""
import argparse,gzip,hashlib,json,pathlib

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--episode',action='append');a=p.parse_args()
 root=a.package.resolve();a.output.mkdir(parents=True,exist_ok=False);selected=set(a.episode or []);counts={'episodes':0,'main':0,'timing':0}
 def resolve(path,digest=None):
  f=(root/path).resolve()
  if not f.is_relative_to(root) or not f.is_file():raise ValueError('Missing or unsafe media: '+path)
  if digest and hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Media checksum mismatch: '+path)
  return str(f)
 def nested(x):
  if isinstance(x,dict):return {k:nested(v) for k,v in x.items()}
  if isinstance(x,list):return [nested(v) for v in x]
  if isinstance(x,str) and x.startswith('assets/'):return resolve(x)
  return x
 ids=set()
 with (a.output/'episodes.jsonl').open('w') as manifest:
  for shard in sorted((root/'shards').glob('episode-*.json.gz')):
   with gzip.open(shard,'rt') as f:row=json.load(f)
   ident=row['id']
   if selected and ident not in selected:continue
   if pathlib.Path(ident).name!=ident or ident in ('.','..') or ident in ids:raise ValueError('Unsafe or duplicate ID')
   ids.add(ident);data=row['payload']
   for m in data['media'].values():m['path']=resolve(m['path'],m.get('sha256'))
   data=nested(data);dest=a.output/'episodes'/ident;dest.mkdir(parents=True);(dest/'bundle.json').write_text(json.dumps(data))
   manifest.write(json.dumps(dict(root=str(dest.resolve()),episode=ident,source=row['source'],scene=row['scene']))+'\n');counts['episodes']+=1
 if selected-ids:raise ValueError('Unknown episode IDs')
 for family in ('main','timing'):
  with (a.output/(family+'.scoring.jsonl')).open('w') as out:
   for shard in sorted((root/'shards').glob(family+'-*.jsonl.gz')):
    with gzip.open(shard,'rt') as f:
     for line in f:
      row=json.loads(line)
      if row['episode'] not in ids:continue
      row['episode_bundle']=str((a.output/'episodes'/row['episode']/'bundle.json').resolve());out.write(json.dumps(row)+'\n');counts[family]+=1
 (a.output/'summary.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts))
if __name__=='__main__':main()
