"""Expand annotation shards into native training bundles without copying images.

Run once for each package. Media must already be downloaded/extracted into its
package root. The output includes SFT, QA-replay, and memory-replay manifests.
"""
import argparse,functools,gzip,hashlib,json,pathlib

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--shards',type=pathlib.Path,nargs='+',required=True)
 p.add_argument('--media-root',type=pathlib.Path,required=True)
 p.add_argument('--output',type=pathlib.Path,required=True)
 p.add_argument('--verify-hashes',action='store_true')
 a=p.parse_args();root=a.media_root.resolve();a.output.mkdir(parents=True,exist_ok=False);(a.output/'episodes').mkdir()
 @functools.lru_cache(maxsize=200000)
 def resolve(path,digest):
  target=(root/path).resolve()
  if not target.is_relative_to(root):raise ValueError('Media escapes package root')
  if not target.is_file():raise FileNotFoundError(target)
  if a.verify_hashes:
   if not digest:raise ValueError('Missing content checksum')
   if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Media checksum mismatch')
  return str(target)
 counts={'streams':0,'qa_turns':0,'memory_write':0,'use_memory':0};seen=set()
 with (a.output/'sft.jsonl').open('w') as sft,(a.output/'qa_replay.jsonl').open('w') as qa,(a.output/'memory_replay.jsonl').open('w') as mem:
  for shard in a.shards:
   with gzip.open(shard,'rt') as source:
    for line in source:
     row=json.loads(line);ident=row['id']
     if ident in seen:raise ValueError('Duplicate example ID: '+ident)
     if pathlib.Path(ident).name!=ident or ident in ('.','..'):raise ValueError('Unsafe example ID')
     seen.add(ident)
     data=row.get('payload') or {'events':row['events'],'media':row['media'],'schedule':[],'answers':[],'actions':[]}
     for m in data['media'].values():m['path']=resolve(m['path'],m.get('sha256'))
     dest=a.output/'episodes'/ident;dest.mkdir();(dest/'bundle.json').write_text(json.dumps(data,ensure_ascii=False))
     record=json.dumps({'directory':'episodes/'+ident})+'\n';sft.write(record)
     kinds=[e['payload'].get('type',e.get('kind')) for e in data['events'] if e['role']=='assistant' and e.get('loss',False)]
     if any(k!='memory_write' for k in kinds):qa.write(record)
     if 'memory_write' in kinds:mem.write(record)
     counts['streams']+=1
     for kind,key in [('answer','qa_turns'),('memory_write','memory_write'),('use_memory','use_memory')]:counts[key]+=kinds.count(kind)
 (a.output/'summary.json').write_text(json.dumps(counts,indent=2))
 print(json.dumps(counts))
if __name__=='__main__':main()
