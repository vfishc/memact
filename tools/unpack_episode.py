"""Materialize one portable annotation record using separately prepared media."""
import argparse,gzip,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shard',type=Path,required=True);p.add_argument('--id',required=True);p.add_argument('--media-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.media_root.resolve()
with gzip.open(a.shard,'rt') as f:
 for line in f:
  row=json.loads(line)
  if row['id']!=a.id:continue
  data=row.get('payload')
  if data is None:
   data={'events':row['events'],'media':row['media'],
         'schedule':[],'answers':[],'actions':[]}
  for media in data['media'].values():
   path=(root/media['path']).resolve()
   if not path.is_relative_to(root):raise ValueError('media path escapes media root')
   if not path.is_file():raise FileNotFoundError(path)
   if media.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=media['sha256']:raise ValueError('media checksum mismatch')
   media['path']=str(path)
  a.output.mkdir(parents=True,exist_ok=False);(a.output/'bundle.json').write_text(json.dumps(data,ensure_ascii=False));break
 else:raise ValueError('episode ID not in shard')
