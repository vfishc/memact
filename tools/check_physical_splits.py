"""Check recording IDs and physical groups separately before releasing data.

Inputs are JSONL rows with source and scene. ARKitScenes requires its official
video_id/visit_id CSV. Unknown visits are reported, never merged into one group.
"""
import argparse,collections,csv,json,pathlib,re

def physical_group(source, scene, arkit):
 scene=str(scene)
 if source=='real_scannet':
  m=re.fullmatch(r'(scene\d+)_\d+(?:_part\d+)?',scene)
  if not m:raise ValueError('Unrecognized ScanNet recording ID')
  return source,m.group(1)
 if source=='real_arkit':
  video=re.sub(r'_part\d+$','',scene);visit=arkit.get(video)
  return None if visit in (None,'','NA','nan') else (source,visit)
 if source=='real_scannetpp':return source,scene.split('_chunk_')[0]
 if source=='sim_front3d':return source,scene
 raise ValueError('Unrecognized source')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',type=pathlib.Path,required=True);p.add_argument('--evaluation',type=pathlib.Path,required=True);p.add_argument('--arkit-map',type=pathlib.Path,required=True);p.add_argument('--report',type=pathlib.Path,required=True);a=p.parse_args()
 with a.arkit_map.open() as f:arkit={x['video_id']:x['visit_id'] for x in csv.DictReader(f)}
 sets={};unknown={};rows={}
 for name,path in [('train',a.train),('evaluation',a.evaluation)]:
  rows[name]=[json.loads(l) for l in path.open() if l.strip()];sets[name]=set();unknown[name]=set()
  for x in rows[name]:
   group=physical_group(x['source'],x['scene'],arkit)
   if group is None:unknown[name].add((x['source'],str(x['scene'])))
   else:sets[name].add(group)
 overlap=sets['train']&sets['evaluation'];affected=[x for x in rows['train'] if physical_group(x['source'],x['scene'],arkit) in overlap]
 report=dict(status='fail' if overlap else ('incomplete' if any(unknown.values()) else 'pass'),overlapping_groups=sorted(overlap),affected_training_rows=len(affected),affected_qa=sum(x.get('qa',x.get('qa_turns',1)) for x in affected),unknown_groups={k:sorted(v) for k,v in unknown.items()})
 with a.report.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report));return 0 if report['status']=='pass' else 2
if __name__=='__main__':raise SystemExit(main())
