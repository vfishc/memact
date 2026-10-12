"""Generate additional spatial/memory QA candidates from aligned episode facts.
Outputs require subsequent stream assembly and acceptance, not just counting.
"""
import argparse,itertools,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'pipeline'
sys.path[:0]=[str(root/'plan26_spatial_qa_memory_action'),str(root)]
from extra_candidates import spatial_candidates,area_candidates,memory_candidates,EpisodeFacts
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--facts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
f=EpisodeFacts(json.loads(a.facts.read_text()))
with a.output.open('x') as out:
 for s in itertools.chain(memory_candidates(f),spatial_candidates(f),area_candidates(f)):out.write(json.dumps(s,ensure_ascii=False)+'\n')
