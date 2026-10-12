from pathlib import Path
import sys
r=Path(__file__).resolve().parents[1]/"pipeline"
for p in [r,r/"plan26_spatial_qa_memory_action",r/"plan26_spatial_qa_memory_action/multiround_rl_v1"]: sys.path.insert(0,str(p))
