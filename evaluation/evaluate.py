"""Main QA evaluation entry point."""
import sys, runpy
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / 'pipeline'
path = ROOT / 'plan26_spatial_qa_memory_action' / 'evaluate_joint.py'
sys.path[:0] = [str(path.parent), str(ROOT)]
sys.argv[0] = str(path)
runpy.run_path(str(path), run_name='__main__')
