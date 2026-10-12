"""Run a supported data-generation entry point from any working directory."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'pipeline'
ENTRY={'generate':'generate_routed_memory_qa.py','generate-batch':'batch_routed_memory_qa.py','export-action':'export_action_streams.py','metadata':'../build_episode_metadata_cache.py'}
if len(sys.argv)<2 or sys.argv[1] not in ENTRY:raise SystemExit('Usage: python generation/run.py {'+','.join(ENTRY)+'} [arguments]')
name=sys.argv.pop(1);path=(ROOT/'plan26_spatial_qa_memory_action'/ENTRY[name]).resolve();sys.path[:0]=[str(path.parent),str(ROOT/'plan26_spatial_qa_memory_action'),str(ROOT)];sys.argv[0]=str(path);runpy.run_path(str(path),run_name='__main__')
