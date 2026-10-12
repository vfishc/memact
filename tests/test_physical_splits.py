import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'tools/check_physical_splits.py'
s=importlib.util.spec_from_file_location('split_check',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_scannet_rescans_share_physical_group():
 assert m.physical_group('real_scannet','scene0002_00',{})==m.physical_group('real_scannet','scene0002_02_part3',{})
def test_arkit_videos_share_visit_not_unknown_group():
 mapping={'10':'100','11':'100','12':'NA'}
 assert m.physical_group('real_arkit','10_part001',mapping)==m.physical_group('real_arkit','11',mapping)
 assert m.physical_group('real_arkit','12',mapping) is None
 assert m.physical_group('real_arkit','13',mapping) is None
def test_scannetpp_chunks_share_scene():
 assert m.physical_group('real_scannetpp','abc_chunk_0',{})==m.physical_group('real_scannetpp','abc_chunk_7',{})
