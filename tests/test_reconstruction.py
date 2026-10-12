"""Source-frame alignment and safe reconstruction outputs."""
import importlib.util
import pathlib
import struct
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'reconstruction'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_scannet_failed_decode_does_not_advance_output_clock(tmp_path):
    import cv2,numpy as np
    raw=tmp_path/'sample.sens'
    with raw.open('wb') as f:
        f.write(struct.pack('<IQ',4,0));f.write(bytes(256))
        f.write(struct.pack('<iiiiifi',2,1,640,480,640,480.,1000))
        f.write(struct.pack('<Q',10))
        for i in range(10):
            image=np.full((480,640,3),i*10,dtype=np.uint8)
            ok,data=cv2.imencode('.jpg',image);assert ok
            data=b'not-a-jpeg' if i==2 else data.tobytes()
            f.write(bytes(64));f.write(struct.pack('<QQQQ',i*33333,i*33333,len(data),0));f.write(data)
    rows=list(module('reconstruct_real_video').scannet_frames(raw))
    assert [x[1]['source_frame'] for x in rows]==[0,3,4,5,6,7,8,9]
    assert rows[-1][1]['source_color_timestamp_us']==9*33333

def test_truncated_sensor_rejected(tmp_path):
    p=tmp_path/'truncated.sens';p.write_bytes(struct.pack('<I',4))
    with pytest.raises(ValueError,match='Truncated'):
        list(module('reconstruct_real_video').scannet_frames(p))

@pytest.mark.parametrize('path',['../outside.jpg','/absolute.jpg'])
def test_frame_paths_cannot_escape_root(tmp_path,path):
    with pytest.raises(ValueError):module('reconstruct_frames').contained(tmp_path,path)
