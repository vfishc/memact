#!/usr/bin/env python3
"""Rebuild a selected prepared-video chunk from upstream RGB data.

Profiles preserve the historical video preprocessing; derived geometry and QA
are supplied separately. Outputs are new files, never replacements of inputs.
"""
import argparse, hashlib, io, json, os, pathlib, struct, tempfile, zipfile
from fractions import Fraction

def exact(f, count):
    data = f.read(count)
    if len(data) != count:
        raise ValueError('Truncated source file')
    return data

def scannet_frames(path):
    import cv2
    import numpy as np
    with path.open('rb') as f:
        version, = struct.unpack('<I', exact(f, 4))
        if version != 4:
            raise ValueError('Expected ScanNet .sens version 4')
        n, = struct.unpack('<Q', exact(f, 8))
        exact(f, n)
        exact(f, 256)
        metadata = struct.unpack('<iiiiifi', exact(f, 28))
        count, = struct.unpack('<Q', exact(f, 8))
        target = 0.0
        for index in range(count):
            exact(f, 64)
            color_time, depth_time = struct.unpack('<QQ', exact(f, 16))
            color_n, depth_n = struct.unpack('<QQ', exact(f, 16))
            if index / 30.0 >= target - .005:
                image = cv2.imdecode(np.frombuffer(exact(f, color_n), np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    if image.shape[1] != 640:
                        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                    yield image, dict(source_frame=index, source_color_timestamp_us=color_time,
                                      selection_time_s=index/30.0)
                    target += 1/24.0
            else:
                f.seek(color_n, 1)
            f.seek(depth_n, 1)

def scannetpp_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot open upstream ScanNet++ video')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 3)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 3)
    width -= width % 2
    height -= height % 2
    target = 0.0
    index = -1
    try:
        while cap.grab():
            index += 1
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms == 0 and index > 0:
                ms = index/fps*1000
            if ms < target - 1.0:
                continue
            ok, image = cap.retrieve()
            if not ok:
                raise ValueError(f'Failed source retrieval at frame {index}')
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(image, (width,height), interpolation=cv2.INTER_AREA)
            yield image, dict(source_frame=index, source_timestamp_s=ms/1000)
            target += 1000/24.0
    finally:
        cap.release()

def encode_cv(args, target):
    import cv2
    generator = scannet_frames(args.input) if args.profile == 'scannet' else scannetpp_frames(args.input)
    begin = args.chunk_index*2160
    records = []
    writer = None
    try:
        for index, (image, provenance) in enumerate(generator):
            if index < begin:
                continue
            if index >= begin+2160:
                break
            if writer is None:
                height,width = image.shape[:2]
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (width,height))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create MPEG-4 video')
            writer.write(image)
            records.append(dict(frame_id=len(records), timestamp_s=len(records)/24.0, **provenance))
    finally:
        generator.close()
        if writer is not None:
            writer.release()
    return records

def encode_arkit(args, target):
    import av
    import numpy as np
    from PIL import Image
    if args.mapping is None:
        raise ValueError('ARKit requires the supplied episode frame mapping')
    records = json.loads(args.mapping.read_text())
    if not records or [x['frame_id'] for x in records] != list(range(len(records))):
        raise ValueError('Mapping must contain consecutive frame IDs')
    with zipfile.ZipFile(args.input) as rgb, av.open(str(target), 'w') as output:
        stream = output.add_stream('libx264', rate=24)
        stream.width,stream.height,stream.pix_fmt = 640,480,'yuv420p'
        stream.options = {'crf':'18','preset':'fast'}
        stream.thread_count = 2
        for row in records:
            image = Image.open(io.BytesIO(rgb.read(row['rgb_member']))).convert('RGB')
            image = np.asarray(image.resize((640,480), Image.Resampling.LANCZOS))
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts, frame.time_base = row['frame_id'], Fraction(1,24)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return records

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['scannet','scannetpp','arkit'],required=True)
    p.add_argument('--input',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--mapping',type=pathlib.Path)
    p.add_argument('--chunk-index',type=int,default=0,help='Zero-based selected chunk, 2160 frames per chunk')
    p.add_argument('--expected-frames',type=int)
    p.add_argument('--report',type=pathlib.Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.report.exists():
        p.error('Output/report already exists; choose a new destination')
    if args.chunk_index<0:
        p.error('Negative chunk index')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rebuild-',dir=args.output.parent) as d:
        temp=pathlib.Path(d)/'output.mp4'
        records=encode_arkit(args,temp) if args.profile=='arkit' else encode_cv(args,temp)
        if not records or (args.expected_frames is not None and len(records)!=args.expected_frames):
            raise ValueError('Empty or unexpected frame count')
        import cv2
        cap=cv2.VideoCapture(str(temp));n=0
        try:
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
        finally:cap.release()
        if n!=len(records):raise ValueError('Encoded video decoding/count check failed')
        digest=hashlib.sha256(temp.read_bytes()).hexdigest()
        os.link(temp,args.output)
    args.report.write_text(json.dumps(dict(status='pass',profile=args.profile,frames=len(records),fps=24,
        sha256=digest,frame_mapping=records,scope='video reconstruction; validate training-frame hashes separately'),indent=2))
if __name__=='__main__':main()
