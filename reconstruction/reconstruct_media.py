#!/usr/bin/env python3
"""Batch upstream-video reconstruction using released video_jobs.jsonl."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,sys
from reconstruct_frames import contained

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=pathlib.Path,required=True)
    p.add_argument('--raw-root',type=pathlib.Path,required=True)
    p.add_argument('--output-root',type=pathlib.Path,required=True)
    p.add_argument('--report',type=pathlib.Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--profile',choices=['arkit','scannet','scannetpp'])
    p.add_argument('--limit',type=int)
    a=p.parse_args();jobs=[json.loads(l) for l in a.manifest.open() if l.strip()]
    if a.profile:jobs=[x for x in jobs if x['profile']==a.profile]
    if a.limit is not None:jobs=jobs[:a.limit]
    destinations=[str(contained(a.output_root,j['output'])) for j in jobs]
    if len(set(destinations))!=len(destinations):raise ValueError('Duplicate video output')
    def run(j):
        source=contained(a.raw_root,j['input']);out=contained(a.output_root,j['output']);receipt=out.with_suffix('.reconstruction.json')
        if out.exists() and receipt.exists():
            x=json.loads(receipt.read_text())
            if x['status']=='pass' and digest(out)==x['sha256']:return dict(video=j['output'],status='verified_existing')
            raise ValueError('Existing output checksum mismatch')
        if not source.is_file():raise FileNotFoundError(str(source))
        cmd=[sys.executable,str(pathlib.Path(__file__).with_name('reconstruct_real_video.py')),'--profile',j['profile'],'--input',str(source),'--output',str(out),'--chunk-index',str(j['chunk_index']),'--report',str(receipt)]
        if j.get('mapping'):cmd+=['--mapping',str(contained(a.manifest.parent,j['mapping']))]
        if j.get('expected_frames') is not None:cmd+=['--expected-frames',str(j['expected_frames'])]
        process=subprocess.run(cmd,text=True,capture_output=True)
        if process.returncode:raise RuntimeError(process.stderr[-3000:])
        return dict(video=j['output'],status='reconstructed')
    results=[];errors=[]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        fs={pool.submit(run,j):j['video_id'] for j in jobs}
        for f in concurrent.futures.as_completed(fs):
            try:results.append(f.result())
            except Exception as e:errors.append(dict(video_id=fs[f],error=repr(e)))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(dict(status='fail' if errors else 'pass',jobs=len(jobs),results=results,errors=errors),indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
