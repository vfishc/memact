#!/usr/bin/env python3
"""Restore selected training JPEGs from prepared videos using a relative manifest.

Each JSONL row: {video: relative_path, frames: [{index, output, sha256?, width?, height?}]}.
Paths are relative to --video-root and --output-root. Existing files are verified.
"""
import argparse, concurrent.futures, hashlib, json, os, pathlib, tempfile

def contained(root, relative):
    p = pathlib.Path(relative)
    if p.is_absolute():
        raise ValueError('Manifest paths must be relative: '+relative)
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: '+relative)
    return target

def restore(job, video_root, output_root):
    import cv2
    video = contained(video_root, job['video'])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('Cannot open video: '+str(video))
    restored = reused = 0
    try:
        for row in sorted(job['frames'], key=lambda x: x['index']):
            index = row['index']
            if not isinstance(index, int) or index < 0:
                raise ValueError('Invalid frame index')
            dest = contained(output_root, row['output'])
            expected = row.get('sha256')
            if dest.exists():
                if not expected or hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Existing file cannot be verified: '+str(dest))
                reused += 1
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode frame {index} in {video}')
            h, w = image.shape[:2]
            if row.get('width', w) != w or row.get('height', h) != h:
                raise RuntimeError('Frame dimension mismatch')
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            data = encoded.tobytes()
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f'JPEG hash mismatch: {video}, frame {index}; check source video and decoder version')
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.frame-', dir=dest.parent)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(data)
                # Never overwrite a concurrent writer's result.
                try:
                    os.link(tmp, dest)
                except FileExistsError:
                    if dest.read_bytes() != data:
                        raise RuntimeError('Concurrent output differs')
            finally:
                os.unlink(tmp)
            restored += 1
    finally:
        capture.release()
    return dict(video=job['video'], restored=restored, reused=reused)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=pathlib.Path, required=True)
    p.add_argument('--video-root', type=pathlib.Path, required=True)
    p.add_argument('--output-root', type=pathlib.Path, required=True)
    p.add_argument('--report', type=pathlib.Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    jobs = [json.loads(line) for line in args.manifest.open() if line.strip()]
    seen = set()
    for job in jobs:
        contained(args.video_root, job['video'])
        for row in job['frames']:
            dest = str(contained(args.output_root, row['output']))
            if dest in seen:
                raise ValueError('Duplicate output in manifest: '+dest)
            seen.add(dest)
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(restore, job, args.video_root, args.output_root): job['video'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                errors.append(dict(video=futures[future], error=repr(e)))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(status='fail' if errors else 'pass', results=results, errors=errors), indent=2))
    if errors:
        raise SystemExit(1)
if __name__ == '__main__':
    main()
