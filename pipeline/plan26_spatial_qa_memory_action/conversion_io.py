"""Resume isolated official ARKit scene conversions with bounded CPU concurrency."""

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from arkit_rebuild import sha


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verified_rows(destination, scene):
    report = json.loads((destination / "generation_report.json").read_text())
    if report["failed_scenes"] or not report["episodes"]:
        raise ValueError("scene failed or produced no episodes")
    rows = report["episodes"]
    for row in rows:
        receipt_path = Path(row["check"])
        receipt = json.loads(receipt_path.read_text())
        if row["scene_id"] != scene or receipt["source_scene"] != scene:
            raise ValueError("scene identity mismatch")
        if receipt["fps"] != 24 or receipt["size"] != [640, 480] or not 0 < receipt["frames"] <= 2160 or not receipt["official_orientation"]:
            raise ValueError("output contract mismatch")
        for name, expected in receipt["output_sha256"].items():
            if sha(receipt_path.parent / name) != expected:
                raise ValueError("output hash mismatch")
    return rows


def convert(args, scene):
    scene_root = args.output / "scenes" / scene
    scene_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for previous in sorted(scene_root.glob("attempt_*")):
        try:
            return {"scene": scene, "status": "complete", "reused": True,
                    "rows": verified_rows(previous, scene), "attempt": str(previous)}
        except (OSError, ValueError, KeyError):
            pass
    started = time.monotonic()
    for retry in range(args.retries + 1):
        destination = scene_root / f"attempt_{time.time_ns()}"
        command = [sys.executable, str(args.generator), "--manifest", str(args.manifest),
                   "--assets", str(args.assets), "--output", str(destination), "--scene", scene]
        log = args.state / "logs" / f"{scene}_{destination.name}.log"
        try:
            with log.open("w") as handle:
                subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True,
                               timeout=args.timeout, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            rows = verified_rows(destination, scene)
            return {"scene": scene, "status": "complete", "rows": rows, "attempt": str(destination),
                    "seconds": time.monotonic() - started, "previous_failures": failures}
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failures.append({"attempt": str(destination), "log": str(log), "error": repr(error)})
    return {"scene": scene, "status": "failed", "failures": failures, "seconds": time.monotonic() - started}


def run(args):
    args.output = args.output.resolve()
    args.state = args.state.resolve()
    args.manifest = args.manifest.resolve()
    args.assets = args.assets.resolve()
    args.generator = Path(__file__).with_name("arkit_official_generate.py").resolve()
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = {"manifest_sha256": sha(args.manifest), "assets": str(args.assets), "output": str(args.output),
                         "code_hashes": {path.name: sha(path) for path in Path(__file__).parent.glob("*.py")
                                         if path.name in ("conversion_io.py", "arkit_official_generate.py", "arkit_rebuild.py",
                                                          "arkit_annotations.py", "arkit_speed_probe.py", "projection_audit.py")},
                         "contract": "arkit_official_native_v2", "fps": 24, "size": [640, 480], "max_seconds": 90}
        configuration_path = args.state / "configuration.json"
        if configuration_path.exists():
            if json.loads(configuration_path.read_text()) != configuration:
                raise ValueError("resume configuration/code mismatch; use a new run")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            save(configuration_path, configuration)
        (args.state / "logs").mkdir(exist_ok=True)
        (args.state / "results").mkdir(exist_ok=True)
        scenes = sorted({item["scene"] for item in json.loads(args.manifest.read_text())["episodes"]})
        selected = scenes[:args.limit] if args.limit else scenes
        status = {"status": "running", "pid": os.getpid(), "started_unix": time.time(), "workers": args.workers,
                  "total_scenes": len(scenes), "selected_scenes": len(selected), "completed": 0, "failed": 0, "episodes": 0}
        save(args.state / "status.json", status)
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(convert, args, scene): scene for scene in selected}
            for job in concurrent.futures.as_completed(jobs):
                try:
                    result = job.result()
                except Exception as error:
                    result = {"scene": jobs[job], "status": "failed", "error": repr(error)}
                save(args.state / "results" / (result["scene"] + ".json"), result)
                if result["status"] == "complete":
                    status["completed"] += 1
                    rows.extend(result["rows"])
                else:
                    status["failed"] += 1
                status["episodes"] = len(rows)
                status["elapsed_seconds"] = time.time() - status["started_unix"]
                save(args.state / "status.json", status)
                print(json.dumps(status), flush=True)
        rows.sort(key=lambda row: row["episode_id"])
        temporary = args.output / "registry.jsonl.tmp"
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        temporary.replace(args.output / "registry.jsonl")
        status["status"] = "complete_with_failures" if status["failed"] else "complete"
        status["full_manifest_complete"] = not args.limit and not status["failed"]
        save(args.state / "status.json", status)
        return int(status["failed"] > 0)


