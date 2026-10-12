"""Versioned process-parallel Current/Memory generation from frozen input rows."""

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import time

from arkit_rebuild import sha
from conversion_io import save
from check_routed_memory_qa import check_episode
from generate_routed_memory_qa import generate_episode
from prepare_scene_split_manifest import base_scene
from routed_facts import rebuild_record, official_arkit_record
from routed_intrinsics import attach_intrinsics


def generate_one(job):
    item, root, settings = job
    started = time.monotonic()
    episode = item["episode_id"]
    try:
        if item.get("official_episode"):
            record = official_arkit_record(item["official_episode"])
        else:
            record = rebuild_record(item["record"])
            record = attach_intrinsics(record, settings)
        if record["episode_id"] != episode:
            raise ValueError("episode identity mismatch")
        record["split"] = item["split"]
        visibility = json.loads(Path(record["paths"]["visibility"]).read_text())
        rows = visibility if isinstance(visibility, list) else visibility.get("frames", [])
        if any(abs(row.get("visibility_source_frame_id", row["frame_id"]) - row["frame_id"]) > record["fps"] for row in rows):
            raise ValueError("visibility extrapolation exceeds observation interval")
        destination = Path(root) / "episodes" / episode / f"attempt_{time.time_ns()}"
        summary = generate_episode(record, destination, max_seconds=90, deduplicate=True)
        checked = check_episode(destination)
        return {"episode": episode, "status": "pass", "source": record["source"], "split": item["split"],
                "destination": str(destination), "summary": summary, "checks": checked,
                "seconds": time.monotonic() - started}
    except Exception as error:
        return {"episode": episode, "status": "failed", "error": repr(error), "seconds": time.monotonic() - started}


def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "batch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        items = [json.loads(line) for line in args.manifest.open()]
        mapping = {(row["source"], row["base_scene_id"]): row["split"]
                   for row in (json.loads(line) for line in args.splits.open())}
        seen = set()
        for item in items:
            episode = item["episode_id"]
            if episode in seen or Path(episode).name != episode or episode in (".", ".."):
                raise ValueError("duplicate or invalid episode ID")
            seen.add(episode)
            key = item["source"], base_scene(item["source"], item["scene_id"])
            if key not in mapping:
                raise ValueError("missing base-scene split")
            item["split"] = mapping[key]
        configuration = {"manifest_sha256": sha(args.manifest), "splits_sha256": sha(args.splits),
                         "sim_settings_sha256": sha(args.sim_render_settings), "deduplicate": True,
                         "code_hashes": {path.name: sha(path) for path in Path(__file__).parent.glob("*.py")}}
        receipt = args.output / "configuration.json"
        if receipt.exists() and json.loads(receipt.read_text()) != configuration:
            raise ValueError("resume code/input mismatch; use new output")
        save(receipt, configuration)
        results_dir = args.output / "results"
        results_dir.mkdir(exist_ok=True)
        results, pending = [], []
        for item in items:
            previous = results_dir / (item["episode_id"] + ".json")
            if previous.exists():
                result = json.loads(previous.read_text())
                if result["status"] == "pass":
                    check_episode(Path(result["destination"]))
                    results.append(result)
                    continue
            pending.append((item, str(args.output), str(args.sim_render_settings)))
        started = time.monotonic()
        def status():
            return {"pid": os.getpid(), "total": len(items), "completed": sum(row["status"] == "pass" for row in results),
                    "failed": sum(row["status"] == "failed" for row in results), "workers": args.workers,
                    "elapsed_seconds": time.monotonic() - started, "training_tested": False}
        save(args.output / "status.json", {**status(), "status": "running"})
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(generate_one, job): job[0]["episode_id"] for job in pending}
            for future in concurrent.futures.as_completed(jobs):
                try:
                    result = future.result()
                except Exception as error:
                    result = {"episode": jobs[future], "status": "failed", "error": repr(error)}
                save(results_dir / (result["episode"] + ".json"), result)
                results.append(result)
                save(args.output / "status.json", {**status(), "status": "running"})
        with (args.output / "dataset.jsonl").open("w") as handle:
            for result in sorted(results, key=lambda row: row["episode"]):
                if result["status"] == "pass":
                    handle.write(json.dumps(result) + "\n")
        final = {**status(), "status": "complete_with_failures" if status()["failed"] else "complete"}
        save(args.output / "status.json", final)
        print(json.dumps(final))
        return int(final["failed"] > 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "splits", "output", "sim-render-settings"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    raise SystemExit(run(args))
