"""Export existing Action bundles using physical-scene splits and resume receipts."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import time

from action_stream_adapter import adapt
from memory_runtime import digest
from routed_facts import file_hash


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def export(bundle, splits, output, receipts):
    sample = bundle["sample_id"]
    if Path(sample).name != sample or sample in (".", ".."):
        raise ValueError("invalid sample ID")
    receipt = receipts / (sample + ".json")
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result["bundle_hash"] != digest(bundle):
            raise ValueError("changed source bundle")
        if result["status"] == "pass" and any(file_hash(Path(result["destination"]) / name) != value for name, value in result["hashes"].items()):
            raise ValueError("resume output corrupted")
        return result
    try:
        image = Path(bundle["events"][0]["media_ref"])
        root = next(parent for parent in image.parents if (parent / "passive/states.json").exists())
        states = json.loads((root / "states.json").read_text())
        matched = [row for row in states if row["state_id"] == bundle["state_id"]]
        if len(matched) != 1:
            raise ValueError("missing or duplicate action state")
        scene = matched[0]["scene_id"]
        split = splits[("sim_front3d", scene)]
        if not all(row["scene_id"] == scene for row in states):
            raise ValueError("mixed scene states")
        data = adapt(bundle, root, split)
        destination = output / sample / f"attempt_{time.time_ns()}"
        destination.mkdir(parents=True)
        for name, value in data.items():
            save(destination / (name + ".json"), value)
        result = {"status": "pass", "episode": sample, "source_episode": bundle["audit"]["episode_id"],
                  "source": "sim_front3d", "family": "action", "scene_id": scene, "split": split,
                  "template": bundle["template_id"], "destination": str(destination), "bundle_hash": digest(bundle),
                  "hashes": {path.name: file_hash(path) for path in destination.glob("*.json")},
                  "summary": data["check"], "formal_training_eligible": False}
    except Exception as error:
        result = {"status": "failed", "episode": sample, "bundle_hash": digest(bundle), "error": repr(error)}
    save(receipt, result)
    return result


def main(args):
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configuration = {"pairs": file_hash(args.pairs), "splits": file_hash(args.splits), "output": str(args.output.resolve()),
                         "limit": args.limit, "code": {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py")}}
        receipt = args.state / "configuration.json"
        if receipt.exists():
            if json.loads(receipt.read_text()) != configuration:
                raise ValueError("resume configuration mismatch")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            save(receipt, configuration)
        splits = {(row["source"], row["base_scene_id"]): row["split"] for row in (json.loads(line) for line in args.splits.read_text().splitlines())}
        bundles = {}
        for line in args.pairs.read_text().splitlines():
            bundle = json.loads(line)
            if bundle["sample_id"] in bundles and bundles[bundle["sample_id"]] != bundle:
                raise ValueError("conflicting duplicate sample")
            bundles[bundle["sample_id"]] = bundle
        selected = list(bundles.values())[:args.limit or None]
        receipts = args.state / "results"
        receipts.mkdir(exist_ok=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(export, bundle, splits, args.output, receipts) for bundle in selected]
            for future in as_completed(futures):
                results.append(future.result())
                counts = Counter(row["status"] for row in results)
                save(args.state / "status.json", {"status": "running", "pid": os.getpid(), "total": len(selected), **counts, "updated_unix": time.time()})
        with (args.output / "dataset.jsonl").open("w") as handle:
            for row in sorted(results, key=lambda row: row["episode"]):
                if row["status"] == "pass":
                    handle.write(json.dumps(row) + "\n")
        save(args.state / "status.json", {"status": "complete_with_failures" if any(row["status"] != "pass" for row in results) else "complete",
             "total": len(selected), **Counter(row["status"] for row in results),
             "splits": dict(Counter(row["split"] for row in results if row["status"] == "pass"))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "splits", "output", "state"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
