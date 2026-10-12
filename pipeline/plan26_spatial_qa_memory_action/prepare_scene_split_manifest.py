"""Prepare conservative base-scene splits without overwriting historical manifests."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from arkit_rebuild import sha


def base_scene(source, scene):
    return re.sub(r"_(?:part|chunk)_?\d+$", "", scene) if source.startswith("real_") else scene


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for line in source.open():
        row = json.loads(line)
        if row["s3_split"] not in ("train", "val", "test"):
            raise ValueError("unknown source split")
        groups[row["source"], base_scene(row["source"], row["scene_id"])].append(row)
    conflicts = []
    counts = Counter()
    with (output / "scenes.jsonl").open("w") as handle:
        for (domain_source, scene), rows in sorted(groups.items()):
            splits = sorted({row["s3_split"] for row in rows})
            split = max(splits, key={"train": 0, "val": 1, "test": 2}.get)
            entry = {"source": domain_source, "base_scene_id": scene, "split": split,
                     "original_splits": splits, "original_scene_ids": sorted({row["scene_id"] for row in rows})}
            if len(splits) > 1:
                conflicts.append(entry)
            counts[domain_source + "/" + split] += 1
            handle.write(json.dumps(entry) + "\n")
    report = {"status": "prepared", "input_sha256": sha(source), "scenes": len(groups),
              "counts": dict(counts), "conflicts": conflicts,
              "policy": "test > val > train; never move historical held-out scene into training",
              "unmapped_scene_policy": "exclude pending assignment; no implicit train default",
              "formal_evaluation_equivalence": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"scenes": len(groups), "conflicts": len(conflicts), "counts": dict(counts)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
