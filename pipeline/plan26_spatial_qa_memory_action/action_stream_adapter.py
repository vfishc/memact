"""Export internally consistent rendered Action prefixes into shared streaming traces."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from memory_runtime import MemoryManager, digest
from routed_facts import file_hash
from routed_stream import ACTION_SYSTEM_INSTRUCTION, check_public, validate_stream, evaluation_schedule
from qa_semantic_contract import visual_category

ACTION_MENU = [{"type": "action", "name": name, "arguments": {"angle_deg": 45} if name.startswith("rotate_") else {"distance_m": 0.5}}
               for name in ("rotate_left_45", "rotate_right_45", "move_forward_d", "move_left_d", "move_right_d")]


def read(path):
    return json.loads(Path(path).read_text())


class CachedActionExecutor:
    def __init__(self, entries):
        self.entries = entries
        self.executed = False
        self.attempts = 0

    def begin_question(self, index, payload):
        self.executed = False
        self.attempts = 0

    def __call__(self, command):
        if set(command) != {"type", "name", "arguments"} or command["type"] != "action":
            raise ValueError("invalid action command")
        if self.executed:
            return {"status": "action_unavailable", "reason": "no_second_step_cache"}
        result = next((row["result"] for row in self.entries if row["action"] == command), None)
        if result is None:
            from action_graph import ActionCoverageError
            raise ActionCoverageError("no_matching_cached_observation")
        self.executed = True
        self.attempts += 1
        return copy.deepcopy(result)


def adapt(bundle, episode_root, split):
    root = Path(episode_root)
    states = read(root / "passive/states.json")
    passive = read(root / "passive/results.json")
    chosen = next(row for row in read(root / "states.json") if row["state_id"] == bundle["state_id"])
    anchor = bundle["audit"]["frame_id"]
    prefix = sorted((row for row in states if row["frame_id"] <= anchor), key=lambda row: row["frame_id"])
    if not prefix or prefix[0]["timestamp_sec"] != 0 or prefix[-1]["frame_id"] != anchor:
        raise ValueError("incomplete action prefix")
    if any(abs(row["timestamp_sec"] - index) > 1e-6 for index, row in enumerate(prefix)):
        raise ValueError("expected complete1FPS prefix")
    if any(row["episode_id"] != bundle["audit"]["episode_id"] or row["source_render_contract"] != chosen["source_render_contract"] for row in prefix):
        raise ValueError("mixed episode or render environment")
    if "poses" in chosen and "poses" in prefix[-1] and chosen["poses"]["base"] != prefix[-1]["poses"]["base"]:
        raise ValueError("action anchor pose differs from passive prefix")
    by_state = {row["state_id"]: row["branches"]["base"] for row in passive["states"]}
    objects = chosen["objects"]
    events, media, pending = [], {}, []
    manager = MemoryManager()
    start = 0
    def emit(role, kind, payload):
        check_public(payload)
        events.append({"seq": len(events), "prefix_hash": digest(events[-1]) if events else None,
                       "role": role, "kind": kind, "loss": role == "assistant", "payload": payload})
    def bind(frame, path):
        path = Path(path)
        with Image.open(path) as image:
            image.load()
        media[frame] = {"path": str(path.resolve()), "sha256": file_hash(path)}
    action_result = next(row for row in read(next(root.glob("scenes/*/results.json")))["states"] if row["state_id"] == bundle["state_id"])
    original = bundle["events"]
    if bundle["template_id"] == "object_size_m":
        truth = bundle["audit"].get("size_truth", {})
        if truth.get("source") != "rendered_mesh_transformed_local_bbox_edges_v1" or abs(original[-1]["answer"] - truth["longest_side_m"]) > 0.000501:
            raise ValueError("size truth repair required")
    command = {"type": "action", "name": original[2]["name"], "arguments": original[2]["arguments"]}
    emit("system", "system_prompt", {"instruction": ACTION_SYSTEM_INSTRUCTION})
    for index, state in enumerate(prefix):
        if index and index % 5 == 0:
            emit("user", "write_prompt", {"mode": "[MEMORY_WRITE]", "window": [start, index], "frames": [row["frame"] for row in pending]})
            memory = {"type": "memory_write", "observations": pending, "relations": []}
            emit("assistant", "memory_write", memory)
            emit("tool", "memory_ack", manager.write(memory, start, index))
            pending = []
            start = index
        branch = by_state[state["state_id"]]
        frame = f"f{index:06d}"
        if index == len(prefix) - 1:
            branch = action_result["branches"].get("base", branch)
            bind(frame, original[0]["media_ref"])
            if media[frame]["sha256"] != file_hash(branch["rgb"]):
                raise ValueError("action base differs from recorded base")
        else:
            bind(frame, branch["rgb"])
        counts = Counter()
        for target, evidence in branch["objects"].items():
            category = visual_category(objects.get(target, {}).get("category", ""))
            if category and evidence["pixels"] > 0:
                counts[category] += 1
        manager.observe(frame, state["timestamp_sec"])
        emit("user", "observation", {"frame": frame, "timestamp": state["timestamp_sec"]})
        pending.append({"frame": frame, "objects": [{"category": name, "count": count} for name, count in sorted(counts.items())]})
    entries = []
    for name, branch in sorted(action_result["branches"].items()):
        if name == "base" or not branch.get("rendered") or not branch.get("collision", {}).get("valid"):
            continue
        action = next((row for row in ACTION_MENU if row["name"] == name), None)
        if action is None:
            continue
        frame = f"f{len(prefix) + len(entries):06d}"
        bind(frame, branch["rgb"])
        entries.append({"action": action, "result": {"status": "ok", "action": action, "frame": frame}})
    selected = next((row for row in entries if row["action"] == command), None)
    if selected is None or media[selected["result"]["frame"]]["sha256"] != file_hash(original[3]["media_ref"]):
        raise ValueError("missing matching action outcome")
    question_index = len(events)
    emit("user", "question", {"mode": "[QA]", "question": original[1]["question"], "options": None,
                               "actions": ACTION_MENU})
    emit("assistant", "action", command)
    emit("tool", "action_result", selected["result"])
    emit("user", "continue_prompt", {"mode": "[QA_CONTINUE]"})
    emit("assistant", "answer", {"type": "answer", "answer": original[-1]["answer"]})
    audit = [{"question_seq": question_index, "answer_seq": len(events) - 1, "template": bundle["template_id"], "sample": 0, "turn": 0}]
    schedule, answers = evaluation_schedule(events, audit)
    return {"events": events, "media": media, "schedule": schedule, "answers": answers, "cached_actions": entries,
            "check": validate_stream(events, media), "split": split, "source_bundle_sha256": digest(bundle),
            "source_training_eligible": bundle["training_eligible"], "formal_training_eligible": False,
            "pending_observation_count_at_terminal": len(pending)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = next(json.loads(line) for line in args.pairs.open() if json.loads(line)["sample_id"] == args.sample)
    data = adapt(bundle, args.episode_root, args.split)
    args.output.mkdir(parents=True, exist_ok=False)
    for key, value in data.items():
        (args.output / (key + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(data["check"]))
