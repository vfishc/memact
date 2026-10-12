"""Shared, causal RGB materialization for training and native evaluation."""
import hashlib
import json
from pathlib import Path

from evaluate_routed_memory_qa import FrameResolver
from routed_stream import check_public


def _load_episode(root):
    root = Path(root)
    if (root / 'bundle.json').exists():
        return json.loads((root / 'bundle.json').read_text())
    action = (root / "events.json").exists()
    def read(name):
        return json.loads((root / name).read_text())
    def lines(name):
        return [json.loads(line) for line in (root / name).read_text().splitlines()]
    return {
        "events": read("events.json") if action else lines("stream.teacher.jsonl"),
        "schedule": (read("schedule.json") if (root / "schedule.json").exists() else []) if action else lines("evaluation_schedule.jsonl"),
        "answers": (read("answers.json") if (root / "answers.json").exists() else []) if action else lines("evaluation_answers.private.jsonl"),
        "media": read("media.json") if action else read("private_media.json"),
        "actions": read("cached_actions.json") if action and (root / "cached_actions.json").exists() else [],
    }


def load_episode(root):
    """Resolve portable media paths against the episode, never the caller's cwd."""
    data = _load_episode(root)
    graph_path = Path(root) / "action_graph.json"
    if graph_path.exists():
        data["action_graph"] = json.loads(graph_path.read_text())
    root = Path(root).resolve()
    for row in data["media"].values():
        path = Path(row["path"])
        if not path.is_absolute():
            path = (root / path).resolve()
            if not path.is_relative_to(root):
                raise ValueError("media path escapes episode directory")
            row["path"] = str(path)
    return data


def materialize(context, media, max_pixels=200704):
    """Private paths never reach textual prompts; receipts stay outside messages."""
    messages, receipts = [], []
    for index, message in enumerate(FrameResolver(media).materialize(context)):
        check_public(message["content"])
        content = [{"type": "text", "text": json.dumps(message["content"], ensure_ascii=False)}]
        for row in message["images"]:
            rgb = row["image"]
            content.append({"type": "image", "image": rgb, "max_pixels": max_pixels})
            receipts.append({"message": index, "frame": row["frame"], "size": list(rgb.size),
                             "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        messages.append({"role": message["role"], "content": content})
    return messages, receipts


def event_context(events):
    return [{"role": row["role"], "content": row["payload"]} for row in events]


def check_visual_batch(batch, receipts, processor):
    grid = batch.get("image_grid_thw")
    if not receipts:
        token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        if grid is not None or "pixel_values" in batch or bool((batch["input_ids"] == token).any()):
            raise ValueError("unexpected visual inputs in a text-only checkpoint")
        return {"decoded_images": 0, "visual_tokens": 0, "pixel_values_shape": [], "media_receipts": []}
    if grid is None or len(grid) != len(receipts) or "pixel_values" not in batch:
        raise ValueError("decoded RGB / processor grid mismatch")
    expected = int(grid.prod(dim=-1).sum().item()) // processor.image_processor.merge_size**2
    token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    actual = int((batch["input_ids"] == token).sum().item())
    if expected != actual:
        raise ValueError("image tokens / vision grid mismatch")
    return {"decoded_images": len(receipts), "visual_tokens": actual,
            "pixel_values_shape": list(batch["pixel_values"].shape),
            "media_receipts": receipts}
