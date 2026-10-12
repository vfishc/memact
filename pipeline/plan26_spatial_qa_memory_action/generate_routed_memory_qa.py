"""Bounded existing-video-only P26 memory/QA generation; never invokes a renderer."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

from memory_runtime import digest
from routed_facts import EpisodeFacts, file_hash, rebuild_record, official_arkit_record
from routed_qa import CURRENT, MEMORY, QAGenerator
from routed_stream import SCHEMA, build_stream, evaluation_schedule, validate_stream
from qa_semantic_contract import VISUAL_CATEGORY_CONTRACT_HASH
from qa_difficulty_contract import QA_DIFFICULTY_CONTRACT_HASH
from routed_intrinsics import attach_intrinsics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def extract_frames(facts, output):
    output.mkdir()
    capture = cv2.VideoCapture(facts.record["paths"]["video"])
    if not capture.isOpened():
        raise ValueError("cannot open existing video")
    measured_fps = capture.get(cv2.CAP_PROP_FPS)
    if abs(measured_fps - facts.fps) > 0.05:
        capture.release()
        raise ValueError("video/cache FPS mismatch")
    media = {}
    try:
        for row in facts.frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, row["source_frame"])
            success, image = capture.read()
            if not success:
                raise ValueError("video frame decode failed")
            path = output / (row["frame"] + ".jpg")
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise ValueError("frame write failed")
            media[row["frame"]] = {"path": str(path.resolve()), "source_frame": row["source_frame"],
                                    "timestamp": row["timestamp"], "sha256": file_hash(path)}
    finally:
        capture.release()
    return media


def generate_episode(record, output, sample_fps=1.0, max_seconds=45, page_size=8, frame_budget=2, deduplicate=False):
    output.mkdir(parents=True, exist_ok=False)
    facts = EpisodeFacts(record, sample_fps, max_seconds)
    media = extract_frames(facts, output / "frames")
    generator = QAGenerator(facts)
    samples = generator.generate()
    original_samples = len(samples)
    if deduplicate:
        unique = {}
        for sample in samples:
            unique.setdefault(sample["semantic_key"], sample)
        samples = list(unique.values())
    events, audit, rejected = build_stream(facts, samples, page_size, frame_budget)
    check = validate_stream(events, media, page_size, frame_budget)
    schedule, answers = evaluation_schedule(events, audit)
    accepted_samples = {row["sample"] for row in audit}
    counts = Counter(samples[index]["template"] for index in accepted_samples)
    write_json(output / "private_facts.json", record)
    write_json(output / "private_media.json", media)
    write_jsonl(output / "qa_candidates.private.jsonl", samples)
    write_jsonl(output / "audit.private.jsonl", audit)
    write_jsonl(output / "stream.teacher.jsonl", events)
    write_jsonl(output / "evaluation_schedule.jsonl", schedule)
    write_jsonl(output / "evaluation_answers.private.jsonl", answers)
    write_jsonl(output / "memory_targets.jsonl", [event["payload"] for event in events if event["kind"] == "memory_write"])
    write_json(output / "rejections.json", {"generation": [{"template": key[0], "reason": key[1], "count": count}
                                                          for key, count in sorted(generator.rejections.items())], "routing": rejected})
    check.update({"status": "fixture_pass", "schema": SCHEMA, "sample_fps": sample_fps, "duration_seconds": facts.end,
                  "page_size": page_size, "frame_budget": frame_budget,
                  "template_counts": {template: counts[template] for template in CURRENT + MEMORY},
                  "accepted_samples": len(accepted_samples), "candidate_samples": len(samples),
                  "semantic_duplicates_removed": original_samples - len(samples),
                  "semantic_unique_samples": len({samples[index]["semantic_key"] for index in accepted_samples}),
                  "intrinsics_provenance": record.get("intrinsics_provenance"),
                  "input_hash": digest(record), "stream_hash": file_hash(output / "stream.teacher.jsonl"),
                  "category_contract_hash": VISUAL_CATEGORY_CONTRACT_HASH,
                  "difficulty_contract_hash": QA_DIFFICULTY_CONTRACT_HASH,
                  "model_inference_tested": False, "tokenizer_budget_pass": False,
                  "event_json_characters": sum(len(json.dumps(event["payload"])) for event in events),
                  "relations": sum(len(event["payload"]["relations"]) for event in events if event["kind"] == "memory_write"),
                  "geometry_status": "rebuilt_existing_annotation_not_new_visual_acceptance",
                  "formal_training_eligible": False})
    write_json(output / "check.json", check)
    write_json(output / "fixture_manifest.json", {"schema": SCHEMA, "episode_id": record["episode_id"],
               "scene_id": record["scene_id"], "source": record["source"], "split": record["split"],
               "stream": "stream.teacher.jsonl", "schedule": "evaluation_schedule.jsonl", "media": "private_media.json",
               "input_hash": digest(record), "formal_training_eligible": False})
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cache-jsonl", type=Path)
    inputs.add_argument("--official-arkit-episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", default="sim_front3d")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=45)
    parser.add_argument("--sample-fps", type=float, default=1)
    parser.add_argument("--sim-visibility", type=Path)
    parser.add_argument("--sim-render-settings", type=Path)
    parser.add_argument("--scannet-raw-root", type=Path)
    parser.add_argument("--scannetpp-raw-root", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_episodes <= 8 or not 0 < args.max_seconds <= 90:
        parser.error("This entry is bounded to 1..8 episodes and <=90 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.official_arkit_episode:
        record = official_arkit_record(args.official_arkit_episode)
        check = generate_episode(record, args.output / record["episode_id"], args.sample_fps, args.max_seconds)
        write_json(args.output / "summary.json", {"episodes": [check], "status": "fixture_pass", "renders": 0,
                   "command": sys.argv, "source_receipt_hash": file_hash(args.official_arkit_episode / ".complete.json")})
        print(json.dumps(check))
        return 0
    summaries = []
    failures = []
    scenes = {}
    attempted = 0
    for line in args.cache_jsonl.open():
        record = json.loads(line)
        if record["source"] not in args.sources.split(","):
            continue
        key = (record["source"], record["scene_id"])
        if key in scenes and scenes[key] != record["split"]:
            raise ValueError("scene split conflict")
        scenes[key] = record["split"]
        if attempted >= args.max_episodes:
            break
        attempted += 1
        try:
            record = rebuild_record(record, args.sim_visibility if record["source"] == "sim_front3d" else None)
            raw_root = args.scannet_raw_root if record["source"] == "real_scannet" else args.scannetpp_raw_root
            record = attach_intrinsics(record, args.sim_render_settings, raw_root)
            destination = args.output / record["episode_id"]
            summaries.append({"episode": record["episode_id"], **generate_episode(record, destination, args.sample_fps, args.max_seconds)})
        except Exception as error:
            failures.append({"episode": record["episode_id"], "error": str(error), "error_type": type(error).__name__})
    source_hashes = {path.name: file_hash(path) for path in Path(__file__).parent.glob("*.py") if path.name in
                     ("memory_runtime.py", "routed_facts.py", "routed_qa.py", "routed_stream.py", Path(__file__).name,
                      "check_routed_memory_qa.py", "evaluate_routed_memory_qa.py", "routed_intrinsics.py")}
    write_json(args.output / "summary.json", {"episodes": summaries, "failures": failures, "code_hashes": source_hashes,
               "cache_hash": file_hash(args.cache_jsonl), "command": sys.argv, "renders": 0,
               "status": "fixture_pass" if summaries and not failures else "failed"})
    print(json.dumps({"episodes": len(summaries), "failures": failures, "samples": sum(row["accepted_samples"] for row in summaries)}))
    return 0 if summaries and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
