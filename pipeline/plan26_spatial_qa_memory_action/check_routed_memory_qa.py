"""Reopen a generated episode and verify replay, mathematical labels and media."""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from routed_facts import EpisodeFacts, file_hash
from routed_qa import QAGenerator
from routed_stream import evidence_satisfied, validate_stream
from qa_semantic_contract import clean_category_options, visual_category


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def check_episode(root):
    configuration = json.loads((root / "check.json").read_text())
    record = json.loads((root / "private_facts.json").read_text())
    media = json.loads((root / "private_media.json").read_text())
    events = read_rows(root / "stream.teacher.jsonl")
    facts = EpisodeFacts(record, configuration["sample_fps"], configuration["duration_seconds"])
    samples = read_rows(root / "qa_candidates.private.jsonl")
    audit = read_rows(root / "audit.private.jsonl")
    generator = QAGenerator(facts)
    result = validate_stream(events, media, configuration["page_size"], configuration["frame_budget"])
    for sample in samples:
        regenerated = generator.make(sample["template"], sample["turns"][0]["index"])
        if regenerated != sample:
            raise ValueError("QA disagrees with sampled facts")
        for turn in sample["turns"]:
            options = turn["options"]
            if options is not None:
                _, cleaned, _ = clean_category_options(turn["answer"], options)
                if not 4 <= len(options) <= 5 or len(cleaned) != len(options) or set(cleaned) != set(options) or any(visual_category(option) != option for option in options):
                    raise ValueError("category option contract violation")
                first = {category: position for position, category in facts.first_categories(turn["index"])}
                if sample["template"] in ("first_seen_object_category", "seen_category_order", "first_seen_category_then_timestamp"):
                    times = {option: first[option] for option in options if option in first}
                    if turn["answer"] not in times or [option for option, position in times.items() if position == min(times.values())] != [turn["answer"]]:
                        raise ValueError("ambiguous first-seen options")
    turn_counts = Counter()
    for row in audit:
        sample = samples[row["sample"]]
        turn = sample["turns"][row["turn"]]
        question = events[row["question_seq"]]
        answer = events[row["answer_seq"]]
        if question["payload"]["question"] != turn["question"] or answer["payload"]["answer"] != turn["answer"]:
            raise ValueError("audit/stream QA mismatch")
        previous_observation = next(event for event in reversed(events[:row["question_seq"]]) if event["kind"] == "observation")
        if previous_observation["payload"]["frame"] != facts.frames[turn["index"]]["frame"]:
            raise ValueError("question anchored at wrong time")
        returned = {item["frame"] for event in events[row["question_seq"]:row["answer_seq"]]
                    if event["kind"] == "memory_result" for item in event["payload"]["frames"]}
        if not evidence_satisfied(facts, sample, turn, returned):
            raise ValueError("incomplete historical image evidence")
        turn_counts[row["sample"]] += 1
    if any(count != len(samples[index]["turns"]) for index, count in turn_counts.items()):
        raise ValueError("incomplete chain")
    for entry in media.values():
        path = Path(entry["path"])
        if file_hash(path) != entry["sha256"]:
            raise ValueError("media hash mismatch")
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                raise ValueError("invalid image")
    result.update({"status": "fixture_pass", "decoded_images": len(media), "recomputed_candidates": len(samples),
                   "formal_training_eligible": False, "independent_semantic_audit": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_episode(args.episode)))


if __name__ == "__main__":
    main()
