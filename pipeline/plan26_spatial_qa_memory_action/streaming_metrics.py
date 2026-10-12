"""Score complete streaming evaluation manifests without invoking a model."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from evaluate_routed_memory_qa import score_answers, score_memory


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def relation_scores(truth, prediction):
    def relations(writes):
        return {(row["frame"], row["subject"]["category"], row["subject"]["region"], row["predicate"],
                 row["object"]["category"], row["object"]["region"])
                for write in writes for row in write.get("relations", [])}
    expected, actual = relations(truth), relations(prediction)
    common = len(expected & actual)
    return {"precision": ratio(common, len(actual)), "recall": ratio(common, len(expected)),
            "f1": ratio(2 * common, len(expected) + len(actual)), "expected": len(expected), "predicted": len(actual)}


def trace_metrics(context):
    counts = Counter()
    per_question = []
    active = None
    for message in context:
        role, payload = message["role"], message["content"]
        if role == "user" and payload.get("mode") == "[QA]":
            counts["questions"] += 1
            active = 0
            per_question.append(0)
        if role == "assistant":
            kind = payload.get("type")
            if kind in ("use_memory", "action", "answer", "memory_write"):
                counts[kind] += 1
        if role == "tool":
            if payload.get("status") in ("invalid_memory", "invalid_query"):
                counts[payload["status"]] += 1
            if "frames" in payload:
                size = len(payload["frames"])
                counts["retrieved_images"] += size
                counts["per_call_cap_violations"] += int(size > 2)
                if active is not None:
                    active += size
                    per_question[-1] = active
                elif size:
                    counts["retrieval_outside_qa"] += 1
            if "action" in payload and payload.get("status") == "ok":
                counts["successful_actions"] += 1
        if role == "assistant" and payload.get("type") == "answer":
            active = None
    return {"counts": dict(counts), "max_retrieved_images_per_question": max(per_question, default=0),
            "per_question_cap_violations": sum(value > 4 for value in per_question),
            "action_execution_rate": ratio(counts["successful_actions"], counts["action"]),
            "mean_memory_calls_per_question": ratio(counts["use_memory"], counts["questions"])}


def qa_totals(expected, predictions):
    detailed = score_answers(expected, predictions)
    actual = {row["schedule_index"]: row.get("answer") for row in predictions}
    correct = numeric = valid = 0
    mra = 0.0
    choice_count = choice_correct = invalid_format = 0
    from qa_eight_contract import valid_answer
    for row in expected:
        target = row["answer"]
        answer = actual.get(row["schedule_index"])
        if row.get('answer_type'):
            if row['schedule_index'] in actual and not valid_answer(answer,row):invalid_format+=1
            if row['answer_type']=='choice':
                choice_count+=1;choice_correct+=int(answer==target)
            if not valid_answer(answer,row):answer=None
        if type(target) in (int, float):
            if not math.isfinite(target):
                raise ValueError("nonfinite ground truth")
            numeric += 1
            if type(answer) in (int, float) and math.isfinite(answer):
                valid += 1
                correct += int(answer == target)
                if target == 0:
                    mra += float(answer == 0)
                else:
                    error = abs(answer - target) / abs(target)
                    mra += sum(error < 0.5 - index * 0.05 for index in range(10)) / 10
        else:
            correct += int(isinstance(answer, str) and answer == target)
    return {"count": len(expected), "correct": correct, "exact_accuracy": ratio(correct, len(expected)),
            "choice_count":choice_count,"choice_accuracy":ratio(choice_correct,choice_count),
            "invalid_format_count":invalid_format,"invalid_format_rate":ratio(invalid_format,len(expected)),
            "answer_coverage": ratio(len(actual), len(expected)), "numeric_count": numeric,
            "numeric_valid": valid, "numeric_mra": ratio(mra, numeric), **detailed}


def evaluate(episodes):
    identities = set()
    expected_all, predictions_all = [], []
    groups = defaultdict(lambda: ([], []))
    reports = []
    for episode in episodes:
        identity = episode["episode_id"]
        if identity in identities:
            raise ValueError("duplicate episode")
        identities.add(identity)
        expected = episode["expected"]
        if len({row["schedule_index"] for row in expected}) != len(expected):
            raise ValueError("duplicate expected question")
        predictions = episode.get("predictions", [])
        report = {"episode_id": identity, "qa": qa_totals(expected, predictions),
                  "trace": trace_metrics(episode["context"]) if "context" in episode else None}
        if "teacher_writes" in episode:
            teacher, predicted = episode["teacher_writes"], episode.get("predicted_writes", [])
            memory = score_memory(teacher, predicted)
            precision, recall = memory["category_precision"], memory["category_recall"]
            memory["category_f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            memory["relations"] = relation_scores(teacher, predicted)
            report["memory"] = memory
        reports.append(report)
        renamed = [{**row, "schedule_index": (identity, row["schedule_index"]), "sample": (identity, row["sample"])} for row in expected]
        renamed_predictions = [{**row, "schedule_index": (identity, row["schedule_index"])} for row in predictions]
        expected_all.extend(renamed)
        predictions_all.extend(renamed_predictions)
        for field in ("source", "family"):
            group = groups[field + ":" + episode.get(field, "unspecified")]
            group[0].extend(renamed)
            group[1].extend(renamed_predictions)
    return {"metric_version": "p26_streaming_metrics_v1", "numeric_policy": "shared_vsibench_mra", "overall": qa_totals(expected_all, predictions_all),
            "groups": {name: qa_totals(*values) for name, values in groups.items()}, "episodes": reports,
            "route_accuracy": None, "latency": None, "counterfactual_action_gain": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = []
    for line in args.manifest.read_text().splitlines():
        entry = json.loads(line)
        for key in ("expected", "predictions", "teacher_writes", "predicted_writes", "context"):
            if isinstance(entry.get(key), str):
                path = Path(entry[key])
                if not path.is_absolute():
                    path = args.manifest.parent / path
                entry[key] = [json.loads(row) for row in path.read_text().splitlines()] if path.suffix == ".jsonl" else json.loads(path.read_text())
        episodes.append(entry)
    with args.output.open("x") as handle:
        json.dump(evaluate(episodes), handle, indent=2, allow_nan=False)
