"""Offline scoring and media binding; no GT is passed to a model callback."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


class FrameResolver:
    def __init__(self, private_media):
        self.media = private_media

    def materialize(self, context):
        result = []
        observed = set()
        pending_action = False
        for message in context:
            content = message["content"]
            handles = []
            if message["role"] == "assistant" and content.get("type") == "action":
                pending_action = True
            if message["role"] == "user" and "frame" in content and "timestamp" in content:
                observed.add(content["frame"])
                handles = [content["frame"]]
            elif message["role"] == "tool" and "frames" in content:
                handles = [row["frame"] for row in content["frames"]]
            elif message["role"] == "tool" and content.get("status") == "ok" and "action" in content and "frame" in content:
                observed.add(content["frame"])
                handles = [content["frame"]]
                pending_action = False
            elif message["role"] == "tool" and content.get("type") == "action_result":
                if not pending_action:
                    raise ValueError("action result without a preceding action")
                pending_action = False
                if content.get("status", "ok") == "ok":
                    if "frame" not in content:
                        raise ValueError("successful action result is missing its frame")
                    observed.add(content["frame"])
                    handles = [content["frame"]]
            images = []
            for frame in handles:
                if frame not in observed or frame not in self.media:
                    raise ValueError("unobserved media request")
                with Image.open(self.media[frame]["path"]) as image:
                    images.append({"frame": frame, "image": image.convert("RGB").copy()})
            result.append({**message, "images": images})
        return result


def score_answers(expected, predictions):
    actual = {}
    for row in predictions:
        if row["schedule_index"] in actual:
            raise ValueError("duplicate prediction")
        actual[row["schedule_index"]] = row.get("answer")
    if actual.keys() - {row["schedule_index"] for row in expected}:
        raise ValueError("prediction does not belong to schedule")
    groups = defaultdict(list)
    chains = defaultdict(list)
    for row in expected:
        target = row["answer"]
        answer = actual.get(row["schedule_index"])
        numeric = type(target) in (int, float)
        valid_numeric = numeric and type(answer) in (int, float) and math.isfinite(answer)
        if row.get('unit')=='count' and type(answer) is not int:valid_numeric=False
        exact = answer == target and (valid_numeric if numeric else isinstance(answer, str))
        item = {"exact": exact, "missing": row["schedule_index"] not in actual,
                "invalid_numeric": numeric and not valid_numeric}
        if valid_numeric:
            item["absolute_error"] = abs(answer - target)
            if target != 0:
                item["relative_error"] = abs(answer - target) / abs(target)
        groups[row["template"]].append(item)
        chains[row["sample"]].append(exact)
    report = {}
    for template, values in groups.items():
        row = {"count": len(values), "exact_accuracy": sum(value["exact"] for value in values) / len(values),
               "missing": sum(value["missing"] for value in values), "invalid_numeric": sum(value["invalid_numeric"] for value in values)}
        for metric in ("absolute_error", "relative_error"):
            errors = [value[metric] for value in values if metric in value]
            if errors:
                row["mean_" + metric] = sum(errors) / len(errors)
                row[metric + "_denominator"] = len(errors)
        report[template] = row
    chains = [values for values in chains.values() if len(values) == 2]
    return {"templates": report, "two_turn_chains": len(chains),
            "two_turn_exact_success": sum(all(values) for values in chains) / len(chains) if chains else None,
            "numeric_tolerance_accuracy": None, "numeric_tolerances_status": "not_frozen"}


def score_memory(teacher_writes, predicted_writes):
    def flatten(writes):
        frames = {}
        for write in writes:
            for row in write["observations"]:
                if row["frame"] in frames:
                    raise ValueError("duplicate memory frame")
                frames[row["frame"]] = {item["category"]: item["count"] for item in row["objects"]}
        return frames
    expected, actual = flatten(teacher_writes), flatten(predicted_writes)
    true_positive = false_positive = false_negative = count_error = pairs = 0
    for frame in expected.keys() | actual.keys():
        truth, prediction = expected.get(frame, {}), actual.get(frame, {})
        true_positive += len(truth.keys() & prediction.keys())
        false_positive += len(prediction.keys() - truth.keys())
        false_negative += len(truth.keys() - prediction.keys())
        for category in truth.keys() | prediction.keys():
            count_error += abs(truth.get(category, 0) - prediction.get(category, 0))
            pairs += 1
    return {"category_precision": true_positive / max(1, true_positive + false_positive),
            "category_recall": true_positive / max(1, true_positive + false_negative),
            "frame_category_count_mae": count_error / max(1, pairs),
            "missing_frames": len(expected.keys() - actual.keys()), "extra_frames": len(actual.keys() - expected.keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = [json.loads(line) for line in args.answers.open()]
    predictions = [json.loads(line) for line in args.predictions.open()]
    with args.output.open("x") as handle:
        json.dump(score_answers(expected, predictions), handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
