"""Versioned difficulty and wording contract for Plan 23 QA."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence


QA_DIFFICULTY_CONTRACT_VERSION = "s3_qa_difficulty_v1"
CORE_CHOICE_MIN_OPTIONS = 4
CORE_CHOICE_MAX_OPTIONS = 5

CAMERA_DISTANCE_BINS = (
    ("very near (at most 1.25 meters)", None, 1.25),
    ("near (1.5 to 2.5 meters)", 1.5, 2.5),
    ("mid-range (3 to 4.5 meters)", 3.0, 4.5),
    ("far (at least 5 meters)", 5.0, None),
)
OBJECT_DISTANCE_BINS = (
    ("very near (at most 0.75 meters)", None, 0.75),
    ("near (1 to 1.75 meters)", 1.0, 1.75),
    ("mid-range (2 to 3 meters)", 2.0, 3.0),
    ("far (at least 3.5 meters)", 3.5, None),
)
CAMERA_MOTION_BINS = (
    ("almost stationary (at most 0.2 meters)", None, 0.2),
    ("small (0.35 to 0.75 meters)", 0.35, 0.75),
    ("moderate (1 to 1.5 meters)", 1.0, 1.5),
    ("large (at least 2 meters)", 2.0, None),
)
DISTANCE_CHANGE_BINS = (
    ("much closer (decrease of at least 0.75 meters)", None, -0.75),
    ("slightly closer (decrease of 0.2 to 0.5 meters)", -0.5, -0.2),
    ("slightly farther (increase of 0.2 to 0.5 meters)", 0.2, 0.5),
    ("much farther (increase of at least 0.75 meters)", 0.75, None),
)
DIRECTION_OPTIONS = (
    "far to the left",
    "slightly to the left",
    "in front",
    "slightly to the right",
    "far to the right",
)

DIAGNOSTIC_CHOICE_TEMPLATES = frozenset({"object_category_then_later_visibility"})

_CONTRACT_PAYLOAD = {
    "version": QA_DIFFICULTY_CONTRACT_VERSION,
    "choice_options": [CORE_CHOICE_MIN_OPTIONS, CORE_CHOICE_MAX_OPTIONS],
    "camera_distance_bins": CAMERA_DISTANCE_BINS,
    "object_distance_bins": OBJECT_DISTANCE_BINS,
    "camera_motion_bins": CAMERA_MOTION_BINS,
    "distance_change_bins": DISTANCE_CHANGE_BINS,
    "direction_options": DIRECTION_OPTIONS,
    "diagnostic_choice_templates": sorted(DIAGNOSTIC_CHOICE_TEMPLATES),
}
QA_DIFFICULTY_CONTRACT_HASH = hashlib.sha256(
    json.dumps(_CONTRACT_PAYLOAD, sort_keys=True, separators=(",", ":")).encode("ascii")
).hexdigest()


def bucket_label(value: Optional[float], bins: Sequence[Sequence[Any]]) -> Optional[str]:
    if value is None or value < 0:
        return None
    for label, lower, upper in bins:
        if lower is None and value <= float(upper):
            return str(label)
        if upper is None and value >= float(lower):
            return str(label)
        if lower is not None and upper is not None and float(lower) <= value <= float(upper):
            return str(label)
    return None


def distance_change_label(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if value <= -0.75:
        return str(DISTANCE_CHANGE_BINS[0][0])
    if -0.5 <= value <= -0.2:
        return str(DISTANCE_CHANGE_BINS[1][0])
    if 0.2 <= value <= 0.5:
        return str(DISTANCE_CHANGE_BINS[2][0])
    if value >= 0.75:
        return str(DISTANCE_CHANGE_BINS[3][0])
    return None


def question_wording(question: str, *, split: str, identity: Sequence[Any]) -> tuple[str, str]:
    split = str(split)
    if split == "test":
        return f"Considering only evidence available up to this timestamp, {question}", "heldout_test_v1"
    prefixes = ("", "Using the causal observations so far, ", "From the observed stream up to now, ")
    digest = hashlib.sha1("\0".join(str(value) for value in identity).encode("utf-8")).digest()
    index = int.from_bytes(digest[:2], "big") % len(prefixes)
    return f"{prefixes[index]}{question}", f"train_family_{index + 1}"


def difficulty_contract_metadata() -> Mapping[str, Any]:
    return {
        "version": QA_DIFFICULTY_CONTRACT_VERSION,
        "hash": QA_DIFFICULTY_CONTRACT_HASH,
        "core_choice_min_options": CORE_CHOICE_MIN_OPTIONS,
        "core_choice_max_options": CORE_CHOICE_MAX_OPTIONS,
    }
