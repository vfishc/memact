"""Shared observability rules for metadata-derived VLM supervision."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from qa_semantic_contract import (
    CATEGORY_CHOICE_TEMPLATES,
    VISUAL_CATEGORY_CONTRACT_VERSION,
    categories_conflict,
    category_conflict_family,
    clean_category_options,
    question_category_issues,
    visual_category,
    visual_category_rejection,
)
from qa_difficulty_contract import (
    CAMERA_DISTANCE_BINS,
    CAMERA_MOTION_BINS,
    CORE_CHOICE_MAX_OPTIONS,
    CORE_CHOICE_MIN_OPTIONS,
    DIRECTION_OPTIONS,
    OBJECT_DISTANCE_BINS,
    bucket_label,
)


QUALITY_POLICY = "vlm_observable_v3_semantic"

GENERIC_OBJECT_CATEGORIES = {
    "accessory",
    "accessories",
    "background",
    "ceiling",
    "floor",
    "misc",
    "miscellaneous",
    "object",
    "objects",
    "other",
    "others",
    "unknown",
    "wall",
}

ROOM_TYPE_ALIASES = {
    "aisle": "hallway",
    "bathroom": "bathroom",
    "bedroom": "bedroom",
    "bedroomroom": "bedroom",
    "diningroom": "dining room",
    "elderlyroom": "bedroom",
    "entrance": "entryway",
    "entryway": "entryway",
    "hallway": "hallway",
    "kidsroom": "bedroom",
    "kitchen": "kitchen",
    "laundryroom": "laundry room",
    "library": "study",
    "livingdiningroom": "living and dining room",
    "livingroom": "living room",
    "masterbedroom": "bedroom",
    "secondbedroom": "bedroom",
    "storageroom": "storage room",
    "study": "study",
    "terrace": "terrace",
}

ROOM_TYPE_VOCABULARY = (
    "bathroom",
    "bedroom",
    "dining room",
    "entryway",
    "hallway",
    "kitchen",
    "laundry room",
    "living and dining room",
    "living room",
    "storage room",
    "study",
    "terrace",
)
ROOM_TYPE_CONFLICTS = {
    "living and dining room": frozenset({"living room", "dining room"}),
    "living room": frozenset({"living and dining room"}),
    "dining room": frozenset({"living and dining room"}),
}

CAMERA_DISTANCE_OPTIONS = tuple(str(row[0]) for row in CAMERA_DISTANCE_BINS)
OBJECT_DISTANCE_OPTIONS = tuple(str(row[0]) for row in OBJECT_DISTANCE_BINS)
MOTION_OPTIONS = tuple(str(row[0]) for row in CAMERA_MOTION_BINS)
PROXIMITY_OPTIONS = OBJECT_DISTANCE_OPTIONS
YES_NO_OPTIONS = ("yes", "no")

ROOM_TEMPLATE_IDS = {
    "current_room_type",
    "visited_room_type_first",
    "visited_room_type_count",
    "room_type_then_closest_category",
}

OBJECT_GEOMETRY_TEMPLATE_IDS = {
    "camera_to_object_distance_bucket",
    "camera_to_object_distance_m",
    "closest_object_category",
    "object_relative_to_camera",
    "object_to_object_distance_bucket",
    "object_to_object_distance_m",
    "object_size_m",
    "object_spatial_proximity",
    "approached_object_category",
    "closer_than_before",
    "closest_category_then_distance",
    "first_seen_category_then_direction",
    "closest_category_then_relation",
    "object_category_then_later_visibility",
    "first_seen_timestamp_s",
    "camera_path_length_m",
    "camera_displacement_m",
    "camera_rotation_deg",
}

_VISIBLE_INTERNAL_PATTERNS = (
    ("allroom", re.compile(r"\ballroom\b", re.IGNORECASE)),
    ("scene_id", re.compile(r"\bscene\d{3,}(?:[_-]\d+)?\b", re.IGNORECASE)),
    ("opaque_hash", re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)),
    (
        "instance_id",
        re.compile(r"\b[a-z][a-z0-9_ -]*(?<!\d)[._-]\d{2,}\b", re.IGNORECASE),
    ),
)
_ANSWER_INSTANCE_SUFFIX = re.compile(r"^[a-z][a-z _-]*\s+\d+$", re.IGNORECASE)

ALLOWED_MEMORY_FIELDS = {
    "object_slots": {
        "slot_id",
        "slot_type",
        "category",
        "seen_count",
        "current_visible",
        "current_visible_count",
        "first_seen_order",
        "branch_observation",
    },
    "room_slots": {
        "slot_id",
        "slot_type",
        "room_type",
        "observed_room_count",
        "visit_count",
        "first_visit_order",
    },
}


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def observable_category_name(value: Any) -> Optional[str]:
    return visual_category(value)


def observable_object_category(record: Mapping[str, Any], object_id: str) -> Optional[str]:
    objects = record.get("objects", {})
    obj = objects.get(object_id, {}) if isinstance(objects, Mapping) else {}
    return observable_category_name(obj.get("category") or object_id)


def category_groups(record: Mapping[str, Any], object_ids: Iterable[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for object_id in sorted({str(value) for value in object_ids if str(value)}):
        category = observable_object_category(record, object_id)
        if category:
            groups[category].append(object_id)
    return dict(groups)


def unique_category_label(record: Mapping[str, Any], object_id: str, candidate_ids: Iterable[str]) -> Optional[str]:
    category = observable_object_category(record, object_id)
    if not category:
        return None
    groups = category_groups(record, candidate_ids)
    return category if groups.get(category) == [object_id] else None


def category_options(
    record: Mapping[str, Any],
    target_id: str,
    candidate_ids: Iterable[str],
    *,
    require_unique_target: bool,
    min_options: int = CORE_CHOICE_MIN_OPTIONS,
    max_options: int = CORE_CHOICE_MAX_OPTIONS,
) -> Optional[Tuple[str, List[str]]]:
    ordered_ids = [str(value) for value in candidate_ids if str(value)]
    groups = category_groups(record, ordered_ids)
    answer = observable_object_category(record, target_id)
    if not answer or answer not in groups:
        return None
    if require_unique_target and len(groups[answer]) != 1:
        return None
    ranked_categories = []
    for object_id in ordered_ids:
        category = observable_object_category(record, object_id)
        if category and category not in ranked_categories:
            ranked_categories.append(category)
    _clean_answer, options, _reasons = clean_category_options(answer, [answer, *ranked_categories])
    if len(options) < min_options:
        return None
    if len(options) > max_options:
        distractors = [value for value in ranked_categories if value in options and value != answer][: max_options - 1]
        options = sorted([answer, *distractors])
    return answer, options


def semantic_room_type(record: Mapping[str, Any], room_id: Any) -> Optional[str]:
    if str(record.get("source", "")) != "sim_front3d":
        return None
    raw = str(room_id or "").strip()
    if not raw:
        return None
    prefix = re.sub(r"[-_. ]*\d+$", "", raw)
    key = re.sub(r"[^a-z]", "", prefix.lower())
    return ROOM_TYPE_ALIASES.get(key)


def room_type_options(answer: str, max_options: int = 5) -> List[str]:
    conflicts = ROOM_TYPE_CONFLICTS.get(answer, frozenset())
    distractors = [value for value in ROOM_TYPE_VOCABULARY if value != answer and value not in conflicts]
    rank = sum(ord(char) for char in answer) % max(1, len(distractors))
    rotated = distractors[rank:] + distractors[:rank]
    selected = [answer]
    for candidate in rotated:
        if any(
            candidate in ROOM_TYPE_CONFLICTS.get(value, frozenset())
            or value in ROOM_TYPE_CONFLICTS.get(candidate, frozenset())
            for value in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_options:
            break
    return sorted(selected)


def camera_distance_bucket(distance: Optional[float]) -> Optional[str]:
    return bucket_label(distance, CAMERA_DISTANCE_BINS)


def object_distance_bucket(distance: Optional[float]) -> Optional[str]:
    return bucket_label(distance, OBJECT_DISTANCE_BINS)


def camera_motion_bucket(distance: Optional[float]) -> Optional[str]:
    return bucket_label(distance, CAMERA_MOTION_BINS)


def proximity_label(distance: Optional[float], threshold: float = 1.5, margin: float = 0.3) -> Optional[str]:
    if distance is None or distance < 0:
        return None
    if distance <= threshold - margin:
        return "near"
    if distance >= threshold + margin:
        return "far"
    return None


def valid_direction(value: Optional[str]) -> Optional[str]:
    return value if value in DIRECTION_OPTIONS else None


def has_valid_geometry(values: Sequence[Optional[Sequence[float]]]) -> bool:
    normalized = []
    for value in values:
        if value is None or len(value) < 3:
            return False
        point = tuple(round(float(component), 6) for component in value[:3])
        normalized.append(point)
    if any(point == (0.0, 0.0, 0.0) for point in normalized):
        return False
    return len(set(normalized)) == len(normalized)


def category_counts(record: Mapping[str, Any], object_ids: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for object_id in object_ids:
        category = observable_object_category(record, str(object_id))
        if category:
            counts[category] += 1
    return counts


def source_supports_template(record: Mapping[str, Any], template_id: str) -> bool:
    """Reserve source-level gates for capabilities that cannot be checked per sample."""

    return True


def visible_text_issues(value: Any, *, allow_plain_number: bool = False) -> List[str]:
    """Return internal-identifier issues in model-visible text."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [] if allow_plain_number else ["unexpected_numeric_text"]
    text = str(value or "").strip()
    issues = [name for name, pattern in _VISIBLE_INTERNAL_PATTERNS if pattern.search(text)]
    if _ANSWER_INSTANCE_SUFFIX.fullmatch(text):
        issues.append("numbered_instance_label")
    return issues


def qa_sample_quality_issues(sample: Mapping[str, Any]) -> List[str]:
    """Check the model-visible part of one observable QA sample."""

    issues: List[str] = []
    source = str(sample.get("source", ""))
    template = str(sample.get("template_id", ""))
    if template and not source_supports_template(sample, template):
        issues.append("source_template_not_supported")
    if source.startswith("real_") and template in ROOM_TEMPLATE_IDS:
        issues.append("real_source_room_qa")

    question = str(sample.get("question", "")).strip()
    if not question:
        issues.append("empty_question")
    for issue in visible_text_issues(question, allow_plain_number=True):
        if issue != "numbered_instance_label":
            issues.append(f"question:{issue}")
    issues.extend(question_category_issues(question))

    def check_answer_options(answer: Any, options: Any, prefix: str) -> None:
        if answer is None or str(answer).strip() == "":
            issues.append(f"{prefix}:empty_answer")
            return
        for issue in visible_text_issues(answer, allow_plain_number=True):
            issues.append(f"{prefix}:answer:{issue}")
        if not isinstance(options, list) or not options:
            if not isinstance(answer, (int, float)) or isinstance(answer, bool):
                issues.append(f"{prefix}:missing_options")
            return
        normalized = [str(value).strip().lower() for value in options]
        if len(options) < 2:
            issues.append(f"{prefix}:too_few_options")
        if len(normalized) != len(set(normalized)):
            issues.append(f"{prefix}:duplicate_options")
        if str(answer).strip().lower() not in normalized:
            issues.append(f"{prefix}:answer_not_in_options")
        for value in options:
            for issue in visible_text_issues(value, allow_plain_number=True):
                issues.append(f"{prefix}:option:{issue}")

    labels = sample.get("labels", {})
    answer = labels.get("answer") if isinstance(labels, Mapping) else None
    check_answer_options(answer, sample.get("options", []), "sample")

    def check_category_choices(category_answer: Any, category_options: Any, prefix: str) -> None:
        rejection = visual_category_rejection(category_answer)
        if rejection:
            issues.append(f"{prefix}:answer:{rejection}")
        if not isinstance(category_options, list):
            return
        seen_families: Dict[str, str] = {}
        for option in category_options:
            option_rejection = visual_category_rejection(option)
            if option_rejection:
                issues.append(f"{prefix}:option:{option_rejection}")
                continue
            family = category_conflict_family(option)
            if family and family in seen_families:
                issues.append(f"{prefix}:option_conflict_family:{family}")
            elif family:
                seen_families[family] = str(option)

    if template in CATEGORY_CHOICE_TEMPLATES:
        check_category_choices(answer, sample.get("options", []), "sample")

    turns = sample.get("turns")
    if isinstance(turns, list):
        for index, turn in enumerate(turns, 1):
            if not isinstance(turn, Mapping):
                issues.append(f"turn_{index}:not_object")
                continue
            turn_question = str(turn.get("question", "")).strip()
            if not turn_question:
                issues.append(f"turn_{index}:empty_question")
            for issue in visible_text_issues(turn_question, allow_plain_number=True):
                if issue != "numbered_instance_label":
                    issues.append(f"turn_{index}:question:{issue}")
            issues.extend(
                f"turn_{index}:{issue}"
                for issue in question_category_issues(turn_question)
            )
            check_answer_options(turn.get("answer"), turn.get("options", []), f"turn_{index}")
            category_turn = (
                template == "closest_category_then_relation"
                and index in {1, 2}
            ) or (
                index == 1
                and template in {
                    "closest_category_then_distance",
                    "first_seen_category_then_direction",
                    "object_category_then_later_visibility",
                }
            ) or (index == 2 and template == "room_type_then_closest_category")
            if category_turn:
                check_category_choices(
                    turn.get("answer"), turn.get("options", []), f"turn_{index}"
                )
        if template == "object_category_then_later_visibility" and len(turns) >= 3:
            second = turns[1] if isinstance(turns[1], Mapping) else {}
            if str(second.get("answer", "")).strip().lower() == "no":
                issues.append("later_visibility_no_has_followup")
        if template == "closest_category_then_relation" and len(turns) >= 2:
            first = turns[0] if isinstance(turns[0], Mapping) else {}
            second = turns[1] if isinstance(turns[1], Mapping) else {}
            if categories_conflict(first.get("answer"), second.get("answer")):
                issues.append("refchain_same_family_category_pair")
    return sorted(set(issues))


def memory_patch_quality_issues(patch: Mapping[str, Any]) -> List[str]:
    """Enforce the compact category-only memory target contract."""

    issues: List[str] = []
    operation_count = 0
    for slot_type in ("object_slots", "room_slots", "event_slots", "reference_slots"):
        section = patch.get(slot_type)
        if not isinstance(section, Mapping):
            issues.append(f"{slot_type}:missing_section")
            continue
        upserts = section.get("upsert")
        deletes = section.get("delete")
        if not isinstance(upserts, list) or not isinstance(deletes, list):
            issues.append(f"{slot_type}:invalid_operations")
            continue
        operation_count += len(upserts) + len(deletes)
        if slot_type in {"event_slots", "reference_slots"}:
            if upserts or deletes:
                issues.append(f"{slot_type}:must_be_empty")
            continue
        allowed = ALLOWED_MEMORY_FIELDS[slot_type]
        expected_prefix = "category:" if slot_type == "object_slots" else "room_type:"
        for index, slot in enumerate(upserts):
            if not isinstance(slot, Mapping):
                issues.append(f"{slot_type}:upsert_{index}:not_object")
                continue
            extra = sorted(set(slot) - allowed)
            if extra:
                issues.append(f"{slot_type}:upsert_{index}:forbidden_fields:{','.join(extra)}")
            slot_id = str(slot.get("slot_id", ""))
            if not slot_id.startswith(expected_prefix):
                issues.append(f"{slot_type}:upsert_{index}:nonsemantic_slot_id")
            visible_value = slot.get("category") if slot_type == "object_slots" else slot.get("room_type")
            for issue in visible_text_issues(visible_value, allow_plain_number=True):
                issues.append(f"{slot_type}:upsert_{index}:label:{issue}")
            if slot_type == "object_slots":
                rejection = visual_category_rejection(visible_value)
                if rejection:
                    issues.append(f"{slot_type}:upsert_{index}:label:{rejection}")
        for value in deletes:
            if not str(value).startswith(expected_prefix):
                issues.append(f"{slot_type}:nonsemantic_delete_id")
    if operation_count == 0:
        issues.append("empty_noop_patch")
    return sorted(set(issues))
