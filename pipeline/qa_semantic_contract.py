"""Versioned visual-category and answer contracts for Plan 19."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VISUAL_CATEGORY_CONTRACT_VERSION = "s3_visual_category_v2"
ANSWER_CONTRACT_VERSION = "s3_answer_contract_v2"

# Frozen from the visual labels accepted by the Plan 18 semantic audit. New
# labels must be reviewed and added explicitly; they never enter by default.
VISUAL_CATEGORY_ALLOWLIST = frozenset({
    "armchair", "art", "barstool", "basin", "basket", "bath", "bathtub",
    "bed", "bed frame", "blanket", "bookshelf", "bowl", "bucket", "bunk bed",
    "cabinet", "ceiling lamp", "ceiling light", "chair", "chaise longue sofa",
    "children cabinet", "classic chinese chair", "clock", "coat hanger",
    "computer mouse", "computer tower", "counter", "crate", "cup", "curtain",
    "cushion", "cutting board", "desk", "dining chair", "dining table",
    "dishwasher", "door", "double bed", "dressing chair", "dressing table",
    "exhaust fan", "file folder", "fireplace", "headphones", "heater", "kettle",
    "keyboard", "kids bed", "kitchen cabinet", "l-shaped sofa", "laptop",
    "lazy sofa", "lighting", "media unit", "microwave", "mirror", "monitor",
    "nightstand", "outdoor furniture", "oven", "pan", "paper bag", "pendant lamp",
    "picture", "pillow", "plant", "power strip", "printer", "rack",
    "refrigerator", "round end table", "shelf", "shoe rack", "shoes",
    "shower curtain", "single bed", "sink", "slippers", "sofa", "stair",
    "stapler", "stool", "storage unit", "stove", "suitcase", "table",
    "table lamp", "tea table", "telephone", "toilet", "trash can", "tv",
    "tv monitor", "tv stand", "two-seat sofa", "wardrobe", "washer",
    "whiteboard", "window", "wine cooler",
})

TAXONOMY_LABEL_RE = re.compile(r"^\d{3}\s*-\s*", re.IGNORECASE)
TAXONOMY_LABEL_SEARCH_RE = re.compile(r"(?<!\w)\d{3}\s*-\s*[a-z]", re.IGNORECASE)
VAGUE_VISUAL_CATEGORIES = frozenset({
    "build element", "recreation", "electronics", "appliance",
})
GENERIC_VISUAL_CATEGORIES = frozenset({
    "accessory", "accessories", "background", "ceiling", "floor", "misc",
    "miscellaneous", "object", "objects", "other", "others", "unknown", "wall",
})

CATEGORY_CONFLICT_FAMILIES: Mapping[str, frozenset[str]] = {
    "table": frozenset({"table", "dining table", "tea table", "round end table", "dressing table"}),
    "bed": frozenset({"bed", "bed frame", "bunk bed", "double bed", "kids bed", "single bed"}),
    "lighting": frozenset({"lighting", "ceiling lamp", "ceiling light", "pendant lamp", "table lamp"}),
    "display": frozenset({"tv", "tv monitor", "monitor"}),
    "plant": frozenset({"plant"}),
    "chair": frozenset({"armchair", "chair", "classic chinese chair", "dining chair", "dressing chair"}),
    "stool": frozenset({"barstool", "stool"}),
    "sofa": frozenset({"chaise longue sofa", "l-shaped sofa", "lazy sofa", "sofa", "two-seat sofa"}),
    "cabinet_storage": frozenset({"cabinet", "children cabinet", "kitchen cabinet", "storage unit", "wardrobe"}),
    "shelf_rack": frozenset({"bookshelf", "rack", "shelf", "shoe rack"}),
    "media_unit": frozenset({"media unit", "tv stand"}),
    "bath": frozenset({"bath", "bathtub"}),
    "sink": frozenset({"basin", "sink"}),
    "curtain": frozenset({"curtain", "shower curtain"}),
    "art": frozenset({"art", "picture"}),
}

CATEGORY_ALIASES = {"plants": "plant"}

_CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in CATEGORY_CONFLICT_FAMILIES.items()
    for category in categories
}

CATEGORY_CHOICE_TEMPLATES = frozenset({
    "approached_object_category", "became_visible_category", "closest_object_category",
    "disappeared_category", "first_seen_object_category", "seen_category_order",
})
CATEGORY_PAIR_TEMPLATES = frozenset({
    "object_spatial_proximity", "object_to_object_distance_bucket",
    "object_to_object_distance_m", "seen_category_order",
})

ALL_QA_TEMPLATE_IDS = (
    "approached_object_category", "became_visible_category", "camera_motion_bucket",
    "camera_to_object_distance_bucket", "closer_than_before",
    "closest_category_then_distance", "closest_category_then_relation",
    "closest_object_category", "current_room_type", "disappeared_category",
    "first_seen_category_then_direction", "first_seen_object_category",
    "object_category_then_later_visibility", "object_relative_to_camera",
    "object_spatial_proximity", "object_to_object_distance_bucket",
    "room_type_then_closest_category", "seen_category_order", "seen_count_by_category",
    "visited_room_type_count", "visited_room_type_first", "camera_to_object_distance_m",
    "object_to_object_distance_m", "object_size_m", "first_seen_timestamp_s",
    "camera_path_length_m", "camera_displacement_m", "camera_rotation_deg",
)

NUMERIC_TEMPLATE_UNITS = {
    "seen_count_by_category": "objects",
    "visited_room_type_count": "distinct room types",
    "camera_to_object_distance_m": "meters",
    "object_to_object_distance_m": "meters",
    "object_size_m": "meters",
    "first_seen_timestamp_s": "seconds",
    "camera_path_length_m": "meters",
    "camera_displacement_m": "meters",
    "camera_rotation_deg": "degrees",
}
INTEGER_TEMPLATE_IDS = frozenset({
    "seen_count_by_category",
    "visited_room_type_count",
})
NUMERIC_TEMPLATE_IDS = frozenset(NUMERIC_TEMPLATE_UNITS) - INTEGER_TEMPLATE_IDS
DISTANCE_TEMPLATE_IDS = frozenset({
    "camera_motion_bucket", "camera_to_object_distance_bucket",
    "closest_category_then_distance", "closest_category_then_relation",
    "object_spatial_proximity", "object_to_object_distance_bucket",
    "camera_to_object_distance_m", "object_to_object_distance_m",
    "object_size_m", "camera_path_length_m", "camera_displacement_m",
})
SPATIAL_METER_TEMPLATE_IDS = frozenset(DISTANCE_TEMPLATE_IDS)

ANSWER_CONTRACT: Dict[str, Any] = {
    "version": ANSWER_CONTRACT_VERSION,
    "templates": {
        template_id: {
            "response_type": (
                "integer" if template_id in INTEGER_TEMPLATE_IDS
                else "numeric" if template_id in NUMERIC_TEMPLATE_IDS
                else "multiple_choice"
            ),
            "unit": NUMERIC_TEMPLATE_UNITS.get(template_id, "meters" if template_id in DISTANCE_TEMPLATE_IDS else None),
        }
        for template_id in ALL_QA_TEMPLATE_IDS
    },
    "multiple_choice": "return the complete option value, never an option letter",
    "numeric_metric": "mean_relative_accuracy",
    "choice_metric": "accuracy",
}


def _contract_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


VISUAL_CATEGORY_CONTRACT_HASH = _contract_hash({
    "version": VISUAL_CATEGORY_CONTRACT_VERSION,
    "allowlist": sorted(VISUAL_CATEGORY_ALLOWLIST),
    "conflict_families": {key: sorted(value) for key, value in sorted(CATEGORY_CONFLICT_FAMILIES.items())},
    "vague": sorted(VAGUE_VISUAL_CATEGORIES),
})
ANSWER_CONTRACT_HASH = _contract_hash(ANSWER_CONTRACT)


def normalize_semantic_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"\s+", " ", text)
    return CATEGORY_ALIASES.get(normalized, normalized)


def visual_category_rejection(value: Any) -> Optional[str]:
    category = normalize_semantic_label(value)
    if not category:
        return "empty_category"
    if TAXONOMY_LABEL_RE.match(category):
        return "taxonomy_category"
    if category in VAGUE_VISUAL_CATEGORIES:
        return "vague_category"
    if category in GENERIC_VISUAL_CATEGORIES or category.startswith("other "):
        return "generic_category"
    if "/" in category or re.fullmatch(r"(?:object|item|thing)\s*\d*", category):
        return "generic_category"
    if re.fullmatch(r"[a-z][a-z _-]*[._-]\d+", category):
        return "instance_category"
    if category not in VISUAL_CATEGORY_ALLOWLIST:
        return "unknown_category"
    return None


def visual_category(value: Any) -> Optional[str]:
    category = normalize_semantic_label(value)
    return None if visual_category_rejection(category) else category


def category_conflict_family(value: Any) -> Optional[str]:
    return _CATEGORY_TO_FAMILY.get(normalize_semantic_label(value))


def categories_conflict(first: Any, second: Any) -> bool:
    first_family = category_conflict_family(first)
    return bool(first_family and first_family == category_conflict_family(second))


def clean_category_options(answer: Any, options: Sequence[Any]) -> Tuple[Optional[str], List[str], List[str]]:
    """Clean category choices while preserving at most one label per conflict family."""

    reasons: List[str] = []
    clean_answer = visual_category(answer)
    if clean_answer is None:
        return None, [], [f"answer:{visual_category_rejection(answer)}"]
    answer_family = category_conflict_family(clean_answer)
    retained: List[str] = [clean_answer]
    seen_values = {clean_answer}
    seen_families = {answer_family} if answer_family else set()
    for raw in options:
        value = normalize_semantic_label(raw)
        if value == clean_answer:
            continue
        rejection = visual_category_rejection(value)
        if rejection:
            reasons.append(f"distractor:{rejection}")
            continue
        if value in seen_values:
            reasons.append("distractor:duplicate")
            continue
        family = category_conflict_family(value)
        if family and family in seen_families:
            reasons.append("distractor:conflict_family")
            continue
        retained.append(value)
        seen_values.add(value)
        if family:
            seen_families.add(family)
    return clean_answer, retained, sorted(set(reasons))


def question_category_issues(question: Any) -> List[str]:
    text = normalize_semantic_label(question)
    issues: List[str] = []
    if TAXONOMY_LABEL_SEARCH_RE.search(text):
        issues.append("question:taxonomy_category")
    for category in sorted(VAGUE_VISUAL_CATEGORIES):
        if re.search(rf"(?<!\w){re.escape(category)}(?!\w)", text):
            issues.append("question:vague_category")
    return issues


def answer_contract_metadata() -> Dict[str, str]:
    return {"version": ANSWER_CONTRACT_VERSION, "hash": ANSWER_CONTRACT_HASH}


def visual_category_contract_metadata() -> Dict[str, str]:
    return {"version": VISUAL_CATEGORY_CONTRACT_VERSION, "hash": VISUAL_CATEGORY_CONTRACT_HASH}


def template_id_from_event(event: Mapping[str, Any]) -> str:
    provenance = event.get("provenance") if isinstance(event.get("provenance"), Mapping) else {}
    event_input = event.get("input") if isinstance(event.get("input"), Mapping) else {}
    return str(provenance.get("template_id") or event_input.get("template_id") or "")


def answer_contract_for_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    template_id = template_id_from_event(event)
    event_input = event.get("input") if isinstance(event.get("input"), Mapping) else {}
    options = event_input.get("options")
    has_options = isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and bool(options)
    unit = NUMERIC_TEMPLATE_UNITS.get(template_id)
    if has_options:
        result: Dict[str, Any] = {
            "version": ANSWER_CONTRACT_VERSION,
            "response_type": "multiple_choice",
            "instruction": "Return the complete value of exactly one supplied option, not its letter.",
        }
    else:
        result = {
            "version": ANSWER_CONTRACT_VERSION,
            "response_type": (
                "integer" if template_id in INTEGER_TEMPLATE_IDS
                else "numeric" if template_id in NUMERIC_TEMPLATE_IDS
                else "value"
            ),
            "instruction": (
                "Return an integer count in the answer field."
                if template_id in INTEGER_TEMPLATE_IDS
                else "Return one numeric value in the answer field."
                if template_id in NUMERIC_TEMPLATE_IDS
                else "Return only the answer value in the answer field."
            ),
        }
    if unit:
        result["unit"] = unit
        if template_id in INTEGER_TEMPLATE_IDS:
            result["instruction"] = f"Return an integer count in {unit}."
        elif template_id in NUMERIC_TEMPLATE_IDS:
            result["instruction"] = f"Return one numeric value in {unit}."
    if template_id in SPATIAL_METER_TEMPLATE_IDS:
        result["spatial_unit"] = "meters"
    return result


def answer_contract_prompt(event: Mapping[str, Any]) -> str:
    contract = answer_contract_for_event(event)
    lines = [f"Answer contract {ANSWER_CONTRACT_VERSION}: {contract['instruction']}"]
    if contract.get("unit"):
        lines.append(f"The required numeric unit is {contract['unit']}.")
    if contract.get("spatial_unit"):
        lines.append("All spatial distances and motion magnitudes are in meters.")
    return " ".join(lines)


def all_categories_are_allowed(values: Iterable[Any]) -> bool:
    return all(visual_category(value) is not None for value in values)
