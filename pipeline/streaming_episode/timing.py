"""Evidence-gated answer-timing contract for streaming QA."""

from __future__ import annotations

import math
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence


TIMING_CONTRACT_VERSION = "s3_evidence_timing_v2"
FIXED_HORIZON_DIAGNOSTIC_VERSION = "s3_fixed_horizon_timing_diagnostic_v1"
TIMING_EVENT_TYPE = "timing_decision"
TIMING_MODES = frozenset({"wait", "answer"})
TIMING_ELIGIBLE_TEMPLATES = frozenset({
    "became_visible_category",
    "disappeared_category",
})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _forward_question(template_id: str, source_question: str, meta: Mapping[str, Any]) -> str:
    questions = {
        "became_visible_category": "Which listed object category becomes newly and stably visible first?",
        "disappeared_category": "Which listed object category stops being stably visible first?",
    }
    question = questions.get(template_id, str(source_question).strip())
    return (
        "Monitor the causal video. Return wait while the listed alternatives are not yet visually "
        "distinguishable, then answer at the first check with sufficient visual evidence. " + question
    )


def timing_spec_from_delta(sample: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a validated Delta QA into a forward-monitoring timing target."""

    template_id = str(sample.get("template_id", ""))
    if template_id not in TIMING_ELIGIBLE_TEMPLATES:
        return None
    labels = _mapping(sample.get("labels"))
    source_qa = _mapping(labels.get("source_qa"))
    meta = _mapping(source_qa.get("meta"))
    before_frame = meta.get("before_frame")
    answer_frame = meta.get("after_frame", sample.get("frame_id"))
    delta_seconds = meta.get("delta_seconds")
    if (
        not isinstance(before_frame, int)
        or not isinstance(answer_frame, int)
        or before_frame < 0
        or answer_frame <= before_frame
    ):
        return None
    try:
        duration = float(delta_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    answer = labels.get("answer")
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        return None
    options = sample.get("options")
    normalized_options = list(options) if isinstance(options, list) else []
    return {
        "version": TIMING_CONTRACT_VERSION,
        "source_sample_id": str(sample.get("sample_id", "")),
        "template_id": template_id,
        "question_frame": before_frame,
        "first_answerable_frame": answer_frame,
        "delta_seconds": duration,
        "trigger": "first_sufficient_visual_evidence",
        "question": _forward_question(template_id, str(sample.get("question", "")), meta),
        "options": normalized_options,
        "answer": answer,
    }


def timing_check_frames(
    question_frame: int,
    first_answerable_frame: int,
    *,
    fps: float,
    interval_seconds: float = 1.0,
    max_wait_events: int = 3,
) -> List[int]:
    if question_frame < 0 or first_answerable_frame <= question_frame:
        raise ValueError("timing frame bounds are invalid")
    if fps <= 0 or interval_seconds <= 0 or max_wait_events < 1:
        raise ValueError("timing cadence configuration is invalid")
    step = max(1, int(round(fps * interval_seconds)))
    waits = list(range(question_frame, first_answerable_frame, step))
    if question_frame not in waits:
        waits.insert(0, question_frame)
    if len(waits) > max_wait_events:
        if max_wait_events == 1:
            waits = [waits[0]]
        else:
            indices = [
                round(index * (len(waits) - 1) / (max_wait_events - 1))
                for index in range(max_wait_events)
            ]
            waits = [waits[index] for index in indices]
    return sorted(set([*waits, first_answerable_frame]))


def timing_target(spec: Mapping[str, Any], frame_id: int) -> Dict[str, Any]:
    first_answerable = int(spec["first_answerable_frame"])
    if frame_id < first_answerable:
        return {"type": TIMING_EVENT_TYPE, "mode": "wait"}
    return {
        "type": TIMING_EVENT_TYPE,
        "mode": "answer",
        "answer": spec.get("answer"),
    }


def validate_timing_target(target: Mapping[str, Any]) -> None:
    if target.get("type") != TIMING_EVENT_TYPE:
        raise ValueError("timing target type mismatch")
    mode = str(target.get("mode", ""))
    if mode not in TIMING_MODES:
        raise ValueError("timing target mode must be wait or answer")
    if mode == "wait" and "answer" in target:
        raise ValueError("wait timing target cannot contain an answer")
    if mode == "answer" and (
        "answer" not in target
        or target.get("answer") is None
        or (isinstance(target.get("answer"), str) and not str(target["answer"]).strip())
    ):
        raise ValueError("answer timing target requires a non-empty answer")


def normalize_timing_prediction(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping):
            value = decoded
    if isinstance(value, Mapping):
        mode = str(value.get("mode", "")).strip().lower()
        if mode in TIMING_MODES:
            output = {"mode": mode}
            if mode == "answer" and "answer" in value:
                output["answer"] = value.get("answer")
            return output
    text = str(value or "").strip()
    lowered = text.lower()
    if re.fullmatch(r"(?:wait|\{?\s*[\"']?mode[\"']?\s*:\s*[\"']?wait[\"']?\s*\}?)", lowered):
        return {"mode": "wait"}
    return {"mode": "answer", "answer": value}


def timing_protocol_payload() -> Dict[str, Any]:
    return {
        "version": TIMING_CONTRACT_VERSION,
        "modes": ["wait", "answer"],
        "instruction": (
            "Return {\"mode\":\"wait\"} while evidence is insufficient. At the first "
            "sufficient check, return {\"mode\":\"answer\",\"answer\":...}."
        ),
        "target_duration_visible": False,
    }
