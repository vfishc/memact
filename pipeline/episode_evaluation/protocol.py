"""Model-independent causal context and prediction contract for episode evaluation."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from qa_semantic_contract import answer_contract_prompt
from streaming_episode.timing import normalize_timing_prediction, timing_protocol_payload


ACTION_TYPES = (
    "answer_now",
    "move_forward_d",
    "move_left_d",
    "move_right_d",
    "move_forward",
    "move_left",
    "move_right",
    "rotate_left_45",
    "rotate_right_45",
    "scan_forward_90",
    "sweep_360",
    "wait_for_evidence",
)
FRAME_POLICIES = ("recent", "uniform")
OPTION_ORDERS = ("original", "shuffled")


@dataclass(frozen=True)
class EngineCapabilities:
    structured_memory: bool
    persistent_cache: bool
    model_history: bool = True
    continuous_timing: bool = False


def _frame_key(frame: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(frame.get("kind", "")),
        str(frame.get("video", frame.get("path", ""))),
        str(frame.get("frame_id", frame.get("pose_id", ""))),
    )


def observation_frames(event: Mapping[str, Any]) -> List[Dict[str, Any]]:
    event_input = event.get("input") if isinstance(event.get("input"), Mapping) else {}
    observation = event_input.get("observation") if isinstance(event_input.get("observation"), Mapping) else None
    if observation is None:
        return []
    event_frame = int(event.get("frame_id", 0))
    kind = str(observation.get("kind", ""))
    frames: List[Dict[str, Any]] = []
    if kind == "video_frames":
        video = str(observation.get("video", ""))
        for frame_id in observation.get("frame_ids", []):
            frame_id = int(frame_id)
            if frame_id > event_frame:
                raise ValueError(f"future frame {frame_id} at event frame {event_frame}")
            frames.append({
                "kind": "video_frame",
                "video": video,
                "frame_id": frame_id,
                "source_fps": float(observation.get("source_fps", 24.0) or 24.0),
                "sample_fps": float(observation.get("sample_fps", 1.0) or 1.0),
                "event_id": str(event.get("event_id", "")),
            })
    elif kind == "image_sequence":
        paths = list(observation.get("paths", []))
        pose_ids = list(observation.get("pose_ids", []))
        if len(paths) != len(pose_ids):
            raise ValueError("image_sequence paths and pose_ids do not align")
        for path, pose_id in zip(paths, pose_ids):
            frames.append({
                "kind": "image",
                "path": str(path),
                "pose_id": str(pose_id),
                "frame_id": event_frame,
                "sample_fps": float(observation.get("sample_fps", 1.0) or 1.0),
                "event_id": str(event.get("event_id", "")),
            })
    else:
        raise ValueError(f"unsupported observation kind: {kind}")
    return frames


def select_frames(frames: Sequence[Mapping[str, Any]], *, policy: str, max_frames: int) -> List[Dict[str, Any]]:
    if policy not in FRAME_POLICIES:
        raise ValueError(f"unknown frame policy: {policy}")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    values = [copy.deepcopy(dict(frame)) for frame in frames]
    if len(values) <= max_frames:
        return values
    if policy == "recent":
        return values[-max_frames:]
    if max_frames == 1:
        return [values[-1]]
    indices = [round(index * (len(values) - 1) / (max_frames - 1)) for index in range(max_frames)]
    return [values[index] for index in indices]


def with_option_order(
    event: Mapping[str, Any],
    *,
    order: str,
    seed: int,
) -> Dict[str, Any]:
    if order not in OPTION_ORDERS:
        raise ValueError(f"unknown option order: {order}")
    result = copy.deepcopy(dict(event))
    event_input = result.get("input") if isinstance(result.get("input"), Mapping) else None
    if order == "original" or event_input is None:
        return result
    options = event_input.get("options")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)) or len(options) < 2:
        return result
    event_id = str(result.get("event_id", ""))
    ranked = []
    for index, option in enumerate(options):
        payload = f"{seed}\0{event_id}\0{index}\0{option}".encode("utf-8", errors="ignore")
        ranked.append((hashlib.sha1(payload).digest(), index, copy.deepcopy(option)))
    updated_input = dict(event_input)
    updated_input["options"] = [option for _digest, _index, option in sorted(ranked)]
    result["input"] = updated_input
    return result


def _extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            return value
    return stripped


def normalize_prediction(value: Any, event_type: str) -> Any:
    parsed = _extract_json(value) if isinstance(value, str) else value
    if event_type == "qa_turn":
        if isinstance(parsed, Mapping):
            if "answer" in parsed:
                return dict(parsed)
            if "prediction" in parsed:
                return {"mode": "answer", "answer": parsed.get("prediction")}
        return {"mode": "answer", "answer": str(parsed).strip()}
    if event_type == "active_decision":
        if isinstance(parsed, Mapping):
            if isinstance(parsed.get("action"), Mapping):
                return dict(parsed)
            action_type = str(parsed.get("type", parsed.get("answer", ""))).strip()
        else:
            normalized = str(parsed).strip().lower().replace(" ", "_")
            action_type = next((action for action in ACTION_TYPES if action in normalized), normalized)
        return {
            "mode": "answer" if action_type == "answer_now" else "act",
            "action": {"type": action_type},
        }
    if event_type == "timing_decision":
        return normalize_timing_prediction(parsed)
    return parsed


def build_baseline_prompt(
    event: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    include_history: bool,
) -> str:
    event_type = str(event.get("event_type", ""))
    event_input = event.get("input") if isinstance(event.get("input"), Mapping) else {}
    lines = [
        "Use only the supplied causal visual observations. Do not assume future events.",
    ]
    if include_history and history:
        lines.append("Previous model dialogue:")
        for turn in history:
            lines.append(f"User: {turn.get('question', '')}")
            lines.append(f"Assistant: {turn.get('answer', '')}")
    question = str(event_input.get("question", event_input.get("instruction", "")))
    if event_type == "qa_turn":
        lines.append(f"Current question: {question}")
        options = event_input.get("options")
        if isinstance(options, Sequence) and not isinstance(options, (str, bytes)) and options:
            lines.append("Candidate answers:")
            for index, option in enumerate(options):
                lines.append(f"{chr(ord('A') + index)}. {option}")
            lines.append("Choose exactly one candidate and return its complete value, not its letter.")
        else:
            lines.append("For a numeric answer, return a JSON number without explanatory text.")
        lines.append(answer_contract_prompt(event))
        lines.append('Return exactly one JSON object with keys "mode" and "answer"; use mode "answer".')
    elif event_type == "active_decision":
        lines.extend([
            f"Current question: {question}",
            "Choose one action type from: " + ", ".join(ACTION_TYPES) + ".",
            'Use mode "answer" only with answer_now; use mode "act" for every other action type.',
            "Only choose the decision type. Target frame, timing, and motion are generated by the environment.",
            'Return exactly one JSON object: {"mode":"act|answer","action":{"type":"..."}}',
        ])
    elif event_type == "timing_decision":
        protocol = timing_protocol_payload()
        lines.extend([
            f"Current monitored question: {question}",
            answer_contract_prompt(event),
            str(protocol["instruction"]),
            'Return exactly one JSON object: {"mode":"wait"} or {"mode":"answer","answer":...}.',
        ])
    else:
        raise ValueError(f"baseline prompt is unsupported for event type: {event_type}")
    return "\n".join(lines)


def frame_reference_text(frames: Sequence[Mapping[str, Any]]) -> str:
    references = []
    for index, frame in enumerate(frames, 1):
        if frame.get("kind") == "video_frame":
            references.append(f"frame:{frame.get('frame_id')}")
        else:
            references.append(f"action_result_image:{index}@frame:{frame.get('frame_id')}")
    return "Causal visual references: " + ", ".join(references) if references else "Causal visual references: none"


@dataclass
class CausalContext:
    frames: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    _frame_keys: set[Tuple[str, str, str]] = field(default_factory=set)

    def reset(self) -> None:
        self.frames.clear()
        self.history.clear()
        self._frame_keys.clear()

    def observe(self, event: Mapping[str, Any]) -> None:
        for frame in observation_frames(event):
            key = _frame_key(frame)
            if key in self._frame_keys:
                continue
            self._frame_keys.add(key)
            self.frames.append(frame)

    def selected_frames(self, *, policy: str, max_frames: int) -> List[Dict[str, Any]]:
        return select_frames(self.frames, policy=policy, max_frames=max_frames)

    def record_prediction(self, event: Mapping[str, Any], prediction: Any) -> None:
        if event.get("event_type") != "qa_turn":
            return
        normalized = normalize_prediction(prediction, "qa_turn")
        answer = normalized.get("answer", "") if isinstance(normalized, Mapping) else normalized
        event_input = event.get("input") if isinstance(event.get("input"), Mapping) else {}
        self.history.append({
            "event_id": str(event.get("event_id", "")),
            "question": str(event_input.get("question", "")),
            "answer": answer,
        })

    def request(
        self,
        manifest: Mapping[str, Any],
        stream: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        frame_policy: str,
        max_frames: int,
        include_history: bool,
    ) -> Dict[str, Any]:
        event_type = str(event.get("event_type", ""))
        if event_type not in {"qa_turn", "active_decision", "timing_decision"}:
            raise ValueError(f"request event must be QA, action, or timing: {event_type}")
        selected = self.selected_frames(policy=frame_policy, max_frames=max_frames)
        prompt = build_baseline_prompt(event, self.history, include_history=include_history)
        prompt = frame_reference_text(selected) + "\n" + prompt
        return {
            "schema_version": "s3_episode_baseline_request_v1.0",
            "request_id": str(event.get("event_id", "")),
            "episode_id": str(manifest.get("episode_id", "")),
            "stream_id": str(stream.get("stream_id", "")),
            "stream_type": str(stream.get("stream_type", "")),
            "event_id": str(event.get("event_id", "")),
            "event_type": event_type,
            "frame_id": int(event.get("frame_id", 0)),
            "timestamp": float(event.get("timestamp", 0.0)),
            "frame_policy": frame_policy,
            "max_frames": max_frames,
            "frames": selected,
            "input": copy.deepcopy(dict(event.get("input") or {})),
            "history_event_ids": [str(turn.get("event_id", "")) for turn in self.history],
            "history": copy.deepcopy(self.history),
            "include_history": bool(include_history),
            "prompt": prompt,
            "provenance": {
                "split": manifest.get("split", ""),
                "domain": manifest.get("domain", ""),
                "source": manifest.get("source", ""),
                **dict(event.get("provenance") or {}),
            },
        }
