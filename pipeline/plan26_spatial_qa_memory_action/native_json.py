"""Optional transport-only JSON fence handling; never repair payload fields."""
import json


def parse_native_json(raw, allow_json_fence=False):
    text = raw.strip()
    normalized = False
    if allow_json_fence:
        for prefix in ("```json\n", "```\n"):
            if text.startswith(prefix) and text.endswith("\n```"):
                text = text[len(prefix):-4].strip()
                normalized = True
                break
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        return {"type": "invalid_output", "text": raw}, normalized
    return value, normalized
