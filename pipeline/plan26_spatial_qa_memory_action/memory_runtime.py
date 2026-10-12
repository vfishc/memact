"""GT-free structured memory storage and causal text/image retrieval."""

import copy
import hashlib
import json
import math


REGIONS = tuple(
    f"{vertical}_{horizontal}" if (vertical, horizontal) != ("middle", "center") else "center"
    for vertical in ("top", "middle", "bottom")
    for horizontal in ("left", "center", "right")
)
PREDICATES = ("left_of_in_image", "right_of_in_image", "above_in_image", "below_in_image")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exact_keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise ValueError("invalid fields")


def category_valid(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip().lower()


class MemoryManager:
    def __init__(self, page_size=8, frame_budget=2, question_frame_budget=4):
        if page_size < 1 or not 1 <= frame_budget <= 2 or not 1 <= question_frame_budget <= 4:
            raise ValueError("positive budgets required")
        self.page_size = page_size
        self.frame_budget = frame_budget
        self.question_frame_budget = question_frame_budget
        self.returned_frames = set()
        self.question_index = 0
        self.frames = {}
        self.summaries = []
        self.receipts = {}
        self.cursors = {}
        self.last_end = 0.0

    def begin_question(self):
        self.question_index += 1
        self.returned_frames.clear()
        self.cursors.clear()

    def observe(self, frame, timestamp):
        if not isinstance(frame, str) or not frame.startswith("f") or not frame[1:].isdigit():
            raise ValueError("invalid frame handle")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp):
            raise ValueError("invalid timestamp")
        if timestamp < 0 or frame in self.frames or (self.frames and timestamp <= next(reversed(self.frames.values()))):
            raise ValueError("nonmonotonic observation")
        self.frames[frame] = float(timestamp)

    def write(self, payload, start, end):
        exact_keys(payload, ("type", "observations", "relations"))
        if payload["type"] != "memory_write" or not isinstance(payload["observations"], list) or not isinstance(payload["relations"], list):
            raise ValueError("invalid memory schema")
        signature = digest(payload)
        key = (start, end)
        if key in self.receipts:
            if self.receipts[key] != signature:
                raise ValueError("conflicting retry")
            return {"status": "already_stored", "window": [start, end]}
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in key) or start != self.last_end or end <= start:
            raise ValueError("invalid window")
        expected = [frame for frame, timestamp in self.frames.items() if start <= timestamp < end]
        if not expected or any(not isinstance(row, dict) for row in payload["observations"]) or [row.get("frame") for row in payload["observations"]] != expected:
            raise ValueError("missing, duplicate or out-of-window records")
        categories = {}
        for row in payload["observations"]:
            exact_keys(row, ("frame", "objects"))
            if not isinstance(row["objects"], list):
                raise ValueError("invalid objects")
            seen = set()
            for item in row["objects"]:
                exact_keys(item, ("category", "count"))
                if not category_valid(item["category"]) or type(item["count"]) is not int or item["count"] <= 0 or item["category"] in seen:
                    raise ValueError("invalid category/count")
                seen.add(item["category"])
            categories[row["frame"]] = seen
        for relation in payload["relations"]:
            exact_keys(relation, ("frame", "subject", "predicate", "object"))
            if relation["frame"] not in categories or relation["predicate"] not in PREDICATES:
                raise ValueError("invalid relation")
            for role in ("subject", "object"):
                exact_keys(relation[role], ("category", "region"))
                if relation[role]["category"] not in categories[relation["frame"]] or relation[role]["region"] not in REGIONS:
                    raise ValueError("invalid relation reference")
        self.summaries.append({"window": [start, end], "payload": copy.deepcopy(payload)})
        self.receipts[key] = signature
        self.last_end = end
        return {"status": "stored", "window": [start, end]}

    def query(self, request):
        exact_keys(request, ("type", "categories", "with_frames"), ("time_range", "cursor"))
        if request["type"] != "use_memory" or not isinstance(request["categories"], list) or type(request["with_frames"]) is not bool:
            raise ValueError("invalid query")
        if any(not category_valid(category) for category in request["categories"]):
            raise ValueError("invalid query category")
        bounds = request.get("time_range")
        if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2 or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds) or not 0 <= bounds[0] < bounds[1]):
            raise ValueError("invalid query time")
        fingerprint = digest({key: value for key, value in request.items() if key != "cursor"})
        cursor = request.get("cursor")
        if cursor is not None:
            if cursor not in self.cursors or self.cursors[cursor]["fingerprint"] != fingerprint:
                raise ValueError("invalid or query-mismatched cursor")
            state = self.cursors[cursor]
        else:
            selected = set(request["categories"])
            records = []
            all_records = [row for summary in self.summaries for row in summary["payload"]["observations"]]
            relations = [row for summary in self.summaries for row in summary["payload"]["relations"]]
            order = {frame: index for index, frame in enumerate(self.frames)}
            for row in all_records:
                timestamp = self.frames[row["frame"]]
                if bounds is not None and not bounds[0] <= timestamp < bounds[1]:
                    continue
                present = {item["category"] for item in row["objects"]}
                if selected and not selected & present:
                    continue
                records.append({**copy.deepcopy(row), "timestamp": timestamp,
                                "co_visible": bool(selected) and selected <= present})
            runs = []
            for row in records:
                if not runs or order[row["frame"]] != order[runs[-1][-1]["frame"]] + 1:
                    runs.append([])
                runs[-1].append(row)
            representatives = []
            for run in runs:
                candidates = [row for row in run if row["co_visible"]] or run
                midpoint = (candidates[0]["timestamp"] + candidates[-1]["timestamp"]) / 2
                representatives.append(min(candidates, key=lambda row: (abs(row["timestamp"] - midpoint), row["timestamp"]))["frame"])
            selected_counts = lambda row: sum(item["count"] for item in row["objects"] if not selected or item["category"] in selected)
            strongest = sorted(records, key=lambda row: (-selected_counts(row), row["timestamp"]))
            priority = list(dict.fromkeys(representatives + [row["frame"] for row in strongest]))
            state = {"fingerprint": fingerprint, "question_index": self.question_index, "records": records, "relations": relations,
                     "priority": priority if request["with_frames"] else [], "record_offset": 0,
                     "frame_offset": 0, "snapshot_windows": len(self.summaries)}
        record_end = state["record_offset"] + self.page_size
        page = state["records"][state["record_offset"]:record_end]
        remaining = self.question_frame_budget - len(self.returned_frames)
        available = [frame for frame in state["priority"] if frame not in self.returned_frames]
        already_returned = [frame for frame in state["priority"] if frame in self.returned_frames]
        handles = available[:min(self.frame_budget, remaining)]
        self.returned_frames.update(handles)
        returned = {row["frame"] for row in page} | set(handles)
        next_state = {**state, "record_offset": min(record_end, len(state["records"])),
                      "frame_offset": state["frame_offset"] + len(handles)}
        more = next_state["record_offset"] < len(state["records"]) or (len(available) > len(handles) and len(self.returned_frames) < self.question_frame_budget)
        token = digest(next_state)[:24] if more else None
        if token:
            self.cursors[token] = next_state
        return {"records": copy.deepcopy(page),
                "relations": [copy.deepcopy(row) for row in state["relations"] if row["frame"] in returned],
                "frames": [{"frame": frame, "timestamp": self.frames[frame]} for frame in handles],
                "already_returned_frames": already_returned,
                "remaining_frame_budget": self.question_frame_budget - len(self.returned_frames),
                "next_cursor": token, "truncated": more, "total_records": len(state["records"]),
                "total_frames": len(state["priority"]), "snapshot_windows": state["snapshot_windows"]}

    def replay(self):
        restored = MemoryManager(self.page_size, self.frame_budget, self.question_frame_budget)
        for frame, timestamp in self.frames.items():
            restored.observe(frame, timestamp)
        for summary in self.summaries:
            restored.write(summary["payload"], *summary["window"])
        return restored
