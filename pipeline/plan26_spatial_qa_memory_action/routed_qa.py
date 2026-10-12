"""Regenerate the frozen 14 Current / 7 Memory templates from sampled facts."""

import math
from collections import Counter

from routed_facts import EpisodeFacts
from qa_difficulty_contract import CAMERA_DISTANCE_BINS, CAMERA_MOTION_BINS, OBJECT_DISTANCE_BINS, bucket_label, distance_change_label
from qa_semantic_contract import clean_category_options, categories_conflict, VISUAL_CATEGORY_ALLOWLIST
from qa_v1_common import balanced_option_order
from memory_runtime import digest


CURRENT = (
    "camera_to_object_distance_bucket", "camera_to_object_distance_m", "closest_object_category",
    "object_relative_to_camera", "camera_motion_bucket", "camera_path_length_m", "camera_rotation_deg",
    "approached_object_category", "became_visible_category", "disappeared_category", "closer_than_before",
    "closest_category_then_distance", "closest_category_then_relation", "object_category_then_later_visibility",
)
MEMORY = (
    "first_seen_object_category", "first_seen_timestamp_s", "seen_count_by_category", "seen_category_order",
    "first_seen_category_then_timestamp", "single_seen_object_then_closest_category", "single_seen_object_then_closest_distance",
)


class NoSample(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise NoSample(reason)


class QAGenerator:
    def __init__(self, facts: EpisodeFacts, window_seconds=5):
        self.facts = facts
        self.window_seconds = window_seconds
        self.rejections = Counter()
        scene_categories = set(facts.categories.values())
        self.absent_categories = [name for name in sorted(VISUAL_CATEGORY_ALLOWLIST)
                                  if name not in scene_categories and not any(categories_conflict(name, present) for present in scene_categories)]

    def first_choice(self, index):
        first = self.facts.first_categories(index)
        timestamp = self.facts.frames[index]["timestamp"]
        for position, name in first:
            if self.facts.frames[position]["timestamp"] >= timestamp - self.window_seconds:
                continue
            targets = sorted(object_id for object_id in self.facts.prefix(index) if self.facts.categories[object_id] == name)
            later = [other for other_position, other in first if other_position > position]
            answer, choices, _ = clean_category_options(name, later + self.absent_categories)
            if answer is not None and len(choices) >= 4:
                return position, answer, choices[:5], targets
        raise NoSample("no_unambiguous_first_option_with_history")

    def visible_rank(self, index, origin=None):
        ranked = []
        for object_id in sorted(self.facts.frames[index]["visible"]):
            if object_id == origin:
                continue
            distance = self.facts.distance(index, object_id) if origin is None else math.dist(self.facts.center(origin), self.facts.center(object_id))
            ranked.append((distance, object_id))
        return sorted(ranked)

    def nearest(self, index, origin=None):
        ranked = self.visible_rank(index, origin)
        require(ranked, "no_visible_candidate")
        require(len(ranked) == 1 or ranked[1][0] - ranked[0][0] >= 0.15, "nearest_tie")
        return ranked[0]

    def turn(self, index, question, answer, categories=(), frames=(), memory=False, with_frames=False, options=None):
        require(answer is not None, "undefined_answer_or_gray_zone")
        return {"index": index, "question": question, "answer": round(answer, 3) if type(answer) is float else answer,
                "categories": sorted(set(categories)), "evidence_frames": sorted(set(frames)),
                "memory": memory, "with_frames": with_frames, "options": options}

    def make(self, template, index):
        facts = self.facts
        now = facts.frames[index]
        timestamp = now["timestamp"]
        category = facts.categories
        turns, targets = [], []
        if template in CURRENT[:4] or template in CURRENT[-3:]:
            distance, target = self.nearest(index)
            name = category[target]
            targets = [target]
            ref = f"the closest visible {name}"
            first = self.turn(index, "Which visible object category is closest to the camera by 3D center distance?", name)
            if template == "camera_to_object_distance_bucket":
                turns = [self.turn(index, f"What is the camera-to-center distance range for {ref}?", bucket_label(distance, CAMERA_DISTANCE_BINS), options=[row[0] for row in CAMERA_DISTANCE_BINS])]
            elif template == "camera_to_object_distance_m":
                turns = [self.turn(index, f"Estimate the camera-to-center distance to {ref}, in meters.", distance)]
            elif template == "closest_object_category":
                turns = [first]
            elif template == "object_relative_to_camera":
                turns = [self.turn(index, f"Where is {ref} relative to the camera now?", facts.direction(index, target))]
            elif template == "closest_category_then_distance":
                turns = [first, self.turn(index, "What is that selected object's camera-to-center distance range now?", bucket_label(distance, CAMERA_DISTANCE_BINS), options=[row[0] for row in CAMERA_DISTANCE_BINS])]
            elif template == "closest_category_then_relation":
                _, neighbor = self.nearest(index, target)
                targets.append(neighbor)
                turns = [first, self.turn(index, "Which other currently visible object's category is nearest to that object by 3D center distance?", category[neighbor])]
            else:
                later = next((position for position in range(index + 1, len(facts.frames)) if facts.frames[position]["timestamp"] >= timestamp + self.window_seconds), None)
                require(later is not None, "no_later_observation")
                observed = facts.prefix(later)
                require(sum(category[object_id] == name for object_id in observed) == 1, "cross_time_identity_ambiguous")
                turns = [first, self.turn(later, "Is that previously selected object visible now?", "yes" if target in facts.frames[later]["visible"] else "no", options=["yes", "no"])]
        elif template in CURRENT[4:11]:
            before = next((position for position in reversed(range(index)) if facts.frames[position]["timestamp"] <= timestamp - self.window_seconds), None)
            require(before is not None, "insufficient_current_window")
            duration = timestamp - facts.frames[before]["timestamp"]
            if template in CURRENT[4:7]:
                displacement, length, rotation = facts.motion(before, index)
                values = {"camera_motion_bucket": (f"What was the camera's straight-line endpoint displacement range over the last {duration:g} seconds?", bucket_label(displacement, CAMERA_MOTION_BINS)),
                          "camera_path_length_m": (f"Estimate the total camera trajectory length over the last {duration:g} seconds, in meters.", length),
                          "camera_rotation_deg": (f"Estimate the net endpoint camera orientation difference over the last {duration:g} seconds, in degrees.", rotation)}
                question, answer = values[template]
                turns = [self.turn(index, question, answer)]
            elif template in ("became_visible_category", "disappeared_category"):
                old_categories, new_categories = set(facts.counts(before)), set(facts.counts(index))
                changed = new_categories - old_categories if template == "became_visible_category" else old_categories - new_categories
                require(len(changed) == 1, "category_transition_ambiguous")
                verb = "became visible" if template == "became_visible_category" else "disappeared from view"
                turns = [self.turn(index, f"Comparing now with {duration:g} seconds ago, which category {verb}?", next(iter(changed)))]
            else:
                common = facts.frames[before]["visible"] & now["visible"]
                changes = sorted((facts.distance(index, object_id) - facts.distance(before, object_id), object_id) for object_id in common)
                require(changes, "no_common_target")
                if template == "approached_object_category":
                    change, target = changes[0]
                    require(change <= -0.2 and (len(changes) == 1 or changes[1][0] - change >= 0.1), "approach_not_distinct")
                    turns = [self.turn(index, f"Among objects visible both now and {duration:g} seconds ago, which category's object had the greatest camera-to-center distance decrease?", category[target])]
                else:
                    candidates = []
                    for change, target in changes:
                        if distance_change_label(change) is not None:
                            matches = [other for other in common if category[other] == category[target]]
                            if len(matches) == 1:
                                candidates.append((abs(change), target, change))
                    require(candidates, "change_referent_ambiguous_or_gray_zone")
                    _, target, change = max(candidates)
                    turns = [self.turn(index, f"How did the camera-to-center distance to the {category[target]} visible at both times change over the last {duration:g} seconds?", distance_change_label(change))]
                targets = [target]
        else:
            first = facts.first_categories(index)
            require(first, "no_history")
            written_end = math.floor(timestamp / self.window_seconds) * self.window_seconds
            history_indices = [position for position in range(index) if facts.frames[position]["timestamp"] < written_end]
            require(history_indices, "no_written_memory")
            if template in MEMORY[:4]:
                if template in ("first_seen_object_category", "seen_category_order"):
                    _, answer, choices, _ = self.first_choice(index)
                    question = "Which of the listed object categories was seen first in this episode?"
                    turns = [self.turn(index, question, answer, choices, memory=True, options=choices)]
                elif template == "first_seen_timestamp_s":
                    position, name = first[0]
                    require(facts.frames[position]["timestamp"] < timestamp - self.window_seconds, "history_still_current")
                    turns = [self.turn(index, f"At what timestamp did the {name} category first appear, in seconds?", facts.frames[position]["timestamp"], [name], memory=True)]
                else:
                    seen = facts.prefix(index)
                    counts = Counter(category[object_id] for object_id in seen)
                    candidates = sorted((count, name) for name, count in counts.items() if count >= 2)
                    require(candidates, "no_multiple_instances")
                    count, name = candidates[-1]
                    required = {object_id for object_id in seen if category[object_id] == name}
                    historical = set().union(*(facts.frames[position]["visible"] for position in history_indices))
                    require(required <= historical and not required <= now["visible"], "count_not_historical_or_not_written")
                    evidence = [position for position in history_indices if required & facts.frames[position]["visible"]]
                    turns = [self.turn(index, f"How many different {name} objects have been seen so far?", count, [name], evidence, memory=True, with_frames=True)]
                    targets = sorted(required)
            elif template == "first_seen_category_then_timestamp":
                position, name, choices, targets = self.first_choice(index)
                turns = [self.turn(index, "Which of the listed object categories was seen first in this episode?", name, choices, memory=True, options=choices),
                         self.turn(index, "At what timestamp was that category first seen in this episode, in seconds?", facts.frames[position]["timestamp"], [name], memory=True)]
            else:
                counts = Counter(category[object_id] for object_id in facts.prefix(index))
                candidates = sorted(object_id for object_id in facts.prefix(index) - now["visible"] if counts[category[object_id]] == 1)
                choices = []
                for target in candidates:
                    evidence = [position for position in history_indices if target in facts.frames[position]["visible"] and len(facts.frames[position]["visible"]) >= 2]
                    if evidence:
                        position = facts.midpoint(evidence)
                        try:
                            distance, neighbor = self.nearest(position, target)
                            choices.append((target, neighbor, distance, position))
                        except NoSample:
                            continue
                require(choices, "no_unique_historical_target_with_neighbor")
                target, neighbor, distance, position = choices[0]
                targets = [target, neighbor]
                time = facts.frames[position]["timestamp"]
                stem = f"Among the other objects visible at {time:g} seconds with the {category[target]} seen earlier and absent now, "
                question = stem + ("which category was nearest to it by 3D center distance?" if template.endswith("category") else "estimate its 3D center distance to the nearest one, in meters.")
                answer = category[neighbor] if template.endswith("category") else distance
                turns = [self.turn(index, question, answer, [category[target]], [position], memory=True, with_frames=True)]
                turns[0]["time_range"] = [time, time + 1 / facts.sample_fps]
        require(turns, "unimplemented_template")
        category_turns = {
            "closest_object_category": {0}, "approached_object_category": {0},
            "became_visible_category": {0}, "disappeared_category": {0},
            "closest_category_then_distance": {0}, "closest_category_then_relation": {0, 1},
            "object_category_then_later_visibility": {0}, "first_seen_object_category": {0},
            "seen_category_order": {0}, "first_seen_category_then_timestamp": {0},
            "single_seen_object_then_closest_category": {0},
        }
        for turn_index, turn in enumerate(turns):
            if turn_index in category_turns.get(template, set()):
                if template in ("first_seen_object_category", "seen_category_order", "first_seen_category_then_timestamp"):
                    pool = turn["options"]
                else:
                    pool = sorted({category[object_id] for object_id in facts.prefix(turn["index"])}) + self.absent_categories
                clean_answer, choices, _repairs = clean_category_options(turn["answer"], pool)
                require(clean_answer is not None, "filtered_answer_category")
                require(len(choices) >= 4, "insufficient_clean_category_options")
                choices = choices[:5]
                turn["answer"] = clean_answer
                turn["options"] = balanced_option_order(choices, clean_answer,
                    identity_parts=(facts.record["episode_id"], template, index, turn_index))
                if template in ("first_seen_object_category", "seen_category_order", "first_seen_category_then_timestamp"):
                    turn["categories"] = sorted(choices)
            else:
                turn["options"] = None
        semantic_key = digest([{key: (sorted(value) if key == "options" and value else value)
                                for key, value in turn.items() if key in ("index", "question", "answer", "options")} for turn in turns])
        return {"template": template, "family": "memory" if template in MEMORY else "current", "semantic_key": semantic_key,
                "turns": turns, "targets": targets}

    def generate(self):
        samples = []
        for template in CURRENT + MEMORY:
            for start in range(0, len(self.facts.frames), max(1, round(self.window_seconds * self.facts.sample_fps))):
                indices = list(range(start, min(len(self.facts.frames), start + max(1, round(self.window_seconds * self.facts.sample_fps)))))
                candidates = []
                for index in indices:
                    try:
                        candidates.append(self.make(template, index))
                    except (NoSample, ValueError, KeyError, TypeError) as error:
                        self.rejections[(template, str(error))] += 1
                if candidates:
                    positions = [sample["turns"][0]["index"] for sample in candidates]
                    anchor = self.facts.midpoint(positions)
                    def score(sample):
                        position = sample["turns"][0]["index"]
                        target = sample["targets"]
                        if template in MEMORY or template in ("became_visible_category", "disappeared_category") or not target:
                            return (0, -abs(position - anchor), -position)
                        visible = [all(object_id in row["visible"] for object_id in target) for row in self.facts.frames]
                        left = position
                        right = position
                        while left > 0 and visible[left - 1]:
                            left -= 1
                        while right + 1 < len(visible) and visible[right + 1]:
                            right += 1
                        return (min(position - left, right - position), -abs(position - anchor), -position)
                    samples.append(max(candidates, key=score))
        return samples
