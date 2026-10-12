"""Chronological teacher traces and replay validation for a shared session."""

import copy
import json
import math

from memory_runtime import MemoryManager, digest


SCHEMA = "p26_routed_fixture_v2"
SYSTEM_INSTRUCTION = (
    "Consume the observation stream and keep the shared history. Output one JSON object when prompted. "
    "[MEMORY_WRITE]: output type=memory_write, observations=[{frame,objects:[{category,count}]}], relations=[]. "
    "Include each supplied window frame exactly once, including empty objects lists. Counts are simultaneous per-frame counts. "
    "Optional relations have frame, subject={category,region}, predicate, object={category,region}; regions use a 3x3 image grid "
    "(top_left,top_center,top_right,middle_left,center,middle_right,bottom_left,bottom_center,bottom_right), "
    "predicates are left_of_in_image,right_of_in_image,above_in_image,below_in_image. "
    "[QA] and [QA_CONTINUE]: output {type:answer,answer:value} or "
    "{type:use_memory,categories:[category],with_frames:boolean,time_range:[start,end],cursor:token}. "
    "time_range and cursor are optional; empty categories searches all recorded categories. "
    "Use only a cursor returned for the same query. with_frames requests historical RGB; false requests records only. "
    "Retrieve images only when records are insufficient: at most 2 images per call and 4 per question. "
    "Already returned frames remain in shared context and are not sent again. Stop retrieval once evidence is sufficient. "
    "If options exist answer with one complete category option, not a letter. Otherwise answer the requested numeric value "
    "or short direction, distance-range or yes/no value. Never write memory during QA."
)
FORBIDDEN = {"pose", "location", "world_center", "rotation_euler_deg", "object_id", "target_id", "episode_id",
             "template_id", "reward", "oracle_action", "media_ref", "source_frame", "path", "K", "c2w", "w2c"}

ACTION_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION + (
    " During QA you may also output {type:action,name:command,arguments:parameters}. "
    "Use only the provided action menu. A successful action returns a new observed image; "
    "answer using that image and shared history. Do not write memory during QA."
)


def validate_answer(output):
    if not isinstance(output, dict) or set(output) != {"type", "answer"} or output["type"] != "answer":
        raise ValueError("invalid answer schema")
    value = output["answer"]
    if type(value) not in (str, int, float) or (type(value) in (int, float) and not math.isfinite(value)):
        raise ValueError("invalid answer scalar")


def check_public(value):
    if isinstance(value, dict):
        if value.keys() & FORBIDDEN:
            raise ValueError("private fields in public payload")
        for item in value.values():
            check_public(item)
    elif isinstance(value, list):
        for item in value:
            check_public(item)
    elif isinstance(value, str) and (value.startswith(("/data/", "/mnt/")) or "file://" in value):
        raise ValueError("private media path in public payload")


def evidence_satisfied(facts, sample, turn, images):
    if sample["template"] == "seen_count_by_category":
        visible = set().union(*(row["visible"] for row in facts.frames if row["frame"] in images))
        return set(sample["targets"]) <= visible
    return {facts.frames[position]["frame"] for position in turn["evidence_frames"]} <= images


def build_stream(facts, samples, page_size=8, frame_budget=2, max_calls=64):
    manager = MemoryManager(page_size, frame_budget)
    events, audit, rejected = [], [], []
    scheduled = {}
    for sample_index, sample in enumerate(samples):
        for turn_index, turn in enumerate(sample["turns"]):
            scheduled.setdefault(turn["index"], []).append((sample_index, turn_index, sample, turn))

    def emit(role, kind, payload):
        check_public(payload)
        events.append({"seq": len(events), "prefix_hash": digest(events[-1]) if events else None,
                       "role": role, "kind": kind, "loss": role == "assistant", "payload": payload})

    window_start = 0.0
    pending = []
    emit("system", "system_prompt", {"instruction": SYSTEM_INSTRUCTION})

    def flush(end):
        nonlocal window_start, pending
        if pending:
            target = facts.teacher(pending)
            emit("user", "write_prompt", {"mode": "[MEMORY_WRITE]", "window": [window_start, end],
                                           "frames": [facts.frames[index]["frame"] for index in pending]})
            emit("assistant", "memory_write", target)
            emit("tool", "memory_ack", manager.write(target, window_start, end))
            pending = []
        window_start = end

    for index, row in enumerate(facts.frames):
        while row["timestamp"] >= window_start + 5:
            flush(window_start + 5)
        emit("user", "observation", {"frame": row["frame"], "timestamp": row["timestamp"]})
        manager.observe(row["frame"], row["timestamp"])
        pending.append(index)
        for sample_index, turn_index, sample, turn in scheduled.get(index, []):
            manager.begin_question()
            trace = []
            if turn["memory"]:
                query = {"type": "use_memory", "categories": turn["categories"], "with_frames": turn["with_frames"]}
                if "time_range" in turn:
                    query["time_range"] = turn["time_range"]
                images, records = set(), set()
                for _call in range(max_calls):
                    result = manager.query(query)
                    trace.extend([("assistant", "memory_query", copy.deepcopy(query)), ("tool", "memory_result", result)])
                    images.update(item["frame"] for item in result["frames"])
                    records.update(item["frame"] for item in result["records"])
                    if records and evidence_satisfied(facts, sample, turn, images):
                        break
                    if not result["next_cursor"] or (turn["with_frames"] and result["remaining_frame_budget"] == 0):
                        break
                    query = {**query, "cursor": result["next_cursor"]}
                else:
                    manager.question_index -= 1
                    rejected.append({"sample": sample_index, "reason": "call_budget_exhausted"})
                    continue
                if not evidence_satisfied(facts, sample, turn, images) or not records:
                    manager.question_index -= 1
                    rejected.append({"sample": sample_index, "reason": "insufficient_bounded_memory_evidence"})
                    continue
            begin = len(events)
            prompt = {"mode": "[QA]", "question": turn["question"]}
            if turn["options"] is not None:
                prompt["options"] = turn["options"]
            emit("user", "question", prompt)
            for role, kind, payload in trace:
                emit(role, kind, payload)
                if kind == "memory_result":
                    emit("user", "continue_prompt", {"mode": "[QA_CONTINUE]"})
            emit("assistant", "answer", {"type": "answer", "answer": turn["answer"]})
            audit.append({"sample": sample_index, "turn": turn_index, "template": sample["template"],
                          "family": sample["family"], "question_seq": begin, "answer_seq": len(events) - 1,
                          "targets": sample["targets"], "required_frames": turn["evidence_frames"]})
    flush(facts.end)
    accepted = {row["sample"] for row in audit}
    for sample_index in accepted:
        if sum(row["sample"] == sample_index for row in audit) != len(samples[sample_index]["turns"]):
            raise ValueError("partial RefChain emitted")
    if manager.replay().summaries != manager.summaries:
        raise ValueError("memory replay mismatch")
    return events, audit, rejected


def validate_stream(events, media, page_size=8, frame_budget=2):
    from qa_eight_contract import PURE_INSTRUCTION, full_instruction, valid_answer
    manager = MemoryManager(page_size, frame_budget)
    state = "stream"
    expected = None
    window = None
    answers = 0
    pure = events[0]['payload'].get('instruction') == PURE_INSTRUCTION if events else False
    for index, event in enumerate(events):
        if event["seq"] != index or event["prefix_hash"] != (digest(events[index - 1]) if index else None):
            raise ValueError("broken shared prefix")
        if type(event["loss"]) is not bool or event["loss"] != (event["role"] == "assistant"):
            raise ValueError("invalid loss mask")
        payload, kind = event["payload"], event["kind"]
        if pure and kind not in ('system_prompt','observation','question','answer'):
            raise ValueError('QA-only stream contains a tool or memory event')
        check_public(payload)
        roles = {"system_prompt": "system", "observation": "user", "write_prompt": "user", "memory_write": "assistant", "memory_ack": "tool",
                 "question": "user", "memory_query": "assistant", "memory_result": "tool", "continue_prompt": "user", "answer": "assistant",
                 "action": "assistant", "action_result": "tool"}
        if kind not in roles or event["role"] != roles[kind]:
            raise ValueError("invalid event role")
        if kind == "system_prompt":
            if index != 0 or payload.get("instruction") not in (SYSTEM_INSTRUCTION, ACTION_SYSTEM_INSTRUCTION, PURE_INSTRUCTION, full_instruction(SYSTEM_INSTRUCTION), full_instruction(ACTION_SYSTEM_INSTRUCTION)):
                raise ValueError("unexpected system reset or instructions")
        elif kind == "observation":
            if state != "stream" or payload["frame"] not in media:
                raise ValueError("observation state or media missing")
            manager.observe(payload["frame"], payload["timestamp"])
        elif kind == "write_prompt":
            if state != "stream" or payload["mode"] != "[MEMORY_WRITE]":
                raise ValueError("QA cannot write memory")
            window = payload["window"]
            if payload["frames"] != [frame for frame, time in manager.frames.items() if window[0] <= time < window[1]]:
                raise ValueError("write prompt frame mismatch")
            state = "write"
        elif kind == "memory_write":
            if state != "write":
                raise ValueError("unexpected memory write")
            expected = manager.write(payload, *window)
            state = "ack"
        elif kind == "memory_ack":
            if state != "ack" or payload != expected:
                raise ValueError("invalid memory acknowledgment")
            state = "stream"
        elif kind == "question":
            if state != "stream" or payload["mode"] != "[QA]":
                raise ValueError("invalid question state")
            manager.begin_question()
            state = "qa"
        elif kind == "memory_query":
            if state != "qa":
                raise ValueError("unexpected query")
            expected = manager.query(payload)
            state = "result"
        elif kind == "action":
            if state != "qa" or payload.get("type") != "action" or set(payload) != {"type", "name", "arguments"}:
                raise ValueError("unexpected action")
            expected = payload
            state = "action_result"
        elif kind == "action_result":
            if state != "action_result" or payload.get("action") != expected or payload.get("frame") not in media:
                raise ValueError("action result does not match command/media")
            state = "continue"
        elif kind == "memory_result":
            if state != "result" or payload != expected:
                raise ValueError("tool result differs from actual manager")
            for row in payload["frames"]:
                if row["frame"] not in media or row["frame"] not in manager.frames:
                    raise ValueError("future or missing image")
            state = "continue"
        elif kind == "continue_prompt":
            if state != "continue" or payload["mode"] != "[QA_CONTINUE]":
                raise ValueError("unexpected continuation")
            state = "qa"
        elif kind == "answer":
            if state != "qa" or payload.get("type") != "answer":
                raise ValueError("unexpected answer")
            validate_answer(payload)
            answers += 1
            state = "stream"
    if state != "stream":
        raise ValueError("incomplete tool trace")
    return {"events": len(events), "observations": len(manager.frames), "memory_writes": len(manager.summaries), "answers": answers}


def evaluation_schedule(events, audit):
    questions = {row["question_seq"]: row for row in audit}
    schedule, expected = [], []
    for event in events:
        if event["kind"] in ("system_prompt", "observation", "write_prompt", "question"):
            schedule.append({"kind": event["kind"], "payload": copy.deepcopy(event["payload"])})
            if event["kind"] == "question":
                row = questions[event["seq"]]
                expected.append({"schedule_index": len(schedule) - 1, "template": row["template"],
                                 "sample": row["sample"], "turn": row["turn"],
                                 "answer": events[row["answer_seq"]]["payload"]["answer"]})
    return schedule, expected


def run_session(schedule, generate, page_size=8, frame_budget=2, max_calls=64, write_attempts=2, action_executor=None, on_prediction=None, max_actions=3):
    """generate receives one growing public context; no teacher targets are loaded."""
    if max_calls < 0 or max_actions < 0:
        raise ValueError("negative interaction budget")
    manager = MemoryManager(page_size, frame_budget)
    from qa_eight_contract import PURE_INSTRUCTION
    answer_only = bool(schedule and schedule[0]['payload'].get('instruction') == PURE_INSTRUCTION and action_executor is None)
    context, predictions = [], []
    for index, event in enumerate(schedule):
        payload, kind = event["payload"], event["kind"]
        if answer_only and kind not in ('system_prompt','observation','question'):
            raise ValueError('QA-only schedule contains a tool or memory event')
        context.append({"role": "system" if kind == "system_prompt" else "user", "content": copy.deepcopy(payload)})
        if kind == "system_prompt":
            if index != 0:
                raise ValueError("unexpected system reset")
        elif kind == "observation":
            manager.observe(payload["frame"], payload["timestamp"])
        elif kind == "write_prompt":
            for _attempt in range(write_attempts):
                output = generate(copy.deepcopy(context))
                check_public(output)
                context.append({"role": "assistant", "content": output})
                try:
                    acknowledgment = manager.write(output, *payload["window"])
                except ValueError as error:
                    context.append({"role": "tool", "content": {"status": "invalid_memory", "error": str(error)}})
                    context.append({"role": "user", "content": copy.deepcopy(payload)})
                    continue
                context.append({"role": "tool", "content": acknowledgment})
                break
            else:
                raise ValueError("memory repair attempts exhausted")
        elif kind == "question":
            manager.begin_question()
            if action_executor is not None and hasattr(action_executor, "begin_question"):
                action_executor.begin_question(index, payload)
            memory_calls = action_calls = 0
            force_answer = False
            for _call in range(max_calls + max_actions + 2):
                output = generate(copy.deepcopy(context))
                check_public(output)
                context.append({"role": "assistant", "content": output})
                if output.get("type") == "answer":
                    validate_answer(output)
                    predictions.append({"schedule_index": index, "answer": output["answer"]})
                    if on_prediction is not None:
                        on_prediction(copy.deepcopy(predictions[-1]))
                    break
                if answer_only or force_answer or output.get("type") not in ("use_memory", "action"):
                    raise ValueError("invalid QA output or exhausted call budget")
                try:
                    if output.get("type") == "action":
                        if action_executor is None:
                            raise RuntimeError("Action requested without an executable graph")
                        if action_calls >= max_actions:
                            raise ValueError("action budget exhausted")
                        action_calls += 1
                        result = action_executor(output)
                        force_answer = action_calls == max_actions
                        check_public(result)
                    else:
                        if memory_calls >= max_calls:
                            result = {"status": "memory_budget_exhausted"}
                            force_answer = True
                        else:
                            memory_calls += 1
                            result = manager.query(output)
                except ValueError as error:
                    result = {"status": "invalid_query", "error": str(error)}
                context.append({"role": "tool", "content": result})
                context.append({"role": "user", "content": {"mode": "[QA_CONTINUE]",
                    "remaining_actions": max_actions-action_calls,
                    "remaining_memory_calls": max_calls-memory_calls,
                    "instruction": "Return a final answer now." if force_answer else "Continue or answer."}})
            else:
                raise ValueError("missing answer")
        else:
            raise ValueError("unknown schedule event")
    return predictions, context, manager
