"""P26 independent-question causal uniform32 baseline and native inference CLI."""

import argparse
import copy
import json
import math
from pathlib import Path

from episode_evaluation.protocol import select_frames
from routed_stream import check_public
from streaming_metrics import qa_totals


INSTRUCTION = (
    "Answer using the provided chronological video frames and timestamps. "
    'Output exactly one JSON object: {"type":"answer","answer":value}. '
    "If options exist use one complete category option, not a letter. "
    "For numeric questions output a JSON number; otherwise a short answer. "
    "Earlier answers in this question chain are your predictions. "
    "Do not request memory or actions."
)


def requests_from_schedule(schedule, references, max_frames=32):
    if max_frames != 32:
        raise ValueError("formal offline baseline requires uniform32")
    links = {}
    for row in references:
        index = row["schedule_index"]
        if index in links:
            raise ValueError("duplicate reference index")
        links[index] = {"sample": row["sample"], "turn": row["turn"]}
    observations, requests, handles, turns = [], [], set(), {}
    for index, event in enumerate(schedule):
        payload = event["payload"]
        if event["kind"] == "observation":
            timestamp = payload["timestamp"]
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("invalid timestamp")
            if payload["frame"] in handles or (observations and timestamp <= observations[-1]["timestamp"]):
                raise ValueError("nonchronological/duplicate observation")
            check_public(payload)
            observations.append(copy.deepcopy(payload))
            handles.add(payload["frame"])
        elif event["kind"] == "question":
            if index not in links or not observations:
                raise ValueError("question missing reference or causal observation")
            link = links[index]
            sample, turn = link["sample"], link["turn"]
            if type(turn) is not int or turn not in (0, 1) or turn != len(turns.get(sample, [])):
                raise ValueError("incomplete or invalid two-turn chain")
            question = {"mode": "[QA]", "question": payload["question"]}
            for key in ('answer_type','unit'):
                if key in payload:question[key]=payload[key]
            if payload.get("options") is not None:
                question["options"] = copy.deepcopy(payload["options"])
            check_public(question)
            request = {"schedule_index": index, **link, "question": question,
                       "frames": select_frames(observations, policy="uniform", max_frames=max_frames),
                       "causal_observations": len(observations), "question_timestamp": observations[-1]["timestamp"],
                       "previous": list(turns.get(sample, []))}
            requests.append(request)
            turns.setdefault(sample, []).append(index)
    if {row["schedule_index"] for row in requests} != links.keys():
        raise ValueError("reference does not match schedule questions")
    return requests


def run_offline(requests, generate, on_result=None, action_executor=None, max_actions=3):
    completed = {}
    results, predictions = [], []
    questions = {row["schedule_index"]: row["question"] for row in requests}
    for request in requests:
        index = request["schedule_index"]
        row = {"schedule_index": index, "selected_frames": [frame["frame"] for frame in request["frames"]]}
        if any(previous not in completed for previous in request["previous"]):
            row.update(status="predecessor_failed")
        else:
            instruction=INSTRUCTION
            if request['question'].get('answer_type'):
                from qa_eight_contract import ANSWER_INSTRUCTION
                instruction=('Answer using the provided chronological frames and timestamps. '+ANSWER_INSTRUCTION+
                             ' Earlier answers in this question chain are your predictions. Do not request tools.')
            context = [{"role": "system", "content": {"instruction": instruction}}]
            context += [{"role": "user", "content": copy.deepcopy(frame)} for frame in request["frames"]]
            for previous in request["previous"]:
                context.append({"role": "user", "content": copy.deepcopy(questions[previous])})
                context.append({"role": "assistant", "content": copy.deepcopy(completed[previous])})
            context.append({"role": "user", "content": copy.deepcopy(request["question"])})
            try:
                if action_executor is not None:
                    from action_graph import COMMANDS
                    instruction = ('Answer using the supplied chronological frames. Return exactly one JSON object: '
                        '{"type":"answer","answer":value}. '
                        'for labeled options use an uppercase option letter, and for numerical questions use '
                        'a JSON number in the requested unit. You may acquire views using type=action, '
                        'name and arguments from this menu: ' + json.dumps(COMMANDS))
                    context[0]['content']['instruction'] = instruction
                    action_executor.begin_question(index, request['question'])
                    for attempt in range(max_actions + 1):
                        output = generate(copy.deepcopy(context))
                        check_public(output)
                        if output.get('type') == 'answer':
                            break
                        if output.get('type') != 'action' or attempt == max_actions:
                            raise ValueError('invalid active offline output or exhausted action budget')
                        result = action_executor(output)
                        check_public(result)
                        context.extend([{'role':'assistant','content':output},
                                        {'role':'tool','content':result},
                                        {'role':'user','content':{'mode':'[QA_CONTINUE]',
                                         'remaining_actions':max_actions-attempt-1,
                                         'instruction':'No camera attempts remain. Return exactly {"type":"answer","answer":value} now.' if attempt+1 == max_actions else 'Continue or answer.'}}])
                    row['action_attempts'] = action_executor.attempts
                else:
                    output = generate(context)
                if not isinstance(output, dict) or set(output) != {"type", "answer"} or output["type"] != "answer":
                    raise ValueError("offline model must output answer only")
                answer = output["answer"]
                if type(answer) not in (str, int, float) or (type(answer) in (int, float) and not math.isfinite(answer)):
                    raise ValueError("invalid answer scalar")
                completed[index] = copy.deepcopy(output)
                predictions.append({"schedule_index": index, "answer": answer})
                row.update(status="answered", answer=answer)
            except Exception as error:
                row.update(status="failed", error=repr(error))
        results.append(row)
        if on_result:
            on_result(row)
    return predictions, results


def read_rows(path):
    return json.loads(path.read_text()) if path.suffix == ".json" else [json.loads(line) for line in path.read_text().splitlines()]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    action = (args.episode / "events.json").exists()
    expected = read_rows(args.episode / ("answers.json" if action else "evaluation_answers.private.jsonl"))
    schedule = read_rows(args.episode / ("schedule.json" if action else "evaluation_schedule.jsonl"))
    requests = requests_from_schedule(schedule, expected)
    (args.output / "requests.private.json").write_text(json.dumps(requests))
    metadata = {"mode": "offline_uniform32", "questions": len(requests), "max_frames": 32,
                "action_mode": "pre_action_answer_only" if action else None, "native_inference_run": bool(args.model)}
    (args.output / "protocol.json").write_text(json.dumps(metadata))
    if not args.model:
        return
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from model_inference import NativeModelCallback

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True).to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    media = json.loads((args.episode / ("media.json" if action else "private_media.json")).read_text())
    callback = NativeModelCallback(model, processor, media, max_new_tokens=args.max_new_tokens)
    with (args.output / "results.jsonl").open("x") as handle:
        def record(row):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
        predictions, results = run_offline(requests, callback, record)
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    (args.output / "model_calls.json").write_text(json.dumps(callback.calls))
    (args.output / "metrics.json").write_text(json.dumps(qa_totals(expected, predictions), allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    main(parser.parse_args())
