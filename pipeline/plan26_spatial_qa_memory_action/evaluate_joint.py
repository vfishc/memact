"""Unified native Current/Memory/Action and causal offline32 evaluation.

Each manifest row specifies root or destination. No teacher events/answers are
passed to the model. A failed stream keeps accepted predictions and raw calls.
"""
import argparse
import json
from pathlib import Path


def main(args):
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from model_inference import NativeModelCallback
    from evaluation_selection import load_selected_episode
    from routed_stream import run_session
    from action_graph import make_executor
    from offline_uniform32 import requests_from_schedule, run_offline
    from streaming_metrics import qa_totals

    args.output.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                    attn_implementation="sdpa", local_files_only=True).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter) if args.adapter else base
    model.eval()
    text = args.manifest.read_text()
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines()]
    for number, row in enumerate(rows):
        root = row.get("root", row.get("destination"))
        data = load_selected_episode(row)
        schedule = []
        for event in data["schedule"]:
            if args.seconds is not None and event["kind"] == "observation" and event["payload"]["timestamp"] > args.seconds:
                break
            schedule.append(event)
        targets = [a for a in data["answers"] if a["schedule_index"] < len(schedule)]
        output = args.output / str(number)
        output.mkdir()
        predictions = []
        callback = NativeModelCallback(model, processor, data["media"], max_new_tokens=args.max_new_tokens,
                                       allow_json_fence=args.allow_json_fence)
        def generate(context):
            value = callback(context)
            (output / "calls.json").write_text(json.dumps(callback.calls))
            (output / "last_context.json").write_text(json.dumps(context))
            return value
        def record(value):
            predictions.append(value)
            (output / "predictions.json").write_text(json.dumps(predictions))
        result = {"root": root, "mode": args.mode, "status": "pass",
                  "test_subset": row.get("test_subset"),
                  "selected_qa": len(targets),
                  "allow_json_fence": args.allow_json_fence,
                  "persistent_cross_call_kv": False, "teacher_memory_supplied": False}
        try:
            executor = make_executor(data, args.max_actions)
            if args.mode == "streaming":
                _, context, manager = run_session(schedule, generate, max_calls=args.max_calls,
                          action_executor=executor, on_prediction=record, max_actions=args.max_actions)
                result["memory_writes"] = len(manager.summaries)
                (output / "context.json").write_text(json.dumps(context))
            else:
                requests = requests_from_schedule(schedule, targets)
                if args.offline_samples is not None:
                    selected = list(dict.fromkeys(r["sample"] for r in requests if r["question_timestamp"] >= args.offline_min_time))[:args.offline_samples]
                    requests = [r for r in requests if r["sample"] in selected]
                    if not requests:
                        raise ValueError("empty requested offline validation subset")
                    indices = {r["schedule_index"] for r in requests}
                    targets = [t for t in targets if t["schedule_index"] in indices]
                    result["subset_policy"] = {"complete_samples": args.offline_samples, "minimum_question_time": args.offline_min_time}
                (output / "requests.private.json").write_text(json.dumps(requests))
                predictions, diagnostics = run_offline(requests, generate, action_executor=executor, max_actions=args.max_actions)
                (output / "offline_results.json").write_text(json.dumps(diagnostics))
                if any(d["status"] != "answered" for d in diagnostics):
                    result["status"] = "failed"
        except Exception as error:
            result.update(status="failed", error=repr(error))
        result["qa"] = qa_totals(targets, predictions)
        result["calls"] = len(callback.calls)
        (output / "predictions.json").write_text(json.dumps(predictions))
        (output / "report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--max-actions", type=int, choices=(1,2,3), default=3)
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("streaming", "offline32"), default="streaming")
    p.add_argument("--seconds", type=float)
    p.add_argument("--max-calls", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--offline-samples", type=int)
    p.add_argument("--offline-min-time", type=float, default=0)
    p.add_argument("--allow-json-fence", action="store_true")
    main(p.parse_args())
