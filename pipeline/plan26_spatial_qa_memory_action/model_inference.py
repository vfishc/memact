"""Native model callback for P26 shared-context router; no teacher outputs loaded."""

import json

from evaluate_routed_memory_qa import FrameResolver
from joint_episode import materialize, check_visual_batch
from native_json import parse_native_json


class NativeModelCallback:
    def __init__(self, model, processor, media, max_new_tokens=1024, max_pixels=200704, allow_json_fence=False):
        self.model = model
        self.processor = processor
        self.resolver = FrameResolver(media)
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.calls = []
        self.allow_json_fence = allow_json_fence

    def __call__(self, context):
        import torch

        messages, receipts = materialize(context, self.resolver.media, self.max_pixels)
        batch = self.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt",
                                                   add_generation_prompt=True, enable_thinking=False)
        vision = check_visual_batch(batch, receipts, self.processor)
        batch = {key: value.to(self.model.device) if hasattr(value, "to") else value for key, value in batch.items()}
        self.model.eval()
        with torch.inference_mode():
            generated = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
        text = self.processor.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        result, normalized = parse_native_json(text, self.allow_json_fence)
        self.calls.append({"input_tokens": batch["input_ids"].shape[1], "raw_output": text,
                           "json_fence_normalized": normalized,
                           "vision_input": vision,
                           "prefix_policy": "full_context_recompute; intra_call_KV"})
        return result




