"""Assistant-only supervision masks for the Qwen chat template."""
import torch

def assistant_labels(input_ids, tokenizer):
    start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
    values = input_ids[0].tolist()
    labels = torch.full_like(input_ids, -100)
    targets = 0
    for index, value in enumerate(values):
        if value != start or values[index + 1:index + 1 + len(assistant_prefix)] != assistant_prefix:
            continue
        begin = index + 1 + len(assistant_prefix)
        finish = values.index(end, begin) + 1
        labels[0, begin:finish] = input_ids[0, begin:finish]
        targets += finish - begin
    if not targets:
        raise ValueError("no assistant labels; chat template mismatch")
    return labels, targets
