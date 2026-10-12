"""Exact token-mean causal loss with bounded vocabulary-logit intermediates."""

import torch
from torch.utils.checkpoint import checkpoint


def chunked_target_loss(hidden, head, labels, chunk_size=256):
    if hidden.shape[0] != 1 or labels.shape[0] != 1 or chunk_size < 1:
        raise ValueError("batch1 and positive chunk size required")
    positions = (labels[0] != -100).nonzero().flatten()
    if not len(positions) or positions[0] == 0:
        raise ValueError("causal supervised targets require preceding positions")

    def summed_loss(states, targets):
        return torch.nn.functional.cross_entropy(head(states).float(), targets, reduction="sum")

    losses = []
    for offset in range(0, len(positions), chunk_size):
        selected = positions[offset:offset + chunk_size]
        losses.append(checkpoint(summed_loss, hidden[0, selected - 1], labels[0, selected], use_reentrant=False))
    return torch.stack(losses).sum() / len(positions)
