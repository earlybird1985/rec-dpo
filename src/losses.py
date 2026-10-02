"""Reference Bradley-Terry DPO loss and length-normalized ranking.

`trl.DPOTrainer` applies this loss. This module exists so the math is tested
without downloading a model.
"""
from __future__ import annotations

import math


def softplus(x: float) -> float:
    """log(1 + exp(x)), stable for large |x|."""
    if x > 0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def dpo_sigmoid_loss(
    policy_chosen_logps: float,
    policy_rejected_logps: float,
    ref_chosen_logps: float,
    ref_rejected_logps: float,
    beta: float = 0.1,
) -> float:
    """-log σ(β [(log π_w - log π_l) - (log π_ref_w - log π_ref_l)]).

    A zero policy margin against a zero reference margin returns log(2).
    """
    policy_margin = policy_chosen_logps - policy_rejected_logps
    ref_margin = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_margin - ref_margin)
    return softplus(-logits)


def rank_by_logprobs(
    item_ids: list[int],
    sum_logprobs: list[float],
    token_counts: list[int],
    length_normalize: bool,
) -> list[int]:
    """Higher score first. Ties break toward the smaller item id.

    Sum of token log-probs (all ≤ 0) prefers short titles. Dividing by the
    title token count lets a longer title with a better per-token log-prob win.
    """
    scored = []
    for item, total, count in zip(item_ids, sum_logprobs, token_counts):
        count = int(count)
        if length_normalize:
            score = (float(total) / count) if count else float("-inf")
        else:
            score = float(total)
        scored.append((int(item), score))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [item for item, _ in scored]
