"""Length-normalized title log-probs under the shared history prompt (A5 §6)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from losses import rank_by_logprobs
from pairs import as_int_list, title_map
from prompts import history_prompt, pool_labels


def _flat_ids(ids) -> list[int]:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def dpo_token_split(tokenizer, user_prompt: str, completion: str) -> tuple[list[int], list[int]]:
    """Match trl's conversational split: prompt with the generation header, completion = the rest.

    The completion is the title plus the end-of-turn token when the chat template
    makes the prompt a prefix of the full conversation. It is not the user-turn header.
    """
    prompt_messages = [{"role": "user", "content": user_prompt}]
    prompt_ids = _flat_ids(
        tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True)
    )
    full_ids = _flat_ids(
        tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": completion}],
            tokenize=True,
        )
    )
    if len(full_ids) > len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return prompt_ids, full_ids[len(prompt_ids) :]
    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not comp_ids:
        unk = tokenizer.unk_token_id
        comp_ids = [int(unk if unk is not None else 0)]
    return prompt_ids, [int(x) for x in comp_ids]


def _encode_completion(tokenizer, user_prompt: str, completion: str, max_length: int | None):
    prompt_ids, comp_ids = dpo_token_split(tokenizer, user_prompt, completion)
    ids = list(prompt_ids) + list(comp_ids)
    prompt_len = len(prompt_ids)
    if max_length and len(ids) > max_length:
        keep_prompt = max_length - len(comp_ids)
        if keep_prompt < 1:
            ids = list(comp_ids)[-max_length:]
            prompt_len = 0
            comp_ids = ids
        else:
            ids = list(prompt_ids)[-keep_prompt:] + list(comp_ids)
            prompt_len = keep_prompt
    return ids, prompt_len, len(comp_ids)


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.inference_mode()
def completion_logprobs(
    model,
    tokenizer,
    user_prompt: str,
    completions: list[str],
    max_length: int | None = None,
) -> list[tuple[float, int]]:
    """(sum of title-token log-probs, token count) for each completion.

    The chat-template header is context, not part of the count. Empty
    completions contribute a single unk token so the score is defined.
    """
    if not completions:
        return []
    device = model_device(model)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    encoded = [_encode_completion(tokenizer, user_prompt, text, max_length) for text in completions]
    seqs = [ids for ids, _, _ in encoded]
    width = max(len(ids) for ids in seqs)
    input_ids = torch.full((len(seqs), width), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    for i, ids in enumerate(seqs):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, : len(ids)] = 1
    logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = input_ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    out = []
    for i, (_, prompt_len, n_comp) in enumerate(encoded):
        # log-prob of input index p lives at token_lp index p-1
        lp_start = prompt_len - 1
        lp_end = prompt_len + n_comp - 1
        if lp_start < 0:
            lp_start = 0
        sl = token_lp[i, lp_start:lp_end]
        out.append((float(sl.sum().item()) if sl.numel() else 0.0, int(sl.numel())))
    return out


def pair_scores(
    model,
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int | None,
) -> dict:
    (chosen_sum, chosen_n), (rejected_sum, rejected_n) = completion_logprobs(
        model, tokenizer, prompt, [chosen, rejected], max_length
    )
    chosen_norm = chosen_sum / chosen_n if chosen_n else float("-inf")
    rejected_norm = rejected_sum / rejected_n if rejected_n else float("-inf")
    return {
        "sum_margin": chosen_sum - rejected_sum,
        "norm_margin": chosen_norm - rejected_norm,
        "norm_chosen_wins": chosen_norm > rejected_norm,
    }


def rank_pool(
    model,
    tokenizer,
    history_titles: list[str],
    cand_items: list[int],
    titles: dict[int, str],
    max_title_chars: int,
    max_length: int | None,
    microbatch: int = 8,
) -> list[int]:
    if not cand_items:
        return []
    prompt = history_prompt(history_titles, max_title_chars)
    labels = pool_labels([titles.get(i, "") for i in cand_items], cand_items, max_title_chars)
    stats: list[tuple[float, int]] = []
    for start in range(0, len(labels), microbatch):
        stats.extend(
            completion_logprobs(model, tokenizer, prompt, labels[start : start + microbatch], max_length)
        )
    sums = [s for s, _ in stats]
    counts = [n for _, n in stats]
    return rank_by_logprobs(cand_items, sums, counts, length_normalize=True)


def history_titles_for(history_ids, titles: dict[int, str], max_title_chars: int) -> list[str]:
    from prompts import item_label

    return [item_label(titles.get(i, ""), i, max_title_chars) for i in history_ids]


def rank_candidate_frame(
    model,
    tokenizer,
    cand_df,
    items,
    max_title_chars: int,
    max_length: int | None,
    max_users: int | None = None,
) -> list[list[int]]:
    titles = title_map(items)
    ranked = []
    for n, record in enumerate(cand_df.itertuples(index=False)):
        if max_users is not None and n >= max_users:
            break
        history = as_int_list(record.history_items)
        cands = as_int_list(record.cand_items)
        hist_titles = history_titles_for(history, titles, max_title_chars)
        ranked.append(
            rank_pool(model, tokenizer, hist_titles, cands, titles, max_title_chars, max_length)
        )
    return ranked


def score_pairs(model, tokenizer, pairs: list[dict], max_length: int | None, max_pairs: int | None) -> list[dict]:
    if max_pairs is not None:
        pairs = pairs[: int(max_pairs)]
    return [
        pair_scores(model, tokenizer, row["prompt"], row["chosen"], row["rejected"], max_length)
        for row in pairs
    ]
