"""One shared user prompt and title display strings (A5 §3–§4).

DPO compares two completions under the same prompt. The completion is the
product title, not A4's Yes/No token.
"""
from __future__ import annotations


def clip_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return text[:1]
    return text[: max_chars - 1] + "…"


def item_label(title: str, item_idx: int, max_chars: int) -> str:
    """Display string for a completion. Blank titles stay distinct per item."""
    text = clip_text(title, max_chars)
    if not text:
        text = f"item {int(item_idx)}"
    return text


def disambiguate(label_a: str, idx_a: int, label_b: str, idx_b: int) -> tuple[str, str]:
    """Identical completions give DPO a zero gradient. Break the tie with ids."""
    if label_a != label_b:
        return label_a, label_b
    return f"{label_a} [id:{int(idx_a)}]", f"{label_b} [id:{int(idx_b)}]"


def pool_labels(titles: list[str], item_ids: list[int], max_chars: int) -> list[str]:
    """Labels for a full candidate pool. Only colliding strings gain an id suffix."""
    raw = [item_label(title, item, max_chars) for title, item in zip(titles, item_ids)]
    counts: dict[str, int] = {}
    for lab in raw:
        counts[lab] = counts.get(lab, 0) + 1
    out = []
    for lab, item in zip(raw, item_ids):
        if counts[lab] > 1:
            out.append(f"{lab} [id:{int(item)}]")
        else:
            out.append(lab)
    return out


def history_prompt(history_titles: list[str], max_title_chars: int) -> str:
    lines = []
    for i, title in enumerate(history_titles, start=1):
        lines.append(f"{i}. {clip_text(title, max_title_chars)}")
    hist = "\n".join(lines) if lines else "(no history)"
    return (
        "You are recommending the next product for a shopper.\n"
        "The shopper recently interacted with (oldest to newest):\n"
        f"{hist}\n"
        "Which product will they interact with next? Answer with the product title only."
    )
