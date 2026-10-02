"""1-vs-1 preference rows: one positive title, k hard-negative titles (A5 §2–§3)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prompts import disambiguate, history_prompt, item_label


def as_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(x) for x in list(value)]


def title_map(items: pd.DataFrame) -> dict[int, str]:
    out = {}
    for row in items.itertuples(index=False):
        title = "" if pd.isna(row.title) else str(row.title)
        out[int(row.item_idx)] = title
    return out


def eligible_negatives(cand_items, target: int, history) -> list[int]:
    """Pool items that are not the target and not in the history, first-seen order."""
    hist = set(as_int_list(history))
    target = int(target)
    out: list[int] = []
    for item in as_int_list(cand_items):
        if item == target or item in hist or item in out:
            continue
        out.append(item)
    return out


def build_preference_pairs(
    cand_df: pd.DataFrame,
    items: pd.DataFrame,
    negatives_per_positive: int,
    rng: np.random.Generator,
    max_title_chars: int,
) -> list[dict]:
    """k DPO rows per context. Prompt is identical across those k rows."""
    titles = title_map(items)
    k = int(negatives_per_positive)
    rows: list[dict] = []
    for record in cand_df.itertuples(index=False):
        target = int(record.target_item)
        history = as_int_list(record.history_items)
        negs = eligible_negatives(record.cand_items, target, history)
        if not negs or k <= 0:
            continue
        rng.shuffle(negs)
        chosen_negs = negs[:k]
        hist_titles = [item_label(titles.get(i, ""), i, max_title_chars) for i in history]
        prompt = history_prompt(hist_titles, max_title_chars)
        chosen_label = item_label(titles.get(target, ""), target, max_title_chars)
        for neg in chosen_negs:
            neg_label = item_label(titles.get(neg, ""), neg, max_title_chars)
            chosen, rejected = disambiguate(chosen_label, target, neg_label, neg)
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "user_idx": int(record.user_idx),
                    "chosen_item": target,
                    "rejected_item": int(neg),
                }
            )
    return rows


def write_pairs(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_pairs(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
