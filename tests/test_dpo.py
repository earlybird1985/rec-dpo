"""Pair construction, DPO loss, and length-normalized ranking. No model download."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from losses import dpo_sigmoid_loss, rank_by_logprobs
from pairs import build_preference_pairs


def _items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_idx": [0, 1, 2, 3],
            "title": ["", "", "Halo", "Halo"],
        }
    )


def test_pairs_share_prompt_and_reject_hard_negatives():
    cand = pd.DataFrame(
        [
            {
                "user_idx": 7,
                "target_item": 0,
                "history_items": [3],
                "cand_items": [0, 1, 2, 3],
                "cand_scores": [0.0, -1.0, -2.0, -3.0],
            }
        ]
    )
    rng = np.random.default_rng(0)
    rows = build_preference_pairs(cand, _items(), negatives_per_positive=2, rng=rng, max_title_chars=40)
    assert len(rows) == 2
    assert rows[0]["prompt"] == rows[1]["prompt"]
    assert "item 0" in rows[0]["chosen"] or rows[0]["chosen"].startswith("item 0")
    rejected_ids = {row["rejected_item"] for row in rows}
    assert rejected_ids <= {1, 2}
    assert 0 not in rejected_ids
    assert 3 not in rejected_ids  # in history
    for row in rows:
        assert row["chosen"] != row["rejected"]
        assert row["prompt"]
        assert "Answer with the product title only." in row["prompt"]


def test_empty_and_duplicate_titles_do_not_collide():
    cand = pd.DataFrame(
        [
            {
                "user_idx": 1,
                "target_item": 0,
                "history_items": [],
                "cand_items": [0, 1],
                "cand_scores": [0.0, -1.0],
            },
            {
                "user_idx": 2,
                "target_item": 2,
                "history_items": [],
                "cand_items": [2, 3],
                "cand_scores": [0.0, -1.0],
            },
        ]
    )
    rows = build_preference_pairs(
        cand, _items(), negatives_per_positive=1, rng=np.random.default_rng(1), max_title_chars=40
    )
    by_user = {row["user_idx"]: row for row in rows}
    assert by_user[1]["chosen"] != by_user[1]["rejected"]
    assert by_user[1]["chosen"] == "item 0"
    assert by_user[1]["rejected"] == "item 1"
    assert by_user[2]["chosen"] != by_user[2]["rejected"]
    assert "[id:2]" in by_user[2]["chosen"]
    assert "[id:3]" in by_user[2]["rejected"]


def test_dpo_loss_at_zero_margin_and_decreases():
    loss0 = dpo_sigmoid_loss(0.0, 0.0, 0.0, 0.0, beta=0.1)
    assert abs(loss0 - math.log(2)) < 1e-9
    # Policy assigns 1.0 more log-prob to the chosen title than the rejected one.
    loss1 = dpo_sigmoid_loss(0.0, -1.0, 0.0, 0.0, beta=0.1)
    assert loss1 < loss0
    loss_ref_already = dpo_sigmoid_loss(0.0, -1.0, 0.0, -1.0, beta=0.1)
    assert abs(loss_ref_already - math.log(2)) < 1e-9


def test_length_norm_prefers_higher_per_token_not_the_sum():
    # Short title: sum -0.20 over 1 token (per-token -0.20).
    # Long title: sum -0.45 over 3 tokens (per-token -0.15).
    # The sum prefers the short title. The mean prefers the long one.
    item_ids = [1, 2]
    sums = [-0.20, -0.45]
    counts = [1, 3]
    by_sum = rank_by_logprobs(item_ids, sums, counts, length_normalize=False)
    by_mean = rank_by_logprobs(item_ids, sums, counts, length_normalize=True)
    assert by_sum[0] == 1
    assert by_mean[0] == 2


def test_pool_injects_target_on_train_only():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prepare_preferences", ROOT / "data" / "prepare_preferences.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    sequences = pd.DataFrame({"user_idx": [0], "item_idxs": [[10, 11, 12, 13, 14]]})
    items = pd.DataFrame(
        {
            "item_idx": [10, 11, 12, 13, 14, 15, 16],
            "title": ["a", "b", "c", "d", "e", "f", "g"],
        }
    )
    freq = pd.Series({10: 5, 11: 4, 12: 3, 13: 2, 14: 1, 15: 1, 16: 1})
    item_ids = items["item_idx"].to_numpy()
    rng = np.random.default_rng(0)
    train = prep.build_candidate_frame(
        sequences,
        np.array([0]),
        np.array([2]),
        n_cand=4,
        freq=freq,
        item_ids=item_ids,
        rng=rng,
        inject_target=True,
        max_history_len=20,
    )
    assert int(train.iloc[0].target_item) == 12
    assert 12 in as_list(train.iloc[0].cand_items)
    assert set(as_list(train.iloc[0].history_items)) <= {10, 11}
    held = prep.build_candidate_frame(
        sequences,
        np.array([0]),
        np.array([4]),
        n_cand=3,
        freq=freq,
        item_ids=item_ids,
        rng=np.random.default_rng(1),
        inject_target=False,
        max_history_len=2,
    )
    assert list(as_list(held.iloc[0].history_items)) == [12, 13]
    assert int(held.iloc[0].target_item) == 14
    pool = set(as_list(held.iloc[0].cand_items))
    assert pool.isdisjoint({10, 11, 12, 13})
    pairs = build_preference_pairs(train, items, 2, np.random.default_rng(0), 40)
    assert pairs
    assert all(row["chosen_item"] == 12 for row in pairs)
    assert all(row["rejected_item"] != 12 for row in pairs)


def as_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(x) for x in value]


def main():
    test_pairs_share_prompt_and_reject_hard_negatives()
    test_empty_and_duplicate_titles_do_not_collide()
    test_dpo_loss_at_zero_margin_and_decreases()
    test_length_norm_prefers_higher_per_token_not_the_sum()
    test_pool_injects_target_on_train_only()
    print("test_dpo: ok")


if __name__ == "__main__":
    main()
