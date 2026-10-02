"""Before/after DPO: reward margin, pair accuracy, next-item NDCG@K (A5 §6).

Usage:
    python src/eval.py --config configs/dpo.yaml --adapter checkpoints/dpo
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import rank_candidate_frame, score_pairs
from pairs import read_pairs
from utils import load_config, load_json, save_json


def recall_at_k(ranked: list[int], target: int, k: int) -> float:
    return float(target in ranked[:k])


def ndcg_at_k(ranked: list[int], target: int, k: int) -> float:
    for i, item in enumerate(ranked[:k]):
        if int(item) == int(target):
            return 1.0 / math.log2(i + 2)
    return 0.0


def popularity_buckets(train_freq: pd.Series, q: int = 3) -> pd.Series:
    freq = train_freq.astype(float)
    try:
        return pd.qcut(freq, q=q, labels=["long_tail", "mid", "head"], duplicates="drop")
    except ValueError:
        return pd.Series(["mid"] * len(freq), index=freq.index)


def train_target_frequency(train_cand: pd.DataFrame) -> pd.Series:
    return train_cand["target_item"].astype(int).value_counts()


def eval_rankings(
    cand_df: pd.DataFrame,
    ranked_lists: list[list[int]],
    k_list: list[int],
    train_freq: pd.Series | None = None,
) -> dict:
    n = min(len(ranked_lists), len(cand_df))
    frame = cand_df.iloc[:n]
    hits = []
    recs = {k: [] for k in k_list}
    ndcgs = {k: [] for k in k_list}
    buckets = popularity_buckets(train_freq) if train_freq is not None and len(train_freq) else None
    bucket_ndcg: dict[str, list[float]] = {"long_tail": [], "mid": [], "head": []}
    for record, ranked in zip(frame.itertuples(index=False), ranked_lists):
        target = int(record.target_item)
        pool = [int(x) for x in list(record.cand_items)]
        hits.append(int(target in pool))
        for k in k_list:
            recs[k].append(recall_at_k(ranked, target, k))
            ndcgs[k].append(ndcg_at_k(ranked, target, k))
        if buckets is not None and target in buckets.index:
            bname = str(buckets.loc[target])
            if bname in bucket_ndcg:
                k_b = 10 if 10 in k_list else max(k_list)
                bucket_ndcg[bname].append(ndcg_at_k(ranked, target, k_b))
    metrics = {
        "n_eval": n,
        "candidate_hit_rate": float(np.mean(hits)) if n else 0.0,
    }
    for k in k_list:
        metrics[f"recall@{k}"] = float(np.mean(recs[k])) if n else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k])) if n else 0.0
    if buckets is not None:
        for name, vals in bucket_ndcg.items():
            metrics[f"ndcg@10_{name}"] = float(np.mean(vals)) if vals else float("nan")
            metrics[f"n_{name}"] = int(len(vals))
    return metrics


def retriever_order_lists(cand_df: pd.DataFrame, max_users: int | None) -> list[list[int]]:
    frame = cand_df if max_users is None else cand_df.iloc[: int(max_users)]
    return [[int(x) for x in list(record.cand_items)] for record in frame.itertuples(index=False)]


def summarize_pairs(records: list[dict]) -> dict:
    if not records:
        return {"n_pairs": 0, "pair_accuracy": None, "reward_margin": None, "norm_margin": None}
    return {
        "n_pairs": len(records),
        "pair_accuracy": float(np.mean([r["norm_chosen_wins"] for r in records])),
        "reward_margin": float(np.mean([r["sum_margin"] for r in records])),
        "norm_margin": float(np.mean([r["norm_margin"] for r in records])),
    }


def load_tokenizer_and_causal(base_path: str, load_in_4bit: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"low_cpu_mem_usage": True}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype_for_4bit(),
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch_dtype_for_4bit()
    model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)
    if not load_in_4bit:
        model = place_model(model)
    model.eval()
    return tokenizer, model


def torch_dtype_for_4bit():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def place_model(model):
    import torch

    if torch.cuda.is_available():
        return model.to("cuda")
    if torch.backends.mps.is_available():
        return model.to("mps")
    return model


def score_policy(model, tokenizer, cand_df, items, pairs, cfg) -> dict:
    max_users = cfg.get("max_eval_users")
    max_pairs = cfg.get("max_eval_pairs")
    max_length = int(cfg["max_length"])
    max_chars = int(cfg.get("max_title_chars", 160))
    k_list = [int(k) for k in cfg.get("k_list", [5, 10, 20])]
    frame = cand_df if max_users is None else cand_df.iloc[: int(max_users)]
    ranked = rank_candidate_frame(model, tokenizer, frame, items, max_chars, max_length, max_users=None)
    train_path = Path(cfg["processed_dir"]) / "candidates_train.parquet"
    freq = train_target_frequency(pd.read_parquet(train_path)) if train_path.exists() else None
    metrics = eval_rankings(frame, ranked, k_list, freq)
    metrics.update(summarize_pairs(score_pairs(model, tokenizer, pairs, max_length, max_pairs)))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True, help="DPO adapter directory (contains run_meta.json)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = Path(cfg["processed_dir"])
    adapter = Path(args.adapter)
    meta_path = adapter / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}. Train with src/train_dpo.py first.")
    meta = load_json(meta_path)
    base_path = meta.get("base_path") or meta["model_name"]
    from train_dpo import resolve_load_in_4bit

    load_in_4bit = resolve_load_in_4bit(cfg)
    items = pd.read_parquet(processed / "items.parquet")
    cand_df = pd.read_parquet(processed / "candidates_test.parquet")
    pairs = read_pairs(processed / "pairs_test.jsonl") if (processed / "pairs_test.jsonl").exists() else []
    k_list = [int(k) for k in cfg.get("k_list", [5, 10, 20])]
    max_users = cfg.get("max_eval_users")
    train_path = processed / "candidates_train.parquet"
    freq = train_target_frequency(pd.read_parquet(train_path)) if train_path.exists() else None
    frame = cand_df if max_users is None else cand_df.iloc[: int(max_users)]
    retriever = eval_rankings(frame, retriever_order_lists(frame, None), k_list, freq)

    print(f"loading reference from {base_path} (ref_is_sft={meta.get('ref_is_sft')})")
    tokenizer, ref_model = load_tokenizer_and_causal(base_path, load_in_4bit)
    reference = score_policy(ref_model, tokenizer, cand_df, items, pairs, cfg)
    del ref_model

    from peft import PeftModel

    print(f"loading DPO adapter from {adapter}")
    _, policy = load_tokenizer_and_causal(base_path, load_in_4bit)
    policy = PeftModel.from_pretrained(policy, str(adapter))
    if hasattr(policy, "set_adapter") and "default" in getattr(policy, "peft_config", {}):
        policy.set_adapter("default")
    policy.eval()
    dpo = score_policy(policy, tokenizer, cand_df, items, pairs, cfg)

    report = {
        "ref_is_sft": meta.get("ref_is_sft"),
        "base_path": base_path,
        "adapter": str(adapter),
        "retriever": retriever,
        "reference": reference,
        "dpo": dpo,
    }
    out = adapter / "test_metrics.json"
    save_json(report, out)
    print(report)
    print("wrote", out)


if __name__ == "__main__":
    main()
