"""Build DPO preference pairs from A4 candidate parquets, or from Amazon reviews.

Usage:
    python data/prepare_preferences.py --config configs/dpo.yaml
    python data/prepare_preferences.py --config configs/dpo_small.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pairs import as_int_list, build_preference_pairs, write_pairs
from utils import load_config, save_json

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"


def review_url(category: str) -> str:
    return f"{BASE_URL}/review_categories/{category}.jsonl.gz"


def meta_url(category: str) -> str:
    return f"{BASE_URL}/meta_categories/meta_{category}.jsonl.gz"


def download_if_missing(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest} already downloaded")
        return dest
    print(f"Downloading {url} -> {dest} (can be several hundred MB)")
    urlretrieve(url, dest)
    return dest


def k_core_filter(reviews: pd.DataFrame, k: int, item_col: str, max_iters: int = 20) -> pd.DataFrame:
    prev_len = None
    for i in range(max_iters):
        item_counts = reviews[item_col].value_counts()
        user_counts = reviews["user_id"].value_counts()
        keep_items = item_counts[item_counts >= k].index
        keep_users = user_counts[user_counts >= k].index
        filtered = reviews[reviews[item_col].isin(keep_items) & reviews["user_id"].isin(keep_users)]
        print(
            f"  k-core iter {i + 1}: {len(reviews):,} -> {len(filtered):,} rows, "
            f"{filtered['user_id'].nunique():,} users, {filtered[item_col].nunique():,} items"
        )
        if prev_len == len(filtered):
            return filtered.reset_index(drop=True)
        reviews = filtered
        prev_len = len(filtered)
    return reviews.reset_index(drop=True)


def categories_to_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        parts = []
        for piece in val:
            if isinstance(piece, (list, tuple)):
                parts.append(" > ".join(str(x) for x in piece))
            else:
                parts.append(str(piece))
        return " | ".join(p for p in parts if p)[:500]
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)[:500]


def load_reviews(path: Path) -> pd.DataFrame:
    cols = ["user_id", "parent_asin", "timestamp"]
    chunks = []
    for chunk in pd.read_json(path, lines=True, chunksize=500_000, compression="gzip", convert_dates=False):
        keep = [c for c in cols if c in chunk.columns]
        chunks.append(chunk[keep])
    return pd.concat(chunks, ignore_index=True)


def load_meta(path: Path) -> pd.DataFrame:
    wanted = ["parent_asin", "title", "categories"]
    chunks = []
    for chunk in pd.read_json(path, lines=True, chunksize=200_000, compression="gzip"):
        keep = [c for c in wanted if c in chunk.columns]
        chunks.append(chunk[keep])
    meta = pd.concat(chunks, ignore_index=True)
    if "title" not in meta.columns:
        meta["title"] = ""
    if "categories" not in meta.columns:
        meta["categories"] = ""
    meta["parent_asin"] = meta["parent_asin"].astype(str)
    meta["title"] = meta["title"].fillna("").astype(str)
    meta["categories"] = meta["categories"].map(categories_to_str)
    return meta.drop_duplicates("parent_asin", keep="first")


def build_split(sequences: pd.DataFrame) -> dict:
    """Leave-last-two-out. Train targets are positions 1 .. n-3; val n-2; test n-1."""
    train_u, train_p = [], []
    val_u, val_p = [], []
    test_u, test_p = [], []
    too_short = 0
    for row in sequences.itertuples(index=False):
        user = int(row.user_idx)
        n = len(row.item_idxs)
        if n < 3:
            too_short += 1
            continue
        for pos in range(1, n - 2):
            train_u.append(user)
            train_p.append(pos)
        val_u.append(user)
        val_p.append(n - 2)
        test_u.append(user)
        test_p.append(n - 1)
    if too_short:
        print(f"dropped {too_short} users with seq length < 3")
    print(
        f"split: {len(train_u):,} train examples, {len(val_u):,} val users, {len(test_u):,} test users"
    )
    return {
        "train_user_idx": np.asarray(train_u, dtype=np.int64),
        "train_target_pos": np.asarray(train_p, dtype=np.int64),
        "val_user_idx": np.asarray(val_u, dtype=np.int64),
        "val_target_pos": np.asarray(val_p, dtype=np.int64),
        "test_user_idx": np.asarray(test_u, dtype=np.int64),
        "test_target_pos": np.asarray(test_p, dtype=np.int64),
    }


def train_item_frequency(sequences: pd.DataFrame, split: dict) -> pd.Series:
    seq_by_user = {
        int(r.user_idx): as_int_list(r.item_idxs) for r in sequences.itertuples(index=False)
    }
    counts: dict[int, int] = {}
    for user, pos in zip(split["train_user_idx"].tolist(), split["train_target_pos"].tolist()):
        item = int(seq_by_user[int(user)][int(pos)])
        counts[item] = counts.get(item, 0) + 1
    return pd.Series(counts, dtype="int64")


def popularity_random_pool(
    target: int,
    n_cand: int,
    item_ids: np.ndarray,
    freq: pd.Series,
    rng: np.random.Generator,
    exclude: set[int],
    inject_target: bool,
) -> tuple[list[int], list[float]]:
    catalog = [int(i) for i in item_ids if int(i) not in exclude and int(i) != int(target)]
    weights = np.array([float(freq.get(i, 1.0)) for i in catalog], dtype=np.float64)
    weights = weights / weights.sum() if len(catalog) and weights.sum() > 0 else None
    n_draw = n_cand - (1 if inject_target else 0)
    n_draw = max(0, min(n_draw, len(catalog)))
    picked: list[int] = []
    if n_draw and catalog:
        picked = [int(x) for x in rng.choice(catalog, size=n_draw, replace=False, p=weights).tolist()]
    if inject_target and target not in picked:
        if len(picked) >= n_cand:
            picked[-1] = int(target)
        else:
            picked.append(int(target))
    out: list[int] = []
    for item in picked:
        if item not in out:
            out.append(item)
        if len(out) >= n_cand:
            break
    scores = [float(-i) for i in range(len(out))]
    return out, scores


def build_candidate_frame(
    sequences: pd.DataFrame,
    user_idx: np.ndarray,
    target_pos: np.ndarray,
    n_cand: int,
    freq: pd.Series,
    item_ids: np.ndarray,
    rng: np.random.Generator,
    inject_target: bool,
    max_history_len: int,
) -> pd.DataFrame:
    seq_by_user = {
        int(r.user_idx): as_int_list(r.item_idxs) for r in sequences.itertuples(index=False)
    }
    rows = []
    for user, pos in zip(user_idx.tolist(), target_pos.tolist()):
        seq = seq_by_user[int(user)]
        pos = int(pos)
        target = int(seq[pos])
        full_hist = seq[:pos]
        pool, scores = popularity_random_pool(
            target, n_cand, item_ids, freq, rng, set(full_hist), inject_target
        )
        rows.append(
            {
                "user_idx": int(user),
                "target_pos": pos,
                "target_item": target,
                "history_items": full_hist[-max_history_len:],
                "cand_items": pool,
                "cand_scores": scores,
                "hit": int(target in pool),
            }
        )
    return pd.DataFrame(rows)


def subsample_positions(
    user_idx: np.ndarray, target_pos: np.ndarray, max_examples: int | None, rng: np.random.Generator
):
    if not max_examples or len(user_idx) <= int(max_examples):
        return user_idx, target_pos
    sel = np.sort(rng.choice(len(user_idx), size=int(max_examples), replace=False))
    return user_idx[sel], target_pos[sel]


def hit_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    hits = []
    for record in frame.itertuples(index=False):
        hits.append(int(int(record.target_item) in set(as_int_list(record.cand_items))))
    return float(np.mean(hits))


def write_split_pairs(frame: pd.DataFrame, items: pd.DataFrame, split: str, cfg: dict, processed: Path, rng):
    frame.to_parquet(processed / f"candidates_{split}.parquet", index=False)
    pairs = build_preference_pairs(
        frame,
        items,
        int(cfg.get("negatives_per_positive", 4)),
        rng,
        int(cfg.get("max_title_chars", 160)),
    )
    write_pairs(pairs, processed / f"pairs_{split}.jsonl")
    rate = hit_rate(frame)
    print(f"{split}: {len(frame):,} lists, hit rate {rate:.3f}, {len(pairs):,} pairs")
    return {"n_lists": int(len(frame)), "hit_rate": rate, "n_pairs": int(len(pairs))}


def a4_dir(cfg: dict) -> Path | None:
    raw = cfg.get("a4_processed_dir")
    if not raw:
        return None
    path = Path(raw)
    needed = ["items.parquet", "candidates_train.parquet", "candidates_val.parquet", "candidates_test.parquet"]
    if all((path / name).exists() for name in needed):
        return path
    print(f"A4 processed dir not usable ({path}); building pools from Amazon reviews")
    return None


def prepare_from_a4(src: Path, cfg: dict) -> None:
    processed = Path(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(src / "items.parquet")
    items.to_parquet(processed / "items.parquet", index=False)
    nonempty = float((items["title"].fillna("").astype(str).str.strip() != "").mean()) if len(items) else 0.0
    print(f"reused A4 items from {src} ({len(items):,} items, title non-empty {nonempty:.3f})")
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    stats = {"source": "a4", "a4_processed_dir": str(src), "title_nonempty_rate": nonempty}
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(src / f"candidates_{split}.parquet")
        if split == "train":
            user_idx = frame["user_idx"].to_numpy()
            # sample rows, not users — train is one row per prefix
            if cfg.get("max_train_examples") and len(frame) > int(cfg["max_train_examples"]):
                sel = np.sort(rng.choice(len(frame), size=int(cfg["max_train_examples"]), replace=False))
                frame = frame.iloc[sel].reset_index(drop=True)
                print(f"capped train lists to {len(frame):,}")
            del user_idx
        stats[split] = write_split_pairs(frame, items, split, cfg, processed, rng)
    save_json(stats, processed / "prep_stats.json")
    print("wrote", processed)


def prepare_from_amazon(cfg: dict) -> None:
    category = cfg["category"]
    raw_dir = Path(cfg["raw_dir"])
    processed = Path(cfg["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    k_core = int(cfg.get("k_core", 5))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    review_path = download_if_missing(review_url(category), raw_dir / f"{category}.jsonl.gz")
    meta_path = download_if_missing(meta_url(category), raw_dir / f"meta_{category}.jsonl.gz")
    print(f"Loading reviews for category={category} ...")
    reviews = load_reviews(review_path)
    n_raw = len(reviews)
    print(f"  raw interactions: {n_raw:,}")
    reviews["user_id"] = reviews["user_id"].astype(str)
    reviews["parent_asin"] = reviews["parent_asin"].astype(str)
    reviews["timestamp"] = reviews["timestamp"].astype("int64")
    reviews = reviews.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")

    a1 = cfg.get("a1_processed_dir")
    item2idx = None
    if a1:
        item2idx_path = Path(a1) / "item2idx.json"
        if not item2idx_path.exists():
            raise FileNotFoundError(f"a1_processed_dir set but {item2idx_path} is missing")
        with open(item2idx_path) as f:
            item2idx = {str(k): int(v) for k, v in json.load(f).items()}
        reviews["item_idx"] = reviews["parent_asin"].map(item2idx)
        n_unmapped = int(reviews["item_idx"].isna().sum())
        reviews = reviews.dropna(subset=["item_idx"]).astype({"item_idx": int})
        print(f"  mapped through A1 item2idx; dropped {n_unmapped:,} interactions outside that catalog")
        item_col = "item_idx"
    else:
        item_col = "parent_asin"
        n_unmapped = 0

    print(f"iterative {k_core}-core filter ...")
    reviews = k_core_filter(reviews, k_core, item_col)
    if reviews.empty:
        raise RuntimeError(f"{k_core}-core produced no rows. Try a smaller k_core.")

    sample_users = cfg.get("sample_users")
    if sample_users:
        users = reviews["user_id"].unique()
        if len(users) > int(sample_users):
            keep = set(rng.choice(users, size=int(sample_users), replace=False))
            reviews = reviews[reviews["user_id"].isin(keep)].reset_index(drop=True)
            print(f"sampled {int(sample_users):,} users -> {len(reviews):,} interactions")

    if item2idx is None:
        asins = sorted(reviews["parent_asin"].unique())
        item2idx = {asin: i for i, asin in enumerate(asins)}
        reviews["item_idx"] = reviews["parent_asin"].map(item2idx).astype(int)
    save_json(item2idx, processed / "item2idx.json")

    print("Loading item metadata ...")
    meta = load_meta(meta_path)
    items = (
        reviews[["item_idx", "parent_asin"]]
        .drop_duplicates("item_idx")
        .merge(meta, on="parent_asin", how="left")
    )
    items["title"] = items["title"].fillna("").astype(str)
    items["categories"] = items["categories"].fillna("").astype(str)
    nonempty = float((items["title"].str.strip() != "").mean()) if len(items) else 0.0
    print(f"  items {len(items):,}, title non-empty {nonempty:.3f}")
    items.to_parquet(processed / "items.parquet", index=False)

    reviews = reviews.sort_values(["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)
    grouped = reviews.groupby("user_id", sort=True)["item_idx"].apply(list)
    sequences = grouped.reset_index().rename(columns={"item_idx": "item_idxs"})
    sequences["user_idx"] = np.arange(len(sequences), dtype=np.int64)
    user2idx = {str(user): int(i) for user, i in zip(sequences["user_id"], sequences["user_idx"])}
    save_json(user2idx, processed / "user2idx.json")
    sequences.to_parquet(processed / "user_sequences.parquet", index=False)

    split = build_split(sequences)
    np.savez_compressed(processed / "split.npz", **split)
    freq = train_item_frequency(sequences, split)
    item_ids = items["item_idx"].to_numpy()
    max_hist = int(cfg.get("max_history_len", 20))
    n_cand = int(cfg.get("n_cand", 50))
    inject_eval = bool(cfg.get("inject_target_at_eval", False))

    train_u, train_p = subsample_positions(
        split["train_user_idx"], split["train_target_pos"], cfg.get("max_train_examples"), rng
    )
    specs = {
        "train": (train_u, train_p, True),
        "val": (split["val_user_idx"], split["val_target_pos"], inject_eval),
        "test": (split["test_user_idx"], split["test_target_pos"], inject_eval),
    }
    stats = {
        "source": "amazon",
        "category": category,
        "n_raw_reviews": int(n_raw),
        "n_unmapped_dropped": int(n_unmapped),
        "n_users": int(len(sequences)),
        "n_items": int(len(items)),
        "title_nonempty_rate": nonempty,
        "k_core": k_core,
        "a1_processed_dir": a1,
    }
    for split_name, (users, positions, inject) in specs.items():
        frame = build_candidate_frame(
            sequences, users, positions, n_cand, freq, item_ids, rng, inject, max_hist
        )
        stats[split_name] = write_split_pairs(frame, items, split_name, cfg, processed, rng)
    save_json(stats, processed / "prep_stats.json")
    print("wrote", processed)
    print(stats)


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    src = a4_dir(cfg)
    if src is not None:
        prepare_from_a4(src, cfg)
    else:
        prepare_from_amazon(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
