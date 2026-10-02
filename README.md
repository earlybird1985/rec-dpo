# Preference Optimization for Recommendation (DPO)

LoRA DPO of `Qwen/Qwen2.5-1.5B-Instruct` on next-item preference pairs.
The prompt is the user's history. The chosen completion is the held-out
next item's title. The rejected completion is a hard negative from that
user's candidate list.

This is **Project A5** of a larger LLM-for-recommendation learning roadmap —
see [`A5_preference_optimization_implementation.md`](A5_preference_optimization_implementation.md)
for the full design doc (why these pairs are not logged skips, why DPO cannot
reuse A4's pointwise Yes/No prompts, the reference-policy recipe, length-normalized
scoring, and the causal note).

Amazon Reviews 2023 has no impression log. A rejected title was sampled from
the pool. It was not observed as a skip.

**Does not require A2 or A3.** If [`llm-reranking`](../llm-reranking) (A4) has
already written `candidates_*.parquet` and `items.parquet`, pair building
reuses them. Otherwise this repo downloads `Video_Games` and builds a
popularity/random pool. An A4 LoRA adapter is optional: with it, the frozen
reference is the ranking-SFT policy; without it, the reference is the base
Instruct model (`run_meta.json` records which).

## Status

- [x] Repo layout + configs matching the A5 plan
- [x] Pair builder: one shared prompt, chosen = next-item title, rejected = hard negative
- [x] Reference DPO loss and length-normalized ranking (unit-tested, no model download)
- [x] `prepare_preferences.py`: A4 parquets or own Amazon pools, leave-last-two-out
- [x] LoRA DPO via `trl.DPOTrainer` (`learning_rate` 5e-6, `beta` 0.1)
- [x] Before/after eval: reward margin, pair accuracy, NDCG@K, popularity buckets
- [x] Notebook for pairs, the loss shape, and where metrics land
- [ ] End-to-end train on a GPU or Mac (`dpo_small`, then full)
- [ ] `RESULTS.md` write-up, including the causal note

## Repo layout

```
├── configs/dpo.yaml                 # full Video_Games
├── configs/dpo_small.yaml           # subsample, 4-bit off
├── data/prepare_preferences.py
├── src/
│   ├── prompts.py                   # one user prompt
│   ├── pairs.py                     # 1 positive × k negatives
│   ├── losses.py                    # Bradley-Terry loss (tests; trl trains)
│   ├── train_dpo.py                 # PEFT + DPOTrainer
│   ├── infer.py                     # length-norm title log-probs
│   └── eval.py                      # margin, pair accuracy, NDCG@K
├── tests/test_dpo.py
├── notebooks/rec_dpo.ipynb
└── checkpoints/                     # gitignored
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bitsandbytes` 4-bit is CUDA / Linux only. On macOS the trainer forces
`load_in_4bit: false`. `configs/dpo.yaml` still says `true` for a Linux GPU;
`dpo_small.yaml` is off.

## Usage

```bash
# unit tests (no Amazon download, no Qwen)
python tests/test_dpo.py

# 1. Pairs. Reuses ../llm-reranking/data/processed when those parquets exist.
python data/prepare_preferences.py --config configs/dpo_small.yaml

# 2. Smoke train
python src/train_dpo.py --config configs/dpo_small.yaml --max_steps 20 --max_pairs 32

# 3. Full run
python data/prepare_preferences.py --config configs/dpo.yaml
python src/train_dpo.py --config configs/dpo.yaml
```

Score a finished adapter against its reference policy:

```bash
python src/eval.py --config configs/dpo.yaml --adapter checkpoints/dpo
```

Point `sft_adapter` at an A4 LoRA directory before training if you want the
reference to be the ranking-SFT model rather than the base Instruct model.
The trainer merges that adapter, saves `checkpoints/.../sft_merged/`, and
trains a new LoRA on top (zero init, so the forward pass starts as SFT).
`trl` freezes a copy of that initial adapter as `ref` and trains `default`.

**Jupyter:** open `notebooks/rec_dpo.ipynb` from the repo root and run all
cells. It rebuilds synthetic pairs and the DPO loss curve. It does not
download Qwen. Training stays on the CLI.

## Design notes

- **One prompt, two completions.** A4's pointwise Yes/No rows use a different
  prompt per candidate. DPO's Bradley-Terry term needs a shared `x`.
- **Rejected ≠ skipped.** Hard negative from the candidate list, excluded if
  it is the target or in the history.
- **Train injects the target; val/test do not** when `inject_target_at_eval`
  is false. Read `candidate_hit_rate` before reading NDCG.
- **Length-normalized log-prob** ranks candidates. The sum of token log-probs
  prefers short titles.
- **DPO learning rate is `5e-6`.** `2e-4` is the A4 SFT rate and will erase
  the reference policy.
- **S-DPO and A3 are out of scope.** `trl.DPOTrainer` is 1-vs-1 on a causal LM.
