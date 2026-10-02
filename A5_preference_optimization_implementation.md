# A5 Implementation Plan: Preference Optimization for Recommendation (DPO)

> Companion doc to `llm_career_pivot_project_roadmap.md`. This expands A5 into a step-by-step build plan: Bradley-Terry preference pairs from Amazon Reviews 2023, LoRA DPO of `Qwen/Qwen2.5-1.5B-Instruct` with Hugging Face `trl.DPOTrainer`, and a before/after table on the same next-item candidate lists A4 reranks.
>
> **This DPO-tunes the A4 reranker, not A3's encoder-decoder.** `trl.DPOTrainer` fits a causal LM. A3's from-scratch generative retriever is not that model. Multi-negative S-DPO (Chen et al.) is the paper that *would* sit on A3; it is cited below and left out of this repo.
>
> **Amazon Reviews has no impression log.** There is no observed "shown but skipped." The rejected item is a hard negative from the candidate pool. That is an assumed contrast, not a logged skip. The causal write-up in §7 is about that gap.

## 0. Scope and success criteria

**What "done" looks like:**
- Preference pairs `(prompt, chosen, rejected)` on Amazon Reviews 2023 (`Video_Games` by default). One shared prompt per user context (history as titles). `chosen` is the held-out next item's title. `rejected` is a hard negative from that user's candidate list, not in the history and not the target.
- LoRA DPO of `Qwen/Qwen2.5-1.5B-Instruct` via `peft` + `trl.DPOTrainer`. If an A4 SFT adapter is present, merge it first and train a new LoRA so the frozen reference policy is the ranking-SFT model. If it is absent, the reference is the base Instruct model and `RESULTS.md` says so.
- Before/after on the **same** lists:
  - pair accuracy and reward margin `log π(y_w|x) − log π(y_l|x)`
  - NDCG@K / Recall@K from **length-normalized** title log-probs (next-item, leave-last-out, same protocol as A4)
  - popularity-bucketed NDCG@10 from **training** target frequency
- A short write-up (`RESULTS.md`): the table, whether the margin moved with NDCG or against it, and the causal note (reference policy as logging policy, hard negatives as stand-in controls, what real skips would change).

**What "done" does NOT require:** beating A4's LightGBM on NDCG; multi-negative S-DPO; PPO / a reward model (that's Track B5); DPO on A3 semantic-ID sequences; inverse-propensity weighting (no propensities exist in this data); vLLM serving (A7).

---

## 1. Repo layout

New repo, `rec-dpo/` — sits next to `llm-reranking` (A4) so candidate dumps are easy to reuse:

```
rec-dpo/
├── README.md
├── RESULTS.md
├── requirements.txt
├── configs/
│   ├── dpo.yaml                        # full Video_Games, Qwen2.5-1.5B-Instruct
│   └── dpo_small.yaml                  # subsample, short sequences, CPU/Mac smoke
├── data/
│   ├── prepare_preferences.py          # A4 parquets if present, else own pools
│   ├── raw/
│   └── processed/
│       ├── items.parquet
│       ├── user_sequences.parquet
│       ├── split.npz
│       ├── candidates_{train,val,test}.parquet
│       └── pairs_{train,val,test}.jsonl
├── src/
│   ├── utils.py
│   ├── prompts.py                      # one user prompt; title completions
│   ├── pairs.py                        # 1 positive × k hard negatives
│   ├── losses.py                       # reference DPO loss (tests; trl trains)
│   ├── train_dpo.py                    # PEFT LoRA + trl.DPOTrainer
│   ├── infer.py                        # length-norm title log-probs
│   └── eval.py                         # margin, pair accuracy, NDCG@K
├── notebooks/
│   └── rec_dpo.ipynb                   # pairs + loss shape; how to read metrics
├── tests/
│   └── test_dpo.py
└── checkpoints/                        # adapter, merged SFT, test_metrics.json
```

**Papers and code to read (not copy)** before writing the pair builder and the training loop:

| Work | What to take | What not to cargo-cult |
|---|---|---|
| [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., NeurIPS 2023); code in [`huggingface/trl`](https://github.com/huggingface/trl) `DPOTrainer` | Bradley-Terry loss on a frozen reference; implicit reward `β log π/π_ref`; no separate reward model | HH-RLHF chat preferences as if they were recsys impressions |
| [Zephyr](https://arxiv.org/abs/2310.16944) (Tunstall et al., 2023) | SFT first, then DPO from that policy; DPO learning rate much smaller than SFT (`5e-6` vs `2e-4`) | Distilling GPT-4 UltraFeedback as the pair source |
| [S-DPO](https://arxiv.org/abs/2406.00333) (Chen et al., 2024) | The recsys reason to prefer hard negatives over random catalog items; multi-negative softmax is the extension when one rejected item is too weak | Plugging S-DPO into this repo. It targets a generative item decoder and a K-way softmax. `trl.DPOTrainer` is 1-vs-1. A3 would be the home for that loss |
| [Schnabel et al.](https://arxiv.org/abs/1602.05352) (WSDM 2016), recommendations as treatments | Exposure is MNAR. A clicked item is not a random draw from the catalog. Unweighted "skipped" labels confound popularity and position | Implementing IPS. We do not have propensities. The point is to *name* the bias, not pretend DPO removed it |

Write `src/pairs.py` and `src/losses.py` first and unit-test them with no model download. Then wire `train_dpo.py` to `trl`.

---

## 2. Data: preference pairs from candidate lists

### 2.1 Why the rejected item is not a skip

The roadmap's sentence is: for a context, item A (clicked / long-dwell) is preferred over item B (shown, skipped). Amazon Reviews 2023 records a review, a star rating, and a timestamp. It does not record the slate that was shown, dwell, or an explicit skip.

So this project builds the pair the data can support:

- **Chosen** = the leave-last-out next item (the same single relevant item as A3 and A4).
- **Rejected** = an item in that user's candidate pool that is not the target and not in the history.

Call the rejected item a **hard negative**, not a skip. If you write "the user skipped B" in `RESULTS.md`, the claim is false. The honest sentence is: "B was drawn from the retriever pool and does not appear as this user's next item."

A high-vs-low star-rating contrast (rating ≥ 4 vs ≤ 2) is a different preference, and it is not this repo's training signal. Ratings are not required for the pipeline.

### 2.2 Where the pools come from

`data/prepare_preferences.py` prefers A4 artifacts when they exist:

```
../llm-reranking/data/processed/items.parquet
../llm-reranking/data/processed/candidates_{train,val,test}.parquet
```

Columns used: `user_idx`, `target_item`, `history_items`, `cand_items`, and `title` / `categories` on items. Indices are A4's. Do not re-index.

If those files are missing, build the same objects here:

```
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{category}.jsonl.gz
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz
```

Join on `parent_asin`. Iterative 5-core. Leave-last-two-out (train targets are prefixes ending before `n-2`; val is `n-2`; test is `n-1`). History clipped to `max_history_len`.

Candidate pool when A1/A4 is absent: popularity-weighted sample from **training-target** frequency plus the uniform fallback inside that sampler, size `n_cand`. **Inject the target on train** so every training context has a real chosen item in the slate. **Do not inject on val/test** when `inject_target_at_eval: false`. Log hit rate. A test NDCG near 0 with a tiny hit rate is a retrieval failure.

Optional `a1_processed_dir`: if `item2idx.json` is there, map through it verbatim (same trap as A3 §2.1 / A4 §2.2). Otherwise this repo owns its `item_idx` and must not pretend it matches A1.

### 2.3 Pair schema

For each candidate row, shuffle negatives and keep `negatives_per_positive` (default 4). One DPO row per negative:

| field | meaning |
|---|---|
| `prompt` | shared user text (history). Identical across the k rows of one context |
| `chosen` | display string of the target title |
| `rejected` | display string of one negative |
| `user_idx`, `chosen_item`, `rejected_item` | bookkeeping; not model inputs |

Drop a row if the pool has no eligible negative. If the raw title is empty, the display string is `item {item_idx}`, so two blank titles do not become the same completion. If two display strings still match, append ` [id:{item_idx}]` to both. DPO on identical strings is a zero gradient.

Write `pairs_{train,val,test}.jsonl`.

### 2.4 Leakage

Popularity features and frequency tertiles use **training targets only**. Val and test items may be rare; that is the long tail, not a reason to count them in `freq`. Do not fit anything on test pairs. Val pairs are for the margin / pair-accuracy checkpoint signal, not for picking a test-set story after the fact — choose `beta` and the checkpoint on val, then report test once.

---

## 3. Why this is not A4's pointwise Yes/No

DPO's loss is a Bradley-Terry comparison of two completions under **one** prompt:

```
L = -log σ( β [ log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x) ] )
```

A4 pointwise SFT uses a different prompt per candidate (`Candidate: <title> … Answer Yes or No`). Those rows are not `(x, y_w, y_l)` pairs. Feeding them to `DPOTrainer` would compare unrelated prompts.

A5's completion is the **title itself**:

```
prompt:   history + "Which product will they interact with next? Answer with the product title only."
chosen:   <target title>
rejected: <negative title>
```

`trl` applies the tokenizer chat template when `prompt` / `chosen` / `rejected` are message lists (`user` message, then `assistant` message). Store plain strings in jsonl; `train_dpo.py` wraps them. Loss is on the assistant tokens, which is the title.

One positive and k negatives produces k independent 1-vs-1 rows. That is what `DPOTrainer` implements. It is weaker than S-DPO's single softmax over k negatives (the partition function there includes every negative at once). k separate pairs is the right default until the 1-vs-1 run is real.

---

## 4. Prompts (`src/prompts.py`)

Keep the template in one function. History stays chronological (oldest first). Do not shuffle history. Negatives are shuffled only when *sampling* which items become rejected rows.

```
You are recommending the next product for a shopper.
The shopper recently interacted with (oldest to newest):
1. <title>
2. <title>
...
Which product will they interact with next? Answer with the product title only.
```

Clip each title to `max_title_chars` (160 full / 80 small). Categories are not in the default prompt: the completion is a title, and extra fields make the chosen and rejected strings harder to score as exact titles. A4's pointwise prompt can keep categories; this one should not.

At inference the same user prompt is scored against **every** candidate title, not only the pair that was trained. The training negative is a sample; the ranking is over the full pool.

---

## 5. DPO fine-tune (`src/train_dpo.py`)

### 5.1 Model and reference policy

Default: `Qwen/Qwen2.5-1.5B-Instruct`.

LoRA, same targets as A4:

```python
from peft import LoraConfig, TaskType

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
```

Reference-model recipe:

1. If `sft_adapter` points at an A4 LoRA directory, load the base, `PeftModel.from_pretrained`, `merge_and_unload`. That merged stack is `π_ref`.
2. Attach a **new** LoRA. Default LoRA init is a zero update, so the forward pass still equals the merged weights. Pass `ref_model=None`. Current `trl` copies that initial adapter into a frozen adapter named `ref` and trains `default`. The copy stays at zero, so the reference forward pass is the merged SFT policy. That is the Zephyr "before." (Older `trl` got the same effect by disabling the adapter. Do not pass the same `PeftModel` object as `ref_model`; the trainer rejects that.)
3. If `sft_adapter` is null, the new LoRA sits on the raw Instruct model and the frozen `ref` adapter is the zero init, i.e. the base model. The run is still valid DPO. It is not "DPO on top of ranking SFT." `run_meta.json` records `ref_is_sft: false`.

Do not import A4's training modules. Read its adapter directory or its parquets only.

4-bit NF4 (`bitsandbytes`) is CUDA-only. On macOS set `load_in_4bit: false` and train LoRA in fp32/bf16. Merging an adapter into a 4-bit base is not supported: if both `sft_adapter` and `load_in_4bit` are set, merge in bf16, save the merged weights under `checkpoints/`, then reload that directory in 4-bit and attach the new LoRA. The saved merge is gitignored.

### 5.2 Optimizer hyperparameters

DPO learning rates are an order of magnitude below SFT. A4's `2e-4` will overwrite the reference policy and the implicit KL term stops meaning "stay close to a ranking model."

| knob | full | small |
|---|---|---|
| `learning_rate` | `5e-6` | `5e-6` |
| `beta` | `0.1` | `0.1` |
| epochs | 1 | 1 |
| `max_length` / `max_prompt_length` | 1024 / 768 | 512 / 384 |
| effective batch | 16 (2 × accum 8) | smaller |

`beta` is the temperature on the implicit reward `r(x, y) = β log π(y|x) / π_ref(y|x)`. Larger `beta` punishes departure from the reference harder. Pick it on **val pair accuracy and val NDCG**, not on train loss alone. Loss can fall while the policy only learns "shorter title" or "more popular title."

`trl` logs `rewards/chosen`, `rewards/rejected`, and `rewards/margins`. Turn W&B on for a real run (`use_wandb: true`). A margin that grows while val NDCG falls is the failure mode in §9, not a success.

Save the adapter only (`adapter_model.safetensors` + `adapter_config.json`) plus `run_meta.json` (`model_name`, `ref_is_sft`, `merged_sft_dir`).

---

## 6. Inference and evaluation

### 6.1 Length-normalized title log-prob (`src/infer.py`)

For a fixed user prompt `x` and a candidate title `y` with `T` tokens, the ranking score is

```
s(x, y) = (1 / T) Σ_{t=1..T} log π(y_t | x, y_<t)
```

`T` is the completion span `trl` trains: tokens after the generation prompt (the title plus the end-of-turn token), not the user-turn header. Rank the pool by `s` descending.

Why divide by `T`: token log-probs are ≤ 0, so the **sum** `Σ log π` prefers short titles. A long title can be more likely per token and still lose on the sum. Length normalization is what makes "higher per-token probability" win. The unit test locks this: a longer title with a better mean log-prob ranks above a shorter title whose sum is higher (closer to zero) only because it has fewer tokens.

Score the reference policy (adapters off, or the base / merged SFT) and the DPO policy on the same pools.

### 6.2 Metrics (`src/eval.py`)

Next-item metrics, identical definitions to A4:

```python
def recall_at_k(ranked, target, k):
    return float(target in ranked[:k])

def ndcg_at_k(ranked, target, k):
    for i, item in enumerate(ranked[:k]):
        if item == target:
            return 1.0 / math.log2(i + 2)
    return 0.0
```

Report, for reference and for DPO:

- `candidate_hit_rate`
- NDCG@5/10/20 and Recall@10
- `ndcg@10_{long_tail,mid,head}` from training-target frequency tertiles
- `pair_accuracy`: fraction of val/test pairs with `s(x, y_w) > s(x, y_l)`
- `reward_margin`: mean of `log π(y_w|x) − log π(y_l|x)` using the **sum** (this is the DPO logit, up to `beta` and the reference subtraction). Also log the length-normalized margin so it matches the ranker.

`pair_accuracy` moving from 0.5 toward 1.0 means the policy learned the pairs. NDCG moving is the engagement proxy the roadmap asks for. They can diverge; report both.

Retriever order (the candidate list as stored) is the third column, so a DPO model that ignores the prompt cannot hide behind a strong pool.

---

## 7. Causal parallel (the interview note)

Write this in `RESULTS.md` next to the numbers, in your own words. The skeleton:

**Same object, different name.** DPO never fits a reward model. It uses the closed form of the Bradley-Terry model: the optimal policy under a KL penalty to `π_ref` has reward `r(x, y) = β log π(y|x) − β log π_ref(y|x) + β log Z(x)`. The partition term cancels in the pairwise difference, which is why the loss in §3 has no `Z`. A click model `P(click | user, item)` is also a pairwise (or logistic) comparison of "this item vs not." The algebra is the cousin of a logistic treatment model. It is not an average treatment effect.

**`π_ref` is the logging policy.** In off-policy bandits the logging policy chose what was shown. Here the reference is whatever policy you froze: A4's SFT ranker, or the base Instruct model. The KL term keeps the new policy from wandering off the support of that reference. That is the same *role* as a propensity score (stay near the policy that generated the data), and it is not a propensity. Nothing in the loss divides by `P(item shown | user)`.

**The rejected sample is not a control.** A control in the causal sense was eligible to be chosen and was observed not chosen. Our rejected title was sampled from a candidate generator after the fact. Popular items are over-represented in a popularity-weighted pool, so the policy is pushed to prefer the next-item title *over popular titles*. That can look like long-tail lift and can also be an artifact of how negatives were drawn. If NDCG@10 on the head bucket falls while the long-tail bucket rises, say that, and say the negative sampler is a confounder.

**What would change with real skips.** An impression log (item B shown, no click) would make `y_l` a real control. Position bias would still remain; Schnabel-style IPS would reweight pairs by the inverse propensity of being shown. We cannot estimate that propensity from reviews alone. Do not report an "unbiased DPO" number.

**What DPO does not buy you.** It does not identify a causal effect of showing item A. It changes a language model so that the preferred title scores higher than the sampled alternative, close to the reference. The engagement proxy is NDCG of that scoring rule on held-out next items. Treat a gain as a ranking gain under this pair definition.

---

## 8. Milestone checklist

- [ ] Pairs jsonl written; every row has the same prompt for its `(chosen, rejected)`; rejected item is in the pool and is not the target; empty titles do not collapse to one string.
- [ ] `tests/test_dpo.py` passes with no model download: DPO loss at margin 0 is `log(2)` and falls as the policy margin grows; length-norm ranking disagrees with the sum when a longer title has a better per-token log-prob.
- [ ] `prepare` logs k-core before/after (own-data path) or logs that it reused A4 parquets; train hit rate ~100% with inject on; test hit rate logged with inject off.
- [ ] LoRA DPO runs at least `--max_steps 20` on `dpo_small` (GPU or Mac, 4-bit off on Darwin).
- [ ] `run_meta.json` records whether the reference was A4 SFT or the base Instruct model.
- [ ] Val margin and pair accuracy computed for reference and for the adapter. Margin is not stuck at 0.
- [ ] Test NDCG table: retriever order vs reference vs DPO, plus popularity buckets, written to `checkpoints/.../test_metrics.json`.
- [ ] `RESULTS.md` filled: table, margin vs NDCG, causal note (§7) tied to those numbers.

---

## 9. Common failure modes (debug playbook)

| Symptom | Likely cause | Fix |
|---|---|---|
| Train loss flat at `log(2)` | Chosen and rejected strings are identical, or the reference and policy scores match because the adapter is not training | Check display-title collision; confirm LoRA params require grad |
| Loss → 0 in a few steps, generations are nonsense | Learning rate is the SFT value `2e-4` | `5e-6`; one epoch; raise `beta` toward 0.2–0.5 if val format collapses |
| Margin ↑, NDCG ↓ | Policy learned pair artifacts (short titles, head items in the negative sampler) rather than the next item | Rank by length-normalized log-prob; inspect head vs long-tail NDCG; lower lr; check negative sampler |
| Short titles always win | Ranking uses the sum of token log-probs | Divide by title token count (§6.1) |
| Test NDCG ≈ 0 for every method | Target not in the pool (`inject_target_at_eval: false` and a weak pool) | Read `candidate_hit_rate` before touching LoRA |
| DPO equals the reference | `beta` very large, or eval loaded the base model for both columns | Read `run_meta.json`; the DPO column must load the adapter (`set_adapter("default")`) |
| `ref_is_sft: false` and you wanted A4 | `sft_adapter` null or the directory has no `adapter_config.json` | Point at A4's adapter; merge happens before the new LoRA |
| CUDA OOM / Mac 4-bit crash | bitsandbytes is not for Metal | `load_in_4bit: false` on Darwin; `dpo_small` lengths; batch 1 |
| A4 indices don't match titles | `item_idx` rebuilt while reading A4 parquets | Read `items.parquet` from the same processed dir as the candidate files |

---

## 10. Suggested order of work (time-boxed)

1. **Day 1:** `prompts.py`, `pairs.py`, `losses.py`, `tests/test_dpo.py`. No download.
2. **Day 2:** `prepare_preferences.py` on `dpo_small` (own pools if A4 parquets are absent). Read a few jsonl rows by hand.
3. **Day 3:** `train_dpo.py` for `--max_steps 20`. Confirm `run_meta.json` and an adapter directory.
4. **Day 4:** `infer.py` length-norm scoring; reference vs adapter pair accuracy on val.
5. **Day 5:** Full small-config train; test table with hit rate. Fix short-title bias or class of negatives if margin and NDCG disagree.
6. **Day 6:** Point `sft_adapter` at A4 if that adapter exists; rerun the comparison so the reference is ranking-SFT.
7. **Day 7:** `RESULTS.md` with §7 tied to the actual head/tail numbers. Notebook that rebuilds pairs and reads `test_metrics.json`.

About a week part-time if A4 pools already exist. Budget a day for "loss fell, NDCG did not" — that is this project's version of A2 codebook collapse and A4 always-No.

---

## 11. What this sets up for A6–A7

- **A6** wants one token stream of actions and items. The preference pairs here are still two full completions. Keep `pairs_*.jsonl` and the candidate parquets; do not assume A6 will speak DPO. The useful leftover is the same leave-last-out sequences and the honesty about missing impressions.
- **A7** will time the scorer. Length-norm scoring is `O(n_cand)` forward passes per user, the same cost shape as A4 pointwise. Log wall-clock next to NDCG even though this repo does not serve.
- The adapter is a normal PEFT folder. A later serving run can load it with the base model name in `run_meta.json`.

---

## 12. References

1. Rafailov et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. https://arxiv.org/abs/2305.18290
2. Tunstall et al. *Zephyr: Direct Distillation of LM Alignment.* 2023. https://arxiv.org/abs/2310.16944 — SFT then DPO, small DPO learning rate.
3. Chen et al. *On Softmax Direct Preference Optimization for Recommendation.* 2024. https://arxiv.org/abs/2406.00333 — multi-negative DPO for generative recommendation. Not this repo's trainer.
4. Schnabel et al. *Recommendations as Treatments: Debiasing Learning and Evaluation.* WSDM 2016. https://arxiv.org/abs/1602.05352 — MNAR exposure.
5. Hugging Face `trl` `DPOTrainer`: https://huggingface.co/docs/trl/dpo_trainer
6. Roadmap A4 handoff: candidate lists with one positive and many retrieved negatives are the pairs. A4 pointwise Yes/No prompts are not.
