# Results

Not written yet — fill in after a run (`configs/dpo.yaml`, or `dpo_small.yaml` for a smoke test). Small-config numbers are not comparable to the full table.

## Reference policy

| | |
|---|---|
| `ref_is_sft` | |
| base | `Qwen/Qwen2.5-1.5B-Instruct` or merged A4 adapter |
| beta | 0.1 |
| learning rate | 5e-6 |

If `ref_is_sft` is false, this run is DPO from the base Instruct model, not from a ranking-SFT policy.

## Candidate pool

| split | n_users | n_cand | hit rate (target in pool) |
|---|---|---|---|
| train | | | |
| val | | | |
| test | | | |

Train injects the target. Val/test do not, unless `inject_target_at_eval: true`.

## Next-item NDCG / Recall (leave-last-out)

Length-normalized title log-prob. Same protocol as A4, not A1's set retrieval.

| method | NDCG@10 | Recall@10 | pair accuracy | reward margin |
|---|---|---|---|---|
| retriever order | | | — | — |
| reference policy | | | | |
| DPO LoRA | | | | |

`reward margin` is the mean of `log π(y_w|x) − log π(y_l|x)` (sum over title tokens). Pair accuracy uses the length-normalized score.

## Popularity buckets (NDCG@10)

Tertiles of **training-target** frequency.

| bucket | n | retriever | reference | DPO |
|---|---|---|---|---|
| long_tail | | | | |
| mid | | | | |
| head | | | | |

## Causal note

Tie these numbers to the argument in the implementation doc §7. Do not call a rejected item a skip.

- Did the margin move with NDCG, or against it?
- Did the head bucket and the long-tail bucket move together? The negative sampler is a confounder if they split.
- What would change if `y_l` were a logged impression that was not chosen?

## What broke

(identical titles, short-title bias before length-norm, learning rate too high, 4-bit on Mac, target missing from the pool, reference was not the A4 adapter, …)
