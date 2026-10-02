"""LoRA DPO of Qwen2.5-1.5B-Instruct with trl.DPOTrainer (A5 §5).

Reference policy:
- sft_adapter set: merge that A4 LoRA into the base, save it, train a new LoRA
  on top. The new LoRA starts at zero, so the forward pass still equals SFT.
- sft_adapter null: the base is the Instruct model. run_meta.json records
  ref_is_sft false.
- ref_model=None. Current trl copies the initial LoRA into a frozen adapter
  named "ref" and trains "default". The copy stays at zero, so the reference
  forward pass is the merged SFT weights, or the base Instruct model.

Usage:
    python src/train_dpo.py --config configs/dpo_small.yaml --max_steps 20 --max_pairs 32
"""
from __future__ import annotations

import argparse
import inspect
import platform
import sys
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pairs import read_pairs
from utils import load_config, save_json, set_seed


def resolve_load_in_4bit(cfg: dict) -> bool:
    if not cfg.get("load_in_4bit", False):
        return False
    if platform.system() == "Darwin":
        print("load_in_4bit ignored on macOS (bitsandbytes 4-bit is not for Metal)")
        return False
    return True


def precision_flags() -> tuple[torch.dtype, dict]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, {"bf16": True, "fp16": False}
    if torch.cuda.is_available():
        return torch.float16, {"bf16": False, "fp16": True}
    return torch.float32, {"bf16": False, "fp16": False}


def _bnb_config(dtype: torch.dtype) -> BitsAndBytesConfig:
    compute = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute,
    )


def merge_sft_adapter(model_name: str, sft_adapter: str, merged_dir: Path, tokenizer) -> None:
    if (merged_dir / "config.json").exists():
        print(f"[skip] merged SFT already at {merged_dir}")
        return
    print(f"merging SFT adapter {sft_adapter} -> {merged_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    from peft import PeftModel

    merged = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    del merged, base


def load_base(cfg: dict, tokenizer):
    """Return (model, ref_is_sft, base_path_for_reload)."""
    load_in_4bit = resolve_load_in_4bit(cfg)
    dtype, _ = precision_flags()
    model_name = cfg["model_name"]
    sft = cfg.get("sft_adapter") or None
    output_dir = Path(cfg["output_dir"])
    if sft:
        sft_path = Path(sft)
        if not (sft_path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"sft_adapter={sft} has no adapter_config.json. "
                "Point at an A4 LoRA directory, or set sft_adapter: null."
            )
        merged_dir = output_dir / "sft_merged"
        merge_sft_adapter(model_name, str(sft_path), merged_dir, tokenizer)
        load_from = str(merged_dir)
        ref_is_sft = True
    else:
        load_from = model_name
        ref_is_sft = False
        print("no sft_adapter; reference policy is the base Instruct model, not a ranking-SFT policy")

    kwargs = {"low_cpu_mem_usage": True}
    if load_in_4bit:
        kwargs["quantization_config"] = _bnb_config(dtype)
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(load_from, **kwargs)
    return model, ref_is_sft, load_from


def lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )


def to_conversational(rows: list[dict]) -> list[dict]:
    """trl applies the chat template when prompt/chosen/rejected are messages."""
    out = []
    for row in rows:
        out.append(
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            }
        )
    return out


def filter_fields(cls, raw: dict) -> dict:
    fields = set(getattr(cls, "__dataclass_fields__", {}))
    params = set(inspect.signature(cls.__init__).parameters)
    allowed = fields | params
    kept = {k: v for k, v in raw.items() if k in allowed}
    dropped = sorted(set(raw) - set(kept))
    if dropped:
        print("DPOConfig ignored:", ", ".join(dropped))
    return kept


def build_trainer(model, tokenizer, train_rows: list[dict], cfg: dict, max_steps: int | None):
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    _, prec = precision_flags()
    raw = {
        "output_dir": cfg["output_dir"],
        "beta": float(cfg.get("beta", 0.1)),
        "learning_rate": float(cfg["learning_rate"]),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1)),
        "max_length": int(cfg["max_length"]),
        "max_prompt_length": int(cfg["max_prompt_length"]),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.05)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "wandb" if cfg.get("use_wandb") else "none",
        "run_name": cfg.get("wandb_run_name"),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "seed": int(cfg.get("seed", 42)),
        "bf16": prec["bf16"],
        "fp16": prec["fp16"],
        "loss_type": "sigmoid",
    }
    if max_steps is not None:
        raw["max_steps"] = int(max_steps)
    args = DPOConfig(**filter_fields(DPOConfig, raw))
    dataset = Dataset.from_list(to_conversational(train_rows))
    init_kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": dataset,
    }
    sig = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in sig:
        init_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        init_kwargs["tokenizer"] = tokenizer
    return DPOTrainer(**init_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = Path(cfg["processed_dir"]) / "pairs_train.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"missing {pairs_path}. Run data/prepare_preferences.py --config {args.config} first."
        )
    rows = read_pairs(pairs_path)
    if args.max_pairs is not None:
        rows = rows[: int(args.max_pairs)]
    if not rows:
        raise RuntimeError(f"no training pairs in {pairs_path}")
    print(f"training pairs: {len(rows):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, ref_is_sft, load_from = load_base(cfg, tokenizer)
    model = get_peft_model(model, lora_config(cfg))
    model.print_trainable_parameters()
    if cfg.get("gradient_checkpointing", True):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainer = build_trainer(model, tokenizer, rows, cfg, args.max_steps)
    trainer.train()
    # trl may have added a frozen "ref" adapter. Save the trained one.
    trained = trainer.model
    if hasattr(trained, "set_adapter") and "default" in getattr(trained, "peft_config", {}):
        trained.set_adapter("default")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    meta = {
        "model_name": cfg["model_name"],
        "ref_is_sft": ref_is_sft,
        "base_path": load_from,
        "merged_sft_dir": str(output_dir / "sft_merged") if ref_is_sft else None,
        "sft_adapter": cfg.get("sft_adapter"),
        "beta": float(cfg.get("beta", 0.1)),
        "learning_rate": float(cfg["learning_rate"]),
        "n_train_pairs": len(rows),
        "max_steps": args.max_steps,
    }
    save_json(meta, output_dir / "run_meta.json")
    print("wrote", output_dir)
    print(meta)


if __name__ == "__main__":
    main()
