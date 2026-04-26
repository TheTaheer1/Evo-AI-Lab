"""
train_grpo.py
DPO fine-tuning using JSONL from EvoAI Lab (chosen / rejected pairs).
Designed for Google Colab + GPU: set HF_TOKEN for gated models, EVOAI_TRAIN_DATA for JSONL path.

Install (Colab example):
  pip install torch transformers trl datasets peft accelerate bitsandbytes huggingface_hub

Upload `data/training_pairs.jsonl` from your EvoAI Lab run (or set EVOAI_TRAIN_DATA).
"""
import json
import sys
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

_DATA_PATH = os.environ.get("EVOAI_TRAIN_DATA", "data/training_pairs.jsonl")
_REWARD_LOG_PATH = os.environ.get("EVOAI_REWARD_LOG", "data/reward_log.json")
_MODEL_ID = os.environ.get("EVOAI_BASE_MODEL", "meta-llama/Meta-Llama-3-8B")
_OUTPUT_DIR = os.environ.get("EVOAI_OUTPUT_DIR", "./evoai-dpo-output")


def _maybe_hf_login() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("[train_grpo] No HF_TOKEN — public models only, or set HF_TOKEN for Llama gates.")
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("[train_grpo] Hugging Face Hub login OK.")
    except Exception as e:
        print(f"[train_grpo] HF login failed (continuing if model is public): {e}")


def load_dataset_from_disk(path: str = _DATA_PATH) -> Dataset:
    """
    Read JSONL training pairs and convert to HuggingFace Dataset.
    Exits gracefully if file doesn't exist or is empty.
    """
    data_path = Path(path)
    if not data_path.exists():
        print(
            f"[train_grpo] ERROR: Training data not found at '{path}'.\n"
            "Run the EvoAI Lab training loop first to generate data:\n"
            "  uvicorn app:app --reload  (then trigger steps via the UI or /api/run-steps)"
        )
        sys.exit(1)

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[train_grpo] Warning: skipping malformed line: {e}")

    if not records:
        print(f"[train_grpo] ERROR: No valid records found in '{path}'. Cannot train.")
        sys.exit(1)

    print(f"[train_grpo] Loaded {len(records)} training pairs from {path}")

    cleaned = []
    for r in records:
        prompt = (r.get("prompt") or "").strip()
        chosen = (r.get("chosen") or "").strip()
        rejected = (r.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            continue
        if chosen == rejected:
            continue
        cleaned.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "reward": float(r.get("reward", 0.0)),
        })

    if not cleaned:
        print(f"[train_grpo] ERROR: No valid prompt/chosen/rejected rows in '{path}'.")
        sys.exit(1)

    print(f"[train_grpo] Using {len(cleaned)} non-empty contrastive pairs (skipped incomplete rows).")
    return Dataset.from_list(cleaned)


def main():
    _maybe_hf_login()
    print(f"[train_grpo] Training data: {_DATA_PATH}")
    print(f"[train_grpo] Loading base model: {_MODEL_ID}")
    print(f"[train_grpo] CUDA available: {torch.cuda.is_available()}")

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset = load_dataset_from_disk(_DATA_PATH)

    # ── Load model and tokenizer ───────────────────────────────────────────
    # Use 4-bit quantisation if bitsandbytes is available (for Colab / limited GPU)
    load_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("[train_grpo] 4-bit quantisation enabled via bitsandbytes.")
    except ImportError:
        print("[train_grpo] bitsandbytes not available — loading in full precision.")

    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, trust_remote_code=True, **load_kwargs)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── DPO config ────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        beta=0.1,
        save_steps=50,
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        bf16=False,
        fp16=True,
    )

    # ── Trainer (TRL API differs slightly by version) ─────────────────────
    _trainer_kw = dict(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
    )
    try:
        trainer = DPOTrainer(processing_class=tokenizer, **_trainer_kw)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tokenizer, **_trainer_kw)

    print("[train_grpo] Starting DPO training...")
    trainer.train()

    # ── Save final model ───────────────────────────────────────────────────
    final_dir = f"{_OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[train_grpo] Model saved to {final_dir}")

    # ── Post-training summary ──────────────────────────────────────────────
    reward_log = []
    if Path(_REWARD_LOG_PATH).exists():
        try:
            with open(_REWARD_LOG_PATH, "r", encoding="utf-8") as f:
                reward_log = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if reward_log:
        rewards = [r.get("reward", 0.0) for r in reward_log]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5
        positive_count = sum(1 for r in reward_log if r.get("is_positive", False))

        print(f"\n{'='*50}")
        print(f"  EvoAI Lab — GRPO Training Summary")
        print(f"{'='*50}")
        print(f"  Training pairs:  {len(dataset)}")
        print(f"  Mean reward:     {mean_reward:.4f}")
        print(f"  Reward std:      {std_reward:.4f}")
        print(f"  Positive steps:  {positive_count}/{len(reward_log)}")
        print(f"  Model output:    {final_dir}")
        print(f"{'='*50}")
    else:
        print("[train_grpo] No reward log found — skipping summary statistics.")


if __name__ == "__main__":
    main()
