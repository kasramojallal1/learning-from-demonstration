# training/train_lora_llama32_3b.py
# Fine-tune meta-llama/Llama-3.2-3B-Instruct with LoRA on Mac (MPS) for bin-packing LfD.

import os
from typing import Dict, List

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# --------------------
# Config
# --------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_ROOT, "data", "processed"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_ROOT, "checkpoints", "lfd-lora-llama32-3b"))

# LoRA (tuned for Llama arch)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))  # try 512; go 384/320 if OOM
BATCH_SIZE = 1
GRAD_ACCUM = 48
LR = 2e-4
EPOCHS = 3
MAX_GRAD_NORM = 1.0
LOG_STEPS = 50
SAVE_STEPS = 500
SEED = 42

# MPS allocator hints
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def build_messages(row: Dict[str, str]):
    """
    Format one training example as Llama-3.2 chat turns.
    Keeps the same two-pair structure as your Gemma training:
      (user_pick -> assistant_pick) and (user_path -> assistant_path)
    """
    return [
        {"role": "system", "content": "You are a bin-packing assistant. Respond in STRICT JSON only."},
        {"role": "user", "content": row["user_pick"]},
        {"role": "assistant", "content": row["assistant_pick"]},
        {"role": "user", "content": row["user_path"]},
        {"role": "assistant", "content": row["assistant_path"]},
    ]


def main():
    set_seed(SEED)

    # Devices
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps") if use_mps else torch.device("cpu")
    print("✅ Using MPS device." if use_mps else "⚠️ MPS not available; using CPU.")

    # Tokenizer (with Llama 3.2 chat template)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load base model on CPU (fp16) to save RAM, then move to MPS
    print("🔧 Loading base model on CPU (fp16), then moving to MPS...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,  # CPU first
    )
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Processed dataset not found at '{DATA_DIR}'. "
            "Run: python training/sft_prepare.py --input data/demos/binpack_lfd.jsonl --out data/processed --train_ratio 0.9"
        )
    ds = load_from_disk(DATA_DIR)
    train_raw = ds["train"]
    test_raw = ds.get("test")

    # Tokenization via Llama-3.2 chat template
    def map_batch(batch: Dict[str, List[str]]):
        n = len(batch["user_pick"])
        rows = [{k: batch[k][i] for k in batch.keys()} for i in range(n)]
        # Build chat-formatted strings using the tokenizer's template
        texts = [
            tok.apply_chat_template(
                build_messages(r),
                tokenize=False,
                add_generation_prompt=False,  # training: full supervised targets already in messages
            )
            for r in rows
        ]
        enc = tok(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        # Teacher forcing over the whole sequence (mirrors your Gemma setup)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("📝 Tokenizing train set with Llama-3.2 chat template...")
    train_ds = train_raw.map(map_batch, batched=True, remove_columns=train_raw.column_names)
    test_ds = test_raw.map(map_batch, batched=True, remove_columns=test_raw.column_names) if test_raw else None
    print("🔍 Sample keys:", train_ds[0].keys(), " | seq len:", len(train_ds[0]["input_ids"]))

    # TrainingArguments — IMPORTANT: keep fp16/bf16 False on MPS
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds if (test_ds and args.eval_strategy != "no") else None,
        processing_class=tok,  # matches your previous script usage
        data_collator=default_data_collator,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed.")

    print(f"💾 Saving to {OUTPUT_DIR} ...")
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)
    print("✅ Saved adapter and tokenizer.")
    print("🎉 Done.")


if __name__ == "__main__":
    main()