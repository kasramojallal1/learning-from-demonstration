# training/train_lora.py
# Fine-tune google/gemma-2-2b-it with LoRA on Mac (MPS) for bin-packing LfD.

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
MODEL_NAME = "google/gemma-2-2b-it"
DATA_DIR = "data/processed"           # produced by sft_prepare.py
OUTPUT_DIR = "checkpoints/lfd-lora-gemma2-2b"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
MAX_LEN = int(os.environ.get("MAX_LEN", "384"))  # try 512 if fits; 320/256 if OOM
BATCH_SIZE = 1
GRAD_ACCUM = 48
LR = 2e-4
EPOCHS = 3
MAX_GRAD_NORM = 1.0
LOG_STEPS = 50
SAVE_STEPS = 500
EVAL_STRATEGY = "no"
SEED = 42

# MPS allocator hint: allow immediate reuse (helps fragmentation)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
# Optional (usually harmless): let ops fall back to CPU if not implemented
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def build_text(row: Dict[str, str]) -> str:
    return (
        "<|system|> You are a bin-packing assistant. Respond in STRICT JSON only.<|end|>\n"
        f"<|user|>{row['user_pick']}<|end|>\n"
        f"<|assistant|>{row['assistant_pick']}<|end|>\n"
        f"<|user|>{row['user_path']}<|end|>\n"
        f"<|assistant|>{row['assistant_path']}<|end|>"
    )


def main():
    set_seed(SEED)

    # --------------------
    # Devices
    # --------------------
    use_mps = torch.backends.mps.is_available()
    mps = torch.device("mps") if use_mps else torch.device("cpu")
    cpu = torch.device("cpu")
    print("✅ Using MPS device." if use_mps else "⚠️ MPS not available; using CPU.")

    # --------------------
    # Tokenizer
    # --------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --------------------
    # Load base model on CPU in FP16 to avoid MPS warmup allocation
    # --------------------
    print("🔧 Loading base model on CPU (fp16), then moving to MPS...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,     # cut memory in half vs fp32
        low_cpu_mem_usage=True,
        device_map=None,               # ensure load happens on CPU
    )
    # Now move to MPS (or stay CPU)
    model.to(mps if use_mps else cpu)

    # Reduce runtime memory
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --------------------
    # Apply LoRA
    # --------------------
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

    # --------------------
    # Load dataset
    # --------------------
    ds = load_from_disk(DATA_DIR)  # DatasetDict with "train" and "test"
    train_raw = ds["train"]
    test_raw = ds.get("test", None)

    # --------------------
    # Tokenization (batched)
    # --------------------
    def map_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        n = len(batch["user_pick"])
        rows = [{k: batch[k][i] for k in batch.keys()} for i in range(n)]
        texts = [build_text(r) for r in rows]
        enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("📝 Tokenizing train set...")
    train_ds = train_raw.map(map_batch, batched=True, remove_columns=train_raw.column_names)

    if test_raw is not None:
        print("📝 Tokenizing test set...")
        test_ds = test_raw.map(map_batch, batched=True, remove_columns=test_raw.column_names)
    else:
        test_ds = None

    print("🔍 Sample tokenized keys:", train_ds[0].keys())
    print("🔍 input_ids len:", len(train_ds[0]["input_ids"]), " (should be MAX_LEN)")

    # --------------------
    # TrainingArguments
    # --------------------
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
        evaluation_strategy=EVAL_STRATEGY,
        report_to="none",
        bf16=False,        # not on MPS
        fp16=True,         # we run fp16 on MPS/CPU to fit memory
        dataloader_pin_memory=False,  # safer on MPS
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds if (test_ds and EVAL_STRATEGY != "no") else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # --------------------
    # Train
    # --------------------
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training completed.")

    # --------------------
    # Save adapter + tokenizer
    # --------------------
    print(f"💾 Saving to {OUTPUT_DIR} ...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Saved adapter and tokenizer.")

    # Info
    steps = len(trainer.get_train_dataloader())
    total_tokens_per_update = BATCH_SIZE * MAX_LEN * GRAD_ACCUM
    print(f"ℹ️ Steps/epoch: {steps} | tokens/optimizer-step (approx): {total_tokens_per_update}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
