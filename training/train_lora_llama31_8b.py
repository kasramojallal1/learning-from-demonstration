# training/train_lora_llama31_8b_gpu_standalone.py
# Standalone LoRA fine-tuning for meta-llama/Llama-3.1-8B-Instruct on NVIDIA GPUs.
# No environment variables required. Uses QLoRA (4-bit) by default — recommended for 16 GB GPUs.

# https://wormhole.app/PpModY#Qi72xK7_ep3g07fkcSJCFw

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
# Fixed config (tweak in-file if you want)
# --------------------
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Paths relative to repo root (this file is in training/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "lfd-lora-llama31-8b-gpu")

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training knobs (safe defaults for 16 GB with 4-bit)
MAX_LEN = 512
BATCH_SIZE = 1         # try 2 after a clean pass
GRAD_ACCUM = 64        # effective batch size = BATCH_SIZE * GRAD_ACCUM
LR = 2e-4
EPOCHS = 3
MAX_GRAD_NORM = 1.0
LOG_STEPS = 50
SAVE_STEPS = 500
SEED = 42

def build_messages(row: Dict[str, str]):
    """Format dataset rows as Llama chat turns."""
    return [
        {"role": "system", "content": "You are a bin-packing assistant. Respond in STRICT JSON only."},
        {"role": "user", "content": row["user_pick"]},
        {"role": "assistant", "content": row["assistant_pick"]},
        {"role": "user", "content": row["user_path"]},
        {"role": "assistant", "content": row["assistant_path"]},
    ]

def main():
    # ---- Sanity checks
    assert torch.cuda.is_available(), "CUDA GPU not detected. Run on the NVIDIA machine."
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
        from peft.utils.other import prepare_model_for_kbit_training  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit QLoRA on the 8B model. "
            "Install it first:  pip install bitsandbytes"
        ) from e

    set_seed(SEED)

    # ---- Perf toggles
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print("🟢 CUDA device:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    # ---- Tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---- Data
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Processed dataset not found at '{DATA_DIR}'.\n"
            "Expected columns: user_pick, assistant_pick, user_path, assistant_path"
        )
    ds = load_from_disk(DATA_DIR)
    train_raw = ds["train"]
    test_raw = ds.get("test")

    def map_batch(batch: Dict[str, List[str]]):
        n = len(batch["user_pick"])
        rows = [{k: batch[k][i] for k in batch.keys()} for i in range(n)]
        texts = [
            tok.apply_chat_template(
                build_messages(r),
                tokenize=False,
                add_generation_prompt=False,
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
        enc["labels"] = enc["input_ids"].copy()
        return enc

    print("📝 Tokenizing train set...")
    train_ds = train_raw.map(map_batch, batched=True, remove_columns=train_raw.column_names)
    test_ds = test_raw.map(map_batch, batched=True, remove_columns=test_raw.column_names) if test_raw else None
    print("🔍 Sample keys:", train_ds[0].keys(), "| seq len:", len(train_ds[0]["input_ids"]))

    # ---- Model (QLoRA 4-bit)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("🔧 Loading 8B base in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        # If you later install flash-attn, you can add: attn_implementation="flash_attention_2"
    )

    # Gradient checkpointing + k-bit prep
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapters
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

    # ---- Optimizer choice
    fused_ok = torch.cuda.get_device_capability(0)[0] >= 8
    optim_name = "adamw_torch_fused" if fused_ok else "adamw_torch"

    # ---- Training args
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
        eval_strategy="no" if test_ds is None else "steps",
        eval_steps=SAVE_STEPS,
        report_to="none",
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim=optim_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds if (test_ds and args.eval_strategy != "no") else None,
        processing_class=tok,
        data_collator=default_data_collator,
    )

    print("🚀 Starting QLoRA training on CUDA...")
    trainer.train()
    print("✅ Training completed.")

    print(f"💾 Saving to {OUTPUT_DIR} ...")
    trainer.save_model()
    tok.save_pretrained(OUTPUT_DIR)
    print("✅ Saved adapter and tokenizer.")
    print("🎉 Done.")

if __name__ == "__main__":
    main()